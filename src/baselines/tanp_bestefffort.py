"""tanp_bestefffort.py -- TaNP (Lin et al., WWW 2021, "Task-adaptive Neural Process for User Cold-Start
Recommendation") BEST-EFFORT core.

This is an HONEST best-effort row, NOT a verbatim reimplementation: TaNP targets a different benchmark
and protocol (episodic meta-learning for rating prediction over MovieLens-1M / LastFM / Gowalla cold
users, with a global task memory + a task-adaptive "customization" module modulating the decoder). We
port the NEURAL-PROCESS CORE and adapt it to the CASPER Liang-ML-25M implicit strong-generalization
ruler. EVERY substitution from the paper is documented below; flag best-effort at review (no ML-20M/25M
number to snap to).

NEURAL-PROCESS CORE (kept):
  * Encoder over the support set: each observed like is a (item, rating) pair; we mean-pool item
    embeddings over the support to form the context r_C, then amortize a latent posterior
    q(z | C) = N(mu(r_C), diag(sigma^2(r_C)))  (reparameterized).                       [NP encoder]
  * Decoder conditioned on z: an MLP maps z to a user taste vector u in item-embedding space; catalog
    scores = u @ E^T + b. This plays the role of TaNP's task-adaptive decoder (the "customization"
    modulation is reduced to z-conditioning -- see substitutions).                       [NP decoder]
  * Training objective: multinomial log-likelihood of the TARGET items under the decoder (the implicit
    analogue of the paper's per-item likelihood) + beta * KL(q(z|C) || N(0, I)).         [NP ELBO]

SUBSTITUTIONS FROM THE PAPER (documented on purpose):
  1. Implicit binary feedback (rating := 1 for every observed like) instead of explicit 1-5 ratings --
     matches our ruler (rating>3.5 binarization). The rating channel is therefore constant; the signal
     is the item set. (A rating scalar is still concatenated in the encoder for architectural fidelity.)
  2. Multinomial catalog decoder (score all 18,359 items, multinomial NLL) instead of per-(user,item)
     rating regression -- so the metric is the same full-catalog NDCG@10 every other baseline uses.
  3. DROPPED the global task memory + the task-adaptive customization/clustering module (the paper's
     namesake "task-adaptive" part). Reduced to the amortized NP with z-conditioned decoding. This is
     the single biggest deviation and the reason the row is labeled "best-effort core", not "TaNP".
  4. Episodes = per-train-user random support/target splits (support ~ 1-p, target ~ p of the user's
     items) drawn fresh each epoch, instead of the paper's fixed meta support/query sets.
  5. Strong-generalization fold-in eval: at test, the held-out user's fold-in items are the support;
     z = posterior MEAN (no sampling); decode -> catalog scores. Order-free, so it fits metrics.evaluate
     directly (no timestamp adaptation needed, unlike sasrec).

SIZE: kept deliberately SMALL/honest -- item_emb dim d (default 200, = the VAE latent), one-hidden-layer
  encoder and decoder. Modest training job; early stop on val FULL NDCG@10; checkpoint/resume.

Interface: fit(train, n_items, evaluator=None, args=None, ckpt=None, log=print) -> predict_fn
CLI: python src/baselines/tanp_bestefffort.py --smoke   # SYNTHETIC, code-path only (NOT the real split)
     python src/baselines/tanp_bestefffort.py            # full train+eval on the Liang ML-25M split
"""
import os
import sys
import json
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import metrics as M

torch.manual_seed(98765)
np.random.seed(98765)


def _defaults():
    return argparse.Namespace(
        d=200, hidden=512, zdim=64, dropout=0.5, target_prop=0.5, beta=0.2,
        lr=1e-3, batch=500, l2=0.0, epochs=200, patience=15, max_minutes=1e9, resume=True)


class TaNPCore(nn.Module):
    def __init__(self, n_items, d, hidden, zdim, dropout):
        super().__init__()
        self.n_items = n_items; self.d = d; self.zdim = zdim
        self.item_emb = nn.Embedding(n_items, d)
        nn.init.normal_(self.item_emb.weight, std=0.01)
        self.item_bias = nn.Parameter(torch.zeros(n_items))
        # encoder: mean-pooled context (d) + mean rating scalar (1) -> mu, logvar of z
        self.enc = nn.Sequential(nn.Linear(d + 1, hidden), nn.ReLU(), nn.Dropout(dropout))
        self.enc_mu = nn.Linear(hidden, zdim)
        self.enc_lv = nn.Linear(hidden, zdim)
        # decoder: z -> user taste vector in item-embedding space
        self.dec = nn.Sequential(nn.Linear(zdim, hidden), nn.ReLU(), nn.Dropout(dropout),
                                 nn.Linear(hidden, d))

    def encode(self, support):                          # support: (B, n_items) multi-hot float
        deg = support.sum(1, keepdim=True).clamp(min=1.0)
        ctx = (support @ self.item_emb.weight) / deg     # (B, d) mean item embedding
        rating = torch.ones(support.shape[0], 1, device=support.device)  # implicit rating = 1
        h = self.enc(torch.cat([ctx, rating], dim=1))
        return self.enc_mu(h), self.enc_lv(h)

    def reparam(self, mu, lv):
        if self.training:
            return mu + torch.randn_like(mu) * torch.exp(0.5 * lv)
        return mu

    def decode(self, z):
        u = self.dec(z)                                  # (B, d)
        return u @ self.item_emb.weight.T + self.item_bias   # (B, n_items)

    def forward(self, support):
        mu, lv = self.encode(support)
        z = self.reparam(mu, lv)
        return self.decode(z), mu, lv


def _make_predict(model):
    def predict(X_csr):
        model.eval()
        with torch.no_grad():
            s = torch.tensor(X_csr.toarray(), dtype=torch.float32)
            mu, _ = model.encode(s)
            return model.decode(mu).cpu().numpy().astype(np.float32)
    return predict


def _episode(Xbatch, target_prop, rng):
    """Split each row's items into support (kept) and target (held-out) multi-hot tensors."""
    B, n = Xbatch.shape
    support = np.zeros((B, n), dtype=np.float32)
    target = np.zeros((B, n), dtype=np.float32)
    Xc = Xbatch.tocsr()
    for r in range(B):
        cols = Xc.indices[Xc.indptr[r]:Xc.indptr[r + 1]]
        if len(cols) == 0:
            continue
        if len(cols) == 1:
            support[r, cols] = 1.0; continue             # too few to split -> all support
        nt = max(1, int(target_prop * len(cols)))
        perm = rng.permutation(len(cols))
        target[r, cols[perm[:nt]]] = 1.0
        support[r, cols[perm[nt:]]] = 1.0
    return torch.from_numpy(support), torch.from_numpy(target)


def loss_fn(logits, target, mu, lv, beta):
    log_sm = F.log_softmax(logits, dim=1)
    nll = -torch.mean(torch.sum(log_sm * target, dim=1))
    kld = -0.5 * torch.mean(torch.sum(1 + lv - mu.pow(2) - lv.exp(), dim=1))
    return nll + beta * kld, nll, kld


def fit(train, n_items, evaluator=None, args=None, ckpt=None, log=print):
    a = args or _defaults()
    model = TaNPCore(n_items, a.d, a.hidden, a.zdim, a.dropout)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr, weight_decay=a.l2)
    rng = np.random.RandomState(98765)
    N = train.shape[0]
    state = {"epoch": 0, "best": -1.0, "best_state": None, "best_epoch": 0, "history": []}
    if ckpt and getattr(a, "resume", True) and os.path.exists(ckpt):
        blob = torch.load(ckpt, map_location="cpu")
        model.load_state_dict(blob["model"]); opt.load_state_dict(blob["opt"]); state = blob["state"]
        log(f"[tanp] RESUMED ep{state['epoch']} best={state['best']:.4f}")
    predict = _make_predict(model)
    idxlist = np.arange(N)
    t0 = time.time()
    for epoch in range(state["epoch"], a.epochs):
        model.train(); np.random.shuffle(idxlist); ep_loss = 0.0; nb = 0
        for st in range(0, N, a.batch):
            Xb = train[idxlist[st:min(st + a.batch, N)]]
            support, target = _episode(Xb, a.target_prop, rng)
            if target.sum() == 0:
                continue
            opt.zero_grad()
            logits, mu, lv = model(support)
            loss, _, _ = loss_fn(logits, target, mu, lv, a.beta)
            loss.backward(); opt.step()
            ep_loss += float(loss); nb += 1
        state["epoch"] = epoch + 1
        sc, _ = evaluator(predict) if evaluator is not None else (-1.0, {})
        el = (time.time() - t0) / 60.0
        state["history"].append({"epoch": epoch + 1, "loss": ep_loss / max(nb, 1), "val": sc,
                                 "min": round(el, 1)})
        log(f"[tanp] ep{epoch+1:3d} loss {ep_loss/max(nb,1):.4f} val {sc:.4f} {el:.1f}m")
        if sc > state["best"]:
            state["best"] = sc; state["best_epoch"] = epoch + 1
            state["best_state"] = {k: v.clone() for k, v in model.state_dict().items()}
        if ckpt:
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "state": state,
                        "args": vars(a)}, ckpt)
        if evaluator is not None and (epoch + 1) - state["best_epoch"] >= a.patience:
            log(f"[tanp] early stop @ep{epoch+1} best {state['best']:.4f} @ep{state['best_epoch']}")
            break
        if el >= a.max_minutes:
            log(f"[tanp] wall budget {a.max_minutes}m hit @ep{epoch+1}; re-run to resume")
            break
    if state.get("best_state") is not None:
        model.load_state_dict(state["best_state"])
    predict.best_val = state["best"]; predict.best_epoch = state["best_epoch"]
    predict.hp = dict(d=a.d, hidden=a.hidden, zdim=a.zdim, dropout=a.dropout,
                      target_prop=a.target_prop, beta=a.beta, best_val=state["best"],
                      best_epoch=state["best_epoch"], role="TaNP best-effort NP core (no task memory)")
    return predict


# --------------------------------------------------------------------------- CLI / smoke
def _head_mask(train, n_items):
    cnt = np.asarray(train.sum(axis=0)).ravel().astype(np.float64)
    order = np.argsort(-cnt)
    cumfrac = np.cumsum(cnt[order]) / cnt.sum()
    head = np.zeros(n_items, dtype=bool)
    head[order[:np.searchsorted(cumfrac, 0.33) + 1]] = True
    return head


def _smoke():
    print("[tanp][SMOKE] synthetic tiny data (code-path only, NOT the Liang split)")
    from scipy import sparse
    rng = np.random.RandomState(0)
    n_users, n_items = 200, 130
    X = (sparse.random(n_users, n_items, density=0.15, random_state=rng,
                       data_rvs=lambda s: np.ones(s)) > 0).astype(np.float32).tocsr()
    tr = X[:150]
    te_tr = X[150:]
    te_te = (sparse.random(50, n_items, density=0.1, random_state=rng,
                           data_rvs=lambda s: np.ones(s)) > 0).astype(np.float32).tocsr()
    a = _defaults(); a.d = 16; a.hidden = 32; a.zdim = 8; a.epochs = 3; a.batch = 50; a.patience = 99
    hm = _head_mask(tr, n_items)

    def ev(pred):
        m = M.evaluate(pred, te_tr, te_te, batch_size=10, head_mask=hm)
        return m["ndcg@10"], m
    pr = fit(tr, n_items, evaluator=ev, args=a, ckpt=None, log=print)
    res = M.evaluate(pr, te_tr, te_te, batch_size=10, head_mask=hm)
    print(f"[tanp][SMOKE] OK full@10={res['ndcg@10']:.4f} tail@10={res['tail_ndcg@10']:.4f} "
          f"best_val={pr.best_val:.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--max_minutes", type=float, default=1e9)
    ap.add_argument("--out", default=None)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        _smoke(); return
    _HERE = os.path.dirname(os.path.abspath(__file__))
    PROC = os.path.abspath(os.path.join(_HERE, "..", "..", "data", "ml-25m", "proc"))
    meta = M.load_meta(PROC); n_items = meta["n_items"]
    train = M.load_train(n_items, PROC)
    va_tr, va_te = M.load_val(n_items, PROC)
    te_tr, te_te = M.load_test(n_items, PROC)
    hm = _head_mask(train, n_items)
    a = _defaults(); a.epochs = args.epochs; a.max_minutes = args.max_minutes
    ckpt = os.path.join(_ROOT_CKPT(), "tanp_ml25m_liang.pt")
    os.makedirs(os.path.dirname(ckpt), exist_ok=True)

    def ev(pred):
        m = M.evaluate(pred, va_tr, va_te, batch_size=500, head_mask=hm)
        return m["ndcg@10"], m
    t0 = time.time()
    predict = fit(train, n_items, evaluator=ev, args=a, ckpt=ckpt)
    res = M.evaluate(predict, te_tr, te_te, batch_size=500, head_mask=hm)
    res["seconds"] = time.time() - t0; res["hp"] = predict.hp
    print(f"[tanp] test full@10={res['ndcg@10']:.4f} tail@10={res['tail_ndcg@10']:.4f} "
          f"ndcg@100={res['ndcg@100']:.4f}")
    if args.out:
        json.dump(res, open(args.out, "w"), indent=2)


def _ROOT_CKPT():
    _HERE = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(_HERE, "..", "..", ".cache", "baselines"))


if __name__ == "__main__":
    main()
