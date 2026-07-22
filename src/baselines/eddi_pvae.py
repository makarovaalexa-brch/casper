"""eddi_pvae.py -- Partial-VAE (EDDI, Ma et al., "EDDI: Efficient Dynamic Discovery of High-Value
Information with Partial VAE", ICML 2019) adapted to implicit-feedback CF on the Liang ML-25M split.

WHAT EDDI'S PARTIAL-VAE IS (and how it maps to CF):
  The Partial-VAE encoder is a PERMUTATION-INVARIANT set encoder over the OBSERVED coordinates only.
  Ma et al. (Sec. 3, the "PNP" / point-net encoder): each observed feature o is embedded as
      s_o = h( [ e_o , x_o ] )              e_o = a learned per-feature (per-item) embedding,
                                            x_o = its observed value,
  then the set is pooled permutation-invariantly  c = SUM_o s_o  ("PointNet plus"), and c -> (mu,logvar)
  of a Gaussian latent z.  A decoder reconstructs the FULL feature vector from z.  This is exactly a
  set-input VAE that ingests a partially-observed row -- the property CASPER needs for cold interview.

  CF adaptation (documented deviations, faithful where stated):
    * Feature = item.  Observed value x_o = 1 (binary implicit like).  Because the value is constant,
      s_o = h([e_o, 1]) depends only on the item, so the pooled code c = SUM_{o in S} phi(e_o) is a
      sparse gather-sum:  c = X @ Phi  where Phi = phi(E) (n_items x k).  This is the SAME PointNet-plus
      pooling, evaluated exactly (no truncation) -- it is not an approximation, it is the binary-value
      reduction of Ma et al.'s encoder.  (phi keeps a nonlinearity so the encoder is not merely linear.)
    * Decoder + likelihood: MULTINOMIAL over the catalogue with the Mult-VAE^PR objective (Liang WWW2018)
      -- the standard implicit-CF likelihood; EDDI's original per-feature Gaussian/Bernoulli heads are for
      low-dim UCI tables, not a 18k-item catalogue.  Flag at review: the ENCODER is EDDI; the DECODER/
      LIKELIHOOD is Mult-VAE-class (there is no published EDDI-on-ML number to snap to).

HYPERPARAMS: EDDI's paper dims (latent 10, tiny MLPs) are for UCI; for an 18k catalogue we use
  Mult-VAE-class defaults, documented: item-embedding dim = 200, PointNet feature k = 600, latent 200,
  encoder MLP 600, KL beta annealed 0->0.2 over 200000 updates (as Mult-VAE), lr 1e-3, batch 500,
  input dropout 0.5 on the observed set.  Early stop on val FULL NDCG@10.  Checkpoint/resume like multvae.

Interface: fit(train, n_items, evaluator=None, args=None, ckpt=None, log=print) -> predict_fn
  predict_fn(fold_in_csr) -> dense np.float32 (batch x n_items) scores.
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


class PartialVAE(nn.Module):
    """EDDI PointNet-plus set encoder (binary-value CF reduction) + Mult-VAE^PR multinomial decoder."""

    def __init__(self, n_items, emb=200, feat=600, hidden=600, latent=200, dropout=0.5):
        super().__init__()
        self.n_items, self.latent, self.dropout = n_items, latent, dropout
        # per-item embedding e_o (EDDI's learned feature id embedding)
        self.item_emb = nn.Parameter(torch.empty(n_items, emb))
        nn.init.xavier_normal_(self.item_emb)
        # phi: point-wise MLP h([e_o, x_o]); x_o=1 constant -> a bias column folds the value in.
        self.phi1 = nn.Linear(emb, feat)
        self.phi2 = nn.Linear(feat, feat)
        # encoder head over the pooled set-code c -> (mu, logvar)
        self.enc1 = nn.Linear(feat, hidden)
        self.enc_mu = nn.Linear(hidden, latent)
        self.enc_lv = nn.Linear(hidden, latent)
        # multinomial decoder (Mult-VAE^PR): latent -> hidden -> n_items
        self.dec1 = nn.Linear(latent, hidden)
        self.dec2 = nn.Linear(hidden, n_items)
        for m in [self.phi1, self.phi2, self.enc1, self.enc_mu, self.enc_lv, self.dec1, self.dec2]:
            nn.init.xavier_normal_(m.weight); nn.init.normal_(m.bias, std=0.001)

    def _phi(self):
        # Phi = phi(E): (n_items x feat). The bias in phi1 carries the constant observed value x_o=1.
        return self.phi2(torch.tanh(self.phi1(self.item_emb)))

    def encode(self, x):
        # x: (batch x n_items) binary. Permutation-invariant pooled code c = X @ Phi (PointNet SUM).
        if self.training and self.dropout > 0:
            x = F.dropout(x, p=self.dropout, training=True)   # denoising dropout on the observed set
        c = x @ self._phi()                                   # (batch x feat) -- sparse gather-sum
        h = torch.tanh(self.enc1(c))
        return self.enc_mu(h), self.enc_lv(h)

    def decode(self, z):
        return self.dec2(torch.tanh(self.dec1(z)))

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            return mu + torch.randn_like(std) * std
        return mu

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


def loss_fn(recon, x, mu, logvar, anneal):
    log_softmax = F.log_softmax(recon, dim=1)
    neg_ll = -torch.mean(torch.sum(log_softmax * x, dim=1))
    kld = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
    return neg_ll + anneal * kld, neg_ll, kld


def make_predict_fn(model):
    def predict(X_csr):
        model.eval()
        with torch.no_grad():
            x = torch.tensor(X_csr.toarray(), dtype=torch.float32)
            recon, _, _ = model(x)
            return recon.cpu().numpy().astype(np.float32)
    return predict


def _defaults():
    return argparse.Namespace(
        epochs=200, batch=500, lr=1e-3, anneal_cap=0.2, total_anneal_steps=200000,
        emb=200, feat=600, hidden=600, latent=200, dropout=0.5,
        patience=15, max_minutes=1e9, resume=True)


def fit(train, n_items, evaluator=None, args=None, ckpt=None, log=print):
    a = args or _defaults()
    model = PartialVAE(n_items, emb=a.emb, feat=a.feat, hidden=a.hidden, latent=a.latent, dropout=a.dropout)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    N = train.shape[0]
    state = {"epoch": 0, "update": 0, "best": -1.0, "best_state": None, "best_epoch": 0, "history": []}
    if ckpt and getattr(a, "resume", True) and os.path.exists(ckpt):
        blob = torch.load(ckpt, map_location="cpu")
        model.load_state_dict(blob["model"]); opt.load_state_dict(blob["opt"]); state = blob["state"]
        log(f"[eddi] RESUMED ep{state['epoch']} best={state['best']:.4f}")
    predict = make_predict_fn(model)
    idxlist = np.arange(N); t0 = time.time()
    for epoch in range(state["epoch"], a.epochs):
        model.train(); np.random.shuffle(idxlist); ep_loss = 0.0; nb = 0
        for st in range(0, N, a.batch):
            X = train[idxlist[st:min(st + a.batch, N)]]
            x = torch.tensor(X.toarray(), dtype=torch.float32)
            anneal = (min(a.anneal_cap, 1.0 * state["update"] / a.total_anneal_steps)
                      if a.total_anneal_steps > 0 else a.anneal_cap)
            opt.zero_grad()
            recon, mu, logvar = model(x)
            loss, _, _ = loss_fn(recon, x, mu, logvar, anneal)
            loss.backward(); opt.step()
            ep_loss += float(loss); nb += 1; state["update"] += 1
        state["epoch"] = epoch + 1
        sc, _ = evaluator(predict) if evaluator is not None else (-1.0, {})
        el = (time.time() - t0) / 60.0
        state["history"].append({"epoch": epoch + 1, "loss": ep_loss / max(nb, 1), "val": sc, "min": round(el, 1)})
        log(f"[eddi] ep{epoch+1:3d} loss {ep_loss/max(nb,1):.4f} anneal {anneal:.3f} val {sc:.4f} {el:.1f}m")
        if sc > state["best"]:
            state["best"] = sc; state["best_epoch"] = epoch + 1
            state["best_state"] = {k: v.clone() for k, v in model.state_dict().items()}
        if ckpt:
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "state": state,
                        "args": vars(a)}, ckpt)
        if evaluator is not None and (epoch + 1) - state["best_epoch"] >= a.patience:
            log(f"[eddi] early stop @ep{epoch+1} best {state['best']:.4f} @ep{state['best_epoch']}")
            break
        if el >= a.max_minutes:
            log(f"[eddi] wall budget {a.max_minutes}m hit @ep{epoch+1}; re-run to resume")
            break
    if state.get("best_state") is not None:
        model.load_state_dict(state["best_state"])
    predict.best_val = state["best"]; predict.best_epoch = state["best_epoch"]
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
    print("[eddi][SMOKE] synthetic tiny data (code-path only, NOT the Liang split)")
    from scipy import sparse
    rng = np.random.RandomState(0)
    n_users, n_items = 120, 130   # >100 items: metrics.evaluate uses NDCG@100 / Recall@50
    X = (sparse.random(n_users, n_items, density=0.2, random_state=rng,
                       data_rvs=lambda s: np.ones(s)) > 0).astype(np.float32).tocsr()
    tr, te_tr = X[:90], X[90:]
    te_te = (sparse.random(30, n_items, density=0.12, random_state=rng,
                           data_rvs=lambda s: np.ones(s)) > 0).astype(np.float32).tocsr()
    a = _defaults(); a.epochs = 3; a.batch = 30; a.emb = 16; a.feat = 24; a.hidden = 24
    a.latent = 8; a.total_anneal_steps = 20; a.patience = 99
    hm = _head_mask(tr, n_items)
    pr = fit(tr, n_items, evaluator=lambda p: (M.evaluate(p, te_tr, te_te, batch_size=15,
                                                          head_mask=hm)["ndcg@10"], {}), args=a, ckpt=None)
    res = M.evaluate(pr, te_tr, te_te, batch_size=15, head_mask=hm)
    print(f"[eddi][SMOKE] OK  full@10={res['ndcg@10']:.4f} tail@10={res['tail_ndcg@10']:.4f} "
          f"best_val={pr.best_val:.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--max_minutes", type=float, default=1e9)
    ap.add_argument("--tag", default="eddi_ml25m_liang")
    ap.add_argument("--out", default=None)
    ap.add_argument("--smoke", action="store_true")
    args0 = ap.parse_args()
    if args0.smoke:
        _smoke(); return
    _HERE = os.path.dirname(os.path.abspath(__file__))
    PROC = os.path.abspath(os.path.join(_HERE, "..", "..", "data", "ml-25m", "proc"))
    a = _defaults(); a.epochs = args0.epochs; a.max_minutes = args0.max_minutes
    meta = M.load_meta(PROC); n_items = meta["n_items"]
    train = M.load_train(n_items, PROC)
    va_tr, va_te = M.load_val(n_items, PROC)
    te_tr, te_te = M.load_test(n_items, PROC)
    hm = _head_mask(train, n_items)
    ckpt = os.path.join(_HERE, "..", "..", ".cache", "baselines", f"{args0.tag}.pt")
    ckpt = os.path.abspath(ckpt); os.makedirs(os.path.dirname(ckpt), exist_ok=True)

    def evaluator(predict):
        m = M.evaluate(predict, va_tr, va_te, batch_size=a.batch, head_mask=hm)
        return m["ndcg@10"], m
    t0 = time.time()
    predict = fit(train, n_items, evaluator=evaluator, args=a, ckpt=ckpt)
    res = M.evaluate(predict, te_tr, te_te, batch_size=a.batch, head_mask=hm)
    res["seconds"] = time.time() - t0; res["best_val"] = predict.best_val; res["best_epoch"] = predict.best_epoch
    print(f"[eddi] test full@10={res['ndcg@10']:.4f} tail@10={res['tail_ndcg@10']:.4f}")
    if args0.out:
        json.dump(res, open(args0.out, "w"), indent=2)


if __name__ == "__main__":
    main()
