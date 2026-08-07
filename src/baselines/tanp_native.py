r"""tanp_native.py -- TaNP as PUBLISHED: episodic meta-learning with RATING REGRESSION.

WHY A SEPARATE MODULE. `tanp_bestefffort.py` is a documented *implicit-feedback adaptation* -- its own
header says "Implicit binary feedback (rating := 1 for every observed like) instead of explicit 1-5
ratings ... the rating channel is therefore constant; the signal is the item set". It trains a softmax
cross-entropy over a multi-hot target. That is a legitimate row, and it is the one the LIANG column
reports. It is NOT TaNP as Lin et al. built it, and handing it a graded matrix does not make it so: its
episode builder overwrites every value with 1.0, so a "graded" run trained on binary supports over an
all-bands item set -- neither the paper's method nor our stated adaptation. That produced the void arm of
2026-08-06 and is why this module exists rather than a flag.

WHAT IS FAITHFUL HERE (Lin et al., TaNP, WWW 2021):
  * an episode is a user; the support set is observed (item, RATING) pairs, the target set is held-out
    (item, RATING) pairs;
  * the encoder reads support VALUES, not just membership;
  * the decoder predicts a RATING for each target item;
  * the loss is squared error on the target ratings, plus the KL term of the latent.
Only OBSERVED target entries contribute to the loss -- an unrated item is missing data, not a zero. This
is the single most important line in the file: an unmasked MSE over all 18,359 items would train the
model to predict 0 everywhere and is the classic way to get a degenerate rating regressor.

HOW IT IS SCORED, and why this is a fair comparison rather than an objective mismatch. The model is
trained to predict ratings, as published. It is then scored the way EVERY row on this ruler is scored:
rank the full catalogue by the model's own output and take NDCG@10 against the held-out LIKES. Ranking by
predicted rating is the natural read-out of a rating regressor. If a rating-trained model ranks worse than
an implicit-trained one, that is a finding about rating prediction versus top-N (Cremonesi et al. 2010),
not a defect of the harness -- it is precisely what running each system on its own contract is for.

  python src/baselines/tanp_native.py --smoke
"""
import os
import time
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _defaults():
    # beta RESCALED for the change of reconstruction objective, 2026-08-07. The implicit row's softmax
    # NLL runs at ~600; this module's masked MSE runs at ~0.88 -- a factor of ~680. Carrying beta=0.2
    # across unchanged made the KL term dominate the loss, the encoder collapsed to the prior, and the
    # decoder emitted item bias for everyone: 39 distinct top-10 lists across 1,000 users, predictions
    # spanning only 2.45..3.92 over the whole catalogue, val RMSE a healthy 0.845 (item means alone
    # predict ratings well) and NDCG 0.0146, an order of magnitude BELOW the popularity floor.
    # beta is therefore set to preserve the SAME reconstruction-to-KL ratio as the implicit row,
    # 0.2 * (0.88/600) ~ 3e-4, rather than being re-guessed -- the training contract is the only thing
    # this module is supposed to vary.
    return argparse.Namespace(
        d=200, hidden=512, zdim=64, dropout=0.5, target_prop=0.5, beta=3e-4,
        lr=1e-3, batch=500, l2=0.0, epochs=200, patience=15, max_minutes=1e9, resume=True)


class TaNPNative(nn.Module):
    """Same capacity as the best-effort core; the difference is what the head predicts."""

    def __init__(self, n_items, d, hidden, zdim, dropout):
        super().__init__()
        self.n_items, self.d, self.zdim = n_items, d, zdim
        self.item_emb = nn.Embedding(n_items, d)
        nn.init.normal_(self.item_emb.weight, std=0.01)
        self.item_bias = nn.Parameter(torch.zeros(n_items))
        self.global_bias = nn.Parameter(torch.zeros(1))
        # encoder reads the rating-weighted context AND the user's mean rating
        self.enc = nn.Sequential(nn.Linear(d + 1, hidden), nn.ReLU(), nn.Dropout(dropout))
        self.enc_mu = nn.Linear(hidden, zdim)
        self.enc_lv = nn.Linear(hidden, zdim)
        self.dec = nn.Sequential(nn.Linear(zdim, hidden), nn.ReLU(), nn.Dropout(dropout),
                                 nn.Linear(hidden, d))

    def encode(self, support_vals, support_mask):
        cnt = support_mask.sum(1, keepdim=True).clamp(min=1.0)
        ctx = (support_vals @ self.item_emb.weight) / cnt      # rating-weighted mean embedding
        mean_r = support_vals.sum(1, keepdim=True) / cnt        # the user's observed mean rating
        h = self.enc(torch.cat([ctx, mean_r], dim=1))
        return self.enc_mu(h), self.enc_lv(h)

    def reparam(self, mu, lv):
        return mu + torch.randn_like(mu) * torch.exp(0.5 * lv) if self.training else mu

    def decode(self, z):
        u = self.dec(z)
        return u @ self.item_emb.weight.T + self.item_bias + self.global_bias   # predicted RATING

    def forward(self, support_vals, support_mask):
        mu, lv = self.encode(support_vals, support_mask)
        return self.decode(self.reparam(mu, lv)), mu, lv


def loss_fn(pred, target_vals, target_mask, mu, lv, beta):
    """MASKED squared error on observed target ratings + KL. Unrated items are MISSING, not zero."""
    se = ((pred - target_vals) ** 2) * target_mask
    n_obs = target_mask.sum().clamp(min=1.0)
    mse = se.sum() / n_obs
    kld = -0.5 * torch.mean(torch.sum(1 + lv - mu.pow(2) - lv.exp(), dim=1))
    return mse + beta * kld, mse, kld


def _episode(Xbatch, target_prop, rng):
    """Split each user's OBSERVED (item, rating) pairs into support and target, keeping the VALUES."""
    B, n = Xbatch.shape
    sv = np.zeros((B, n), dtype=np.float32); sm = np.zeros((B, n), dtype=np.float32)
    tv = np.zeros((B, n), dtype=np.float32); tm = np.zeros((B, n), dtype=np.float32)
    Xc = Xbatch.tocsr()
    for r in range(B):
        lo, hi = Xc.indptr[r], Xc.indptr[r + 1]
        cols, vals = Xc.indices[lo:hi], Xc.data[lo:hi]
        if len(cols) == 0:
            continue
        if len(cols) == 1:
            sv[r, cols] = vals; sm[r, cols] = 1.0; continue
        nt = max(1, int(target_prop * len(cols)))
        perm = rng.permutation(len(cols))
        ti, si = perm[:nt], perm[nt:]
        tv[r, cols[ti]] = vals[ti]; tm[r, cols[ti]] = 1.0
        sv[r, cols[si]] = vals[si]; sm[r, cols[si]] = 1.0
    return (torch.from_numpy(sv), torch.from_numpy(sm),
            torch.from_numpy(tv), torch.from_numpy(tm))


def _make_predict(model):
    def predict(X_csr):
        model.eval()
        with torch.no_grad():
            sv = torch.tensor(X_csr.toarray(), dtype=torch.float32)
            sm = (sv != 0).float()
            mu, _ = model.encode(sv, sm)
            return model.decode(mu).cpu().numpy().astype(np.float32)   # predicted ratings = the score
    return predict


def fit(train, n_items, evaluator=None, args=None, ckpt=None, log=print):
    a = args or _defaults()
    model = TaNPNative(n_items, a.d, a.hidden, a.zdim, a.dropout)
    # Start the decoder ON THE RATING SCALE. Zero-init biases mean the model opens predicting ~0 against
    # targets averaging ~3.5, so the first epochs are spent climbing to the intercept rather than learning
    # taste -- standard practice in rating factorisation is to seed the global bias at the global mean.
    # Inert for ranking (a constant added to every item), so this buys optimisation, not accuracy.
    with torch.no_grad():
        model.global_bias.fill_(float(train.data.mean()) if train.nnz else 0.0)
    log(f"[tanp-native] global bias initialised at the training mean rating "
        f"{float(train.data.mean()) if train.nnz else 0.0:.3f}")
    opt = torch.optim.Adam(model.parameters(), lr=a.lr, weight_decay=a.l2)
    state = {"epoch": 0, "best": -1.0, "best_state": None, "best_epoch": 0, "history": []}
    if ckpt and getattr(a, "resume", True) and os.path.exists(ckpt):
        blob = torch.load(ckpt, map_location="cpu")
        model.load_state_dict(blob["model"]); opt.load_state_dict(blob["opt"]); state = blob["state"]
        log(f"[tanp-native] RESUMED ep{state['epoch']} best={state['best']:.4f}")
    predict = _make_predict(model)
    rng = np.random.RandomState(98765)
    N = train.shape[0]; idx = np.arange(N); t0 = time.time()

    for epoch in range(state["epoch"], a.epochs):
        model.train(); rng.shuffle(idx); ep = 0.0; nb = 0
        for st in range(0, N, a.batch):
            Xb = train[idx[st:st + a.batch]]
            sv, sm, tv, tm = _episode(Xb, a.target_prop, rng)
            if tm.sum() == 0:
                continue
            opt.zero_grad()
            pred, mu, lv = model(sv, sm)
            loss, _, _ = loss_fn(pred, tv, tm, mu, lv, a.beta)
            loss.backward(); opt.step()
            ep += float(loss); nb += 1
        el = (time.time() - t0) / 60.0
        sc, _ = evaluator(predict) if evaluator is not None else (-1.0, {})
        state["epoch"] = epoch + 1
        state["history"].append({"epoch": epoch + 1, "loss": ep / max(nb, 1), "val": sc})
        log(f"[tanp-native] ep{epoch+1:3d} loss {ep/max(nb,1):.4f} val {sc:.4f} {el:.1f}m")
        if sc > state["best"]:
            state["best"] = sc; state["best_epoch"] = epoch + 1
            state["best_state"] = {k: v.clone() for k, v in model.state_dict().items()}
            if ckpt:
                torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "state": state,
                            "args": vars(a)}, ckpt)
        if evaluator is not None and (epoch + 1) - state["best_epoch"] >= a.patience:
            log(f"[tanp-native] early stop @ep{epoch+1} best {state['best']:.4f} @ep{state['best_epoch']}")
            break
        if el > a.max_minutes:
            log(f"[tanp-native] time cap {a.max_minutes}m"); break

    if state.get("best_state") is not None:
        model.load_state_dict(state["best_state"])
    predict.best_val, predict.best_epoch = state["best"], state["best_epoch"]
    predict.hp = dict(d=a.d, hidden=a.hidden, zdim=a.zdim, beta=a.beta, target_prop=a.target_prop,
                      best_val=state["best"], best_epoch=state["best_epoch"],
                      role="TaNP as published: episodic meta-learning, RATING REGRESSION")
    return predict


def _smoke():
    from scipy import sparse
    rng = np.random.RandomState(0)
    n_users, n_items = 200, 60
    X = sparse.random(n_users, n_items, density=0.2, random_state=rng, format="csr")
    X.data = rng.choice([0.5, 1.5, 2.5, 3.5, 4.5, 5.0], size=X.nnz).astype(np.float32)
    a = _defaults(); a.d = 8; a.hidden = 16; a.zdim = 4; a.epochs = 3; a.batch = 50; a.patience = 99
    pr = fit(X, n_items, evaluator=None, args=a, ckpt=None)
    s = pr(X[:5])
    print("[tanp-native][SMOKE] scores", s.shape, "range %.2f..%.2f" % (s.min(), s.max()))
    print("[tanp-native][SMOKE] predictions land in a rating-like range:",
          bool(0.0 < np.median(s) < 6.0))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--smoke", action="store_true")
    if ap.parse_args().smoke:
        _smoke()
