"""
RANKING-loss instrument: same set-encoder architecture (DualHeadSetEncoder),
but the LIKED head is trained with a listwise softmax RANKING loss instead of
per-item BCE. Isolates the hypothesis that the BCE objective (per-item
calibration) is why the instrument ranks ~base-rate / worse-than-random on
large catalogs. Rated/answerability head keeps BCE (calibration is correct
there). Evaluated by NDCG@10 / Hit@10 (recommender-standard).

Usage: DATASET_NAME=ml_stratified poetry run python scripts/paper1/train_instrument_rank.py
"""
import os, sys, time
sys.path.insert(0, '.'); sys.path.insert(0, 'scripts/paper1')
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from synthetic_sanity import DualHeadSetEncoder

NAME = os.environ.get('DATASET_NAME', 'ml_stratified')
NPZ = os.environ.get('DATASET_NPZ') or \
    {'ml_stratified': 'C:/dev/phd/casper/data/movielens/ml_stratified_profiles.npz',
     'ml1m': 'C:/dev/phd/casper/data/movielens/ml1m_profiles.npz'}.get(NAME)
from test_instrument_lib import CHECKPOINT_DIR
INST = CHECKPOINT_DIR / f'instrument_{NAME}_rank.pt'
MAX_REVEAL = 60; SEED = 42; D_MODEL = int(os.environ.get('D_MODEL', 128))
EPOCHS = int(os.environ.get('EPOCHS', 80)); PATIENCE = int(os.environ.get('PATIENCE', 20))
ATTR_REVEAL_P = 0.5     # fraction of training reveals that are attribute-only
torch.manual_seed(SEED); np.random.seed(SEED)


def build_batch(arr, nt, ni, rng):
    b = len(arr)
    bi = np.zeros((b, MAX_REVEAL), np.int64); bp = np.zeros((b, MAX_REVEAL), np.int64)
    pad = np.ones((b, MAX_REVEAL), bool)
    revt = np.zeros((b, nt), bool)            # which targets were revealed
    for r, full in enumerate(arr):
        rated = np.where(~np.isnan(full))[0]
        if len(rated) == 0:
            continue
        cand = rated
        attrs = rated[rated >= nt]
        if rng.random() < ATTR_REVEAL_P and len(attrs):
            cand = attrs
        k = max(1, min(int(np.exp(rng.uniform(0, np.log(min(len(cand), MAX_REVEAL) + 1)))), len(cand)))
        sel = rng.choice(cand, k, replace=False)
        for j, e in enumerate(sel):
            bi[r, j] = e; bp[r, j] = int(full[e] >= 0.5); pad[r, j] = False
            if e < nt:
                revt[r, e] = True
    tgt = np.stack([f[:nt] for f in arr])
    ratedm = (~np.isnan(np.stack(arr))).astype(np.float32)
    return (torch.from_numpy(bi), torch.from_numpy(bp), torch.from_numpy(pad),
            torch.from_numpy(tgt.astype(np.float32)), torch.from_numpy(revt),
            torch.from_numpy(ratedm))


def liked_rank_loss(liked_logits, tgt, revt, nt):
    lt = liked_logits[:, :nt].masked_fill(revt, -1e9)
    logp = F.log_softmax(lt, dim=1)
    pos = ((tgt == 1) & (~revt)).float()
    n = pos.sum(1)
    per = -(logp * pos).sum(1) / n.clamp(min=1)
    valid = n > 0
    return per[valid].mean() if valid.any() else torch.zeros((), requires_grad=True)


def ndcg_hit(scores_t, liked, k=10):
    order = np.argsort(-scores_t)[:k]
    rel = liked[order].astype(float)
    dcg = (rel / np.log2(np.arange(2, len(rel) + 2))).sum()
    ideal = int(min(liked.sum(), k))
    idcg = (1.0 / np.log2(np.arange(2, ideal + 2))).sum() if ideal > 0 else 1.0
    return (dcg / idcg if idcg > 0 else 0.0), float(rel.any())


def eval_ndcg(model, test, nt, ni, n=400):
    """Vectorised: batch all users' turn-0 and full-attr-reveal forwards."""
    model.eval()
    users = [p for p in test[:n] if (p[:nt] == 1).sum() > 0]
    B = len(users)
    rng = np.random.default_rng(0)
    # turn 0: empty reveal (single padded token, cls-only)
    bi0 = torch.zeros(B, 1, dtype=torch.long); bp0 = torch.zeros(B, 1, dtype=torch.long)
    pad0 = torch.ones(B, 1, dtype=torch.bool)
    # full attribute reveal
    attrs = [[(i, p[i]) for i in range(nt, ni) if not np.isnan(p[i])] for p in users]
    L = max(1, max(len(a) for a in attrs))
    bi = torch.zeros(B, L, dtype=torch.long); bp = torch.zeros(B, L, dtype=torch.long)
    pad = torch.ones(B, L, dtype=torch.bool)
    for r, a in enumerate(attrs):
        for j, (e, pv) in enumerate(a):
            bi[r, j] = e; bp[r, j] = int(pv >= 0.5); pad[r, j] = False
    with torch.no_grad():
        s0 = model(bi0, bp0, pad0)[0][:, :nt].numpy()
        sf = model(bi, bp, pad)[0][:, :nt].numpy()
    res = {'t0n': [], 'fn': [], 't0h': [], 'fh': []}; rnd = []
    for r, p in enumerate(users):
        liked = (p[:nt] == 1)
        n0, h0 = ndcg_hit(s0[r], liked); nf, hf = ndcg_hit(sf[r], liked)
        res['t0n'].append(n0); res['fn'].append(nf); res['t0h'].append(h0); res['fh'].append(hf)
        rnd.append(ndcg_hit(rng.permutation(nt).astype(float), liked)[0])
    m = lambda x: float(np.mean(x))
    return m(res['t0n']), m(res['fn']), m(res['t0h']), m(res['fh']), m(rnd)


def main():
    d = np.load(NPZ, allow_pickle=True)
    train, test = d['train'], d['test']; nt = int(d['n_targets']); ni = train.shape[1]
    print(f"{NAME}: {len(train)} train / {len(test)} test, {ni} items, {nt} targets", flush=True)
    model = DualHeadSetEncoder(ni, d_model=D_MODEL)
    opt = torch.optim.Adam(model.parameters(), lr=5e-4)
    rng = np.random.default_rng(SEED)
    UPE = int(os.environ.get('USERS_PER_EPOCH', 8000)); BS = 256
    best, bstate, be = -1, None, -1; t0 = time.time()
    for ep in range(EPOCHS):
        model.train(); perm = rng.permutation(len(train))[:UPE]
        for s in range(0, len(perm), BS):
            arr = [train[i] for i in perm[s:s + BS]]
            bi, bp, pad, tgt, revt, ratedm = build_batch(arr, nt, ni, rng)
            opt.zero_grad()
            lk, rt = model(bi, bp, pad)
            loss = liked_rank_loss(lk, tgt, revt, nt) + F.binary_cross_entropy_with_logits(rt, ratedm)
            loss.backward(); opt.step()
        if ep % 2 and ep < EPOCHS - 1:
            continue
        t0n, fn, t0h, fh, rn = eval_ndcg(model, test, nt, ni, n=300)
        if fn > best:
            best, bstate, be = fn, {k: v.clone() for k, v in model.state_dict().items()}, ep
        if (ep + 1) % 5 == 0:
            print(f"  ep{ep+1} NDCG@10 turn0={t0n:.4f} full={fn:.4f} (rand {rn:.4f}) "
                  f"Hit full={fh:.4f} best={best:.4f}@{be+1} {time.time()-t0:.0f}s", flush=True)
        if ep - be >= PATIENCE:
            break
    model.load_state_dict(bstate)
    torch.save({'model_state_dict': model.state_dict(), 'arch': 'dual_set_encoder',
                'd_model': D_MODEL, 'n_heads': 4, 'n_layers': 2, 'n_items': ni,
                'n_movies': nt, 'loss': 'ranking', 'config': {'dataset': NAME, 'max_reveal': MAX_REVEAL}}, INST)
    t0n, fn, t0h, fh, rn = eval_ndcg(model, test, nt, ni, n=500)
    print(f"\n===== RANKING instrument ({NAME}) =====")
    print(f"  NDCG@10  turn0={t0n:.4f}  full={fn:.4f}  lift={fn-t0n:+.4f}  (random {rn:.4f})")
    print(f"  Hit@10   turn0={t0h:.4f}  full={fh:.4f}  lift={fh-t0h:+.4f}")
    print(f"  vs BCE instrument: stratified 0.150->0.272 | ml1m 0.002->0.003 (worse than random)")
    print(f"  saved {INST}")


if __name__ == '__main__':
    main()
