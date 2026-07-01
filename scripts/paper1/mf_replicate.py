"""
STEP 1 — replicate textbook biased-MF RMSE on MovieLens-1M.

Standard biased matrix factorization (Koren 2009): r_ui = mu + b_u + b_i + p_u . q_i,
SGD with L2 reg, random 90/10 split. Published ballpark: RMSE ~ 0.86-0.88.
If we don't land there, the pipeline is wrong -- fix that before anything else.
No bespoke slates, no binarization, no custom metric: explicit ratings, RMSE.
"""
import os, sys, time
import numpy as np

DS = os.environ.get('DS', 'ml100k')
SEED = 42; D = int(os.environ.get('D', 32)); EPOCHS = int(os.environ.get('EPOCHS', 30))
LR = float(os.environ.get('LR', 0.008)); REG = float(os.environ.get('REG', 0.05))
BS = 20000
rng = np.random.default_rng(SEED)

# ---- load (dataset-aware; ml100k uses canonical u1 split) ----
def parse(path, sep):
    U, I, R = [], [], []
    with open(path) as f:
        for line in f:
            a = line.strip().split(sep)
            if len(a) < 3: continue
            U.append(int(a[0])); I.append(int(a[1])); R.append(float(a[2]))
    return np.array(U), np.array(I), np.array(R, np.float32)

base = 'C:/dev/phd/casper/data/movielens'
if DS == 'ml100k':
    ub, ib, rb = parse(f'{base}/ml-100k/u1.base', '\t')
    uvr, ivr, rvr = parse(f'{base}/ml-100k/u1.test', '\t')
    allu = np.concatenate([ub, uvr]); alli = np.concatenate([ib, ivr])
    uids = {u: k for k, u in enumerate(np.unique(allu))}; iids = {i: k for k, i in enumerate(np.unique(alli))}
    ut = np.array([uids[x] for x in ub]); it = np.array([iids[x] for x in ib]); rt = rb
    uv = np.array([uids[x] for x in uvr]); iv = np.array([iids[x] for x in ivr]); rv = rvr
    nu, ni = len(uids), len(iids)
    print(f"ML-100k (canonical u1 split): {len(rt)} train / {len(rv)} test, {nu} users, {ni} movies", flush=True)
else:
    u_raw, i_raw, r = parse(f'{base}/ml-1m/ratings.dat', '::')
    uids = {u: k for k, u in enumerate(np.unique(u_raw))}; iids = {i: k for k, i in enumerate(np.unique(i_raw))}
    u = np.array([uids[x] for x in u_raw]); i = np.array([iids[x] for x in i_raw])
    nu, ni = len(uids), len(iids)
    perm = rng.permutation(len(r)); ntest = len(r) // 10
    te, tr = perm[:ntest], perm[ntest:]
    ut, it, rt = u[tr], i[tr], r[tr]; uv, iv, rv = u[te], i[te], r[te]
    print(f"ML-1M (random 90/10): {len(rt)} train / {len(rv)} test, {nu} users, {ni} movies", flush=True)
mu = float(rt.mean())

# ---- init ----
bu = np.zeros(nu); bi = np.zeros(ni)
P = 0.1 * rng.standard_normal((nu, D)).astype(np.float64)
Q = 0.1 * rng.standard_normal((ni, D)).astype(np.float64)

def rmse(uu, ii, rr):
    pred = mu + bu[uu] + bi[ii] + np.sum(P[uu] * Q[ii], axis=1)
    pred = np.clip(pred, 1.0, 5.0)
    return float(np.sqrt(np.mean((rr - pred) ** 2)))

print("baseline (predict global mean) test RMSE:", round(float(np.sqrt(np.mean((rv - mu) ** 2))), 4), flush=True)
t0 = time.time()
for ep in range(EPOCHS):
    order = rng.permutation(len(rt)); lr = LR / (1 + 0.05 * ep)
    for s in range(0, len(order), BS):
        b = order[s:s + BS]; uu = ut[b]; ii = it[b]; rr = rt[b]
        pred = mu + bu[uu] + bi[ii] + np.sum(P[uu] * Q[ii], axis=1)
        e = (rr - pred).astype(np.float64)
        # scatter-add (handle duplicate u/i within batch)
        np.add.at(bu, uu, lr * (e - REG * bu[uu]))
        np.add.at(bi, ii, lr * (e - REG * bi[ii]))
        gP = e[:, None] * Q[ii] - REG * P[uu]
        gQ = e[:, None] * P[uu] - REG * Q[ii]
        np.add.at(P, uu, lr * gP)
        np.add.at(Q, ii, lr * gQ)
    if (ep + 1) % 5 == 0 or ep == 0:
        print(f"  ep{ep+1}/{EPOCHS} train RMSE={rmse(ut,it,rt):.4f}  test RMSE={rmse(uv,iv,rv):.4f}  ({time.time()-t0:.0f}s)", flush=True)

print(f"\nFINAL test RMSE = {rmse(uv,iv,rv):.4f}  (published biased-MF on ML-1M ~ 0.86-0.88)", flush=True)
