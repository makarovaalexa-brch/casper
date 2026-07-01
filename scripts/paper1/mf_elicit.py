"""
STEP 2 — established cold-start elicitation curve on ML-100k (Golbandi protocol).

Train biased MF on WARM users (item factors Q, biases). Hold out COLD users.
For each cold user: interview from an answerable pool (top-N most-rated movies),
reveal true ratings, ridge fold-in -> (b_u, u), predict held-out ratings.
Report held-out RMSE vs number of questions for:
  random  /  most-popular  /  adaptive (D-optimal variance reduction)
Gate: reproduce the known ordering adaptive <= popular <= random.

Recommender held FIXED (same MF fold-in) across strategies, so only the
question-selection varies (testbed isolation). No bespoke slate, explicit
ratings, RMSE -- all established.
"""
import os, sys, time
import numpy as np

base = 'C:/dev/phd/casper/data/movielens/ml-100k'
SEED = 42; D = int(os.environ.get('D', 32)); EPOCHS = int(os.environ.get('EPOCHS', 30))
LR = 0.008; REG = 0.05; REGF = float(os.environ.get('REGF', 8.0))   # fold-in ridge
N_COLD = int(os.environ.get('N_COLD', 200)); POOL = int(os.environ.get('POOL', 300))
T = int(os.environ.get('T', 20)); BS = 20000
rng = np.random.default_rng(SEED)

# ---- load all ratings ----
U, I, R = [], [], []
with open(f'{base}/u.data') as f:
    for line in f:
        a = line.strip().split('\t'); U.append(int(a[0])); I.append(int(a[1])); R.append(float(a[2]))
U = np.array(U); I = np.array(I); R = np.array(R, np.float32)
uids = {u: k for k, u in enumerate(np.unique(U))}; iids = {i: k for k, i in enumerate(np.unique(I))}
u = np.array([uids[x] for x in U]); i = np.array([iids[x] for x in I])
nu, ni = len(uids), len(iids)

# ---- warm / cold USER split ----
cold_users = set(rng.choice(nu, N_COLD, replace=False).tolist())
warm = np.array([k for k in range(len(u)) if u[k] not in cold_users])
ut, it, rt = u[warm], i[warm], R[warm]
mu = float(rt.mean())
print(f"ML-100k: {nu} users ({nu-N_COLD} warm / {N_COLD} cold), {ni} movies; warm ratings {len(rt)}", flush=True)

# ---- train MF on warm users ----
bu = np.zeros(nu); bi = np.zeros(ni)
P = 0.1 * rng.standard_normal((nu, D)); Q = 0.1 * rng.standard_normal((ni, D))
for ep in range(EPOCHS):
    order = rng.permutation(len(ut)); lr = LR / (1 + 0.05 * ep)
    for s in range(0, len(order), BS):
        b = order[s:s + BS]; uu, ii, rr = ut[b], it[b], rt[b]
        e = (rr - (mu + bu[uu] + bi[ii] + np.sum(P[uu] * Q[ii], 1))).astype(np.float64)
        np.add.at(bu, uu, lr * (e - REG * bu[uu])); np.add.at(bi, ii, lr * (e - REG * bi[ii]))
        np.add.at(P, uu, lr * (e[:, None] * Q[ii] - REG * P[uu]))
        np.add.at(Q, ii, lr * (e[:, None] * P[uu] - REG * Q[ii]))
print(f"MF trained (warm). item-factor norm {np.linalg.norm(Q,axis=1).mean():.3f}", flush=True)

# ask space = FULL catalogue (so answer-rate distinguishes strategies)
cnt = np.bincount(it, minlength=ni)
pop_order = list(np.argsort(-cnt))                       # popularity over all items
# HELF-style: log-popularity x rating-spread (answerable AND informative)
mean_r = np.array([rt[it == j].mean() if cnt[j] > 0 else mu for j in range(ni)])
var_r = np.array([rt[it == j].var() if cnt[j] > 1 else 0.0 for j in range(ni)])
lf = np.log1p(cnt) / np.log1p(cnt.max() + 1e-9)
helf = 2 * lf * var_r / (lf + var_r + 1e-9)
helf_order = list(np.argsort(-helf))

# cold-user data: map item->rating
cold_ratings = {}
for k in range(len(u)):
    if u[k] in cold_users:
        cold_ratings.setdefault(u[k], {})[i[k]] = R[k]

# ---- fold-in: given answered [(item,rating)], solve [b_u, u] ridge ----
def foldin(ans):
    if not ans: return 0.0, np.zeros(D)
    A = np.array([[1.0] + Q[it_].tolist() for it_, _ in ans])
    y = np.array([r - mu - bi[it_] for it_, r in ans])
    M = A.T @ A + REGF * np.eye(D + 1)
    x = np.linalg.solve(M, A.T @ y)
    return float(x[0]), x[1:]

def predict(bu_, u_, items):
    items = np.array(items)
    return np.clip(mu + bu_ + bi[items] + Q[items] @ u_, 1, 5)

def rmse(bu_, u_, held):
    if not held: return np.nan
    items = [h for h, _ in held]; truth = np.array([r for _, r in held])
    return float(np.sqrt(np.mean((truth - predict(bu_, u_, items)) ** 2)))

def run_static(order, ratings, held, forbid):
    ans = []; curve = []; t = 0; idx = 0
    bu_, u_ = foldin(ans); curve.append(rmse(bu_, u_, held))
    while t < T:
        # advance to next askable item (not forbidden/test)
        while idx < len(order) and order[idx] in forbid: idx += 1
        q = order[idx] if idx < len(order) else None; idx += 1
        if q is not None and q in ratings: ans.append((q, ratings[q]))
        bu_, u_ = foldin(ans); curve.append(rmse(bu_, u_, held)); t += 1
    return curve

def run_adaptive(ratings, held, forbid):
    ans = []; Sigma = np.eye(D + 1) / REGF; asked = set(forbid); curve = []
    bu_, u_ = foldin(ans); curve.append(rmse(bu_, u_, held))
    A = np.concatenate([np.ones((ni, 1)), Q], 1)        # [ni, D+1] augmented item rows
    for _ in range(T):
        v = np.einsum('ij,jk,ik->i', A, Sigma, A)        # predictive variance per item
        v[list(asked)] = -1
        best = int(np.argmax(v)); asked.add(best)
        a = A[best]; Sigma = Sigma - np.outer(Sigma @ a, a @ Sigma) / (1.0 + a @ Sigma @ a)
        if best in ratings: ans.append((best, ratings[best]))
        bu_, u_ = foldin(ans); curve.append(rmse(bu_, u_, held))
    return curve

# ---- evaluate: random held-out test split; ask from FULL catalogue ----
curves = {'random': [], 'popular': [], 'HELF': [], 'adaptive': []}; used = 0
for cu, ratings in cold_ratings.items():
    items_u = list(ratings.keys())
    if len(items_u) < 10: continue
    rng.shuffle(items_u); ncut = max(5, int(0.3 * len(items_u)))
    test_items = set(items_u[:ncut]); forbid = test_items                 # never ask test items
    held = [(it_, ratings[it_]) for it_ in test_items]
    randord = pop_order[:]; rng.shuffle(randord)
    curves['random'].append(run_static(randord, ratings, held, forbid))
    curves['popular'].append(run_static(pop_order, ratings, held, forbid))
    curves['HELF'].append(run_static(helf_order, ratings, held, forbid))
    curves['adaptive'].append(run_adaptive(ratings, held, forbid))
    used += 1
print(f"evaluated {used} cold users (random 30% held-out; ask full catalogue)\n", flush=True)

arr = {k: np.nanmean(np.array(v), 0) for k, v in curves.items()}
print(f"{'#q':>3}  {'random':>8} {'popular':>8} {'HELF':>8} {'adaptive':>8}")
for t in [0, 1, 2, 3, 5, 10, 15, 20]:
    print(f"{t:>3}  {arr['random'][t]:>8.4f} {arr['popular'][t]:>8.4f} {arr['HELF'][t]:>8.4f} {arr['adaptive'][t]:>8.4f}")
for k in curves:
    print(f"  {k:>9}: RMSE {arr[k][0]:.4f} -> {arr[k][20]:.4f}  (drop {arr[k][0]-arr[k][20]:+.4f})")
print("(want big drops; answerable strategies (popular/HELF) beat random; warm-MF RMSE ~0.94 is the floor)")
