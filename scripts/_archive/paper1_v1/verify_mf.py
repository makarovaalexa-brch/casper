"""
Adversarial verification of the instrument-vs-MF comparison.
(1) unit-test the metric functions on hand-computable inputs;
(2) RATED-NESS ARTIFACT control: rank held-out DISLIKED items vs unrated
    negatives -- a fair recommender ranks them LOW; if the instrument ranks
    them HIGH, its NDCG-on-liked win is inflated by boosting rated items;
(3) WRMF tuning sweep (alpha, d) to ensure the MF baseline is not undertrained.
"""
import os, sys, time
sys.path.insert(0, '.'); sys.path.insert(0, 'scripts/paper1')
import numpy as np
from test_instrument_lib import load_instrument_by_name

NPZ = 'C:/dev/phd/casper/data/movielens/ml_stratified_profiles.npz'
INST = 'instrument_ml_stratified_rankcal'
SEED = 42; N_USERS = int(os.environ.get('N_USERS', 150)); HOLD = 0.40
rng = np.random.default_rng(SEED)
d = np.load(NPZ, allow_pickle=True); train, test = d['train'], d['test']
nt = int(d['n_targets']); ni = train.shape[1]
pop = (~np.isnan(train[:, :nt])).mean(0); p_rated = (~np.isnan(train)).mean(0)
movies = list(range(nt))
LAM = 0.1; D = 24


# ---------- (1) metric unit tests ----------
def ndcg_at(rank, k=10): return 1.0 / np.log2(rank + 1) if rank <= k else 0.0
def auc(scores, labels):
    pos = scores[labels == 1]; neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0: return np.nan
    alls = np.concatenate([pos, neg]); order = alls.argsort()
    ranks = np.empty(len(alls)); ranks[order] = np.arange(1, len(alls) + 1)
    return (ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))

print("== (1) metric unit tests ==")
# perfect ranking: target above all negs -> rank 1 -> NDCG 1.0; AUC 1.0
s = np.array([0.9, 0.1, 0.2, 0.3]); lab = np.array([1, 0, 0, 0])
print(f"  perfect: NDCG@10(rank1)={ndcg_at(1):.3f} (exp 1.000)  AUC={auc(s,lab):.3f} (exp 1.000)")
# worst: target below all -> rank 4 -> NDCG 0 (k=10 still counts: 1/log2(5)=0.43) use k=3
print(f"  rank4@k3 -> {ndcg_at(4,3):.3f} (exp 0.000)   rank2 -> {ndcg_at(2):.3f} (exp 0.631)")
s2 = np.array([0.1, 0.9, 0.8, 0.7]); print(f"  worst-AUC={auc(s2,lab):.3f} (exp 0.000)")
s3 = np.array([0.5, 0.5, 0.5, 0.5]); print(f"  tie-AUC={auc(s3,lab):.3f} (exp 0.500)")


# ---------- build cases ----------
cases = []
for prof in test:
    rated = np.where(~np.isnan(prof[:nt]))[0]
    if len(rated) < 8: continue
    liked = rated[prof[rated] == 1]; disl = rated[prof[rated] == 0]
    if len(liked) < 4 or len(disl) < 3: continue
    perm = rng.permutation(rated); k = max(2, int(HOLD * len(rated)))
    H = perm[:k]; askable = set(int(x) for x in perm[k:])
    Hl = [int(x) for x in H if prof[x] == 1]; Hd = [int(x) for x in H if prof[x] == 0]
    if len(Hl) < 1 or len(Hd) < 1: continue
    cases.append((prof, askable, Hl, Hd))
    if len(cases) >= N_USERS: break
print(f"\nusers={len(cases)} (each has held-out liked AND disliked)")


# ---------- WRMF ----------
class WRMF:
    def __init__(s, Q, alpha): s.Q = Q; s.QtQ = Q.T @ Q; s.alpha = alpha; s.n_movies = nt
    def _f(s, rev):
        dd = s.Q.shape[1]; A = s.QtQ + LAM*np.eye(dd); rhs = np.zeros(dd)
        for e, p in rev:
            q = s.Q[e]; A = A + s.alpha*np.outer(q, q)
            if p >= 0.5: rhs = rhs + (1+s.alpha)*q
        return np.linalg.solve(A, rhs)
    def predict_full(s, rev): return s.Q @ s._f(rev)

def train_wrmf(alpha, dd, iters, nuser):
    sub = rng.choice(len(train), min(nuser, len(train)), replace=False); R = train[sub]; nu = len(R)
    pos = (R == 1); obs = ~np.isnan(R)
    P = 0.01*rng.standard_normal((nu, dd)); Q = 0.01*rng.standard_normal((ni, dd))
    uo = [np.where(obs[u])[0] for u in range(nu)]; up = [np.where(pos[u])[0] for u in range(nu)]
    io = [np.where(obs[:, i])[0] for i in range(ni)]; ip = [np.where(pos[:, i])[0] for i in range(ni)]
    Id = LAM*np.eye(dd)
    for it in range(iters):
        QtQ = Q.T @ Q
        for u in range(nu):
            A = QtQ + Id + alpha*(Q[uo[u]].T@Q[uo[u]]); rhs = (1+alpha)*Q[up[u]].sum(0) if len(up[u]) else np.zeros(dd)
            P[u] = np.linalg.solve(A, rhs)
        PtP = P.T @ P
        for i in range(ni):
            A = PtP + Id + alpha*(P[io[i]].T@P[io[i]]); rhs = (1+alpha)*P[ip[i]].sum(0) if len(ip[i]) else np.zeros(dd)
            Q[i] = np.linalg.solve(A, rhs)
    return WRMF(Q, alpha)


# ---------- (2)+(3) eval: liked-NDCG AND disliked-in-top10 (artifact) ----------
def evaluate(rec, name):
    liked_ndcg = []; dis_top10 = []; dis_pct = []; liked_top10 = []
    for (prof, askable, Hl, Hd) in cases:
        rated_set = set(int(x) for x in np.where(~np.isnan(prof[:nt]))[0])
        rev = [(int(e), float(prof[e])) for e in askable if e < nt]
        bel = np.asarray(rec.predict_full(rev))[:nt]
        negs = np.array([m for m in movies if m not in rated_set])
        for h in Hl:
            r = 1 + int((bel[negs] >= bel[h]).sum())
            liked_ndcg.append(ndcg_at(r)); liked_top10.append(1.0 if r <= 10 else 0.0)
        for hd in Hd:                                  # CONTROL: should rank LOW
            r = 1 + int((bel[negs] >= bel[hd]).sum())
            dis_top10.append(1.0 if r <= 10 else 0.0); dis_pct.append(r / (len(negs)+1))
    print(f"  {name:<26} liked: NDCG@10={np.mean(liked_ndcg):.4f} top10={np.mean(liked_top10):.3f} | "
          f"DISLIKED: top10={np.mean(dis_top10):.3f} (want LOW) meanPct={np.mean(dis_pct):.3f} (want HIGH)")

print("\n== (3) WRMF tuning sweep ==")
configs = [(40, 24, 12, 12000), (100, 32, 15, 15000), (200, 48, 18, 20000)]
wrmfs = []
for (a, dd, it, nu) in configs:
    t0 = time.time(); w = train_wrmf(a, dd, it, nu)
    wrmfs.append((f"WRMF a{a} d{dd}", w)); print(f"  trained WRMF alpha={a} d={dd} iters={it} ({time.time()-t0:.0f}s)", flush=True)

print("\n== (2) artifact control + ranking (held-out LIKED vs DISLIKED, both vs unrated) ==")
print("  fair recommender: liked top10 HIGH, disliked top10 LOW, disliked meanPct HIGH (~>0.5)")
inst, _ = load_instrument_by_name(INST)
evaluate(inst, "Learned instrument")
for nm, w in wrmfs:
    evaluate(w, nm)
# popularity reference
class Pop:
    n_movies = nt
    def predict_full(s, rev): return p_rated.copy()
evaluate(Pop(), "popularity-rank")
