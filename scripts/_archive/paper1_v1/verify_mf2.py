"""
Artifact-FREE head-to-head with tie-correct AUC: rank held-out LIKED above
held-out DISLIKED (both rated -> no rated-ness confound). Also a balanced
ranking: liked target vs (disliked + equal#unrated) so rated-ness cancels.
"""
import os, sys, time
sys.path.insert(0, '.'); sys.path.insert(0, 'scripts/paper1')
import numpy as np
from test_instrument_lib import load_instrument_by_name

NPZ = 'C:/dev/phd/casper/data/movielens/ml_stratified_profiles.npz'
INST = 'instrument_ml_stratified_rankcal'
SEED = 42; N_USERS = int(os.environ.get('N_USERS', 150)); HOLD = 0.40; LAM = 0.1
rng = np.random.default_rng(SEED)
d = np.load(NPZ, allow_pickle=True); train, test = d['train'], d['test']
nt = int(d['n_targets']); ni = train.shape[1]
movies = list(range(nt))


def auc(scores, labels):                       # tie-correct (average ranks)
    pos = scores[labels == 1]; neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0: return np.nan
    alls = np.concatenate([pos, neg]); order = np.argsort(alls, kind='mergesort')
    sa = alls[order]; ranks = np.empty(len(alls))
    i = 0
    while i < len(alls):                        # average ranks within ties
        j = i
        while j + 1 < len(alls) and sa[j + 1] == sa[i]: j += 1
        ranks[order[i:j + 1]] = (i + 1 + j + 1) / 2.0; i = j + 1
    return (ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))

# unit re-test
print("tie-AUC:", auc(np.array([.5, .5, .5, .5]), np.array([1, 0, 0, 0])), "(exp 0.5)")
print("perfect:", auc(np.array([.9, .1, .2]), np.array([1, 0, 0])), "(exp 1.0)")

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
    cases.append((prof, askable, Hl, Hd));
    if len(cases) >= N_USERS: break
print(f"users={len(cases)}")


class WRMF:
    def __init__(s, Q, alpha): s.Q = Q; s.QtQ = Q.T @ Q; s.alpha = alpha
    def predict_full(s, rev):
        dd = s.Q.shape[1]; A = s.QtQ + LAM*np.eye(dd); rhs = np.zeros(dd)
        for e, p in rev:
            q = s.Q[e]; A = A + s.alpha*np.outer(q, q)
            if p >= 0.5: rhs = rhs + (1+s.alpha)*q
        return s.Q @ np.linalg.solve(A, rhs)
def train_wrmf(alpha, dd, iters, nuser):
    sub = rng.choice(len(train), min(nuser, len(train)), replace=False); R = train[sub]; nu = len(R)
    pos = (R == 1); obs = ~np.isnan(R)
    P = .01*rng.standard_normal((nu, dd)); Q = .01*rng.standard_normal((ni, dd))
    uo=[np.where(obs[u])[0] for u in range(nu)]; up=[np.where(pos[u])[0] for u in range(nu)]
    io=[np.where(obs[:,i])[0] for i in range(ni)]; ip=[np.where(pos[:,i])[0] for i in range(ni)]
    Id = LAM*np.eye(dd)
    for it in range(iters):
        QtQ = Q.T@Q
        for u in range(nu):
            A=QtQ+Id+alpha*(Q[uo[u]].T@Q[uo[u]]); P[u]=np.linalg.solve(A,(1+alpha)*Q[up[u]].sum(0) if len(up[u]) else np.zeros(dd))
        PtP=P.T@P
        for i in range(ni):
            A=PtP+Id+alpha*(P[io[i]].T@P[io[i]]); Q[i]=np.linalg.solve(A,(1+alpha)*P[ip[i]].sum(0) if len(ip[i]) else np.zeros(dd))
    return WRMF(Q, alpha)


def evalrec(rec, name):
    ld_auc = []          # liked-vs-disliked AUC (artifact-free)
    bal_ndcg = []        # liked vs (disliked + equal# unrated): rated-ness cancels
    for (prof, askable, Hl, Hd) in cases:
        rated_set = set(int(x) for x in np.where(~np.isnan(prof[:nt]))[0])
        rev = [(int(e), float(prof[e])) for e in askable if e < nt]
        bel = np.asarray(rec.predict_full(rev))[:nt]
        # liked vs disliked (held-out, both rated)
        sc = np.array([bel[i] for i in Hl] + [bel[i] for i in Hd])
        lab = np.array([1]*len(Hl) + [0]*len(Hd))
        a = auc(sc, lab)
        if not np.isnan(a): ld_auc.append(a)
        # balanced ranking: each liked target vs (all held-out disliked + equal # unrated)
        unrated = np.array([m for m in movies if m not in rated_set])
        for h in Hl:
            nud = list(Hd) + list(rng.choice(unrated, min(len(Hd), len(unrated)), replace=False))
            negsc = bel[np.array(nud)]
            r = 1 + int((negsc >= bel[h]).sum())
            bal_ndcg.append(1.0/np.log2(r+1) if r <= 10 else 0.0)
    print(f"  {name:<22} liked-vs-disliked AUC={np.mean(ld_auc):.4f}  balanced-NDCG@10={np.mean(bal_ndcg):.4f}")

print("\n== artifact-free comparison (tie-correct) ==")
inst, _ = load_instrument_by_name(INST)
evalrec(inst, "Learned instrument")
for a, dd, it, nu in [(40, 24, 12, 12000), (100, 32, 15, 15000)]:
    t0 = time.time(); w = train_wrmf(a, dd, it, nu)
    evalrec(w, f"WRMF a{a} d{dd}")
