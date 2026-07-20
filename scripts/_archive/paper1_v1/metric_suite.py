"""
METRIC SUITE for the CASPER testbed (no training; re-scores saved policies).

Fixes the coarse single-target Hit@10:
  * FOLDS: hold out 40% of each user's rated movies (reserved set H, forbidden to
    ask). Every LIKED item in H is a retrieval target -> many measurements/user.
  * PROPER SPLIT: policy asks only from the other 60% + unrated -> we measure
    generalization to unasked preferences, not memorization of reveals.
  * TWO metric families, per turn -> area-under-curve + t0/t5/t15:
      RECONSTRUCTION over H (calibrated belief vs truth): Brier v, NLL v, ROC-AUC ^
      RETRIEVAL full-slate (rank each liked target vs all non-interacted movies):
          NDCG@10 ^, MRR ^, Hit@10 ^ (reference)
  * Ceiling = FULL-REVEAL of all askable rated movies (upper bound for this split).

All policies movies-only (deployed setting; attributes hurt the student).
"""
import os, sys, time
sys.path.insert(0, '.'); sys.path.insert(0, 'scripts/paper1'); sys.path.insert(0, 'scripts/paper2')
import numpy as np, torch
from test_instrument_lib import load_instrument_by_name
from equivariant_actor import EquivariantActor

INST = 'instrument_ml_stratified_rankcal'
NPZ = 'C:/dev/phd/casper/data/movielens/ml_stratified_profiles.npz'
T = 15; N_USERS = int(os.environ.get('N_USERS', 250)); SEED = 42; HOLD = 0.40
w, _ = load_instrument_by_name(INST)
d = np.load(NPZ, allow_pickle=True); train, test = d['train'], d['test']
nt = int(d['n_targets']); ni = train.shape[1]
pop = (~np.isnan(train[:, :nt])).mean(0); p_all = (~np.isnan(train)).mean(0)
rng = np.random.default_rng(SEED)
movies = list(range(nt))
poolmask = np.full(ni, -1e9, np.float32); poolmask[:nt] = 0.0   # movies-only

# ---- build per-user folds ----
cases = []
for prof in test:
    rated = np.where(~np.isnan(prof[:nt]))[0]
    if len(rated) < 8: continue
    liked = rated[prof[rated] == 1]; disl = rated[prof[rated] == 0]
    if len(liked) < 4 or len(disl) < 2: continue
    perm = rng.permutation(rated); k = max(2, int(HOLD * len(rated)))
    H = perm[:k]; askable = set(int(x) for x in perm[k:])
    Hl = [int(x) for x in H if prof[x] == 1]; Hd = [int(x) for x in H if prof[x] == 0]
    if len(Hl) < 1 or len(Hd) < 1: continue
    cases.append((prof, askable, np.array(H), np.array(Hl)))
    if len(cases) >= N_USERS: break
print(f"users={len(cases)}  held-out/user: mean |H|={np.mean([len(c[2]) for c in cases]):.1f} "
      f"(liked {np.mean([len(c[3]) for c in cases]):.1f})", flush=True)

# unrated negatives per user for full-slate retrieval
def auc(scores, labels):
    pos = scores[labels == 1]; neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0: return np.nan
    alls = np.concatenate([pos, neg]); order = alls.argsort()
    ranks = np.empty_like(order, float); ranks[order] = np.arange(1, len(alls) + 1)
    # average ties
    return (ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))

def recon(belief, prof, H):
    p = np.clip(belief[H], 1e-6, 1 - 1e-6); y = prof[H]
    brier = float(np.mean((p - y) ** 2))
    nll = float(np.mean(-(y * np.log(p) + (1 - y) * np.log(1 - p))))
    return brier, nll, float(auc(belief[H], y))

def retrieval(belief, prof, Hl, rated_set):
    negs = np.array([m for m in movies if m not in rated_set])    # non-interacted
    ndcgs = []; mrrs = []; hits = []
    for h in Hl:
        sc_h = belief[h]; sc_n = belief[negs]
        rank = 1 + int((sc_n >= sc_h).sum())
        ndcgs.append(1.0 / np.log2(rank + 1) if rank <= 10 else 0.0)
        mrrs.append(1.0 / rank); hits.append(1.0 if rank <= 10 else 0.0)
    return float(np.mean(ndcgs)), float(np.mean(mrrs)), float(np.mean(hits))

def belief_of(rev):
    return np.asarray(w.predict_full(rev))[:nt]

# ---- policies (return next movie to ask, or None) ----
porder = [int(e) for e in np.argsort(-pop)]
def pol_pop(belief, rev, asked, forb):
    return next((e for e in porder if e not in asked and e not in forb), None)
def pol_bel(belief, rev, asked, forb):
    b = belief.copy()
    for e in asked: b[e] = -1e9
    for e in forb: b[e] = -1e9
    return int(np.argmax(b))
def ent(p): p = np.clip(p, 1e-6, 1 - 1e-6); return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))
gpool = [m for m in movies]  # movies-only greedy
def pol_gig(belief, rev, asked, forb):
    cur = belief; h = ent(cur).sum()
    cands = [e for e in gpool if e not in asked and e not in forb]
    sets = [rev + [(e, 1.0)] for e in cands] + [rev + [(e, 0.0)] for e in cands]
    preds = w.predict_batch(sets); ncl = len(cands); best, bs = cands[0], -1e9
    for j, e in enumerate(cands):
        pl = float(cur[e]); eig = p_all[e] * (pl * (h - ent(preds[j][:nt]).sum()) + (1 - pl) * (h - ent(preds[ncl + j][:nt]).sum()))
        if eig > bs: bs, best = eig, e
    return int(best)
actor = EquivariantActor(ni, 'dual')
actor.load_state_dict(torch.load('C:/dev/phd/casper/experiments/paper2/casper_clair_ml_stratified.pt')['actor_state_dict'])
actor.eval()
def state_from(rev):
    bel = np.asarray(w.predict_full(rev)); rat = np.asarray(w.predict_rated(rev))
    s = np.zeros((ni, 3), np.float32); s[:, 2] = 1
    for (e, p) in rev: s[e, 2] = 0; s[e, 1 if p >= 0.5 else 0] = 1
    return np.concatenate([s.reshape(-1), bel.astype(np.float32), rat.astype(np.float32)])
def pol_clair(belief, rev, asked, forb):
    with torch.no_grad(): lg = actor(torch.from_numpy(state_from(rev)).unsqueeze(0))[0].numpy()
    lg = lg + poolmask
    for e in asked: lg[e] = -1e9
    for e in forb: lg[e] = -1e9
    return int(np.argmax(lg))

POLICIES = [('popularity', pol_pop), ('belief-greedy', pol_bel),
            ('greedy-infogain', pol_gig), ('CASPER-CLAIR', pol_clair)]

def run(name, pol):
    t0 = time.time()
    M = {k: np.zeros(T + 1) for k in ['brier', 'nll', 'auc', 'ndcg', 'mrr', 'hit']}
    cnt = np.zeros(T + 1)
    for (prof, askable, H, Hl) in cases:
        rated_set = set(int(x) for x in np.where(~np.isnan(prof[:nt]))[0])
        forb = set(int(x) for x in H)
        rev = []; asked = set()
        for t in range(T + 1):
            belief = belief_of(rev)
            br, nl, au = recon(belief, prof, H); nd, mr, hi = retrieval(belief, prof, Hl, rated_set)
            for k, v in zip(['brier', 'nll', 'auc', 'ndcg', 'mrr', 'hit'], [br, nl, au, nd, mr, hi]):
                if not np.isnan(v): M[k][t] += v
            cnt[t] += 1
            if t == T: break
            a = pol(belief, rev, asked, forb)
            if a is None: continue
            asked.add(a)
            if a in askable and not np.isnan(prof[a]): rev.append((a, float(prof[a])))
    for k in M: M[k] = M[k] / cnt
    print(f"\n{name}  ({time.time()-t0:.0f}s)", flush=True)
    print(f"  RECON   Brier {M['brier'].mean():.4f} (t0 {M['brier'][0]:.3f} t15 {M['brier'][-1]:.3f}) | "
          f"NLL {M['nll'].mean():.4f} (t15 {M['nll'][-1]:.3f}) | AUC {M['auc'].mean():.4f} (t0 {M['auc'][0]:.3f} t15 {M['auc'][-1]:.3f})", flush=True)
    print(f"  RETR    NDCG {M['ndcg'].mean():.4f} (t0 {M['ndcg'][0]:.3f} t5 {M['ndcg'][5]:.3f} t15 {M['ndcg'][-1]:.3f}) | "
          f"MRR {M['mrr'].mean():.4f} (t0 {M['mrr'][0]:.3f} t15 {M['mrr'][-1]:.3f}) | Hit@10 {M['hit'].mean():.4f} (t15 {M['hit'][-1]:.3f})", flush=True)
    return M

print("="*100)
for name, pol in POLICIES:
    run(name, pol)
# ceiling: full reveal of all askable rated movies
print("\n" + "="*100)
M = {k: 0.0 for k in ['brier', 'nll', 'auc', 'ndcg', 'mrr', 'hit']}; c = 0
for (prof, askable, H, Hl) in cases:
    rated_set = set(int(x) for x in np.where(~np.isnan(prof[:nt]))[0])
    rev = [(int(e), float(prof[e])) for e in askable if e < nt]
    belief = belief_of(rev)
    br, nl, au = recon(belief, prof, H); nd, mr, hi = retrieval(belief, prof, Hl, rated_set)
    for k, v in zip(['brier','nll','auc','ndcg','mrr','hit'], [br,nl,au,nd,mr,hi]):
        if not np.isnan(v): M[k] += v
    c += 1
print(f"CEILING (full-reveal askable): Brier {M['brier']/c:.4f} NLL {M['nll']/c:.4f} AUC {M['auc']/c:.4f} | "
      f"NDCG {M['ndcg']/c:.4f} MRR {M['mrr']/c:.4f} Hit@10 {M['hit']/c:.4f}")
