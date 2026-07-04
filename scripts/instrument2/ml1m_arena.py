"""
ml1m_arena.py -- the CANONICAL CASPER ML-1M arena (identical to
scripts/paper2/continuous_policy2_st.py), exposed for INSTRUMENT 2.0 Phase 2.

Protocol conventions (mirrored VERBATIM from the V1 harness so RecVAE numbers are
directly comparable to the V1 papers):
  - items = np.unique(movieId) over ratings.dat -> ni=3706, same ordinal ids as V1.
  - users = np.unique(userId) -> nu=6040.
  - like (implicit positive) = rating >= 4  (the V1 harness like definition).
  - keep users with >=5 likes; rng(0) shuffle; trU = keep[:80%], te = keep[90%:].
  - te usable (>=6 rated items under the SEED profile-split): te[:300] = VAL cohort,
    te[300:] = TEST cohort (304 users) -- disjoint from val.
  - profile-split (SEED, default 123): shuffle a user's ALL rated items, first half =
    profile (fold-in / candidate-exclusion), second half = held-out; tlike = held-out
    items with rating>=4 = the disjoint targets. FULL-catalogue ranking, profile items
    excluded, NDCG@10. tail = Cremonesi head-33% masked (headmask).
  - MOSTPOP q0 anchor = popb = log(train-like-count+1). (V1 ref: 0.310.)
  - V1 full-profile ref: 0.407 full / 0.216 tail (fold the profile-half through the V1
    encoder). This module's metric reproduces those exactly (fidelity check in ml1m_bars).
"""
import os, numpy as np

BASE = 'C:/dev/phd/casper/data/movielens'
ML = f'{BASE}/ml-1m'


def load_arena(seed=123):
    U, I, Rr = [], [], []
    with open(f'{ML}/ratings.dat') as f:
        for line in f:
            a = line.strip().split('::'); U.append(int(a[0])); I.append(int(a[1])); Rr.append(float(a[2]))
    U = np.array(U); I = np.array(I); Rr = np.array(Rr, np.float32)
    uids = {x: k for k, x in enumerate(np.unique(U))}
    iids = {x: k for k, x in enumerate(np.unique(I))}
    nu, ni = len(uids), len(iids)
    uu = np.array([uids[x] for x in U]); ii = np.array([iids[x] for x in I])
    rat_by_u = {}
    for k in range(len(uu)):
        rat_by_u.setdefault(uu[k], []).append((ii[k], float(Rr[k])))
    likes_by_u = {x: [j for j, r in v if r >= 4] for x, v in rat_by_u.items()}
    rng = np.random.default_rng(0)
    keep = [x for x in range(nu) if len(likes_by_u.get(x, [])) >= 5]
    rng.shuffle(keep); nK = len(keep)
    trU = keep[:int(0.8 * nK)]; te = keep[int(0.9 * nK):]
    # popularity over train-user likes
    cnt = np.zeros(ni)
    for x in trU:
        for j in likes_by_u.get(x, []):
            cnt[j] += 1
    popb = np.log(cnt + 1.0).astype(np.float32)
    # head-33% cumulative-popularity mask (tail = remainder)
    order_pop = np.argsort(-cnt); cum = np.cumsum(cnt[order_pop]) / cnt.sum()
    headmask = np.zeros(ni, bool); headmask[order_pop[:np.searchsorted(cum, 0.33) + 1]] = True
    # per-seed profile-split over te (val = te[:300], test = te[300:])
    _rs = np.random.default_rng(seed); SPL = {}
    for x in te:
        its = list(dict(rat_by_u[x]))
        if len(its) >= 6:
            il = its[:]; _rs.shuffle(il); SPL[x] = (set(il[:len(il) // 2]), il[len(il) // 2:])
    tsel = [x for x in te if x in SPL]
    val_users = tsel[:300]; test_users = tsel[300:]
    return dict(nu=nu, ni=ni, uu=uu, ii=ii, rat_by_u=rat_by_u, likes_by_u=likes_by_u,
                trU=trU, te=te, cnt=cnt, popb=popb, headmask=headmask, SPL=SPL,
                val_users=val_users, test_users=test_users)


_W = 1.0 / np.log2(np.arange(2, 12))


def ndcg_at10(score, tlike, profset, headmask, tail):
    """score: (ni,) dense. Excludes profset, NDCG@10 over tlike; tail masks headmask.
    Matches continuous_policy2_st.metr exactly."""
    s = score.copy()
    s[list(profset)] = -1e9
    if tail:
        s[headmask] = -1e9
        rel = set(t for t in tlike if not headmask[t])
    else:
        rel = set(tlike)
    if not rel:
        return None
    o = np.argsort(-s)[:10]
    dcg = sum(_W[p] for p, t in enumerate(o) if int(t) in rel)
    idcg = _W[:min(10, len(rel))].sum() + 1e-12
    return dcg / idcg
