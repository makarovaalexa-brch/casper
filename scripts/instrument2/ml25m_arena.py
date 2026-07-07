"""
ml25m_arena.py -- the ML-25M INSTRUMENT 2.0 arena, mirroring ml1m_arena.py but at 25M scale.

Reuses the EXISTING byte-stable full ML-25M split cached in
`data/movielens/.cache/ml25m/meta.npz` (built by scripts/paper2/ml25m_build_svd_full.py):
  - items  = movies with >= 20 total ratings -> ni=18430, dense-remapped ordinal ids (keepI order).
  - users  = all 162541; rng(0) permutation, LAST 1000 held out -> va = first 500, te = last 500;
             trU = the remaining 161541 (the Paper-B ML-25M full-build convention == the ML-1M
             te[:300]/te[300:] analog, kept separate here as va/te cohorts).
  - like (implicit positive) = rating >= 4.0 (0.5-5 scale).

Protocol conventions mirrored VERBATIM from ml1m_arena.py:
  - popb  = log(train-like-count + 1)  (MOSTPOP anchor).
  - headmask = head-33% cumulative-popularity mask over train-like counts (tail = remainder).
  - profile-split (SEED, default 123): shuffle a user's ALL rated items, first half = profile
    (fold-in / candidate-exclusion), second half = held-out; tlike = held-out items rating>=4
    = the disjoint targets. FULL-catalogue ranking, profile items excluded, NDCG@10. Requires
    >=6 rated items (identical to ml1m_arena).
  - val cohort = usable va users; test cohort = usable te users (disjoint).

Exposes the SAME api surface ml1m_recvae.py / p3_gates.py consume:
  load_arena(seed) -> dict(ni, trU, likes_by_u(train only), rat_by_u(eval only),
                           cnt, popb, headmask, SPL, val_users, test_users)
  ndcg_at10(score, tlike, profset, headmask, tail)   (identical metric to ml1m_arena)
"""
import os, numpy as np

META = 'C:/dev/phd/casper/data/movielens/.cache/ml25m/meta.npz'

_CACHE = {}  # keyed by nothing-relevant-to-seed heavy structures, so we build once, reslice SPL per seed


def _load_base():
    if 'base' in _CACHE:
        return _CACHE['base']
    d = np.load(META)
    uu = d['uu']; ii = d['ii']; rr = d['rr']
    ni = int(d['ni']); nu = int(d['nu'])
    trU = d['trU'].astype(np.int64); va = d['va'].astype(np.int64); te = d['te'].astype(np.int64)
    cnt = d['cnt'].astype(np.float64)
    # popb + headmask over train-like popularity (cnt is already train-like counts from the full build)
    popb = np.log(cnt + 1.0).astype(np.float32)
    order_pop = np.argsort(-cnt); cum = np.cumsum(cnt[order_pop]) / cnt.sum()
    headmask = np.zeros(ni, bool); headmask[order_pop[:np.searchsorted(cum, 0.33) + 1]] = True
    # train-like CSR rows: build likes_by_u only for TRAIN users (for the RecVAE input matrix)
    trmask = np.zeros(nu, bool); trmask[trU] = True
    like = rr >= 4.0
    sel_tr = like & trmask[uu]
    tr_u = uu[sel_tr]; tr_i = ii[sel_tr]
    # eval cohorts: full rating dicts for va + te users only
    evalset = np.zeros(nu, bool); evalset[va] = True; evalset[te] = True
    sel_ev = evalset[uu]
    ev_u = uu[sel_ev]; ev_i = ii[sel_ev]; ev_r = rr[sel_ev]
    rat_by_u = {}
    for k in range(len(ev_u)):
        rat_by_u.setdefault(int(ev_u[k]), []).append((int(ev_i[k]), float(ev_r[k])))
    base = dict(ni=ni, nu=nu, trU=trU, va=va, te=te, cnt=cnt, popb=popb, headmask=headmask,
                tr_u=tr_u, tr_i=tr_i, rat_by_u=rat_by_u)
    _CACHE['base'] = base
    return base


def load_arena(seed=123):
    b = _load_base()
    rat_by_u = b['rat_by_u']
    _rs = np.random.default_rng(seed); SPL = {}
    for x, v in rat_by_u.items():
        its = list(dict(v))  # unique rated items, insertion order
        if len(its) >= 6:
            il = its[:]; _rs.shuffle(il)
            SPL[x] = (set(il[:len(il) // 2]), il[len(il) // 2:])
    va = b['va'].tolist(); te = b['te'].tolist()
    val_users = [x for x in va if x in SPL]
    test_users = [x for x in te if x in SPL]
    ar = dict(ni=b['ni'], nu=b['nu'], trU=b['trU'], cnt=b['cnt'], popb=b['popb'],
              headmask=b['headmask'], tr_u=b['tr_u'], tr_i=b['tr_i'], rat_by_u=rat_by_u,
              SPL=SPL, val_users=val_users, test_users=test_users)
    return ar


_W = 1.0 / np.log2(np.arange(2, 12))


def ndcg_at10(score, tlike, profset, headmask, tail):
    """Identical to ml1m_arena.ndcg_at10."""
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
