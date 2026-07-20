"""
squeeze_r3_eig.py -- INSTRUMENT 2.0 "squeeze arc" rung 3: a REALIZABLE, NON-PEEKING
best-subset / best-direction selector that attacks the honest realizable gaps the R0
compass identified: the SELECTION gap (full-profile 0.5541 -> best-8-item-subset oracle
0.7546) and then the CONTINUITY gap (0.7546 -> answer-peek direction oracle 0.8782).

KEY R1 LESSON (obeyed here): EIG / optimal design in the LATENT or POPULATION metric
collapses to the popularity axis (reproduces the PCA-z* 0.169 failure). Informativeness
MUST come from the DECODER geometry. So the criterion is a *ranking-aware* decoder-metric
EIG: reduce the posterior variance of the SCORES of the items currently contending for the
top-K (contenders = current belief's top-M items, decoder rows w_i), never the global
latent variance. This is (a) decoder-metric (uses w_i^T P q), (b) genuinely per-user
adaptive (contenders move as the belief sharpens -- a NONLINEAR objective, the only way to
escape the linear-Gaussian "optimal design is non-adaptive" trap), and (c) non-peeking
(never touches held-out TEST NDCG -- that is the off-limits L3 oracle).

Belief:  mean z tracked by the ADDITIVE operator z' = z + eta*a*q (R1: additive re-inflation
beats Kalman-mean shrinkage), covariance P in the DECODER metric (P0 eigenvectors =
decoder-SVD dirs, eigenvalues ~ singular-value^2), Kalman rank-1 downdate for selection only.
Answer a = cos(z*, q) graded. Decode from z.

  DISCRETE selector : pick q from the pool (subsampled item-decoder rows + decoder-SVD dirs,
                      the squeeze_r01 pool) maximizing f(q)=sum_i u_i (w_i^T P q)^2/(q^T P q+R).
  CONTINUOUS selector: emit the OFF-CATALOG continuous q maximizing the same criterion
                      (top generalized eigenvector of the P-whitened contender scatter) ->
                      targets the 0.755->0.878 continuity gap.

Frozen invariants: instrument .cache/instrument2/ml1m_recvae_d512_best.pt; operator z'=z+16*a*q,
z0=0; arena ml1m_arena byte-identical splits; seeds {1,2,3,7,11}; te[300:] TEST (304 users);
T=8; graded a=cos(z*,q); NDCG@10 full + Cremonesi tail. Anchors: full-profile 0.5541,
actor 0.4907/0.2982, SVD-8 0.4709/0.2916, item-8 0.4625.

Usage: python scripts/instrument2/squeeze_r3_eig.py [run|noisy]
"""
import os, sys, json, time, argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ml1m_arena as A
import p4a_battery as P
from p4a_bootstrap import boot

DEVICE = torch.device('cpu'); torch.set_num_threads(4)
CK = '.cache/instrument2'
NI = 3706
ETA = 16.0
T = 8
SEEDS = [1, 2, 3, 7, 11]
M_CONT = 50                       # number of top-belief contender items driving the EIG
OUT = f'{CK}/squeeze_r3_eig.json'
CHAN = f'{CK}/p4c_channel.json'

# gap anchors (R0 compass)
FULL_PROFILE = 0.5541
SEL_ORACLE = 0.7546               # best-8-item-subset fold oracle (selection ceiling)
DIR_ORACLE = 0.8782              # answer-peek direction oracle (continuity ceiling)
SVD8_FULL = 0.4709; SVD8_TAIL = 0.2916
ITEM8_FULL = 0.4625
ACTOR_FULL = 0.4907; ACTOR_TAIL = 0.2982


# ------------------------------------------------------------------ io
def _save(obj):
    json.dump(obj, open(OUT, 'w'), indent=2)
    print(f'[saved] {OUT}', flush=True)


# ------------------------------------------------------------------ decoder-metric prior P0
def train_zstar_scale(model, rd_all):
    """mean ||z*|| over train-like encodings (the natural latent magnitude ~17.68)."""
    ar = A.load_arena(seed=123)
    trU = [x for x in ar['trU'] if sum(1 for j, r in rd_all[x].items() if r >= 4) >= 4]
    d = model.decoder.in_features
    norms = []
    for st in range(0, len(trU), 500):
        ch = trU[st:st + 500]; Xd = np.zeros((len(ch), NI), np.float32)
        for r, x in enumerate(ch):
            lk = [j for j, rr in rd_all[x].items() if rr >= 4]; Xd[r, lk] = 1.0
        Z = P.enc_mu(model, Xd); norms += list(np.linalg.norm(Z, axis=1))
    return float(np.mean(norms))


def decoder_metric_P0(model, scale):
    """P0 = V diag(lam) V^T ; V = decoder-SVD basis (right sing vecs of centered W),
    lam ~ sigma^2 normalized so trace(P0)=scale^2 (prior variance budget in decoder geometry)."""
    d = model.decoder.in_features
    W = model.decoder.weight.detach().numpy()
    Wc = W - W.mean(0, keepdims=True)
    _, sv, Vt = np.linalg.svd(Wc, full_matrices=False)         # Vt:(d,d) rows=eigvecs
    lam = (sv[:d] ** 2).astype(np.float64)
    lam = lam / lam.sum() * (scale ** 2)
    P0 = (Vt.T * lam[None, :]) @ Vt
    return P0.astype(np.float64)


# ------------------------------------------------------------------ candidate pool (squeeze_r01 pool: items + SVD dirs)
def build_pool(model, rng, n_item=1024):
    W = model.decoder.weight.detach().numpy()
    D = W / (np.linalg.norm(W, axis=1, keepdims=True) + 1e-9)
    isel = rng.choice(D.shape[0], min(n_item, D.shape[0]), replace=False)
    svd = P.decoder_svd_dirs(model, 512)
    Q = np.concatenate([D[isel], svd], 0).astype(np.float64)
    return Q


# ------------------------------------------------------------------ ranking-aware decoder-metric EIG selector
def eig_select_user(zs, W, bdec, profset, P0, R, pool, mode, cont_whiten=True,
                    answer_fn=None, warm=0, warm_dirs=None):
    """Run the T-turn selector for one user. mode in {'discrete','continuous'}.
    warm>0 : first `warm` turns use the fixed decoder-SVD basis warm_dirs[t] (globally
    informative) to seed the belief before switching to contender-adaptive EIG -- avoids the
    cold-start popularity trap (top-M contenders of the z=0 belief = most-popular items).
    Returns final decode score S (ni,). answer_fn(q)->a overrides graded cos (noisy channel)."""
    d = W.shape[1]
    z = np.zeros(d); Pc = P0.copy()
    nz = np.linalg.norm(zs) + 1e-9
    prof_list = list(profset)
    u_disc = 1.0 / np.log2(np.arange(M_CONT) + 2.0)            # rank discount over contenders
    for t in range(T):
        if t < warm:                                           # SVD warm-start turn
            q = warm_dirs[t]
            a = (zs @ q / nz) if answer_fn is None else answer_fn(q)
            z = z + ETA * a * q
            Pq = Pc @ q; Pc = Pc - np.outer(Pq, Pq) / float(q @ Pq + R)
            continue
        S = z @ W.T + bdec
        S[prof_list] = -1e18
        cont = np.argpartition(-S, M_CONT)[:M_CONT]
        order = np.argsort(-S[cont]); cont = cont[order]        # ranked top-M
        Wc = W[cont]                                            # (M,d) contender decoder rows
        u = u_disc
        PWc = Wc @ Pc                                           # (M,d) = w_i^T P
        if mode == 'discrete':
            proj = pool @ PWc.T                                # (C,M) = w_i^T P q
            num = (proj * proj * u[None, :]).sum(1)            # (C,)
            PQ = pool @ Pc                                      # (C,d)
            den = (pool * PQ).sum(1) + R                        # q^T P q + R
            q = pool[int(np.argmax(num / den))]
        else:
            # continuous: max q^T (P G P) q / (q^T (P+R I) q), G=sum u_i w_i w_i^T
            B = Pc + R * np.eye(d)
            Psi = np.sqrt(u)[:, None] * PWc                    # (M,d): sqrt(u_i) w_i^T P
            # reduced M x M generalized problem: q ∝ B^{-1} Psi^T c, c=top eig of Psi B^{-1} Psi^T
            BinvPsiT = np.linalg.solve(B, Psi.T)               # (d,M)
            Gred = Psi @ BinvPsiT                              # (M,M)
            w, V = np.linalg.eigh((Gred + Gred.T) / 2)
            c = V[:, -1]
            q = BinvPsiT @ c
            if not cont_whiten:                                # plain contender-scatter axis (P=I twist off)
                Psi2 = np.sqrt(u)[:, None] * Wc
                G = Psi2.T @ Psi2
                wv, Vv = np.linalg.eigh(G)
                q = Vv[:, -1]
            q = q / (np.linalg.norm(q) + 1e-9)
        a = (zs @ q / nz) if answer_fn is None else answer_fn(q)
        z = z + ETA * a * q
        # Kalman covariance downdate (selection only)
        Pq = Pc @ q; sden = float(q @ Pq + R)
        Pc = Pc - np.outer(Pq, Pq) / sden
    S = z @ W.T + bdec
    return S


def run_cohort(model, W, bdec, ar, users, zst, P0, R, pool, mode, cont_whiten=True,
               warm=0, warm_dirs=None):
    """Returns (full, tail, peruser_full dict)."""
    af = at = 0.0; mf = mt = 0; pu = {}
    for i, x in enumerate(users):
        profset, tst = ar['SPL'][x]; rd = ar['rat_by_u_dict'][x]
        rel = set(j for j in tst if rd[j] >= 4)
        if not rel:
            continue
        S = eig_select_user(zst[i], W, bdec, profset, P0, R, pool, mode, cont_whiten,
                            warm=warm, warm_dirs=warm_dirs)
        nf = A.ndcg_at10(S, rel, profset, ar['headmask'], False)
        ntl = A.ndcg_at10(S, rel, profset, ar['headmask'], True)
        if nf is not None: af += nf; mf += 1; pu[x] = nf
        if ntl is not None: at += ntl; mt += 1
    return af / max(mf, 1), at / max(mt, 1), pu


# ------------------------------------------------------------------ anchors (per-user full for bootstrap)
def anchor_peruser(model, W, bdec, ar, users, zst, kind, svd8=None, actor=None, rng=None):
    Wn = W; bn = bdec; d = W.shape[1]; pu = {}
    af = at = 0.0; mf = mt = 0
    zt = torch.tensor(zst) if actor is not None else None
    if kind == 'actor':
        nz = zt.norm(dim=1, keepdim=True) + 1e-9; z = torch.zeros(len(users), d)
        Wt = torch.tensor(W.T, dtype=torch.float32); bt = torch.tensor(bdec, dtype=torch.float32)
        with torch.no_grad():
            for t in range(T):
                q = actor(z, t); a = (zt * q).sum(1, keepdim=True) / nz; z = z + ETA * a * q
            Sall = (z @ Wt + bt).numpy().astype(np.float64)
    for i, x in enumerate(users):
        profset, tst = ar['SPL'][x]; rd = ar['rat_by_u_dict'][x]
        rel = set(j for j in tst if rd[j] >= 4)
        if not rel: continue
        if kind == 'svd8':
            z = P.unroll_static(zst[i], svd8); S = z @ Wn.T + bn
        elif kind == 'item8':
            lk = [j for j in profset if rd[j] >= 4]
            if len(lk) > 8: lk = list(rng.choice(lk, 8, replace=False))
            z = P.enc_mu(model, P.bag_from_likes(lk)[None, :])[0] if lk else np.zeros(d, np.float32)
            S = z @ Wn.T + bn
        else:
            S = Sall[i]
        nf = A.ndcg_at10(S, rel, profset, ar['headmask'], False)
        ntl = A.ndcg_at10(S, rel, profset, ar['headmask'], True)
        if nf is not None: af += nf; mf += 1; pu[x] = nf
        if ntl is not None: at += ntl; mt += 1
    return af / max(mf, 1), at / max(mt, 1), pu


# ------------------------------------------------------------------ paired bootstrap over seed-avg per-user
def paired(Ad, Bd):
    keys = [k for k in Ad if k in Bd]
    return boot([Ad[k] - Bd[k] for k in keys])


def seedavg(pu):
    return {x: float(np.mean(v)) for x, v in pu.items()}


# ================================================================== main run (clean channel)
def run(fixedR=None, fixed_whiten=None):
    t0 = time.time()
    model, d = P.load_model()
    W = model.decoder.weight.detach().numpy().astype(np.float64)
    bdec = model.decoder.bias.detach().numpy().astype(np.float64)
    rd_all = {x: dict(v) for x, v in A.load_arena(seed=123)['rat_by_u'].items()}
    scale = train_zstar_scale(model, rd_all)
    P0 = decoder_metric_P0(model, scale)
    pool = build_pool(model, np.random.default_rng(0))
    svd8 = P.decoder_svd_dirs(model, 8)
    blob = torch.load(f'{CK}/p4a_actor_s0.pt', map_location=DEVICE)
    actor = P.Actor(d); actor.load_state_dict(blob['state']); actor.eval()
    print(f'[r3] scale={scale:.3f} pool={pool.shape} P0 trace={np.trace(P0):.1f} '
          f'[{time.time()-t0:.0f}s]', flush=True)

    # --- R selection on seed-1 VAL full-NDCG (discrete selector) ---
    if fixedR is not None:
        bestR = fixedR; bestf = -1; cont_whiten = fixed_whiten
        print(f'[r3] using cached bestR={bestR} cont_whiten={cont_whiten} (val sweep skipped)', flush=True)
    else:
        arv = P.arena_seed(1, rd_all); val = arv['val_users']; zstv = P.build_zstar(model, arv, val)
        Rgrid = [0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0]
        bestR = None; bestf = -1
        for R in Rgrid:
            f, _, _ = run_cohort(model, W, bdec, arv, val, zstv, P0, R, pool, 'discrete')
            print(f'[r3 valR] R={R:5.1f} discrete val_full={f:.4f} [{time.time()-t0:.0f}s]', flush=True)
            if f > bestf: bestf = f; bestR = R
        fc_w, _, _ = run_cohort(model, W, bdec, arv, val, zstv, P0, bestR, pool, 'continuous', True)
        fc_p, _, _ = run_cohort(model, W, bdec, arv, val, zstv, P0, bestR, pool, 'continuous', False)
        cont_whiten = fc_w >= fc_p
        print(f'[r3] bestR={bestR} (val {bestf:.4f}); cont whiten_val={fc_w:.4f} plain_val={fc_p:.4f} '
              f'-> whiten={cont_whiten}', flush=True)

    # --- 5-seed TEST ---
    perseed = {k: {'f': [], 't': []} for k in
               ['eig_discrete', 'eig_continuous', 'svd8', 'item8', 'actor']}
    pu = {k: {} for k in perseed}
    for sd in SEEDS:
        ar = P.arena_seed(sd, rd_all); test = ar['test_users']; zst = P.build_zstar(model, ar, test)
        fd, td, pud = run_cohort(model, W, bdec, ar, test, zst, P0, bestR, pool, 'discrete')
        fk, tk, puk = run_cohort(model, W, bdec, ar, test, zst, P0, bestR, pool, 'continuous', cont_whiten)
        fs, ts, pus = anchor_peruser(model, W, bdec, ar, test, zst, 'svd8', svd8=svd8)
        fi, ti, pui = anchor_peruser(model, W, bdec, ar, test, zst, 'item8', rng=np.random.default_rng(sd))
        fa, ta, pua = anchor_peruser(model, W, bdec, ar, test, zst, 'actor', actor=actor)
        for k, (fv, tv, p) in [('eig_discrete', (fd, td, pud)), ('eig_continuous', (fk, tk, puk)),
                               ('svd8', (fs, ts, pus)), ('item8', (fi, ti, pui)),
                               ('actor', (fa, ta, pua))]:
            perseed[k]['f'].append(fv); perseed[k]['t'].append(tv)
            for x, v in p.items(): pu[k].setdefault(x, []).append(v)
        print(f'[r3 TEST] seed{sd}: disc {fd:.4f}/{td:.4f} | cont {fk:.4f}/{tk:.4f} | '
              f'svd8 {fs:.4f} | item8 {fi:.4f} | actor {fa:.4f} [{time.time()-t0:.0f}s]', flush=True)

    def agg(a):
        return {'full': float(np.mean(a['f'])), 'tail': float(np.mean(a['t'])),
                'full_sd': float(np.std(a['f'])), 'tail_sd': float(np.std(a['t'])),
                'per_seed_full': [float(x) for x in a['f']]}
    res = {'invariants': {'eta': ETA, 'T': T, 'seeds': SEEDS, 'M_contenders': M_CONT,
                          'scale': scale, 'bestR': bestR, 'cont_whiten': bool(cont_whiten),
                          'pool_shape': list(pool.shape)},
           'anchors': {'full_profile': FULL_PROFILE, 'sel_oracle': SEL_ORACLE,
                       'dir_oracle': DIR_ORACLE, 'svd8': [SVD8_FULL, SVD8_TAIL],
                       'item8': ITEM8_FULL, 'actor': [ACTOR_FULL, ACTOR_TAIL]},
           'test': {k: agg(v) for k, v in perseed.items()}}

    # --- gap-closing fractions ---
    def frac_sel(x): return (x - FULL_PROFILE) / (SEL_ORACLE - FULL_PROFILE)
    def frac_cont(x): return (x - SEL_ORACLE) / (DIR_ORACLE - SEL_ORACLE)
    ed = res['test']['eig_discrete']['full']; ec = res['test']['eig_continuous']['full']
    res['gap_closing'] = {
        'discrete_full': ed, 'continuous_full': ec,
        'discrete_sel_gap_frac_vs_fullprofile': frac_sel(ed),
        'continuous_sel_gap_frac_vs_fullprofile': frac_sel(ec),
        'discrete_vs_svd8_frac_of_selgap': (ed - SVD8_FULL) / (SEL_ORACLE - SVD8_FULL),
        'continuity_gap_frac_beyond_seloracle_discrete': frac_cont(ed),
        'continuity_gap_frac_beyond_seloracle_continuous': frac_cont(ec)}

    # --- bootstraps (seed-avg per-user full) ---
    sa = {k: seedavg(pu[k]) for k in pu}
    res['bootstrap'] = {
        'eig_discrete_minus_svd8': paired(sa['eig_discrete'], sa['svd8']),
        'eig_discrete_minus_item8': paired(sa['eig_discrete'], sa['item8']),
        'eig_discrete_minus_actor': paired(sa['eig_discrete'], sa['actor']),
        'eig_continuous_minus_eig_discrete': paired(sa['eig_continuous'], sa['eig_discrete']),
        'eig_continuous_minus_actor': paired(sa['eig_continuous'], sa['actor']),
        'eig_continuous_minus_svd8': paired(sa['eig_continuous'], sa['svd8'])}
    _save(res)

    # --- summary ---
    print('\n================ SQUEEZE R3-EIG SUMMARY ================', flush=True)
    for k in ['eig_discrete', 'eig_continuous', 'svd8', 'item8', 'actor']:
        a = res['test'][k]
        print(f"  {k:16s} full {a['full']:.4f} (sd {a['full_sd']:.4f}) tail {a['tail']:.4f}", flush=True)
    gc = res['gap_closing']
    print(f"\n  discrete captures {100*gc['discrete_sel_gap_frac_vs_fullprofile']:.1f}% of the "
          f"selection gap (0.554->0.755); vs SVD-8 baseline {100*gc['discrete_vs_svd8_frac_of_selgap']:.1f}% "
          f"of the SVD8->oracle gap", flush=True)
    print(f"  continuous continuity-gap fraction beyond sel-oracle: "
          f"{100*gc['continuity_gap_frac_beyond_seloracle_continuous']:.1f}%", flush=True)
    for nm, b in res['bootstrap'].items():
        print(f"  {nm:38s} dfull={b['mean_diff']:+.4f} CI{[round(x,4) for x in b['ci95']]} "
              f"p(>0)={b['p_gt0']:.3f}", flush=True)
    print(f"[r3] done [{time.time()-t0:.0f}s]", flush=True)
    return res


# ================================================================== optional: noisy-channel spot-check
def _load_channel():
    ch = json.load(open(CHAN))
    edges = np.array([(-np.inf if e is None and i == 0 else (np.inf if e is None else e))
                      for i, e in enumerate(ch['bin_edges'])], np.float64)
    edges[0] = -np.inf; edges[-1] = np.inf
    return {'edges': edges, 'bin_mean': np.array(ch['bin_mean_rating']),
            'bin_dist': np.array(ch['bin_dist_12345'])}


def _sample_one(s, ch, rng, center):
    edges = ch['edges']; bin_mean = ch['bin_mean']; bin_dist = ch['bin_dist']; NB = len(bin_mean)
    b = int(np.clip(np.digitize([s], edges)[0] - 1, 0, NB - 1))
    r0 = rng.choice([1, 2, 3, 4, 5], p=bin_dist[b])
    r = bin_mean[b] + (r0 - bin_mean[b])
    return float(np.clip((r - center) / 2.0, -1.0, 1.0))


def run_noisy(winner_mode='eig_discrete'):
    """Spot-check the clean-channel winner under the P4C empirical noisy channel (scale-matched,
    per-user centered), 5-seed. Reports the selector vs the R2 noise-adapted static-repeat number."""
    t0 = time.time()
    model, d = P.load_model()
    W = model.decoder.weight.detach().numpy().astype(np.float64)
    bdec = model.decoder.bias.detach().numpy().astype(np.float64)
    rd_all = {x: dict(v) for x, v in A.load_arena(seed=123)['rat_by_u'].items()}
    scale = train_zstar_scale(model, rd_all); P0 = decoder_metric_P0(model, scale)
    pool = build_pool(model, np.random.default_rng(0))
    ch = _load_channel()
    prev = json.load(open(OUT)) if os.path.exists(OUT) else {}
    bestR = prev.get('invariants', {}).get('bestR', 4.0)
    cw = prev.get('invariants', {}).get('cont_whiten', True)
    mode = 'discrete' if winner_mode == 'eig_discrete' else 'continuous'

    def um_of(ar, users):
        um = {}
        for x in users:
            profset, _ = ar['SPL'][x]; rr = [rd_all[x][j] for j in profset]
            um[x] = float(np.mean(rr)) if rr else 3.0
        return um

    ff = []; tt = []
    for sd in SEEDS:
        ar = P.arena_seed(sd, rd_all); test = ar['test_users']; zst = P.build_zstar(model, ar, test)
        um = um_of(ar, test)
        # scale-match cmul via a clean-answer trajectory pass (like squeeze_r01.compute_cmul)
        rng = np.random.default_rng(1000 * sd)
        s_all = []; a_all = []
        for i, x in enumerate(test):
            zs = zst[i]; nz = np.linalg.norm(zs) + 1e-9
            z = np.zeros(d); Pc = P0.copy(); prof = list(ar['SPL'][x][0])
            for t in range(T):
                S = z @ W.T + bdec; S[prof] = -1e18
                cont = np.argpartition(-S, M_CONT)[:M_CONT]
                Wc = W[cont]; u = 1.0 / np.log2(np.arange(M_CONT) + 2.0); PWc = Wc @ Pc
                proj = pool @ PWc.T; num = (proj * proj * u[None, :]).sum(1)
                PQ = pool @ Pc; den = (pool * PQ).sum(1) + bestR
                q = pool[int(np.argmax(num / den))]
                s = float(zs @ q / nz); a = _sample_one(s, ch, rng, um[x])
                s_all.append(s); a_all.append(a)
                z = z + ETA * s * q
                Pq = Pc @ q; Pc = Pc - np.outer(Pq, Pq) / float(q @ Pq + bestR)
        cmul = np.sqrt(np.mean(np.square(s_all)) / (np.mean(np.square(a_all)) + 1e-12))
        # noisy eval pass with scale-matched answers
        rng2 = np.random.default_rng(1000 * sd)
        af = at = 0.0; mf = mt = 0
        for i, x in enumerate(test):
            profset, tst = ar['SPL'][x]; rd = rd_all[x]
            rel = set(j for j in tst if rd[j] >= 4)
            if not rel: continue
            ans = lambda q, i=i, x=x: cmul * _sample_one(float(zst[i] @ q / (np.linalg.norm(zst[i]) + 1e-9)),
                                                         ch, rng2, um[x])
            S = eig_select_user(zst[i], W, bdec, profset, P0, bestR, pool, mode, cw, answer_fn=ans)
            nf = A.ndcg_at10(S, rel, profset, ar['headmask'], False)
            ntl = A.ndcg_at10(S, rel, profset, ar['headmask'], True)
            if nf is not None: af += nf; mf += 1
            if ntl is not None: at += ntl; mt += 1
        ff.append(af / max(mf, 1)); tt.append(at / max(mt, 1))
        print(f'[r3 noisy] seed{sd} cmul={cmul:.3f} {winner_mode} {ff[-1]:.4f}/{tt[-1]:.4f} '
              f'[{time.time()-t0:.0f}s]', flush=True)
    out = {'mode': winner_mode, 'full': float(np.mean(ff)), 'tail': float(np.mean(tt)),
           'full_sd': float(np.std(ff)), 'tail_sd': float(np.std(tt)),
           'per_seed_full': [float(x) for x in ff],
           'context': {'r2_noiseadapted_static_repeat_full': 0.3143, 'r2_noisy_actor_full': 0.2844}}
    prev['noisy_spotcheck'] = out; json.dump(prev, open(OUT, 'w'), indent=2)
    print(f'[saved] {OUT} :: noisy_spotcheck', flush=True)
    print(f'\n[r3 noisy] {winner_mode} full {out["full"]:.4f} tail {out["tail"]:.4f} '
          f"(vs R2 noise-adapted static-repeat 0.3143, noisy actor 0.2844)", flush=True)


# ================================================================== SVD warm-start hybrid
def run_hybrid(warm=4):
    """Best realizable shot: seed the belief with `warm` fixed decoder-SVD probes (globally
    informative), then switch to contender-adaptive decoder-metric EIG (discrete + continuous).
    Reuses bestR / cont_whiten from the pure run. 5-seed TEST + bootstrap vs SVD-8 / actor."""
    t0 = time.time()
    model, d = P.load_model()
    W = model.decoder.weight.detach().numpy().astype(np.float64)
    bdec = model.decoder.bias.detach().numpy().astype(np.float64)
    rd_all = {x: dict(v) for x, v in A.load_arena(seed=123)['rat_by_u'].items()}
    scale = train_zstar_scale(model, rd_all); P0 = decoder_metric_P0(model, scale)
    pool = build_pool(model, np.random.default_rng(0))
    svd8 = P.decoder_svd_dirs(model, 8)
    wdirs = P.decoder_svd_dirs(model, T).astype(np.float64)     # warm-start basis (T dirs)
    blob = torch.load(f'{CK}/p4a_actor_s0.pt', map_location=DEVICE)
    actor = P.Actor(d); actor.load_state_dict(blob['state']); actor.eval()
    prev = json.load(open(OUT)) if os.path.exists(OUT) else {}
    bestR = prev.get('invariants', {}).get('bestR', 0.5)
    cw = prev.get('invariants', {}).get('cont_whiten', True)
    print(f'[r3 hybrid] warm={warm} bestR={bestR} cw={cw} [{time.time()-t0:.0f}s]', flush=True)

    perseed = {'hyb_discrete': {'f': [], 't': []}, 'hyb_continuous': {'f': [], 't': []},
               'svd8': {'f': [], 't': []}, 'actor': {'f': [], 't': []}}
    pu = {k: {} for k in perseed}
    for sd in SEEDS:
        ar = P.arena_seed(sd, rd_all); test = ar['test_users']; zst = P.build_zstar(model, ar, test)
        fd, td, pud = run_cohort(model, W, bdec, ar, test, zst, P0, bestR, pool, 'discrete',
                                 warm=warm, warm_dirs=wdirs)
        fk, tk, puk = run_cohort(model, W, bdec, ar, test, zst, P0, bestR, pool, 'continuous', cw,
                                 warm=warm, warm_dirs=wdirs)
        fs, ts, pus = anchor_peruser(model, W, bdec, ar, test, zst, 'svd8', svd8=svd8)
        fa, ta, pua = anchor_peruser(model, W, bdec, ar, test, zst, 'actor', actor=actor)
        for k, (fv, tv, p) in [('hyb_discrete', (fd, td, pud)), ('hyb_continuous', (fk, tk, puk)),
                               ('svd8', (fs, ts, pus)), ('actor', (fa, ta, pua))]:
            perseed[k]['f'].append(fv); perseed[k]['t'].append(tv)
            for x, v in p.items(): pu[k].setdefault(x, []).append(v)
        print(f'[r3 hybrid TEST] seed{sd}: hyb-disc {fd:.4f}/{td:.4f} | hyb-cont {fk:.4f}/{tk:.4f} | '
              f'svd8 {fs:.4f} | actor {fa:.4f} [{time.time()-t0:.0f}s]', flush=True)

    def agg(a):
        return {'full': float(np.mean(a['f'])), 'tail': float(np.mean(a['t'])),
                'full_sd': float(np.std(a['f'])), 'tail_sd': float(np.std(a['t'])),
                'per_seed_full': [float(x) for x in a['f']]}
    sa = {k: seedavg(pu[k]) for k in pu}
    hd = float(np.mean(perseed['hyb_discrete']['f'])); hc = float(np.mean(perseed['hyb_continuous']['f']))
    out = {'warm': warm, 'bestR': bestR, 'test': {k: agg(v) for k, v in perseed.items()},
           'gap_closing': {
               'hyb_discrete_full': hd, 'hyb_continuous_full': hc,
               'hyb_discrete_sel_gap_frac': (hd - FULL_PROFILE) / (SEL_ORACLE - FULL_PROFILE),
               'hyb_discrete_vs_svd8_frac_of_selgap': (hd - SVD8_FULL) / (SEL_ORACLE - SVD8_FULL),
               'hyb_continuous_continuity_frac': (hc - SEL_ORACLE) / (DIR_ORACLE - SEL_ORACLE)},
           'bootstrap': {
               'hyb_discrete_minus_svd8': paired(sa['hyb_discrete'], sa['svd8']),
               'hyb_discrete_minus_actor': paired(sa['hyb_discrete'], sa['actor']),
               'hyb_continuous_minus_hyb_discrete': paired(sa['hyb_continuous'], sa['hyb_discrete']),
               'hyb_continuous_minus_actor': paired(sa['hyb_continuous'], sa['actor'])}}
    prev['hybrid'] = out; json.dump(prev, open(OUT, 'w'), indent=2)
    print(f'[saved] {OUT} :: hybrid', flush=True)
    print('\n============ SQUEEZE R3-EIG HYBRID (SVD warm-start) ============', flush=True)
    for k in ['hyb_discrete', 'hyb_continuous', 'svd8', 'actor']:
        a = out['test'][k]
        print(f"  {k:16s} full {a['full']:.4f} (sd {a['full_sd']:.4f}) tail {a['tail']:.4f}", flush=True)
    for nm, b in out['bootstrap'].items():
        print(f"  {nm:36s} dfull={b['mean_diff']:+.4f} CI{[round(x,4) for x in b['ci95']]} "
              f"p(>0)={b['p_gt0']:.3f}", flush=True)
    print(f"[r3 hybrid] done [{time.time()-t0:.0f}s]", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', nargs='?', default='run', choices=['run', 'noisy', 'hybrid'])
    ap.add_argument('--warm', type=int, default=4)
    ap.add_argument('--fast', action='store_true', help='skip val sweep, use cached bestR=0.5 whiten')
    args = ap.parse_args()
    if args.stage == 'run':
        run(fixedR=0.5, fixed_whiten=True) if args.fast else run()
    elif args.stage == 'hybrid':
        run_hybrid(args.warm)
    else:
        run_noisy()


if __name__ == '__main__':
    main()
