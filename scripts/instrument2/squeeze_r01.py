"""
squeeze_r01.py -- INSTRUMENT 2.0 "squeeze arc" rungs 0-1 on the certified RecVAE-d512
instrument (ML-1M). Sizes the realizable prize (privileged oracles) then builds a
zero-training Bayesian (Kalman + D-optimal) elicitation arm.

Instrument (frozen): .cache/instrument2/ml1m_recvae_d512_best.pt
Operator (P3 winner, additive baseline): z' = z + eta*a*q, eta=16, z0=0 cold seed.
Graded answer: a = cos(z*, q), z* = enc(profile-half likes).
Arena ml1m_arena (byte-identical V1 splits). Eval seeds {1,2,3,7,11}, te[300:] TEST (304 users).
NDCG@10 full + Cremonesi tail. T=8 turns.

Anchors (P4a): actor 0.4907/0.2982 | SVD-basis-8 0.4709/0.2916 | item-8 fold 0.4625/0.2796
             | full-profile 0.5541.

Stages:
  oracle_dir   RUNG 0a: L3 greedy answer-peek DIRECTION oracle (per-user, per-turn greedily pick
               the candidate direction that maximizes test NDCG after the additive update).
  oracle_item  RUNG 0b: L3 greedy best-8-item-subset fold oracle (greedy over profile items).
  bayes        RUNG 1: Kalman belief + D-optimal / A-optimal policy; operator-swap decomposition
               (Kalman x {D-optimal, actor dirs, SVD-8}) vs (additive x same); paired bootstrap.
  bayes_noisy  RUNG 1d: rerun the Bayesian headline under the P4C empirical noisy channel.

Usage: OMP_NUM_THREADS=3 python scripts/instrument2/squeeze_r01.py {oracle_dir|oracle_item|bayes|bayes_noisy}
"""
import os, sys, json, time, argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ml1m_arena as A
import p4a_battery as P

DEVICE = torch.device('cpu'); torch.set_num_threads(3)
CK = '.cache/instrument2'
NI = 3706
ETA = 16.0
T = 8
SEEDS = [1, 2, 3, 7, 11]
OUT = f'{CK}/squeeze_r01.json'
CHAN = f'{CK}/p4c_channel.json'
_W10 = A._W[:10]


# ------------------------------------------------------------------ io
def _save(stage, obj):
    all_ = json.load(open(OUT)) if os.path.exists(OUT) else {}
    all_[stage] = obj
    json.dump(all_, open(OUT, 'w'), indent=2)
    print(f'[saved] {OUT} :: {stage}', flush=True)


# ------------------------------------------------------------------ vectorized NDCG over candidates
def ndcg_batch_full(Scand, profset, rel_set):
    """Scand:(C,NI) -> NDCG@10 (full) per candidate row (C,). rel_set = held-out likes."""
    if not rel_set:
        return None
    S = Scand.copy()
    S[:, list(profset)] = -1e18
    relmask = np.zeros(NI, bool); relmask[list(rel_set)] = True
    C = S.shape[0]
    top = np.argpartition(-S, 10, axis=1)[:, :10]                 # (C,10)
    row = np.arange(C)[:, None]
    order = np.argsort(-S[row, top], axis=1)
    topo = top[row, order]                                        # ordered top-10 item ids
    hit = relmask[topo].astype(np.float64)                       # (C,10)
    dcg = (hit * _W10[None, :]).sum(1)
    idcg = _W10[:min(10, len(rel_set))].sum() + 1e-12
    return dcg / idcg


# ================================================================== RUNG 0a: direction oracle
def build_candidate_dirs(model, rng, n_rand=512):
    """~ (3706 items + 512 SVD + n_rand random) unit directions in z-space."""
    W = model.decoder.weight.detach().numpy()
    D = W / (np.linalg.norm(W, axis=1, keepdims=True) + 1e-9)     # 3706 item rows
    svd = P.decoder_svd_dirs(model, 512)                         # 512 informative
    R = rng.standard_normal((n_rand, W.shape[1])).astype(np.float32)
    R /= (np.linalg.norm(R, axis=1, keepdims=True) + 1e-9)
    Q = np.concatenate([D, svd, R], 0).astype(np.float32)
    return Q


def stage_oracle_dir():
    model, d = P.load_model()
    W = model.decoder.weight.detach().numpy().astype(np.float64)
    bdec = model.decoder.bias.detach().numpy().astype(np.float64)
    rd_all = {x: dict(v) for x, v in A.load_arena(seed=123)['rat_by_u'].items()}
    Q = build_candidate_dirs(model, np.random.default_rng(0))    # (C,d)
    C = Q.shape[0]
    print(f'[oracle_dir] {C} candidate directions; precompute QW...', flush=True)
    QW = (Q.astype(np.float64) @ W.T)                            # (C,NI)  q.W^T  shared over users/turns
    res = {'full': [], 'tail': []}
    for sd in SEEDS:
        ar = P.arena_seed(sd, rd_all); test = ar['test_users']
        zst = P.build_zstar(model, ar, test)
        t0 = time.time(); af = at = 0.0; mf = mt = 0
        for i, x in enumerate(test):
            profset, tst = ar['SPL'][x]; rd = ar['rat_by_u_dict'][x]
            rel = set(j for j in tst if rd[j] >= 4)
            if not rel:
                continue
            zs = zst[i]; nz = np.linalg.norm(zs) + 1e-9
            a_u = (Q.astype(np.float64) @ zs) / nz               # (C,) graded answer per direction
            Delta = ETA * a_u[:, None] * QW                     # (C,NI) score contribution of each dir
            Sbase = np.zeros(NI)                                # z=0 -> decode bias only... plus bias
            Sbase = Sbase + bdec                                # z0=0 => decode = bias
            for t in range(T):
                Scand = Sbase[None, :] + Delta                 # (C,NI)
                nd = ndcg_batch_full(Scand, profset, rel)
                c = int(np.argmax(nd))
                Sbase = Scand[c]                               # new decode = chosen candidate row
            # final decode = Sbase ; full+tail single-user ndcg
            nf = A.ndcg_at10(Sbase, rel, profset, ar['headmask'], False)
            ntl = A.ndcg_at10(Sbase, rel, profset, ar['headmask'], True)
            if nf is not None: af += nf; mf += 1
            if ntl is not None: at += ntl; mt += 1
        res['full'].append(af / max(mf, 1)); res['tail'].append(at / max(mt, 1))
        print(f'[oracle_dir] seed{sd} full {res["full"][-1]:.4f} tail {res["tail"][-1]:.4f} '
              f'[{time.time()-t0:.0f}s]', flush=True)
    out = {'n_candidates': C, 'select_metric': 'full_NDCG@10', 'repeats_allowed': True,
           'full': float(np.mean(res['full'])), 'tail': float(np.mean(res['tail'])),
           'full_sd': float(np.std(res['full'])), 'tail_sd': float(np.std(res['tail'])),
           'per_seed_full': res['full'], 'per_seed_tail': res['tail']}
    _save('oracle_dir', out); print(json.dumps(out, indent=2))


# ================================================================== RUNG 0b: item-subset fold oracle
def stage_oracle_item(seeds=None):
    seeds = seeds or SEEDS
    model, d = P.load_model()
    W = model.decoder.weight.detach().numpy().astype(np.float64)
    bdec = model.decoder.bias.detach().numpy().astype(np.float64)
    rd_all = {x: dict(v) for x, v in A.load_arena(seed=123)['rat_by_u'].items()}
    res = {'full': [], 'tail': []}
    for sd in seeds:
        ar = P.arena_seed(sd, rd_all); test = ar['test_users']
        t0 = time.time(); af = at = 0.0; mf = mt = 0
        for i, x in enumerate(test):
            profset, tst = ar['SPL'][x]; rd = ar['rat_by_u_dict'][x]
            rel = set(j for j in tst if rd[j] >= 4)
            if not rel:
                continue
            likes = [j for j in profset if rd[j] >= 4]            # answerable positive pool
            if not likes:
                nf = A.ndcg_at10(bdec, rel, profset, ar['headmask'], False)
                ntl = A.ndcg_at10(bdec, rel, profset, ar['headmask'], True)
                if nf is not None: af += nf; mf += 1
                if ntl is not None: at += ntl; mt += 1
                continue
            chosen = []; avail = list(likes)
            kmax = min(8, len(likes))
            for _ in range(kmax):
                # batch-encode bag(chosen + cand) for every remaining candidate
                B = np.zeros((len(avail), NI), np.float32)
                for r, cand in enumerate(avail):
                    B[r, chosen + [cand]] = 1.0
                Z = P.enc_mu(model, B)                            # (n_avail,d)
                S = Z.astype(np.float64) @ W.T + bdec            # (n_avail,NI)
                nd = ndcg_batch_full(S, profset, rel)
                bi = int(np.argmax(nd))
                chosen.append(avail.pop(bi))
            # final decode with the chosen subset
            b = np.zeros((1, NI), np.float32); b[0, chosen] = 1.0
            zf = P.enc_mu(model, b)[0]
            Sf = zf.astype(np.float64) @ W.T + bdec
            nf = A.ndcg_at10(Sf, rel, profset, ar['headmask'], False)
            ntl = A.ndcg_at10(Sf, rel, profset, ar['headmask'], True)
            if nf is not None: af += nf; mf += 1
            if ntl is not None: at += ntl; mt += 1
        res['full'].append(af / max(mf, 1)); res['tail'].append(at / max(mt, 1))
        print(f'[oracle_item] seed{sd} full {res["full"][-1]:.4f} tail {res["tail"][-1]:.4f} '
              f'[{time.time()-t0:.0f}s]', flush=True)
    out = {'pool': 'profile_likes', 'kmax': 8, 'select_metric': 'full_NDCG@10',
           'seeds': seeds, 'full': float(np.mean(res['full'])), 'tail': float(np.mean(res['tail'])),
           'full_sd': float(np.std(res['full'])), 'tail_sd': float(np.std(res['tail'])),
           'per_seed_full': res['full'], 'per_seed_tail': res['tail']}
    _save('oracle_item', out); print(json.dumps(out, indent=2))


# ================================================================== RUNG 1: Bayesian arm
def population_zstar_cov(model, rd_all):
    """diag + full population covariance of train-user z* (for prior + PCA directions)."""
    ar = A.load_arena(seed=123)
    trU = [x for x in ar['trU'] if sum(1 for j, r in rd_all[x].items() if r >= 4) >= 4]
    d = model.decoder.in_features
    Z = np.zeros((len(trU), d), np.float32)
    for st in range(0, len(trU), 500):
        ch = trU[st:st + 500]; Xd = np.zeros((len(ch), NI), np.float32)
        for r, x in enumerate(ch):
            lk = [j for j, rr in rd_all[x].items() if rr >= 4]; Xd[r, lk] = 1.0
        Z[st:st + len(ch)] = P.enc_mu(model, Xd)
    mu = Z.mean(0); Zc = Z - mu
    cov = (Zc.T @ Zc) / Zc.shape[0]
    return Z, mu.astype(np.float64), cov.astype(np.float64)


def make_prior(kind, model, cov_pop, scale, svd_all=None, sv=None):
    """Return prior covariance P0 (d,d) for the Kalman belief (prior mean = 0)."""
    d = cov_pop.shape[0]
    if kind == 'iso':
        return np.eye(d) * (scale ** 2 / d)
    if kind == 'pop_diag':
        return np.diag(np.diag(cov_pop))
    if kind == 'pop_full':
        return cov_pop.copy()
    if kind == 'decoder':
        # eigenvectors = decoder-SVD dirs, eigenvalues ~ singular-value^2 (informative metric)
        lam = (sv ** 2); lam = lam / lam.sum() * np.trace(cov_pop)
        return (svd_all.T * lam[None, :]) @ svd_all
    raise ValueError(kind)


def kalman_run(zst, dirs_fn, P0, R, scale, decode_fn, ar, users, model, W, bdec,
               answer_fn=None):
    """Run Kalman belief for a cohort. dirs_fn(t, mu, Pcur) -> unit q (d,).
    answer_fn(i, q) -> observation a (default graded cos). Returns (full, tail)."""
    d = P0.shape[0]
    af = at = 0.0; mf = mt = 0
    for i, x in enumerate(users):
        profset, tst = ar['SPL'][x]; rd = ar['rat_by_u_dict'][x]
        rel = set(j for j in tst if rd[j] >= 4)
        if not rel:
            continue
        zs = zst[i]; nz = np.linalg.norm(zs) + 1e-9
        mu = np.zeros(d); Pc = P0.copy()
        for t in range(T):
            q = dirs_fn(t, mu, Pc)                                # unit dir
            a = (zs @ q / nz) if answer_fn is None else answer_fn(i, q)
            y = a * scale                                        # linear obs y = q^T z + noise
            Pq = Pc @ q
            S = float(q @ Pq + R)
            K = Pq / S
            innov = y - float(q @ mu)
            mu = mu + K * innov
            Pc = Pc - np.outer(K, Pq)
        S = mu @ W.T + bdec
        nf = A.ndcg_at10(S, rel, profset, ar['headmask'], False)
        ntl = A.ndcg_at10(S, rel, profset, ar['headmask'], True)
        if nf is not None: af += nf; mf += 1
        if ntl is not None: at += ntl; mt += 1
    return af / max(mf, 1), at / max(mt, 1)


def dopt_sequence(P0, R, T):
    """D-optimal (== A-optimal here) direction sequence: top eigenvector of the running
    covariance. Independent of measurements => a FIXED sequence given (P0,R)."""
    Pc = P0.copy(); seq = []
    for t in range(T):
        w, V = np.linalg.eigh(Pc)
        q = V[:, -1]; q = q / (np.linalg.norm(q) + 1e-9)
        seq.append(q.astype(np.float64))
        Pq = Pc @ q; S = float(q @ Pq + R); K = Pq / S
        Pc = Pc - np.outer(K, Pq)
    return seq


def load_actor(d, name):
    blob = torch.load(f'{CK}/{name}.pt', map_location=DEVICE)
    a = P.Actor(d); a.load_state_dict(blob['state']); a.eval()
    return a, blob.get('best_val_tail')


def additive_run(zst, dirs_fn, ar, users, model, W, bdec, answer_fn=None):
    """Baseline additive operator z'=z+eta*a*q with a direction policy dirs_fn(t,z)->q."""
    d = W.shape[1]; af = at = 0.0; mf = mt = 0
    for i, x in enumerate(users):
        profset, tst = ar['SPL'][x]; rd = ar['rat_by_u_dict'][x]
        rel = set(j for j in tst if rd[j] >= 4)
        if not rel:
            continue
        zs = zst[i]; nz = np.linalg.norm(zs) + 1e-9
        z = np.zeros(d)
        for t in range(T):
            q = dirs_fn(t, z)
            a = (zs @ q / nz) if answer_fn is None else answer_fn(i, q)
            z = z + ETA * a * q
        S = z @ W.T + bdec
        nf = A.ndcg_at10(S, rel, profset, ar['headmask'], False)
        ntl = A.ndcg_at10(S, rel, profset, ar['headmask'], True)
        if nf is not None: af += nf; mf += 1
        if ntl is not None: at += ntl; mt += 1
    return af / max(mf, 1), at / max(mt, 1)


def actor_dir_fn(actor):
    def f(t, z):
        with torch.no_grad():
            q = actor(torch.tensor(z[None, :], dtype=torch.float32), t).numpy()[0]
        return q.astype(np.float64)
    return f


def collect_peruser_full(build_final, ar, users, zst, W, bdec):
    """Return dict user->full NDCG for bootstrap. build_final(i,zs)->z_final (d,)."""
    out = {}
    for i, x in enumerate(users):
        profset, tst = ar['SPL'][x]; rd = ar['rat_by_u_dict'][x]
        rel = set(j for j in tst if rd[j] >= 4)
        if not rel:
            continue
        z = build_final(i, zst[i])
        nf = A.ndcg_at10(z @ W.T + bdec, rel, profset, ar['headmask'], False)
        if nf is not None:
            out[x] = nf
    return out


def paired_boot(Ad, Bd, nboot=2000):
    keys = [k for k in Ad if k in Bd]
    da = np.array([Ad[k] for k in keys]); db = np.array([Bd[k] for k in keys])
    diff = da - db; n = len(diff); rng = np.random.default_rng(0)
    means = np.array([diff[rng.integers(0, n, n)].mean() for _ in range(nboot)])
    lo, hi = np.percentile(means, [2.5, 97.5])
    return {'mean_diff': float(diff.mean()), 'ci95': [float(lo), float(hi)],
            'p_gt0': float((means > 0).mean()), 'n_users': int(n)}


def stage_bayes(noisy=False):
    model, d = P.load_model()
    W = model.decoder.weight.detach().numpy().astype(np.float64)
    bdec = model.decoder.bias.detach().numpy().astype(np.float64)
    rd_all = {x: dict(v) for x, v in A.load_arena(seed=123)['rat_by_u'].items()}
    Zpop, mu_pop, cov_pop = population_zstar_cov(model, rd_all)
    scale = float(np.linalg.norm(Zpop, axis=1).mean())           # 17.2 population mean ||z*||
    svd_all = P.decoder_svd_dirs(model, d).astype(np.float64)    # informative basis (d dirs)
    Wd = model.decoder.weight.detach().numpy()
    Wc = Wd - Wd.mean(0, keepdims=True)
    sv = np.linalg.svd(Wc, full_matrices=False, compute_uv=False)[:d].astype(np.float64)
    svd8 = svd_all[:8]
    actor, _ = load_actor(d, 'p4a_actor_s0')

    # --- noisy channel setup (rung 1d) ---
    chan = None
    if noisy:
        chan = _load_channel()

    priors = ['iso', 'pop_diag', 'pop_full', 'decoder']
    # R swept on VAL (seed 1) for the D-optimal x each prior; pick best-full R
    Rgrid = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0]
    arv = P.arena_seed(1, rd_all); val = arv['val_users']; zstv = P.build_zstar(model, arv, val)
    bestR = {}
    for pk in priors:
        P0 = make_prior(pk, model, cov_pop, scale, svd_all, sv)
        bf = -1; bR = 1.0
        for R in Rgrid:
            seq = dopt_sequence(P0, R, T)
            f, _ = kalman_run(zstv, lambda t, mu, Pc, seq=seq: seq[t], P0, R, scale,
                              None, arv, val, model, W, bdec)
            if f > bf: bf = f; bR = R
        bestR[pk] = bR
        print(f'[bayes] prior {pk}: val-best R={bR} (val full {bf:.4f})', flush=True)

    def agg(lst):
        a = np.array(lst)
        return {'full': float(a[:, 0].mean()), 'tail': float(a[:, 1].mean()),
                'full_sd': float(a[:, 0].std()), 'tail_sd': float(a[:, 1].std())}

    # answer_fn factory for noisy channel (per-seed rng, per-user center)
    def make_answer_fn(ar, users, zst, seed):
        if not noisy:
            return lambda: None
        rng = np.random.default_rng(seed)
        um = _user_means(ar, users, rd_all)
        centers = np.array([um[u] for u in users])
        nzs = np.linalg.norm(zst, axis=1) + 1e-9

        def answer_fn(i, q):
            s = float(zst[i] @ q / nzs[i])
            a = _sample_channel(np.array([s]), chan, 1.0, rng, center=centers[i])[0]
            return a
        return answer_fn

    arms = {}
    # 2x2 decomposition + priors: (update x directions)
    keys = []
    for pk in priors:
        keys.append((f'kalman_dopt_{pk}', 'kalman', 'dopt', pk))
    keys += [('kalman_actor', 'kalman', 'actor', 'pop_full'),
             ('kalman_svd8', 'kalman', 'svd8', 'pop_full'),
             ('additive_actor', 'additive', 'actor', None),
             ('additive_svd8', 'additive', 'svd8', None),
             ('additive_dopt_decoder', 'additive', 'dopt', 'decoder'),
             ('additive_dopt_pop_full', 'additive', 'dopt', 'pop_full')]

    boot = {}                                                    # per-arm per-user full ndcg (seed-avg)
    for name, upd, dirs, pk in keys:
        arms[name] = []
        boot[name] = {}
    for sd in SEEDS:
        ar = P.arena_seed(sd, rd_all); test = ar['test_users']; zst = P.build_zstar(model, ar, test)
        afn = make_answer_fn(ar, test, zst, 1000 * sd)() if noisy else None
        for name, upd, dirs, pk in keys:
            if dirs == 'dopt':
                P0 = make_prior(pk, model, cov_pop, scale, svd_all, sv); R = bestR[pk]
                seq = dopt_sequence(P0, R, T)
                dfn = lambda t, a, b, seq=seq: seq[t]
            elif dirs == 'actor':
                dfn_add = actor_dir_fn(actor)
                dfn = lambda t, mu, Pc: dfn_add(t, mu)           # kalman feeds mu as state
            elif dirs == 'svd8':
                dfn = lambda t, a, b: svd8[t]
            if upd == 'kalman':
                P0 = make_prior(pk, model, cov_pop, scale, svd_all, sv); R = bestR.get(pk, 1.0)
                f, t = kalman_run(zst, dfn, P0, R, scale, None, ar, test, model, W, bdec,
                                  answer_fn=afn)

                def bf_k(i, zs, dfn=dfn, P0=P0, R=R, afn=afn, zst=zst):
                    mu = np.zeros(d); Pc = P0.copy(); nz = np.linalg.norm(zs) + 1e-9
                    for tt in range(T):
                        q = dfn(tt, mu, Pc)
                        a = (zs @ q / nz) if afn is None else afn(i, q)
                        y = a * scale; Pq = Pc @ q; Sd = float(q @ Pq + R); K = Pq / Sd
                        mu = mu + K * (y - float(q @ mu)); Pc = Pc - np.outer(K, Pq)
                    return mu
                bu = collect_peruser_full(bf_k, ar, test, zst, W, bdec)
            else:
                dfn2 = (actor_dir_fn(actor) if dirs == 'actor'
                        else (lambda t, z: svd8[t]) if dirs == 'svd8'
                        else (lambda t, z, seq=dopt_sequence(make_prior(pk, model, cov_pop, scale, svd_all, sv), bestR.get(pk, 1.0), T): seq[t]))
                f, t = additive_run(zst, dfn2, ar, test, model, W, bdec, answer_fn=afn)

                def bf_a(i, zs, dfn2=dfn2, afn=afn):
                    z = np.zeros(d); nz = np.linalg.norm(zs) + 1e-9
                    for tt in range(T):
                        q = dfn2(tt, z)
                        a = (zs @ q / nz) if afn is None else afn(i, q)
                        z = z + ETA * a * q
                    return z
                bu = collect_peruser_full(bf_a, ar, test, zst, W, bdec)
            arms[name].append((f, t))
            for u, v in bu.items():
                boot[name].setdefault(u, []).append(v)
        print(f'[bayes{"_noisy" if noisy else ""}] seed{sd} done', flush=True)

    out = {'scale': scale, 'bestR': bestR, 'arms': {k: agg(v) for k, v in arms.items()}}
    # collapse per-user seed lists to means for bootstrap
    bmean = {k: {u: float(np.mean(vs)) for u, vs in d_.items()} for k, d_ in boot.items()}
    # actor anchor per-user (additive+actor) is the reference
    ref = bmean['additive_actor']; refsvd = bmean['additive_svd8']
    out['bootstrap'] = {}
    for name in ['kalman_dopt_decoder', 'kalman_dopt_pop_full', 'kalman_actor', 'kalman_svd8']:
        out['bootstrap'][f'{name}_vs_additive_actor'] = paired_boot(bmean[name], ref)
        out['bootstrap'][f'{name}_vs_additive_svd8'] = paired_boot(bmean[name], refsvd)
    _save('bayes_noisy' if noisy else 'bayes', out)
    print(json.dumps(out, indent=2))


# ------------------------------------------------------------------ P4C channel helpers (inline)
def _load_channel():
    ch = json.load(open(CHAN))
    edges = np.array([(-np.inf if e is None and i == 0 else (np.inf if e is None else e))
                      for i, e in enumerate(ch['bin_edges'])], np.float64)
    edges[0] = -np.inf; edges[-1] = np.inf
    return {'edges': edges, 'bin_mean': np.array(ch['bin_mean_rating']),
            'bin_dist': np.array(ch['bin_dist_12345'])}


def _sample_channel(s, ch, noise_scale, rng, center=3.0):
    edges = ch['edges']; bin_mean = ch['bin_mean']; bin_dist = ch['bin_dist']
    NB = len(bin_mean)
    b = np.clip(np.digitize(s, edges) - 1, 0, NB - 1)
    U = len(s); r0 = np.zeros(U)
    for k in np.unique(b):
        idx = np.where(b == k)[0]
        r0[idx] = rng.choice([1, 2, 3, 4, 5], size=len(idx), p=bin_dist[k])
    bm = bin_mean[b]; r = bm + noise_scale * (r0 - bm)
    return np.clip((r - np.asarray(center)) / 2.0, -1.0, 1.0)


def _user_means(ar, users, rd_all):
    um = {}
    for x in users:
        profset, _ = ar['SPL'][x]; rd = rd_all[x]
        rr = [rd[j] for j in profset]
        um[x] = float(np.mean(rr)) if rr else 3.0
    return um


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', choices=['oracle_dir', 'oracle_item', 'bayes', 'bayes_noisy'])
    ap.add_argument('--seeds', type=str, default='')
    args = ap.parse_args()
    if args.stage == 'oracle_dir':
        stage_oracle_dir()
    elif args.stage == 'oracle_item':
        sds = [int(s) for s in args.seeds.split(',')] if args.seeds else None
        stage_oracle_item(sds)
    elif args.stage == 'bayes':
        stage_bayes(noisy=False)
    else:
        stage_bayes(noisy=True)


if __name__ == '__main__':
    main()
