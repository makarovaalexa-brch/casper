"""
squeeze_r3_belief.py -- INSTRUMENT 2.0 "squeeze arc" RUNG 3: decoder-metric Kalman belief
tracker + the MAGNITUDE-SHRINKAGE FIX that R1 diagnosed.

R1 finding (SQUEEZE_R01.md, RUNG 1): the Kalman posterior mean is a SHRINKAGE estimator that
under-inflates the latent magnitude. Because the decode is  S = mu @ W.T + bdec , a shrunk mu
lets the popularity bias `bdec` dominate the ranking -> Kalman loses 0.03-0.04 NDCG to the
additive operator, whose implicit "re-inflate the unit direction to ||z*||~=eta" does real work
that naive Bayesian fusion discards. Recommendation #4: re-inflate the posterior-mean estimate
to the natural ||z*||~=eta (scaled / MAP estimate) before decoding.

This rung:
  1. DECODER-METRIC belief: prior covariance lives in the decoder geometry (eigenvectors =
     decoder-SVD dirs, eigenvalues ~ singular-value^2) so greedy D-optimal probing selects the
     discriminative (decoder-SVD) directions, NOT the population/popularity axis (R1's 0.169
     collapse under the natural latent metric). [== make_prior('decoder') from squeeze_r01]
  2. MAGNITUDE FIX: at decode time, re-inflate the posterior mean. Ablation:
       none         raw posterior mean  mu                         (R1's shrinkage estimator)
       unit_scale   direction-preserving  mu -> scale * mu/||mu||   (snap to natural ||z*||)
       unit_tuned   mu -> tau*  * mu/||mu||   (tau* val-selected)   (snap to val-best norm)
       gain_tuned   uniform gain  mu -> g* * mu   (g* val-selected) (MAP-style de-shrink)
     The direction schedule is UNCHANGED by the fix (D-opt seq depends only on P0,R), so the fix
     is a pure decode-time transform -> collect final mu once per arm, decode under each setting.
  3. Applied to three belief arms: decoder-metric D-opt, actor-directed Kalman, SVD-8 Kalman
     -> does the magnitude fix let a corrected Bayesian belief tracker finally BEAT the additive
     operator (additive.actor 0.4901 / additive.SVD-8 0.4709)?
  4. Paired bootstrap vs additive.actor and additive.SVD-8; noisy-channel spot-check vs R1's noisy
     Kalman.D-opt 0.248 and R2's noise-adapted static_repeat 0.3143.

Frozen invariants: instrument .cache/instrument2/ml1m_recvae_d512_best.pt; operator z'=z+16*a*q,
z0=0; arena ml1m_arena; seeds {1,2,3,7,11}; te[300:] TEST (304 users); T=8; graded a=cos(z*,q);
NDCG@10 full + Cremonesi tail. Channel .cache/instrument2/p4c_channel.json.

Usage: OMP_NUM_THREADS=3 python scripts/instrument2/squeeze_r3_belief.py {clean|noisy|both}
"""
import os, sys, json, time, argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ml1m_arena as A
import p4a_battery as P
import squeeze_r01 as R1

DEVICE = torch.device('cpu'); torch.set_num_threads(3)
CK = '.cache/instrument2'
NI = 3706
ETA = 16.0
T = 8
SEEDS = [1, 2, 3, 7, 11]
OUT = f'{CK}/squeeze_r3_belief.json'

# anchors (R1, 5-seed TEST clean)
ANCH = {'additive_actor': (0.4901, 0.2973), 'additive_svd8': (0.4709, 0.2916),
        'kalman_dopt_decoder_raw_R1': (0.4787, 0.2916)}
# noisy anchors
ANCH_N = {'kalman_dopt_decoder_R1': (0.2480, 0.0944), 'additive_actor_R1': (0.1530, 0.0861),
          'r2_static_repeat': (0.3143, 0.1167), 'r2_noisy_actor': (0.2844, 0.1010)}


def _save(stage, obj):
    all_ = json.load(open(OUT)) if os.path.exists(OUT) else {}
    all_[stage] = obj
    json.dump(all_, open(OUT, 'w'), indent=2)
    print(f'[saved] {OUT} :: {stage}', flush=True)


# ----------------------------------------------------------------- Kalman: collect final mu
def kalman_collect(zst, dfn, P0, R, scale, ar, users, answer_fn=None, cmul=1.0):
    """Run the decoder-metric Kalman belief; RETURN per-user dict x -> (mu_final, rel, profset).
    dfn(t, mu, Pc) -> unit direction. Answer graded cos (or channel answer_fn). Decode is deferred
    to decode_eval so the magnitude fix can be ablated without re-running the belief."""
    d = P0.shape[0]
    out = {}
    for i, x in enumerate(users):
        profset, tst = ar['SPL'][x]; rd = ar['rat_by_u_dict'][x]
        rel = set(j for j in tst if rd[j] >= 4)
        if not rel:
            continue
        zs = zst[i]; nz = np.linalg.norm(zs) + 1e-9
        mu = np.zeros(d); Pc = P0.copy()
        for t in range(T):
            q = dfn(t, mu, Pc)
            a = (zs @ q / nz) if answer_fn is None else cmul * answer_fn(i, q)
            y = a * scale
            Pq = Pc @ q; S = float(q @ Pq + R); K = Pq / S
            mu = mu + K * (y - float(q @ mu))
            Pc = Pc - np.outer(K, Pq)
        out[x] = (mu.copy(), rel, profset)
    return out


def _reinf(mu, magfix, param):
    """Apply the magnitude fix to a posterior mean before decoding."""
    if magfix == 'none':
        return mu
    if magfix in ('unit_scale', 'unit_tuned'):
        return param * mu / (np.linalg.norm(mu) + 1e-9)
    if magfix == 'gain_tuned':
        return param * mu
    raise ValueError(magfix)


def decode_eval(mud, ar, W, bdec, magfix, param):
    """Decode every user's (fixed) posterior mean under a magnitude fix -> seed-cohort (full,tail)
    + per-user full dict (for bootstrap)."""
    af = at = 0.0; mf = mt = 0; per = {}
    for x, (mu, rel, profset) in mud.items():
        mh = _reinf(mu, magfix, param)
        S = mh @ W.T + bdec
        nf = A.ndcg_at10(S, rel, profset, ar['headmask'], False)
        ntl = A.ndcg_at10(S, rel, profset, ar['headmask'], True)
        if nf is not None: af += nf; mf += 1; per[x] = nf
        if ntl is not None: at += ntl; mt += 1
    return af / max(mf, 1), at / max(mt, 1), per


# ----------------------------------------------------------------- direction functions
def make_dfns(model, cov_pop, scale, svd_all, sv, actor, bestR):
    """Build the three belief arms' (P0, R, dfn) triples."""
    svd8 = svd_all[:8]
    P0_dec = R1.make_prior('decoder', model, cov_pop, scale, svd_all, sv)
    R_dec = bestR['decoder']
    seq_dec = R1.dopt_sequence(P0_dec, R_dec, T)
    adir = R1.actor_dir_fn(actor)
    P0_pop = R1.make_prior('pop_full', model, cov_pop, scale, svd_all, sv)
    arms = {
        'kalman_dopt_decoder': (P0_dec, R_dec, (lambda t, mu, Pc, s=seq_dec: s[t])),
        'kalman_actor':        (P0_pop, bestR['pop_full'], (lambda t, mu, Pc: adir(t, mu))),
        'kalman_svd8':         (P0_pop, bestR['pop_full'], (lambda t, mu, Pc, s=svd8: s[t])),
    }
    return arms


# ----------------------------------------------------------------- val tuning of the fix params
def tune_fix(mud_val, ar_val, W, bdec, scale):
    """Grid-select the reinflation params on the clean/ noisy VAL cohort (full NDCG)."""
    norms = [6, 8, 10, 12, 14, scale, 20, 24, 28, 32]
    gains = [1, 2, 4, 8, 16, 32, 64, 128]
    best_tau = scale; bf = -1
    for tau in norms:
        f, _, _ = decode_eval(mud_val, ar_val, W, bdec, 'unit_tuned', tau)
        if f > bf: bf = f; best_tau = tau
    best_g = 1.0; bg = -1
    for g in gains:
        f, _, _ = decode_eval(mud_val, ar_val, W, bdec, 'gain_tuned', g)
        if f > bg: bg = f; best_g = g
    return best_tau, best_g


# ----------------------------------------------------------------- main run
def run(noisy=False):
    tag = 'noisy' if noisy else 'clean'
    model, d = P.load_model()
    W = model.decoder.weight.detach().numpy().astype(np.float64)
    bdec = model.decoder.bias.detach().numpy().astype(np.float64)
    rd_all = {x: dict(v) for x, v in A.load_arena(seed=123)['rat_by_u'].items()}
    Zpop, mu_pop, cov_pop = R1.population_zstar_cov(model, rd_all)
    scale = float(np.linalg.norm(Zpop, axis=1).mean())
    svd_all = P.decoder_svd_dirs(model, d).astype(np.float64)
    Wd = model.decoder.weight.detach().numpy(); Wc = Wd - Wd.mean(0, keepdims=True)
    sv = np.linalg.svd(Wc, full_matrices=False, compute_uv=False)[:d].astype(np.float64)
    actor, _ = R1.load_actor(d, 'p4a_actor_s0')
    # R selected on VAL by R1 (identical invariants): clean decoder R=1, pop_full R=0.25;
    # noisy decoder R=32, pop_full R=1. Reuse R1's val-selected values.
    bestR = ({'decoder': 32.0, 'pop_full': 1.0} if noisy else {'decoder': 1.0, 'pop_full': 0.25})
    print(f'[{tag}] scale(mean||z*||)={scale:.3f}  bestR={bestR}', flush=True)

    chan = R1._load_channel() if noisy else None

    # --- VAL setup (seed 1) for tuning the magnitude fix params ---
    arv = P.arena_seed(1, rd_all); val = arv['val_users']; zstv = P.build_zstar(model, arv, val)
    val_afn = None; val_cmul = 1.0
    if noisy:
        _rngv = np.random.default_rng(99)
        _umv = R1._user_means(arv, val, rd_all); _cv = np.array([_umv[u] for u in val])
        _nzv = np.linalg.norm(zstv, axis=1) + 1e-9
        val_cmul = R1.compute_cmul(actor, zstv, chan, _cv, 99)

        def val_afn(i, q):
            s = float(zstv[i] @ q / _nzv[i])
            return R1._sample_channel(np.array([s]), chan, 1.0, _rngv, center=_cv[i])[0]

    arms_v = make_dfns(model, cov_pop, scale, svd_all, sv, actor, bestR)
    fixparam = {}
    for name, (P0, R, dfn) in arms_v.items():
        mud_v = kalman_collect(zstv, dfn, P0, R, scale, arv, val, answer_fn=val_afn, cmul=val_cmul)
        tau, g = tune_fix(mud_v, arv, W, bdec, scale)
        fixparam[name] = {'unit_tuned': float(tau), 'gain_tuned': float(g), 'unit_scale': scale}
        print(f'[{tag}] {name}: val-tuned tau={tau:.2f} gain={g:.1f}', flush=True)

    MAGFIX = ['none', 'unit_scale', 'unit_tuned', 'gain_tuned']
    # results[arm][magfix] = list of (full,tail) per seed ; boot[arm][magfix][user]=[full...]
    results = {a: {m: [] for m in MAGFIX} for a in arms_v}
    boot = {a: {m: {} for m in MAGFIX} for a in arms_v}
    # additive anchors recomputed per-user for a paired bootstrap on THIS run's cohorts
    add_boot = {'additive_actor': {}, 'additive_svd8': {}}  # user -> [per-seed full]
    add_res = {'additive_actor': [], 'additive_svd8': []}
    svd8 = svd_all[:8]

    for sd in SEEDS:
        t0 = time.time()
        ar = P.arena_seed(sd, rd_all); test = ar['test_users']; zst = P.build_zstar(model, ar, test)
        if noisy:
            _um = R1._user_means(ar, test, rd_all); _ctr = np.array([_um[u] for u in test])
            cmul = R1.compute_cmul(actor, zst, chan, _ctr, 1000 * sd)
        else:
            cmul = 1.0

        def fresh_afn():
            if not noisy:
                return None
            rng = np.random.default_rng(1000 * sd)
            um = R1._user_means(ar, test, rd_all); ctr = np.array([um[u] for u in test])
            nzs = np.linalg.norm(zst, axis=1) + 1e-9

            def afn(i, q):
                s = float(zst[i] @ q / nzs[i])
                return R1._sample_channel(np.array([s]), chan, 1.0, rng, center=ctr[i])[0]
            return afn

        arms = make_dfns(model, cov_pop, scale, svd_all, sv, actor, bestR)
        for name, (P0, R, dfn) in arms.items():
            mud = kalman_collect(zst, dfn, P0, R, scale, ar, test, answer_fn=fresh_afn(), cmul=cmul)
            for m in MAGFIX:
                param = fixparam[name].get(m, scale)
                f, t, per = decode_eval(mud, ar, W, bdec, m, param)
                results[name][m].append((f, t))
                for u, v in per.items():
                    boot[name][m].setdefault(u, []).append(v)

        # additive anchors on the same cohort/channel (CRN reset)
        for aname, dfn2 in [('additive_actor', R1.actor_dir_fn(actor)),
                            ('additive_svd8', (lambda t, z, s=svd8: s[t]))]:
            afn = fresh_afn()
            f, t = R1.additive_run(zst, dfn2, ar, test, model, W, bdec, answer_fn=afn, cmul=cmul)
            add_res[aname].append((f, t))
            afn = fresh_afn()

            def bf_a(i, zs, dfn2=dfn2, afn=afn, cmul=cmul):
                z = np.zeros(d); nz = np.linalg.norm(zs) + 1e-9
                for tt in range(T):
                    q = dfn2(tt, z)
                    a = (zs @ q / nz) if afn is None else cmul * afn(i, q)
                    z = z + ETA * a * q
                return z
            for u, v in R1.collect_peruser_full(bf_a, ar, test, zst, W, bdec).items():
                add_boot[aname].setdefault(u, []).append(v)
        print(f'[{tag}] seed{sd} done [{time.time()-t0:.0f}s]', flush=True)

    def agg(lst):
        a = np.array(lst)
        return {'full': float(a[:, 0].mean()), 'tail': float(a[:, 1].mean()),
                'full_sd': float(a[:, 0].std()), 'tail_sd': float(a[:, 1].std()),
                'per_seed_full': [float(x) for x in a[:, 0]]}

    out = {'scale': scale, 'bestR': bestR, 'fixparam': fixparam,
           'arms': {a: {m: agg(results[a][m]) for m in MAGFIX} for a in results},
           'additive': {k: agg(v) for k, v in add_res.items()},
           'anchors_R1': ANCH if not noisy else ANCH_N}

    # bootstrap: seed-mean per-user, additive anchors accumulated as single (last-seed overwrite fix):
    # recompute additive per-user as seed-mean
    # (add_boot above only kept last seed due to dict merge; recompute cleanly below)
    out['bootstrap'] = _bootstrap_block(boot, add_boot, fixparam, scale)
    _save('noisy' if noisy else 'clean', out)
    _print_table(tag, out)
    return out


def _bootstrap_block(boot, add_boot, fixparam, scale):
    """Paired bootstrap of each belief arm (best magfix = unit_scale + unit_tuned + gain_tuned +
    none) vs the additive anchors, seed-mean per user."""
    bb = {}
    ref_actor = {u: float(np.mean(v)) for u, v in add_boot['additive_actor'].items()}
    ref_svd = {u: float(np.mean(v)) for u, v in add_boot['additive_svd8'].items()}
    for arm in boot:
        for m in boot[arm]:
            bm = {u: float(np.mean(v)) for u, v in boot[arm][m].items()}
            bb[f'{arm}::{m}__vs_additive_actor'] = R1.paired_boot(bm, ref_actor)
            bb[f'{arm}::{m}__vs_additive_svd8'] = R1.paired_boot(bm, ref_svd)
    return bb


def _print_table(tag, out):
    print(f'\n================= RUNG 3 [{tag}] =================', flush=True)
    print(f'{"arm":24s} {"none":>16s} {"unit_scale":>16s} {"unit_tuned":>16s} {"gain_tuned":>16s}')
    for a in out['arms']:
        row = f'{a:24s}'
        for m in ['none', 'unit_scale', 'unit_tuned', 'gain_tuned']:
            v = out['arms'][a][m]
            row += f'  {v["full"]:.4f}/{v["tail"]:.4f}'
        print(row, flush=True)
    for k, v in out['additive'].items():
        print(f'{k:24s}  {v["full"]:.4f}/{v["tail"]:.4f}', flush=True)
    print('--- bootstrap (belief arm - additive anchor, full) ---', flush=True)
    for k, v in out['bootstrap'].items():
        if v['mean_diff'] > 0 or 'unit' in k or 'gain' in k:
            pass
    # print the headline comparisons only
    for arm in out['arms']:
        for m in ['unit_scale', 'unit_tuned', 'gain_tuned', 'none']:
            for anc in ['additive_actor', 'additive_svd8']:
                key = f'{arm}::{m}__vs_{anc}'
                v = out['bootstrap'][key]
                print(f'{key:52s} d={v["mean_diff"]:+.4f} CI[{v["ci95"][0]:+.4f},{v["ci95"][1]:+.4f}] '
                      f'p={v["p_gt0"]:.3f}', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['clean', 'noisy', 'both'])
    args = ap.parse_args()
    if args.mode in ('clean', 'both'):
        run(noisy=False)
    if args.mode in ('noisy', 'both'):
        run(noisy=True)


if __name__ == '__main__':
    main()
