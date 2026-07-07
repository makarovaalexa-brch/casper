"""
squeeze_r2.py -- INSTRUMENT 2.0 "squeeze arc" rung 2: the LINCHPIN noise-channel test.

Question: does the adaptive (learned) actor still WIN under the REALISTIC empirical answer
channel once the STATIC comparator is ALSO adapted to that noisy channel? P4C compared the
noise-trained actor (0.287) only to CLEAN-era static baselines (SVD-8 = 0.159). This rung builds
a fair NOISE-ADAPTED static -- a direction basis greedily forward-selected UNDER the deployment
channel (noisy val), with the repeat-probing option (re-probe an informative axis to average down
channel noise, the D-opt(decoder) R=32 trick from squeeze_r01) -- and re-runs the margin.

Frozen invariants: instrument .cache/instrument2/ml1m_recvae_d512_best.pt; operator z'=z+16*a*q,
z0=0; arena ml1m_arena byte-identical splits; eval seeds {1,2,3,7,11}; te[300:] TEST (304 users);
T=8; NDCG@10 full + Cremonesi tail; empirical channel .cache/instrument2/p4c_channel.json
(corr(s,rating)=0.516). Answers scale-matched + per-user centered exactly as p4c_answer_sources.py.

Noise-trained actors: p4c_actor_noisy_s{0,1,2}.pt (seed-avg). Static candidate pool = decoder-SVD
dirs (P.decoder_svd_dirs), the same informative basis squeeze_r01's D-opt(decoder) arm uses.

Usage: python scripts/instrument2/squeeze_r2.py
"""
import os, sys, json, time, argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ml1m_arena as A
import p4a_battery as P
import p4c_answer_sources as C4
from p4a_bootstrap import boot

DEVICE = torch.device('cpu'); torch.set_num_threads(4)
CK = '.cache/instrument2'
NI = 3706
ETA = 16.0
T = 8
SEEDS = [1, 2, 3, 7, 11]
NOISE = 1.0
POOL = 32                       # top-POOL decoder-SVD dirs = candidate pool for greedy select
VALSEED = 4242                  # fixed CRN seed for greedy val comparisons
OUT = f'{CK}/squeeze_r2_noise.json'

# channel (loaded once, module-global for the static two-pass helper)
_CH, EDGES, BIN_MEAN, BIN_DIST = C4.load_channel()


def load_noisy_actors(d):
    """The three noise-trained actors p4c_actor_noisy_s{0,1,2}.pt."""
    acts = {}
    for s in range(3):
        p = f'{CK}/p4c_actor_noisy_s{s}.pt'
        blob = torch.load(p, map_location=DEVICE)
        a = P.Actor(d); a.load_state_dict(blob['state']); a.eval()
        acts[f's{s}'] = (a, blob.get('best_val_tail'), blob.get('best_ep'))
    return acts


def run_static_noisy_peruser(model, W, bdec, ar, users, zst, schedule, rng, um):
    """Static direction schedule under the SAMPLED empirical channel, scale-matched two-pass
    (exactly mirrors C4.run_static_arm: collect s,a -> global c=rms(s)/rms(a) -> apply, then
    per-user NDCG). Returns (full, tail, c, peruser_full_dict)."""
    d = W.shape[1]
    seqs = []; s_all = []; a_all = []
    for i, x in enumerate(users):
        zs = zst[i]; nz = np.linalg.norm(zs) + 1e-9
        seq = []
        for t in range(len(schedule)):
            q = schedule[t]; s = float(zs @ q / nz)
            a = C4.sample_channel_a(np.array([s]), EDGES, BIN_MEAN, BIN_DIST, NOISE, rng,
                                    center=um[x])[0]
            seq.append((q, a)); s_all.append(s); a_all.append(a)
        seqs.append((i, x, seq))
    rs = np.sqrt(np.mean(np.square(s_all)) + 1e-12)
    ra = np.sqrt(np.mean(np.square(a_all)) + 1e-12)
    c = rs / ra if ra > 0 else 1.0
    U = len(users); Z = np.zeros((U, d), np.float32)
    for i, x, seq in seqs:
        z = np.zeros(d, np.float32)
        for (q, a) in seq:
            z = z + ETA * (c * a) * q
        Z[i] = z
    S = (Z @ W.T + bdec).astype(np.float64)
    af = at = 0.0; mf = mt = 0; pu = {}
    for i, x in enumerate(users):
        nf = C4.ndcg_user(S[i], ar, x, False); nt = C4.ndcg_user(S[i], ar, x, True)
        if nf is not None: af += nf; mf += 1; pu[x] = nf
        if nt is not None: at += nt; mt += 1
    return af / max(mf, 1), at / max(mt, 1), float(c), pu


def actor_noisy_peruser(actor, model, W, bdec, ar, users, zst, ans_fn, c):
    """run_actor_arm but returns (full, tail, peruser_full_dict)."""
    d = W.shape[1]; U = len(users)
    zt = zst; nz = np.linalg.norm(zt, axis=1) + 1e-9
    z = np.zeros((U, d), np.float32)
    for t in range(T):
        with torch.no_grad():
            q = actor(torch.tensor(z, dtype=torch.float32), t).numpy()
        s = (zt * q).sum(1) / nz
        a = ans_fn(t, q, s, users)
        z = z + ETA * (c * a)[:, None] * q
    S = (z @ W.T + bdec).astype(np.float64)
    af = at = 0.0; mf = mt = 0; pu = {}
    for i, x in enumerate(users):
        nf = C4.ndcg_user(S[i], ar, x, False); nt = C4.ndcg_user(S[i], ar, x, True)
        if nf is not None: af += nf; mf += 1; pu[x] = nf
        if nt is not None: at += nt; mt += 1
    return af / max(mf, 1), at / max(mt, 1), pu


# ================================================================== greedy noise-adapted static
def greedy_select(model, W, bdec, poolQ, allow_repeat):
    """Greedy forward-select an 8-direction schedule from poolQ, each slot chosen to MAXIMIZE
    full NDCG on the NOISY val cohort (te[:300], sampled empirical channel), CRN across candidates
    (rng reset to VALSEED per candidate). allow_repeat: a direction may be re-selected."""
    rd_all = {x: dict(v) for x, v in A.load_arena(seed=123)['rat_by_u'].items()}
    arv = P.arena_seed(1, rd_all); val = arv['val_users']
    zstv = P.build_zstar(model, arv, val)
    umv = C4.user_means(arv, val, rd_all)
    chosen_idx = []; chosen_dirs = []
    val_curve = []
    tag = 'repeat' if allow_repeat else 'distinct'
    for slot in range(T):
        best_j = -1; best_f = -1e9
        cand = range(poolQ.shape[0]) if allow_repeat else \
            [j for j in range(poolQ.shape[0]) if j not in chosen_idx]
        for j in cand:
            sched = chosen_dirs + [poolQ[j]]
            rng = np.random.default_rng(VALSEED)          # CRN: identical noise stream per cand
            f, _, _, _ = run_static_noisy_peruser(model, W, bdec, arv, val, zstv, sched, rng, umv)
            if f > best_f:
                best_f = f; best_j = j
        chosen_idx.append(best_j); chosen_dirs.append(poolQ[best_j])
        val_curve.append(best_f)
        print(f'[greedy {tag}] slot{slot} pick idx={best_j} val_full={best_f:.4f} '
              f'schedule={chosen_idx}', flush=True)
    return chosen_idx, np.array(chosen_dirs, np.float32), val_curve


# ================================================================== main (rung 2)
def run_r2():
    t0 = time.time()
    model, d = P.load_model()
    W = model.decoder.weight.detach().numpy().astype(np.float64)
    bdec = model.decoder.bias.detach().numpy().astype(np.float64)
    Wf = model.decoder.weight.detach().numpy(); bf = model.decoder.bias.detach().numpy()
    rd_all = {x: dict(v) for x, v in A.load_arena(seed=123)['rat_by_u'].items()}

    poolQ = P.decoder_svd_dirs(model, POOL)               # (POOL,d) candidate pool
    svd8_clean = P.decoder_svd_dirs(model, 8)             # clean-era static (top-8 SVD)
    nacts = load_noisy_actors(d)
    print(f'[r2] noisy actors: ' + ', '.join(
        f'{k} val_tail={v[1]:.4f}@ep{v[2]}' for k, v in nacts.items()), flush=True)

    res = {'invariants': {'eta': ETA, 'T': T, 'seeds': SEEDS, 'noise': NOISE, 'pool': POOL},
           'noisy_actor_ckpts': {k: {'best_val_tail': v[1], 'best_ep': v[2]}
                                 for k, v in nacts.items()}}

    # ---------- STEP 3: build noise-adapted static (greedy on noisy val) ----------
    print('\n=== greedy noise-adapted static (distinct-8) ===', flush=True)
    idx_d, sched_d, curve_d = greedy_select(model, W, bdec, poolQ, allow_repeat=False)
    print('\n=== greedy noise-adapted static (repeat-allowed) ===', flush=True)
    idx_r, sched_r, curve_r = greedy_select(model, W, bdec, poolQ, allow_repeat=True)
    res['greedy'] = {
        'distinct': {'schedule_idx': idx_d, 'val_curve': [float(x) for x in curve_d]},
        'repeat': {'schedule_idx': idx_r, 'val_curve': [float(x) for x in curve_r]}}
    val_win = 'repeat' if curve_r[-1] > curve_d[-1] else 'distinct'
    res['greedy']['val_winner'] = val_win
    print(f'[r2] val winner = {val_win} (distinct {curve_d[-1]:.4f} vs repeat {curve_r[-1]:.4f})',
          flush=True)
    _save(res)

    # ---------- STEP 2 + 3c: TEST eval, 5-seed, seed-avg ----------
    # per-user accumulators: x -> list of full-NDCG over (train-actor x eval-seed) / (eval-seed)
    pu_actor = {}; pu_static_d = {}; pu_static_r = {}; pu_svd8clean = {}
    perseed = {'actor': {'f': [], 't': []}, 'static_distinct': {'f': [], 't': []},
               'static_repeat': {'f': [], 't': []}, 'svd8_clean': {'f': [], 't': []}}
    for sd in SEEDS:
        ar = P.arena_seed(sd, rd_all); test = ar['test_users']
        zst = P.build_zstar(model, ar, test)
        um = C4.user_means(ar, test, rd_all)

        # --- actor seed-avg (3 noisy actors) under sampled channel; rng mirrors part1_trainnoisy ---
        rng_a = np.random.default_rng(sd)
        af = lambda t, q, s, users: C4.sample_channel_a(
            s, EDGES, BIN_MEAN, BIN_DIST, NOISE, rng_a, center=np.array([um[u] for u in users]))
        afs = []; ats = []; puA = {}
        for k in ['s0', 's1', 's2']:
            act = nacts[k][0]
            c = C4.native_scale_for_actor(act, model, Wf, bf, ar, test, zst, af)
            f, t, pu = actor_noisy_peruser(act, model, Wf, bf, ar, test, zst, af, c)
            afs.append(f); ats.append(t)
            for x, v in pu.items():
                puA.setdefault(x, []).append(v)
        perseed['actor']['f'].append(float(np.mean(afs)))
        perseed['actor']['t'].append(float(np.mean(ats)))
        for x, vs in puA.items():                          # avg over 3 train actors this seed
            pu_actor.setdefault(x, []).append(float(np.mean(vs)))

        # --- static arms (fresh rng per arm, same seed = common noise start) ---
        rng_sd = np.random.default_rng(sd)
        fd, td, _, pud = run_static_noisy_peruser(model, W, bdec, ar, test, zst, sched_d, rng_sd, um)
        perseed['static_distinct']['f'].append(fd); perseed['static_distinct']['t'].append(td)
        for x, v in pud.items(): pu_static_d.setdefault(x, []).append(v)

        rng_sr = np.random.default_rng(sd)
        fr, tr, _, pur = run_static_noisy_peruser(model, W, bdec, ar, test, zst, sched_r, rng_sr, um)
        perseed['static_repeat']['f'].append(fr); perseed['static_repeat']['t'].append(tr)
        for x, v in pur.items(): pu_static_r.setdefault(x, []).append(v)

        rng_sc = np.random.default_rng(sd)
        fc, tc, _, puc = run_static_noisy_peruser(model, W, bdec, ar, test, zst, svd8_clean, rng_sc, um)
        perseed['svd8_clean']['f'].append(fc); perseed['svd8_clean']['t'].append(tc)
        for x, v in puc.items(): pu_svd8clean.setdefault(x, []).append(v)

        print(f'[r2 TEST] seed{sd}: actor {perseed["actor"]["f"][-1]:.4f}/'
              f'{perseed["actor"]["t"][-1]:.4f} | static-d {fd:.4f}/{td:.4f} | '
              f'static-r {fr:.4f}/{tr:.4f} | svd8clean {fc:.4f}/{tc:.4f} '
              f'[{time.time()-t0:.0f}s]', flush=True)

    def agg(a):
        return {'full': float(np.mean(a['f'])), 'tail': float(np.mean(a['t'])),
                'full_sd': float(np.std(a['f'])), 'tail_sd': float(np.std(a['t'])),
                'per_seed_full': [float(x) for x in a['f']],
                'per_seed_tail': [float(x) for x in a['t']]}
    res['test'] = {k: agg(v) for k, v in perseed.items()}
    _save(res)

    # ---------- STEP 4: paired bootstrap (per-user full, seed-avg) ----------
    def seedavg(pu):
        return {x: float(np.mean(vs)) for x, vs in pu.items()}
    A_ = seedavg(pu_actor); Sd_ = seedavg(pu_static_d); Sr_ = seedavg(pu_static_r)
    Sc_ = seedavg(pu_svd8clean)
    static_adapt = Sr_ if val_win == 'repeat' else Sd_

    def paired(A1, B1):
        keys = [k for k in A1 if k in B1]
        return boot([A1[k] - B1[k] for k in keys])
    res['bootstrap'] = {
        'actor_minus_noiseadapted_static': paired(A_, static_adapt),
        'actor_minus_static_distinct': paired(A_, Sd_),
        'actor_minus_static_repeat': paired(A_, Sr_),
        'actor_minus_svd8_clean': paired(A_, Sc_)}
    _save(res)

    # ---------- summary ----------
    t = res['test']
    print('\n================ SQUEEZE R2 SUMMARY ================', flush=True)
    print(f"actor (noise-trained, seed-avg s0/s1/s2):   {t['actor']['full']:.4f} "
          f"(sd {t['actor']['full_sd']:.4f}) / tail {t['actor']['tail']:.4f}", flush=True)
    print(f"static distinct-8 (noise-adapted):          {t['static_distinct']['full']:.4f} / "
          f"tail {t['static_distinct']['tail']:.4f}", flush=True)
    print(f"static repeat-allowed (noise-adapted):      {t['static_repeat']['full']:.4f} / "
          f"tail {t['static_repeat']['tail']:.4f}", flush=True)
    print(f"svd8 clean-era static (context):            {t['svd8_clean']['full']:.4f} / "
          f"tail {t['svd8_clean']['tail']:.4f}", flush=True)
    b = res['bootstrap']['actor_minus_noiseadapted_static']
    print(f"\nActor - noise-adapted static ({val_win}): dfull={b['mean_diff']:+.4f} "
          f"CI{b['ci95']} p(>0)={b['p_gt0']:.3f}", flush=True)
    bc = res['bootstrap']['actor_minus_svd8_clean']
    print(f"Actor - clean-era SVD-8:                   dfull={bc['mean_diff']:+.4f} "
          f"CI{bc['ci95']} p(>0)={bc['p_gt0']:.3f}", flush=True)
    print(f"[r2] done [{time.time()-t0:.0f}s] -> {OUT}", flush=True)


def _save(obj):
    json.dump(obj, open(OUT, 'w'), indent=2)
    print(f'[saved] {OUT}', flush=True)


# ================================================================== rung 2b: tie-by-construction
def eval_actor_noisy(actor, model, Wf, bf, ar, users, zst, um, seed):
    """Actor under sampled empirical channel (native-scale-matched), returns (full, tail, pu_full)."""
    rng = np.random.default_rng(seed)
    af = lambda t, q, s, users: C4.sample_channel_a(
        s, EDGES, BIN_MEAN, BIN_DIST, NOISE, rng, center=np.array([um[u] for u in users]))
    c = C4.native_scale_for_actor(actor, model, Wf, bf, ar, users, zst, af)
    return actor_noisy_peruser(actor, model, Wf, bf, ar, users, zst, af, c)


def finetune_noisy(actor, model, d, alpha, beta, sigma, tseed=0, epochs=20):
    """Fine-tune a (schedule-warmed) actor under the channel's effective law in the unroll
    (a = alpha*s + beta + N(0,sigma), noise stop-grad) -- same recipe as part1_trainnoisy but
    starting from the given init. Val-SELECT on noisy val FULL NDCG (the R2 objective).
    Returns (best_state, best_val_full, best_val_tail_at_best, best_ep)."""
    W = torch.tensor(model.decoder.weight.detach().numpy(), dtype=torch.float32)
    bdc = torch.tensor(model.decoder.bias.detach().numpy(), dtype=torch.float32)
    ar123 = A.load_arena(seed=123); rd_all = {x: dict(v) for x, v in ar123['rat_by_u'].items()}
    trU = [x for x in ar123['trU'] if sum(1 for j, r in rd_all[x].items() if r >= 4) >= 4]
    Zstar = np.zeros((len(trU), d), np.float32)
    for st in range(0, len(trU), 500):
        chk = trU[st:st + 500]; Xd = np.zeros((len(chk), NI), np.float32)
        for r, x in enumerate(chk):
            lk = [j for j, rr in rd_all[x].items() if rr >= 4]; Xd[r, lk] = 1.0
        Zstar[st:st + len(chk)] = P.enc_mu(model, Xd)
    Zstar_t = torch.tensor(Zstar); znorm = Zstar_t.norm(dim=1, keepdim=True) + 1e-9
    with torch.no_grad():
        Steach = Zstar_t @ W.T + bdc
    topk = torch.topk(Steach, 20, dim=1).indices
    arv = P.arena_seed(1, rd_all); val = arv['val_users']
    zsv = torch.tensor(P.build_zstar(model, arv, val))

    torch.manual_seed(2000 + tseed); np.random.seed(2000 + tseed)
    opt = torch.optim.Adam(actor.parameters(), lr=3e-4)
    Ntr = len(trU); B = 256; rng = np.random.default_rng(100 + tseed)

    def val_noisy():
        actor.eval()
        with torch.no_grad():
            z = torch.zeros(len(val), d)
            for t in range(T):
                q = actor(z, t); s = (zsv * q).sum(1) / (zsv.norm(dim=1) + 1e-9)
                a = alpha * s + beta + sigma * torch.randn_like(s)
                z = z + ETA * a[:, None] * q
            S = (z @ W.T + bdc).numpy().astype(np.float64)
        af = at = 0.0; mf = mt = 0
        for i, x in enumerate(val):
            profset, tst = arv['SPL'][x]; rd = arv['rat_by_u_dict'][x]
            tl = set(j for j in tst if rd[j] >= 4)
            if not tl: continue
            nf = A.ndcg_at10(S[i], tl, profset, arv['headmask'], False)
            nt = A.ndcg_at10(S[i], tl, profset, arv['headmask'], True)
            if nf is not None: af += nf; mf += 1
            if nt is not None: at += nt; mt += 1
        return af / max(mf, 1), at / max(mt, 1)

    best_f = -1; best_t = -1; best_ep = -1
    best_state = {k: v.clone() for k, v in actor.state_dict().items()}
    t0 = time.time()
    for ep in range(epochs):
        actor.train(); perm = rng.permutation(Ntr)
        for st in range(0, Ntr, B):
            idx = perm[st:st + B]; zs = Zstar_t[idx]; nz = znorm[idx]
            z = torch.zeros(len(idx), d)
            for t in range(T):
                q = actor(z, t)
                s = (zs * q).sum(1, keepdim=True) / nz
                noise = sigma * torch.randn_like(s)
                a = alpha * s + beta + noise.detach()
                z = z + ETA * a * q
            cosT = (z * zs).sum(1) / (z.norm(dim=1) + 1e-9) / nz.squeeze(1)
            L_rec = (1.0 - cosT).mean()
            pos = topk[idx]; neg = torch.randint(0, NI, (len(idx), 108))
            cand = torch.cat([pos, neg], 1)
            sc = (z[:, None, :] * W[cand]).sum(2) + bdc[cand]
            rel = torch.zeros_like(sc); rel[:, :20] = 1.0
            L = L_rec + 0.3 * P.approx_ndcg_loss(sc, rel)
            opt.zero_grad(); L.backward(); opt.step()
        vf, vt = val_noisy()
        if vf > best_f:
            best_f = vf; best_t = vt; best_ep = ep
            best_state = {k: v.clone() for k, v in actor.state_dict().items()}
        print(f'[r2b-ft ep{ep:2d}] val full {vf:.4f} tail {vt:.4f} best_full {best_f:.4f}@{best_ep} '
              f'[{time.time()-t0:.0f}s]', flush=True)
    return best_state, best_f, best_t, best_ep


def run_r2b():
    t0 = time.time()
    model, d = P.load_model()
    W = model.decoder.weight.detach().numpy().astype(np.float64)
    bdec = model.decoder.bias.detach().numpy().astype(np.float64)
    Wf = model.decoder.weight.detach().numpy(); bf = model.decoder.bias.detach().numpy()
    rd_all = {x: dict(v) for x, v in A.load_arena(seed=123)['rat_by_u'].items()}
    poolQ = P.decoder_svd_dirs(model, POOL)

    res = _load()
    idx_r = res['greedy']['repeat']['schedule_idx']            # winning noise-adapted schedule
    static_repeat_test = res['test']['static_repeat']['full']
    print(f'[r2b] winning repeat schedule idx={idx_r}; static-repeat TEST full={static_repeat_test:.4f}',
          flush=True)
    # channel effective law (stored on the noisy actor ckpts)
    blob0 = torch.load(f'{CK}/p4c_actor_noisy_s0.pt', map_location=DEVICE)
    alpha, beta, sigma = blob0['alpha'], blob0['beta'], blob0['sigma']

    # ---- construct the tie actor: BC-warm to emit the fixed repeat schedule per turn ----
    sched_target = np.stack([poolQ[j] for j in idx_r]).astype(np.float32)   # (T,d)
    torch.manual_seed(0); np.random.seed(0)
    act_con = P.Actor(d); P.bc_warm(act_con, sched_target, d, steps=2000)
    # verify by-construction fidelity: emitted dir . target per turn (belief-independent probe)
    with torch.no_grad():
        zc = torch.zeros(4, d)
        fid = [float((act_con(zc, t)[0] @ torch.tensor(sched_target[t])).item()) for t in range(T)]
    print(f'[r2b] BC-warm per-turn cos-to-target = {[round(x,3) for x in fid]}', flush=True)

    # ---- optional fine-tune under the noisy channel on top of the construction ----
    torch.manual_seed(0); np.random.seed(0)
    act_ft = P.Actor(d); P.bc_warm(act_ft, sched_target, d, steps=2000)
    ft_state, ft_val_full, ft_val_tail, ft_ep = finetune_noisy(act_ft, model, d, alpha, beta, sigma)
    act_ft.load_state_dict(ft_state); act_ft.eval()
    torch.save({'state': ft_state, 'd': d, 'best_ep': ft_ep, 'best_val_full': ft_val_full,
                'best_val_tail': ft_val_tail, 'schedule_idx': idx_r,
                'alpha': alpha, 'beta': beta, 'sigma': sigma},
               f'{CK}/p4c_actor_tie_ft.pt')

    # ---- VAL (te[:300]) eval of both tie actors + static-repeat, for the by-construction check ----
    arv = P.arena_seed(1, rd_all); val = arv['val_users']
    zstv = P.build_zstar(model, arv, val); umv = C4.user_means(arv, val, rd_all)
    vf_con, vt_con, _ = eval_actor_noisy(act_con, model, Wf, bf, arv, val, zstv, umv, VALSEED)
    vf_ft, vt_ft, _ = eval_actor_noisy(act_ft, model, Wf, bf, arv, val, zstv, umv, VALSEED)
    rng_vs = np.random.default_rng(VALSEED)
    vf_st, vt_st, _, _ = run_static_noisy_peruser(model, W, bdec, arv, val, zstv, sched_target,
                                                  rng_vs, umv)
    print(f'[r2b VAL] construct-only {vf_con:.4f}/{vt_con:.4f} | finetuned {vf_ft:.4f}/{vt_ft:.4f} '
          f'| static-repeat {vf_st:.4f}/{vt_st:.4f}', flush=True)

    # ---- TEST 5-seed (same protocol/CRN as R2), per-user for bootstrap vs static-repeat ----
    sched_r = sched_target
    perseed = {'tie_construct': {'f': [], 't': []}, 'tie_finetune': {'f': [], 't': []},
               'static_repeat': {'f': [], 't': []}}
    pu_con = {}; pu_ft = {}; pu_st = {}
    for sd in SEEDS:
        ar = P.arena_seed(sd, rd_all); test = ar['test_users']
        zst = P.build_zstar(model, ar, test); um = C4.user_means(ar, test, rd_all)
        fc, tc, puc = eval_actor_noisy(act_con, model, Wf, bf, ar, test, zst, um, sd)
        ff, tf, puf = eval_actor_noisy(act_ft, model, Wf, bf, ar, test, zst, um, sd)
        rng_sr = np.random.default_rng(sd)
        fs, ts, _, pus = run_static_noisy_peruser(model, W, bdec, ar, test, zst, sched_r, rng_sr, um)
        perseed['tie_construct']['f'].append(fc); perseed['tie_construct']['t'].append(tc)
        perseed['tie_finetune']['f'].append(ff); perseed['tie_finetune']['t'].append(tf)
        perseed['static_repeat']['f'].append(fs); perseed['static_repeat']['t'].append(ts)
        for x, v in puc.items(): pu_con.setdefault(x, []).append(v)
        for x, v in puf.items(): pu_ft.setdefault(x, []).append(v)
        for x, v in pus.items(): pu_st.setdefault(x, []).append(v)
        print(f'[r2b TEST] seed{sd}: construct {fc:.4f}/{tc:.4f} | finetune {ff:.4f}/{tf:.4f} '
              f'| static-r {fs:.4f}/{ts:.4f} [{time.time()-t0:.0f}s]', flush=True)

    def agg(a):
        return {'full': float(np.mean(a['f'])), 'tail': float(np.mean(a['t'])),
                'full_sd': float(np.std(a['f'])), 'tail_sd': float(np.std(a['t'])),
                'per_seed_full': [float(x) for x in a['f']],
                'per_seed_tail': [float(x) for x in a['t']]}

    def seedavg(pu): return {x: float(np.mean(vs)) for x, vs in pu.items()}
    A_con = seedavg(pu_con); A_ft = seedavg(pu_ft); S_ = seedavg(pu_st)

    def paired(A1, B1):
        keys = [k for k in A1 if k in B1]
        return boot([A1[k] - B1[k] for k in keys])

    res['r2b'] = {
        'construction': {'schedule_idx': idx_r, 'bc_per_turn_cos': fid,
                         'finetune_best_ep': ft_ep, 'finetune_val_full': ft_val_full,
                         'finetune_val_tail': ft_val_tail},
        'val': {'tie_construct': {'full': vf_con, 'tail': vt_con},
                'tie_finetune': {'full': vf_ft, 'tail': vt_ft},
                'static_repeat': {'full': vf_st, 'tail': vt_st}},
        'test': {k: agg(v) for k, v in perseed.items()},
        'bootstrap': {
            'tie_construct_minus_static_repeat': paired(A_con, S_),
            'tie_finetune_minus_static_repeat': paired(A_ft, S_),
            'tie_finetune_minus_tie_construct': paired(A_ft, A_con)}}
    _save(res)

    tb = res['r2b']
    print('\n================ SQUEEZE R2b SUMMARY ================', flush=True)
    print(f"tie construct-only:  VAL {vf_con:.4f}/{vt_con:.4f}  TEST "
          f"{tb['test']['tie_construct']['full']:.4f}/{tb['test']['tie_construct']['tail']:.4f}", flush=True)
    print(f"tie fine-tuned:      VAL {vf_ft:.4f}/{vt_ft:.4f}  TEST "
          f"{tb['test']['tie_finetune']['full']:.4f}/{tb['test']['tie_finetune']['tail']:.4f}", flush=True)
    print(f"static-repeat (ref): VAL {vf_st:.4f}/{vt_st:.4f}  TEST "
          f"{tb['test']['static_repeat']['full']:.4f}/{tb['test']['static_repeat']['tail']:.4f}", flush=True)
    for nm in ['tie_construct_minus_static_repeat', 'tie_finetune_minus_static_repeat',
               'tie_finetune_minus_tie_construct']:
        b = tb['bootstrap'][nm]
        print(f"{nm}: dfull={b['mean_diff']:+.4f} CI{[round(x,4) for x in b['ci95']]} "
              f"p(>0)={b['p_gt0']:.3f}", flush=True)
    print(f"[r2b] done [{time.time()-t0:.0f}s]", flush=True)


def _load():
    return json.load(open(OUT)) if os.path.exists(OUT) else {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', nargs='?', default='r2', choices=['r2', 'r2b'])
    args = ap.parse_args()
    if args.stage == 'r2':
        run_r2()
    else:
        run_r2b()


if __name__ == '__main__':
    main()
