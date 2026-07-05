"""
p4c_answer_sources.py -- INSTRUMENT 2.0 Phase 4c: THE ANSWER-SOURCE ABLATION.

Closes the two non-human holes in the P4a flagship result:
  (H1) "perfect geometric answerer": P4a answers are a = cos(z*, q) -- a noiseless
       geometric oracle. Do the claims survive a REALISTIC answer channel fitted on
       real ML-1M ratings (quantization + noise + user bias)?  -> PART 1
  (H2) "u* circularity": the answer uses the SAME encoder geometry the instrument folds
       with. Do the claims survive answers from a FOREIGN geometry (V1 encoder / EASE)
       -> PART 2, and from RAW OBSERVED DATA with no model in the answer path (which of
       two items did the user actually rate higher) -> PART 3.

Instrument (frozen): .cache/instrument2/ml1m_recvae_d512_best.pt.
Operator (P3 winner): z' = z + eta*a*q, eta=16, z0=0 cold seed.
Arena: ml1m_arena (byte-identical V1 splits). Eval seeds {1,2,3,7,11}, te[300:] TEST.
Reuses P4a helpers + actors (p4a_actor_s{0,1,2}.pt) and ml1m_bars V1/EASE bars.

SCALE-MATCHING NOTE (load-bearing, stated in the writeup):
  The native geometric answer is literally a = cos(z*,q) = s. A foreign / empirical channel
  maps to a different raw scale (e.g. rating->[-1,1]); with a FIXED operator step eta this
  miscalibrates the update and would collapse purely from step-size, not information loss.
  We therefore rescale each foreign/empirical answer stream by a single global scalar so its
  cohort RMS matches the native s-RMS (== choosing eta for the channel). This isolates the
  answer's INFORMATION CONTENT from its raw scale. Raw-scale (unmatched) numbers are reported
  once to show the miscalibration, then all claim tables use scale-matched answers.

Stages:
  channel  -- PART 1a: fit + report the empirical P(rating|s) channel.
  part1    -- PART 1b: rerun key arms under SAMPLED empirical answers (+ noise x0.5/1/2,
              + train-noisy seed-0 retrain, + snap-loss on noisy actor).
  part2    -- PART 2: V1-geometry and EASE-percentile FOREIGN answers.
  part3    -- PART 3: RAW-DATA pair answers (which item rated higher), zero circularity.

Usage: python scripts/instrument2/p4c_answer_sources.py {channel|part1|part2|part3}
"""
import os, sys, json, time, argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ml1m_arena as A
import ml1m_bars as MB
import p4a_battery as P
from recvae import RecVAE

DEVICE = torch.device('cpu'); torch.set_num_threads(4)
CK = '.cache/instrument2'
NI = 3706
ETA = 16.0
T = 8
SEEDS = [1, 2, 3, 7, 11]
OUT = f'{CK}/p4c_answer_sources.json'
CHAN = f'{CK}/p4c_channel.json'


# ------------------------------------------------------------------ shared helpers
def load_actors(d):
    acts = {}
    for s in range(3):
        p = f'{CK}/p4a_actor_s{s}.pt'
        if os.path.exists(p):
            blob = torch.load(p, map_location=DEVICE)
            a = P.Actor(d); a.load_state_dict(blob['state']); a.eval()
            acts[f's{s}'] = (a, blob.get('best_val_tail'))
    return acts


def unit_rows(M):
    return M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-9)


def ndcg_user(S, ar, x, tail):
    profset, tst = ar['SPL'][x]; rd = ar['rat_by_u_dict'][x]
    tlike = set(j for j in tst if rd[j] >= 4)
    if not tlike:
        return None
    return A.ndcg_at10(S, tlike, profset, ar['headmask'], tail)


def cohort_ndcg(model, W, bdec, ar, users, Zfinal):
    """Zfinal: (U,d). Decode + NDCG full/tail seed-mean over cohort."""
    S = (Zfinal @ W.T + bdec).astype(np.float64)
    af = at = 0.0; mf = mt = 0
    for i, x in enumerate(users):
        nf = ndcg_user(S[i], ar, x, False); nt = ndcg_user(S[i], ar, x, True)
        if nf is not None: af += nf; mf += 1
        if nt is not None: at += nt; mt += 1
    return af / max(mf, 1), at / max(mt, 1)


# ================================================================== PART 1a: fit channel
def fit_channel():
    model, d = P.load_model()
    W = model.decoder.weight.detach().numpy()
    D = unit_rows(W)                                              # (NI,d) unit item dirs
    ar = A.load_arena(seed=123)
    rd_all = {x: dict(v) for x, v in ar['rat_by_u'].items()}
    trU = [x for x in ar['trU'] if sum(1 for j, r in rd_all[x].items() if r >= 4) >= 4]

    s_all = []; r_all = []
    user_means = []; user_stds = []
    for x in trU:
        rd = rd_all[x]
        lk = [j for j, r in rd.items() if r >= 4]
        z = P.enc_mu(model, P.bag_from_likes(lk)[None, :])[0]
        nz = np.linalg.norm(z) + 1e-9
        its = np.array(list(rd.keys())); rts = np.array([rd[j] for j in its], np.float32)
        s = (D[its] @ z) / nz                                    # cos(z*, d_i) for each rated item
        s_all.append(s); r_all.append(rts)
        user_means.append(rts.mean()); user_stds.append(rts.std())
    s_all = np.concatenate(s_all); r_all = np.concatenate(r_all)
    gmean = float(r_all.mean())
    corr = float(np.corrcoef(s_all, r_all)[0, 1])

    # quantile bins of s
    NB = 12
    qs = np.quantile(s_all, np.linspace(0, 1, NB + 1))
    qs[0] = -np.inf; qs[-1] = np.inf
    edges = qs
    b = np.clip(np.digitize(s_all, edges) - 1, 0, NB - 1)
    bin_mean = np.zeros(NB); bin_dist = np.zeros((NB, 5)); bin_ent = np.zeros(NB); bin_n = np.zeros(NB, int)
    for k in range(NB):
        m = b == k
        bin_n[k] = int(m.sum())
        rr = r_all[m]
        bin_mean[k] = rr.mean() if len(rr) else gmean
        h = np.array([(rr == v).sum() for v in [1, 2, 3, 4, 5]], np.float64)
        p = h / (h.sum() + 1e-12); bin_dist[k] = p
        bin_ent[k] = float(-(p * np.log2(p + 1e-12)).sum())

    ch = {
        'n_pairs': int(len(s_all)), 'n_users': len(trU),
        'global_mean_rating': gmean, 'corr_s_rating': corr,
        's_quantile_edges': [float(x) for x in edges[1:-1]],
        'bin_edges': [float(x) if np.isfinite(x) else None for x in edges],
        'bin_mean_rating': [float(x) for x in bin_mean],
        'bin_entropy_bits': [float(x) for x in bin_ent],
        'bin_dist_12345': [[float(y) for y in row] for row in bin_dist],
        'bin_n': [int(x) for x in bin_n],
        'user_mean_rating_meanstd': [float(np.mean(user_means)), float(np.std(user_means))],
        'user_dispersion_meanstd': [float(np.mean(user_stds)), float(np.std(user_stds))],
        'mean_abs_bin_entropy': float(bin_ent.mean()),
    }
    json.dump(ch, open(CHAN, 'w'), indent=2)
    print(json.dumps(ch, indent=2))
    print(f'\n[channel] corr(s,rating)={corr:.3f}  mean bin entropy={bin_ent.mean():.3f} bits '
          f'(max {np.log2(5):.3f})  global mean rating={gmean:.3f}', flush=True)
    return ch


def load_channel():
    ch = json.load(open(CHAN))
    edges = np.array([(-np.inf if e is None and i == 0 else (np.inf if e is None else e))
                      for i, e in enumerate(ch['bin_edges'])], np.float64)
    edges[0] = -np.inf; edges[-1] = np.inf
    return ch, edges, np.array(ch['bin_mean_rating']), np.array(ch['bin_dist_12345'])


def sample_channel_a(s, edges, bin_mean, bin_dist, noise_scale, rng, center=3.0):
    """s:(U,) -> graded answer a:(U,) in [-1,1] via empirical P(rating|bin(s)).
    `center` (scalar or (U,) array) = the user's rating baseline; the graded answer is the
    signed deviation (rating - center)/2. Per-user centering realizes the fitted user-bias law
    ("did you rate this above/below YOUR average") and feeds the additive operator a properly
    signed preference (absolute ratings carry a +0.29 offset the operator is not built for)."""
    NB = len(bin_mean)
    b = np.clip(np.digitize(s, edges) - 1, 0, NB - 1)
    U = len(s); r0 = np.zeros(U)
    for k in np.unique(b):
        idx = np.where(b == k)[0]
        r0[idx] = rng.choice([1, 2, 3, 4, 5], size=len(idx), p=bin_dist[k])
    bm = bin_mean[b]
    r = bm + noise_scale * (r0 - bm)
    a = np.clip((r - np.asarray(center)) / 2.0, -1.0, 1.0)
    return a


def user_means(ar, users, rd_all):
    """Per-user rating baseline from PROFILE ratings (observable, no target leak)."""
    um = {}
    for x in users:
        profset, _ = ar['SPL'][x]; rd = rd_all[x]
        rr = [rd[j] for j in profset]
        um[x] = float(np.mean(rr)) if rr else 3.0
    return um


# ================================================================== generic static-arm runner
def run_static_arm(model, W, bdec, ar, users, zst_np, get_dirs, a_fn, scale_match=True):
    """get_dirs(i,x,zs)->(Q8[K,d], items[K] or None). a_fn(x,t,q,item,zs,s)->raw a.
    Two-pass: collect s,a -> global scale c = rms(s)/rms(a) -> apply. Returns (full,tail,c)."""
    d = W.shape[1]
    seqs = []; s_all = []; a_all = []
    for i, x in enumerate(users):
        zs = zst_np[i]; nz = np.linalg.norm(zs) + 1e-9
        Q8, items = get_dirs(i, x, zs)
        seq = []
        for t in range(len(Q8)):
            q = Q8[t]; it = None if items is None else items[t]
            s = float(zs @ q / nz)
            a = a_fn(x, t, q, it, zs, s)
            seq.append((q, a)); s_all.append(s); a_all.append(a)
        seqs.append((i, x, seq))
    c = 1.0
    if scale_match and len(a_all):
        rs = np.sqrt(np.mean(np.square(s_all)) + 1e-12)
        ra = np.sqrt(np.mean(np.square(a_all)) + 1e-12)
        c = rs / ra if ra > 0 else 1.0
    Z = np.zeros((len(users), d), np.float32)
    for i, x, seq in seqs:
        z = np.zeros(d, np.float32)
        for (q, a) in seq:
            z = z + ETA * (c * a) * q
        Z[i] = z
    f, t = cohort_ndcg(model, W, bdec, ar, users, Z)
    return f, t, float(c)


# ================================================================== generic actor-arm runner
def run_actor_arm(actor, model, W, bdec, ar, users, zst_np, ans_fn, c=1.0,
                  realize=None, collect=False):
    """ans_fn(t,q[U,d],s[U],users)->a[U] (raw). realize(t,q[U,d],users)->q_use[U,d] (e.g. pair/item
    realization; None = use q). Applies scale c. Returns (full,tail, s_rec[T,U], a_rec[T,U], qused)."""
    d = W.shape[1]; U = len(users)
    zt = zst_np; nz = np.linalg.norm(zt, axis=1) + 1e-9
    z = np.zeros((U, d), np.float32)
    s_rec = np.zeros((T, U)); a_rec = np.zeros((T, U))
    for t in range(T):
        with torch.no_grad():
            q = actor(torch.tensor(z, dtype=torch.float32), t).numpy()
        q_use = q if realize is None else realize(t, q, users)
        s = (zt * q_use).sum(1) / nz
        a = ans_fn(t, q_use, s, users)
        s_rec[t] = s; a_rec[t] = a
        z = z + ETA * (c * a)[:, None] * q_use
    f, t2 = cohort_ndcg(model, W, bdec, ar, users, z)
    if collect:
        return f, t2, s_rec, a_rec
    return f, t2


def native_scale_for_actor(actor, model, W, bdec, ar, users, zst_np, emp_ans_fn, realize=None):
    """Run actor with NATIVE answers to get the trajectory; compute c = rms(s)/rms(a_emp along it)."""
    def native(t, q, s, users):
        return s
    _, _, s_rec, _ = run_actor_arm(actor, model, W, bdec, ar, users, zst_np, native,
                                   c=1.0, realize=realize, collect=True)
    # recompute empirical a along that native trajectory to size the scale
    # (rerun to capture q; cheap)
    d = W.shape[1]; U = len(users); zt = zst_np; nz = np.linalg.norm(zt, axis=1) + 1e-9
    z = np.zeros((U, d), np.float32); a_emp_all = []
    for t in range(T):
        with torch.no_grad():
            q = actor(torch.tensor(z, dtype=torch.float32), t).numpy()
        q_use = q if realize is None else realize(t, q, users)
        s = (zt * q_use).sum(1) / nz
        a_emp_all.append(emp_ans_fn(t, q_use, s, users))
        z = z + ETA * s[:, None] * q_use
    s_flat = s_rec.reshape(-1); a_flat = np.concatenate(a_emp_all)
    rs = np.sqrt(np.mean(s_flat ** 2) + 1e-12); ra = np.sqrt(np.mean(a_flat ** 2) + 1e-12)
    return rs / ra if ra > 0 else 1.0


# ------------------------------------------------------------------ direction designs (cached)
class Designs:
    def __init__(self, model, d):
        self.model = model; self.d = d
        self.svd8 = P.decoder_svd_dirs(model, 8)
        self.Qc, self.item_tag, self.tag_mass, self.names = P.concept_dirs(model)
        W = model.decoder.weight.detach().numpy()
        self.D = unit_rows(W)                                    # item decoder dirs (unit)

    def lift_concepts(self, x, ar, rd_all):
        profset, _ = ar['SPL'][x]; rd = rd_all[x]
        prof = [j for j in profset if rd[j] >= 4]
        aff = self.item_tag[prof].sum(0) if prof else np.zeros(self.item_tag.shape[1])
        lift = aff / (self.tag_mass + 1e-9)
        sel = [c for c in np.argsort(-lift)[:8] if lift[c] > 0]
        return sel, (self.Qc[sel] if sel else np.zeros((0, self.d), np.float32))

    def item8(self, x, ar, rd_all, rng):
        """8 profile like items (indices) + their unit decoder dirs."""
        profset, _ = ar['SPL'][x]; rd = rd_all[x]
        lk = [j for j in profset if rd[j] >= 4]
        if len(lk) > 8: lk = list(rng.choice(lk, 8, replace=False))
        return lk, (self.D[lk] if lk else np.zeros((0, self.d), np.float32))


# ================================================================== PART 1b: empirical channel arms
def part1():
    model, d = P.load_model()
    W = model.decoder.weight.detach().numpy(); bdec = model.decoder.bias.detach().numpy()
    dz = Designs(model, d)
    ch, edges, bin_mean, bin_dist = load_channel()
    rd_all = {x: dict(v) for x, v in A.load_arena(seed=123)['rat_by_u'].items()}
    actors = load_actors(d)
    prim = max(actors, key=lambda n: actors[n][1] if actors[n][1] is not None else -1)
    print(f'[part1] primary actor={prim}', flush=True)

    def emp_ans(noise, rng, um):
        return lambda t, q, s, users: sample_channel_a(
            s, edges, bin_mean, bin_dist, noise, rng, center=np.array([um[u] for u in users]))

    res = {}

    # ---- main arms under empirical channel (noise x1), seed-avg, 2 sampling reps ----
    for noise in [0.5, 1.0, 2.0]:
        tag = f'noise_x{noise}'
        arms = {'actor_seedavg': {'f': [], 't': [], 'c': []},
                'svd8_static': {'f': [], 't': [], 'c': []},
                'concept8_lift': {'f': [], 't': [], 'c': []},
                'item8_realrating_op': {'f': [], 't': [], 'c': []},
                'item8_fold_native': {'f': [], 't': [], 'c': []}}
        for sd in SEEDS:
            ar = P.arena_seed(sd, rd_all); test = ar['test_users']
            zst = P.build_zstar(model, ar, test)
            um = user_means(ar, test, rd_all)
            rng_item = np.random.default_rng(sd)
            for rep in range(1):                                 # one answer/user (realistic); 5 seeds avg
                rng = np.random.default_rng(1000 * sd + rep)
                af = emp_ans(noise, rng, um)
                # actor(s), seed-avg over 3 train seeds
                afs = []; ats = []; cs = []
                for nm, (act, _) in actors.items():
                    c = native_scale_for_actor(act, model, W, bdec, ar, test, zst, af)
                    f, t = run_actor_arm(act, model, W, bdec, ar, test, zst, af, c=c)
                    afs.append(f); ats.append(t); cs.append(c)
                arms['actor_seedavg']['f'].append(np.mean(afs))
                arms['actor_seedavg']['t'].append(np.mean(ats)); arms['actor_seedavg']['c'].append(np.mean(cs))

                # svd8 static
                f, t, c = run_static_arm(model, W, bdec, ar, test, zst,
                                         lambda i, x, zs: (dz.svd8, None),
                                         lambda x, t_, q, it, zs, s: sample_channel_a(
                                             np.array([s]), edges, bin_mean, bin_dist, noise, rng, center=um[x])[0])
                arms['svd8_static']['f'].append(f); arms['svd8_static']['t'].append(t); arms['svd8_static']['c'].append(c)

                # concept8 lift
                def cget(i, x, zs):
                    sel, Q = dz.lift_concepts(x, ar, rd_all); return Q, None
                f, t, c = run_static_arm(model, W, bdec, ar, test, zst, cget,
                                         lambda x, t_, q, it, zs, s: sample_channel_a(
                                             np.array([s]), edges, bin_mean, bin_dist, noise, rng, center=um[x])[0])
                arms['concept8_lift']['f'].append(f); arms['concept8_lift']['t'].append(t); arms['concept8_lift']['c'].append(c)

                # item8 with REAL ratings via operator (fully-real answer path)
                def iget(i, x, zs):
                    lk, Q = dz.item8(x, ar, rd_all, rng_item); return Q, lk
                def irat(x, t_, q, it, zs, s):
                    r = rd_all[x][it]; return np.clip((r - um[x]) / 2.0, -1, 1)
                f, t, c = run_static_arm(model, W, bdec, ar, test, zst, iget, irat)
                arms['item8_realrating_op']['f'].append(f); arms['item8_realrating_op']['t'].append(t)
                arms['item8_realrating_op']['c'].append(c)

                # item8 native fold reference (no answer channel; encode bag)
                def ifold(i, x, zs):
                    lk, _ = dz.item8(x, ar, rd_all, rng_item)
                    if not lk:
                        return np.zeros(d, np.float32)
                    return P.enc_mu(model, P.bag_from_likes(lk)[None, :])[0]
                Z = np.zeros((len(test), d), np.float32)
                for i, x in enumerate(test): Z[i] = ifold(i, x, zst[i])
                ff, tt = cohort_ndcg(model, W, bdec, ar, test, Z)
                arms['item8_fold_native']['f'].append(ff); arms['item8_fold_native']['t'].append(tt)
                arms['item8_fold_native']['c'].append(1.0)
            print(f'[part1 {tag}] seed{sd} done', flush=True)
        res[tag] = {k: {'full': float(np.mean(v['f'])), 'tail': float(np.mean(v['t'])),
                        'full_sd': float(np.std(v['f'])), 'scale_c': float(np.mean(v['c']))}
                    for k, v in arms.items()}
        _save('part1', res)

    # ---- binary answers on empirical channel (graded>binary check) ----
    bin_arms = {'actor_binary': [], 'svd8_binary': []}
    for sd in SEEDS:
        ar = P.arena_seed(sd, rd_all); test = ar['test_users']; zst = P.build_zstar(model, ar, test)
        um = user_means(ar, test, rd_all)
        rng = np.random.default_rng(sd)
        def af_bin(t, q, s, users):
            a = sample_channel_a(s, edges, bin_mean, bin_dist, 1.0, rng,
                                 center=np.array([um[u] for u in users])); return np.sign(a)
        c = native_scale_for_actor(actors[prim][0], model, W, bdec, ar, test, zst, af_bin)
        f, t = run_actor_arm(actors[prim][0], model, W, bdec, ar, test, zst, af_bin, c=c)
        bin_arms['actor_binary'].append((f, t))
        fs, ts, cs = run_static_arm(model, W, bdec, ar, test, zst, lambda i, x, zs: (dz.svd8, None),
                                    lambda x, t_, q, it, zs, s: np.sign(sample_channel_a(
                                        np.array([s]), edges, bin_mean, bin_dist, 1.0, rng, center=um[x])[0]))
        bin_arms['svd8_binary'].append((fs, ts))
    res['binary_channel'] = {k: {'full': float(np.mean([a for a, _ in v])),
                                 'tail': float(np.mean([b for _, b in v]))} for k, v in bin_arms.items()}
    _save('part1', res)

    # ---- snap-loss on the empirical-channel actor (noise x1) ----
    Wn = dz.D; Qc = dz.Qc
    snap = {'unsnapped': [], 'concept_snap': [], 'item_snap': []}
    for sd in SEEDS:
        ar = P.arena_seed(sd, rd_all); test = ar['test_users']; zst = P.build_zstar(model, ar, test)
        um = user_means(ar, test, rd_all)
        rng = np.random.default_rng(sd)
        af = emp_ans(1.0, rng, um)
        act = actors[prim][0]
        c = native_scale_for_actor(act, model, W, bdec, ar, test, zst, af)
        f, t = run_actor_arm(act, model, W, bdec, ar, test, zst, af, c=c)
        snap['unsnapped'].append((f, t))
        for bank, key in [(Qc, 'concept_snap'), (Wn, 'item_snap')]:
            def realize(t_, q, users, bank=bank):
                sims = q @ bank.T; return bank[sims.argmax(1)]
            c2 = native_scale_for_actor(act, model, W, bdec, ar, test, zst, af, realize=realize)
            fs, ts = run_actor_arm(act, model, W, bdec, ar, test, zst, af, c=c2, realize=realize)
            snap[key].append((fs, ts))
    def m(v, j): return float(np.mean([x[j] for x in v]))
    res['snap_loss_noisy'] = {
        'unsnapped': {'full': m(snap['unsnapped'], 0), 'tail': m(snap['unsnapped'], 1)},
        'concept_snap': {'full': m(snap['concept_snap'], 0), 'tail': m(snap['concept_snap'], 1)},
        'item_snap': {'full': m(snap['item_snap'], 0), 'tail': m(snap['item_snap'], 1)},
        'delta_concept_full': m(snap['concept_snap'], 0) - m(snap['unsnapped'], 0),
        'delta_concept_tail': m(snap['concept_snap'], 1) - m(snap['unsnapped'], 1),
        'delta_item_full': m(snap['item_snap'], 0) - m(snap['unsnapped'], 0),
        'delta_item_tail': m(snap['item_snap'], 1) - m(snap['unsnapped'], 1)}
    _save('part1', res)
    print(json.dumps(res, indent=2))
    return res


# ================================================================== PART 1c: train-noisy retrain
def part1_trainnoisy(tseed=0):
    """Retrain seed-0 actor with the empirical channel's EFFECTIVE law in the unroll:
    a = alpha*s + beta + N(0,sigma) (differentiable in s via the linear response; noise stop-grad).
    Tests train-noisy vs eval-noisy. Saves p4c_actor_noisy_s{seed}.pt, then eval reruns pick it up."""
    model, d = P.load_model()
    ch, edges, bin_mean, bin_dist = load_channel()
    # linear response E[a|s] and residual sigma from the fitted channel
    NB = len(bin_mean)
    gmean = ch['global_mean_rating']
    bin_a = (bin_mean - gmean) / 2.0                             # centered response (matches eval centering)
    centers = np.array([(edges[k] + edges[k + 1]) / 2 for k in range(NB)])
    centers[0] = edges[1] - 0.05; centers[-1] = edges[-2] + 0.05
    Wc = np.array(ch['bin_n'], np.float64)
    A_ = np.polyfit(centers, bin_a, 1, w=Wc)                     # [alpha, beta]
    alpha, beta = float(A_[0]), float(A_[1])
    # residual sigma of a around linear fit, weighted
    var = 0.0
    for k in range(NB):
        p = bin_dist[k]; av = (np.array([1, 2, 3, 4, 5]) - 3) / 2
        mean_a = (p * av).sum(); ev = (p * (av - mean_a) ** 2).sum()
        var += Wc[k] * ev
    sigma = float(np.sqrt(var / Wc.sum()))
    print(f'[trainnoisy] linear response a~={alpha:.3f}*s+{beta:.3f}, channel sigma_a={sigma:.3f}', flush=True)

    W = torch.tensor(model.decoder.weight.detach().numpy(), dtype=torch.float32)
    bdc = torch.tensor(model.decoder.bias.detach().numpy(), dtype=torch.float32)
    svd8 = P.decoder_svd_dirs(model, 8)
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

    torch.manual_seed(1000 + tseed); np.random.seed(1000 + tseed)
    actor = P.Actor(d); P.bc_warm(actor, svd8, d)
    opt = torch.optim.Adam(actor.parameters(), lr=3e-4)
    Ntr = len(trU); B = 256; EPOCHS = 20
    rng = np.random.default_rng(tseed)
    # val under the SAME noisy law (eval-noisy val for selection)
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
    best_val = -1; best_state = {k: v.clone() for k, v in actor.state_dict().items()}; best_ep = -1
    t0 = time.time()
    for ep in range(EPOCHS):
        actor.train(); perm = rng.permutation(Ntr)
        for st in range(0, Ntr, B):
            idx = perm[st:st + B]; zs = Zstar_t[idx]; nz = znorm[idx]
            z = torch.zeros(len(idx), d)
            for t in range(T):
                q = actor(z, t)
                s = (zs * q).sum(1, keepdim=True) / nz
                noise = sigma * torch.randn_like(s)               # stop-grad sample
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
        if vt > best_val:
            best_val = vt; best_ep = ep; best_state = {k: v.clone() for k, v in actor.state_dict().items()}
        print(f'[trainnoisy s{tseed}] ep{ep:2d} val full {vf:.4f} tail {vt:.4f} best {best_val:.4f}@{best_ep} '
              f'[{time.time()-t0:.0f}s]', flush=True)
    torch.save({'state': best_state, 'd': d, 'best_ep': best_ep, 'best_val_tail': best_val,
                'alpha': alpha, 'beta': beta, 'sigma': sigma},
               f'{CK}/p4c_actor_noisy_s{tseed}.pt')
    print(f'[trainnoisy] saved p4c_actor_noisy_s{tseed}.pt', flush=True)

    # eval: train-noisy actor vs eval-noisy BC-warm actor, both under sampled empirical channel
    W_np = model.decoder.weight.detach().numpy(); b_np = model.decoder.bias.detach().numpy()
    na = P.Actor(d); na.load_state_dict(best_state); na.eval()
    bc = load_actors(d)['s0'][0]
    ev = {'trainnoisy': [], 'bcwarm_evalnoisy': []}
    for sd in SEEDS:
        ar = P.arena_seed(sd, rd_all); test = ar['test_users']; zst = P.build_zstar(model, ar, test)
        um = user_means(ar, test, rd_all)
        rng = np.random.default_rng(sd)
        af = lambda t, q, s, users: sample_channel_a(s, edges, bin_mean, bin_dist, 1.0, rng,
                                                     center=np.array([um[u] for u in users]))
        for key, act in [('trainnoisy', na), ('bcwarm_evalnoisy', bc)]:
            c = native_scale_for_actor(act, model, W_np, b_np, ar, test, zst, af)
            f, t = run_actor_arm(act, model, W_np, b_np, ar, test, zst, af, c=c)
            ev[key].append((f, t))
    out = {k: {'full': float(np.mean([a for a, _ in v])), 'tail': float(np.mean([b for _, b in v]))}
           for k, v in ev.items()}
    out['channel_response'] = {'alpha': alpha, 'beta': beta, 'sigma': sigma}
    res = _load('part1'); res['train_noisy'] = out; _save('part1', res)
    print(json.dumps(out, indent=2))
    return out


# ================================================================== PART 2: foreign geometry
def part2():
    model, d = P.load_model()
    W = model.decoder.weight.detach().numpy(); bdec = model.decoder.bias.detach().numpy()
    dz = Designs(model, d)
    rd_all = {x: dict(v) for x, v in A.load_arena(seed=123)['rat_by_u'].items()}
    actors = load_actors(d)
    prim = max(actors, key=lambda n: actors[n][1] if actors[n][1] is not None else -1)

    # V1 geometry
    ar123 = A.load_arena(seed=123)
    Qv1, Qlv1, resid, enc_u = MB.load_v1(ar123)
    QlN = unit_rows(Qlv1)                                         # (NI,64) unit item scoring dirs
    # V1 concept dirs (normalized mean of Ql over member items), aligned to dz.Qc concept order
    Mbag = 50
    v1conc = np.zeros((dz.item_tag.shape[1], 64), np.float32)
    for c in range(dz.item_tag.shape[1]):
        rel = dz.item_tag[:, c]; top = np.argpartition(-rel, Mbag)[:Mbag]
        v = Qlv1[top].mean(0); v1conc[c] = v / (np.linalg.norm(v) + 1e-9)

    # EASE B (val-selected lam=1000 per ml1m_bars)
    X = MB.build_like_matrix(ar123, ar123['trU'])
    Bease = MB.ease_B(X, 1000.0)

    Dunit = dz.D
    res = {}

    for sd in SEEDS:
        ar = P.arena_seed(sd, rd_all); test = ar['test_users']; zst = P.build_zstar(model, ar, test)
        # V1 user vectors u* (profile-half) + EASE percentile score arrays per user
        uV1 = np.zeros((len(test), 64), np.float32); pe = {}
        for i, x in enumerate(test):
            profset, _ = ar['SPL'][x]
            u = enc_u([(Qv1[j], resid[x][j]) for j in profset])
            uV1[i] = u / (np.linalg.norm(u) + 1e-9)
            lk = [j for j in profset if rd_all[x][j] >= 4]
            esc = Bease[lk].sum(0) if lk else np.zeros(NI)
            order = np.argsort(esc); pct = np.empty(NI); pct[order] = np.linspace(0, 1, NI)
            pe[x] = pct
        idx_of = {x: i for i, x in enumerate(test)}

        def v1_item_ans(x, item):
            return float(uV1[idx_of[x]] @ QlN[item])

        def ease_item_ans(x, item):
            return float(2 * pe[x][item] - 1)

        seed_res = {}
        # --- V1 answers: item-8, concept-8, basis-8(item-realized), actor(item-realized) ---
        # item-8 (operator, real item dirs, V1 answer)
        rng_item = np.random.default_rng(sd)
        def iget(i, x, zs):
            lk, Q = dz.item8(x, ar, rd_all, rng_item); return Q, lk
        f, t, c = run_static_arm(model, W, bdec, ar, test, zst, iget,
                                 lambda x, t_, q, it, zs, s: v1_item_ans(x, it))
        seed_res['V1_item8'] = (f, t, c)
        # concept-8 (operator, I2 concept dirs, V1 concept answer)
        def cget(i, x, zs):
            sel, Q = dz.lift_concepts(x, ar, rd_all); return Q, sel
        def v1_conc_ans(x, t_, q, sel_c, zs, s):
            return float(uV1[idx_of[x]] @ v1conc[sel_c])
        f, t, c = run_static_arm(model, W, bdec, ar, test, zst, cget, v1_conc_ans)
        seed_res['V1_concept8'] = (f, t, c)
        # basis-8 static, ITEM-REALIZED: snap each basis dir to nearest I2 item row, V1 answer
        snap_item_of = Dunit[np.argmax(dz.svd8 @ Dunit.T, axis=1)]  # not needed; realize per dir
        basis_items = np.argmax(dz.svd8 @ Dunit.T, axis=1)          # (8,)
        def bget(i, x, zs):
            return Dunit[basis_items], list(basis_items)             # use item dirs of realized items
        f, t, c = run_static_arm(model, W, bdec, ar, test, zst, bget,
                                 lambda x, t_, q, it, zs, s: v1_item_ans(x, it))
        seed_res['V1_basis8_itemreal'] = (f, t, c)
        # actor item-realized: snap each q_t to nearest item row, V1 answer for that item
        def realize_item(t_, q, users):
            it = np.argmax(q @ Dunit.T, axis=1); return Dunit[it]
        def actor_v1_ans(t_, q_use, s, users):
            it = np.argmax(q_use @ Dunit.T, axis=1)
            return np.array([v1_item_ans(users[u], it[u]) for u in range(len(users))])
        act = actors[prim][0]
        c = native_scale_for_actor(act, model, W, bdec, ar, test, zst, actor_v1_ans, realize=realize_item)
        f, t = run_actor_arm(act, model, W, bdec, ar, test, zst, actor_v1_ans, c=c, realize=realize_item)
        seed_res['V1_actor_itemreal'] = (f, t, c)

        # --- EASE answers (item questions only): item-8, basis-8(item-real), actor(item-real) ---
        f, t, c = run_static_arm(model, W, bdec, ar, test, zst, iget,
                                 lambda x, t_, q, it, zs, s: ease_item_ans(x, it))
        seed_res['EASE_item8'] = (f, t, c)
        f, t, c = run_static_arm(model, W, bdec, ar, test, zst, bget,
                                 lambda x, t_, q, it, zs, s: ease_item_ans(x, it))
        seed_res['EASE_basis8_itemreal'] = (f, t, c)
        def actor_ease_ans(t_, q_use, s, users):
            it = np.argmax(q_use @ Dunit.T, axis=1)
            return np.array([ease_item_ans(users[u], it[u]) for u in range(len(users))])
        c = native_scale_for_actor(act, model, W, bdec, ar, test, zst, actor_ease_ans, realize=realize_item)
        f, t = run_actor_arm(act, model, W, bdec, ar, test, zst, actor_ease_ans, c=c, realize=realize_item)
        seed_res['EASE_actor_itemreal'] = (f, t, c)

        for k, (f, t, c) in seed_res.items():
            res.setdefault(k, {'f': [], 't': [], 'c': []})
            res[k]['f'].append(f); res[k]['t'].append(t); res[k]['c'].append(c)
        print(f'[part2] seed{sd} done', flush=True)

    out = {k: {'full': float(np.mean(v['f'])), 'tail': float(np.mean(v['t'])),
               'full_sd': float(np.std(v['f'])), 'scale_c': float(np.mean(v['c']))}
           for k, v in res.items()}
    _save('part2', out)
    print(json.dumps(out, indent=2))
    return out


# ================================================================== PART 3: raw-data pairs
def pair_snap_profile(q, cand_items, D):
    """Best pair (i,j) among cand_items maximizing cos(q, D[i]-D[j]). Returns (i,j,dir_unit)."""
    if len(cand_items) < 2:
        return None
    Di = D[cand_items]                                            # (m,d)
    proj = Di @ q                                                 # (m,)
    i = int(np.argmax(proj)); j = int(np.argmin(proj))
    if i == j:
        return None
    diff = D[cand_items[i]] - D[cand_items[j]]
    n = np.linalg.norm(diff) + 1e-9
    return cand_items[i], cand_items[j], diff / n


def part3():
    model, d = P.load_model()
    W = model.decoder.weight.detach().numpy(); bdec = model.decoder.bias.detach().numpy()
    dz = Designs(model, d); Dunit = dz.D
    rd_all = {x: dict(v) for x, v in A.load_arena(seed=123)['rat_by_u'].items()}
    actors = load_actors(d)
    prim = max(actors, key=lambda n: actors[n][1] if actors[n][1] is not None else -1)
    act = actors[prim][0]

    res = {'pair_raw_profile': {'f': [], 't': []}, 'pair_geom_profile': {'f': [], 't': []},
           'pair_raw_global': {'f': [], 't': [], 'cov': []},
           'svd8_static_native': {'f': [], 't': []}}
    K_GLOBAL = 20

    for sd in SEEDS:
        ar = P.arena_seed(sd, rd_all); test = ar['test_users']; zst = P.build_zstar(model, ar, test)
        U = len(test)
        # profile candidate items per user (all rated in profile half)
        prof_items = []
        for x in test:
            profset, _ = ar['SPL'][x]
            prof_items.append(np.array(sorted(profset)))
        # ---- pair-realized actor, PROFILE-restricted, RAW answers ----
        for mode in ['raw', 'geom']:
            zt = zst; nz = np.linalg.norm(zt, axis=1) + 1e-9
            z = np.zeros((U, d), np.float32)
            for t in range(T):
                with torch.no_grad():
                    q = act(torch.tensor(z, dtype=torch.float32), t).numpy()
                for u in range(U):
                    x = test[u]; ci = prof_items[u]
                    pr = pair_snap_profile(q[u], ci, Dunit)
                    if pr is None:
                        continue
                    ii, jj, dirv = pr
                    if mode == 'raw':
                        a = np.clip((rd_all[x][ii] - rd_all[x][jj]) / 4.0, -1, 1)
                    else:
                        a = float(zt[u] @ dirv / nz[u])
                    z[u] = z[u] + ETA * a * dirv
            f, tt = cohort_ndcg(model, W, bdec, ar, test, z)
            key = 'pair_raw_profile' if mode == 'raw' else 'pair_geom_profile'
            res[key]['f'].append(f); res[key]['t'].append(tt)

        # ---- pair-realized actor, GLOBAL snap, RAW answers (coverage-limited) ----
        zt = zst; nz = np.linalg.norm(zt, axis=1) + 1e-9
        z = np.zeros((U, d), np.float32); cov_hit = 0; cov_tot = 0
        for t in range(T):
            with torch.no_grad():
                q = act(torch.tensor(z, dtype=torch.float32), t).numpy()
            proj = q @ Dunit.T                                    # (U,NI)
            for u in range(U):
                x = test[u]; rated = set(dict(ar['rat_by_u'][x]).keys())
                top = np.argpartition(-proj[u], K_GLOBAL)[:K_GLOBAL]
                bot = np.argpartition(proj[u], K_GLOBAL)[:K_GLOBAL]
                best = None; bestc = -2
                for ii in top:
                    for jj in bot:
                        if ii == jj: continue
                        diff = Dunit[ii] - Dunit[jj]; dn = np.linalg.norm(diff) + 1e-9
                        cc = q[u] @ diff / dn
                        if cc > bestc: bestc = cc; best = (ii, jj, diff / dn)
                cov_tot += 1
                if best is None: continue
                ii, jj, dirv = best
                profset = ar['SPL'][x][0]
                if ii in profset and jj in profset:               # both revealed -> answerable
                    cov_hit += 1
                    a = np.clip((rd_all[x][ii] - rd_all[x][jj]) / 4.0, -1, 1)
                    z[u] = z[u] + ETA * a * dirv
        f, tt = cohort_ndcg(model, W, bdec, ar, test, z)
        res['pair_raw_global']['f'].append(f); res['pair_raw_global']['t'].append(tt)
        res['pair_raw_global']['cov'].append(cov_hit / max(cov_tot, 1))

        # ---- static SVD-8 native reference ----
        Z = np.zeros((U, d), np.float32)
        for i in range(U):
            Z[i] = P.unroll_static(zst[i], dz.svd8)
        f, tt = cohort_ndcg(model, W, bdec, ar, test, Z)
        res['svd8_static_native']['f'].append(f); res['svd8_static_native']['t'].append(tt)
        print(f'[part3] seed{sd} done', flush=True)

    out = {}
    for k, v in res.items():
        out[k] = {'full': float(np.mean(v['f'])), 'tail': float(np.mean(v['t'])),
                  'full_sd': float(np.std(v['f']))}
        if 'cov' in v:
            out[k]['coverage'] = float(np.mean(v['cov']))
    _save('part3', out)
    print(json.dumps(out, indent=2))
    return out


# ------------------------------------------------------------------ io
def _save(stage, obj):
    all_ = json.load(open(OUT)) if os.path.exists(OUT) else {}
    all_[stage] = obj; json.dump(all_, open(OUT, 'w'), indent=2)
    print(f'[saved] {OUT} :: {stage}', flush=True)


def _load(stage):
    if os.path.exists(OUT):
        return json.load(open(OUT)).get(stage, {})
    return {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', choices=['channel', 'part1', 'trainnoisy', 'part2', 'part3'])
    ap.add_argument('--tseed', type=int, default=0)
    args = ap.parse_args()
    if args.stage == 'channel': fit_channel()
    elif args.stage == 'part1': part1()
    elif args.stage == 'trainnoisy': part1_trainnoisy(args.tseed)
    elif args.stage == 'part2': part2()
    else: part3()


if __name__ == '__main__':
    main()
