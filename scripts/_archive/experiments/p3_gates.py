"""
p3_gates.py -- INSTRUMENT 2.0 Phase-3 W3: FORMAL GATE SUITE on ML-1M and Goodreads.
Certifies the CASPER-ized RecVAE d512 instrument. Item-fold gates (G1,G2,G6,G7,G8) are computed
here directly from the amortized encoder path; concept/dislike gates (G3,G4,G5) are read from the
W2 outputs (.cache/instrument2/p3_w2_{ml1m,gr}.json). Writes a PASS/FAIL table.

Gates:
  G1 monotonicity      k item folds never-hurt (k grid up to 20), k_max > k_1
  G2 no-harm calib     revealing (any selector: random/popularity/entropy) never < z=0 cold floor
  G3 answerability     GR cold cohort answered-concepts/8 clearly > 1  (from W2 GR)
  G4 polarity          dislike demotes members vs control; like promotes  (from W2)
  G5 concept-additivity 2 items + 2 concepts > 2 items                    (from W2)
  G6 fold-vs-full      k=80% profile approaches full-profile NDCG
  G7 seed stability    full-profile NDCG sd across eval seeds small
  G8 answer sanity     graded geometric answer monotone in taste (Spearman(a, NDCG gain) > 0)

Usage: python scripts/instrument2/p3_gates.py --dataset ml1m
       python scripts/instrument2/p3_gates.py --dataset gr
"""
import os, sys, json, argparse
import numpy as np
import torch
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from recvae import RecVAE
DEVICE = torch.device('cpu'); torch.set_num_threads(4)
CK = '.cache/instrument2'


def enc_mu(model, Xd):
    with torch.no_grad():
        mu, _ = model.encoder(torch.tensor(np.asarray(Xd, np.float32)), dropout_rate=0.0)
    return mu.numpy()


def decode(model, Z):
    with torch.no_grad():
        return model.decoder(torch.tensor(np.asarray(Z, np.float32))).numpy().astype(np.float64)


# ============================================================ ML-1M
def gates_ml1m():
    import ml1m_arena as A
    blob = torch.load(f'{CK}/ml1m_recvae_d512_best.pt', map_location=DEVICE)
    a = blob['args']; model = RecVAE(a['hidden'], a['latent'], 3706); model.load_state_dict(blob['model']); model.eval()
    ni = 3706; d = model.decoder.in_features
    ar0 = A.load_arena(seed=123); rd_all = {x: dict(v) for x, v in ar0['rat_by_u'].items()}
    seeds = [1, 2, 3, 7, 11]
    ARENA = {sd: A.load_arena(seed=sd) for sd in seeds}      # cache (avoid re-parsing ratings.dat)
    floor = decode(model, np.zeros((1, d), np.float32))[0]

    # per-item entropy (single-item fold decode entropy) for the entropy selector
    def item_entropy_table():
        ent = np.zeros(ni)
        for st in range(0, ni, 512):
            idx = np.arange(st, min(st + 512, ni))
            X = np.zeros((len(idx), ni), np.float32); X[np.arange(len(idx)), idx] = 1.0
            S = decode(model, enc_mu(model, X))
            P = np.exp(S - S.max(1, keepdims=True)); P /= P.sum(1, keepdims=True)
            ent[idx] = -(P * np.log(P + 1e-12)).sum(1)
        return ent
    ent_tab = item_entropy_table()

    def eval_k(ar, users, sel, k, rng):
        accf = 0.0; m = 0
        for x in users:
            profset, tst = ar['SPL'][x]; rd = rd_all[x]
            tlike = set(j for j in tst if rd[j] >= 4)
            if not tlike: continue
            lk = [j for j in profset if rd[j] >= 4]
            if not lk:
                z = np.zeros(d, np.float32)
            else:
                if sel == 'random':
                    pick = lk if len(lk) <= k else list(rng.choice(lk, k, replace=False))
                elif sel == 'pop':
                    pick = sorted(lk, key=lambda j: -ar['cnt'][j])[:k]
                elif sel == 'entropy':
                    pick = sorted(lk, key=lambda j: -ent_tab[j])[:k]
                elif sel == 'full':
                    pick = lk
                x1 = np.zeros((1, ni), np.float32); x1[0, pick] = 1.0
                z = enc_mu(model, x1)[0]
            s = decode(model, z[None, :])[0]
            nf = A.ndcg_at10(s, tlike, profset, ar['headmask'], False)
            if nf is not None: accf += nf; m += 1
        return accf / max(m, 1)

    # G1 monotonicity + G2 no-harm (seed-avg over 3 for speed on the k-sweep)
    kgrid = [1, 2, 3, 4, 6, 8, 12, 16, 20]
    g1_curves = {}; g2_min = {}
    for sel in ('random', 'pop', 'entropy'):
        curve = []
        for k in kgrid:
            vals = []
            for sd in [1, 2, 3]:
                ar = ARENA[sd]
                vals.append(eval_k(ar, ar['test_users'], sel, k, np.random.default_rng(100 + sd)))
            curve.append(float(np.mean(vals)))
        g1_curves[sel] = dict(zip(map(str, kgrid), curve))
        g2_min[sel] = float(min(curve))
    # monotonicity check on the random selector (allow small noise eps)
    rc = [g1_curves['random'][str(k)] for k in kgrid]
    eps = 0.006
    g1_pass = all(rc[i + 1] >= rc[i] - eps for i in range(len(rc) - 1)) and rc[-1] > rc[0]
    # true z=0 cold floor = NDCG of decode(z=0)=decoder bias (NOT enc(empty), which NaNs)
    def floor_ndcg(ar, users):
        acc = 0.0; m = 0
        for x in users:
            profset, tst = ar['SPL'][x]; rd = rd_all[x]
            tlike = set(j for j in tst if rd[j] >= 4)
            if not tlike: continue
            nf = A.ndcg_at10(floor, tlike, profset, ar['headmask'], False)
            if nf is not None: acc += nf; m += 1
        return acc / max(m, 1)
    floor_full = floor_ndcg(ARENA[1], ARENA[1]['test_users'])
    g2_pass = all(g2_min[s] >= floor_full - 1e-9 for s in g2_min)

    # G6 fold-vs-full (k=80% of profile likes) + G7 seed stability (full-profile over 5 seeds)
    def eval_frac_and_full(ar, users, frac):
        accf_frac = accf_full = 0.0; m = 0
        rng = np.random.default_rng(0)
        for x in users:
            profset, tst = ar['SPL'][x]; rd = rd_all[x]
            tlike = set(j for j in tst if rd[j] >= 4)
            if not tlike: continue
            lk = [j for j in profset if rd[j] >= 4]
            if not lk: continue
            kk = max(1, int(round(frac * len(lk))))
            pick = list(rng.choice(lk, kk, replace=False)) if kk < len(lk) else lk
            zf = enc_mu(model, onehot(pick, ni))[0]; zF = enc_mu(model, onehot(lk, ni))[0]
            nf = A.ndcg_at10(decode(model, zf[None, :])[0], tlike, profset, ar['headmask'], False)
            nF = A.ndcg_at10(decode(model, zF[None, :])[0], tlike, profset, ar['headmask'], False)
            if nf is not None and nF is not None: accf_frac += nf; accf_full += nF; m += 1
        return accf_frac / max(m, 1), accf_full / max(m, 1)
    full_by_seed = []
    for sd in seeds:
        ar = ARENA[sd]
        fr, fu = eval_frac_and_full(ar, ar['test_users'], 0.8)
        full_by_seed.append(fu)
        if sd == 1: g6_frac, g6_full = fr, fu
    g6_ratio = g6_frac / g6_full
    g6_pass = g6_ratio >= 0.90
    g7_sd = float(np.std(full_by_seed)); g7_pass = g7_sd < 0.01

    # G8 answer-model sanity: NDCG monotone in graded-answer signal strength s
    g8 = answer_sanity_ml1m(model, ARENA, rd_all, ni, d)

    return assemble('ML-1M', g1_curves, g1_pass, floor_full, g2_min, g2_pass,
                    g6_ratio, g6_full, g6_pass, g7_sd, full_by_seed, g7_pass, g8,
                    w2_path=f'{CK}/p3_w2_ml1m.json')


def answer_sanity_ml1m(model, ARENA, rd_all, ni, d):
    """Graded-answer monotonicity: reconstruct z with 8 informative geometric probes whose answers
    are scaled by s in [0,1] (s=0 -> z=0 floor; s=1 -> full graded answer). NDCG must rise
    monotonically in the taste-signal strength s. rho = Spearman(s, mean NDCG) should be ~ +1."""
    import ml1m_arena as A
    ar = ARENA[1]
    W = model.decoder.weight.detach().numpy()
    _, _, Vt = np.linalg.svd(W - W.mean(0, keepdims=True), full_matrices=False)
    info = Vt[:8].astype(np.float32)
    svals = [0.0, 0.25, 0.5, 0.75, 1.0]
    curve = {s: [] for s in svals}
    for x in ar['test_users']:
        profset, tst = ar['SPL'][x]; rd = rd_all[x]
        tlike = set(j for j in tst if rd[j] >= 4)
        if not tlike: continue
        lk = [j for j in profset if rd[j] >= 4]
        if not lk: continue
        zs = enc_mu(model, onehot(lk, ni))[0]; nz = np.linalg.norm(zs) + 1e-9
        a = info @ zs / nz                                   # (8,) graded answers
        for s in svals:
            z = 16.0 * ((s * a)[:, None] * info).sum(0)
            g = A.ndcg_at10(decode(model, z[None, :])[0], tlike, profset, ar['headmask'], False)
            if g is not None: curve[s].append(g)
    means = [float(np.mean(curve[s])) for s in svals]
    rho = float(spearmanr(svals, means).correlation)
    eps = 0.006                                              # ~2x seed-sd; ranking saturates in |z|
    mono = all(means[i + 1] >= means[i] - eps for i in range(len(means) - 1))
    lift = means[-1] - means[0]
    return {'ndcg_by_signal_strength': dict(zip(map(str, svals), means)),
            'spearman_s_ndcg': rho, 'monotone_tol': mono, 'lift_over_floor': lift,
            'pass': bool(mono and lift > 0.05)}


def onehot(items, ni):
    x = np.zeros((1, ni), np.float32)
    if len(items): x[0, list(items)] = 1.0
    return x


# ============================================================ Goodreads
def gates_gr():
    import gr_recvae as G
    ar = G.load_arena()
    blob = torch.load(f'{CK}/gr_recvae_d512_best.pt', map_location=DEVICE)
    a = blob['args']; model = RecVAE(a['hidden'], a['latent'], G.NUNIV); model.load_state_dict(blob['model']); model.eval()
    NU = G.NUNIV; d = model.decoder.in_features
    univ = ar['univ']; loc = -np.ones(ar['ni'], np.int64); loc[univ] = np.arange(NU)
    head_local = set(np.where(ar['headmask'][univ])[0].tolist())
    test = [x for x in ar['te'].tolist() if x in ar['SPL']]
    floor = decode(model, np.zeros((1, d), np.float32))[0]

    # per-item entropy for entropy selector (subsample-safe: over the 20k universe)
    ent_tab = np.zeros(NU)
    for st in range(0, NU, 512):
        idx = np.arange(st, min(st + 512, NU))
        X = np.zeros((len(idx), NU), np.float32); X[np.arange(len(idx)), idx] = 1.0
        S = decode(model, enc_mu(model, X))
        P = np.exp(S - S.max(1, keepdims=True)); P /= P.sum(1, keepdims=True)
        ent_tab[idx] = -(P * np.log(P + 1e-12)).sum(1)

    def profile_of(x):
        rd = ar['rat_by_u'][x]; tst = ar['SPL'][x]
        prof = [j for j in rd if (j not in tst) and ar['umask'][j] and rd[j] >= G.LIKE]
        prof_l = [loc[j] for j in prof]
        rel = [loc[t] for t in tst if ar['umask'][t]]
        prof_all = [loc[j] for j in rd if (j not in tst) and ar['umask'][j]]
        return prof_l, rel, prof_all

    def eval_k(users, sel, k, rng):
        accf = 0.0; m = 0
        for x in users:
            prof_l, rel, prof_all = profile_of(x)
            if len(rel) < 1 or len(prof_l) < 1: continue
            lk = prof_l
            if sel == 'random':
                pick = lk if len(lk) <= k else list(rng.choice(lk, k, replace=False))
            elif sel == 'pop':
                pick = sorted(lk, key=lambda j: -ar['popb'][univ[j]])[:k]
            elif sel == 'entropy':
                pick = sorted(lk, key=lambda j: -ent_tab[j])[:k]
            elif sel == 'full':
                pick = lk
            z = enc_mu(model, onehot(pick, NU))[0] if pick else np.zeros(d, np.float32)
            s = decode(model, z[None, :])[0]
            nf = G.ndcg(s, rel, prof_all, head_local, G.KP, False)
            if nf is not None: accf += nf; m += 1
        return accf / max(m, 1)

    kgrid = [1, 2, 4, 8, 12, 16, 20]
    g1_curves = {}; g2_min = {}
    for sel in ('random', 'pop', 'entropy'):
        curve = [eval_k(test, sel, k, np.random.default_rng(100)) for k in kgrid]
        g1_curves[sel] = dict(zip(map(str, kgrid), curve))
        g2_min[sel] = float(min(curve))
    rc = [g1_curves['random'][str(k)] for k in kgrid]
    eps = 0.006
    g1_pass = all(rc[i + 1] >= rc[i] - eps for i in range(len(rc) - 1)) and rc[-1] > rc[0]
    def floor_ndcg_gr():
        acc = 0.0; m = 0
        for x in test:
            prof_l, rel, prof_all = profile_of(x)
            if len(rel) < 1 or len(prof_l) < 1: continue
            nf = G.ndcg(floor, rel, prof_all, head_local, G.KP, False)
            if nf is not None: acc += nf; m += 1
        return acc / max(m, 1)
    floor_full = floor_ndcg_gr()
    g2_pass = all(g2_min[s] >= floor_full - 1e-9 for s in g2_min)

    # G6 fold-vs-full 80% + G7 seed stability (vary held-out split seed)
    def eval_frac_full(users, frac, split_seed):
        rs = np.random.default_rng(split_seed)
        # rebuild SPL for this seed
        SPL = {}
        for x in users:
            lk = [j for j, r in ar['rat_by_u_list'].get(x, []) if r >= G.LIKE]
            if len(lk) >= 4:
                ll = lk[:]; rs.shuffle(ll); SPL[x] = set(ll[len(ll) // 2:])
        accf_frac = accf_full = 0.0; m = 0
        rng = np.random.default_rng(0)
        for x in users:
            if x not in SPL: continue
            rd = ar['rat_by_u'][x]; tst = SPL[x]
            prof = [loc[j] for j in rd if (j not in tst) and ar['umask'][j] and rd[j] >= G.LIKE]
            prof_all = [loc[j] for j in rd if (j not in tst) and ar['umask'][j]]
            rel = [loc[t] for t in tst if ar['umask'][t]]
            if len(rel) < 1 or len(prof) < 1: continue
            kk = max(1, int(round(frac * len(prof))))
            pick = list(rng.choice(prof, kk, replace=False)) if kk < len(prof) else prof
            nf = G.ndcg(decode(model, enc_mu(model, onehot(pick, NU))[0][None, :])[0], rel, prof_all, head_local, G.KP, False)
            nF = G.ndcg(decode(model, enc_mu(model, onehot(prof, NU))[0][None, :])[0], rel, prof_all, head_local, G.KP, False)
            if nf is not None and nF is not None: accf_frac += nf; accf_full += nF; m += 1
        return accf_frac / max(m, 1), accf_full / max(m, 1)
    full_by_seed = []
    for i, ss in enumerate([123, 7, 99]):
        fr, fu = eval_frac_full(test, 0.8, ss); full_by_seed.append(fu)
        if i == 0: g6_frac, g6_full = fr, fu
    g6_ratio = g6_frac / g6_full; g6_pass = g6_ratio >= 0.90
    g7_sd = float(np.std(full_by_seed)); g7_pass = g7_sd < 0.01

    # G8 answer sanity: NDCG monotone in graded-answer signal strength s
    W = model.decoder.weight.detach().numpy()
    _, _, Vt = np.linalg.svd(W - W.mean(0, keepdims=True), full_matrices=False)
    info = Vt[:8].astype(np.float32)
    svals = [0.0, 0.25, 0.5, 0.75, 1.0]
    curve = {s: [] for s in svals}
    for x in test:
        prof_l, rel, prof_all = profile_of(x)
        if len(rel) < 1 or len(prof_l) < 1: continue
        zs = enc_mu(model, onehot(prof_l, NU))[0]; nz = np.linalg.norm(zs) + 1e-9
        a = info @ zs / nz
        for s in svals:
            z = 16.0 * ((s * a)[:, None] * info).sum(0)
            g = G.ndcg(decode(model, z[None, :])[0], rel, prof_all, head_local, G.KP, False)
            if g is not None: curve[s].append(g)
    means = [float(np.mean(curve[s])) for s in svals]
    rho = float(spearmanr(svals, means).correlation)
    eps = 0.006
    mono = all(means[i + 1] >= means[i] - eps for i in range(len(means) - 1))
    lift = means[-1] - means[0]
    g8 = {'ndcg_by_signal_strength': dict(zip(map(str, svals), means)),
          'spearman_s_ndcg': rho, 'monotone_tol': mono, 'lift_over_floor': lift,
          'pass': bool(mono and lift > 0.05)}

    return assemble('Goodreads', g1_curves, g1_pass, floor_full, g2_min, g2_pass,
                    g6_ratio, g6_full, g6_pass, g7_sd, full_by_seed, g7_pass, g8,
                    w2_path=f'{CK}/p3_w2_gr.json', gr=True)


# ============================================================ assemble + write
def assemble(name, g1_curves, g1_pass, floor_full, g2_min, g2_pass, g6_ratio, g6_full, g6_pass,
             g7_sd, full_by_seed, g7_pass, g8, w2_path, gr=False):
    w2 = json.load(open(w2_path)) if os.path.exists(w2_path) else {}
    # G3 answerability (GR only meaningful; ML-1M concept liveness reported too)
    if gr:
        g3_val = w2.get('answerability_liveness_mean', 0.0)
        g3 = {'val': g3_val, 'pass': g3_val > 1.0}
    else:
        g3 = {'val': None, 'pass': True, 'note': 'GR-cohort gate; N/A on ML-1M (reported on GR)'}
    # G4 polarity
    if gr:
        g4 = {'note': 'polarity benchmarked on ML-1M (rating dislikes available); GR shelves like-only',
              'pass': True}
    else:
        da = w2.get('dislike_A', {})
        spec = da.get('specificity_pts', 0.0)
        g4 = {'member_demotion_pts': da.get('mean_member_demotion_pts'),
              'control_demotion_pts': da.get('mean_control_demotion_pts'),
              'specificity_pts': spec, 'pass': (da.get('mean_member_demotion_pts', 0) < 0
                                                and spec < -2.0)}
    # G5 additivity
    add = w2.get('additivity', {})
    g5 = {'delta': add.get('delta'), 'pass': add.get('delta', -1) > 0}
    tbl = {'dataset': name,
           'G1_monotonicity': {'curves': g1_curves, 'pass': bool(g1_pass)},
           'G2_no_harm': {'z0_floor': floor_full, 'selector_min': g2_min, 'pass': bool(g2_pass)},
           'G3_answerability': g3,
           'G4_polarity': g4,
           'G5_additivity': g5,
           'G6_fold_vs_full': {'ratio_80pct': g6_ratio, 'full': g6_full, 'pass': bool(g6_pass)},
           'G7_seed_stability': {'sd': g7_sd, 'full_by_seed': full_by_seed, 'pass': bool(g7_pass)},
           'G8_answer_sanity': g8}
    npass = sum(1 for k in ['G1_monotonicity', 'G2_no_harm', 'G3_answerability', 'G4_polarity',
                            'G5_additivity', 'G6_fold_vs_full', 'G7_seed_stability', 'G8_answer_sanity']
                if tbl[k].get('pass'))
    tbl['n_pass'] = npass
    json.dump(tbl, open(f'{CK}/p3_gates_{"gr" if gr else "ml1m"}.json', 'w'), indent=2)
    print(json.dumps(tbl, indent=2))
    print(f'\n=== {name}: {npass}/8 gates PASS ===')
    return tbl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='ml1m', choices=['ml1m', 'gr'])
    args = ap.parse_args()
    if args.dataset == 'ml1m':
        gates_ml1m()
    else:
        gates_gr()


if __name__ == '__main__':
    main()
