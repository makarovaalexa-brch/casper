"""
ml25m_gates.py -- INSTRUMENT 2.0 Phase-3 W3 FORMAL GATE SUITE on the ML-25M arena.
Mirrors p3_gates.gates_ml1m (G1,G2,G6,G7,G8 from the amortized encoder path) and reads the
concept/dislike gates (G3 answerability, G4 polarity, G5 additivity) from p3_w2_ml25m.json.
Instrument = .cache/instrument2/ml25m_recvae_d512_best.pt. Writes p3_gates_ml25m.json.

Usage: python scripts/instrument2/ml25m_gates.py
"""
import os, sys, json
import numpy as np
import torch
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from recvae import RecVAE
import ml25m_arena as A

DEVICE = torch.device('cpu'); torch.set_num_threads(4)
CK = '.cache/instrument2'
NI = 18430


def enc_mu(model, Xd):
    with torch.no_grad():
        mu, _ = model.encoder(torch.tensor(np.asarray(Xd, np.float32)), dropout_rate=0.0)
    return mu.numpy()


def decode(model, Z):
    with torch.no_grad():
        return model.decoder(torch.tensor(np.asarray(Z, np.float32))).numpy().astype(np.float64)


def onehot(items, ni):
    x = np.zeros((1, ni), np.float32)
    if len(items): x[0, list(items)] = 1.0
    return x


def main():
    blob = torch.load(f'{CK}/ml25m_recvae_d512_best.pt', map_location=DEVICE)
    a = blob['args']; model = RecVAE(a['hidden'], a['latent'], NI)
    model.load_state_dict(blob['model']); model.eval()
    ni = NI; d = model.decoder.in_features
    seeds = [1, 2, 3, 7, 11]
    ARENA = {sd: A.load_arena(seed=sd) for sd in seeds}
    rd_all = {x: dict(v) for x, v in ARENA[1]['rat_by_u'].items()}
    floor = decode(model, np.zeros((1, d), np.float32))[0]

    # per-item single-fold entropy (for the entropy selector)
    ent_tab = np.zeros(ni)
    for st in range(0, ni, 512):
        idx = np.arange(st, min(st + 512, ni))
        X = np.zeros((len(idx), ni), np.float32); X[np.arange(len(idx)), idx] = 1.0
        S = decode(model, enc_mu(model, X))
        P = np.exp(S - S.max(1, keepdims=True)); P /= P.sum(1, keepdims=True)
        ent_tab[idx] = -(P * np.log(P + 1e-12)).sum(1)
    print('[gates] entropy table done', flush=True)

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
                z = enc_mu(model, onehot(pick, ni))[0]
            s = decode(model, z[None, :])[0]
            nf = A.ndcg_at10(s, tlike, profset, ar['headmask'], False)
            if nf is not None: accf += nf; m += 1
        return accf / max(m, 1)

    # G1 monotonicity + G2 no-harm
    kgrid = [1, 2, 3, 4, 6, 8, 12, 16, 20]
    g1_curves = {}; g2_min = {}
    for sel in ('random', 'pop', 'entropy'):
        curve = []
        for k in kgrid:
            vals = [eval_k(ARENA[sd], ARENA[sd]['test_users'], sel, k, np.random.default_rng(100 + sd))
                    for sd in [1, 2, 3]]
            curve.append(float(np.mean(vals)))
        g1_curves[sel] = dict(zip(map(str, kgrid), curve)); g2_min[sel] = float(min(curve))
        print(f'[gates] G1/G2 {sel} done', flush=True)
    rc = [g1_curves['random'][str(k)] for k in kgrid]; eps = 0.006
    g1_pass = all(rc[i + 1] >= rc[i] - eps for i in range(len(rc) - 1)) and rc[-1] > rc[0]

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

    # G6 fold-vs-full (80%) + G7 seed stability
    def eval_frac_and_full(ar, users, frac):
        accf_frac = accf_full = 0.0; m = 0; rng = np.random.default_rng(0)
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
        fr, fu = eval_frac_and_full(ARENA[sd], ARENA[sd]['test_users'], 0.8)
        full_by_seed.append(fu)
        if sd == 1: g6_frac, g6_full = fr, fu
    g6_ratio = g6_frac / g6_full; g6_pass = g6_ratio >= 0.90
    g7_sd = float(np.std(full_by_seed)); g7_pass = g7_sd < 0.01
    print('[gates] G6/G7 done', flush=True)

    # G8 answer sanity
    ar1 = ARENA[1]
    W = model.decoder.weight.detach().numpy()
    _, _, Vt = np.linalg.svd(W - W.mean(0, keepdims=True), full_matrices=False)
    info = Vt[:8].astype(np.float32)
    svals = [0.0, 0.25, 0.5, 0.75, 1.0]; curve = {s: [] for s in svals}
    for x in ar1['test_users']:
        profset, tst = ar1['SPL'][x]; rd = rd_all[x]
        tlike = set(j for j in tst if rd[j] >= 4)
        if not tlike: continue
        lk = [j for j in profset if rd[j] >= 4]
        if not lk: continue
        zs = enc_mu(model, onehot(lk, ni))[0]; nz = np.linalg.norm(zs) + 1e-9
        av = info @ zs / nz
        for s in svals:
            z = 16.0 * ((s * av)[:, None] * info).sum(0)
            g = A.ndcg_at10(decode(model, z[None, :])[0], tlike, profset, ar1['headmask'], False)
            if g is not None: curve[s].append(g)
    means = [float(np.mean(curve[s])) for s in svals]
    rho = float(spearmanr(svals, means).correlation)
    mono = all(means[i + 1] >= means[i] - 0.006 for i in range(len(means) - 1))
    lift = means[-1] - means[0]
    g8 = {'ndcg_by_signal_strength': dict(zip(map(str, svals), means)), 'spearman_s_ndcg': rho,
          'monotone_tol': mono, 'lift_over_floor': lift, 'pass': bool(mono and lift > 0.05)}
    print('[gates] G8 done', flush=True)

    # G3/G4/G5 from W2
    w2 = json.load(open(f'{CK}/p3_w2_ml25m.json')) if os.path.exists(f'{CK}/p3_w2_ml25m.json') else {}
    g3v = w2.get('answerability_liveness_mean', None)
    g3 = {'val': g3v, 'pass': bool(g3v is not None and g3v > 1.0)}
    da = w2.get('dislike_A', {})
    spec = da.get('specificity_pts', 0.0)
    g4 = {'member_demotion_pts': da.get('mean_member_demotion_pts'),
          'control_demotion_pts': da.get('mean_control_demotion_pts'), 'specificity_pts': spec,
          'pass': bool(da.get('mean_member_demotion_pts', 0) < 0 and spec < -2.0)}
    add = w2.get('additivity', {})
    g5 = {'delta': add.get('delta'), 'pass': bool(add.get('delta', -1) > 0)}

    tbl = {'dataset': 'ML-25M',
           'G1_monotonicity': {'curves': g1_curves, 'pass': bool(g1_pass)},
           'G2_no_harm': {'z0_floor': floor_full, 'selector_min': g2_min, 'pass': bool(g2_pass)},
           'G3_answerability': g3, 'G4_polarity': g4, 'G5_additivity': g5,
           'G6_fold_vs_full': {'ratio_80pct': g6_ratio, 'full': g6_full, 'pass': bool(g6_pass)},
           'G7_seed_stability': {'sd': g7_sd, 'full_by_seed': full_by_seed, 'pass': bool(g7_pass)},
           'G8_answer_sanity': g8}
    npass = sum(1 for k in ['G1_monotonicity', 'G2_no_harm', 'G3_answerability', 'G4_polarity',
                            'G5_additivity', 'G6_fold_vs_full', 'G7_seed_stability', 'G8_answer_sanity']
                if tbl[k].get('pass'))
    tbl['n_pass'] = npass
    json.dump(tbl, open(f'{CK}/p3_gates_ml25m.json', 'w'), indent=2)
    print(json.dumps(tbl, indent=2))
    print(f'\n=== ML-25M: {npass}/8 gates PASS ===')


if __name__ == '__main__':
    main()
