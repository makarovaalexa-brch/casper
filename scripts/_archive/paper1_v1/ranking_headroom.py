"""
Is the elicitation signal hidden by the metric? Compare per-movie BINARY
accuracy headroom vs RANKING headroom (NDCG@10 / Hit@10) on the same
instrument. Recommenders are judged by ranking; binary accuracy is base-rate
dominated. Targets are held out (we reveal only ATTRIBUTE prefs, never the
movies we rank), so no leakage.

Usage: DATASET_NAME=ml1m poetry run python scripts/paper1/ranking_headroom.py
"""
import os, sys
sys.path.insert(0, '.'); sys.path.insert(0, 'scripts/paper1')
import numpy as np
from test_instrument_lib import load_instrument_by_name

NAME = os.environ.get('DATASET_NAME', 'ml1m')
NPZ = {'ml1m': 'C:/dev/phd/casper/data/movielens/ml1m_profiles.npz',
       'ml_stratified': 'C:/dev/phd/casper/data/movielens/ml_stratified_profiles.npz'}[NAME]
INST = os.environ.get('INST_NAME', f'instrument_{NAME}')
K = 10
N_USERS = 500


def ndcg_hit(preds, liked, k=K):
    order = np.argsort(-preds)
    top = order[:k]
    rel = liked[top].astype(float)
    dcg = (rel / np.log2(np.arange(2, k + 2))).sum()
    ideal = int(min(liked.sum(), k))
    idcg = (1.0 / np.log2(np.arange(2, ideal + 2))).sum() if ideal > 0 else 1.0
    return (dcg / idcg if idcg > 0 else 0.0), float(rel.any())


def acc(preds, gt, filt):
    return float(((preds[filt] > 0.5) == gt[filt]).mean())


def main():
    w, _ = load_instrument_by_name(INST)
    d = np.load(NPZ, allow_pickle=True)
    test = d['test'][:N_USERS]; nm = int(d['n_targets'])
    rng = np.random.default_rng(0)
    rows = {'turn0': [], 'full': []}; hit = {'turn0': [], 'full': []}
    accs = {'turn0': [], 'full': []}; rnd = []
    for prof in test:
        gt = prof[:nm]; filt = ~np.isnan(gt)
        liked = (gt == 1)
        if liked.sum() == 0 or filt.sum() == 0:
            continue
        # turn 0: no reveal
        p0 = np.asarray(w.predict([]))[:nm]
        # full ATTRIBUTE reveal (targets stay held out -> no leakage)
        attr_rev = [(int(i), float(prof[i])) for i in range(nm, w.n_items)
                    if not np.isnan(prof[i])]
        pf = np.asarray(w.predict(attr_rev))[:nm]
        n0, h0 = ndcg_hit(p0, liked); nf, hf = ndcg_hit(pf, liked)
        rows['turn0'].append(n0); rows['full'].append(nf)
        hit['turn0'].append(h0); hit['full'].append(hf)
        accs['turn0'].append(acc(p0, gt, filt)); accs['full'].append(acc(pf, gt, filt))
        rp = rng.permutation(nm).astype(float); rnd.append(ndcg_hit(rp, liked)[0])
    def m(x): return float(np.mean(x))
    print(f"=== {NAME} / {INST}  (n={len(rows['turn0'])} users, K={K}) ===")
    print(f"  BINARY acc      turn0={m(accs['turn0']):.4f}  full={m(accs['full']):.4f}  "
          f"lift={m(accs['full'])-m(accs['turn0']):+.4f}")
    print(f"  NDCG@{K}        turn0={m(rows['turn0']):.4f}  full={m(rows['full']):.4f}  "
          f"lift={m(rows['full'])-m(rows['turn0']):+.4f}   (random {m(rnd):.4f})")
    print(f"  Hit@{K}         turn0={m(hit['turn0']):.4f}  full={m(hit['full']):.4f}  "
          f"lift={m(hit['full'])-m(hit['turn0']):+.4f}")
    print(f"  >>> ranking lift / binary lift ratio = "
          f"{(m(rows['full'])-m(rows['turn0']))/max(m(accs['full'])-m(accs['turn0']),1e-6):.1f}x")


if __name__ == '__main__':
    main()
