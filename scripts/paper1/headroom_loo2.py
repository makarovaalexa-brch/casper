"""
Harder, honest LOO eval: rank the held-out target against POPULARITY-MATCHED
negatives (same popularity decile), and include an explicit POPULARITY baseline
(rank by global popularity, ignoring the user). If the instrument barely beats
popularity under matched negatives, it's weak / riding popularity. If it clearly
beats popularity AND elicitation lifts it further, it genuinely personalizes.

Usage: DATASET_NAME=ml1m INST_NAME=instrument_ml1m_rank \
       poetry run python scripts/paper1/headroom_loo2.py
"""
import os, sys
sys.path.insert(0, '.'); sys.path.insert(0, 'scripts/paper1')
import numpy as np
from test_instrument_lib import load_instrument_by_name

NAME = os.environ.get('DATASET_NAME', 'ml1m')
INST = os.environ.get('INST_NAME', f'instrument_{NAME}_rank')
NPZ = os.environ.get('DATASET_NPZ') or \
    {'ml1m': 'C:/dev/phd/casper/data/movielens/ml1m_profiles.npz',
     'ml_stratified': 'C:/dev/phd/casper/data/movielens/ml_stratified_profiles.npz'}.get(NAME)
KS = [int(x) for x in os.environ.get('KS', '0,20').split(',')]
N_NEG = 50; N_USERS = 600


def metrics(scores, n_neg):
    rank = 1 + int((scores[1:] >= scores[0]).sum())
    return (1.0 if rank <= 10 else 0.0,
            1.0 / np.log2(rank + 1) if rank <= 10 else 0.0,
            1.0 / rank)


def main():
    w, _ = load_instrument_by_name(INST)
    d = np.load(NPZ, allow_pickle=True)
    train, test = d['train'], d['test']; nt = int(d['n_targets'])
    pop = (~np.isnan(train[:, :nt])).mean(0)
    decile = np.zeros(nt, int)
    order = np.argsort(pop)
    for q in range(10):
        decile[order[q * nt // 10:(q + 1) * nt // 10]] = q
    by_dec = {q: set(np.where(decile == q)[0].tolist()) for q in range(10)}
    rng = np.random.default_rng(0)
    inst = {k: {'h': [], 'n': [], 'm': []} for k in KS}
    popb = {'h': [], 'n': [], 'm': []}
    for prof in test[:N_USERS]:
        liked = np.where(prof[:nt] == 1)[0]
        rated = set(np.where(~np.isnan(prof[:nt]))[0].tolist())
        if len(liked) < 3:
            continue
        rng.shuffle(liked); target = int(liked[0])
        cand_pool = [i for i in by_dec[decile[target]] if i not in rated]
        if len(cand_pool) < N_NEG:
            continue
        negs = list(rng.choice(cand_pool, N_NEG, replace=False))
        cand = np.array([target] + negs)
        pool = [int(x) for x in liked[1:]] + [int(i) for i in np.where(prof[:nt] == 0)[0]]
        rng.shuffle(pool)
        # popularity baseline (ignores user)
        h, n, m = metrics(pop[cand], N_NEG); popb['h'].append(h); popb['n'].append(n); popb['m'].append(m)
        for K in KS:
            rev = [(e, float(prof[e])) for e in pool[:K] if e != target]
            sc = np.asarray(w.predict(rev))[cand]
            h, n, m = metrics(sc, N_NEG)
            inst[K]['h'].append(h); inst[K]['n'].append(n); inst[K]['m'].append(m)
    f = lambda x: float(np.mean(x))
    print(f"=== HARD LOO: popularity-matched negatives ({N_NEG}), {INST} ({NAME}) ===")
    print(f"{'method':>22}{'Hit@10':>10}{'NDCG@10':>10}{'MRR':>10}")
    print(f"{'popularity baseline':>22}{f(popb['h']):>10.4f}{f(popb['n']):>10.4f}{f(popb['m']):>10.4f}")
    for K in KS:
        print(f"{'instrument K='+str(K):>22}{f(inst[K]['h']):>10.4f}{f(inst[K]['n']):>10.4f}{f(inst[K]['m']):>10.4f}")
    print(f"\n  instrument(K=0) - popularity:  Hit {f(inst[0]['h'])-f(popb['h']):+.4f}  (does it beat popularity at all?)")
    print(f"  elicitation uplift K0->K20:    Hit {f(inst[KS[-1]]['h'])-f(inst[0]['h']):+.4f}")
    print(f"  random-guess Hit@10 = {10/(N_NEG+1):.3f}")


if __name__ == '__main__':
    main()
