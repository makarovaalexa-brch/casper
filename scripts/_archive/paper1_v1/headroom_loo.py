"""
Honest, PUBLISHED-COMPARABLE elicitation eval: leave-one-out with sampled
negatives (single held-out target ranked against 100 random un-rated items),
as in EAR/UNICORN/PEBOL. Reveal K of the user's OTHER ratings -> does the
held-out target rise? Reports Hit@10 / NDCG@10 / MRR vs K. Negatives are
sampled from un-rated items (mostly long-tail) so popularity can't game it.

Reuses a trained instrument; no training.
Usage: DATASET_NAME=ml1m INST_NAME=instrument_ml1m_rank \
       poetry run python scripts/paper1/headroom_loo.py
"""
import os, sys
sys.path.insert(0, '.'); sys.path.insert(0, 'scripts/paper1')
import numpy as np
from test_instrument_lib import load_instrument_by_name

NAME = os.environ.get('DATASET_NAME', 'ml1m')
INST = os.environ.get('INST_NAME', f'instrument_{NAME}_rank')
NPZ = {'ml1m': 'C:/dev/phd/casper/data/movielens/ml1m_profiles.npz',
       'ml_stratified': 'C:/dev/phd/casper/data/movielens/ml_stratified_profiles.npz'}[NAME]
KS = [0, 5, 10, 20]
N_NEG = 100
N_USERS = 600


def main():
    w, _ = load_instrument_by_name(INST)
    d = np.load(NPZ, allow_pickle=True)
    train, test = d['train'], d['test']; nt = int(d['n_targets'])
    rng = np.random.default_rng(0)
    res = {k: {'hit': [], 'ndcg': [], 'mrr': []} for k in KS}
    for prof in test[:N_USERS]:
        liked = np.where(prof[:nt] == 1)[0]
        rated = set(np.where(~np.isnan(prof[:nt]))[0].tolist())
        unrated = [i for i in range(nt) if i not in rated]
        if len(liked) < 3 or len(unrated) < N_NEG:
            continue
        rng.shuffle(liked)
        target = int(liked[0])                       # held-out target (never revealed)
        pool = [int(x) for x in liked[1:]] + \
               [int(i) for i in np.where(prof[:nt] == 0)[0]]   # other ratings to reveal
        rng.shuffle(pool)
        negs = list(rng.choice(unrated, N_NEG, replace=False))
        cand = np.array([target] + negs)
        for K in KS:
            rev = [(e, float(prof[e])) for e in pool[:K] if e != target]
            scores = np.asarray(w.predict(rev))[cand]
            rank = 1 + int((scores[1:] >= scores[0]).sum())   # rank of target (1=best)
            res[K]['hit'].append(1.0 if rank <= 10 else 0.0)
            res[K]['ndcg'].append(1.0 / np.log2(rank + 1) if rank <= 10 else 0.0)
            res[K]['mrr'].append(1.0 / rank)
    m = lambda x: float(np.mean(x))
    print(f"=== leave-one-out + {N_NEG} sampled negatives: {INST} ({NAME}) ===")
    print(f"{'K revealed':>11}{'Hit@10':>10}{'NDCG@10':>10}{'MRR':>10}")
    for K in KS:
        print(f"{K:>11}{m(res[K]['hit']):>10.4f}{m(res[K]['ndcg']):>10.4f}{m(res[K]['mrr']):>10.4f}")
    print(f"\n  uplift Hit@10 (K=0->{KS[-1]}): {m(res[KS[-1]]['hit'])-m(res[KS[0]]['hit']):+.4f}")
    print(f"  reference (sampled-neg, published ML-1M): strong models Hit@10~0.6-0.8, NDCG@10~0.35-0.45")


if __name__ == '__main__':
    main()
