"""
Proper headroom re-check (gut check): reveal K of the user's actual MOVIE
ratings (item-level elicitation, not just coarse attributes) and measure
ranking of held-out items vs K. Also splits OVERALL vs LONG-TAIL NDCG to test
the 'elicitation surfaces obscure items' (novelty) hypothesis.

Reuses a trained instrument; no training.
Usage: DATASET_NAME=ml1m INST_NAME=instrument_ml1m_rank \
       poetry run python scripts/paper1/headroom_item.py
"""
import os, sys
sys.path.insert(0, '.'); sys.path.insert(0, 'scripts/paper1')
import numpy as np
from test_instrument_lib import load_instrument_by_name

NAME = os.environ.get('DATASET_NAME', 'ml1m')
INST = os.environ.get('INST_NAME', f'instrument_{NAME}_rank')
NPZ = {'ml1m': 'C:/dev/phd/casper/data/movielens/ml1m_profiles.npz',
       'ml_stratified': 'C:/dev/phd/casper/data/movielens/ml_stratified_profiles.npz'}[NAME]
KS = [0, 3, 5, 10, 20]
N_USERS = 400


def ndcg(scores, relevant_mask, cand_mask, k=10):
    cand = np.where(cand_mask)[0]
    if len(cand) == 0 or relevant_mask[cand].sum() == 0:
        return np.nan
    order = cand[np.argsort(-scores[cand])][:k]
    rel = relevant_mask[order].astype(float)
    dcg = (rel / np.log2(np.arange(2, len(rel) + 2))).sum()
    ideal = int(min(relevant_mask[cand].sum(), k))
    idcg = (1.0 / np.log2(np.arange(2, ideal + 2))).sum()
    return float(dcg / idcg) if idcg > 0 else 0.0


def main():
    w, _ = load_instrument_by_name(INST)
    d = np.load(NPZ, allow_pickle=True)
    train, test = d['train'], d['test']; nt = int(d['n_targets'])
    pop = (~np.isnan(train[:, :nt])).mean(0)          # item popularity proxy
    tail_thresh = np.median(pop)
    tail = pop < tail_thresh                          # bottom-half popularity
    rng = np.random.default_rng(0)
    overall = {k: [] for k in KS}; tailn = {k: [] for k in KS}
    for prof in test[:N_USERS]:
        rated = np.where(~np.isnan(prof[:nt]))[0]
        liked = (prof[:nt] == 1)
        if liked.sum() < 4:
            continue
        rng.shuffle(rated)
        for K in KS:
            rev = [(int(e), float(prof[e])) for e in rated[:K]]
            revset = set(int(e) for e in rated[:K])
            scores = np.asarray(w.predict(rev))[:nt]
            cand = np.ones(nt, bool)
            for e in revset:
                cand[e] = False
            overall[K].append(ndcg(scores, liked, cand))
            tailn[K].append(ndcg(scores, liked & tail, cand & tail))
    print(f"=== item-reveal headroom: {INST} ({nt} movies, tail=bottom-50% pop) ===")
    print(f"{'K reveals':>10}{'NDCG@10 overall':>18}{'NDCG@10 long-tail':>20}")
    for K in KS:
        print(f"{K:>10}{np.nanmean(overall[K]):>18.4f}{np.nanmean(tailn[K]):>20.4f}")
    o0, oN = np.nanmean(overall[KS[0]]), np.nanmean(overall[KS[-1]])
    t0, tN = np.nanmean(tailn[KS[0]]), np.nanmean(tailn[KS[-1]])
    print(f"\n  overall lift (K=0->{KS[-1]}):  {oN-o0:+.4f}")
    print(f"  long-tail lift (K=0->{KS[-1]}): {tN-t0:+.4f}   "
          f"({(tN-t0)/max(oN-o0,1e-6):.1f}x the overall lift)")


if __name__ == '__main__':
    main()
