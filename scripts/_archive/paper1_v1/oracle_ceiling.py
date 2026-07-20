"""
Adaptive CEILING diagnostic: a clairvoyant-greedy oracle that, each turn,
picks the question maximizing TRUE post-reveal accuracy (uses ground-truth
labels). This is an upper bound on what ANY 1-step adaptive policy -- incl.
a perfectly-trained RL agent -- could achieve via per-user routing.

If oracle-adaptive ~= best-static, static is genuinely near-optimal and a
static RL solution is correct (no tweaking helps). If oracle-adaptive >>
best-static, real adaptive headroom exists that the RL agent failed to find.

Usage: DATASET_NAME=ml_stratified poetry run python scripts/paper1/oracle_ceiling.py
"""
import os, sys
sys.path.insert(0, '.'); sys.path.insert(0, 'scripts/paper1')
import numpy as np
from test_instrument_lib import load_instrument_by_name
from testbed import _movie_metrics

NAME = os.environ.get('DATASET_NAME', 'ml_stratified')
NPZ = {'ml_stratified': 'C:/dev/phd/casper/data/movielens/ml_stratified_profiles.npz',
       'yelp_multicity': 'C:/dev/phd/casper/data/yelp/yelp_multicity_profiles.npz'}[NAME]
N_TURNS = 15
N_USERS = int(os.environ.get('ORACLE_USERS', 80))


def answer_polarity(v):
    if np.isnan(v):
        return None
    return 1.0 if v >= 0.5 else 0.0


def main():
    w, _ = load_instrument_by_name(f'instrument_{NAME}')
    d = np.load(NPZ, allow_pickle=True)
    test = d['test'][:N_USERS]
    nm = w.n_movies
    n_items = w.n_items

    oracle_auac, static_branch = [], []
    seqs = set()
    for prof in test:
        revealed = []
        asked = set()
        accs = [_movie_metrics(w.predict(revealed), prof, nm)[0]]
        qseq = []
        for _ in range(N_TURNS):
            cands = [q for q in range(n_items) if q not in asked]
            # build hypothetical reveal sets (unknown -> revealed unchanged)
            sets, pol = [], {}
            for q in cands:
                p = answer_polarity(prof[q])
                pol[q] = p
                sets.append(revealed + ([(q, p)] if p is not None else []))
            preds = w.predict_batch(sets)            # [n_cands, n_movies]
            gt = prof[:nm]; filt = ~np.isnan(gt)
            y = gt[filt]
            acc = ((preds[:, filt] > 0.5) == y).mean(1)  # accuracy per candidate
            best = int(np.argmax(acc))
            q = cands[best]
            asked.add(q); qseq.append(q)
            if pol[q] is not None:
                revealed.append((q, pol[q]))
            accs.append(_movie_metrics(w.predict(revealed), prof, nm)[0])
        oracle_auac.append(float(np.mean(accs)))
        seqs.add(tuple(qseq))
    print(f"{NAME}: oracle clairvoyant-greedy over {len(test)} users")
    print(f"  ORACLE-ADAPTIVE AUAC = {np.mean(oracle_auac):.4f}  "
          f"(final-turn acc {np.mean([a for a in oracle_auac]):.4f})")
    print(f"  distinct oracle sequences = {len(seqs)}/{len(test)} "
          f"(adaptive routing => near-unique per user)")


if __name__ == '__main__':
    main()
