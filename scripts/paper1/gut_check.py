"""
Qualitative gut-check of the instrument: reveal a concrete preference and print
the top-10 recommended movies. If the instrument is sane, "likes Gladiator"
should surface action/historical epics, "likes Toy Story" family/animation, etc.

Usage: DATASET_NAME=ml1m INST_NAME=instrument_ml1m_rank \
       poetry run python scripts/paper1/gut_check.py
"""
import os, sys
sys.path.insert(0, '.'); sys.path.insert(0, 'scripts/paper1')
import numpy as np
from test_instrument_lib import load_instrument_by_name

NAME = os.environ.get('DATASET_NAME', 'ml1m')
INST = os.environ.get('INST_NAME', f'instrument_{NAME}_rank')
NPZ = {'ml1m': 'C:/dev/phd/casper/data/movielens/ml1m_profiles.npz',
       'ml_stratified': 'C:/dev/phd/casper/data/movielens/ml_stratified_profiles.npz'}[NAME]


def main():
    w, _ = load_instrument_by_name(INST)
    d = np.load(NPZ, allow_pickle=True)
    items = [tuple(x) for x in d['items'].tolist()]
    nt = int(d['n_targets'])
    names = [str(it[2]) for it in items]
    typ = [it[0] for it in items]

    def find(sub, kind=None):
        sub = sub.lower()
        for i, nm in enumerate(names):
            if sub in nm.lower() and (kind is None or typ[i] == kind):
                return i
        return None

    def recommend(reveals, k=10):
        rev = []
        for sub, pol, kind in reveals:
            idx = find(sub, kind)
            if idx is None:
                print(f"    [!] '{sub}' not found"); continue
            rev.append((idx, 1.0 if pol == 'like' else 0.0))
        scores = np.asarray(w.predict(rev))[:nt]
        revealed_idx = {i for i, _ in rev}
        order = [i for i in np.argsort(-scores) if i not in revealed_idx][:k]
        shown = ", ".join(f"{names[i]}" for i in order)
        desc = "; ".join(f"{'+' if p=='like' else '-'}{s}" for s, p, _ in reveals)
        print(f"\n  GIVEN [{desc}]  ->  TOP {k}:")
        for i in order:
            print(f"     {names[i]}")

    print(f"=== gut check: {INST} ({nt} movies) ===")
    recommend([("Gladiator", 'like', 'movie')])
    recommend([("Toy Story", 'like', 'movie')])
    recommend([("Silence of the Lambs", 'like', 'movie'),
               ("Toy Story", 'dislike', 'movie')])
    recommend([("Matrix", 'like', 'movie')])
    # attribute-level reveals
    recommend([("Romance", 'like', 'genre')])
    recommend([("Horror", 'like', 'genre'), ("Comedy", 'dislike', 'genre')])


if __name__ == '__main__':
    main()
