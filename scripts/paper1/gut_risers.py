"""
Permutation gut-check (the right one): reveal ONE entity as +liked vs -disliked
and show the titles whose score MOVES the most (top risers / fallers) vs the
no-reveal baseline. Strips the popularity floor (which is constant across the
permutation), so it isolates what the instrument actually learned about an entity.
e.g. +Gladiator should lift other Russell-Crowe / epic titles.

Usage: DATASET_NAME=ml1m INST_NAME=instrument_ml1m_rank \
       poetry run python scripts/paper1/gut_risers.py
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
    items = [tuple(x) for x in d['items'].tolist()]; nt = int(d['n_targets'])
    names = [str(it[2]) for it in items]; typ = [it[0] for it in items]

    def find(sub, kind=None):
        s = sub.lower()
        hits = [i for i, n in enumerate(names) if s in n.lower() and (kind is None or typ[i] == kind)]
        return hits[0] if hits else None

    s0 = np.asarray(w.predict([]))[:nt]

    def risers(sub, kind=None, k=10):
        idx = find(sub, kind)
        if idx is None:
            print(f"\n  [!] '{sub}' not found"); return
        sp = np.asarray(w.predict([(idx, 1.0)]))[:nt]
        sm = np.asarray(w.predict([(idx, 0.0)]))[:nt]
        up = sp - s0; down = sm - s0
        if idx < nt:
            up[idx] = -1e9; down[idx] = -1e9
        top_up = np.argsort(-up)[:k]
        top_dn = np.argsort(down)[:k]   # most negative = falls most when disliked
        print(f"\n  ENTITY: {names[idx]}  ({typ[idx]})")
        print(f"    titles that RISE most when +liked:")
        for i in top_up:
            print(f"       +{up[i]:.3f}  {names[i]}")
        print(f"    titles that FALL most when -disliked:")
        for i in top_dn:
            print(f"       {down[i]:.3f}  {names[i]}")

    print(f"=== permutation risers/fallers: {INST} ===")
    # Russell Crowe titles in ML-1M: Gladiator, L.A. Confidential, The Insider
    risers("Gladiator", 'movie')
    risers("L.A. Confidential", 'movie')
    risers("Silence of the Lambs", 'movie')
    risers("Toy Story", 'movie')
    # genre / tag level
    risers("Romance", 'genre')
    risers("Horror", 'genre')
    # any actor tags present?
    for actor in ['crowe', 'hanks', 'schwarzenegger']:
        if find(actor, 'tag') is not None:
            risers(actor, 'tag')


if __name__ == '__main__':
    main()
