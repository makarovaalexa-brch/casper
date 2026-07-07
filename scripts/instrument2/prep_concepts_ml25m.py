"""
prep_concepts_ml25m.py -- genome-tag concept vocabulary aligned to the ML-25M INSTRUMENT 2.0 arena
(ni=18430, ordinal ids = rank in np.sort(keepI) from meta.npz). Mirrors prep_concepts_ml1m.py.

Output: .cache/instrument2/concepts_ml25m.npz  (item_tag, tag_ids, tag_names, coverage, tag_mass).
EVAL-ONLY prep; no training, no commits.
"""
import os
import numpy as np
import pandas as pd

BASE = 'C:/dev/phd/casper/data/movielens'
GENOME = f'{BASE}/genome-scores.csv'
TAGS = f'{BASE}/genome-tags.csv'
META = f'{BASE}/.cache/ml25m/meta.npz'
OUT = os.path.join('.cache', 'instrument2', 'concepts_ml25m.npz')
N_TAGS = 200
COV_THRESH = 0.5


def main():
    keepI = np.load(META)['keepI']
    order = np.sort(keepI)
    mid2ord = {int(m): k for k, m in enumerate(order)}
    ni = len(order)
    print(f'[concepts-ml25m] ni={ni}', flush=True)

    gs = pd.read_csv(GENOME, dtype={'movieId': np.int64, 'tagId': np.int32, 'relevance': np.float32})
    gs = gs[gs['movieId'].isin(mid2ord)].copy()
    gs['oid'] = gs['movieId'].map(mid2ord).astype(np.int64)
    print(f'[concepts-ml25m] genome rows in vocab {len(gs):,} over {gs["movieId"].nunique()} movies '
          f'({100*gs["movieId"].nunique()/ni:.1f}%)', flush=True)

    hi = gs[gs['relevance'] >= COV_THRESH]
    cov = hi.groupby('tagId').size().sort_values(ascending=False)
    sel = cov.head(N_TAGS)
    sel_tags = sel.index.values.astype(np.int32)
    tag2col = {int(t): j for j, t in enumerate(sel_tags)}

    tagnames = pd.read_csv(TAGS).set_index('tagId')['tag'].to_dict()
    names = np.array([str(tagnames.get(int(t), str(t))) for t in sel_tags])

    item_tag = np.zeros((ni, len(sel_tags)), dtype=np.float32)
    sub = gs[gs['tagId'].isin(tag2col)]
    item_tag[sub['oid'].values, sub['tagId'].map(tag2col).values] = sub['relevance'].values
    tag_mass = item_tag.sum(0).astype(np.float32)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    np.savez_compressed(OUT, item_tag=item_tag, tag_ids=sel_tags, tag_names=names,
                        coverage=sel.values.astype(np.int32), tag_mass=tag_mass)
    print(f'[concepts-ml25m] wrote {OUT}  item_tag {item_tag.shape} '
          f'nnz/col mean {(item_tag>0).sum(0).mean():.0f}', flush=True)
    print('  sample tags:', ', '.join(names[:12]), flush=True)


if __name__ == '__main__':
    main()
