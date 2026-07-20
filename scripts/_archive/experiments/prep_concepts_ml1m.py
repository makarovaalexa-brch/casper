"""
prep_concepts_ml1m.py -- build a genome-tag concept vocabulary aligned to the CANONICAL
CASPER ML-1M arena (ni=3706 V1 ordinal ids), for INSTRUMENT 2.0 Phase-3 (W2/W3).

Genome scores (data/movielens/genome-scores.csv: movieId,tagId,relevance) share movieIds
with ML-1M (94.8% of the 3706-item vocab covered). We map movieId -> ordinal id exactly as
ml1m_arena.load_arena does (iids = {movieId: rank in np.unique(movieId over ratings.dat)}).
Select N_TAGS=200 highest-coverage tags (coverage = #vocab items with relevance>=0.5), store a
dense item_tag relevance matrix (ni x n_sel) + global tag mass for LIFT-based per-user selection.

Output: .cache/instrument2/concepts_ml1m.npz
  item_tag   (ni x n_sel) float32 relevance (0 where no genome / tag not selected)
  tag_ids    (n_sel,) int32 genome tagId
  tag_names  (n_sel,) str
  coverage   (n_sel,) int32
  tag_mass   (n_sel,) float32   global sum of relevance over the vocab (for lift = affinity/mass)
EVAL-ONLY prep, no training, no commits.
"""
import os
import numpy as np
import pandas as pd

BASE = 'C:/dev/phd/casper/data/movielens'
ML = f'{BASE}/ml-1m'
GENOME = f'{BASE}/genome-scores.csv'
TAGS = f'{BASE}/genome-tags.csv'
OUT = os.path.join('.cache', 'instrument2', 'concepts_ml1m.npz')
N_TAGS = 200
COV_THRESH = 0.5


def main():
    # reconstruct ML-1M ordinal ids EXACTLY as ml1m_arena does
    I = []
    for line in open(f'{ML}/ratings.dat'):
        a = line.strip().split('::'); I.append(int(a[1]))
    I = np.array(I)
    uniq = np.unique(I); ni = len(uniq)
    mid2ord = {int(m): k for k, m in enumerate(uniq)}
    print(f'[concepts-ml1m] ni={ni}', flush=True)

    gs = pd.read_csv(GENOME, dtype={'movieId': np.int64, 'tagId': np.int32, 'relevance': np.float32})
    gs = gs[gs['movieId'].isin(mid2ord)].copy()
    gs['oid'] = gs['movieId'].map(mid2ord).astype(np.int64)
    print(f'[concepts-ml1m] genome rows in vocab {len(gs):,} over {gs["movieId"].nunique()} movies '
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
    print(f'[concepts-ml1m] wrote {OUT}  item_tag {item_tag.shape} '
          f'nnz/col mean {(item_tag>0).sum(0).mean():.0f}', flush=True)
    print('  sample tags:', ', '.join(names[:12]), flush=True)


if __name__ == '__main__':
    main()
