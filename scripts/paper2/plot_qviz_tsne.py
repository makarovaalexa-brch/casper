#!/usr/bin/env python
"""Joint 2-D t-SNE of the BEST model's emitted QUESTIONS together with MOVIES and TAGS (concepts),
in the recommender's R^64 space (cosine metric). Shows the learned questions live OFF-manifold,
between the named movies and tags. Input: data/movielens/.cache/qviz.npz (from QVIZ block)."""
import sys
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

BASE = 'C:/dev/phd/casper'
NPZ = sys.argv[1] if len(sys.argv) > 1 else f'{BASE}/data/movielens/.cache/qviz.npz'
OUT = sys.argv[2] if len(sys.argv) > 2 else 'C:/dev/phd/papers/paper3_casper/fig_qspace_tsne'
NMOV = 1800   # subsample movies for legibility

d = np.load(NPZ, allow_pickle=True)
ques, qturn = d['questions'], d['qturn']
mov, mpop = d['movies'], d['movie_pop']
con = d['concepts']

rng = np.random.default_rng(0)
# subsample movies (keep a popularity-stratified sample so the catalog shape shows)
mi = rng.choice(len(mov), size=min(NMOV, len(mov)), replace=False)
mov_s = mov[mi]
# subsample questions a touch if huge
if len(ques) > 1800:
    qi = rng.choice(len(ques), size=1800, replace=False); ques, qturn = ques[qi], qturn[qi]

X = np.vstack([mov_s, con, ques]).astype(np.float32)
nmov, ncon, nque = len(mov_s), len(con), len(ques)
print(f't-SNE on {len(X)} pts: {nmov} movies + {ncon} tags + {nque} questions (cosine)...')
emb = TSNE(n_components=2, metric='cosine', init='pca', perplexity=40,
           learning_rate='auto', random_state=0).fit_transform(X)
Em, Ec_, Eq = emb[:nmov], emb[nmov:nmov+ncon], emb[nmov+ncon:]

fig, ax = plt.subplots(figsize=(8.2, 7.0))
ax.scatter(Em[:,0], Em[:,1], s=5,  c='#c8c8c8', alpha=0.45, lw=0, label=f'movies (n={len(mov)})', zorder=1)
ax.scatter(Ec_[:,0], Ec_[:,1], s=22, c='#e69138', alpha=0.80, lw=0, marker='^', label=f'tags / concepts (n={ncon})', zorder=2)
sc = ax.scatter(Eq[:,0], Eq[:,1], s=16, c=qturn, cmap='winter', alpha=0.75, lw=0, label=f'learned questions (n={nque})', zorder=3)
cb = fig.colorbar(sc, ax=ax, fraction=0.035, pad=0.02); cb.set_label('question turn (0=opener → 7)', fontsize=8)
ax.set_xticks([]); ax.set_yticks([])
ax.set_title('Where the learned questions live: off-manifold, between movies and tags\n(joint t-SNE in the recommender\'s embedding space, cosine)', fontsize=11)
ax.legend(loc='upper right', fontsize=9, framealpha=0.92, markerscale=1.4)
fig.tight_layout()
for ext in ('pdf', 'png'):
    fig.savefig(f'{OUT}.{ext}', bbox_inches='tight', dpi=160); print('wrote', f'{OUT}.{ext}')
