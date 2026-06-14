"""
Stratified MovieLens slate — tests whether 'static wins' was an artifact
of popularity-biased slate construction.

Prior slates took the TOP-M most popular items: popular targets the prior
already predicts (turn-0 acc ~0.70), answerable by everyone, so static
'ask popular things' looks optimal. Real recommendation value is in the
TAIL — niche items the prior can't predict, where you MUST elicit.

This slate samples movies EVENLY across popularity-rank bands (head -> long
tail), so targets include prior-unpredictable niche items and answerable
sets are genuinely heterogeneous. If adaptive beats static HERE, the
earlier null was a slate-construction artifact.

Run: poetry run python scripts/paper1/build_movielens_stratified.py
"""

import sys
sys.path.insert(0, '.'); sys.path.insert(0, 'scripts/paper1')
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd
from adaptivity_headroom import headroom, verdict

DATA = Path('C:/dev/phd/casper/data/movielens')
EXP = Path('C:/dev/phd/casper/experiments/paper1')
OUT = DATA / 'ml_stratified_profiles.npz'
SEED = 42
# popularity-rank bands -> movies sampled per band (even across the spectrum)
BANDS = [(0, 100, 50), (100, 400, 60), (400, 1000, 60),
         (1000, 2500, 60), (2500, 6000, 70)]   # 300 movies, head..tail
N_TAGS = 40
MIN_USER_ITEMS = 8
SEED = 42
rng = np.random.default_rng(SEED)


def main():
    ratings = pd.read_csv(DATA / 'ratings.csv')
    movies = pd.read_csv(DATA / 'movies.csv')
    counts = ratings['movieId'].value_counts()          # popularity rank
    ranked = counts.index.tolist()
    chosen = []
    for lo, hi, k in BANDS:
        band = ranked[lo:hi]
        pick = rng.choice(band, size=min(k, len(band)), replace=False)
        chosen.extend(int(m) for m in pick)
    chosen = list(dict.fromkeys(chosen))
    movie_idx = {m: i for i, m in enumerate(chosen)}
    n_movies = len(chosen)
    print(f"stratified slate: {n_movies} movies across bands "
          f"(ranks {BANDS[0][0]}-{BANDS[-1][1]}); "
          f"pop range {counts[chosen].min()}-{counts[chosen].max()} ratings/movie")

    # genres + top genome tags over chosen movies as attributes
    genres = sorted({g for _, row in movies[movies.movieId.isin(movie_idx)].iterrows()
                     if pd.notna(row.genres)
                     for g in row.genres.split('|') if g != '(no genres listed)'})
    gscores = pd.read_csv(DATA / 'genome-scores.csv')
    gtags = pd.read_csv(DATA / 'genome-tags.csv').set_index('tagId')['tag']
    gs = gscores[gscores.movieId.isin(movie_idx)]
    top_tags = gs.groupby('tagId')['relevance'].sum().nlargest(N_TAGS).index.tolist()
    g_in = {g: n_movies + i for i, g in enumerate(genres)}
    t_in = {t: n_movies + len(genres) + i for i, t in enumerate(top_tags)}
    n_items = n_movies + len(genres) + len(top_tags)
    genres_by_movie = {r.movieId: set(str(r.genres).split('|'))
                       for r in movies[movies.movieId.isin(movie_idx)].itertuples()}
    tag_movies = defaultdict(list)
    for r in gs[gs.tagId.isin(top_tags) & (gs.relevance >= 0.5)].itertuples():
        tag_movies[r.tagId].append(r.movieId)
    movie_attrs = defaultdict(list)
    for m in movie_idx:
        for g in genres_by_movie.get(m, set()):
            if g in g_in: movie_attrs[m].append(g_in[g])
    for t, ms in tag_movies.items():
        for m in ms:
            if m in movie_idx: movie_attrs[m].append(t_in[t])

    rf = ratings[ratings.movieId.isin(movie_idx)]
    print(f"slate ratings: {len(rf)}  building profiles...", flush=True)
    profiles = []
    for uid, grp in rf.groupby('userId'):
        if len(grp) < MIN_USER_ITEMS:
            continue
        vec = np.full(n_items, np.nan, dtype=np.float32)
        asum = np.zeros(n_items); acnt = np.zeros(n_items)
        for r in grp.itertuples():
            vec[movie_idx[r.movieId]] = 1.0 if r.rating >= 4 else 0.0
            for ai in movie_attrs[r.movieId]:
                asum[ai] += r.rating; acnt[ai] += 1
        has = acnt >= 3
        means = np.full(n_items, np.nan); means[has] = asum[has]/acnt[has]
        vec[means >= 4] = 1.0; vec[(means < 4) & has] = 0.0
        profiles.append(vec)
    profiles = np.stack(profiles)
    nt = (~np.isnan(profiles[:, :n_movies])).sum(1)
    print(f"users={len(profiles)}, mean movies/user={nt.mean():.1f}, slate={n_items} "
          f"({n_movies} movies + {len(genres)} genres + {len(top_tags)} tags)")

    idx = rng.permutation(len(profiles)); split = int(0.8*len(idx))
    items_meta = ([('movie', int(m), f'movie_{m}') for m in chosen] +
                  [('genre', g, g) for g in genres] +
                  [('tag', int(t), str(gtags.loc[t])) for t in top_tags])
    np.savez(OUT, train=profiles[idx[:split]], test=profiles[idx[split:]],
             items=np.array(items_meta, dtype=object), n_targets=n_movies)

    samp = list(profiles[rng.choice(len(profiles), min(3000, len(profiles)), replace=False)])
    h = headroom(samp); h['verdict'] = verdict(h)
    allh = json.loads((EXP/'adaptivity_headroom.json').read_text())
    allh['movielens_stratified'] = h
    (EXP/'adaptivity_headroom.json').write_text(json.dumps(allh, indent=2))
    print(f"\nSTRATIFIED HEADROOM@10={h['headroom@10']:.3f} jaccard={h['pair_jaccard_answerable']:.3f} "
          f"ans_rate={h['answer_rate_mean']:.2f} static@10={h['static_cov@10']:.3f}")
    print("  (top-popular anchors: slate1 0.169, slate2 0.241)")


if __name__ == '__main__':
    main()
