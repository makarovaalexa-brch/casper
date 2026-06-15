"""
MovieLens-1M COMPLETE catalog -> profiles npz. Standard, recognised, complete
(no bespoke slate): every movie in ML-1M is a target. Askable attributes =
18 genres + decade buckets + (optional) ML-25M genome tags joined by title.

This replaces the bespoke stratified-300 slate with a standard complete dataset
for the reviewer-proof main result.

Run: poetry run python scripts/paper1/build_movielens_1m.py
Env: USE_GENOME=1 (default) joins genome tags; N_TAGS=60.
"""
import os, sys
sys.path.insert(0, '.'); sys.path.insert(0, 'scripts/paper1')
import re
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd
from adaptivity_headroom import headroom, verdict

DATA = Path('C:/dev/phd/casper/data/movielens')
ML1M = DATA / 'ml-1m'
EXP = Path('C:/dev/phd/casper/experiments/paper1')
OUT = DATA / 'ml1m_profiles.npz'
USE_GENOME = os.environ.get('USE_GENOME', '1') == '1'
N_TAGS = int(os.environ.get('N_TAGS', 60))
MIN_USER_ITEMS = 8
MIN_ATTR_SUPPORT = 3
SEED = 42
rng = np.random.default_rng(SEED)
YEAR = re.compile(r'\((\d{4})\)\s*$')


def main():
    ratings = pd.read_csv(ML1M / 'ratings.dat', sep='::', engine='python',
                          names=['userId', 'movieId', 'rating', 'ts'], encoding='latin-1')
    movies = pd.read_csv(ML1M / 'movies.dat', sep='::', engine='python',
                         names=['movieId', 'title', 'genres'], encoding='latin-1')
    chosen = sorted(ratings['movieId'].unique().tolist())
    movie_idx = {m: i for i, m in enumerate(chosen)}
    n_movies = len(chosen)
    title = dict(zip(movies.movieId, movies.title))
    print(f"ML-1M complete catalog: {n_movies} movies, {ratings.userId.nunique()} users, "
          f"{len(ratings)} ratings", flush=True)

    genres = sorted({g for gs in movies.genres for g in str(gs).split('|')
                     if g and g != '(no genres listed)'})
    decades = sorted({f"{int(m.group(1))//10*10}s" for t in title.values()
                      if (m := YEAR.search(str(t)))})
    g_in = {g: n_movies + i for i, g in enumerate(genres)}
    d_in = {dd: n_movies + len(genres) + i for i, dd in enumerate(decades)}
    base_attr = n_movies + len(genres) + len(decades)

    movie_genres = {r.movieId: set(str(r.genres).split('|')) for r in movies.itertuples()}
    movie_decade = {}
    for mid, t in title.items():
        mm = YEAR.search(str(t))
        if mm:
            movie_decade[mid] = f"{int(mm.group(1))//10*10}s"

    # optional: genome tags joined ML-1M title -> ML-25M movieId -> top tags
    t_in = {}; tag_movies = defaultdict(list); tag_name = {}
    if USE_GENOME:
        m25 = pd.read_csv(DATA / 'movies.csv')
        title2id25 = dict(zip(m25.title, m25.movieId))
        mapped = {mid: title2id25[str(t)] for mid, t in title.items() if str(t) in title2id25}
        print(f"  genome: matched {len(mapped)}/{n_movies} movies to ML-25M by title", flush=True)
        ids25 = set(mapped.values())
        gs = pd.read_csv(DATA / 'genome-scores.csv')
        gs = gs[gs.movieId.isin(ids25)]
        top_tags = gs.groupby('tagId')['relevance'].sum().nlargest(N_TAGS).index.tolist()
        gtags = pd.read_csv(DATA / 'genome-tags.csv').set_index('tagId')['tag']
        t_in = {t: base_attr + i for i, t in enumerate(top_tags)}
        tag_name = {t: str(gtags.loc[t]) for t in top_tags}
        id25_to_1m = {v: k for k, v in mapped.items()}
        rel = gs[gs.tagId.isin(top_tags) & (gs.relevance >= 0.5)]
        for r in rel.itertuples():
            tag_movies[r.tagId].append(id25_to_1m[r.movieId])
    n_items = base_attr + len(t_in)

    movie_attrs = defaultdict(list)
    for m in movie_idx:
        for g in movie_genres.get(m, set()):
            if g in g_in: movie_attrs[m].append(g_in[g])
        if m in movie_decade: movie_attrs[m].append(d_in[movie_decade[m]])
    for t, ms in tag_movies.items():
        for m in ms:
            if m in movie_idx: movie_attrs[m].append(t_in[t])
    print(f"  slate: {n_items} items = {n_movies} movies + {len(genres)} genres + "
          f"{len(decades)} decades + {len(t_in)} genome tags", flush=True)

    profiles = []
    for uid, grp in ratings.groupby('userId'):
        if len(grp) < MIN_USER_ITEMS:
            continue
        vec = np.full(n_items, np.nan, dtype=np.float32)
        asum = np.zeros(n_items); acnt = np.zeros(n_items)
        for r in grp.itertuples():
            vec[movie_idx[r.movieId]] = 1.0 if r.rating >= 4 else 0.0
            for ai in movie_attrs[r.movieId]:
                asum[ai] += r.rating; acnt[ai] += 1
        has = acnt >= MIN_ATTR_SUPPORT
        means = np.full(n_items, np.nan); means[has] = asum[has] / acnt[has]
        vec[means >= 4] = 1.0; vec[(means < 4) & has] = 0.0
        profiles.append(vec)
    profiles = np.stack(profiles)
    nt = (~np.isnan(profiles[:, :n_movies])).sum(1)
    print(f"  users={len(profiles)}, mean movies/user={nt.mean():.1f}", flush=True)

    idx = rng.permutation(len(profiles)); split = int(0.8 * len(idx))
    items_meta = ([('movie', int(m), str(title.get(m, m))) for m in chosen] +
                  [('genre', g, g) for g in genres] +
                  [('decade', dd, dd) for dd in decades] +
                  [('tag', int(t), tag_name[t]) for t in t_in])
    np.savez(OUT, train=profiles[idx[:split]], test=profiles[idx[split:]],
             items=np.array(items_meta, dtype=object), n_targets=n_movies)
    print(f"  saved {OUT}  (n_targets={n_movies}, n_items={n_items})")

    samp = list(profiles[rng.choice(len(profiles), min(3000, len(profiles)), replace=False)])
    h = headroom(samp); h['verdict'] = verdict(h)
    allh = {}
    if (EXP / 'adaptivity_headroom.json').exists():
        import json; allh = json.loads((EXP / 'adaptivity_headroom.json').read_text())
    allh['ml1m'] = h
    import json; (EXP / 'adaptivity_headroom.json').write_text(json.dumps(allh, indent=2))
    print(f"  HEADROOM@10={h['headroom@10']:.3f} jaccard={h['pair_jaccard_answerable']:.3f} "
          f"ans_rate={h['answer_rate_mean']:.2f}")


if __name__ == '__main__':
    main()
