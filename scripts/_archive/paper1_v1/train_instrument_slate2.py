"""
Paper 1 / #14: Slate 2 -- the less-popular, larger slate (core experiment).

Motivation (from the harmonized Slate-1 results): on a top-100 blockbuster
slate, popularity-style static playlists are near-optimal because the most
informative questions are the same for every user, and the accuracy band is
compressed (prior 0.70, ceiling 0.85). Slate 2 changes the regime:

  - MOVIES: popularity ranks 101-400 (300 mid-popularity movies):
    answerability now varies strongly by user, so static playlists waste
    turns and adaptivity should pay.
  - ATTRIBUTES: 19 genres + the top 60 genome tags by total relevance over
    slate movies (tag matches movie iff relevance >= 0.6): a much richer
    semantic question vocabulary than cast/crew.

Profiles: users with >= 20 rated slate movies; movie label = rating >= 4;
attribute label = absolute mean over matching rated movies (>= 4) with
support >= 3 (canonical Slate-1 scheme).

Instrument: SetEncoderInstrument (v5 architecture), reveal-subset training,
same recipe and acceptance philosophy as Slate 1.

Run from casper root:
    poetry run python scripts/paper1/train_instrument_slate2.py
Output: data/movielens/.cache/checkpoints/instrument_slate2_set.pt
"""

import sys
sys.path.insert(0, '.')
sys.path.insert(0, 'scripts/paper1')

import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from train_instrument_v5 import SetEncoderInstrument

DATA_DIR = Path('C:/dev/phd/casper/data/movielens')
CHECKPOINT_DIR = DATA_DIR / '.cache' / 'checkpoints'
OUT_PATH = CHECKPOINT_DIR / 'instrument_slate2_set.pt'

SEED = 42
MOVIE_RANK_START, MOVIE_RANK_END = 100, 400   # ranks 101..400
N_TAGS = 60
TAG_MATCH_THRESHOLD = 0.6
MIN_USER_RATINGS = 20
ATTR_MIN_SUPPORT = 3

N_EPOCHS = 80
BATCH_SIZE = 64
LEARNING_RATE = 5e-4
D_MODEL = 128
MAX_TRAIN_USERS = 12000
MAX_VAL_USERS = 1500
ATTR_ONLY_PROB = 0.15
EMPTY_PROB = 0.05
EARLY_STOP_PATIENCE = 10
MAX_REVEAL = 60

np.random.seed(SEED)
torch.manual_seed(SEED)


def build_slate():
    ratings = pd.read_csv(DATA_DIR / 'ratings.csv')
    movies_df = pd.read_csv(DATA_DIR / 'movies.csv')
    counts = ratings['movieId'].value_counts()
    slate_movies = counts.iloc[MOVIE_RANK_START:MOVIE_RANK_END].index.tolist()
    titles = movies_df.set_index('movieId')['title'].to_dict()
    genres_df = movies_df.set_index('movieId')['genres'].to_dict()

    # genres present in slate
    genres = sorted({g for mid in slate_movies
                     for g in str(genres_df.get(mid, '')).split('|')
                     if g and g != '(no genres listed)'})

    # genome tags: top N by total relevance over slate movies
    gscores = pd.read_csv(DATA_DIR / 'genome-scores.csv')
    gtags = pd.read_csv(DATA_DIR / 'genome-tags.csv').set_index('tagId')['tag']
    gs = gscores[gscores['movieId'].isin(slate_movies)]
    tag_total = gs.groupby('tagId')['relevance'].sum().nlargest(N_TAGS)
    top_tags = tag_total.index.tolist()

    # tag -> matching slate movies
    gs_top = gs[gs['tagId'].isin(top_tags) & (gs['relevance'] >= TAG_MATCH_THRESHOLD)]
    tag_movies = gs_top.groupby('tagId')['movieId'].apply(list).to_dict()

    items = [('movie', mid, titles.get(mid, f'Movie {mid}')) for mid in slate_movies]
    items += [('genre', g, g) for g in genres]
    items += [('tag', int(t), str(gtags.loc[t])) for t in top_tags]

    # movie -> attribute indices
    n_movies = len(slate_movies)
    movie_pos = {mid: i for i, mid in enumerate(slate_movies)}
    attr_of_movie = {mid: [] for mid in slate_movies}
    for j, (itype, ival, _) in enumerate(items):
        if itype == 'genre':
            for mid in slate_movies:
                if ival in str(genres_df.get(mid, '')).split('|'):
                    attr_of_movie[mid].append(j)
        elif itype == 'tag':
            for mid in tag_movies.get(ival, []):
                if mid in movie_pos:
                    attr_of_movie[mid].append(j)

    ratings_f = ratings[ratings['movieId'].isin(movie_pos)]
    return items, n_movies, movie_pos, attr_of_movie, ratings_f


def build_profiles_slate2(items, n_items, movie_pos, attr_of_movie,
                          ratings_f, user_ids):
    profiles = {}
    grouped = ratings_f[ratings_f['userId'].isin(set(user_ids))].groupby('userId')
    for uid, grp in grouped:
        vec = np.full(n_items, np.nan, dtype=np.float32)
        s = np.zeros(n_items)
        c = np.zeros(n_items)
        for r in grp.itertuples():
            vec[movie_pos[r.movieId]] = 1.0 if r.rating >= 4 else 0.0
            for ai in attr_of_movie[r.movieId]:
                s[ai] += r.rating
                c[ai] += 1
        has = c >= ATTR_MIN_SUPPORT
        means = np.full(n_items, np.nan)
        means[has] = s[has] / c[has]
        vec[means >= 4.0] = 1.0
        vec[(means < 4.0) & has] = 0.0
        profiles[uid] = vec
    return profiles


def main():
    print("Building slate 2...")
    t0 = time.time()
    items, n_movies, movie_pos, attr_of_movie, ratings_f = build_slate()
    n_items = len(items)
    attr_indices = np.array([i for i, it in enumerate(items) if it[0] != 'movie'])
    print(f"  {n_items} items ({n_movies} movies, {len(attr_indices)} attributes) "
          f"in {time.time() - t0:.0f}s")

    user_counts = ratings_f.groupby('userId').size()
    dense = user_counts[user_counts >= MIN_USER_RATINGS].index.to_numpy()
    print(f"  users with >= {MIN_USER_RATINGS} slate ratings: {len(dense)}")
    np.random.seed(SEED)
    shuffled = dense.copy()
    np.random.shuffle(shuffled)
    split = int(0.8 * len(shuffled))
    train_users = shuffled[:split][:MAX_TRAIN_USERS]
    val_users = shuffled[split:][:MAX_VAL_USERS]

    print("Building profiles...")
    t0 = time.time()
    profiles = build_profiles_slate2(items, n_items, movie_pos, attr_of_movie,
                                     ratings_f,
                                     np.concatenate([train_users, val_users]))
    print(f"  {len(profiles)} profiles in {time.time() - t0:.0f}s")

    class DS(torch.utils.data.Dataset):
        def __init__(self, uids):
            self.uids = [u for u in uids if u in profiles]

        def __len__(self):
            return len(self.uids)

        def __getitem__(self, i):
            full = profiles[self.uids[i]]
            rated = np.where(~np.isnan(full))[0]
            if np.random.random() < EMPTY_PROB or len(rated) == 0:
                sel = np.array([], dtype=int)
            else:
                cand = rated
                if np.random.random() < ATTR_ONLY_PROB:
                    ac = np.intersect1d(rated, attr_indices)
                    if len(ac):
                        cand = ac
                k = int(np.exp(np.random.uniform(
                    0, np.log(min(len(cand), MAX_REVEAL) + 1))))
                k = max(1, min(k, len(cand)))
                sel = np.random.choice(cand, size=k, replace=False)
            idx = np.zeros(MAX_REVEAL, dtype=np.int64)
            pol = np.zeros(MAX_REVEAL, dtype=np.int64)
            pad = np.ones(MAX_REVEAL, dtype=bool)
            for j, e in enumerate(sel):
                idx[j] = e
                pol[j] = int(full[e] >= 0.5)
                pad[j] = False
            return (torch.from_numpy(idx), torch.from_numpy(pol),
                    torch.from_numpy(pad), torch.FloatTensor(full))

    def masked_bce(pred, tgt):
        mask = torch.isnan(tgt)
        pred = torch.where(mask, torch.zeros_like(pred), pred)
        tgt = torch.where(mask, torch.zeros_like(tgt), tgt)
        per = nn.functional.binary_cross_entropy_with_logits(
            pred, tgt, reduction='none')
        per = torch.where(mask, torch.zeros_like(per), per)
        return per.sum() / (~mask).float().sum().clamp(min=1)

    model = SetEncoderInstrument(n_items, d_model=D_MODEL)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    opt = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    tl = torch.utils.data.DataLoader(DS(train_users), batch_size=BATCH_SIZE,
                                     shuffle=True)
    vl = torch.utils.data.DataLoader(DS(val_users), batch_size=BATCH_SIZE)

    best, best_epoch = float('inf'), -1
    t0 = time.time()
    for epoch in range(N_EPOCHS):
        model.train()
        tr = 0.0
        for idx, pol, pad, tgt in tl:
            opt.zero_grad()
            loss = masked_bce(model(idx, pol, pad), tgt)
            loss.backward()
            opt.step()
            tr += loss.item()
        tr /= max(len(tl), 1)
        model.eval()
        va = 0.0
        with torch.no_grad():
            for idx, pol, pad, tgt in vl:
                va += masked_bce(model(idx, pol, pad), tgt).item()
        va /= max(len(vl), 1)
        marker = ''
        if va < best:
            best, best_epoch = va, epoch
            marker = ' *'
            torch.save({
                'model_state_dict': model.state_dict(),
                'arch': 'set_encoder', 'd_model': D_MODEL,
                'n_heads': 4, 'n_layers': 2,
                'n_items': n_items, 'n_movies': n_movies, 'items': items,
                'val_loss': va, 'epochs': epoch + 1,
                'config': {'slate': 'ranks 101-400 + genres + genome tags',
                           'labels': 'absolute, attr support >= 3',
                           'max_reveal': MAX_REVEAL, 'seed': SEED},
            }, OUT_PATH)
        print(f"Epoch {epoch + 1:3d}/{N_EPOCHS} | train {tr:.4f} | "
              f"val {va:.4f}{marker} | {time.time() - t0:.0f}s", flush=True)
        if epoch - best_epoch >= EARLY_STOP_PATIENCE:
            print("Early stop")
            break

    print(f"\nBest val {best:.4f} (epoch {best_epoch + 1}); saved {OUT_PATH}")


if __name__ == '__main__':
    main()
