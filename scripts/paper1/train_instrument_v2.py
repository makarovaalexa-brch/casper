"""
Paper 1 / M1: Train instrument v2 (onehot extrapolation model with
reveal-subset augmentation).

Diagnosis from acceptance tests on the January checkpoints:
- onehot_paper_config passes monotonicity (rho=1.0, 66->92%) and franchise
  flips (91.7%) but scores 68% on single-attribute polarity, because the
  original training revealed the user's FULL profile at every sample: inputs
  with only 1-5 revealed items were never in-distribution.
- Fix: at each training sample, reveal only a random subset of k rated items
  (log-uniform k, 15% attribute-only episodes); target remains the full
  rated profile. Slate order shuffled per sample; loss at every LSTM
  position (each prefix is a smaller reveal -> built-in curriculum).

Same architecture, same item slate (from existing checkpoint, authoritative
ordering), same user filtering and 80/20 split (seed 42).

Run from casper root:
    poetry run python scripts/paper1/train_instrument_v2.py
Output: data/movielens/.cache/checkpoints/instrument_v2_onehot.pt
"""

import sys
sys.path.insert(0, '.')
sys.path.insert(0, 'scripts/paper1')

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from test_instrument_lib import ExtrapolationModel

DATA_DIR = Path('C:/dev/phd/casper/data/movielens')
CHECKPOINT_DIR = DATA_DIR / '.cache' / 'checkpoints'
OUT_PATH = CHECKPOINT_DIR / 'instrument_v2_onehot.pt'

SEED = 42
N_EPOCHS = 60
BATCH_SIZE = 32
LEARNING_RATE = 0.001
MAX_TRAIN_USERS = 12000
MAX_VAL_USERS = 1500
ATTR_ONLY_PROB = 0.15
EARLY_STOP_PATIENCE = 10

np.random.seed(SEED)
torch.manual_seed(SEED)

# ---------------------------------------------------------------------------
# Item slate: authoritative ordering from the existing checkpoint
# ---------------------------------------------------------------------------

ref_ckpt = torch.load(CHECKPOINT_DIR / 'onehot_paper_config.pt', weights_only=False)
ITEMS = ref_ckpt['items']
N_ITEMS = len(ITEMS)
N_MOVIES = ref_ckpt.get('n_movies', 100)
movie_to_idx = {it[1]: i for i, it in enumerate(ITEMS) if it[0] == 'movie'}
attr_indices = [i for i, it in enumerate(ITEMS) if it[0] != 'movie']
print(f"Slate: {N_ITEMS} items ({N_MOVIES} movies, {len(attr_indices)} attributes)")

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

print("Loading ratings...")
t0 = time.time()
ratings = pd.read_csv(DATA_DIR / 'ratings.csv')
movies_df = pd.read_csv(DATA_DIR / 'movies.csv')
print(f"  {time.time() - t0:.0f}s")

credits_path = DATA_DIR / '.cache' / 'credits_top100_actors5.json'
credits_data = json.loads(credits_path.read_text()) if credits_path.exists() else {}

# movie -> attribute slate indices
genres_in_slate = {it[1]: i for i, it in enumerate(ITEMS) if it[0] == 'genre'}
actors_in_slate = {it[1]: i for i, it in enumerate(ITEMS) if it[0] == 'actor'}
directors_in_slate = {it[1]: i for i, it in enumerate(ITEMS) if it[0] == 'director'}

movie_attrs = {}
for mid in movie_to_idx:
    attrs = []
    row = movies_df[movies_df['movieId'] == mid]
    if len(row) and pd.notna(row.iloc[0]['genres']):
        attrs += [genres_in_slate[g] for g in row.iloc[0]['genres'].split('|')
                  if g in genres_in_slate]
    mstr = str(mid)
    attrs += [actors_in_slate[a] for a in credits_data.get('movie_actors', {}).get(mstr, [])[:5]
              if a in actors_in_slate]
    attrs += [directors_in_slate[d] for d in credits_data.get('movie_directors', {}).get(mstr, [])
              if d in directors_in_slate]
    movie_attrs[mid] = attrs

ratings_f = ratings[ratings['movieId'].isin(movie_to_idx)].copy()
user_counts = ratings_f.groupby('userId').size()
dense_users = user_counts[user_counts >= 25].index.to_numpy()
print(f"Users with >=25 ratings on slate movies: {len(dense_users)}")

# match the original split protocol (seed 42 shuffle, 80/20), then cap
np.random.seed(SEED)
shuffled = dense_users.copy()
np.random.shuffle(shuffled)
split = int(0.8 * len(shuffled))
train_users = shuffled[:split][:MAX_TRAIN_USERS]
val_users = shuffled[split:][:MAX_VAL_USERS]
print(f"Train users: {len(train_users)}, Val users: {len(val_users)}")

print("Building per-user profiles (movies + derived attribute labels)...")
t0 = time.time()
profiles = {}
keep = set(train_users) | set(val_users)
grouped = ratings_f[ratings_f['userId'].isin(keep)].groupby('userId')
for uid, grp in grouped:
    vec = np.full(N_ITEMS, np.nan, dtype=np.float32)
    attr_sums = np.zeros(N_ITEMS)
    attr_cnts = np.zeros(N_ITEMS)
    for r in grp.itertuples():
        idx = movie_to_idx[r.movieId]
        vec[idx] = 1.0 if r.rating >= 4 else 0.0
        for ai in movie_attrs[r.movieId]:
            attr_sums[ai] += r.rating
            attr_cnts[ai] += 1
    has = attr_cnts > 0
    vec[has] = (attr_sums[has] / attr_cnts[has] >= 4).astype(np.float32)
    profiles[uid] = vec
print(f"  {time.time() - t0:.0f}s, {len(profiles)} profiles")


class RevealSubsetDataset(torch.utils.data.Dataset):
    """Reveals a random subset of k rated items; target = full rated profile."""

    def __init__(self, user_ids):
        self.user_ids = [u for u in user_ids if u in profiles]

    def __len__(self):
        return len(self.user_ids)

    def __getitem__(self, i):
        full = profiles[self.user_ids[i]]
        rated = np.where(~np.isnan(full))[0]

        if np.random.random() < ATTR_ONLY_PROB:
            cand = np.intersect1d(rated, attr_indices)
            if len(cand) == 0:
                cand = rated
        else:
            cand = rated
        # log-uniform reveal count in [1, len(cand)]
        k = int(np.exp(np.random.uniform(0, np.log(len(cand) + 1))))
        k = max(1, min(k, len(cand)))
        revealed = set(np.random.choice(cand, size=k, replace=False).tolist())

        order = np.random.permutation(N_ITEMS)
        rating_input = np.zeros((N_ITEMS, 3), dtype=np.float32)
        for pos, item_idx in enumerate(order):
            if item_idx in revealed:
                if full[item_idx] >= 0.5:
                    rating_input[pos, 1] = 1
                else:
                    rating_input[pos, 0] = 1
            else:
                rating_input[pos, 2] = 1

        # target at every position: full profile, columns in shuffled order
        target = np.tile(full, (N_ITEMS, 1))[:, order]
        return (
            torch.LongTensor(order),
            torch.FloatTensor(rating_input),
            torch.FloatTensor(target),
        )


class MaskedBCE(nn.Module):
    def forward(self, y_pred, y_true):
        mask = torch.isnan(y_true)
        y_pred = torch.where(mask, torch.zeros_like(y_pred), y_pred)
        y_true = torch.where(mask, torch.zeros_like(y_true), y_true)
        per_elem = nn.functional.binary_cross_entropy_with_logits(
            y_pred, y_true, reduction='none')
        per_elem = torch.where(mask, torch.zeros_like(per_elem), per_elem)
        return per_elem.sum() / (~mask).float().sum().clamp(min=1)


def main():
    model = ExtrapolationModel(N_ITEMS)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = MaskedBCE()
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    train_loader = torch.utils.data.DataLoader(
        RevealSubsetDataset(train_users), batch_size=BATCH_SIZE, shuffle=True,
        num_workers=0)
    val_loader = torch.utils.data.DataLoader(
        RevealSubsetDataset(val_users), batch_size=BATCH_SIZE, shuffle=False,
        num_workers=0)

    best_val, best_epoch = float('inf'), -1
    t0 = time.time()
    for epoch in range(N_EPOCHS):
        model.train()
        tr_loss = 0.0
        for order, rating_input, target in train_loader:
            optimizer.zero_grad()
            out = model(order, rating_input)
            loss = loss_fn(out, target)
            loss.backward()
            optimizer.step()
            tr_loss += loss.item()
        tr_loss /= max(len(train_loader), 1)

        model.eval()
        va_loss = 0.0
        with torch.no_grad():
            for order, rating_input, target in val_loader:
                va_loss += loss_fn(model(order, rating_input), target).item()
        va_loss /= max(len(val_loader), 1)

        marker = ''
        if va_loss < best_val:
            best_val, best_epoch = va_loss, epoch
            marker = ' *'
            torch.save({
                'model_state_dict': model.state_dict(),
                'n_items': N_ITEMS,
                'n_movies': N_MOVIES,
                'items': ITEMS,
                'val_loss': va_loss,
                'epochs': epoch + 1,
                'config': {
                    'augmentation': 'reveal-subset log-uniform',
                    'attr_only_prob': ATTR_ONLY_PROB,
                    'train_users': len(train_users),
                    'seed': SEED,
                },
            }, OUT_PATH)
        print(f"Epoch {epoch + 1:3d}/{N_EPOCHS} | train {tr_loss:.4f} | "
              f"val {va_loss:.4f}{marker} | {time.time() - t0:.0f}s", flush=True)

        if epoch - best_epoch >= EARLY_STOP_PATIENCE:
            print(f"Early stop (no improvement for {EARLY_STOP_PATIENCE} epochs)")
            break

    print(f"\nBest val {best_val:.4f} (epoch {best_epoch + 1}); saved {OUT_PATH}")


if __name__ == '__main__':
    main()
