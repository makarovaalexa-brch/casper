"""
Paper 1 / M1: Instrument v4 — warm-start fine-tune.

Generations so far:
  v1 (onehot_paper_config): genuine attribute structure (genre DiD 100%,
      franchise 91.7%) but full-profile training -> dialogue regime OOD.
  v2 (+reveal-subset, scratch): regime fixed (L/D overlap 0.7%) but the
      single-attribute response collapsed onto rater harshness (genre DiD
      37.5%).
  v3 (+taste-relative labels, scratch): sparser labels weakened the entire
      reveal response (liked ~ prior, franchise 33%).

v4 hypothesis: v1's learned attribute structure is worth keeping; what it
lacks is exposure to partial reveals. Warm-start from v1's weights and
fine-tune briefly with reveal-subset augmentation at low LR, using
absolute attribute labels with a support threshold (>=3 matching movies)
-- the label scheme v1 was trained under, minus its noisiest labels.

Run from casper root:
    poetry run python scripts/paper1/train_instrument_v4.py
Output: data/movielens/.cache/checkpoints/instrument_v4_onehot.pt
"""

import sys
sys.path.insert(0, '.')
sys.path.insert(0, 'scripts/paper1')

import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from test_instrument_lib import CHECKPOINT_DIR, ExtrapolationModel
from testbed import build_profiles

OUT_PATH = CHECKPOINT_DIR / 'instrument_v4_onehot.pt'

SEED = 42
N_EPOCHS = 15
BATCH_SIZE = 32
LEARNING_RATE = 3e-4   # low LR: adapt, don't forget
MAX_TRAIN_USERS = 12000
MAX_VAL_USERS = 1500
ATTR_ONLY_PROB = 0.15
EARLY_STOP_PATIENCE = 5

np.random.seed(SEED)
torch.manual_seed(SEED)

ref_ckpt = torch.load(CHECKPOINT_DIR / 'onehot_paper_config.pt', weights_only=False)
ITEMS = ref_ckpt['items']
N_ITEMS = len(ITEMS)
N_MOVIES = ref_ckpt.get('n_movies', 100)
attr_indices = [i for i, it in enumerate(ITEMS) if it[0] != 'movie']

# same split protocol as v2/v3
import pandas as pd
DATA_DIR = Path('C:/dev/phd/casper/data/movielens')
ratings = pd.read_csv(DATA_DIR / 'ratings.csv')
movie_ids = {it[1] for it in ITEMS if it[0] == 'movie'}
ratings_f = ratings[ratings['movieId'].isin(movie_ids)]
user_counts = ratings_f.groupby('userId').size()
dense_users = user_counts[user_counts >= 25].index.to_numpy()
np.random.seed(SEED)
shuffled = dense_users.copy()
np.random.shuffle(shuffled)
split = int(0.8 * len(shuffled))
train_users = shuffled[:split][:MAX_TRAIN_USERS]
val_users = shuffled[split:][:MAX_VAL_USERS]
del ratings, ratings_f

print("Building profiles (absolute labels, support >= 3)...")
t0 = time.time()
profiles = build_profiles(ITEMS, user_ids=np.concatenate([train_users, val_users]),
                          attr_min_support=3, taste_margin=None)
print(f"  {time.time() - t0:.0f}s, {len(profiles)} profiles")


class RevealSubsetDataset(torch.utils.data.Dataset):
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
        k = int(np.exp(np.random.uniform(0, np.log(len(cand) + 1))))
        k = max(1, min(k, len(cand)))
        revealed = set(np.random.choice(cand, size=k, replace=False).tolist())

        order = np.random.permutation(N_ITEMS)
        rating_input = np.zeros((N_ITEMS, 3), dtype=np.float32)
        for pos, item_idx in enumerate(order):
            if item_idx in revealed:
                rating_input[pos, 1 if full[item_idx] >= 0.5 else 0] = 1
            else:
                rating_input[pos, 2] = 1
        target = np.tile(full, (N_ITEMS, 1))[:, order]
        return (torch.LongTensor(order), torch.FloatTensor(rating_input),
                torch.FloatTensor(target))


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
    model.load_state_dict(ref_ckpt['model_state_dict'])  # warm start from v1
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = MaskedBCE()

    train_loader = torch.utils.data.DataLoader(
        RevealSubsetDataset(train_users), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = torch.utils.data.DataLoader(
        RevealSubsetDataset(val_users), batch_size=BATCH_SIZE, shuffle=False)

    best_val, best_epoch = float('inf'), -1
    t0 = time.time()
    for epoch in range(N_EPOCHS):
        model.train()
        tr = 0.0
        for order, rating_input, target in train_loader:
            optimizer.zero_grad()
            loss = loss_fn(model(order, rating_input), target)
            loss.backward()
            optimizer.step()
            tr += loss.item()
        tr /= max(len(train_loader), 1)

        model.eval()
        va = 0.0
        with torch.no_grad():
            for order, rating_input, target in val_loader:
                va += loss_fn(model(order, rating_input), target).item()
        va /= max(len(val_loader), 1)

        marker = ''
        if va < best_val:
            best_val, best_epoch = va, epoch
            marker = ' *'
            torch.save({
                'model_state_dict': model.state_dict(),
                'n_items': N_ITEMS, 'n_movies': N_MOVIES, 'items': ITEMS,
                'val_loss': va, 'epochs': epoch + 1,
                'config': {'init': 'onehot_paper_config (warm start)',
                           'augmentation': 'reveal-subset log-uniform',
                           'labels': 'absolute, attr support >= 3',
                           'lr': LEARNING_RATE, 'seed': SEED},
            }, OUT_PATH)
        print(f"Epoch {epoch + 1:2d}/{N_EPOCHS} | train {tr:.4f} | "
              f"val {va:.4f}{marker} | {time.time() - t0:.0f}s", flush=True)
        if epoch - best_epoch >= EARLY_STOP_PATIENCE:
            print("Early stop")
            break

    print(f"\nBest val {best_val:.4f} (epoch {best_epoch + 1}); saved {OUT_PATH}")


if __name__ == '__main__':
    main()
