"""
Paper 1 / #14 + #24: Slate-2 instrument, dual-head version.

The single-head slate-2 instrument (instrument_slate2_set.pt) trained
before the answerability discovery; benchmarking on it would repeat the
representational failure proven on the synthetic indicator world. This
retrains with the dual head -- P(rated) and P(liked|rated) -- whose
rated-ness targets are free (the nan mask).

Run from casper root:
    poetry run python scripts/paper1/train_instrument_slate2_dual.py
Output: data/movielens/.cache/checkpoints/instrument_slate2_dual.pt
"""

import sys
sys.path.insert(0, '.')
sys.path.insert(0, 'scripts/paper1')

import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from synthetic_sanity import DualHeadSetEncoder
from train_instrument_slate2 import (build_slate, build_profiles_slate2,
                                     CHECKPOINT_DIR, MIN_USER_RATINGS)

OUT_PATH = CHECKPOINT_DIR / 'instrument_slate2_dual.pt'

SEED = 42
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


def main():
    print("Building slate 2...")
    items, n_movies, movie_pos, attr_of_movie, ratings_f = build_slate()
    n_items = len(items)
    attr_indices = np.array([i for i, it in enumerate(items) if it[0] != 'movie'])
    print(f"  {n_items} items ({n_movies} movies)")

    user_counts = ratings_f.groupby('userId').size()
    dense = user_counts[user_counts >= MIN_USER_RATINGS].index.to_numpy()
    np.random.seed(SEED)
    shuffled = dense.copy()
    np.random.shuffle(shuffled)
    split = int(0.8 * len(shuffled))
    train_users = shuffled[:split][:MAX_TRAIN_USERS]
    val_users = shuffled[split:][:MAX_VAL_USERS]

    print("Building profiles...")
    profiles = build_profiles_slate2(items, n_items, movie_pos, attr_of_movie,
                                     ratings_f,
                                     np.concatenate([train_users, val_users]))
    print(f"  {len(profiles)} profiles")

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

    def loss_fn(liked_logits, rated_logits, tgt):
        is_rated = ~torch.isnan(tgt)
        tgt0 = torch.where(is_rated, tgt, torch.zeros_like(tgt))
        per = nn.functional.binary_cross_entropy_with_logits(
            liked_logits, tgt0, reduction='none')
        liked_loss = (per * is_rated.float()).sum() / \
            is_rated.float().sum().clamp(min=1)
        rated_loss = nn.functional.binary_cross_entropy_with_logits(
            rated_logits, is_rated.float())
        return liked_loss + rated_loss

    model = DualHeadSetEncoder(n_items, d_model=D_MODEL)
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
            lk, rt = model(idx, pol, pad)
            loss = loss_fn(lk, rt, tgt)
            loss.backward()
            opt.step()
            tr += loss.item()
        tr /= max(len(tl), 1)
        model.eval()
        va = 0.0
        with torch.no_grad():
            for idx, pol, pad, tgt in vl:
                lk, rt = model(idx, pol, pad)
                va += loss_fn(lk, rt, tgt).item()
        va /= max(len(vl), 1)
        marker = ''
        if va < best:
            best, best_epoch = va, epoch
            marker = ' *'
            torch.save({
                'model_state_dict': model.state_dict(),
                'arch': 'dual_set_encoder', 'd_model': D_MODEL,
                'n_heads': 4, 'n_layers': 2,
                'n_items': n_items, 'n_movies': n_movies, 'items': items,
                'val_loss': va, 'epochs': epoch + 1,
                'config': {'slate': 'ranks 101-400 + genres + genome tags',
                           'labels': 'absolute, attr support >= 3',
                           'heads': 'liked + rated', 'seed': SEED},
            }, OUT_PATH)
        print(f"Epoch {epoch + 1:3d}/{N_EPOCHS} | train {tr:.4f} | "
              f"val {va:.4f}{marker} | {time.time() - t0:.0f}s", flush=True)
        if epoch - best_epoch >= EARLY_STOP_PATIENCE:
            print("Early stop")
            break

    print(f"\nBest val {best:.4f} (epoch {best_epoch + 1}); saved {OUT_PATH}")


if __name__ == '__main__':
    main()
