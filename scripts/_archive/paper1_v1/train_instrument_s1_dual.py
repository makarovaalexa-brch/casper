"""
Dual-head instrument for SLATE 1 (decision-aid for dual-belief policies).

Note on role: slate-1 MEASUREMENT remains the accepted single-head v5
instrument; this dual-head model supplies the policy-side answerability
beliefs only, so leaderboard accuracy stays comparable across rows.

Run from casper root: poetry run python scripts/paper1/train_instrument_s1_dual.py
Output: data/movielens/.cache/checkpoints/instrument_s1_dual.pt
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
from test_instrument_lib import CHECKPOINT_DIR
from testbed import build_profiles, get_user_splits

OUT_PATH = CHECKPOINT_DIR / 'instrument_s1_dual.pt'

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

ref = torch.load(CHECKPOINT_DIR / 'instrument_v5_set.pt', weights_only=False)
ITEMS = ref['items']
N_ITEMS = len(ITEMS)
N_MOVIES = ref.get('n_movies', 100)
attr_indices = np.array([i for i, it in enumerate(ITEMS) if it[0] != 'movie'])


def main():
    train_users, ival, _ = get_user_splits(ITEMS)
    train_users = train_users[:MAX_TRAIN_USERS]
    val_users = ival[:MAX_VAL_USERS]
    print("Building profiles...")
    profiles = build_profiles(ITEMS, user_ids=np.concatenate([train_users, val_users]),
                              attr_min_support=3, taste_margin=None)
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

    def loss_fn(lk, rt, tgt):
        is_rated = ~torch.isnan(tgt)
        tgt0 = torch.where(is_rated, tgt, torch.zeros_like(tgt))
        per = nn.functional.binary_cross_entropy_with_logits(lk, tgt0,
                                                             reduction='none')
        liked = (per * is_rated.float()).sum() / is_rated.float().sum().clamp(min=1)
        rated = nn.functional.binary_cross_entropy_with_logits(
            rt, is_rated.float())
        return liked + rated

    model = DualHeadSetEncoder(N_ITEMS, d_model=D_MODEL)
    opt = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    tl = torch.utils.data.DataLoader(DS(train_users), batch_size=BATCH_SIZE,
                                     shuffle=True)
    vl = torch.utils.data.DataLoader(DS(val_users), batch_size=BATCH_SIZE)

    best, best_epoch = float('inf'), -1
    t0 = time.time()
    for epoch in range(N_EPOCHS):
        model.train()
        for idx, pol, pad, tgt in tl:
            opt.zero_grad()
            lk, rt = model(idx, pol, pad)
            loss_fn(lk, rt, tgt).backward()
            opt.step()
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
            torch.save({'model_state_dict': model.state_dict(),
                        'arch': 'dual_set_encoder', 'd_model': D_MODEL,
                        'n_heads': 4, 'n_layers': 2,
                        'n_items': N_ITEMS, 'n_movies': N_MOVIES,
                        'items': ITEMS, 'val_loss': va, 'epochs': epoch + 1,
                        'config': {'slate': 'slate1',
                                   'labels': 'absolute, attr support >= 3',
                                   'heads': 'liked + rated', 'seed': SEED}},
                       OUT_PATH)
        print(f"Epoch {epoch + 1:3d}/{N_EPOCHS} | val {va:.4f}{marker} | "
              f"{time.time() - t0:.0f}s", flush=True)
        if epoch - best_epoch >= EARLY_STOP_PATIENCE:
            print("Early stop")
            break
    print(f"Best val {best:.4f}; saved {OUT_PATH}")


if __name__ == '__main__':
    main()
