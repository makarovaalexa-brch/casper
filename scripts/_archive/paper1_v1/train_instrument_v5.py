"""
Paper 1 / M1: Instrument v5 — set-encoder architecture (root-cause fix).

Diagnosis across v1-v4: the ExtrapolationModel processes the slate as a
187-step LSTM SEQUENCE in which revealed items are rare tokens among ~184
'unseen' placeholders. With 1-3 reveals, the signal must survive a long
recurrence through a 93-dim hidden state dominated by unseen tokens --
which is why every generation responds weakly or pathologically to single
reveals, whatever the training distribution or labels.

v5 encodes ONLY the revealed (entity, polarity) pairs:
    token_i = item_embedding(e_i) + polarity_embedding(a_i)
    [CLS] + tokens -> 2-layer transformer encoder -> CLS state
    CLS state -> MLP -> logits for all 187 items (loss masked to rated)
Permutation-invariant by construction, no dilution, single reveals are
first-class citizens, and the empty set yields a learned prior. This is
architecturally the user-tower of Makarova et al. (IJCNN 2024), applied
as the testbed instrument.

Labels: absolute with attribute support >= 3 (the v4 scheme).
Run from casper root:
    poetry run python scripts/paper1/train_instrument_v5.py
Output: data/movielens/.cache/checkpoints/instrument_v5_set.pt
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

from test_instrument_lib import CHECKPOINT_DIR
from testbed import build_profiles

OUT_PATH = CHECKPOINT_DIR / 'instrument_v5_set.pt'

SEED = 42
N_EPOCHS = 80
BATCH_SIZE = 64
LEARNING_RATE = 5e-4
D_MODEL = 128
N_HEADS = 4
N_LAYERS = 2
MAX_TRAIN_USERS = 12000
MAX_VAL_USERS = 1500
ATTR_ONLY_PROB = 0.15
EMPTY_PROB = 0.05          # teach the prior explicitly
EARLY_STOP_PATIENCE = 10
MAX_REVEAL = 60            # cap sequence length for batching efficiency

np.random.seed(SEED)
torch.manual_seed(SEED)

ref_ckpt = torch.load(CHECKPOINT_DIR / 'onehot_paper_config.pt', weights_only=False)
ITEMS = ref_ckpt['items']
N_ITEMS = len(ITEMS)
N_MOVIES = ref_ckpt.get('n_movies', 100)
attr_indices = np.array([i for i, it in enumerate(ITEMS) if it[0] != 'movie'])

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


class SetEncoderInstrument(nn.Module):
    """Attention over revealed (entity, polarity) tokens; CLS -> all-item logits."""

    def __init__(self, n_items, d_model=D_MODEL, n_heads=N_HEADS,
                 n_layers=N_LAYERS):
        super().__init__()
        self.n_items = n_items
        self.item_emb = nn.Embedding(n_items, d_model)
        self.pol_emb = nn.Embedding(2, d_model)   # 0 disliked, 1 liked
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=4 * d_model,
            dropout=0.1, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.head = nn.Sequential(
            nn.Linear(d_model, 2 * d_model), nn.ReLU(),
            nn.Linear(2 * d_model, n_items))
        nn.init.normal_(self.cls, std=0.02)

    def forward(self, idx, pol, pad_mask):
        """idx, pol: [B, L] long; pad_mask: [B, L] True where PADDING."""
        b = idx.shape[0]
        tok = self.item_emb(idx) + self.pol_emb(pol)
        x = torch.cat([self.cls.expand(b, -1, -1), tok], dim=1)
        full_mask = torch.cat(
            [torch.zeros(b, 1, dtype=torch.bool, device=idx.device), pad_mask],
            dim=1)
        out = self.encoder(x, src_key_padding_mask=full_mask)
        return self.head(out[:, 0])


class RevealSetDataset(torch.utils.data.Dataset):
    def __init__(self, user_ids):
        self.user_ids = [u for u in user_ids if u in profiles]

    def __len__(self):
        return len(self.user_ids)

    def __getitem__(self, i):
        full = profiles[self.user_ids[i]]
        rated = np.where(~np.isnan(full))[0]

        if np.random.random() < EMPTY_PROB:
            revealed = np.array([], dtype=int)
        else:
            if np.random.random() < ATTR_ONLY_PROB:
                cand = np.intersect1d(rated, attr_indices)
                if len(cand) == 0:
                    cand = rated
            else:
                cand = rated
            k = int(np.exp(np.random.uniform(0, np.log(min(len(cand), MAX_REVEAL) + 1))))
            k = max(1, min(k, len(cand)))
            revealed = np.random.choice(cand, size=k, replace=False)

        idx = np.zeros(MAX_REVEAL, dtype=np.int64)
        pol = np.zeros(MAX_REVEAL, dtype=np.int64)
        pad = np.ones(MAX_REVEAL, dtype=bool)
        for j, e in enumerate(revealed):
            idx[j] = e
            pol[j] = 1 if full[e] >= 0.5 else 0
            pad[j] = False
        return (torch.from_numpy(idx), torch.from_numpy(pol),
                torch.from_numpy(pad), torch.FloatTensor(full))


class MaskedBCE(nn.Module):
    def forward(self, y_pred, y_true):
        mask = torch.isnan(y_true)
        y_pred = torch.where(mask, torch.zeros_like(y_pred), y_pred)
        y_true = torch.where(mask, torch.zeros_like(y_true), y_true)
        per = nn.functional.binary_cross_entropy_with_logits(
            y_pred, y_true, reduction='none')
        per = torch.where(mask, torch.zeros_like(per), per)
        return per.sum() / (~mask).float().sum().clamp(min=1)


def main():
    model = SetEncoderInstrument(N_ITEMS)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = MaskedBCE()

    train_loader = torch.utils.data.DataLoader(
        RevealSetDataset(train_users), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = torch.utils.data.DataLoader(
        RevealSetDataset(val_users), batch_size=BATCH_SIZE, shuffle=False)

    best_val, best_epoch = float('inf'), -1
    t0 = time.time()
    for epoch in range(N_EPOCHS):
        model.train()
        tr = 0.0
        for idx, pol, pad, target in train_loader:
            optimizer.zero_grad()
            loss = loss_fn(model(idx, pol, pad), target)
            loss.backward()
            optimizer.step()
            tr += loss.item()
        tr /= max(len(train_loader), 1)

        model.eval()
        va = 0.0
        with torch.no_grad():
            for idx, pol, pad, target in val_loader:
                va += loss_fn(model(idx, pol, pad), target).item()
        va /= max(len(val_loader), 1)

        marker = ''
        if va < best_val:
            best_val, best_epoch = va, epoch
            marker = ' *'
            torch.save({
                'model_state_dict': model.state_dict(),
                'arch': 'set_encoder',
                'd_model': D_MODEL, 'n_heads': N_HEADS, 'n_layers': N_LAYERS,
                'n_items': N_ITEMS, 'n_movies': N_MOVIES, 'items': ITEMS,
                'val_loss': va, 'epochs': epoch + 1,
                'config': {'labels': 'absolute, attr support >= 3',
                           'max_reveal': MAX_REVEAL, 'seed': SEED},
            }, OUT_PATH)
        print(f"Epoch {epoch + 1:3d}/{N_EPOCHS} | train {tr:.4f} | "
              f"val {va:.4f}{marker} | {time.time() - t0:.0f}s", flush=True)
        if epoch - best_epoch >= EARLY_STOP_PATIENCE:
            print("Early stop")
            break

    print(f"\nBest val {best_val:.4f} (epoch {best_epoch + 1}); saved {OUT_PATH}")


if __name__ == '__main__':
    main()
