"""
Paper 1 / M1: Acceptance tests for the measurement instrument.

Tests whether a candidate recommender model can serve as the fixed
measurement instrument of the CASPER elicitation testbed. A model passes if:

  T1 (polarity):     top-10 movie recommendations for "user likes E1..Ek"
                     vs "user dislikes E1..Ek" overlap < 30%
  T2 (monotonicity): held-out rating-prediction accuracy increases with the
                     number of revealed preferences (Spearman rho > 0.9 over
                     timesteps, and final > first)
  T3 (rating flip):  flipping the rating of one franchise anchor (Star Wars
                     Ep IV) moves related movies in opposite directions
  T4 (SNR):          per-reveal accuracy gain / between-user std at matched
                     timestep -- reported for the paper (no hard threshold,
                     but >0.1 needed for the testbed to discriminate policies)

Run from casper root:  poetry run python scripts/paper1/test_instrument.py
Results: experiments/paper1/instrument_acceptance.json
"""

import sys
sys.path.insert(0, '.')

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import sys as _sys
_sys.path.insert(0, 'scripts/paper1')
from test_instrument_lib import load_instruments

DATA_DIR = Path('C:/dev/phd/casper/data/movielens')
CHECKPOINT_DIR = DATA_DIR / '.cache' / 'checkpoints'
OUT_DIR = Path('C:/dev/phd/casper/experiments/paper1')
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_MOVIES = 100
SEED = 42
TOP_K = 10
N_POLARITY_TRIALS = 200
TIMESTEPS = [1, 3, 5, 10, 20, 30, 50]
N_EVAL_USERS = 200

rng = np.random.default_rng(SEED)


# ---------------------------------------------------------------------------
# Model definitions (identical to scripts/unified_evaluation.py)
# ---------------------------------------------------------------------------

class ExtrapolationModel(nn.Module):
    def __init__(self, n_items):
        super().__init__()
        self.n_items = n_items
        hidden_dim = n_items // 2
        self.embedding = nn.Embedding(n_items + 1, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim + 3, hidden_dim, 1, batch_first=True)
        self.dense1 = nn.Linear(hidden_dim, n_items)
        self.dense2 = nn.Linear(n_items, hidden_dim)
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=1, batch_first=True)
        self.output = nn.Linear(hidden_dim * 2, n_items)

    def forward(self, index_input, rating_input):
        x = self.embedding(index_input)
        x = torch.cat((x, rating_input), dim=-1)
        encoder_output, _ = self.lstm(x)
        x = torch.relu(self.dense1(encoder_output))
        x = torch.relu(self.dense2(x))
        attention, _ = self.attention(encoder_output, x, x)
        x = torch.cat((x, attention), dim=-1)
        return self.output(x)


class ConceptEmbeddingModel(nn.Module):
    def __init__(self, n_items, embedding_dim=384, hidden_dim=None):
        super().__init__()
        self.n_items = n_items
        hidden_dim = hidden_dim or n_items // 2
        self.concept_proj = nn.Linear(embedding_dim, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim + 3, hidden_dim, 1, batch_first=True)
        self.dense1 = nn.Linear(hidden_dim, n_items)
        self.dense2 = nn.Linear(n_items, hidden_dim)
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=1, batch_first=True)
        self.output = nn.Linear(hidden_dim * 2, n_items)

    def forward(self, item_embeddings, rating_input):
        x = self.concept_proj(item_embeddings)
        x = torch.cat((x, rating_input), dim=-1)
        encoder_output, _ = self.lstm(x)
        x = torch.relu(self.dense1(encoder_output))
        x = torch.relu(self.dense2(x))
        attention, _ = self.attention(encoder_output, x, x)
        x = torch.cat((x, attention), dim=-1)
        return self.output(x)


# ---------------------------------------------------------------------------
# Wrappers exposing a single predict(revealed) -> movie scores interface
# ---------------------------------------------------------------------------

class InstrumentWrapper:
    """revealed: list of (item_idx, polarity) with polarity in {1.0 liked, 0.0 disliked}."""

    def __init__(self, model, n_items, n_movies, item_embeddings=None):
        self.model = model
        self.n_items = n_items
        self.n_movies = n_movies
        self.item_embeddings = item_embeddings
        if item_embeddings is not None:
            self._emb = torch.FloatTensor(item_embeddings).unsqueeze(0)
        else:
            self._idx = torch.arange(n_items).unsqueeze(0)

    def predict(self, revealed):
        rating_input = torch.zeros(1, self.n_items, 3)
        rating_input[:, :, 2] = 1
        for idx, pol in revealed:
            rating_input[:, idx, 2] = 0
            if pol >= 0.5:
                rating_input[:, idx, 1] = 1
            else:
                rating_input[:, idx, 0] = 1
        with torch.no_grad():
            if self.item_embeddings is not None:
                out = self.model(self._emb, rating_input)
            else:
                out = self.model(self._idx, rating_input)
        return out[:, -1, :self.n_movies].sigmoid().numpy().flatten()


def _local_load_instruments_unused():
    instruments = {}

    ckpt_path = CHECKPOINT_DIR / 'onehot_paper_config.pt'
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, weights_only=False)
        m = ExtrapolationModel(ckpt['n_items'])
        m.load_state_dict(ckpt['model_state_dict'])
        m.eval()
        instruments['onehot'] = (
            InstrumentWrapper(m, ckpt['n_items'], ckpt.get('n_movies', N_MOVIES)),
            ckpt,
        )

    ckpt_path = CHECKPOINT_DIR / 'concept_paper_config.pt'
    emb_path = CHECKPOINT_DIR / 'concept_embeddings_paper_config.npy'
    if ckpt_path.exists() and emb_path.exists():
        ckpt = torch.load(ckpt_path, weights_only=False)
        emb = np.load(emb_path)
        hidden_dim = ckpt['model_state_dict']['concept_proj.bias'].shape[0]
        m = ConceptEmbeddingModel(ckpt['n_items'], ckpt.get('embedding_dim', 384), hidden_dim)
        m.load_state_dict(ckpt['model_state_dict'])
        m.eval()
        instruments['concept'] = (
            InstrumentWrapper(m, ckpt['n_items'], ckpt.get('n_movies', N_MOVIES), emb),
            ckpt,
        )

    return instruments


# ---------------------------------------------------------------------------
# Shared data (identical filtering to unified_evaluation.py)
# ---------------------------------------------------------------------------

print("Loading MovieLens ratings (large file, ~1 min)...")
t0 = time.time()
ratings = pd.read_csv(DATA_DIR / 'ratings.csv')
movies = pd.read_csv(DATA_DIR / 'movies.csv')
print(f"  loaded in {time.time() - t0:.0f}s")

movie_counts = ratings.groupby('movieId').size().reset_index(name='count')
movies_with_counts = movies.merge(movie_counts, on='movieId', how='left')
movies_with_counts['count'] = movies_with_counts['count'].fillna(0)
top_movies_df = movies_with_counts.nlargest(N_MOVIES, 'count')
top_movies = top_movies_df['movieId'].tolist()
movie_to_idx = {mid: i for i, mid in enumerate(top_movies)}
idx_to_title = {i: movies_with_counts.set_index('movieId').loc[mid, 'title']
                for i, mid in enumerate(top_movies)}

ratings_filtered = ratings[ratings['movieId'].isin(top_movies)].copy()
user_rating_counts = ratings_filtered.groupby('userId').size()
active_users = user_rating_counts[user_rating_counts >= 25].index.tolist()

np.random.seed(SEED)
shuffled = active_users.copy()
np.random.shuffle(shuffled)
val_users = shuffled[int(0.8 * len(shuffled)):][:N_EVAL_USERS]
print(f"Eval users: {len(val_users)}")

user_ratings_map = {
    uid: ratings_filtered[ratings_filtered['userId'] == uid]
    for uid in val_users
}


# ---------------------------------------------------------------------------
# T1: polarity overlap
# ---------------------------------------------------------------------------

def test_polarity_overlap(wrapper, ckpt):
    """Same entity set, all-liked vs all-disliked: top-K movie overlap."""
    n_items = wrapper.n_items
    # askable entities = attribute positions (beyond the movie slate), if any;
    # otherwise sample movies themselves
    attr_indices = list(range(wrapper.n_movies, n_items))
    pool = attr_indices if len(attr_indices) >= 10 else list(range(n_items))

    overlaps = []
    for _ in range(N_POLARITY_TRIALS):
        k = int(rng.integers(3, 9))
        entities = rng.choice(pool, size=min(k, len(pool)), replace=False)
        liked = wrapper.predict([(int(e), 1.0) for e in entities])
        disliked = wrapper.predict([(int(e), 0.0) for e in entities])
        top_l = set(np.argsort(-liked)[:TOP_K])
        top_d = set(np.argsort(-disliked)[:TOP_K])
        overlaps.append(len(top_l & top_d) / TOP_K)

    mean_overlap = float(np.mean(overlaps))
    return {
        'mean_overlap': mean_overlap,
        'std': float(np.std(overlaps)),
        'pass': mean_overlap < 0.30,
    }


# ---------------------------------------------------------------------------
# T2 + T4: monotonicity and SNR
# ---------------------------------------------------------------------------

def test_monotonicity_snr(wrapper, ckpt):
    acc_by_t = {t: [] for t in TIMESTEPS}
    per_user_curves = {}

    for uid in val_users:
        ur = user_ratings_map[uid]
        item_list = [(movie_to_idx[r.movieId], 1.0 if r.rating >= 4 else 0.0)
                     for r in ur.itertuples() if r.movieId in movie_to_idx]
        if len(item_list) < max(TIMESTEPS):
            continue
        gt = np.full(wrapper.n_movies, np.nan)
        for idx, pol in item_list:
            gt[idx] = pol

        np.random.seed(uid)
        np.random.shuffle(item_list)
        curve = {}
        for t in TIMESTEPS:
            preds = wrapper.predict(item_list[:t])
            filt = ~np.isnan(gt)
            acc = float(np.mean((preds[filt] > 0.5) == gt[filt]))
            acc_by_t[t].append(acc)
            curve[t] = acc
        per_user_curves[uid] = curve

    means = {t: float(np.mean(v)) for t, v in acc_by_t.items() if v}
    stds = {t: float(np.std(v)) for t, v in acc_by_t.items() if v}
    ts = sorted(means.keys())

    # Spearman rho between timestep order and mean accuracy order
    from scipy.stats import spearmanr
    rho, _ = spearmanr(ts, [means[t] for t in ts])

    # SNR: per-reveal gain over the 1->20 window vs between-user std at t=10
    gain = (means[20] - means[1]) / 19 if 20 in means and 1 in means else np.nan
    snr = float(gain * 19 / stds[10]) if 10 in stds and stds[10] > 0 else np.nan

    return {
        'accuracy_by_timestep': means,
        'std_by_timestep': stds,
        'spearman_rho': float(rho),
        'total_gain_1_to_20': float(means.get(20, np.nan) - means.get(1, np.nan)),
        'signal_to_noise_1_to_20': snr,
        'n_users': len(per_user_curves),
        'pass': bool(rho > 0.9 and means[ts[-1]] > means[ts[0]]),
    }


# ---------------------------------------------------------------------------
# T3: rating flip on a franchise anchor
# ---------------------------------------------------------------------------

def test_rating_flip(wrapper, ckpt):
    anchor_idx, related = None, []
    for i, title in idx_to_title.items():
        t = str(title)
        if 'Star Wars: Episode IV' in t:
            anchor_idx = i
        elif 'Star Wars' in t:
            related.append(i)
    if anchor_idx is None or not related:
        return {'pass': None, 'note': 'Star Wars movies not in slate'}

    liked = wrapper.predict([(anchor_idx, 1.0)])
    disliked = wrapper.predict([(anchor_idx, 0.0)])

    def ranks(scores):
        order = np.argsort(-scores)
        return {int(m): int(np.where(order == m)[0][0]) for m in related}

    r_l, r_d = ranks(liked), ranks(disliked)
    moves = {idx_to_title[m]: r_d[m] - r_l[m] for m in related}
    n_correct = sum(1 for v in moves.values() if v > 0)
    return {
        'rank_drop_when_disliked': moves,
        'fraction_correct_direction': n_correct / len(moves),
        'pass': n_correct / len(moves) >= 0.75,
    }


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

def main():
    instruments = load_instruments()
    if not instruments:
        print("NO CHECKPOINTS FOUND -- retraining required")
        return

    results = {}
    for name, (wrapper, ckpt) in instruments.items():
        print(f"\n{'=' * 60}\nINSTRUMENT: {name} "
              f"(n_items={wrapper.n_items}, n_movies={wrapper.n_movies})\n{'=' * 60}")

        r = {}
        print("T1 polarity overlap...")
        r['T1_polarity'] = test_polarity_overlap(wrapper, ckpt)
        print(f"  overlap={r['T1_polarity']['mean_overlap']:.2%} "
              f"pass={r['T1_polarity']['pass']}")

        print("T2/T4 monotonicity + SNR...")
        r['T2_T4_monotonicity_snr'] = test_monotonicity_snr(wrapper, ckpt)
        m = r['T2_T4_monotonicity_snr']
        print(f"  acc: {m['accuracy_by_timestep']}")
        print(f"  rho={m['spearman_rho']:.3f} gain={m['total_gain_1_to_20']:+.4f} "
              f"SNR={m['signal_to_noise_1_to_20']:.2f} pass={m['pass']}")

        print("T3 rating flip...")
        r['T3_rating_flip'] = test_rating_flip(wrapper, ckpt)
        print(f"  {r['T3_rating_flip']}")

        passes = [v.get('pass') for v in r.values() if v.get('pass') is not None]
        r['ACCEPTED'] = all(passes)
        print(f"\n  ==> {'ACCEPTED' if r['ACCEPTED'] else 'REJECTED'} as instrument")
        results[name] = r

    out = OUT_DIR / 'instrument_acceptance.json'
    with open(out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out}")


if __name__ == '__main__':
    main()
