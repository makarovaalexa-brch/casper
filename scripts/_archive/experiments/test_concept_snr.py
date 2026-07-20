"""
Signal-to-Noise Ratio analysis for the Concept model reward signal.

Computes BCE loss, accuracy, and NDCG at each step as preferences are
revealed. Reports per-step deltas, per-user std, and SNR for each metric.

Uses saved concept_paper_config.pt checkpoint.
Pool=100, 20 steps, 50 users (more users for better SNR estimate).

Run: python scripts/test_concept_snr.py
"""

import sys
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))
DATA_DIR = PROJECT_ROOT / 'data' / 'movielens'
CACHE_DIR = DATA_DIR / '.cache'
CHECKPOINT_DIR = CACHE_DIR / 'checkpoints'
RESULTS_DIR = PROJECT_ROOT / 'experiments' / 'diagnostics'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

N_MOVIES = 100
SEED = 42


class ConceptEmbeddingModel(nn.Module):
    def __init__(self, n_items, embedding_dim=384, hidden_dim=None):
        super().__init__()
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
        enc, _ = self.lstm(x)
        x = torch.relu(self.dense1(enc))
        x = torch.relu(self.dense2(x))
        att, _ = self.attention(enc, x, x)
        return self.output(torch.cat((x, att), dim=-1))


def load_data():
    movies_df = pd.read_csv(DATA_DIR / 'movies.csv')
    ratings_df = pd.read_csv(
        DATA_DIR / 'ratings.csv',
        dtype={'userId': 'int32', 'movieId': 'int32', 'rating': 'float32'},
        usecols=['userId', 'movieId', 'rating'],
    )
    movie_counts = ratings_df.groupby('movieId').size().reset_index(name='count')
    movies_merged = movies_df.merge(movie_counts, on='movieId', how='left').fillna(0)
    top_movies_df = movies_merged.nlargest(N_MOVIES, 'count')
    top_movies = top_movies_df['movieId'].tolist()
    movie_to_idx = {mid: i for i, mid in enumerate(top_movies)}
    top_set = set(top_movies)
    ratings_filtered = ratings_df[ratings_df['movieId'].isin(top_set)].copy()
    return top_movies, movie_to_idx, ratings_filtered


def load_model():
    ckpt = torch.load(CHECKPOINT_DIR / 'concept_paper_config.pt',
                      map_location='cpu', weights_only=False)
    hidden_dim = ckpt['model_state_dict']['concept_proj.bias'].shape[0]
    n_items = ckpt['n_items']
    model = ConceptEmbeddingModel(n_items, 384, hidden_dim)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    emb = np.load(CHECKPOINT_DIR / 'concept_embeddings_paper_config.npy')
    print(f"  Concept model: n_items={n_items}, hidden={hidden_dim}, "
          f"epochs={ckpt.get('epochs', 0)}")
    return model, n_items, emb


def build_profiles(ratings_filtered, movie_to_idx, min_targets=5, max_n=1000):
    df = ratings_filtered[ratings_filtered['movieId'].isin(movie_to_idx)].copy()
    df['liked'] = (df['rating'] >= 4.0).astype('int8')
    user_liked = df.groupby('userId')['liked'].sum()
    eligible = user_liked[user_liked >= min_targets].index.tolist()
    np.random.seed(SEED)
    sampled = np.random.choice(eligible, min(max_n, len(eligible)), replace=False)
    grouped = {uid: grp for uid, grp in df[df['userId'].isin(sampled)].groupby('userId')}
    profiles = []
    for uid in sampled:
        if uid not in grouped:
            continue
        user_df = grouped[uid]
        # Ground truth: binary (1 if rating >= 4, 0 otherwise) for each pool movie
        gt = np.full(N_MOVIES, np.nan)
        items = []
        for _, row in user_df.iterrows():
            mid = int(row['movieId'])
            if mid in movie_to_idx:
                idx = movie_to_idx[mid]
                gt[idx] = 1.0 if row['rating'] >= 4.0 else 0.0
                items.append((idx, float(row['rating']), mid))
        profiles.append({
            'user_id': int(uid),
            'items': items,
            'gt': gt,
        })
    return profiles


def calc_metrics(preds, gt):
    """Compute BCE loss, accuracy, NDCG over rated items (same as notebook)."""
    mask = ~np.isnan(gt)
    if mask.sum() == 0:
        return np.nan, np.nan, np.nan

    y_true = gt[mask]
    y_pred = np.clip(preds[mask], 1e-7, 1 - 1e-7)

    # BCE loss
    loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    # Accuracy
    acc = np.mean((y_pred > 0.5) == y_true)

    # NDCG@10 over rated items
    rated_idx = np.where(mask)[0]
    sorted_idx = np.argsort(-preds[rated_idx])[:10]
    rel = gt[rated_idx][sorted_idx]
    dcg = np.sum(rel / np.log2(np.arange(2, len(rel) + 2)))
    ideal = np.sort(gt[rated_idx])[::-1][:10]
    idcg = np.sum(ideal / np.log2(np.arange(2, len(ideal) + 2)))
    ndcg = dcg / idcg if idcg > 0 else 0

    return loss, acc, ndcg


def predict(model, emb, n_items, revealed):
    """revealed = list of (model_idx, is_liked_bool)"""
    item_emb = torch.FloatTensor(emb).unsqueeze(0)
    rating_input = torch.zeros(1, n_items, 3)
    rating_input[:, :, 2] = 1
    for idx, is_liked in revealed:
        if idx < n_items:
            rating_input[:, idx, 2] = 0
            rating_input[:, idx, 1 if is_liked else 0] = 1
    with torch.no_grad():
        return model(item_emb, rating_input)[:, -1, :N_MOVIES].sigmoid().numpy().flatten()


def main():
    print("=" * 60)
    print("Concept Model: Signal-to-Noise Ratio Analysis")
    print("=" * 60)

    print("\nLoading data...")
    top_movies, movie_to_idx, ratings_filtered = load_data()

    print("\nLoading model...")
    model, n_items, emb = load_model()

    print("\nBuilding profiles...")
    profiles = build_profiles(ratings_filtered, movie_to_idx, min_targets=5)
    np.random.seed(SEED)
    indices = np.random.permutation(len(profiles))
    profiles = [profiles[i] for i in indices]

    N_USERS = 50
    MAX_STEPS = 20
    sample = profiles[:N_USERS]
    print(f"  {len(sample)} test users, {MAX_STEPS} steps")

    # Collect per-user curves for all 3 metrics
    all_loss = []
    all_acc = []
    all_ndcg = []

    for profile in tqdm(sample, desc="Evaluating", unit="user"):
        items = list(profile['items'])
        gt = profile['gt']

        np.random.seed(profile['user_id'])
        np.random.shuffle(items)

        # Interleave 2 liked : 1 disliked
        liked = [(idx, r, mid) for idx, r, mid in items if r >= 4.0]
        disliked = [(idx, r, mid) for idx, r, mid in items if r < 2.5]
        seq = []
        li, di = 0, 0
        while li < len(liked) or di < len(disliked):
            if li < len(liked):
                seq.append((liked[li][0], True))
                li += 1
            if li < len(liked):
                seq.append((liked[li][0], True))
                li += 1
            if di < len(disliked):
                seq.append((disliked[di][0], False))
                di += 1

        loss_curve = []
        acc_curve = []
        ndcg_curve = []
        revealed = []

        for step in range(MAX_STEPS + 1):
            if step > 0 and step <= len(seq):
                revealed.append(seq[step - 1])

            preds = predict(model, emb, n_items, revealed)
            loss, acc, ndcg = calc_metrics(preds, gt)

            if not np.isnan(loss):
                loss_curve.append(loss)
                acc_curve.append(acc)
                ndcg_curve.append(ndcg)
            else:
                loss_curve.append(loss_curve[-1] if loss_curve else 0)
                acc_curve.append(acc_curve[-1] if acc_curve else 0.5)
                ndcg_curve.append(ndcg_curve[-1] if ndcg_curve else 0)

        all_loss.append(loss_curve)
        all_acc.append(acc_curve)
        all_ndcg.append(ndcg_curve)

    all_loss = np.array(all_loss)
    all_acc = np.array(all_acc)
    all_ndcg = np.array(all_ndcg)

    # ─── Results ─────────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print("STEP-BY-STEP RESULTS (mean ± std over 50 users)")
    print(f"{'='*80}")
    print(f"{'Step':<5} {'Loss':>8} {'±Std':>8} {'Acc':>8} {'±Std':>8} "
          f"{'NDCG':>8} {'±Std':>8}")
    print(f"{'-'*55}")

    for s in range(MAX_STEPS + 1):
        print(f"{s:<5} {all_loss[:,s].mean():>8.4f} {all_loss[:,s].std():>8.4f} "
              f"{all_acc[:,s].mean():>8.4f} {all_acc[:,s].std():>8.4f} "
              f"{all_ndcg[:,s].mean():>8.4f} {all_ndcg[:,s].std():>8.4f}")

    # ─── Signal-to-Noise Analysis ────────────────────────────────────
    print(f"\n{'='*80}")
    print("SIGNAL-TO-NOISE RATIO ANALYSIS")
    print(f"{'='*80}")

    for name, curves in [("BCE Loss", all_loss), ("Accuracy", all_acc), ("NDCG", all_ndcg)]:
        # Per-user deltas at each step
        deltas = np.diff(curves, axis=1)  # [n_users, max_steps]

        # For loss, improvement = negative delta (loss decreasing is good)
        if name == "BCE Loss":
            deltas = -deltas

        mean_delta = deltas.mean(axis=0)        # mean per-step improvement
        std_per_user = curves.std(axis=0)       # cross-user std at each step
        mean_delta_overall = mean_delta.mean()  # avg per-step improvement
        std_overall = std_per_user.mean()       # avg cross-user std

        # Total improvement
        total_per_user = curves[:, -1] - curves[:, 0]
        if name == "BCE Loss":
            total_per_user = -total_per_user
        total_mean = total_per_user.mean()
        total_std = total_per_user.std()

        # SNR = mean improvement / std
        snr_per_step = mean_delta_overall / std_overall if std_overall > 0 else 0
        snr_total = total_mean / total_std if total_std > 0 else 0
        se_total = total_std / np.sqrt(len(curves))

        print(f"\n  {name}:")
        print(f"    Step 0 mean:          {curves[:,0].mean():.4f} ± {curves[:,0].std():.4f}")
        print(f"    Step 20 mean:         {curves[:,-1].mean():.4f} ± {curves[:,-1].std():.4f}")
        print(f"    Total improvement:    {total_mean:+.4f} ± {total_std:.4f}")
        print(f"    SE of total:          {se_total:.4f}")
        print(f"    95% CI:               [{total_mean - 1.96*se_total:+.4f}, "
              f"{total_mean + 1.96*se_total:+.4f}]")
        print(f"    Avg per-step delta:   {mean_delta_overall:+.5f}")
        print(f"    Avg cross-user std:   {std_overall:.4f}")
        print(f"    SNR (per-step):       {snr_per_step:.4f}")
        print(f"    SNR (total):          {snr_total:.4f}")
        print(f"    Monotonic steps:      "
              f"{sum(1 for d in mean_delta if d > 0)}/{len(mean_delta)}")

    # ─── Comparison with LLM noise ───────────────────────────────────
    print(f"\n{'='*80}")
    print("COMPARISON: CONCEPT MODEL vs LLM (from prior diagnostic)")
    print(f"{'='*80}")
    print(f"{'Metric':<20} {'Concept':>12} {'LLM (Mixed)':>12} {'LLM (Attr)':>12}")
    print(f"{'-'*58}")

    # Concept model stats
    loss_total = -(all_loss[:, -1] - all_loss[:, 0]).mean()
    loss_snr = loss_total / (all_loss[:, -1] - all_loss[:, 0]).std()
    acc_total = (all_acc[:, -1] - all_acc[:, 0]).mean()
    ndcg_total = (all_ndcg[:, -1] - all_ndcg[:, 0]).mean()
    ndcg_std = (all_ndcg[:, -1] - all_ndcg[:, 0]).std()

    print(f"{'NDCG improve':<20} {ndcg_total:>+12.4f} {'~+0.102':>12} {'~+0.025':>12}")
    print(f"{'NDCG per-user std':<20} {ndcg_std:>12.4f} {'~0.20':>12} {'~0.20':>12}")
    print(f"{'Loss improve':<20} {loss_total:>+12.4f} {'N/A':>12} {'N/A':>12}")
    print(f"{'Acc improve':<20} {acc_total:>+12.4f} {'N/A':>12} {'N/A':>12}")
    print(f"{'Loss SNR (total)':<20} {abs(loss_snr):>12.4f} {'N/A':>12} {'N/A':>12}")

    print(f"\nKey insight: Loss is dense (over all rated items) vs NDCG (top-10 only).")
    print(f"If loss SNR >> NDCG SNR, loss is a better RL reward signal.")

    # Save
    results = {
        'n_users': N_USERS,
        'max_steps': MAX_STEPS,
        'loss': {'mean': all_loss.mean(axis=0).tolist(),
                 'std': all_loss.std(axis=0).tolist()},
        'accuracy': {'mean': all_acc.mean(axis=0).tolist(),
                     'std': all_acc.std(axis=0).tolist()},
        'ndcg': {'mean': all_ndcg.mean(axis=0).tolist(),
                 'std': all_ndcg.std(axis=0).tolist()},
    }
    out_file = RESULTS_DIR / 'concept_model_snr_analysis.json'
    with open(out_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to: {out_file}")


if __name__ == '__main__':
    main()
