"""
Test: LSTM+Attention models as reward signal (vs LLM diagnostic).

Runs the same diagnostic as test_preference_types.py but using the trained
neural models (One-Hot LSTM+Attention and Concept SBERT+LSTM) instead of LLM.

Reveals user preferences one at a time and measures NDCG@10 at each step.
Comparable setup: pool=100, 20 steps, 30 users, min_eval_targets=5.

Run: python scripts/test_lstm_reward_signal.py
"""

import sys
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
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


# ─── Model Definitions (from RS_Model_Comparison notebook) ──────────────────

class ExtrapolationModel(nn.Module):
    """One-hot LSTM+Attention model (paper baseline)."""
    def __init__(self, n_items):
        super().__init__()
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
        enc, _ = self.lstm(x)
        x = torch.relu(self.dense1(enc))
        x = torch.relu(self.dense2(x))
        att, _ = self.attention(enc, x, x)
        return self.output(torch.cat((x, att), dim=-1))


class ConceptEmbeddingModel(nn.Module):
    """SBERT-based concept embeddings model."""
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


# ─── Data loading ───────────────────────────────────────────────────────────

def load_data():
    """Load movies, ratings, build pool. Same as notebook and LLM diagnostic."""
    movies_df = pd.read_csv(DATA_DIR / 'movies.csv')
    ratings_df = pd.read_csv(
        DATA_DIR / 'ratings.csv',
        dtype={'userId': 'int32', 'movieId': 'int32', 'rating': 'float32'},
        usecols=['userId', 'movieId', 'rating'],
    )

    # Top 100 movies by rating count
    movie_counts = ratings_df.groupby('movieId').size().reset_index(name='count')
    movies_merged = movies_df.merge(movie_counts, on='movieId', how='left').fillna(0)
    top_movies_df = movies_merged.nlargest(N_MOVIES, 'count')
    top_movies = top_movies_df['movieId'].tolist()

    movie_to_idx = {mid: i for i, mid in enumerate(top_movies)}
    idx_to_movie = {i: mid for mid, i in movie_to_idx.items()}
    movie_titles = {row['movieId']: row['title'] for _, row in movies_df.iterrows()}

    top_set = set(top_movies)
    ratings_filtered = ratings_df[ratings_df['movieId'].isin(top_set)].copy()

    return top_movies, movie_to_idx, idx_to_movie, movie_titles, ratings_filtered


def load_models():
    """Load model checkpoints."""
    models = {}

    # One-Hot model
    ckpt_path = CHECKPOINT_DIR / 'onehot_paper_config.pt'
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        n_items = ckpt['n_items']
        model = ExtrapolationModel(n_items)
        model.load_state_dict(ckpt['model_state_dict'])
        model.eval()
        models['onehot'] = {
            'model': model,
            'n_items': n_items,
            'epochs': ckpt.get('epochs', 0),
            'val_loss': ckpt.get('val_loss', 0),
        }
        print(f"  One-Hot: n_items={n_items}, epochs={ckpt.get('epochs', 0)}, "
              f"val_loss={ckpt.get('val_loss', 0):.4f}")

    # Concept model
    ckpt_path = CHECKPOINT_DIR / 'concept_paper_config.pt'
    emb_path = CHECKPOINT_DIR / 'concept_embeddings_paper_config.npy'
    if ckpt_path.exists() and emb_path.exists():
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        hidden_dim = ckpt['model_state_dict']['concept_proj.bias'].shape[0]
        n_items = ckpt['n_items']
        model = ConceptEmbeddingModel(n_items, 384, hidden_dim)
        model.load_state_dict(ckpt['model_state_dict'])
        model.eval()
        emb = np.load(emb_path)
        models['concept'] = {
            'model': model,
            'n_items': n_items,
            'embeddings': emb,
            'epochs': ckpt.get('epochs', 0),
            'val_loss': ckpt.get('val_loss', 0),
        }
        print(f"  Concept: n_items={n_items}, hidden_dim={hidden_dim}, "
              f"epochs={ckpt.get('epochs', 0)}, val_loss={ckpt.get('val_loss', 0):.4f}")

    return models


# ─── Prediction helpers ─────────────────────────────────────────────────────

def predict_onehot(model, n_items, revealed):
    """Get predictions from one-hot model. revealed = [(idx, rating), ...]"""
    indices = torch.arange(n_items).unsqueeze(0)
    rating_input = torch.zeros(1, n_items, 3)
    rating_input[:, :, 2] = 1  # All unknown
    for idx, rating in revealed:
        if idx < n_items:
            rating_input[:, idx, 2] = 0
            rating_input[:, idx, 1 if rating >= 4 else 0] = 1
    with torch.no_grad():
        return model(indices, rating_input)[:, -1, :N_MOVIES].sigmoid().numpy().flatten()


def predict_concept(model, emb, n_items, revealed):
    """Get predictions from concept model. revealed = [(idx, rating), ...]"""
    item_emb = torch.FloatTensor(emb).unsqueeze(0)
    rating_input = torch.zeros(1, n_items, 3)
    rating_input[:, :, 2] = 1  # All unknown
    for idx, rating in revealed:
        if idx < n_items:
            rating_input[:, idx, 2] = 0
            rating_input[:, idx, 1 if rating >= 4 else 0] = 1
    with torch.no_grad():
        return model(item_emb, rating_input)[:, -1, :N_MOVIES].sigmoid().numpy().flatten()


# ─── NDCG calculation (same as LLM diagnostic) ─────────────────────────────

def calculate_ndcg(recommended_ids, relevant_ids, k=10):
    """Binary relevance NDCG@k."""
    if not relevant_ids:
        return 0.0
    relevant_set = set(relevant_ids)
    dcg = 0.0
    for i, rec_id in enumerate(recommended_ids[:k]):
        if rec_id in relevant_set:
            dcg += 1.0 / np.log2(i + 2)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(k, len(relevant_ids))))
    return dcg / idcg if idcg > 0 else 0.0


def scores_to_top_k_ids(scores, idx_to_movie, k=10):
    """Convert model scores to top-k movie IDs."""
    top_indices = np.argsort(-scores)[:k]
    return [idx_to_movie[idx] for idx in top_indices if idx in idx_to_movie]


# ─── Main evaluation ────────────────────────────────────────────────────────

def build_user_profiles(ratings_filtered, movie_to_idx, min_eval_targets=5):
    """Build user profiles comparable to LLM diagnostic."""
    user_groups = ratings_filtered.groupby('userId')

    profiles = []
    for uid, group in user_groups:
        items = []
        liked_ids = []
        for _, row in group.iterrows():
            mid = row['movieId']
            if mid in movie_to_idx:
                idx = movie_to_idx[mid]
                items.append((idx, float(row['rating']), mid))
                if row['rating'] >= 4.0:
                    liked_ids.append(mid)

        if len(liked_ids) >= min_eval_targets:
            profiles.append({
                'user_id': int(uid),
                'items': items,
                'liked_ids': liked_ids,
            })

    return profiles


def run_model_eval(model_name, model_info, idx_to_movie, profiles,
                   max_steps=20, n_users=30):
    """Run step-by-step preference revelation eval for a model."""
    sample = profiles[:n_users]
    print(f"\n  Model: {model_name} ({len(sample)} users, {max_steps} steps)")

    model = model_info['model']
    n_items = model_info['n_items']

    all_curves = []

    for profile in tqdm(sample, desc=model_name, unit="user"):
        items = list(profile['items'])
        liked_ids = profile['liked_ids']

        # Shuffle revelation order (seeded by user_id)
        np.random.seed(profile['user_id'])
        np.random.shuffle(items)

        curve = []

        # Step 0: baseline (all unknown)
        if model_name == 'onehot':
            preds = predict_onehot(model, n_items, [])
        else:
            preds = predict_concept(model, model_info['embeddings'], n_items, [])

        rec_ids = scores_to_top_k_ids(preds, idx_to_movie)
        ndcg = calculate_ndcg(rec_ids, liked_ids)
        curve.append(ndcg)

        # Steps 1..max_steps
        for step in range(1, max_steps + 1):
            revealed = [(idx, rating) for idx, rating, mid in items[:min(step, len(items))]]

            if model_name == 'onehot':
                preds = predict_onehot(model, n_items, revealed)
            else:
                preds = predict_concept(model, model_info['embeddings'], n_items, revealed)

            rec_ids = scores_to_top_k_ids(preds, idx_to_movie)
            ndcg = calculate_ndcg(rec_ids, liked_ids)
            curve.append(ndcg)

        all_curves.append(curve)

    all_curves = np.array(all_curves)
    mean_curve = np.mean(all_curves, axis=0)
    std_curve = np.std(all_curves, axis=0)

    print(f"\n  {'Step':<6} {'NDCG':>8} {'Std':>8} {'Delta':>8}")
    print(f"  {'-'*32}")
    for step in range(max_steps + 1):
        delta = mean_curve[step] - mean_curve[step - 1] if step > 0 else 0.0
        print(f"  {step:<6} {mean_curve[step]:>8.4f} {std_curve[step]:>8.4f} {delta:>+8.4f}")

    total = mean_curve[-1] - mean_curve[0]
    peak_step = int(np.argmax(mean_curve))
    mono = sum(1 for i in range(1, len(mean_curve)) if mean_curve[i] >= mean_curve[i - 1])

    print(f"\n  Baseline: {mean_curve[0]:.4f}  Peak: {mean_curve[peak_step]:.4f} (s{peak_step})")
    print(f"  Final: {mean_curve[-1]:.4f}  Total: {total:+.4f}  Monotonic: {mono}/{max_steps}")

    return {
        'name': model_name,
        'n_users': len(all_curves),
        'mean_curve': mean_curve.tolist(),
        'std_curve': std_curve.tolist(),
        'total_improvement': float(total),
        'peak': float(mean_curve[peak_step]),
        'peak_step': peak_step,
        'monotonic_steps': mono,
    }


# ─── Main ───────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("LSTM+Attention Model Reward Signal Diagnostic")
    print("(Comparable to LLM diagnostic, pool=100, 20 steps)")
    print("=" * 60)

    print("\nLoading data...")
    top_movies, movie_to_idx, idx_to_movie, movie_titles, ratings_filtered = load_data()

    print("\nLoading models...")
    models = load_models()

    if not models:
        print("ERROR: No model checkpoints found in", CHECKPOINT_DIR)
        return

    # Build user profiles (same filtering as LLM diagnostic)
    profiles = build_user_profiles(ratings_filtered, movie_to_idx, min_eval_targets=5)
    np.random.seed(42)
    indices = np.random.permutation(len(profiles))
    profiles = [profiles[i] for i in indices]
    print(f"\n  {len(profiles)} users with >= 5 liked movies in pool=100")

    results = {}

    for name, info in models.items():
        results[name] = run_model_eval(name, info, idx_to_movie, profiles,
                                       max_steps=20, n_users=30)

    # Summary
    print(f"\n{'='*70}")
    print("NEURAL MODEL RESULTS")
    print(f"{'='*70}")
    print(f"{'Model':<12} {'Base':>7} {'Peak':>7} {'Final':>7} {'Improv':>8} "
          f"{'Mono':>6} {'Std@0':>7} {'Std@20':>7}")
    print(f"{'-'*68}")
    for key, r in results.items():
        print(f"{r['name']:<12} {r['mean_curve'][0]:>7.4f} {r['peak']:>7.4f} "
              f"{r['mean_curve'][-1]:>7.4f} {r['total_improvement']:>+8.4f} "
              f"{r['monotonic_steps']:>3}/20 {r['std_curve'][0]:>7.4f} "
              f"{r['std_curve'][-1]:>7.4f}")

    # LLM comparison
    llm_ref = RESULTS_DIR / 'preference_type_comparison.json'
    if llm_ref.exists():
        with open(llm_ref) as f:
            llm_data = json.load(f)
        print(f"\n{'='*70}")
        print("LLM DIAGNOSTIC REFERENCE (pool=100, same setup)")
        print(f"{'='*70}")
        print(f"{'Condition':<30} {'Base':>7} {'Final':>7} {'Improv':>8} {'Mono':>6}")
        print(f"{'-'*60}")
        for key, r in llm_data.items():
            print(f"{r['name']:<30} {r['mean_curve'][0]:>7.4f} "
                  f"{r['mean_curve'][-1]:>7.4f} "
                  f"{r['total_improvement']:>+8.4f} {r['monotonic_steps']:>3}/20")

    # Save
    out_file = RESULTS_DIR / 'lstm_reward_signal_diagnostic.json'
    with open(out_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {out_file}")


if __name__ == '__main__':
    main()
