"""
Unified evaluation across all models using IDENTICAL users and setup.

Updated for Paper Configuration:
- 100 movies, 50 actors, 20 directors, 20 genres = 187 items
- 100 epochs training
- Same 500 validation users for ALL models
"""

import sys
sys.path.insert(0, '.')

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
import json

# =============================================================================
# CONFIGURATION (Match paper config)
# =============================================================================

DATA_DIR = Path('C:/dev/phd/casper/data/movielens')
CHECKPOINT_DIR = DATA_DIR / '.cache' / 'checkpoints'

N_MOVIES = 100  # Paper config
N_USERS_EVAL = 500
SEED = 42

# =============================================================================
# LOAD SHARED DATA
# =============================================================================

print("=" * 80)
print("UNIFIED EVALUATION - PAPER CONFIGURATION")
print("=" * 80)
print(f"Config: {N_MOVIES} movies, {N_USERS_EVAL} eval users")

# Load data once
ratings = pd.read_csv(DATA_DIR / 'ratings.csv')
movies = pd.read_csv(DATA_DIR / 'movies.csv')

# Get top movies (same as training)
movie_counts = ratings.groupby('movieId').size().reset_index(name='count')
movies_with_counts = movies.merge(movie_counts, on='movieId', how='left')
movies_with_counts['count'] = movies_with_counts['count'].fillna(0)
top_movies_df = movies_with_counts.nlargest(N_MOVIES, 'count')
top_movies = top_movies_df['movieId'].tolist()
movie_to_idx = {mid: i for i, mid in enumerate(top_movies)}
movie_titles = {row['movieId']: row['title'] for _, row in movies.iterrows()}

# Filter ratings
ratings_filtered = ratings[ratings['movieId'].isin(top_movies)].copy()
ratings_filtered['itemIdx'] = ratings_filtered['movieId'].map(movie_to_idx)

# Get users with at least 25 ratings on top 100 movies (paper: "quarter of ratings available")
user_rating_counts = ratings_filtered.groupby('userId').size()
active_users = user_rating_counts[user_rating_counts >= 25].index.tolist()
print(f"Users with >= 25 ratings on top {N_MOVIES} movies: {len(active_users)}")

# 80/20 split - SAME as training (with fixed seed for reproducibility)
np.random.seed(SEED)
shuffled_users = active_users.copy()
np.random.shuffle(shuffled_users)
train_users = shuffled_users[:int(0.8 * len(shuffled_users))]
val_users = shuffled_users[int(0.8 * len(shuffled_users)):]

# Use SAME N users for ALL evaluations
EVAL_USERS = val_users[:N_USERS_EVAL]
print(f"Evaluation users: {len(EVAL_USERS)}")

# =============================================================================
# MODEL DEFINITIONS (copy from training scripts for loading)
# =============================================================================

class ExtrapolationModel(nn.Module):
    """One-hot model matching paper architecture."""
    def __init__(self, n_items, lr=0.001):
        super(ExtrapolationModel, self).__init__()
        self.n_items = n_items
        hidden_dim = n_items // 2

        self.embedding = nn.Embedding(n_items + 1, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim + 3, hidden_dim, 1, batch_first=True, bidirectional=False)
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
        x = self.output(x)
        return x


class ConceptEmbeddingModel(nn.Module):
    """Concept model with SBERT embeddings."""
    def __init__(self, n_items, embedding_dim=384, hidden_dim=None, lr=0.001):
        super(ConceptEmbeddingModel, self).__init__()
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

# =============================================================================
# HELPER: Calculate metrics
# =============================================================================

def calc_bce_loss(predictions, ground_truth):
    """BCE loss on rated items."""
    filt = ~np.isnan(ground_truth)
    if filt.sum() == 0:
        return np.nan

    y_true = ground_truth[filt]
    y_pred = np.clip(predictions[filt], 1e-7, 1 - 1e-7)
    bce = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    return bce


def calc_accuracy(predictions, ground_truth):
    """Accuracy on rated items."""
    filt = ~np.isnan(ground_truth)
    if filt.sum() == 0:
        return np.nan

    pred_binary = (predictions[filt] > 0.5).astype(float)
    return np.mean(pred_binary == ground_truth[filt])

# =============================================================================
# 1. ONE-HOT MODEL EVALUATION
# =============================================================================

def evaluate_onehot(timesteps=[1, 3, 5, 10, 20, 30, 50, 70, 100]):
    """Evaluate one-hot model using paper methodology."""
    print("\n" + "=" * 60)
    print("ONE-HOT MODEL (Paper Config: 187 items, 100 epochs)")
    print("=" * 60)

    checkpoint_path = CHECKPOINT_DIR / 'onehot_paper_config.pt'
    if not checkpoint_path.exists():
        print(f"Checkpoint not found: {checkpoint_path}")
        return {}

    checkpoint = torch.load(checkpoint_path, weights_only=False)
    n_items = checkpoint['n_items']
    n_movies = checkpoint.get('n_movies', N_MOVIES)

    model = ExtrapolationModel(n_items)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"  Loaded: {n_items} items, {n_movies} movies")
    print(f"  Val loss: {checkpoint.get('val_loss', 'N/A'):.4f}")

    results = {t: {'loss': [], 'accuracy': []} for t in timesteps}
    users_evaluated = 0

    for user_id in EVAL_USERS:
        user_ratings = ratings_filtered[ratings_filtered['userId'] == user_id]

        if len(user_ratings) < max(timesteps):
            continue

        # Build ground truth for ALL user's rated items
        ground_truth = np.full(n_movies, np.nan)
        item_list = []
        for _, row in user_ratings.iterrows():
            if row['movieId'] in movie_to_idx:
                idx = movie_to_idx[row['movieId']]
                ground_truth[idx] = 1.0 if row['rating'] >= 4 else 0.0
                item_list.append((idx, row['rating']))

        if len(item_list) < max(timesteps):
            continue

        # Shuffle item order
        np.random.seed(user_id)
        np.random.shuffle(item_list)
        users_evaluated += 1

        for n_prefs in timesteps:
            if n_prefs > len(item_list):
                continue

            # Build input with n_prefs revealed
            indices = torch.arange(n_items).unsqueeze(0)
            rating_input = torch.zeros(1, n_items, 3)
            rating_input[:, :, 2] = 1  # All not_seen

            for i in range(n_prefs):
                idx, rating = item_list[i]
                rating_input[:, idx, 2] = 0
                if rating >= 4:
                    rating_input[:, idx, 1] = 1
                else:
                    rating_input[:, idx, 0] = 1

            # Predict
            with torch.no_grad():
                preds = model(indices, rating_input)[:, -1, :n_movies].sigmoid().numpy().flatten()

            loss = calc_bce_loss(preds, ground_truth)
            accuracy = calc_accuracy(preds, ground_truth)

            if not np.isnan(loss):
                results[n_prefs]['loss'].append(loss)
            if not np.isnan(accuracy):
                results[n_prefs]['accuracy'].append(accuracy)

    print(f"\n  Evaluated {users_evaluated} users")
    print(f"\n  {'Inputs':<8} | {'Loss':<8} | {'Accuracy':<8} | N")
    print("  " + "-" * 40)

    summary = {}
    for t in sorted(timesteps):
        if results[t]['loss']:
            avg_loss = np.mean(results[t]['loss'])
            avg_acc = np.mean(results[t]['accuracy']) if results[t]['accuracy'] else 0
            summary[t] = {'loss': avg_loss, 'accuracy': avg_acc}
            print(f"  {t:<8} | {avg_loss:.4f}   | {avg_acc:.4f}   | {len(results[t]['loss'])}")

    return summary

# =============================================================================
# 2. CONCEPT MODEL EVALUATION
# =============================================================================

def evaluate_concept(timesteps=[1, 3, 5, 10, 20, 30, 50, 70, 100]):
    """Evaluate concept embedding model."""
    print("\n" + "=" * 60)
    print("CONCEPT (SBERT) MODEL (Paper Config)")
    print("=" * 60)

    checkpoint_path = CHECKPOINT_DIR / 'concept_paper_config.pt'
    embeddings_path = CHECKPOINT_DIR / 'concept_embeddings_paper_config.npy'

    if not checkpoint_path.exists():
        print(f"Checkpoint not found: {checkpoint_path}")
        return {}

    checkpoint = torch.load(checkpoint_path, weights_only=False)
    n_items = checkpoint['n_items']
    n_movies = checkpoint.get('n_movies', N_MOVIES)
    embedding_dim = checkpoint.get('embedding_dim', 384)

    # Load embeddings
    if embeddings_path.exists():
        item_embeddings = np.load(embeddings_path)
    else:
        print(f"Embeddings not found: {embeddings_path}")
        return {}

    # Get hidden_dim from checkpoint
    hidden_dim = checkpoint['model_state_dict']['concept_proj.bias'].shape[0]

    model = ConceptEmbeddingModel(n_items, embedding_dim, hidden_dim=hidden_dim)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"  Loaded: {n_items} items, embedding_dim={embedding_dim}")
    print(f"  Val loss: {checkpoint.get('val_loss', 'N/A'):.4f}")

    item_emb_tensor = torch.FloatTensor(item_embeddings).unsqueeze(0)

    results = {t: {'loss': [], 'accuracy': []} for t in timesteps}
    users_evaluated = 0

    for user_id in EVAL_USERS:
        user_ratings = ratings_filtered[ratings_filtered['userId'] == user_id]

        if len(user_ratings) < max(timesteps):
            continue

        ground_truth = np.full(n_movies, np.nan)
        item_list = []
        for _, row in user_ratings.iterrows():
            if row['movieId'] in movie_to_idx:
                idx = movie_to_idx[row['movieId']]
                ground_truth[idx] = 1.0 if row['rating'] >= 4 else 0.0
                item_list.append((idx, row['rating']))

        if len(item_list) < max(timesteps):
            continue

        np.random.seed(user_id)
        np.random.shuffle(item_list)
        users_evaluated += 1

        for n_prefs in timesteps:
            if n_prefs > len(item_list):
                continue

            rating_input = torch.zeros(1, n_items, 3)
            rating_input[:, :, 2] = 1

            for i in range(n_prefs):
                idx, rating = item_list[i]
                rating_input[:, idx, 2] = 0
                if rating >= 4:
                    rating_input[:, idx, 1] = 1
                else:
                    rating_input[:, idx, 0] = 1

            with torch.no_grad():
                preds = model(item_emb_tensor, rating_input)[:, -1, :n_movies].sigmoid().numpy().flatten()

            loss = calc_bce_loss(preds, ground_truth)
            accuracy = calc_accuracy(preds, ground_truth)

            if not np.isnan(loss):
                results[n_prefs]['loss'].append(loss)
            if not np.isnan(accuracy):
                results[n_prefs]['accuracy'].append(accuracy)

    print(f"\n  Evaluated {users_evaluated} users")
    print(f"\n  {'Inputs':<8} | {'Loss':<8} | {'Accuracy':<8} | N")
    print("  " + "-" * 40)

    summary = {}
    for t in sorted(timesteps):
        if results[t]['loss']:
            avg_loss = np.mean(results[t]['loss'])
            avg_acc = np.mean(results[t]['accuracy']) if results[t]['accuracy'] else 0
            summary[t] = {'loss': avg_loss, 'accuracy': avg_acc}
            print(f"  {t:<8} | {avg_loss:.4f}   | {avg_acc:.4f}   | {len(results[t]['loss'])}")

    return summary

# =============================================================================
# 3. TWO-TOWER MODEL EVALUATION
# =============================================================================

def evaluate_two_tower(timesteps=[1, 3, 5, 10, 20, 30, 50]):
    """Evaluate two-tower model using InfoNCE loss."""
    print("\n" + "=" * 60)
    print("TWO-TOWER MODEL (Paper Config)")
    print("=" * 60)

    checkpoint_path = CHECKPOINT_DIR / 'two_tower_paper_config.pt'

    if not checkpoint_path.exists():
        print(f"Checkpoint not found: {checkpoint_path}")
        return {}

    try:
        from sentence_transformers import SentenceTransformer
        import torch.nn.functional as F
    except ImportError as e:
        print(f"Import error: {e}")
        return {}

    checkpoint = torch.load(checkpoint_path, weights_only=False)
    print(f"  Val loss: {checkpoint.get('val_loss', 'N/A'):.4f}")
    print(f"  Epochs: {checkpoint.get('epoch', 'N/A')}")

    # Two-tower uses different loss metric (InfoNCE), hard to compare directly
    # For now just report training metrics
    print("\n  NOTE: Two-Tower uses InfoNCE loss (not BCE)")
    print("  Direct loss comparison not applicable")
    print("  See checkpoint for training metrics")

    return {'training_loss': checkpoint.get('val_loss', float('nan'))}

# =============================================================================
# 4. LLM EVALUATION
# =============================================================================

def evaluate_llm(timesteps=[1, 3, 5, 10], use_cache=True):
    """Evaluate LLM recommender."""
    print("\n" + "=" * 60)
    print("LLM (GPT-4o-mini) MODEL")
    print("=" * 60)

    try:
        from experiment_llm_recommender import call_llm, create_recommendation_prompt, parse_llm_recommendations
    except ImportError as e:
        print(f"Import error: {e}")
        return {}

    print("  LLM evaluation uses NDCG (not BCE loss)")
    print("  Skipping for now - uncomment in main to enable")
    return {}

# =============================================================================
# MAIN
# =============================================================================

def run_unified_evaluation():
    """Run unified evaluation on all available models."""
    print(f"\nUsing {len(EVAL_USERS)} identical users for all models")

    all_results = {}
    timesteps = [1, 3, 5, 10, 20, 30, 50, 70, 100]

    # Run evaluations
    all_results['One-Hot'] = evaluate_onehot(timesteps)
    all_results['Concept'] = evaluate_concept(timesteps)
    all_results['Two-Tower'] = evaluate_two_tower()
    # all_results['LLM'] = evaluate_llm()

    # Summary
    print("\n" + "=" * 80)
    print("COMPARISON SUMMARY")
    print("=" * 80)

    models = [m for m in ['One-Hot', 'Concept'] if all_results.get(m)]

    if models:
        print("\n--- BCE Loss by # Inputs (should DECREASE) ---")
        print(f"{'Inputs':<10}", end="")
        for m in models:
            print(f"{m:<15}", end="")
        print()
        print("-" * 50)

        for t in timesteps:
            print(f"{t:<10}", end="")
            for m in models:
                if t in all_results[m]:
                    loss = all_results[m][t].get('loss', 'N/A')
                    if isinstance(loss, float):
                        print(f"{loss:<15.4f}", end="")
                    else:
                        print(f"{'N/A':<15}", end="")
                else:
                    print(f"{'N/A':<15}", end="")
            print()

        # RL Signal Summary
        print("\n--- RL Signal (Loss Reduction) ---")
        for m in models:
            ts = [t for t in timesteps if t in all_results[m]]
            if len(ts) >= 2:
                t_min, t_max = min(ts), max(ts)
                loss_start = all_results[m][t_min].get('loss', 0)
                loss_end = all_results[m][t_max].get('loss', 0)
                reduction = loss_start - loss_end
                pct = (reduction / loss_start * 100) if loss_start > 0 else 0
                print(f"  {m}: {loss_start:.4f} -> {loss_end:.4f} (Δ = {reduction:.4f}, {pct:.1f}% reduction)")

    # Save results
    results_file = DATA_DIR / '.cache' / 'unified_evaluation_results.json'
    with open(results_file, 'w') as f:
        # Convert numpy to native types
        def convert(obj):
            if isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            return obj
        json.dump(convert(all_results), f, indent=2)
    print(f"\nResults saved to: {results_file}")

    return all_results


if __name__ == "__main__":
    run_unified_evaluation()
