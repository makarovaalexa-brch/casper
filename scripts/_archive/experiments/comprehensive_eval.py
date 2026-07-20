"""
Comprehensive Evaluation of All 3 Models

This script evaluates:
1. RL Signal - Loss/Accuracy by timestep (# of revealed preferences)
2. Full set vs Holdout evaluation (Paper Fig 2a vs 2b)
3. NDCG@10 ranking quality
4. Sanity checks:
   - Star Wars correlation test
   - Liked vs Disliked input effects
"""

import sys
sys.path.insert(0, '.')

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import json
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

# =============================================================================
# CONFIGURATION
# =============================================================================

DATA_DIR = Path('C:/dev/phd/casper/data/movielens')
CHECKPOINT_DIR = DATA_DIR / '.cache' / 'checkpoints'

N_MOVIES = 100
N_USERS_EVAL = 200  # Use 200 users for faster eval
SEED = 42

np.random.seed(SEED)
torch.manual_seed(SEED)

# =============================================================================
# MODEL DEFINITIONS
# =============================================================================

class ExtrapolationModel(nn.Module):
    """One-hot model."""
    def __init__(self, n_items, lr=0.001):
        super(ExtrapolationModel, self).__init__()
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


class UserTower(nn.Module):
    def __init__(self, state_dim=384, user_emb_dim=128, dropout=0.1):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, user_emb_dim)
        )

    def forward(self, state):
        return self.network(state)


class ItemTower(nn.Module):
    def __init__(self, item_dim=384, item_emb_dim=128, dropout=0.1):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(item_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, item_emb_dim)
        )

    def forward(self, item_features):
        return self.network(item_features)


class TwoTowerModel(nn.Module):
    def __init__(self, state_dim=384, embedding_dim=128, dropout=0.1):
        super().__init__()
        self.user_tower = UserTower(state_dim, embedding_dim, dropout)
        self.item_tower = ItemTower(state_dim, embedding_dim, dropout)

    def forward(self, user_states, item_features):
        user_emb = self.user_tower(user_states)
        item_emb = self.item_tower(item_features)
        user_emb = F.normalize(user_emb, p=2, dim=-1)
        item_emb = F.normalize(item_emb, p=2, dim=-1)
        return (user_emb * item_emb).sum(dim=1)

    def get_all_scores(self, user_state, all_item_features):
        """Get scores for one user against all items."""
        user_emb = self.user_tower(user_state)
        user_emb = F.normalize(user_emb, p=2, dim=-1)
        item_emb = self.item_tower(all_item_features)
        item_emb = F.normalize(item_emb, p=2, dim=-1)
        return torch.matmul(user_emb, item_emb.T)


# =============================================================================
# METRICS
# =============================================================================

def calc_bce_loss(predictions, ground_truth, mask=None):
    """BCE loss on rated items."""
    if mask is None:
        mask = ~np.isnan(ground_truth)
    if mask.sum() == 0:
        return np.nan

    y_true = ground_truth[mask]
    y_pred = np.clip(predictions[mask], 1e-7, 1 - 1e-7)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))


def calc_accuracy(predictions, ground_truth, mask=None):
    """Accuracy on rated items."""
    if mask is None:
        mask = ~np.isnan(ground_truth)
    if mask.sum() == 0:
        return np.nan

    pred_binary = (predictions[mask] > 0.5).astype(float)
    return np.mean(pred_binary == ground_truth[mask])


def calc_ndcg(predictions, ground_truth, k=10, mask=None):
    """NDCG@k on rated items."""
    if mask is None:
        mask = ~np.isnan(ground_truth)
    if mask.sum() == 0:
        return np.nan

    # Get indices of rated items sorted by prediction score
    rated_indices = np.where(mask)[0]
    preds_rated = predictions[rated_indices]
    gt_rated = ground_truth[rated_indices]

    # Sort by prediction score (descending)
    sorted_idx = np.argsort(-preds_rated)[:k]
    relevance = gt_rated[sorted_idx]

    # DCG
    dcg = np.sum(relevance / np.log2(np.arange(2, len(relevance) + 2)))

    # Ideal DCG
    ideal_relevance = np.sort(gt_rated)[::-1][:k]
    idcg = np.sum(ideal_relevance / np.log2(np.arange(2, len(ideal_relevance) + 2)))

    return dcg / idcg if idcg > 0 else 0


# =============================================================================
# LOAD DATA
# =============================================================================

print("=" * 80)
print("COMPREHENSIVE MODEL EVALUATION")
print("=" * 80)

# Load data
ratings = pd.read_csv(DATA_DIR / 'ratings.csv')
movies = pd.read_csv(DATA_DIR / 'movies.csv')

# Top movies
movie_counts = ratings.groupby('movieId').size().reset_index(name='count')
movies_with_counts = movies.merge(movie_counts, on='movieId', how='left')
movies_with_counts['count'] = movies_with_counts['count'].fillna(0)
top_movies_df = movies_with_counts.nlargest(N_MOVIES, 'count')
top_movies = top_movies_df['movieId'].tolist()
movie_to_idx = {mid: i for i, mid in enumerate(top_movies)}
idx_to_movie = {i: mid for mid, i in movie_to_idx.items()}
movie_titles = {row['movieId']: row['title'] for _, row in movies.iterrows()}

# Filter ratings
ratings_filtered = ratings[ratings['movieId'].isin(top_movies)].copy()

# Active users
user_rating_counts = ratings_filtered.groupby('userId').size()
active_users = user_rating_counts[user_rating_counts >= 25].index.tolist()

# Split users
np.random.seed(SEED)
shuffled_users = active_users.copy()
np.random.shuffle(shuffled_users)
val_users = shuffled_users[int(0.8 * len(shuffled_users)):]
EVAL_USERS = val_users[:N_USERS_EVAL]

print(f"Total active users: {len(active_users)}")
print(f"Evaluation users: {len(EVAL_USERS)}")

# Find Star Wars movies
star_wars_movies = []
for mid in top_movies:
    title = movie_titles.get(mid, "")
    if "Star Wars" in title:
        star_wars_movies.append((mid, movie_to_idx[mid], title))

print(f"\nStar Wars movies in top 100:")
for mid, idx, title in star_wars_movies:
    print(f"  [{idx}] {title}")

# =============================================================================
# LOAD MODELS
# =============================================================================

print("\n" + "=" * 80)
print("LOADING MODELS")
print("=" * 80)

models = {}

# One-Hot Model
onehot_path = CHECKPOINT_DIR / 'onehot_paper_config.pt'
if onehot_path.exists():
    ckpt = torch.load(onehot_path, map_location='cpu', weights_only=False)
    n_items = ckpt['n_items']
    onehot_model = ExtrapolationModel(n_items)
    onehot_model.load_state_dict(ckpt['model_state_dict'])
    onehot_model.eval()
    models['onehot'] = {
        'model': onehot_model,
        'n_items': n_items,
        'n_movies': ckpt.get('n_movies', N_MOVIES),
        'epochs': ckpt.get('epochs', 0),
        'val_loss': ckpt.get('val_loss', 0),
        'items': ckpt.get('items', [])
    }
    print(f"One-Hot: {n_items} items, epoch {ckpt.get('epochs', 0)}, val_loss={ckpt.get('val_loss', 0):.4f}")

# Concept Model
concept_path = CHECKPOINT_DIR / 'concept_paper_config.pt'
concept_emb_path = CHECKPOINT_DIR / 'concept_embeddings_paper_config.npy'
if concept_path.exists() and concept_emb_path.exists():
    ckpt = torch.load(concept_path, map_location='cpu', weights_only=False)
    n_items = ckpt['n_items']
    embedding_dim = ckpt.get('embedding_dim', 384)
    hidden_dim = ckpt['model_state_dict']['concept_proj.bias'].shape[0]

    concept_model = ConceptEmbeddingModel(n_items, embedding_dim, hidden_dim)
    concept_model.load_state_dict(ckpt['model_state_dict'])
    concept_model.eval()

    item_embeddings = np.load(concept_emb_path)

    models['concept'] = {
        'model': concept_model,
        'n_items': n_items,
        'n_movies': ckpt.get('n_movies', N_MOVIES),
        'epochs': ckpt.get('epochs', 0),
        'val_loss': ckpt.get('val_loss', 0),
        'embeddings': item_embeddings,
        'items': ckpt.get('items', [])
    }
    print(f"Concept: {n_items} items, epoch {ckpt.get('epochs', 0)}, val_loss={ckpt.get('val_loss', 0):.4f}")

# Two-Tower Model
twotower_path = CHECKPOINT_DIR / 'two_tower_paper_config.pt'
if twotower_path.exists():
    ckpt = torch.load(twotower_path, map_location='cpu', weights_only=False)

    twotower_model = TwoTowerModel(state_dim=384, embedding_dim=128)
    twotower_model.user_tower.load_state_dict(ckpt['user_tower'])
    twotower_model.item_tower.load_state_dict(ckpt['item_tower'])
    twotower_model.eval()

    # Load SBERT encoder for two-tower
    print("Loading SBERT encoder for Two-Tower...")
    sbert_encoder = SentenceTransformer('all-MiniLM-L6-v2')

    # Pre-compute movie embeddings
    movie_texts = [movie_titles.get(mid, f"Movie {mid}") for mid in top_movies]
    movie_embeddings_tt = sbert_encoder.encode(movie_texts, show_progress_bar=False)
    movie_embeddings_tt = torch.FloatTensor(movie_embeddings_tt)

    models['twotower'] = {
        'model': twotower_model,
        'n_movies': ckpt.get('n_movies', N_MOVIES),
        'epochs': ckpt.get('epoch', 0),
        'val_loss': ckpt.get('val_loss', 0),
        'encoder': sbert_encoder,
        'movie_embeddings': movie_embeddings_tt
    }
    print(f"Two-Tower: {N_MOVIES} movies, epoch {ckpt.get('epoch', 0)}, val_loss={ckpt.get('val_loss', 0):.4f}")


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_user_data(user_id):
    """Get user's ratings as ground truth array."""
    user_ratings = ratings_filtered[ratings_filtered['userId'] == user_id]
    ground_truth = np.full(N_MOVIES, np.nan)
    item_list = []

    for _, row in user_ratings.iterrows():
        if row['movieId'] in movie_to_idx:
            idx = movie_to_idx[row['movieId']]
            ground_truth[idx] = 1.0 if row['rating'] >= 4 else 0.0
            item_list.append((idx, row['rating'], row['movieId']))

    return ground_truth, item_list


def predict_onehot(model, n_items, revealed_items, n_movies=N_MOVIES):
    """Get predictions from one-hot model given revealed items."""
    indices = torch.arange(n_items).unsqueeze(0)
    rating_input = torch.zeros(1, n_items, 3)
    rating_input[:, :, 2] = 1  # All not_seen

    for idx, rating, _ in revealed_items:
        rating_input[:, idx, 2] = 0
        if rating >= 4:
            rating_input[:, idx, 1] = 1
        else:
            rating_input[:, idx, 0] = 1

    with torch.no_grad():
        preds = model(indices, rating_input)[:, -1, :n_movies].sigmoid().numpy().flatten()
    return preds


def predict_concept(model, embeddings, revealed_items, n_movies=N_MOVIES):
    """Get predictions from concept model given revealed items."""
    n_items = embeddings.shape[0]
    item_emb = torch.FloatTensor(embeddings).unsqueeze(0)
    rating_input = torch.zeros(1, n_items, 3)
    rating_input[:, :, 2] = 1

    for idx, rating, _ in revealed_items:
        rating_input[:, idx, 2] = 0
        if rating >= 4:
            rating_input[:, idx, 1] = 1
        else:
            rating_input[:, idx, 0] = 1

    with torch.no_grad():
        preds = model(item_emb, rating_input)[:, -1, :n_movies].sigmoid().numpy().flatten()
    return preds


def predict_twotower(model, encoder, movie_embeddings, revealed_items):
    """Get predictions from two-tower model given revealed items."""
    # Create preference text
    liked = [movie_titles.get(mid, "Unknown")[:30] for idx, r, mid in revealed_items if r >= 4][:10]
    disliked = [movie_titles.get(mid, "Unknown")[:30] for idx, r, mid in revealed_items if r < 4][:5]

    parts = []
    if liked:
        parts.append(f"likes: {', '.join(liked)}")
    if disliked:
        parts.append(f"dislikes: {', '.join(disliked)}")
    pref_text = ' | '.join(parts) if parts else "no preferences"

    # Encode user state
    user_state = encoder.encode([pref_text], show_progress_bar=False)
    user_state = torch.FloatTensor(user_state)

    # Get scores for all movies
    with torch.no_grad():
        scores = model.get_all_scores(user_state, movie_embeddings)
        # Convert to probabilities via sigmoid
        preds = torch.sigmoid(scores * 5).numpy().flatten()  # Scale for sigmoid
    return preds


# =============================================================================
# EVALUATION 1: TIMESTEP-BASED (RL SIGNAL)
# =============================================================================

print("\n" + "=" * 80)
print("EVALUATION 1: RL SIGNAL BY TIMESTEP")
print("(Paper Figure 2 - Loss/Accuracy as more preferences revealed)")
print("=" * 80)

timesteps = [1, 3, 5, 10, 20, 30, 50]
results_by_model = {}

for model_name in ['onehot', 'concept', 'twotower']:
    if model_name not in models:
        continue

    print(f"\n--- {model_name.upper()} ---")

    results = {t: {'loss': [], 'accuracy': [], 'ndcg': [],
                   'holdout_loss': [], 'holdout_acc': [], 'holdout_ndcg': []}
               for t in timesteps}

    for user_id in tqdm(EVAL_USERS, desc=f"Evaluating {model_name}"):
        ground_truth, item_list = get_user_data(user_id)

        if len(item_list) < max(timesteps):
            continue

        np.random.seed(user_id)
        np.random.shuffle(item_list)

        for n_prefs in timesteps:
            if n_prefs > len(item_list):
                continue

            revealed = item_list[:n_prefs]
            revealed_indices = set(idx for idx, _, _ in revealed)

            # Get predictions
            if model_name == 'onehot':
                preds = predict_onehot(models['onehot']['model'],
                                       models['onehot']['n_items'], revealed)
            elif model_name == 'concept':
                preds = predict_concept(models['concept']['model'],
                                        models['concept']['embeddings'], revealed)
            else:  # twotower
                preds = predict_twotower(models['twotower']['model'],
                                         models['twotower']['encoder'],
                                         models['twotower']['movie_embeddings'], revealed)

            # Full set metrics (Paper Fig 2a)
            loss = calc_bce_loss(preds, ground_truth)
            acc = calc_accuracy(preds, ground_truth)
            ndcg = calc_ndcg(preds, ground_truth)

            if not np.isnan(loss):
                results[n_prefs]['loss'].append(loss)
            if not np.isnan(acc):
                results[n_prefs]['accuracy'].append(acc)
            if not np.isnan(ndcg):
                results[n_prefs]['ndcg'].append(ndcg)

            # Holdout metrics (Paper Fig 2b) - exclude revealed items
            holdout_mask = ~np.isnan(ground_truth) & np.array([i not in revealed_indices for i in range(N_MOVIES)])
            if holdout_mask.sum() > 0:
                h_loss = calc_bce_loss(preds, ground_truth, holdout_mask)
                h_acc = calc_accuracy(preds, ground_truth, holdout_mask)
                h_ndcg = calc_ndcg(preds, ground_truth, mask=holdout_mask)

                if not np.isnan(h_loss):
                    results[n_prefs]['holdout_loss'].append(h_loss)
                if not np.isnan(h_acc):
                    results[n_prefs]['holdout_acc'].append(h_acc)
                if not np.isnan(h_ndcg):
                    results[n_prefs]['holdout_ndcg'].append(h_ndcg)

    # Print results table
    print(f"\n{'Inputs':<8} | {'Loss':<8} | {'Acc':<8} | {'NDCG@10':<8} || {'H-Loss':<8} | {'H-Acc':<8} | {'H-NDCG':<8}")
    print("-" * 85)

    summary = {}
    for t in timesteps:
        if results[t]['loss']:
            avg_loss = np.mean(results[t]['loss'])
            avg_acc = np.mean(results[t]['accuracy'])
            avg_ndcg = np.mean(results[t]['ndcg'])

            avg_h_loss = np.mean(results[t]['holdout_loss']) if results[t]['holdout_loss'] else np.nan
            avg_h_acc = np.mean(results[t]['holdout_acc']) if results[t]['holdout_acc'] else np.nan
            avg_h_ndcg = np.mean(results[t]['holdout_ndcg']) if results[t]['holdout_ndcg'] else np.nan

            summary[t] = {
                'loss': avg_loss, 'accuracy': avg_acc, 'ndcg': avg_ndcg,
                'holdout_loss': avg_h_loss, 'holdout_acc': avg_h_acc, 'holdout_ndcg': avg_h_ndcg
            }

            print(f"{t:<8} | {avg_loss:.4f}   | {avg_acc:.4f}   | {avg_ndcg:.4f}   || "
                  f"{avg_h_loss:.4f}   | {avg_h_acc:.4f}   | {avg_h_ndcg:.4f}")

    # RL Signal
    if 1 in summary and max(timesteps) in summary:
        loss_improvement = summary[1]['loss'] - summary[max(timesteps)]['loss']
        acc_improvement = summary[max(timesteps)]['accuracy'] - summary[1]['accuracy']
        print(f"\nRL Signal (1 -> {max(timesteps)} inputs):")
        print(f"  Loss reduction: {loss_improvement:+.4f}")
        print(f"  Accuracy gain:  {acc_improvement:+.4f}")

        if not np.isnan(summary[1]['holdout_loss']) and not np.isnan(summary[max(timesteps)]['holdout_loss']):
            h_loss_imp = summary[1]['holdout_loss'] - summary[max(timesteps)]['holdout_loss']
            h_acc_imp = summary[max(timesteps)]['holdout_acc'] - summary[1]['holdout_acc']
            print(f"  Holdout loss reduction: {h_loss_imp:+.4f}")
            print(f"  Holdout accuracy gain:  {h_acc_imp:+.4f}")

    results_by_model[model_name] = summary


# =============================================================================
# EVALUATION 2: STAR WARS SANITY CHECK
# =============================================================================

print("\n" + "=" * 80)
print("EVALUATION 2: STAR WARS CORRELATION TEST")
print("(If user likes one Star Wars, do others rank highly?)")
print("=" * 80)

if star_wars_movies:
    test_movie = star_wars_movies[0]  # First Star Wars movie
    test_idx = test_movie[1]
    test_title = test_movie[2]

    print(f"\nTest: Input '{test_title}' as LIKED")
    print("Expected: Other Star Wars movies should rank high\n")

    for model_name in ['onehot', 'concept', 'twotower']:
        if model_name not in models:
            continue

        print(f"--- {model_name.upper()} ---")

        # Create single revealed item (liked)
        revealed = [(test_idx, 5.0, test_movie[0])]  # rating 5 = liked

        if model_name == 'onehot':
            preds = predict_onehot(models['onehot']['model'],
                                   models['onehot']['n_items'], revealed)
        elif model_name == 'concept':
            preds = predict_concept(models['concept']['model'],
                                    models['concept']['embeddings'], revealed)
        else:
            preds = predict_twotower(models['twotower']['model'],
                                     models['twotower']['encoder'],
                                     models['twotower']['movie_embeddings'], revealed)

        # Get top 10 predictions
        top_indices = np.argsort(-preds)[:15]
        print(f"Top 15 predictions (excluding input):")
        rank = 1
        sw_ranks = []
        for idx in top_indices:
            if idx == test_idx:
                continue
            mid = idx_to_movie.get(idx, -1)
            title = movie_titles.get(mid, "Unknown")
            is_sw = "*** STAR WARS ***" if "Star Wars" in title else ""
            print(f"  {rank:2d}. [{idx:2d}] {title[:50]:<50} ({preds[idx]:.3f}) {is_sw}")
            if "Star Wars" in title:
                sw_ranks.append(rank)
            rank += 1
            if rank > 10:
                break

        # Check other Star Wars rankings
        print(f"\nOther Star Wars movies rankings:")
        for mid, idx, title in star_wars_movies:
            if idx != test_idx:
                all_ranks = np.argsort(-preds)
                movie_rank = np.where(all_ranks == idx)[0][0] + 1
                print(f"  {title[:40]:<40} -> Rank {movie_rank}")
        print()


# =============================================================================
# EVALUATION 3: LIKED vs DISLIKED EFFECT
# =============================================================================

print("\n" + "=" * 80)
print("EVALUATION 3: LIKED vs DISLIKED INPUT EFFECT")
print("(How does rating polarity affect predictions?)")
print("=" * 80)

# Use first Star Wars movie for test
if star_wars_movies:
    test_movie = star_wars_movies[0]
    test_idx = test_movie[1]
    test_title = test_movie[2]

    print(f"\nTest movie: '{test_title}'")

    for model_name in ['onehot', 'concept', 'twotower']:
        if model_name not in models:
            continue

        print(f"\n--- {model_name.upper()} ---")

        # LIKED input
        revealed_liked = [(test_idx, 5.0, test_movie[0])]
        if model_name == 'onehot':
            preds_liked = predict_onehot(models['onehot']['model'],
                                         models['onehot']['n_items'], revealed_liked)
        elif model_name == 'concept':
            preds_liked = predict_concept(models['concept']['model'],
                                          models['concept']['embeddings'], revealed_liked)
        else:
            preds_liked = predict_twotower(models['twotower']['model'],
                                           models['twotower']['encoder'],
                                           models['twotower']['movie_embeddings'], revealed_liked)

        # DISLIKED input
        revealed_disliked = [(test_idx, 1.0, test_movie[0])]
        if model_name == 'onehot':
            preds_disliked = predict_onehot(models['onehot']['model'],
                                            models['onehot']['n_items'], revealed_disliked)
        elif model_name == 'concept':
            preds_disliked = predict_concept(models['concept']['model'],
                                             models['concept']['embeddings'], revealed_disliked)
        else:
            preds_disliked = predict_twotower(models['twotower']['model'],
                                              models['twotower']['encoder'],
                                              models['twotower']['movie_embeddings'], revealed_disliked)

        # Compare predictions
        diff = preds_liked - preds_disliked

        print(f"\nTop 10 movies MOST AFFECTED by like vs dislike:")
        most_affected = np.argsort(-np.abs(diff))[:10]
        for idx in most_affected:
            if idx == test_idx:
                continue
            mid = idx_to_movie.get(idx, -1)
            title = movie_titles.get(mid, "Unknown")
            print(f"  [{idx:2d}] {title[:40]:<40} | Liked: {preds_liked[idx]:.3f} | Disliked: {preds_disliked[idx]:.3f} | Diff: {diff[idx]:+.3f}")

        # Check other Star Wars
        print(f"\nStar Wars movies effect:")
        for mid, idx, title in star_wars_movies:
            if idx != test_idx:
                print(f"  {title[:40]:<40} | Liked: {preds_liked[idx]:.3f} | Disliked: {preds_disliked[idx]:.3f} | Diff: {diff[idx]:+.3f}")


# =============================================================================
# SUMMARY COMPARISON
# =============================================================================

print("\n" + "=" * 80)
print("SUMMARY COMPARISON")
print("=" * 80)

print("\n--- Model Info ---")
for name in ['onehot', 'concept', 'twotower']:
    if name in models:
        m = models[name]
        print(f"{name.upper():12} | Epoch: {m.get('epochs', m.get('epoch', 0)):3d}/100 | Val Loss: {m['val_loss']:.4f}")

print("\n--- Best Metrics at 50 Inputs ---")
print(f"{'Model':<12} | {'Loss':<8} | {'Accuracy':<8} | {'NDCG@10':<8} | {'H-Loss':<8} | {'H-Acc':<8}")
print("-" * 75)
for name in ['onehot', 'concept', 'twotower']:
    if name in results_by_model and 50 in results_by_model[name]:
        r = results_by_model[name][50]
        print(f"{name.upper():<12} | {r['loss']:.4f}   | {r['accuracy']:.4f}   | {r['ndcg']:.4f}   | "
              f"{r['holdout_loss']:.4f}   | {r['holdout_acc']:.4f}")

print("\n--- RL Signal (Loss Reduction 1->50 inputs) ---")
for name in ['onehot', 'concept', 'twotower']:
    if name in results_by_model:
        r = results_by_model[name]
        if 1 in r and 50 in r:
            loss_red = r[1]['loss'] - r[50]['loss']
            acc_gain = r[50]['accuracy'] - r[1]['accuracy']
            print(f"{name.upper():<12} | Loss: {loss_red:+.4f} | Acc: {acc_gain:+.4f}")

print("\n" + "=" * 80)
print("EVALUATION COMPLETE")
print("=" * 80)
