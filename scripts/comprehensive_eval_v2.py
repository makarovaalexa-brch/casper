"""
Comprehensive Evaluation v2 - Fixing Issues

Changes:
1. Same user group for all timesteps (don't exclude based on rating count)
2. Paper Figure 5 replication - pairwise correlation test
3. Paper Table I replication - attribute effect on movies
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
N_USERS_EVAL = 300  # Use more users
SEED = 42

np.random.seed(SEED)
torch.manual_seed(SEED)

# =============================================================================
# MODEL DEFINITIONS
# =============================================================================

class ExtrapolationModel(nn.Module):
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
        user_emb = self.user_tower(user_state)
        user_emb = F.normalize(user_emb, p=2, dim=-1)
        item_emb = self.item_tower(all_item_features)
        item_emb = F.normalize(item_emb, p=2, dim=-1)
        return torch.matmul(user_emb, item_emb.T)


# =============================================================================
# METRICS
# =============================================================================

def calc_bce_loss(predictions, ground_truth, mask=None):
    if mask is None:
        mask = ~np.isnan(ground_truth)
    if mask.sum() == 0:
        return np.nan
    y_true = ground_truth[mask]
    y_pred = np.clip(predictions[mask], 1e-7, 1 - 1e-7)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))


def calc_accuracy(predictions, ground_truth, mask=None):
    if mask is None:
        mask = ~np.isnan(ground_truth)
    if mask.sum() == 0:
        return np.nan
    pred_binary = (predictions[mask] > 0.5).astype(float)
    return np.mean(pred_binary == ground_truth[mask])


def calc_ndcg(predictions, ground_truth, k=10, mask=None):
    if mask is None:
        mask = ~np.isnan(ground_truth)
    if mask.sum() == 0:
        return np.nan
    rated_indices = np.where(mask)[0]
    preds_rated = predictions[rated_indices]
    gt_rated = ground_truth[rated_indices]
    sorted_idx = np.argsort(-preds_rated)[:k]
    relevance = gt_rated[sorted_idx]
    dcg = np.sum(relevance / np.log2(np.arange(2, len(relevance) + 2)))
    ideal_relevance = np.sort(gt_rated)[::-1][:k]
    idcg = np.sum(ideal_relevance / np.log2(np.arange(2, len(ideal_relevance) + 2)))
    return dcg / idcg if idcg > 0 else 0


# =============================================================================
# LOAD DATA
# =============================================================================

print("=" * 80)
print("COMPREHENSIVE MODEL EVALUATION v2")
print("=" * 80)

ratings = pd.read_csv(DATA_DIR / 'ratings.csv')
movies = pd.read_csv(DATA_DIR / 'movies.csv')

movie_counts = ratings.groupby('movieId').size().reset_index(name='count')
movies_with_counts = movies.merge(movie_counts, on='movieId', how='left')
movies_with_counts['count'] = movies_with_counts['count'].fillna(0)
top_movies_df = movies_with_counts.nlargest(N_MOVIES, 'count')
top_movies = top_movies_df['movieId'].tolist()
movie_to_idx = {mid: i for i, mid in enumerate(top_movies)}
idx_to_movie = {i: mid for mid, i in movie_to_idx.items()}
movie_titles = {row['movieId']: row['title'] for _, row in movies.iterrows()}

ratings_filtered = ratings[ratings['movieId'].isin(top_movies)].copy()

user_rating_counts = ratings_filtered.groupby('userId').size()
active_users = user_rating_counts[user_rating_counts >= 10].index.tolist()  # Lower threshold

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
    print("Loading SBERT encoder for Two-Tower...")
    sbert_encoder = SentenceTransformer('all-MiniLM-L6-v2')
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
    indices = torch.arange(n_items).unsqueeze(0)
    rating_input = torch.zeros(1, n_items, 3)
    rating_input[:, :, 2] = 1
    for idx, rating, _ in revealed_items:
        rating_input[:, idx, 2] = 0
        if rating >= 4:
            rating_input[:, idx, 1] = 1
        else:
            rating_input[:, idx, 0] = 1
    with torch.no_grad():
        preds = model(indices, rating_input)[:, -1, :n_movies].sigmoid().numpy().flatten()
    return preds


def predict_onehot_with_rating(model, n_items, item_idx, is_liked, n_movies=N_MOVIES):
    """Predict with a single item as input with explicit liked/disliked."""
    indices = torch.arange(n_items).unsqueeze(0)
    rating_input = torch.zeros(1, n_items, 3)
    rating_input[:, :, 2] = 1  # All unknown
    rating_input[:, item_idx, 2] = 0
    if is_liked:
        rating_input[:, item_idx, 1] = 1
    else:
        rating_input[:, item_idx, 0] = 1
    with torch.no_grad():
        preds = model(indices, rating_input)[:, -1, :n_movies].sigmoid().numpy().flatten()
    return preds


def predict_concept(model, embeddings, revealed_items, n_movies=N_MOVIES):
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


def predict_concept_with_rating(model, embeddings, item_idx, is_liked, n_movies=N_MOVIES):
    """Predict with a single item as input with explicit liked/disliked."""
    n_items = embeddings.shape[0]
    item_emb = torch.FloatTensor(embeddings).unsqueeze(0)
    rating_input = torch.zeros(1, n_items, 3)
    rating_input[:, :, 2] = 1
    rating_input[:, item_idx, 2] = 0
    if is_liked:
        rating_input[:, item_idx, 1] = 1
    else:
        rating_input[:, item_idx, 0] = 1
    with torch.no_grad():
        preds = model(item_emb, rating_input)[:, -1, :n_movies].sigmoid().numpy().flatten()
    return preds


def predict_twotower(model, encoder, movie_embeddings, revealed_items):
    liked = [movie_titles.get(mid, "Unknown")[:30] for idx, r, mid in revealed_items if r >= 4][:10]
    disliked = [movie_titles.get(mid, "Unknown")[:30] for idx, r, mid in revealed_items if r < 4][:5]
    parts = []
    if liked:
        parts.append(f"likes: {', '.join(liked)}")
    if disliked:
        parts.append(f"dislikes: {', '.join(disliked)}")
    pref_text = ' | '.join(parts) if parts else "no preferences"
    user_state = encoder.encode([pref_text], show_progress_bar=False)
    user_state = torch.FloatTensor(user_state)
    with torch.no_grad():
        scores = model.get_all_scores(user_state, movie_embeddings)
        preds = torch.sigmoid(scores * 5).numpy().flatten()
    return preds


# =============================================================================
# EVALUATION 1: RL SIGNAL BY TIMESTEP (FIXED - SAME USERS FOR ALL TIMESTEPS)
# =============================================================================

print("\n" + "=" * 80)
print("EVALUATION 1: RL SIGNAL BY TIMESTEP (FIXED)")
print("(Same user group for all timesteps - use all available data)")
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

        if len(item_list) < 5:  # Need at least 5 ratings
            continue

        np.random.seed(user_id)
        np.random.shuffle(item_list)

        for n_prefs in timesteps:
            # USE ALL AVAILABLE DATA if user has fewer than n_prefs
            actual_n = min(n_prefs, len(item_list))
            revealed = item_list[:actual_n]
            revealed_indices = set(idx for idx, _, _ in revealed)

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

            # Full set metrics
            loss = calc_bce_loss(preds, ground_truth)
            acc = calc_accuracy(preds, ground_truth)
            ndcg = calc_ndcg(preds, ground_truth)

            if not np.isnan(loss):
                results[n_prefs]['loss'].append(loss)
            if not np.isnan(acc):
                results[n_prefs]['accuracy'].append(acc)
            if not np.isnan(ndcg):
                results[n_prefs]['ndcg'].append(ndcg)

            # Holdout metrics (exclude revealed items)
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

    print(f"\n{'Inputs':<8} | {'Loss':<8} | {'Acc':<8} | {'NDCG@10':<8} || {'H-Loss':<8} | {'H-Acc':<8} | {'N':<5}")
    print("-" * 80)

    summary = {}
    for t in timesteps:
        if results[t]['loss']:
            avg_loss = np.mean(results[t]['loss'])
            avg_acc = np.mean(results[t]['accuracy'])
            avg_ndcg = np.mean(results[t]['ndcg'])
            avg_h_loss = np.mean(results[t]['holdout_loss']) if results[t]['holdout_loss'] else np.nan
            avg_h_acc = np.mean(results[t]['holdout_acc']) if results[t]['holdout_acc'] else np.nan

            summary[t] = {
                'loss': avg_loss, 'accuracy': avg_acc, 'ndcg': avg_ndcg,
                'holdout_loss': avg_h_loss, 'holdout_acc': avg_h_acc
            }

            print(f"{t:<8} | {avg_loss:.4f}   | {avg_acc:.4f}   | {avg_ndcg:.4f}   || "
                  f"{avg_h_loss:.4f}   | {avg_h_acc:.4f}   | {len(results[t]['loss'])}")

    if 1 in summary and max(timesteps) in summary:
        loss_improvement = summary[1]['loss'] - summary[max(timesteps)]['loss']
        acc_improvement = summary[max(timesteps)]['accuracy'] - summary[1]['accuracy']
        print(f"\nRL Signal (1 -> {max(timesteps)} inputs):")
        print(f"  Loss reduction: {loss_improvement:+.4f}")
        print(f"  Accuracy gain:  {acc_improvement:+.4f}")

    results_by_model[model_name] = summary


# =============================================================================
# EVALUATION 2: PAPER FIGURE 5 - PAIRWISE CORRELATION TEST
# =============================================================================

print("\n" + "=" * 80)
print("EVALUATION 2: PAPER FIGURE 5 REPLICATION")
print("Predicting Episode IV rating based on SINGLE other movie input")
print("(Loss should be lower when Episode V or VI is input vs random movies)")
print("=" * 80)

if star_wars_movies and 'onehot' in models:
    # Find Episode IV, V, VI
    ep4 = next((m for m in star_wars_movies if "Episode IV" in m[2] or "A New Hope" in m[2]), None)
    ep5 = next((m for m in star_wars_movies if "Episode V" in m[2] or "Empire Strikes" in m[2]), None)
    ep6 = next((m for m in star_wars_movies if "Episode VI" in m[2] or "Return of the Jedi" in m[2]), None)

    if ep4 and ep5 and ep6:
        target_idx = ep4[1]  # Episode IV index
        target_title = ep4[2]

        print(f"\nTarget movie to predict: {target_title} (idx={target_idx})")

        # Get users who have rated Episode IV
        users_with_ep4 = []
        for user_id in EVAL_USERS:
            gt, items = get_user_data(user_id)
            if not np.isnan(gt[target_idx]):
                users_with_ep4.append((user_id, gt[target_idx], gt, items))

        print(f"Users who rated Episode IV: {len(users_with_ep4)}")

        # Test movies: Episode V, Episode VI, and some random movies
        test_inputs = [
            (ep5[1], ep5[2], "Episode V"),
            (ep6[1], ep6[2], "Episode VI"),
        ]
        # Add some random movies
        random_movies = [(0, movie_titles.get(idx_to_movie[0], ""), "Forrest Gump"),
                         (4, movie_titles.get(idx_to_movie[4], ""), "Matrix"),
                         (10, movie_titles.get(idx_to_movie[10], ""), "Movie 10"),
                         (30, movie_titles.get(idx_to_movie[30], ""), "Movie 30")]
        test_inputs.extend([(idx, "", name) for idx, _, name in random_movies])

        for model_name in ['onehot', 'concept']:
            if model_name not in models:
                continue

            print(f"\n--- {model_name.upper()} ---")
            print(f"{'Input Movie':<45} | {'Loss predicting EP IV':<20} | {'Pred':<8}")
            print("-" * 80)

            for input_idx, _, input_name in test_inputs:
                losses = []
                preds_list = []

                for user_id, ep4_true, gt, items in users_with_ep4:
                    # Check if user rated this input movie
                    user_rated_input = not np.isnan(gt[input_idx])
                    if not user_rated_input:
                        continue

                    # Use the user's actual rating for the input movie
                    input_rating = gt[input_idx]

                    # Predict using ONLY this one movie as input
                    revealed = [(input_idx, 5.0 if input_rating == 1 else 1.0, idx_to_movie[input_idx])]

                    if model_name == 'onehot':
                        preds = predict_onehot(models['onehot']['model'],
                                               models['onehot']['n_items'], revealed)
                    else:
                        preds = predict_concept(models['concept']['model'],
                                                models['concept']['embeddings'], revealed)

                    # Calculate loss ONLY on Episode IV
                    pred_ep4 = preds[target_idx]
                    loss_ep4 = -(ep4_true * np.log(max(pred_ep4, 1e-7)) +
                                 (1 - ep4_true) * np.log(max(1 - pred_ep4, 1e-7)))
                    losses.append(loss_ep4)
                    preds_list.append(pred_ep4)

                if losses:
                    avg_loss = np.mean(losses)
                    avg_pred = np.mean(preds_list)
                    marker = "*** SW ***" if "Episode" in input_name else ""
                    print(f"{input_name:<45} | {avg_loss:<20.4f} | {avg_pred:.4f}   {marker}")


# =============================================================================
# EVALUATION 3: PAPER TABLE I - ATTRIBUTE EFFECT (TOP CHANGED MOVIES)
# =============================================================================

print("\n" + "=" * 80)
print("EVALUATION 3: PAPER TABLE I REPLICATION")
print("Top movies with HIGHEST RATING CHANGE when input flips from LIKED to DISLIKED")
print("=" * 80)

# Find Russell Crowe in items (for concept model with attributes)
# First check if we have actor items
if 'onehot' in models and models['onehot']['items']:
    items = models['onehot']['items']
    n_movies_in_items = sum(1 for i in items if i[0] == 'movie')
    n_actors = sum(1 for i in items if i[0] == 'actor')

    print(f"\nItems breakdown: {n_movies_in_items} movies, {n_actors} actors")

    # Find Russell Crowe actor
    crowe_idx = None
    for i, item in enumerate(items):
        if item[0] == 'actor' and 'Crowe' in item[2]:
            crowe_idx = i
            print(f"Found Russell Crowe at index {i}: {item[2]}")
            break

    if crowe_idx:
        print(f"\n--- Testing Russell Crowe attribute effect ---")

        for model_name in ['onehot', 'concept']:
            if model_name not in models:
                continue

            print(f"\n{model_name.upper()}:")

            # Predict with Russell Crowe LIKED
            if model_name == 'onehot':
                preds_liked = predict_onehot_with_rating(
                    models['onehot']['model'], models['onehot']['n_items'],
                    crowe_idx, is_liked=True)
                preds_disliked = predict_onehot_with_rating(
                    models['onehot']['model'], models['onehot']['n_items'],
                    crowe_idx, is_liked=False)
            else:
                preds_liked = predict_concept_with_rating(
                    models['concept']['model'], models['concept']['embeddings'],
                    crowe_idx, is_liked=True)
                preds_disliked = predict_concept_with_rating(
                    models['concept']['model'], models['concept']['embeddings'],
                    crowe_idx, is_liked=False)

            # Calculate difference
            diff = preds_liked - preds_disliked

            # Top 10 most increased when Russell Crowe is LIKED
            top_increased = np.argsort(-diff)[:15]

            print(f"\nTop 15 movies MOST INCREASED when Russell Crowe is LIKED:")
            print(f"(Expected: Gladiator, A Beautiful Mind, L.A. Confidential)")
            print(f"{'Rank':<5} {'Movie':<50} | {'Liked':<8} | {'Disliked':<8} | {'Diff':<8}")
            print("-" * 85)

            for rank, idx in enumerate(top_increased, 1):
                if idx >= N_MOVIES:
                    continue  # Skip non-movie items
                mid = idx_to_movie.get(idx, -1)
                title = movie_titles.get(mid, "Unknown")
                # Check if Russell Crowe is in this movie
                is_crowe = any(t in title.lower() for t in ['gladiator', 'beautiful mind', 'l.a. confidential', 'insider'])
                marker = "*** RUSSELL CROWE MOVIE ***" if is_crowe else ""
                print(f"{rank:<5} {title[:48]:<50} | {preds_liked[idx]:.4f}   | {preds_disliked[idx]:.4f}   | {diff[idx]:+.4f}   {marker}")
                if rank >= 10:
                    break


# =============================================================================
# EVALUATION 4: TOP CHANGED MOVIES FOR STAR WARS
# =============================================================================

print("\n" + "=" * 80)
print("EVALUATION 4: TOP CHANGED MOVIES WHEN STAR WARS LIKED vs DISLIKED")
print("=" * 80)

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
        # DISLIKED input
        revealed_disliked = [(test_idx, 1.0, test_movie[0])]

        if model_name == 'onehot':
            preds_liked = predict_onehot(models['onehot']['model'],
                                         models['onehot']['n_items'], revealed_liked)
            preds_disliked = predict_onehot(models['onehot']['model'],
                                            models['onehot']['n_items'], revealed_disliked)
        elif model_name == 'concept':
            preds_liked = predict_concept(models['concept']['model'],
                                          models['concept']['embeddings'], revealed_liked)
            preds_disliked = predict_concept(models['concept']['model'],
                                             models['concept']['embeddings'], revealed_disliked)
        else:
            preds_liked = predict_twotower(models['twotower']['model'],
                                           models['twotower']['encoder'],
                                           models['twotower']['movie_embeddings'], revealed_liked)
            preds_disliked = predict_twotower(models['twotower']['model'],
                                              models['twotower']['encoder'],
                                              models['twotower']['movie_embeddings'], revealed_disliked)

        diff = preds_liked - preds_disliked

        # Top 10 most INCREASED (positive diff = higher when liked)
        top_increased = np.argsort(-diff)[:12]

        print(f"\nTop 10 movies MOST INCREASED when Star Wars EP IV is LIKED:")
        print(f"(Expected: Other Star Wars, Indiana Jones, sci-fi movies)")
        print(f"{'Rank':<5} {'Movie':<50} | {'Liked':<8} | {'Disliked':<8} | {'Diff':<8}")
        print("-" * 85)

        rank = 1
        for idx in top_increased:
            if idx == test_idx:
                continue
            mid = idx_to_movie.get(idx, -1)
            title = movie_titles.get(mid, "Unknown")
            is_sw = "*** STAR WARS ***" if "Star Wars" in title else ""
            print(f"{rank:<5} {title[:48]:<50} | {preds_liked[idx]:.4f}   | {preds_disliked[idx]:.4f}   | {diff[idx]:+.4f}   {is_sw}")
            rank += 1
            if rank > 10:
                break

        # Also show Star Wars movies specifically
        print(f"\nAll Star Wars movies:")
        for mid, idx, title in star_wars_movies:
            if idx != test_idx:
                print(f"  {title[:45]:<45} | Liked: {preds_liked[idx]:.4f} | Disliked: {preds_disliked[idx]:.4f} | Diff: {diff[idx]:+.4f}")


# =============================================================================
# SUMMARY
# =============================================================================

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

print("\n--- Model Status ---")
for name in ['onehot', 'concept', 'twotower']:
    if name in models:
        m = models[name]
        print(f"{name.upper():12} | Epoch: {m.get('epochs', m.get('epoch', 0)):3d}/100 | Val Loss: {m['val_loss']:.4f}")

print("\n--- RL Signal (Loss Reduction 1->50 inputs) ---")
for name in ['onehot', 'concept', 'twotower']:
    if name in results_by_model:
        r = results_by_model[name]
        if 1 in r and 50 in r:
            loss_red = r[1]['loss'] - r[50]['loss']
            acc_gain = r[50]['accuracy'] - r[1]['accuracy']
            print(f"{name.upper():<12} | Full Loss: {loss_red:+.4f} | Full Acc: {acc_gain:+.4f}")

print("\n" + "=" * 80)
print("EVALUATION COMPLETE")
print("=" * 80)
