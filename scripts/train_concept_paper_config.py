"""
Train CONCEPT EMBEDDINGS model with PAPER configuration.

Same setup as One-Hot but with SBERT embeddings instead of learned embeddings.
"""

import sys
sys.path.insert(0, '.')

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from tqdm import tqdm
import time
import json
from sentence_transformers import SentenceTransformer

# =============================================================================
# PAPER CONFIGURATION
# =============================================================================

DATA_DIR = Path('C:/dev/phd/casper/data/movielens')
CHECKPOINT_DIR = DATA_DIR / '.cache' / 'checkpoints'
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

N_MOVIES = 100
N_ACTORS = 50
N_DIRECTORS = 20
N_GENRES = 20

N_EPOCHS = 100
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'
EMBEDDING_DIM = 384

MIN_USER_RATINGS = 25  # Paper: "at least a quarter of ratings available"
SEED = 42

np.random.seed(SEED)
torch.manual_seed(SEED)

# =============================================================================
# LOAD DATA (same as one-hot)
# =============================================================================

print("=" * 80)
print("TRAINING CONCEPT EMBEDDINGS WITH PAPER CONFIGURATION")
print("=" * 80)
print(f"Config: {N_MOVIES} movies, {N_ACTORS} actors, {N_DIRECTORS} directors")
print(f"Embedding model: {EMBEDDING_MODEL}")
print(f"Epochs: {N_EPOCHS}")
print("=" * 80)

ratings = pd.read_csv(DATA_DIR / 'ratings.csv')
movies_df = pd.read_csv(DATA_DIR / 'movies.csv')

movie_counts = ratings['movieId'].value_counts()
top_movies = movie_counts.head(N_MOVIES).index.tolist()
movie_titles = {row['movieId']: row['title'] for _, row in movies_df.iterrows()}

credits_cache_path = DATA_DIR / '.cache' / 'credits_top100_actors5.json'
credits_data = {}
if credits_cache_path.exists():
    print(f"Loading credits from cache...")
    with open(credits_cache_path, 'r') as f:
        credits_data = json.load(f)

# Count actor/director appearances
actor_counts = {}
director_counts = {}

if 'movie_actors' in credits_data:
    for movie_id_str, actors in credits_data['movie_actors'].items():
        movie_id = int(movie_id_str)
        if movie_id in top_movies:
            for actor in actors[:5]:
                actor_counts[actor] = actor_counts.get(actor, 0) + 1

if 'movie_directors' in credits_data:
    for movie_id_str, directors in credits_data['movie_directors'].items():
        movie_id = int(movie_id_str)
        if movie_id in top_movies:
            for director in directors:
                director_counts[director] = director_counts.get(director, 0) + 1

top_actors = sorted(actor_counts.keys(), key=lambda x: actor_counts[x], reverse=True)[:N_ACTORS]
top_directors = sorted(director_counts.keys(), key=lambda x: director_counts[x], reverse=True)[:N_DIRECTORS]

# Extract genres
all_genres = set()
for _, row in movies_df[movies_df['movieId'].isin(top_movies)].iterrows():
    if pd.notna(row['genres']):
        for genre in row['genres'].split('|'):
            all_genres.add(genre)
genres = sorted(list(all_genres))[:N_GENRES]

# Build item list
items = []
for movie_id in top_movies:
    items.append(('movie', movie_id, movie_titles.get(movie_id, f'Movie {movie_id}')))
for genre in genres:
    items.append(('genre', genre, genre))
for actor in top_actors:
    items.append(('actor', actor, actor))
for director in top_directors:
    items.append(('director', director, director))

N_ITEMS = len(items)
print(f"\nTotal items: {N_ITEMS}")

item_to_idx = {(item[0], item[1]): i for i, item in enumerate(items)}
movie_to_idx = {item[1]: i for i, item in enumerate(items) if item[0] == 'movie'}

# =============================================================================
# GENERATE SBERT EMBEDDINGS
# =============================================================================

print("\nGenerating SBERT embeddings...")
encoder = SentenceTransformer(EMBEDDING_MODEL)

item_descriptions = []
for item_type, item_id, item_name in items:
    if item_type == 'movie':
        item_descriptions.append(f"Movie: {item_name}")
    elif item_type == 'genre':
        item_descriptions.append(f"Genre: {item_name}")
    elif item_type == 'actor':
        item_descriptions.append(f"Actor: {item_name}")
    elif item_type == 'director':
        item_descriptions.append(f"Director: {item_name}")

item_embeddings = encoder.encode(item_descriptions, show_progress_bar=True)
print(f"Embedding shape: {item_embeddings.shape}")

# Save embeddings
np.save(CHECKPOINT_DIR / 'concept_embeddings_paper_config.npy', item_embeddings)

# =============================================================================
# BUILD DATASET (same as one-hot)
# =============================================================================

ratings_filtered = ratings[ratings['movieId'].isin(top_movies)].copy()
ratings_filtered['itemIdx'] = ratings_filtered['movieId'].map(movie_to_idx)

# Get users with at least 25 ratings on top 100 movies (paper criteria)
user_movie_counts = ratings_filtered.groupby('userId').size()
dense_users = user_movie_counts[user_movie_counts >= MIN_USER_RATINGS].index.tolist()
print(f"Users with >= {MIN_USER_RATINGS} ratings on top {N_MOVIES} movies: {len(dense_users)}")


def get_movie_attributes(movie_id):
    attrs = []
    movie_row = movies_df[movies_df['movieId'] == movie_id]
    if len(movie_row) > 0 and pd.notna(movie_row.iloc[0]['genres']):
        for genre in movie_row.iloc[0]['genres'].split('|'):
            if genre in genres:
                attrs.append(('genre', genre))
    movie_id_str = str(movie_id)
    if 'movie_actors' in credits_data and movie_id_str in credits_data['movie_actors']:
        for actor in credits_data['movie_actors'][movie_id_str][:5]:
            if actor in top_actors:
                attrs.append(('actor', actor))
    if 'movie_directors' in credits_data and movie_id_str in credits_data['movie_directors']:
        for director in credits_data['movie_directors'][movie_id_str]:
            if director in top_directors:
                attrs.append(('director', director))
    return attrs


print("Building attribute ratings...")
user_attr_ratings = {}

for user_id in tqdm(dense_users, desc="Building attribute ratings"):
    user_movies = ratings_filtered[ratings_filtered['userId'] == user_id]
    attr_ratings = {}
    for _, row in user_movies.iterrows():
        movie_id = row['movieId']
        rating = row['rating']
        for attr in get_movie_attributes(movie_id):
            if attr not in attr_ratings:
                attr_ratings[attr] = []
            attr_ratings[attr].append(rating)
    user_attr_ratings[user_id] = {attr: np.mean(ratings) for attr, ratings in attr_ratings.items()}


def create_user_data(user_id):
    user_movies = ratings_filtered[ratings_filtered['userId'] == user_id]
    all_ratings = np.full(N_ITEMS, np.nan)
    for _, row in user_movies.iterrows():
        idx = movie_to_idx.get(row['movieId'])
        if idx is not None:
            all_ratings[idx] = 1.0 if row['rating'] >= 4 else 0.0
    if user_id in user_attr_ratings:
        for attr, avg_rating in user_attr_ratings[user_id].items():
            idx = item_to_idx.get(attr)
            if idx is not None:
                all_ratings[idx] = 1.0 if avg_rating >= 4 else 0.0
    return all_ratings


active_users = []
for user_id in tqdm(list(user_attr_ratings.keys()), desc="Filtering users"):
    user_data = create_user_data(user_id)
    n_rated = np.sum(~np.isnan(user_data))
    if n_rated >= 30:
        active_users.append(user_id)

print(f"Active users: {len(active_users)}")

np.random.shuffle(active_users)
train_users = active_users[:int(0.8 * len(active_users))]
val_users = active_users[int(0.8 * len(active_users)):]
print(f"Train: {len(train_users)}, Val: {len(val_users)}")

# =============================================================================
# MODEL (Concept Embeddings)
# =============================================================================

class ConceptEmbeddingModel(nn.Module):
    """Model using SBERT embeddings instead of learned embeddings."""

    def __init__(self, n_items, embedding_dim=384, hidden_dim=None, lr=0.001):
        super(ConceptEmbeddingModel, self).__init__()
        self.n_items = n_items
        hidden_dim = hidden_dim or n_items // 2

        # Project SBERT embeddings to hidden dim
        self.concept_proj = nn.Linear(embedding_dim, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim + 3, hidden_dim, 1, batch_first=True)
        self.dense1 = nn.Linear(hidden_dim, n_items)
        self.dense2 = nn.Linear(n_items, hidden_dim)
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=1, batch_first=True)
        # Output takes concatenated x and attention: 2 * hidden_dim
        self.output = nn.Linear(hidden_dim * 2, n_items)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    def forward(self, item_embeddings, rating_input):
        # item_embeddings: [batch, n_items, embedding_dim]
        # rating_input: [batch, n_items, 3]
        x = self.concept_proj(item_embeddings)
        x = torch.cat((x, rating_input), dim=-1)
        encoder_output, _ = self.lstm(x)
        x = torch.relu(self.dense1(encoder_output))
        x = torch.relu(self.dense2(x))
        attention, _ = self.attention(encoder_output, x, x)
        x = torch.cat((x, attention), dim=-1)
        return self.output(x)


class BCEWithLogitsLossNan(nn.Module):
    def forward(self, y_pred, y_true):
        mask = torch.isnan(y_true)
        y_pred = torch.where(mask, torch.zeros_like(y_true), y_pred)
        y_true = torch.where(mask, torch.zeros_like(y_true), y_true)
        return nn.functional.binary_cross_entropy_with_logits(y_pred, y_true)


# =============================================================================
# DATASET
# =============================================================================

item_emb_tensor = torch.FloatTensor(item_embeddings)


class ConceptDataset(torch.utils.data.Dataset):
    def __init__(self, user_ids, n_items, create_user_data_fn, item_embeddings):
        self.user_ids = user_ids
        self.n_items = n_items
        self.create_user_data = create_user_data_fn
        self.item_embeddings = item_embeddings

    def __len__(self):
        return len(self.user_ids)

    def __getitem__(self, idx):
        user_id = self.user_ids[idx]
        full_ratings = self.create_user_data(user_id)

        # For concept model, we keep the embeddings in order (no shuffling needed due to attention)
        # But shuffle for consistency with paper approach
        indices = np.random.permutation(self.n_items)

        # Reorder embeddings
        shuffled_embeddings = self.item_embeddings[indices]

        rating_input = np.zeros((self.n_items, 3))
        for i, item_idx in enumerate(indices):
            if np.isnan(full_ratings[item_idx]):
                rating_input[i, 2] = 1
            elif full_ratings[item_idx] == 0:
                rating_input[i, 0] = 1
            else:
                rating_input[i, 1] = 1

        # Output needs to be reordered to match shuffled input
        output = np.tile(full_ratings, (self.n_items, 1))
        # Reorder output columns to match shuffled input
        output = output[:, indices]

        return (
            torch.FloatTensor(shuffled_embeddings),
            torch.FloatTensor(rating_input),
            torch.FloatTensor(output),
            torch.LongTensor(indices)  # Keep track of original indices
        )


# =============================================================================
# TRAINING
# =============================================================================

print("\n" + "=" * 80)
print("TRAINING")
print("=" * 80)

model = ConceptEmbeddingModel(N_ITEMS, EMBEDDING_DIM, lr=LEARNING_RATE)
loss_fn = BCEWithLogitsLossNan()

train_dataset = ConceptDataset(train_users, N_ITEMS, create_user_data, item_embeddings)
val_dataset = ConceptDataset(val_users, N_ITEMS, create_user_data, item_embeddings)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

best_val_loss = float('inf')
start_time = time.time()

for epoch in range(N_EPOCHS):
    model.train()
    train_loss = 0
    for embeddings, ratings, targets, indices in train_loader:
        model.optimizer.zero_grad()
        outputs = model(embeddings, ratings)
        loss = loss_fn(outputs, targets)
        loss.backward()
        model.optimizer.step()
        train_loss += loss.item()
    train_loss /= len(train_loader)

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for embeddings, ratings, targets, indices in val_loader:
            outputs = model(embeddings, ratings)
            loss = loss_fn(outputs, targets)
            val_loss += loss.item()
    val_loss /= len(val_loader)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_marker = " *"
        torch.save({
            'model_state_dict': model.state_dict(),
            'n_items': N_ITEMS,
            'n_movies': N_MOVIES,
            'embedding_dim': EMBEDDING_DIM,
            'val_loss': val_loss,
            'epochs': epoch + 1,
            'items': items,
        }, CHECKPOINT_DIR / 'concept_paper_config.pt')
    else:
        best_marker = ""

    elapsed = time.time() - start_time
    if (epoch + 1) % 10 == 0 or epoch == 0:
        print(f"Epoch {epoch+1:3d}/{N_EPOCHS} | Train: {train_loss:.4f} | Val: {val_loss:.4f}{best_marker} | Time: {elapsed:.1f}s")

print(f"\nTraining complete in {time.time() - start_time:.1f}s")
print(f"Best val loss: {best_val_loss:.4f}")
print(f"Checkpoint: {CHECKPOINT_DIR / 'concept_paper_config.pt'}")

# =============================================================================
# EVALUATION
# =============================================================================

print("\n" + "=" * 80)
print("EVALUATION: Loss by Timestep")
print("=" * 80)

model.eval()
from sklearn.metrics import log_loss


def calculate_loss(predictions, ground_truth):
    filt = ~np.isnan(ground_truth)
    if filt.sum() == 0:
        return np.nan
    preds = np.clip(predictions[filt], 1e-7, 1 - 1e-7)
    gt = ground_truth[filt]
    return log_loss(gt, preds, labels=[0, 1])


timesteps = [1, 3, 5, 10, 20, 30, 50, 70, 100]
losses_by_timestep = {t: [] for t in timesteps if t <= N_ITEMS}
accuracy_by_timestep = {t: [] for t in timesteps if t <= N_ITEMS}

for user_id in tqdm(val_users[:100], desc="Evaluating"):
    user_data = create_user_data(user_id)
    n_rated = np.sum(~np.isnan(user_data))
    if n_rated < 30:
        continue

    rated_indices = np.where(~np.isnan(user_data))[0]
    np.random.seed(user_id)
    np.random.shuffle(rated_indices)

    for n_revealed in losses_by_timestep.keys():
        if n_revealed > len(rated_indices):
            continue

        # Build input
        emb_input = torch.FloatTensor(item_embeddings).unsqueeze(0)
        rating_input = torch.zeros(1, N_ITEMS, 3)
        rating_input[:, :, 2] = 1

        for i in range(n_revealed):
            idx = rated_indices[i]
            rating_input[:, idx, 2] = 0
            if user_data[idx] == 1:
                rating_input[:, idx, 1] = 1
            else:
                rating_input[:, idx, 0] = 1

        with torch.no_grad():
            preds = model(emb_input, rating_input)[:, -1, :].sigmoid().numpy().flatten()

        loss = calculate_loss(preds, user_data)
        if not np.isnan(loss):
            losses_by_timestep[n_revealed].append(loss)

        filt = ~np.isnan(user_data)
        pred_binary = (preds[filt] > 0.5).astype(float)
        acc = np.mean(pred_binary == user_data[filt])
        accuracy_by_timestep[n_revealed].append(acc)

print(f"\n{'Timestep':<12} {'Loss':<15} {'Accuracy':<15}")
print("-" * 45)
for t in sorted(losses_by_timestep.keys()):
    if losses_by_timestep[t]:
        print(f"{t:<12} {np.mean(losses_by_timestep[t]):<15.4f} {np.mean(accuracy_by_timestep[t]):<15.4f}")

t_min = min(losses_by_timestep.keys())
t_max = max(t for t in losses_by_timestep.keys() if losses_by_timestep[t])
if losses_by_timestep[t_min] and losses_by_timestep[t_max]:
    loss_improvement = np.mean(losses_by_timestep[t_min]) - np.mean(losses_by_timestep[t_max])
    acc_improvement = np.mean(accuracy_by_timestep[t_max]) - np.mean(accuracy_by_timestep[t_min])
    print(f"\nRL Signal:")
    print(f"  Loss improvement ({t_min} -> {t_max}): {loss_improvement:+.4f}")
    print(f"  Accuracy improvement: {acc_improvement:+.4f}")
