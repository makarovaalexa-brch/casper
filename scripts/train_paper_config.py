"""
Train models with PAPER configuration for proper comparison.

Paper config (CSMAI-19):
- 100 movies (top most-rated)
- 50 actors (by popularity)
- 10-20 directors
- 20 genres
- ~130 total items
- 100 epochs
- Users with >=25% item ratings available
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
import pickle

# =============================================================================
# PAPER CONFIGURATION
# =============================================================================

DATA_DIR = Path('C:/dev/phd/casper/data/movielens')
CHECKPOINT_DIR = DATA_DIR / '.cache' / 'checkpoints'
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

# Match paper exactly
N_MOVIES = 100
N_ACTORS = 50
N_DIRECTORS = 20
N_GENRES = 20  # Will be whatever is in data
N_TAGS = 0  # Paper doesn't use genome tags

N_EPOCHS = 100
BATCH_SIZE = 32  # Paper uses 1 but that's very slow
LEARNING_RATE = 0.001

MIN_USER_RATINGS = 25  # Paper: "at least a quarter of ratings available" = 25 of 100 movies
SEED = 42

np.random.seed(SEED)
torch.manual_seed(SEED)

# =============================================================================
# LOAD DATA
# =============================================================================

print("=" * 80)
print("TRAINING WITH PAPER CONFIGURATION")
print("=" * 80)
print(f"Config: {N_MOVIES} movies, {N_ACTORS} actors, {N_DIRECTORS} directors")
print(f"Epochs: {N_EPOCHS}, Batch size: {BATCH_SIZE}")
print("=" * 80)

ratings = pd.read_csv(DATA_DIR / 'ratings.csv')
movies_df = pd.read_csv(DATA_DIR / 'movies.csv')

# Get top N_MOVIES
movie_counts = ratings['movieId'].value_counts()
top_movies = movie_counts.head(N_MOVIES).index.tolist()
movie_titles = {row['movieId']: row['title'] for _, row in movies_df.iterrows()}

print(f"\nTop {N_MOVIES} movies selected (by rating count)")

# =============================================================================
# LOAD ATTRIBUTES (Actors, Directors from TMDB cache)
# =============================================================================

import json

credits_cache_path = DATA_DIR / '.cache' / 'credits_top100_actors5.json'

credits_data = {}
if credits_cache_path.exists():
    print(f"Loading credits from cache...")
    with open(credits_cache_path, 'r') as f:
        credits_data = json.load(f)
else:
    print("Credits cache not found.")

# Count actor/director appearances in top movies
actor_counts = {}
director_counts = {}

if 'movie_actors' in credits_data:
    for movie_id_str, actors in credits_data['movie_actors'].items():
        movie_id = int(movie_id_str)
        if movie_id in top_movies:
            for actor in actors[:5]:  # Top 5 actors per movie
                actor_counts[actor] = actor_counts.get(actor, 0) + 1

if 'movie_directors' in credits_data:
    for movie_id_str, directors in credits_data['movie_directors'].items():
        movie_id = int(movie_id_str)
        if movie_id in top_movies:
            for director in directors:
                director_counts[director] = director_counts.get(director, 0) + 1

# Get top actors/directors
top_actors = sorted(actor_counts.keys(), key=lambda x: actor_counts[x], reverse=True)[:N_ACTORS]
top_directors = sorted(director_counts.keys(), key=lambda x: director_counts[x], reverse=True)[:N_DIRECTORS]

print(f"Loaded {len(top_actors)} actors, {len(top_directors)} directors from credits cache")

# =============================================================================
# EXTRACT GENRES
# =============================================================================

all_genres = set()
for _, row in movies_df[movies_df['movieId'].isin(top_movies)].iterrows():
    if pd.notna(row['genres']):
        for genre in row['genres'].split('|'):
            all_genres.add(genre)

genres = sorted(list(all_genres))[:N_GENRES]
print(f"Genres: {len(genres)}")

# =============================================================================
# BUILD ITEM INDEX
# =============================================================================

# Items: movies + genres + actors + directors
items = []

# Movies first (indices 0 to N_MOVIES-1)
for movie_id in top_movies:
    items.append(('movie', movie_id, movie_titles.get(movie_id, f'Movie {movie_id}')))

# Genres (indices N_MOVIES to N_MOVIES+len(genres)-1)
for genre in genres:
    items.append(('genre', genre, genre))

# Actors
for actor in top_actors:
    items.append(('actor', actor, actor))

# Directors
for director in top_directors:
    items.append(('director', director, director))

N_ITEMS = len(items)
print(f"\nTotal items: {N_ITEMS} ({N_MOVIES} movies + {len(genres)} genres + {len(top_actors)} actors + {len(top_directors)} directors)")

# Create index mappings
item_to_idx = {(item[0], item[1]): i for i, item in enumerate(items)}
movie_to_idx = {item[1]: i for i, item in enumerate(items) if item[0] == 'movie'}

# =============================================================================
# BUILD ATTRIBUTE RATINGS (derive from movie ratings)
# =============================================================================

print("\nBuilding attribute ratings from movie ratings...")

# Filter to top movies
ratings_filtered = ratings[ratings['movieId'].isin(top_movies)].copy()
ratings_filtered['itemIdx'] = ratings_filtered['movieId'].map(movie_to_idx)

# Get users with at least 25 ratings on top 100 movies (paper: "quarter of ratings available")
user_movie_counts = ratings_filtered.groupby('userId').size()
dense_users = user_movie_counts[user_movie_counts >= MIN_USER_RATINGS].index.tolist()
print(f"Users with >= {MIN_USER_RATINGS} ratings on top {N_MOVIES} movies: {len(dense_users)}")

# For each user, derive attribute ratings from their movie ratings
print("Deriving attribute ratings...")

def get_movie_attributes(movie_id):
    """Get genres, actors, directors for a movie."""
    attrs = []

    # Genres
    movie_row = movies_df[movies_df['movieId'] == movie_id]
    if len(movie_row) > 0 and pd.notna(movie_row.iloc[0]['genres']):
        for genre in movie_row.iloc[0]['genres'].split('|'):
            if genre in genres:
                attrs.append(('genre', genre))

    # Actors/Directors from credits cache
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

# Build attribute ratings for each user
user_attr_ratings = {}  # user_id -> {(type, name): [ratings]}

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

    # Average ratings for each attribute
    user_attr_ratings[user_id] = {attr: np.mean(ratings) for attr, ratings in attr_ratings.items()}

# =============================================================================
# CREATE DATASET
# =============================================================================

print("\nCreating training dataset...")

# For each user, create (item_indices, rating_one_hot, ground_truth)
def create_user_data(user_id):
    """Create training data for a user."""
    user_movies = ratings_filtered[ratings_filtered['userId'] == user_id]

    # All item ratings (NaN for unseen)
    all_ratings = np.full(N_ITEMS, np.nan)

    # Movie ratings
    for _, row in user_movies.iterrows():
        idx = movie_to_idx.get(row['movieId'])
        if idx is not None:
            all_ratings[idx] = 1.0 if row['rating'] >= 4 else 0.0

    # Attribute ratings
    if user_id in user_attr_ratings:
        for attr, avg_rating in user_attr_ratings[user_id].items():
            idx = item_to_idx.get(attr)
            if idx is not None:
                all_ratings[idx] = 1.0 if avg_rating >= 4 else 0.0

    return all_ratings

# Get users with enough ratings
active_users = []
for user_id in tqdm(list(user_attr_ratings.keys()), desc="Filtering users"):
    user_data = create_user_data(user_id)
    n_rated = np.sum(~np.isnan(user_data))
    if n_rated >= 30:  # At least 30 items rated
        active_users.append(user_id)

print(f"Active users with >=30 item ratings: {len(active_users)}")

# Split train/val
np.random.shuffle(active_users)
train_users = active_users[:int(0.8 * len(active_users))]
val_users = active_users[int(0.8 * len(active_users)):]

print(f"Train users: {len(train_users)}, Val users: {len(val_users)}")

# =============================================================================
# MODEL (Same architecture as paper)
# =============================================================================

class ExtrapolationModel(nn.Module):
    """Exact copy of paper's ExtrapolationModel."""

    def __init__(self, n_items, lr=0.001):
        super(ExtrapolationModel, self).__init__()
        self.n_items = n_items
        hidden_dim = n_items // 2

        self.embedding = nn.Embedding(n_items + 1, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim + 3, hidden_dim, 1, batch_first=True, bidirectional=False)
        self.dense1 = nn.Linear(hidden_dim, n_items)
        self.dense2 = nn.Linear(n_items, hidden_dim)
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=1, batch_first=True)
        # Output takes concatenated x and attention: 2 * hidden_dim
        self.output = nn.Linear(hidden_dim * 2, n_items)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)

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


class BCEWithLogitsLossNan(nn.Module):
    """BCE loss that masks NaN outputs."""
    def forward(self, y_pred, y_true):
        mask = torch.isnan(y_true)
        y_pred = torch.where(mask, torch.zeros_like(y_true), y_pred)
        y_true = torch.where(mask, torch.zeros_like(y_true), y_true)
        return nn.functional.binary_cross_entropy_with_logits(y_pred, y_true)


# =============================================================================
# DATASET CLASS
# =============================================================================

class UserDataset(torch.utils.data.Dataset):
    def __init__(self, user_ids, n_items, create_user_data_fn):
        self.user_ids = user_ids
        self.n_items = n_items
        self.create_user_data = create_user_data_fn

    def __len__(self):
        return len(self.user_ids)

    def __getitem__(self, idx):
        user_id = self.user_ids[idx]
        full_ratings = self.create_user_data(user_id)

        # Shuffle item order
        indices = np.random.permutation(self.n_items)

        # Create rating input (one-hot: [disliked, liked, not_seen])
        rating_input = np.zeros((self.n_items, 3))
        for i, item_idx in enumerate(indices):
            if np.isnan(full_ratings[item_idx]):
                rating_input[i, 2] = 1  # not_seen
            elif full_ratings[item_idx] == 0:
                rating_input[i, 0] = 1  # disliked
            else:
                rating_input[i, 1] = 1  # liked

        # Output: ratings tiled for each timestep
        output = np.tile(full_ratings, (self.n_items, 1))

        return (
            torch.LongTensor(indices),
            torch.FloatTensor(rating_input),
            torch.FloatTensor(output)
        )


# =============================================================================
# TRAINING
# =============================================================================

print("\n" + "=" * 80)
print("TRAINING")
print("=" * 80)

model = ExtrapolationModel(N_ITEMS, lr=LEARNING_RATE)
loss_fn = BCEWithLogitsLossNan()

train_dataset = UserDataset(train_users, N_ITEMS, create_user_data)
val_dataset = UserDataset(val_users, N_ITEMS, create_user_data)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"Training batches: {len(train_loader)}, Val batches: {len(val_loader)}")

best_val_loss = float('inf')
start_time = time.time()

for epoch in range(N_EPOCHS):
    # Training
    model.train()
    train_loss = 0
    for indices, ratings, targets in train_loader:
        model.optimizer.zero_grad()
        outputs = model(indices, ratings)
        loss = loss_fn(outputs, targets)
        loss.backward()
        model.optimizer.step()
        train_loss += loss.item()
    train_loss /= len(train_loader)

    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for indices, ratings, targets in val_loader:
            outputs = model(indices, ratings)
            loss = loss_fn(outputs, targets)
            val_loss += loss.item()
    val_loss /= len(val_loader)

    # Track best
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_marker = " *"

        # Save checkpoint
        torch.save({
            'model_state_dict': model.state_dict(),
            'n_items': N_ITEMS,
            'n_movies': N_MOVIES,
            'val_loss': val_loss,
            'epochs': epoch + 1,
            'items': items,
            'config': {
                'n_movies': N_MOVIES,
                'n_actors': len(top_actors),
                'n_directors': len(top_directors),
                'n_genres': len(genres),
            }
        }, CHECKPOINT_DIR / 'onehot_paper_config.pt')
    else:
        best_marker = ""

    elapsed = time.time() - start_time
    if (epoch + 1) % 10 == 0 or epoch == 0:
        print(f"Epoch {epoch+1:3d}/{N_EPOCHS} | Train: {train_loss:.4f} | Val: {val_loss:.4f}{best_marker} | Time: {elapsed:.1f}s")

print(f"\nTraining complete in {time.time() - start_time:.1f}s")
print(f"Best val loss: {best_val_loss:.4f}")
print(f"Checkpoint saved: {CHECKPOINT_DIR / 'onehot_paper_config.pt'}")

# =============================================================================
# EVALUATION: Loss by Timestep (Paper Fig 2b)
# =============================================================================

print("\n" + "=" * 80)
print("EVALUATION: Loss by Timestep (Paper Fig 2b)")
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

    # Get rated item indices (shuffled)
    rated_indices = np.where(~np.isnan(user_data))[0]
    np.random.seed(user_id)
    np.random.shuffle(rated_indices)

    for n_revealed in losses_by_timestep.keys():
        if n_revealed > len(rated_indices):
            continue

        # Build input with n_revealed ratings
        indices = torch.arange(N_ITEMS).unsqueeze(0)
        rating_input = torch.zeros(1, N_ITEMS, 3)
        rating_input[:, :, 2] = 1  # All unknown

        for i in range(n_revealed):
            idx = rated_indices[i]
            rating_input[:, idx, 2] = 0
            if user_data[idx] == 1:
                rating_input[:, idx, 1] = 1  # liked
            else:
                rating_input[:, idx, 0] = 1  # disliked

        # Predict
        with torch.no_grad():
            preds = model(indices, rating_input)[:, -1, :].sigmoid().numpy().flatten()

        # Loss on ALL rated items
        loss = calculate_loss(preds, user_data)
        if not np.isnan(loss):
            losses_by_timestep[n_revealed].append(loss)

        # Accuracy
        filt = ~np.isnan(user_data)
        pred_binary = (preds[filt] > 0.5).astype(float)
        acc = np.mean(pred_binary == user_data[filt])
        accuracy_by_timestep[n_revealed].append(acc)

print(f"\n{'Timestep':<12} {'Loss':<15} {'Accuracy':<15} {'N users'}")
print("-" * 55)
for t in sorted(losses_by_timestep.keys()):
    if losses_by_timestep[t]:
        avg_loss = np.mean(losses_by_timestep[t])
        avg_acc = np.mean(accuracy_by_timestep[t])
        print(f"{t:<12} {avg_loss:<15.4f} {avg_acc:<15.4f} {len(losses_by_timestep[t])}")

# Calculate improvement
t_min = min(losses_by_timestep.keys())
t_max = max(t for t in losses_by_timestep.keys() if losses_by_timestep[t])
if losses_by_timestep[t_min] and losses_by_timestep[t_max]:
    loss_improvement = np.mean(losses_by_timestep[t_min]) - np.mean(losses_by_timestep[t_max])
    acc_improvement = np.mean(accuracy_by_timestep[t_max]) - np.mean(accuracy_by_timestep[t_min])
    print(f"\nRL Signal:")
    print(f"  Loss improvement ({t_min} -> {t_max}): {loss_improvement:+.4f}")
    print(f"  Accuracy improvement: {acc_improvement:+.4f}")
