"""
Two-Tower V2 Recommender Training - Separate Like/Dislike Encodings.

Key difference from V1:
- V1: "likes: Movie1, Movie2 | dislikes: Movie3" -> single SBERT encoding -> user tower
- V2: likes and dislikes encoded SEPARATELY, then projected and concatenated

Hypothesis: V1 loses polarity information when SBERT encodes mixed text.
            V2 preserves like/dislike signal by keeping them separate.
"""

import sys
sys.path.insert(0, '../src')

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import json
import random

# =============================================================================
# CONFIGURATION (Match paper exactly)
# =============================================================================
N_MOVIES = 100
N_ACTORS = 50
N_DIRECTORS = 20
N_GENRES = 20
N_EPOCHS = 100
BATCH_SIZE = 64
LEARNING_RATE = 0.0003
NUM_NEGATIVES = 16

DATA_DIR = Path('../data/movielens')
CHECKPOINT_DIR = DATA_DIR / '.cache' / 'checkpoints'
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("TRAINING TWO-TOWER V2 - SEPARATE LIKE/DISLIKE ENCODINGS")
print("=" * 80)
print(f"Config: {N_MOVIES} movies, {N_ACTORS} actors, {N_DIRECTORS} directors")
print(f"Epochs: {N_EPOCHS}, Batch size: {BATCH_SIZE}")
print("Key change: Likes and dislikes encoded separately, then concatenated")
print("=" * 80)

# =============================================================================
# LOAD DATA (same as other models)
# =============================================================================

# Movies
movies = pd.read_csv(DATA_DIR / 'movies.csv')
ratings = pd.read_csv(DATA_DIR / 'ratings.csv')

# Rating counts per movie
rating_counts = ratings.groupby('movieId').size().reset_index(name='count')
movies = movies.merge(rating_counts, on='movieId', how='left')
movies['count'] = movies['count'].fillna(0).astype(int)

# Top 100 movies by rating count
top_movies = movies.nlargest(N_MOVIES, 'count')
top_movie_ids = set(top_movies['movieId'].tolist())
print(f"Top {N_MOVIES} movies selected (by rating count)")

# Create movie ID to index mapping
movie_id_to_idx = {mid: idx for idx, mid in enumerate(top_movies['movieId'])}
idx_to_movie_id = {idx: mid for mid, idx in movie_id_to_idx.items()}

# Get movie titles for creating text
movie_titles = {row['movieId']: row['title'] for _, row in top_movies.iterrows()}

# Genres
all_genres = set()
for genres_str in movies['genres'].dropna():
    all_genres.update(genres_str.split('|'))
all_genres.discard('(no genres listed)')
all_genres = sorted(list(all_genres))[:N_GENRES]
print(f"Genres: {len(all_genres)}")

# Load credits
credits_cache_path = DATA_DIR / '.cache' / 'credits_top100_actors5.json'
if credits_cache_path.exists():
    print("Loading credits from cache...")
    with open(credits_cache_path, 'r') as f:
        credits_data = json.load(f)

    actor_counts = credits_data.get('actor_counts', {})
    director_counts = credits_data.get('director_counts', {})
    movie_actors = credits_data.get('movie_actors', {})
    movie_directors = credits_data.get('movie_directors', {})

    top_actors = sorted(actor_counts.keys(), key=lambda x: actor_counts[x], reverse=True)[:N_ACTORS]
    top_directors = sorted(director_counts.keys(), key=lambda x: director_counts[x], reverse=True)[:N_DIRECTORS]
    print(f"Loaded {len(top_actors)} actors, {len(top_directors)} directors from credits cache")
else:
    print("Credits cache not found. Using genres only.")
    top_actors = []
    top_directors = []
    movie_actors = {}
    movie_directors = {}

print(f"\nTotal: {N_MOVIES} movies + {len(all_genres)} genres + {len(top_actors)} actors + {len(top_directors)} directors")

# =============================================================================
# PREPARE USER DATA
# =============================================================================

# Filter ratings to top movies
filtered_ratings = ratings[ratings['movieId'].isin(top_movie_ids)].copy()
filtered_ratings['liked'] = filtered_ratings['rating'] >= 4.0

print(f"\nFiltered ratings: {len(filtered_ratings)}")

# Users with at least 25 ratings on top 100 movies
user_rating_counts = filtered_ratings.groupby('userId').size()
active_users = user_rating_counts[user_rating_counts >= 25].index.tolist()
print(f"Active users (>=25 ratings on top {N_MOVIES} movies): {len(active_users)}")

# Build user preference data
print("\nBuilding user preference data...")
user_data = {}
for user_id in tqdm(active_users, desc="Processing users"):
    user_ratings = filtered_ratings[filtered_ratings['userId'] == user_id]
    liked = user_ratings[user_ratings['liked']]['movieId'].tolist()
    disliked = user_ratings[~user_ratings['liked']]['movieId'].tolist()
    if len(liked) >= 5:  # Need some likes to train on
        user_data[user_id] = {
            'liked': liked,
            'disliked': disliked
        }

print(f"Users with sufficient preferences: {len(user_data)}")

# Train/val split (use fixed seed for reproducibility with V1)
random.seed(42)
user_ids = list(user_data.keys())
random.shuffle(user_ids)
train_users = user_ids[:int(0.8 * len(user_ids))]
val_users = user_ids[int(0.8 * len(user_ids)):]
print(f"Train: {len(train_users)}, Val: {len(val_users)}")

# =============================================================================
# ENCODE MOVIES WITH SBERT
# =============================================================================

print("\nLoading SentenceBERT encoder...")
encoder = SentenceTransformer('all-MiniLM-L6-v2')

# Create movie text descriptions
movie_texts = []
for movie_id in top_movies['movieId']:
    row = top_movies[top_movies['movieId'] == movie_id].iloc[0]
    title = row['title'].split('(')[0].strip()
    genres = row['genres'].replace('|', ', ')
    text = f"{title}: {genres}"
    movie_texts.append(text)

print(f"Encoding {len(movie_texts)} movies...")
movie_embeddings = encoder.encode(movie_texts, convert_to_numpy=True, show_progress_bar=True)
movie_embeddings_tensor = torch.FloatTensor(movie_embeddings)
print(f"Movie embedding shape: {movie_embeddings.shape}")

# Pre-encode "no preferences" placeholder
no_pref_embedding = encoder.encode(["no preferences"], convert_to_numpy=True)[0]
no_pref_tensor = torch.FloatTensor(no_pref_embedding)

# =============================================================================
# TWO-TOWER V2 MODEL - SEPARATE LIKE/DISLIKE ENCODINGS
# =============================================================================

class UserTowerV2(nn.Module):
    """User tower with separate like/dislike projections."""
    def __init__(self, state_dim=384, proj_dim=128, dropout=0.1):
        super().__init__()
        # Separate projection for likes
        self.like_proj = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, proj_dim)
        )
        # Separate projection for dislikes
        self.dislike_proj = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, proj_dim)
        )
        # Final user embedding from concatenated projections
        self.user_head = nn.Sequential(
            nn.Linear(proj_dim * 2, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, proj_dim)
        )

    def forward(self, like_state, dislike_state):
        like_emb = self.like_proj(like_state)
        dislike_emb = self.dislike_proj(dislike_state)
        # Concatenate like and dislike projections
        combined = torch.cat([like_emb, dislike_emb], dim=-1)
        return self.user_head(combined)


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


class TwoTowerModelV2(nn.Module):
    """Two-tower with SEPARATE like/dislike encodings."""
    def __init__(self, state_dim=384, embedding_dim=128, learning_rate=0.0003, dropout=0.1):
        super().__init__()
        self.user_tower = UserTowerV2(state_dim, embedding_dim, dropout)
        self.item_tower = ItemTower(state_dim, embedding_dim, dropout)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate, weight_decay=1e-4)

    def forward(self, like_states, dislike_states, item_features):
        user_emb = self.user_tower(like_states, dislike_states)
        item_emb = self.item_tower(item_features)

        # L2 normalize
        user_emb = F.normalize(user_emb, p=2, dim=-1)
        item_emb = F.normalize(item_emb, p=2, dim=-1)

        scores = (user_emb * item_emb).sum(dim=1)
        return scores

    def get_embeddings(self, like_states, dislike_states, item_features):
        user_emb = self.user_tower(like_states, dislike_states)
        item_emb = self.item_tower(item_features)
        user_emb = F.normalize(user_emb, p=2, dim=-1)
        item_emb = F.normalize(item_emb, p=2, dim=-1)
        return user_emb, item_emb


# =============================================================================
# DATASET - V2 with separate like/dislike text
# =============================================================================

def create_like_text(liked_ids, movie_titles_dict, max_items=10):
    """Create text for liked movies only."""
    if not liked_ids:
        return "no liked movies"
    liked_names = [movie_titles_dict.get(mid, "Unknown")[:30] for mid in liked_ids[:max_items]]
    return f"liked: {', '.join(liked_names)}"


def create_dislike_text(disliked_ids, movie_titles_dict, max_items=5):
    """Create text for disliked movies only."""
    if not disliked_ids:
        return "no disliked movies"
    disliked_names = [movie_titles_dict.get(mid, "Unknown")[:30] for mid in disliked_ids[:max_items]]
    return f"disliked: {', '.join(disliked_names)}"


class TwoTowerDatasetV2(Dataset):
    def __init__(self, user_ids, user_data, movie_titles, encoder, movie_embeddings_tensor, num_negatives=16):
        self.user_ids = user_ids
        self.user_data = user_data
        self.movie_titles = movie_titles
        self.encoder = encoder
        self.movie_embeddings = movie_embeddings_tensor
        self.num_negatives = num_negatives
        self.all_movie_ids = list(movie_titles.keys())

    def __len__(self):
        return len(self.user_ids)

    def __getitem__(self, idx):
        user_id = self.user_ids[idx]
        data = self.user_data[user_id]

        liked = data['liked']
        disliked = data['disliked']

        # Sample subset of preferences to reveal (same strategy as V1)
        n_reveal = random.randint(1, min(10, len(liked)))
        revealed_liked = random.sample(liked, n_reveal)
        revealed_disliked = random.sample(disliked, min(len(disliked), 3)) if disliked else []

        # Create SEPARATE like and dislike texts
        like_text = create_like_text(revealed_liked, self.movie_titles)
        dislike_text = create_dislike_text(revealed_disliked, self.movie_titles)

        # Sample positive (from unrevealed liked movies)
        remaining_liked = [m for m in liked if m not in revealed_liked]
        if remaining_liked:
            pos_id = random.choice(remaining_liked)
        else:
            pos_id = random.choice(liked)
        pos_idx = movie_id_to_idx.get(pos_id, 0)

        # Sample negatives (mix of disliked + random)
        neg_ids = []
        likes_set = set(liked)

        if len(disliked) >= 2:
            neg_ids.extend(random.sample(disliked, min(2, len(disliked))))

        remaining = self.num_negatives - len(neg_ids)
        unseen = [m for m in self.all_movie_ids if m not in likes_set]
        if len(unseen) >= remaining:
            neg_ids.extend(random.sample(unseen, remaining))
        else:
            neg_ids.extend(random.choices(unseen if unseen else self.all_movie_ids, k=remaining))

        neg_indices = [movie_id_to_idx.get(mid, 0) for mid in neg_ids]

        return like_text, dislike_text, pos_idx, neg_indices


def collate_fn_v2(batch, encoder, movie_embeddings):
    """Custom collate function to encode like/dislike texts separately."""
    like_texts, dislike_texts, pos_indices, neg_indices_list = zip(*batch)

    # Encode like texts
    like_states = encoder.encode(list(like_texts), convert_to_numpy=True, show_progress_bar=False)
    like_states = torch.FloatTensor(like_states)

    # Encode dislike texts
    dislike_states = encoder.encode(list(dislike_texts), convert_to_numpy=True, show_progress_bar=False)
    dislike_states = torch.FloatTensor(dislike_states)

    # Get positive item embeddings
    pos_embeddings = movie_embeddings[list(pos_indices)]

    # Get negative item embeddings
    neg_embeddings = torch.stack([movie_embeddings[neg_idx] for neg_idx_list in neg_indices_list for neg_idx in neg_idx_list])

    return like_states, dislike_states, pos_embeddings, neg_embeddings


# =============================================================================
# TRAINING
# =============================================================================

print("\n" + "=" * 80)
print("TRAINING")
print("=" * 80)

model = TwoTowerModelV2(state_dim=384, embedding_dim=128, learning_rate=LEARNING_RATE)
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

train_dataset = TwoTowerDatasetV2(train_users, user_data, movie_titles, encoder, movie_embeddings_tensor, NUM_NEGATIVES)
val_dataset = TwoTowerDatasetV2(val_users, user_data, movie_titles, encoder, movie_embeddings_tensor, NUM_NEGATIVES)

def my_collate(batch):
    return collate_fn_v2(batch, encoder, movie_embeddings_tensor)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=my_collate)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=my_collate)

print(f"Training batches: {len(train_loader)}, Val batches: {len(val_loader)}")

best_val_loss = float('inf')
temperature = 0.1

for epoch in range(N_EPOCHS):
    # Training
    model.train()
    train_losses = []

    for like_states, dislike_states, pos_embeddings, neg_embeddings in tqdm(train_loader, desc=f"Epoch {epoch+1}/{N_EPOCHS}", leave=False):
        batch_size = like_states.size(0)
        num_neg = neg_embeddings.size(0) // batch_size

        # Get embeddings
        user_emb, pos_emb = model.get_embeddings(like_states, dislike_states, pos_embeddings)

        # Positive scores
        pos_scores = (user_emb * pos_emb).sum(dim=-1, keepdim=True)

        # Negative scores
        neg_emb = model.item_tower(neg_embeddings)
        neg_emb = F.normalize(neg_emb, p=2, dim=-1)
        neg_emb = neg_emb.view(batch_size, num_neg, -1)
        neg_scores = torch.bmm(neg_emb, user_emb.unsqueeze(-1)).squeeze(-1)

        # InfoNCE loss
        all_scores = torch.cat([pos_scores, neg_scores], dim=1) / temperature
        labels = torch.zeros(batch_size, dtype=torch.long)
        loss = F.cross_entropy(all_scores, labels)

        # Optimize
        model.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        model.optimizer.step()

        train_losses.append(loss.item())

    avg_train_loss = np.mean(train_losses)

    # Validation
    model.eval()
    val_losses = []

    with torch.no_grad():
        for like_states, dislike_states, pos_embeddings, neg_embeddings in val_loader:
            batch_size = like_states.size(0)
            num_neg = neg_embeddings.size(0) // batch_size

            user_emb, pos_emb = model.get_embeddings(like_states, dislike_states, pos_embeddings)
            pos_scores = (user_emb * pos_emb).sum(dim=-1, keepdim=True)

            neg_emb = model.item_tower(neg_embeddings)
            neg_emb = F.normalize(neg_emb, p=2, dim=-1)
            neg_emb = neg_emb.view(batch_size, num_neg, -1)
            neg_scores = torch.bmm(neg_emb, user_emb.unsqueeze(-1)).squeeze(-1)

            all_scores = torch.cat([pos_scores, neg_scores], dim=1) / temperature
            labels = torch.zeros(batch_size, dtype=torch.long)
            loss = F.cross_entropy(all_scores, labels)
            val_losses.append(loss.item())

    avg_val_loss = np.mean(val_losses)

    # Track best
    is_best = avg_val_loss < best_val_loss
    if is_best:
        best_val_loss = avg_val_loss

    print(f"Epoch {epoch+1:3d}/{N_EPOCHS}: Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}{' (best)' if is_best else ''}")

    # Save checkpoint every 10 epochs or on best
    if (epoch + 1) % 10 == 0 or is_best:
        checkpoint = {
            'epoch': epoch + 1,
            'n_movies': N_MOVIES,
            'model_version': 'v2_separate_encodings',
            'user_tower': model.user_tower.state_dict(),
            'item_tower': model.item_tower.state_dict(),
            'val_loss': avg_val_loss,
            'train_loss': avg_train_loss,
            'best_val_loss': best_val_loss
        }
        torch.save(checkpoint, CHECKPOINT_DIR / 'two_tower_v2_paper_config.pt')

print("\n" + "=" * 80)
print("TRAINING COMPLETE")
print(f"Best val loss: {best_val_loss:.4f}")
print(f"Checkpoint saved to: {CHECKPOINT_DIR / 'two_tower_v2_paper_config.pt'}")
print("=" * 80)
