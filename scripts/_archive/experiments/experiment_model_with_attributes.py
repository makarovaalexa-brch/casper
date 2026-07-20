"""
EXPERIMENT 1: Own RL Model with Rich Attributes

Model trained on movies + genres + genome tags (as proxy for actors/directors/concepts).
Genome tags provide rich semantic attributes like "dark hero", "twist ending", "based on book".

This creates a richer item space for learning correlations.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import ndcg_score
import time

# =============================================================================
# Model
# =============================================================================

class ExtrapolationModel(nn.Module):
    def __init__(self, n_items, lr=0.001):
        super(ExtrapolationModel, self).__init__()
        self.n_items = n_items
        hidden_dim = max(n_items // 2, 64)  # Min 64 for smaller item sets

        self.embedding = nn.Embedding(n_items + 1, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim + 3, hidden_dim, 1, batch_first=True)
        self.dense1 = nn.Linear(hidden_dim, n_items)
        self.dense2 = nn.Linear(n_items, hidden_dim)
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=1, batch_first=True)
        self.output = nn.Linear(hidden_dim * 2, n_items * 2)

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
    def forward(self, y_pred, y_true):
        mask = torch.isnan(y_true)
        y_pred = torch.where(mask, torch.zeros_like(y_true), y_pred)
        y_true = torch.where(mask, torch.zeros_like(y_true), y_true)
        return torch.nn.functional.binary_cross_entropy_with_logits(y_pred, y_true)


# =============================================================================
# Dataset with Movies + Genres + Genome Tags
# =============================================================================

class RichAttributeDataset(Dataset):
    """
    Dataset with movies, genres, AND genome tags as items.

    Item layout:
    - [0:n_movies] = movies
    - [n_movies:n_movies+n_genres] = genres
    - [n_movies+n_genres:] = genome tags
    """
    def __init__(self, ratings_df, n_movies, n_genres, n_tags,
                 movie_genres_map, movie_tags_map, tag_threshold=0.5):
        self.ratings_df = ratings_df
        self.n_movies = n_movies
        self.n_genres = n_genres
        self.n_tags = n_tags
        self.n_items = n_movies + n_genres + n_tags
        self.movie_genres_map = movie_genres_map
        self.movie_tags_map = movie_tags_map
        self.tag_threshold = tag_threshold
        self.user_ids = ratings_df['userId'].unique()

    def __len__(self):
        return len(self.user_ids)

    def __getitem__(self, idx):
        user_id = self.user_ids[idx]
        user_ratings = self.ratings_df[self.ratings_df['userId'] == user_id]

        full_ratings = np.full(self.n_items, np.nan)
        genre_likes = {}
        tag_likes = {}

        for _, row in user_ratings.iterrows():
            movie_idx = int(row['itemIdx'])
            rating = 1.0 if row['rating'] >= 4 else 0.0
            full_ratings[movie_idx] = rating

            # Propagate to genres
            if movie_idx in self.movie_genres_map:
                for genre_idx in self.movie_genres_map[movie_idx]:
                    if genre_idx not in genre_likes:
                        genre_likes[genre_idx] = []
                    genre_likes[genre_idx].append(rating)

            # Propagate to tags (only if tag relevance > threshold)
            if movie_idx in self.movie_tags_map:
                for tag_idx, relevance in self.movie_tags_map[movie_idx]:
                    if relevance >= self.tag_threshold:
                        if tag_idx not in tag_likes:
                            tag_likes[tag_idx] = []
                        tag_likes[tag_idx].append(rating)

        # Set genre ratings
        for genre_idx, ratings_list in genre_likes.items():
            if len(ratings_list) >= 2:
                genre_item_idx = self.n_movies + genre_idx
                full_ratings[genre_item_idx] = np.median(ratings_list)

        # Set tag ratings
        for tag_idx, ratings_list in tag_likes.items():
            if len(ratings_list) >= 2:
                tag_item_idx = self.n_movies + self.n_genres + tag_idx
                if tag_item_idx < self.n_items:
                    full_ratings[tag_item_idx] = np.median(ratings_list)

        indices = np.random.permutation(self.n_items)

        rating_input = np.zeros((self.n_items, 3))
        for i, item_idx in enumerate(indices):
            if np.isnan(full_ratings[item_idx]):
                rating_input[i, 2] = 1
            elif full_ratings[item_idx] == 0:
                rating_input[i, 0] = 1
            else:
                rating_input[i, 1] = 1

        explicit = np.tile(full_ratings, (self.n_items, 1))
        implicit = (~np.isnan(explicit)).astype(float)
        output = np.concatenate([explicit, implicit], axis=1)

        return (
            torch.LongTensor(indices),
            torch.FloatTensor(rating_input),
            torch.FloatTensor(output)
        )


# =============================================================================
# Helper Functions
# =============================================================================

def calculate_ndcg(predictions, ground_truth, k=10):
    filt = ~np.isnan(ground_truth)
    if filt.sum() < k:
        return np.nan
    relevance = ground_truth[filt]
    scores = predictions[filt]
    return ndcg_score([relevance], [scores], k=k)


# =============================================================================
# Main Experiment
# =============================================================================

def run_model_experiment(
    n_movies: int = 100,
    n_top_tags: int = 50,
    max_users: int = 300,
    n_epochs: int = 50,
    verbose: bool = True
):
    """
    Run the trained model experiment.
    """

    print("=" * 80)
    print("EXPERIMENT 1: OWN RL MODEL WITH RICH ATTRIBUTES")
    print("=" * 80)

    # Load data
    DATA_DIR = Path('C:/dev/phd/casper/data/movielens')
    ratings = pd.read_csv(DATA_DIR / 'ratings.csv')
    movies = pd.read_csv(DATA_DIR / 'movies.csv')
    genome_scores = pd.read_csv(DATA_DIR / 'genome-scores.csv')
    genome_tags = pd.read_csv(DATA_DIR / 'genome-tags.csv')

    # Get top movies
    movie_counts = ratings['movieId'].value_counts()
    top_movies = movie_counts.head(n_movies).index.tolist()

    ratings_filtered = ratings[ratings['movieId'].isin(top_movies)]

    # Select users with 25+ liked ratings
    liked_ratings = ratings_filtered[ratings_filtered['rating'] >= 4]
    user_liked_counts = liked_ratings.groupby('userId').size()
    active_users = user_liked_counts[user_liked_counts >= 25].index.tolist()

    if len(active_users) > max_users:
        np.random.seed(42)
        active_users = np.random.choice(active_users, max_users, replace=False).tolist()

    ratings_filtered = ratings_filtered[ratings_filtered['userId'].isin(active_users)]

    # Create movie index mapping
    movie_to_idx = {mid: i for i, mid in enumerate(top_movies)}
    ratings_filtered = ratings_filtered.copy()
    ratings_filtered['itemIdx'] = ratings_filtered['movieId'].map(movie_to_idx)

    # Extract genres
    all_genres = set()
    movie_genres_map = {}
    movie_titles = {}

    for _, row in movies[movies['movieId'].isin(top_movies)].iterrows():
        movie_idx = movie_to_idx[row['movieId']]
        genres = row['genres'].split('|')
        all_genres.update(genres)
        movie_titles[movie_idx] = row['title']

    all_genres.discard('(no genres listed)')
    all_genres = sorted(list(all_genres))
    genre_to_idx = {g: i for i, g in enumerate(all_genres)}
    N_GENRES = len(all_genres)

    for _, row in movies[movies['movieId'].isin(top_movies)].iterrows():
        movie_idx = movie_to_idx[row['movieId']]
        genres = row['genres'].split('|')
        movie_genres_map[movie_idx] = [genre_to_idx[g] for g in genres if g in genre_to_idx]

    # Extract TOP genome tags (most discriminative)
    # Filter to our movies
    genome_filtered = genome_scores[genome_scores['movieId'].isin(top_movies)]

    # Find tags with high variance (most discriminative)
    tag_variance = genome_filtered.groupby('tagId')['relevance'].var()
    top_tag_ids = tag_variance.nlargest(n_top_tags).index.tolist()

    tag_to_idx = {tid: i for i, tid in enumerate(top_tag_ids)}
    tag_names = {tid: genome_tags[genome_tags['tagId'] == tid]['tag'].values[0]
                 for tid in top_tag_ids}
    N_TAGS = len(top_tag_ids)

    print(f"\nTop {N_TAGS} discriminative genome tags:")
    for i, tid in enumerate(top_tag_ids[:10]):
        print(f"  {tag_names[tid]}")
    print("  ...")

    # Build movie -> tags mapping
    movie_tags_map = {}
    for movie_id in top_movies:
        movie_idx = movie_to_idx[movie_id]
        movie_genome = genome_filtered[genome_filtered['movieId'] == movie_id]
        tags = []
        for _, row in movie_genome.iterrows():
            if row['tagId'] in tag_to_idx:
                tags.append((tag_to_idx[row['tagId']], row['relevance']))
        movie_tags_map[movie_idx] = tags

    N_ITEMS = n_movies + N_GENRES + N_TAGS
    print(f"\nDataset: {n_movies} movies + {N_GENRES} genres + {N_TAGS} tags = {N_ITEMS} items")
    print(f"Users: {len(active_users)}")

    # Split
    train_users = active_users[:int(0.8 * len(active_users))]
    val_users = active_users[int(0.8 * len(active_users)):]

    train_df = ratings_filtered[ratings_filtered['userId'].isin(train_users)]
    val_df = ratings_filtered[ratings_filtered['userId'].isin(val_users)]

    train_dataset = RichAttributeDataset(
        train_df, n_movies, N_GENRES, N_TAGS, movie_genres_map, movie_tags_map
    )
    val_dataset = RichAttributeDataset(
        val_df, n_movies, N_GENRES, N_TAGS, movie_genres_map, movie_tags_map
    )

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    # Create model
    model = ExtrapolationModel(N_ITEMS, lr=0.001)
    loss_fn = BCEWithLogitsLossNan()

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Train
    print("\n" + "=" * 80)
    print("TRAINING")
    print("=" * 80)

    start_time = time.time()

    for epoch in range(n_epochs):
        model.train()
        train_loss = 0
        for indices, ratings_batch, targets in train_loader:
            model.optimizer.zero_grad()
            outputs = model(indices, ratings_batch)
            loss = loss_fn(outputs, targets)
            loss.backward()
            model.optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for indices, ratings_batch, targets in val_loader:
                outputs = model(indices, ratings_batch)
                loss = loss_fn(outputs, targets)
                val_loss += loss.item()
        val_loss /= len(val_loader)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            elapsed = time.time() - start_time
            print(f"Epoch {epoch+1:3d} | Train: {train_loss:.4f} | Val: {val_loss:.4f} | Time: {elapsed:.1f}s")

    total_time = time.time() - start_time
    print(f"\nTotal training time: {total_time:.1f}s")

    # =============================================================================
    # TEST: NDCG BY TIMESTEP (RL Signal)
    # =============================================================================
    print("\n" + "=" * 80)
    print("TEST: NDCG@10 BY TIMESTEP (RL SIGNAL)")
    print("=" * 80)

    model.eval()
    indices_ordered = torch.arange(N_ITEMS).unsqueeze(0)

    timesteps = [1, 3, 5, 10, 20]
    ndcg_by_timestep = {t: [] for t in timesteps}

    for user_id in val_users[:30]:
        user_ratings = val_df[val_df['userId'] == user_id]
        user_items = user_ratings['itemIdx'].tolist()
        user_scores = user_ratings['rating'].tolist()

        if len(user_items) < 10:
            continue

        ground_truth = np.full(n_movies, np.nan)
        for item_idx, rating in zip(user_items, user_scores):
            if item_idx < n_movies:
                ground_truth[item_idx] = 1.0 if rating >= 4 else 0.0

        for n_revealed in timesteps:
            if n_revealed > len(user_items):
                continue

            ratings_input = torch.zeros(1, N_ITEMS, 3)
            ratings_input[:, :, 2] = 1

            for i in range(min(n_revealed, len(user_items))):
                item_idx = user_items[i]
                rating = user_scores[i]
                ratings_input[:, item_idx, 2] = 0
                if rating >= 4:
                    ratings_input[:, item_idx, 1] = 1
                else:
                    ratings_input[:, item_idx, 0] = 1

            with torch.no_grad():
                preds = model(indices_ordered, ratings_input)[:, -1, :n_movies].sigmoid().numpy().flatten()

            ndcg = calculate_ndcg(preds, ground_truth, k=10)
            if not np.isnan(ndcg):
                ndcg_by_timestep[n_revealed].append(ndcg)

    print("\n  Timestep | NDCG@10")
    print("  " + "-" * 25)
    results = {}
    prev_ndcg = None
    for t in sorted(timesteps):
        if ndcg_by_timestep[t]:
            avg_ndcg = np.mean(ndcg_by_timestep[t])
            delta = f"({avg_ndcg - prev_ndcg:+.4f})" if prev_ndcg else ""
            print(f"  {t:8d} | {avg_ndcg:.4f} {delta}")
            results[t] = avg_ndcg
            prev_ndcg = avg_ndcg

    if results:
        first_t = min(results.keys())
        last_t = max(results.keys())
        improvement = results[last_t] - results[first_t]
        print(f"\n  Improvement ({first_t} -> {last_t} prefs): {improvement:+.4f}")
        if improvement > 0.02:
            print("  >>> RL SIGNAL PRESENT!")

    return results, model, {
        'n_movies': n_movies,
        'n_genres': N_GENRES,
        'n_tags': N_TAGS,
        'n_items': N_ITEMS,
        'tag_names': tag_names,
        'movie_titles': movie_titles,
        'all_genres': all_genres
    }


if __name__ == "__main__":
    results, model, config = run_model_experiment(
        n_movies=100,
        n_top_tags=50,
        max_users=300,
        n_epochs=50
    )
