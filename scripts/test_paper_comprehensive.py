"""
Comprehensive tests replicating ALL evaluation from the CSMAI-19 notebook.

Key Tests:
1. Loss Decrease by Timestep - THE RL signal (more prefs = better recs)
2. Single Item Liked vs Disliked - Basic sanity check
3. Strongly vs Weakly Correlated Movies - Cluster detection
4. NDCG metric - Formal evaluation
5. Training speed investigation
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import log_loss, ndcg_score
from scipy.stats import spearmanr
import time

# =============================================================================
# Model (exact copy from CSMAI-19)
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
# Dataset
# =============================================================================

class MovieAttributeDataset(Dataset):
    """Dataset with movies, genres, tags, actors, directors."""
    def __init__(self, ratings_df, n_movies, n_genres, n_tags, n_actors, n_directors,
                 movie_genres_map, movie_tags_map=None, movie_actors_map=None, movie_directors_map=None):
        self.ratings_df = ratings_df
        self.n_movies = n_movies
        self.n_genres = n_genres
        self.n_tags = n_tags
        self.n_actors = n_actors
        self.n_directors = n_directors
        self.n_items = n_movies + n_genres + n_tags + n_actors + n_directors
        self.movie_genres_map = movie_genres_map
        self.movie_tags_map = movie_tags_map or {}
        self.movie_actors_map = movie_actors_map or {}
        self.movie_directors_map = movie_directors_map or {}
        self.user_ids = ratings_df['userId'].unique()

        # Item index offsets
        self.genre_offset = n_movies
        self.tag_offset = n_movies + n_genres
        self.actor_offset = n_movies + n_genres + n_tags
        self.director_offset = n_movies + n_genres + n_tags + n_actors

    def __len__(self):
        return len(self.user_ids)

    def __getitem__(self, idx):
        user_id = self.user_ids[idx]
        user_ratings = self.ratings_df[self.ratings_df['userId'] == user_id]

        full_ratings = np.full(self.n_items, np.nan)
        genre_likes = {}
        tag_likes = {}
        actor_likes = {}
        director_likes = {}

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

            # Propagate to tags
            if movie_idx in self.movie_tags_map:
                for tag_idx, relevance in self.movie_tags_map[movie_idx]:
                    if relevance > 0.5:  # Only high-relevance tags
                        if tag_idx not in tag_likes:
                            tag_likes[tag_idx] = []
                        tag_likes[tag_idx].append(rating)

            # Propagate to actors
            if movie_idx in self.movie_actors_map:
                for actor_idx in self.movie_actors_map[movie_idx]:
                    if actor_idx not in actor_likes:
                        actor_likes[actor_idx] = []
                    actor_likes[actor_idx].append(rating)

            # Propagate to directors
            if movie_idx in self.movie_directors_map:
                for director_idx in self.movie_directors_map[movie_idx]:
                    if director_idx not in director_likes:
                        director_likes[director_idx] = []
                    director_likes[director_idx].append(rating)

        # Aggregate genre ratings
        for genre_idx, ratings_list in genre_likes.items():
            if len(ratings_list) >= 2:
                full_ratings[self.genre_offset + genre_idx] = np.median(ratings_list)

        # Aggregate tag ratings
        for tag_idx, ratings_list in tag_likes.items():
            if len(ratings_list) >= 2:
                full_ratings[self.tag_offset + tag_idx] = np.median(ratings_list)

        # Aggregate actor ratings
        for actor_idx, ratings_list in actor_likes.items():
            if len(ratings_list) >= 2:
                full_ratings[self.actor_offset + actor_idx] = np.median(ratings_list)

        # Aggregate director ratings
        for director_idx, ratings_list in director_likes.items():
            if len(ratings_list) >= 2:
                full_ratings[self.director_offset + director_idx] = np.median(ratings_list)

        # Scramble order
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
# Helper Functions (from notebook)
# =============================================================================

def calculate_loss(predictions, ground_truth):
    """BCE loss with NaN masking - from notebook."""
    filt = ~np.isnan(ground_truth)
    if filt.sum() == 0:
        return np.nan
    preds = np.clip(predictions[filt], 1e-7, 1 - 1e-7)
    gt = ground_truth[filt]
    return log_loss(gt, preds, labels=[0, 1])


def calculate_ndcg(predictions, ground_truth, k=10):
    """Calculate NDCG@k."""
    filt = ~np.isnan(ground_truth)
    if filt.sum() < k:
        return np.nan
    # For NDCG, we need relevance scores (binary: liked or not)
    relevance = ground_truth[filt]
    scores = predictions[filt]
    # ndcg_score expects 2D arrays
    return ndcg_score([relevance], [scores], k=k)


# =============================================================================
# Main
# =============================================================================

def run_comprehensive_test(n_movies=50, n_epochs=50, max_users=5000, min_user_total_ratings=200,
                          n_tags=20, include_actors=False, include_directors=False,
                          n_actors=30, n_directors=20):
    """
    Run comprehensive one-hot model test with configurable parameters.

    Args:
        n_movies: Number of top movies to include
        n_epochs: Training epochs
        max_users: Maximum users to sample
        min_user_total_ratings: Minimum total ratings for dense user filtering
        n_tags: Number of genome tags to include
        include_actors: Include actors as items
        include_directors: Include directors as items
        n_actors: Max number of actors
        n_directors: Max number of directors
    """
    import json

    print("=" * 80)
    print("COMPREHENSIVE PAPER REPLICATION TESTS (ONE-HOT)")
    print("=" * 80)

    # Load data
    DATA_DIR = Path('C:/dev/phd/casper/data/movielens')
    ratings = pd.read_csv(DATA_DIR / 'ratings.csv')
    movies = pd.read_csv(DATA_DIR / 'movies.csv')
    genome_scores = pd.read_csv(DATA_DIR / 'genome-scores.csv')
    genome_tags = pd.read_csv(DATA_DIR / 'genome-tags.csv')

    N_MOVIES = n_movies
    MIN_LIKED_RATINGS = 25

    # Get top movies
    movie_counts = ratings['movieId'].value_counts()
    top_movies = movie_counts.head(N_MOVIES).index.tolist()

    # Filter for DENSE users first (users with many total ratings)
    user_total_counts = ratings['userId'].value_counts()
    dense_users = user_total_counts[user_total_counts >= min_user_total_ratings].index.tolist()
    print(f"Found {len(dense_users):,} dense users (>={min_user_total_ratings} total ratings)")

    # Filter ratings to top movies and dense users
    ratings_filtered = ratings[
        (ratings['movieId'].isin(top_movies)) &
        (ratings['userId'].isin(dense_users))
    ]

    # Select users with 25+ liked ratings on top movies
    liked_ratings = ratings_filtered[ratings_filtered['rating'] >= 4]
    user_liked_counts = liked_ratings.groupby('userId').size()
    active_users = user_liked_counts[user_liked_counts >= MIN_LIKED_RATINGS].index.tolist()

    print(f"Users with >= {MIN_LIKED_RATINGS} liked ratings on top movies: {len(active_users)}")

    if len(active_users) > max_users:
        np.random.seed(42)
        active_users = np.random.choice(active_users, max_users, replace=False).tolist()

    ratings_filtered = ratings_filtered[ratings_filtered['userId'].isin(active_users)]

    # Create mappings
    movie_to_idx = {mid: i for i, mid in enumerate(top_movies)}
    ratings_filtered = ratings_filtered.copy()
    ratings_filtered['itemIdx'] = ratings_filtered['movieId'].map(movie_to_idx)

    # Extract genres
    all_genres = set()
    for _, row in movies[movies['movieId'].isin(top_movies)].iterrows():
        genres = row['genres'].split('|')
        all_genres.update(genres)
    all_genres.discard('(no genres listed)')
    all_genres = sorted(list(all_genres))
    genre_to_idx = {g: i for i, g in enumerate(all_genres)}
    N_GENRES = len(all_genres)

    # Movie -> genres mapping
    movie_genres_map = {}
    movie_titles = {}
    for _, row in movies[movies['movieId'].isin(top_movies)].iterrows():
        movie_idx = movie_to_idx[row['movieId']]
        genres = row['genres'].split('|')
        genre_idxs = [genre_to_idx[g] for g in genres if g in genre_to_idx]
        movie_genres_map[movie_idx] = genre_idxs
        movie_titles[movie_idx] = row['title']

    # Extract genome tags
    genome_filtered = genome_scores[genome_scores['movieId'].isin(top_movies)]
    tag_variance = genome_filtered.groupby('tagId')['relevance'].var()
    top_tag_ids = tag_variance.nlargest(n_tags).index.tolist()
    tag_to_idx = {tid: i for i, tid in enumerate(top_tag_ids)}
    N_TAGS = len(top_tag_ids)

    # Movie -> tags mapping
    movie_tags_map = {}
    for movie_id in top_movies:
        movie_idx = movie_to_idx[movie_id]
        movie_genome = genome_filtered[genome_filtered['movieId'] == movie_id]
        tags = []
        for _, row in movie_genome.iterrows():
            if row['tagId'] in tag_to_idx:
                tags.append((tag_to_idx[row['tagId']], row['relevance']))
        movie_tags_map[movie_idx] = tags

    # Load actors and directors
    N_ACTORS = 0
    N_DIRECTORS = 0
    actor_names = []
    director_names = []
    movie_actors_map = {}
    movie_directors_map = {}

    if include_actors or include_directors:
        cache_path = DATA_DIR / '.cache' / 'credits_top100_actors5.json'
        if cache_path.exists():
            with open(cache_path, 'r') as f:
                credits_data = json.load(f)

            if include_actors:
                actor_counts = {}
                for actors in credits_data['movie_actors'].values():
                    for a in actors:
                        actor_counts[a] = actor_counts.get(a, 0) + 1
                top_actors = sorted(actor_counts.keys(), key=lambda x: -actor_counts[x])[:n_actors]
                actor_names = top_actors
                N_ACTORS = len(actor_names)
                actor_to_idx = {a: i for i, a in enumerate(actor_names)}

                for movie_id in top_movies:
                    movie_idx = movie_to_idx[movie_id]
                    movie_id_str = str(movie_id)
                    if movie_id_str in credits_data['movie_actors']:
                        actors = credits_data['movie_actors'][movie_id_str]
                        movie_actors_map[movie_idx] = [actor_to_idx[a] for a in actors if a in actor_to_idx]

            if include_directors:
                director_counts = {}
                for directors in credits_data['movie_directors'].values():
                    for d in directors:
                        director_counts[d] = director_counts.get(d, 0) + 1
                top_directors = sorted(director_counts.keys(), key=lambda x: -director_counts[x])[:n_directors]
                director_names = top_directors
                N_DIRECTORS = len(director_names)
                director_to_idx = {d: i for i, d in enumerate(director_names)}

                for movie_id in top_movies:
                    movie_idx = movie_to_idx[movie_id]
                    movie_id_str = str(movie_id)
                    if movie_id_str in credits_data['movie_directors']:
                        directors = credits_data['movie_directors'][movie_id_str]
                        movie_directors_map[movie_idx] = [director_to_idx[d] for d in directors if d in director_to_idx]

            print(f"Loaded {N_ACTORS} actors, {N_DIRECTORS} directors from TMDB cache")

    N_ITEMS = N_MOVIES + N_GENRES + N_TAGS + N_ACTORS + N_DIRECTORS

    # Rating distribution
    n_liked = (ratings_filtered['rating'] >= 4).sum()
    n_disliked = (ratings_filtered['rating'] < 4).sum()
    item_str = f"{N_MOVIES} movies + {N_GENRES} genres + {N_TAGS} tags"
    if N_ACTORS > 0:
        item_str += f" + {N_ACTORS} actors"
    if N_DIRECTORS > 0:
        item_str += f" + {N_DIRECTORS} directors"
    print(f"\nDataset: {item_str} = {N_ITEMS} items")
    print(f"Users: {len(active_users)}")
    print(f"Ratings: {len(ratings_filtered)} ({100*n_liked/len(ratings_filtered):.1f}% liked)")

    # Split
    train_users = active_users[:int(0.8 * len(active_users))]
    val_users = active_users[int(0.8 * len(active_users)):]

    train_df = ratings_filtered[ratings_filtered['userId'].isin(train_users)]
    val_df = ratings_filtered[ratings_filtered['userId'].isin(val_users)]

    train_dataset = MovieAttributeDataset(
        train_df, N_MOVIES, N_GENRES, N_TAGS, N_ACTORS, N_DIRECTORS,
        movie_genres_map, movie_tags_map, movie_actors_map, movie_directors_map
    )
    val_dataset = MovieAttributeDataset(
        val_df, N_MOVIES, N_GENRES, N_TAGS, N_ACTORS, N_DIRECTORS,
        movie_genres_map, movie_tags_map, movie_actors_map, movie_directors_map
    )

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Create model
    model = ExtrapolationModel(N_ITEMS, lr=0.001)
    loss_fn = BCEWithLogitsLossNan()

    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")

    # =============================================================================
    # TRAINING (with timing)
    # =============================================================================
    print("\n" + "=" * 80)
    print("TRAINING")
    print("=" * 80)

    N_EPOCHS = n_epochs
    start_time = time.time()

    for epoch in range(N_EPOCHS):
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
    print(f"\nTotal training time: {total_time:.1f}s ({total_time/N_EPOCHS:.2f}s/epoch)")

    # Save checkpoint
    checkpoint_dir = Path('C:/dev/phd/casper/data/movielens/.cache/checkpoints')
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f'onehot_model_{N_MOVIES}movies_{N_ITEMS}items_{len(active_users)}users.pt'
    torch.save({
        'model_state_dict': model.state_dict(),
        'n_items': N_ITEMS,
        'n_movies': N_MOVIES,
        'val_loss': val_loss,
        'epochs': N_EPOCHS
    }, checkpoint_path)
    print(f"Checkpoint saved to: {checkpoint_path}")

    # =============================================================================
    # TEST 1: Loss Decrease by Timestep (THE RL SIGNAL)
    # =============================================================================
    print("\n" + "=" * 80)
    print("TEST 1: LOSS DECREASE BY TIMESTEP (RL Signal)")
    print("=" * 80)
    print("Does adding more preferences IMPROVE predictions?")

    model.eval()
    indices_ordered = torch.arange(N_ITEMS).unsqueeze(0)

    # For each validation user, calculate loss at different timesteps
    losses_by_timestep = {t: [] for t in [1, 2, 3, 5, 10, 20, 30, 50]}

    for user_id in val_users[:30]:  # Test on 30 users
        user_ratings = val_df[val_df['userId'] == user_id]
        user_items = user_ratings['itemIdx'].tolist()
        user_scores = user_ratings['rating'].tolist()

        if len(user_items) < 10:
            continue

        # Ground truth for this user
        ground_truth = np.full(N_MOVIES, np.nan)
        for item_idx, rating in zip(user_items, user_scores):
            if item_idx < N_MOVIES:
                ground_truth[item_idx] = 1.0 if rating >= 4 else 0.0

        for n_revealed in losses_by_timestep.keys():
            if n_revealed > len(user_items):
                continue

            # Build input with n_revealed ratings
            ratings_input = torch.zeros(1, N_ITEMS, 3)
            ratings_input[:, :, 2] = 1  # All unknown

            for i in range(min(n_revealed, len(user_items))):
                item_idx = user_items[i]
                rating = user_scores[i]
                ratings_input[:, item_idx, 2] = 0
                if rating >= 4:
                    ratings_input[:, item_idx, 1] = 1
                else:
                    ratings_input[:, item_idx, 0] = 1

            with torch.no_grad():
                preds = model(indices_ordered, ratings_input)[:, -1, :N_MOVIES].sigmoid().numpy().flatten()

            loss = calculate_loss(preds, ground_truth)
            if not np.isnan(loss):
                losses_by_timestep[n_revealed].append(loss)

    print("\n  Timestep | Avg Loss | Std")
    print("  " + "-" * 35)
    prev_loss = None
    for t in sorted(losses_by_timestep.keys()):
        if losses_by_timestep[t]:
            avg_loss = np.mean(losses_by_timestep[t])
            std_loss = np.std(losses_by_timestep[t])
            delta = f"({avg_loss - prev_loss:+.4f})" if prev_loss else ""
            print(f"  {t:8d} | {avg_loss:.4f}   | {std_loss:.4f} {delta}")
            prev_loss = avg_loss

    # Check if loss decreases
    losses_1 = np.mean(losses_by_timestep[1]) if losses_by_timestep[1] else np.nan
    losses_20 = np.mean(losses_by_timestep[20]) if losses_by_timestep[20] else np.nan
    if not np.isnan(losses_1) and not np.isnan(losses_20):
        improvement = losses_1 - losses_20
        print(f"\n  Loss improvement (1 -> 20 timesteps): {improvement:+.4f}")
        if improvement > 0.05:
            print("  >>> GOOD: Adding preferences significantly improves predictions!")
        elif improvement > 0:
            print("  >>> OK: Adding preferences slightly improves predictions")
        else:
            print("  >>> PROBLEM: Adding preferences doesn't improve predictions!")

    # =============================================================================
    # TEST 2: NDCG at Different Timesteps
    # =============================================================================
    print("\n" + "=" * 80)
    print("TEST 2: NDCG@10 BY TIMESTEP")
    print("=" * 80)

    ndcg_by_timestep = {t: [] for t in [1, 5, 10, 20]}

    for user_id in val_users[:30]:
        user_ratings = val_df[val_df['userId'] == user_id]
        user_items = user_ratings['itemIdx'].tolist()
        user_scores = user_ratings['rating'].tolist()

        if len(user_items) < 10:
            continue

        ground_truth = np.full(N_MOVIES, np.nan)
        for item_idx, rating in zip(user_items, user_scores):
            if item_idx < N_MOVIES:
                ground_truth[item_idx] = 1.0 if rating >= 4 else 0.0

        for n_revealed in ndcg_by_timestep.keys():
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
                preds = model(indices_ordered, ratings_input)[:, -1, :N_MOVIES].sigmoid().numpy().flatten()

            ndcg = calculate_ndcg(preds, ground_truth, k=10)
            if not np.isnan(ndcg):
                ndcg_by_timestep[n_revealed].append(ndcg)

    print("\n  Timestep | NDCG@10")
    print("  " + "-" * 25)
    for t in sorted(ndcg_by_timestep.keys()):
        if ndcg_by_timestep[t]:
            avg_ndcg = np.mean(ndcg_by_timestep[t])
            print(f"  {t:8d} | {avg_ndcg:.4f}")

    # =============================================================================
    # TEST 3: Single Item Liked vs Disliked
    # =============================================================================
    print("\n" + "=" * 80)
    print("TEST 3: SINGLE ITEM LIKED vs DISLIKED")
    print("=" * 80)

    test_movies = [0, 1, 2, 3, 4]  # Top 5 most popular
    print("\nTesting top 5 movies:")

    for movie_idx in test_movies:
        single_indices = torch.tensor([[movie_idx]])
        liked_onehot = torch.tensor([[[0., 1., 0.]]])
        disliked_onehot = torch.tensor([[[1., 0., 0.]]])

        with torch.no_grad():
            liked_preds = model(single_indices, liked_onehot)[:, -1, :N_MOVIES].sigmoid().numpy().flatten()
            disliked_preds = model(single_indices, disliked_onehot)[:, -1, :N_MOVIES].sigmoid().numpy().flatten()

        score_diff = np.mean(liked_preds) - np.mean(disliked_preds)
        print(f"  {movie_titles.get(movie_idx, f'Movie {movie_idx}')[:35]:35s} | diff: {score_diff:+.4f}")

    # =============================================================================
    # TEST 4: Correlated Movie Clusters (Star Wars test)
    # =============================================================================
    print("\n" + "=" * 80)
    print("TEST 4: MOVIE CLUSTER DETECTION")
    print("=" * 80)

    # Find Sci-Fi movies (should be correlated)
    scifi_idx = genre_to_idx.get('Sci-Fi', 0)
    scifi_movies = [m for m, g in movie_genres_map.items() if scifi_idx in g][:5]

    # Find Drama movies (different cluster)
    drama_idx = genre_to_idx.get('Drama', 0)
    drama_movies = [m for m, g in movie_genres_map.items()
                    if drama_idx in g and scifi_idx not in g][:5]

    if len(scifi_movies) >= 2:
        target_scifi = scifi_movies[0]
        print(f"\nTarget: {movie_titles.get(target_scifi, 'Sci-Fi movie')}")

        # Predict target when user likes OTHER Sci-Fi movies
        scifi_predictor = scifi_movies[1]
        single_indices = torch.tensor([[scifi_predictor]])
        liked_onehot = torch.tensor([[[0., 1., 0.]]])

        with torch.no_grad():
            preds = model(single_indices, liked_onehot)[:, -1, :N_MOVIES].sigmoid().numpy().flatten()

        scifi_pred = preds[target_scifi]
        print(f"  When user likes '{movie_titles.get(scifi_predictor, 'Sci-Fi')[:30]}': pred = {scifi_pred:.4f}")

        # Compare with drama predictor
        if drama_movies:
            drama_predictor = drama_movies[0]
            single_indices = torch.tensor([[drama_predictor]])

            with torch.no_grad():
                preds = model(single_indices, liked_onehot)[:, -1, :N_MOVIES].sigmoid().numpy().flatten()

            drama_pred = preds[target_scifi]
            print(f"  When user likes '{movie_titles.get(drama_predictor, 'Drama')[:30]}': pred = {drama_pred:.4f}")
            print(f"  Cluster lift: {scifi_pred - drama_pred:+.4f}")

            if scifi_pred > drama_pred:
                print("  >>> GOOD: Model detects Sci-Fi cluster!")
            else:
                print("  >>> PROBLEM: Model doesn't detect genre clusters")

    # =============================================================================
    # SUMMARY
    # =============================================================================
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    loss_improvement = losses_1 - losses_20 if not np.isnan(losses_1) and not np.isnan(losses_20) else 0
    ndcg_1 = np.mean(ndcg_by_timestep[1]) if ndcg_by_timestep[1] else 0
    ndcg_20 = np.mean(ndcg_by_timestep[20]) if ndcg_by_timestep[20] else 0

    print(f"""
    Training:
    - {N_EPOCHS} epochs in {total_time:.1f}s ({total_time/N_EPOCHS:.2f}s/epoch)
    - Final val loss: {val_loss:.4f}

    RL Signal (Loss by Timestep):
    - 1 pref:  {losses_1:.4f}
    - 20 prefs: {losses_20:.4f}
    - Improvement: {loss_improvement:+.4f}

    NDCG@10:
    - 1 pref:  {ndcg_1:.4f}
    - 20 prefs: {ndcg_20:.4f}

    KEY QUESTION: Does adding preferences improve predictions?
    {'YES - Loss decreases, RL signal present!' if loss_improvement > 0.02 else 'WEAK - Need investigation'}
    """)

    # Return results for notebook use
    results = {
        'loss_results': {t: np.mean(v) for t, v in losses_by_timestep.items() if v},
        'ndcg_results': {t: np.mean(v) for t, v in ndcg_by_timestep.items() if v},
        'loss_improvement': loss_improvement,
        'ndcg_improvement': ndcg_20 - ndcg_1,
        'single_item_diff': np.mean([liked_preds.mean() - disliked_preds.mean()
                                     for _ in range(1)]),  # Simplified
        'config': {
            'n_movies': N_MOVIES,
            'n_genres': N_GENRES,
            'n_items': N_ITEMS,
            'n_users': len(active_users),
            'n_epochs': N_EPOCHS
        }
    }
    return results


def main():
    """CLI entry point with default dense user settings."""
    import argparse
    parser = argparse.ArgumentParser(description="One-Hot Model Comprehensive Test")
    parser.add_argument('--movies', type=int, default=50, help='Number of movies (default: 50)')
    parser.add_argument('--epochs', type=int, default=50, help='Training epochs (default: 50)')
    parser.add_argument('--users', type=int, default=5000, help='Max users (default: 5000)')
    parser.add_argument('--min-ratings', type=int, default=200, help='Min total ratings for dense users (default: 200)')
    parser.add_argument('--tags', type=int, default=20, help='Number of genome tags (default: 20)')
    parser.add_argument('--actors', action='store_true', help='Include actors as items')
    parser.add_argument('--directors', action='store_true', help='Include directors as items')
    parser.add_argument('--n-actors', type=int, default=30, help='Max actors (default: 30)')
    parser.add_argument('--n-directors', type=int, default=20, help='Max directors (default: 20)')
    args = parser.parse_args()

    return run_comprehensive_test(
        n_movies=args.movies,
        n_epochs=args.epochs,
        max_users=args.users,
        min_user_total_ratings=args.min_ratings,
        n_tags=args.tags,
        include_actors=args.actors,
        include_directors=args.directors,
        n_actors=args.n_actors,
        n_directors=args.n_directors
    )


if __name__ == "__main__":
    main()
