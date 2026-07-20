"""
Paper replication WITH GENRE ATTRIBUTES.

Key insight from CSMAI-19 paper:
- Items include BOTH movies AND genres (and actors/directors)
- When you rate a movie, you implicitly rate its genres
- This creates correlations the model can learn

Example: If user likes "The Matrix" -> implicitly likes "Sci-Fi"
Then model can predict other Sci-Fi movies for users who like Sci-Fi
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset, DataLoader

# =============================================================================
# Model (exact copy from CSMAI-19 notebook)
# =============================================================================

class ExtrapolationModel(nn.Module):
    def __init__(self, n_items, lr=0.001):
        super(ExtrapolationModel, self).__init__()
        self.n_items = n_items
        hidden_dim = n_items // 2

        self.embedding = nn.Embedding(n_items + 1, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim + 3, hidden_dim, 1, batch_first=True, bidirectional=False)
        self.dense1 = nn.Linear(hidden_dim, n_items)
        self.dense2 = nn.Linear(n_items, hidden_dim)
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=1, batch_first=True)
        # After concat: hidden_dim * 2, not n_items (they differ when n_items is odd)
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
# Dataset with Genre Attributes
# =============================================================================

class MovieGenreDataset(Dataset):
    """
    Dataset that includes movies AND genres as items.
    Genre ratings are inferred from movie ratings.
    """
    def __init__(self, ratings_df, n_movies, n_genres, movie_genres_map):
        """
        Args:
            ratings_df: DataFrame with userId, itemIdx (movie), rating
            n_movies: Number of movies
            n_genres: Number of genres
            movie_genres_map: Dict mapping movie_idx -> list of genre_idxs
        """
        self.ratings_df = ratings_df
        self.n_movies = n_movies
        self.n_genres = n_genres
        self.n_items = n_movies + n_genres
        self.movie_genres_map = movie_genres_map
        self.user_ids = ratings_df['userId'].unique()

    def __len__(self):
        return len(self.user_ids)

    def __getitem__(self, idx):
        user_id = self.user_ids[idx]
        user_ratings = self.ratings_df[self.ratings_df['userId'] == user_id]

        # Create full rating vector (NaN for unseen)
        # [0:n_movies] = movies, [n_movies:n_movies+n_genres] = genres
        full_ratings = np.full(self.n_items, np.nan)

        # First, set movie ratings
        genre_likes = {}  # genre_idx -> list of ratings
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

        # Set genre ratings as median of movie ratings in that genre
        for genre_idx, ratings_list in genre_likes.items():
            if len(ratings_list) >= 2:  # Need at least 2 ratings to infer
                genre_item_idx = self.n_movies + genre_idx
                full_ratings[genre_item_idx] = np.median(ratings_list)

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

        # Output
        explicit = np.tile(full_ratings, (self.n_items, 1))
        implicit = (~np.isnan(explicit)).astype(float)
        output = np.concatenate([explicit, implicit], axis=1)

        return (
            torch.LongTensor(indices),
            torch.FloatTensor(rating_input),
            torch.FloatTensor(output)
        )


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 80)
    print("PAPER REPLICATION WITH GENRE ATTRIBUTES")
    print("=" * 80)

    # Load data
    DATA_DIR = Path('C:/dev/phd/casper/data/movielens')
    ratings = pd.read_csv(DATA_DIR / 'ratings.csv')
    movies = pd.read_csv(DATA_DIR / 'movies.csv')

    N_MOVIES = 100  # Match paper exactly
    MIN_LIKED_RATINGS = 25  # Paper's criterion: users with at least 25 LIKED ratings

    # Get top N most-rated movies (match paper)
    movie_counts = ratings['movieId'].value_counts()
    top_movies = movie_counts.head(N_MOVIES).index.tolist()

    # Filter to top movies
    ratings_filtered = ratings[ratings['movieId'].isin(top_movies)]

    # KEY DIFFERENCE: Select users with at least MIN_LIKED_RATINGS LIKED items (not total)
    liked_ratings = ratings_filtered[ratings_filtered['rating'] >= 4]
    user_liked_counts = liked_ratings.groupby('userId').size()
    active_users = user_liked_counts[user_liked_counts >= MIN_LIKED_RATINGS].index.tolist()

    print(f"Users with >= {MIN_LIKED_RATINGS} liked ratings: {len(active_users)}")

    # Limit users for faster iteration (keep quality users, just fewer)
    MAX_USERS = 200
    if len(active_users) > MAX_USERS:
        np.random.seed(42)
        active_users = np.random.choice(active_users, MAX_USERS, replace=False).tolist()
        print(f"Sampled {MAX_USERS} users for faster iteration")

    # Filter to active users
    ratings_filtered = ratings_filtered[ratings_filtered['userId'].isin(active_users)]

    # Create movie index mapping
    movie_to_idx = {mid: i for i, mid in enumerate(top_movies)}
    idx_to_movie = {i: mid for mid, i in movie_to_idx.items()}
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

    print(f"Genres: {all_genres}")
    print(f"N_GENRES: {N_GENRES}")

    # Build movie -> genres mapping
    movie_genres_map = {}
    for _, row in movies[movies['movieId'].isin(top_movies)].iterrows():
        movie_idx = movie_to_idx[row['movieId']]
        genres = row['genres'].split('|')
        genre_idxs = [genre_to_idx[g] for g in genres if g in genre_to_idx]
        movie_genres_map[movie_idx] = genre_idxs

    # Also store movie titles for later
    movie_titles = {}
    for _, row in movies[movies['movieId'].isin(top_movies)].iterrows():
        movie_idx = movie_to_idx[row['movieId']]
        movie_titles[movie_idx] = row['title']

    N_ITEMS = N_MOVIES + N_GENRES
    N_USERS = len(active_users)
    print(f"\nDataset: {N_MOVIES} movies + {N_GENRES} genres = {N_ITEMS} items")
    print(f"Users: {len(active_users)}")
    print(f"Total movie ratings: {len(ratings_filtered)}")

    # DIAGNOSTIC: Check liked vs disliked distribution
    n_liked = (ratings_filtered['rating'] >= 4).sum()
    n_disliked = (ratings_filtered['rating'] < 4).sum()
    print(f"\nRating distribution:")
    print(f"  Liked (>=4): {n_liked} ({100*n_liked/len(ratings_filtered):.1f}%)")
    print(f"  Disliked (<4): {n_disliked} ({100*n_disliked/len(ratings_filtered):.1f}%)")

    # Split train/val
    train_users = active_users[:int(0.8 * len(active_users))]
    val_users = active_users[int(0.8 * len(active_users)):]

    train_df = ratings_filtered[ratings_filtered['userId'].isin(train_users)]
    val_df = ratings_filtered[ratings_filtered['userId'].isin(val_users)]

    train_dataset = MovieGenreDataset(train_df, N_MOVIES, N_GENRES, movie_genres_map)
    val_dataset = MovieGenreDataset(val_df, N_MOVIES, N_GENRES, movie_genres_map)

    # Paper uses batch_size=1!
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    # Create model
    model = ExtrapolationModel(N_ITEMS, lr=0.001)
    loss_fn = BCEWithLogitsLossNan()

    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Train
    print("\n" + "=" * 80)
    print("TRAINING")
    print("=" * 80)

    N_EPOCHS = 30  # Reduced for faster iteration with batch_size=1

    for epoch in range(N_EPOCHS):
        model.train()
        train_loss = 0
        for batch_idx, (indices, ratings_batch, targets) in enumerate(train_loader):
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
            print(f"Epoch {epoch+1:3d} | Train: {train_loss:.4f} | Val: {val_loss:.4f}")

    # =============================================================================
    # PAPER'S EXACT TEST: Single entity liked vs disliked
    # From notebook: entity = 'Russell Crowe', liked = [[1, 0, 0]], disliked = [[0, 1, 0]]
    # =============================================================================
    print("\n" + "=" * 80)
    print("PAPER'S EXACT TEST: Single Item Liked vs Disliked")
    print("=" * 80)

    model.eval()

    # Test with a single popular movie
    test_movie_idx = 0  # Most popular movie
    test_movie_name = movie_titles.get(test_movie_idx, "Movie 0")
    print(f"\nTest item: {test_movie_name} (idx={test_movie_idx})")

    single_indices = torch.tensor([[test_movie_idx]])  # [1, 1]

    # CRITICAL: Check encoding
    # Paper data uses: np.eye(3)[rating] where 0=disliked, 1=liked, 2=not_seen
    # So: disliked(0) -> [1,0,0], liked(1) -> [0,1,0], not_seen(2) -> [0,0,1]

    # Data encoding: [disliked, liked, not_seen]
    liked_onehot = torch.tensor([[[0., 1., 0.]]])  # liked is index 1
    disliked_onehot = torch.tensor([[[1., 0., 0.]]])  # disliked is index 0

    with torch.no_grad():
        liked_preds = model(single_indices, liked_onehot)[:, -1, :N_MOVIES].sigmoid().numpy().flatten()
        disliked_preds = model(single_indices, disliked_onehot)[:, -1, :N_MOVIES].sigmoid().numpy().flatten()

    print(f"\n  Predictions when {test_movie_name} is LIKED:")
    top5_liked = np.argsort(liked_preds)[-5:][::-1]
    for idx in top5_liked:
        if idx != test_movie_idx:
            print(f"    {movie_titles.get(idx, f'Movie {idx}')[:40]:40s}: {liked_preds[idx]:.4f}")

    print(f"\n  Predictions when {test_movie_name} is DISLIKED:")
    top5_disliked = np.argsort(disliked_preds)[-5:][::-1]
    for idx in top5_disliked:
        if idx != test_movie_idx:
            print(f"    {movie_titles.get(idx, f'Movie {idx}')[:40]:40s}: {disliked_preds[idx]:.4f}")

    # Score difference
    single_score_diff = np.mean(liked_preds) - np.mean(disliked_preds)
    print(f"\n  Mean score diff (liked - disliked): {single_score_diff:+.4f}")
    if abs(single_score_diff) < 0.01:
        print("  >>> PROBLEM: Predictions don't change with liked vs disliked!")
    else:
        print("  >>> OK: Predictions change based on liked vs disliked")

    # =============================================================================
    # SANITY TEST: Likes vs Dislikes for MULTIPLE MOVIES of a genre
    # =============================================================================
    print("\n" + "=" * 80)
    print("SANITY TEST: Like vs Dislike MULTIPLE Sci-Fi Movies")
    print("=" * 80)

    model.eval()

    scifi_idx = genre_to_idx.get('Sci-Fi', 0)
    romance_idx = genre_to_idx.get('Romance', 1)

    # Get Sci-Fi and non-Sci-Fi movies
    scifi_movie_idxs = [m for m, g in movie_genres_map.items() if scifi_idx in g]
    non_scifi_idxs = [m for m in range(N_MOVIES) if m not in scifi_movie_idxs]

    print(f"\nSci-Fi movies: {len(scifi_movie_idxs)}")
    print(f"Non-Sci-Fi movies: {len(non_scifi_idxs)}")

    # Test with 5 Sci-Fi movies liked vs disliked
    N_REVEAL = 5
    test_scifi = scifi_movie_idxs[:N_REVEAL]
    test_other = [m for m in non_scifi_idxs if m not in test_scifi]  # Movies to predict

    print(f"\nRevealing {N_REVEAL} Sci-Fi movies:")
    for m in test_scifi:
        print(f"  {movie_titles.get(m, f'Movie {m}')[:50]}")

    indices = torch.arange(N_ITEMS).unsqueeze(0)

    # Scenario 1: User LIKES these Sci-Fi movies
    ratings_liked = torch.zeros(1, N_ITEMS, 3)
    ratings_liked[:, :, 2] = 1  # All unknown
    for m in test_scifi:
        ratings_liked[:, m, 2] = 0
        ratings_liked[:, m, 1] = 1  # Liked
    # Also mark the Sci-Fi genre as liked (since it would be inferred)
    ratings_liked[:, N_MOVIES + scifi_idx, 2] = 0
    ratings_liked[:, N_MOVIES + scifi_idx, 1] = 1

    # Scenario 2: User DISLIKES these Sci-Fi movies
    ratings_disliked = torch.zeros(1, N_ITEMS, 3)
    ratings_disliked[:, :, 2] = 1
    for m in test_scifi:
        ratings_disliked[:, m, 2] = 0
        ratings_disliked[:, m, 0] = 1  # Disliked
    # Genre also disliked
    ratings_disliked[:, N_MOVIES + scifi_idx, 2] = 0
    ratings_disliked[:, N_MOVIES + scifi_idx, 0] = 1

    with torch.no_grad():
        out_liked = model(indices, ratings_liked)[:, -1, :N_MOVIES].sigmoid()
        out_disliked = model(indices, ratings_disliked)[:, -1, :N_MOVIES].sigmoid()

    liked_scores = out_liked.numpy().flatten()
    disliked_scores = out_disliked.numpy().flatten()

    # Compare predictions for OTHER Sci-Fi movies (not revealed)
    other_scifi = [m for m in scifi_movie_idxs if m not in test_scifi]

    print(f"\nPredictions for OTHER {len(other_scifi)} Sci-Fi movies:")
    scifi_liked_avg = np.mean([liked_scores[m] for m in other_scifi])
    scifi_disliked_avg = np.mean([disliked_scores[m] for m in other_scifi])
    print(f"  When Sci-Fi LIKED:    {scifi_liked_avg:.4f}")
    print(f"  When Sci-Fi DISLIKED: {scifi_disliked_avg:.4f}")
    scifi_diff = scifi_liked_avg - scifi_disliked_avg
    print(f"  Difference: {scifi_diff:+.4f}")

    print(f"\nPredictions for non-Sci-Fi movies:")
    non_scifi_liked_avg = np.mean([liked_scores[m] for m in test_other[:20]])
    non_scifi_disliked_avg = np.mean([disliked_scores[m] for m in test_other[:20]])
    print(f"  When Sci-Fi LIKED:    {non_scifi_liked_avg:.4f}")
    print(f"  When Sci-Fi DISLIKED: {non_scifi_disliked_avg:.4f}")
    non_scifi_diff = non_scifi_liked_avg - non_scifi_disliked_avg
    print(f"  Difference: {non_scifi_diff:+.4f}")

    lift_diff = scifi_diff - non_scifi_diff
    print(f"\n  CRITICAL: Sci-Fi lift vs non-Sci-Fi lift: {lift_diff:+.4f}")
    if lift_diff > 0.05:
        print("  >>> GOOD: Model gives Sci-Fi higher lift when user likes Sci-Fi!")
    elif lift_diff > 0.01:
        print("  >>> OK: Some lift difference detected")
    else:
        print("  >>> PROBLEM: No meaningful lift difference!")

    # =============================================================================
    # SANITY TEST 2: Top-10 overlap - Sci-Fi fan vs Romance fan (multiple movies)
    # =============================================================================
    print("\n" + "=" * 80)
    print("SANITY TEST 2: Top-10 Overlap (Sci-Fi fan vs Romance fan)")
    print("=" * 80)

    # Get Romance movies
    romance_movie_idxs = [m for m, g in movie_genres_map.items() if romance_idx in g]
    print(f"Romance movies: {len(romance_movie_idxs)}")

    # Reveal 5 Sci-Fi movies as liked
    test_scifi_movies = scifi_movie_idxs[:5]
    ratings_scifi_fan = torch.zeros(1, N_ITEMS, 3)
    ratings_scifi_fan[:, :, 2] = 1
    for m in test_scifi_movies:
        ratings_scifi_fan[:, m, 2] = 0
        ratings_scifi_fan[:, m, 1] = 1
    ratings_scifi_fan[:, N_MOVIES + scifi_idx, 2] = 0
    ratings_scifi_fan[:, N_MOVIES + scifi_idx, 1] = 1

    # Reveal 5 Romance movies as liked
    test_romance_movies = romance_movie_idxs[:5]
    ratings_romance_fan = torch.zeros(1, N_ITEMS, 3)
    ratings_romance_fan[:, :, 2] = 1
    for m in test_romance_movies:
        ratings_romance_fan[:, m, 2] = 0
        ratings_romance_fan[:, m, 1] = 1
    ratings_romance_fan[:, N_MOVIES + romance_idx, 2] = 0
    ratings_romance_fan[:, N_MOVIES + romance_idx, 1] = 1

    with torch.no_grad():
        scifi_scores = model(indices, ratings_scifi_fan)[:, -1, :N_MOVIES].sigmoid().numpy().flatten()
        romance_scores = model(indices, ratings_romance_fan)[:, -1, :N_MOVIES].sigmoid().numpy().flatten()

    # Exclude revealed movies from ranking
    all_revealed = set(test_scifi_movies + test_romance_movies)
    available_movies = [m for m in range(N_MOVIES) if m not in all_revealed]

    scifi_ranking = sorted(available_movies, key=lambda m: scifi_scores[m], reverse=True)
    romance_ranking = sorted(available_movies, key=lambda m: romance_scores[m], reverse=True)

    scifi_top10 = set(scifi_ranking[:10])
    romance_top10 = set(romance_ranking[:10])
    overlap = len(scifi_top10 & romance_top10)

    print(f"\nTop-10 when liking Sci-Fi movies vs Romance movies:")
    print(f"  Overlap: {overlap}/10")

    print("\nSci-Fi fan top-10:")
    for idx in scifi_ranking[:5]:
        genres = [all_genres[g] for g in movie_genres_map.get(idx, [])]
        print(f"  {movie_titles.get(idx, f'Movie {idx}')[:40]:40s} | {genres}")

    print("\nRomance fan top-10:")
    for idx in romance_ranking[:5]:
        genres = [all_genres[g] for g in movie_genres_map.get(idx, [])]
        print(f"  {movie_titles.get(idx, f'Movie {idx}')[:40]:40s} | {genres}")

    if overlap >= 7:
        print("\n  >>> PROBLEM: High overlap - genres not affecting recommendations!")
    elif overlap >= 5:
        print("\n  >>> WARNING: Moderate overlap")
    else:
        print("\n  >>> GOOD: Low overlap - different genres give different recommendations!")

    # =============================================================================
    # SANITY TEST 3: Held-out Rating Prediction (Paper's actual evaluation)
    # =============================================================================
    print("\n" + "=" * 80)
    print("SANITY TEST 3: Held-out Rating Prediction")
    print("=" * 80)

    # For each validation user, hide one rating and predict it
    correct_liked = 0
    correct_disliked = 0
    total_liked = 0
    total_disliked = 0

    for user_id in val_users[:20]:  # Test on 20 users
        user_ratings = val_df[val_df['userId'] == user_id]
        if len(user_ratings) < 5:
            continue

        user_items = user_ratings['itemIdx'].tolist()
        user_scores = user_ratings['rating'].tolist()

        # Hold out last rating
        holdout_idx = user_items[-1]
        holdout_rating = user_scores[-1]
        train_items = user_items[:-1]
        train_scores = user_scores[:-1]

        # Build input
        ratings_input = torch.zeros(1, N_ITEMS, 3)
        ratings_input[:, :, 2] = 1  # All unknown

        for item_idx, rating in zip(train_items, train_scores):
            ratings_input[:, item_idx, 2] = 0
            if rating >= 4:
                ratings_input[:, item_idx, 1] = 1
            else:
                ratings_input[:, item_idx, 0] = 1

            # Also set genre ratings
            if item_idx < N_MOVIES and item_idx in movie_genres_map:
                for genre_idx in movie_genres_map[item_idx]:
                    genre_item_idx = N_MOVIES + genre_idx
                    ratings_input[:, genre_item_idx, 2] = 0
                    if rating >= 4:
                        ratings_input[:, genre_item_idx, 1] = 1
                    else:
                        ratings_input[:, genre_item_idx, 0] = 1

        # Predict
        with torch.no_grad():
            out = model(indices, ratings_input)[:, -1, :N_ITEMS].sigmoid()
            pred_score = out[0, holdout_idx].item()

        # Check: did we predict correctly?
        true_liked = holdout_rating >= 4
        pred_liked = pred_score > 0.5

        if true_liked:
            total_liked += 1
            if pred_liked:
                correct_liked += 1
        else:
            total_disliked += 1
            if not pred_liked:
                correct_disliked += 1

    print(f"\nHeld-out prediction (20 users):")
    print(f"  Liked items: {correct_liked}/{total_liked} correct ({100*correct_liked/max(total_liked,1):.1f}%)")
    print(f"  Disliked items: {correct_disliked}/{total_disliked} correct ({100*correct_disliked/max(total_disliked,1):.1f}%)")
    total_acc = (correct_liked + correct_disliked) / max(total_liked + total_disliked, 1)
    print(f"  Overall accuracy: {total_acc*100:.1f}%")

    # Compare to random baseline
    like_rate = total_liked / max(total_liked + total_disliked, 1)
    random_acc = max(like_rate, 1-like_rate)
    print(f"  Random baseline (always predict majority): {random_acc*100:.1f}%")
    if total_acc > random_acc + 0.1:
        print("  >>> GOOD: Model beats random baseline significantly!")
    elif total_acc > random_acc:
        print("  >>> OK: Model slightly beats random baseline")
    else:
        print("  >>> PROBLEM: Model doesn't beat random baseline!")

    # =============================================================================
    # Summary
    # =============================================================================
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"""
    Architecture: ExtrapolationModel with Genre Attributes
    - {N_MOVIES} movies + {N_GENRES} genres = {N_ITEMS} items
    - {len(active_users)} users, {N_EPOCHS} epochs
    - Genre ratings inferred from movie ratings (median)

    Key Test Results:
    - Sci-Fi lift difference (liked vs disliked): {lift_diff:+.4f}
    - Top-10 overlap (Sci-Fi vs Romance fans): {overlap}/10
    - Held-out prediction accuracy: {total_acc*100:.1f}% (vs {random_acc*100:.1f}% random)

    Target:
    - Lift difference > 0.05
    - Top-10 overlap < 5/10
    - Held-out accuracy > random + 10%
    """)


if __name__ == "__main__":
    main()
