"""
Exact replication of the CSMAI-19 ExtrapolationModel.

Key differences from my two-tower approach:
1. Output: Softmax over ALL items (not user embedding + dot product)
2. Loss: BCE with NaN masking (not InfoNCE/contrastive)
3. Architecture: Learned embeddings for items (not SBERT)
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
    """
    Builds a model given movie vector dimension.
    Exact copy from the paper's notebook.
    """
    def __init__(self, n_items, lr=0.001):
        super(ExtrapolationModel, self).__init__()
        self.n_items = n_items
        hidden_dim = n_items // 2

        self.embedding = nn.Embedding(n_items + 1, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim + 3, hidden_dim, 1, batch_first=True, bidirectional=False)
        self.dense1 = nn.Linear(hidden_dim, n_items)
        self.dense2 = nn.Linear(n_items, hidden_dim)
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=1, batch_first=True)
        self.output = nn.Linear(n_items, n_items * 2)  # explicit + implicit

        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    def forward(self, index_input, rating_input):
        """
        index_input: [batch, seq_len] - item indices
        rating_input: [batch, seq_len, 3] - one-hot ratings [disliked, liked, not_seen]

        Returns: [batch, seq_len, n_items*2] - predictions for all items at each timestep
        """
        # Encoder
        x = self.embedding(index_input)  # [batch, seq, hidden]
        x = torch.cat((x, rating_input), dim=-1)  # [batch, seq, hidden+3]
        encoder_output, _ = self.lstm(x)  # [batch, seq, hidden]

        # Decoder
        x = torch.relu(self.dense1(encoder_output))  # [batch, seq, n_items]
        x = torch.relu(self.dense2(x))  # [batch, seq, hidden]
        attention, _ = self.attention(encoder_output, x, x)  # [batch, seq, hidden]
        x = torch.cat((x, attention), dim=-1)  # [batch, seq, n_items]
        x = self.output(x)  # [batch, seq, n_items*2]

        return x


class BCEWithLogitsLossNan(nn.Module):
    """BCE loss that masks NaN outputs."""
    def forward(self, y_pred, y_true):
        mask = torch.isnan(y_true)
        y_pred = torch.where(mask, torch.zeros_like(y_true), y_pred)
        y_true = torch.where(mask, torch.zeros_like(y_true), y_true)
        return torch.nn.functional.binary_cross_entropy_with_logits(y_pred, y_true)


# =============================================================================
# Dataset
# =============================================================================

class MovieLensDataset(Dataset):
    """
    Simple MovieLens dataset for testing.
    """
    def __init__(self, ratings_df, n_items):
        self.ratings_df = ratings_df
        self.n_items = n_items
        self.user_ids = ratings_df['userId'].unique()

    def __len__(self):
        return len(self.user_ids)

    def __getitem__(self, idx):
        user_id = self.user_ids[idx]
        user_ratings = self.ratings_df[self.ratings_df['userId'] == user_id]

        # Create full rating vector (NaN for unseen)
        full_ratings = np.full(self.n_items, np.nan)
        for _, row in user_ratings.iterrows():
            item_idx = int(row['itemIdx'])
            # Convert 1-5 rating to 0/1: >=4 is liked (1), <4 is disliked (0)
            full_ratings[item_idx] = 1.0 if row['rating'] >= 4 else 0.0

        # Shuffle item order (attention will handle order-invariance)
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

        # Output: explicit ratings tiled for each timestep + implicit (seen/not seen)
        explicit = np.tile(full_ratings, (self.n_items, 1))  # [seq, n_items]
        implicit = (~np.isnan(explicit)).astype(float)
        output = np.concatenate([explicit, implicit], axis=1)  # [seq, n_items*2]

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
    print("EXACT PAPER REPLICATION TEST")
    print("=" * 80)

    # Load MovieLens data
    DATA_DIR = Path('C:/dev/phd/casper/data/movielens')
    ratings = pd.read_csv(DATA_DIR / 'ratings.csv')
    movies = pd.read_csv(DATA_DIR / 'movies.csv')

    # Use top N most-rated movies for speed
    N_ITEMS = 100  # Small for fast iteration
    N_USERS = 200

    # Get top N most-rated movies
    movie_counts = ratings['movieId'].value_counts()
    top_movies = movie_counts.head(N_ITEMS).index.tolist()

    # Filter ratings to only include top movies
    ratings_filtered = ratings[ratings['movieId'].isin(top_movies)]

    # Create movie index mapping
    movie_to_idx = {mid: i for i, mid in enumerate(top_movies)}
    ratings_filtered = ratings_filtered.copy()
    ratings_filtered['itemIdx'] = ratings_filtered['movieId'].map(movie_to_idx)

    # Get users with enough ratings
    user_counts = ratings_filtered['userId'].value_counts()
    active_users = user_counts[user_counts >= 10].index.tolist()[:N_USERS]
    ratings_filtered = ratings_filtered[ratings_filtered['userId'].isin(active_users)]

    print(f"Dataset: {N_ITEMS} items, {len(active_users)} users")
    print(f"Total ratings: {len(ratings_filtered)}")

    # Split train/val
    train_users = active_users[:int(0.8 * len(active_users))]
    val_users = active_users[int(0.8 * len(active_users)):]

    train_df = ratings_filtered[ratings_filtered['userId'].isin(train_users)]
    val_df = ratings_filtered[ratings_filtered['userId'].isin(val_users)]

    train_dataset = MovieLensDataset(train_df, N_ITEMS)
    val_dataset = MovieLensDataset(val_df, N_ITEMS)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    # Create model
    model = ExtrapolationModel(N_ITEMS, lr=0.001)
    loss_fn = BCEWithLogitsLossNan()

    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Hidden dim: {N_ITEMS // 2}")

    # Train
    print("\n" + "=" * 80)
    print("TRAINING")
    print("=" * 80)

    N_EPOCHS = 150  # Try much more epochs

    for epoch in range(N_EPOCHS):
        model.train()
        train_loss = 0
        for batch_idx, (indices, ratings, targets) in enumerate(train_loader):
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

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

    # =============================================================================
    # SANITY TEST: Likes vs Dislikes (with MULTIPLE ratings for stronger signal)
    # =============================================================================
    print("\n" + "=" * 80)
    print("SANITY TEST: Likes vs Dislikes")
    print("=" * 80)

    model.eval()

    # Test with multiple ratings revealed for stronger signal
    # Scenario: User likes items 0-9 vs User dislikes items 0-9
    indices = torch.arange(N_ITEMS).unsqueeze(0)

    N_REVEALED = 10

    # Version 1: User LIKES items 0 to N_REVEALED-1
    ratings_liked = torch.zeros(1, N_ITEMS, 3)
    ratings_liked[:, :, 2] = 1  # All unknown
    for i in range(N_REVEALED):
        ratings_liked[:, i, 2] = 0  # Not unknown
        ratings_liked[:, i, 1] = 1  # Liked

    # Version 2: User DISLIKES items 0 to N_REVEALED-1
    ratings_disliked = torch.zeros(1, N_ITEMS, 3)
    ratings_disliked[:, :, 2] = 1  # All unknown
    for i in range(N_REVEALED):
        ratings_disliked[:, i, 2] = 0  # Not unknown
        ratings_disliked[:, i, 0] = 1  # Disliked

    with torch.no_grad():
        out_liked = model(indices, ratings_liked)[:, -1, :N_ITEMS].sigmoid()
        out_disliked = model(indices, ratings_disliked)[:, -1, :N_ITEMS].sigmoid()

    liked_scores = out_liked.numpy().flatten()
    disliked_scores = out_disliked.numpy().flatten()

    # Compare predictions for UN-REVEALED items only
    unrevealed_liked = liked_scores[N_REVEALED:]
    unrevealed_disliked = disliked_scores[N_REVEALED:]

    liked_top10 = set(np.argsort(unrevealed_liked)[-10:])
    disliked_top10 = set(np.argsort(unrevealed_disliked)[-10:])
    overlap = len(liked_top10 & disliked_top10)

    score_diff = np.abs(unrevealed_liked - unrevealed_disliked).mean()

    print(f"\n  Revealed {N_REVEALED} items as liked vs disliked")
    print(f"  Predictions for OTHER {N_ITEMS - N_REVEALED} items:")
    print(f"    Top-10 overlap: {overlap}/10")
    print(f"    Avg score diff: {score_diff:.4f}")

    # Also check: are predictions for REVEALED items different?
    revealed_liked = liked_scores[:N_REVEALED]
    revealed_disliked = disliked_scores[:N_REVEALED]
    revealed_diff = np.mean(revealed_liked - revealed_disliked)
    print(f"\n  Predictions for REVEALED items (should differ):")
    print(f"    Liked mean: {np.mean(revealed_liked):.4f}")
    print(f"    Disliked mean: {np.mean(revealed_disliked):.4f}")
    print(f"    Diff: {revealed_diff:+.4f}")

    # Now test single-item sensitivity
    print("\n  Single-item test (weaker signal):")
    test_items = [0, 10, 20, 30, 40]
    single_item_diffs = []

    for test_item in test_items:
        indices = torch.arange(N_ITEMS).unsqueeze(0)

        ratings_liked = torch.zeros(1, N_ITEMS, 3)
        ratings_liked[:, :, 2] = 1
        ratings_liked[:, test_item, 2] = 0
        ratings_liked[:, test_item, 1] = 1

        ratings_disliked = torch.zeros(1, N_ITEMS, 3)
        ratings_disliked[:, :, 2] = 1
        ratings_disliked[:, test_item, 2] = 0
        ratings_disliked[:, test_item, 0] = 1

        with torch.no_grad():
            out_liked = model(indices, ratings_liked)[:, -1, :N_ITEMS].sigmoid().numpy().flatten()
            out_disliked = model(indices, ratings_disliked)[:, -1, :N_ITEMS].sigmoid().numpy().flatten()

        score_diff = np.abs(out_liked - out_disliked).mean()
        single_item_diffs.append(score_diff)
        print(f"    Item {test_item:3d}: avg score diff = {score_diff:.4f}")

    avg_overlap = overlap  # Use multi-item test

    if avg_overlap >= 7:
        print("  >>> PROBLEM: High overlap - model can't distinguish likes/dislikes!")
    elif avg_overlap >= 5:
        print("  >>> WARNING: Moderate overlap")
    elif avg_overlap >= 3:
        print("  >>> OK: Low-moderate overlap")
    else:
        print("  >>> GOOD: Low overlap - model distinguishes likes/dislikes!")

    # =============================================================================
    # SANITY TEST 2: Sequential Predictions (from paper Fig 2)
    # =============================================================================
    print("\n" + "=" * 80)
    print("SANITY TEST 2: Sequential Predictions")
    print("=" * 80)
    print("Does the model's output change as we reveal more preferences?")

    # Take a random user from validation set
    test_user = val_df[val_df['userId'] == val_users[0]]
    user_items = test_user['itemIdx'].tolist()
    user_ratings = test_user['rating'].tolist()

    print(f"\nUser has {len(user_items)} ratings")

    # Start with all unknown, then reveal one rating at a time
    indices = torch.arange(N_ITEMS).unsqueeze(0)

    # Track predictions for a specific target item
    target_item = user_items[-1]  # Last item (we'll predict this)
    target_rating = user_ratings[-1]
    print(f"Target item: {target_item}, True rating: {target_rating}")

    predictions_over_time = []

    for n_revealed in [0, 1, 3, 5, 10, min(20, len(user_items)-1)]:
        ratings_input = torch.zeros(1, N_ITEMS, 3)
        ratings_input[:, :, 2] = 1  # All unknown

        # Reveal n ratings (excluding target)
        for i in range(min(n_revealed, len(user_items)-1)):
            item_idx = user_items[i]
            if item_idx == target_item:
                continue
            rating = user_ratings[i]
            ratings_input[:, item_idx, 2] = 0  # Not unknown
            if rating >= 4:
                ratings_input[:, item_idx, 1] = 1  # Liked
            else:
                ratings_input[:, item_idx, 0] = 1  # Disliked

        with torch.no_grad():
            out = model(indices, ratings_input)[:, -1, :N_ITEMS].sigmoid()
            pred = out[0, target_item].item()
            predictions_over_time.append((n_revealed, pred))
            print(f"  After {n_revealed:2d} ratings revealed: prediction for target = {pred:.4f}")

    # Check if predictions change
    pred_range = max(p[1] for p in predictions_over_time) - min(p[1] for p in predictions_over_time)
    print(f"\n  Prediction range: {pred_range:.4f}")
    if pred_range < 0.01:
        print("  >>> PROBLEM: Predictions don't change with more info!")
    else:
        print("  >>> OK: Predictions change as preferences are revealed")

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"""
    Architecture: ExtrapolationModel (exact paper copy)
    - {N_ITEMS} items, {N_USERS} users
    - {N_EPOCHS} epochs
    - BCE loss with NaN masking
    - Learned item embeddings (not SBERT)

    Likes vs Dislikes Overlap: {avg_overlap:.1f}/10 ({avg_overlap*10:.0f}%)
    Prediction Range (sequential test): {pred_range:.4f}
    Target: overlap < 30% (3/10), range > 0.01
    """)


if __name__ == "__main__":
    main()
