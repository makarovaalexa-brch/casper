"""
MovieLens data loader for CASPER.

Loads MovieLens 25M ratings and creates train/test user splits.
"""

import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple


class MovieLensLoader:
    """
    Load and process MovieLens 25M dataset.

    Provides train/test user profiles for RL training and evaluation.
    """

    def __init__(self, data_path: str, min_ratings: int = 20):
        """
        Initialize MovieLens loader.

        Args:
            data_path: Path to MovieLens dataset directory
            min_ratings: Minimum ratings per user to include
        """
        self.data_path = Path(data_path)
        self.min_ratings = min_ratings

        # Load movies metadata
        movies_path = self.data_path / "movies.csv"
        if not movies_path.exists():
            raise FileNotFoundError(
                f"movies.csv not found at {data_path}. "
                "Please download MovieLens 25M from https://grouplens.org/datasets/movielens/25m/"
            )

        self.movies_df = pd.read_csv(movies_path)

        # Create item index mapping
        self.item_to_idx = {
            int(row['movieId']): idx
            for idx, row in self.movies_df.iterrows()
        }
        self.idx_to_item = {v: k for k, v in self.item_to_idx.items()}

    def load_data(self, test_split: float = 0.3) -> Tuple[List[Dict], List[Dict]]:
        """
        Load and split MovieLens data into train/test users.

        Args:
            test_split: Fraction of users for testing (default: 0.3)

        Returns:
            (train_users, test_users): Lists of user profile dicts

        Each user profile dict has:
            - user_id: int
            - ratings: Dict[str, float] (movie_title -> rating)
        """
        # Try to load cached train/test split first (ensures deterministic splits across runs)
        split_cache_path = self.data_path.parent / "processed" / f"train_test_split_{self.min_ratings}_{test_split}.pkl"

        if split_cache_path.exists():
            print(f"Loading cached train/test split from {split_cache_path}", flush=True)
            import pickle
            with open(split_cache_path, 'rb') as f:
                cached_data = pickle.load(f)
            print(f"  Train users: {len(cached_data['train_users'])}", flush=True)
            print(f"  Test users: {len(cached_data['test_users'])}", flush=True)
            return cached_data['train_users'], cached_data['test_users']

        # Try to load from cache first (much faster)
        cache_path = self.data_path.parent / "processed" / "ratings_subset.pkl"

        if cache_path.exists():
            print(f"Loading from cache: {cache_path}", flush=True)
            ratings_df = pd.read_pickle(cache_path)
            print(f"Cached ratings loaded: {len(ratings_df)} rows", flush=True)
        else:
            # Load ratings from CSV (slow)
            ratings_path = self.data_path / "ratings.csv"
            if not ratings_path.exists():
                raise FileNotFoundError(f"ratings.csv not found at {self.data_path}")

            print(f"Loading ratings from {ratings_path}...", flush=True)
            ratings_df = pd.read_csv(ratings_path)
            print(f"Ratings loaded: {len(ratings_df)} rows", flush=True)

        # Filter users with sufficient ratings
        print(f"Filtering users...", flush=True)
        user_counts = ratings_df['userId'].value_counts()
        eligible_users = user_counts[user_counts >= self.min_ratings].index.tolist()

        print(f"Found {len(eligible_users)} users with >={self.min_ratings} ratings", flush=True)

        # DETERMINISTIC split: sort user IDs to ensure consistent ordering
        eligible_users_sorted = sorted(eligible_users)

        # Split into train/test
        test_size = int(len(eligible_users_sorted) * test_split)
        test_user_ids = eligible_users_sorted[:test_size]
        train_user_ids = eligible_users_sorted[test_size:]

        # Create user profiles
        train_users = self._create_profiles(ratings_df, train_user_ids)
        test_users = self._create_profiles(ratings_df, test_user_ids)

        # Cache the split for future runs
        split_cache_path.parent.mkdir(parents=True, exist_ok=True)
        import pickle
        with open(split_cache_path, 'wb') as f:
            pickle.dump({
                'train_users': train_users,
                'test_users': test_users,
                'min_ratings': self.min_ratings,
                'test_split': test_split
            }, f)
        print(f"Train/test split cached to {split_cache_path}", flush=True)

        return train_users, test_users

    def _create_profiles(self, ratings_df: pd.DataFrame, user_ids: List[int]) -> List[Dict]:
        """
        Create user profiles from ratings data.

        Returns list of dicts with structure:
        {
            'user_id': int,
            'ratings': {movie_title: rating, ...}
        }
        """
        # Filter to selected users
        user_ratings = ratings_df[ratings_df['userId'].isin(user_ids)]

        # Merge with movie titles once
        user_movies = user_ratings.merge(
            self.movies_df[['movieId', 'title']],
            on='movieId',
            how='left'
        )

        # Group by user
        profiles = []
        for user_id, group in user_movies.groupby('userId'):
            # Build ratings dict
            ratings_dict = {}
            for _, row in group.iterrows():
                if pd.notna(row['title']):
                    # Clean title (remove year)
                    clean_title = row['title'].split('(')[0].strip()
                    ratings_dict[clean_title] = float(row['rating'])

            profile = {
                'user_id': int(user_id),
                'ratings': ratings_dict
            }
            profiles.append(profile)

        return profiles
