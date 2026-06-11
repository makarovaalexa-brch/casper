"""
LLM-based reward calculation for CASPER training.

Uses GPT-4o-mini to generate recommendations and calculates NDCG
against held-out ground truth movies.

Reward = NDCG(t) - NDCG(t-1)
"""

import json
import hashlib
import re
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import pandas as pd


class LLMRewardCalculator:
    """
    Calculate rewards using LLM-based recommendations.

    Uses GPT to recommend movies based on discovered preferences,
    then calculates NDCG against user's held-out movies.
    """

    def __init__(self, data_path: str, top_n_movies: int = 100, cache_dir: Optional[Path] = None):
        """
        Initialize LLM reward calculator.

        Args:
            data_path: Path to MovieLens data directory
            top_n_movies: Number of top movies to use for LLM recommendation pool
            cache_dir: Directory for caching LLM responses (defaults to data_path/.cache/llm_responses)
        """
        self.data_path = Path(data_path)
        self.top_n_movies = top_n_movies

        # Setup cache
        if cache_dir is None:
            cache_dir = self.data_path / '.cache' / 'llm_responses'
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Load movie data
        self._load_movie_data()

    def _load_movie_data(self):
        """Load and prepare movie data."""
        movies_df = pd.read_csv(self.data_path / 'movies.csv')
        ratings_df = pd.read_csv(self.data_path / 'ratings.csv')

        # Get top N movies by rating count
        movie_counts = ratings_df.groupby('movieId').size().reset_index(name='count')
        movies_merged = movies_df.merge(movie_counts, on='movieId', how='left').fillna(0)
        self.top_movies_df = movies_merged.nlargest(self.top_n_movies, 'count')

        # Movie title list for LLM prompts
        self.movie_list = [row['title'] for _, row in self.top_movies_df.iterrows()]

        # Title -> ID mapping (with year)
        self.movie_id_map = {row['title']: row['movieId'] for _, row in self.top_movies_df.iterrows()}

        # Normalized title mapping (handles with/without year)
        self.movie_id_map_normalized = {}
        for full_title, movie_id in self.movie_id_map.items():
            normalized = self._normalize_title(full_title)
            self.movie_id_map_normalized[normalized] = movie_id
            self.movie_id_map_normalized[full_title] = movie_id

    @staticmethod
    def _normalize_title(title: str) -> str:
        """Strip year suffix like ' (1994)' from movie title."""
        return re.sub(r'\s*\(\d{4}\)\s*$', '', title).strip()

    def get_movie_id(self, title: str) -> Optional[int]:
        """Get movie ID from title (handles with/without year)."""
        if title in self.movie_id_map:
            return self.movie_id_map[title]
        normalized = self._normalize_title(title)
        return self.movie_id_map_normalized.get(normalized)

    def _cache_key(self, prompt: str, model: str = "gpt-4o-mini") -> str:
        """Generate cache key for LLM response."""
        return hashlib.sha256(f"{model}:{prompt}".encode()).hexdigest()[:16]

    def _call_llm_cached(self, prompt: str, model: str = "gpt-4o-mini") -> Optional[str]:
        """Call LLM with caching to reduce API costs."""
        import openai

        key = self._cache_key(prompt, model)
        cache_file = self.cache_dir / f"{key}.json"

        if cache_file.exists():
            with open(cache_file, 'r') as f:
                return json.load(f).get('response')

        try:
            client = openai.OpenAI()
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=500
            )
            result = resp.choices[0].message.content

            with open(cache_file, 'w') as f:
                json.dump({'model': model, 'prompt': prompt[:200], 'response': result}, f)

            return result
        except Exception as e:
            print(f"LLM error: {e}")
            return None

    def _parse_recommendations(self, response: str) -> List[str]:
        """Parse LLM response into ordered movie list."""
        if not response:
            return []

        rankings = []
        for line in response.split('\n'):
            for movie in self.movie_list:
                # Match by first 20 chars (handles truncation)
                if movie[:20].lower() in line.lower():
                    if movie not in rankings:
                        rankings.append(movie)
                    break
        return rankings

    def get_recommendations(self, preferences_text: str, top_k: int = 10) -> List[int]:
        """
        Get LLM recommendations given user preferences.

        Args:
            preferences_text: Text describing user preferences
            top_k: Number of recommendations to return

        Returns:
            List of movie IDs (ordered by recommendation rank)
        """
        prompt = f"""Based on these movie preferences, recommend the top {top_k} movies this user would enjoy.

User preferences:
{preferences_text}

Choose from these movies ONLY:
{', '.join(self.movie_list[:50])}
{', '.join(self.movie_list[50:])}

Return ONLY a numbered list of {top_k} movie titles, most recommended first."""

        response = self._call_llm_cached(prompt)
        recommendations = self._parse_recommendations(response)

        # Convert to movie IDs
        rec_ids = [self.movie_id_map.get(title) for title in recommendations[:top_k]
                   if title in self.movie_id_map]
        return rec_ids

    @staticmethod
    def calculate_ndcg(recommended_ids: List[int], relevant_ids: List[int], k: int = 10) -> float:
        """
        Calculate NDCG@k.

        Args:
            recommended_ids: Ordered list of recommended movie IDs
            relevant_ids: Set of relevant (ground truth) movie IDs
            k: Cutoff for NDCG calculation

        Returns:
            NDCG@k score (0.0 to 1.0)
        """
        if not relevant_ids:
            return 0.0

        relevant_set = set(relevant_ids)

        # DCG
        dcg = 0.0
        for i, rec_id in enumerate(recommended_ids[:k]):
            if rec_id in relevant_set:
                dcg += 1.0 / np.log2(i + 2)

        # IDCG (ideal: all relevant items at top)
        idcg = sum(1.0 / np.log2(i + 2) for i in range(min(k, len(relevant_ids))))

        return dcg / idcg if idcg > 0 else 0.0

    def get_eval_target_ids(self, held_out_titles: List[str],
                            training_ratings: Dict[str, float]) -> List[int]:
        """
        Get evaluation target IDs from held-out movies or high-rated training movies.

        Args:
            held_out_titles: Titles of held-out movies
            training_ratings: Dict of movie title -> rating from training set

        Returns:
            List of movie IDs to use as evaluation targets
        """
        # Try held-out movies first
        held_out_ids = [self.get_movie_id(title) for title in held_out_titles]
        held_out_ids = [mid for mid in held_out_ids if mid is not None]

        if held_out_ids:
            return held_out_ids

        # Fallback: high-rated movies from training set
        return [
            self.get_movie_id(movie) for movie, rating in training_ratings.items()
            if rating >= 4.0 and self.get_movie_id(movie) is not None
        ]

    @staticmethod
    def preferences_to_text(preferences: Dict) -> str:
        """Convert discovered preferences dict to text for LLM."""
        parts = []
        liked = preferences.get('liked', [])
        disliked = preferences.get('disliked', [])
        not_seen = preferences.get('not_seen', [])

        if liked:
            parts.append(f"Liked: {', '.join(liked[:10])}")
        if disliked:
            parts.append(f"Disliked: {', '.join(disliked[:5])}")
        if not_seen:
            parts.append(f"Haven't seen: {', '.join(not_seen[:5])}")

        return '\n'.join(parts) if parts else "No preferences yet"

    def calculate_turn_reward(self, preferences: Dict, eval_target_ids: List[int],
                              prev_ndcg: float, top_k: int = 10) -> Tuple[float, float]:
        """
        Calculate reward for a single turn.

        Args:
            preferences: Current discovered preferences dict
            eval_target_ids: Ground truth movie IDs for evaluation
            prev_ndcg: NDCG from previous turn
            top_k: Number of recommendations

        Returns:
            (current_ndcg, reward) tuple where reward = current_ndcg - prev_ndcg
        """
        pref_text = self.preferences_to_text(preferences)
        rec_ids = self.get_recommendations(pref_text, top_k=top_k)
        current_ndcg = self.calculate_ndcg(rec_ids, eval_target_ids, k=top_k)
        reward = current_ndcg - prev_ndcg
        return current_ndcg, reward

    def calculate_baseline_ndcg(self, eval_target_ids: List[int], top_k: int = 10) -> float:
        """
        Calculate baseline NDCG (with no preferences).

        This is the NDCG you'd get before asking any questions.
        """
        return self.calculate_turn_reward(
            preferences={},
            eval_target_ids=eval_target_ids,
            prev_ndcg=0.0,
            top_k=top_k
        )[0]

    def prepare_user_for_evaluation(
        self,
        user_profile: Dict,
        holdout_ratio: float = 0.0,  # Default: no holdout, use all
        min_rating: float = 4.0,
        seed: int = None
    ) -> Tuple[Dict, List[int], List[str]]:
        """
        Prepare user profile for evaluation.

        Default behavior (holdout_ratio=0): Use ALL user's high-rated movies
        in TOP_N as ground truth. This gives more eval targets.

        The episode runner should exclude movies explicitly mentioned in
        conversation from the eval targets dynamically.

        Args:
            user_profile: User profile with 'ratings' dict
            holdout_ratio: Fraction to hold out (0 = use all)
            min_rating: Minimum rating to consider as "liked"
            seed: Random seed for reproducible splits

        Returns:
            (training_profile, eval_target_ids, eval_movie_titles) tuple
        """
        # Get user's high-rated movies that are IN TOP_N
        high_rated_in_top = []
        for movie, rating in user_profile.get('ratings', {}).items():
            if rating >= min_rating:
                movie_id = self.get_movie_id(movie)
                if movie_id is not None:
                    high_rated_in_top.append((movie, movie_id, rating))

        if not high_rated_in_top:
            return user_profile, [], []

        if holdout_ratio > 0:
            # Split mode: hold out some movies
            rng = np.random.RandomState(seed)
            rng.shuffle(high_rated_in_top)
            split_point = max(1, int(len(high_rated_in_top) * (1 - holdout_ratio)))
            eval_movies = high_rated_in_top[split_point:]
        else:
            # No holdout: use all high-rated movies as ground truth
            eval_movies = high_rated_in_top

        eval_target_ids = [movie_id for _, movie_id, _ in eval_movies]
        eval_movie_titles = [movie for movie, _, _ in eval_movies]

        # Training profile is the full profile (user sim can respond about any movie)
        return user_profile, eval_target_ids, eval_movie_titles

    def count_eval_targets(self, user_profile: Dict, min_rating: float = 4.0) -> int:
        """
        Count how many eval targets a user has (high-rated movies in TOP_N).

        Use this to pre-filter users before training.

        Args:
            user_profile: User profile with 'ratings' dict
            min_rating: Minimum rating to consider as "liked"

        Returns:
            Number of eval targets
        """
        count = 0
        for movie, rating in user_profile.get('ratings', {}).items():
            if rating >= min_rating:
                if self.get_movie_id(movie) is not None:
                    count += 1
        return count

    def filter_users_by_eval_targets(
        self,
        user_profiles: List[Dict],
        min_eval_targets: int = 10
    ) -> List[Dict]:
        """
        Filter users to only those with sufficient eval targets.

        Args:
            user_profiles: List of user profiles
            min_eval_targets: Minimum number of high-rated movies in TOP_N

        Returns:
            Filtered list of user profiles
        """
        filtered = []
        for profile in user_profiles:
            count = self.count_eval_targets(profile)
            if count >= min_eval_targets:
                filtered.append(profile)
        return filtered

    def filter_mentioned_movies(
        self,
        eval_target_ids: List[int],
        mentioned_items: List[str]
    ) -> List[int]:
        """
        Remove mentioned movies from eval targets.

        Call this during episode to exclude movies user explicitly
        talked about (don't give credit for obvious recommendations).

        Args:
            eval_target_ids: Original eval target IDs
            mentioned_items: Items mentioned in conversation (from discovered_preferences)

        Returns:
            Filtered eval target IDs
        """
        mentioned_ids = set()
        for item in mentioned_items:
            movie_id = self.get_movie_id(item)
            if movie_id is not None:
                mentioned_ids.add(movie_id)

        return [mid for mid in eval_target_ids if mid not in mentioned_ids]
