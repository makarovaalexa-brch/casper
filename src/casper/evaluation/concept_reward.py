"""
Concept model-based reward calculation for CASPER training.

Uses the saved ConceptEmbeddingModel (SBERT+LSTM+Attention) to predict
user preferences and calculates accuracy as the reward signal.

Reward = Accuracy(t) - Accuracy(t-1)

Key advantages over LLM-based reward:
- No API calls (fast, free, deterministic)
- Accuracy has SNR=1.69 (vs NDCG SNR=0.81) — more learnable signal
- Perfectly monotonic (20/20 steps improve) — consistent gradient
"""

import re
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, List, Optional, Tuple


class ConceptEmbeddingModel(nn.Module):
    """ConceptEmbeddingModel from RS_Model_Comparison notebook."""

    def __init__(self, n_items, embedding_dim=384, hidden_dim=None):
        super().__init__()
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
        enc, _ = self.lstm(x)
        x = torch.relu(self.dense1(enc))
        x = torch.relu(self.dense2(x))
        att, _ = self.attention(enc, x, x)
        return self.output(torch.cat((x, att), dim=-1))


class ConceptModelRewardCalculator:
    """
    Calculate rewards using Concept model accuracy.

    Drop-in replacement for LLMRewardCalculator with same API.
    Uses saved ConceptEmbeddingModel checkpoint for fast, deterministic
    predictions. Returns accuracy (fraction of rated items correctly
    predicted) instead of NDCG.
    """

    def __init__(
        self,
        data_path: str,
        top_n_movies: int = 100,
        checkpoint_name: str = 'concept_paper_config.pt',
        embeddings_name: str = 'concept_embeddings_paper_config.npy',
    ):
        self.data_path = Path(data_path)
        self.top_n_movies = top_n_movies

        # Load model first to get n_items
        self._load_model(checkpoint_name, embeddings_name)

        # Load movie data with correct n_items
        self._load_movie_data()

        # Current user ground truth (set by prepare_user_for_evaluation)
        self._current_gt = None

    def _load_model(self, checkpoint_name, embeddings_name):
        """Load concept model checkpoint and embeddings."""
        ckpt_dir = self.data_path / '.cache' / 'checkpoints'

        ckpt = torch.load(
            ckpt_dir / checkpoint_name,
            map_location='cpu',
            weights_only=False,
        )

        self.n_items = ckpt['n_items']
        hidden_dim = ckpt['model_state_dict']['concept_proj.bias'].shape[0]

        self.model = ConceptEmbeddingModel(self.n_items, 384, hidden_dim)
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.model.eval()

        self.item_embeddings = np.load(ckpt_dir / embeddings_name)

        print(
            f"  Concept model loaded: n_items={self.n_items}, hidden={hidden_dim}, "
            f"epochs={ckpt.get('epochs', 0)}"
        )

    def _load_movie_data(self):
        """Load movie data and build title-to-index mappings."""
        movies_df = pd.read_csv(self.data_path / 'movies.csv')
        ratings_df = pd.read_csv(
            self.data_path / 'ratings.csv',
            dtype={'userId': 'int32', 'movieId': 'int32', 'rating': 'float32'},
            usecols=['userId', 'movieId', 'rating'],
        )

        movie_counts = ratings_df.groupby('movieId').size().reset_index(name='count')
        movies_merged = movies_df.merge(movie_counts, on='movieId', how='left').fillna(0)

        # Full model item list (n_items=187, matching checkpoint)
        self.all_items_df = movies_merged.nlargest(self.n_items, 'count')
        all_movie_ids = self.all_items_df['movieId'].tolist()
        self.movie_id_to_idx = {mid: i for i, mid in enumerate(all_movie_ids)}

        # Title -> model index mappings (all 187 items)
        self.title_to_idx = {}
        for _, row in self.all_items_df.iterrows():
            idx = self.movie_id_to_idx[row['movieId']]
            self.title_to_idx[row['title']] = idx
            normalized = self._normalize_title(row['title'])
            self.title_to_idx[normalized] = idx

        # For compatibility with LLMRewardCalculator interface
        self.top_movies_df = self.all_items_df.head(self.top_n_movies)
        self.movie_list = [row['title'] for _, row in self.top_movies_df.iterrows()]

        # movie_id_map covers all items (needed for get_movie_id)
        self.movie_id_map = {
            row['title']: row['movieId']
            for _, row in self.all_items_df.iterrows()
        }
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

    def _title_to_model_idx(self, title: str) -> Optional[int]:
        """Map movie title to concept model index."""
        if title in self.title_to_idx:
            return self.title_to_idx[title]
        normalized = self._normalize_title(title)
        return self.title_to_idx.get(normalized)

    def _predict(self, revealed: list) -> np.ndarray:
        """Run concept model prediction.

        Args:
            revealed: list of (model_idx, is_liked_bool)

        Returns:
            Sigmoid predictions for top_n_movies items.
        """
        item_emb = torch.FloatTensor(self.item_embeddings).unsqueeze(0)
        rating_input = torch.zeros(1, self.n_items, 3)
        rating_input[:, :, 2] = 1  # All start as "not seen"

        for idx, is_liked in revealed:
            if idx < self.n_items:
                rating_input[:, idx, 2] = 0
                rating_input[:, idx, 1 if is_liked else 0] = 1

        with torch.no_grad():
            preds = self.model(item_emb, rating_input)
            return preds[:, -1, :self.top_n_movies].sigmoid().numpy().flatten()

    def _calc_accuracy(self, preds: np.ndarray, gt: np.ndarray) -> float:
        """Compute accuracy over rated items."""
        mask = ~np.isnan(gt)
        if mask.sum() == 0:
            return 0.5  # No rated items, return chance level

        y_true = gt[mask]
        y_pred = preds[mask]
        return float(np.mean((y_pred > 0.5) == y_true))

    # ─── LLMRewardCalculator-compatible API ────────────────────────────

    def prepare_user_for_evaluation(
        self,
        user_profile: Dict,
        holdout_ratio: float = 0.0,
        min_rating: float = 4.0,
        seed: int = None,
    ) -> Tuple[Dict, List[int], List[str]]:
        """
        Prepare user profile and build ground truth vector.

        Ground truth: 1.0 if rating >= min_rating, 0.0 otherwise,
        NaN for unrated items. Evaluated over top_n_movies items.
        """
        gt = np.full(self.top_n_movies, np.nan)
        eval_target_ids = []
        eval_movie_titles = []

        for movie, rating in user_profile.get('ratings', {}).items():
            idx = self._title_to_model_idx(movie)
            movie_id = self.get_movie_id(movie)

            if idx is not None and idx < self.top_n_movies:
                gt[idx] = 1.0 if rating >= min_rating else 0.0
                if movie_id is not None:
                    eval_target_ids.append(movie_id)
                    eval_movie_titles.append(movie)

        # Store ground truth for calculate_turn_reward
        self._current_gt = gt

        return user_profile, eval_target_ids, eval_movie_titles

    def calculate_turn_reward(
        self,
        preferences: Dict,
        eval_target_ids: List[int],
        prev_ndcg: float,
        top_k: int = 10,
    ) -> Tuple[float, float]:
        """
        Calculate accuracy-based reward for a single turn.

        Returns (current_accuracy, reward) where reward = accuracy - prev.
        Parameter names kept for API compatibility (prev_ndcg = prev_accuracy).
        """
        if self._current_gt is None:
            return 0.5, 0.0

        # Convert preferences to revealed list
        revealed = []
        for title in preferences.get('liked', []):
            idx = self._title_to_model_idx(title)
            if idx is not None:
                revealed.append((idx, True))
        for title in preferences.get('disliked', []):
            idx = self._title_to_model_idx(title)
            if idx is not None:
                revealed.append((idx, False))

        preds = self._predict(revealed)
        current_acc = self._calc_accuracy(preds, self._current_gt)
        reward = current_acc - prev_ndcg

        return current_acc, reward

    def calculate_baseline_ndcg(
        self, eval_target_ids: List[int], top_k: int = 10
    ) -> float:
        """Calculate baseline accuracy (no preferences revealed)."""
        return self.calculate_turn_reward(
            preferences={}, eval_target_ids=eval_target_ids, prev_ndcg=0.0
        )[0]

    def count_eval_targets(
        self, user_profile: Dict, min_rating: float = 4.0
    ) -> int:
        """Count rated movies in the evaluation pool."""
        count = 0
        for movie, rating in user_profile.get('ratings', {}).items():
            idx = self._title_to_model_idx(movie)
            if idx is not None and idx < self.top_n_movies:
                count += 1
        return count

    def filter_users_by_eval_targets(
        self,
        user_profiles: List[Dict],
        min_eval_targets: int = 10,
    ) -> List[Dict]:
        """Filter users with sufficient rated movies in the pool."""
        return [
            p for p in user_profiles
            if self.count_eval_targets(p) >= min_eval_targets
        ]

    def filter_mentioned_movies(
        self,
        eval_target_ids: List[int],
        mentioned_items: List[str],
    ) -> List[int]:
        """Pass-through: concept model accuracy doesn't need filtering."""
        return eval_target_ids

    @staticmethod
    def preferences_to_text(preferences: Dict) -> str:
        """Convert discovered preferences dict to text (for logging)."""
        parts = []
        liked = preferences.get('liked', [])
        disliked = preferences.get('disliked', [])
        if liked:
            parts.append(f"Liked: {', '.join(liked[:10])}")
        if disliked:
            parts.append(f"Disliked: {', '.join(disliked[:5])}")
        return '\n'.join(parts) if parts else "No preferences yet"
