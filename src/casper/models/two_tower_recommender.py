"""
Two-Tower Recommender for CASPER.

Takes the same dialogue state as RL model (384-dim SentenceBERT) and recommends movies.

STATE ENCODING:
- State = Extracted preferences only (clean, structured)
- LLM extracts from conversation: {"liked": ["Nolan", "Inception"], "disliked": ["horror"]}
- Convert to text: "likes: Nolan, Inception | dislikes: horror"
- Encode with SentenceBERT: Text → 384-dim vector
- No raw conversation text (avoids duplication, cleaner signal)

Architecture:
    User Tower: Dialogue State (384-dim) → User Embedding (128-dim)
    Item Tower: Movie Metadata (384-dim) → Movie Embedding (128-dim)
    Score: Dot Product(user_emb, movie_emb)

Training:
- NOT pretrained
- Trained from scratch using BPR loss on MovieLens data
- Training data: Conversation texts + Liked movies from user profiles
- Loss: BPR (Bayesian Personalized Ranking)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple, Optional
from sentence_transformers import SentenceTransformer
import pandas as pd
from pathlib import Path
import random
import pickle
import hashlib


class UserTower(nn.Module):
    """
    User tower: Maps dialogue state to user embedding.

    Input: Conversation state (384-dim from SentenceBERT)
    Output: User embedding (128-dim)
    """

    def __init__(self, state_dim: int = 384, user_emb_dim: int = 128):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, user_emb_dim),
            nn.LayerNorm(user_emb_dim)  # Normalize for dot product
        )

    def forward(self, state):
        """
        Args:
            state: Dialogue state tensor [batch_size, 384]

        Returns:
            User embedding [batch_size, 128]
        """
        return self.network(state)


class ItemTower(nn.Module):
    """
    Item tower: Maps movie metadata to item embedding.

    Input: Movie title + genres encoded with SentenceBERT (384-dim)
    Output: Item embedding (128-dim)
    """

    def __init__(self, item_dim: int = 384, item_emb_dim: int = 128):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(item_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, item_emb_dim),
            nn.LayerNorm(item_emb_dim)  # Normalize for dot product
        )

    def forward(self, item_features):
        """
        Args:
            item_features: Movie feature tensor [batch_size, 384]

        Returns:
            Item embedding [batch_size, 128]
        """
        return self.network(item_features)


class TwoTowerRecommender(nn.Module):
    """
    Two-Tower recommendation model.

    Shares the same dialogue state representation as RL model.
    """

    def __init__(
        self,
        state_dim: int = 384,
        embedding_dim: int = 128,
        learning_rate: float = 0.001
    ):
        super().__init__()

        self.user_tower = UserTower(state_dim, embedding_dim)
        self.item_tower = ItemTower(state_dim, embedding_dim)

        self.optimizer = torch.optim.Adam(
            self.parameters(),
            lr=learning_rate,
            weight_decay=1e-5
        )

    def forward(self, user_states, item_features):
        """
        Compute scores for user-item pairs.

        Args:
            user_states: [batch_size, 384] dialogue states
            item_features: [batch_size, 384] movie features

        Returns:
            scores: [batch_size] dot product scores
        """
        user_emb = self.user_tower(user_states)
        item_emb = self.item_tower(item_features)

        # Dot product for scoring
        scores = (user_emb * item_emb).sum(dim=1)

        return scores

    def predict_scores(self, user_state, candidate_items):
        """
        Predict scores for all candidate items.

        Args:
            user_state: [384] single dialogue state
            candidate_items: [num_items, 384] candidate movie features

        Returns:
            scores: [num_items] predicted scores
        """
        with torch.no_grad():
            # Expand user state to match batch size
            user_states = user_state.unsqueeze(0).expand(len(candidate_items), -1)

            # Get scores
            scores = self.forward(user_states, candidate_items)

        return scores

    def train_bpr_step(
        self,
        user_states,
        positive_items,
        negative_items
    ):
        """
        Training step with BPR (Bayesian Personalized Ranking) loss.

        BPR assumes positive items should be ranked higher than negative items.

        Args:
            user_states: [batch_size, 384] dialogue states
            positive_items: [batch_size, 384] liked movie features
            negative_items: [batch_size, 384] disliked/random movie features

        Returns:
            loss value
        """
        # Get scores
        pos_scores = self.forward(user_states, positive_items)
        neg_scores = self.forward(user_states, negative_items)

        # BPR loss: encourage pos_score > neg_score
        # loss = -log(sigmoid(pos_score - neg_score))
        loss = -F.logsigmoid(pos_scores - neg_scores).mean()

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
        self.optimizer.step()

        return loss.item()


class MovieCatalog:
    """
    Handles movie metadata and feature encoding.

    Encodes movies using SentenceBERT for the item tower.
    Uses caching to avoid re-encoding movies on subsequent runs.
    """

    def __init__(
        self,
        movielens_data_path: Optional[str] = None,
        encoder: Optional[SentenceTransformer] = None
    ):
        """
        Initialize movie catalog.

        Args:
            movielens_data_path: Path to MovieLens data
            encoder: Optional pre-initialized SentenceTransformer to reuse (for efficiency)
        """
        # Reuse encoder or create new one
        if encoder is not None:
            self.encoder = encoder
            print("  Using shared encoder for MovieCatalog")
        else:
            self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
            print("  Creating new encoder for MovieCatalog")

        # Store data path for caching
        self.data_path = Path(movielens_data_path) if movielens_data_path else None

        # Load MovieLens data
        self.movies = self._load_movies(movielens_data_path)

        # Pre-compute movie embeddings (with caching)
        self._encode_movies()

        print(f"  Loaded {len(self.movies)} movies")

    def _load_movies(self, data_path: Optional[str]) -> pd.DataFrame:
        """
        Load MovieLens movies.

        Filtered dataset (movies_filtered.csv):
        - Subset of top-rated/most-rated movies for performance
        - Faster training, reduced memory usage

        Requires MovieLens data - no fallback sample data.
        """
        if not data_path:
            raise ValueError("movielens_data_path is required. Download MovieLens dataset first.")

        # Try filtered dataset first (subset of top movies for performance)
        movies_path = Path(data_path) / "movies_filtered.csv"
        if movies_path.exists():
            movies = pd.read_csv(movies_path)
            print(f"Using filtered dataset: {len(movies)} movies")
            return movies

        # Fall back to full dataset
        movies_path = Path(data_path) / "movies.csv"
        if movies_path.exists():
            movies = pd.read_csv(movies_path)
            print(f"Using full dataset: {len(movies)} movies")
            return movies

        raise FileNotFoundError(
            f"MovieLens data not found at {data_path}. "
            f"Expected movies_filtered.csv or movies.csv. "
            f"Download MovieLens 25M dataset from https://grouplens.org/datasets/movielens/"
        )

    def _encode_movies(self):
        """
        Pre-encode all movies with SentenceBERT.

        Uses disk caching to avoid re-encoding on subsequent runs.
        Cache invalidates if movies data changes.
        """
        # Create movieId -> index mapping for O(1) lookups
        self.movie_id_to_idx = {
            row['movieId']: idx
            for idx, row in self.movies.iterrows()
        }

        # Check cache first
        cache_file = self._get_embeddings_cache_path()

        if cache_file and cache_file.exists():
            print(f"  Loading movie embeddings from cache: {cache_file}")
            with open(cache_file, 'rb') as f:
                cache_data = pickle.load(f)
            self.movie_embeddings = cache_data['embeddings']
            self.movie_embeddings_tensor = torch.FloatTensor(self.movie_embeddings)
            print(f"  Loaded {len(self.movie_embeddings)} movie embeddings from cache")
            return

        # Cache miss - encode movies from scratch
        print(f"  Encoding {len(self.movies)} movies with SentenceBERT...")
        print(f"  (This will be cached for future runs)")

        movie_texts = []
        for idx, row in self.movies.iterrows():
            # Create text representation: title + genres
            title = row['title'].split('(')[0].strip()  # Remove year
            genres = row['genres'].replace('|', ', ')
            text = f"{title}: {genres}"
            movie_texts.append(text)

        # Encode all movies
        self.movie_embeddings = self.encoder.encode(
            movie_texts,
            convert_to_numpy=True,
            show_progress_bar=True
        )

        # Convert to tensor
        self.movie_embeddings_tensor = torch.FloatTensor(self.movie_embeddings)

        # Save to cache
        if cache_file:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_file, 'wb') as f:
                pickle.dump({
                    'embeddings': self.movie_embeddings,
                    'num_movies': len(self.movies)
                }, f)
            print(f"  Movie embeddings cached to: {cache_file}")

    def _get_embeddings_cache_path(self) -> Optional[Path]:
        """Get cache file path for movie embeddings."""
        if not self.data_path:
            return None

        # Create hash of movie data to detect changes
        movies_csv = self.data_path / "movies_filtered.csv"
        if not movies_csv.exists():
            movies_csv = self.data_path / "movies.csv"

        if not movies_csv.exists():
            return None

        # Hash based on number of movies and first few rows
        hash_str = f"{len(self.movies)}_{self.movies.head(5).to_json()}"
        data_hash = hashlib.md5(hash_str.encode()).hexdigest()[:8]

        cache_dir = self.data_path / '.cache'
        cache_file = cache_dir / f'movie_embeddings_{len(self.movies)}_{data_hash}.pkl'
        return cache_file

    def get_movie_features(self, movie_ids: List[int]) -> torch.Tensor:
        """Get movie features by IDs (O(1) lookup using pre-computed mapping)."""
        indices = [self.movie_id_to_idx[mid] for mid in movie_ids if mid in self.movie_id_to_idx]
        return self.movie_embeddings_tensor[indices]

    def get_all_movie_features(self) -> torch.Tensor:
        """Get all movie features."""
        return self.movie_embeddings_tensor

    def get_movie_ids(self) -> List[int]:
        """Get all movie IDs."""
        return self.movies['movieId'].tolist()


class RecommenderTrainer:
    """
    Trainer for two-tower recommender.

    Trains on conversation → liked movies pairs.
    """

    def __init__(
        self,
        recommender: TwoTowerRecommender,
        movie_catalog: MovieCatalog,
        encoder: SentenceTransformer
    ):
        self.recommender = recommender
        self.movie_catalog = movie_catalog
        self.encoder = encoder

    def create_training_batch(
        self,
        conversations: List[str],
        liked_movies: List[List[int]],
        batch_size: int = 32
    ):
        """
        Create training batches from conversation-movie pairs.

        Args:
            conversations: List of conversation texts
            liked_movies: List of lists of liked movie IDs
            batch_size: Batch size

        Yields:
            (user_states, positive_items, negative_items) tensors
        """
        all_movie_ids = self.movie_catalog.get_movie_ids()

        for i in range(0, len(conversations), batch_size):
            batch_convs = conversations[i:i+batch_size]
            batch_likes = liked_movies[i:i+batch_size]

            # Encode conversations
            user_states = self.encoder.encode(
                batch_convs,
                convert_to_numpy=True,
                show_progress_bar=False
            )
            user_states = torch.FloatTensor(user_states)

            # Get positive and negative items
            positive_ids = []
            negative_ids = []

            for likes in batch_likes:
                # Sample one positive
                pos_id = random.choice(likes) if likes else random.choice(all_movie_ids)
                positive_ids.append(pos_id)

                # Sample one negative (not in likes)
                neg_candidates = [mid for mid in all_movie_ids if mid not in likes]
                neg_id = random.choice(neg_candidates) if neg_candidates else random.choice(all_movie_ids)
                negative_ids.append(neg_id)

            # Get movie features
            positive_items = self.movie_catalog.get_movie_features(positive_ids)
            negative_items = self.movie_catalog.get_movie_features(negative_ids)

            yield user_states, positive_items, negative_items

    def train(
        self,
        conversations: List[str],
        liked_movies: List[List[int]],
        epochs: int = 10,
        batch_size: int = 32,
        val_split: float = 0.1
    ):
        """
        Train recommender on conversation data with validation monitoring.

        Args:
            conversations: List of conversation texts
            liked_movies: List of lists of liked movie IDs per conversation
            epochs: Number of training epochs
            batch_size: Batch size
            val_split: Fraction of data to use for validation
        """
        from tqdm import tqdm
        import torch

        # Train/val split
        n_val = int(len(conversations) * val_split)
        n_train = len(conversations) - n_val

        train_convs = conversations[:n_train]
        train_likes = liked_movies[:n_train]
        val_convs = conversations[n_train:]
        val_likes = liked_movies[n_train:]

        print(f"\n{'='*60}")
        print("TRAINING TWO-TOWER RECOMMENDER")
        print(f"{'='*60}")
        print(f"Training examples: {n_train}")
        print(f"Validation examples: {n_val}")
        print(f"Epochs: {epochs}")
        print(f"Batch size: {batch_size}\n")

        best_val_ndcg = 0.0

        for epoch in tqdm(range(epochs), desc="Training recommender", unit="epoch"):
            # Training
            epoch_losses = []
            for user_states, pos_items, neg_items in self.create_training_batch(
                train_convs, train_likes, batch_size
            ):
                loss = self.recommender.train_bpr_step(
                    user_states, pos_items, neg_items
                )
                epoch_losses.append(loss)

            avg_loss = np.mean(epoch_losses) if epoch_losses else 0.0

            # Validation
            if n_val > 0:
                val_ndcg, val_hit_rate = self._validate(val_convs, val_likes, k=10)

                # Track best model
                if val_ndcg > best_val_ndcg:
                    best_val_ndcg = val_ndcg
                    best_marker = " (best)"
                else:
                    best_marker = ""

                tqdm.write(
                    f"Epoch {epoch+1}/{epochs} - "
                    f"Loss: {avg_loss:.4f} | "
                    f"Val NDCG@10: {val_ndcg:.4f} | "
                    f"Val Hit@10: {val_hit_rate:.4f}{best_marker}"
                )
            else:
                tqdm.write(f"Epoch {epoch+1}/{epochs} - BPR Loss: {avg_loss:.4f}")

        print(f"\n{'='*60}")
        print("TRAINING COMPLETE")
        print(f"Best Val NDCG@10: {best_val_ndcg:.4f}")
        print(f"{'='*60}\n")

    def _validate(self, conversations: List[str], liked_movies: List[List[int]], k: int = 10):
        """
        Validate recommender on held-out data.

        Args:
            conversations: Validation conversation texts
            liked_movies: Validation liked movie IDs
            k: Top-k for metrics

        Returns:
            (ndcg@k, hit_rate@k)
        """
        import torch

        ndcgs = []
        hits = []

        # Encode all conversations
        user_states = self.encoder.encode(
            conversations,
            convert_to_numpy=True,
            show_progress_bar=False
        )
        user_states = torch.FloatTensor(user_states)

        # Get all movie features
        all_movie_features = self.movie_catalog.get_all_movie_features()
        all_movie_ids = self.movie_catalog.get_movie_ids()

        for i, target_movies in enumerate(liked_movies):
            if not target_movies:
                continue

            # Get top-k recommendations
            scores = self.recommender.predict_scores(user_states[i], all_movie_features)
            top_indices = torch.argsort(scores, descending=True)[:k]
            recommended_ids = [all_movie_ids[idx] for idx in top_indices]

            # Calculate NDCG@k
            dcg = 0.0
            for rank, movie_id in enumerate(recommended_ids):
                if movie_id in target_movies:
                    dcg += 1.0 / np.log2(rank + 2)

            # IDCG (ideal)
            ideal_relevances = [1.0] * min(len(target_movies), k)
            idcg = sum(rel / np.log2(rank + 2) for rank, rel in enumerate(ideal_relevances))
            ndcg = dcg / idcg if idcg > 0 else 0.0
            ndcgs.append(ndcg)

            # Hit rate
            hit = 1.0 if any(mid in target_movies for mid in recommended_ids) else 0.0
            hits.append(hit)

        return np.mean(ndcgs) if ndcgs else 0.0, np.mean(hits) if hits else 0.0

    def recommend(
        self,
        conversation_state: np.ndarray,
        top_k: int = 10
    ) -> List[Tuple[int, str, float]]:
        """
        Get top-k movie recommendations for a conversation state.

        Args:
            conversation_state: [384] dialogue state from SentenceBERT
            top_k: Number of recommendations

        Returns:
            List of (movie_id, title, score) tuples
        """
        # Convert to tensor
        state_tensor = torch.FloatTensor(conversation_state)

        # Get all movie features
        all_movie_features = self.movie_catalog.get_all_movie_features()

        # Predict scores
        scores = self.recommender.predict_scores(state_tensor, all_movie_features)

        # Get top-k
        top_indices = torch.argsort(scores, descending=True)[:top_k]

        # Get movie details
        recommendations = []
        for idx in top_indices:
            idx = idx.item()
            movie_id = self.movie_catalog.movies.iloc[idx]['movieId']
            title = self.movie_catalog.movies.iloc[idx]['title']
            score = scores[idx].item()
            recommendations.append((movie_id, title, score))

        return recommendations


