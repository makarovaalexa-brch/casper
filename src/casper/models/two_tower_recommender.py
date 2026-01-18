"""
Two-Tower Recommender for CASPER.

Takes the same dialogue state as RL model (384-dim SentenceBERT) and recommends movies.

STATE ENCODING:
- State = Extracted preferences only (clean, structured)
- LLM extracts from conversation: {"liked": ["Nolan", "Inception"], "disliked": ["horror"]}
- Convert to text: "likes: Nolan, Inception | dislikes: horror"
- Encode with SentenceBERT: Text → 384-dim vector
- No raw conversation text (avoids duplication, cleaner signal)

Architecture (after 2026-01 improvements):
    User Tower: State (384) → Linear(256) → LayerNorm → ReLU → Dropout → Linear(128)
    Item Tower: Movie (384) → Linear(256) → LayerNorm → ReLU → Dropout → Linear(128)
    Embeddings: L2 normalized (prevents collapse)
    Score: Dot Product(user_emb, movie_emb)

Training:
- Trained from scratch on MovieLens data
- Loss: InfoNCE (contrastive) - much better than BPR
- Negatives: 16 per positive (mix of random + explicitly disliked)
- Best result: Val NDCG@10 = 0.28 (5k users, 30 epochs)
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

    def __init__(self, state_dim: int = 384, user_emb_dim: int = 128,
                 deep: bool = True, dropout: float = 0.1):
        super().__init__()
        self.deep = deep

        if deep:
            # Deep tower with LayerNorm + ReLU (proven to help)
            self.network = nn.Sequential(
                nn.Linear(state_dim, 256),
                nn.LayerNorm(256),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(256, user_emb_dim)
            )
        else:
            # Simple linear (preserves sign for signed encoding)
            self.network = nn.Linear(state_dim, user_emb_dim)

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

    def __init__(self, item_dim: int = 384, item_emb_dim: int = 128,
                 deep: bool = True, dropout: float = 0.1):
        super().__init__()

        if deep:
            # Deep tower with LayerNorm + ReLU
            self.network = nn.Sequential(
                nn.Linear(item_dim, 256),
                nn.LayerNorm(256),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(256, item_emb_dim)
            )
        else:
            # Simple linear projection
            self.network = nn.Linear(item_dim, item_emb_dim)

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

    Improvements from experiments (2026-01):
    - Deep towers with LayerNorm + ReLU
    - L2 normalization before dot product (prevents collapse)
    - InfoNCE loss (much better than BPR)
    - 16 negatives per positive
    """

    def __init__(
        self,
        state_dim: int = 384,
        embedding_dim: int = 128,
        learning_rate: float = 0.0003,
        deep: bool = True,
        dropout: float = 0.1,
        normalize: bool = True  # L2 normalize embeddings
    ):
        super().__init__()
        self.normalize = normalize

        self.user_tower = UserTower(state_dim, embedding_dim, deep=deep, dropout=dropout)
        self.item_tower = ItemTower(state_dim, embedding_dim, deep=deep, dropout=dropout)

        self.optimizer = torch.optim.Adam(
            self.parameters(),
            lr=learning_rate,
            weight_decay=1e-4  # Increased from 1e-5
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

        # L2 normalize to prevent embedding collapse
        if self.normalize:
            user_emb = F.normalize(user_emb, p=2, dim=-1)
            item_emb = F.normalize(item_emb, p=2, dim=-1)

        # Dot product for scoring
        scores = (user_emb * item_emb).sum(dim=1)

        return scores

    def get_embeddings(self, user_states, item_features):
        """Get normalized embeddings (for InfoNCE loss)."""
        user_emb = self.user_tower(user_states)
        item_emb = self.item_tower(item_features)

        if self.normalize:
            user_emb = F.normalize(user_emb, p=2, dim=-1)
            item_emb = F.normalize(item_emb, p=2, dim=-1)

        return user_emb, item_emb

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
        Supports multiple negatives per positive (batch_size * num_negatives).

        Args:
            user_states: [batch_size, 384] dialogue states
            positive_items: [batch_size, 384] liked movie features
            negative_items: [batch_size * num_negatives, 384] disliked/random movie features

        Returns:
            loss value
        """
        batch_size = user_states.size(0)
        num_negatives = negative_items.size(0) // batch_size

        # Get positive scores
        pos_scores = self.forward(user_states, positive_items)  # [batch_size]

        # Repeat user states for each negative
        user_states_repeated = user_states.repeat_interleave(num_negatives, dim=0)  # [batch_size * num_negatives, 384]

        # Get negative scores
        neg_scores = self.forward(user_states_repeated, negative_items)  # [batch_size * num_negatives]

        # Reshape to [batch_size, num_negatives]
        neg_scores = neg_scores.view(batch_size, num_negatives)

        # Expand pos_scores to compare with all negatives
        pos_scores_expanded = pos_scores.unsqueeze(1)  # [batch_size, 1]

        # BPR loss: encourage pos_score > neg_score for ALL negatives
        # loss = -log(sigmoid(pos_score - neg_score)) averaged over all pairs
        loss = -F.logsigmoid(pos_scores_expanded - neg_scores).mean()

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
        self.optimizer.step()

        return loss.item()

    def train_infonce_step(
        self,
        user_states,
        positive_items,
        negative_items,
        temperature: float = 0.1
    ):
        """
        Training step with InfoNCE (contrastive) loss.

        InfoNCE provides stronger gradients than BPR and led to 0.18 → 0.28 NDCG
        improvement in experiments.

        Args:
            user_states: [batch_size, 384] dialogue states
            positive_items: [batch_size, 384] liked movie features
            negative_items: [batch_size * num_negatives, 384] disliked/random movie features
            temperature: Softmax temperature (lower = sharper)

        Returns:
            loss value
        """
        batch_size = user_states.size(0)
        num_negatives = negative_items.size(0) // batch_size

        # Get embeddings
        user_emb, pos_emb = self.get_embeddings(user_states, positive_items)

        # Reshape negatives and get embeddings
        neg_items_reshaped = negative_items.view(batch_size, num_negatives, -1)

        # Compute positive scores
        pos_scores = (user_emb * pos_emb).sum(dim=-1, keepdim=True)  # [batch, 1]

        # Compute negative scores for each user
        neg_emb = self.item_tower(neg_items_reshaped.view(-1, neg_items_reshaped.size(-1)))
        if self.normalize:
            neg_emb = F.normalize(neg_emb, p=2, dim=-1)
        neg_emb = neg_emb.view(batch_size, num_negatives, -1)  # [batch, num_neg, emb_dim]

        neg_scores = torch.bmm(neg_emb, user_emb.unsqueeze(-1)).squeeze(-1)  # [batch, num_neg]

        # InfoNCE: softmax over positive + all negatives
        all_scores = torch.cat([pos_scores, neg_scores], dim=1) / temperature  # [batch, 1+num_neg]
        labels = torch.zeros(batch_size, dtype=torch.long, device=user_states.device)  # positive is index 0

        loss = F.cross_entropy(all_scores, labels)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
        self.optimizer.step()

        return loss.item()

    def calculate_bpr_loss(
        self,
        user_state,
        positive_items,
        negative_items
    ):
        """
        Calculate BPR loss WITHOUT training (for RL reward evaluation).

        Args:
            user_state: [384] single dialogue state
            positive_items: [num_pos, 384] liked movie features
            negative_items: [num_neg, 384] random/disliked movie features

        Returns:
            loss value (float)
        """
        # Ensure model is in eval mode (disables dropout for deterministic inference)
        self.eval()

        with torch.no_grad():
            # Expand user state to match batch sizes
            user_states_pos = user_state.unsqueeze(0).expand(len(positive_items), -1)
            user_states_neg = user_state.unsqueeze(0).expand(len(negative_items), -1)

            # Get scores
            pos_scores = self.forward(user_states_pos, positive_items)
            neg_scores = self.forward(user_states_neg, negative_items)

            # BPR loss: -log(sigmoid(pos_score - neg_score))
            # Average over all positive-negative pairs
            total_loss = 0.0
            count = 0
            for pos_score in pos_scores:
                for neg_score in neg_scores:
                    total_loss += -F.logsigmoid(pos_score - neg_score).item()
                    count += 1

            avg_loss = total_loss / count if count > 0 else 0.0

        return avg_loss


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
        user_states: np.ndarray,
        liked_movies: List[List[int]],
        disliked_movies: List[List[int]] = None,
        batch_size: int = 32,
        show_progress: bool = False,
        num_negatives: int = 16  # Increased from 4 (sharper discrimination)
    ):
        """
        Create training batches with smart negative sampling.

        Args:
            user_states: Pre-encoded conversation embeddings [N, 384]
            liked_movies: List of lists of liked movie IDs
            disliked_movies: List of lists of disliked movie IDs (NEW!)
            batch_size: Batch size
            show_progress: Show progress bar for batches
            num_negatives: Number of negatives per positive (default: 4)

        Yields:
            (user_states_batch, positive_items, negative_items) tensors
        """
        all_movie_ids = self.movie_catalog.get_movie_ids()
        num_batches = (len(user_states) + batch_size - 1) // batch_size

        iterator = range(0, len(user_states), batch_size)
        if show_progress:
            from tqdm import tqdm
            iterator = tqdm(list(iterator), desc="Batches", total=num_batches, leave=True, unit="batch")

        for i in iterator:
            batch_states = user_states[i:i+batch_size]
            batch_likes = liked_movies[i:i+batch_size]
            batch_dislikes = disliked_movies[i:i+batch_size] if disliked_movies else [[] for _ in range(len(batch_likes))]

            batch_states_tensor = torch.FloatTensor(batch_states)

            # Get positive and negative items
            positive_ids = []
            negative_ids = []

            for likes, dislikes in zip(batch_likes, batch_dislikes):
                # Sample one positive
                pos_id = random.choice(likes) if likes else random.choice(all_movie_ids)
                positive_ids.append(pos_id)

                # SMART NEGATIVE SAMPLING (Literature-based)
                # Mix explicitly disliked (strong signal) + random unseen (generalization)
                likes_set = set(likes)
                dislikes_set = set(dislikes) if dislikes else set()
                unseen_movies = [mid for mid in all_movie_ids if mid not in likes_set]

                sampled_negatives = []

                # 1. Prefer explicitly disliked movies (strongest signal!)
                if len(dislikes) >= 2:
                    # Sample 2 from dislikes
                    sampled_negatives.extend(random.sample(dislikes, min(2, len(dislikes))))
                    # Sample 2 from random unseen
                    remaining = num_negatives - len(sampled_negatives)
                    sampled_negatives.extend(random.sample(unseen_movies, remaining))
                else:
                    # Fallback: all random if not enough dislikes
                    sampled_negatives = random.sample(unseen_movies, num_negatives)

                negative_ids.extend(sampled_negatives)

            # Get movie features
            positive_items = self.movie_catalog.get_movie_features(positive_ids)
            negative_items = self.movie_catalog.get_movie_features(negative_ids)

            yield batch_states_tensor, positive_items, negative_items

    def train(
        self,
        conversations: List[str],
        liked_movies: List[List[int]],
        disliked_movies: List[List[int]] = None,
        epochs: int = 10,
        batch_size: int = 64,  # CHANGED: 32 → 64 (less noisy gradients)
        val_split: float = 0.1,
        checkpoint_path: Optional[str] = None
    ):
        """
        Train recommender with smart negative sampling.

        Args:
            conversations: List of conversation texts
            liked_movies: List of lists of liked movie IDs per conversation
            disliked_movies: List of lists of disliked movie IDs (NEW!)
            epochs: Number of training epochs
            batch_size: Batch size (default: 64)
            val_split: Fraction of data to use for validation
        """
        from tqdm import tqdm
        import torch

        # Train/val split
        n_val = int(len(conversations) * val_split)
        n_train = len(conversations) - n_val

        train_convs = conversations[:n_train]
        train_likes = liked_movies[:n_train]
        train_dislikes = disliked_movies[:n_train] if disliked_movies else None
        val_convs = conversations[n_train:]
        val_likes = liked_movies[n_train:]
        val_dislikes = disliked_movies[n_train:] if disliked_movies else None

        print(f"\n{'='*60}")
        print("TRAINING TWO-TOWER RECOMMENDER")
        print(f"{'='*60}")
        print(f"Training examples: {n_train}")
        print(f"Validation examples: {n_val}")
        print(f"Epochs: {epochs}")
        print(f"Batch size: {batch_size}\n")

        # Pre-encode all conversations with SentenceBERT
        print(f"Encoding {n_train} training conversations...")

        train_states = self.encoder.encode(
            train_convs,
            convert_to_numpy=True,
            show_progress_bar=False,
            batch_size=64
        )

        if n_val > 0:
            print(f"Encoding {n_val} validation conversations...")
            val_states = self.encoder.encode(
                val_convs,
                convert_to_numpy=True,
                show_progress_bar=False,
                batch_size=64
            )

        best_val_ndcg = 0.0
        start_epoch = 0

        # Try to resume from checkpoint
        if checkpoint_path:
            from pathlib import Path
            checkpoint_file = Path(checkpoint_path)
            if checkpoint_file.exists():
                checkpoint = torch.load(checkpoint_file)
                self.recommender.user_tower.load_state_dict(checkpoint['user_tower'])
                self.recommender.item_tower.load_state_dict(checkpoint['item_tower'])
                start_epoch = checkpoint.get('epoch', 0)
                best_val_ndcg = checkpoint.get('best_val_ndcg', 0.0)
                print(f"Resuming from epoch {start_epoch} (best NDCG: {best_val_ndcg:.4f})\n")

        for epoch in tqdm(range(start_epoch, epochs), desc="Training recommender", unit="epoch", initial=start_epoch, total=epochs):
            # Training
            epoch_losses = []
            for user_states_batch, pos_items, neg_items in self.create_training_batch(
                train_states, train_likes, train_dislikes, batch_size, show_progress=False
            ):
                # Use InfoNCE loss (better than BPR: 0.18 → 0.28 NDCG)
                loss = self.recommender.train_infonce_step(
                    user_states_batch, pos_items, neg_items
                )
                epoch_losses.append(loss)

            avg_loss = np.mean(epoch_losses) if epoch_losses else 0.0

            # Validation
            if n_val > 0:
                val_ndcg, val_hit_rate = self._validate(val_states, val_likes, k=10)

                # Track best model
                is_best = val_ndcg > best_val_ndcg
                if is_best:
                    best_val_ndcg = val_ndcg
                    best_marker = " (best)"
                else:
                    best_marker = ""

                # Print epoch results so they stack for comparison
                tqdm.write(
                    f"Epoch {epoch+1}/{epochs} - "
                    f"Loss: {avg_loss:.4f} | "
                    f"Val NDCG@10: {val_ndcg:.4f} | "
                    f"Val Hit@10: {val_hit_rate:.4f}{best_marker}"
                )

                # Save checkpoint after each epoch
                if checkpoint_path:
                    torch.save({
                        'user_tower': self.recommender.user_tower.state_dict(),
                        'item_tower': self.recommender.item_tower.state_dict(),
                        'epoch': epoch + 1,
                        'best_val_ndcg': best_val_ndcg,
                        'val_ndcg': val_ndcg,
                        'val_hit_rate': val_hit_rate,
                        'loss': avg_loss
                    }, checkpoint_path)
            else:
                tqdm.write(f"Epoch {epoch+1}/{epochs} - BPR Loss: {avg_loss:.4f}")

        print(f"\n{'='*60}")
        print("TRAINING COMPLETE")
        print(f"Best Val NDCG@10: {best_val_ndcg:.4f}")
        print(f"{'='*60}\n")

    def _validate(self, user_states: np.ndarray, liked_movies: List[List[int]], k: int = 10):
        """
        Validate recommender on held-out data (batched for speed).

        Args:
            user_states: Pre-encoded user states [N, 384]
            liked_movies: Validation liked movie IDs
            k: Top-k for metrics

        Returns:
            (ndcg@k, hit_rate@k)
        """
        import torch

        ndcgs = []
        hits = []

        # Convert to tensor (already encoded!)
        user_states_tensor = torch.FloatTensor(user_states)

        # Get all movie features
        all_movie_features = self.movie_catalog.get_all_movie_features()
        all_movie_ids = self.movie_catalog.get_movie_ids()

        # Batch validation for speed
        batch_size = 100
        for batch_start in range(0, len(user_states), batch_size):
            batch_end = min(batch_start + batch_size, len(user_states))
            batch_states = user_states_tensor[batch_start:batch_end]
            batch_targets = liked_movies[batch_start:batch_end]

            # Score all users in batch against all movies at once
            with torch.no_grad():
                user_tower_out = self.recommender.user_tower(batch_states)  # [batch, embed_dim]
                item_tower_out = self.recommender.item_tower(all_movie_features)  # [num_movies, embed_dim]
                scores = torch.matmul(user_tower_out, item_tower_out.T)  # [batch, num_movies]

            # Get top-k for each user in batch
            top_indices = torch.argsort(scores, dim=1, descending=True)[:, :k]  # [batch, k]

            for i, target_movies in enumerate(batch_targets):
                if not target_movies:
                    continue

                recommended_ids = [all_movie_ids[idx.item()] for idx in top_indices[i]]

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
        # Ensure model is in eval mode (disables dropout for deterministic inference)
        self.recommender.eval()

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


