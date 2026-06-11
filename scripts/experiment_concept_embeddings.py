"""
EXPERIMENT 3: Model with Concept Embeddings (SBERT)

Replace one-hot item encoding with semantic embeddings from SBERT.
This enables:
1. Natural language understanding of items
2. Zero-shot transfer to new items
3. Integration with LLM-based dialogue systems

Key change: Instead of learning item embeddings from scratch,
we use pre-trained SBERT embeddings and learn transformations on top.
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
# Model with Concept Embeddings
# =============================================================================

class ConceptEmbeddingModel(nn.Module):
    """
    Model that uses pre-computed concept embeddings instead of learned embeddings.

    Input: (concept_embeddings, rating_flags)
    - rating_flags: [disliked, liked, not_seen] one-hot (3 dims) - kept as-is like original
    - concept_embeddings: pre-computed SBERT embeddings for items

    The model learns a transformation on top of concept embeddings.
    """
    def __init__(self, n_items, embedding_dim=384, hidden_dim=128, lr=0.001):
        super(ConceptEmbeddingModel, self).__init__()
        self.n_items = n_items
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        # Project concept embeddings to hidden dimension
        self.concept_proj = nn.Linear(embedding_dim, hidden_dim)

        # Rating is kept as simple 3-dim one-hot, just like original model
        # No projection - concatenate directly
        lstm_input_dim = hidden_dim + 3

        self.lstm = nn.LSTM(lstm_input_dim, hidden_dim, num_layers=1, batch_first=True)
        self.dense1 = nn.Linear(hidden_dim, n_items)
        self.dense2 = nn.Linear(n_items, hidden_dim)
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=1, batch_first=True)

        # Output: predict ratings for all items (explicit + implicit)
        self.output_proj = nn.Linear(hidden_dim * 2, n_items * 2)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    def forward(self, concept_embeddings, rating_input):
        """
        Args:
            concept_embeddings: (batch, seq_len, embedding_dim) - SBERT embeddings
            rating_input: (batch, seq_len, 3) - one-hot rating flags

        Returns:
            predictions: (batch, seq_len, n_items * 2) - explicit + implicit predictions
        """
        # Project concept embeddings
        x_concept = self.concept_proj(concept_embeddings)

        # Concatenate with raw rating one-hot (like original model)
        x = torch.cat([x_concept, rating_input], dim=-1)

        # LSTM encoding
        encoder_output, _ = self.lstm(x)

        # Dense layers (like original model)
        x = torch.relu(self.dense1(encoder_output))
        x = torch.relu(self.dense2(x))

        # Attention (like original model)
        attention_out, _ = self.attention(encoder_output, x, x)

        # Combine and output (like original model)
        x = torch.cat([x, attention_out], dim=-1)
        output = self.output_proj(x)

        return output


class BCEWithLogitsLossNan(nn.Module):
    def forward(self, y_pred, y_true):
        mask = torch.isnan(y_true)
        y_pred = torch.where(mask, torch.zeros_like(y_true), y_pred)
        y_true = torch.where(mask, torch.zeros_like(y_true), y_true)
        return torch.nn.functional.binary_cross_entropy_with_logits(y_pred, y_true)


# =============================================================================
# Dataset with Concept Embeddings
# =============================================================================

class ConceptEmbeddingDataset(Dataset):
    """
    Dataset that provides pre-computed concept embeddings for items.
    """
    def __init__(self, ratings_df, n_items, item_embeddings):
        """
        Args:
            ratings_df: DataFrame with userId, itemIdx, rating
            n_items: total number of items
            item_embeddings: numpy array (n_items, embedding_dim) - SBERT embeddings
        """
        self.ratings_df = ratings_df
        self.n_items = n_items
        self.item_embeddings = torch.FloatTensor(item_embeddings)
        self.embedding_dim = item_embeddings.shape[1]
        self.user_ids = ratings_df['userId'].unique()

    def __len__(self):
        return len(self.user_ids)

    def __getitem__(self, idx):
        user_id = self.user_ids[idx]
        user_ratings = self.ratings_df[self.ratings_df['userId'] == user_id]

        # Full rating array
        full_ratings = np.full(self.n_items, np.nan)
        for _, row in user_ratings.iterrows():
            item_idx = int(row['itemIdx'])
            rating = 1.0 if row['rating'] >= 4 else 0.0
            full_ratings[item_idx] = rating

        # Random permutation of items
        indices = np.random.permutation(self.n_items)

        # Get concept embeddings in permuted order
        concept_embeddings = self.item_embeddings[indices]

        # Rating input: one-hot [disliked, liked, not_seen]
        rating_input = np.zeros((self.n_items, 3))
        for i, item_idx in enumerate(indices):
            if np.isnan(full_ratings[item_idx]):
                rating_input[i, 2] = 1  # not_seen
            elif full_ratings[item_idx] == 0:
                rating_input[i, 0] = 1  # disliked
            else:
                rating_input[i, 1] = 1  # liked

        # Output: explicit ratings + implicit (seen/not seen)
        explicit = np.tile(full_ratings, (self.n_items, 1))
        implicit = (~np.isnan(explicit)).astype(float)
        output = np.concatenate([explicit, implicit], axis=1)

        return (
            concept_embeddings,
            torch.FloatTensor(rating_input),
            torch.FloatTensor(output)
        )


# =============================================================================
# SBERT Embedding Generation
# =============================================================================

# Supported embedding models with their configurations
EMBEDDING_MODELS = {
    'minilm': {
        'name': 'sentence-transformers/all-MiniLM-L6-v2',
        'dim': 384,
        'desc': 'Fast, general-purpose (22M params)'
    },
    'e5-small': {
        'name': 'intfloat/e5-small-v2',
        'dim': 384,
        'desc': 'E5 small - good accuracy, low latency'
    },
    'e5-base': {
        'name': 'intfloat/e5-base-v2',
        'dim': 768,
        'desc': 'E5 base - balanced performance'
    },
    'e5-large': {
        'name': 'intfloat/e5-large-v2',
        'dim': 1024,
        'desc': 'E5 large - best E5 performance'
    },
    'bge-small': {
        'name': 'BAAI/bge-small-en-v1.5',
        'dim': 384,
        'desc': 'BGE small - fast, competitive'
    },
    'bge-base': {
        'name': 'BAAI/bge-base-en-v1.5',
        'dim': 768,
        'desc': 'BGE base - strong performance'
    },
    'bge-large': {
        'name': 'BAAI/bge-large-en-v1.5',
        'dim': 1024,
        'desc': 'BGE large - top open-source model'
    },
    'movie-minilm': {
        'name': 'AventIQ-AI/Movie-Recommendation-Using-Sentence-Transormer',
        'dim': 384,
        'desc': 'Fine-tuned on MovieLens (based on MiniLM)'
    }
}


def generate_embeddings(texts, model_key='minilm'):
    """
    Generate embeddings for list of texts using specified model.

    Args:
        texts: List of text strings
        model_key: Key from EMBEDDING_MODELS dict

    Returns:
        numpy array of embeddings
    """
    import sys

    if model_key not in EMBEDDING_MODELS:
        print(f"Unknown model: {model_key}. Available: {list(EMBEDDING_MODELS.keys())}")
        model_key = 'minilm'

    model_config = EMBEDDING_MODELS[model_key]
    print(f"Using embedding model: {model_config['name']}", flush=True)
    print(f"  {model_config['desc']}", flush=True)

    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(model_config['name'])
        embeddings = model.encode(texts, show_progress_bar=True)
        return embeddings, model_config['dim']
    except ImportError:
        print("sentence-transformers not installed. Using random embeddings as fallback.")
        print("Install with: pip install sentence-transformers")
        np.random.seed(42)
        dim = model_config['dim']
        return np.random.randn(len(texts), dim).astype(np.float32), dim


def create_item_descriptions(movie_titles, genres_list, tag_names,
                             actor_names=None, director_names=None,
                             enrich=False, movie_genres_map=None,
                             movie_tags_map=None, tag_names_dict=None):
    """
    Create text descriptions for all items.

    For ORPHANED concept learning, items should be simple names that the embedding
    model can understand in isolation. The model learns correlations between them.

    Args:
        enrich: If True, add metadata to movie descriptions (for comparison)
                If False (default), use simple names for orphaned concept learning

    Returns list of descriptions in order:
    [movie_0, ..., genre_0, ..., tag_0, ..., actor_0, ..., director_0, ...]
    """
    descriptions = []

    # Movies: just title (or enriched if specified)
    for i, title in enumerate(movie_titles):
        if enrich and movie_genres_map and i in movie_genres_map:
            genre_indices = movie_genres_map[i]
            genres = [genres_list[gi] for gi in genre_indices if gi < len(genres_list)]
            if genres:
                descriptions.append(f"{title} ({', '.join(genres)})")
            else:
                descriptions.append(title)
        else:
            descriptions.append(title)

    # Genres: simple name
    for genre in genres_list:
        descriptions.append(genre)

    # Tags: simple name (genome tags are already descriptive)
    for tag in tag_names:
        descriptions.append(tag)

    # Actors: simple name
    if actor_names:
        for actor in actor_names:
            descriptions.append(actor)

    # Directors: simple name
    if director_names:
        for director in director_names:
            descriptions.append(director)

    return descriptions


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

def run_concept_embedding_experiment(
    n_movies: int = 100,
    n_top_tags: int = 50,
    max_users: int = 300,
    n_epochs: int = 50,
    embedding_model: str = 'minilm',
    include_actors: bool = False,
    include_directors: bool = False,
    n_actors: int = 50,
    n_directors: int = 30,
    min_user_total_ratings: int = 200,
    verbose: bool = True
):
    """
    Run the concept embedding experiment.

    Args:
        embedding_model: One of 'minilm', 'e5-small', 'e5-base', 'e5-large',
                        'bge-small', 'bge-base', 'bge-large'
        include_actors: Include actors as items
        include_directors: Include directors as items
        n_actors: Max number of actors to include
        n_directors: Max number of directors to include
        min_user_total_ratings: Minimum total ratings for a user (dense users)
    """
    import sys

    print("=" * 80, flush=True)
    print("EXPERIMENT 3: MODEL WITH CONCEPT EMBEDDINGS", flush=True)
    print("=" * 80, flush=True)

    # Load data
    DATA_DIR = Path('C:/dev/phd/casper/data/movielens')
    ratings = pd.read_csv(DATA_DIR / 'ratings.csv')
    movies = pd.read_csv(DATA_DIR / 'movies.csv')
    genome_scores = pd.read_csv(DATA_DIR / 'genome-scores.csv')
    genome_tags = pd.read_csv(DATA_DIR / 'genome-tags.csv')

    # Get top movies
    movie_counts = ratings['movieId'].value_counts()
    top_movies = movie_counts.head(n_movies).index.tolist()

    # Filter for DENSE users first (users with many total ratings)
    user_total_counts = ratings['userId'].value_counts()
    dense_users = user_total_counts[user_total_counts >= min_user_total_ratings].index.tolist()
    print(f"Found {len(dense_users):,} dense users (>={min_user_total_ratings} total ratings)", flush=True)

    # Filter ratings to top movies and dense users
    ratings_filtered = ratings[
        (ratings['movieId'].isin(top_movies)) &
        (ratings['userId'].isin(dense_users))
    ]

    # Select users with 25+ liked ratings on top movies
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

    # Extract movie titles
    movie_titles = []
    for mid in top_movies:
        title = movies[movies['movieId'] == mid]['title'].values[0]
        movie_titles.append(title)

    # Extract genres and build movie->genres mapping
    all_genres = set()
    movie_genres_map = {}
    genre_to_idx = {}

    for _, row in movies[movies['movieId'].isin(top_movies)].iterrows():
        movie_idx = movie_to_idx[row['movieId']]
        genres = row['genres'].split('|')
        all_genres.update(genres)

    all_genres.discard('(no genres listed)')
    all_genres = sorted(list(all_genres))
    genre_to_idx = {g: i for i, g in enumerate(all_genres)}
    N_GENRES = len(all_genres)

    for _, row in movies[movies['movieId'].isin(top_movies)].iterrows():
        movie_idx = movie_to_idx[row['movieId']]
        genres = row['genres'].split('|')
        movie_genres_map[movie_idx] = [genre_to_idx[g] for g in genres if g in genre_to_idx]

    # Extract top genome tags and build movie->tags mapping
    genome_filtered = genome_scores[genome_scores['movieId'].isin(top_movies)]
    tag_variance = genome_filtered.groupby('tagId')['relevance'].var()
    top_tag_ids = tag_variance.nlargest(n_top_tags).index.tolist()

    tag_to_idx = {tid: i for i, tid in enumerate(top_tag_ids)}
    tag_names_list = []
    tag_names_dict = {}  # tag_idx -> tag_name
    for i, tid in enumerate(top_tag_ids):
        tag_name = genome_tags[genome_tags['tagId'] == tid]['tag'].values[0]
        tag_names_list.append(tag_name)
        tag_names_dict[i] = tag_name
    N_TAGS = len(top_tag_ids)

    # Build movie->tags mapping
    movie_tags_map = {}
    for movie_id in top_movies:
        movie_idx = movie_to_idx[movie_id]
        movie_genome = genome_filtered[genome_filtered['movieId'] == movie_id]
        tags = []
        for _, row in movie_genome.iterrows():
            if row['tagId'] in tag_to_idx:
                tags.append((tag_to_idx[row['tagId']], row['relevance']))
        movie_tags_map[movie_idx] = tags

    # Load actors and directors if requested
    actor_names = []
    director_names = []
    movie_actors_map = {}
    movie_directors_map = {}
    N_ACTORS = 0
    N_DIRECTORS = 0

    if include_actors or include_directors:
        # Try to load from TMDB cache first, then fall back to static data
        import json
        cache_path = DATA_DIR / '.cache' / 'credits_top100_actors5.json'

        if cache_path.exists():
            with open(cache_path, 'r') as f:
                credits_data = json.load(f)

            if include_actors:
                # Count actor frequency to get most common ones
                actor_counts = {}
                for actors in credits_data['movie_actors'].values():
                    for a in actors:
                        actor_counts[a] = actor_counts.get(a, 0) + 1
                # Get top n_actors by frequency
                top_actors = sorted(actor_counts.keys(), key=lambda x: -actor_counts[x])[:n_actors]
                actor_names = top_actors
                N_ACTORS = len(actor_names)
                actor_to_idx = {a: i for i, a in enumerate(actor_names)}

                # Build movie->actors mapping using movieId
                for movie_id in top_movies:
                    movie_idx = movie_to_idx[movie_id]
                    movie_id_str = str(movie_id)
                    if movie_id_str in credits_data['movie_actors']:
                        actors = credits_data['movie_actors'][movie_id_str]
                        movie_actors_map[movie_idx] = [actor_to_idx[a] for a in actors if a in actor_to_idx]

            if include_directors:
                # Count director frequency to get most common ones
                director_counts = {}
                for directors in credits_data['movie_directors'].values():
                    for d in directors:
                        director_counts[d] = director_counts.get(d, 0) + 1
                # Get top n_directors by frequency
                top_directors = sorted(director_counts.keys(), key=lambda x: -director_counts[x])[:n_directors]
                director_names = top_directors
                N_DIRECTORS = len(director_names)
                director_to_idx = {d: i for i, d in enumerate(director_names)}

                # Build movie->directors mapping using movieId
                for movie_id in top_movies:
                    movie_idx = movie_to_idx[movie_id]
                    movie_id_str = str(movie_id)
                    if movie_id_str in credits_data['movie_directors']:
                        directors = credits_data['movie_directors'][movie_id_str]
                        movie_directors_map[movie_idx] = [director_to_idx[d] for d in directors if d in director_to_idx]

            print(f"Loaded {N_ACTORS} actors, {N_DIRECTORS} directors from TMDB cache", flush=True)
        else:
            # Fallback to static data
            try:
                from credits_fallback import TOP_ACTORS, TOP_DIRECTORS, MOVIE_ACTORS, MOVIE_DIRECTORS

                if include_actors:
                    actor_names = TOP_ACTORS
                    N_ACTORS = len(actor_names)
                    actor_to_idx = {a: i for i, a in enumerate(actor_names)}
                    for movie_idx, title in enumerate(movie_titles):
                        if title in MOVIE_ACTORS:
                            actors = MOVIE_ACTORS[title]
                            movie_actors_map[movie_idx] = [actor_to_idx[a] for a in actors if a in actor_to_idx]

                if include_directors:
                    director_names = TOP_DIRECTORS
                    N_DIRECTORS = len(director_names)
                    director_to_idx = {d: i for i, d in enumerate(director_names)}
                    for movie_idx, title in enumerate(movie_titles):
                        if title in MOVIE_DIRECTORS:
                            directors = MOVIE_DIRECTORS[title]
                            movie_directors_map[movie_idx] = [director_to_idx[d] for d in directors if d in director_to_idx]

                print(f"Loaded {N_ACTORS} actors, {N_DIRECTORS} directors from fallback data", flush=True)
            except ImportError:
                print("Warning: No credits data available, skipping actors/directors", flush=True)

    N_ITEMS = n_movies + N_GENRES + N_TAGS + N_ACTORS + N_DIRECTORS
    item_counts = f"{n_movies} movies + {N_GENRES} genres + {N_TAGS} tags"
    if N_ACTORS > 0:
        item_counts += f" + {N_ACTORS} actors"
    if N_DIRECTORS > 0:
        item_counts += f" + {N_DIRECTORS} directors"
    print(f"\nDataset: {item_counts} = {N_ITEMS} items", flush=True)
    print(f"Users: {len(active_users)}", flush=True)

    # Generate item descriptions (simple names for orphaned concept learning)
    print("\nGenerating item descriptions (simple names for orphaned concepts)...", flush=True)
    descriptions = create_item_descriptions(
        movie_titles, all_genres, tag_names_list,
        actor_names=actor_names if include_actors else None,
        director_names=director_names if include_directors else None,
        enrich=False  # Simple names, not enriched - model learns correlations
    )

    print(f"Sample descriptions:", flush=True)
    print(f"  Movie: {descriptions[0]}", flush=True)
    print(f"  Movie: {descriptions[1]}", flush=True)
    print(f"  Genre: {descriptions[n_movies]}", flush=True)
    print(f"  Tag: {descriptions[n_movies + N_GENRES]}", flush=True)

    # Generate embeddings
    print(f"\nGenerating embeddings with model: {embedding_model}", flush=True)
    item_embeddings, embedding_dim = generate_embeddings(descriptions, model_key=embedding_model)
    print(f"Embedding shape: {item_embeddings.shape}", flush=True)

    # Split users
    train_users = active_users[:int(0.8 * len(active_users))]
    val_users = active_users[int(0.8 * len(active_users)):]

    train_df = ratings_filtered[ratings_filtered['userId'].isin(train_users)]
    val_df = ratings_filtered[ratings_filtered['userId'].isin(val_users)]

    train_dataset = ConceptEmbeddingDataset(train_df, N_ITEMS, item_embeddings)
    val_dataset = ConceptEmbeddingDataset(val_df, N_ITEMS, item_embeddings)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Create model
    # Use smaller hidden_dim like original model: max(n_items // 2, 64)
    # Smaller bottleneck forces model to learn tighter relationships
    hidden_dim = max(N_ITEMS // 2, 64)
    model = ConceptEmbeddingModel(N_ITEMS, embedding_dim=embedding_dim, hidden_dim=hidden_dim, lr=0.001)
    loss_fn = BCEWithLogitsLossNan()

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}", flush=True)

    # Training
    print("\n" + "=" * 80, flush=True)
    print("TRAINING", flush=True)
    print("=" * 80, flush=True)

    start_time = time.time()

    for epoch in range(n_epochs):
        model.train()
        train_loss = 0
        for concept_emb, ratings_batch, targets in train_loader:
            model.optimizer.zero_grad()
            outputs = model(concept_emb, ratings_batch)
            loss = loss_fn(outputs, targets)
            loss.backward()
            model.optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for concept_emb, ratings_batch, targets in val_loader:
                outputs = model(concept_emb, ratings_batch)
                loss = loss_fn(outputs, targets)
                val_loss += loss.item()
        val_loss /= len(val_loader)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            elapsed = time.time() - start_time
            print(f"Epoch {epoch+1:3d} | Train: {train_loss:.4f} | Val: {val_loss:.4f} | Time: {elapsed:.1f}s", flush=True)

    total_time = time.time() - start_time
    print(f"\nTotal training time: {total_time:.1f}s", flush=True)

    # Save checkpoint
    checkpoint_dir = Path('C:/dev/phd/casper/data/movielens/.cache/checkpoints')
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f'concept_model_{embedding_model}_{N_ITEMS}items_{len(active_users)}users.pt'
    torch.save({
        'model_state_dict': model.state_dict(),
        'n_items': N_ITEMS,
        'n_movies': n_movies,
        'embedding_dim': item_embeddings.shape[1],
        'val_loss': val_loss,
        'epochs': n_epochs,
        'embedding_model': embedding_model
    }, checkpoint_path)
    print(f"Checkpoint saved to: {checkpoint_path}")

    # Save item embeddings separately (useful for analysis)
    embeddings_path = checkpoint_dir / f'concept_embeddings_{embedding_model}_{N_ITEMS}items.npy'
    np.save(embeddings_path, item_embeddings)

    # =============================================================================
    # TEST: NDCG BY TIMESTEP (RL Signal)
    # =============================================================================
    print("\n" + "=" * 80)
    print("TEST: NDCG@10 BY TIMESTEP (RL SIGNAL)")
    print("=" * 80)

    model.eval()

    # Fixed item order for testing (same as training)
    item_embeddings_tensor = torch.FloatTensor(item_embeddings).unsqueeze(0)

    timesteps = [1, 3, 5, 10, 20]
    ndcg_by_timestep = {t: [] for t in timesteps}

    for user_id in val_users[:30]:
        user_ratings = val_df[val_df['userId'] == user_id]
        user_items = user_ratings['itemIdx'].tolist()
        user_scores = user_ratings['rating'].tolist()

        if len(user_items) < 10:
            continue

        # Ground truth for movies only
        ground_truth = np.full(n_movies, np.nan)
        for item_idx, rating in zip(user_items, user_scores):
            if item_idx < n_movies:
                ground_truth[item_idx] = 1.0 if rating >= 4 else 0.0

        for n_revealed in timesteps:
            if n_revealed > len(user_items):
                continue

            # Build rating input
            ratings_input = torch.zeros(1, N_ITEMS, 3)
            ratings_input[:, :, 2] = 1  # All not_seen initially

            for i in range(min(n_revealed, len(user_items))):
                item_idx = user_items[i]
                rating = user_scores[i]
                ratings_input[:, item_idx, 2] = 0
                if rating >= 4:
                    ratings_input[:, item_idx, 1] = 1  # liked
                else:
                    ratings_input[:, item_idx, 0] = 1  # disliked

            with torch.no_grad():
                preds = model(item_embeddings_tensor, ratings_input)[:, -1, :n_movies].sigmoid().numpy().flatten()

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
        print(f"\n  Improvement ({first_t} -> {last_t} prefs): {improvement:+.4f}", flush=True)
        if improvement > 0.02:
            print("  >>> RL SIGNAL PRESENT!", flush=True)
        elif improvement > 0:
            print("  >>> Weak RL signal detected", flush=True)

    # =============================================================================
    # SANITY TEST: Single Movie Input (Liked vs Disliked)
    # =============================================================================
    print("\n" + "=" * 80, flush=True)
    print("SANITY TEST: SINGLE MOVIE INPUT", flush=True)
    print("=" * 80, flush=True)

    # Pick a few popular movies to test
    test_movies = [0, 1, 2, 3, 4]  # First 5 movies (most popular)

    liked_scores = []
    disliked_scores = []

    for movie_idx in test_movies:
        movie_name = descriptions[movie_idx]

        # Test with LIKED
        ratings_liked = torch.zeros(1, N_ITEMS, 3)
        ratings_liked[:, :, 2] = 1  # All not_seen
        ratings_liked[:, movie_idx, 2] = 0
        ratings_liked[:, movie_idx, 1] = 1  # liked

        # Test with DISLIKED
        ratings_disliked = torch.zeros(1, N_ITEMS, 3)
        ratings_disliked[:, :, 2] = 1  # All not_seen
        ratings_disliked[:, movie_idx, 2] = 0
        ratings_disliked[:, movie_idx, 0] = 1  # disliked

        with torch.no_grad():
            pred_liked = model(item_embeddings_tensor, ratings_liked)[:, -1, :n_movies].sigmoid().mean().item()
            pred_disliked = model(item_embeddings_tensor, ratings_disliked)[:, -1, :n_movies].sigmoid().mean().item()

        liked_scores.append(pred_liked)
        disliked_scores.append(pred_disliked)

        diff = pred_liked - pred_disliked
        print(f"  {movie_name[:40]:<40} | Liked: {pred_liked:.4f} | Disliked: {pred_disliked:.4f} | Diff: {diff:+.4f}", flush=True)

    avg_diff = np.mean(liked_scores) - np.mean(disliked_scores)
    print(f"\n  Average difference (liked - disliked): {avg_diff:+.4f}", flush=True)
    if avg_diff > 0.01:
        print("  >>> Model distinguishes liked from disliked!", flush=True)
    else:
        print("  >>> WARNING: Model may not distinguish liked from disliked", flush=True)

    return results, model, {
        'n_movies': n_movies,
        'n_genres': N_GENRES,
        'n_tags': N_TAGS,
        'n_items': N_ITEMS,
        'embedding_dim': embedding_dim,
        'item_embeddings': item_embeddings,
        'descriptions': descriptions
    }


if __name__ == "__main__":
    import argparse

    # List available models
    model_choices = list(EMBEDDING_MODELS.keys())

    parser = argparse.ArgumentParser(description="Concept Embedding Experiment")
    parser.add_argument('--model', type=str, default='minilm', choices=model_choices,
                       help=f'Embedding model: {model_choices}')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--users', type=int, default=5000, help='Number of users (default: 5000)')
    parser.add_argument('--movies', type=int, default=50, help='Number of movies (default: 50)')
    parser.add_argument('--tags', type=int, default=20, help='Number of genome tags (default: 20)')
    parser.add_argument('--actors', action='store_true', help='Include actors as items')
    parser.add_argument('--directors', action='store_true', help='Include directors as items')
    parser.add_argument('--n-actors', type=int, default=30, help='Max actors (default: 30)')
    parser.add_argument('--n-directors', type=int, default=20, help='Max directors (default: 20)')
    parser.add_argument('--min-ratings', type=int, default=200, help='Min user total ratings for dense users (default: 200)')
    parser.add_argument('--list-models', action='store_true', help='List available embedding models')
    args = parser.parse_args()

    if args.list_models:
        print("\nAvailable embedding models:")
        for key, config in EMBEDDING_MODELS.items():
            print(f"  {key:15s} - {config['desc']} (dim={config['dim']})")
        exit(0)

    results, model, config = run_concept_embedding_experiment(
        n_movies=args.movies,
        n_top_tags=args.tags,
        max_users=args.users,
        n_epochs=args.epochs,
        embedding_model=args.model,
        include_actors=args.actors,
        include_directors=args.directors,
        n_actors=args.n_actors,
        n_directors=args.n_directors,
        min_user_total_ratings=args.min_ratings
    )
