"""
SentenceBERT embedding space for movie entities.

Handles multi-word entities (e.g., "The Dark Knight", "Christopher Nolan")
with no vocabulary limitations - embeds any text on-the-fly.
"""

import numpy as np
from typing import List, Tuple, Optional
from sentence_transformers import SentenceTransformer
from pathlib import Path
import pandas as pd
import yaml


class SentenceBERTEmbeddingSpace:
    """
    SentenceBERT embedding space for movie entities.

    Handles multi-word entities: "The Dark Knight", "Christopher Nolan"
    No vocabulary limitations - can embed any text on-the-fly.
    Uses FAISS for efficient similarity search.
    """

    def __init__(
        self,
        movielens_data_path: Optional[str] = None,
        config_path: Optional[str] = None
    ):
        """
        Initialize SentenceBERT embedding space.

        Args:
            movielens_data_path: Path to MovieLens data for entity extraction
            config_path: Path to config.yaml (auto-detected if None)
        """
        print("Initializing SentenceBERT embedding space...")

        # Store data path for caching
        self._data_path = movielens_data_path

        # Load config
        if config_path is None:
            project_root = Path(__file__).parent.parent.parent.parent
            config_path = str(project_root / "config" / "config.yaml")

        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        embedding_config = config['models']['embedding_space']

        # Load SentenceBERT with configured model
        model_name = embedding_config['model_name']
        self.encoder = SentenceTransformer(model_name)
        self.embedding_dim = embedding_config['embedding_dim']

        print(f"Loaded {model_name}: {self.embedding_dim}-dim")

        # Extract MovieLens entities (movies, actors, directors)
        self.movie_entities = self._extract_movielens_entities(movielens_data_path)
        print(f"Extracted {len(self.movie_entities)} MovieLens entities")

        # Pre-compute entity embeddings for fast lookup
        self._cache_entity_embeddings()

    def _extract_movielens_entities(self, data_path: Optional[str]) -> List[str]:
        """
        Extract entities from MovieLens dataset.

        Entity space composition (~6,200 entities):
        - 1,128 genome tags (rich descriptors from MovieLens)
        - ~5,000 movie titles (top-rated movies only)
        - ~50 popular directors/actors (curated list)

        Returns:
            List of entity strings for semantic search
        """
        entities = []

        if not data_path:
            print("Warning: No MovieLens path provided, using minimal entity set")
            return self._get_fallback_entities()

        data_path = Path(data_path)

        # 1. Load MovieLens Genome Tags (1,128 rich tags)
        genome_tags_path = data_path / "genome-tags.csv"
        if genome_tags_path.exists():
            try:
                tags_df = pd.read_csv(genome_tags_path)
                genome_tags = tags_df['tag'].tolist()
                entities.extend(genome_tags)
                print(f"Loaded {len(genome_tags)} genome tags from MovieLens")
            except Exception as e:
                print(f"Warning: Could not load genome tags: {e}")
        else:
            print(f"Warning: genome-tags.csv not found at {genome_tags_path}")

        # 2. Load top 5,000 most-rated movies (not all 62K)
        movies_path = data_path / "movies.csv"
        ratings_path = data_path / "ratings.csv"

        if movies_path.exists() and ratings_path.exists():
            try:
                movies_df = pd.read_csv(movies_path)
                ratings_df = pd.read_csv(ratings_path)

                # Count ratings per movie
                rating_counts = ratings_df['movieId'].value_counts()

                # Get top 5000 most-rated movies
                top_movie_ids = rating_counts.head(5000).index.tolist()
                top_movies = movies_df[movies_df['movieId'].isin(top_movie_ids)]

                # Extract clean titles
                for title in top_movies['title']:
                    # Remove year: "Inception (2010)" -> "Inception"
                    clean_title = title.split('(')[0].strip()
                    entities.append(clean_title)

                print(f"Loaded {len(top_movies)} top-rated movie titles")

            except Exception as e:
                print(f"Warning: Could not load movies: {e}")
        else:
            print("Warning: movies.csv or ratings.csv not found")

        # 3. Add curated directors & actors (~50 popular names)
        popular_people = [
            # Directors
            "Christopher Nolan", "Quentin Tarantino", "Steven Spielberg",
            "Martin Scorsese", "Stanley Kubrick", "Alfred Hitchcock",
            "David Fincher", "Wes Anderson", "Denis Villeneuve", "Ridley Scott",
            "James Cameron", "Kathryn Bigelow", "Greta Gerwig", "Jordan Peele",
            "Bong Joon-ho", "Guillermo del Toro", "Paul Thomas Anderson",
            "Coen Brothers", "Wong Kar-wai", "Hayao Miyazaki",

            # Actors
            "Leonardo DiCaprio", "Tom Hanks", "Meryl Streep", "Denzel Washington",
            "Brad Pitt", "Scarlett Johansson", "Samuel L. Jackson", "Morgan Freeman",
            "Cate Blanchett", "Robert De Niro", "Al Pacino", "Natalie Portman",
            "Christian Bale", "Frances McDormand", "Joaquin Phoenix", "Viola Davis",
            "Daniel Day-Lewis", "Tilda Swinton", "Michael Caine", "Anthony Hopkins"
        ]
        entities.extend(popular_people)

        print(f"Total entities: {len(entities)} (genome tags + movies + people)")

        return entities

    def _get_fallback_entities(self) -> List[str]:
        """Minimal entity set if MovieLens data not available."""
        return [
            # Core genres
            "action", "comedy", "drama", "thriller", "sci-fi", "horror",
            "romance", "animation", "documentary", "fantasy",

            # Basic moods
            "dark", "uplifting", "intense", "funny", "emotional",
            "suspenseful", "thought-provoking", "inspiring"
        ]

    def _cache_entity_embeddings(self):
        """Pre-compute embeddings for all known entities and build FAISS index."""
        # Try to load from cache first
        cache_dir = Path(self.movie_entities[0]).parent if hasattr(self.movie_entities[0], 'parent') else Path.cwd() / 'data'
        if hasattr(self, '_data_path') and self._data_path:
            cache_dir = Path(self._data_path).parent / 'processed'
        else:
            # Fallback: use project root
            cache_dir = Path(__file__).parent.parent.parent.parent / 'data' / 'processed'

        cache_dir.mkdir(parents=True, exist_ok=True)

        # Create cache filename based on model and entity count
        import hashlib
        entity_hash = hashlib.md5(''.join(sorted(self.movie_entities[:100])).encode()).hexdigest()[:8]
        cache_file = cache_dir / f'embeddings_{self.encoder._modules["0"].auto_model.name_or_path.replace("/", "_")}_{len(self.movie_entities)}_{entity_hash}.npz'

        if cache_file.exists():
            print(f"Loading cached embeddings from {cache_file}...")
            try:
                cached_data = np.load(cache_file, allow_pickle=True)
                self.entity_list = cached_data['entity_list'].tolist()
                self.entity_embeddings = cached_data['entity_embeddings']
                self.entity_embeddings_normalized = cached_data['entity_embeddings_normalized']

                # Rebuild FAISS index from cached embeddings
                try:
                    import faiss
                    self.faiss_index = faiss.IndexFlatIP(self.embedding_dim)
                    self.faiss_index.add(self.entity_embeddings_normalized.astype('float32'))
                    self.use_faiss = True
                    print(f"Loaded {len(self.entity_list)} cached embeddings with FAISS index")
                except ImportError:
                    self.use_faiss = False
                    print(f"Loaded {len(self.entity_list)} cached embeddings (FAISS not available)")

                return
            except Exception as e:
                print(f"Warning: Failed to load cache ({e}), rebuilding embeddings...")

        print("Pre-computing entity embeddings...")

        # Encode all entities at once (batch encoding is faster)
        self.entity_list = self.movie_entities
        self.entity_embeddings = self.encoder.encode(
            self.entity_list,
            convert_to_numpy=True,
            show_progress_bar=True
        )

        # Normalize embeddings for cosine similarity
        norms = np.linalg.norm(self.entity_embeddings, axis=1, keepdims=True)
        self.entity_embeddings_normalized = self.entity_embeddings / (norms + 1e-8)

        # Save to cache
        try:
            np.savez_compressed(
                cache_file,
                entity_list=np.array(self.entity_list, dtype=object),
                entity_embeddings=self.entity_embeddings,
                entity_embeddings_normalized=self.entity_embeddings_normalized
            )
            print(f"Embeddings cached to {cache_file}")
        except Exception as e:
            print(f"Warning: Failed to cache embeddings ({e})")

        # Build FAISS index for fast similarity search
        try:
            import faiss
            self.faiss_index = faiss.IndexFlatIP(self.embedding_dim)  # Inner product (cosine sim with normalized vectors)
            self.faiss_index.add(self.entity_embeddings_normalized.astype('float32'))
            self.use_faiss = True
            print(f"Cached {len(self.entity_list)} entity embeddings with FAISS index")
        except ImportError:
            print(f"FAISS not available, using numpy fallback")
            self.use_faiss = False
            print(f"Cached {len(self.entity_list)} entity embeddings")

    def get_embedding(self, text: str) -> np.ndarray:
        """
        Get SentenceBERT embedding for any text.

        Works for:
        - Single words: "action"
        - Multi-word entities: "The Dark Knight"
        - Phrases: "intense psychological thriller"
        """
        return self.encoder.encode(text, convert_to_numpy=True)

    def find_nearest_entities(
        self,
        embedding: np.ndarray,
        top_k: int = 5
    ) -> List[Tuple[str, float]]:
        """
        Find nearest movie entities to a given embedding using FAISS.

        Uses cosine similarity (via normalized embeddings + inner product).
        Always returns top_k entities by similarity (no threshold).

        Args:
            embedding: Query embedding (384-dim)
            top_k: How many entities to return

        Returns:
            List of (entity, similarity) tuples, sorted by similarity (descending)
        """
        if len(self.entity_list) == 0:
            return []

        # Normalize query embedding
        embedding_norm = embedding / (np.linalg.norm(embedding) + 1e-8)
        embedding_norm = embedding_norm.astype('float32').reshape(1, -1)

        if self.use_faiss:
            # Use FAISS for fast similarity search
            similarities, indices = self.faiss_index.search(embedding_norm, min(top_k, len(self.entity_list)))

            results = []
            for sim, idx in zip(similarities[0], indices[0]):
                entity = self.entity_list[idx]
                results.append((entity, float(sim)))
        else:
            # Numpy fallback (if FAISS not available)
            similarities = np.dot(self.entity_embeddings_normalized, embedding_norm.flatten())

            # Get top-k using argpartition (O(n) instead of O(n log n))
            k = min(top_k, len(similarities))
            top_indices = np.argpartition(similarities, -k)[-k:]
            # Sort only the top-k
            top_indices = top_indices[np.argsort(similarities[top_indices])][::-1]

            results = []
            for idx in top_indices:
                entity = self.entity_list[idx]
                results.append((entity, float(similarities[idx])))

        return results
