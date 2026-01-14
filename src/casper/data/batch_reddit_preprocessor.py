"""
Batch preprocess Reddit data with hybrid extraction.

- User preferences: LLM (extract ALL concepts with sentiment)
- Response concepts: Vocab matching (match against embedding space vocabulary)
"""

import json
import asyncio
import os
import re
import pandas as pd
from pathlib import Path
from typing import List, Dict, Set
import openai
from tqdm.asyncio import tqdm as async_tqdm


class BatchRedditPreprocessor:
    """Preprocess Reddit data with LLM + vocabulary matching."""

    def __init__(self, batch_size: int = 50, movielens_path: str = "data/movielens"):
        self.batch_size = batch_size
        self.client = openai.AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        # Load EXACT vocabulary used in embedding space
        print("Loading embedding space vocabulary...")
        self.vocabulary = self._load_embedding_vocabulary(Path(movielens_path))
        print(f"Loaded {len(self.vocabulary)} entities (genome tags + movies + people)")

    def _load_embedding_vocabulary(self, data_path: Path) -> Set[str]:
        """
        Load the EXACT vocabulary used in SentenceBERTEmbeddingSpace.

        This matches what's in src/casper/models/embedding_space.py:
        - 1,128 genome tags
        - ~5,000 top-rated movie titles
        - ~50 curated directors/actors
        """
        vocab = set()

        # 1. Load genome tags (1,128 rich descriptors)
        genome_tags_path = data_path / "genome-tags.csv"
        if genome_tags_path.exists():
            tags_df = pd.read_csv(genome_tags_path)
            for tag in tags_df['tag']:
                tag_clean = str(tag).strip().lower()
                if tag_clean:  # Filter out empty strings
                    vocab.add(tag_clean)

        # 2. Load top 5,000 most-rated movies
        movies_path = data_path / "movies.csv"
        ratings_path = data_path / "ratings.csv"

        if movies_path.exists() and ratings_path.exists():
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
                clean_title = str(title).split('(')[0].strip().lower()
                if clean_title:  # Filter out empty strings
                    vocab.add(clean_title)

        # 3. Add curated directors & actors (same as embedding_space.py)
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
        for name in popular_people:
            vocab.add(name.lower())

        return vocab

    def extract_concepts_from_text(self, text: str) -> List[str]:
        """
        Extract concepts by matching against embedding space vocabulary.

        Matches against ~6K entities:
        - 1,128 genome tags (themes, genres, moods)
        - ~5,000 top movie titles
        - ~50 directors/actors

        This is the EXACT vocabulary used in the embedding space.
        """
        if not text:
            return []

        text_lower = text.lower()
        matched = []

        # Sort vocabulary by length (longest first) to match multi-word phrases before single words
        # e.g., "Christopher Nolan" before "Christopher"
        vocab_sorted = sorted(self.vocabulary, key=len, reverse=True)

        for entity in vocab_sorted:
            # Word boundary matching to avoid partial matches
            # e.g., "action" matches but not "action-packed" -> "action"
            pattern = r'\b' + re.escape(entity) + r'\b'

            if re.search(pattern, text_lower):
                matched.append(entity)

        return matched

    async def extract_preferences(self, text: str, debug: bool = False) -> Dict:
        """
        Extract ALL movie-related concepts from user post using LLM.

        This is used for BOTH user posts AND responses - extracts everything mentioned.
        """
        # Improved prompt with concrete examples including genome-type tags
        prompt = f"""Extract ALL movie-related concepts from this text and classify their sentiment.

Text: "{text[:500]}"

Extract these types of concepts (ONLY if explicitly mentioned):
- Movie titles: "Inception", "Ex Machina", "Oldboy", "Devil Wears Prada"
- Genres: "sci-fi", "thriller", "horror", "comedy", "drama", "action", "rom com"
- Directors: "Christopher Nolan", "Tarantino", "Spielberg"
- Actors: "Leonardo DiCaprio", "Meryl Streep"
- Themes/tags: "revenge", "dystopian", "time travel", "coming of age", "mind-bending", "suspenseful", "dark", "uplifting", "strong female protagonist", "work culture", "female friendship", "immigrant experience", "sisterhood", "adulting"
- Time periods: "90s", "early 2000s", "1980s", "classic"
- Other: "foreign films", "indie", "based on true story"

Sentiment rules:
- "liked": positive OR user requests (love, enjoyed, favorite, great, like, best, looking for, want, need, prefer, interested in)
- "neutral": just mentioned without preference (have you seen, check out, similar to, also, mentions)
- "disliked": negative (hate, dislike, not a fan, avoid, no rom coms, not into, boring)

Keep each concept 1-3 words. Extract EVERYTHING mentioned.

Return JSON:
{{"liked": [], "neutral": [], "disliked": []}}"""

        try:
            # For gpt-5 reasoning models:
            # - Reasoning tokens are SEPARATE from completion tokens
            # - Model automatically allocates reasoning budget
            # - max_completion_tokens = tokens for actual output AFTER reasoning
            # - Need enough tokens to ensure output isn't truncated
            response = await self.client.chat.completions.create(
                model="gpt-5-nano",
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=10000  # Very high to ensure reasoning + output never truncate
            )
            content = response.choices[0].message.content

            if debug:
                print(f"  [DEBUG] LLM raw response: '{content}'")
                if hasattr(response, 'usage'):
                    usage = response.usage
                    print(f"  [DEBUG] Token usage:")
                    print(f"    completion_tokens: {usage.completion_tokens}")
                    if hasattr(usage, 'completion_tokens_details'):
                        details = usage.completion_tokens_details
                        print(f"    reasoning_tokens: {details.reasoning_tokens}")
                    print(f"    finish_reason: {response.choices[0].finish_reason}")

            if not content or not content.strip():
                if debug:
                    print(f"  [DEBUG] Empty response from LLM!")
                return {"liked": [], "neutral": [], "disliked": []}

            content = content.strip()
            return json.loads(content)
        except Exception as e:
            if debug:
                print(f"  [DEBUG] LLM Error: {e}")
            return {"liked": [], "neutral": [], "disliked": []}

    async def process_pair(self, pair: Dict, debug: bool = False) -> Dict:
        """
        Process one conversation pair.

        - User post: LLM extraction (need sentiment analysis)
        - Response: LLM extraction (get ALL concepts, even if not in vocab)
        """
        user_post = pair.get("user_post", "")
        response = pair.get("response", "")

        # Extract user preferences with LLM (need sentiment analysis)
        user_prefs = await self.extract_preferences(user_post, debug=debug)

        # Normalize LLM output
        if isinstance(user_prefs, Exception) or not isinstance(user_prefs, dict):
            user_prefs = {"liked": [], "neutral": [], "disliked": []}
        else:
            user_prefs = {
                "liked": user_prefs.get("liked", []) if isinstance(user_prefs.get("liked", []), list) else [],
                "neutral": user_prefs.get("neutral", []) if isinstance(user_prefs.get("neutral", []), list) else [],
                "disliked": user_prefs.get("disliked", []) if isinstance(user_prefs.get("disliked", []), list) else []
            }

        # Extract response concepts with LLM (same extraction, just flatten to list)
        response_extraction = await self.extract_preferences(response, debug=debug)

        # Normalize and flatten all concepts (don't need sentiment for responses)
        if isinstance(response_extraction, Exception) or not isinstance(response_extraction, dict):
            response_concepts = []
        else:
            # Combine all concepts from all sentiment categories
            response_concepts = []
            for sentiment in ["liked", "neutral", "disliked"]:
                concepts = response_extraction.get(sentiment, [])
                if isinstance(concepts, list):
                    response_concepts.extend(concepts)

        return {
            "user_preferences": user_prefs,
            "response_concepts": response_concepts
        }

    async def process_all(self, pairs: List[Dict], output_path: Path) -> List[Dict]:
        """Process all pairs with progress bar and intermediate saves."""
        all_results = []

        # Load existing results if any
        if output_path.exists():
            with open(output_path, 'r', encoding='utf-8') as f:
                all_results = json.load(f)
            print(f"Resuming: {len(all_results)} pairs already processed")
            pairs = pairs[len(all_results):]

        pbar = async_tqdm(total=len(pairs), desc="Processing", unit="pair")

        for i in range(0, len(pairs), self.batch_size):
            batch = pairs[i:i+self.batch_size]

            tasks = [self.process_pair(p) for p in batch]
            batch_results = await asyncio.gather(*tasks)

            all_results.extend(batch_results)

            # Save after each batch
            valid_count = sum(1 for r in all_results if r.get("response_concepts"))
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(all_results, f, indent=2, ensure_ascii=False)

            pbar.update(len(batch))
            pbar.set_postfix({"valid": valid_count, "total": len(all_results)})

        pbar.close()
        return all_results

    def process_reddit_file(self, input_path: Path, output_path: Path):
        """Load, process, and save Reddit data."""
        print(f"Loading {input_path.name}...")

        with open(input_path, 'r', encoding='utf-8') as f:
            pairs = json.load(f)

        print(f"Processing {len(pairs)} pairs (batch={self.batch_size})...")

        try:
            loop = asyncio.get_running_loop()
            # We're in Jupyter - use nest_asyncio
            import nest_asyncio
            nest_asyncio.apply()
            all_results = loop.run_until_complete(self.process_all(pairs, output_path))
        except RuntimeError:
            # No running loop - create new one
            all_results = asyncio.run(self.process_all(pairs, output_path))

        # Filter: ONLY need non-empty response_concepts (target)
        # Empty user_preferences is VALID (represents "no preferences known yet")
        valid_pairs = [p for p in all_results if p.get("response_concepts")]

        # Save final filtered version
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(valid_pairs, f, indent=2, ensure_ascii=False)

        print(f"\nComplete: {len(all_results)} processed | {len(valid_pairs)} with concepts")
        print(f"Saved to {output_path}")

        return valid_pairs


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python batch_reddit_preprocessor.py <reddit_data_dir>")
        sys.exit(1)

    reddit_dir = Path(sys.argv[1])
    processor = BatchRedditPreprocessor(batch_size=50)

    for json_file in reddit_dir.glob("*_conversations.json"):
        output_file = json_file.parent / f"{json_file.stem}_processed.json"
        processor.process_reddit_file(json_file, output_file)
