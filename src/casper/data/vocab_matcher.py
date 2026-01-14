"""
Fast vocabulary matcher for Reddit data.

Instead of using LLM to extract concepts, just match against MovieLens vocab:
- 62,423 movie titles
- 1,128 genome tags (themes, genres, moods, directors, etc.)

Much faster, more reliable, no API costs!
"""

import re
import pandas as pd
from pathlib import Path
from typing import List, Set
from functools import lru_cache


class VocabMatcher:
    """Match movie concepts against MovieLens vocabulary."""

    def __init__(self, movielens_path: str):
        self.movielens_path = Path(movielens_path)
        self._load_vocab()

    def _load_vocab(self):
        """Load MovieLens titles and tags."""
        # Load movie titles
        movies = pd.read_csv(self.movielens_path / 'movies.csv')

        # Extract clean titles (remove year)
        self.movie_titles = set()
        for title in movies['title']:
            # Remove year: "Toy Story (1995)" -> "Toy Story"
            clean = re.sub(r'\s*\(\d{4}\).*$', '', title).strip()
            self.movie_titles.add(clean.lower())

            # Also add without articles
            for article in ['the ', 'a ', 'an ']:
                if clean.lower().startswith(article):
                    self.movie_titles.add(clean[len(article):].lower())

        # Load genome tags
        tags = pd.read_csv(self.movielens_path / 'genome-tags.csv')
        self.genome_tags = set(tag.lower() for tag in tags['tag'])

        # Combined vocab
        self.vocab = self.movie_titles | self.genome_tags

        # Pre-sort by length (longest first) for efficient matching
        self.sorted_vocab = sorted(self.vocab, key=len, reverse=True)

        print(f"Loaded vocab: {len(self.movie_titles)} movies + {len(self.genome_tags)} tags = {len(self.vocab)} total")

    def extract_concepts(self, text: str) -> List[str]:
        """
        Extract concepts by matching text against vocab.

        Fast approach: only check vocab items that share words with text.
        """
        if not text:
            return []

        text_lower = text.lower()

        # Extract words from text
        text_words = set(re.findall(r'\b\w+\b', text_lower))

        # Filter vocab to only candidates (shares at least one word with text)
        candidates = []
        for concept in self.sorted_vocab:
            concept_words = set(re.findall(r'\b\w+\b', concept))
            if concept_words & text_words:  # Intersection
                candidates.append(concept)

        # Now match candidates (much smaller list!)
        matched = []
        for concept in candidates:
            # Skip if too short
            if len(concept) < 4:  # At least 4 chars to avoid noise
                continue

            # Exact phrase match with word boundaries
            pattern = r'\b' + re.escape(concept) + r'\b'
            if re.search(pattern, text_lower):
                matched.append(concept)

        return matched

    def extract_user_post_concepts(self, text: str) -> dict:
        """
        Extract concepts from user post with sentiment.

        Analyzes context to determine liked/neutral/disliked.
        """
        concepts = self.extract_concepts(text)

        if not concepts:
            return {"liked": [], "neutral": [], "disliked": []}

        text_lower = text.lower()

        liked = []
        neutral = []
        disliked = []

        for concept in concepts:
            # Find context around concept
            pos = text_lower.find(concept)
            if pos == -1:
                neutral.append(concept)
                continue

            # Get surrounding context (50 chars before/after)
            start = max(0, pos - 50)
            end = min(len(text_lower), pos + len(concept) + 50)
            context = text_lower[start:end]

            # Sentiment indicators
            if any(phrase in context for phrase in [
                'love', 'favorite', 'enjoyed', 'amazing', 'great', 'excellent',
                'adore', 'like', 'prefer', 'best'
            ]):
                liked.append(concept)
            elif any(phrase in context for phrase in [
                'hate', 'dislike', 'not a fan', 'avoid', 'boring', 'bad',
                'worst', 'terrible', 'no rom com', 'not interested'
            ]):
                disliked.append(concept)
            else:
                neutral.append(concept)

        return {
            "liked": list(set(liked)),
            "neutral": list(set(neutral)),
            "disliked": list(set(disliked))
        }

    def extract_response_concepts(self, text: str) -> List[str]:
        """
        Extract concepts from expert response.

        Just returns all matched concepts (no sentiment needed).
        """
        return self.extract_concepts(text)
