"""
Simple, fast movie title matcher.

Just match movie titles from MovieLens - no LLM needed!
Much faster and more reliable than LLM extraction.
"""

import re
import pandas as pd
from pathlib import Path
from typing import List, Dict


class SimpleMovieMatcher:
    """Fast movie title matcher using regex."""

    def __init__(self, movielens_path: str):
        print("Loading MovieLens titles...")
        movies = pd.read_csv(Path(movielens_path) / 'movies.csv')

        # Build title patterns
        self.title_patterns = []

        for _, row in movies.iterrows():
            # Extract title without year
            title = row['title']
            clean_title = re.sub(r'\s*\(\d{4}\).*$', '', title).strip()

            # Create pattern (case-insensitive, word boundaries)
            # Escape special regex chars
            escaped = re.escape(clean_title)
            pattern = re.compile(r'\b' + escaped + r'\b', re.IGNORECASE)

            self.title_patterns.append((clean_title, pattern))

        print(f"Loaded {len(self.title_patterns)} movie titles")

    def extract_movie_titles(self, text: str) -> List[str]:
        """Extract movie titles from text."""
        if not text:
            return []

        matched = []
        for title, pattern in self.title_patterns:
            if pattern.search(text):
                matched.append(title)

        return matched

    def process_reddit_pair(self, user_post: str, response: str) -> Dict:
        """
        Process a Reddit conversation pair.

        Returns: {user_preferences, response_concepts}
        """
        # Extract from user post
        user_titles = self.extract_movie_titles(user_post)
        user_prefs = {
            "liked": user_titles,  # Simplified: assume mentioned = liked
            "neutral": [],
            "disliked": []
        }

        # Extract from response
        response_titles = self.extract_movie_titles(response)

        return {
            "user_preferences": user_prefs,
            "response_concepts": response_titles
        }
