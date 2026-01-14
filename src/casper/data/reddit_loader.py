"""
Reddit data loader for training.

Loads preprocessed Reddit conversation data for RL actor pretraining.
"""

from pathlib import Path
from typing import List, Dict
import json


class RedditDataLoader:
    """
    Loads and processes Reddit conversation data for actor pretraining.

    Expected data format (JSON):
    {
        "user_post": "I love Inception",
        "response": "Have you seen Interstellar?",
        "response_concepts": ["Interstellar", "Nolan", "sci-fi"]
    }
    """

    def __init__(self, data_path: str):
        self.data_path = Path(data_path)

    def load(self) -> List[Dict]:
        """Load Reddit posts from JSON files, filtering out removed/deleted content."""
        if not self.data_path.exists():
            raise FileNotFoundError(
                f"Reddit data not found at {self.data_path}. "
                f"Run reddit_scraper.py first to collect training data."
            )

        all_posts = []
        removed_count = 0

        for json_file in self.data_path.glob("*.json"):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    posts = json.load(f)

                    # Filter out posts/responses with [removed] or [deleted]
                    for post in posts:
                        user_post = post.get("user_post", "").lower()
                        response = post.get("response", "").lower()

                        if "[removed]" in user_post or "[deleted]" in user_post:
                            removed_count += 1
                            continue
                        if "[removed]" in response or "[deleted]" in response:
                            removed_count += 1
                            continue

                        all_posts.append(post)

            except Exception as e:
                print(f"Error loading {json_file}: {e}")

        if removed_count > 0:
            print(f"Filtered out {removed_count} removed/deleted posts")

        return all_posts
