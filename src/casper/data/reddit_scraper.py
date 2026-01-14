"""
Reddit conversation data scraper for CASPER supervised pretraining.

PURPOSE:
Scrapes USER POST + RESPONSE pairs from movie subreddits to train the RL actor
to predict which movie concepts/entities to ask about next.

TRAINING PIPELINE:
1. Scrape Reddit conversations: User preference → Expert response
2. Extract concepts from RESPONSE (what expert asked about)
3. Train RL actor: Given user state → Predict embedding of next concept to explore

Example:
    User: "I love Inception"
    Expert Response: "Have you seen other Nolan films like Interstellar?"
    → Extract concepts: ["Christopher Nolan", "Interstellar", "director", "sci-fi"]
    → Train: state_embedding("love Inception") → concept_embedding("Nolan")

This teaches the RL actor which movie concepts are good follow-ups to user preferences.

Subreddits:
- r/MovieSuggestions (recommendation requests)
- r/movies (general discussion)
- r/TrueFilm (deep analysis)
- r/criterion (classic/art films)
"""

import json
import time
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime
import requests
import numpy as np
from sentence_transformers import SentenceTransformer


class RedditScraper:
    """
    Scrapes movie conversation data from Reddit using Pushshift API.

    No authentication needed - uses public Pushshift archive.
    """

    def __init__(self, output_dir: str = "data/reddit", use_semantic_filter: bool = True):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Pushshift API endpoint (public archive)
        self.api_base = "https://api.pullpush.io/reddit/search"

        # Semantic filtering
        self.use_semantic_filter = use_semantic_filter
        if use_semantic_filter:
            print("Initializing semantic filter with SentenceBERT...")
            self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
            # Pre-compute query embedding for movie recommendation requests
            self.query_embedding = self.encoder.encode(
                "movie recommendation request suggestion looking for films",
                convert_to_numpy=True
            )
            print("Semantic filter ready")

    def scrape_subreddit_questions(
        self,
        subreddit: str,
        max_posts: int = 100,
        keywords: List[str] = None,
        disable_semantic_filter: bool = False
    ) -> List[Dict]:
        """
        Scrape question posts from a subreddit.

        Args:
            subreddit: Subreddit name (without r/)
            max_posts: Maximum number of VALID posts (after filtering [removed]/[deleted])
            keywords: Filter posts containing these keywords
            disable_semantic_filter: If True, skip semantic filtering (use for r/MovieSuggestions)

        Returns:
            List of post dictionaries with questions and responses
        """
        from tqdm import tqdm

        if keywords is None:
            keywords = [
                "recommend", "suggestion", "looking for", "similar to",
                "what should", "any recommendations", "help me find"
            ]

        # MovieSuggestions is 100% recommendation posts, no need for semantic filter
        if subreddit.lower() == "moviesuggestions":
            disable_semantic_filter = True

        print(f"Scraping r/{subreddit}... (target: {max_posts} valid posts)")
        if disable_semantic_filter:
            print(f"  Semantic filter: DISABLED (subreddit is recommendation-focused)")

        posts = []
        after = None
        total_checked = 0
        filtered_count = 0

        pbar = tqdm(total=max_posts, desc=f"r/{subreddit}", unit="valid posts")

        while len(posts) < max_posts:
            # Build query
            params = {
                "subreddit": subreddit,
                "size": 100,  # Always fetch 100 to account for filtering
                "sort": "desc",
                "sort_type": "created_utc",
            }

            if after:
                params["after"] = after

            try:
                # Get submissions
                response = requests.get(
                    f"{self.api_base}/submission",
                    params=params,
                    timeout=30
                )

                if response.status_code != 200:
                    pbar.write(f"API error: {response.status_code}")
                    break

                data = response.json()

                if "data" not in data or not data["data"]:
                    pbar.write(f"No more posts found (scraped {len(posts)} valid posts)")
                    break

                # Filter for question posts
                for post in data["data"]:
                    total_checked += 1

                    title = post.get("title", "")
                    selftext = post.get("selftext", "")
                    text = (title + " " + selftext).lower()

                    # Skip removed/deleted posts
                    if "[removed]" in text or "[deleted]" in text:
                        filtered_count += 1
                        continue

                    # Semantic filtering (if enabled and not disabled for this subreddit)
                    if self.use_semantic_filter and not disable_semantic_filter:
                        # Compute semantic similarity
                        text_embedding = self.encoder.encode(text, convert_to_numpy=True)
                        similarity = np.dot(self.query_embedding, text_embedding) / (
                            np.linalg.norm(self.query_embedding) * np.linalg.norm(text_embedding)
                        )

                        # Keep if similarity > threshold (0.4 = relaxed, includes casual language)
                        if similarity > 0.4:
                            posts.append({
                                "id": post.get("id"),
                                "title": post.get("title"),
                                "text": post.get("selftext"),
                                "author": post.get("author"),
                                "created_utc": post.get("created_utc"),
                                "score": post.get("score"),
                                "num_comments": post.get("num_comments"),
                                "url": f"https://reddit.com{post.get('permalink')}",
                                "similarity": float(similarity)
                            })

                            # Update progress bar
                            pbar.update(1)
                            pbar.set_postfix({
                                'checked': total_checked,
                                'filtered': filtered_count
                            })

                            if len(posts) >= max_posts:
                                break

                    elif disable_semantic_filter:
                        # No semantic filtering - accept all non-removed posts
                        posts.append({
                            "id": post.get("id"),
                            "title": post.get("title"),
                            "text": post.get("selftext"),
                            "author": post.get("author"),
                            "created_utc": post.get("created_utc"),
                            "score": post.get("score"),
                            "num_comments": post.get("num_comments"),
                            "url": f"https://reddit.com{post.get('permalink')}"
                        })

                        # Update progress bar
                        pbar.update(1)
                        pbar.set_postfix({
                            'checked': total_checked,
                            'filtered': filtered_count
                        })

                        if len(posts) >= max_posts:
                            break

                    else:
                        # Keyword filtering (fallback)
                        if any(keyword in text for keyword in keywords):
                            posts.append({
                                "id": post.get("id"),
                                "title": post.get("title"),
                                "text": post.get("selftext"),
                                "author": post.get("author"),
                                "created_utc": post.get("created_utc"),
                                "score": post.get("score"),
                                "num_comments": post.get("num_comments"),
                                "url": f"https://reddit.com{post.get('permalink')}"
                            })

                            # Update progress bar
                            pbar.update(1)
                            pbar.set_postfix({
                                'checked': total_checked,
                                'filtered': filtered_count
                            })

                            if len(posts) >= max_posts:
                                break

                # Update pagination
                if data["data"]:
                    after = data["data"][-1]["created_utc"]

                # Rate limiting
                time.sleep(1)

            except Exception as e:
                pbar.write(f"Error scraping: {e}")
                break

        pbar.close()

        print(f"\nScraping complete:")
        print(f"  Total posts checked: {total_checked}")
        print(f"  Filtered ([removed]/[deleted]): {filtered_count}")
        print(f"  Valid posts collected: {len(posts)}")

        return posts

    def get_post_comments(self, post_id: str, subreddit: str) -> List[Dict]:
        """
        Get comments for a specific post.

        Args:
            post_id: Reddit post ID
            subreddit: Subreddit name

        Returns:
            List of comment dictionaries
        """
        params = {
            "link_id": post_id,
            "subreddit": subreddit,
            "size": 50,
            "sort": "asc"
        }

        try:
            response = requests.get(
                f"{self.api_base}/comment",
                params=params,
                timeout=30
            )

            if response.status_code == 200:
                data = response.json()
                return data.get("data", [])

        except Exception as e:
            print(f"Error getting comments: {e}")

        return []

    def scrape_conversation_pairs(
        self,
        subreddit: str,
        max_pairs: int = 100
    ) -> List[Dict]:
        """
        Scrape conversation pairs: user post + helpful response.

        Creates MULTIPLE pairs per post if there are multiple good responses,
        maximizing training data from each thread.

        Args:
            subreddit: Subreddit name
            max_pairs: Maximum conversation pairs to collect

        Returns:
            List of conversation pairs with structure:
            {
                "id": "post123_5",  # Unique pair ID
                "post_id": "post123",  # Original post ID
                "user_post": "I love Inception",
                "response": "Have you seen Interstellar?",
                "response_score": 15,  # Upvotes
                "response_concepts": ["Interstellar", "Nolan", "sci-fi"]  # To be extracted
            }
        """
        print(f"Scraping conversation pairs from r/{subreddit}...")

        # First get posts
        posts = self.scrape_subreddit_questions(subreddit, max_posts=max_pairs * 2)

        conversation_pairs = []

        for post in posts:
            if len(conversation_pairs) >= max_pairs:
                break

            # Get comments for this post
            comments = self.get_post_comments(post["id"], subreddit)

            if not comments:
                continue

            # Find good responses (non-bot, substantive, upvoted)
            good_responses = []
            for comment in comments:
                body = comment.get("body", "")
                score = comment.get("score", 0)
                author = comment.get("author", "")

                # Filter criteria
                if (
                    len(body) > 50 and  # Substantive response
                    score >= 1 and  # At least 1 upvote
                    author != "AutoModerator" and
                    "[deleted]" not in body and
                    "[removed]" not in body
                ):
                    good_responses.append({
                        "body": body,
                        "score": score
                    })

            if not good_responses:
                continue

            # Create conversation pairs from ALL good responses (not just best one)
            # This gives us more training data
            user_text = post["title"] + " " + post.get("text", "")

            # Skip if post itself is removed/deleted
            if "[removed]" in user_text.lower() or "[deleted]" in user_text.lower():
                continue

            for response in good_responses:
                if len(conversation_pairs) >= max_pairs:
                    break

                conversation_pairs.append({
                    "id": f"{post['id']}_{len(conversation_pairs)}",  # Unique ID per pair
                    "post_id": post["id"],  # Original post ID
                    "user_post": user_text.strip(),
                    "response": response["body"],
                    "response_score": response["score"],
                    "subreddit": subreddit,
                    "post_url": post.get("url", "")
                })

            # Rate limiting
            time.sleep(0.5)

        print(f"Collected {len(conversation_pairs)} conversation pairs")
        return conversation_pairs

    def scrape_movie_subreddits(
        self,
        max_posts_per_subreddit: int = 100
    ) -> Dict[str, List[Dict]]:
        """
        Scrape multiple movie-related subreddits for conversation pairs.

        Returns:
            Dictionary mapping subreddit name to list of conversation pairs
        """
        subreddits = [
            "MovieSuggestions",
            "movies",
            "TrueFilm",
            "criterion",
            "Letterboxd"
        ]

        all_data = {}

        for subreddit in subreddits:
            pairs = self.scrape_conversation_pairs(
                subreddit,
                max_pairs=max_posts_per_subreddit
            )
            all_data[subreddit] = pairs

            # Save intermediate results
            self.save_data(pairs, f"{subreddit}_conversations.json")

            print(f"Completed r/{subreddit}: {len(pairs)} conversation pairs")
            time.sleep(2)  # Be nice to the API

        return all_data

    def save_data(self, data: List[Dict], filename: str):
        """Save scraped data to JSON file."""
        output_path = self.output_dir / filename

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        print(f"Saved to {output_path}")

    def load_data(self, filename: str) -> List[Dict]:
        """Load previously scraped data."""
        input_path = self.output_dir / filename

        if not input_path.exists():
            print(f"File not found: {input_path}")
            return []

        with open(input_path, 'r', encoding='utf-8') as f:
            return json.load(f)
