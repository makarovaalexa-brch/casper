"""
Build training data for two-tower recommender from MovieLens.

Extracts genome tags + genres from user ratings to create structured preferences.
"""

import pandas as pd
from tqdm import tqdm
from pathlib import Path
from typing import Tuple, List


def create_recommender_training_data(
    ratings_path: Path,
    movies_path: Path,
    genome_scores_path: Path,
    genome_tags_path: Path,
    num_users: int
) -> Tuple[List[str], List[List[int]], List[List[int]]]:
    """
    Create training data in PRODUCTION format.

    Format: "likes: movie title, sci-fi, action | dislikes: horror"
    This MATCHES what preference_extractor.preferences_to_text() produces!

    Args:
        ratings_path: Path to ratings.csv
        movies_path: Path to movies.csv
        genome_scores_path: Path to genome-scores.csv
        genome_tags_path: Path to genome-tags.csv
        num_users: Number of top users to sample

    Returns:
        (preference_texts, liked_movie_ids, disliked_movie_ids) tuple
    """
    print("Loading MovieLens data...")
    ratings_df = pd.read_csv(ratings_path)
    movies_df = pd.read_csv(movies_path)

    print("Loading genome data (15M+ rows)...")
    genome_scores_df = pd.read_csv(genome_scores_path)
    genome_tags_df = pd.read_csv(genome_tags_path)

    # Build tag ID -> tag name mapping
    tag_id_to_name = dict(zip(genome_tags_df['tagId'], genome_tags_df['tag']))

    # OPTIMIZATION: Pre-group genome scores by movieId for O(1) lookup
    print("Indexing genome scores by movieId (makes extraction 1000x faster)...")
    genome_by_movie = genome_scores_df.groupby('movieId')

    preference_texts = []
    liked_movie_ids = []
    disliked_movie_ids = []  # NEW: Track explicitly disliked movies

    user_counts = ratings_df['userId'].value_counts()
    sampled_users = user_counts.head(num_users).index.tolist()

    print(f"Extracting preferences from {num_users} users...")
    for user_id in tqdm(sampled_users, desc="Processing users", unit="user"):
        user_ratings = ratings_df[ratings_df['userId'] == user_id]

        # High ratings (>= 4.0) → liked, Low ratings (< 2.5) → disliked
        high_ratings = user_ratings[user_ratings['rating'] >= 4.0]
        low_ratings = user_ratings[user_ratings['rating'] < 2.5]

        if len(high_ratings) < 2:
            continue

        # Sample movies
        liked_sample = high_ratings.sample(min(5, len(high_ratings)), random_state=42)
        disliked_sample = low_ratings.sample(min(3, len(low_ratings)), random_state=42) if len(low_ratings) > 0 else pd.DataFrame()

        liked_concepts = []
        disliked_concepts = []

        # Extract from LIKED movies
        for movie_id in liked_sample['movieId']:
            movie_row = movies_df[movies_df['movieId'] == movie_id]
            if len(movie_row) == 0:
                continue

            title = movie_row.iloc[0]['title']
            liked_concepts.append(title.lower())

            genres_str = movie_row.iloc[0]['genres']
            if pd.notna(genres_str) and genres_str != '(no genres listed)':
                genres = [g.lower() for g in genres_str.split('|')]
                liked_concepts.extend(genres)

            try:
                movie_tags = genome_by_movie.get_group(movie_id)
                top_tags = movie_tags[movie_tags['relevance'] > 0.5].nlargest(5, 'relevance')
                for _, tag_row in top_tags.iterrows():
                    tag_name = tag_id_to_name.get(tag_row['tagId'], '').lower()
                    if tag_name and len(tag_name) > 2:
                        liked_concepts.append(tag_name)
            except KeyError:
                pass

        # Extract from DISLIKED movies
        if len(disliked_sample) > 0:
            for movie_id in disliked_sample['movieId']:
                movie_row = movies_df[movies_df['movieId'] == movie_id]
                if len(movie_row) == 0:
                    continue

                title = movie_row.iloc[0]['title']
                disliked_concepts.append(title.lower())

                genres_str = movie_row.iloc[0]['genres']
                if pd.notna(genres_str) and genres_str != '(no genres listed)':
                    genres = [g.lower() for g in genres_str.split('|')]
                    disliked_concepts.extend(genres)

                try:
                    movie_tags = genome_by_movie.get_group(movie_id)
                    top_tags = movie_tags[movie_tags['relevance'] > 0.5].nlargest(3, 'relevance')
                    for _, tag_row in top_tags.iterrows():
                        tag_name = tag_id_to_name.get(tag_row['tagId'], '').lower()
                        if tag_name and len(tag_name) > 2:
                            disliked_concepts.append(tag_name)
                except KeyError:
                    pass

        # Remove duplicates, keep top concepts
        liked_concepts = list(dict.fromkeys(liked_concepts))[:10]
        disliked_concepts = list(dict.fromkeys(disliked_concepts))[:5]

        if not liked_concepts:
            continue

        # Format as preferences_to_text() does
        parts = []
        if liked_concepts:
            parts.append(f"likes: {', '.join(liked_concepts)}")
        if disliked_concepts:
            parts.append(f"dislikes: {', '.join(disliked_concepts)}")

        pref_text = " | ".join(parts)
        target_movie_ids = high_ratings['movieId'].tolist()
        disliked_ids = low_ratings['movieId'].tolist() if len(low_ratings) > 0 else []

        preference_texts.append(pref_text)
        liked_movie_ids.append(target_movie_ids)
        disliked_movie_ids.append(disliked_ids)  # NEW: Collect disliked IDs

    return preference_texts, liked_movie_ids, disliked_movie_ids
