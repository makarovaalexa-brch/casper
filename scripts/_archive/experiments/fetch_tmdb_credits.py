"""
Fetch actors and directors from TMDB API for MovieLens movies.

Uses the links.csv file which contains tmdbId for each movie.

Set TMDB_API_KEY environment variable or create .env file with:
TMDB_API_KEY=your_api_key_here

Get API key from: https://www.themoviedb.org/settings/api
"""

import os
import json
import time
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple

from pathlib import Path
try:
    from dotenv import load_dotenv
    # Load from casper/.env
    env_path = Path(__file__).parent.parent / '.env'
    if env_path.exists():
        load_dotenv(env_path)
except ImportError:
    pass

import requests


def get_movie_credits(tmdb_id: int, api_key: str) -> Dict:
    """Fetch credits (cast and crew) for a movie from TMDB."""
    url = f"https://api.themoviedb.org/3/movie/{tmdb_id}/credits"
    params = {"api_key": api_key}

    try:
        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 404:
            return None  # Movie not found
        else:
            print(f"Error {response.status_code} for tmdb_id {tmdb_id}")
            return None
    except Exception as e:
        print(f"Request failed for tmdb_id {tmdb_id}: {e}")
        return None


def extract_top_actors(credits: Dict, n: int = 5) -> List[str]:
    """Extract top N actors from credits."""
    if not credits or 'cast' not in credits:
        return []

    cast = credits['cast']
    # Sort by order (billing order)
    sorted_cast = sorted(cast, key=lambda x: x.get('order', 999))
    return [actor['name'] for actor in sorted_cast[:n]]


def extract_directors(credits: Dict) -> List[str]:
    """Extract director(s) from credits."""
    if not credits or 'crew' not in credits:
        return []

    directors = [
        person['name']
        for person in credits['crew']
        if person.get('job') == 'Director'
    ]
    return directors


def fetch_credits_for_movies(
    movie_ids: List[int],
    links_df: pd.DataFrame,
    api_key: str,
    n_actors: int = 5,
    rate_limit_delay: float = 0.25
) -> Tuple[Dict[int, List[str]], Dict[int, List[str]], set, set]:
    """
    Fetch actors and directors for a list of MovieLens movie IDs.

    Returns:
        movie_actors: Dict[movieId, List[actor_names]]
        movie_directors: Dict[movieId, List[director_names]]
        all_actors: set of all unique actor names
        all_directors: set of all unique director names
    """
    movie_actors = {}
    movie_directors = {}
    all_actors = set()
    all_directors = set()

    total = len(movie_ids)

    for i, movie_id in enumerate(movie_ids):
        # Get TMDB ID
        tmdb_row = links_df[links_df['movieId'] == movie_id]
        if tmdb_row.empty:
            print(f"  No TMDB ID for movie {movie_id}")
            continue

        tmdb_id = tmdb_row['tmdbId'].values[0]
        if pd.isna(tmdb_id):
            continue

        tmdb_id = int(tmdb_id)

        # Fetch credits
        credits = get_movie_credits(tmdb_id, api_key)

        if credits:
            actors = extract_top_actors(credits, n_actors)
            directors = extract_directors(credits)

            movie_actors[movie_id] = actors
            movie_directors[movie_id] = directors
            all_actors.update(actors)
            all_directors.update(directors)

            if (i + 1) % 10 == 0 or i == total - 1:
                print(f"  Processed {i+1}/{total} movies, "
                      f"{len(all_actors)} actors, {len(all_directors)} directors")

        # Rate limiting
        time.sleep(rate_limit_delay)

    return movie_actors, movie_directors, all_actors, all_directors


def save_credits_cache(
    movie_actors: Dict,
    movie_directors: Dict,
    all_actors: set,
    all_directors: set,
    cache_path: Path
):
    """Save credits to cache file."""
    cache_data = {
        'movie_actors': {str(k): v for k, v in movie_actors.items()},
        'movie_directors': {str(k): v for k, v in movie_directors.items()},
        'all_actors': sorted(list(all_actors)),
        'all_directors': sorted(list(all_directors))
    }

    with open(cache_path, 'w') as f:
        json.dump(cache_data, f, indent=2)

    print(f"Saved credits cache to {cache_path}")


def load_credits_cache(cache_path: Path) -> Tuple[Dict, Dict, List, List]:
    """Load credits from cache file."""
    with open(cache_path, 'r') as f:
        cache_data = json.load(f)

    movie_actors = {int(k): v for k, v in cache_data['movie_actors'].items()}
    movie_directors = {int(k): v for k, v in cache_data['movie_directors'].items()}
    all_actors = cache_data['all_actors']
    all_directors = cache_data['all_directors']

    return movie_actors, movie_directors, all_actors, all_directors


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Fetch TMDB credits for MovieLens movies")
    parser.add_argument('--n-movies', type=int, default=100, help='Number of top movies')
    parser.add_argument('--n-actors', type=int, default=5, help='Top actors per movie')
    parser.add_argument('--force', action='store_true', help='Force re-fetch even if cache exists')
    args = parser.parse_args()

    # Check API key
    api_key = os.environ.get('TMDB_API_KEY')
    if not api_key:
        print("ERROR: TMDB_API_KEY not set")
        print("Get API key from: https://www.themoviedb.org/settings/api")
        print("Set it with: export TMDB_API_KEY=your_key")
        return

    # Paths
    DATA_DIR = Path('C:/dev/phd/casper/data/movielens')
    CACHE_DIR = DATA_DIR / '.cache'
    CACHE_DIR.mkdir(exist_ok=True)

    cache_file = CACHE_DIR / f'credits_top{args.n_movies}_actors{args.n_actors}.json'

    # Check cache
    if cache_file.exists() and not args.force:
        print(f"Loading from cache: {cache_file}")
        movie_actors, movie_directors, all_actors, all_directors = load_credits_cache(cache_file)
        print(f"Loaded: {len(movie_actors)} movies, {len(all_actors)} actors, {len(all_directors)} directors")
        return movie_actors, movie_directors, all_actors, all_directors

    # Load data
    print("Loading MovieLens data...")
    ratings = pd.read_csv(DATA_DIR / 'ratings.csv')
    links = pd.read_csv(DATA_DIR / 'links.csv')

    # Get top movies
    movie_counts = ratings['movieId'].value_counts()
    top_movies = movie_counts.head(args.n_movies).index.tolist()

    print(f"Fetching credits for top {len(top_movies)} movies...")

    movie_actors, movie_directors, all_actors, all_directors = fetch_credits_for_movies(
        top_movies, links, api_key, n_actors=args.n_actors
    )

    # Save cache
    save_credits_cache(movie_actors, movie_directors, all_actors, all_directors, cache_file)

    # Print summary
    print(f"\nSummary:")
    print(f"  Movies with credits: {len(movie_actors)}")
    print(f"  Unique actors: {len(all_actors)}")
    print(f"  Unique directors: {len(all_directors)}")

    print(f"\nTop 10 most common actors:")
    actor_counts = {}
    for actors in movie_actors.values():
        for actor in actors:
            actor_counts[actor] = actor_counts.get(actor, 0) + 1

    for actor, count in sorted(actor_counts.items(), key=lambda x: -x[1])[:10]:
        print(f"  {actor}: {count} movies")

    return movie_actors, movie_directors, all_actors, all_directors


if __name__ == "__main__":
    main()
