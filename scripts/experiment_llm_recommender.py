"""
EXPERIMENT 2: LLM as Recommender

Use LLM's implicit knowledge to generate recommendations based on user preferences.
Measure NDCG at different timesteps to verify RL signal exists.

The idea: LLM has learned movie correlations from training data.
Given likes/dislikes, it should produce better recommendations as more info is revealed.
Delta in NDCG = reward signal for RL.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import ndcg_score
import json
import re
import os
import hashlib
from typing import List, Dict, Tuple, Optional

# Cache directory for LLM responses
CACHE_DIR = Path('C:/dev/phd/casper/data/movielens/.cache/llm_responses')
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Load .env file from project root
from pathlib import Path
try:
    from dotenv import load_dotenv
    # Try multiple locations for .env
    env_paths = [
        Path(__file__).parent.parent / '.env',  # casper/.env
        Path(__file__).parent / '.env',          # scripts/.env
        Path.cwd() / '.env'                       # current directory
    ]
    for env_path in env_paths:
        if env_path.exists():
            load_dotenv(env_path)
            print(f"Loaded .env from {env_path}")
            break
except ImportError:
    print("python-dotenv not installed. Install with: pip install python-dotenv")

# =============================================================================
# LLM Interface (abstracted for different providers)
# =============================================================================

def _get_cache_key(prompt: str, model: str) -> str:
    """Generate cache key from prompt and model."""
    content = f"{model}:{prompt}"
    return hashlib.sha256(content.encode()).hexdigest()[:16]


def _get_cached_response(cache_key: str) -> Optional[str]:
    """Try to load cached LLM response."""
    cache_file = CACHE_DIR / f"{cache_key}.json"
    if cache_file.exists():
        try:
            with open(cache_file, 'r') as f:
                data = json.load(f)
                return data.get('response')
        except:
            pass
    return None


def _save_cached_response(cache_key: str, prompt: str, response: str, model: str):
    """Save LLM response to cache."""
    cache_file = CACHE_DIR / f"{cache_key}.json"
    with open(cache_file, 'w') as f:
        json.dump({
            'model': model,
            'prompt': prompt[:500],  # Truncate for readability
            'response': response
        }, f, indent=2)


def call_llm(prompt: str, model: str = "gpt-4o-mini", use_cache: bool = True) -> str:
    """
    Call LLM API with caching. Supports OpenAI and can be extended.
    Set OPENAI_API_KEY environment variable.

    Args:
        prompt: The prompt to send
        model: Model name
        use_cache: Whether to use cached responses (default: True)
    """
    # Check cache first
    cache_key = _get_cache_key(prompt, model)
    if use_cache:
        cached = _get_cached_response(cache_key)
        if cached:
            return cached

    try:
        import openai
        client = openai.OpenAI()
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,  # Lower for more consistent rankings
            max_tokens=500
        )
        result = response.choices[0].message.content

        # Cache the response
        if use_cache and result:
            _save_cached_response(cache_key, prompt, result, model)

        return result
    except ImportError:
        print("OpenAI not installed. Install with: pip install openai")
        return None
    except Exception as e:
        print(f"LLM API error: {e}")
        return None


def call_llm_anthropic(prompt: str, model: str = "claude-3-haiku-20240307", use_cache: bool = True) -> str:
    """Alternative: Anthropic Claude API with caching."""
    # Check cache first
    cache_key = _get_cache_key(prompt, model)
    if use_cache:
        cached = _get_cached_response(cache_key)
        if cached:
            return cached

    try:
        import anthropic
        client = anthropic.Anthropic()
        response = client.messages.create(
            model=model,
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}]
        )
        result = response.content[0].text

        # Cache the response
        if use_cache and result:
            _save_cached_response(cache_key, prompt, result, model)

        return result
    except ImportError:
        print("Anthropic not installed. Install with: pip install anthropic")
        return None
    except Exception as e:
        print(f"Anthropic API error: {e}")
        return None


# =============================================================================
# Recommendation Prompt & Parsing
# =============================================================================

def create_recommendation_prompt(
    liked_movies: List[str],
    disliked_movies: List[str],
    candidate_movies: List[str],
    n_recommendations: int = 10
) -> str:
    """Create prompt for LLM to rank movies."""

    liked_str = "\n".join(f"  - {m}" for m in liked_movies) if liked_movies else "  (none yet)"
    disliked_str = "\n".join(f"  - {m}" for m in disliked_movies) if disliked_movies else "  (none yet)"
    candidates_str = "\n".join(f"  {i+1}. {m}" for i, m in enumerate(candidate_movies))

    prompt = f"""Based on a user's movie preferences, rank which movies from the candidate list they would most likely enjoy.

USER'S PREFERENCES:
Movies they LIKED:
{liked_str}

Movies they DISLIKED:
{disliked_str}

CANDIDATE MOVIES TO RANK:
{candidates_str}

Please rank the top {n_recommendations} movies this user would most likely enjoy.
Return ONLY a JSON array of movie titles in order from most to least recommended.
Example format: ["Movie A", "Movie B", "Movie C"]

Your ranking:"""

    return prompt


def parse_llm_recommendations(response: str, candidate_movies: List[str], verbose: bool = False) -> List[str]:
    """
    Parse LLM response to extract movie rankings.

    Handles multiple formats:
    - JSON arrays: ["Movie A", "Movie B"]
    - Numbered lists: 1. Movie A\n2. Movie B
    - Bulleted lists: - Movie A\n- Movie B
    - Plain movie names separated by newlines
    """
    if response is None:
        return []

    if verbose:
        print(f"  [DEBUG] LLM response: {response[:300]}...")

    # Build lookup structures for matching
    candidate_lower = {m.lower(): m for m in candidate_movies}
    # Also build lookup without year: "The Matrix (1999)" -> "the matrix"
    candidate_base = {}
    for m in candidate_movies:
        base = re.sub(r'\s*\(\d{4}\)\s*$', '', m).lower().strip()
        candidate_base[base] = m

    def find_match(text: str) -> Optional[str]:
        """Find best matching candidate for a text string."""
        text = text.strip()
        text_lower = text.lower()

        # Exact match
        if text in candidate_movies:
            return text
        # Case-insensitive match
        if text_lower in candidate_lower:
            return candidate_lower[text_lower]
        # Match without year
        text_base = re.sub(r'\s*\(\d{4}\)\s*$', '', text).lower().strip()
        if text_base in candidate_base:
            return candidate_base[text_base]
        # Partial match (text contains candidate or vice versa)
        for cand in candidate_movies:
            cand_lower = cand.lower()
            cand_base = re.sub(r'\s*\(\d{4}\)\s*$', '', cand).lower().strip()
            if text_lower in cand_lower or cand_lower in text_lower:
                return cand
            if text_base and (text_base in cand_base or cand_base in text_base):
                return cand
        return None

    valid_rankings = []

    # Strategy 1: Try JSON array parsing
    try:
        # Look for JSON array pattern (greedy to capture full array)
        match = re.search(r'\[[\s\S]*?\]', response)
        if match:
            json_str = match.group()
            # Fix common LLM JSON issues
            json_str = re.sub(r',\s*]', ']', json_str)  # Remove trailing comma
            json_str = json_str.replace("'", '"')  # Single to double quotes
            rankings = json.loads(json_str)

            for r in rankings:
                if isinstance(r, str):
                    matched = find_match(r)
                    if matched and matched not in valid_rankings:
                        valid_rankings.append(matched)

            if verbose:
                print(f"  [DEBUG] JSON parsed {len(rankings)} items, {len(valid_rankings)} valid matches")

            if len(valid_rankings) >= 5:  # Good enough
                return valid_rankings
    except (json.JSONDecodeError, Exception) as e:
        if verbose:
            print(f"  [DEBUG] JSON decode error: {e}")

    # Strategy 2: Parse numbered/bulleted lists
    # Patterns: "1. Movie", "1) Movie", "- Movie", "* Movie"
    list_pattern = re.compile(r'^\s*(?:\d+[\.\)]\s*|\-\s*|\*\s*|\•\s*)(.+?)(?:\s*[-–—]\s*.+)?$', re.MULTILINE)
    for match in list_pattern.finditer(response):
        movie_text = match.group(1).strip()
        # Remove trailing punctuation or explanation
        movie_text = re.sub(r'\s*[-–—:].+$', '', movie_text)
        movie_text = movie_text.strip('"\'')

        matched = find_match(movie_text)
        if matched and matched not in valid_rankings:
            valid_rankings.append(matched)

    if verbose:
        print(f"  [DEBUG] List pattern parsed {len(valid_rankings)} movies")

    if len(valid_rankings) >= 5:
        return valid_rankings

    # Strategy 3: Scan each line for movie mentions
    for line in response.split('\n'):
        line = line.strip()
        if not line or len(line) < 3:
            continue

        # Skip lines that look like explanations
        if any(skip in line.lower() for skip in ['based on', 'because', 'since', 'would enjoy', 'recommend']):
            continue

        for movie in candidate_movies:
            movie_lower = movie.lower()
            movie_base = re.sub(r'\s*\(\d{4}\)\s*$', '', movie).lower().strip()
            line_lower = line.lower()

            if movie_lower in line_lower or (movie_base and movie_base in line_lower):
                if movie not in valid_rankings:
                    valid_rankings.append(movie)
                    break

    if verbose:
        print(f"  [DEBUG] Line scan found {len(valid_rankings)} movies")

    return valid_rankings


# =============================================================================
# Evaluation
# =============================================================================

def calculate_ndcg_from_ranking(
    predicted_ranking: List[str],
    ground_truth: Dict[str, float],  # movie -> rating (1=liked, 0=disliked)
    k: int = 10
) -> float:
    """Calculate NDCG given predicted ranking and ground truth preferences."""

    if len(predicted_ranking) < k:
        return np.nan

    # Build relevance scores based on ground truth
    relevance = []
    for movie in predicted_ranking[:k]:
        if movie in ground_truth:
            relevance.append(ground_truth[movie])
        else:
            relevance.append(0.5)  # Unknown = neutral

    # Ideal ranking: all liked movies first
    ideal = sorted(relevance, reverse=True)

    if sum(ideal) == 0:
        return 1.0  # No relevant items

    # Calculate DCG
    def dcg(scores):
        return sum(s / np.log2(i + 2) for i, s in enumerate(scores))

    return dcg(relevance) / dcg(ideal) if dcg(ideal) > 0 else 0


# =============================================================================
# Main Experiment
# =============================================================================

def run_llm_recommender_experiment(
    llm_func=call_llm,
    n_users: int = 10,
    timesteps: List[int] = [1, 3, 5, 10],
    n_recommendations: int = 10,
    verbose: bool = True
):
    """
    Run LLM recommender experiment.

    Returns dict with NDCG at each timestep.
    """

    print("=" * 80)
    print("EXPERIMENT 2: LLM AS RECOMMENDER")
    print("=" * 80)

    # Load data
    DATA_DIR = Path('C:/dev/phd/casper/data/movielens')
    ratings = pd.read_csv(DATA_DIR / 'ratings.csv')
    movies = pd.read_csv(DATA_DIR / 'movies.csv')

    # Get top 50 most popular movies (LLM will know these)
    N_MOVIES = 50
    movie_counts = ratings['movieId'].value_counts()
    top_movies = movie_counts.head(N_MOVIES).index.tolist()

    # Create movie_id -> title mapping
    movie_titles = {}
    for _, row in movies[movies['movieId'].isin(top_movies)].iterrows():
        movie_titles[row['movieId']] = row['title']

    # Get users with enough ratings in top movies
    ratings_filtered = ratings[ratings['movieId'].isin(top_movies)]
    user_counts = ratings_filtered.groupby('userId').size()
    active_users = user_counts[user_counts >= 15].index.tolist()

    if len(active_users) > n_users * 2:
        np.random.seed(42)
        active_users = np.random.choice(active_users, n_users * 2, replace=False).tolist()

    # Split users for experiment
    test_users = active_users[:n_users]

    print(f"\nDataset: {N_MOVIES} popular movies")
    print(f"Test users: {len(test_users)}")
    print(f"Timesteps to test: {timesteps}")

    # Results storage
    ndcg_by_timestep = {t: [] for t in timesteps}

    for user_idx, user_id in enumerate(test_users):
        user_ratings = ratings_filtered[ratings_filtered['userId'] == user_id]

        # Build ground truth for this user
        user_liked = []
        user_disliked = []
        ground_truth = {}

        for _, row in user_ratings.iterrows():
            movie_title = movie_titles.get(row['movieId'])
            if movie_title:
                if row['rating'] >= 4:
                    user_liked.append(movie_title)
                    ground_truth[movie_title] = 1.0
                else:
                    user_disliked.append(movie_title)
                    ground_truth[movie_title] = 0.0

        if len(user_liked) < 5:
            continue

        # Shuffle for random reveal order
        np.random.shuffle(user_liked)
        np.random.shuffle(user_disliked)

        all_user_movies = user_liked + user_disliked

        if verbose:
            print(f"\n--- User {user_idx + 1} ---")
            print(f"Liked: {len(user_liked)}, Disliked: {len(user_disliked)}")

        for n_revealed in timesteps:
            if n_revealed > len(all_user_movies):
                continue

            # Reveal first n preferences
            revealed_liked = [m for m in all_user_movies[:n_revealed] if m in user_liked]
            revealed_disliked = [m for m in all_user_movies[:n_revealed] if m in user_disliked]

            # Candidates = movies not yet revealed
            candidates = [movie_titles[mid] for mid in top_movies
                         if movie_titles.get(mid) and
                         movie_titles[mid] not in all_user_movies[:n_revealed]]

            if len(candidates) < n_recommendations:
                continue

            # Get LLM recommendations
            prompt = create_recommendation_prompt(
                revealed_liked, revealed_disliked, candidates, n_recommendations
            )

            response = llm_func(prompt)
            rankings = parse_llm_recommendations(response, candidates, verbose=False)

            if len(rankings) >= n_recommendations:
                # Calculate NDCG against user's actual preferences
                ndcg = calculate_ndcg_from_ranking(rankings, ground_truth, k=n_recommendations)
                ndcg_by_timestep[n_revealed].append(ndcg)

                if verbose:
                    print(f"  {n_revealed} prefs -> NDCG@{n_recommendations}: {ndcg:.4f}")
            else:
                if verbose:
                    print(f"  {n_revealed} prefs -> Failed to parse LLM response")

    # Summary
    print("\n" + "=" * 80)
    print("RESULTS: NDCG@10 BY TIMESTEP")
    print("=" * 80)

    results = {}
    prev_ndcg = None
    for t in sorted(ndcg_by_timestep.keys()):
        if ndcg_by_timestep[t]:
            avg_ndcg = np.mean(ndcg_by_timestep[t])
            std_ndcg = np.std(ndcg_by_timestep[t])
            delta = f"({avg_ndcg - prev_ndcg:+.4f})" if prev_ndcg else ""
            print(f"  {t:2d} preferences | NDCG: {avg_ndcg:.4f} +/- {std_ndcg:.4f} {delta}")
            results[t] = {'mean': avg_ndcg, 'std': std_ndcg}
            prev_ndcg = avg_ndcg

    # RL Signal check
    if results:
        first_t = min(results.keys())
        last_t = max(results.keys())
        improvement = results[last_t]['mean'] - results[first_t]['mean']
        print(f"\n  Improvement ({first_t} -> {last_t} prefs): {improvement:+.4f}")
        if improvement > 0.02:
            print("  >>> RL SIGNAL PRESENT: More preferences = better recommendations!")
        elif improvement > 0:
            print("  >>> Weak RL signal detected")
        else:
            print("  >>> No RL signal - LLM doesn't improve with more info")

    # Save results to cache
    results_file = CACHE_DIR.parent / 'llm_experiment_results.json'
    with open(results_file, 'w') as f:
        json.dump({
            'n_users': n_users,
            'timesteps': timesteps,
            'results': {str(k): v for k, v in results.items()}
        }, f, indent=2)
    print(f"\nResults saved to: {results_file}")

    return results


# =============================================================================
# Mock LLM for testing without API
# =============================================================================

def mock_llm(prompt: str) -> str:
    """Mock LLM for testing - returns random rankings."""
    # Extract candidate movies from prompt
    candidates = []
    in_candidates = False
    for line in prompt.split('\n'):
        if 'CANDIDATE MOVIES' in line:
            in_candidates = True
            continue
        if in_candidates:
            # Match any numbered line
            match = re.match(r'\s*(\d+)\.\s*(.+)', line.strip())
            if match:
                candidates.append(match.group(2).strip())
        if 'Your ranking' in line or 'Please rank' in line:
            break

    if not candidates:
        # Fallback: look for movie-like patterns
        for line in prompt.split('\n'):
            if '(' in line and ')' in line:  # Movie titles often have (year)
                match = re.search(r'([^(]+\(\d{4}\))', line)
                if match:
                    candidates.append(match.group(1).strip())

    # Return random subset as mock recommendation
    candidates = list(set(candidates))  # Deduplicate
    np.random.shuffle(candidates)
    return json.dumps(candidates[:10])


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--mock', action='store_true', help='Use mock LLM (no API)')
    parser.add_argument('--provider', default='openai', choices=['openai', 'anthropic'])
    parser.add_argument('--users', type=int, default=5)
    args = parser.parse_args()

    if args.mock:
        print("Using MOCK LLM (random baseline)")
        llm_func = mock_llm
    elif args.provider == 'anthropic':
        print("Using Anthropic Claude")
        llm_func = call_llm_anthropic
    else:
        print("Using OpenAI GPT-4o-mini")
        llm_func = call_llm

    results = run_llm_recommender_experiment(
        llm_func=llm_func,
        n_users=args.users,
        timesteps=[1, 3, 5, 10],
        verbose=True
    )
