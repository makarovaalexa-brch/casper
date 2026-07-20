"""
Diagnostic: NDCG Baseline & Reward Signal Analysis

Tests whether the LLM reward signal is usable for RL training by:
1. Measuring baseline NDCG (0 preferences) across different user filters
2. Simulating attribute-based dialogue (20 steps) to check NDCG growth
3. Varying movie pool size (100, 200, 500)

Run: python scripts/diagnose_reward_signal.py
"""

import sys
import os
import json
import hashlib
import re
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from tqdm import tqdm

# Setup paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))
DATA_DIR = PROJECT_ROOT / 'data' / 'movielens'

# Ensure API key
from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / '.env')
if not os.getenv('OPENAI_API_KEY'):
    raise RuntimeError("OPENAI_API_KEY not found in .env file")

RESULTS_DIR = PROJECT_ROOT / 'experiments' / 'diagnostics'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ─── LLM Helper (self-contained, no dependency on llm_reward.py) ────────────

CACHE_DIR = DATA_DIR / '.cache' / 'diagnostic_llm'
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def call_llm_cached(prompt: str, model: str = "gpt-4o-mini") -> Optional[str]:
    """Call LLM with caching."""
    key = hashlib.sha256(f"{model}:{prompt}".encode()).hexdigest()[:16]
    cache_file = CACHE_DIR / f"{key}.json"

    if cache_file.exists():
        with open(cache_file, 'r') as f:
            return json.load(f).get('response')

    try:
        import openai
        client = openai.OpenAI()
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=500
        )
        result = resp.choices[0].message.content
        with open(cache_file, 'w') as f:
            json.dump({'model': model, 'prompt': prompt[:200], 'response': result}, f)
        return result
    except Exception as e:
        print(f"LLM error: {e}")
        return None


def parse_recommendations(response: str, movie_list: List[str]) -> List[str]:
    """Parse LLM response into ordered movie list."""
    if not response:
        return []
    rankings = []
    for line in response.split('\n'):
        for movie in movie_list:
            if movie[:20].lower() in line.lower():
                if movie not in rankings:
                    rankings.append(movie)
                break
    return rankings


def normalize_title(title: str) -> str:
    return re.sub(r'\s*\(\d{4}\)\s*$', '', title).strip()


def calculate_ndcg(recommended_ids: List[int], relevant_ids: List[int], k: int = 10) -> float:
    """NDCG@k - same formula as llm_reward.py."""
    if not relevant_ids:
        return 0.0
    relevant_set = set(relevant_ids)
    dcg = 0.0
    for i, rec_id in enumerate(recommended_ids[:k]):
        if rec_id in relevant_set:
            dcg += 1.0 / np.log2(i + 2)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(k, len(relevant_ids))))
    return dcg / idcg if idcg > 0 else 0.0


# ─── Data Loading ────────────────────────────────────────────────────────────

def load_movie_pool(top_n: int = 100) -> Tuple[pd.DataFrame, List[str], Dict[str, int]]:
    """Load top N movies by rating count."""
    movies_df = pd.read_csv(DATA_DIR / 'movies.csv')
    ratings_df = pd.read_csv(DATA_DIR / 'ratings.csv')

    movie_counts = ratings_df.groupby('movieId').size().reset_index(name='count')
    movies_merged = movies_df.merge(movie_counts, on='movieId', how='left').fillna(0)
    top_movies_df = movies_merged.nlargest(top_n, 'count')

    movie_list = [row['title'] for _, row in top_movies_df.iterrows()]
    movie_id_map = {row['title']: row['movieId'] for _, row in top_movies_df.iterrows()}
    # Also map normalized titles
    for title, mid in list(movie_id_map.items()):
        movie_id_map[normalize_title(title)] = mid

    return top_movies_df, movie_list, movie_id_map


def load_user_profiles(min_ratings: int = 20, max_profiles: int = 500) -> List[Dict]:
    """Load user profiles (same as UserSimulator)."""
    movies_df = pd.read_csv(DATA_DIR / 'movies.csv')
    ratings_df = pd.read_csv(DATA_DIR / 'ratings.csv')

    user_counts = ratings_df['userId'].value_counts()
    eligible = user_counts[user_counts >= min_ratings].index.tolist()

    np.random.seed(42)
    sampled = np.random.choice(eligible, size=min(max_profiles, len(eligible)), replace=False)

    profiles = []
    for uid in tqdm(sampled, desc="Loading profiles", unit="user"):
        user_df = ratings_df[ratings_df['userId'] == uid]
        user_movies = user_df.merge(movies_df[['movieId', 'title']], on='movieId', how='left')
        ratings = {}
        for _, row in user_movies.iterrows():
            if pd.notna(row['title']):
                ratings[row['title']] = float(row['rating'])
        profiles.append({'user_id': int(uid), 'ratings': ratings})

    return profiles


def get_eval_targets(profile: Dict, movie_id_map: Dict[str, int], min_rating: float = 4.0) -> List[int]:
    """Get user's high-rated movies that are in the movie pool."""
    targets = []
    for movie, rating in profile.get('ratings', {}).items():
        if rating >= min_rating:
            mid = movie_id_map.get(movie) or movie_id_map.get(normalize_title(movie))
            if mid is not None:
                targets.append(mid)
    return targets


def get_user_genres(profile: Dict, movies_df: pd.DataFrame, movie_id_map: Dict[str, int],
                    min_rating: float = 4.0) -> Tuple[List[str], List[str]]:
    """Extract liked and disliked genres from user profile."""
    genre_scores = {}  # genre -> list of ratings

    for movie, rating in profile.get('ratings', {}).items():
        # Find this movie in movies_df
        mid = movie_id_map.get(movie) or movie_id_map.get(normalize_title(movie))
        if mid is None:
            continue
        movie_row = movies_df[movies_df['movieId'] == mid]
        if movie_row.empty:
            continue
        genres = movie_row.iloc[0]['genres'].split('|')
        for g in genres:
            if g == '(no genres listed)':
                continue
            if g not in genre_scores:
                genre_scores[g] = []
            genre_scores[g].append(rating)

    liked_genres = []
    disliked_genres = []
    for genre, scores in genre_scores.items():
        avg = np.mean(scores)
        if avg >= 4.0:
            liked_genres.append(genre)
        elif avg < 2.5:
            disliked_genres.append(genre)

    return liked_genres, disliked_genres


def get_user_attribute_sequence(profile: Dict, movies_df: pd.DataFrame,
                                movie_id_map: Dict[str, int]) -> List[Dict]:
    """
    Build a sequence of attributes to reveal, simulating a dialogue.

    Returns list of preference snapshots at each step.
    Each step reveals one more piece of info (genre like/dislike or specific movie).
    """
    # Collect all user's movies in the pool
    pool_movies_liked = []
    pool_movies_disliked = []

    for movie, rating in profile.get('ratings', {}).items():
        mid = movie_id_map.get(movie) or movie_id_map.get(normalize_title(movie))
        if mid is None:
            continue
        if rating >= 4.0:
            pool_movies_liked.append(movie)
        elif rating < 2.5:
            pool_movies_disliked.append(movie)

    # Get genre preferences
    liked_genres, disliked_genres = get_user_genres(profile, movies_df, movie_id_map)

    # Build reveal sequence: interleave genres and specific movies
    # Start with genres (broad attributes), then specific movies
    sequence = []

    # Phase 1: genres (the "attribute" part - this is what CASPER actually asks about)
    for g in liked_genres:
        sequence.append(('genre_like', g))
    for g in disliked_genres:
        sequence.append(('genre_dislike', g))

    # Phase 2: specific movies
    np.random.seed(profile['user_id'])
    np.random.shuffle(pool_movies_liked)
    np.random.shuffle(pool_movies_disliked)

    for m in pool_movies_liked:
        sequence.append(('movie_like', m))
    for m in pool_movies_disliked:
        sequence.append(('movie_dislike', m))

    return sequence


def build_preferences_text(revealed: List[Tuple[str, str]]) -> str:
    """Build preferences text from revealed items (no not_seen)."""
    liked = []
    disliked = []

    for kind, item in revealed:
        if kind in ('genre_like', 'movie_like'):
            liked.append(item)
        elif kind in ('genre_dislike', 'movie_dislike'):
            disliked.append(item)

    parts = []
    if liked:
        parts.append(f"Liked: {', '.join(liked[:15])}")
    if disliked:
        parts.append(f"Disliked: {', '.join(disliked[:10])}")

    return '\n'.join(parts) if parts else "No preferences yet"


def get_llm_recommendations(preferences_text: str, movie_list: List[str],
                            movie_id_map: Dict[str, int], top_k: int = 10) -> List[int]:
    """Get LLM recommendations (same prompt structure as llm_reward.py)."""
    prompt = f"""Based on these movie preferences, recommend the top {top_k} movies this user would enjoy.

User preferences:
{preferences_text}

Choose from these movies ONLY:
{', '.join(movie_list[:50])}
{', '.join(movie_list[50:])}

Return ONLY a numbered list of {top_k} movie titles, most recommended first."""

    response = call_llm_cached(prompt)
    recommendations = parse_recommendations(response, movie_list)
    rec_ids = [movie_id_map.get(title) for title in recommendations[:top_k]
               if title in movie_id_map]
    return [r for r in rec_ids if r is not None]


# ─── Test 1: Baseline NDCG Distribution ──────────────────────────────────────

def test_baseline_ndcg(profiles: List[Dict], movie_list: List[str],
                       movie_id_map: Dict[str, int],
                       min_eval_targets_list: List[int],
                       pool_size: int) -> Dict:
    """Test baseline NDCG for different eval target filters."""
    print(f"\n{'='*60}")
    print(f"TEST 1: Baseline NDCG Distribution (pool={pool_size})")
    print(f"{'='*60}")

    results = {}

    for min_targets in min_eval_targets_list:
        # Filter users
        filtered = []
        for p in profiles:
            targets = get_eval_targets(p, movie_id_map)
            if len(targets) >= min_targets:
                filtered.append((p, targets))

        if not filtered:
            print(f"\n  min_targets={min_targets}: 0 users qualify, skipping")
            results[min_targets] = {'n_users': 0}
            continue

        # Sample up to 50 users for LLM calls
        sample = filtered[:50]

        baseline_ndcgs = []
        target_counts = []

        for profile, targets in tqdm(sample, desc=f"Baseline (min={min_targets})", unit="user"):
            rec_ids = get_llm_recommendations("No preferences yet", movie_list, movie_id_map)
            ndcg = calculate_ndcg(rec_ids, targets)
            baseline_ndcgs.append(ndcg)
            target_counts.append(len(targets))

        results[min_targets] = {
            'n_users': len(filtered),
            'n_sampled': len(sample),
            'ndcg_mean': float(np.mean(baseline_ndcgs)),
            'ndcg_std': float(np.std(baseline_ndcgs)),
            'ndcg_median': float(np.median(baseline_ndcgs)),
            'ndcg_min': float(np.min(baseline_ndcgs)),
            'ndcg_max': float(np.max(baseline_ndcgs)),
            'avg_targets': float(np.mean(target_counts)),
            'values': baseline_ndcgs
        }

        print(f"\n  min_targets={min_targets}: {len(filtered)} users qualify ({len(sample)} sampled)")
        print(f"    Eval targets: mean={np.mean(target_counts):.1f}")
        print(f"    Baseline NDCG: {np.mean(baseline_ndcgs):.4f} +/- {np.std(baseline_ndcgs):.4f}")
        print(f"    Range: [{np.min(baseline_ndcgs):.4f}, {np.max(baseline_ndcgs):.4f}]")

    return results


# ─── Test 2: Simulated Attribute Dialogue ─────────────────────────────────────

def test_attribute_dialogue(profiles: List[Dict], movies_df: pd.DataFrame,
                            movie_list: List[str], movie_id_map: Dict[str, int],
                            min_eval_targets: int, pool_size: int,
                            max_steps: int = 20, n_users: int = 30) -> Dict:
    """Simulate revealing attributes step by step, measure NDCG at each step."""
    print(f"\n{'='*60}")
    print(f"TEST 2: Simulated Attribute Dialogue (pool={pool_size}, steps={max_steps})")
    print(f"{'='*60}")

    # Filter users
    filtered = []
    for p in profiles:
        targets = get_eval_targets(p, movie_id_map)
        if len(targets) >= min_eval_targets:
            filtered.append((p, targets))

    sample = filtered[:n_users]
    print(f"  Users: {len(filtered)} qualify, sampling {len(sample)}")

    # Track NDCG at each step for each user
    all_curves = []  # list of lists, each inner list = NDCG at steps 0..max_steps

    for profile, targets in tqdm(sample, desc="Dialogue sim", unit="user"):
        sequence = get_user_attribute_sequence(profile, movies_df, movie_id_map)
        if not sequence:
            continue

        curve = []

        # Step 0: no preferences
        rec_ids = get_llm_recommendations("No preferences yet", movie_list, movie_id_map)
        ndcg = calculate_ndcg(rec_ids, targets)
        curve.append(ndcg)

        # Steps 1..max_steps: reveal one more attribute each step
        for step in range(1, max_steps + 1):
            if step <= len(sequence):
                revealed = sequence[:step]
            else:
                revealed = sequence  # All revealed

            pref_text = build_preferences_text(revealed)
            rec_ids = get_llm_recommendations(pref_text, movie_list, movie_id_map)
            ndcg = calculate_ndcg(rec_ids, targets)
            curve.append(ndcg)

        all_curves.append(curve)

    # Aggregate
    all_curves = np.array(all_curves)  # shape: (n_users, max_steps+1)
    mean_curve = np.mean(all_curves, axis=0)
    std_curve = np.std(all_curves, axis=0)

    print(f"\n  NDCG progression (mean over {len(all_curves)} users):")
    print(f"  {'Step':<6} {'NDCG':>8} {'Std':>8} {'Delta':>8}")
    print(f"  {'-'*32}")
    for step in range(max_steps + 1):
        delta = mean_curve[step] - mean_curve[step - 1] if step > 0 else 0.0
        print(f"  {step:<6} {mean_curve[step]:>8.4f} {std_curve[step]:>8.4f} {delta:>+8.4f}")

    result = {
        'n_users': len(all_curves),
        'min_eval_targets': min_eval_targets,
        'mean_curve': mean_curve.tolist(),
        'std_curve': std_curve.tolist(),
        'step_0_ndcg': float(mean_curve[0]),
        'step_20_ndcg': float(mean_curve[-1]),
        'total_improvement': float(mean_curve[-1] - mean_curve[0]),
        'monotonic_steps': int(sum(1 for i in range(1, len(mean_curve))
                                   if mean_curve[i] >= mean_curve[i-1])),
    }

    print(f"\n  Baseline (step 0): {mean_curve[0]:.4f}")
    print(f"  Final (step {max_steps}): {mean_curve[-1]:.4f}")
    print(f"  Total improvement: {mean_curve[-1] - mean_curve[0]:+.4f}")
    print(f"  Monotonic steps: {result['monotonic_steps']}/{max_steps}")

    return result


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("CASPER Reward Signal Diagnostic")
    print("=" * 60)

    # Load user profiles once
    print("\nLoading user profiles...")
    profiles = load_user_profiles(min_ratings=20, max_profiles=500)
    print(f"  Loaded {len(profiles)} profiles")

    movies_df = pd.read_csv(DATA_DIR / 'movies.csv')

    all_results = {}

    # Test across different pool sizes
    for pool_size in [100, 200, 500]:
        print(f"\n{'#'*60}")
        print(f"# POOL SIZE: {pool_size} movies")
        print(f"{'#'*60}")

        top_movies_df, movie_list, movie_id_map = load_movie_pool(pool_size)

        # Test 1: Baseline NDCG with different eval target filters
        # Scale min_eval_targets relative to pool size
        if pool_size == 100:
            filters = [5, 10, 15, 20, 25]
        elif pool_size == 200:
            filters = [5, 10, 15, 20, 25]
        else:
            filters = [5, 10, 15, 20, 25]

        baseline_results = test_baseline_ndcg(
            profiles, movie_list, movie_id_map, filters, pool_size
        )

        # Test 2: Attribute dialogue with a reasonable filter
        # Pick the filter that gives baseline ~0.2-0.3 (sweet spot)
        best_filter = 10  # default
        for f in filters:
            r = baseline_results.get(f, {})
            if r.get('ndcg_mean', 1.0) <= 0.35 and r.get('n_users', 0) >= 20:
                best_filter = f
                break

        dialogue_results = test_attribute_dialogue(
            profiles, movies_df, movie_list, movie_id_map,
            min_eval_targets=best_filter,
            pool_size=pool_size,
            max_steps=20,
            n_users=30
        )

        all_results[pool_size] = {
            'baseline': baseline_results,
            'dialogue': dialogue_results,
            'best_filter': best_filter
        }

    # ─── Summary ──────────────────────────────────────────────────────────────

    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")

    print(f"\n{'Pool':<6} {'Filter':<8} {'Users':<7} {'Base NDCG':<12} {'Final NDCG':<12} {'Improvement':<12} {'Monotonic'}")
    print(f"{'-'*75}")

    for pool_size, res in all_results.items():
        f = res['best_filter']
        d = res['dialogue']
        b = res['baseline'].get(f, {})
        print(f"{pool_size:<6} {f:<8} {d['n_users']:<7} "
              f"{d['step_0_ndcg']:<12.4f} {d['step_20_ndcg']:<12.4f} "
              f"{d['total_improvement']:<+12.4f} {d['monotonic_steps']}/20")

    # Save results
    output_file = RESULTS_DIR / 'reward_signal_diagnostic.json'
    # Convert numpy types for JSON serialization
    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=convert)
    print(f"\nResults saved to: {output_file}")

    # Print baseline table for all filters
    print(f"\n{'='*60}")
    print(f"BASELINE NDCG BY POOL SIZE AND EVAL TARGET FILTER")
    print(f"{'='*60}")
    print(f"{'Pool':<6} {'Filter':<8} {'Users':<7} {'NDCG Mean':<10} {'NDCG Std':<10} {'Avg Targets'}")
    print(f"{'-'*55}")
    for pool_size, res in all_results.items():
        for f, b in sorted(res['baseline'].items()):
            if b.get('n_users', 0) == 0:
                continue
            print(f"{pool_size:<6} {f:<8} {b['n_users']:<7} "
                  f"{b['ndcg_mean']:<10.4f} {b['ndcg_std']:<10.4f} {b.get('avg_targets', 0):<10.1f}")


if __name__ == '__main__':
    main()
