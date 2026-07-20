"""
Test: Attributes vs Movie Titles vs Mixed on pool=100

Isolates the effect of preference type on NDCG growth.
3 conditions, all pool=100, 20 steps:
  A) Movie titles only (like the notebook)
  B) Attributes/genres only (like our diagnostic)
  C) Mixed: interleave attributes and titles

Run: python scripts/test_preference_types.py
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

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))
DATA_DIR = PROJECT_ROOT / 'data' / 'movielens'

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / '.env')
if not os.getenv('OPENAI_API_KEY'):
    raise RuntimeError("OPENAI_API_KEY not found")

RESULTS_DIR = PROJECT_ROOT / 'experiments' / 'diagnostics'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CACHE_DIR = DATA_DIR / '.cache' / 'diagnostic_llm'
CACHE_DIR.mkdir(parents=True, exist_ok=True)


# ─── Shared helpers (from diagnose_reward_signal.py) ─────────────────────────

def call_llm_cached(prompt: str, model: str = "gpt-4o-mini") -> Optional[str]:
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
    if not relevant_ids:
        return 0.0
    relevant_set = set(relevant_ids)
    dcg = 0.0
    for i, rec_id in enumerate(recommended_ids[:k]):
        if rec_id in relevant_set:
            dcg += 1.0 / np.log2(i + 2)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(k, len(relevant_ids))))
    return dcg / idcg if idcg > 0 else 0.0


def load_movie_pool(top_n: int = 100):
    movies_df = pd.read_csv(DATA_DIR / 'movies.csv')
    ratings_df = pd.read_csv(DATA_DIR / 'ratings.csv')
    movie_counts = ratings_df.groupby('movieId').size().reset_index(name='count')
    movies_merged = movies_df.merge(movie_counts, on='movieId', how='left').fillna(0)
    top_movies_df = movies_merged.nlargest(top_n, 'count')
    movie_list = [row['title'] for _, row in top_movies_df.iterrows()]
    movie_id_map = {row['title']: row['movieId'] for _, row in top_movies_df.iterrows()}
    for title, mid in list(movie_id_map.items()):
        movie_id_map[normalize_title(title)] = mid
    return top_movies_df, movie_list, movie_id_map


def load_user_profiles(min_ratings: int = 20, max_profiles: int = 500):
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


def get_eval_targets(profile: Dict, movie_id_map: Dict[str, int], min_rating: float = 4.0):
    targets = []
    for movie, rating in profile.get('ratings', {}).items():
        if rating >= min_rating:
            mid = movie_id_map.get(movie) or movie_id_map.get(normalize_title(movie))
            if mid is not None:
                targets.append(mid)
    return targets


def get_llm_recommendations(preferences_text: str, movie_list: List[str],
                            movie_id_map: Dict[str, int], top_k: int = 10) -> List[int]:
    prompt = f"""Based on these movie preferences, recommend the top {top_k} movies this user would enjoy.

User preferences:
{preferences_text}

Choose from these movies ONLY:
{', '.join(movie_list)}

Return ONLY a numbered list of {top_k} movie titles, most recommended first."""

    response = call_llm_cached(prompt)
    recommendations = parse_recommendations(response, movie_list)
    rec_ids = [movie_id_map.get(title) for title in recommendations[:top_k]
               if title in movie_id_map]
    return [r for r in rec_ids if r is not None]


# ─── Sequence builders for each condition ─────────────────────────────────────

def get_user_genres(profile: Dict, movies_df: pd.DataFrame, movie_id_map: Dict[str, int]):
    genre_scores = {}
    for movie, rating in profile.get('ratings', {}).items():
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


def build_sequence_titles_only(profile: Dict, movie_id_map: Dict[str, int]):
    """Condition A: reveal movie titles only (like the notebook)."""
    liked = []
    disliked = []
    for movie, rating in profile.get('ratings', {}).items():
        mid = movie_id_map.get(movie) or movie_id_map.get(normalize_title(movie))
        if mid is None:
            continue
        if rating >= 4.0:
            liked.append(movie)
        elif rating < 2.5:
            disliked.append(movie)

    np.random.seed(profile['user_id'])
    np.random.shuffle(liked)
    np.random.shuffle(disliked)

    # Interleave: 2 liked, 1 disliked (similar to notebook which does liked[:10], disliked[:5])
    seq = []
    li, di = 0, 0
    while li < len(liked) or di < len(disliked):
        if li < len(liked):
            seq.append(('title_like', liked[li]))
            li += 1
        if li < len(liked):
            seq.append(('title_like', liked[li]))
            li += 1
        if di < len(disliked):
            seq.append(('title_dislike', disliked[di]))
            di += 1
    return seq


def build_sequence_attributes_only(profile: Dict, movies_df: pd.DataFrame,
                                    movie_id_map: Dict[str, int]):
    """Condition B: reveal genre attributes only (like our diagnostic)."""
    liked_genres, disliked_genres = get_user_genres(profile, movies_df, movie_id_map)

    seq = []
    for g in liked_genres:
        seq.append(('genre_like', g))
    for g in disliked_genres:
        seq.append(('genre_dislike', g))

    # If we run out of genres before 20 steps, pad with specific movies
    liked_movies = []
    disliked_movies = []
    for movie, rating in profile.get('ratings', {}).items():
        mid = movie_id_map.get(movie) or movie_id_map.get(normalize_title(movie))
        if mid is None:
            continue
        if rating >= 4.0:
            liked_movies.append(movie)
        elif rating < 2.5:
            disliked_movies.append(movie)

    np.random.seed(profile['user_id'])
    np.random.shuffle(liked_movies)
    np.random.shuffle(disliked_movies)

    for m in liked_movies:
        seq.append(('movie_like', m))
    for m in disliked_movies:
        seq.append(('movie_dislike', m))
    return seq


def build_sequence_mixed(profile: Dict, movies_df: pd.DataFrame,
                         movie_id_map: Dict[str, int]):
    """Condition C: interleave genres and titles."""
    liked_genres, disliked_genres = get_user_genres(profile, movies_df, movie_id_map)

    liked_movies = []
    disliked_movies = []
    for movie, rating in profile.get('ratings', {}).items():
        mid = movie_id_map.get(movie) or movie_id_map.get(normalize_title(movie))
        if mid is None:
            continue
        if rating >= 4.0:
            liked_movies.append(movie)
        elif rating < 2.5:
            disliked_movies.append(movie)

    np.random.seed(profile['user_id'])
    np.random.shuffle(liked_movies)
    np.random.shuffle(disliked_movies)

    # Interleave: genre, title, genre, title...
    seq = []
    gi_l, gi_d = 0, 0
    mi_l, mi_d = 0, 0

    while (gi_l < len(liked_genres) or gi_d < len(disliked_genres) or
           mi_l < len(liked_movies) or mi_d < len(disliked_movies)):
        # One liked genre
        if gi_l < len(liked_genres):
            seq.append(('genre_like', liked_genres[gi_l]))
            gi_l += 1
        # One liked movie title
        if mi_l < len(liked_movies):
            seq.append(('title_like', liked_movies[mi_l]))
            mi_l += 1
        # One disliked genre
        if gi_d < len(disliked_genres):
            seq.append(('genre_dislike', disliked_genres[gi_d]))
            gi_d += 1
        # One disliked movie title
        if mi_d < len(disliked_movies):
            seq.append(('title_dislike', disliked_movies[mi_d]))
            mi_d += 1
    return seq


# ─── Build preferences text ──────────────────────────────────────────────────

def build_pref_text(revealed: List[Tuple[str, str]]) -> str:
    """Build preference text, keeping titles and genres in the same liked/disliked buckets."""
    liked = []
    disliked = []
    for kind, item in revealed:
        if kind in ('title_like', 'genre_like', 'movie_like'):
            liked.append(item)
        elif kind in ('title_dislike', 'genre_dislike', 'movie_dislike'):
            disliked.append(item)

    parts = []
    if liked:
        parts.append(f"Liked: {', '.join(liked[:15])}")
    if disliked:
        parts.append(f"Disliked: {', '.join(disliked[:10])}")
    return '\n'.join(parts) if parts else "No preferences yet"


# ─── Run one condition ────────────────────────────────────────────────────────

def run_condition(name: str, profiles_with_targets, movies_df, movie_list,
                  movie_id_map, build_fn, max_steps=20, n_users=30):
    """Run a single condition and return NDCG curve."""
    sample = profiles_with_targets[:n_users]
    print(f"\n  Condition: {name} ({len(sample)} users, {max_steps} steps)")

    all_curves = []

    for profile, targets in tqdm(sample, desc=name, unit="user"):
        if build_fn.__code__.co_varnames[:3] == ('profile', 'movies_df', 'movie_id_map'):
            sequence = build_fn(profile, movies_df, movie_id_map)
        else:
            sequence = build_fn(profile, movie_id_map)

        if not sequence:
            continue

        curve = []

        # Step 0: baseline
        rec_ids = get_llm_recommendations("No preferences yet", movie_list, movie_id_map)
        ndcg = calculate_ndcg(rec_ids, targets)
        curve.append(ndcg)

        # Steps 1..max_steps
        for step in range(1, max_steps + 1):
            revealed = sequence[:min(step, len(sequence))]
            pref_text = build_pref_text(revealed)
            rec_ids = get_llm_recommendations(pref_text, movie_list, movie_id_map)
            ndcg = calculate_ndcg(rec_ids, targets)
            curve.append(ndcg)

        all_curves.append(curve)

    all_curves = np.array(all_curves)
    mean_curve = np.mean(all_curves, axis=0)
    std_curve = np.std(all_curves, axis=0)

    print(f"\n  {'Step':<6} {'NDCG':>8} {'Std':>8} {'Delta':>8}")
    print(f"  {'-'*32}")
    for step in range(max_steps + 1):
        delta = mean_curve[step] - mean_curve[step - 1] if step > 0 else 0.0
        print(f"  {step:<6} {mean_curve[step]:>8.4f} {std_curve[step]:>8.4f} {delta:>+8.4f}")

    total = mean_curve[-1] - mean_curve[0]
    mono = sum(1 for i in range(1, len(mean_curve)) if mean_curve[i] >= mean_curve[i-1])
    print(f"\n  Baseline: {mean_curve[0]:.4f}  Final: {mean_curve[-1]:.4f}")
    print(f"  Total: {total:+.4f}  Monotonic: {mono}/{max_steps}")

    return {
        'name': name,
        'n_users': len(all_curves),
        'mean_curve': mean_curve.tolist(),
        'std_curve': std_curve.tolist(),
        'total_improvement': float(total),
        'monotonic_steps': mono,
    }


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Preference Type Comparison (pool=100, 20 steps)")
    print("=" * 60)

    profiles = load_user_profiles(min_ratings=20, max_profiles=500)
    movies_df = pd.read_csv(DATA_DIR / 'movies.csv')
    top_movies_df, movie_list, movie_id_map = load_movie_pool(100)

    # Filter to users with >= 5 eval targets (same as previous diagnostic)
    filtered = []
    for p in profiles:
        targets = get_eval_targets(p, movie_id_map)
        if len(targets) >= 5:
            filtered.append((p, targets))

    print(f"\n  {len(filtered)} users with >= 5 eval targets in pool=100")
    np.random.seed(42)
    # Shuffle to get diverse sample, but use same users across conditions
    indices = np.random.permutation(len(filtered))
    filtered = [filtered[i] for i in indices]

    results = {}

    # A) Movie titles only
    results['titles'] = run_condition(
        "A: Movie Titles Only", filtered, movies_df, movie_list,
        movie_id_map, build_sequence_titles_only)

    # B) Attributes only
    results['attributes'] = run_condition(
        "B: Attributes Only", filtered, movies_df, movie_list,
        movie_id_map, build_sequence_attributes_only)

    # C) Mixed
    results['mixed'] = run_condition(
        "C: Mixed (Genres + Titles)", filtered, movies_df, movie_list,
        movie_id_map, build_sequence_mixed)

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY (pool=100)")
    print(f"{'='*60}")
    print(f"{'Condition':<30} {'Baseline':>8} {'Final':>8} {'Improve':>8} {'Mono':>6}")
    print(f"{'-'*62}")
    for key, r in results.items():
        print(f"{r['name']:<30} {r['mean_curve'][0]:>8.4f} {r['mean_curve'][-1]:>8.4f} "
              f"{r['total_improvement']:>+8.4f} {r['monotonic_steps']:>3}/20")

    # Save
    out_file = RESULTS_DIR / 'preference_type_comparison.json'
    with open(out_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {out_file}")


if __name__ == '__main__':
    main()
