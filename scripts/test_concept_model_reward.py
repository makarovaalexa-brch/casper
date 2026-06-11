"""
Test: Concept (SBERT+LSTM+Attention) model as reward signal.

Uses the SAVED checkpoint from RS_Model_Comparison notebook:
  - concept_paper_config.pt (187 items, ~100 epochs, BCE/CE loss)
  - concept_embeddings_paper_config.npy (187 x 384 SBERT embeddings)

Three conditions (analogous to LLM diagnostic):
  A) Movie titles only — reveal individual movie ratings
  B) Attributes/genres only — reveal all movies of a genre at once
  C) Mixed — interleave genre and movie revelations

Pool=100, 20 steps, 30 users.

Run: python scripts/test_concept_model_reward.py
"""

import sys
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))
DATA_DIR = PROJECT_ROOT / 'data' / 'movielens'
CACHE_DIR = DATA_DIR / '.cache'
CHECKPOINT_DIR = CACHE_DIR / 'checkpoints'
RESULTS_DIR = PROJECT_ROOT / 'experiments' / 'diagnostics'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

N_MOVIES = 100
SEED = 42


# ─── Model Definition (from RS_Model_Comparison notebook) ───────────────────

class ConceptEmbeddingModel(nn.Module):
    """SBERT-based concept embeddings model with LSTM+Attention."""
    def __init__(self, n_items, embedding_dim=384, hidden_dim=None):
        super().__init__()
        hidden_dim = hidden_dim or n_items // 2
        self.concept_proj = nn.Linear(embedding_dim, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim + 3, hidden_dim, 1, batch_first=True)
        self.dense1 = nn.Linear(hidden_dim, n_items)
        self.dense2 = nn.Linear(n_items, hidden_dim)
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=1, batch_first=True)
        self.output = nn.Linear(hidden_dim * 2, n_items)

    def forward(self, item_embeddings, rating_input):
        x = self.concept_proj(item_embeddings)
        x = torch.cat((x, rating_input), dim=-1)
        enc, _ = self.lstm(x)
        x = torch.relu(self.dense1(enc))
        x = torch.relu(self.dense2(x))
        att, _ = self.attention(enc, x, x)
        return self.output(torch.cat((x, att), dim=-1))


# ─── Data loading ───────────────────────────────────────────────────────────

def load_data():
    movies_df = pd.read_csv(DATA_DIR / 'movies.csv')
    ratings_df = pd.read_csv(
        DATA_DIR / 'ratings.csv',
        dtype={'userId': 'int32', 'movieId': 'int32', 'rating': 'float32'},
        usecols=['userId', 'movieId', 'rating'],
    )

    # Top 100 movies by rating count (same as notebook)
    movie_counts = ratings_df.groupby('movieId').size().reset_index(name='count')
    movies_merged = movies_df.merge(movie_counts, on='movieId', how='left').fillna(0)
    top_movies_df = movies_merged.nlargest(N_MOVIES, 'count')
    top_movies = top_movies_df['movieId'].tolist()

    movie_to_idx = {mid: i for i, mid in enumerate(top_movies)}
    movie_titles = {row['movieId']: row['title'] for _, row in top_movies_df.iterrows()}
    movie_genres = {}
    for _, row in top_movies_df.iterrows():
        genres = row['genres'].split('|') if pd.notna(row['genres']) else []
        movie_genres[row['movieId']] = [g for g in genres if g != '(no genres listed)']

    top_set = set(top_movies)
    ratings_filtered = ratings_df[ratings_df['movieId'].isin(top_set)].copy()

    return top_movies, movie_to_idx, movie_titles, movie_genres, ratings_filtered


def load_model():
    """Load the saved Concept model checkpoint."""
    ckpt_path = CHECKPOINT_DIR / 'concept_paper_config.pt'
    emb_path = CHECKPOINT_DIR / 'concept_embeddings_paper_config.npy'

    if not ckpt_path.exists() or not emb_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    hidden_dim = ckpt['model_state_dict']['concept_proj.bias'].shape[0]
    n_items = ckpt['n_items']
    model = ConceptEmbeddingModel(n_items, 384, hidden_dim)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    emb = np.load(emb_path)

    print(f"  Loaded Concept model: n_items={n_items}, hidden_dim={hidden_dim}, "
          f"epochs={ckpt.get('epochs', 0)}, val_loss={ckpt.get('val_loss', 0):.4f}")
    print(f"  Embeddings shape: {emb.shape}")

    return model, n_items, emb


def build_user_profiles(ratings_filtered, movie_to_idx, min_eval_targets=5,
                        max_profiles=1000):
    """Build user profiles."""
    df = ratings_filtered[ratings_filtered['movieId'].isin(movie_to_idx)].copy()
    df['liked'] = (df['rating'] >= 4.0).astype('int8')

    user_liked_counts = df.groupby('userId')['liked'].sum()
    eligible_users = user_liked_counts[user_liked_counts >= min_eval_targets].index.tolist()

    np.random.seed(SEED)
    sampled = np.random.choice(eligible_users,
                               size=min(max_profiles, len(eligible_users)),
                               replace=False)
    grouped = {uid: grp for uid, grp in df[df['userId'].isin(sampled)].groupby('userId')}

    profiles = []
    for uid in sampled:
        if uid not in grouped:
            continue
        user_df = grouped[uid]
        items = []
        liked_ids = []
        for _, row in user_df.iterrows():
            mid = int(row['movieId'])
            r = float(row['rating'])
            if mid in movie_to_idx:
                items.append((mid, r))
                if r >= 4.0:
                    liked_ids.append(mid)
        profiles.append({
            'user_id': int(uid),
            'items': items,
            'liked_ids': liked_ids,
        })
    return profiles


# ─── Prediction helper ──────────────────────────────────────────────────────

def predict_concept(model, emb, n_items, revealed_items):
    """
    Get predictions from concept model.
    revealed_items: list of (movie_idx_in_model, is_liked_bool)
    """
    item_emb = torch.FloatTensor(emb).unsqueeze(0)  # [1, n_items, 384]
    rating_input = torch.zeros(1, n_items, 3)
    rating_input[:, :, 2] = 1  # All unknown

    for idx, is_liked in revealed_items:
        if idx < n_items:
            rating_input[:, idx, 2] = 0
            rating_input[:, idx, 1 if is_liked else 0] = 1

    with torch.no_grad():
        preds = model(item_emb, rating_input)[:, -1, :N_MOVIES].sigmoid().numpy().flatten()
    return preds


# ─── NDCG calculation ──────────────────────────────────────────────────────

def calculate_ndcg(recommended_ids, relevant_ids, k=10):
    if not relevant_ids:
        return 0.0
    relevant_set = set(relevant_ids)
    dcg = sum(1.0 / np.log2(i + 2) for i, rid in enumerate(recommended_ids[:k]) if rid in relevant_set)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(k, len(relevant_ids))))
    return dcg / idcg if idcg > 0 else 0.0


# ─── Sequence builders ──────────────────────────────────────────────────────

def build_seq_titles(profile, movie_to_idx):
    """Condition A: reveal individual movie ratings one at a time."""
    liked = [(mid, True) for mid, r in profile['items'] if r >= 4.0 and mid in movie_to_idx]
    disliked = [(mid, False) for mid, r in profile['items'] if r < 2.5 and mid in movie_to_idx]

    np.random.seed(profile['user_id'])
    np.random.shuffle(liked)
    np.random.shuffle(disliked)

    # Interleave 2 liked : 1 disliked
    seq = []
    li, di = 0, 0
    while li < len(liked) or di < len(disliked):
        if li < len(liked):
            mid, is_liked = liked[li]
            seq.append((movie_to_idx[mid], is_liked))
            li += 1
        if li < len(liked):
            mid, is_liked = liked[li]
            seq.append((movie_to_idx[mid], is_liked))
            li += 1
        if di < len(disliked):
            mid, is_liked = disliked[di]
            seq.append((movie_to_idx[mid], is_liked))
            di += 1
    return seq


def build_seq_attributes(profile, movie_to_idx, movie_genres, top_movies):
    """
    Condition B: reveal genre preferences.
    Each step = mark all movies of one genre as liked/disliked.
    """
    # Compute user's genre preferences
    genre_ratings = {}
    for mid, rating in profile['items']:
        for g in movie_genres.get(mid, []):
            if g not in genre_ratings:
                genre_ratings[g] = []
            genre_ratings[g].append(rating)

    liked_genres = [(g, True) for g, scores in genre_ratings.items() if np.mean(scores) >= 4.0]
    disliked_genres = [(g, False) for g, scores in genre_ratings.items() if np.mean(scores) < 2.5]

    # Each "step" reveals all movies of one genre
    seq = []
    for genre, is_liked in liked_genres + disliked_genres:
        # Collect all pool movies of this genre
        genre_movies = []
        for mid in top_movies[:N_MOVIES]:
            if genre in movie_genres.get(mid, []) and mid in movie_to_idx:
                genre_movies.append((movie_to_idx[mid], is_liked))
        if genre_movies:
            seq.append(genre_movies)  # One step = list of (idx, is_liked)

    # Pad with individual movie revelations if needed
    liked_movies = [(mid, True) for mid, r in profile['items'] if r >= 4.0 and mid in movie_to_idx]
    disliked_movies = [(mid, False) for mid, r in profile['items'] if r < 2.5 and mid in movie_to_idx]
    np.random.seed(profile['user_id'])
    np.random.shuffle(liked_movies)
    np.random.shuffle(disliked_movies)
    for mid, is_liked in liked_movies + disliked_movies:
        seq.append([(movie_to_idx[mid], is_liked)])

    return seq


def build_seq_mixed(profile, movie_to_idx, movie_genres, top_movies):
    """Condition C: alternate genre reveals and individual movies."""
    # Genre steps
    genre_ratings = {}
    for mid, rating in profile['items']:
        for g in movie_genres.get(mid, []):
            if g not in genre_ratings:
                genre_ratings[g] = []
            genre_ratings[g].append(rating)

    liked_genres = [(g, True) for g, scores in genre_ratings.items() if np.mean(scores) >= 4.0]
    disliked_genres = [(g, False) for g, scores in genre_ratings.items() if np.mean(scores) < 2.5]

    genre_steps = []
    for genre, is_liked in liked_genres + disliked_genres:
        genre_movies = []
        for mid in top_movies[:N_MOVIES]:
            if genre in movie_genres.get(mid, []) and mid in movie_to_idx:
                genre_movies.append((movie_to_idx[mid], is_liked))
        if genre_movies:
            genre_steps.append(genre_movies)

    # Individual movie steps
    liked_movies = [(mid, True) for mid, r in profile['items'] if r >= 4.0 and mid in movie_to_idx]
    disliked_movies = [(mid, False) for mid, r in profile['items'] if r < 2.5 and mid in movie_to_idx]
    np.random.seed(profile['user_id'])
    np.random.shuffle(liked_movies)
    np.random.shuffle(disliked_movies)
    movie_steps = [[(movie_to_idx[mid], is_liked)] for mid, is_liked in liked_movies + disliked_movies]

    # Interleave: 1 genre, 1 movie, 1 genre, 1 movie...
    seq = []
    gi, mi = 0, 0
    while gi < len(genre_steps) or mi < len(movie_steps):
        if gi < len(genre_steps):
            seq.append(genre_steps[gi])
            gi += 1
        if mi < len(movie_steps):
            seq.append(movie_steps[mi])
            mi += 1
    return seq


# ─── Run one condition ─────────────────────────────────────────────────────

def run_condition(name, model, emb, n_items, profiles, build_fn,
                  top_movies, movie_to_idx, max_steps=20, n_users=30):
    """Run one condition. build_fn returns a list of steps, where each step
    is a list of (model_idx, is_liked) tuples to reveal."""
    sample = profiles[:n_users]
    print(f"\n  Condition: {name} ({len(sample)} users, {max_steps} steps)")

    all_curves = []
    idx_to_movie = {i: mid for mid, i in movie_to_idx.items()}

    for profile in tqdm(sample, desc=name, unit="user"):
        sequence = build_fn(profile)
        if not sequence:
            continue

        liked_ids = profile['liked_ids']
        curve = []

        # Accumulate revealed items across steps
        revealed = []

        # Step 0: baseline (all unknown)
        preds = predict_concept(model, emb, n_items, [])
        top10 = [top_movies[i] for i in np.argsort(-preds)[:10]]
        curve.append(calculate_ndcg(top10, liked_ids))

        # Steps 1..max_steps
        for step in range(1, max_steps + 1):
            if step <= len(sequence):
                step_items = sequence[step - 1]
                # For titles condition: step_items is a single (idx, is_liked) tuple
                if isinstance(step_items, tuple):
                    revealed.append(step_items)
                else:
                    # For genres/mixed: step_items is a list of tuples
                    revealed.extend(step_items)

            preds = predict_concept(model, emb, n_items, revealed)
            top10 = [top_movies[i] for i in np.argsort(-preds)[:10]]
            curve.append(calculate_ndcg(top10, liked_ids))

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
    peak_step = int(np.argmax(mean_curve))
    mono = sum(1 for i in range(1, len(mean_curve)) if mean_curve[i] >= mean_curve[i - 1])

    print(f"\n  Baseline: {mean_curve[0]:.4f}  Peak: {mean_curve[peak_step]:.4f} (s{peak_step})")
    print(f"  Final: {mean_curve[-1]:.4f}  Total: {total:+.4f}  Monotonic: {mono}/{max_steps}")

    return {
        'name': name,
        'n_users': len(all_curves),
        'mean_curve': mean_curve.tolist(),
        'std_curve': std_curve.tolist(),
        'total_improvement': float(total),
        'peak': float(mean_curve[peak_step]),
        'peak_step': peak_step,
        'monotonic_steps': mono,
    }


# ─── Main ──────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Concept Model (SBERT+LSTM+Attention) Reward Signal Test")
    print("Using SAVED checkpoint from RS_Model_Comparison notebook")
    print("=" * 60)

    print("\nLoading data...")
    top_movies, movie_to_idx, movie_titles, movie_genres, ratings_filtered = load_data()

    print("\nLoading model checkpoint...")
    model, n_items, emb = load_model()

    print("\nBuilding user profiles...")
    all_profiles = build_user_profiles(ratings_filtered, movie_to_idx, min_eval_targets=5)
    np.random.seed(SEED)
    indices = np.random.permutation(len(all_profiles))
    all_profiles = [all_profiles[i] for i in indices]
    print(f"  {len(all_profiles)} users with >= 5 liked movies in pool=100")

    test_profiles = all_profiles[:30]

    print("\n" + "=" * 60)
    print("EVALUATION")
    print("=" * 60)

    results = {}

    # A) Movie titles only (1 movie per step)
    results['titles'] = run_condition(
        "A: Movie Titles Only", model, emb, n_items, test_profiles,
        lambda p: build_seq_titles(p, movie_to_idx),
        top_movies, movie_to_idx,
    )

    # B) Attributes/genres only (1 genre = N movies per step)
    results['attributes'] = run_condition(
        "B: Attributes/Genres Only", model, emb, n_items, test_profiles,
        lambda p: build_seq_attributes(p, movie_to_idx, movie_genres, top_movies),
        top_movies, movie_to_idx,
    )

    # C) Mixed
    results['mixed'] = run_condition(
        "C: Mixed (Genres + Titles)", model, emb, n_items, test_profiles,
        lambda p: build_seq_mixed(p, movie_to_idx, movie_genres, top_movies),
        top_movies, movie_to_idx,
    )

    # Summary
    print(f"\n{'='*70}")
    print("CONCEPT MODEL RESULTS (saved checkpoint, pool=100)")
    print(f"{'='*70}")
    print(f"{'Condition':<30} {'Base':>7} {'Peak':>7} {'Final':>7} "
          f"{'Improv':>8} {'Mono':>6} {'Std@0':>7} {'Std@20':>7}")
    print(f"{'-'*75}")
    for key, r in results.items():
        print(f"{r['name']:<30} {r['mean_curve'][0]:>7.4f} {r['peak']:>7.4f} "
              f"{r['mean_curve'][-1]:>7.4f} {r['total_improvement']:>+8.4f} "
              f"{r['monotonic_steps']:>3}/20 {r['std_curve'][0]:>7.4f} "
              f"{r['std_curve'][-1]:>7.4f}")

    # LLM comparison
    llm_ref = RESULTS_DIR / 'preference_type_comparison.json'
    if llm_ref.exists():
        with open(llm_ref) as f:
            llm_data = json.load(f)
        print(f"\n{'='*70}")
        print("LLM DIAGNOSTIC REFERENCE (pool=100, same setup)")
        print(f"{'='*70}")
        print(f"{'Condition':<30} {'Base':>7} {'Final':>7} {'Improv':>8} {'Mono':>6}")
        print(f"{'-'*60}")
        for key, r in llm_data.items():
            print(f"{r['name']:<30} {r['mean_curve'][0]:>7.4f} "
                  f"{r['mean_curve'][-1]:>7.4f} "
                  f"{r['total_improvement']:>+8.4f} {r['monotonic_steps']:>3}/20")

    # Save
    out_file = RESULTS_DIR / 'concept_model_reward_diagnostic.json'
    with open(out_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {out_file}")


if __name__ == '__main__':
    main()
