"""
Test: LSTMAttentionRecommender (SBERT two-tower) as reward signal.

Trains the LSTMAttentionRecommender from scratch, then runs the same
three preference type conditions as the LLM diagnostic:
  A) Movie titles only
  B) Attributes/genres only
  C) Mixed (genres + titles interleaved)

Pool=100, 20 steps, 30 users. Directly comparable to LLM results.

Run: python scripts/test_lstm_recommender_reward.py
"""

import sys
import json
import numpy as np
import pandas as pd
import torch
import re
from pathlib import Path
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))
DATA_DIR = PROJECT_ROOT / 'data' / 'movielens'
RESULTS_DIR = PROJECT_ROOT / 'experiments' / 'diagnostics'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

N_MOVIES = 100
SEED = 42

from casper.models.lstm_attention_recommender import (
    LSTMAttentionRecommender,
    create_rating_one_hot,
)


# ─── Data loading ───────────────────────────────────────────────────────────

def load_data():
    """Load movies, ratings, build pool."""
    movies_df = pd.read_csv(DATA_DIR / 'movies.csv')
    ratings_df = pd.read_csv(
        DATA_DIR / 'ratings.csv',
        dtype={'userId': 'int32', 'movieId': 'int32', 'rating': 'float32'},
        usecols=['userId', 'movieId', 'rating'],
    )

    # Top 100 movies by rating count (same as notebook and LLM diagnostic)
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


def build_user_profiles(ratings_filtered, movie_to_idx, min_eval_targets=5,
                        max_profiles=1000):
    """Build user profiles with items in pool (vectorized, capped)."""
    df = ratings_filtered[ratings_filtered['movieId'].isin(movie_to_idx)].copy()
    df['liked'] = (df['rating'] >= 4.0).astype('int8')
    df['disliked'] = (df['rating'] < 2.5).astype('int8')

    # Filter to users with enough liked movies
    user_liked_counts = df.groupby('userId')['liked'].sum()
    eligible_users = user_liked_counts[user_liked_counts >= min_eval_targets].index.tolist()
    print(f"  {len(eligible_users)} eligible users, sampling {min(max_profiles, len(eligible_users))}")

    np.random.seed(SEED)
    sampled_users = np.random.choice(eligible_users,
                                     size=min(max_profiles, len(eligible_users)),
                                     replace=False)

    # Pre-group for fast lookup
    grouped = {uid: grp for uid, grp in df[df['userId'].isin(sampled_users)].groupby('userId')}

    profiles = []
    for uid in sampled_users:
        if uid not in grouped:
            continue
        user_df = grouped[uid]
        items = list(zip(user_df['movieId'].values, user_df['rating'].values))
        liked_ids = user_df[user_df['liked'] == 1]['movieId'].tolist()
        disliked_ids = user_df[user_df['disliked'] == 1]['movieId'].tolist()
        profiles.append({
            'user_id': int(uid),
            'items': [(int(m), float(r)) for m, r in items],
            'liked_ids': [int(m) for m in liked_ids],
            'disliked_ids': [int(m) for m in disliked_ids],
        })
    return profiles


# ─── SBERT encoding ────────────────────────────────────────────────────────

def encode_all_concepts(movie_titles, movie_genres, sbert):
    """Encode all movie titles and genre names with SBERT."""
    # Movie title embeddings
    title_texts = list(movie_titles.values())
    title_ids = list(movie_titles.keys())
    print(f"  Encoding {len(title_texts)} movie titles...")
    title_embs = sbert.encode(title_texts, show_progress_bar=False)
    movie_title_emb = {mid: torch.tensor(emb) for mid, emb in zip(title_ids, title_embs)}

    # Genre embeddings
    all_genres = set()
    for genres in movie_genres.values():
        all_genres.update(genres)
    genre_list = sorted(all_genres)
    print(f"  Encoding {len(genre_list)} genres...")
    genre_embs = sbert.encode(genre_list, show_progress_bar=False)
    genre_emb = {g: torch.tensor(emb) for g, emb in zip(genre_list, genre_embs)}

    # Item embeddings matrix for scoring (all pool movies)
    item_emb_matrix = torch.stack([movie_title_emb[mid] for mid in title_ids])

    return movie_title_emb, genre_emb, item_emb_matrix, title_ids


# ─── Training ──────────────────────────────────────────────────────────────

def train_model(profiles, movie_titles, movie_genres, movie_title_emb,
                genre_emb, item_emb_matrix, n_train_users=500, n_epochs=30):
    """Train LSTMAttentionRecommender with InfoNCE + contrastive like/dislike loss."""
    import torch.nn.functional as F

    model = LSTMAttentionRecommender(
        concept_dim=384, item_dim=384, hidden_dim=256, output_dim=128,
        num_lstm_layers=1, num_attention_heads=4, dropout=0.1,
        learning_rate=0.0005, temperature=0.1, normalize=True,
    )

    train_profiles = profiles[:n_train_users]
    all_movie_ids = list(movie_title_emb.keys())

    # Pre-build training data for speed
    print(f"\n  Preparing training examples...")
    train_examples = []
    for profile in train_profiles:
        liked = [m for m in profile['liked_ids'] if m in movie_title_emb]
        disliked = [m for m in profile['disliked_ids'] if m in movie_title_emb]
        if len(liked) < 2:
            continue

        # Build concept sequences at multiple lengths (1, 3, 5, 10 concepts)
        all_concepts = []
        all_ratings = []
        for mid, rating in profile['items']:
            if mid not in movie_title_emb:
                continue
            all_concepts.append(movie_title_emb[mid])
            all_ratings.append("liked" if rating >= 4.0 else "disliked")
            for g in movie_genres.get(mid, []):
                if g in genre_emb:
                    all_concepts.append(genre_emb[g])
                    all_ratings.append("liked" if rating >= 4.0 else "disliked")

        if len(all_concepts) < 2:
            continue

        # Create examples at varying sequence lengths for diversity
        for seq_len in [1, 2, 3, 5, 8, 12, 20, min(30, len(all_concepts))]:
            if seq_len > len(all_concepts):
                continue
            train_examples.append({
                'concepts': all_concepts[:seq_len],
                'ratings': all_ratings[:seq_len],
                'liked': liked,
                'disliked': disliked,
            })

    print(f"  {len(train_examples)} training examples from {len(train_profiles)} users")
    print(f"\n  Training for {n_epochs} epochs")
    print(f"  {'Epoch':>5} | {'InfoNCE':>8} | {'Contr':>8} | {'Val NDCG':>10}")
    print(f"  {'-'*42}")

    NUM_NEGATIVES = 32

    for epoch in range(n_epochs):
        model.train()
        infonce_total = 0
        contrastive_total = 0
        n_batches = 0

        np.random.shuffle(train_examples)

        for ex in train_examples:
            concept_emb = torch.stack(ex['concepts']).unsqueeze(0)
            rating_oh = create_rating_one_hot(ex['ratings']).unsqueeze(0)

            liked = ex['liked']
            disliked = ex['disliked']

            # === InfoNCE loss: liked movie should score high ===
            pos_id = np.random.choice(liked)
            pos_item = movie_title_emb[pos_id].unsqueeze(0)

            # Hard negatives: mix of disliked + random
            liked_set = set(liked)
            neg_items_list = []
            for m in disliked[:NUM_NEGATIVES // 2]:
                neg_items_list.append(movie_title_emb[m])
            random_pool = [m for m in all_movie_ids if m not in liked_set]
            n_random = NUM_NEGATIVES - len(neg_items_list)
            for m in np.random.choice(random_pool, min(n_random, len(random_pool)), replace=False):
                neg_items_list.append(movie_title_emb[m])

            if len(neg_items_list) < 4:
                continue
            neg_items = torch.stack(neg_items_list)

            loss_infonce = model.train_infonce_step(
                concept_emb, rating_oh, pos_item, neg_items
            )

            # === Contrastive like/dislike loss ===
            # Same concept with flipped rating should produce different user embeddings
            # that score the SAME item differently
            if len(ex['concepts']) >= 1:
                model.optimizer.zero_grad()

                # Original: as-is
                user_emb_orig = model.encode_user(concept_emb, rating_oh)

                # Flipped: swap liked<->disliked
                flipped_ratings = []
                for r in ex['ratings']:
                    flipped_ratings.append("disliked" if r == "liked" else "liked")
                flipped_oh = create_rating_one_hot(flipped_ratings).unsqueeze(0)
                user_emb_flipped = model.encode_user(concept_emb, flipped_oh)

                # The original should be closer to liked items, flipped to disliked
                target_item = movie_title_emb[pos_id]
                item_emb = model.encode_items(target_item.unsqueeze(0))

                score_orig = (user_emb_orig * item_emb).sum()
                score_flipped = (user_emb_flipped * item_emb).sum()

                # Margin loss: original should score higher than flipped by margin
                margin = 0.3
                loss_contrast = F.relu(score_flipped - score_orig + margin)

                loss_contrast.backward()
                model.optimizer.step()
                contrastive_total += loss_contrast.item()

            infonce_total += loss_infonce
            n_batches += 1

        infonce_avg = infonce_total / max(n_batches, 1)
        contrast_avg = contrastive_total / max(n_batches, 1)

        # Validation NDCG (step-by-step, same as eval)
        model.eval()
        val_ndcgs = []
        val_profiles = profiles[n_train_users:n_train_users + 50]
        with torch.no_grad():
            for profile in val_profiles:
                liked = [m for m in profile['liked_ids'] if m in movie_title_emb]
                if len(liked) < 3:
                    continue

                concepts = []
                ratings_list = []
                for mid, rating in profile['items'][:10]:
                    if mid in movie_title_emb:
                        concepts.append(movie_title_emb[mid])
                        ratings_list.append("liked" if rating >= 4.0 else "disliked")
                if not concepts:
                    continue

                ce = torch.stack(concepts).unsqueeze(0)
                ro = create_rating_one_hot(ratings_list).unsqueeze(0)
                scores = model.predict_scores(ce, ro, item_emb_matrix).numpy().flatten()

                top10_ids = [all_movie_ids[i] for i in np.argsort(-scores)[:10]]
                liked_set = set(liked)
                dcg = sum(1.0 / np.log2(i + 2) for i, mid in enumerate(top10_ids) if mid in liked_set)
                idcg = sum(1.0 / np.log2(i + 2) for i in range(min(10, len(liked))))
                val_ndcgs.append(dcg / idcg if idcg > 0 else 0)

        val_ndcg = np.mean(val_ndcgs) if val_ndcgs else 0
        print(f"  {epoch+1:>5} | {infonce_avg:>8.4f} | {contrast_avg:>8.4f} | {val_ndcg:>10.4f}")

    return model


# ─── Sequence builders (same logic as LLM diagnostic) ──────────────────────

def get_user_genre_prefs(profile, movie_genres):
    """Get user's genre preferences from their ratings."""
    genre_scores = {}
    for mid, rating in profile['items']:
        for g in movie_genres.get(mid, []):
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


def build_seq_titles(profile, movie_title_emb):
    """Condition A: movie titles only."""
    liked = [(mid, r) for mid, r in profile['items'] if r >= 4.0 and mid in movie_title_emb]
    disliked = [(mid, r) for mid, r in profile['items'] if r < 2.5 and mid in movie_title_emb]

    np.random.seed(profile['user_id'])
    np.random.shuffle(liked)
    np.random.shuffle(disliked)

    # Interleave 2 liked : 1 disliked
    seq = []
    li, di = 0, 0
    while li < len(liked) or di < len(disliked):
        if li < len(liked):
            seq.append((movie_title_emb[liked[li][0]], "liked"))
            li += 1
        if li < len(liked):
            seq.append((movie_title_emb[liked[li][0]], "liked"))
            li += 1
        if di < len(disliked):
            seq.append((movie_title_emb[disliked[di][0]], "disliked"))
            di += 1
    return seq


def build_seq_attributes(profile, movie_genres, genre_emb, movie_title_emb):
    """Condition B: genres/attributes only, pad with movie titles if needed."""
    liked_genres, disliked_genres = get_user_genre_prefs(profile, movie_genres)

    seq = []
    for g in liked_genres:
        if g in genre_emb:
            seq.append((genre_emb[g], "liked"))
    for g in disliked_genres:
        if g in genre_emb:
            seq.append((genre_emb[g], "disliked"))

    # Pad with movie title preferences if we run out of genres
    liked = [(mid, r) for mid, r in profile['items'] if r >= 4.0 and mid in movie_title_emb]
    disliked = [(mid, r) for mid, r in profile['items'] if r < 2.5 and mid in movie_title_emb]
    np.random.seed(profile['user_id'])
    np.random.shuffle(liked)
    np.random.shuffle(disliked)
    for mid, _ in liked:
        seq.append((movie_title_emb[mid], "liked"))
    for mid, _ in disliked:
        seq.append((movie_title_emb[mid], "disliked"))
    return seq


def build_seq_mixed(profile, movie_genres, genre_emb, movie_title_emb):
    """Condition C: interleave genres and titles."""
    liked_genres, disliked_genres = get_user_genre_prefs(profile, movie_genres)
    liked = [(mid, r) for mid, r in profile['items'] if r >= 4.0 and mid in movie_title_emb]
    disliked = [(mid, r) for mid, r in profile['items'] if r < 2.5 and mid in movie_title_emb]

    np.random.seed(profile['user_id'])
    np.random.shuffle(liked)
    np.random.shuffle(disliked)

    seq = []
    gl, gd, ml, md = 0, 0, 0, 0
    while (gl < len(liked_genres) or gd < len(disliked_genres) or
           ml < len(liked) or md < len(disliked)):
        if gl < len(liked_genres) and liked_genres[gl] in genre_emb:
            seq.append((genre_emb[liked_genres[gl]], "liked"))
            gl += 1
        if ml < len(liked):
            seq.append((movie_title_emb[liked[ml][0]], "liked"))
            ml += 1
        if gd < len(disliked_genres) and disliked_genres[gd] in genre_emb:
            seq.append((genre_emb[disliked_genres[gd]], "disliked"))
            gd += 1
        if md < len(disliked):
            seq.append((movie_title_emb[disliked[md][0]], "disliked"))
            md += 1
    return seq


# ─── NDCG calculation ──────────────────────────────────────────────────────

def calculate_ndcg(recommended_ids, relevant_ids, k=10):
    if not relevant_ids:
        return 0.0
    relevant_set = set(relevant_ids)
    dcg = sum(1.0 / np.log2(i + 2) for i, rid in enumerate(recommended_ids[:k]) if rid in relevant_set)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(k, len(relevant_ids))))
    return dcg / idcg if idcg > 0 else 0.0


# ─── Run one condition ─────────────────────────────────────────────────────

def run_condition(name, model, profiles, build_fn, item_emb_matrix,
                  pool_movie_ids, max_steps=20, n_users=30):
    """Run a single condition and return NDCG curve."""
    sample = profiles[:n_users]
    print(f"\n  Condition: {name} ({len(sample)} users, {max_steps} steps)")

    model.eval()
    all_curves = []

    for profile in tqdm(sample, desc=name, unit="user"):
        sequence = build_fn(profile)
        if not sequence:
            continue

        liked_ids = profile['liked_ids']
        curve = []

        with torch.no_grad():
            # Step 0: baseline — single "unknown" concept
            unknown_emb = torch.zeros(1, 1, 384)
            unknown_rating = create_rating_one_hot(["unknown"]).unsqueeze(0)
            scores = model.predict_scores(
                unknown_emb, unknown_rating, item_emb_matrix
            ).numpy().flatten()
            top10 = [pool_movie_ids[i] for i in np.argsort(-scores)[:10]]
            curve.append(calculate_ndcg(top10, liked_ids))

            # Steps 1..max_steps
            for step in range(1, max_steps + 1):
                revealed = sequence[:min(step, len(sequence))]
                embs = torch.stack([e for e, _ in revealed]).unsqueeze(0)
                ratings = create_rating_one_hot([r for _, r in revealed]).unsqueeze(0)

                scores = model.predict_scores(
                    embs, ratings, item_emb_matrix
                ).numpy().flatten()
                top10 = [pool_movie_ids[i] for i in np.argsort(-scores)[:10]]
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
    print("LSTMAttentionRecommender Reward Signal Diagnostic")
    print("(SBERT two-tower, pool=100, 20 steps)")
    print("=" * 60)

    print("\nLoading data...")
    top_movies, movie_to_idx, movie_titles, movie_genres, ratings_filtered = load_data()

    print("\nLoading SBERT encoder...")
    from sentence_transformers import SentenceTransformer
    sbert = SentenceTransformer('all-MiniLM-L6-v2')

    print("\nEncoding concepts...")
    movie_title_emb, genre_emb, item_emb_matrix, pool_movie_ids = encode_all_concepts(
        movie_titles, movie_genres, sbert
    )

    print("\nBuilding user profiles...")
    all_profiles = build_user_profiles(ratings_filtered, movie_to_idx, min_eval_targets=5)
    np.random.seed(SEED)
    indices = np.random.permutation(len(all_profiles))
    all_profiles = [all_profiles[i] for i in indices]
    print(f"  {len(all_profiles)} users with >= 5 liked movies in pool=100")

    # Reserve first 30 for testing, rest for training
    test_profiles = all_profiles[:30]
    train_profiles = all_profiles[30:]

    print("\n" + "=" * 60)
    print("TRAINING")
    print("=" * 60)
    model = train_model(
        train_profiles, movie_titles, movie_genres,
        movie_title_emb, genre_emb, item_emb_matrix,
        n_train_users=min(500, len(train_profiles)), n_epochs=15,
    )

    # ─── Run conditions ─────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("EVALUATION")
    print("=" * 60)

    results = {}

    # A) Movie titles only
    results['titles'] = run_condition(
        "A: Movie Titles Only", model, test_profiles,
        lambda p: build_seq_titles(p, movie_title_emb),
        item_emb_matrix, pool_movie_ids,
    )

    # B) Attributes/genres only
    results['attributes'] = run_condition(
        "B: Attributes Only", model, test_profiles,
        lambda p: build_seq_attributes(p, movie_genres, genre_emb, movie_title_emb),
        item_emb_matrix, pool_movie_ids,
    )

    # C) Mixed
    results['mixed'] = run_condition(
        "C: Mixed (Genres + Titles)", model, test_profiles,
        lambda p: build_seq_mixed(p, movie_genres, genre_emb, movie_title_emb),
        item_emb_matrix, pool_movie_ids,
    )

    # ─── Summary ────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("LSTM ATTENTION RECOMMENDER RESULTS (SBERT two-tower)")
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
    out_file = RESULTS_DIR / 'lstm_recommender_reward_diagnostic.json'
    with open(out_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {out_file}")


if __name__ == '__main__':
    main()
