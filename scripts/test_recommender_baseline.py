"""
Test the two-tower recommender baseline.
1. Train on small dataset
2. Check NDCG
3. Run sanity tests: likes vs dislikes overlap
"""

import sys
sys.path.insert(0, 'C:/dev/phd/casper/src')

import torch
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from casper.models.two_tower_recommender import TwoTowerRecommender
from casper.data.recommender_data_builder import create_recommender_training_data

# =============================================================================
# Build dataset
# =============================================================================
print("="*80)
print("BUILDING DATASET")
print("="*80)

DATA_DIR = Path('C:/dev/phd/casper/data/movielens')

pref_texts, liked_ids, disliked_ids = create_recommender_training_data(
    ratings_path=DATA_DIR / 'ratings.csv',
    movies_path=DATA_DIR / 'movies.csv',
    genome_scores_path=DATA_DIR / 'genome-scores.csv',
    genome_tags_path=DATA_DIR / 'genome-tags.csv',
    num_users=500,  # Small for quick testing
    include_titles=True,
    holdout_ratio=0.1,
    random_users=True
)

print(f"Created {len(pref_texts)} training examples")
print(f"\nExample preference text:")
print(f"  {pref_texts[0][:100]}...")

# =============================================================================
# Encode with SBERT
# =============================================================================
print("\n" + "="*80)
print("ENCODING WITH SBERT")
print("="*80)

sbert = SentenceTransformer('all-MiniLM-L6-v2')

# Load movies for item embeddings
import pandas as pd
movies_df = pd.read_csv(DATA_DIR / 'movies.csv')

# Create movie id -> embedding cache
# Only encode movies that appear in our training data for speed
all_movie_ids_needed = set()
for ids in liked_ids + disliked_ids:
    all_movie_ids_needed.update(ids)

print(f"Encoding {len(all_movie_ids_needed)} relevant movies...")
movie_embeddings = {}
movies_to_encode = movies_df[movies_df['movieId'].isin(all_movie_ids_needed)]
for _, row in movies_to_encode.iterrows():
    movie_id = row['movieId']
    movie_text = f"movie: {row['title']}"
    movie_embeddings[movie_id] = torch.tensor(sbert.encode(movie_text))

print(f"Encoded {len(movie_embeddings)} movies")

# Encode user preference texts
print("Encoding user preferences...")
user_state_embeddings = torch.tensor(sbert.encode(pref_texts, show_progress_bar=True))
print(f"Encoded {len(user_state_embeddings)} user states")

# =============================================================================
# Prepare training data
# =============================================================================
print("\n" + "="*80)
print("PREPARING TRAINING DATA")
print("="*80)

# Split train/val
n_train = int(len(pref_texts) * 0.9)
train_indices = list(range(n_train))
val_indices = list(range(n_train, len(pref_texts)))

print(f"Train: {len(train_indices)}, Val: {len(val_indices)}")

# =============================================================================
# Train the model
# =============================================================================
print("\n" + "="*80)
print("TRAINING TWO-TOWER MODEL")
print("="*80)

model = TwoTowerRecommender(
    state_dim=384,
    embedding_dim=128,
    learning_rate=0.0003,
    deep=True,
    dropout=0.1,
    normalize=True
)

NUM_NEGATIVES = 16

def sample_negatives(positive_ids, disliked_ids_list, all_movie_ids, num_neg):
    """Sample negative movies (mix of random + explicitly disliked)."""
    negatives = []
    pos_set = set(positive_ids)

    # Add disliked movies if available
    if disliked_ids_list:
        for mid in disliked_ids_list[:num_neg//2]:
            if mid in movie_embeddings:
                negatives.append(movie_embeddings[mid].clone())

    # Fill rest with random movies
    available = [m for m in all_movie_ids if m not in pos_set and m in movie_embeddings]
    for mid in np.random.choice(available, min(num_neg - len(negatives), len(available)), replace=False):
        negatives.append(movie_embeddings[mid].clone())

    return negatives

all_movie_ids = list(movie_embeddings.keys())

print("\nTraining...")
print("-"*80)
print(f"{'Epoch':>5} | {'Loss':>8} | {'Val NDCG@10':>12}")
print("-"*80)

best_val_ndcg = 0
for epoch in range(15):
    # Training
    model.train()
    train_loss = 0
    n_batches = 0

    np.random.shuffle(train_indices)
    for idx in train_indices:
        user_state = user_state_embeddings[idx].clone().unsqueeze(0)

        # Get positive item
        pos_movie_ids = liked_ids[idx]
        if not pos_movie_ids:
            continue

        pos_id = np.random.choice([m for m in pos_movie_ids if m in movie_embeddings])
        pos_item = movie_embeddings[pos_id].clone().unsqueeze(0)

        # Get negatives
        neg_items = sample_negatives(pos_movie_ids, disliked_ids[idx], all_movie_ids, NUM_NEGATIVES)
        if len(neg_items) < NUM_NEGATIVES:
            continue
        neg_items = torch.stack(neg_items)

        loss = model.train_infonce_step(user_state, pos_item, neg_items)
        train_loss += loss
        n_batches += 1

    train_loss /= max(n_batches, 1)

    # Validation NDCG
    model.eval()
    ndcg_scores = []
    with torch.no_grad():
        for idx in val_indices[:50]:  # Sample for speed
            user_state = user_state_embeddings[idx]
            pos_movie_ids = [m for m in liked_ids[idx] if m in movie_embeddings]

            if len(pos_movie_ids) < 2:
                continue

            # Score a candidate set
            candidate_ids = pos_movie_ids[:5] + list(np.random.choice(
                [m for m in all_movie_ids if m not in pos_movie_ids],
                min(95, len(all_movie_ids) - len(pos_movie_ids)),
                replace=False
            ))

            candidate_items = torch.stack([movie_embeddings[m] for m in candidate_ids])
            scores = model.predict_scores(user_state, candidate_items).numpy()

            # Calculate NDCG@10
            ranked_ids = [candidate_ids[i] for i in np.argsort(scores)[::-1]]
            dcg = 0
            for i, mid in enumerate(ranked_ids[:10]):
                if mid in pos_movie_ids:
                    dcg += 1 / np.log2(i + 2)

            idcg = sum(1 / np.log2(i + 2) for i in range(min(10, len(pos_movie_ids))))
            ndcg = dcg / idcg if idcg > 0 else 0
            ndcg_scores.append(ndcg)

    val_ndcg = np.mean(ndcg_scores) if ndcg_scores else 0
    marker = " *" if val_ndcg > best_val_ndcg else ""
    if val_ndcg > best_val_ndcg:
        best_val_ndcg = val_ndcg

    print(f"{epoch+1:>5} | {train_loss:>8.4f} | {val_ndcg:>12.4f}{marker}")

print("-"*80)
print(f"\nBest Val NDCG@10: {best_val_ndcg:.4f}")

# =============================================================================
# SANITY TEST: Likes vs Dislikes
# =============================================================================
print("\n" + "="*80)
print("SANITY TEST: Likes vs Dislikes Overlap")
print("="*80)

test_concepts = [
    "Science Fiction",
    "Action",
    "Romance",
    "Horror",
    "The Matrix (1999)",
    "Titanic (1997)",
]

# Get candidate movies (subset for speed)
candidate_ids = list(movie_embeddings.keys())[:200]
candidate_items = torch.stack([movie_embeddings[m] for m in candidate_ids])

print(f"\nTesting with {len(candidate_ids)} movies...")
print(f"Comparing top-10 recommendations for 'likes: X' vs 'dislikes: X'\n")

overlaps = []
model.eval()
for concept in test_concepts:
    likes_state = f"likes: {concept}"
    dislikes_state = f"dislikes: {concept}"

    likes_emb = torch.tensor(sbert.encode(likes_state))
    dislikes_emb = torch.tensor(sbert.encode(dislikes_state))

    with torch.no_grad():
        likes_scores = model.predict_scores(likes_emb, candidate_items).numpy()
        dislikes_scores = model.predict_scores(dislikes_emb, candidate_items).numpy()

    likes_top10 = set(np.argsort(likes_scores)[-10:])
    dislikes_top10 = set(np.argsort(dislikes_scores)[-10:])

    overlap = len(likes_top10 & dislikes_top10)
    overlaps.append(overlap)
    print(f"  {concept:25s}: {overlap}/10 overlap ({overlap*10}%)")

avg_overlap = np.mean(overlaps)
print(f"\n  Average overlap: {avg_overlap:.1f}/10 ({avg_overlap*10:.0f}%)")

if avg_overlap >= 7:
    print("\n  >>> PROBLEM: High overlap - model can't distinguish likes/dislikes!")
elif avg_overlap >= 5:
    print("\n  >>> WARNING: Moderate overlap")
else:
    print("\n  >>> GOOD: Low overlap")

# =============================================================================
# SANITY TEST: User embedding variance
# =============================================================================
print("\n" + "="*80)
print("SANITY TEST: User Embedding Variance")
print("="*80)

test_states = [
    "likes: Science Fiction, Action",
    "likes: Romance, Drama",
    "likes: Horror, Thriller",
    "dislikes: Comedy",
    "likes: Christopher Nolan movies",
]

embeddings = []
for state in test_states:
    emb = torch.tensor(sbert.encode(state)).unsqueeze(0)
    with torch.no_grad():
        user_emb = model.user_tower(emb)
        embeddings.append(user_emb.numpy())

embeddings = np.vstack(embeddings)
std = np.std(embeddings, axis=0).mean()

print(f"\n  User embedding std: {std:.4f}")
if std < 0.1:
    print("  >>> WARNING: Low variance - possible embedding collapse!")
else:
    print("  >>> OK: Reasonable variance")

# =============================================================================
# Summary
# =============================================================================
print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"""
BASELINE TWO-TOWER RECOMMENDER:
  - Best Val NDCG@10: {best_val_ndcg:.4f}
  - Likes vs Dislikes overlap: {avg_overlap:.1f}/10 ({avg_overlap*10:.0f}%)
  - User embedding std: {std:.4f}

TARGET:
  - NDCG@10 >= 0.30
  - Likes vs Dislikes overlap < 30%
  - User embedding std > 0.15

CONCLUSION:
  {'PASS' if best_val_ndcg >= 0.30 and avg_overlap < 3 else 'NEEDS IMPROVEMENT'}
""")
