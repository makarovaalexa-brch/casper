"""
Test the LSTM + Attention recommender with explicit rating encoding.

This should solve the likes/dislikes overlap problem from the baseline
by using explicit one-hot rating encoding instead of text encoding.
"""

import sys
sys.path.insert(0, 'C:/dev/phd/casper/src')

import torch
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from casper.models.lstm_attention_recommender import (
    LSTMAttentionRecommender,
    create_rating_one_hot,
    pad_sequences,
)
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
    num_users=500,
    include_titles=True,
    holdout_ratio=0.1,
    random_users=True
)

print(f"Created {len(pref_texts)} training examples")

# =============================================================================
# Parse preference texts into (concept, rating) sequences
# =============================================================================
print("\n" + "="*80)
print("PARSING PREFERENCES INTO SEQUENCES")
print("="*80)

def parse_pref_text(pref_text: str):
    """
    Parse preference text into sequence of (concept, rating) pairs.

    Input: "likes: matrix, sci-fi, action | dislikes: romance, drama"
    Output: [("matrix", "liked"), ("sci-fi", "liked"), ("romance", "disliked"), ...]
    """
    preferences = []

    # Split by likes/dislikes sections
    parts = pref_text.split(" | ")
    for part in parts:
        part = part.strip()
        if part.startswith("likes:"):
            rating = "liked"
            concepts_str = part[6:].strip()  # Remove "likes:"
        elif part.startswith("dislikes:"):
            rating = "disliked"
            concepts_str = part[9:].strip()  # Remove "dislikes:"
        else:
            continue

        # Split concepts by comma
        concepts = [c.strip() for c in concepts_str.split(",") if c.strip()]
        for concept in concepts:
            preferences.append((concept, rating))

    return preferences

# Parse all preference texts
user_preferences = []
for text in pref_texts:
    prefs = parse_pref_text(text)
    if prefs:
        user_preferences.append(prefs)
    else:
        # Fallback: treat entire text as single liked concept
        user_preferences.append([(text, "liked")])

print(f"Parsed {len(user_preferences)} user preference sequences")
print(f"Example: {user_preferences[0][:5]}...")

# =============================================================================
# Encode concepts with SBERT
# =============================================================================
print("\n" + "="*80)
print("ENCODING WITH SBERT")
print("="*80)

sbert = SentenceTransformer('all-MiniLM-L6-v2')

# Collect all unique concepts
all_concepts = set()
for prefs in user_preferences:
    for concept, _ in prefs:
        all_concepts.add(concept)

print(f"Found {len(all_concepts)} unique concepts")

# Encode all concepts
print("Encoding concepts...")
concept_list = list(all_concepts)
concept_embeddings_np = sbert.encode(concept_list, show_progress_bar=True)
concept_to_embedding = {c: torch.tensor(e) for c, e in zip(concept_list, concept_embeddings_np)}

# Load movies for item embeddings
import pandas as pd
movies_df = pd.read_csv(DATA_DIR / 'movies.csv')

# Encode movies needed for training
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

# =============================================================================
# Prepare user preference sequences as tensors
# =============================================================================
print("\n" + "="*80)
print("PREPARING TRAINING DATA")
print("="*80)

def prepare_user_sequence(prefs, concept_to_embedding, max_len=50):
    """Convert (concept, rating) list to tensors."""
    # Limit sequence length
    prefs = prefs[:max_len]

    # Get embeddings and ratings
    embeddings = []
    ratings = []
    for concept, rating in prefs:
        if concept in concept_to_embedding:
            embeddings.append(concept_to_embedding[concept])
            ratings.append(rating)

    if not embeddings:
        return None, None

    concept_emb = torch.stack(embeddings)  # [seq_len, 384]
    rating_one_hot = create_rating_one_hot(ratings)  # [seq_len, 3]

    return concept_emb, rating_one_hot

# Prepare all user sequences
user_sequences = []
valid_indices = []
for i, prefs in enumerate(user_preferences):
    concept_emb, rating_one_hot = prepare_user_sequence(prefs, concept_to_embedding)
    if concept_emb is not None:
        user_sequences.append((concept_emb, rating_one_hot))
        valid_indices.append(i)

print(f"Prepared {len(user_sequences)} valid user sequences")

# Map valid indices to liked/disliked ids
valid_liked_ids = [liked_ids[i] for i in valid_indices]
valid_disliked_ids = [disliked_ids[i] for i in valid_indices]

# Train/val split
n_train = int(len(user_sequences) * 0.9)
train_indices = list(range(n_train))
val_indices = list(range(n_train, len(user_sequences)))

print(f"Train: {len(train_indices)}, Val: {len(val_indices)}")

# =============================================================================
# Train the model
# =============================================================================
print("\n" + "="*80)
print("TRAINING LSTM+ATTENTION MODEL")
print("="*80)

model = LSTMAttentionRecommender(
    concept_dim=384,
    item_dim=384,
    hidden_dim=256,
    output_dim=128,
    num_lstm_layers=1,
    num_attention_heads=4,
    dropout=0.1,
    learning_rate=0.001,
    temperature=0.07,
    normalize=True,
)

NUM_NEGATIVES = 16
all_movie_ids = list(movie_embeddings.keys())

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
        concept_emb, rating_one_hot = user_sequences[idx]

        # Get positive item
        pos_movie_ids = valid_liked_ids[idx]
        if not pos_movie_ids:
            continue

        valid_pos_ids = [m for m in pos_movie_ids if m in movie_embeddings]
        if not valid_pos_ids:
            continue

        pos_id = np.random.choice(valid_pos_ids)
        pos_item = movie_embeddings[pos_id].clone()

        # Get negatives
        neg_items = sample_negatives(pos_movie_ids, valid_disliked_ids[idx], all_movie_ids, NUM_NEGATIVES)
        if len(neg_items) < NUM_NEGATIVES:
            continue
        neg_items = torch.stack(neg_items)

        # Train step
        loss = model.train_infonce_step(
            concept_embeddings=concept_emb.unsqueeze(0),  # [1, seq_len, 384]
            rating_one_hots=rating_one_hot.unsqueeze(0),  # [1, seq_len, 3]
            positive_items=pos_item.unsqueeze(0),  # [1, 384]
            negative_items=neg_items,  # [num_neg, 384]
        )
        train_loss += loss
        n_batches += 1

    train_loss /= max(n_batches, 1)

    # Validation NDCG
    model.eval()
    ndcg_scores = []
    with torch.no_grad():
        for idx in val_indices[:50]:
            concept_emb, rating_one_hot = user_sequences[idx]
            pos_movie_ids = [m for m in valid_liked_ids[idx] if m in movie_embeddings]

            if len(pos_movie_ids) < 2:
                continue

            # Score a candidate set
            candidate_ids = pos_movie_ids[:5] + list(np.random.choice(
                [m for m in all_movie_ids if m not in pos_movie_ids],
                min(95, len(all_movie_ids) - len(pos_movie_ids)),
                replace=False
            ))

            candidate_items = torch.stack([movie_embeddings[m] for m in candidate_ids])

            scores = model.predict_scores(
                concept_emb, rating_one_hot, candidate_items
            ).numpy()

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

# Get candidate movies
candidate_ids = list(movie_embeddings.keys())[:200]
candidate_items = torch.stack([movie_embeddings[m] for m in candidate_ids])

print(f"\nTesting with {len(candidate_ids)} movies...")
print(f"Comparing top-10 recommendations for 'likes: X' vs 'dislikes: X'\n")

overlaps = []
model.eval()
for concept in test_concepts:
    # Encode concept
    concept_emb = torch.tensor(sbert.encode(concept)).unsqueeze(0)  # [1, 384]

    # "Likes" version
    likes_rating = create_rating_one_hot(["liked"]).unsqueeze(0)  # [1, 1, 3]
    with torch.no_grad():
        likes_scores = model.predict_scores(
            concept_emb.unsqueeze(0),  # [1, 1, 384]
            likes_rating,
            candidate_items
        ).numpy().flatten()

    # "Dislikes" version
    dislikes_rating = create_rating_one_hot(["disliked"]).unsqueeze(0)
    with torch.no_grad():
        dislikes_scores = model.predict_scores(
            concept_emb.unsqueeze(0),
            dislikes_rating,
            candidate_items
        ).numpy().flatten()

    likes_top10 = set(np.argsort(likes_scores)[-10:].tolist())
    dislikes_top10 = set(np.argsort(dislikes_scores)[-10:].tolist())

    overlap = len(likes_top10 & dislikes_top10)
    overlaps.append(overlap)
    print(f"  {concept:25s}: {overlap}/10 overlap ({overlap*10}%)")

avg_overlap = np.mean(overlaps)
print(f"\n  Average overlap: {avg_overlap:.1f}/10 ({avg_overlap*10:.0f}%)")

if avg_overlap >= 7:
    print("\n  >>> PROBLEM: High overlap - model can't distinguish likes/dislikes!")
elif avg_overlap >= 5:
    print("\n  >>> WARNING: Moderate overlap")
elif avg_overlap >= 3:
    print("\n  >>> OK: Low-moderate overlap")
else:
    print("\n  >>> GOOD: Low overlap - model distinguishes likes/dislikes!")

# =============================================================================
# SANITY TEST: User Embedding Variance
# =============================================================================
print("\n" + "="*80)
print("SANITY TEST: User Embedding Variance")
print("="*80)

test_states = [
    ("Science Fiction, Action", ["liked", "liked"]),
    ("Romance, Drama", ["liked", "liked"]),
    ("Horror, Thriller", ["liked", "liked"]),
    ("Comedy", ["disliked"]),
    ("Christopher Nolan movies", ["liked"]),
]

embeddings = []
for concepts_str, ratings in test_states:
    concepts = [c.strip() for c in concepts_str.split(",")]
    concept_embs = torch.stack([torch.tensor(sbert.encode(c)) for c in concepts])
    rating_one_hot = create_rating_one_hot(ratings)

    with torch.no_grad():
        user_emb = model.encode_user(
            concept_embs.unsqueeze(0),
            rating_one_hot.unsqueeze(0)
        )
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
LSTM+ATTENTION RECOMMENDER:
  - Best Val NDCG@10: {best_val_ndcg:.4f}
  - Likes vs Dislikes overlap: {avg_overlap:.1f}/10 ({avg_overlap*10:.0f}%)
  - User embedding std: {std:.4f}

TARGET:
  - NDCG@10 >= 0.30
  - Likes vs Dislikes overlap < 30%
  - User embedding std > 0.15

BASELINE (Two-Tower):
  - NDCG@10: 0.3886
  - Likes vs Dislikes overlap: 68%
  - User embedding std: 0.3539

CONCLUSION:
  {'PASS' if best_val_ndcg >= 0.30 and avg_overlap < 3 else 'NEEDS IMPROVEMENT'}
""")
