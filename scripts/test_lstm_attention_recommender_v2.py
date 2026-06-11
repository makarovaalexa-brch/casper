"""
Test the LSTM + Attention recommender with CONTRASTIVE training.

Key insight: We must explicitly train the model to distinguish likes from dislikes
by creating contrastive training pairs where the same concept is marked as
liked vs disliked and the model must produce different rankings.
"""

import sys
sys.path.insert(0, 'C:/dev/phd/casper/src')

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from casper.models.lstm_attention_recommender import (
    LSTMAttentionRecommender,
    create_rating_one_hot,
)
from casper.data.recommender_data_builder import create_recommender_training_data

# =============================================================================
# Build dataset
# =============================================================================
print("="*80, flush=True)
print("BUILDING DATASET", flush=True)
print("="*80, flush=True)

DATA_DIR = Path('C:/dev/phd/casper/data/movielens')

pref_texts, liked_ids, disliked_ids = create_recommender_training_data(
    ratings_path=DATA_DIR / 'ratings.csv',
    movies_path=DATA_DIR / 'movies.csv',
    genome_scores_path=DATA_DIR / 'genome-scores.csv',
    genome_tags_path=DATA_DIR / 'genome-tags.csv',
    num_users=200,
    include_titles=True,
    holdout_ratio=0.1,
    random_users=True
)

print(f"Created {len(pref_texts)} training examples")

# =============================================================================
# Load SBERT and encode movies
# =============================================================================
print("\n" + "="*80)
print("ENCODING WITH SBERT")
print("="*80)

sbert = SentenceTransformer('all-MiniLM-L6-v2')

# Load movies
import pandas as pd
movies_df = pd.read_csv(DATA_DIR / 'movies.csv')

# Build movie_id -> title mapping
movie_id_to_title = dict(zip(movies_df['movieId'], movies_df['title']))

# Encode movies needed for training
all_movie_ids_needed = set()
for ids in liked_ids + disliked_ids:
    all_movie_ids_needed.update(ids)

print(f"Encoding {len(all_movie_ids_needed)} relevant movies (batch)...")
movie_embeddings = {}
movies_to_encode = movies_df[movies_df['movieId'].isin(all_movie_ids_needed)]

# Batch encode for speed
movie_ids_list = movies_to_encode['movieId'].tolist()
movie_texts = [f"movie: {row['title']}" for _, row in movies_to_encode.iterrows()]
movie_emb_np = sbert.encode(movie_texts, show_progress_bar=True)

print(f"SBERT encoding complete. Converting {len(movie_ids_list)} tensors...", flush=True)
for i, movie_id in enumerate(movie_ids_list):
    movie_embeddings[movie_id] = torch.tensor(movie_emb_np[i])
    if (i + 1) % 1000 == 0:
        print(f"  Converted {i+1}/{len(movie_ids_list)} tensors...", flush=True)

print(f"Encoded {len(movie_embeddings)} movies", flush=True)

# =============================================================================
# Create contrastive training data
# =============================================================================
print("\n" + "="*80, flush=True)
print("CREATING CONTRASTIVE TRAINING DATA", flush=True)
print("="*80, flush=True)

# For each user, create training examples:
# 1. (liked_movie_concept, "liked") -> liked_movie should rank high
# 2. (liked_movie_concept, "disliked") -> liked_movie should rank LOW
# This explicitly teaches the like/dislike distinction

training_examples = []

print(f"Processing {len(pref_texts)} users for contrastive examples...", flush=True)
for user_idx in range(len(pref_texts)):
    user_liked = [m for m in liked_ids[user_idx] if m in movie_embeddings]
    user_disliked = [m for m in disliked_ids[user_idx] if m in movie_embeddings]

    if not user_liked:
        continue

    # Create examples from liked movies
    # NOTE: Reuse movie_embeddings since concept is same as movie
    for movie_id in user_liked[:5]:  # Limit to 5 per user
        # Reuse movie embedding as concept embedding (they're the same text)
        concept_emb = movie_embeddings[movie_id].clone()

        # Positive example: (concept, liked) -> this movie should rank HIGH
        training_examples.append({
            'concept_emb': concept_emb,
            'rating': 'liked',
            'target_movie_id': movie_id,
            'target_is_positive': True,  # Should rank HIGH
            'user_liked': user_liked,
            'user_disliked': user_disliked,
        })

        # Contrastive example: (concept, disliked) -> this movie should rank LOW
        training_examples.append({
            'concept_emb': concept_emb,
            'rating': 'disliked',
            'target_movie_id': movie_id,
            'target_is_positive': False,  # Should rank LOW
            'user_liked': user_liked,
            'user_disliked': user_disliked,
        })

    # Also create examples from disliked movies (if available)
    for movie_id in user_disliked[:3]:
        # Reuse movie embedding as concept embedding
        concept_emb = movie_embeddings[movie_id].clone()

        # (disliked_movie, liked) -> should NOT rank high
        training_examples.append({
            'concept_emb': concept_emb,
            'rating': 'liked',
            'target_movie_id': movie_id,
            'target_is_positive': False,  # This movie is actually disliked
            'user_liked': user_liked,
            'user_disliked': user_disliked,
        })

        # (disliked_movie, disliked) -> confirms this movie should rank low
        training_examples.append({
            'concept_emb': concept_emb,
            'rating': 'disliked',
            'target_movie_id': movie_id,
            'target_is_positive': True,  # Correctly identified as disliked
            'user_liked': user_liked,
            'user_disliked': user_disliked,
        })

print(f"Created {len(training_examples)} contrastive training examples", flush=True)

# Split
np.random.shuffle(training_examples)
n_train = int(len(training_examples) * 0.9)
train_examples = training_examples[:n_train]
val_examples = training_examples[n_train:]

print(f"Train: {len(train_examples)}, Val: {len(val_examples)}", flush=True)

# =============================================================================
# Train the model with contrastive loss
# =============================================================================
print("\n" + "="*80, flush=True)
print("TRAINING LSTM+ATTENTION MODEL (CONTRASTIVE)", flush=True)
print("="*80, flush=True)

model = LSTMAttentionRecommender(
    concept_dim=384,
    item_dim=384,
    hidden_dim=256,
    output_dim=128,
    num_lstm_layers=1,
    num_attention_heads=4,
    dropout=0.1,
    learning_rate=0.001,
    temperature=0.1,  # Higher temperature for more gradient signal
    normalize=True,
)

all_movie_ids = list(movie_embeddings.keys())
NUM_NEGATIVES = 8

def get_contrastive_negatives(example, all_ids, num_neg):
    """Get negative items for contrastive learning."""
    target_id = example['target_movie_id']
    is_positive = example['target_is_positive']

    if is_positive:
        # Target should rank high, negatives should rank low
        # Use random movies as negatives
        available = [m for m in all_ids if m != target_id and m in movie_embeddings]
    else:
        # Target should rank low, negatives should be other low-ranking items
        available = [m for m in all_ids if m != target_id and m in movie_embeddings]

    neg_ids = np.random.choice(available, min(num_neg, len(available)), replace=False)
    return [movie_embeddings[m].clone() for m in neg_ids]

print("\nTraining with contrastive loss...", flush=True)
print("-"*80, flush=True)
print(f"{'Epoch':>5} | {'Loss':>8} | {'Contrast Acc':>12}", flush=True)
print("-"*80, flush=True)

for epoch in range(20):
    model.train()
    train_loss = 0
    correct = 0
    total = 0

    np.random.shuffle(train_examples)

    for ex in train_examples:
        concept_emb = ex['concept_emb'].unsqueeze(0).unsqueeze(0)  # [1, 1, 384]
        rating_one_hot = create_rating_one_hot([ex['rating']]).unsqueeze(0)  # [1, 1, 3]
        target_item = movie_embeddings[ex['target_movie_id']].clone().unsqueeze(0)  # [1, 384]

        # Get negatives
        neg_items = get_contrastive_negatives(ex, all_movie_ids, NUM_NEGATIVES)
        if len(neg_items) < NUM_NEGATIVES:
            continue
        neg_items = torch.stack(neg_items)  # [num_neg, 384]

        model.optimizer.zero_grad()

        # Encode user state
        user_emb = model.encode_user(concept_emb, rating_one_hot)  # [1, 128]

        # Encode items
        target_emb = model.encode_items(target_item)  # [1, 128]
        neg_emb = model.encode_items(neg_items)  # [num_neg, 128]

        # Compute scores
        target_score = (user_emb * target_emb).sum(dim=-1) / model.temperature  # [1]
        neg_scores = torch.matmul(user_emb, neg_emb.T) / model.temperature  # [1, num_neg]

        if ex['target_is_positive']:
            # Target should score higher than negatives
            logits = torch.cat([target_score.unsqueeze(1), neg_scores], dim=1)  # [1, 1+num_neg]
            labels = torch.zeros(1, dtype=torch.long)
            loss = F.cross_entropy(logits, labels)

            # Check if target ranked first
            if target_score.item() > neg_scores.max().item():
                correct += 1
        else:
            # Target should score LOWER than negatives
            # Reverse: want negatives to score higher
            logits = torch.cat([neg_scores, target_score.unsqueeze(1)], dim=1)  # [1, num_neg+1]
            # Any negative winning is good, so use margin loss instead
            margin = 0.5
            # Loss: max(0, target_score - min_neg_score + margin)
            loss = F.relu(target_score - neg_scores.min() + margin).mean()

            # Check if target ranked last (lower than all negatives)
            if target_score.item() < neg_scores.min().item():
                correct += 1

        total += 1

        loss.backward()
        model.optimizer.step()
        train_loss += loss.item()

    train_loss /= max(total, 1)
    acc = correct / max(total, 1)

    print(f"{epoch+1:>5} | {train_loss:>8.4f} | {acc:>12.4f}")

print("-"*80)

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
    concept_emb = torch.tensor(sbert.encode(concept)).unsqueeze(0).unsqueeze(0)  # [1, 1, 384]

    # Likes version
    likes_rating = create_rating_one_hot(["liked"]).unsqueeze(0)
    with torch.no_grad():
        likes_scores = model.predict_scores(
            concept_emb, likes_rating, candidate_items
        ).numpy().flatten()

    # Dislikes version
    dislikes_rating = create_rating_one_hot(["disliked"]).unsqueeze(0)
    with torch.no_grad():
        dislikes_scores = model.predict_scores(
            concept_emb, dislikes_rating, candidate_items
        ).numpy().flatten()

    likes_top10 = set(np.argsort(likes_scores)[-10:].tolist())
    dislikes_top10 = set(np.argsort(dislikes_scores)[-10:].tolist())

    overlap = len(likes_top10 & dislikes_top10)
    overlaps.append(overlap)

    # Also show score difference
    score_diff = np.mean(likes_scores) - np.mean(dislikes_scores)
    print(f"  {concept:25s}: {overlap}/10 overlap | score_diff: {score_diff:+.4f}")

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
    ("Science Fiction", "liked"),
    ("Science Fiction", "disliked"),
    ("Romance", "liked"),
    ("Romance", "disliked"),
    ("Action", "liked"),
]

embeddings = []
for concept_str, rating in test_states:
    concept_emb = torch.tensor(sbert.encode(concept_str)).unsqueeze(0).unsqueeze(0)
    rating_one_hot = create_rating_one_hot([rating]).unsqueeze(0)

    with torch.no_grad():
        user_emb = model.encode_user(concept_emb, rating_one_hot)
        embeddings.append(user_emb.numpy())
    print(f"  '{concept_str}' ({rating}): norm={np.linalg.norm(user_emb.numpy()):.4f}")

embeddings = np.vstack(embeddings)
std = np.std(embeddings, axis=0).mean()

print(f"\n  User embedding std: {std:.4f}")
if std < 0.1:
    print("  >>> WARNING: Low variance - possible embedding collapse!")
else:
    print("  >>> OK: Reasonable variance")

# Show cosine similarity between liked and disliked
print("\n  Cosine similarities:")
for i in range(0, len(test_states)-1, 2):
    if i+1 < len(test_states):
        sim = np.dot(embeddings[i], embeddings[i+1]) / (np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[i+1]))
        print(f"    {test_states[i][0]} liked vs disliked: {sim:.4f}")

# =============================================================================
# Summary
# =============================================================================
print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"""
LSTM+ATTENTION (CONTRASTIVE) RECOMMENDER:
  - Likes vs Dislikes overlap: {avg_overlap:.1f}/10 ({avg_overlap*10:.0f}%)
  - User embedding std: {std:.4f}

TARGET:
  - Likes vs Dislikes overlap < 30%
  - User embedding std > 0.15

BASELINE (Two-Tower):
  - Likes vs Dislikes overlap: 68%
  - User embedding std: 0.3539

CONCLUSION:
  {'PASS' if avg_overlap < 3 and std > 0.15 else 'NEEDS IMPROVEMENT'}
""")
