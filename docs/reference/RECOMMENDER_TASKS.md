# Two-Tower Recommender Improvement Plan

**Reference**: IEEE Paper "Learning a Strategy for Preference Elicitation in Conversational Recommender Systems" + CSMAI_19 notebook

## Problem Statement

### Current State
- **NDCG@10 achieved**: ~0.25-0.28 (best: 0.2571 with linear tower)
- **IMDB baselines for reference**: Collaborative filtering on MovieLens typically achieves 0.35-0.45 NDCG@10

### Core Issue: Preferences Not Used Properly
The model achieves decent aggregate NDCG but **fails to properly use user preference signals in predictions**. This is a **hard blocker for RL** because:

1. The RL policy relies on the recommender to update predictions based on newly revealed preferences
2. If "likes: Sci-Fi" doesn't significantly change Sci-Fi movie rankings, the agent gets no useful reward signal
3. Sanity tests show 8/10 overlap between recommendations for users who "like X" vs "dislike X"

### Why This Matters for CASPER
The conversational recommendation premise is that asking about preferences helps narrow down recommendations. If the model can't use preference information to change its predictions, the RL policy cannot learn meaningful question-asking behavior.

### Key Insight from IEEE Paper
> "The attention mechanism allows the network to learn the relationships between inputs and outputs directly based on embedded indexes of items and independently of the order of inputs"

The paper shows that:
- Loss decreases **consistently** with each new preference input (Fig 2)
- Model learns item-attribute correlations (Russell Crowe → Gladiator, A Beautiful Mind)
- Trained policy significantly outperforms random baseline (Fig 7)

---

## IEEE Paper / CSMAI-19 Approach: LSTM + Attention

### System Architecture (from Paper Fig 1)
```
┌─────────────────────────────────────────────────────────────────┐
│                        BOT-PLAY LOOP                            │
│                                                                 │
│  ┌─────────┐    "Do you like thrillers?"    ┌─────────┐        │
│  │  QBot   │ ──────────────────────────────→│  ABot   │        │
│  │         │←────────────────────────────── │         │        │
│  │ policy  │         "Yes"                  │ prefs   │        │
│  │ network │                                │ history │        │
│  └────┬────┘                                └─────────┘        │
│       │                                                         │
│       ▼                                                         │
│  ┌─────────────────┐      ┌──────────────────────┐             │
│  │   Preference    │      │   Recommendation     │             │
│  │ Representation  │─────→│   Network (LSTM +    │─→ reward    │
│  │ (item, rating)  │      │   Attention)         │   (ΔLoss)   │
│  └─────────────────┘      └──────────────────────┘             │
└─────────────────────────────────────────────────────────────────┘
```

### Recommendation Module Architecture (ExtrapolationModel)
```python
class ExtrapolationModel(nn.Module):
    def __init__(self, y_n=N_ITEMS):
        # Encoder
        self.embedding = nn.Embedding(y_n+1, y_n//2)  # learned item embeddings
        self.lstm = nn.LSTM(y_n//2 + 3, y_n//2, batch_first=True)  # +3 for rating one-hot

        # Decoder
        self.dense1 = nn.Linear(y_n//2, y_n)
        self.dense2 = nn.Linear(y_n, y_n//2)
        self.attention = nn.MultiheadAttention(y_n//2, num_heads=1)
        self.output = nn.Linear(y_n, y_n*2)  # explicit + implicit predictions

    def forward(self, index_input, rating_input):
        # Encode: embed item index, concat rating, process with LSTM
        x = self.embedding(index_input)
        x = torch.cat((x, rating_input), dim=-1)  # rating: [liked, disliked, not_seen]
        encoder_output, _ = self.lstm(x)

        # Decode: dense layers + attention connects encoder to outputs
        x = self.dense1(encoder_output)
        x = self.dense2(x)
        attention, _ = self.attention(encoder_output, x, x)
        x = torch.cat((x, attention), dim=-1)
        return self.output(x)  # predictions for ALL items
```

### Key Design Principles (from Paper)
1. **Item-to-item approach**: No user ID needed → **cold-start friendly**
2. **Explicit rating encoding**: One-hot [liked, disliked, not_seen] - **no ambiguity**
3. **Attention over encoder**: Learns item correlations **regardless of input order**
4. **Bottleneck architecture**: Forces learning of item-to-item correlations
5. **Cross-entropy loss with masking**: Only penalize predictions for rated items

### Reward Signal for RL Policy
```
r_t = ΔL = L_{t-1} - L_t
```
- Reward = reduction in recommendation loss after observing new preference
- Incentivizes questions that **maximize information gain**

### Paper Results
- **Attribute-to-movie learning works**: "Russell Crowe" → improves predictions for Gladiator (+0.89), A Beautiful Mind (+0.91)
- **Correlation clusters learned**: Star Wars episodes form distinct cluster (Fig 4)
- **Policy beats random**: Significant loss reduction in fewer turns (Fig 7)

---

## Adapting for SBERT Embeddings (CASPER Integration)

### Current Two-Tower Approach (BROKEN)
```
User state text: "likes: Sci-Fi | Action, dislikes: Romance"
    → SBERT encode → 384-dim vector
    → User Tower → 128-dim user embedding
    → Dot product with item embeddings

PROBLEM: SBERT conflates "likes: X" and "dislikes: X" → 80% overlap in recommendations
```

### Proposed SBERT + LSTM + Attention Approach
```
User preferences as sequence:
    [("Sci-Fi", liked), ("Action", liked), ("Romance", disliked)]

    For each preference:
        concept → SBERT encode → 384-dim
        rating → one-hot encode → 3-dim [liked, disliked, unknown]
        concat → 387-dim input

    Sequence → LSTM → encoder states
    Encoder states → Attention → pooled user representation
    Pooled → Dense → 128-dim user embedding
    User embedding → Dot product with item embeddings
```

### Key Differences
| Aspect | Current Two-Tower | Proposed SBERT+LSTM+Attention |
|--------|------------------|-------------------------------|
| Input format | Single text string | Sequence of (concept, rating) tuples |
| Like/dislike | Implicit in text (SBERT must infer) | **Explicit one-hot encoding** |
| Aggregation | SBERT mean pooling (fixed) | LSTM + Attention (learned) |
| Order | Fixed text format | Order-invariant via attention |
| Cold-start | Needs at least some text | Works with zero preferences |

### Why SBERT + LSTM Instead of Pure Learned Embeddings?
The paper uses **learned item embeddings** (nn.Embedding). For CASPER, we use **SBERT concept embeddings** because:
1. Concepts are **open vocabulary** (not fixed item set)
2. SBERT captures **semantic similarity** between concepts
3. Enables **zero-shot generalization** to new concepts
4. Pre-trained embeddings have richer representations

### Concept Vocabulary for CASPER
Input concepts should match what we expect from conversations:
- **Movie titles**: `"movie: The Matrix (1999)"` or `"movie: Inception (2010)"`
- **Genres**: `"genre: Science Fiction"` or `"genre: Action"`
- **Actors**: `"actor: Keanu Reeves"` or `"actor: Leonardo DiCaprio"`
- **Directors**: `"director: Christopher Nolan"`
- **Keywords/themes**: `"theme: time travel"` or `"plot: heist"`

**SBERT Encoding Sanity Check Results** (from `scripts/test_sbert_encoding.py`):

| Test | Finding |
|------|---------|
| Movie format | `"movie: Title (Year)"` works well (0.99 sim with `"film: Title (Year)"`) |
| Similar movies | WEAK clustering - within-group: 0.38-0.43, between-group: 0.37-0.39 |
| Genre encoding | Better clustering - Horror-Thriller: 0.80, Drama-Comedy: 0.71 |
| Concept types | Genres cluster (0.64-0.67), Actors cluster (0.45-0.56), Movies don't cluster well |
| **Like vs Dislike** | **0.75-0.78 similarity** - confirms need for explicit one-hot encoding! |
| Movie-Genre | LOW correlation (0.10-0.30) - SBERT doesn't know Matrix = Sci-Fi |

**Key Insight**: SBERT captures semantics but NOT like/dislike distinction. Must use explicit rating encoding!

### Scope
- **IN SCOPE**: LSTM+Attention recommender with SBERT concept embeddings
- **OUT OF SCOPE**: Bot-play, RL policy training (future work)

---

## Test Suite Requirements

### 1. Metric Evaluations
| Metric | Description | Target |
|--------|-------------|--------|
| NDCG@10 | Normalized Discounted Cumulative Gain | >= 0.30 |
| Hit@10 | Whether true positive in top 10 | >= 0.70 |
| U_std | User embedding standard deviation | > 0.15 |
| I_std | Item embedding standard deviation | > 0.20 |

### 2. Sanity Tests (Critical for RL)
| Test | Description | Expected |
|------|-------------|----------|
| Likes vs Dislikes Overlap | Top-10 for "likes: X" vs "dislikes: X" | < 30% overlap |
| Add Preference Delta | Ranking change after adding preference | Relevant items move significantly |
| Contradictory Preferences | "likes: X, dislikes: X" | Different from either alone |
| Sequential Update | Predictions after each new preference | Monotonic improvement |

### 3. Bot-Play Tests (from CSMAI-19)
- QBot asks questions, observes ratings
- Reward = delta in prediction error (log loss)
- Track loss improvement over dialogue turns
- Compare trained policy vs random policy

---

## Dataset Requirements

### Paper Dataset (MovieLens + Attributes)
From the IEEE paper:
```
- 100 most frequently rated movies
- 37,139 users with >= 25% ratings available
- Enriched with attributes:
  - Genres (from MovieLens)
  - Top 50 actors (by popularity score)
  - Top 50 directors (by popularity score)
  - Plot keywords (common keywords from TMDB)
```

### Quick Iteration Dataset for CASPER
```
- Top 100-500 most-rated movies (start small!)
- Users with >= 25 ratings (enough signal to learn from)
- Enriched with SBERT-encoded concepts for each movie
- Train/Val split: 80/20 by user
```

### Data Format (from CSMAI-19 notebook)
```python
class CustomDataset(Dataset):
    def __getitem__(self, index):
        user_data = self.get_user_ratings(index)

        # Shuffle order (attention will learn to handle this)
        index_input = random_permutation(user_data.movie_indices)
        rating_input = one_hot_encode(user_data.ratings)  # [liked, disliked, not_seen]

        # Output: predictions for ALL movies (masked for unrated)
        explicit_output = user_data.full_ratings  # what they rated
        implicit_output = user_data.seen_mask     # whether they've seen it

        return index_input, rating_input, (explicit_output, implicit_output)
```

### Attribute Ratings (Key Innovation from Paper)
The paper creates **synthetic attribute ratings** from movie ratings:
```python
# If user likes Gladiator and Braveheart (both Action)
# → Derive: user likes "Action" genre
# This allows QBot to ask about attributes, not just movies!
```

---

## Hypothesis Table

| # | Hypothesis | Approach | Status | Result |
|---|-----------|----------|--------|--------|
| 1 | Text concat loses like/dislike distinction | Current approach | Tested | 6.8/10 overlap (68%) - **fails** |
| 2 | Signed encoding (likes - dislikes) works | Subtraction | Tested | 0.28 NDCG, sanity fails |
| 3 | RNN+Attention preserves like/dislike | CSMAI-19 approach | Tested | NDCG 0.43, but **100% overlap** - training doesn't teach like/dislike contrast |
| 4 | SBERT concepts work with RNN | Replace movie embeddings | Tested | Works for encoding, but needs contrastive training |
| 5 | Attention pooling beats mean pooling | Compare aggregation | **TODO** | - |
| 6 | Contrastive training teaches like/dislike | Explicit pos/neg pairs | **TESTED** | 98% overlap - **FAILED** (embedding collapse) |
| 7 | Direct rating concat (stronger signal) | Concat rating directly, no cross-attn | **TODO** | - |
| 8 | Reproduce paper exactly (learned embeds) | Skip SBERT, use paper's approach | **TODO** | - |
| 9 | Rating-conditioned item embeddings | Different item tower per rating | **TODO** | - |

---

## Baseline Results (2025-01-18)

**Test configuration**: 500 random users, 15 epochs, 16 negatives per positive

| Metric | Result | Target | Status |
|--------|--------|--------|--------|
| NDCG@10 | **0.3886** | >= 0.30 | ✓ PASS |
| Likes vs Dislikes overlap | **68%** | < 30% | ✗ FAIL |
| User embedding std | **0.3539** | > 0.15 | ✓ PASS |

**Per-concept overlap breakdown**:
- Science Fiction: 70%
- Action: 60%
- Romance: 70%
- Horror: 70%
- The Matrix (1999): 70%
- Titanic (1997): 70%

**Conclusion**: Model achieves good NDCG but cannot distinguish likes from dislikes. This confirms the need for explicit one-hot rating encoding via LSTM+Attention architecture.

---

## LSTM+Attention Experiment Results (2025-01-18)

### Initial LSTM+Attention (Simple Training)
**Script**: `scripts/test_lstm_attention_recommender.py`

| Metric | Result | Target | Status |
|--------|--------|--------|--------|
| NDCG@10 | **0.4276** | >= 0.30 | ✓ PASS |
| Likes vs Dislikes overlap | **100%** | < 30% | ✗ FAIL |
| User embedding std | **0.0003** | > 0.15 | ✗ FAIL (collapse!) |

**Problem identified**: The standard InfoNCE training only uses "liked" examples from parsed text. The model never learns what "disliked" means because:
1. Training positives come from `liked_ids`
2. Negatives are random movies, not explicitly "disliked" items
3. The rating one-hot is always [1,0,0] (liked) during training

**Result**: Severe embedding collapse - model outputs same embedding regardless of input.

### Contrastive Training Approach (V2)
**Script**: `scripts/test_lstm_attention_recommender_v2.py`

**Key insight**: Must explicitly train with contrastive pairs:
- `(concept, "liked")` → target movie should rank HIGH
- `(concept, "disliked")` → same target movie should rank LOW

This teaches the model that the rating dimension (liked vs disliked) should flip the ranking.

**Training Results (2025-01-18)**:
- 200 users, 2862 contrastive examples, 20 epochs
- Training contrast accuracy: ~59% (improved from 18% to 59% during training)
- Loss: 1.38 → 0.86 (decreasing)

| Metric | Result | Target | Status |
|--------|--------|--------|--------|
| Likes vs Dislikes overlap | **98%** | < 30% | ✗ FAIL (worse than baseline!) |
| User embedding std | **0.0006** | > 0.15 | ✗ FAIL (severe collapse) |

**Per-concept overlap** (testing on held-out concepts):
- Science Fiction: 10/10 overlap (100%)
- Action: 9/10 overlap (90%)
- Romance: 10/10 overlap (100%)
- Horror: 10/10 overlap (100%)
- The Matrix (1999): 10/10 overlap (100%)
- Titanic (1997): 10/10 overlap (100%)

**Cosine similarities** (should be low for liked vs disliked):
- Science Fiction liked vs disliked: **1.0000** (identical!)
- Romance liked vs disliked: **0.9999** (identical!)

**Diagnosis**: **Embedding collapse during inference**. The model learned the contrastive objective during training (59% accuracy) but the learned representations **did not generalize** to novel concepts. At inference time, the model outputs nearly identical embeddings regardless of the rating input.

**Why it failed**:
1. Training used movie embeddings as concept embeddings (reused same vector)
2. Model may have memorized training examples rather than learning general principle
3. The rating one-hot (3 dims) may be overwhelmed by the concept embedding (384 dims)
4. Cross-attention may not be strong enough to modulate embeddings based on rating

---

## Implementation Plan

### Phase 0: Run Current Baselines
- [x] Verify current two-tower model achieves ~0.28 NDCG → **Achieved 0.3886**
- [x] Run sanity test: likes vs dislikes overlap → **68% overlap (target < 30%)**
- [x] Create simple test script for quick iteration → `scripts/test_recommender_baseline.py`

### Phase 1: Port CSMAI-19 Architecture
- [x] Create `lstm_attention_recommender.py` → `src/casper/models/lstm_attention_recommender.py`
- [x] Implement LSTM + Attention user tower with explicit rating one-hot
- [x] Create test script → `scripts/test_lstm_attention_recommender.py`
- [x] **Finding**: Standard InfoNCE training causes embedding collapse (100% overlap)
- [ ] Try exact paper reproduction with learned embeddings (skip SBERT for now)

### Phase 2: Contrastive Training (COMPLETED - FAILED)
- [x] Created contrastive training script → `scripts/test_lstm_attention_recommender_v2.py`
- [x] Run full contrastive training experiment → **98% overlap (worse than baseline!)**
- [x] **Finding**: Model suffers embedding collapse - outputs identical embeddings for liked vs disliked
- [x] Training accuracy reached 59%, but doesn't generalize to novel concepts

### Phase 2b: Alternative Approaches (TODO)
- [ ] **Hypothesis 7**: Direct rating concatenation with stronger signal weighting
- [ ] **Hypothesis 8**: Reproduce paper exactly with learned embeddings (skip SBERT)
- [ ] **Hypothesis 9**: Rating-conditioned item embeddings (separate tower per rating)

### Phase 3: Sanity Tests
- [x] Implement likes vs dislikes overlap test
- [ ] Implement sequential update test (loss should decrease with each input)
- [ ] Implement attribute manipulation test
- [ ] **Target**: < 30% overlap for opposite preferences

### Phase 4: Integration with CASPER
- [ ] Replace two-tower user encoder with LSTM+Attention encoder
- [ ] Verify RL reward signal (ΔLoss) is meaningful
- [ ] Train policy network via bot-play
- [ ] Compare with random policy baseline

---

## Model Architecture (Proposed)

```python
class RNNAttentionUserTower(nn.Module):
    def __init__(self, concept_dim=384, hidden_dim=256, output_dim=128):
        self.lstm = nn.LSTM(concept_dim + 3, hidden_dim, batch_first=True)  # +3 for rating one-hot
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=4)
        self.output = nn.Linear(hidden_dim, output_dim)

    def forward(self, concept_embeddings, rating_one_hots):
        # concept_embeddings: [batch, seq_len, 384] - SBERT embeddings
        # rating_one_hots: [batch, seq_len, 3] - [liked, disliked, unknown]

        x = torch.cat([concept_embeddings, rating_one_hots], dim=-1)
        lstm_out, _ = self.lstm(x)

        # Self-attention over LSTM outputs
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)

        # Pool to single user embedding
        user_emb = attn_out.mean(dim=1)  # or use last hidden state
        return self.output(user_emb)
```

---

## Key Insight from IEEE Paper / CSMAI-19

The fundamental difference is **how preferences are represented**:

**Current CASPER approach (BROKEN)**:
- All preferences collapsed into single text → SBERT encodes as single vector
- Like/dislike distinction relies on SBERT understanding "likes:" vs "dislikes:" prefixes
- **Result**: 80% overlap between opposite preferences - useless for RL!

**IEEE Paper / CSMAI-19 approach (WORKS)**:
- Each preference is a separate (item, rating) tuple in a sequence
- Rating is **explicitly** encoded as one-hot [liked, disliked, not_seen]
- LSTM processes sequence, attention captures item correlations
- Model **cannot** confuse likes and dislikes because they have different one-hot codes
- **Result**: Loss decreases monotonically with each new preference (Fig 2)

### Why This Solves the RL Problem
From the paper:
> "A new observation of an item-rating pair reduces the loss most when it is uncorrelated with previous observations and information gain is maximised."

This means:
1. **Reward signal is meaningful**: ΔLoss actually reflects preference learning
2. **Policy can learn**: Asking informative questions gets higher rewards
3. **Cold-start works**: No user ID needed, just preference history

---

## Next Steps

1. **Run current baselines** - verify ~0.28 NDCG, document 80% overlap issue
2. **Port CSMAI-19 architecture** - exact reproduction with learned embeddings
3. **Verify paper results** - loss curves (Fig 2), attribute manipulation (Table I)
4. **Adapt for SBERT** - replace learned embeddings with SBERT concept embeddings
5. **Run sanity tests** - verify < 30% overlap for opposite preferences
6. **Integrate into CASPER** - replace user tower, verify RL reward signal
