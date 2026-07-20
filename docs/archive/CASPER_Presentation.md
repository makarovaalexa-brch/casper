# CASPER - Presentation Slides

---

## Slide 1: What is CASPER?

**Continuous Action Space Preference Elicitation via Reinforcement**

A conversational movie recommender that learns *what questions to ask* using RL in continuous semantic space.

**Key Innovation:** Instead of selecting from fixed question templates, the RL agent predicts embeddings that guide natural language question generation.

---

## Slide 2: System Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                              CASPER SYSTEM                              │
├────────────────────┬────────────────────┬───────────────────────────────┤
│                    │                    │                               │
│   PREFERENCE       │      RL POLICY     │        RECOMMENDER            │
│   ELICITATION      │                    │                               │
│                    │                    │                               │
│  ┌──────────────┐  │  ┌──────────────┐  │  ┌─────────────────────────┐  │
│  │ Preference   │  │  │ RL State     │  │  │ Recommender State       │  │
│  │ Extractor    │──┼─→│ Encoder      │  │  │ Encoder                 │  │
│  │ (LLM)        │  │  └──────┬───────┘  │  │ (may differ from RL)    │  │
│  └──────────────┘  │         │          │  └───────────┬─────────────┘  │
│         │          │         ▼          │              │                │
│         │          │  ┌──────────────┐  │              ▼                │
│  extracted prefs   │  │ RL Actor     │  │  ┌─────────────────────────┐  │
│         │          │  │ (DDPG)       │  │  │ Item Ranker             │  │
│         │          │  └──────┬───────┘  │  └───────────┬─────────────┘  │
│         │          │         │          │              │                │
│         │          │    embedding       │          Top-10               │
│         │          │         │          │              │                │
│         │          │         ▼          │              ▼                │
│         │          │  ┌──────────────┐  │  ┌─────────────────────────┐  │
│         │          │  │ FAISS        │  │  │ NDCG Calculation        │  │
│         │          │  │ Lookup       │  │  └───────────┬─────────────┘  │
│         │          │  └──────┬───────┘  │              │                │
│         │          │         │          │         Reward signal         │
│         │          │      concept       │              │                │
│         │          │         │          │              │                │
│         │          │         ▼          │              │                │
│         │          │  ┌──────────────┐  │              │                │
│         │          │  │ Question     │  │              │                │
│         │          │  │ Generator    │◄─┼──────────────┘                │
│         │          │  │ (LLM)        │  │                               │
│         │          │  └──────────────┘  │                               │
│         │          │                    │                               │
└─────────┼──────────┴────────────────────┴───────────────────────────────┘
          │
          ▼
   ┌──────────────┐          ┌──────────────┐
   │   QBot       │ question │   ABot       │
   │   CASPER     │─────────→│   User Sim   │
   │              │←─────────│   (MovieLens)│
   └──────────────┘ response └──────────────┘
```

---

## Slide 3: Component Interaction Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                      CONVERSATION LOOP                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. USER RESPONSE                                               │
│     Question ──> User Simulator ──> "I loved Inception!"        │
│                                                                 │
│  2. PREFERENCE EXTRACTION                                       │
│     Response ──> LLM ──> {liked: [...], disliked: [...]}        │
│                                                                 │
│  3. RL STATE ENCODING                                           │
│     Extracted preferences ──> RL Encoder* ──> 384-dim state     │
│     (* SBERT text OR LSTM+Attention - separate from recommender)│
│                                                                 │
│  4. ACTION PREDICTION                                           │
│     State ──> RL Actor (DDPG) ──> Target embedding (384-dim)    │
│                                                                 │
│  5. ENTITY MAPPING                                              │
│     Target embedding ──> FAISS ──> Nearest concept              │
│     (e.g., "sci-fi", "Tom Hanks", "The Matrix")                 │
│                                                                 │
│  6. QUESTION GENERATION                                         │
│     Concept + History ──> LLM ──> Natural language question     │
│                                                                 │
│  7. RECOMMENDATION & REWARD (separate encoder)                  │
│     Extracted preferences ──> Recommender Encoder** ──> Top-10  │
│     Top-10 vs held-out ratings ──> NDCG ──> Reward to RL        │
│     (** may use different architecture than RL encoder)         │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**Key Design Decision:** RL encoder and Recommender encoder are **separate components** that may have different architectures and are trained independently.

---

## Slide 4: Core Model Architectures

### RL Policy: DDPG (Deep Deterministic Policy Gradient)
Enables RL in **continuous action spaces** - instead of choosing from discrete actions, the agent outputs a 384-dim embedding directly.

- **Actor:** Predicts which concept to ask about (as an embedding)
  - State (384-dim) → Hidden (512) → Action embedding (384-dim)
- **Critic:** Estimates how good that action is (for training)
  - State + Action → Hidden (512) → Q-value (scalar)
- **Target networks:** Slowly-updating copies prevent training instability
- **Exploration:** Gaussian noise added to actions during training

### Recommender: Two-Tower
Two separate neural networks encode users and items into a shared embedding space. Recommendations = nearest items to user.

```
User preferences (text) → SBERT → User Tower → 128-dim user embedding
Movie metadata (text)   → SBERT → Item Tower → 128-dim item embedding
Score = dot_product(user_emb, item_emb)
```
- **Pro:** Fast inference (precompute item embeddings)
- **Con:** SBERT can't distinguish "likes: X" from "dislikes: X" (68% overlap)

### Recommender: LSTM + Attention
Processes preferences as a **sequence** with explicit rating signals. From IEEE paper on preference elicitation.

```
[(concept₁, liked), (concept₂, disliked), ...]
    → SBERT + rating one-hot [1,0,0] or [0,1,0]
    → LSTM processes sequence
    → Attention pools to user embedding
    → Predict scores for all items
```
- **Pro:** Explicit rating encoding - model can't confuse like/dislike
- **Con:** Current implementation suffers embedding collapse (100% overlap)

### Question Generator (LLM)
- Input: selected concept + conversation history
- Output: contextual natural language question

---

## Slide 5: Recommender Results

### Architecture Comparison

| Architecture | NDCG@10 | Like/Dislike Overlap | Status |
|--------------|---------|----------------------|--------|
| Two-Tower (SBERT text) | **0.28** | 68% | Sanity fails |
| LSTM+Attention | 0.43 | 100% | Embedding collapse |
| Paper Exact (learned embeds) | - | - | TODO |
| Target benchmark | 0.35-0.45 | <30% | - |

### Two-Tower Progression
| Configuration | NDCG@10 |
|---------------|---------|
| Baseline (text only) | 0.035 |
| + Genre features | 0.15 |
| + L2 normalization | 0.22 |
| Linear towers | **0.28** |

### Key Finding
- **Like/Dislike Overlap: 68-100%** across all architectures tested
- Root cause: SBERT conflates "likes: X" and "dislikes: X" (0.75-0.78 similarity)
- Needed: explicit rating one-hot encoding [liked, disliked, not_seen]

---

## Slide 6: RL Training Run #1 - Baseline Reward

**Date:** January 20, 2025
**Episodes:** 95 (interrupted)
**Status:** Failed

### Configuration
- Reward = Final NDCG score
- No reward shaping

### Results
| Metric | Value |
|--------|-------|
| Mean Reward (last 10) | **-0.25** |
| Mean NDCG (last 10) | 0.36 |

### Problem
- Baseline NDCG already high (~0.6-0.8) before conversation
- Revealing preferences often *dropped* NDCG
- Agent learned to ask uninformative questions

---

## Slide 7: RL Training Run #2 - Shaped Reward

**Date:** January 2026
**Episodes:** 106
**Status:** Failed

### Configuration
- Reward = NDCG_delta + preference_bonus - not_seen_penalty
- Per-turn reward signal

### Results
- Agent got stuck in repetitive loops
- Same entity asked 5+ times per episode
- Example: "Children of the Corn" asked every turn

### Problem
- State encoder only tracks *what* user likes
- No memory of *what questions were asked*
- Fix applied: added `asked_entities` tracking

---

## Slide 8: RL Training Run #3 - Similarity Penalty

**Date:** January 2026
**Episodes:** 90
**Status:** Failed

### Configuration
- Added similarity penalty for exploring regions similar to known preferences
- Penalty = 0.1 × max_similarity (if similarity > 0.7 threshold)

### Results
| Metric | Early | Late | Trend |
|--------|-------|------|-------|
| Mean Reward | 0.71 | 0.57 | Declining |
| Mean NDCG | 0.50 | 0.44 | Declining |
| Unique Entities | 35% (274/783) | - | Low diversity |

### Problem
- Model converged to **fixed policy** for all users
- Top entities repeated across episodes:
  - "archaeology" (8x), "masterpiece" (22x), "history" (22x)
- No user-adaptive learning emerged

---

## Slide 9: Why RL Training Failed - Root Cause Analysis

```
Problem: Early turns have generic state → same prediction for all users

Turn 1:  State = [empty]           → Predict "masterpiece"    (all users)
Turn 2:  State = [1 preference]    → Predict "history"        (all users)
Turn 3:  State = [2 preferences]   → Predict "archaeology"    (all users)
         ...
         Model learns fixed sequence, not user-adaptive strategy
```

### Contributing Factors
1. **State representation** doesn't distinguish users early in conversation
2. **Exploration** insufficient - exploitation dominates
3. **Reward signal** doesn't penalize user-agnostic behavior

---

## Slide 10: Next Steps

### Short-term Fixes
- [ ] Implement LSTM+Attention encoder with explicit rating one-hot
- [ ] Add user ID embedding to initial state (breaks generalization but tests hypothesis)
- [ ] Increase exploration noise in early training

### Architecture Changes
- [ ] Separate "what to ask about" from "how to ask" (hierarchical RL)
- [ ] Add memory of asked entities to state representation
- [ ] Try curiosity-driven exploration bonus

### Recommender Improvements
- [ ] Train on larger user subset (currently 10K of 162K)
- [ ] Add actor/director features from TMDB
- [ ] Experiment with cross-attention between towers

### Evaluation
- [ ] Human evaluation study (current: simulator only)
- [ ] Compare against pure-LLM baseline (no RL)
- [ ] Ablation study on reward components

---

## Slide 11: Summary

| Component | Status | Performance |
|-----------|--------|-------------|
| Embedding Space (FAISS) | Working | Lookup <10ms |
| Question Generator (LLM) | Working | Coherent questions |
| Preference Extractor (LLM) | Working | ~90% accuracy |
| RL State Encoder | Partial | SBERT works, LSTM untested |
| Recommender (3 architectures) | Partial | Best NDCG 0.28, all fail sanity |
| RL Training | Not Working | Converges to fixed policy |

**Key Insight:** The core pipeline works, but two blockers remain:
1. **Recommender** can't distinguish likes from dislikes (68%+ overlap)
2. **RL policy** learns fixed question sequence, not user-adaptive strategy

**Critical Path:** Fix recommender sanity → Fix RL state encoder → Retrain
