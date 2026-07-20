# V3: Similarity Penalty for Exploration

**Starting from:** Pretrained LSTM actor (actor_pretrained_lstm.pt) - compatible, no architecture change
**Previous runs:** V1 (negative rewards), V2 (repetition bug)
**Status:** FAILED - see `v3_similarity_penalty/README.md`

## Outcome

Run failed after ~90 episodes. Model converged to a fixed policy, asking the same questions ("archaeology", "unusual plot structure") to every user regardless of their preferences.

**Root cause:** Similarity penalty only prevents repetition WITHIN episodes, not ACROSS episodes. Model found local optimum and exploited it.

## The Problem (V2 Failure)

V2 failed because the RL agent got stuck in repetitive loops, asking about the same entities repeatedly:
- "Children of the Corn" asked 5 times across consecutive episodes
- "based on a comic" asked 4 times within a single episode

Root cause: The model predicted embeddings in the same region of space, leading to the same entities being selected.

## The Solution: Similarity Penalty

Instead of changing the state encoder architecture, we add a **reward penalty** when the model's predicted embedding is too similar to things it already knows about.

**Key insight**: The information is already in the state (preferences discovered). The model just needs to be guided to USE this information by penalizing predictions that explore already-known regions.

### How It Works

```python
# After model predicts embedding
predicted_emb = actor(state)

# Get embeddings of everything already discovered
known_items = liked + disliked + not_seen
known_embs = encode(known_items)

# Find max similarity to any known item
max_sim = max(cosine_sim(predicted_emb, k) for k in known_embs)

# Penalize if too similar
if max_sim > threshold:
    reward -= penalty_weight * max_sim
```

### Reward Formula (V3)

```
reward = ndcg_weight * NDCG_delta
       + new_pref_bonus * (new_liked + new_disliked)
       - not_seen_penalty * new_not_seen
       - similarity_penalty * max_similarity    <-- NEW
```

Where:
- `similarity_penalty` only applies if `max_similarity > threshold`
- Default: `threshold=0.7`, `penalty_weight=0.1`

## Why This Works

1. **No architecture change** - Uses existing state encoder, compatible with pretrained weights
2. **Direct signal** - Model learns: "predicting in same region → negative reward"
3. **Exploration pressure** - Forces model to spread out in embedding space
4. **Natural learning** - Model discovers that novel regions lead to better rewards

## Files Changed

### `episode_runner.py`
- Added `similarity_threshold` and `similarity_penalty` parameters
- Added `_calculate_similarity_penalty()` helper method
- Added similarity penalty to reward calculation
- Added `similarity_penalty` and `max_similarity` to turn logs

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `similarity_threshold` | 0.7 | Only penalize if similarity > this |
| `similarity_penalty` | 0.1 | Penalty = this * max_similarity |

### Tuning Guidelines

- **Higher threshold (0.8)**: Only penalize very similar predictions
- **Lower threshold (0.5)**: Penalize moderately similar predictions
- **Higher penalty (0.2)**: Stronger push for exploration
- **Lower penalty (0.05)**: Gentler exploration pressure

## Comparison with Previous Approaches

### V2 Attempt: Action-Outcome Encoding (Reverted)
- Tried adding "asked" type to state encoder (4-dim instead of 3-dim)
- Problem: Redundant - outcomes often ARE the things we asked about
- Required architecture change, incompatible with pretrained weights

### V3: Similarity Penalty (Implemented)
- No architecture change
- Works with existing pretrained model
- Direct reward signal for exploration
- Simple and interpretable

## Debugging

Check turn logs for similarity penalty:

```json
{
  "turn": 3,
  "reward": {
    "ndcg_delta": 0.05,
    "pref_bonus": 0.05,
    "not_seen_penalty": 0.0,
    "similarity_penalty": 0.08,   // <-- V3: Penalty applied
    "max_similarity": 0.82,       // <-- V3: Too similar to known
    "total": 0.02
  }
}
```

If `max_similarity` is consistently high and `similarity_penalty` is being applied, the model is being pushed to explore new regions.

## Expected Behavior

With similarity penalty, the model should:

1. **Avoid repetition naturally** - Predicting near known items → penalty
2. **Explore diverse concepts** - Spread across embedding space
3. **Learn faster** - Clear signal for what NOT to do
4. **No external filter needed** - Can eventually remove `asked_entities` set

## Backup Filter

The `asked_entities` set in `casper_agent.py` is kept as a backup safety net. Once training shows the model naturally avoids repetition via the similarity penalty, this filter can be removed.
