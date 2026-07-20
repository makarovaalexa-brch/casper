# Training Run V3: Similarity Penalty

**Date**: January 2026
**Episodes Run**: ~90
**Status**: Failed - model converged to fixed policy, no NDCG improvement

## Changes from V2

Added **similarity penalty** to reward shaping:
- Compare predicted embedding to all known items (liked + disliked + not_seen)
- If max cosine similarity > 0.7, subtract `0.1 * max_similarity` from reward
- Goal: Encourage exploration of new embedding regions

```python
reward = ndcg_weight * NDCG_delta
       + new_pref_bonus * (new_liked + new_disliked)
       - not_seen_penalty * new_not_seen
       - similarity_penalty * max_similarity  # NEW
```

## What Worked

1. **Within-episode diversity**: Perfect (9/9 unique entities per episode)
2. **No within-episode repetition**: Similarity penalty + asked_entities filter worked
3. **Positive rewards**: Consistently in 0.4-1.3 range (unlike V1)

## What Failed

**Model converged to a FIXED policy** - asks same questions to every user:

### First Entity Selected (Turn 2, last 20 episodes)
```
archaeology:              8 times (40%)
unusual plot structure:   5 times (25%)
others:                   7 times (35%)
```

### Top Entities Overall (783 selections)
```
masterpiece:     32 times
history:         29 times
cartoon:         22 times
dark hero:       22 times
archaeology:     22 times
```

Only 274 unique entities out of 783 selections (35% unique).

## Root Cause Analysis

1. **Similarity penalty only works WITHIN episodes** - doesn't prevent asking "archaeology" across 90 different episodes

2. **Exploration noise decayed too much** - started at 0.2, decayed to 0.14

3. **State is too generic early on** - after Turn 1, state only has 1-2 preferences, so model predicts similar embeddings for all users

4. **Local optimum exploitation** - "archaeology" got positive rewards early, model exploits it forever

## Key Insight

The model is not learning a **user-adaptive** strategy. It's learning a **fixed sequence** of questions:
1. Ask "archaeology" or "unusual plot structure"
2. Ask other broad concepts
3. Collect preferences

This works okay (rewards are positive) but doesn't IMPROVE because:
- Same questions → same types of responses → no new learning signal
- Can't discover that different users need different questioning strategies

## Metrics

| Metric | Early (ep 1-10) | Late (ep 80-90) | Change |
|--------|-----------------|-----------------|--------|
| Mean Reward | 0.71 | 0.57 | -0.14 |
| Mean NDCG | 0.50 | 0.44 | -0.06 |
| Noise Scale | 0.20 | 0.14 | -0.06 |

## Potential Fixes (Not Implemented)

1. **Increase minimum exploration noise** - prevent decay below 0.15-0.2
2. **Cross-episode diversity penalty** - penalize asking same entity across multiple episodes
3. **User-specific initial state** - encode something about the user before Turn 1
4. **Entropy bonus** - add reward for diverse predictions
5. **Periodic noise reset** - reset exploration every N episodes

## Files

- `rl_episode_*.pt` - Model checkpoints every 10 episodes
- `training_progress.csv` - Episode-level metrics
- `training_stats.json` - Summary statistics
- `training_detailed.jsonl` - Full turn-by-turn logs
- `training_conversations.jsonl` - Conversation summaries
