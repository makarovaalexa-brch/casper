# V1: Baseline Reward Training Run

**Date:** January 20, 2025
**Episodes:** 95 (interrupted)

## Configuration

- **Reward Signal:** Pure NDCG delta (no shaping)
  - `reward = NDCG(t) - NDCG(t-1)`
  - Baseline NDCG calculated from LLM with "No preferences yet"
  - No bonus for new preferences, no penalty for "not seen"

- **Question Generation:**
  - Used top 3 entities from RL prediction
  - No preference for broad vs specific entities
  - Often generated double-barreled questions

- **User Simulator:**
  - Did not always use tools for genre questions
  - Sometimes hallucinated preferences without checking ratings

- **State Encoder:** LSTM + Attention

## Issues Identified

1. **Negative rewards:** Baseline NDCG was high (~0.6-0.8), revealing preferences often DROPPED NDCG
2. **Hallucinated responses:** User simulator answered genre questions without tool calls
3. **Double-barreled questions:** "Have you seen X? What about Y?"
4. **Too many "not seen" responses:** Specific movie questions often yielded no info
5. **Meaningless embedding samples in logs:** First 5 dims of 384-dim vector

## Results

- Mean reward (last 10): -0.25 (negative due to baseline issue)
- Mean NDCG (last 10): 0.36
- No clear learning signal observed

## Files

- `rl_episode_*.pt` - Checkpoints every 10 episodes
- `rl_interrupted_95.pt` - Final checkpoint before stop
- `training_progress.csv` - Episode-by-episode metrics
- `training_detailed.jsonl` - Full conversation logs
- `training_conversations.jsonl` - Episode summaries
