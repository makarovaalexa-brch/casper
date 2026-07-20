# V2: Shaped Reward Training Run

**Starting from:** Pretrained LSTM actor (actor_pretrained_lstm.pt)
**Replay buffer:** Cleared (incompatible with new reward signal)
**Status:** FAILED - see `v2_repetition_bug/README.md`

## Outcome

Run failed after ~106 episodes. RL agent got stuck in repetitive loops, asking about the same entities (e.g., "Children of the Corn" 5 consecutive episodes, "based on a comic" 4 times in single episode).

**Root cause:** State encoder only tracks discovered preferences, not asked entities. Fixed by adding `asked_entities` tracking in casper_agent.py.

## Fixes Applied

### 1. Reward Shaping (baseline=0, bonuses, penalties)
```
reward = ndcg_weight * NDCG_delta
       + new_pref_bonus * (new_liked + new_disliked)
       - not_seen_penalty * new_not_seen
```
- `use_baseline=False`: Start NDCG from 0, so improvement is always positive
- `new_pref_bonus=0.05`: Reward for each useful preference discovered
- `not_seen_penalty=0.05`: Penalty for wasted "haven't seen" questions

### 2. User Simulator (forced tool usage)
- Updated prompt to REQUIRE tool calls before responding
- Must use `query_rating`, `search_ratings`, or `get_all_ratings`
- No more hallucinated genre preferences

### 3. Question Generation (single entity, prefer broad)
- Select only TOP 1 entity (not 3) for clearer questions
- `find_nearest_entities_prefer_broad()` boosts genome tags
- Prompt enforces single question, max 15 words
- No more double-barreled questions

### 4. Improved Logging
- Logical flow: RL Decision -> Agent Question -> User Response -> Preferences -> Reward
- Shows `selected_entity` and `is_broad` flag
- Tracks `new_this_turn` preferences (not just cumulative)
- Reward breakdown shows each component

## Expected Improvements

1. **Positive rewards** when preferences are discovered
2. **Cleaner questions** about single concepts
3. **Grounded responses** from user simulator
4. **Better learning signal** for RL

## Comparison with V1

| Aspect | V1 | V2 |
|--------|----|----|
| Baseline NDCG | LLM baseline (~0.6-0.8) | 0 (start fresh) |
| Pref bonus | None | +0.05 per liked/disliked |
| Not seen penalty | None | -0.05 per not_seen |
| Entities per question | 3 (often combined) | 1 (single focus) |
| Entity preference | None | Broad concepts boosted |
| User tool usage | Optional | Required |
