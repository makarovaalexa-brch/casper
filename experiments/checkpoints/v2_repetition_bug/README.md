# Training Run v2: Entity Repetition Bug

**Date**: January 2026
**Episodes Run**: ~106
**Status**: Failed - no improvement trend

## Summary

This run attempted to fix the negative reward issues from v1 by implementing proper reward shaping. While rewards became positive, the RL agent got stuck in repetitive loops, asking about the same entities repeatedly.

## Changes from v1

1. **Reward shaping**: `reward = ndcg_weight * NDCG_delta + new_pref_bonus * (new_liked + new_disliked) - not_seen_penalty * new_not_seen`
2. **Baseline = 0**: First turn improvements are rewarded (not penalized vs high LLM baseline)
3. **User simulator**: Forced tool calls, natural language (no rating mentions)
4. **Question generator**: Single entity questions only

## Observed Problem

The RL agent kept asking about the same entities repeatedly:

### Cross-episode repetition
```
Episode 101: "Children of the Corn"
Episode 102: "Children of the Corn"
Episode 103: "Children of the Corn"
Episode 104: "Children of the Corn"
Episode 105: "Children of the Corn"
```

### Within-episode repetition
```
Episode X:
  Turn 2: "based on a comic"
  Turn 3: "based on a comic"
  Turn 4: "based on a comic"
  Turn 5: "based on a comic"
```

## Root Cause Analysis

The LSTM+Attention state encoder only encodes **discovered preferences** (liked/disliked/not_seen movies). It has **no memory of what entities were asked about**.

State encoder input:
- Preference embeddings (liked movies, disliked movies, not_seen)
- Attention over preference sequence

Missing from state:
- What questions were asked
- What entities have been explored

This means:
1. If user says "haven't seen Children of the Corn", it gets added to not_seen
2. Next turn, state encoder sees the same not_seen list
3. RL predicts same embedding → same nearest entity → same question
4. Loop continues until user happens to mention something new

## Fix Applied

Added entity tracking in `casper_agent.py`:

```python
def reset_conversation(self):
    self.conversation_history = []
    self.discovered_preferences = {}
    self.preference_sequence = []
    self.embedding_history = []
    self.asked_entities = set()  # NEW: Track asked entities

def ask_question(self, ...):
    # Filter out already-asked entities
    if not hasattr(self, 'asked_entities'):
        self.asked_entities = set()

    filtered_entities = [
        (entity, score) for entity, score in nearest_entities
        if entity.lower() not in self.asked_entities
    ]

    if filtered_entities:
        top_entity = filtered_entities[0][0]
    else:
        top_entity = nearest_entities[0][0] if nearest_entities else None

    if top_entity:
        self.asked_entities.add(top_entity.lower())
```

## Artifacts

**Note**: Training artifacts (checkpoints, CSV, JSONL) were cleared before archiving. This README documents the run for reference.

## Next Steps

Run v3 with entity tracking fix to prevent repetitive questions.
