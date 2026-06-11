# CASPER Reward Signal Diagnostic Report

**Date**: 2026-02-09
**Scripts**: `scripts/diagnose_reward_signal.py`, `scripts/test_preference_types.py`

## Background

CASPER training V1-V3 showed broken reward signal: high baseline NDCG (~0.52), NDCG declining as more preferences revealed, policy collapsing to fixed questions. Investigation aimed to identify root causes.

## Test 1: Baseline NDCG by Pool Size and User Filter

**Question**: Why is baseline NDCG so high before any preferences are revealed?

Measured NDCG when LLM receives zero user preferences ("No preferences yet"), varying movie pool size and minimum eval target filter.

| Pool | min_targets=5 | =10   | =15   | =20   | =25 (training default) |
|------|---------------|-------|-------|-------|------------------------|
| 100  | 0.315         | 0.412 | 0.460 | 0.508 | **0.519**              |
| 200  | 0.275         | 0.295 | 0.420 | 0.430 | 0.463                  |
| 500  | 0.253         | 0.286 | 0.295 | 0.377 | 0.417                  |

**n=50 users per cell. Std ~0.20 across all conditions.**

**Finding**: Current training config (pool=100, min_targets=25) starts at 0.52 baseline. This is because users with 25+ liked movies in a pool of 100 means ~39% of the pool is ground truth. LLM's default popular picks score well by chance. Larger pools and lower filters reduce this selection bias.

## Test 2: Simulated Attribute Dialogue (Genres First, Then Movies)

**Question**: Does NDCG grow as we reveal genre attributes step by step?

Revealed user preferences one at a time over 20 steps (liked genres first, then disliked genres, then specific movie likes/dislikes). min_eval_targets=5 for all.

| Pool | n    | Baseline | Final  | Improvement | Monotonic Steps |
|------|------|----------|--------|-------------|-----------------|
| 100  | 30   | 0.289    | 0.263  | **-0.027**  | 7/20            |
| 200  | 30   | 0.268    | 0.282  | +0.014      | 11/20           |
| 500  | 30   | 0.262    | 0.362  | **+0.099**  | 11/20           |

Pool=500 step-by-step (mean NDCG):
```
Step 0: 0.262  Step 5: 0.316  Step 10: 0.378  Step 14: 0.410 (peak)  Step 20: 0.362
```

**Finding**: Attribute-based preferences only produce positive NDCG growth at pool=500. At pool=100, NDCG peaks around step 5 then declines below baseline.

## Test 3: Preference Type Comparison (Pool=100 Only)

**Question**: Is the problem attributes vs movie titles, or pool size?

Tested 3 preference types on pool=100, 20 steps, same 30 users across all conditions:

- **A: Movie Titles Only** - reveal liked/disliked movie names (like the RS_Model_Comparison notebook)
- **B: Attributes Only** - reveal liked/disliked genres (like CASPER training)
- **C: Mixed** - interleave genres and movie titles

| Condition                  | Baseline | Peak         | Final  | Improvement | Monotonic |
|----------------------------|----------|--------------|--------|-------------|-----------|
| A: Movie Titles Only       | 0.323    | 0.394 (s3)   | 0.215  | **-0.109**  | 6/20      |
| B: Attributes Only         | 0.323    | 0.429 (s8)   | 0.348  | +0.025      | 11/20     |
| C: Mixed (Genres + Titles) | 0.323    | 0.451 (s5)   | 0.425  | **+0.102**  | 11/20     |

**n=30 users per condition. Std ~0.20, standard error ~0.037, 95% CI ~+-0.074.**

**Findings**:
- Movie titles alone are worst - they overconstrain the LLM, causing NDCG to crash after step 3
- Mixed preferences (genres + titles interleaved) perform best at +0.102
- The RS_Model_Comparison notebook's LLM test (titles only, t=1 to t=10) was misleading because it stopped before the decline
- The notebook also used only first 50 of 100 movies in the prompt, different user filtering (>=10 ratings), and n=50 users

## Comparison to Notebook Results

The notebook's cached LLM results (pool=100, titles only, n=50):

| Timestep | NDCG  |
|----------|-------|
| 1        | 0.352 |
| 3        | 0.382 |
| 5        | 0.426 |
| 10       | 0.423 |

This showed apparent growth but: (1) only went to t=10, (2) used first 50 movies as candidates not all 100, (3) different user pool and selection criteria.

## Statistical Concerns

All tests suffer from high per-user variance (std ~0.20) relative to the effects being measured. At n=30:
- Standard error of mean: ~0.037
- 95% CI: ~+-0.074
- Only the title decline (-0.109) and mixed improvement (+0.102) are marginally outside noise
- Attribute-only improvement (+0.025) is well within noise

**For RL training specifically**: The agent sees individual episodes, not 30-user averages. Per-turn NDCG deltas of ~0.005 avg are buried in per-user noise of ~0.20. Signal-to-noise ratio for per-step RL rewards is very poor.

## Conclusions

1. **Pool=100 + min_targets=25 is broken**: Baseline NDCG of 0.52 leaves no headroom for improvement
2. **Pool size matters more than preference type**: Pool=500 with attributes (+0.099) works; pool=100 with attributes doesn't
3. **Mixed preferences work best at pool=100** (+0.102), matching pool=500 attribute-only performance
4. **The LLM reward signal is very noisy**: Std ~0.20 per user, per-step deltas ~0.005, making per-turn RL reward deltas effectively unlearnable at the individual episode level
5. **The notebook LLM test was misleading**: Stopped at t=10 before the decline became visible, used restricted candidate pool

## Files

- `casper/scripts/diagnose_reward_signal.py` - Pool size and attribute dialogue tests
- `casper/scripts/test_preference_types.py` - Preference type comparison (A/B/C)
- `casper/experiments/diagnostics/reward_signal_diagnostic.json` - Raw results (Test 1 & 2)
- `casper/experiments/diagnostics/preference_type_comparison.json` - Raw results (Test 3)
