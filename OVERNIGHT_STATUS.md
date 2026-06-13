# Overnight Adaptivity-Headroom Report

Generated 2026-06-13 23:35 (70s).

Tier-0 model-free diagnostic: HEADROOM@10 = (per-user adaptive liked-coverage) - (best fixed-set coverage) in 10 questions. Large => adaptivity can pay; train learned policies. Small => static near-optimal.

| Dataset | users | items | answer-rate | pair-Jaccard | static@10 | adaptive@10 | HEADROOM@10 | verdict |
|---|---|---|---|---|---|---|---|---|
| movielens_slate1 | 3000 | 187 | 0.38 | 0.373 | 0.148 | 0.317 | **0.169** | MODERATE headroom |
| movielens_slate2 | 3000 | 379 | 0.33 | 0.419 | 0.070 | 0.311 | **0.241** | MODERATE headroom |
| synthetic_indicator | 3000 | 97 | 0.36 | 0.283 | 0.142 | 0.640 | **0.498** | STRONG adaptivity headroom |
| lastfm | 1715 | 300 | 0.07 | 0.055 | 0.149 | 0.834 | **0.685** | STRONG adaptivity headroom |
| yelp | — | — | — | — | — | — | — | SKIPPED: Yelp Open Dataset requires a signup form |

## Validation check (should retrodict known results)
- synthetic_indicator: expect LARGE headroom (disjoint user groups).
- movielens_slate1: expect SMALL headroom (everyone likes the hits).
- movielens_slate2: expect small-moderate (mid-pop, more varied).
If these hold, the diagnostic is trustworthy for new datasets.

## Recommendation
Train Tier-2 learned policies on any NEW dataset with HEADROOM@10 materially above movielens_slate2's; that is the dataset most likely to demonstrate learned adaptive > static.
## LastFM Tier-1 results (real-data adaptivity demonstrator)
instrument accepted=True (rho=1.00, overlap=0.42); prior=0.569

| policy | final | AUAC | answer | adaptive |
|---|---|---|---|---|
| random | 0.5749 | 0.5710 | 17% | True |
| popularity | 0.6016 | 0.5860 | 94% | False |
| greedy_infogain | 0.6089 | 0.5896 | 66% | True |
| scpr_entropy | 0.6019 | 0.5880 | 87% | True |
| thompson | 0.5964 | 0.5807 | 43% | True |

Adaptive heuristic best AUAC=0.5896 vs popularity 0.5860 (gap +0.0036). Tier-2 (PPO/bot-play) is the recommended next step.
## LastFM diagnosis (IMPORTANT — read before trusting v1 numbers)
v1 used a median-split target (liked = plays >= own median). Result:
oracle-ceiling (all reveals) = 0.637, prior 0.570 => dynamic range only
0.068. The instrument is WEAK because the median-split target is near-
noise; the benchmark (greedy 0.590 vs popularity 0.586 AUAC) is therefore
INCONCLUSIVE, NOT a real "adaptivity doesn't help" finding — it is the
weak-instrument compression failure mode (cf. synthetic ceiling gate).
Tier-0 headroom (0.685, Jaccard 0.055) shows the DATA has strong routing
structure; the median target just can't express it.

FIX IN PROGRESS: tercile target (liked = top-third plays = genuine
favourites, disliked = bottom-third, middle = unknown), mirroring the
movie liked=rating>=4 design, plus an oracle-ceiling acceptance gate
(>0.72). If the tercile instrument clears the ceiling gate, its benchmark
is the real LastFM adaptivity test; results land in
benchmark_lastfm_tercile.json.

METHODOLOGICAL NOTE for the paper: the oracle-ceiling gate must be a
STANDARD acceptance criterion — v1 passed monotonicity + overlap yet was
too weak to benchmark. Instrument acceptance needs: polarity + reveal-
monotonicity + liked/dislike overlap + ANSWERABILITY (dual head) +
ORACLE CEILING.
