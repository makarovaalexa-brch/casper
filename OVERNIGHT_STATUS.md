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