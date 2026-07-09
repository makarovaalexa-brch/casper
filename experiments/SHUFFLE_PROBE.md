# 3-USER SHUFFLE REPEATABILITY PROBE

Judge `gpt-5.4-mini-2026-03-17` temp 0.0, split seed 123. Same user, same 250 item questions re-sent 3 times with shuffled question order AND shuffled profile-line order (seeds [1, 2, 3]); content identical, presentation differs (no caching). Reuses scripts/answerer_smallbatch.py judge machinery.

Cost: 9 calls, 43737 prompt + 18099 completion tokens, **$0.0471**.

## User 85673 (low; orig know-well rate 0.611, n_items 250, profile 41/82)

- **(b) know-well RATE**: original 0.972 | repeats 1=0.772, 2=0.112, 3=0.488 -> **spread 0.660**, max|rep-orig| 0.860
- **(a) cell knowledge-label agreement**: repeat-vs-repeat identical 0.501 (kappa 0.166); vs original identical 0.473 (kappa 0.047)
- **(c) stars MAE on know-well cells**: repeat-vs-repeat 0.177 | vs original 0.201 stars

## User 2389 (mid; orig know-well rate 0.725, n_items 250, profile 101/203)

- **(b) know-well RATE**: original 0.908 | repeats 1=0.832, 2=0.840, 3=0.992 -> **spread 0.160**, max|rep-orig| 0.084
- **(a) cell knowledge-label agreement**: repeat-vs-repeat identical 0.856 (kappa 0.240); vs original identical 0.860 (kappa 0.190)
- **(c) stars MAE on know-well cells**: repeat-vs-repeat 0.205 | vs original 0.279 stars

## User 23228 (high; orig know-well rate 0.858, n_items 250, profile 70/141)

- **(b) know-well RATE**: original 0.964 | repeats 1=0.812, 2=0.888, 3=0.348 -> **spread 0.540**, max|rep-orig| 0.616
- **(a) cell knowledge-label agreement**: repeat-vs-repeat identical 0.621 (kappa 0.284); vs original identical 0.708 (kappa 0.182)
- **(c) stars MAE on know-well cells**: repeat-vs-repeat 0.117 | vs original 0.213 stars

## (d) Verdict

| user | tier | rate spread | max|rep-orig| | rep-vs-rep cell agree | vs-orig cell agree | rep MAE | verdict |
|---|---|--:|--:|--:|--:|--:|---|
| 85673 | low | 0.660 | 0.860 | 0.501 | 0.473 | 0.177 | BIG |
| 2389 | mid | 0.160 | 0.084 | 0.856 | 0.860 | 0.205 | BIG |
| 23228 | high | 0.540 | 0.616 | 0.621 | 0.708 | 0.117 | BIG |

**Overall: BIG flutter (a shuffle-order effect is visible).**

Criterion: SMALL = rate spread <= 0.03 AND repeat-vs-repeat cell knowledge agreement > 0.85 for EVERY user; otherwise BIG.