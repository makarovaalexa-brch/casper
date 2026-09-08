# A9 — opener-state arms on the reserved TEST cohort (DS-A9, 2026-09-08)

**Status: COMPLETE.** Design `docs/design/DS_A9_OPENER_ARMS_ON_TEST.md`. Code `src/instrument/m6d_eval.py`
(`--static_train`, commit `272cf98`, committed before launch). Artifacts
`experiments/battery/m6/m6generalise_B_test.json` (rewritten with the new keys; the 29 Aug file kept as
`m6generalise_B_test_2026-08-29.json`), `m6generalise_B_test_peruser.npz` (150 per-user arrays, so no further
reading of test is needed), log `m6generalise_B_test_dsa9.out`. Run time 2.5 min. Conclusions Fable's own.

**Gate: PASSED.** Every policy, tree and cold mean at K=1, 4, 15 in all six groups reproduces the 29 August test
file to 0 (max |diff| = 0). The extra arm and contrasts were read only after that.

**One-line verdict: on TEST, pooled over the 3,877 users who volunteer something, the order static < cold < opener
holds at K=15 and the opener-fitted tree beats the cold tree by +0.01041 [+0.00583, +0.01520] full NDCG@10; the
validation figure was +0.00688. Per group, the ordering holds in four of six and the interval clears zero in the
two largest (arthouse, inspiring), as on validation.**

## Pooled (3,877 test users, one group each, K=15, full NDCG@10)

| arm | full@10 |
|---|---|
| static_train (train-fitted 15-question order) | 0.27721 |
| cold tree (train-fitted) | 0.27993 |
| **opener tree (own group, train-fitted)** | **0.29035** |
| policy (no group label) | 0.28939 |

| contrast | K=1 | K=4 | K=15 |
|---|---|---|---|
| opener − cold | +0.00497 [+0.00180, +0.00816]* | +0.00402 [−0.00011, +0.00823] | **+0.01041 [+0.00583, +0.01520]*** |
| cold − static | 0 (by construction) | +0.00533 [+0.00168, +0.00886]* | +0.00272 [−0.00162, +0.00707] |
| opener − static | +0.00497 [+0.00180, +0.00816]* | +0.00935 [+0.00519, +0.01357]* | **+0.01313 [+0.00831, +0.01794]*** |
| policy − cold | +0.00428 [+0.00132, +0.00718]* | +0.00231 [−0.00145, +0.00617] | +0.00946 [+0.00520, +0.01373]* |
| policy − static | +0.00428* | +0.00764 [+0.00358, +0.01169]* | +0.01218 [+0.00751, +0.01686]* |

`*` = paired percentile bootstrap interval (10,000 resamples, seed 98765) excludes zero.

## Per group, K=15

| group | n | static | cold | opener | opener − cold | ordering |
|---|---|---|---|---|---|---|
| arthouse | 464 | .27251 | .27964 | **.30046** | +.02082 [+.00676, +.03504]* | yes |
| dialogue | 395 | .21750 | .21894 | **.22940** | +.01045 [−.00176, +.02301] | yes |
| marvel | 599 | .40369 | .41933 | **.42534** | +.00601 [−.00515, +.01722] | yes (cold − static +.01564*) |
| gore | 391 | .32950 | .32595 | **.33208** | +.00613 [−.00848, +.02042] | no: static > cold |
| teenagers | 502 | .24170 | .23712 | **.24179** | +.00466 [−.00990, +.01931] | no: static ≈ opener > cold |
| inspiring | 1,526 | .24274 | .24338 | **.25534** | +.01196 [+.00455, +.01930]* | yes |

## Reading
- Validation (`tab:volunteered`) and test agree in direction everywhere and in magnitude pooled (+0.00688 val vs
  +0.01041 test for opener − cold; the test figure is larger, not smaller). The two groups that separated on
  validation (arthouse, inspiring) separate on test.
- cold − static is the weak link: +0.00272 pooled at K=15, interval includes zero. On these opener states the cold
  tree is not reliably better than a fixed order; the opener-fitted tree is. That is the same shape as validation.
- The mix-table numbers (`tab:policy` a) are unchanged; this run adds columns, it does not move anything.

## What changes in the chapter
`tab:volunteered` can now be stated on test with the same three contrasts, and the requests section's protocol
sentence becomes "fitted on train, scored once on test (one look, 29 Aug, re-read for the static arm 8 Sep)".
The chapter is NOT edited here (author's chapter; edit on instruction).

## Declared
Second reading of the same test look: same users, same banked material, one added arm. Nothing was refitted;
no arm was changed after reading test. The 29 Aug validation of the assignment rule against the banked labels
licenses the test assignment, as before.
