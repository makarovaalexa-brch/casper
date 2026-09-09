# A10 — the bandits' free constants, chosen on validation (2026-09-09)

**Status: COMPLETE.** Design `docs/design/DS_A10_BANDIT_CONSTANTS_ON_VALIDATION.md`. Code
`src/baselines/adaptive_harness.py` (`--cohort validation` via `arm_n.load_arm_n(cohort=...)`, `--ts_scale`),
committed before launch (`3f160fc`). Artifacts `experiments/baselines/interview/adaptive_table_VAL_a10_*.json`
(grid), `adaptive_table_a10_test_*.json` + `_per_user.npz` (chosen constants on test),
`a10_tuned_vs_static_test.json` (paired intervals). Conclusions Fable's own; runs on Opus's watch.

**Why it exists.** A simulated ECIR reviewer's principal objection to the draft: the static comparator is a
greedy search on held-out NDCG@10 on the scoring recommender, while every ported adaptive arm runs with its
published constants. The comparison is asymmetric, and the paper could not say how much of the gap that
asymmetry explains. Now it can.

**One-line verdict: the objection has substance and does not change the finding. The published constant is
not the best for either bandit, but choosing the best one on validation buys GLM-UCB +0.00578 and Thompson
+0.00180 full NDCG@10, against a gap to the static order of 0.037–0.044 on the same cohort. Tuned and run on
test, both remain far below the static order, every interval separating.**

## The grid (validation, K=16, consumer = certified tower, full NDCG@10)

| selector | constant | value | full | tail | answers/16 |
|---|---|---|---|---|---|
| GLM-UCB | alpha | 0.25 | 0.21202 | 0.08405 | 5.33 |
| | | 0.5 | 0.21355 | 0.08444 | 5.31 |
| | | **1.0 (published)** | 0.21281 | 0.08411 | 5.24 |
| | | 2.0 | 0.21472 | 0.08194 | 5.08 |
| | | **4.0 (chosen)** | **0.21859** | 0.08138 | 4.46 |
| | | 8.0 | 0.20322 | 0.07380 | 3.35 |
| | | 16.0 | 0.20559 | 0.07561 | 2.91 |
| Thompson | posterior scale | 0.1 | 0.21015 | 0.08495 | 5.34 |
| | | 0.25 | 0.21153 | 0.08519 | 5.27 |
| | | **0.5 (chosen)** | **0.21171** | 0.08817 | 4.79 |
| | | **1.0 (published)** | 0.20991 | 0.08494 | 3.37 |
| | | 2.0 | 0.20251 | 0.08175 | 2.47 |
| ConUCB | lambda | 0.1 | 0.18225 | 0.07092 | 1.68 |
| | | **0.5 (published, chosen)** | **0.18247** | 0.07017 | 1.77 |

**Both bandits have an INTERIOR optimum**, which is what makes the choice defensible: GLM-UCB rises to
alpha 4.0 and falls at 8.0 and 16.0; Thompson peaks at scale 0.5 and falls either side. The grids were
pre-registered as {0.25 … 4.0} and {0.5, 1.0, 2.0} and EXTENDED after both proved monotone at their edge.
The extension direction strengthens a baseline, i.e. works against this paper's claim, which is the
conservative direction to extend; it is recorded as an extension, not presented as the original grid.

⚠ **ConUCB's grid is incomplete: the lambda = 2.0 point failed (exit 1) and was not re-run.** The published
value already wins over 0.1, so the arm's conclusion is unaffected, but the grid is two points, not three,
and must be described that way.

## The comparison the grid is read against (same validation cohort, K=16)

| arm | full | gap to static |
|---|---|---|
| combined greedy static order | 0.25561 | — |
| GLM-UCB, best constant (4.0) | 0.21859 | **−0.03702** |
| Thompson, best constant (0.5) | 0.21171 | **−0.04390** |

Tuning recovers roughly an eighth of the distance for GLM-UCB (+0.00578 of 0.0428) and a twentieth for
Thompson. **The asymmetry is real and is not the explanation.**

## The chosen constants on TEST (10,000 users, consumer = certified tower)

| selector | K=1 | K=2 | K=4 | K=8 | K=16 | answers/16 |
|---|---|---|---|---|---|---|
| GLM-UCB alpha=4.0 | .16788/.03756 | .17789/.05204 | .18616/.05382 | .19231/.05682 | **.22338/.08413** | 4.48 |
| GLM-UCB alpha=1.0 (published, banked) | .1606/.0400 | .1635/.0452 | .1756/.0578 | .1902/.0673 | .2161/.0833 | 5.24 |
| Thompson scale=0.5 | .16165/.03868 | .16760/.04712 | .17789/.05762 | .19311/.07063 | **.21582/.08812** | 4.83 |
| Thompson scale=1.0 (published, banked) | .1627/.0376 | .1676/.0437 | .1748/.0551 | .1908/.0677 | .2135/.0878 | 3.39 |

Tuning is worth **+0.0073 full** to GLM-UCB and **+0.0023** to Thompson at sixteen questions on test,
consistent with the validation gain that selected it.

**Against the combined static order on test, paired percentile bootstrap, 10,000 resamples clustered over users:**

| tuned arm | full | tail |
|---|---|---|
| GLM-UCB alpha=4.0 | −0.03562 [−0.03893, −0.03241]* | −0.03957 [−0.04312, −0.03603]* |
| Thompson scale=0.5 | −0.04318 [−0.04665, −0.03974]* | −0.03558 [−0.03916, −0.03192]* |

`*` = interval excludes zero. Against the *items-only* static order (.2447 full), the correct comparator for
item-only selectors, the tuned GLM-UCB deficit is −0.0213 rather than the published arm's −0.0281.

## What goes in the paper
One sentence in "Ports and tuning": the constants were chosen on validation over a grid containing the
published value, both bandits have an interior optimum, and the chosen values are the ones reported.
One sentence in Limitations, replacing the current concession: tuning is worth +0.006 and +0.002 against a
gap of 0.036 and 0.043, so the fitting asymmetry does not account for the result. Table 1 gains the tuned
rows or a footnote carrying them.

## Mechanism worth one clause
More exploration collects FEWER answers (GLM-UCB 5.24 → 4.46 answers as alpha goes 1 → 4) and still scores
higher, up to a point. The arm is trading answer count for answer informativeness, and past alpha = 4 the
trade stops paying. That is consistent with the chapter's answerability finding rather than in tension with it.
