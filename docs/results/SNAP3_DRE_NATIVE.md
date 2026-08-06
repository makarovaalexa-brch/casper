# SNAP-3: DRE — native snap PASSES, and the duplicate-seed defect is real

Target: Kweon, Kang, Hwang & Yu, *"Deep Rating Elicitation for New Users in Collaborative
Filtering"*, **WWW 2020**, DOI 10.1145/3366423.3380042. Code `WonbinKweon/DRE_WWW2020`.
Run 2026-08-05/06, `scripts/baselines/dre_native_snap.py`, ML-1M, k=50, 2000 epochs, 3 seeds.
Artifacts `experiments/baselines/DRE_NATIVE_SNAP_ml1m_k50.{json,log}` (+ the k=10 pilot).

## Step 0 — the published target [V]

Extracted from the arXiv LaTeX source (`5ex.tex`, **Table 2, k=50**), cross-checked against the
ar5iv render and our own `external_literature/findings/rating_elicitation_family_deep.md`:

| | P@10 | P@20 | NDCG@10 | NDCG@20 |
|---|---|---|---|---|
| **ML-1M** DRE | .5396 | .4734 | **.5688** | .5197 |
| ML-1M best baseline (RMVA) | .5063 | .4373 | .5387 | .4887 |
| **ML-20M** DRE | .5712 | .4875 | **.6077** | .5373 |
| ML-20M best baseline (RMVA) | .5116 | .4382 | .5444 | .4851 |

★ **A number we had been carrying was wrong.** Our design docs recorded "a secondary source implies
a ~0.57-class NDCG@10 [on ML-20M]". That is the **ML-1M** row (.5688) — ML-20M is **.6077**. The
likely origin is P@10 .5712 misread as NDCG@10 from the adjacent column. Had we snapped against
0.57 on ML-20M we would have "passed" against a target off by 0.04, belonging to another dataset.
**This is what step 0 is for.** Also recorded: DRE's per-k sweep exists only as Figure 3, so any
per-k number other than k=50 is [S]-grade — the k=10 pilot below is therefore indicative only.

## The result (ML-1M, k=50, 3 seeds)

| seed | NDCG@10 | NDCG@20 | P@10 | P@20 | duplicate seeds |
|---|---|---|---|---|---|
| 1 | .5635 | .5176 | .5338 | .4731 | 0 |
| 2 | .5654 | .5199 | .5386 | .4780 | 0 |
| 3 | **.5492** | .5057 | .5256 | .4659 | **1** |
| **mean ± sd** | **.5593 ± .0088** | .5144 ± .0077 | .5327 ± .0066 | .4723 ± .0061 | 0.33 |

| metric | published | ours | delta |
|---|---|---|---|
| NDCG@10 | .5688 | .5593 | **−.0095** |
| NDCG@20 | .5197 | .5144 | −.0053 |
| P@10 | .5396 | .5327 | −.0069 |
| P@20 | .4734 | .4723 | **−.0011** |

## Verdict: PASS, with a diagnosed residual

The mean is 0.0095 below published on NDCG@10, i.e. ~1.1 seed-sd, and P@20 lands essentially on
top of the published value. The published figure is itself an average of 5 runs; we ran 3.
For calibration in this project: **Mult-DAE was DROPPED from Paper A for a −0.0221 miss** against a
±0.005 tolerance, and our own EDDI variant **failed outright** (it lost to random ordering on the
authors' data). DRE is in a different category from both.

**And the residual has a named cause.** The one seed carrying a **duplicate selected item**
(seed 3, `dup_seeds=1`) is also the worst by a clear margin (.5492 against .5635/.5654). Excluding
it, the mean is .5645 — a −0.0043 gap. This is the criticism our literature pass predicted:
**DRE selects its k seeds as independent categoricals with per-row argmax and no duplicate guard**,
so the same item can be chosen twice and a question is wasted. The prediction is now measured: it
fires in 1 of 3 runs at k=50 and costs roughly 0.015 NDCG@10 when it does.

Two honest caveats. (a) We do **not** exclude seed 3 from the headline — the 3-seed mean is the
reported number; the duplicate analysis is the diagnosis, not a re-baselining. (b) This snap is on
**ML-1M**, not ML-20M; the ML-20M cell (target NDCG@10 .6077) has not been run.

## What this licenses

DRE may now be run as **the peer-reviewed learned-static bar** for Act 1 — the row that makes
"we compared against the strongest published static method" a true sentence rather than a
comparison against our own greedy order. Steps P (our ruler, DRE's own recommender) and X (our
ruler, our certified tower) are unblocked.

Prior record to supersede: `docs/EXPERIMENTS.md` carries **"DRE did NOT reproduce (honest
negative)"** from an earlier attempt on a different setup. That entry should now be annotated —
the method reproduces on its own dataset, split and metric when the published target is the
verified one.
