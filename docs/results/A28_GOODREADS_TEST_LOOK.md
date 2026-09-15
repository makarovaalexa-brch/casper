# A28 — Goodreads, the test look: both of Paper B's levers transfer (2026-09-14)

The single authorised test look on the Goodreads cohort. Certified pair: tower `t2i26g_ep53`, concept
net `cvg_ep10`. 10,000 canonical test users, policy-B pool throughout, learned decoder bias, full
catalogue, no floor. Nothing was refitted or reselected for this look.

**Goodreads has no policy-A problem.** `experiments/battery_goodreads/greedy_static_i26c.json` carries
`pool_policy: B`, and the three separately rebuilt channel orders reproduce its `items-only`,
`concepts-only` and `combined` arms **bit-for-bit** — a free contract check that passed. So the
ML-25M two-stage dance (`ndcg_tree_policyB_test.py`) is not needed here: the tree's comparator is
already the policy-B combined order, and `channel_orders_on_test.py` reads its comparator payload
from the test look's own `cmp_*` bank. The banked assert passed at all five budgets on both axes.

## The two headline numbers, on test

| | ML-25M | Goodreads |
|---|---|---|
| ask nothing (m=0) | .1586 / .0282 | **.0940 / .0058** |
| greedy static combined, K=16 | .2590 / .1237 | **.1713 / .0323** |
| CASPER (tree), K=16 | .2694 / .1307 | **.1823 / .0429** |
| **RQ2 — CASPER − static** | +.01035\* / +.00705\* | **+.01100\* [+.00844,+.01357] / +.01059\* [+.00837,+.01285]** |
| **RQ1 — items → items+concepts** | +.01430\* / +.02199\* | **+.00610\* [+.00373,+.00842] / +.01105\* [+.00936,+.01274]** |

CASPER separates from **K=4 onward** on both axes (K=4 +.00536/+.00432, K=8 +.01072/+.00806). K=1 is
exactly 0.0 by construction and K=2 is a null both ways, as on ML-25M.

Single-channel orders at K=16: items .16522/.02121, concepts .15092/.04287, combined .17132/.03226.

## What differs between the domains

- **Validation overstates, again, and by more.** Validation said +0.03168 full; test says +0.01100.
  ML-25M's own validation figure was +0.02794 against +0.01035 on test. Same direction, same lesson.
- **The channel mix inverts.** The joint order gives concepts **5 of 16** slots on Goodreads against
  **10 of 16** on ML-25M, and sixteen concept questions obtain **9.46** answers against **12.36**.
- **Concepts still win the tail on both.** `concepts − items` at K=16 is +0.02166 tail on Goodreads
  and +0.01436 on ML-25M. ⚠ An earlier draft sentence claiming "items carry the tail on films" was
  **wrong** — it came from the A26 validation *arm* table (items-pop vs conc-member-mass), which is a
  different construction from the greedy orders. Corrected in the paper before it shipped.
- **The bank result shrinks to roughly half** on both axes; the criterion result transfers intact.

## Harness faults fixed to get here

The Goodreads tree is depth 11 with 116 leaves against ML-25M's depth 5 and 15, and every fault was a
consequence of that shape:

1. **Unseen answer patterns.** 8 of 10,000 test users follow a branch no validation user took, so no
   leaf exists for them. Author ruling: they receive the **static comparator order**, unconditioned.
   Asserted below 1% of the cohort and reported.
2. That fallback leaf has no validation cohort, so every consumer of `val_gain_at16` had to exclude
   it or carry NaN — patched at all of them in one pass after finding them one crash at a time.
3. `ndcg_tree_test16.py` read `Ev["pool_contract"]` after `del Ev`; now captured before the release.
4. `channel_orders_on_test.py` enumerated banked leaves from any key with two pipes, which picked up
   leaves that received **no** test users; now keyed off `test|<nid>|rows`.
5. The tree was grown at `--total_L 16` directly, so the 15→16 extension had nothing to extend and
   inheriting leaves had no CELF heap to resume; the extension block is now conditional on the tree's
   own budget.

## Artifacts

`experiments/battery_goodreads/ndcgtree16test.json` + `_per_user.npz` ·
`channel_orders_policyB_test.json` · logs `experiments/goodreads/ndcgtree16test.out`,
`channel_orders_test.out`.

---

## The criterion ablation on Goodreads (2026-09-14, `splitcrit` label)

The gap the author identified: the first pass put only OUR greedy orders and OUR tree on the second
dataset, so it replicated the least contested claim. `split_criterion_ablation.py` ported and run:
**37.8 min**, five arms regrown at min_leaf 200 / L=16 on validation, measured on both cohorts
against that dataset's own greedy static order.

Test gain at K=16, full/tail:

| criterion | ML-25M | Goodreads |
|---|---|---|
| **measured NDCG@10 (ours)** | **+.0104\*/+.0070\*** | **+.0110\*/+.0106\*** |
| rating variance (Golbandi) | −.0306\*/−.0225\* | −.0256\*/**+.0023\*** |
| pairwise rank loss (Hu & Yu) | −.0304\*/−.0234\* | −.0384\*/−.0030\* |
| logistic loss (Hu & Yu) | −.0863\*/−.0555\* | −.0550\*/−.0099\* |
| uniform random pick | −.0393\*/−.0213\* | −.0598\*/−.0104\* |
| shuffle, min-leaf-respecting | −.0482\*/−.0341\* | −.0449\*/−.0041\* |

**On full NDCG@10 — the axis every order is selected on — exactly one of six is positive on both
datasets.** Gates: both intercept snaps exact, UNSEEN fallback 8 users (0.08%), and the `ndcg` arm
re-measured through this file's code path reproduces `ndcgtree16test.json` at **max deviation
0.00e+00**.

### Two cells move, and both are in the paper

1. ⚠ **A sub-claim was ML-only.** The chapter/paper sentence "Hu and Yu's pairwise rank loss and
   Golbandi's rating variance are indistinguishable… so replacing a rating surrogate with a ranking
   surrogate changed nothing" rests on ML-25M's +0.00020 [−0.00331,+0.00371]. The paired contrast on
   Goodreads is **−0.01286 [−0.01565,−0.01015]**, separating: the ranking surrogate is *worse*. The
   headline survives (it still does not help) but the sentence is now marked as this dataset's.
2. ⚠ **A second positive on the Goodreads TAIL.** Rating variance reaches **+0.00230\*** there,
   against −0.02246 on ML. So "exactly one of six" is a **full-axis** statement on Goodreads. Tail is
   descriptive in this work — no order is ever selected on it — and ours is 4.6x larger.

### Port faults fixed (all the same family as A28's first half)

`OUTDIR` hardcoded to `experiments/battery` (would have overwritten the certified ML ablation and its
five grown trees — HARD RULE 11e); the structure/orders artifact split (two files on ML, one on
Goodreads); the `_vs_policyB` cohort-key suffix; **two** canonical snaps, not one (the smoke run
caught the validation one I missed by grepping for a single pattern); the leaf-order field name
(`own_sequence` vs `sequence`); the heaps directory, which had to be derived from the structure
artifact because Goodreads has both `ndcgtree_heaps` and `ndcgtree16_heaps` and the first-that-exists
rule would have read the wrong search.

**Not run on Goodreads:** the left half of Table 1 (the correlation exhibit) — 85 of 176 nodes have
banked heaps there, since inheriting leaves run no search of their own. And the 23-arm battery, which
is a port of thirteen published methods and is not feasible before the deadline.

---

## Adversarial ECIR panel, 2026-09-15 — and a stale table found because of it

Three Fable reviewers with distinct personas (senior evaluation methodologist, CRS-subfield insider,
final-year PhD student), PDF only, no repo, explicitly adversarial brief. Scores: **weak reject (4),
weak accept (4), weak accept (3)**. The brief asked for hostility and withheld the artifacts, so the
scores are not comparable with earlier panels; the findings are.

### ⭐ TABLE 3'S TWO BANDIT ROWS WERE STALE — found by checking, not by agreeing

Two reviewers said the p.10 GLM-UCB contrast did not reconcile with Table 3. **The prose was right and
the table was wrong.** Recomputed from `act1_per_user_vectors.npz`:

  items-only − GLM-UCB @K=16 full = **+0.02808 [+0.02475, +0.03139]** = the prose's `.0281 [.0247,.0314]`

GLM-UCB's true mean is **.21662**; Table 3 printed **.2161**. Checking every banked row against the
table found the two tuned-bandit rows stale and **every other row exact**:

| row | printed | correct |
|---|---|---|
| GLM-UCB K=16 | .2161/.0833 | **.2166/.0839** |
| Thompson K=1 | .1627/.0376 | **.1613/.0378** |
| Thompson K=16 | .2135/.0878 | **.2126/.0878** |

Tuning deltas recomputed against the corrected values: **+.0068 / +.0032**, not +.0073 / +.0023.
(Tuned arms: GLM-UCB α=4 .22338/.08413, Thompson scale=0.5 .21582/.08812; both vs combined static
−.03562 / −.04318, which the paper already had right.)

★ **A number that appears in prose and in a table is two artifacts, not one.** These drifted apart
and three review rounds plus my own full read did not catch it; recomputing from the per-user bank
did, in two minutes.

### The refusal probe — R2's hypothesis refuted

`src/instrument/refusal_fold_probe.py`, artifact `refusal_fold_probe.json`. Question: does folding a
refusal move the ranking below the m=0 state, which would charge an unanswerable question twice and
inflate every answer-rate effect including RQ1's?

15 probes across both channels (the first pass tested no concepts at all — the concept pool is ranked
BY ANSWERABILITY, so its head has zero refusers by construction; probes must be spread down the
ranking). **6 of 15 separate: 2 down, 4 up. Largest absolute shift 0.00317, and it is positive.**

So a refusal is **not state-preserving** — the token carries information, by design — but it is **not
a systematic penalty**, and at ≤0.003 against effects of +0.0143 and +0.0104 it cannot account for
them. It also does **not** explain the sub-empty K=1 cells (Golbandi .1557 against .1586): those sit
0.0029 below across the whole cohort where refusal shifts are ≤0.0012 and usually positive. A dislike
answer legitimately moving a user off popular items is the likelier cause.

### Other fixes this panel produced

Counts made checkable against the tables (the battery is **eleven ported selectors + three literature
heuristics**; "thirteen" was the comparator audit's count being reused for the battery, and no
defensible thirteen exists for the battery). Related Work no longer claims fMF, Liu et al. and Nguyen
et al. were "re-implemented and run here" — there is no selector module for any of them. The port gate
no longer claims ports were required to beat popularity, which seven do not. Table 2 drops RecVAE and
RBMF: `.0000/.0000` is a degenerate ranking, not a measurement, and two of the "four of five"
consumers leaned on it — the interface limitation survives as one sentence. The candidate-bank
asymmetry is disclosed (ported methods select over the full catalogue, ours over 1,000 — a
restriction, not a privilege). "138 cells" counts every other selector, not every adaptive one.
