# Hu & Yu (RecSys 2013): the replication, and the split-criterion ablation

Plan item **B1 / TODO 1.12** of `docs/design/PAPER_B_PLAN_2026-08-12.md`. Written 2026-08-12.

AUTHOR SAID (2026-08-12), verbatim: *"do 4 reaaally thoroughly, i think they are the only ones who do
something similar to us."*

Hu & Yu 2013 is the nearest published rival to this chapter's centrepiece. Our tree is Golbandi's
ternary interview tree with exactly one change — the split criterion is measured held-out NDCG@10
instead of a rating-variance surrogate — and Hu & Yu built an interview tree whose split loss is
general and explicitly lists **pairwise rank loss**. So *"first ranking objective in an interview
tree's split"* is not available to us and is not claimed anywhere. This file establishes what IS.

Three parts: **A** the faithful snap on their protocol (HARD RULE 2); **B** the objective ablation
inside our harness, which is the real deliverable; **C** positioning — the four separations, verified
line by line, written as chapter-ready prose. **§D is a live run log** — if you are picking this up
in the morning, read §D first: it says what ran, what did not, and what it cost.

---

## 0. DESIGN SHEET (pre-registered; written before any run)

### 0.1 The question

Inside our tree harness, with everything else held fixed, how much of the NDCG tree's held-out gain is
attributable to the split criterion being **the measured reported metric** rather than a value-free
or untruncated surrogate — and specifically, does Hu & Yu's ranking loss already get most of it?

### 0.2 The arms — exactly one thing varies

| arm | criterion | source |
|---|---|---|
| (a) `variance` | within-node squared error of RATING prediction, Eq. (4) | Golbandi, Koren & Lempel, WSDM 2011 |
| (b) `pairwise` | pairwise rank loss, scored by the Taylor / diagonal-Hessian split gain Eq. (8)–(12) | Hu & Yu, RecSys 2013 (the loss their *framework* lists) |
| (b′) `logistic` | logistic loss, same Taylor gain | Hu & Yu, RecSys 2013 (the loss their *experiments* actually ran) |
| (c) `ndcg` | node-size-weighted mean of the three children's cohort-mean MEASURED held-out full NDCG@10 | ours (`src/instrument/ndcg_tree.py`, banked) |
| (r) `random` | uniform draw over the candidate pool, seed 98765 | shuffle control |
| (r′) `random_ml` | uniform draw restricted to candidates whose ternary split clears min-leaf | the *fair* shuffle control — isolates "which question" from "does it split at all" |

`(b′)` is run because the full re-read (§C.1) showed our previous summary was wrong about which loss
they ran. Reporting only one of the two would look like choosing the version that flatters us.

### 0.3 Held fixed, by construction and by assertion

Same tree code (this job imports `ndcg_tree.split_rows` and `ndcg_tree.branch_of`, so the ternary
split rule is literally the same function); same ternary LIKE / DISLIKE / DID-NOT-ANSWER branch rule;
same MIN_LEAF = 200 as the only stopping rule, no depth cap; same charged behavioural answers (an
asked-but-unrated item folds as the i26 native UNSEEN exposure token, an unanswerable concept as a
cv3 refusal token, and the turn is burned either way); same certified pair (`t2i26c_ep31` +
`concept_cv3_ep08`); same policy-B candidate mask; same 1,000-candidate combined pool (500
popularity-ranked items + 500 answerable concepts, items first, from the same two `_ranked_pool`
calls); same 10,000 canonical VALIDATION users to grow on and 10,000 disjoint canonical TEST users to
report on; same 16-question budget compared question-for-question at K ∈ {1, 2, 4, 8, 16}; same
static comparator `greedy_static_policyB_combined`.

### 0.4 The two design decisions the brief asked to be made explicitly

**(i) What the pairwise rank loss is computed over.** FOLD-IN EVIDENCE ONLY. For user *u*,
`POS_u` = fold-in items rated at level ≥ 7 (≥ 4.0 stars, the Liang like boundary) and
`NEG_u` = fold-in items rated below it; the loss sums over **every** (i, j) ∈ POS_u × NEG_u —
all pairs, nothing sampled (HARD RULE 1). Users with an empty POS or NEG contribute exactly zero and
are counted, not dropped. `logistic` uses the same items as labelled examples (1 for POS, 0 for NEG).
`variance` uses the user's whole fold-in rating vector, user-bias corrected, exactly as Eq. (4).

*Why fold-in and not sampled negatives.* Hu & Yu sample negatives for training (l. 325–328 of the
source text). We do not, for two reasons: sampling on our ruler would need author sign-off, and their
sampled negatives are a substitute for the observed dislikes that our graded fold-in actually has.
Using real observed dislikes is strictly more information and strictly less arbitrary.

*The leak check, which is run and must pass before any number is written* (`split_criteria.assert_no_leak`):
(1) the fold-in rating matrix shares no (user, item) cell with the held-out target matrix; (2) per
user, `POS_u` and `NEG_u` are set-disjoint from that user's targets; (3) `POS_u ∪ NEG_u` partitions
that user's fold-in items exactly — nothing invented, nothing lost. This is on top of
`ndcg_tree.leak_assert`, which proves the *routing* lookup is target-free.

**(ii) Training loss over a node's users, or the expected value of asking.** We report the
**FAITHFUL** one: a training loss over the node's users, exactly as Hu & Yu have it. Their split score
is Eq. (12) — a second-order Taylor expansion, with the Hessian replaced by its diagonal, of how far
the training loss on the node's own users would fall if each child received its optimal factor
increment. Because `Σ_children L̃(0)` is the node's own loss and is candidate-independent, the argmin
of the total child loss is the argmax of

    Gain(q) = Σ_{C ∈ {like, dislike, noanswer}} ½ Σ_k g_C[k]² / (H_C[k] + 2λ_u),   λ_u = 10

which is what we compute. `u_P` in their model is one vector shared by the node; in our harness the
user factor is the certified tower's latent `z_u`, folded from that user's own answers to the path,
so `u_P` is per-user and we use `z_u`. That is the minimal transplant: swapping in their criterion
must not also swap out our recommender, or the ablation would confound the criterion with the model.
Scores are the recommender's own, `s_uj = (z_u + Δu)ᵀ W_j + b_j`, with `W`, `b` the certified tower's
LEARNED decoder weight and bias.

The **expected-value-of-asking** variant (`pairwise_heldout`: Hu & Yu's loss under our protocol —
fold, then measure the pairwise rank loss of held-out targets against the policy-B candidate pool) is
the clean single-factor test of separation (2). It is **specified and QUEUED, not run**: it needs a
tower fold per candidate, i.e. it costs what arm (c) costs (the banked NDCG tree build was 37,191 s =
10.3 h), and the box is committed elsewhere. See §B.5 for the cheap exact substitute that IS run.

### 0.5 Leaf tails

A node that the min-leaf rule stops still owes questions up to the 16-budget. Arm (c)'s tail is its
own criterion applied sequentially (greedy argmax of measured NDCG on the leaf's users). Arms (a),
(b), (b′) get the analogue: **the same criterion applied to a refining partition**. Everyone in the
leaf is asked the same next question — there is no branching, so min-leaf does not apply — but the
criterion is summed over the blocks that the tail's earlier answers have already produced. Without
this, a partition criterion evaluated on an unchanging user set would return the same argmax at every
tail step and the arm would ask one question sixteen times. Stated because it is a real choice.

### 0.6 Ns, metric, MDE

10,000 validation users to grow, 10,000 disjoint test users to report. Full AND tail NDCG@10, learned
decoder bias, no floor, policy-B candidate pool, full 18,359-item catalogue, nothing sampled. Paired
percentile bootstrap over per-user difference vectors, 10,000 resamples, seed 98765 — never the
unpaired `1.96·√2·SE` form. At n = 10,000 the tree's own K=16 interval is ±0.0029 wide, so effects
below ≈0.003 are not resolvable and will be reported as nulls rather than as ties.

### 0.7 Controls (HARD RULE 5)

- **Canonical snap.** The m=0 intercept is asserted to reproduce the banked policy-B values before any
  arm is scored: validation `0.15686538118354684` exactly, test `0.15855 / 0.02819` (which is
  `MASK_POLICY_B_MIGRATION.md`'s figure, recorded long before this run).
- **Banked-headline assert.** Arm (c) is re-measured through *this file's* code path and must
  reproduce every published number in `ndcgtree16test_policyB.json` to < 1e-6, at all five budgets on
  both axes. If it does not, the ablation is not on the tree's ruler and the run stops.
- **Shuffle controls.** `random` and `random_ml` as above, plus, in the correlation exhibit, a
  permuted-criterion Spearman and the expected regret of a uniform random pick.
- **Leak check.** §0.4(i), plus `ndcg_tree.leak_assert` on both cohorts.
- **Maths self-test**, run before any environment is built (`src/instrument/test_split_criteria.py`):
  both gradients and both **diagonal Hessians** against central finite differences (the pairwise
  Hessian's cross term `−2 Σ_ij c_ij W_i W_j` is computed exactly, and the test fails loudly if it is
  dropped); the Taylor gain against a brute-force minimisation of the quadratic model; the variance
  criterion against a brute-force per-child recomputation of Eq. (4). All pass — see §B.1.

### 0.8 PRE-REGISTERED INTERPRETATION — fixed before the run, not revisable after seeing it

- **(c) > (b) > (a)** → the mechanism claim is airtight: a ranking loss beats a rating loss, and
  *measuring* the reported metric beats optimising a smooth surrogate for it.
- **(b) ≈ (c)** → the novelty **narrows** to "the exact reported metric". The field-level claim
  (value-free and variance surrogates lose) survives with that nuance stated explicitly. This outcome
  is reported as a narrowing and is **not** marketed away.
- **(b) > (c)** → a major finding against us, reported immediately and prominently, at the top of this
  file and in the report to the author.

### 0.9 What this run can and cannot test

| separation | testable here? |
|---|---|
| (1) untruncated/undiscounted loss vs truncated discounted NDCG@10 | only jointly with (2) — the clean test is the queued `pairwise_heldout` arm |
| (2) training loss vs measured expected value of asking | same |
| (3) 10 sampled negatives vs the full 18,359-item catalogue | **no** — our metric is full-catalogue by construction. This is a statement about *their evaluation*, verified by reading, not by running |
| (4) static comparator = top-k popular vs an order built by maximising the reported metric | **no** — this is a property of their baseline set, verified by reading (§C.2) |

The cheap exact substitute for (1)+(2) that *is* run: §B.5.

### 0.10 Artefacts

Code `src/instrument/split_criteria.py`, `src/instrument/split_criterion_ablation.py`,
`src/instrument/test_split_criteria.py` (commit `7d3e302`, committed **before** any run, HARD RULE 10).
Outputs `experiments/battery/splitcrit.json`, `splitcrit_tree_<criterion>.json`,
`splitcrit_per_user.npz`. Part A: `scripts/baselines/huyu_eip_ml1m.py`,
`experiments/baselines/huyu2013/`.

---

## A. The faithful snap on their protocol

*(pending — filled when the Part A run lands)*

### A.0 What "matching" can and cannot mean here

It is a **4-page short paper with figure-only results**. There is no scalar table anywhere in it: the
three MAP panels and the one RMSE panel of Figure 2 are MATLAB bar charts, and every number we could
snap against has to be read off a bar height. The tick labels are rasterised, so they cannot be
extracted as text; they were digitised from a 400 dpi render.

Therefore "matching" here can mean:
- reproducing the **ordering** of the arms (Warm-MF > EIP > FixQ on ML-1M);
- reproducing the **direction** with budget (MAP rises with depth 3 → 10);
- reproducing the **gap sizes** to the precision the figure supports (EIP − FixQ ≈ 0.018–0.021 MAP on
  ML-1M; Warm-MF − EIP ≈ 0.057–0.064);
and it cannot mean matching a published digit, because there is no published digit. Any claim of
agreement below ≈0.002 MAP is a claim about our pixel measurement, not about their result.

Two further limits stated up front: the latent dimension `d` is never given in the paper, and neither
is the train/test user split size or the number of coordinate-descent iterations. Those are choices we
had to make, and they are recorded with the run rather than tuned to the target.

### A.1 Digitised figure values

`scripts/baselines/huyu_figure_digitise.py`. Page 4 rendered via pdfplumber; each panel's axis frame
auto-detected; row→value calibrated from the y-tick **label centres**; each bar's top read from the
contiguous **axis-anchored** colour run — that anchoring is what rejects the legend patches, which sit
in the same pixel columns as the right-most bar group and corrupted a first pass. Calibration residual
≤0.00012 in every panel, and **400 dpi and 800 dpi agree to ≤0.0003 MAP in every cell of (a)–(c)**, so
these are quoted at **±0.0005**.

| panel | arm | 3 | 5 | 8 | 10 |
|---|---|---|---|---|---|
| (a) ML-1M MAP | FixQ | .7176 | .7239 | .7283 | .7306 |
| | EIP | .7381 | .7406 | .7443 | .7451 |
| | Warm-MF | .8017 flat | | | |
| (b) Netflix MAP | FixQ | .7421 | .7450 | .7470 | .7488 |
| | EIP | .7589 | .7627 | .7674 | .7711 |
| | Warm-MF | .8133 flat | | | |
| (c) Yahoo MAP (depths 3/4/5/6) | FixQ | .8194 | .8237 | .8248 | .8264 |
| | EIP | .8790 | .8798 | .8802 | .8825 |
| | Warm-MF | .9061 flat | | | |
| (d) ML-1M RMSE | FixQ | .9940 | .9995 | .9850 | .9838 |
| | fMF | .9735 | .9686 | .9674 | .9723 |
| | EIP | .9656 | .9626 | .9638 | .9626 |
| | Warm-MF | .8489 flat | | | |

**How to read their percentages: absolute MAP points, not relative.** (a) Warm−EIP is +.0597 absolute
= +8.05% relative, and the text says "about 6%"; (b) Warm−EIP is +.0483 = +6.31%, text says "5%".
Relative fails badly on both. The one loose statement is Yahoo's "about 7%" for EIP−FixQ, which is
+.0568 absolute (5.7 points) and +6.90% relative — there the relative reading fits better. So:
**absolute, with Yahoo imprecise.**

**Two text errors in one four-page results section, both confirmed by pixel measurement.** (i) §3.2.3
claims "EIP outperforms Warm-MF by a bit less than 3%" on Yahoo! Music; panel (c) has Warm-MF flat at
**0.9061** against EIP 0.8790–0.8825, i.e. **Warm-MF is ABOVE EIP by 0.024–0.027 at every depth**. The
magnitude is right and the direction is backwards — the two names are swapped. The figure is
authoritative, so the paper's "our method almost reaches the rank performance of warm-start" reading
of Yahoo is unsupported by its own figure. (ii) §3.2.2's "the MAP of EIP is still higher than that of
EIP by about 2%" is a plain typo for FixQ.

Netflix and Yahoo! Music are **not** replicated — we do not hold those datasets. ML-1M only.

### A.2 The snap

**Ruler, one for every arm.** ML-1M `ratings.dat`, 1,000,209 ratings / 6040 users / 3706 rated items.
Users split 75/25 → 4530 train / 1510 test (seed 12345). Each test user's ratings halved at random
(seed 24680) → 121,495 answer / 122,238 evaluation ratings. Training negatives 10:1 (seed 13579); test
negatives 10:1 — **theirs** (seed 20130101); candidate tie-break by a fixed uniform key, **identical
across arms so every comparison is paired** (seed 4242). d = 20, λ_v = 10, λ_u = 10 (theirs), min-leaf
30, 8 alternations × 3 coordinate sweeps.

Choices the paper does not specify, recorded rather than tuned: the 75/25 user split; random rather
than temporal halves; 10:1 *training* negatives (symmetric with their test ratio); min-leaf 30; d = 20;
λ_v = 10. FixQ's features are intercept + a 3-way (>3 / ≤3 / unknown) one-hot per question, which
deliberately gives FixQ **the same information EIP gets** rather than mere rated/not — the stronger
reading of their baseline.

**The method is implemented as specified, including the Taylor trick.** The Eq. (12) diagonal-Hessian
gain is computed with the *true* Hessian, searched jointly over all 3706 items × 4 thresholds, and used
**only to pick the question**; the chosen node's Δu is then re-fitted by coordinate descent, exactly as
lines 280–283 require. The one deliberate deviation is in the *fitting*: each coordinate step is taken
on the **majorising** curvature (¼ for logistic) rather than the true one, because the undamped Newton
step diverged (training loss 0.43 → 7.59 → 3.44 → 5.66 → 7.45). Four optimisers were tried; the
majorised cyclic descent is the one that is monotone. A separate silent trap was found and fixed: a
V initialisation at 0.05 leaves both Newton steps ridge-dominated and the fit stuck at the zero-scale
saddle (Warm-MF MAP 0.189, below random-ish).

**Metric.** Headline is **MAP(r≥4)**. Their Eq. (13) states "P@n calculates the fraction of top n
recommended items that are **positive**", and precision requires a positive/negative labelling, so
"observed item set" means observed *positives*. MAP over all observed ratings is kept as a secondary
column so the choice is visible.

| arm | MAP(r≥4) at 3 / 5 / 8 / 10 | MAP(all observed) | **theirs (digitised)** |
|---|---|---|---|
| random | .1073 | .1011 *(analytic .1040 ✓)* | — |
| **most-popular, no interview** | **.5129** | .4468 | — |
| FixQ | .4910 / .4880 / .4777 / .4729 | .4323 / .4276 / .4206 / .4199 | .7176 / .7239 / .7283 / .7306 |
| EIP (logistic) | .5277 / .5223 / — / — | .4589 / .4568 | .7381 / .7406 / .7443 / .7451 |
| Warm-MF | .6926 | .6329 | .8017 |

### A.3 How close did we get — plainly

**What reproduces.**
- **The arm ordering, Warm-MF > EIP > FixQ**, at every depth measured.
- **The EIP − FixQ gap in sign and order of magnitude**: +0.0367 (d=3) and +0.0343 (d=5) on the r≥4
  ruler, +0.0266 / +0.0292 on all-observed, against their +0.0205 / +0.0167. Ours is if anything
  *larger* than theirs.

**What does not reproduce.**
- **The absolute level**, by 0.19–0.24 MAP everywhere. Three candidate explanations were tested and
  all three are excluded. *Not the metric*: the analytic E[MAP] of a random permutation over 81
  positives and 810 negatives is 0.1040 against a measured 0.1011. *Not the negative pool*: negatives
  are already drawn only from items the user never rated, which is the easier of the two readings.
  *Not latent capacity*: the Warm-MF calibration curve on a held-out validation cohort runs 0.6538
  (d=10) → 0.6823 (d=20) → 0.6985 (d=50) → **0.7047 (d=100)**, increments decaying +0.0285, +0.0162,
  +0.0062, converging to about 0.71 — quadrupling d from 20 to 100 buys 0.022 and no achievable d
  reaches their 0.8018. **The residual is unexplained and is reported as unexplained.**
- **The direction with depth.** Their MAP rises with the number of interview questions for both cold
  arms. Ours falls: FixQ .4910 → .4729 monotonically, EIP flat-to-falling .5277 → .5223.

**The direction failure is now EXPLAINED, and the explanation is a property of their stated protocol
rather than of our code.** A *test* user's answers must come from their first half only, while a
*train* user's come from all their ratings. On the top-10 popular questions test users answer **~50%
as often as train users at every rank** (0.283 vs 0.575 at rank 1); at k=3 a train user answers 1.58
of 3 questions and a **test user answers 0.78** — most test users answer nothing at all. The model is
therefore fitted on feature vectors roughly twice as populated as the ones it is asked to serve, and
each additional question widens the mismatch.

**The symmetric variant — halve the train users' ratings too, so answer availability matches at fit
time and serve time — flips the trend.** Everything else identical:

| FixQ, MAP(r≥4) | depth 3 | depth 5 | trend |
|---|---|---|---|
| as their protocol states it (asymmetric) | .4910 | .4880 | **falling** |
| **symmetric answers** | **.5164** | **.5224** | **rising** |
| *theirs (digitised)* | *.7176* | *.7239* | *rising* |

Two things change at once and both move toward the paper: the level rises by ~0.025, and **the
direction reverses to match their published trend**. It also lifts FixQ just above the most-popular
baseline (.5164 vs .5129), where the asymmetric version sat below it.

**This is a finding about the paper, not merely a fix to our harness.** Their reported result that MAP
rises with interview length is, on our reading of their protocol, contingent on training users being
about twice as answerable as the users the method is actually deployed on. Under the deployment
condition their own evaluation imposes — a cold user who has answered only what the interview elicited
— asking more questions makes their fixed-question baseline *worse*, not better. That is worth stating
in the chapter, and it is independent of the unexplained level gap.

Caveat carried: this is measured for **FixQ at depths 3 and 5** so far. The symmetric EIP cells had not
returned at the time of writing, so the claim is currently about the baseline arm.

**One incidental finding worth keeping.** On our ruler **FixQ is worse than simply ranking by
popularity** (.4910 vs .5129 at d=3) — asking the top-k popular items and folding the answers back in
makes it *worse* than not asking. EIP does clear most-popular (.5277). That is our Act 1 pattern
appearing inside their own baseline set.

### A.4 What "matching" therefore means here, stated once

We reproduce **what our chapter actually cites them for** — the arm ordering and the EIP−FixQ gap — on
an internally consistent, leak-checked, analytically-verified ruler. We do **not** reproduce their
absolute level or their depth trend, and we say so rather than tuning toward the figure. For a
four-page paper that never states its latent dimension, its train/test ratio, its alternation count or
its training negative ratio, that is the honest ceiling on a replication, and the ablation of Part B
does not depend on it: **Part B runs their criterion inside our harness on our ruler, so it is
unaffected by anything unresolved here.**

### A.5 Controls

| control | result |
|---|---|
| Leak check | answer ∩ eval pairs **0**; train ∩ eval pairs **0**; train ∩ test users **0**; test users appearing in train ratings **0** |
| Canonical snap, reported first | most-popular .5129 / .4468; random .1011 against an **analytic .1040** — the metric implementation is verified independently of any model |
| Shuffle control (random-question tree) | implemented, **NOT YET RETURNED.** This is the one required control still outstanding. No claim in §A rests on an arm it would invalidate, but every EIP number here is uncontrolled until it lands, and the file says so. |
| Symmetric-answers contrast | run for FixQ at depths 3 and 5 — see §A.3; it reverses the depth trend |

**Still running at the time of writing**, persisting one JSON per cell to
`experiments/baselines/huyu2013/cells/`: EIP-logistic depths 8 and 10, the random-question shuffle
control, the pairwise-loss arm, and the RMSE panel with fMF.

Code: `scripts/baselines/huyu_eip_ml1m.py`, `scripts/baselines/huyu_figure_digitise.py`.

---

## B. The objective ablation — the real deliverable

### B.1 The criterion maths is verified before the harness runs

`src/instrument/test_split_criteria.py`, on synthetic data, no environment, one second:

```
[ok] pairwise: gradient and diagonal Hessian match central finite differences
     (max |dg| 4.47e-10, max |dH| 1.15e-04)
[ok] logistic: gradient and diagonal Hessian match central finite differences
     (max |dg| 8.26e-11, max |dH| 1.76e-05)
[ok] _taylor_gain == sum_C [Ltilde(0) - min_du Ltilde(du)] under the diagonal Hessian
     (0.487490218 vs 0.487490218)
[ok] variance_scores == brute-force Golbandi Eq. (4) variance REDUCTION on all 5 candidates
ALL SELF-TESTS PASSED
```

This matters more than it looks. The pairwise diagonal Hessian expands to two separable terms plus a
cross term `−2 Σ_ij c_ij W_ik W_jk`; dropping the cross term is the obvious shortcut and would leave a
criterion that is *plausible*, *fast* and *not Hu & Yu's*. The finite-difference check is what stops a
clean-looking, entirely fictitious ablation.

### B.2 The three trees side by side — THE RESULT

**10,000 held-out TEST users, K=16, against the same policy-B static comparator, paired percentile
bootstrap, 10,000 resamples, seed 98765. Only the split criterion differs.**

| criterion | full@10 | gain vs static | 95% CI | tail@10 | tail gain | 95% CI |
|---|---|---|---|---|---|---|
| **(c) measured NDCG@10 (ours)** | **0.26936** | **+0.01035** | [+0.00749, +0.01328] | **0.13074** | **+0.00705** | [+0.00394, +0.01022] |
| (b) pairwise rank loss (Hu & Yu) | 0.22859 | **−0.03041** | [−0.03368, −0.02718] | 0.10031 | **−0.02338** | [−0.02672, −0.02002] |
| (a) variance (Golbandi) | 0.22840 | **−0.03061** | [−0.03415, −0.02714] | 0.10124 | **−0.02246** | [−0.02614, −0.01877] |
| (b′) logistic loss (Hu & Yu, as run) | 0.17274 | **−0.08627** | [−0.08977, −0.08258] | 0.06817 | **−0.05552** | [−0.05902, −0.05207] |

Every interval excludes zero. **The measured-metric criterion is the only one that beats the static
order; both published surrogates lose to it, and lose by three times the margin ours wins by.**

Full curves, test, gain vs the static comparator (full / tail):

| K | (c) ours | (b) pairwise | (a) variance | (b′) logistic |
|---|---|---|---|---|
| 1 | **0.00000 / 0.00000** (nested) | −0.01523 / −0.01743 | −0.01842 / −0.01701 | **0.00000 / 0.00000** (nested) |
| 2 | +0.00528 / −0.00262 | −0.02321 / −0.02478 | −0.03380 / −0.02494 | −0.00896 / −0.00565 |
| 4 | +0.00891 / +0.00482 | −0.02824 / −0.02106 | −0.04253 / −0.02951 | −0.02777 / −0.01756 |
| 8 | +0.01062 / +0.00835 | −0.03222 / −0.01915 | −0.03190 / −0.02411 | −0.05605 / −0.03991 |
| 16 | +0.01035 / +0.00705 | −0.03041 / −0.02338 | −0.03061 / −0.02246 | −0.08627 / −0.05552 |

Validation (the selection cohort) at K=16, for the shrinkage record: ours **+0.02794 / +0.01055**,
pairwise −0.03255 / −0.02397, variance −0.03273 / −0.02199, logistic −0.08292 / −0.05533. Ours shrinks
val→test as already documented; **the surrogate arms do not shrink, because they were never fitted to
anything — they are simply wrong on both cohorts.**

### Applying the pre-registered interpretation, without adjustment

The pre-registered cases were (c)>(b)>(a), (b)≈(c), and (b)>(c). What happened is **(c) ≫ (b) ≈ (a)**:

- **(c) > (b) is emphatic and on the right side of zero.** +0.01035 against −0.03041 — a gap of
  0.04076 full, roughly four times the size of our own headline conditioning gain.
- **(b) ≈ (a).** Hu & Yu's pairwise rank loss (−0.03041) and Golbandi's rating variance (−0.03061) are
  statistically indistinguishable at K=16, 0.0002 apart with intervals ~0.0065 wide. **Moving from a
  rating surrogate to a ranking surrogate buys nothing measurable.** That is a sharper statement than
  our pre-registration anticipated and it belongs in the chapter: the ranking-vs-rating axis is not
  what separates these methods; the surrogate-vs-measured axis is.
- **The adverse case did not occur.** (b) does not beat (c) anywhere, at any budget, on either axis.

**So the mechanism claim stands in its strongest form.** Hold the tree code, the ternary split rule,
min-leaf, the charged behavioural answers, the certified recommender, the candidate mask, the pool,
both cohorts and the budget all fixed, and change nothing but the scalar a candidate question is
scored by: the sign of the conditioning gain flips. Adaptivity does not fail here and it does not
succeed here — the objective decides which.

### The logistic arm is the sharpest single exhibit in the file

`logistic` opens on **exactly our question** (concept 114, identical ternary partition, K=1 difference
identically zero on all 20,000 users), and then degrades monotonically: −0.00896 at K=2, −0.02777 at
K=4, −0.05605 at K=8, −0.08627 at K=16. Its absolute full@10 **peaks at K=2 (0.18723) and falls to
0.17274 by K=16** — under Hu & Yu's own experimental loss, asking more questions makes the
recommendation worse, on held-out users, with every question charged and folded.

That is the objective-mismatch thesis in one row: a criterion can agree with the metric on the first,
easiest, largest-cohort decision and still walk the interview steadily downhill afterwards. It also
explains the correlation exhibit's shape (§B.5): logistic has regret exactly 0.0 at the root and no
better than random below it.

### B.6 Separation (1) isolated — PRELIMINARY, and it moves the claim

`pairwise_heldout_root.py` changes **only the loss**. At the root state, for each candidate, it folds
once and computes from the *same* score matrix both full NDCG@10 (through the same
`ndcg10_from_scores` funnel, same mask contract) and the held-out pairwise rank loss in its 0-1 (AUC)
form over the full 18,359-item catalogue — untruncated, undiscounted, ties counted as errors. Same
fold, same state, same candidates, same protocol; only the scalarisation differs. Its m=0 gate against
the tree's banked root heap passes at ≤1e-12 and every recomputed NDCG gain matches the banked build
at **0.0e+00**.

**Status: RUNNING — 140 of 1,000 candidates at the time of writing, and the prefix is not yet
channel-balanced (110 items, 30 concepts), because the first 80 were banked before the evaluation
order was interleaved. The numbers below are therefore PRELIMINARY and must not be quoted as the
full-pool result.** `pwheldout_partial_report.py` recomputes them from the partial at any moment and
always reports its own coverage.

| at the root, 140/1000 candidates | value |
|---|---|
| Spearman(measured NDCG@10 gain, measured pairwise-loss reduction) | **+0.637** |
| Kendall | +0.468 |
| Spearman, items only (n=110) | +0.615 |
| Spearman, concepts only (n=30) | +0.855 |
| top-10 recall | 0.50 |
| regret of following the measured pairwise loss | **+0.00371** |
| expected regret of a uniform random pick | +0.01444 |
| rank the pairwise loss gives the NDCG argmax | **2** of 140 |

**If this holds to 1,000 candidates it relocates the mechanism, and in our favour but not where we
expected.** Put it beside the *training-loss* pairwise criterion at the same node (§B.5): that one has
Spearman **−0.149**, ranks the metric's argmax **75th**, and carries regret +0.01473 — i.e.
indistinguishable from guessing. The *measured* pairwise loss, on the same pairwise objective at the
same state, has Spearman +0.637, ranks it 2nd, and carries regret +0.00371, about a quarter of random.

So the two separations are not equal partners:

- **Separation (2) — training loss versus the measured expected value of asking — is doing almost all
  of the work.** Hu & Yu's Taylor-approximated training-loss reduction is nearly uninformative about
  held-out ranking; the same loss *measured on held-out targets* is a good proxy for it.
- **Separation (1) — truncation and positional discounting — is real but second-order.** An
  untruncated, undiscounted pairwise loss, measured properly, gets you most of the way to NDCG@10 —
  but not all of it: it still forgoes +0.00371 of NDCG@10 at the root, it disagrees on the argmax, and
  its top ten overlap the metric's top ten only half the time.

That is a **more precise and more defensible** claim than "ranking losses are the wrong objective",
and it is the one the chapter should make. It also has a constructive consequence for workstream E:
if the measured pairwise loss tracks NDCG@10 at ρ≈0.64, it is a candidate *cheap surrogate reward* for
the parametric policy — one that can be computed without a top-k sort — provided the final selection
is still made on the reported metric. That is a design lead we did not have before.

**Do not write any of this into the chapter until the run completes.** 140 candidates, item-heavy, is
a direction, not a number.

### B.4 Controls — all passed

| control | result |
|---|---|
| Canonical snap, validation m=0 | 0.15686538 full — reproduces the banked policy-B intercept **exactly** |
| Canonical snap, test m=0 | 0.15855 / 0.02819 — reproduces `MASK_POLICY_B_MIGRATION.md`'s banked figure |
| **Banked-headline assert** | the `ndcg` arm re-measured through *this file's* code path reproduces **every** published number in `ndcgtree16test_policyB.json`, max deviation **2.81e-07** (float summation order across leaves). The ablation is provably on the tree's ruler. |
| Leak check, routing | 0/10,000 users on both cohorts have a held-out target inside the routing lookup |
| Leak check, criteria | 0/10,000 users leak: the fold-in rating matrix shares no cell with the targets; POS_u and NEG_u are target-free for every user and partition each user's fold-in items exactly |
| Nothing sampled | the pairwise loss summed over **137,591,707** fold-in pairs, all of them |
| Shuffle control, `random` | degenerates to depth 0 (root question answered by 4.5% of users, 9,552 into one branch) — the confound that makes `random_ml` the informative control |
| Shuffle control, `random_ml` | reaches depth 7, 27 leaves, 286 distinct questions — **deeper and more varied than every real criterion**, which refutes any "deeper tree = better" reading |
| Criterion maths | gradients and diagonal Hessians against central finite differences; Taylor gain against brute-force minimisation; Eq. (4) against brute-force per-child recomputation — all pass (§B.1) |
| Untruncated pairwise loss | matches a brute-force double loop over all (target, non-target) pairs on 200 random cases, including masked candidates and ties |

### B.3 Structural comparison

All six trees, same pool, same cohort, same min-leaf 200, same 16-question budget:

| criterion | depth | nodes | leaves | root splitter | root answer rate | root children (L/D/U) | distinct questions asked |
|---|---|---|---|---|---|---|---|
| (a) variance (Golbandi) | 5 | 25 | 17 | concept 731 | 0.916 | 4861 / 4297 / 842 | 145 |
| (b) pairwise (Hu & Yu) | 2 | 7 | 5 | concept 511 | 0.688 | 1209 / 5668 / 3123 | 71 |
| (b′) logistic (Hu & Yu, as run) | **1** | 4 | 3 | **concept 114** | 0.742 | 2606 / 4814 / 2580 | 46 |
| **(c) measured NDCG@10 (ours)** | **5** | **22** | **15** | **concept 114** | 0.742 | 2606 / 4814 / 2580 | **159** |
| (r) random | 0 | 1 | 1 | item 1246 | 0.045 | 267 / 181 / 9552 | 16 |
| (r′) random, min-leaf respecting | 7 | 40 | 27 | item 458 | 0.186 | 639 / 1223 / 8138 | 286 |

Four things worth stating in the chapter.

**The criteria do not merely disagree, they share almost nothing.** Pairwise splitter overlap between
any two criteria is **0 or 1 splitter out of the whole tree**, including ours against Golbandi's (0)
and ours against Hu & Yu's pairwise (1). Question-set overlaps are `ndcg|variance` 32, `ndcg|pairwise`
15, `ndcg|logistic` 7. Same 1,000 candidates, same 10,000 users, same states — different interviews.

**The logistic criterion opens exactly where we do and then stops.** Its root splitter is concept 114
with an identical ternary partition to ours, so at K=1 it is nested in our tree *and* in the static
comparator and the difference is exactly zero by construction. It then reaches depth 1 and quits: with
only 3 leaves and 46 distinct questions, it is close to a static order with one branch point.

**Depth is not virtue.** `random_ml` — a uniform draw among candidates that clear min-leaf — reaches
depth 7 with 27 leaves and 286 distinct questions, deeper and more varied than every real criterion.
Any argument of the form "our tree is better because it is deeper / asks more distinct questions" is
refuted by this row, and that is exactly why the fair random control was worth running.

**The plain random control collapses, as expected and as designed.** Its root question is answered by
4.5% of users, dumping 9,552 of 10,000 into a single branch, so min-leaf refuses to split and it
degenerates to a static 16-question order. This is the confound that makes `random_ml` the informative
control rather than this one.

### B.4 Controls

*(pending)*

### B.5 Criterion vs MEASURED NDCG at every node — the free ground truth

The `ndcgtree` build persisted a CELF heap sidecar per node. The seed entries of each sidecar
(`upd == 0`) are the **exact measured cohort-mean NDCG@10 marginal gains** of every candidate at that
node's prefix, over that node's own users — 22 nodes × ~1,000 candidates of ground truth for the
quantity our criterion measures and the other criteria approximate, at states the real tree actually
visited, at zero additional NDCG compute.

For each node and each criterion we report: Spearman and Kendall correlation against the measured
gain; whether the criterion's argmax is the measured argmax; the **regret** (measured NDCG gain
forgone by following that criterion instead of the metric) and its user-weighted mean; the rank the
measured argmax receives under the criterion; and top-10 recall. Controls: the Spearman of a permuted
criterion, and the expected regret of a uniform random pick — a criterion worth having must sit well
below the latter.

This is the direct, quantitative form of Act 2's "1 of 21 rules optimises the reported quantity"
(`BASELINE_OBJECTIVE_AUDIT.md`): it turns a description of what each rule optimises into a measurement
of how far that surrogate is from the thing being reported, at real states.

### RESULT — the surrogates are barely distinguishable from guessing

22 nodes, ~1,000 candidates each, exact measured NDCG@10 marginal gains from the real 10.3 h build.

| criterion | mean Spearman | nodes with rho>0 | argmax == NDCG argmax | mean top-10 recall | **user-weighted regret** |
|---|---|---|---|---|---|
| (a) variance (Golbandi) | **+0.010** | 13/22 | **1/22** | 0.03 | **+0.01664** |
| (b) pairwise rank loss (Hu & Yu) | **−0.060** | 10/22 | **0/22** | 0.01 | **+0.01738** |
| (b′) logistic loss (Hu & Yu, as run) | **−0.104** | 9/22 | **1/22** | 0.03 | **+0.01282** |
| *control: uniform random pick* | — | — | — | 0.10 by construction | **+0.01955** |

Read the last column first. **Following any of these three criteria forgoes between 66% and 89% of the
NDCG@10 that a uniform random pick would forgo.** Not 5%, not 20% — they recover a third of the way
from random to optimal at best. The rank correlations are ~0 and two of the three are *negative*, so
these are not weak-but-aligned surrogates; at the states this tree actually visits they are close to
uninformative about the quantity the field reports. Golbandi's criterion picks the measured argmax at
1 node of 22; Hu & Yu's pairwise loss picks it at **0 of 22**; top-10 recall is 0.01–0.03 against 0.10
for a random draw, i.e. **the pairwise criterion's top ten candidates overlap the metric's top ten
less often than chance**.

This is the cleanest exhibit in the file, and it is free: the ground truth was already banked.

### The root is a real and instructive exception

| criterion at the ROOT | its argmax | measured NDCG gain of its pick | regret | rank it gives the metric's argmax | Spearman |
|---|---|---|---|---|---|
| variance | concept 731 | +0.00414 | +0.01635 | **250** of 1,000 | −0.212 |
| pairwise | concept 511 | +0.00576 | +0.01473 | **75** of 1,000 | −0.149 |
| **logistic** | **concept 114** | **+0.02049** | **0.00000** | **1** | **+0.187** |
| measured NDCG@10 (ours) | concept 114 | +0.02049 | — | 1 | — |

**Hu & Yu's actually-run loss picks exactly the right first question and then loses the plot.** At the
root the logistic criterion's argmax *is* the metric's argmax — concept 114, the same question our
tree and the policy-B static order both open with — with regret identically zero and top-10 recall
0.3. One level down its mean regret is no better than the others. So the honest statement is not
"their criterion is bad", it is **"their criterion agrees with the metric where the cohort is largest
and the signal is strongest, and stops agreeing exactly where an adaptive method has to earn its
keep."** That is a more interesting claim than a flat dismissal and it is what the numbers support.

Note also what this does to the variance criterion: it ranks the metric's own best question **250th of
1,000** at the root. The gap between a rating-variance surrogate and a ranking metric is not a
rescaling.

---

## C. Positioning

### C.1 What the full re-read changed

*(see `external_literature/findings/hu_yu_2013_interview_process_learning.md`, section "CORRECTED
2026-08-12"; summarised here for the chapter)*

1. **Their ranking experiments ran a LOGISTIC loss, not the pairwise rank loss.** §3.2.1: "we evaluate
   the MAP for EIP, FixQ and Warm-MF on MovieLens since fMF cannot handle the logistic loss." Pairwise
   rank loss occurs exactly once in the paper, in the "square loss, logistic loss, pairwise rank loss,
   etc." list attached to Eq. (1) — the *basic* MF objective, one subsection before the tree exists. It
   is never instantiated, tuned, ablated or evaluated. Our previous summary described it as what they
   ran. It is not. Hence arm (b′).
2. **Their metric is untruncated and undiscounted.** Eq. (13) is average precision over each test
   user's whole observed set. No cutoff, no positional discount. Separation (1) is therefore stronger
   than we had written: nothing in their pipeline — training loss, split criterion, or reported metric
   — is truncated or position-discounted.
3. **Negative sampling is in their TRAINING data too**, not only the evaluation, so their split
   criterion is fitted against sampled negatives as well.
4. **Warm-MF beats EIP on two of three datasets, and on the third the text contradicts the figure.**
   The paper's "almost reaches the rank performance of warm-start" sentence for Yahoo! Music is
   contradicted by its own Figure 2(c), which shows Warm-MF above EIP at every depth.
5. **Depth is the budget, and their own tree stops paying at 6** on Yahoo ("the MAP begins to
   decrease, which means our model becomes overfitting"). Their budgets are 3/5/8/10 and 3/4/5/6;
   ours is 16.
6. **The Taylor trick is the interesting engineering fact.** Because `Σ_C L̃(0)` is
   candidate-independent, `g` and `H` are per-user quantities that do not depend on the candidate: one
   pass per node, then every candidate is a masked sum. That is the published route to a scale our
   exact objective cannot reach — our tree pays a tower fold per candidate (10.3 h), theirs pays none.
   It is also the obvious skeleton for the parametric policy of workstream E.

### C.2 The four separations, as chapter-ready prose

> The nearest prior construction to the tree we build is Hu & Yu's *Interview Process Learning for
> Top-N Recommendation* (RecSys 2013), which fits a ternary interview tree in the functional
> matrix-factorisation lineage and whose split loss is stated generally enough to admit a pairwise
> rank loss. We therefore do not claim to be first to put a ranking objective inside an interview
> tree's split, and we state four separations up front rather than leaving them to be found.
>
> *First, nothing they optimise or report is truncated or position-discounted.* The loss their
> framework lists is a smooth pairwise surrogate over all pairs; the loss their ranking experiments
> actually run is a pointwise logistic loss; the metric they report is untruncated average precision.
> NDCG@10 has a top-10 cutoff and a positional discount and reduces to none of the three. This is the
> same distinction that lets an A-optimality derivation go through for an untruncated, undiscounted
> loss and not for ours.
>
> *Second, they optimise a second-order approximation of a training loss over the tree fit; we measure
> the expected value of asking.* Their split score is a Taylor expansion, with the Hessian replaced by
> its diagonal, of how far the training loss on a node's own users would fall if each child received
> its optimal factor increment — used, in their words, "only to find the interview question". Ours
> folds each candidate's answers through a frozen certified recommender and ranks the users' held-out
> targets. The two are not the same object: one is a bound on how well the model will fit what it has
> already seen, the other is the number the paper reports.
>
> *Third, they evaluate on ten sampled negatives per positive; we evaluate on the full 18,359-item
> catalogue.* Sampled-negative ranking metrics are known to reorder methods relative to full-catalogue
> ones, so their numbers are not comparable to ours and their ordering is not evidence about ours.
>
> *Fourth, and most consequential for this chapter, their static comparator is popularity.* FixQ asks
> the top-k most popular items. Nothing in that paper — or, as far as our audit of twenty-one
> selection rules can establish, in the surrounding literature — tests an adaptive selector against a
> static order that was itself built by maximising the metric being reported. That comparator is the
> one this chapter constructs, and it is the reason our Act 1 result is not pre-empted.

### C.3 Verdict on what novelty remains

**Our novelty claim survives, and the ablation strengthens it rather than narrowing it.**

What is NOT available, and is not claimed anywhere: *first ranking objective inside an interview
tree's split.* Hu & Yu have it, in 2013, and they are cited as the nearest prior construction.

What IS available, and is now measured rather than argued:

> An interview tree whose split criterion is the **measured held-out value of asking** — cohort-mean
> NDCG@10 under a frozen certified recommender, on the full 18,359-item catalogue with no negative
> sampling, grown on a validation cohort and reported on a disjoint test cohort — beats a static order
> built by maximising that same quantity. **Substituting the published surrogates into the identical
> machinery reverses the result: Hu & Yu's pairwise rank loss loses to the static order by −0.03041,
> Golbandi's rating variance by −0.03061, and Hu & Yu's own experimental logistic loss by −0.08627,
> against our +0.01035.** The field's null about adaptive elicitation is a statement about objectives,
> not about adaptivity.

Three further claims the ablation licenses, none of which were available before this run:

1. **The ranking-vs-rating axis is not the one that matters.** Golbandi's rating-variance criterion and
   Hu & Yu's pairwise rank loss are statistically indistinguishable on our metric (−0.03061 vs
   −0.03041, 0.0002 apart, intervals ~0.0065 wide). Upgrading a rating surrogate to a ranking surrogate
   bought nothing. What separates the methods is *measured versus surrogate*, not *ranking versus
   rating*. This is the single most useful thing the exercise produced and it was not predicted.
2. **The surrogates are close to uninformative about the reported metric at real states.** Over 22
   nodes × ~1,000 candidates of exact ground truth, following any of them forgoes 66–89% of the
   NDCG@10 that a uniform random pick forgoes; mean rank correlations are ~0 and two of three are
   negative; Hu & Yu's pairwise loss picks the metric's argmax at 0 of 22 nodes, and its top ten
   candidates overlap the metric's top ten *less often than chance*.
3. **A criterion can be right where it is easy and wrong where it matters.** The logistic criterion
   picks exactly the right first question — regret 0.0 at the root, the same concept 114 our tree and
   the static order open with — and then walks the interview downhill, its absolute full@10 peaking at
   K=2 and falling by K=16. Agreement on the largest, coldest, most-shared decision is not evidence of
   agreement where an adaptive method has to earn its keep.

**What must travel with the claim, honestly stated.**

- The comparison is on OUR ruler — our recommender, our answers, our catalogue, our metric. It shows
  that *their criterion, transplanted into our harness*, loses. It is not a refutation of their
  published result on their own protocol, and Part A is the separate exercise of establishing what
  that result is.
- Their criterion is the **cheap** one, and that is a genuine advantage we do not have. Their Taylor
  expansion makes the gradient and Hessian candidate-independent, so 1,000 candidates cost one pass;
  our exact objective pays a tower fold per candidate and cost 37,191 s for one tree. If the chapter
  claims our objective is better it must also concede that theirs is orders of magnitude cheaper per
  node — and that this is precisely why the parametric policy (workstream E) is the method rather than
  the tree.
- The `logistic` arm's collapse is partly structural: it stops at depth 1, so most of its 16 questions
  come from a leaf tail chosen by a criterion anti-correlated with the metric. The depth is itself a
  consequence of the criterion, so this is not a confound, but the mechanism should be stated rather
  than left as a mystery.
- The min-leaf rule applies to all three branches including an empty DID-NOT-ANSWER branch, which
  penalises criteria drawn to fully answerable questions. Held identical across arms by design; the
  `nonempty` robustness mode is implemented and unrun (TODO 1.18 / plan A10).

---

## D. Run log and current state (live — newest entry last)

Kept so a morning review can start from this file rather than from a chat message. Every entry
records what ran, what it cost, and what it changed.

| when | what | outcome |
|---|---|---|
| 22:13 | ML-1M downloaded (6040 users / 3952 movie ids / 1,000,209 ratings — matches the paper's stated counts exactly) | Part A can use their own dataset. Netflix and Yahoo! Music are NOT held and are not replicated. |
| 22:18 | Figure 2 digitised at 400 and 800 dpi | values in §A.1 / the findings file |
| 22:2x | Criterion maths self-test | PASS (§B.1) |
| 22:28 | Part A canonical snap | most-popular MAP 0.4468, random MAP 0.1011, leak check clean (0 answer/eval pair overlap, 0 train/eval pair overlap, 0 train/test user overlap) |
| 23:12 | Part B grow stage v1 launched | **KILLED at 23:17.** Not a correctness fault: the refining leaf tail was going quadratic in block count, because after a few tail questions most blocks are singletons and the cost was dominated by per-block call overhead. |
| 23:19 | Part B grow stage v2 launched after two exact fixes — dense membership matrices, and skipping singleton blocks (exactly zero contribution for the variance criterion, a candidate-independent constant for the Hu & Yu criteria) | **KILLED at 23:22.** Still too slow, and the real cause was different: each sparse product's DENSE output is `(chunk x 18,359)` no matter how few users the block holds, so a three-user block paid the same 37 MB allocation as the root. |
| 00:00 | **Part B grow COMPLETE** — five trees, 33 min | variance depth 5 / 17 leaves; pairwise depth 2 / 5; logistic depth 1 / 3; random depth 0 / 1; random_ml depth 7 / 27 |
| 00:02 | **Correlation exhibit COMPLETE** | §B.5 — the surrogates are barely distinguishable from guessing |
| 00:27 | **Part B measure COMPLETE**, 24.6 min; banked-headline assert PASS at 2.81e-07 | §B.2 — **(c) ≫ (b) ≈ (a)**, ours the only arm above the static order |
| 00:27 | Separation-(1) root probe launched, 1,000 candidates | running |
| 23:22 | Part B grow stage v3, third exact fix: the variance criterion now restricts the item axis to the items the block's users actually rated. Eq. (4) sums over items with `n_j > 0`, so an item nobody in the block rated contributes zero to the node's e2 and to every child's — dropping it changes no number. | running |
| ~23:20 | Part A first cells | **RED FLAG, see below** |

### Part A: a red flag that must be resolved before any snap is reported

`eip_logistic_d3` MAP = **0.4455**, `canonical` most-popular MAP = **0.4468**, `warm_logistic` MAP =
**0.5996**, random MAP = 0.1011. Their Figure 2(a) at depth 3 has FixQ 0.7175, EIP 0.7383,
Warm-MF 0.8018.

Two separate problems, and only the second is fatal:
1. *The level is far below theirs* (≈0.45 vs ≈0.72). The random baseline at 0.1011 is almost exactly
   the analytic 1/11 expected for ten sampled negatives per positive, so **the negative-sampling ratio
   and the AP implementation are right**; the gap is model strength or the definition of `O_i`, not
   the metric plumbing.
2. **EIP is not beating most-popular** — it is 0.0013 *below* it. A three-question interview tree that
   loses to ranking by popularity is not reproducing a method whose entire claim is that it beats
   top-k-popular. Under the project's own rule ("a surprising number is a harness bug until proven
   otherwise") this is a harness fault until shown otherwise. The leading suspect is the latent
   dimension: the cells record `d = 10`, the paper never states `d`, and a d=10 factorisation on ML-1M
   is close to a popularity model. A d sweep is under way. Second suspect: the Warm-MF logistic loss
   trace is not monotone (0.210 → 0.196 → 0.182 → 0.190 → 0.188 → 0.173 → 0.183 → 0.182), which
   suggests the alternating fit is not converging — and Warm-MF is the *ceiling* in their figure, so
   an under-fitted Warm-MF corrupts every comparison.

**FixQ has not been run yet and is the comparison that matters** — their ranking claim is exactly
"EIP outperforms FixQ ... by 2% averagely", and our chapter's positioning rests on FixQ being top-k
popular. Note FixQ is a *stronger* baseline than most-popular ranking: it asks the top-k popular items
as an interview and folds the answers into a feature-based MF.

### Things already established, before any citable ablation number

- The **variance (Golbandi) criterion does grow a real tree** on our pool under min-leaf 200: its
  root splitter is **concept 731** (answer rate 0.916, children 4861 / 4297 / 842), and it splits
  again at depth 1 on both the like and dislike branches. It is not degenerate, so arm (a) is a
  genuine comparison and not a straw man.
- Its root choice is **already different from ours**: our NDCG tree's root is concept 114
  (answer rate 0.742, children 2606 / 4814 / 2580). Two criteria, same 1,000 candidates, same
  10,000 users, same state — different first question.
- **Cost of the criteria, measured:** ~17 s to score all 1,000 candidates at 10,000 users for
  `variance`; the Hu & Yu criteria add one candidate-independent gradient pass (~11 s at 10,000
  users, 137,591,707 pairs, nothing sampled) and are then masked sums. Our exact NDCG criterion
  needs a tower fold per candidate and cost 37,191 s for the whole tree. **That ratio is the
  Taylor trick's entire value proposition, and it is now measured rather than asserted.**
- **A caveat that must travel with arms (b)/(b′):** the Hu & Yu criteria are drawn to
  100%-answerable concepts, whose DID-NOT-ANSWER branch is empty; the all-three-branches min-leaf
  rule then refuses the split outright. The rule is held identical across all arms (the brief
  requires it, and arm (c)'s tree cannot be regrown), but a `nonempty` min-leaf mode is implemented
  as a robustness check, because Golbandi's own selector constrains only the like and hate branches
  (`golbandi_tree_selector._best_split`: `(cL >= min) & (cH >= min)`). This is the same interaction
  as TODO 1.18 / plan A10.
