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
inside our harness, which is the real deliverable; **C** positioning — the four separations, the
Part A / Part B reconciliation, the surviving novelty as a conjunction, and chapter-ready prose.
**§D is the run log** — it says what ran, what did not, and what it cost.

**STATUS 2026-08-13: FINALISED for the chapter.** Part B is complete and citable
(`experiments/battery/splitcrit.json`, `citable = true`, `SMOKE = false`); the separation-(1) root
probe completed at 05:46 (`experiments/battery/pwheldout_root.json`, 1,000/1,000 candidates) and §B.6
is no longer preliminary.
The pre-registered interpretation of §0.8 fired on the **(c) ≫ (b)** leg and NOT on the **(b) > (a)**
leg; the outcome is written as **(c) ≫ (b) ≈ (a)** throughout and never as "(c) > (b) > (a)". Part A is
closed at depths 3 and 5 on both readings of the answerability protocol; depths 8/10, the Part A
pairwise arm, the fMF RMSE panel, and Netflix / Yahoo! Music are **unrun**, listed as such in §A.6 and
§B.7, and never quietly omitted.

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

| FixQ, MAP(r≥4) | depth 3 | depth 5 | depth 8 | depth 10 | trend |
|---|---|---|---|---|---|
| as their protocol states it (asymmetric) | .4910 | .4880 | .4777 | .4729 | **falling, monotonically** |
| **symmetric answers** | **.5164** | **.5224** | **.5274** | **.5312** | **rising, monotonically** |
| *theirs (digitised)* | *.7176* | *.7239* | *.7283* | *.7306* | *rising, monotonically* |

The full sweep is now in and the reversal is clean at every depth, not just the first two: their
protocol as stated gives a monotone decline across 3/5/8/10, and the symmetric variant gives a
monotone rise across the same four depths, with the same shape as their figure.

Two things change at once and both move toward the paper: the level rises by ~0.025, and **the
direction reverses to match their published trend**. It also lifts FixQ just above the most-popular
baseline (.5164 vs .5129), where the asymmetric version sat below it.

**This is a finding about the paper, not merely a fix to our harness.** Their reported result that MAP
rises with interview length is, on our reading of their protocol, contingent on training users being
about twice as answerable as the users the method is actually deployed on. Under the deployment
condition their own evaluation imposes — a cold user who has answered only what the interview elicited
— asking more questions makes their fixed-question baseline *worse*, not better. That is worth stating
in the chapter, and it is independent of the unexplained level gap.

**The symmetric EIP and RandQ cells have now returned at depths 3 and 5, and the reversal carries to
both** (§A.6): EIP .52121 → .52717 and RandQ .51460 → .51851, i.e. rising with depth, against EIP
.52770 → .52227 (falling) on the asymmetric ruler. So the direction reversal is a property of the
answerability asymmetry across every arm measured, not of the FixQ baseline alone. Depths 8 and 10
are measured for FixQ only.

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
| **Shuffle control (random-question tree)** | **RETURNED ON BOTH RULERS. Asymmetric: passes at depth 3 (+0.0136) and collapses by depth 5 (+0.0029). Symmetric: +0.0066 at depth 3 and +0.0087 at depth 5 — it does NOT collapse. See §A.6.** |
| Symmetric-answers contrast | run for FixQ at all four depths and for EIP/RandQ at depths 3 and 5 — see §A.3 and §A.6; it reverses the depth trend monotonically |

### A.6 The shuffle control, and what it decomposes

Same budget, same min-leaf, same fitting procedure, same alternations, same ruler and the same paired
candidate lists as EIP; only the split criterion is replaced by a uniform draw over the candidates
that satisfy min-leaf. **The 2×2 is now complete** — both depths on both readings of the answerability
protocol. MAP(r≥4), most-popular (no interview) = .5129 on every cell:

| arm | asym d3 | asym d5 | sym d3 | sym d5 |
|---|---|---|---|---|
| EIP (Hu & Yu's criterion) | **.52770** | **.52227** | **.52121** | **.52717** |
| **RandQ (shuffle control)** | **.51405** | **.51939** | **.51460** | **.51851** |
| FixQ (top-k popular as the interview) | .49097 | .48797 | .51636 | .52240 |
| **EIP − RandQ** | **+0.01365** | **+0.00288** | **+0.00661** | **+0.00866** |
| **EIP − FixQ** | +0.03673 | +0.03430 | +0.00485 | +0.00477 |
| **RandQ − FixQ** | +0.02309 | +0.03142 | −0.00176 | −0.00389 |

("asym" = the protocol as literally stated; "sym" = `--sym-answers`, training users' interview answers
restricted to their first half too. Cell files `{eip_logistic,randq,fixq_logistic}[_sym]_d{3,5}.json`.)

> **PROVENANCE WARNING, and it bit during collection.** The runner's log line prints the arm tag
> *without* the `_sym` suffix, so a symmetric run reads as `eip_logistic depth=5` in `run_main.log`
> exactly as an asymmetric one does — lines 205 and 303 of that log are the two different rulers
> under the identical string. **Only the cell FILENAMES distinguish them**, and the cell JSON records
> no `symmetric` field either: `--sym-answers` (behaviour) and `--tag-suffix` (filename) are separate
> flags that could in principle disagree. Every number in this table was read from the JSON files, and
> the two depth-5 symmetric cells were additionally confirmed against the live process command lines
> (`--sym-answers --tag-suffix _sym --depths 5` on both PIDs). **Never difference these cells off a log
> line.** A `symmetric` field in `write_cell` would close this properly and should be added before the
> next sweep.

**Three findings. The first is a correction to what this file previously said.**

**(1) The depth-5 collapse is an artefact of the answerability asymmetry — it does not survive fixing
it.** This file previously reported that EIP's advantage over a coin-flip criterion collapses from
+0.0136 at three questions to +0.0029 at five, and flagged the symmetric cell as the one that would
separate "the criterion is weak" from "the protocol starves it". **That cell has now returned and it
came down on the second reading.** Under symmetric answers the advantage is +0.00661 at depth 3 and
**+0.00866 at depth 5** — smaller than the asymmetric depth-3 figure, but *rising* with depth rather
than collapsing. **The harder sentence is therefore NOT licensed and is not written.** What is:

> On ML-1M, under Hu & Yu's protocol as literally stated — where a training user's interview answers
> come from all their ratings but a test user's come from their first half only — the advantage of
> their split criterion over a random-question interview tree of the same shape falls from +0.0136 at
> three questions to +0.0029 at five. That collapse is a property of the answerability asymmetry, not
> of the criterion: once training and test users are made equally answerable, the advantage is smaller
> at three questions (+0.0066) but does not decay by five (+0.0087). Scope: ML-1M, MAP over their ten
> sampled negatives per positive, depths 3 and 5, their protocol, 1,510 test users, single seed.

**(2) But fixing the asymmetry guts their headline claim instead.** Their §3.2.1 result is
"EIP outperforms FixQ … by 2% averagely" — +0.0205 absolute at depth 3 by our digitisation. On the
asymmetric ruler we *over*-reproduce it (+0.0367 / +0.0343). On the symmetric ruler it shrinks to
**+0.0048 at both depths, under a quarter of their reported gap**, because FixQ is the arm that
benefits most from making training and test users equally answerable (.4910 → .5164 at depth 3). So
the two readings trade off against each other, and neither flatters the paper on both axes: the
literal protocol makes the criterion look strong against FixQ and worthless against a coin flip, and
the symmetric protocol makes it look real against a coin flip and marginal against FixQ.

**(3) The contribution decomposition REVERSES between the two readings.** On the asymmetric ruler,
of the +0.0367 EIP beats FixQ by, **+0.0231 — roughly two thirds — survives replacing the split
criterion with a coin flip**: most of the advantage is *having a conditional tree at all*, not the
criterion the paper's contribution rests on. On the symmetric ruler that decomposition inverts —
**RandQ falls BELOW FixQ** (−0.0018 at depth 3, −0.0039 at depth 5), so a tree with randomly chosen
questions is *worse* than simply asking the top-k popular items, and **all** of EIP's (smaller) edge
over FixQ is attributable to the criterion. Which reading you take decides whether Hu & Yu's
contribution is "the tree" or "the criterion", and the paper does not say which reading it intends.

**The limitation that travels with all three.** These are single-seed point differences on 1,510 test
users with **no confidence interval** — the Part A harness persists only cell-level MAP, not per-user
average precisions, so no paired bootstrap is available without a re-run. +0.0087 against a spread of
only ~0.014 between most-popular and the best cell is not a difference we should quote as if it were
interval-backed. Treat the *directions* (collapse under asymmetric, no collapse under symmetric,
RandQ crossing FixQ) as the findings and the magnitudes as indicative. Part B, which is the actual
deliverable, is paired-bootstrapped on 10,000 test users and does not depend on any of this.

Note also that on the asymmetric ruler the random-question tree is level with most-popular (.5141 vs
.5129) while FixQ sits *below* it (.4910); under symmetric answers both clear it.

**This cuts both ways, and it applies to us.** It is why Part B carries its own `random_ml` control,
and why Part B compares against a static order built by maximising the *same* objective rather than
against popularity. Part B's controls answer the corresponding question for our tree: `random_ml`
reaches depth 7 with 27 leaves and 286 distinct questions — structurally richer than our tree — and
our criterion still wins, while both published criteria lose to the static order outright.

**Part A collection is now stopped and the box is free.** Both detached jobs exited cleanly after
writing `eip_logistic_sym_d5.json` and `randq_sym_d5.json` (2026-08-13). Banked cells in
`experiments/baselines/huyu2013/cells/`: FixQ at depths 3/5/8/10 on both rulers; EIP and RandQ at
depths 3 and 5 on both rulers; `canonical`, `warm_logistic`, and the two d-sweeps.

**What was commissioned but never returned, and is NOT reported anywhere above:** EIP and RandQ at
depths 8 and 10 (the asymmetric depth-8 EIP run reached iteration 1 of 8 and was superseded by the
symmetric sweep), the Part A pairwise-loss arm, and the RMSE panel with fMF (`run_rmse.log` stops at
its CONFIG line). None of these is needed for §A.6's verdict or for Part B; the depth-8/10 cells are
the only ones that would extend the depth trend past 5, so **no claim about EIP beyond five questions
is licensed on either ruler.**

Code: `scripts/baselines/huyu_eip_ml1m.py`, `scripts/baselines/huyu_figure_digitise.py`.

---

## B. The objective ablation — the real deliverable

**Status: COMPLETE and CITABLE.** `experiments/battery/splitcrit.json` (410 KB, written 2026-08-13
00:27, `citable = true`, `SMOKE = false`, `seconds = 1477.5` for the measure stage on top of a 33-minute
grow stage), per-arm trees `splitcrit_tree_{variance,pairwise,logistic,random,random_ml}.json`, per-user
vectors `splitcrit_per_user.npz`. Run identity, read from the artefact: `tower = t2i26c_ep31.pt`,
`concept_ckpt = concept_cv3_ep08.pt`, `mask_policy = B`, `objective = full`, `total_L = 16`,
`min_leaf = 200`, `minleaf_mode = all3`, `budgets = [1, 2, 4, 8, 16]`, `lam_u = 10.0`, `seed = 98765`,
`static_comparator.label = greedy_static_policyB_combined`,
`bootstrap = {kind: "paired percentile over per-user differences", n_resamples: 10000, seed: 98765}`.

### B.0 The contract check that licenses every comparison in this section

The ablation is only interesting if it sits on the **same ruler** as the published tree result. The run
carries a `banked_headline_assert` that re-measures arm (c) through *this file's* code path and
differences it against `experiments/battery/ndcgtree16test_policyB.json`. Read from the artefact:

| cell | recomputed | banked | abs deviation |
|---|---|---|---|
| K1 full | 0.0 | 0.0 | **0.0** |
| K1 tail | 0.0 | 0.0 | **0.0** |
| K2 full | 0.005281998589634895 | 0.005281998589634895 | **0.0** |
| K2 tail | −0.0026241775986231447 | −0.0026241775986231447 | **0.0** |
| K4 full | 0.008908980266749859 | 0.008908980266749859 | **0.0** |
| K4 tail | 0.004817837561664236 | 0.004817837561664236 | **0.0** |
| K8 full | 0.010619475589692593 | 0.010619475589692593 | **0.0** |
| K8 tail | 0.008346395585322712 | 0.008346395585322712 | **0.0** |
| K16 full | 0.010352365604788066 | 0.010352646841853858 | 2.81e-07 |
| K16 tail | 0.007045511512352518 | 0.007045511512352518 | **0.0** |

**Nine of the ten cells reproduce at exactly 0.0**; the tenth deviates by 2.8e-07, four orders of
magnitude below the ±0.0029 the K=16 interval is wide, and its cause is float summation order across
leaves. Say this explicitly in the chapter: *the ablation arms and the published tree number are
differences against the same static order, computed by the same funnel, on the same 10,000 test users,
so they may be placed in one table.* Without this gate they could not be (see
`banked-global-greedy-is-a-policy-A-artifact` for what happens when banked numbers from two rulers are
compared).

Two further identity checks, run over `splitcrit_per_user.npz` for this write-up:

- Re-assembling every arm's per-user vector from its leaf blocks and re-differencing reproduces all
  **120 published cohort cells** (6 arms × 5 budgets × 2 axes × 2 cohorts) to a maximum absolute
  deviation of **1.17e-10**.
- The static comparator's per-user vector is **bit-identical across all six arms** (max abs difference
  0.0), which is what makes the arm-versus-arm contrasts of §B.2.4 paired rather than merely aligned.

*A reporting detail that must travel with every tail number in this file.* The tail metric is undefined
for users with no tail targets, and those users are excluded from it: the tail mean and interval are
over **9,281 of 10,000 test users** and **9,217 of 10,000 validation users**, identical masks across all
six arms. The artefact's `n_users` field reads `10000` on tail cells as well as full cells, which is
wrong for the tail; it is a labelling defect, not a numerical one — the recomputed tail means match the
published ones to 4e-11 once the mask is applied.

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

### B.2 THE RESULT — six arms side by side

**10,000 held-out TEST users (10,000 disjoint validation users to grow), against the same policy-B
static comparator `greedy_static_policyB_combined`, paired percentile bootstrap over per-user
differences, 10,000 resamples, seed 98765 — never the unpaired `1.96·√2·SE` form. Only the split
criterion differs.**

#### B.2.1 Headline, K=16, test

| criterion | full@10 | gain vs static | 95% CI | tail@10 | tail gain | 95% CI |
|---|---|---|---|---|---|---|
| **(c) measured NDCG@10 (ours)** | **0.26936** | **+0.01035** | [+0.00749, +0.01328] | **0.13074** | **+0.00705** | [+0.00394, +0.01022] |
| (b) pairwise rank loss (Hu & Yu) | 0.22859 | **−0.03041** | [−0.03368, −0.02718] | 0.10031 | **−0.02338** | [−0.02672, −0.02002] |
| (a) variance (Golbandi) | 0.22840 | **−0.03061** | [−0.03415, −0.02714] | 0.10124 | **−0.02246** | [−0.02614, −0.01877] |
| (b′) logistic loss (Hu & Yu, as run) | 0.17274 | **−0.08627** | [−0.08977, −0.08258] | 0.06817 | **−0.05552** | [−0.05902, −0.05207] |
| (r′) random, min-leaf respecting | 0.21083 | −0.04817 | [−0.05148, −0.04491] | 0.08958 | −0.03412 | [−0.03770, −0.03058] |
| (r) random | 0.21967 | −0.03934 | [−0.04270, −0.03601] | 0.10243 | −0.02126 | [−0.02449, −0.01799] |

The static comparator itself is 0.25900 full / 0.12370 tail at K=16 on test.

**All twelve intervals in this table exclude zero, and exactly one of the six arms is above the static
order.** Counting the whole budget grid: of the 60 test cells (6 arms × 5 budgets × 2 axes), 4 are
nested at K=1 and identically zero by construction, and **all 56 remaining intervals exclude zero**. On
validation, 55 of the 56 do; the single exception is **our own arm's tail at K=2** (−0.00175 [−0.00402,
+0.00052]), which separates negatively on test (−0.00262 [−0.00477, −0.00046]).

#### B.2.2 Every budget, both axes, both cohorts

**TEST — full@10 gain vs the policy-B static order [95% paired-bootstrap CI]**

| criterion | K=1 | K=2 | K=4 | K=8 | K=16 |
|---|---|---|---|---|---|
| **(c) measured NDCG@10 (ours)** | 0.00000 (nested) | +0.00528 [+0.00325, +0.00728] | +0.00891 [+0.00633, +0.01151] | +0.01062 [+0.00774, +0.01353] | +0.01035 [+0.00749, +0.01328] |
| (b) pairwise rank loss (Hu & Yu) | -0.01523 [-0.01713, -0.01334] | -0.02321 [-0.02552, -0.02092] | -0.02824 [-0.03083, -0.02568] | -0.03222 [-0.03530, -0.02922] | -0.03041 [-0.03368, -0.02718] |
| (a) variance (Golbandi) | -0.01842 [-0.02044, -0.01644] | -0.03380 [-0.03690, -0.03079] | -0.04253 [-0.04583, -0.03912] | -0.03190 [-0.03537, -0.02844] | -0.03061 [-0.03415, -0.02714] |
| (b') logistic loss (Hu & Yu, as run) | 0.00000 (nested) | -0.00896 [-0.01088, -0.00703] | -0.02777 [-0.03018, -0.02537] | -0.05605 [-0.05919, -0.05298] | -0.08627 [-0.08977, -0.08258] |
| (r') random, min-leaf respecting | -0.01441 [-0.01674, -0.01210] | -0.01423 [-0.01676, -0.01167] | -0.02528 [-0.02823, -0.02230] | -0.03764 [-0.04089, -0.03440] | -0.04817 [-0.05148, -0.04491] |
| (r) random | -0.02230 [-0.02423, -0.02039] | -0.03324 [-0.03574, -0.03075] | -0.04933 [-0.05236, -0.04631] | -0.04844 [-0.05174, -0.04518] | -0.03934 [-0.04270, -0.03601] |

*absolute full@10 (static order: K=1 0.18079, K=2 0.19618, K=4 0.21783, K=8 0.23862, K=16 0.25900)*

| criterion | K=1 | K=2 | K=4 | K=8 | K=16 |
|---|---|---|---|---|---|
| **(c) measured NDCG@10 (ours)** | 0.18079 | 0.20147 | 0.22674 | 0.24924 | 0.26936 |
| (b) pairwise rank loss (Hu & Yu) | 0.16556 | 0.17297 | 0.18959 | 0.20640 | 0.22859 |
| (a) variance (Golbandi) | 0.16237 | 0.16238 | 0.17530 | 0.20672 | 0.22840 |
| (b') logistic loss (Hu & Yu, as run) | 0.18079 | 0.18723 | 0.19006 | 0.18258 | 0.17274 |
| (r') random, min-leaf respecting | 0.16638 | 0.18195 | 0.19255 | 0.20098 | 0.21083 |
| (r) random | 0.15849 | 0.16295 | 0.16850 | 0.19018 | 0.21967 |

**TEST — tail@10 gain vs the policy-B static order [95% paired-bootstrap CI]**

| criterion | K=1 | K=2 | K=4 | K=8 | K=16 |
|---|---|---|---|---|---|
| **(c) measured NDCG@10 (ours)** | 0.00000 (nested) | -0.00262 [-0.00477, -0.00046] | +0.00482 [+0.00215, +0.00750] | +0.00835 [+0.00542, +0.01143] | +0.00705 [+0.00394, +0.01022] |
| (b) pairwise rank loss (Hu & Yu) | -0.01743 [-0.01984, -0.01501] | -0.02478 [-0.02762, -0.02204] | -0.02106 [-0.02400, -0.01814] | -0.01915 [-0.02220, -0.01603] | -0.02338 [-0.02672, -0.02002] |
| (a) variance (Golbandi) | -0.01701 [-0.01963, -0.01450] | -0.02494 [-0.02832, -0.02168] | -0.02951 [-0.03312, -0.02598] | -0.02411 [-0.02774, -0.02048] | -0.02246 [-0.02614, -0.01877] |
| (b') logistic loss (Hu & Yu, as run) | 0.00000 (nested) | -0.00565 [-0.00782, -0.00352] | -0.01756 [-0.02010, -0.01492] | -0.03991 [-0.04300, -0.03683] | -0.05552 [-0.05902, -0.05207] |
| (r') random, min-leaf respecting | -0.01464 [-0.01732, -0.01201] | -0.01592 [-0.01861, -0.01327] | -0.02055 [-0.02372, -0.01746] | -0.02553 [-0.02887, -0.02219] | -0.03412 [-0.03770, -0.03058] |
| (r) random | -0.01998 [-0.02232, -0.01763] | -0.02846 [-0.03130, -0.02558] | -0.03621 [-0.03943, -0.03286] | -0.03325 [-0.03642, -0.03003] | -0.02126 [-0.02449, -0.01799] |

*absolute tail@10 (static order: K=1 0.05373, K=2 0.07012, K=4 0.08566, K=8 0.10334, K=16 0.12370)*

| criterion | K=1 | K=2 | K=4 | K=8 | K=16 |
|---|---|---|---|---|---|
| **(c) measured NDCG@10 (ours)** | 0.05373 | 0.06749 | 0.09048 | 0.11168 | 0.13074 |
| (b) pairwise rank loss (Hu & Yu) | 0.03631 | 0.04534 | 0.06461 | 0.08419 | 0.10031 |
| (a) variance (Golbandi) | 0.03673 | 0.04517 | 0.05615 | 0.07922 | 0.10124 |
| (b') logistic loss (Hu & Yu, as run) | 0.05373 | 0.06447 | 0.06811 | 0.06343 | 0.06817 |
| (r') random, min-leaf respecting | 0.03909 | 0.05420 | 0.06512 | 0.07781 | 0.08958 |
| (r) random | 0.03375 | 0.04166 | 0.04945 | 0.07008 | 0.10243 |

**VALIDATION — full@10 gain vs the policy-B static order [95% paired-bootstrap CI]**

| criterion | K=1 | K=2 | K=4 | K=8 | K=16 |
|---|---|---|---|---|---|
| **(c) measured NDCG@10 (ours)** | 0.00000 (nested) | +0.00637 [+0.00436, +0.00835] | +0.01436 [+0.01179, +0.01691] | +0.02159 [+0.01879, +0.02449] | +0.02794 [+0.02495, +0.03099] |
| (b) pairwise rank loss (Hu & Yu) | -0.01473 [-0.01660, -0.01285] | -0.02365 [-0.02598, -0.02130] | -0.02839 [-0.03100, -0.02580] | -0.03502 [-0.03803, -0.03195] | -0.03255 [-0.03577, -0.02928] |
| (a) variance (Golbandi) | -0.01635 [-0.01824, -0.01440] | -0.03427 [-0.03738, -0.03104] | -0.04193 [-0.04538, -0.03844] | -0.03455 [-0.03810, -0.03106] | -0.03273 [-0.03626, -0.02924] |
| (b') logistic loss (Hu & Yu, as run) | 0.00000 (nested) | -0.00814 [-0.01001, -0.00620] | -0.02714 [-0.02956, -0.02476] | -0.05538 [-0.05844, -0.05233] | -0.08292 [-0.08650, -0.07932] |
| (r') random, min-leaf respecting | -0.01391 [-0.01612, -0.01170] | -0.01547 [-0.01813, -0.01293] | -0.02574 [-0.02870, -0.02281] | -0.03706 [-0.04030, -0.03392] | -0.04649 [-0.04993, -0.04315] |
| (r) random | -0.02014 [-0.02208, -0.01825] | -0.03214 [-0.03469, -0.02965] | -0.04886 [-0.05202, -0.04579] | -0.04846 [-0.05175, -0.04517] | -0.04103 [-0.04424, -0.03784] |

*absolute full@10 (static order: K=1 0.17736, K=2 0.19363, K=4 0.21496, K=8 0.23625, K=16 0.25561)*

| criterion | K=1 | K=2 | K=4 | K=8 | K=16 |
|---|---|---|---|---|---|
| **(c) measured NDCG@10 (ours)** | 0.17736 | 0.20000 | 0.22932 | 0.25784 | 0.28355 |
| (b) pairwise rank loss (Hu & Yu) | 0.16263 | 0.16998 | 0.18657 | 0.20123 | 0.22306 |
| (a) variance (Golbandi) | 0.16101 | 0.15937 | 0.17303 | 0.20170 | 0.22288 |
| (b') logistic loss (Hu & Yu, as run) | 0.17736 | 0.18549 | 0.18782 | 0.18087 | 0.17269 |
| (r') random, min-leaf respecting | 0.16345 | 0.17817 | 0.18922 | 0.19919 | 0.20912 |
| (r) random | 0.15722 | 0.16149 | 0.16611 | 0.18779 | 0.21458 |

**VALIDATION — tail@10 gain vs the policy-B static order [95% paired-bootstrap CI]**

| criterion | K=1 | K=2 | K=4 | K=8 | K=16 |
|---|---|---|---|---|---|
| **(c) measured NDCG@10 (ours)** | 0.00000 (nested) | -0.00175 [-0.00402, +0.00052] | +0.00824 [+0.00559, +0.01100] | +0.01195 [+0.00892, +0.01491] | +0.01055 [+0.00738, +0.01367] |
| (b) pairwise rank loss (Hu & Yu) | -0.01652 [-0.01889, -0.01416] | -0.02294 [-0.02570, -0.02016] | -0.01946 [-0.02237, -0.01649] | -0.01848 [-0.02171, -0.01525] | -0.02397 [-0.02742, -0.02046] |
| (a) variance (Golbandi) | -0.01718 [-0.01963, -0.01463] | -0.02576 [-0.02899, -0.02237] | -0.02710 [-0.03062, -0.02356] | -0.02394 [-0.02761, -0.02027] | -0.02199 [-0.02563, -0.01835] |
| (b') logistic loss (Hu & Yu, as run) | 0.00000 (nested) | -0.00369 [-0.00574, -0.00165] | -0.01774 [-0.02031, -0.01525] | -0.04051 [-0.04359, -0.03747] | -0.05533 [-0.05887, -0.05190] |
| (r') random, min-leaf respecting | -0.01577 [-0.01827, -0.01322] | -0.01654 [-0.01935, -0.01375] | -0.02228 [-0.02537, -0.01923] | -0.02754 [-0.03094, -0.02418] | -0.03536 [-0.03881, -0.03192] |
| (r) random | -0.01958 [-0.02194, -0.01723] | -0.02931 [-0.03223, -0.02645] | -0.03818 [-0.04159, -0.03487] | -0.03401 [-0.03739, -0.03070] | -0.02141 [-0.02478, -0.01798] |

*absolute tail@10 (static order: K=1 0.05235, K=2 0.06795, K=4 0.08468, K=8 0.10291, K=16 0.12366)*

| criterion | K=1 | K=2 | K=4 | K=8 | K=16 |
|---|---|---|---|---|---|
| **(c) measured NDCG@10 (ours)** | 0.05235 | 0.06620 | 0.09293 | 0.11486 | 0.13421 |
| (b) pairwise rank loss (Hu & Yu) | 0.03583 | 0.04501 | 0.06523 | 0.08443 | 0.09970 |
| (a) variance (Golbandi) | 0.03518 | 0.04219 | 0.05759 | 0.07897 | 0.10168 |
| (b') logistic loss (Hu & Yu, as run) | 0.05235 | 0.06426 | 0.06694 | 0.06239 | 0.06833 |
| (r') random, min-leaf respecting | 0.03658 | 0.05141 | 0.06240 | 0.07537 | 0.08830 |
| (r) random | 0.03278 | 0.03864 | 0.04651 | 0.06889 | 0.10226 |

#### B.2.3 What `random` and `random_ml` each control for, and why both are needed

They are not redundant, and the pair is what makes the exhibit honest.

- **`random_ml` (uniform draw restricted to candidates whose ternary split clears min-leaf) is the fair
  criterion control.** It holds the *tree shape machinery* fixed and destroys only the *choice of
  question*, so the contrast (c) − (r′) isolates the criterion. It is also the control that refutes the
  laziest reading of our own result: `random_ml` reaches **depth 7, 40 nodes, 27 leaves, 286 distinct
  questions** — strictly deeper and more varied than our own tree (depth 5, 22 nodes, 15 leaves, 159
  questions) — and still loses to the static order by −0.04817 full. *No argument of the form "the NDCG
  tree wins because it is deeper or asks more distinct questions" survives this row.*
- **`random` (uniform draw over the whole 1,000-candidate pool) is the answerability control, and it
  degenerates on purpose.** Its root pick is item 1246, answered by **4.5%** of users, which dumps
  **9,552 of 10,000** into the no-answer branch; min-leaf then refuses every split and the arm collapses
  to a **depth-0, single-leaf, random static 16-question order**. That degeneration is itself the
  measurement: it says that a question drawn without regard to answerability cannot support a partition
  at all in a cold catalogue, which is the confound that would otherwise contaminate `random_ml` and the
  reason `random_ml` is the informative control.
- Because `random` degenerates to a *static* order, it doubles as a **third comparator**: a randomly
  chosen static 16-question order. That accident is what produces the most uncomfortable number in the
  file, in §B.2.4.

#### B.2.4 Arm versus arm, paired — computed for this write-up from the banked per-user vectors

The artefact scores every arm against the static order but not against each other. Both are paired on
the same users, so the arm-versus-arm contrast is available exactly, at no additional model compute.
Same protocol: paired percentile bootstrap over per-user differences, 10,000 resamples, seed 98765,
n = 10,000 (full) / 9,281 (tail), TEST at K=16.

| contrast | full@10 | 95% CI | tail@10 | 95% CI |
|---|---|---|---|---|
| ndcg − pairwise | **+0.04076** | [+0.03741, +0.04412] | **+0.03043** | [+0.02694, +0.03389] |
| ndcg − variance | **+0.04096** | [+0.03740, +0.04446] | **+0.02951** | [+0.02581, +0.03321] |
| ndcg − logistic | **+0.09662** | [+0.09300, +0.10033] | **+0.06257** | [+0.05894, +0.06619] |
| ndcg − random_ml | +0.05852 | [+0.05499, +0.06202] | +0.04117 | [+0.03752, +0.04484] |
| ndcg − random | +0.04969 | [+0.04633, +0.05304] | +0.02831 | [+0.02502, +0.03163] |
| **pairwise − variance** | **+0.00020** | **[−0.00331, +0.00371] ns** | **−0.00092** | **[−0.00448, +0.00277] ns** |
| pairwise − logistic | +0.05586 | [+0.05247, +0.05931] | +0.03214 | [+0.02872, +0.03565] |
| pairwise − random_ml | +0.01776 | [+0.01427, +0.02118] | +0.01074 | [+0.00707, +0.01439] |
| **pairwise − random** | +0.00893 | [+0.00575, +0.01207] | **−0.00212** | **[−0.00540, +0.00127] ns** |
| variance − logistic | +0.05566 | [+0.05203, +0.05937] | +0.03306 | [+0.02929, +0.03686] |
| variance − random_ml | +0.01757 | [+0.01403, +0.02107] | +0.01166 | [+0.00792, +0.01538] |
| **variance − random** | +0.00873 | [+0.00520, +0.01228] | **−0.00120** | **[−0.00486, +0.00245] ns** |
| logistic − random_ml | −0.03809 | [−0.04136, −0.03480] | −0.02140 | [−0.02480, −0.01814] |
| logistic − random | −0.04693 | [−0.04993, −0.04401] | −0.03426 | [−0.03728, −0.03121] |
| random_ml − random | −0.00884 | [−0.01209, −0.00556] | −0.01286 | [−0.01633, −0.00951] |

Three things fall out that the against-static table alone could not say.

1. **(b) ≈ (a) is now interval-backed, not an eyeballed 0.0002.** Hu & Yu's pairwise rank loss and
   Golbandi's rating variance are **+0.00020 [−0.00331, +0.00371] apart on full and −0.00092 [−0.00448,
   +0.00277] on tail** — the only null in the matrix, on both axes. Upgrading a rating surrogate to a
   ranking surrogate is not merely a small gain; it is *no measurable gain*.
2. **On the tail at K=16 neither published criterion is distinguishable from a random static order.**
   pairwise − random = −0.00212 [−0.00540, +0.00127] and variance − random = −0.00120 [−0.00486,
   +0.00245], both ns, both with the point estimate on the *wrong* side of zero. On full they do clear
   it (+0.0089 and +0.0087, both separating), so the honest sentence is: *the published criteria buy a
   small but real amount over a random static order on full NDCG@10, and nothing measurable on the
   tail.* Do not write "they are worse than random"; do not omit this either.
3. **Hu & Yu's actually-run loss is beaten by both shuffle controls, decisively, on both axes.**
   logistic − random_ml = −0.03809 full / −0.02140 tail; logistic − random = −0.04693 / −0.03426. This
   is the strongest statement in the ablation, and it is about (b′), not about (b).

#### B.2.5 Validation, and the shrinkage record

At K=16 on the selection cohort: ours **+0.02794 / +0.01055**, pairwise −0.03255 / −0.02397, variance
−0.03273 / −0.02199, logistic −0.08292 / −0.05533, random_ml −0.04649 / −0.03536, random −0.04103 /
−0.02141. Ours shrinks val→test as already documented (2.70× full, 1.50× tail). **The five losing arms
do not shrink** — four of the six val→test movements are under 0.003 — because none of them was fitted
to the validation cohort's held-out metric in the first place. They are wrong on both cohorts by
roughly the same amount, which is what a mis-specified objective looks like as opposed to an overfitted
one.

#### B.2.6 Applying the pre-registered interpretation, without adjustment

The pre-registered cases (§0.8, fixed before the run) were **(c) > (b) > (a)**, **(b) ≈ (c)**, and
**(b) > (c)**. What happened is **(c) ≫ (b) ≈ (a)**.

- **The first branch fired in its strong form on the (c)-versus-(b) leg.** +0.01035 against −0.03041, a
  paired gap of **+0.04076 [+0.03741, +0.04412]** on full and **+0.03043 [+0.02694, +0.03389]** on tail
  — roughly four times the size of our own headline conditioning gain, on the same users, in the same
  table.
- **It did NOT fire on the (b)-versus-(a) leg.** The pre-registration expected a ranking surrogate to
  beat a rating surrogate. It does not: the two are a measured null on both axes. **The mechanism claim
  is airtight in the sense the pre-registration cared about — measured beats surrogate — but the
  ranking-versus-rating ordering that would have made it "(c) > (b) > (a)" is absent.** State the
  ablation's outcome as **(c) ≫ (b) ≈ (a)** and never as "(c) > (b) > (a)".
- **The narrowing branch did not fire.** (b) ≈ (c) is emphatically false, so the novelty is *not*
  narrowed to "the exact reported metric" by this run.
- **The adverse branch did not fire.** (b) > (c) occurs nowhere: not at K=1, 2, 4, 8 or 16, not on full,
  not on tail, not on validation, not on test.

**So the mechanism claim stands.** Hold the tree code, the ternary split rule, min-leaf, the charged
behavioural answers, the certified recommender, the candidate mask, the 1,000-candidate pool, both
cohorts and the 16-question budget all fixed, and change nothing but the scalar a candidate question is
scored by: **the sign of the conditioning gain flips.** Adaptivity does not fail here and it does not
succeed here — the objective decides which.

**And the unpredicted finding is the more useful one for the chapter.** The separating axis is
**measured versus surrogate**, not **ranking versus rating**. A rating-variance criterion from 2011 and
a pairwise ranking criterion from 2013 land on top of each other on a 2026 ranking metric. That is a
sharper diagnosis than "the field used rating-era objectives", and it was not what we expected to find.

#### B.2.7 The logistic arm is the sharpest single exhibit in the file

`logistic` opens on **exactly our question** — root splitter `[1, 114]`, concept 114, answer rate 0.742,
children 2606 / 4814 / 2580, identical to ours in every field — so at K=1 it is nested in our tree *and*
in the static comparator and the difference is identically zero on all 20,000 users. It then degrades
monotonically: −0.00896 at K=2, −0.02777 at K=4, −0.05605 at K=8, −0.08627 at K=16. Its absolute
full@10 **peaks at K=2 (0.18723) and falls to 0.17274 by K=16**, below its own K=1 value of 0.18079.
Under Hu & Yu's own experimental loss, on held-out users with every question charged and folded,
**asking more questions makes the recommendation worse**.

That is the objective-mismatch thesis in one row: a criterion can agree with the metric on the first,
easiest, largest-cohort decision and still walk the interview steadily downhill afterwards. It also
explains the correlation exhibit's shape (§B.5): logistic has regret exactly 0.0 at the root and no
better than random below it.

**The mechanism, stated rather than left mysterious.** The logistic arm stops at depth 1 with 3 leaves,
so fifteen of its sixteen questions come from a leaf tail chosen by the same criterion — and that
criterion is *anti*-correlated with the metric below the root (mean Spearman −0.104, §B.5). The depth is
itself a consequence of the criterion, so this is not a confound; but the chapter should say the
collapse runs through the tail, not through the branch structure.

### B.3 Structural comparison

All six trees, same 1,000-candidate pool, same 10,000 validation users, same min-leaf 200 in `all3`
mode, same 16-question budget. Every field below is read from `splitcrit.json → structure`.

| criterion | depth | nodes | leaves | root splitter | root answer rate | root children (L/D/U) | distinct questions | question slots (item / concept) |
|---|---|---|---|---|---|---|---|---|
| (a) variance (Golbandi) | 5 | 25 | 17 | concept 731 | 0.9158 | 4861 / 4297 / 842 | 145 | 102 / 170 |
| (b) pairwise (Hu & Yu) | 2 | 7 | 5 | concept 511 | 0.6877 | 1209 / 5668 / 3123 | 71 | 63 / 17 |
| (b′) logistic (Hu & Yu, as run) | **1** | 4 | 3 | **concept 114** | 0.7420 | 2606 / 4814 / 2580 | 46 | 40 / 8 |
| **(c) measured NDCG@10 (ours)** | **5** | **22** | **15** | **concept 114** | 0.7420 | 2606 / 4814 / 2580 | **159** | 95 / 130 |
| (r) random | 0 | 1 | 1 | item 1246 | 0.0448 | 267 / 181 / 9552 | 16 | 8 / 8 |
| (r′) random, min-leaf respecting | 7 | 40 | 27 | item 458 | 0.1862 | 639 / 1223 / 8138 | 286 | 190 / 242 |

Leaf-depth histograms: variance `{1:1, 2:4, 3:4, 4:5, 5:3}`; pairwise `{1:2, 2:3}`; logistic `{1:3}`;
ours `{2:8, 3:2, 4:2, 5:3}`; random `{0:1}`; random_ml `{1:1, 2:4, 3:3, 4:6, 5:8, 6:2, 7:3}`.

**Channel mix of the splitters** (`split_channel_by_depth`, item : concept at each depth):

| criterion | d0 | d1 | d2 | d3 | d4 | d5 | d6 | splitter set |
|---|---|---|---|---|---|---|---|---|
| (a) variance | 0:1 | 2:0 | 1:1 | 0:2 | 0:1 | — | — | items 4, 8, 463; concepts 224, 366, 731, 786, 888 |
| (b) pairwise | 0:1 | 0:1 | — | — | — | — | — | concepts 114, 511 |
| (b′) logistic | 0:1 | — | — | — | — | — | — | concept 114 |
| **(c) ours** | 0:1 | 2:1 | 0:1 | 1:0 | 0:1 | — | — | items 19, 83, 514; concepts 36, 114, 283, 334 |
| (r′) random_ml | 1:0 | 1:1 | 0:2 | 0:3 | 0:3 | 0:1 | 1:0 | items 8, 458, 1140; concepts 11, 251, 369, 436, 482, 487, 502, 638, 684, 934 |

**Every criterion that is allowed to choose picks a CONCEPT at the root; only the two random arms pick
an item, and both of them pick an unanswerable one.** That is an independent, criterion-free
corroboration of Act 4's answerability claim: three different objectives — a rating-variance surrogate,
a pairwise ranking surrogate, a pointwise logistic surrogate — and our measured metric all agree that
the first question in a cold interview must be a concept, and they agree on nothing else.

**The criteria do not merely disagree; they share almost nothing.** Splitter overlap, from
`overlap.splitters` (whole tree, both channels):

| | ndcg | pairwise | variance | logistic | random_ml | random |
|---|---|---|---|---|---|---|
| **ndcg** | — | 1 | 0 | 1 | 0 | 0 |
| **pairwise** | 1 | — | 0 | 1 | 0 | 0 |
| **variance** | 0 | 0 | — | 0 | 1 | 0 |
| **logistic** | 1 | 1 | 0 | — | 0 | 0 |
| **random_ml** | 0 | 0 | 1 | 0 | — | 0 |

The single shared splitter in the ndcg / pairwise / logistic triangle is concept 114 in every case — our
root, logistic's only split, and pairwise's depth-1 split on the no-answer branch. **Ours and Golbandi's
share zero splitters.**

Overlap of the *set of questions actually asked* anywhere in the tree (`overlap.questions_asked`, with
the union in brackets so the reader can read a Jaccard):

| pair | shared | union |
|---|---|---|
| ndcg \| variance | 32 | 272 |
| ndcg \| pairwise | 15 | 215 |
| ndcg \| logistic | 7 | 198 |
| ndcg \| random_ml | 45 | 400 |
| ndcg \| random | 2 | 173 |
| pairwise \| variance | 18 | 198 |
| pairwise \| logistic | 4 | 113 |
| pairwise \| random_ml | 19 | 338 |
| pairwise \| random | 3 | 84 |
| variance \| logistic | 2 | 189 |
| variance \| random_ml | 45 | 386 |
| variance \| random | 2 | 159 |
| logistic \| random_ml | 13 | 319 |
| logistic \| random | 1 | 61 |
| random \| random_ml | 5 | 297 |

Our tree shares **32 of 272** questions with Golbandi's, **15 of 215** with Hu & Yu's pairwise and **7 of
198** with their logistic — and it shares **45 of 400** with a coin flip, i.e. *more* with the random
control than with either published criterion. Same 1,000 candidates, same 10,000 users, same states:
different interviews, not rescaled versions of one interview.

**Depth is not virtue.** `random_ml` is deeper (7 vs 5), has more nodes (40 vs 22), more leaves (27 vs
15) and more distinct questions (286 vs 159) than our tree, and loses to the static order by −0.04817
full / −0.03412 tail. Any "deeper tree / more questions = better" reading is refuted by that single row,
and it is exactly why the fair random control was worth running.

**The plain random control collapses, as designed.** Root question answered by 4.5% of users, 9,552 of
10,000 into one branch, min-leaf refuses to split, depth 0. This is the confound that makes `random_ml`
rather than `random` the informative criterion control — and it is also why `random`'s numbers are
comparable to a *static* order rather than to a tree.

### B.4 Controls — all passed

| control | result |
|---|---|
| Canonical snap, validation m=0 | `intercept_val_m0.full@10 = 0.15686538118354684` — reproduces the banked policy-B intercept **exactly**; `tail@10 = 0.02747733123666661` |
| Canonical snap, test m=0 | `intercept_test_m0 = 0.15855464334646144 / 0.028185191294182937` — reproduces `MASK_POLICY_B_MIGRATION.md`'s banked figure, recorded long before this run |
| **Banked-headline assert** | 9 of 10 cells at **0.0**, the tenth at **2.81e-07** (§B.0). The ablation is provably on the tree's ruler |
| Per-user re-assembly (this write-up) | all 120 cohort cells reproduced from `splitcrit_per_user.npz` to **1.17e-10** |
| Static-comparator identity across arms (this write-up) | per-user static vector **bit-identical** across all six arms (max abs diff 0.0) — the arm-versus-arm contrasts of §B.2.4 are genuinely paired |
| Leak check, routing | 0/10,000 users on both cohorts have a held-out target inside the routing lookup (`ndcg_tree.leak_assert`) |
| Leak check, criteria | 0/10,000 users leak: the fold-in rating matrix shares no cell with the targets; `POS_u` and `NEG_u` are target-free for every user and partition each user's fold-in items exactly |
| Nothing sampled (HARD RULE 1) | the pairwise loss summed over **137,591,707** fold-in pairs, all of them; the metric is over the full 18,359-item catalogue |
| Shuffle control, `random` | degenerates to depth 0 (root answered by 4.5%, 9,552 into one branch) — the confound that makes `random_ml` the informative control |
| Shuffle control, `random_ml` | depth 7, 27 leaves, 286 distinct questions — deeper and more varied than every real criterion, which refutes any "deeper = better" reading |
| Criterion maths | gradients and diagonal Hessians against central finite differences; Taylor gain against brute-force minimisation; Eq. (4) against brute-force per-child recomputation — all pass (§B.1) |
| Untruncated pairwise loss | matches a brute-force double loop over all (target, non-target) pairs on 200 random cases, including masked candidates and ties |

### B.5 Criterion vs MEASURED NDCG at every node — the free ground truth

The `ndcgtree` build persisted a CELF heap sidecar per node. The seed entries of each sidecar
(`upd == 0`) are the **exact measured cohort-mean NDCG@10 marginal gains** of every candidate at that
node's prefix, over that node's own users — 22 nodes × ~1,000 candidates of ground truth for the
quantity our criterion measures and the other criteria approximate, at states the real tree actually
visited, at zero additional NDCG compute.

For each node and each criterion we report Spearman and Kendall against the measured gain; whether the
criterion's argmax is the measured argmax; the **regret** (measured NDCG gain forgone by following that
criterion instead of the metric) and its user-weighted mean; the rank the measured argmax receives under
the criterion; and top-10 recall. Controls: a permuted-criterion Spearman, and the expected regret of a
uniform random pick — a criterion worth having must sit well below the latter.

This is the direct, quantitative form of Act 2's "1 of 21 rules optimises the reported quantity"
(`BASELINE_OBJECTIVE_AUDIT.md`): it turns a description of what each rule optimises into a measurement
of how far that surrogate is from the thing being reported, at real states.

#### RESULT — the surrogates are barely distinguishable from guessing

| criterion | mean Spearman | nodes with rho>0 | argmax == NDCG argmax | mean top-10 recall | **user-weighted regret** |
|---|---|---|---|---|---|
| (a) variance (Golbandi) | **+0.010** | 13/22 | **1/22** | 0.03 | **+0.01664** |
| (b) pairwise rank loss (Hu & Yu) | **−0.060** | 10/22 | **0/22** | 0.01 | **+0.01738** |
| (b′) logistic loss (Hu & Yu, as run) | **−0.104** | 9/22 | **1/22** | 0.03 | **+0.01282** |
| *control: uniform random pick* | — | — | — | 0.10 by construction | **+0.01955** |

Read the last column first. **Following any of these three criteria forgoes between 66% and 89% of the
NDCG@10 that a uniform random pick would forgo.** Not 5%, not 20% — they recover a third of the way from
random to optimal at best. The rank correlations are ~0 and two of the three are *negative*, so these
are not weak-but-aligned surrogates; at the states this tree actually visits they are close to
uninformative about the quantity the field reports. Golbandi's criterion picks the measured argmax at 1
node of 22; Hu & Yu's pairwise loss at **0 of 22**; top-10 recall is 0.01–0.03 against 0.10 for a random
draw, i.e. **the pairwise criterion's top ten candidates overlap the metric's top ten less often than
chance**.

This is the cleanest exhibit in the file, and it is free: the ground truth was already banked.

#### The root is a real and instructive exception

| criterion at the ROOT | its argmax | measured NDCG gain of its pick | regret | rank it gives the metric's argmax | Spearman |
|---|---|---|---|---|---|
| variance | concept 731 | +0.00414 | +0.01635 | **250** of 1,000 | −0.212 |
| pairwise | concept 511 | +0.00576 | +0.01473 | **75** of 1,000 | −0.149 |
| **logistic** | **concept 114** | **+0.02049** | **0.00000** | **1** | **+0.187** |
| measured NDCG@10 (ours) | concept 114 | +0.02049 | — | 1 | — |

**Hu & Yu's actually-run loss picks exactly the right first question and then loses the plot.** At the
root the logistic criterion's argmax *is* the metric's argmax — concept 114, the same question our tree
and the policy-B static order both open with — with regret identically zero and top-10 recall 0.3. One
level down its mean regret is no better than the others. So the honest statement is not "their criterion
is bad", it is **"their criterion agrees with the metric where the cohort is largest and the signal is
strongest, and stops agreeing exactly where an adaptive method has to earn its keep."** That is a more
interesting claim than a flat dismissal and it is what the numbers support.

Note also what this does to the variance criterion: it ranks the metric's own best question **250th of
1,000** at the root. The gap between a rating-variance surrogate and a ranking metric is not a
rescaling.

### B.6 Separation (1) isolated — COMPLETE, and it moves the claim

`pairwise_heldout_root.py` changes **only the loss**. At the root state, for each candidate, it folds
once and computes from the *same* score matrix both full NDCG@10 (through the same `ndcg10_from_scores`
funnel, same mask contract) and the held-out pairwise rank loss in its 0-1 (AUC) form over the full
18,359-item catalogue — untruncated, undiscounted, ties counted as errors. Same fold, same state, same
candidates, same protocol; only the scalarisation differs.

**The run COMPLETED at 05:46 on 2026-08-13** (`experiments/battery/pwheldout_root.json`, `citable =
true`, `n_candidates = 1000`, `n_users = 10000`, `state = "ROOT (empty prefix)"`). The §B.6 numbers
previously in this file were a **140-candidate, item-heavy partial** and are superseded; the partial's
Spearman was +0.637 against the full pool's +0.784, so the partial *understated* the agreement.
Coverage is now the whole pool and channel-balanced: **500 items and 500 concepts.**

| at the root, 1,000 / 1,000 candidates | value |
|---|---|
| base full NDCG@10 (m=0) | 0.15686538118354684 — the canonical intercept, again exactly |
| base pairwise rank loss (m=0) | 0.0428786752243029 |
| banked-NDCG gate: max abs deviation of every recomputed NDCG gain from the banked root heap | **0.0** on all 1,000 candidates |
| **Spearman(measured NDCG@10 gain, measured pairwise-loss reduction)** | **+0.7838** |
| Kendall | +0.5919 |
| Spearman, items only (n = 500) | +0.7317 |
| Spearman, concepts only (n = 500) | +0.7381 |
| NDCG argmax | **concept 114** |
| pairwise-loss argmax | **concept 704** — `argmax_agrees = false` |
| measured NDCG gain of the pairwise argmax | 0.015899 (best available: 0.020491) |
| **regret of following the measured pairwise loss** | **+0.004591** |
| expected regret of a uniform random pick | +0.016082 |
| rank the pairwise loss gives the NDCG argmax | **2** of 1,000 |
| top-10 recall | 0.40 |

**This relocates the mechanism, in our favour but not where we expected.** Put it beside the
*training-loss* pairwise criterion at the same node (§B.5): that one has Spearman **−0.149**, ranks the
metric's argmax **75th** of 1,000, and carries regret **+0.01473** — 92% of what a uniform random pick
forgoes, i.e. indistinguishable from guessing. The *measured* pairwise loss, on the same pairwise
objective at the same state, has Spearman **+0.784**, ranks it **2nd**, and carries regret **+0.00459**,
about **29%** of random.

So the two separations are not equal partners:

- **Separation (2) — training loss versus the measured expected value of asking — is doing almost all of
  the work.** Hu & Yu's Taylor-approximated training-loss reduction is nearly uninformative about
  held-out ranking; the same loss *measured on held-out targets* is a good proxy for it.
- **Separation (1) — truncation and positional discounting — is real but second-order.** An untruncated,
  undiscounted pairwise loss, measured properly, gets you most of the way to NDCG@10 — but not all of
  it: it still forgoes +0.00459 of NDCG@10 at the root, it disagrees on the argmax (concept 704 rather
  than concept 114), and its top ten overlap the metric's top ten only 40% of the time.

That is a **more precise and more defensible** claim than "ranking losses are the wrong objective", and
it is the one the chapter should make. It also has a constructive consequence for workstream E: if the
measured pairwise loss tracks NDCG@10 at ρ ≈ 0.78 without needing a top-k sort, it is a candidate cheap
surrogate *reward* for the parametric policy — provided the final selection is still made on the
reported metric.

**Two limits.** This is **one state** — the root, the largest and easiest cohort in the tree, and the
one state where §B.5 already showed the surrogates do best. Nothing here licenses a claim about ρ deeper
in the tree, and the `pairwise_heldout` **tree** arm (their loss under our protocol, all the way down)
is still specified and unrun; it costs a tower fold per candidate per node, i.e. what arm (c) cost
(37,191 s = 10.3 h).

### B.7 What did NOT run in Part B, listed as unrun rather than omitted

- **`pairwise_heldout` as a full tree arm.** Specified in §0.4(ii), QUEUED, never run. It is the clean
  single-factor test of separation (2) at every state; §B.6 is its root-only substitute.
- **The `nonempty` min-leaf mode.** Implemented, unrun. Golbandi's own selector constrains only the like
  and hate branches (`golbandi_tree_selector._best_split`: `(cL >= min) & (cH >= min)`), whereas our
  `all3` mode also requires the DID-NOT-ANSWER branch to clear 200. That rule penalises criteria drawn
  to 100%-answerable concepts — which the Hu & Yu criteria are — by refusing their splits outright. It
  is held identical across all six arms by design, and arm (c)'s tree cannot be regrown (author ruling),
  so this is a **robustness check we owe, not a defect in the comparison**. Same interaction as TODO
  1.18 / plan A10.
- **Seed variation on the two random arms.** One seed (98765) each. The structural claims (`random`
  degenerates, `random_ml` goes deeper than every real criterion) are single-draw facts.

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
> the top-k most popular items. The comparator this chapter constructs instead — a static order built
> by greedily maximising the reported metric — is not tested against in that paper, nor in any of the
> twenty-one selection rules our objective audit covers, nor anywhere in the modern conversational-,
> bandit- or LLM-elicitation lines. **We say "since the reported metric moved to truncated top-k
> ranking" and not "never", because Golbandi, Koren and Lempel (WSDM 2011) did exactly this comparison
> in the rating era: their interview tree was tested against GreedyExtend (CIKM 2010), a static seed
> set built by greedily minimising RMSE with the recommender refit on seed-restricted data — the
> reported metric of its time. That precedent is evidence *for* the objective-mismatch frame, and we
> present it ourselves rather than leave it to a reviewer.**

### C.3 Reconciling Part A with Part B — what each one licenses

Part A and Part B measure different things on different rulers and they are allowed to disagree. This
section says which sentence rests on which, so that no reader has to guess and no reviewer can pick the
half they prefer.

| | **Part A** | **Part B** |
|---|---|---|
| whose protocol | **theirs** | **ours** |
| dataset | ML-1M, 6,040 users | ML-25M, canonical Liang split |
| metric | MAP(r≥4) over **10 sampled negatives** per positive, untruncated, undiscounted | **full and tail NDCG@10** over the whole 18,359-item catalogue, nothing sampled |
| recommender | our re-implementation of their fMF-lineage tree, d=20, fitted per run | frozen certified pair `t2i26c_ep31` + `concept_cv3_ep08` |
| cohort | 1,510 test users, single seed | 10,000 test users grown on 10,000 disjoint validation users |
| uncertainty | **none** — cell-level MAP only, no per-user APs, no bootstrap available without a re-run | paired percentile bootstrap, 10,000 resamples, seed 98765, on every cell |
| static comparator | **FixQ = top-k popular as the interview** (theirs) | a static order built by **maximising the reported metric** (ours) |
| budgets | depths 3 and 5 (FixQ also 8 and 10) | K ∈ {1, 2, 4, 8, 16} |
| what it can settle | whether *their published result* reproduces, and what it is contingent on | whether *their criterion*, transplanted, beats a metric-optimised static order on our ruler |

**What Part A licenses.**

1. **Their arm ordering reproduces** — Warm-MF > EIP > FixQ at every depth measured — and so does the
   sign and order of magnitude of their headline EIP − FixQ gap on the protocol as literally stated
   (+0.0367 at d=3, +0.0343 at d=5 on MAP(r≥4), against their digitised +0.0205 / +0.0167).
2. **Their absolute level does not reproduce**, by 0.19–0.24 MAP, and three candidate explanations
   (metric implementation, negative pool, latent capacity) were tested and excluded. **Reported as
   unexplained.**
3. **Their depth trend is contingent on an answerability asymmetry in their own protocol.** A training
   user's interview answers come from all their ratings; a test user's come from their first half only.
   Test users answer the top-10 popular questions at roughly half the rate of training users (0.283 vs
   0.575 at rank 1). Under the protocol as stated, MAP *falls* with depth (FixQ .4910 → .4729 over
   3/5/8/10); once training and test users are made equally answerable it *rises* monotonically (.5164 →
   .5312), matching their figure's shape.
4. **CORRECTED, and this file previously said the opposite: their criterion survives a shuffle control
   once answerability is matched.** EIP − RandQ is +0.01365 (d3) → **+0.00288** (d5) under the
   asymmetric protocol — a collapse — but +0.00661 (d3) → **+0.00866** (d5) under symmetric answers,
   i.e. it does not decay. **The depth-5 collapse is an artefact of the asymmetry, not a property of the
   criterion.** The strong sentence ("their criterion does not beat a random tree past three questions")
   is **withdrawn and must not be written**.
5. **But their headline was largely comparator handicap.** EIP − FixQ shrinks to **+0.0048 at both
   depths** under symmetric answers — under a quarter of their claimed "2% averagely" — because FixQ is
   the arm that gains most from the fix (.4910 → .5164 at d=3). And RandQ crosses **below** FixQ
   (−0.0018 / −0.0039), so the contribution decomposition reverses: the asymmetric reading says two
   thirds of EIP's edge is "having a tree at all", the symmetric reading says all of it is the
   criterion. **The paper does not say which reading it intends.**
6. **They never instantiated the pairwise rank loss.** It appears exactly once, in the "square loss,
   logistic loss, pairwise rank loss, etc." list attached to Eq. (1) — the *basic* MF objective, one
   subsection before the tree exists — and §3.2.1 confirms the ranking experiments ran the **logistic**
   loss ("since fMF cannot handle the logistic loss"). It is never instantiated, tuned, ablated or
   evaluated, by them or by any of their seven citers in thirteen years.

**What Part A does NOT license.** Any magnitude quoted as if interval-backed (single seed, 1,510 users,
no CIs, against a ~0.014 spread between most-popular and the best cell — the *directions* are the
findings and the magnitudes are indicative); anything past **five** questions on either ruler (the
depth-8/10 EIP and RandQ cells never returned); any claim about Netflix or Yahoo! Music (not held, not
replicated); any claim about their RMSE panel or fMF (never run).

**What Part B licenses, and it does not depend on Part A at all.** Part B runs their *criterion* inside
our harness on our ruler, so nothing unresolved in Part A — the level gap, the depth trend, the missing
intervals — touches it. It licenses: with the tree code, split rule, min-leaf, answers, recommender,
mask, pool, cohorts and budget all held fixed and **only** the scalar a question is scored by varying,
our measured-metric criterion is the only one of six that beats a metric-optimised static order, and
both published surrogates lose to it by roughly three times the margin we win by, with every interval
excluding zero.

**Where they agree.** Both find that Hu & Yu's criterion is *real but small*. Part A: it beats a
coin-flip tree by +0.0087 on their own ruler once the protocol is made symmetric. Part B: it beats the
fair shuffle control `random_ml` by **+0.01776 [+0.01427, +0.02118]** full on ours. Neither result is a
dismissal of the criterion. Both find that the criterion is *not* what carries their headline: Part A
because the headline shrinks to +0.0048 against a symmetric FixQ, Part B because the criterion loses
outright once the static comparator is built by maximising the reported metric instead of by
popularity.

**Where they disagree, and it is instructive.** On *their* protocol their criterion beats their static
comparator; on *ours* it loses to ours. The single variable that changed is the comparator's
construction, not the criterion — which is the chapter's whole thesis, arriving here as a by-product
rather than as an argument. Part A's FixQ is top-k popular; Part B's static order is greedy-maximised
NDCG@10. Against the first, an interview tree with a surrogate split wins. Against the second, it loses
by −0.03041.

**One incidental Part A finding that belongs in Act 1, not here.** On our ML-1M ruler, FixQ is *worse
than simply ranking by popularity* (.4910 vs .5129 at d=3) — asking the top-k popular items and folding
the answers back in makes it worse than not asking. EIP clears most-popular (.5277). That is our Act 1
pattern appearing inside their own baseline set, on their own dataset.

### C.4 Verdict on what novelty survives — written as a CONJUNCTION, never as a technique

Three prior-art facts bound the claim, and all three are ours to state first rather than a reviewer's to
find (`docs/results/POSITION_VS_MODERN_CRS.md`):

1. **Hu & Yu (RecSys 2013) offered a rank loss inside an interview tree's split** — so *"first ranking
   objective in an interview tree's split"* is **not available and is not claimed anywhere.** They never
   ran it (§C.1), and The FacT (SIGIR 2019) has a BPR term in an interview tree's split, so 2019 is the
   floor even for "instantiated".
2. **Golbandi (WSDM 2011) did compare an adaptive tree against a metric-optimised static order** —
   GreedyExtend (CIKM 2010), built by greedily minimising RMSE with the recommender refit on
   seed-restricted data, which *was* the reported metric of its era. So *"nobody ever compared
   adaptivity against a metric-optimised static order"* is **FALSE and must never be written.** It is
   also evidence *for* our frame and should be presented as such.
3. **Bilgic & Bennett (SIGIR 2012) put DCG@10 / NDCG@10 inside an acquisition function** (arms
   `dcgELOq`, `ndcgELOq`), and greedy forward selection with an NDCG wrapper is ordinary in
   learning-to-rank feature selection. So *"nobody has used a truncated, discounted ranking metric as a
   selection criterion"* is **FALSE and must never be written.**

What survives is a **conjunction**. No single conjunct is new; the conjunction is, and it is the unit
that must be claimed:

> A question-selection criterion that is (i) the **measured value of the answer after it arrives** —
> improvement, not regret over what the current model might lose; (ii) **cohort-mean held-out NDCG@10**,
> the exact truncated, position-discounted quantity the field reports, computed rather than
> approximated; (iii) evaluated over the **full 18,359-item catalogue** with no negative sampling; (iv)
> scoring **a question asked of a cold user**, not a training label acquired for a ranker; (v) used to
> grow an **adaptive interview** on one cohort and reported on a **disjoint held-out cohort**; and (vi)
> tested against a **static order built by maximising that same quantity** rather than by popularity.

And the scoped field-level claim, which is what the chapter actually argues:

> **Since the reported metric moved to truncated top-k ranking, no adaptive selector has been tested
> against a static order built by maximising it.** Golbandi's RMSE-era comparison is the exception that
> proves the rule and we cite it as such.

**What the ablation adds, which is measurement rather than argument:**

> An interview tree whose split criterion is the measured held-out value of asking beats a static order
> built by maximising that same quantity by **+0.01035 [+0.00749, +0.01328] full / +0.00705 [+0.00394,
> +0.01022] tail** at K=16 on 10,000 held-out users. **Substituting the published surrogates into the
> identical machinery reverses the result: Hu & Yu's pairwise rank loss loses by −0.03041, Golbandi's
> rating variance by −0.03061, and Hu & Yu's own experimental logistic loss by −0.08627.** The field's
> null about adaptive elicitation is a statement about objectives, not about adaptivity.

Three further claims the ablation licenses, none of which was available before this run:

1. **The ranking-versus-rating axis is not the one that matters.** Golbandi's rating-variance criterion
   and Hu & Yu's pairwise rank loss are a measured null against each other on both axes: **+0.00020
   [−0.00331, +0.00371]** full and **−0.00092 [−0.00448, +0.00277]** tail (paired, K=16, test).
   Upgrading a rating surrogate to a ranking surrogate bought **nothing measurable**. What separates the
   methods is *measured versus surrogate*. This was **not** predicted by the pre-registration, which
   expected (c) > (b) > (a); the (b) > (a) leg did not fire.
2. **The surrogates are close to uninformative about the reported metric at real states.** Over 22 nodes
   × ~1,000 candidates of exact banked ground truth, following any of them forgoes 66–89% of the NDCG@10
   a uniform random pick forgoes; mean rank correlations are ~0 and two of three are negative; Hu & Yu's
   pairwise loss picks the metric's argmax at **0 of 22** nodes and its top ten overlap the metric's top
   ten *less often than chance*.
3. **The separation is training-loss-versus-measured-value, not truncation.** At the root over all 1,000
   candidates, the *measured* pairwise loss tracks NDCG@10 at Spearman **+0.784** with regret +0.00459
   (29% of random), while the *Taylor-approximated training-loss* pairwise criterion at the same node
   has Spearman **−0.149** and regret +0.01473 (92% of random). Truncation and discounting are real but
   second-order: even measured properly, the untruncated loss disagrees on the argmax and recovers only
   40% of the metric's top ten.

**What must travel with the claim, honestly stated.**

- **The comparison is on OUR ruler.** It shows that *their criterion, transplanted into our harness*,
  loses. It is **not** a refutation of their published result on their own protocol; Part A is the
  separate exercise of establishing what that result is, and §C.3 says which sentence rests on which.
- **Their criterion is the cheap one, and that is a genuine advantage we do not have.** Their Taylor
  expansion makes gradient and Hessian candidate-independent, so 1,000 candidates cost one pass (~11 s
  at 10,000 users over 137,591,707 pairs); the variance criterion costs ~17 s. Our exact objective pays
  a tower fold per candidate and cost **37,191 s** for one tree — a factor of roughly 3,000 per node.
  If the chapter claims our objective is better it must concede in the same breath that theirs is orders
  of magnitude cheaper, **and that this is precisely why the parametric policy (workstream E) is the
  method rather than the tree.**
- **The `logistic` arm's collapse runs through its leaf tails**, not through its branch structure: it
  stops at depth 1, so fifteen of sixteen questions come from a tail chosen by a criterion
  anti-correlated with the metric. The depth is itself a consequence of the criterion, so this is a
  mechanism, not a confound — but state it rather than leave it mysterious.
- **The min-leaf rule is `all3`**, applying to the DID-NOT-ANSWER branch as well, which penalises
  criteria drawn to fully answerable questions — and the Hu & Yu criteria are so drawn. Held identical
  across all six arms by design; the `nonempty` robustness mode is implemented and **unrun** (TODO 1.18
  / plan A10).
- **The published criteria are not worse than a random static order on full**, and the chapter should
  not imply they are: pairwise − random = +0.00893 [+0.00575, +0.01207] and variance − random = +0.00873
  [+0.00520, +0.01228], both separating. On the **tail** at K=16 both are nulls against it. Report both
  halves.

---

### C.5 Chapter-ready prose

Lift these blocks into `chapterB_v2.tex` as they stand; every number in them is quoted from
`experiments/battery/splitcrit.json` or `pwheldout_root.json` and cross-checked in §B.0 and §B.4. The
ablation belongs in **Act 3** (nearest rival); the four separations belong in **Positioning**; the
comparator and objective concessions belong wherever the claim is first stated, and must not be
deferred to a limitations paragraph.

#### C.5.1 The ablation (Act 3)

> If the objective is what decides whether adaptivity pays, then swapping the objective inside a fixed
> tree — and changing nothing else — should decide it. We ran that experiment. The tree code, the
> ternary like / dislike / did-not-answer split rule, the minimum leaf size of 200, the charged
> behavioural answer model, the frozen certified recommender, the candidate mask, the 1,000-candidate
> pool of 500 popularity-ranked items and 500 answerable concepts, the 10,000 validation users the tree
> is grown on, the 10,000 disjoint test users it is reported on, and the sixteen-question budget are all
> held fixed. The only thing that varies is the scalar by which a candidate question is scored.
>
> Six criteria were run. Ours is the measured, cohort-mean, held-out NDCG@10 that follows from actually
> asking the question and folding the answer. Golbandi's is the within-node variance of rating
> prediction error. Hu & Yu's framework admits a pairwise rank loss, scored by their second-order Taylor
> split gain; their experiments in fact ran a pointwise logistic loss, and both are included so that no
> reader has to wonder which version we chose. Two shuffle controls complete the set: a uniform draw
> over the whole pool, and a uniform draw restricted to candidates whose split clears the minimum leaf
> size — the fair control, which holds the tree machinery fixed and destroys only the choice of
> question.
>
> At sixteen questions on held-out users, measured against a static order built by greedily maximising
> the same metric, our criterion gains $+0.01035$ full NDCG@10 (95\% paired-bootstrap CI $[+0.00749,
> +0.01328]$) and $+0.00705$ on the long tail ($[+0.00394, +0.01022]$). Hu & Yu's pairwise rank loss
> \emph{loses} $-0.03041$ ($[-0.03368, -0.02718]$); Golbandi's rating variance loses $-0.03061$
> ($[-0.03415, -0.02714]$); Hu & Yu's own experimental logistic loss loses $-0.08627$ ($[-0.08977,
> -0.08258]$). The min-leaf-respecting shuffle control loses $-0.04817$ and the plain shuffle control
> loses $-0.03934$. Every interval excludes zero, and exactly one of the six criteria places the
> adaptive interview above the static order. Changing nothing but the number a question is scored by
> flips the sign of the conditioning gain.
>
> Two results inside that table were not predicted and are worth more than the headline. First, the
> ranking-versus-rating axis is not the one that separates these methods: Hu & Yu's pairwise rank loss
> and Golbandi's rating variance are indistinguishable from one another on our metric, differing by
> $+0.00020$ full ($[-0.00331, +0.00371]$) and $-0.00092$ tail ($[-0.00448, +0.00277]$) in a paired
> comparison — the only null in the whole matrix. Upgrading a rating surrogate to a ranking surrogate
> bought nothing measurable. What separates the criteria is whether the quantity is \emph{measured} or
> \emph{approximated}, not whether it concerns ranks or ratings. Second, depth is not virtue: the
> min-leaf-respecting shuffle control grows deeper (7 versus 5), wider (27 leaves versus 15) and more
> varied (286 distinct questions versus 159) than the criterion that wins, and still loses by
> $-0.04817$. No argument from tree size survives that row.
>
> The logistic arm is the single sharpest exhibit. It opens on exactly our question — concept 114, with
> an identical ternary partition, so that at one question its difference from our tree and from the
> static order is identically zero on all 20,000 users — and then degrades monotonically: $-0.00896$ at
> two questions, $-0.02777$ at four, $-0.05605$ at eight, $-0.08627$ at sixteen. Its absolute NDCG@10
> peaks at two questions and falls below its own one-question value by sixteen. Under the loss Hu & Yu's
> ranking experiments actually ran, on held-out users, with every question charged and every answer
> folded, asking more questions makes the recommendation worse. A criterion can agree with the metric on
> the first, largest, easiest decision and still walk the interview downhill afterwards.
>
> The mechanism behind that is directly measurable, and it is free: the tree's build banked, at every
> node it visited, the exact measured NDCG@10 marginal gain of all thousand candidates. Across 22 nodes
> we can therefore ask how far each surrogate sits from the quantity the field reports, at states a real
> interview actually reaches. Following Golbandi's criterion forgoes 85\% of the NDCG@10 that a uniform
> random pick forgoes; Hu & Yu's pairwise loss, 89\%; their logistic loss, 66\%. Mean rank correlations
> against the measured gain are $+0.010$, $-0.060$ and $-0.104$. Golbandi's criterion selects the
> metric's own best question at 1 node of 22, Hu & Yu's pairwise loss at 0 of 22, and the pairwise
> criterion's top ten candidates overlap the metric's top ten \emph{less often than a random draw
> would}. These are not weak-but-aligned surrogates; at the states that matter they are close to
> uninformative about the reported quantity.
>
> Finally, the failure is attributable to the approximation rather than to the shape of the loss. At the
> root, over all thousand candidates, the \emph{measured} held-out pairwise rank loss — untruncated,
> undiscounted, over the full catalogue — tracks measured NDCG@10 at Spearman $+0.784$ and forgoes only
> $+0.00459$ of NDCG@10, about 29\% of what a random pick forgoes. Hu & Yu's Taylor-approximated
> \emph{training}-loss version of the same objective, at the same node, correlates at $-0.149$, ranks
> the metric's best question 75th of 1,000, and forgoes $+0.01473$ — 92\% of random. Truncation and
> positional discounting are real but second-order: even measured properly, the untruncated loss picks a
> different first question and recovers only 40\% of the metric's top ten.

#### C.5.2 Scope sentences that must accompany the ablation

> This is a comparison on our ruler. It establishes that these criteria, transplanted into our harness
> and given every advantage we give our own, lose to a static order built by maximising the reported
> metric; it is not a refutation of the published results on their own protocols. It also carries a cost
> asymmetry that runs against us: the Taylor construction makes gradient and Hessian
> candidate-independent, so all thousand candidates are scored in a single pass of about eleven seconds,
> where our exact objective pays a recommender fold per candidate and cost 37,191 seconds to grow one
> tree. That is roughly three orders of magnitude per node, it is a genuine advantage of their method
> over ours, and it is exactly why the parametric policy rather than the tree is the method this chapter
> proposes.
>
> One more number belongs here for symmetry. The published criteria are not worse than a randomly chosen
> static order: on full NDCG@10 they beat the degenerate shuffle control by $+0.00893$ and $+0.00873$,
> both separating. On the long tail at sixteen questions neither separates from it ($-0.00212$
> $[-0.00540, +0.00127]$ and $-0.00120$ $[-0.00486, +0.00245]$). Both halves are reported.

#### C.5.3 The Hu & Yu separations (Positioning) — see §C.2 for the four-separation block

§C.2 carries the four separations as continuous prose and is lifted unchanged. The two sentences that
must appear near it, and that cannot be softened:

> We do not claim to be the first to put a ranking objective inside an interview tree's split: Hu & Yu
> state a general split loss that admits a pairwise rank loss, and The FacT (SIGIR 2019) instantiates a
> BPR term in one. Nor do we claim that a truncated, position-discounted ranking metric has never been
> a selection criterion: Bilgic and Bennett (SIGIR 2012) run DCG@10 and NDCG@10 inside an acquisition
> function, and greedy forward selection with an NDCG wrapper is standard practice in learning-to-rank
> feature selection. Nor, finally, do we claim that adaptivity has never been tested against a
> metric-optimised static order: Golbandi, Koren and Lempel did precisely that against GreedyExtend in
> 2011, when the reported metric was RMSE. What has not been done — and what this chapter does — is the
> conjunction: an adaptive interview whose split criterion is the \emph{measured} value of a question
> asked of a cold user, scored as cohort-mean held-out NDCG@10 over the full catalogue with nothing
> sampled, grown on one cohort, reported on a disjoint one, and tested against a static order built by
> maximising that same quantity.
>
> One curiosity of the replication is worth a sentence. Hu & Yu's pairwise rank loss is offered rather
> than run: it appears exactly once, in a list of admissible losses attached to the basic factorisation
> objective, one subsection before their tree exists, and their ranking experiments use a pointwise
> logistic loss instead. In thirteen years and seven citations, no one has instantiated it. We ran both.

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
| 04:27 | Symmetric depth-3 cells (`eip_logistic_sym_d3`, `randq_sym_d3`) | EIP−RandQ **+0.00661** |
| 05:46 | **Separation-(1) root probe COMPLETE — all 1,000 candidates, 500 items / 500 concepts** (`pwheldout_root.json`, `citable=true`) | §B.6 rewritten from PRELIMINARY to FINAL. Spearman **+0.784** (the 140-candidate partial said +0.637 and understated it), regret **+0.00459** vs +0.01608 random, NDCG argmax concept 114 vs pairwise argmax concept 704 (`argmax_agrees=false`), rank 2 of 1,000, top-10 recall 0.40, banked-NDCG gate 0.0 on all 1,000. |
| 2026-08-13 (finalisation) | **B1 FINALISED for the chapter.** Part B written up in full from the artefact: six arms (the two shuffle controls were computed but unreported), per-budget paired CIs on both axes and both cohorts, structure + channel mix + overlap matrices, arm-versus-arm paired contrasts recomputed from `splitcrit_per_user.npz`, Part A/Part B reconciliation (§C.3), novelty as a conjunction (§C.4), chapter-ready prose (§C.5). | New facts: **(b) vs (a) is an interval-backed null on both axes**; **on the tail at K=16 neither published criterion separates from a random static order**; **logistic loses to BOTH shuffle controls decisively**; the tail metric is over **9,281 / 9,217** users, not 10,000, and the artefact's `n_users` field mislabels this. |
| ~05:2x | **Symmetric depth-5 cells returned — the 2×2 is closed.** `eip_logistic_sym_d5` .52717, `randq_sym_d5` .51851 | **EIP−RandQ +0.00866 — it does NOT collapse under symmetric answers.** §A.6 finding (1) is CORRECTED: the depth-5 collapse (+0.0029) is an artefact of the answerability asymmetry, so the strong "does not beat random past three questions on either reading" sentence is withdrawn and not written. Two new findings fell out: EIP−FixQ shrinks to +0.0048 under symmetric (a quarter of their reported gap), and RandQ crosses BELOW FixQ, reversing the contribution decomposition. Both jobs then exited cleanly; box free. |

### Part A: a red flag that must be resolved before any snap is reported

> **HISTORICAL — RESOLVED. Kept because the resolution is the finding.** This entry was written at
> ~23:20 on 2026-08-12, before the d sweep and before FixQ ran. Both problems named below were closed:
> the latent dimension was raised from d=10 to d=20 and the fit was made monotone by majorised
> coordinate descent (§A.2), after which EIP does clear most-popular (.5277 vs .5129) and FixQ does not
> (.4910) — see §A.3. The absolute level gap remains and is reported as unexplained. **Read §A.2–§A.6
> for the current Part A numbers; nothing in this box is a live claim.**

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
