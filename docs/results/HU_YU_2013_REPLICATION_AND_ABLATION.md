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

See `external_literature/findings/hu_yu_2013_interview_process_learning.md` §"CORRECTED" for the table.
Netflix and Yahoo! Music are **not** replicated — we do not hold those datasets. ML-1M only.

### A.2 The snap

*(pending)*

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

### B.2 The three trees side by side

*(pending)*

### B.3 Structural comparison

*(pending)*

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

*(results pending)*

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

*(pending — written once B lands, and written to whichever of the three pre-registered outcomes
occurs)*

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
