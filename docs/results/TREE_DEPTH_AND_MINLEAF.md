# TREE DEPTH AND THE MIN-LEAF RULE — why the NDCG tree stopped at depth 5

TODO **1.18 + 1.18b** · plan item **A10** · 2026-08-12 · code `src/instrument/tree_minleaf_audit.py` ·
numbers `experiments/battery/tree_minleaf_audit.json` · log
`experiments/battery/tree_minleaf_audit.log`.

**NO RE-SCORING. The recommender was never run.** Everything below is a lookup or an arithmetic
consequence of one. Node membership comes from the tree's own per-user npz, node records from
`ndcgtree.json`, per-candidate gains from the banked CELF heap sidecars, and the root's candidate
table from `ndcgtree_PREFLIGHT.json`. The one thing no artifact banks is the ternary answer label of
*every* candidate for *every* user — needed to know which candidates were feasible at nodes below the
root — so those labels are recomputed here from the data (graded validation fold-in for items, the
certified signed-answer path for concepts). No tower is loaded, no NDCG is computed, and the
recomputation is **gated** before any number is used:

| gate | result |
|---|---|
| G1 — all 1,000 root candidates reproduce the banked preflight ternary sizes **and** answer rate exactly | **1000 / 1000** |
| G2 — all 22 nodes reproduce their banked `child_sizes` and `answer_rate` for the chosen splitter exactly | **22 / 22** |
| G3 — the 15 banked leaf row sets partition the 10,000-user cohort | **pass** |

The labels are therefore the harness's own labels, bit for bit.

---

## VERDICT

> **Every one of the 15 stops was a floor stop with a strictly positive gain already in hand.** Not
> one node stopped because no question helped: the smallest argmax gain among stopped nodes is
> **+0.00672** and the largest **+0.04251**. **8 of the 15 stops (3,125 users, 31.2% of the cohort)
> were decided by node SIZE before any question was looked at**; the other **7 (6,875 users, 68.8%)
> were SHAPE stops**, and in **8 of the 15** the branch that broke the floor was the
> **did-not-answer** branch of a concept **92–100% of the node could answer**.
>
> Realised depth measures the stopping rule, not the data. The chapter may not read depth 5 as
> "conditioning stops paying" — and equally may not claim a deeper tree would pay.

---

## 1. Why the rule is a constraint before it is a measurement

The tree splits three ways (LIKE / DISLIKE / DID-NOT-ANSWER) and refuses a split if any child holds
fewer than `MIN_LEAF = 200` users. Two arithmetic consequences follow immediately, with no data:

* a strictly ternary split needs **n ≥ 600**;
* a *binary-effective* split — a question 100% of the node can answer, so the did-not-answer child is
  empty and the floor never applies to it — needs **n ≥ 400**.

Restated as a single inequality, the floor is a **window on the answer rate**. At a node of size *n*
a candidate needs its did-not-answer child over the floor, i.e. answer rate ≤ 1 − 200/n, and both
answered children over it, which requires at least answer rate ≥ 400/n (and exact balance to achieve
it). The window is `[400/n, 1 − 200/n]` and it **closes at n = 600**:

| n | window on the answer rate | feasible candidates found (of ~1,000) |
|---|---|---|
| 10,000 (root) | [0.040, 0.980] | 876 |
| 2,580 | [0.155, 0.922] | 267 |
| 1,785 | [0.224, 0.888] | 168 |
| 1,256 | [0.318, 0.841] | 99 |
| 1,010 | [0.396, 0.802] | 16 |
| 723 | [0.553, 0.723] | 1 |
| 702 | [0.570, 0.715] | 1 |
| 660 | [0.606, 0.697] | **0** |
| 600 | a single point, 0.667 | — |
| < 600 | **empty** | **0** |

This is why the tree could not have gone much deeper whatever the data said: the pool of *admissible*
questions collapses from 876 to zero over a factor of fifteen in node size.

---

## 2. Stop-cause classification — ALL 15 stopped nodes, every depth

Cause is read off the artifacts, not inferred: SIZE causes are decided by *n* alone (no question can
clear the floor), SHAPE means the node had the users but the argmax splitter put a child under 200,
and "no positive-gain split" would mean the argmax gain was ≤ 0.

| cause | nodes | users | share of cohort |
|---|---|---|---|
| **SIZE, n < 400** — no split of any shape can clear the floor | **5** | 1,602 | 16.0% |
| **SIZE, 400 ≤ n < 600** — ternary impossible; only a fully-answered binary split could have continued | **3** | 1,523 | 15.2% |
| **SHAPE, n ≥ 600** — the argmax splitter puts a child under the floor | **7** | 6,875 | 68.8% |
| **no positive-gain split** | **0** | 0 | 0% |

By depth (**this verifies 1.18b**: of the nine depth-2 nodes, 1 split, 4 stopped on SIZE, 4 on SHAPE):

| depth | split | SIZE n<400 | SIZE 400–599 | SHAPE |
|---|---|---|---|---|
| 0 | 1 | – | – | – |
| 1 | 3 | – | – | – |
| 2 | 1 | 2 | 2 | 4 |
| 3 | 1 | 0 | 1 | 1 |
| 4 | 1 | 1 | 0 | 1 |
| 5 | 0 | 2 | 0 | 1 |

Per node, with the branch that broke the floor:

| leaf | d | n | splitter | answer rate | like / dislike / no-answer | min child | argmax gain | cause |
|---|---|---|---|---|---|---|---|---|
| root.noanswer.noanswer | 2 | 1,785 | item 1 | 0.194 | 223 / 124 / 1,438 | 124 | +0.01634 | SHAPE |
| root.like.noanswer | 2 | 1,256 | concept 377 | **0.971** | 652 / 567 / **37** | 37 | +0.00672 | SHAPE |
| root.like.like | 2 | 1,010 | item 112 | 0.345 | 239 / 109 / 662 | 109 | +0.01797 | SHAPE |
| root.dislike.like | 2 | 723 | concept 312 | **0.927** | 151 / 519 / **53** | 53 | +0.01155 | SHAPE |
| root.dislike.noanswer | 2 | 529 | item 15 | 0.255 | 90 / 45 / 394 | 45 | +0.03386 | SIZE 400–599 |
| root.noanswer.dislike | 2 | 438 | item 1 | 0.717 | 193 / 121 / 124 | 121 | +0.04251 | SIZE 400–599 |
| root.noanswer.like | 2 | 357 | concept 354 | **1.000** | 170 / 187 / **0** | 0 | +0.02351 | SIZE n<400 |
| root.like.dislike | 2 | 340 | concept 911 | **0.941** | 63 / 257 / **20** | 20 | +0.01352 | SIZE n<400 |
| root.dislike.dislike.like | 3 | 660 | item 951 | 0.435 | 169 / 118 / 373 | 118 | +0.02159 | SHAPE |
| root.dislike.dislike.noanswer | 3 | 556 | item 499 | 0.097 | 29 / 25 / 502 | 25 | +0.01621 | SIZE 400–599 |
| root.dislike.dislike.dislike.like | 4 | 702 | item 614 | 0.426 | 125 / 174 / 403 | 125 | +0.01363 | SHAPE |
| root.dislike.dislike.dislike.dislike | 4 | 355 | concept 784 | **1.000** | 42 / 313 / **0** | 0 | +0.01650 | SIZE n<400 |
| …noanswer.dislike | 5 | 739 | concept 251 | **0.976** | 140 / 581 / **18** | 18 | +0.01143 | SHAPE |
| …noanswer.like | 5 | 345 | concept 455 | **0.974** | 30 / 306 / **9** | 9 | +0.02097 | SIZE n<400 |
| …noanswer.noanswer | 5 | 205 | concept 306 | **1.000** | 48 / 157 / **0** | 0 | +0.01436 | SIZE n<400 |

**The anecdote generalises.** In **8 of the 15** stops the binding child is the DID-NOT-ANSWER branch,
and in all eight the splitter is a **concept answered by 92.7–100%** of the node — three of them by
exactly 100%, so their did-not-answer branch is *empty*. The remaining seven are the mirror image: an
**item** whose answer rate is too *low* (0.097–0.435), so the like and dislike branches starve while
the no-answer branch swallows the node. Both failures are the same window closing from opposite
sides.

---

## 3. The answerability penalty across the whole candidate pool

For every node, every candidate in the pool (1,000 at the root, minus the inherited path below it)
was partitioned and tested against the floor. The headline statistic is the mean answer rate of the
**feasible** subset minus the mean over the **whole pool**, with a permutation null (same number of
feasible labels reassigned at random, 10,000 draws, seed 98765).

### At the root the penalty is unambiguous and one-sided

| pool at the root | n | mean AR | median AR | max AR |
|---|---|---|---|---|
| all candidates | 1,000 | 0.456 | 0.457 | 1.000 |
| min-leaf FEASIBLE | 876 | **0.436** | 0.309 | **0.980** |
| min-leaf EXCLUDED | 124 | **0.598** | **0.986** | 1.000 |
| concepts, all | 500 | 0.798 | 0.842 | 1.000 |
| concepts, feasible | 428 | **0.765** | 0.794 | **0.980** |
| items, all | 500 | 0.114 | 0.094 | 0.426 |
| items, feasible | 448 | 0.122 | 0.103 | 0.426 |

Gap over the whole pool **−0.020, permutation p < 1e-4, z = −4.63**; over concepts alone **−0.033,
p < 1e-4, z = −11.03**. The excluded candidates' *median* answer rate is **0.986** against **0.309**
for the survivors, and the most answerable concept that survives sits at **0.9797** — exactly the
arithmetic ceiling 1 − 200/10,000 = 0.98.

**All 72 excluded concepts are excluded on the did-not-answer branch. None — zero — is excluded for a
thin like or dislike branch.** At the root, being excluded from the search *is* being too answerable.

### Below the root the sign flips, and it flips for a reason

The floor is an absolute 200, so as *n* falls it becomes a larger fraction of the node and starts
binding from the other side, eliminating the low-answer-rate items wholesale. Pooled over the 12
nodes that have any feasible candidate the gap is **+0.117** (mean AR 0.475 → 0.602), i.e. the
feasible subset is on average *more* answerable. That is the item channel dominating the count, and
it must be reported rather than buried.

Inside the **concept** channel the penalty stays in the predicted direction almost everywhere —
**9 of the 12 nodes with a feasible concept show a negative gap**, 8 of them at p ≤ 0.0024:

| node | n | concepts: pool AR | feasible AR | gap | p | feasible concepts |
|---|---|---|---|---|---|---|
| root | 10,000 | 0.798 | 0.765 | **−0.033** | <1e-4 | 428 |
| root.dislike | 4,814 | 0.854 | 0.801 | **−0.053** | <1e-4 | 343 |
| root.like | 2,606 | 0.827 | 0.766 | **−0.061** | <1e-4 | 226 |
| root.noanswer | 2,580 | 0.665 | 0.703 | +0.037 | 0.0024 | 237 |
| root.dislike.dislike | 3,562 | 0.896 | 0.837 | **−0.059** | <1e-4 | 208 |
| root.dislike.dislike.dislike | 2,346 | 0.943 | 0.870 | **−0.073** | <1e-4 | 37 |
| root.d.d.d.noanswer | 1,289 | 0.913 | 0.825 | **−0.088** | 0.0015 | 11 |
| root.noanswer.noanswer | 1,785 | 0.641 | 0.708 | +0.067 | <1e-4 | 168 |
| root.like.noanswer | 1,256 | 0.749 | 0.677 | **−0.072** | 1e-4 | 99 |
| root.like.like | 1,010 | 0.897 | 0.752 | **−0.145** | <1e-4 | 16 |
| root.dislike.like | 723 | 0.864 | 0.620 | −0.244 | 0.13 (n=1) | 1 |
| root.d.d.d.like | 702 | 0.982 | — | — | — | 0 concepts feasible |

Ten of the 22 nodes have **no feasible concept at all**, and at those nodes the concept pool's mean
answer rate runs 0.556–0.982. The deeper the tree goes the more answerable its cohort's concepts
become (root 0.798 → `root.dislike.dislike.dislike.like` at depth 4, 0.982) and the harder the
ceiling bites: at the five smallest depth-4/5 nodes (n = 205–739) **493–497 of the ~500 concepts sit
above the node's own ceiling**.

---

## 4. Pricing the two alternative rules — from banked quantities, neither was run

Both are different arms, not fixes; neither is claimed to be better without a run.

### (a) CONSTRAINED SEARCH — search only among candidates whose split clears the floor

*What it changes structurally.* The split criterion stops being "the argmax of the reported metric"
and becomes "the argmax over an admissible set", which reintroduces exactly the kind of side
constraint this chapter criticises other selectors for. It does **not** break the chapter's nesting
gate: at all seven nodes that actually split — the root included — the unconstrained argmax was
already feasible, so the constrained tree has the **same root question** and the K=1 difference stays
exactly 0.

*Which nodes would have continued.* **5 of the 15**, covering 5,476 users (54.8%):
`root.dislike.like` (1 feasible candidate), `root.like.like` (16), `root.like.noanswer` (99),
`root.noanswer.noanswer` (168), `root.dislike.dislike.dislike.like` (1). At two of these the
"choice" is a single forced candidate. **The other 10 stopped nodes have ZERO feasible candidates in
the entire 1,000-candidate pool** — the 8 SIZE nodes plus two SHAPE nodes (`root.dislike.dislike.like`
n=660, `…noanswer.dislike` n=739). **Constrained search cannot save two thirds of the stops.**

*What it costs.* Priced from the banked CELF level-0 gains. A heap entry at update level 0 is the
candidate's exact gain at that node; entries popped during the leaf's greedy tail lost their
node-level value, but a popped entry was the heap maximum when it was popped, so its node-level gain
is ≥ every intact entry's — which brackets the constrained pick rather than guessing it:

| node | n | argmax gain (infeasible) | best FEASIBLE gain | cost of the constraint |
|---|---|---|---|---|
| root.like.noanswer | 1,256 | +0.00672 | [+0.00175, +0.00672) | ≤ 0.00497 (≤74% of the gain) |
| root.dislike.like | 723 | +0.01155 | [+0.00163, +0.01155) | ≤ 0.00993 (≤86%) |
| root.noanswer.noanswer | 1,785 | +0.01634 | [+0.00213, +0.01634) | ≤ 0.01421 (≤87%) |
| root.d.d.d.like | 702 | +0.01363 | [+0.00158, +0.01363) | ≤ 0.01205 (≤88%) |
| root.like.like | 1,010 | +0.01797 | [+0.00189, +0.01797) | ≤ 0.01607 (≤89%) |
| the 7 nodes that split | — | — | = the argmax | **exactly 0** |

So constrained search buys one more level for about half the cohort, at a node-level question cost
bounded above by roughly three quarters to nine tenths of that question's gain, and it changes the
criterion. It is not free and it is not a repair.

### (b) PER-CHILD ROUTING — allow the split, route the undersized child to the parent's greedy tail

*What it changes structurally.* A node's small child stops being a veto and becomes a terminal child
that inherits the parent's greedy tail. Note this costs the routed users **nothing by construction**:
a stopped node's greedy tail already begins with its own splitter (asserted in `ndcg_tree.py`), so a
routed child receives precisely the interview it receives today. What the tree loses is its clean
recursive-partition form — siblings terminate at different depths — and realised depth stops being
comparable to Golbandi's. What it keeps is the empirical warrant for the 200 floor: every child that
goes on to be conditioned still clears it.

*Which nodes would have continued.* **10 of the 15**, covering 8,398 users (84.0%) — every stopped
node except the five with n < 400, where two children over the floor is arithmetically impossible.
Using the tree's own chosen splitters (no re-search), three nodes have two children over the floor
immediately — `root.like.like` (239 / 662 conditioned, 109 routed), `root.like.noanswer` (652 / 567,
37 routed), `root.noanswer.noanswer` (223 / 1,438, 124 routed) — i.e. **3,781 users take one more
conditioned question and 270 are routed**. The other seven need a different (feasible-for-two)
candidate, of which they have between 24 and 364.

*What it costs.* Nothing measurable on the banked evidence, and that is exactly why it cannot be
sold: the value of the extra conditioning for the 3,781–8,398 users is **unmeasured**. The tree's own
val→test shrinkage (2.70× full / 1.50× tail) and its 6/15 negative test leaves (39.3% of the test
cohort, against the policy-B comparator) say a val-selected
extra split can lose on test. Per-child routing is the more sensible arm, and it is still an arm.

---

## 5. What the chapter MAY and MAY NOT conclude about depth

**MAY claim, all measured above:**

* the tree reached depth 5 on 10,000 validation users under a 200-user floor, and **all 15 stops were
  floor stops** — none was a "no question helps" stop; the argmax gain in hand at the stopping nodes
  ran +0.00672 to +0.04251;
* **8 of 15 stops (31.2% of the cohort) were forced by node size before any question was considered**
  (5 at n < 400, where no split of any shape clears the floor; 3 at 400 ≤ n < 600, where only a
  fully-answered binary split could have continued), and **7 (68.8%) were shape stops**;
* the floor acts as a **window on the answer rate**, `[400/n, 1 − 200/n]`, which closes at n = 600;
  admissible candidates fall from 876/1,000 at the root to 0 at n = 660;
* at the root the excluded candidates are **systematically more answerable** than the survivors
  (median 0.986 vs 0.309; concept gap −0.033, permutation p < 1e-4), and **all 72 excluded concepts
  are excluded on their did-not-answer branch**; within the concept channel the penalty holds at 9 of
  the 12 nodes that have a feasible concept;
* in **8 of the 15 stops the binding child was the did-not-answer branch of a concept 92.7–100% of
  the node could answer** — the property that makes a concept worth asking is the property that ends
  the branch.

**MAY NOT claim:**

* that conditioning stops paying at depth 5, or that depth 5 is a property of the data rather than of
  the rule;
* that a deeper tree would pay. Nothing here measures that, and the banked test-side evidence argues
  against extrapolating: the three depth-5 leaves score +0.028 / −0.005 / −0.006 on test, and leaf
  gain does not track leaf size (Spearman ρ = 0.10, p = 0.71);
* that either alternative rule is an improvement. Constrained search cannot even continue 10 of the
  15 stopped nodes, and per-child routing's benefit is unmeasured.

**One sentence for the chapter:** *Realised depth is a property of the ternary-split-plus-200-floor
stopping rule interacting with how answerable the best questions are — every stop was a floor stop
with a positive gain in hand, a third of them arithmetically forced by cohort size and most of the
rest by a concept nearly everyone could answer — so this run supports no claim, in either direction,
about whether conditioning beyond depth 5 would pay.*
