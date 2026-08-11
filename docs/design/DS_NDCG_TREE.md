# Design sheet — NDCG-objective interview tree (`ndcgtree`), stage 1: build on validation

Pre-registered 2026-08-11, revised 2026-08-12 after the author's redirect (no depth cap, combined
action space). Code: `src/instrument/ndcg_tree.py`. TODO item 1.9.

## The question

Every Act-1 comparison confounds two things: *conditioning on the answers* and *choice of objective*.
The only arm in the battery that maximises the quantity we report — cohort-mean held-out full
NDCG@10 under the certified tower — is also the only static one. Golbandi's tree conditions, but its
splitter minimises within-node squared error of **rating prediction**. This arm conditions **and**
optimises our metric, so conditioning is the only thing that varies.

**Decision the outcome changes.** If a tree that maximises exactly the reported metric still fails to
beat the static order built by maximising that same metric, the Act-1 story stops being "adaptive
methods optimise the wrong thing" and becomes something stronger and narrower about what there is to
condition on at these budgets. If it wins, we have the missing arm and a magnitude for what
conditioning is worth when nothing else differs.

## Nearest prior work (cited, not claimed around)

Hu & Yu, *Interview Process Learning for Top-N Recommendation*, RecSys 2013 — an interview decision
tree whose split loss "can be square loss, logistic loss, **pairwise rank loss**, etc." So *"first
ranking objective inside an interview tree"* is **not available to us**. Four separations, verified
from their text (`external_literature/findings/hu_yu_2013_interview_process_learning.md`):

1. a smooth **pairwise surrogate**, not a truncated discounted top-k metric;
2. optimised as a **training loss** over the tree fit, not as the measured value of asking a question;
3. **MAP on 10× sampled negatives**, not NDCG@10 on the full 18,359-item catalogue;
4. their static comparator is **top-k popular**; ours is the order built by maximising the very
   metric being reported.

## Design

| | |
|---|---|
| Cohort | the **10,000 canonical VALIDATION users**. Forced: `data/ml-25m/proc/` has no train_tr/train_te, so train users have no fold-in/target split, and the tower trained on them. **TEST is not built, not loaded, not touched.** |
| Action space | the **combined** pool — top-500 items by popularity + top-500 concepts by the `answerable` bank, the same two `_ranked_pool` calls the global build makes, concatenated items-first. 1,000 candidates in one argmax. |
| Branch | ternary **LIKE / DISLIKE / DID-NOT-ANSWER**, off the harness's own behavioural rule. Item: absent from the graded fold-in lookup → did-not-answer; level ≥ `LIKE_MIN_LEVEL` (7 = ≥4.0 stars) → like; else dislike. Concept: not `foldable` (unanswerable or refuse band) → did-not-answer; signed value > 0 → like; else dislike (the `meh` band is zeroed under the certified `keep_meh=False` convention and lands in dislike — the count of exact zeros is reported per node). Target leak asserted 0/10,000. |
| Split objective | node-size-weighted mean of the three children's cohort-mean **full NDCG@10**, by actually folding and scoring under the certified tower + concept channel. `argmax` over the pool. |
| Stopping | **the min-leaf rule is the only stopping rule — no depth cap.** No split if any child < **200** validation users. Growth otherwise continues until the budget runs out (a node with one question left has nothing to split into). Realised depth is a **result**. |
| Budget | **always 15 questions.** A node that stops at depth *d* gets a greedy tail of 15−*d* further questions searched on its own users. |
| Charging | **every question is charged and folded.** No free side information anywhere in this arm, and therefore no generosity caveat — unlike the genre arm. |
| Comparators | the banked global 15-step orders, re-scored **in this run**: headline `combined` (same action space, same objective, same cohort), plus `concepts-only` and `items-only` for continuity. Question-for-question at equal K ∈ {1,2,4,8,15}. |

### Why 200

Empirical, from the genre-split experiment (2026-08-11): every genre group **above** 200 users held a
positive gain when its order was carried to the TEST cohort; every group **below** it failed to
generalise — Horror n=51 test −0.0013 (ns), Crime n=79 test −0.0433, Fantasy n=11 test −0.1156. The
rule is applied to the **chosen** splitter, not as a constraint on the search: if the argmax question
shatters the node, the node stops rather than falling back to a worse question.

### What is *not* guaranteed

A leaf's order must carry its ancestors' questions, chosen for their *parents'* cohorts. The global
order is therefore **not** inside a leaf's search space, and a leaf **can** legitimately score below
it at K=15. A negative leaf gain is a result, not a harness failure. The only mechanical guarantee is
the K=1 tie, and only when the root splitter matches the comparator's first question.

## Two facts about the criterion, stated up front

**1. The criterion is algebraically the node mean.** NDCG@10 is per-user, so the node-size-weighted
mean over a partition of the node equals the plain mean over the node. Measured at every node
(`partition_decomposition_max_abs_err`, observed 0.0e+00 – 5.6e-17). Consequence: the split search is
the greedy argmax restricted to the node's users, and this tree is **myopic in exactly the sense
Golbandi's is** — his criterion is also the immediate within-node error. Adaptivity here comes from
the *children diverging*, not from lookahead. This is written into the output JSON, not left for a
reviewer to derive.

**2. The banked comparator is a POLICY-A artifact.** Measured by this run's sanity gate on
2026-08-12: `greedy_static_i26c.json` records `intercept.full = 0.12987336`, which is the **policy-A**
m=0 value; under **policy B** (the 2026-08-05 candidate-masking ruling this harness implements) the
same 10,000 validation users give m=0 = **0.15686538**. Its stored `gains` and `curve` are on a
different ruler and are **not comparable** to anything measured here. Only its **sequences** are used,
and every comparison re-scores them in-run under policy B — the same thing `group_greedy_genre.py`
does. **Residual caveat, not buried:** the comparator's *order* was selected under policy A, so this
is not exactly "same objective, same protocol, only conditioning differs". Closing that gap means
rebuilding the static combined order under policy B (~12 h). Flagged for the author.

## Preflight (measured 2026-08-12, `ndcgtree_PREFLIGHT.json`, not citable)

* **s/eval at n=10,000**: empty 7.4 s, ITEM candidate 8.5 s, CONCEPT candidate 6.9 s (the item-state
  cache is warm across the contiguous concept block, so the tower fold is paid once for all 500).
  Threads make no difference (8.49 s at 8 threads vs 8.51 s at 4) — the bottleneck is not BLAS.
* **Answer rate at the root**: items median **0.094**, concepts median **0.842**. The author's claim
  is confirmed and then some.
* **Smallest-branch fraction p_min**: items median 0.0364 (max 0.1737), concepts median 0.0880
  (max 0.2580). Concepts partition far more evenly.
* **Candidates whose root split already clears min_leaf=200**: 448/500 items, 428/500 concepts.
* **Hard bound on the tree**: every leaf holds ≥200 of 10,000 users ⇒ ≤50 leaves ⇒ (ternary)
  ≤24 internal nodes ⇒ **≤74 nodes total**, at any depth.

### Cost

Measured `s/eval` across the sanity run and the live root pass — n = 285 → 0.24 s, 401 → 0.30,
1,020 → 0.75, 1,753 → 1.36, 2,439 → 1.79, 6,541 → 5.18, 10,000 → **8.53** — is **linear in cohort
size with essentially zero fixed overhead**: `s/eval ≈ 8.5e-4 · n`. That is the number that decides
the bill, and it is better than the pessimistic model: since the nodes at one level partition the
10,000 users, **Σ s/eval over a whole level is ~8.5 s no matter how many nodes it has**, so a full
level costs ≈ 1,000 × 8.5 s = **2.4 h** regardless of tree width. The root's live seed pass confirms
it (`seed 25/1000 (3.6m, 8.53s/eval)`).

Total ≈ 2.4 h × (number of levels still containing internal nodes, weighted by the user fraction
still internal) + ~1.2 h of leaf greedy tails ⇒ **≈10–16 h** for a realised depth of 6–10.

**Nothing is cut.** The run launches with `--wall_hours 0` (no operational limit). Growth is
**level-synchronous**, so the partial tree on disk after every level is exactly the tree a shallower
request would have produced — the run can be stopped at any level and still yields a complete, valid,
citable tree, and it resumes deterministically from its per-node CELF-heap sidecars if the author
wants more depth later.

Nothing is cut: the run launches with `--wall_hours 0` (no operational limit). Growth is
**level-synchronous**, so the partial tree on disk after every level is exactly the tree a shallower
request would have produced — the run can be stopped at any level and still yields a complete, valid,
citable tree, and it resumes deterministically from its per-node CELF-heap sidecars if the author
wants more depth later.

Levers, if the author wants it shorter (none applied without sign-off): raise `--min_leaf` (fewer
nodes, but 200 is the empirically justified value); shrink the pool (HARD RULE 1 — needs sign-off);
or set `--wall_hours` to stop growth at the last completed level.

## Controls and gates

* **Sanity run first** (`--sanity`, ~11 min): budget 3, 6-question pool built **head-first** so it
  contains the comparator's own head (defect 2, which bit an earlier run); every comparison asserts
  equal K on both sides (defect 1); the ternary partition asserts cover + disjointness; the
  partition-decomposition identity is asserted at every node.
* **CELF-heap self-test**: a cold search (`heap_path=None`, i.e. the pre-change code path) vs
  `L=1` + resume-and-extend must agree on the sequence **and every gain bit for bit**; and a sidecar
  carrying a different contract must be **ignored, not merged**. Both verified 2026-08-12:
  `[(1,630),(1,173),(1,242)]`, gains `0.019562633400380808 / 0.017454344784169518 /
  0.008822177330925496`, identical.
* **Leak**: 0/10,000 users have a held-out target inside the routing lookup, asserted at startup.
* **HARD RULE 1**: no user is dropped, capped or sampled anywhere; the three branches partition each
  node exactly and it is asserted.
* **HARD RULE 11**: `*_partial.json` after every pick and every node; per-node contract-signed CELF
  heap sidecars; unique label `ndcgtree`; no output path shared with any other job.

## Report

Tree structure (question and channel at every node, node sizes, the like/dislike/did-not-answer split
and answer rate at every node, which nodes hit the min-leaf rule), realised depth and the leaf-depth
histogram, item-vs-concept split channel by depth, per-leaf gain vs each comparator at every K,
cohort-weighted gain, per-leaf genre composition (face validity), s/eval per node.

**Stage 2 (TEST) is separate and is not authorised by this sheet.**
