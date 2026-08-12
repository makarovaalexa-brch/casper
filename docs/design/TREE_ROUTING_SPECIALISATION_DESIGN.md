# DESIGN SHEET — tree routing / specialisation measurement (TODO 1.20, plan D2)

Written **before** any statistic was computed. Everything below is pre-registered: the estimands, the
controls, the decision rule and the failure outcome. Written 2026-08-12.

AUTHOR SAID (2026-08-12): "static coarse-to-fine makes no sense whatsoever … in static nothing
informs anything."

## 1. The question this decides

The chapter may not claim **routing** ("a broad answer determines which finer question follows")
unless the TREE actually does it. A static order cannot: nothing informs anything, and the observed
concept-first/item-later composition is an ANSWERABILITY fact. The tree is the only object where a
routing claim could live. Two things must be true for the claim to be admissible:

1. **Specialisation** — different branches ask genuinely *different* questions (if every leaf asks
   the same 15 things, the tree partitions users but does not route them).
2. **Answer-congruence** — the questions a branch asks are related to *the answer that defined that
   branch*, beyond what a question of the same kind would give by chance.

Neither is currently supported. There is no channel dominance by depth (5 concepts / 4 items at
depth 2; 8 concepts / 7 items among the 16th picks; TODO 4.5 closed as a null).

**A null here is a perfectly good outcome and is reported as such: "the tree partitions, it does not
route", and every routing sentence stays out of the chapter.**

## 2. Data — BANKED ONLY, nothing is re-scored

- `experiments/battery/ndcgtree.json` — validation-grown tree: 22 nodes, 15 leaves, depth 5, root
  splitter = concept 114, pool = 500 popularity-ranked items + 500 answerable concepts. Provides
  every node's splitter, path, branch label, child sizes and every leaf's 15-question sequence.
- `experiments/battery/ndcgtree_PREFLIGHT.json` — the ternary partition (like / dislike / noanswer
  sizes), answer rate and `p_min` of **all 1,000 pool candidates at the root**. This is the matching
  table for the control; it is banked, so the control costs no scoring.
- `data/movielens/genome-scores.csv` + `data/ml-25m/proc/unique_sid.txt` — concept definitions, on
  the tree's own path: relevance >= 0.5, vocab-restricted, tags with >= 30 members, `tags =
  sorted(members.keys())` (`run_battery_phaseA.load_genome` + `tradeoff_ledger.build_shared`).
  Verified: 1,031 concepts, so concept index c ↔ `tags[c]`; item index = sid = row of
  `unique_sid.txt`.

**No recommender scoring of any kind.** If a statistic below had required re-scoring it would have
been dropped, not run (a heavy job holds the box).

## 3. The similarity — stated once, used everywhere

One space for both channels: the **1,128-dim genome tag-relevance space**.

- an ITEM i is the raw genome relevance row `R[i, :]` (all 1,128 tags, un-thresholded);
- a CONCEPT c is the **member centroid**: the mean of `R[i, :]` over the items with
  `relevance(i, tags[c]) >= 0.5` (the same member set the concept channel is built from);
- `sim(a, b) = cosine(v_a, v_b)`.

Coverage checked before pre-registration: **all 500 pool items and all 71 items the tree asks have
genome rows**, so nothing is dropped and no exclusion rule is needed (HARD RULE 1).

Secondary, interpretable, for the root only: the **direct genome relevance** `R[i, tags[114]]` of
each follow-up item to the root concept.

Note stated up front: raw genome vectors are dense and non-negative, so absolute cosines are large
for any pair. **The absolute number is uninterpretable; only the observed-minus-control contrast is.**
That is exactly why §5 is mandatory, not decorative.

## 4. Pre-specified statistics

### S1 — divergence of leaf question sets (do branches ask different things?)

Over all 15 leaves = 105 unordered pairs:

- **S1a** `|A ∩ B|` of the full 15-question sets. Report the full distribution (min / quartiles /
  median / max + histogram), never only the mean.
- **S1b** the same on **free picks only** — a leaf's questions *minus its inherited ancestor path*
  (`nodes[nid].steps`), since the shared prefix is overlap by construction, not by choice.
- **S1c** positions differing: `#{k : A_k != B_k}` out of 15.
- **S1d** overlap of each leaf with the **global static `combined` order** (15 questions), and the
  overlap of the global order with the union of all tree questions.
- Two chance references, both reported: (i) two random 15-sets from the 1,000-candidate pool
  (E = 0.225); (ii) **matched availability** — two random 15-sets from the union U of questions any
  leaf actually asks (E = 225/|U|), which is the honest reference because the tree only ever shops
  in the useful part of the pool.

**Pre-specified reading:** leaves are called *specialised* if the median free-pick pairwise overlap
is <= 25% of the free-pick length; *not specialised* if it is >= 50%. In between is reported as
partial and does not on its own license a routing claim.

### S2 — answer-congruence (are the follow-ups about the answer that defined the branch?)

For each of the 7 split nodes t (splitter `s_t`) and each branch b ∈ {like, dislike, noanswer}:

- `Q(t, b)` = every question asked strictly **below** the child node — the child's own splitter/tail
  and everything in its subtree — deduplicated, with the inherited ancestor path removed.
- **congruence** `C(t, b) = mean_{q ∈ Q(t,b)} sim(v_{s_t}, v_q)`.
- variant (i) NEXT-ONLY: `sim(s_t, first question of the child)` — one number per branch, no
  averaging-away.
- **contrast** `Δ(t) = C(t, like) − C(t, dislike)`. Predicted sign if routing is answer-congruent:
  **positive** (the branch that liked the concept gets asked about related things). The noanswer
  branch has no pre-registered direction and is exploratory.
- Pooled: **primary = unweighted mean of Δ(t) over the 7 split nodes**; user-weighted by child size
  reported alongside. Root reported separately in full (it is the one the author named).

### S3 — CONTROLS (mandatory, HARD RULE 5)

- **C1, matched-availability random control.** Each `q ∈ Q(t,b)` is replaced by a random pool
  candidate of the **same channel**, drawn from candidates in the same **quintile of root answer
  rate** and the same **quintile of root divisiveness** `D = min(like, dislike) / max(like + dislike, 1)`
  — both read from the banked preflight table. This holds constant "the tree asks answerable,
  divisive questions, and such questions may resemble each other generically". If a matched cell has
  < 5 candidates it widens to channel + answer-rate quintile only, and every widening is counted and
  reported. 2,000 draws, seed 98765. Observed must sit outside the control's central 95% in the
  predicted direction.
- **C2, branch-label shuffle.** Within each split node the three children's question sets are
  permuted across the labels {like, dislike, noanswer}; Δ is recomputed; 2,000 permutations, seed
  98765. This asks whether congruence is tied to *which answer* produced the branch or is just a
  property of the node.

### Decision rule (fixed now, not after seeing numbers)

A **routing claim is admissible only if BOTH**: (a) pooled like-branch congruence exceeds the C1
matched control's 97.5th percentile, and (b) the pooled Δ exceeds the C2 shuffle's 97.5th percentile.
Anything else ⇒ **verdict "partitions, does not route"**, no routing sentence anywhere in the
chapter, and S1 is reported on its own as a descriptive divergence fact.

## 5. What each outcome changes

- Both pass ⇒ Act 4 may state a routing mechanism, sized by Δ and qualified by S1.
- S1 diverges but S2 fails ⇒ the honest sentence is "different branches ask different questions, but
  we cannot show the difference is *about* the answer" — divergence without exploitability, the same
  shape already recorded for the genre arm.
- S1 fails ⇒ no specialisation at all; the tree's gain comes from re-ordering within a shared
  question set, and that is what gets written.

No decision anywhere in the chapter is left unchanged by this measurement, so it runs.
