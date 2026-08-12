# TREE ROUTING / SPECIALISATION — does the NDCG tree ROUTE, or only PARTITION?

TODO **1.20** · plan item **D2** · 2026-08-12 · design sheet
`docs/design/TREE_ROUTING_SPECIALISATION_DESIGN.md` (written before any statistic was computed) ·
code `src/instrument/tree_routing_specialisation.py` · numbers
`experiments/battery/tree_routing_specialisation.json`.

**BANKED ARTIFACTS ONLY. Nothing was re-scored** — the tree structure, all 15 leaf orders and the
ternary partition of all 1,000 pool candidates were already on disk; the only new read is the genome
file, on the tree's own concept path (relevance >= 0.5, vocab-restricted, >= 30 members, 1,031
concepts, `tags = sorted(members.keys())`). All 500 pool items and all 71 items the tree asks have
genome rows, so nothing is dropped.

---

## VERDICT

> **The tree PARTITIONS and DIVERGES, but it does not demonstrably ROUTE.** Different branches ask
> almost disjoint questions — 80 of 105 leaf pairs share *nothing* outside their inherited path — but
> the questions a branch asks are **not** measurably about the answer that created the branch: pooled
> like-branch congruence 0.7705 vs matched control 0.7691 [0.7572, 0.7821], p = 0.82; like−dislike
> contrast +0.0054 vs label-shuffle [−0.0200, +0.0200], p = 0.61. **The control did not separate, so
> under the pre-registered rule no routing claim may appear in the chapter.**

This is divergence *without demonstrable answer-congruence* — the same shape already recorded for the
genre arm (divergence without exploitability), now inside the tree.

---

## 1. Pre-specified statistics (from the design sheet, unchanged)

- **S1 divergence** — pairwise overlap of the 15 leaves' question sets: full 15-question sets, **free
  picks only** (leaf questions minus the inherited ancestor path — the shared prefix is overlap by
  construction, not by choice), positions differing, and overlap with the global static `combined`
  order. Distribution reported, not a mean.
- **S2 answer-congruence** — similarity, in the 1,128-dim genome tag-relevance space (item = raw
  genome row, concept = mean genome row over its members; cosine), between a node's **splitter** and
  every question asked strictly **below** each of its three children. Primary pooled statistic =
  unweighted mean over the 7 split nodes; contrast Δ = like − dislike, predicted **positive**.
- **S3 controls (mandatory)** — **C1** matched-availability random control (replacement drawn from
  the same channel, same root **answer-rate** quintile and same root **divisiveness** quintile — the
  banked preflight table; 2,000 draws, seed 98765) and **C2** branch-label shuffle (the three
  children's question sets permuted across {like, dislike, noanswer} within each node; 2,000
  permutations).
- **Decision rule, fixed in advance:** routing admissible only if (a) pooled like-congruence exceeds
  C1's 97.5th percentile **and** (b) pooled Δ exceeds C2's 97.5th percentile.

Matching note: 0.39% of the 1,014,000 control draws fell in a matched cell with < 5 candidates and
widened to channel + answer-rate quintile only, as pre-registered.

---

## 2. S1 — divergence: leaves ask near-disjoint questions

| statistic (105 leaf pairs) | min | q25 | median | mean | q75 | max |
|---|---|---|---|---|---|---|
| overlap of the full 15-question sets | 1 | 1 | **2** | 2.16 | 3 | 6 |
| overlap of **free picks** (~12 per leaf) | 0 | 0 | **0** | 0.27 | 0 | 3 |
| positions differing (of 15) | 10 | 13 | **14** | 13.25 | 14 | 14 |

- **80 of 105 pairs share no free pick at all**; 23 share one, 1 shares two, 1 shares three. The
  floor of 1 on the full sets is the root splitter (concept 114), which every leaf carries by
  construction.
- 15 leaves × 12 free slots = 180 picks yield **154 distinct questions** (86% distinct); the union
  over full sequences is 159 of the 1,000-candidate pool.
- Chance references: two random 15-sets from the 1,000-pool overlap 0.225; two random free-pick sets
  drawn from the leaves' **own** 154-question union overlap **0.935**. Observed 0.267 is **3.5×
  below** even the matched-availability chance level — leaves are not merely different, their
  question sets are close to mutually exclusive.
- Overlap with the global static `combined` 15-question order: mean 1.47, range 0–4, **four leaves
  share nothing with it**; only 9 of the global order's 15 questions appear anywhere in the tree.

Pre-specified reading (median free-pick overlap <= 25% ⇒ specialised): **specialised, decisively** —
median 0.

## 3. S2 — answer-congruence: null

Splitter-to-follow-up cosine, per split node (n = questions below that child):

| node | d | n users | splitter | like | dislike | noanswer | Δ (like−dis) |
|---|---|---|---|---|---|---|---|
| root | 0 | 10,000 | concept 114 ("better than expected") | 0.8261 (38) | 0.8370 (96) | 0.8305 (39) | −0.0109 |
| root.dislike | 1 | 4,814 | concept 283 ("disturbing") | 0.8524 (13) | 0.7937 (75) | 0.7957 (13) | +0.0588 |
| root.like | 1 | 2,606 | item 83 | 0.7603 (13) | 0.7441 (13) | 0.7593 (13) | +0.0162 |
| root.noanswer | 1 | 2,580 | item 514 | 0.7222 (13) | 0.7089 (13) | 0.6879 (13) | +0.0133 |
| root.dislike.dislike | 2 | 3,562 | concept 36 ("alien") | 0.7718 (12) | 0.7675 (50) | 0.7991 (12) | +0.0043 |
| root.dislike.dislike.dislike | 3 | 2,346 | item 19 | 0.7156 (11) | 0.7412 (11) | 0.7410 (29) | −0.0256 |
| …dislike.noanswer | 4 | 1,289 | concept 334 ("explosions") | 0.7446 (10) | 0.7625 (10) | 0.7897 (10) | −0.0179 |

Pooled (unweighted over the 7 nodes): like **0.7705**, dislike **0.7650**, noanswer **0.7719**,
Δ **+0.0054**. Only 4 of 7 nodes have Δ > 0 — a coin flip.

The absolute cosines are large (~0.77) for the stated reason: raw genome vectors are dense and
non-negative, so *any* pair of catalogue-scale questions is "similar". That number carries no
information on its own; only the contrast against the control does.

## 4. S3 — the controls, and they did not separate

| test | observed | control mean | control central 95% | p (2-sided) | outside? |
|---|---|---|---|---|---|
| **C1** like-branch congruence | 0.7705 | 0.7691 | [0.7572, 0.7821] | 0.824 | no |
| **C1** dislike-branch congruence | 0.7650 | 0.7691 | [0.7591, 0.7793] | 0.437 | no |
| **C1** noanswer-branch congruence | 0.7719 | 0.7694 | [0.7577, 0.7812] | 0.672 | no |
| **C1** Δ like−dislike | +0.0054 | −0.0000 | [−0.0163, +0.0163] | 0.491 | no |
| **C2** Δ like−dislike (label shuffle) | +0.0054 | −0.0001 | [−0.0200, +0.0200] | 0.609 | no |
| **C2** like-branch congruence (label shuffle) | 0.7705 | 0.7690 | [0.7582, 0.7810] | 0.817 | no |

Both legs of the pre-registered rule fail. A question drawn at random from the pool, matched only on
channel, answer rate and divisiveness, is **exactly as related to the splitter** as the question the
tree actually chose. Permuting which answer a branch came from changes nothing either.

`ROUTING_CLAIM_ADMISSIBLE = false`.

## 5. Secondary (root only, does NOT license a claim)

Pre-registered as interpretable colour, with the same matched control attached. Direct genome
relevance of a branch's follow-up **items** to the root concept's own tag (tagId 129, "better than
expected", 868 members):

| root branch | n items | mean relevance | matched control | p | outside 95%? |
|---|---|---|---|---|---|
| like | 20 | **0.4348** | 0.3379 [0.2768, 0.4053] | 0.004 | **yes** |
| dislike | 40 | 0.3273 | 0.3147 [0.2738, 0.3600] | 0.557 | no |
| noanswer | 19 | 0.3112 | 0.3188 [0.2621, 0.3798] | 0.797 | no |
| like − dislike | — | **+0.1075** | +0.0233 [−0.0515, +0.1001] | 0.036 | **yes** |

So on **one node, one channel, one hand-chosen similarity**, the users who *liked* the opener are
subsequently shown items more relevant to it than a matched-random question would be. It is stated
here in full and it changes nothing: it is a single node out of seven, it is one of six secondary
tests with no multiplicity correction, it disappears under the pre-registered primary statistic
(which pools the same node's concepts and items and all seven nodes), and the tag itself
("better than expected") is a generic evaluative tag rather than a content one. Recording it because
suppressing a positive that failed the pre-registered test would be the same error in the other
direction — it is a **hypothesis for the parametric policy (E1/E2)**, not a result.

## 6. What this licenses and forbids

**Forbidden.** Any sentence of the form "a broad answer determines which finer question follows",
"the tree routes users to congruent follow-ups", or any coarse-to-fine/curriculum reading of the
tree. Plan §9 ("no routing/curriculum claim unless D2 separates from its control") binds: it did not.

**Licensed.** The divergence fact, which is strong and is worth stating on its own: *conditioning
makes the branches ask almost entirely different questions — 154 distinct questions across 180 free
slots, 80 of 105 leaf pairs sharing none, and only 9 of the global static order's 15 questions
appearing anywhere in the tree.* That is what the +0.0137 buys: not a comprehensible curriculum but a
question set the static order never reaches. It is consistent with, and independent evidence for, the
already-confirmed 4.6 finding (the no-answer branch's next question, item 514, appears nowhere in
either global order).

**Honest framing for Act 4.** The adaptive gain is real and its questions are genuinely different;
*why* those questions are the right ones for that branch is not something we can currently
characterise in tag space. Say that, and stop.
