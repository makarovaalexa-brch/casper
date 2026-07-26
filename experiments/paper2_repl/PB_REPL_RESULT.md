# Paper B faithful replication — REAL item-vs-concept short-interview table (item-mask bug fixed)

**Date:** 2026-07-26 · **Stack:** canonical ML-25M Liang split (140,768 train / 10,000 val / **10,000 test**,
seed 98765), catalog vocab 18,359 items, 1,031 concepts. Weak biased-SVD recommender + attention fold-in
encoder (the faithful old-Paper-B reconstruction encoder), best-val checkpoint `.cache/paper2_repl/enc.pt`.
Decoder = method's own score `popb + Qp@u`. Ruler = canonical full & tail NDCG@10 (190-item head mask),
fold-in masked, targets = test_te. Results JSON: `experiments/paper2_repl/pb_results.json`.

## The bug that was fixed
The item-ask arms were **degenerate**. Folding was already correct (only fold-in-rated items are folded,
`residmap` = test_tr ratings). But the **masking** used `excl = prof | asked`: every asked popular item was
removed from the ranking. Since `pop_item`/`eig_item` ask the *most popular* items, this masked out popular
**held-out targets**, cratering full-NDCG **0.1345 → 0.0860 @q8**. That was a measurement artifact, not an
item number.

**Fix** (`scripts/paper2_repl/pb_interview.py`, commit `9a2eab7`): `excl = prof` — mask only the fold-in
profile, exactly like the concept arm and the full-profile snap. Faithful Paper B rule now holds: an item is
answerable/foldable **iff rated in the fold-in half**; unanswered popular questions burn budget and fold
nothing; **held-out targets stay un-askable** (never in `residmap` → never folded) **and un-masked** (never
in `prof`). Concept arms are byte-identical after the fix, rng-independent and deterministic, so their
numbers are carried forward from the same-checkpoint run (`ARMS=item`). Buggy JSON preserved at
`pb_results_BUGGY_itemmask.json`.

## Item is cold-UNANSWERABLE (the faithful setup, confirmed)
Item mean_answered per 8-question budget (of 8 asked):

| arm | q1 | q2 | q4 | q6 | q8 |
|---|---|---|---|---|---|
| pop_item / eig_item | 0.35 | 0.66 | 1.26 | 1.81 | **2.29** |
| rand_item | 0.03 | 0.07 | 0.13 | 0.19 | 0.25 |

Only ~2.3 of 8 popular item-questions land in a user's fold-in half — cold-unanswerable, exactly the regime
in which concepts (geometric) beat items in the original. **Item-ask no longer craters; it is flat-to-slightly
negative** (folding popular items nudges the user vector toward popularity, mildly hurting full). The random
control (which folds almost nothing) stays nearest cold — i.e. asking informative items does not help.

## REAL short-interview curves (full / tail NDCG@10, 10k test users)
Full-profile fold ceiling: **0.1443 / 0.0365**. Cold intercept (q0): **0.1345 / 0.0226** (all arms).

**Item-ask (answer-model-independent):**

| arm | q0 | q1 | q2 | q4 | q6 | q8 |
|---|---|---|---|---|---|---|
| pop_item full | 0.1345 | 0.1215 | 0.1158 | 0.1219 | 0.1269 | **0.1268** |
| pop_item tail | 0.0226 | 0.0206 | 0.0193 | 0.0215 | 0.0217 | **0.0222** |
| rand_item full | 0.1345 | 0.1339 | 0.1334 | 0.1329 | 0.1327 | **0.1320** |
| rand_item tail | 0.0226 | 0.0228 | 0.0228 | 0.0233 | 0.0238 | **0.0243** |

(`eig_item` is identical to `pop_item` — the coverage/`p_seen` objective selects the same popular items; this
held in the buggy run too.)

**Concept-ask, GEOMETRIC answers** (original +36% setting — recommender-geometry self-preference, CIRCULAR/UNCITABLE):

| arm | q0 | q1 | q2 | q4 | q6 | q8 |
|---|---|---|---|---|---|---|
| conc_pop full | 0.1345 | 0.0585 | 0.0932 | 0.1362 | 0.1446 | **0.1450** |
| conc_pop tail | 0.0226 | 0.0219 | 0.0258 | 0.0362 | 0.0361 | **0.0364** |

**Concept-ask, BEHAVIORAL (honest signed-SEL) answers** (the headline, recommender-independent):

| arm | q0 | q1 | q2 | q4 | q6 | q8 |
|---|---|---|---|---|---|---|
| conc_pop full | 0.1345 | 0.0837 | 0.1062 | 0.1273 | 0.1321 | **0.1322** |
| conc_pop tail | 0.0226 | 0.0198 | 0.0224 | 0.0254 | 0.0261 | **0.0261** |

## ⇒ CONCEPT − ITEM deltas @q8 (the reproduction of the old claim, both answer models)

Primary comparator = `pop_item` (the item-elicitation arm that actually asks & folds informative items):

| answer model | concept full | item full | **Δfull** | concept tail | item tail | **Δtail** |
|---|---|---|---|---|---|---|
| **GEOMETRIC** | 0.1450 | 0.1268 | **+0.0182 (+14%)** | 0.0364 | 0.0222 | **+0.0142 (+64%)** |
| **BEHAVIORAL** | 0.1322 | 0.1268 | **+0.0054 (+4%)** | 0.0261 | 0.0222 | **+0.0039 (+18%)** |

Robustness vs the *strongest* item arm (`rand_item`, full 0.1320 / tail 0.0243):

| answer model | **Δfull** | **Δtail** |
|---|---|---|
| **GEOMETRIC** | +0.0130 (+10%) | +0.0121 (+50%) |
| **BEHAVIORAL** | +0.0002 (≈tie) | +0.0018 (+7%) |

## Verdict
- **The item number is now real, not degenerate.** Item-ask is flat/cold-unanswerable (~2.3/8 answered),
  not a crater. The old 0.0860 was a masking artifact.
- **Concept beats item — direction reproduced under both answer models.** Under GEOMETRIC answers the
  concept advantage is *large* (+64% tail / +14% full vs pop_item), reproducing — and on this clean stack
  exceeding — the old "+36%" tail claim. But GEOMETRIC is the circular self-preference measure: geometric
  concepts at q8 (0.1450 full) even **exceed the full-profile ceiling (0.1443)** — a tell of circularity.
- **The honest premium is modest.** Under BEHAVIORAL (signed-SEL) answers the concept edge shrinks to
  +18% tail / +4% full vs the item-elicitation arm, and to **≈tie on full / +7% tail** vs the strongest
  (random) item control. The answer model, not the channel, drives most of the geometric advantage.
- **Control:** answer-permutation shuffle on pop_item @q8 = 0.1229 full vs real 0.1268 — item-ask sits barely
  above its own answer-shuffle, confirming items carry little cold signal.

**Bottom line:** on our data the faithful old claim ("concepts beat items in the short cold interview")
**reproduces in direction under both answer models**, is *inflated* by the geometric (circular) answer model,
and is *modest but positive* under honest behavioral answers — concentrated in the TAIL, where items are
cold-unanswerable.
