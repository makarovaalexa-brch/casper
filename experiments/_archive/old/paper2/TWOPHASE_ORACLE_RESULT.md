# TWOPHASE CHANNEL-SWITCH ORACLE CEILING (Exp1, Paper B)

**Date:** 2026-07-03. **Question:** is the user's "concepts anchor → predicted-answerable items refine" idea worth
building a realizable exposure model for? Measure its **oracle ceiling**: 4 static-entropy concept questions, then 4
item questions drawn from the user's *actually-rated* items (answerability peek only — never peek at answer values or
held-out targets), picking the most-informative rated item per turn.

## Protocol (faithful Paper-B ruler)
- Harness: `scripts/paper2/continuous_policy2.py`, new modes `twophase` / `twophase_pop` (env-gated by `MODES`/`EVALBASE`;
  defaults unchanged). Eval-only (`NOBC=1 EP=0`), seed-avg **{1,2,3,7,11}**, `te[300:]` (304 users), binary geometric
  answers, unanswerable asks waste the turn. `QPTS=0,2,4,6,8`.
- **Phase 1 (turns 1–4):** static entropy over the 761 concepts — *identical* ranking/fold to the canonical `entropy`
  baseline (verified: all three modes give byte-identical q0/q2/q4 rows).
- **Phase 2 (turns 5–8):** item questions restricted to the user's rated pool items (`PITEMS ∩ profset` — the
  answerability peek: we only ask items we know the user rated, so every phase-2 turn is answered; ans/8 rises 5.1→6.1).
  Selection signal (no answer/target peek):
  - `twophase` → most-informative rated item by **POOL_IG** (population single-fold info-gain, precomputed offline, no
    per-user leak). This is "the existing item informativeness signal" the request named.
  - `twophase_pop` → most-**popular** rated item by train like-count `cnt`.
- Answers fold the item's real residual `resid[x][j]` (the answer to the question actually asked — legitimate, same as
  concept answers). Selection never uses it.

## Result (seed-avg {1,2,3,7,11}, te[300:], NDCG@10)

| mode | q0 | q2 | q4 | q6 | q8 |
|---|---|---|---|---|---|
| **entropy** FULL | 0.3099 | 0.3482 | 0.3588 | 0.3597 | **0.3609 ± 0.0014** |
| **twophase** FULL | 0.3099 | 0.3482 | 0.3588 | 0.3717 | **0.3728 ± 0.0034** |
| **twophase_pop** FULL | 0.3099 | 0.3482 | 0.3588 | 0.3661 | **0.3677 ± 0.0042** |
| **entropy** TAIL | 0.0808 | 0.1201 | 0.1353 | 0.1394 | **0.1397 ± 0.0039** |
| **twophase** TAIL | 0.0808 | 0.1201 | 0.1353 | 0.1345 | **0.1342 ± 0.0018** |
| **twophase_pop** TAIL | 0.0808 | 0.1201 | 0.1353 | 0.1292 | **0.1259 ± 0.0063** |

**@q8 delta vs entropy (paired over 5 seeds):**
| mode | dFULL | dTAIL |
|---|---|---|
| twophase (IG-selected items) | **+0.0119** (sd 0.0025, **+10.7σ**) | **−0.0056** (sd 0.0027, −4.7σ) |
| twophase_pop (popular items) | +0.0069 (sd 0.0053, +2.9σ) | −0.0138 (sd 0.0066, −4.7σ) |

Sanity: `entropy` @q8 reproduces the canonical **0.3609/0.1397** exactly.

## Verdict — METRIC-SPECIFIC ceiling; alive for FULL, DEAD for TAIL
Against the request's gate (< +0.005 → dead; ≥ +0.01 → pre-register the realizable version):

- **FULL NDCG: ALIVE.** The oracle ceiling is **+0.0119 ≥ +0.01** (10.7σ). Two informative rated-item questions after
  the 4-concept anchor beat eight concept questions on full-catalog NDCG (q6 twophase 0.3717 already > entropy's 8-question
  0.3609). This justifies pre-registering a realizable **exposure-model** version (predict which items the user has
  consumed, then ask the informative ones) **if the target metric is full-catalog NDCG**.
- **TAIL NDCG: DEAD / counterproductive.** Even the *oracle* ceiling is **−0.0056** (twophase) to **−0.0138**
  (twophase_pop), both ≈−4.7σ. The rated-item questions are popular-item directions: they push popular recommendations up
  (helping FULL) but *displace* the tail signal that was the concept channel's entire differentiator vs popularity. Since
  Paper B's headline metric is TAIL, the idea does not survive there — no exposure model can rescue a negative ceiling.

**Bottom line:** the "anchor-then-refine-with-items" idea has real headroom, but only on the head-heavy full metric, and it
buys that head gain by *spending* tail accuracy. It is not a free refinement of the concept channel; it is a head/tail
trade. Recommend: pre-register the realizable exposure-model version **only** if a full-NDCG deployment is the target;
do **not** pursue it for the tail story.

## Why the concept channel saturates at ~4–5 while items keep adding (effective-rank)
`scripts/paper2/effrank_concept_vs_item.py` — eigenspectrum of the 761 concept direction vectors vs the 600 pool-item
direction vectors, both unit-normalized (participation ratio PR = (Σλ)²/Σλ² = effective dimensionality).

| set | PR (uncentered) | PR (centered) | top-4 var | top-5 var | top-8 var |
|---|---|---|---|---|---|
| **concepts (761 tags)** | **2.25 / 64** | 1.96 / 64 | 91.8% | 93.2% | 95.4% |
| **pool items (top-600)** | **27.13 / 64** | 27.60 / 64 | 29.4% | 33.7% | 43.7% |

The concept directions collapse onto a **~2-effective-dimensional** subspace (one dominant axis explains 64–70%; the top
5 explain 93%). Once ~4–5 divisive concepts are answered, additional concepts are near-linear combinations of what's
already known → the concept belief saturates (matches the empirical q4≈q8 plateau: entropy 0.3588→0.3609). Pool items span
a **~27-effective-dimensional** subspace, so each new item question genuinely adds an independent direction — which is why
items keep moving the belief past q4. The catch (see verdict): those extra item directions are popular/head directions,
so the added dimensions help FULL NDCG, not TAIL.

## Repro
```
NOBC=1 EP=0 QPTS=0,2,4,6,8 EVALCKS=, EVALBASE=entropy,twophase,twophase_pop EVALSEEDS=1,2,3,7,11 \
  EVALCSV=<out>.csv python -u scripts/paper2/continuous_policy2.py
python scripts/paper2/effrank_concept_vs_item.py
```
