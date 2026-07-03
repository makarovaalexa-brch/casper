# TWOPHASE E2 — realizable exposure-model channel switch (Paper B)

**Date:** 2026-07-03. **Pre-registered gate:** `PREREG_TWOPHASE_E2.md` (frozen before this run). **Verdict: FAILURE — the
oracle does not translate.** The learned exposure model captures essentially none of the answerability-oracle's full-NDCG
ceiling; the belief signal adds nothing over popularity; and the primary gate is missed on both the effect-size and the
significance criterion.

## What was tested
Replace the answerability ORACLE in `twophase` (turns 1–4 static entropy concepts → turns 5–8 item questions) with a
**realizable learned exposure model** P(user has rated item | belief u_t, item). Turns 5–8 pick
`argmax_unasked  P(rated | u_t, i) × item-IG` (same POOL_IG signal the oracle ranks by, shifted non-negative). A
predicted-answerable item **not in the known profile is a wasted turn** (not folded — also prevents target leak). This is
the standard Paper-B regime.

- **Exposure model (frozen arch, stated in advance).** Small MLP `Linear(nf,32)-ReLU-Linear(32,32)-ReLU-Linear(32,1)`,
  BCE, Adam 1e-3, 20 epochs, seed 0, TRAIN users only (3862 train / 965 val). Partial-reveal state = belief after the
  static entropy-4 concept prefix (fold only answerable concepts, identical to eval phase 1). Label = **item in the user's
  full profile** (frozen prereg). Belief features `[z(u_t·e_i), z(logpop_i), z(‖u_t‖)]` (nf=3). **Popularity-only
  baseline:** same MLP, single feature `[z(logpop_i)]`, no belief (nf=1). Checkpoint saved durably →
  `data/movielens/.cache/exposure_twophase_e2_s0.pt` (both heads + feature stats + val metrics).
- **Ruler:** eval seeds {1,2,3,7,11}, `te[300:]` (304 users), QPTS 0,2,4,6,8, binary geometric answers, unanswerable asks
  waste the turn. Comparators: entropy-8 (0.3609/0.1397), oracle ceiling `twophase` (0.3728/0.1342). Sanity: `entropy` and
  `twophase` reproduce the canonical/oracle numbers exactly.

## Exposure-model validation (TRAIN val, 965 users)
| model | features | val AUC | precision@4 |
|---|---|---|---|
| **belief** | u·e, logpop, ‖u‖ | **0.7025** | **0.5453** |
| pop-only | logpop | 0.6760 | 0.5031 |

The belief model *is* a better predictor of full-profile membership (+0.027 AUC, +0.042 p@4). The failure is that this
predictive edge does **not** survive to downstream NDCG.

## Result (seed-avg {1,2,3,7,11}, te[300:], NDCG@10 @q8)
| mode | FULL q8 | TAIL q8 | phase-2 answered (turns 5–8) |
|---|---|---|---|
| **entropy** (comparator) | 0.3609 ± 0.0014 | 0.1397 ± 0.0039 | 3.04 / 4 |
| **twophase** (answerability oracle, ceiling) | 0.3728 ± 0.0034 | 0.1342 ± 0.0018 | **3.99 / 4** |
| **twophase_exp** (LEARNED belief) | 0.3623 ± 0.0025 | 0.1330 ± 0.0022 | **1.12 / 4** |
| **twophase_poponly** (popularity exposure) | 0.3664 ± 0.0016 | 0.1323 ± 0.0042 | 1.05 / 4 |

**PRIMARY — FULL@q8 vs entropy, paired per-user bootstrap (10k, pooled over seeds, n=304 users):**
| mode | dFULL | 95% CI | p | gate (≥+0.005 & p<0.05) |
|---|---|---|---|---|
| twophase (oracle) | **+0.0119** | [+0.0007, +0.0233] | 0.036 | (ceiling; marginal per-user) |
| **twophase_exp** | **+0.0014** | [−0.0078, +0.0110] | **0.764** | **FAIL** |
| twophase_poponly | +0.0055 | [−0.0043, +0.0161] | 0.280 | FAIL (not significant) |

**TAIL@q8 vs entropy (honest):** twophase_exp −0.0069 (CI [−0.0173,+0.0032], p=0.19); twophase_poponly −0.0075 (p=0.18).
Both negative as expected; neither exceeds the −0.010 "deployment-costly" flag, and neither is significant.

## Verdict against the gate — FAILURE
1. **Primary missed on both criteria.** twophase_exp FULL@q8 = **+0.0014**, below the +0.005 success threshold and far from
   significant (p=0.76). Per the prereg this records as **"oracle does not translate."** No iteration without new prereg
   (and the "2 more training seeds" clause never triggers because the primary did not pass).
2. **The mechanism is the story, and the mechanism breaks.** The oracle answers **3.99/4** phase-2 turns (it peeks at which
   items the user rated); the learned model answers only **1.12/4**. Its val precision@4 (0.55) is measured against
   *full-profile* membership, but in the ruler only the ~half of rated items in the *known* profile can be folded without
   leaking the held-out target — so useful folds collapse to ~1/4. Three of four "predicted-answerable" item questions are
   wasted, which is exactly why the +0.0119 full ceiling evaporates to +0.0014.
3. **Belief does not add anything — if anything it is a popularity trick, and a weak one.** Despite the belief model's
   higher predictive AUC, popularity-only downstream FULL (+0.0055) is *larger* than the belief model's (+0.0014); belief
   loses to popularity on the metric that matters. And popularity-only is itself not significant per-user (p=0.28). So the
   channel switch is neither a learned win nor even a reliable popularity win.
4. **Even the ceiling is marginal on the pre-registered ruler.** On the per-user bootstrap the oracle only reaches p=0.036
   with a CI that nearly touches zero — the +10.7σ figure in `TWOPHASE_ORACLE_RESULT.md` was per-*seed* (5 low-variance
   points); the prereg-named per-*user* test is the stricter and honest one. There was little realizable headroom to begin
   with, and the realization captures ~12% of it (pop-only ~46%), neither significantly.

**Bottom line:** the "anchor with concepts, then refine with predicted-answerable items" idea is DEAD in its realizable
form. The answerability oracle's full-NDCG edge came from a peek (4/4 answered) that no belief-or-popularity exposure model
reproduces (≈1/4 answered), the tail cost persists, and the belief signal does not beat popularity downstream. Do not
pursue.

## Repro
```
# train (seed 0) + save checkpoint durably:  data/movielens/.cache/exposure_twophase_e2_s0.pt
EXPMODEL=1 EXPSEED=0 REGEN=1 NOBC=1 EP=0 QPTS=0,2,4,6,8 EVALCKS=, \
  EVALBASE=entropy,twophase,twophase_exp,twophase_poponly EVALSEEDS=1,2,3,7,11 \
  EVALCSV=<grid>.csv ANSDUMP=<dump>.csv python -u scripts/paper2/continuous_policy2.py
python scripts/paper2/agg_twophase.py <grid>.csv          # seed-avg q-curves + per-seed deltas
python scripts/paper2/twophase_e2_boot.py <dump>.csv      # paired per-user bootstrap @q8 + phase-2 answered-count
```
(Re-running without `REGEN` loads the saved exposure checkpoint — eval is deterministic given the seed splits.)
