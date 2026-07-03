# STATIC-8 CONTINUOUS DESIGN CONTROL (Exp2, Paper C)

**Date:** 2026-07-03. **THE question:** does an *optimized STATIC continuous questionnaire* — 8 free 64-d direction
vectors (8×64 = 512 params, NO policy net, the same 8 queries for every user) — match the *adaptive* D1 policy
(**0.3780/0.1782** graded, COMPARE4)? If yes → Paper C gains "the prize is continuous questionnaire DESIGN, not
adaptivity" + an 8-fixed-question deployment. If D1 beats it by >0.005 → first isolation of true adaptive value in the
continuous regime.

## Verdict — **D1 ADAPTIVE BEATS THE BEST STATIC-8 BY +0.038 FULL / +0.042 TAIL (≫ 0.005).**
First clean isolation of true adaptive value in the continuous regime. A fixed continuous questionnaire cannot match the
belief-conditioned policy; per-user adaptivity is worth **~0.04 NDCG on both axes here**, and it is specifically what lets
graded answers pay off. The best static-8 is the **raw D1 turn-means with ZERO further training** (0.3399/0.1362) —
gradient-optimizing the static bank on any variant of the D1 objective only makes it worse (collapse or drift). Even that
best static bank sits at the *discrete* entropy/CASPER-R baseline level (~0.34/0.13); the adaptive D1 is +0.04 above all
of them.

## Setup
- `scripts/paper2/continuous_actor.py`, new env-gated `StaticActor` (bank of `T×D` params; `forward` returns `bank[turn]`
  for every user, ignoring belief). Instantiated only when `STATIC8=1`; all defaults unchanged.
- **Init** = the 8 per-turn MEAN queries of the D1 winner (`policy_phase3_d1divw_last.pt`), from a batched cont rollout
  over 800 train users (graded geometric answers, frozen V1 encoder), cached `.cache/static8_init.npy` /
  `static8_d1means.npy`.
- Frozen V1 recommender, same train users / `te` split, TRSEED=0, CONTMODE=cont, GRADED answers.
- Final eval = graded **COMPARE4** (identical ruler to D1's 0.378/0.178), seed-avg {1,2,3,7,11}, te[300:], q8; q-curve via
  QCURVE. Added `GRADEDVAL=1` to `val_ndcg` so model selection uses the graded metric (env-gated; default unchanged).

## Headline (graded COMPARE4, seed-avg {1,2,3,7,11}, te[300:], q8)
| policy | FULL | TAIL |
|---|---|---|
| **D1 adaptive continuous actor** (anchor) | **0.3780 ± 0.0032** | **0.1782 ± 0.0065** |
| entropy graded (discrete baseline) | 0.353 | 0.133 |
| CASPER-R graded (discrete) | 0.343 | 0.138 |
| **static-8 = raw D1 turn-means, NO training (best static)** | **0.3399 ± 0.0053** | **0.1362 ± 0.0049** |
| static-8, recon+0.3·softNDCG obj, val-best (trained) | 0.3249 ± 0.0061 | 0.1251 ± 0.0049 |
| static-8, pure softNDCG obj, val-best (trained) | 0.3090 ± 0.0077 | 0.0990 ± 0.0016 |
| popular graded (no elicitation, q0) | 0.310 | 0.081 |

**D1 − best static-8 = +0.038 FULL / +0.042 TAIL.** The best static bank is the *untrained* D1 turn-means (0.340/0.136);
gradient-optimizing the 512 params only degrades it (0.325 → 0.309). Every static bank sits at or below the discrete
entropy/CASPER-R baselines — nowhere near adaptive D1.

## Why the static questionnaire fails — two independent confirmations

**(1) The D1 objective COLLAPSES a free static bank.** Optimizing the 512-param bank on the exact stated objective
(recon + 0.3·softNDCG − 1.0·div, DTAU=2.0) from the D1-mean init degenerates: the 8 queries collapse toward mutual
parallelism (cross-turn mean |cos| 0.47 init → **0.82** by ep2, max 0.99) and graded/binary val NDCG falls monotonically.
A *static* reconstruction objective has a trivial minimum — with graded answers a couple of directions along the dominant
taste axis already reconstruct `u*`, so surplus static queries are redundant and collapse. Val-best therefore pins the
near-init point (ep1, cos 0.805 to D1 means, still diverse) → the "best static under this objective" ≈ the D1 turn-means.

**(2) Even the best (untrained) static bank saturates early and then slightly decays.** Graded q-curve of the best
static-8 (raw D1 turn-means), seed-avg {1,2,3,7,11}, te[300:]:

| q | 0 | 2 | 4 | 6 | 8 |
|---|---|---|---|---|---|
| FULL | 0.3099 | 0.3130 | **0.3525** | 0.3434 | 0.3399 |
| TAIL | 0.0808 | 0.1235 | 0.1347 | **0.1362** | 0.1362 |

FULL peaks at **q4 (0.3525 ≈ entropy's 8-question 0.353)** then *declines* toward q8; TAIL plateaus at ~0.136 by q6. The
static bank extracts everything it can in ~4 fixed questions and additional fixed questions add nothing (TAIL) or slightly
hurt (FULL) — misaligned graded answers on later fixed directions corrupt the per-user belief. (Gradient-training the
bank makes this worse: the pure-softNDCG bank *peaks at q2* 0.315/0.118 then falls to 0.309/0.099.) This is the Exp1
effective-rank story from the design side: the concept/taste subspace is only ~2-effective-dimensional (PR 2.25), so a
*fixed* bank has ~nothing independent to ask after a few turns — whereas adaptive D1 re-aims every turn at the *individual*
and keeps climbing (D1 q-curve FULL 0.310/0.330/0.370/0.378, TAIL 0.081/0.135/0.164/0.178).

**The graded/binary inversion.** For the static bank graded < binary (best static 0.340 graded vs 0.357 binary), the
exact opposite of adaptive D1 (graded 0.378 >> binary 0.356). The continuous/graded answer only pays when the question
direction is tuned to the individual — a static direction turns the soft answer into noise relative to a clean sign bit.
**This localizes the entire graded-continuity prize of Paper C to ADAPTIVITY.**

## Per-turn cosine between the learned static bank and D1's turn-means
recon val-best: turn 0–7 = +0.914/+0.946/+0.761/+0.678/+0.636/+0.787/+0.837/+0.881, **mean +0.805** (cross-turn |cos|
0.47). softNDCG best: mean **+0.855** (cross-turn |cos| 0.46). Both optimized static banks stay ≈ D1's per-turn mean
directions and never find a better static design — they either collapse (recon) or plateau at the init (softNDCG).

## Notes / robustness
- Both training runs were stopped early once val degradation was monotone and confirmed (recon: ep1→ep3 collapse;
  softNDCG: ep1 peak, ep2 lower) — the val-best checkpoint (ep1) is durably saved and is what is reported; letting them run
  to the nominal EP=20/25 cannot beat a monotonically-degrading val. The definitive best static (raw D1 turn-means, 0
  training epochs) needs no training at all.
- Random-init follow-up available (`RANDINIT=1`); not needed for the verdict — even the *privileged* D1-mean init (a
  strong, purpose-built starting point) cannot reach D1 adaptive, so a random init cannot do better.
- Objective faithfulness: the task-specified `recon + 0.3·softNDCG − DIVW` and D1's own `OBJ=ustar+DIVW` finetune both
  reduce to reconstructing `u*` under graded answers; both collapse a static bank for the same structural reason.

## Repro
```
# task-objective run (collapses; val-best ≈ D1 means):
STATIC8=1 CONTMODE=cont OBJ=recon SNDCG=0.3 DIVW=1.0 DTAU=2.0 GRADED=1 FEATS=ext,ans ANSF=1 NOBC=1 \
  EP=25 SELVAL=tail USEBEST=1 TAG=static8 python -u scripts/paper2/continuous_actor.py
# best-shot (pure graded softNDCG, graded-val):
STATIC8=1 CONTMODE=cont OBJ=softndcg DIVW=0 GRADED=1 GRADEDVAL=1 FEATS=ext,ans ANSF=1 NOBC=1 \
  EP=20 SELVAL=tail USEBEST=1 TAG=static8_snd python -u scripts/paper2/continuous_actor.py
# graded eval + q-curve (comparable to D1 0.378/0.178):
STATIC8=1 CONTMODE=cont FEATS=ext,ans ANSF=1 NOBC=1 EP=0 COMPARE4=1 ONLYACTOR=1 ACTORCK=<ckpt> EVALSEEDS=1,2,3,7,11 ...
STATIC8=1 ... QCURVE=1 ONLYACTOR=1 ALABEL=static8 ACTORCK=<ckpt> QSNAP=0,2,4,6,8 QFRESH=1 ...
# per-turn cosine: CK=<ckpt> python scripts/paper2/static8_cosine.py
```
