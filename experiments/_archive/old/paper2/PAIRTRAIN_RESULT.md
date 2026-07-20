# PAIR-NATIVE TRAINING — realizing the query as an item-pair difference IN THE LOOP rescues most of the snap-loss

**Date:** 2026-07-02. **Question:** post-hoc PAIRSNAP (PAIRSNAP_RESULT.md) showed that realizing D1's off-manifold
queries as item-pair differences AFTER training lands on the concept-snap floor (0.337/0.134, −0.041/−0.044 vs free D1).
Is that because the pair-reachable set is genuinely weaker, or because the policy was never trained to act within it?
Here we train PAIR-NATIVELY: inside the differentiable unroll, every emitted query q is replaced by its best item-pair
difference d = (e_i − e_j)/||e_i − e_j|| (straight-through: forward folds d, backward treats d ≈ q), so the policy
learns while experiencing the realized action.
**Verdict: pair-native training recovers ~3/4 of the snap-loss. Warm-started from D1, across 3 training seeds it
reaches 0.3678 ± 0.0018 / 0.1641 ± 0.0023 (each: seed-avg {1,2,3,7,11}, te[300:], q8, graded) — +0.031/+0.030 over
post-hoc pair-snap, at-or-slightly-above uent+GRAW (0.3667/0.1577; tail delta positive at every training seed but NOT
per-user significant, paired bootstrap p=0.30), and −0.010/−0.014 below the free (un-realizable) D1 headline. A
deployable "would you rather A than B?" policy is restored from clearly-losing (post-hoc snap) to the top of the
deployable table. Surprise: the win does NOT come from the policy migrating onto the pair-reachable set — realization
cos(q,d) FELL slightly during training (0.70 → 0.68); the policy instead learns to pick queries whose REALIZED pair
direction folds informatively. From-scratch pair-native training fails (0.344/0.125, cos 0.62); warm-start required.**

## Mechanism (PAIRTRAIN=1 in scripts/paper2/continuous_actor.py)
Identical to the D1 training recipe (differentiable unroll, OBJ=ustar reconstruction loss, divisiveness field
DIVW=1.0 DTAU=2.0 NUMAT=2500, graded geometric answers, CONTMODE=cont, frozen V1, EP=10, LR 5e-4), except each turn
inside `rollout()`:

    emit q  →  qn = q/||q||
    d = argmax_{i≠j} cos(qn, e_i − e_j)   (top/bottom-K=64 trick over the FULL N=3706 item set, == PAIRSNAP search,
                                            torch-batched; verified exact vs the numpy search: 0/50 mismatches)
    qn ← qn + (d − qn).detach()            # STRAIGHT-THROUGH: fwd = d, bwd gradient flows as if d ≈ q
    fold (qn·_CN, graded answer along qn)  # divfield reward + graded answer + encoder all see d in the forward pass

Per-epoch mean realization cos(q,d) is logged. The per-epoch VAL rollout (te[:300], disjoint from test) also
pair-realizes and uses the graded answer along d — validation IS the deployment ruler; best checkpoint by val TAIL
(SELVAL=tail), never test-peeked.

## Result (canonical PAIRSNAP eval harness: seed-avg {1,2,3,7,11}, te[300:], q8, graded answer along d)
| condition | FULL | TAIL |
|---|---|---|
| D1 free / un-snapped (NOT deployable as a named question) | 0.3780 ± 0.0032 | 0.1782 ± 0.0065 |
| **PAIR-NATIVE, warm from D1, TRSEED=0 (best-val ep4)** | **0.3701 ± 0.0047** | **0.1659 ± 0.0049** |
| PAIR-NATIVE, warm, TRSEED=1 (best-val ep9) | 0.3676 ± 0.0035 | 0.1655 ± 0.0036 |
| PAIR-NATIVE, warm, TRSEED=2 (best-val ep5) | 0.3658 ± 0.0044 | 0.1608 ± 0.0039 |
| **PAIR-NATIVE mean ± std across 3 training seeds** | **0.3678 ± 0.0018** | **0.1641 ± 0.0023** |
| uent+GRAW (static entropy, graded, answerable items; canonical) | 0.3667 ± 0.0045 | 0.1577 ± 0.0078 |
| entropy (binary, canonical) | 0.3618 ± 0.0026 | 0.1393 ± 0.0041 |
| D1 post-hoc PAIR-SNAP (no pair-in-the-loop training) | 0.3373 ± 0.0039 | 0.1344 ± 0.0046 |
| D1 post-hoc concept-snap (SNAPLOSS) | 0.3414 | 0.1384 |
| PAIR-NATIVE **from scratch**, TRSEED=0 (diagnostic; cos 0.62) | 0.3442 ± 0.0044 | 0.1248 ± 0.0056 |

- Per-eval-seed (TRSEED=0): s1 0.3713/0.1659, s2 0.3689/0.1651, s3 0.3617/0.1611, s7 0.3736/0.1625, s11 0.3751/0.1751.
- **Training-seed robustness gate PASSED (with the split-leak FIXED — clean splits): tail delta vs uent+GRAW (0.1577)
  is POSITIVE at every training seed: ts0 +0.0082, ts1 +0.0078, ts2 +0.0031. FULL deltas: +0.0034 / +0.0009 / −0.0009 (~tie).**
- vs post-hoc pair-snap: **+0.031 FULL / +0.030 TAIL** (3-seed mean) — ~3/4 of the −0.041/−0.044 snap-loss recovered.
- vs free D1 0.378/0.178: −0.010/−0.014 — the residual, now honest, cost of deployability (asking realizable
  pair questions instead of un-nameable directions).

## Realization-cosine trajectory (the diagnosis)
Warm run, per-epoch mean cos(q,d) over 408 train-rollout turns:
ep1 0.7015 → ep2 0.6994 → ep3 0.6925 → ep4 0.6888 (best-val) → ep5 0.6880 → … → ep10 0.6834.
Eval-time distribution (best ckpt, 2432 queries): mean 0.6836, median 0.6893, p10 0.5990, p90 0.7430, min 0.5150
(D1 post-hoc reference: mean 0.7165).
- **cos did NOT climb — it drifted DOWN ~0.02 while val tail rose 0.114 → 0.124.** The policy does not learn to stay
  reachable; it learns to EXPLOIT the realization map: choose q (possibly less aligned with its own realization) whose
  realized pair-direction d folds well under the graded answer. Straight-through training compensates for the
  realization distortion instead of eliminating it.
- This reframes the PAIRSNAP negative: the off-manifold residual is load-bearing only for a policy trained WITHOUT the
  realizer in the loop. Once the realizer is inside the unroll, the pair-reachable set supports nearly the whole win —
  the failure was an optimization/train-test mismatch, not an expressiveness gap of pair questions.

## Paired per-user bootstrap (PAIRBOOT block; the honest significance test)
A = pair-native ts0 best (pair-realized graded roll) vs B = uent+GRAW (static entropy top-8 unified pool, raw-dot
graded answers), IDENTICAL users/splits across the 5 eval seeds, per-user seed-averaged NDCG, 10k bootstrap resamples
over the 304 users (harness faithfulness: A tail 0.1656 ≈ eval 0.1659; B tail 0.1572 ≈ canonical 0.1577):
- FULL: delta **+0.0035**, 95% CI [−0.0070, +0.0143], **p = 0.51**
- TAIL: delta **+0.0084**, 95% CI [−0.0075, +0.0237], **p = 0.30**
=> The tail edge is consistent in SIGN everywhere (all 5 eval seeds at every training seed, all 3 training seeds) but
is NOT per-user significant — per-user NDCG@10 noise dwarfs a +0.008 mean with n=304. Honest claim: pair-native
training restores the deployable policy to PARITY-OR-SLIGHTLY-ABOVE the best static graded baseline (vs the clear
LOSS of post-hoc snapping, whose −0.023 deficit vs uent+GRAW is the thing rescued); it is NOT a significant win over
uent+GRAW. (Note: the same caveat applies to most seed-level "σ" claims in this project — seed-σ understates
user-level noise.)

## Test-time field-scored pair re-rank (PAIRFS, negative ablation)
Among top-M=32 pairs by cos(q,d), fold the pair maximizing the continuous field score instead of pure cos
(ts0 best checkpoint, same ruler):
| re-rank | FULL | TAIL | realization cos |
|---|---|---|---|
| none (argmax cos, headline) | 0.3701 ± 0.0047 | 0.1659 ± 0.0049 | 0.684 |
| divisiveness only | 0.3565 ± 0.0042 | 0.1551 ± 0.0035 | 0.637 |
| div + pop + rat (w=1,1,1) | 0.3636 ± 0.0066 | 0.1532 ± 0.0042 | 0.643 |
=> Both HURT (−0.011 to −0.014 full, −0.011 to −0.013 tail). The policy's emitted direction already encodes where it
wants to probe (divisiveness was in its training reward); overriding the realization with field greed moves the folded
direction away from the query (cos 0.68→0.64) and loses more than the field gains. Faithful cos-realization is the
right test-time rule; no free margin here.

## From-scratch control (FAILED — warm-start is required)
Same recipe from random init (TRSEED=0): val stuck at 0.303–0.309 / 0.094–0.100 over 5 epochs, realization cos flat
~0.60–0.63; killed at ep5 per the pre-registered gate (val tail < 0.13). Its best-val checkpoint evaluates at
0.3442 ± 0.0044 / 0.1248 ± 0.0056 (canonical ruler, cos 0.6209) — below binary entropy on tail, at the concept-snap
floor. Consistent with the project's "pretraining essential" lesson: the straight-through gradient (biased: bwd
pretends d = q) appears too crude to bootstrap a policy from scratch, but is fine for FINETUNING a good off-manifold
policy onto the reachable set.
(peak files: peak_pairtrain_ts0.txt val 0.3076/0.1002 @ep5; peak_pairtrain_d1warm.txt val 0.3203/0.1237 @ep4;
ts1 0.3244/0.1227 @ep9; ts2 0.3208/0.1252 @ep5.)

## History note (SNAPTRAIN, concept vocabulary)
Concept-snap-in-the-loop WAS tried before (SNAPBANK_RESULT.md): warm-D1 REINFORCE over a categorical snap to the rich
6724-phrase bank tied naive snap (0.334/0.139) — but that was REINFORCE on a frozen-ish D1, not the differentiable
straight-through unroll used here. Pair-native + straight-through is the combination that works (richer realizable set
AND gradients through the realized rollout).

## ⚠ TRSEED SPLIT-LEAK found & fixed (affects prior TRAINSEED_RESULT ts1/ts2 too)
First TRSEED=1 relaunch showed ep1 val 0.360/0.147 (vs ts0's 0.324/0.114) — impossible from init noise alone. Cause:
`TRSEED` seeds the module rng that ALSO shuffles the user list for the train/test split (continuous_actor.py L21), so
TRSEED≠0 MOVES the split → canonical te[300:] users enter TRAINING and the val set changes. FIXED in-place
(2026-07-02): the split is now always the rng(0) order (TRSEED=0 byte-identical; TRSEED still varies torch init +
training sampling). The contaminated ts1 run was killed at ep1 and its checkpoints purged before any eval.
**NOTE: the same leak was present in TRAINSEED_RESULT.md's D1/CASPER-R ts1/ts2 rows** (same TRSEED mechanism in BOTH
continuous_actor.py and continuous_policy2.py; both patched 2026-07-02) — under a reshuffled split ~80% of canonical
test users land inside training; those ts1/ts2 numbers should be re-run. The doc's claim "data splits stay
deterministic (range-based)" was incorrect.

## Honest caveats
1. Same answer-model caveat as PAIRSNAP: pair realization is GEOMETRIC (graded answer along d at the q̂·_CN fold
   magnitude). A deployed "A rather than B?" question needs a comparative answer model; here only the direction is
   realized. Pairs are also unconstrained by answerability/popularity.
2. The straight-through estimator is biased; the from-scratch failure suggests the method only works near a good init.
3. Warm-start = the LOCKED D1 checkpoint (read-only; `policy_phase3_d1divw_last.pt`, sha a63fec1e). Locked artifacts
   untouched; new checkpoints under fresh TAGs `pairtrain_*`.

## Checkpoints / repro
- `data/movielens/.cache/policy_pairtrain_d1warm{_best,_last,_ep1..10}.pt` (TRSEED=0; _best = ep4, the headline),
  `policy_pairtrain_d1warm_ts{1,2}*.pt` (_best = ep9 / ep5), from-scratch `policy_pairtrain_ts0*.pt`.
  Peak files `peak_pairtrain_*.txt`; run rows in `policy_runs.tsv`.
- Logs: `experiments/paper2/pairtrain_ts0.log` (scratch), `pairtrain_d1warm.log` (headline train),
  `pairtrain_d1warm_eval.log` (headline eval), `pairtrain_ts12.log` (TRSEED 1/2 + from-scratch eval; NUL-garbled
  prefix from a killed contaminated run — strip \x00), `pairboot_fs.log` (bootstrap + PAIRFS).
```
# train (headline; ts1/ts2 = TRSEED=1/2, TAG=pairtrain_d1warm_ts{1,2}):
PAIRTRAIN=1 TRSEED=0 INIT=data/movielens/.cache/policy_phase3_d1divw_last.pt DIVW=1.0 DTAU=2.0 NUMAT=2500 \
  CONTMODE=cont GRADED=1 OBJ=ustar NOBC=1 FEATS=ext,ans EP=10 SELVAL=tail TAG=pairtrain_d1warm \
  python scripts/paper2/continuous_actor.py
# eval (canonical ruler):
NOBC=1 EP=0 PAIRSNAP=1 EVALSEEDS=1,2,3,7,11 CONTMODE=cont FEATS=ext,ans ANSF=1 \
  ACTORCK=data/movielens/.cache/policy_pairtrain_d1warm_best.pt python scripts/paper2/continuous_actor.py
# paired per-user bootstrap:  PAIRBOOT=1 (same env as eval)
# field re-rank ablation:     PAIRSNAP=1 PAIRFS=1 PAIRM=32 [FSP=1 FSR=1] DIVW=1.0 (fields needed)
```
