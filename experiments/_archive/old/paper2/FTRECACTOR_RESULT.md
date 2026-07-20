# FTREC × D1-actor zero-shot stack — NEGATIVE: OOD mismatch, no free headline (2026-07-03)

**Question:** the two biggest unclaimed numbers are D1 (continuous actor + divisiveness, frozen V1 recommender,
0.378/0.178) and FTREC (de-OOD fine-tuned recommender + static entropy+graded, 0.385/0.165). Do they stack zero-shot
(actor unchanged, FTREC does the folding/scoring)? Gate: >0.39 = new headline candidate; between = partial transfer;
<0.378 = OOD mismatch → scoping fact for Paper C's FTREC disclosure paragraph.

**Verdict: WELL BELOW both anchors (best variant 0.335/0.125). The FTREC gain is policy-specific and does NOT
transfer to the D1 actor zero-shot.**

## Setup (canonical ruler, eval-only)
- Harness: COMPARE4 ONLYACTOR block of `scripts/paper2/continuous_actor.py` via a patched scratchpad copy
  (`continuous_actor_evalpatch.py`). Seeds {1,2,3,7,11}, te[300:], q8, graded geometric answers.
- Actor: `policy_phase3_d1divw_last.pt` (== locked PAPER_C_CONT_WINNER/policy_cont_actor_DIVW_WINNER.pt, verified
  byte-identical). Recommender swap: `LOADREC=PAPER_C_WINNER/cache/ftrec_best.pt` (sha 5d6406fd = locked; == live
  .cache copy) — replaces the fold encoder AND the Ql readout for scoring, per the existing LOADREC mechanism.
- Protocol fidelity patch (`FROZENANS=1`): the ANSWER always comes from the frozen V1 encoder's u* (exactly the FTREC
  training/eval protocol: "frozen enc generates the answers; the fine-tuned enc builds the belief"). Without this the
  answer model itself would change and the comparison to both anchors would be unfair.
- Two zero-shot wirings: (1) actor conditions on the FTREC belief (full swap); (2) `ACTV1=1` — actor conditions on
  its NATIVE frozen-V1 belief (its training regime), FTREC only folds the transcript and scores.

## Results (FULL / TAIL NDCG@10, seed-avg ±SD)
| config | FULL | TAIL |
|---|---|---|
| SANITY: D1 + V1 on this patched harness | **0.3780 ±0.0032** | **0.1782 ±0.0065** (= canonical exactly) |
| anchor: FTREC + static entropy + graded (README) | 0.385 ±.006 | 0.165 ±.007 |
| anchor: FTRA-mix + entropy (README, ref) | 0.401 | 0.183 |
| **stack v1: D1 actor on FTREC belief, FTREC fold/score** | 0.3131 ±0.0050 | 0.0938 ±0.0028 |
| **stack v2: D1 actor on V1 belief, FTREC fold/score** | 0.3349 ±0.0046 | 0.1248 ±0.0043 |
| (v1 binary / v2 binary) | 0.3188 / 0.3152 | 0.1036 / 0.0982 |

- Baselines verified canonical BEFORE claiming (memory rule): the sanity row reproduces 0.3780/0.1782 to the 4th digit.
- Both wirings are −0.043…−0.065 FULL / −0.053…−0.084 TAIL vs D1+V1. Not "partial transfer" — a hard OOD failure.
- v2 > v1: letting the actor at least SEE its native belief recovers +0.022/+0.031, i.e. part of the failure is the
  actor reading FTREC beliefs it was never trained on. But even with the actor operating fully in-distribution (v2),
  FTREC folding/scoring of the actor's transcript destroys the result.

## Why (mechanism)
FTREC was fine-tuned ONLY on the static entropy questionnaire's belief distribution: tokens = top-divisive CATALOG
entities (pool items + concepts) with graded answers, fold-curriculum t=1..8. The D1 actor emits OFF-MANIFOLD scaled
query directions (qn*_CN) — token vectors FTREC has never folded. Its encoder + adapted Ql readout misfire on that
input distribution (worst on TAIL, where the readout adaptation was concentrated). This is the same
specialization-not-robustness conclusion as the random-policy FTREC control (README 2026-06-27): FTREC's +0.018/+0.008
is the recommender learning the belief distribution it actually sees — change the policy and the gain not only
vanishes, it inverts.

## Consequences
1. **Paper C FTREC disclosure paragraph (scoping fact):** the FTREC gain is claimed only for the policy it was
   co-designed with (static entropy+graded). It does NOT transfer zero-shot to the continuous actor; each headline
   stands on its own recommender (D1 on frozen V1: 0.378/0.178; FTREC/FTRA-mix with entropy: 0.385/0.165, 0.401/0.183).
2. **The stack remains unclaimed and requires TRAINING, not evaluation:** either (a) retrain the actor against the
   FTREC/FTRA-mix recommender (differentiable-unroll or REINFORCE through the new fold — the "fair POLOPT retry"
   already flagged in PAPER_C_WINNER README), or (b) fine-tune the recommender on the D1 transcript distribution
   (FTREC-on-D1, incl. off-manifold tokens in the fold-curriculum). Given (a) has 8 policy-learning negatives behind
   it, (b) is the cheaper, more promising route — flag for the next training campaign (FTRA-mix + off-manifold-token
   curriculum is the natural best-of-both candidate).
3. No dimensionality blocker exists: FTREC's encoder is the same simple-attn Enc class, D=64 in/out; the swap is
   mechanically clean via `LOADREC` — the failure is distributional, not architectural.

## Repro
```
# sanity (canonical): NOBC=1 EP=0 COMPARE4=1 ONLYACTOR=1 ACTORCK=data/movielens/.cache/policy_phase3_d1divw_last.pt \
#   EVALSEEDS=1,2,3,7,11 CONTMODE=cont FEATS=ext,ans ANSF=1 python <patched copy>
# stack v1: + LOADREC=PAPER_C_WINNER/cache/ftrec_best.pt FROZENANS=1
# stack v2: + ACTV1=1
```
Patched copy (FROZENANS/ACTV1, eval-only): session scratchpad `continuous_actor_evalpatch.py`.
