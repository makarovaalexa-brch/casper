# Paper C Phase 2 — realizable continuous policy via DIFFERENTIABLE UNROLL (frozen V1)
OBJ=ustar (reconstruction 1-cos(u,u*)), CONTMODE=cont (fold raw off-pool point q*_CN, NO snap), GRADED geometric answer.
Actor pi(belief u, turn)->q in R^64; trained by backprop through the full 8-step rollout (u* in gradient/simulator ONLY
=> actor belief-only => REALIZABLE). Eval = VALTEST (te[300:], seed 1, live), SELVAL=tail. Frozen V1 encoder.

Epoch curve (full / tail NDCG@10):
 ep1 0.329/0.107  ep2 0.333/0.124  ep3 0.335/0.127  ep4 0.332/0.131  ep5 0.340/0.139
 ep6 0.346/0.149 (PEAK tail)  ep7 0.345/0.141  ep8 0.346/0.146  ep9 0.344/0.141

PEAK = ep6: FULL 0.346 / TAIL 0.149  (policy_phase2_cont_v1_PEAK_ep6.pt)

## Verdict (single-seed, frozen V1)
vs V1 entropy 0.362/0.139 : continuous WINS tail +0.010, LOSES full -0.016.
vs CASPER-R   0.361/0.150 : ties tail (0.149~0.150), loses full.
=> FIRST REALIZABLE continuous win: the differentiable unroll produces a belief-only continuous policy that
   beats entropy on tail (all prior REINFORCE/distill continuous attempts LOST). Captures ~0.010 of the +0.049
   continuity-oracle tail headroom; FULL is sacrificed (tail-selected + reconstruction objective).
LEVERS to close the rest: Phase 0 (continuous-capable encoder — frozen V1 folds off-pool OOD, cos~0.46 plateau caps it)
   + Phase 3 (NDCG-surrogate reward instead of reconstruction; + answerability for adaptivity). Seed-avg pending.
Code: scripts/paper2/continuous_actor.py  CONTMODE=cont OBJ=ustar GRADED=1 NOBC=1 VALTEST=1 SELVAL=tail.

## Phase 3a (FAILED LEVER): OBJ=bce ranking-surrogate reward through the cont unroll
ep1-3 stuck at full ~0.307 / tail ~0.067 (vs Phase 2 ustar 0.346/0.149). The held-out-BCE gradient through the
8-step rollout is too weak/negative-dominated to shape the query. => Reward swap is NOT the lever on frozen V1.
Diagnosis: Phase 2 reconstruction already learns good queries; FULL (0.346<entropy 0.362) is capped by the FROZEN
ENCODER's OOD folding of off-pool points (cos~0.46 plateau), NOT by the objective. => the real lever is PHASE 0
(continuous-capable encoder), not Phase 3 reward. Keep OBJ=ustar.

## SEED-AVERAGE {1,2,3,7,11} (te[300:], policy_phase2_cont_v1_best ep6)
seed123(dev) 0.3455/0.1494 | s1 0.357/0.148 | s2 0.355/0.139 | s3 0.350/0.148 | s7 0.353/0.142 | s11 0.364/0.155
SEED-AVG: FULL 0.3559 +/- 0.0048 | TAIL 0.1463 +/- 0.0056
vs entropy 0.362/0.139 : tail +0.007 (~1.2 sigma, suggestive NOT significant); full -0.006.
vs CASPER-R 0.360/0.152: tail -0.006; full -0.004.
HONEST VERDICT: first realizable continuous policy in the CASPER-R ballpark; marginally beats entropy tail,
NOT yet a clean win (below CASPER-R). Frozen-encoder OOD cap confirmed => Phase 0 (continuous-capable encoder)
is the necessary lever. SEEDAVG eval block added to continuous_actor.py (NOBC=1 SEEDAVG=1 CONTMODE=cont).
