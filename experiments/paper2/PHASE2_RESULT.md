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
