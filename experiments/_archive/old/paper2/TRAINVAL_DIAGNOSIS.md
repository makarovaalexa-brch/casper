# Paper C — overfit/underfit diagnosis (answers: data? neurons?)
## Co-train (collapsing) actors = OVERFIT
COENC runs: train obj improved (1-cos 0.96->0.98) while VAL degraded (tail 0.122->0.085). Train UP + val DOWN =
overfit; trainable encoder memorizes train. Fix = regularize / freeze enc / less freedom, NOT more neurons.

## Good frozen-V1 from-scratch actor = UNDERFIT / CEILING (not overfit, not data-limited)
TRAINVAL (policy_phase2_cont_v1_best, same elicitation NDCG + recon-cos):
  TRAIN (n=400): FULL 0.2907 TAIL 0.1135 recon-cos 0.712
  TEST  (n=304): FULL 0.3606 TAIL 0.1493 recon-cos 0.712
=> recon-cos IDENTICAL train==test (0.712) => ZERO generalization gap => NOT overfit. (test NDCG > train = denser
   train-profile population effect, not overfit.)
CONCLUSIONS:
 - MORE DATA won't help: no train/test gap to close (actor already generalizes perfectly).
 - MORE NEURONS won't help: can't even fit TRAIN past cos 0.71 (not a memorization-capacity problem).
 - The CEILING = the reconstruction objective (1-cos saturates ~0.71) + 8 graded answers + encoder folding.
   The ORACLE hits 0.39 tail with the SAME 8 questions => headroom is in HOW QUESTIONS ARE SCORED, not how many.
=> LEVER = the OBJECTIVE. Need a RANKING-ALIGNED differentiable surrogate the unroll can optimize (listwise/LambdaRank
   or recon+rank hybrid). OBJ=bce (Phase 3a) was right instinct, wrong surrogate (BCE-through-unroll gradient too weak).
Diagnostic: TRAINVAL=1 block in continuous_actor.py.
