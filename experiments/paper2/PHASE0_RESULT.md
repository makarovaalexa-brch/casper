# Paper C Phase 0 — continuous-capable encoder (NCT=6 random off-pool tokens): FAILED LEVER
freeze_concept_encoder_cont.py = V1 recipe (pooled-attn, from scratch, learns enc+Ql+Ec) + NCT=6 CONTINUOUS
off-pool tokens/user (random unit dir, graded a=u*·q). enc_concept_cont.pt. Then Phase 2 continuous policy on it.

Phase 2 on cont encoder (LOADREC=enc_concept_cont, CONTMODE=cont OBJ=ustar GRADED, seed123, EP=20):
 peak ep12 full 0.330 / tail 0.146, then DEGRADES (ep20 0.332/0.127).
vs Phase 2 on FROZEN V1: peak ep6 full 0.346 / tail 0.149.
=> cont encoder WORSE: full -0.016, tail -0.003. PHASE 0 (this variant) FAILED.

DIAGNOSIS: random directions in 64-D are ~orthogonal to taste (u*·q~=0) => near-zero training signal, so the
continuous tokens just DILUTED the base recommender (full dropped) without teaching useful off-pool folding. The
actor at eval emits INFORMATIVE (taste-aligned) directions, not random => train/eval distribution mismatch.
FIX OPTIONS: (a) CO-TRAIN encoder + actor (encoder learns to fold the actor's actual emitted queries); (b) train
on INFORMATIVE off-pool dirs (concept/item interpolations, residual-of-u* dirs), not random; (c) accept frozen V1
as the instrument (realizable continuous policy ~0.356/0.146 seed-avg, ~tied entropy/CASPER-R) and pursue the
ANSWERABILITY/adaptivity lever (Phase 3 bot-play) instead.
Code: scripts/paper2/freeze_concept_encoder_cont.py (NCT knob); continuous_actor.py LOADREC now loads Ec.

## Phase 0 v2 (INFORMATIVE off-pool tokens, RTYPE=info NCT=3 SIGMA=0.7): NEUTRAL — recovers full, matches V1
enc_concept_cont_info.pt = V1 recipe + informative off-pool tokens (u*-aligned+noise & concept-pair interps).
Phase 2 on it (seed123): peak ep10 full 0.342 / tail 0.147; full peaks 0.346 @ep12.
vs frozen V1 0.346/0.149 | vs rand-enc 0.330/0.146.
=> Informative dirs FIX the dilution (full 0.330->0.346) but DON'T beat frozen V1 (tail 0.147 vs 0.149).
CONCLUSION: open-loop encoder training (random OR informative) cannot capture more oracle headroom — the encoder
must learn to fold the ACTOR'S SPECIFIC emitted queries => CO-TRAINING (variant 1) is required. Proceeding to COENC.
