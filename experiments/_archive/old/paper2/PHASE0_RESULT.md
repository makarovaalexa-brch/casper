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

## Variant 1 (COENC co-train encoder+actor, anchor=1.0): COLLAPSED (overfit)
LOADREC=enc_concept_cont_info + COENC=1 COENW=1.0. Train return improved -0.043->-0.017 (cos(u,u*)->1 on TRAIN)
but TEST VAL DEGRADED monotonically: peak ep1 0.299/0.122 -> ep16 0.280/0.085 (<< frozen V1 0.346/0.149, < entropy).
=> co-training lets the encoder WARP the space to fit the actor's queries on train (reward-hack the reconstruction)
   while destroying generalization; anchor=1.0 too weak to prevent it. FAILED.
NET Phase 0: open-loop (v2)=neutral (matches V1); co-train (v1,anchor1.0)=collapse. Frozen V1 remains best instrument;
realizable continuous policy stands at ~0.356/0.146 seed-avg (~= entropy/CASPER-R). Levers left: much STRONGER anchor
(COENW>>1, or freeze enc except a small adapter), OR pivot to answerability/adaptivity (bot-play) lever.

## Variant 1 v3 (COENC warm-actor INIT + ENCLR=1e-4 + anchor): marginal/unstable, NOT a win
INIT=phase2_info_enc actor (warm) + LOADREC=info-enc + COENC ENCLR=1e-4 COENW=1.0. Curve: ep1 0.342/0.139 ->
ep2 0.343/0.145 (tail peak) -> ep4 0.350/0.139 (full peak, +0.004 over info-enc) -> DECLINES ep5-12 to 0.330/0.138.
=> best is full +0.004 @ep4 at the cost of tail (-0.008); unstable (encoder slowly overfits train reconstruction,
   test degrades) even with warm actor + small LR + anchor. NOT a clean win.

OVERALL ENCODER-LEVER CONCLUSION (all variants): random open-loop=diluted(0.330); informative open-loop=neutral(0.346);
co-train anchor1.0 random-actor=collapse(0.280); co-train warm ENCLR1e-4=marginal/unstable(0.350/0.139 peak then decay).
=> reconstruction-based encoder training cannot capture the +0.049 oracle tail headroom (it overfits). Frozen-V1
from-scratch actor 0.356/0.146 remains best realizable. NEXT (architecture, per QPROBE: actor queries cos0.77 to concept
span): CONCEPT-MIXTURE actor head (emit mixture weights over concept bank => query stays IN-DISTRIBUTION for the encoder,
continuous interpolation, no OOD folding by construction). Addresses the diagnosed bottleneck directly.
