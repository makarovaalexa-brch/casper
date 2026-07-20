# Paper C — what the FROM-SCRATCH continuous actor learned to ask (QPROBE)
policy_phase2_cont_v1_best (from-scratch, differentiable-unroll, OBJ=ustar, frozen V1; NOT distilled). te[300:], 304 users.

emitted-query cosine: nearest ITEM 0.596 | nearest CONCEPT 0.772 | to u* 0.121
=> queries hover NEAR the concept manifold (0.77) but do NOT snap (not 1.0) = off-catalog directions BETWEEN named
   concepts; closer to concepts than items; NOT the u*-shortcut (0.12).

Adaptivity (cos of per-user query to the turn centroid; 1.0=fixed across users, <1=adaptive):
 turn0 1.000 (FIXED learned opener) | turn1 0.642 | t2 0.794 | t3 0.727 | t4 0.696 | t5 0.665 | t6 0.646 | t7 0.647
=> learned strategy = a FIXED informative opener, then ADAPTIVE per-user off-catalog concept-space queries.

IMPLICATIONS: (1) continuity is LEARNABLE FROM SCRATCH (not just a re-expression of discrete; reaches CASPER-R
ballpark 0.356/0.146 seed-avg). (2) The policy is genuinely adaptive after the opener. (3) Queries near-but-off
concepts => SNAP-LOSS test (snap to nearest concept vs not) will quantify the off-catalog value. (4) HYBRID ideas:
BC-pretrain actor on CASPER-R discrete picks (Goal-1 distilled ckpt) THEN continuous-unroll fine-tune; or architecture
that emits concept-mixture coefficients (interpolation) rather than a free 64-d point. Run: NOBC=1 QPROBE=1 CONTMODE=cont.
