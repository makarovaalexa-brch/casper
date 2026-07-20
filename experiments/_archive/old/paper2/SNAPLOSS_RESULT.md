# Paper C — SNAP-LOSS (the novelty centerpiece): off-manifold queries are LOAD-BEARING
Continuous actor on canonical harness, GRADED answers, seed-avg {1,2,3,7,11}, te[300:], q8.
SNAP=1 snaps each emitted query to its nearest ANSWERABLE concept (cosine) before folding.

| condition                  | FULL   | TAIL   |
|----------------------------|--------|--------|
| UN-SNAPPED (off-manifold)  | 0.3664 | 0.1622 |
| SNAPPED (to nearest concept)| 0.3276 | 0.1310 |
| cost of snapping           | -0.0388| -0.0312|

VERDICT (decisive):
1. Off-manifold queries are load-bearing: snapping to the nearest nameable concept costs -0.039 full/-0.031 tail.
2. SNAPPED (0.328/0.131) FALLS BELOW CASPER-R (0.360/0.152) => a Wolpertinger/PEBOL-style emit-then-SNAP would LOSE
   to the discrete SOTA. The win REQUIRES not snapping.
3. This is the load-bearing novelty differentiator (novelty agent): every competitor snaps; none claims un-snapped
   beats discrete. SNAP-LOSS is the empirical centerpiece.
Code: SNAP=1 in the 'contactor' mode (continuous_policy2_st.py); cans-scope bug fixed (contactor now in cans list).

## UPDATE 2026-06-30: snap-loss measured on the HEADLINE D1 (divisiveness), not the recon variant
COMPARE4 ONLYACTOR graded, seed-avg {1,2,3,7,11}, ACTSNAP=1 snaps each query to nearest answerable concept:
- D1 un-snapped: 0.3780/0.1782 ; D1 SNAPPED: 0.3414/0.1384 ; snap-loss -0.037 FULL / -0.040 TAIL.
- Snapped D1 (0.341/0.138) < discrete CASPER-R (0.360/0.152) -> emit-then-snap LOSES to discrete for the headline too.
- D1 is concept-LEANING (QVIZ cos 0.58 movie / 0.73 concept) yet the off-concept RESIDUAL is load-bearing (-0.040 tail).
Code: ACTSNAP env in continuous_actor.py COMPARE4 actor roll. Paper tab:snaploss now uses these headline numbers.
