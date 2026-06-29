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
