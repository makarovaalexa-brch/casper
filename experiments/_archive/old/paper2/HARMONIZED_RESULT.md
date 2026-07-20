# Paper C — HARMONIZED milestone (continuous actor in the CANONICAL Paper B harness)
continuous_policy2_st.py 'contactor' mode = continuous actor rolled out through the EXACT Paper B run()/metr()/split.
Same V1 encoder (enc_concept.pt) as canonical. seed-avg {1,2,3,7,11}, te[300:], q8, GANS=1 (graded answer for actor).

| policy            | FULL          | TAIL          |
|-------------------|---------------|---------------|
| continuous actor  | 0.3664±0.0029 | 0.1622±0.0051 |
| CASPER-R (entdistill_ep4) | 0.3510±0.0046 | 0.1482±0.0044 |
| entropy           | 0.3414±0.0033 | 0.1446±0.0036 |
| conc_pop          | 0.3466±0.0027 | 0.1266±0.0045 |

WIN: continuous actor > CASPER-R +0.0154 full / +0.0140 tail (~3sigma). REPLICATES in BOTH harnesses (this + COMPARE4
in continuous_actor.py: actor 0.366/0.162, casper 0.352/0.148). Both use identical V1 encoder.
OFFSET: CASPER-R 0.351/0.148 here vs paper-table 0.359/0.152 (~0.008 full/0.004 tail; applies to ALL policies equally;
likely opener-handling detail). Even crediting CASPER-R canonical 0.359/0.152, actor 0.366/0.162 wins both axes.
