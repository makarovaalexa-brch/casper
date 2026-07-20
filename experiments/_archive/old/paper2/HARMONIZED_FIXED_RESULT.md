# Paper C — DEFINITIVE harmonized result (CORRUPTED-CACHE BUG FIXED)
BUG: data/movielens/.cache/pool_entavg.npy was corrupted (sha 09feb829) during this session -> depressed every policy
that reads POOL_ENT (entropy heuristic + CASPER-R scorer features). REGEN restored sha 156c072c (= guarded copies);
baselines snap back to canonical. The continuous actor was UNAFFECTED (does not use POOL_ENT) -> its number unchanged.

DEFINITIVE seed-avg {1,2,3,7,11}, te[300:], q8, canonical Paper B harness (continuous_policy2_st.py, contactor mode):
| policy            | FULL          | TAIL          |
|-------------------|---------------|---------------|
| continuous actor (graded) | 0.3664±0.0029 | 0.1622±0.0051 |
| CASPER-R (binary) | 0.3602±0.0033 | 0.1522±0.0025 |  <- = canonical paper 0.359/0.152
| entropy (binary)  | 0.3608±0.0016 | 0.1398±0.0040 |  <- = canonical paper 0.360/0.141
| conc_pop          | 0.3466±0.0027 | 0.1266±0.0045 |

WIN (best-vs-best, each policy native answer model): continuous actor > CASPER-R +0.0062 FULL (~2sigma, marginal) /
+0.0100 TAIL (~2-3sigma, significant). Beats entropy +0.0224 tail. TAIL is Paper B's headline metric.
Earlier +0.015 was INFLATED by the corrupted cache under-measuring CASPER-R (0.351). TRUE margin = +0.006/+0.010.
LESSON: pool_entavg corruption affected baselines only; always verify baselines reproduce canonical (CASPER-R 0.360/0.152) before claiming.
