
## STAGE D1 — concept-vs-item on the fixed fold  (2026-07-27 00:54:41)

Best distilled fold: `cd_s1_l10_best.pt`.

Matched-budget realizable concept-ask vs pop item-ask (full@10 | tail@10 | full@100):

| k | concept f@10 | item f@10 | concept t@10 | item t@10 | concept f@100 | item f@100 |
|---|---|---|---|---|---|---|
| 1 | 0.1591 | 0.1374 | 0.0638 | 0.0392 | 0.2345 | 0.2018 |
| 2 | 0.1803 | 0.1493 | 0.0861 | 0.0478 | 0.2595 | 0.2171 |
| 4 | 0.1972 | 0.1839 | 0.1050 | 0.0648 | 0.2813 | 0.2568 |
| 8 | 0.2107 | 0.2274 | 0.1210 | 0.0839 | 0.2984 | 0.3013 |

Concept beats item (full@10) at k = 1; (tail@10) at k = 1.
D1a adaptive_concept_arms rc=0, D1b concepts_only_curve rc=0 (committed-harness @10 crossover + selection arms in their JSONs).
