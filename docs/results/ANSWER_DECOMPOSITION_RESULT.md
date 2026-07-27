# Answer-model decomposition -- FINAL (diagnostic for understanding, not paper)

> Generated 2026-07-26 03:01:46. Stack: frozen t2i25_EP4 tower + signed C-lite (cfold_signed_best.pt). 10000 COLD_SEED users. Cold intercept **0.1279/0.0192** (full/tail NDCG@10). Controls: q0 snap True, shuffle collapses True, leak_users 0. Inputs: {'main': True, 'imputers': True, 'granularity': True}.

## Full table -- full@10 / tail@10 / full@100 per arm x budget

Arms: G geometric(circular/uncitable), B behavioral-SEL(honest), O oracle-B(SEL over full history), K oracle-concept-selection, U utility-oracle(true NDCG ceiling), S SEL+, C content-projection, E ExpoMF, P TagMF.


**CONCEPT-ASK**

| arm | q0 | q1 | q2 | q4 | q8 |
|---|---|---|---|---|---|
| G | 0.1279/0.0192/0.1864 | 0.1322/0.0349/0.2003 | 0.1340/0.0354/0.2020 | 0.1444/0.0409/0.2112 | 0.1450/0.0413/0.2121 |
| B | 0.1279/0.0192/0.1864 | 0.1323/0.0348/0.2007 | 0.1328/0.0343/0.2005 | 0.1433/0.0395/0.2093 | 0.1441/0.0395/0.2097 |
| O | 0.1279/0.0192/0.1864 | 0.1326/0.0350/0.2012 | 0.1330/0.0345/0.2008 | 0.1441/0.0408/0.2105 | 0.1448/0.0404/0.2108 |
| K | 0.1279/0.0192/0.1864 | 0.1603/0.0661/0.2375 | 0.1828/0.0873/0.2636 | 0.1995/0.1085/0.2848 | 0.2132/0.1231/0.3013 |
| U | 0.1279/0.0192/0.1864 | 0.1789/0.0366/0.2251 | 0.2002/0.0410/0.2405 | 0.2422/0.0496/0.2704 | 0.2822/0.0563/0.2981 |
| S | 0.1279/0.0192/0.1864 | 0.1323/0.0348/0.2008 | 0.1326/0.0349/0.2005 | 0.1433/0.0390/0.2088 | 0.1450/0.0407/0.2109 |
| C | 0.1279/0.0192/0.1864 | 0.1331/0.0352/0.2025 | 0.1330/0.0338/0.2013 | 0.1405/0.0343/0.2061 | 0.1421/0.0381/0.2070 |
| E | 0.1279/0.0192/0.1864 | 0.1277/0.0265/0.1910 | 0.1271/0.0255/0.1896 | 0.1303/0.0234/0.1891 | 0.1293/0.0203/0.1865 |
| P | 0.1279/0.0192/0.1864 | 0.1248/0.0212/0.1834 | 0.1243/0.0212/0.1823 | 0.1258/0.0207/0.1825 | 0.1253/0.0205/0.1823 |

**ITEM-ASK**

| arm | q0 | q1 | q2 | q4 | q8 |
|---|---|---|---|---|---|
| G | 0.1279/0.0192/0.1864 | 0.1438/0.0344/0.2080 | 0.1495/0.0412/0.2174 | 0.1506/0.0430/0.2210 | 0.1723/0.0549/0.2408 |
| B | 0.1279/0.0192/0.1864 | 0.1307/0.0266/0.1905 | 0.1305/0.0315/0.1911 | 0.1319/0.0343/0.1936 | 0.1673/0.0504/0.2321 |
| O | 0.1279/0.0192/0.1864 | 0.1113/0.0285/0.1750 | 0.0976/0.0320/0.1651 | 0.0761/0.0344/0.1505 | 0.0992/0.0520/0.1812 |
| U | 0.1279/0.0192/0.1864 | 0.1629/0.0306/0.2113 | 0.1786/0.0359/0.2227 | 0.1960/0.0392/0.2337 | 0.2222/0.0480/0.2580 |
| S | 0.1279/0.0192/0.1864 | 0.1307/0.0266/0.1905 | 0.1305/0.0315/0.1911 | 0.1319/0.0343/0.1936 | 0.1673/0.0504/0.2321 |
| C | 0.1279/0.0192/0.1864 | 0.1307/0.0266/0.1905 | 0.1305/0.0315/0.1911 | 0.1319/0.0343/0.1936 | 0.1673/0.0504/0.2321 |
| E | 0.1279/0.0192/0.1864 | 0.1307/0.0266/0.1905 | 0.1305/0.0315/0.1911 | 0.1319/0.0343/0.1936 | 0.1673/0.0504/0.2321 |
| P | 0.1279/0.0192/0.1864 | 0.1307/0.0266/0.1905 | 0.1305/0.0315/0.1911 | 0.1319/0.0343/0.1936 | 0.1673/0.0504/0.2321 |

**mean_answered (folded per budget)**

| arm | channel | q0 | q1 | q2 | q4 | q8 |
|---|---|---|---|---|---|---|
| G | concept | 0.00 | 0.99 | 1.99 | 3.98 | 7.93 |
| G | item | 0.00 | 0.43 | 0.84 | 1.65 | 3.03 |
| B | concept | 0.00 | 0.99 | 1.99 | 3.98 | 7.93 |
| B | item | 0.00 | 0.43 | 0.84 | 1.65 | 3.03 |
| O | concept | 0.00 | 1.00 | 2.00 | 4.00 | 8.00 |
| O | item | 0.00 | 0.51 | 1.00 | 1.95 | 3.57 |
| K | concept | 0.00 | 1.00 | 2.00 | 4.00 | 8.00 |
| U | concept | 0.00 | 0.73 | 1.35 | 2.57 | 5.06 |
| U | item | 0.00 | 0.34 | 0.63 | 1.15 | 1.99 |
| S | concept | 0.00 | 1.00 | 2.00 | 4.00 | 8.00 |
| S | item | 0.00 | 0.43 | 0.84 | 1.65 | 3.03 |
| C | concept | 0.00 | 1.00 | 2.00 | 4.00 | 8.00 |
| C | item | 0.00 | 0.43 | 0.84 | 1.65 | 3.03 |
| E | concept | 0.00 | 1.00 | 2.00 | 4.00 | 8.00 |
| E | item | 0.00 | 0.43 | 0.84 | 1.65 | 3.03 |
| P | concept | 0.00 | 1.00 | 2.00 | 4.00 | 8.00 |
| P | item | 0.00 | 0.43 | 0.84 | 1.65 | 3.03 |

## Guards (per arm)

| arm | CKA(answer-geom, u*) | Spearman(value, oracle-B) |
|---|---|---|
| G | 0.406 | None |
| B | 0.351 | 0.921 |
| O | 0.358 | 1.000 |
| K | 0.528 | 0.921 |
| U | 0.197 | None |
| S | 0.366 | 0.852 |
| C | 0.365 | 0.751 |
| E | 0.098 | -0.070 |
| P | 0.108 | 0.158 |

(G is the HIGH-CKA circular reference; an honest imputer must not exceed SEL(B)'s CKA while agreeing more with oracle-B.)


## C-lite vs C-full (concept-ask B, polarization, identical answers) -- full@10

| model | q0 | q1 | q2 | q4 | q8 |
|---|---|---|---|---|---|
| sclite | 0.1279 | 0.1323 | 0.1328 | 0.1433 | 0.1441 |
| scfull | 0.1279 | 0.1380 | 0.1356 | 0.1461 | 0.1378 |

## VERDICT -- the four causes + is-SEL-beatable

- **answerability**: Concept questions are answered far more often than item questions (7.9 vs 3.0 folded by q8): popularity-ordered item questions are rarely rated by cold users, so the item belief barely moves until a few answerable items accumulate. Answerability favors concepts.
- **selection**: Oracle per-user concept selection lifts full@10 by +0.0691 (CI [0.06560697628846371, 0.0725852143075269]) over the generic polarization bank at q8 -> a real SELECTION gap: the realizable bank asks sub-optimal concepts.
- **answer_value**: The TRUE answer-channel ceiling (utility oracle) sits +0.1381 above behavioral SEL and +0.1374 above the SEL-oracle at q8 (full@10). The SEL FORMULA is NOT the NDCG ceiling -- a better per-question ANSWER exists.
- **fold_health**: Oracle-selection concept-ask reaches 0.2132 full@10 at q8 vs the ledger's sclite per-answer m8 ~0.163: the fold is HEALTHY. Concepts do move the belief above the 0.1279 intercept.
- **granularity**: Oracle selection asks median-1320.0-member concepts vs the banks' median-4038.5. Per-answer cold lift by granularity (fine/med/broad): -0.005314250177490056/-0.0004918534247220754/0.0019496298986724518; Spearman(lift, member_count)=0.492998836493896. Fine ~= broad (granularity not the lever).
- **is_SEL_beatable**: No cheap model-free imputer (SEL+, content, ExpoMF, TagMF) beats the SEL-oracle on concept NDCG -- SEL is a strong formula ceiling among realizable imputers; but the utility oracle shows the ANSWER channel still has headroom the SEL family does not reach.
