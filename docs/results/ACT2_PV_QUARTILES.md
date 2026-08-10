# Act 2: does the adaptive-minus-static gap track PERSONALISATION VALUE, on the same users?

Run 2026-08-10 19:08:11 by `scripts/act2_pv_quartiles.py` (sha256 `0c3ca8632958519d`).
Design sheet: `docs/design/DS_ACT2_DISPERSION_STRATIFICATION.md` sections 1 and 2.
Criterion machinery imported verbatim from `scripts/reddit/dispersion_leg1.py`.

**Contract of every number below.** Consumer = certified tower `t2i26c_ep31.pt` (arch i26)
with concept channel `concept_cv3_ep08.pt`; **NO-FLOOR**; policy-B / fixed arm-N candidate
pool; **all 10,000 canonical TEST users** (9,281 scored on tail — users
whose held-out set is all head have no tail target and are dropped by the metric, identically for
every arm); **behavioural answers** (the user's own half-star level; asked-but-unrated folds as the
i26 UNSEEN sentinel, the `levels_unseen` contract the consumer is actually fed); no sampling anywhere.
Every selector and BOTH comparators were produced by ONE `adaptive_harness.py` process under one
contract signature (`experiments\baselines\interview\adaptive_table_act2pv.json`), so every difference below is differenced **in-run**.
Paired percentile bootstrap over per-user differences, **10,000 resamples, seed 98765**;
never the unpaired `1.96*sqrt(2)*SE` form.

**Scope of the criterion scoring.** 10,214 states carry a scored criterion. That is
100% of the states the PV arm occupies — every
`fixedorder:entropy0` state at k ∈ 1,2,4,8 — which is everything sections 1–4 need. The remaining
cells in section 6 were enrichment (the criterion evaluated at the states the *adaptive* selectors
themselves reach) and were deliberately not run; their interview and state counts are exact
regardless, because those need no criterion evaluation.

**Per-user alignment.** Every per-user NDCG vector below comes from the single
`experiments\baselines\interview\adaptive_table_act2pv.json` run, so all arms share one user order by construction. Verified rather than
assumed: each stored vector's mean reproduces its aggregate table entry to <1e-9 across all 40
arm x budget x axis combinations, and the `full`/`tail` row counts (10,000 /
9,281) match the metric's own two keep-rules.

## 1. The quantity

`PV(u) = U(q*_u | z_u) − U(q*_pop | z_u)` — what a perfect personaliser would gain over a
one-size-fits-all order, for that user, at that moment. `U(q|z)` is the leg-1 loss-coupled criterion
(expected NDCG@10 of the post-answer slate under the tower's own predictive, marginalised over the
behavioural answer model); the bank is the leg-1 **unified** bank (297
items + 1031 concepts competing in the same expected-NDCG units);
answer-model constant L = 64 (median TEST profile). `q*_pop` is the population's
MODAL argmax at that budget. PV ≥ 0 by construction — asserted per user, not assumed.

**The state PV is read at.** A user's state after k questions depends on which questions were asked,
so PV would be selector-dependent if read off each selector's own interview, and the strata would then
differ per selector. PV is therefore read off the reference static comparator's states
(`fixedorder:entropy0`, k questions asked of everybody, answered behaviourally).

| k | modal q* (population) | modal share | distinct q* | distinct states | distinct PV values | PV=0 share | mean PV |
|---|---|---|---|---|---|---|---|
| 1 | concept 'vengeance' | 0.896 | 4 | 11 | 6 | 0.896 | 0.00050 |
| 2 | concept 'awesome soundtrack' | 0.426 | 12 | 112 | 90 | 0.426 | 0.01100 |
| 4 | concept 'violent' | 0.353 | 31 | 1,463 | 1,171 | 0.353 | 0.00765 |
| 8 | concept 'violent' | 0.252 | 78 | 5,767 | 5,183 | 0.252 | 0.00850 |

## 2. Quartiles

Rank-cut into 4 equal strata of 2,500 users. PV is a function of
the STATE, and the standard opener leaves the population in few distinct states, so PV takes few
distinct values — the tie-splitting is deterministic (by user id) and the realised PV range of each
quartile is printed so the reader can see how much of a cut is real.

### k = 1
| quartile | PV min | PV mean | PV max | own-question gain | pop-question gain | value forfeited by the pop question | mean profile len | mean answer rate |
|---|---|---|---|---|---|---|---|---|
| Q1 | 0.00000 | 0.00000 | 0.00000 | 0.02728 | 0.02728 | 0.0% | 123.6 | 0.371 |
| Q2 | 0.00000 | 0.00000 | 0.00000 | 0.02735 | 0.02735 | 0.0% | 121.6 | 0.378 |
| Q3 | 0.00000 | 0.00000 | 0.00000 | 0.02732 | 0.02732 | 0.0% | 129.5 | 0.374 |
| Q4 | 0.00000 | 0.00198 | 0.00756 | 0.02837 | 0.02639 | 7.0% | 190.1 | 0.624 |

### k = 2
| quartile | PV min | PV mean | PV max | own-question gain | pop-question gain | value forfeited by the pop question | mean profile len | mean answer rate |
|---|---|---|---|---|---|---|---|---|
| Q1 | 0.00000 | 0.00000 | 0.00000 | 0.02310 | 0.02310 | 0.0% | 75.3 | 0.049 |
| Q2 | 0.00000 | 0.00058 | 0.00382 | 0.02491 | 0.02433 | 2.3% | 121.9 | 0.227 |
| Q3 | 0.00382 | 0.01218 | 0.02250 | 0.03148 | 0.01930 | 38.7% | 179.5 | 0.631 |
| Q4 | 0.02250 | 0.03124 | 0.05215 | 0.03881 | 0.00757 | 80.5% | 188.1 | 0.803 |

### k = 4
| quartile | PV min | PV mean | PV max | own-question gain | pop-question gain | value forfeited by the pop question | mean profile len | mean answer rate |
|---|---|---|---|---|---|---|---|---|
| Q1 | 0.00000 | 0.00000 | 0.00000 | 0.02513 | 0.02513 | 0.0% | 116.2 | 0.222 |
| Q2 | 0.00000 | 0.00109 | 0.00364 | 0.02820 | 0.02710 | 3.9% | 179.5 | 0.457 |
| Q3 | 0.00364 | 0.00762 | 0.01200 | 0.03185 | 0.02423 | 23.9% | 156.7 | 0.479 |
| Q4 | 0.01200 | 0.02187 | 0.04121 | 0.03815 | 0.01628 | 57.3% | 112.3 | 0.483 |

### k = 8
| quartile | PV min | PV mean | PV max | own-question gain | pop-question gain | value forfeited by the pop question | mean profile len | mean answer rate |
|---|---|---|---|---|---|---|---|---|
| Q1 | 0.00000 | 0.00000 | 0.00000 | 0.02705 | 0.02705 | 0.0% | 86.0 | 0.164 |
| Q2 | 0.00000 | 0.00246 | 0.00458 | 0.02913 | 0.02666 | 8.5% | 176.8 | 0.449 |
| Q3 | 0.00459 | 0.00717 | 0.01066 | 0.03037 | 0.02320 | 23.6% | 171.3 | 0.444 |
| Q4 | 0.01067 | 0.02436 | 0.07729 | 0.03977 | 0.01542 | 61.2% | 130.7 | 0.492 |

## 3. Adaptive-minus-static gap, by quartile

**Bold** = the 95% paired CI excludes zero. Negative = the adaptive selector LOSES to the static order.

### k = 1 — full NDCG@10
| comparison | Q1 (low PV) | Q2 | Q3 | Q4 (high PV) | all 10k | TREND Q4−Q1 [95% CI] |
|---|---|---|---|---|---|---|
| glm_ucb − entropy0 | +0.0004 [-0.0025, +0.0034] | -0.0014 [-0.0045, +0.0018] | -0.0003 [-0.0033, +0.0027] | **-0.0102 [-0.0143, -0.0060]** | **-0.0029 [-0.0046, -0.0012]** | **-0.0107 [-0.0157, -0.0056]** |
| glm_ucb − greedy_items | **-0.0136 [-0.0173, -0.0098]** | **-0.0171 [-0.0212, -0.0131]** | **-0.0123 [-0.0162, -0.0084]** | **-0.0186 [-0.0231, -0.0140]** | **-0.0154 [-0.0174, -0.0133]** | -0.0050 [-0.0109, +0.0008] |
| thompson − entropy0 | +0.0011 [-0.0021, +0.0045] | -0.0003 [-0.0037, +0.0030] | +0.0014 [-0.0017, +0.0045] | **-0.0062 [-0.0105, -0.0019]** | -0.0010 [-0.0028, +0.0007] | **-0.0073 [-0.0127, -0.0020]** |
| thompson − greedy_items | **-0.0130 [-0.0168, -0.0090]** | **-0.0160 [-0.0202, -0.0119]** | **-0.0105 [-0.0144, -0.0066]** | **-0.0146 [-0.0190, -0.0101]** | **-0.0135 [-0.0156, -0.0115]** | -0.0016 [-0.0076, +0.0044] |
| golbandi_tree − entropy0 | -0.0040 [-0.0080, +0.0001] | **-0.0058 [-0.0099, -0.0016]** | -0.0023 [-0.0062, +0.0017] | **-0.0192 [-0.0242, -0.0142]** | **-0.0078 [-0.0099, -0.0056]** | **-0.0152 [-0.0218, -0.0090]** |
| golbandi_tree − greedy_items | **-0.0180 [-0.0225, -0.0136]** | **-0.0215 [-0.0263, -0.0168]** | **-0.0142 [-0.0189, -0.0093]** | **-0.0276 [-0.0327, -0.0225]** | **-0.0203 [-0.0227, -0.0179]** | **-0.0095 [-0.0164, -0.0027]** |

### k = 1 — tail NDCG@10
| comparison | Q1 (low PV) | Q2 | Q3 | Q4 (high PV) | all 10k | TREND Q4−Q1 [95% CI] |
|---|---|---|---|---|---|---|
| glm_ucb − entropy0 | +0.0014 [-0.0022, +0.0050] | **-0.0050 [-0.0091, -0.0008]** | +0.0018 [-0.0021, +0.0057] | **-0.0114 [-0.0161, -0.0067]** | **-0.0033 [-0.0054, -0.0013]** | **-0.0127 [-0.0187, -0.0070]** |
| glm_ucb − greedy_items | -0.0020 [-0.0062, +0.0022] | -0.0028 [-0.0069, +0.0014] | +0.0006 [-0.0035, +0.0047] | **-0.0095 [-0.0140, -0.0051]** | **-0.0035 [-0.0056, -0.0014]** | **-0.0075 [-0.0137, -0.0014]** |
| thompson − entropy0 | -0.0026 [-0.0064, +0.0012] | **-0.0065 [-0.0106, -0.0024]** | -0.0015 [-0.0054, +0.0022] | **-0.0106 [-0.0152, -0.0061]** | **-0.0053 [-0.0074, -0.0033]** | **-0.0080 [-0.0139, -0.0021]** |
| thompson − greedy_items | **-0.0060 [-0.0094, -0.0026]** | **-0.0043 [-0.0081, -0.0004]** | -0.0028 [-0.0063, +0.0007] | **-0.0088 [-0.0129, -0.0048]** | **-0.0055 [-0.0073, -0.0036]** | -0.0028 [-0.0081, +0.0025] |
| golbandi_tree − entropy0 | -0.0036 [-0.0083, +0.0012] | **-0.0113 [-0.0167, -0.0061]** | **-0.0050 [-0.0097, -0.0002]** | **-0.0126 [-0.0177, -0.0076]** | **-0.0081 [-0.0106, -0.0057]** | **-0.0090 [-0.0159, -0.0023]** |
| golbandi_tree − greedy_items | **-0.0069 [-0.0112, -0.0028]** | **-0.0091 [-0.0132, -0.0049]** | **-0.0063 [-0.0105, -0.0020]** | **-0.0108 [-0.0155, -0.0063]** | **-0.0083 [-0.0104, -0.0061]** | -0.0038 [-0.0102, +0.0022] |

### k = 2 — full NDCG@10
| comparison | Q1 (low PV) | Q2 | Q3 | Q4 (high PV) | all 10k | TREND Q4−Q1 [95% CI] |
|---|---|---|---|---|---|---|
| glm_ucb − entropy0 | **-0.0040 [-0.0067, -0.0013]** | **-0.0142 [-0.0182, -0.0102]** | **-0.0105 [-0.0146, -0.0063]** | -0.0000 [-0.0048, +0.0046] | **-0.0072 [-0.0092, -0.0052]** | +0.0040 [-0.0015, +0.0094] |
| glm_ucb − greedy_items | **-0.0147 [-0.0186, -0.0109]** | **-0.0242 [-0.0291, -0.0195]** | **-0.0290 [-0.0343, -0.0237]** | **-0.0412 [-0.0471, -0.0353]** | **-0.0273 [-0.0298, -0.0247]** | **-0.0265 [-0.0336, -0.0197]** |
| thompson − entropy0 | -0.0015 [-0.0044, +0.0014] | **-0.0051 [-0.0090, -0.0014]** | **-0.0055 [-0.0100, -0.0009]** | +0.0016 [-0.0043, +0.0076] | **-0.0026 [-0.0048, -0.0004]** | +0.0031 [-0.0034, +0.0097] |
| thompson − greedy_items | **-0.0122 [-0.0162, -0.0083]** | **-0.0152 [-0.0198, -0.0105]** | **-0.0240 [-0.0293, -0.0187]** | **-0.0396 [-0.0460, -0.0331]** | **-0.0227 [-0.0253, -0.0201]** | **-0.0274 [-0.0349, -0.0200]** |
| golbandi_tree − entropy0 | +0.0011 [-0.0024, +0.0046] | -0.0031 [-0.0072, +0.0009] | **-0.0219 [-0.0269, -0.0169]** | **-0.0128 [-0.0181, -0.0074]** | **-0.0092 [-0.0115, -0.0069]** | **-0.0139 [-0.0203, -0.0075]** |
| golbandi_tree − greedy_items | **-0.0096 [-0.0142, -0.0052]** | **-0.0132 [-0.0185, -0.0079]** | **-0.0404 [-0.0464, -0.0346]** | **-0.0540 [-0.0606, -0.0471]** | **-0.0293 [-0.0321, -0.0264]** | **-0.0444 [-0.0524, -0.0362]** |

### k = 2 — tail NDCG@10
| comparison | Q1 (low PV) | Q2 | Q3 | Q4 (high PV) | all 10k | TREND Q4−Q1 [95% CI] |
|---|---|---|---|---|---|---|
| glm_ucb − entropy0 | -0.0019 [-0.0047, +0.0010] | **-0.0110 [-0.0151, -0.0070]** | **-0.0085 [-0.0131, -0.0039]** | **-0.0090 [-0.0142, -0.0036]** | **-0.0076 [-0.0097, -0.0054]** | **-0.0071 [-0.0132, -0.0010]** |
| glm_ucb − greedy_items | **-0.0040 [-0.0073, -0.0006]** | **-0.0136 [-0.0176, -0.0099]** | **-0.0215 [-0.0266, -0.0165]** | **-0.0120 [-0.0181, -0.0059]** | **-0.0128 [-0.0152, -0.0105]** | **-0.0080 [-0.0149, -0.0011]** |
| thompson − entropy0 | **-0.0041 [-0.0076, -0.0006]** | **-0.0099 [-0.0146, -0.0054]** | -0.0045 [-0.0093, +0.0003] | **-0.0147 [-0.0213, -0.0083]** | **-0.0083 [-0.0108, -0.0058]** | **-0.0105 [-0.0181, -0.0032]** |
| thompson − greedy_items | **-0.0062 [-0.0093, -0.0032]** | **-0.0126 [-0.0168, -0.0086]** | **-0.0175 [-0.0230, -0.0120]** | **-0.0177 [-0.0239, -0.0116]** | **-0.0136 [-0.0160, -0.0111]** | **-0.0115 [-0.0183, -0.0048]** |
| golbandi_tree − entropy0 | +0.0011 [-0.0032, +0.0055] | -0.0026 [-0.0074, +0.0020] | **-0.0084 [-0.0134, -0.0036]** | **-0.0099 [-0.0157, -0.0042]** | **-0.0050 [-0.0075, -0.0025]** | **-0.0109 [-0.0181, -0.0038]** |
| golbandi_tree − greedy_items | -0.0010 [-0.0052, +0.0032] | **-0.0053 [-0.0103, -0.0004]** | **-0.0214 [-0.0271, -0.0156]** | **-0.0129 [-0.0198, -0.0059]** | **-0.0102 [-0.0130, -0.0075]** | **-0.0119 [-0.0201, -0.0037]** |

### k = 4 — full NDCG@10
| comparison | Q1 (low PV) | Q2 | Q3 | Q4 (high PV) | all 10k | TREND Q4−Q1 [95% CI] |
|---|---|---|---|---|---|---|
| glm_ucb − entropy0 | **-0.0056 [-0.0092, -0.0021]** | **-0.0099 [-0.0138, -0.0059]** | **-0.0144 [-0.0187, -0.0102]** | -0.0021 [-0.0049, +0.0008] | **-0.0080 [-0.0098, -0.0062]** | +0.0036 [-0.0011, +0.0082] |
| glm_ucb − greedy_items | **-0.0227 [-0.0274, -0.0178]** | **-0.0279 [-0.0332, -0.0226]** | **-0.0359 [-0.0412, -0.0306]** | **-0.0397 [-0.0458, -0.0338]** | **-0.0315 [-0.0342, -0.0288]** | **-0.0171 [-0.0248, -0.0094]** |
| thompson − entropy0 | **+0.0048 [+0.0007, +0.0089]** | -0.0038 [-0.0089, +0.0012] | **-0.0118 [-0.0171, -0.0066]** | **-0.0120 [-0.0182, -0.0059]** | **-0.0057 [-0.0083, -0.0031]** | **-0.0168 [-0.0242, -0.0094]** |
| thompson − greedy_items | **-0.0122 [-0.0170, -0.0076]** | **-0.0218 [-0.0275, -0.0162]** | **-0.0333 [-0.0390, -0.0277]** | **-0.0497 [-0.0564, -0.0429]** | **-0.0293 [-0.0322, -0.0263]** | **-0.0375 [-0.0459, -0.0291]** |
| golbandi_tree − entropy0 | **+0.0172 [+0.0120, +0.0225]** | +0.0029 [-0.0023, +0.0082] | -0.0004 [-0.0055, +0.0047] | **-0.0068 [-0.0126, -0.0011]** | **+0.0032 [+0.0006, +0.0059]** | **-0.0240 [-0.0320, -0.0163]** |
| golbandi_tree − greedy_items | +0.0002 [-0.0055, +0.0061] | **-0.0151 [-0.0210, -0.0092]** | **-0.0219 [-0.0277, -0.0162]** | **-0.0444 [-0.0513, -0.0378]** | **-0.0203 [-0.0234, -0.0172]** | **-0.0446 [-0.0536, -0.0359]** |

### k = 4 — tail NDCG@10
| comparison | Q1 (low PV) | Q2 | Q3 | Q4 (high PV) | all 10k | TREND Q4−Q1 [95% CI] |
|---|---|---|---|---|---|---|
| glm_ucb − entropy0 | -0.0017 [-0.0045, +0.0011] | **-0.0051 [-0.0084, -0.0018]** | **-0.0122 [-0.0159, -0.0084]** | -0.0029 [-0.0060, +0.0002] | **-0.0055 [-0.0072, -0.0039]** | -0.0012 [-0.0054, +0.0030] |
| glm_ucb − greedy_items | **-0.0095 [-0.0138, -0.0053]** | **-0.0082 [-0.0130, -0.0035]** | **-0.0209 [-0.0256, -0.0162]** | **-0.0105 [-0.0169, -0.0039]** | **-0.0123 [-0.0148, -0.0097]** | -0.0010 [-0.0088, +0.0067] |
| thompson − entropy0 | -0.0018 [-0.0065, +0.0029] | **-0.0066 [-0.0115, -0.0017]** | **-0.0084 [-0.0131, -0.0036]** | **-0.0153 [-0.0228, -0.0079]** | **-0.0080 [-0.0108, -0.0052]** | **-0.0136 [-0.0223, -0.0047]** |
| thompson − greedy_items | **-0.0096 [-0.0145, -0.0047]** | **-0.0097 [-0.0146, -0.0048]** | **-0.0171 [-0.0219, -0.0124]** | **-0.0229 [-0.0300, -0.0158]** | **-0.0147 [-0.0174, -0.0120]** | **-0.0133 [-0.0221, -0.0044]** |
| golbandi_tree − entropy0 | **+0.0089 [+0.0039, +0.0139]** | +0.0006 [-0.0042, +0.0055] | +0.0000 [-0.0048, +0.0049] | -0.0039 [-0.0099, +0.0021] | +0.0015 [-0.0011, +0.0040] | **-0.0128 [-0.0205, -0.0051]** |
| golbandi_tree − greedy_items | +0.0011 [-0.0042, +0.0063] | -0.0024 [-0.0076, +0.0029] | **-0.0087 [-0.0135, -0.0038]** | **-0.0114 [-0.0185, -0.0043]** | **-0.0053 [-0.0081, -0.0025]** | **-0.0125 [-0.0214, -0.0035]** |

### k = 8 — full NDCG@10
| comparison | Q1 (low PV) | Q2 | Q3 | Q4 (high PV) | all 10k | TREND Q4−Q1 [95% CI] |
|---|---|---|---|---|---|---|
| glm_ucb − entropy0 | **-0.0085 [-0.0118, -0.0052]** | **-0.0148 [-0.0192, -0.0105]** | **-0.0161 [-0.0205, -0.0117]** | **-0.0211 [-0.0263, -0.0159]** | **-0.0151 [-0.0173, -0.0129]** | **-0.0126 [-0.0188, -0.0065]** |
| glm_ucb − greedy_items | **-0.0193 [-0.0238, -0.0150]** | **-0.0366 [-0.0428, -0.0305]** | **-0.0405 [-0.0460, -0.0349]** | **-0.0418 [-0.0478, -0.0358]** | **-0.0345 [-0.0373, -0.0318]** | **-0.0225 [-0.0300, -0.0152]** |
| thompson − entropy0 | +0.0004 [-0.0044, +0.0051] | **-0.0077 [-0.0137, -0.0019]** | **-0.0155 [-0.0211, -0.0098]** | **-0.0285 [-0.0354, -0.0218]** | **-0.0128 [-0.0157, -0.0099]** | **-0.0289 [-0.0372, -0.0207]** |
| thompson − greedy_items | **-0.0104 [-0.0156, -0.0055]** | **-0.0294 [-0.0356, -0.0232]** | **-0.0398 [-0.0456, -0.0341]** | **-0.0492 [-0.0563, -0.0423]** | **-0.0322 [-0.0352, -0.0292]** | **-0.0388 [-0.0475, -0.0302]** |
| golbandi_tree − entropy0 | **+0.0243 [+0.0197, +0.0292]** | +0.0016 [-0.0041, +0.0071] | +0.0000 [-0.0052, +0.0052] | -0.0003 [-0.0061, +0.0057] | **+0.0064 [+0.0037, +0.0091]** | **-0.0246 [-0.0322, -0.0172]** |
| golbandi_tree − greedy_items | **+0.0136 [+0.0078, +0.0194]** | **-0.0202 [-0.0263, -0.0142]** | **-0.0243 [-0.0300, -0.0187]** | **-0.0210 [-0.0272, -0.0151]** | **-0.0130 [-0.0160, -0.0100]** | **-0.0346 [-0.0431, -0.0259]** |

### k = 8 — tail NDCG@10
| comparison | Q1 (low PV) | Q2 | Q3 | Q4 (high PV) | all 10k | TREND Q4−Q1 [95% CI] |
|---|---|---|---|---|---|---|
| glm_ucb − entropy0 | +0.0005 [-0.0025, +0.0035] | **-0.0115 [-0.0154, -0.0076]** | **-0.0088 [-0.0130, -0.0045]** | **-0.0272 [-0.0333, -0.0212]** | **-0.0116 [-0.0139, -0.0094]** | **-0.0277 [-0.0347, -0.0209]** |
| glm_ucb − greedy_items | -0.0046 [-0.0094, +0.0001] | **-0.0070 [-0.0122, -0.0019]** | **-0.0094 [-0.0147, -0.0042]** | **-0.0191 [-0.0258, -0.0122]** | **-0.0099 [-0.0126, -0.0071]** | **-0.0144 [-0.0229, -0.0060]** |
| thompson − entropy0 | -0.0014 [-0.0061, +0.0033] | **-0.0059 [-0.0111, -0.0007]** | **-0.0057 [-0.0111, -0.0004]** | **-0.0262 [-0.0337, -0.0188]** | **-0.0096 [-0.0125, -0.0067]** | **-0.0248 [-0.0337, -0.0158]** |
| thompson − greedy_items | **-0.0065 [-0.0112, -0.0017]** | -0.0014 [-0.0068, +0.0040] | **-0.0064 [-0.0117, -0.0012]** | **-0.0180 [-0.0253, -0.0108]** | **-0.0080 [-0.0108, -0.0050]** | **-0.0116 [-0.0204, -0.0028]** |
| golbandi_tree − entropy0 | **+0.0182 [+0.0132, +0.0230]** | -0.0005 [-0.0055, +0.0045] | **+0.0065 [+0.0011, +0.0121]** | -0.0037 [-0.0104, +0.0029] | **+0.0052 [+0.0024, +0.0079]** | **-0.0219 [-0.0301, -0.0137]** |
| golbandi_tree − greedy_items | **+0.0131 [+0.0078, +0.0184]** | +0.0039 [-0.0013, +0.0092] | **+0.0059 [+0.0002, +0.0116]** | +0.0044 [-0.0025, +0.0115] | **+0.0068 [+0.0039, +0.0098]** | -0.0087 [-0.0175, +0.0003] |

## 4. Controls

**Multiplicity.** The quartile tables above hold 96 cell CIs, so reading
individual cells is a forking-paths invitation. The object the design sheet asks about is the TREND —
one number per comparison — and it carries its own interval (Q1 and Q4 are disjoint user sets, so it
is a two-sample bootstrap: each stratum resampled independently, 10,000 resamples, seed
98765). The trend column is what should be read; the cells are shown so nothing is hidden.

**Shuffle control.** PV is permuted across users and the quartiles re-cut; any gap-vs-quartile trend
must vanish. Every trend, real vs shuffled:

| comparison | axis | k | TREND Q4−Q1 (real PV) | TREND Q4−Q1 (SHUFFLED PV) |
|---|---|---|---|---|
| glm_ucb − entropy0 | full | 1 | **-0.0107 [-0.0157, -0.0056]** | +0.0007 [-0.0042, +0.0056] |
| glm_ucb − entropy0 | full | 2 | +0.0040 [-0.0015, +0.0094] | -0.0042 [-0.0099, +0.0017] |
| glm_ucb − entropy0 | full | 4 | +0.0036 [-0.0011, +0.0082] | -0.0020 [-0.0073, +0.0032] |
| glm_ucb − entropy0 | full | 8 | **-0.0126 [-0.0188, -0.0065]** | -0.0017 [-0.0079, +0.0045] |
| glm_ucb − entropy0 | tail | 1 | **-0.0127 [-0.0187, -0.0070]** | +0.0006 [-0.0052, +0.0065] |
| glm_ucb − entropy0 | tail | 2 | **-0.0071 [-0.0132, -0.0010]** | -0.0029 [-0.0087, +0.0030] |
| glm_ucb − entropy0 | tail | 4 | -0.0012 [-0.0054, +0.0030] | -0.0036 [-0.0086, +0.0012] |
| glm_ucb − entropy0 | tail | 8 | **-0.0277 [-0.0347, -0.0209]** | -0.0006 [-0.0067, +0.0056] |
| glm_ucb − greedy_items | full | 1 | -0.0050 [-0.0109, +0.0008] | +0.0047 [-0.0012, +0.0107] |
| glm_ucb − greedy_items | full | 2 | **-0.0265 [-0.0336, -0.0197]** | -0.0008 [-0.0079, +0.0065] |
| glm_ucb − greedy_items | full | 4 | **-0.0171 [-0.0248, -0.0094]** | +0.0011 [-0.0065, +0.0088] |
| glm_ucb − greedy_items | full | 8 | **-0.0225 [-0.0300, -0.0152]** | +0.0060 [-0.0017, +0.0142] |
| glm_ucb − greedy_items | tail | 1 | **-0.0075 [-0.0137, -0.0014]** | -0.0008 [-0.0069, +0.0051] |
| glm_ucb − greedy_items | tail | 2 | **-0.0080 [-0.0149, -0.0011]** | -0.0013 [-0.0079, +0.0054] |
| glm_ucb − greedy_items | tail | 4 | -0.0010 [-0.0088, +0.0067] | -0.0046 [-0.0122, +0.0026] |
| glm_ucb − greedy_items | tail | 8 | **-0.0144 [-0.0229, -0.0060]** | +0.0022 [-0.0058, +0.0099] |
| thompson − entropy0 | full | 1 | **-0.0073 [-0.0127, -0.0020]** | -0.0006 [-0.0057, +0.0043] |
| thompson − entropy0 | full | 2 | +0.0031 [-0.0034, +0.0097] | -0.0015 [-0.0079, +0.0050] |
| thompson − entropy0 | full | 4 | **-0.0168 [-0.0242, -0.0094]** | -0.0041 [-0.0115, +0.0034] |
| thompson − entropy0 | full | 8 | **-0.0289 [-0.0372, -0.0207]** | -0.0050 [-0.0134, +0.0037] |
| thompson − entropy0 | tail | 1 | **-0.0080 [-0.0139, -0.0021]** | -0.0003 [-0.0059, +0.0053] |
| thompson − entropy0 | tail | 2 | **-0.0105 [-0.0181, -0.0032]** | -0.0022 [-0.0092, +0.0047] |
| thompson − entropy0 | tail | 4 | **-0.0136 [-0.0223, -0.0047]** | -0.0014 [-0.0095, +0.0068] |
| thompson − entropy0 | tail | 8 | **-0.0248 [-0.0337, -0.0158]** | +0.0021 [-0.0061, +0.0102] |
| thompson − greedy_items | full | 1 | -0.0016 [-0.0076, +0.0044] | +0.0034 [-0.0026, +0.0094] |
| thompson − greedy_items | full | 2 | **-0.0274 [-0.0349, -0.0200]** | +0.0020 [-0.0055, +0.0095] |
| thompson − greedy_items | full | 4 | **-0.0375 [-0.0459, -0.0291]** | -0.0009 [-0.0091, +0.0073] |
| thompson − greedy_items | full | 8 | **-0.0388 [-0.0475, -0.0302]** | +0.0028 [-0.0058, +0.0115] |
| thompson − greedy_items | tail | 1 | -0.0028 [-0.0081, +0.0025] | -0.0017 [-0.0071, +0.0036] |
| thompson − greedy_items | tail | 2 | **-0.0115 [-0.0183, -0.0048]** | -0.0007 [-0.0076, +0.0062] |
| thompson − greedy_items | tail | 4 | **-0.0133 [-0.0221, -0.0044]** | -0.0024 [-0.0103, +0.0054] |
| thompson − greedy_items | tail | 8 | **-0.0116 [-0.0204, -0.0028]** | +0.0049 [-0.0035, +0.0130] |
| golbandi_tree − entropy0 | full | 1 | **-0.0152 [-0.0218, -0.0090]** | +0.0019 [-0.0043, +0.0081] |
| golbandi_tree − entropy0 | full | 2 | **-0.0139 [-0.0203, -0.0075]** | -0.0031 [-0.0096, +0.0034] |
| golbandi_tree − entropy0 | full | 4 | **-0.0240 [-0.0320, -0.0163]** | -0.0044 [-0.0117, +0.0029] |
| golbandi_tree − entropy0 | full | 8 | **-0.0246 [-0.0322, -0.0172]** | -0.0027 [-0.0104, +0.0053] |
| golbandi_tree − entropy0 | tail | 1 | **-0.0090 [-0.0159, -0.0023]** | +0.0030 [-0.0038, +0.0099] |
| golbandi_tree − entropy0 | tail | 2 | **-0.0109 [-0.0181, -0.0038]** | -0.0054 [-0.0126, +0.0016] |
| golbandi_tree − entropy0 | tail | 4 | **-0.0128 [-0.0205, -0.0051]** | +0.0032 [-0.0041, +0.0106] |
| golbandi_tree − entropy0 | tail | 8 | **-0.0219 [-0.0301, -0.0137]** | +0.0001 [-0.0076, +0.0079] |
| golbandi_tree − greedy_items | full | 1 | **-0.0095 [-0.0164, -0.0027]** | +0.0059 [-0.0008, +0.0129] |
| golbandi_tree − greedy_items | full | 2 | **-0.0444 [-0.0524, -0.0362]** | +0.0003 [-0.0078, +0.0088] |
| golbandi_tree − greedy_items | full | 4 | **-0.0446 [-0.0536, -0.0359]** | -0.0012 [-0.0099, +0.0071] |
| golbandi_tree − greedy_items | full | 8 | **-0.0346 [-0.0431, -0.0259]** | +0.0051 [-0.0032, +0.0137] |
| golbandi_tree − greedy_items | tail | 1 | -0.0038 [-0.0102, +0.0022] | +0.0016 [-0.0043, +0.0074] |
| golbandi_tree − greedy_items | tail | 2 | **-0.0119 [-0.0201, -0.0037]** | -0.0038 [-0.0116, +0.0042] |
| golbandi_tree − greedy_items | tail | 4 | **-0.0125 [-0.0214, -0.0035]** | +0.0023 [-0.0059, +0.0101] |
| golbandi_tree − greedy_items | tail | 8 | -0.0087 [-0.0175, +0.0003] | +0.0028 [-0.0053, +0.0109] |

Full quartile tables under the shuffled cut:

### k = 4 — full NDCG@10, SHUFFLED PV
| comparison | Q1 (low PV) | Q2 | Q3 | Q4 (high PV) | all 10k | TREND Q4−Q1 [95% CI] |
|---|---|---|---|---|---|---|
| glm_ucb − entropy0 | **-0.0063 [-0.0100, -0.0025]** | **-0.0071 [-0.0108, -0.0035]** | **-0.0104 [-0.0141, -0.0068]** | **-0.0083 [-0.0118, -0.0046]** | **-0.0080 [-0.0098, -0.0062]** | -0.0020 [-0.0073, +0.0032] |
| glm_ucb − greedy_items | **-0.0303 [-0.0357, -0.0250]** | **-0.0299 [-0.0353, -0.0244]** | **-0.0368 [-0.0424, -0.0314]** | **-0.0292 [-0.0345, -0.0238]** | **-0.0315 [-0.0342, -0.0288]** | +0.0011 [-0.0065, +0.0088] |
| thompson − entropy0 | -0.0035 [-0.0087, +0.0018] | **-0.0061 [-0.0112, -0.0010]** | **-0.0057 [-0.0111, -0.0004]** | **-0.0075 [-0.0128, -0.0024]** | **-0.0057 [-0.0083, -0.0031]** | -0.0041 [-0.0115, +0.0034] |
| thompson − greedy_items | **-0.0275 [-0.0332, -0.0218]** | **-0.0289 [-0.0346, -0.0231]** | **-0.0322 [-0.0381, -0.0264]** | **-0.0285 [-0.0343, -0.0227]** | **-0.0293 [-0.0322, -0.0263]** | -0.0009 [-0.0091, +0.0073] |
| golbandi_tree − entropy0 | **+0.0066 [+0.0014, +0.0120]** | +0.0019 [-0.0036, +0.0073] | +0.0024 [-0.0031, +0.0077] | +0.0022 [-0.0030, +0.0074] | **+0.0032 [+0.0006, +0.0059]** | -0.0044 [-0.0117, +0.0029] |
| golbandi_tree − greedy_items | **-0.0175 [-0.0235, -0.0113]** | **-0.0209 [-0.0270, -0.0147]** | **-0.0241 [-0.0303, -0.0181]** | **-0.0187 [-0.0246, -0.0128]** | **-0.0203 [-0.0234, -0.0172]** | -0.0012 [-0.0099, +0.0071] |

### k = 8 — full NDCG@10, SHUFFLED PV
| comparison | Q1 (low PV) | Q2 | Q3 | Q4 (high PV) | all 10k | TREND Q4−Q1 [95% CI] |
|---|---|---|---|---|---|---|
| glm_ucb − entropy0 | **-0.0133 [-0.0177, -0.0091]** | **-0.0124 [-0.0168, -0.0082]** | **-0.0198 [-0.0241, -0.0155]** | **-0.0150 [-0.0194, -0.0106]** | **-0.0151 [-0.0173, -0.0129]** | -0.0017 [-0.0079, +0.0045] |
| glm_ucb − greedy_items | **-0.0372 [-0.0430, -0.0318]** | **-0.0311 [-0.0365, -0.0258]** | **-0.0385 [-0.0442, -0.0329]** | **-0.0312 [-0.0369, -0.0256]** | **-0.0345 [-0.0373, -0.0318]** | +0.0060 [-0.0017, +0.0142] |
| thompson − entropy0 | **-0.0080 [-0.0143, -0.0020]** | **-0.0134 [-0.0191, -0.0077]** | **-0.0168 [-0.0227, -0.0110]** | **-0.0130 [-0.0186, -0.0071]** | **-0.0128 [-0.0157, -0.0099]** | -0.0050 [-0.0134, +0.0037] |
| thompson − greedy_items | **-0.0319 [-0.0385, -0.0257]** | **-0.0322 [-0.0382, -0.0264]** | **-0.0356 [-0.0415, -0.0296]** | **-0.0291 [-0.0349, -0.0231]** | **-0.0322 [-0.0352, -0.0292]** | +0.0028 [-0.0058, +0.0115] |
| golbandi_tree − entropy0 | **+0.0083 [+0.0028, +0.0138]** | **+0.0064 [+0.0012, +0.0116]** | +0.0053 [-0.0001, +0.0107] | +0.0056 [-0.0001, +0.0112] | **+0.0064 [+0.0037, +0.0091]** | -0.0027 [-0.0104, +0.0053] |
| golbandi_tree − greedy_items | **-0.0157 [-0.0218, -0.0097]** | **-0.0123 [-0.0182, -0.0065]** | **-0.0134 [-0.0191, -0.0077]** | **-0.0106 [-0.0167, -0.0044]** | **-0.0130 [-0.0160, -0.0100]** | +0.0051 [-0.0032, +0.0137] |

**Self-difference control.** An arm differenced against itself inside every quartile:
max |delta| + |CI| = 0.0e+00
(exactly zero: True).

**Confound check.** Spearman correlation of PV with profile length and with answer rate:

| k | rho(PV, profile length) | rho(PV, answer rate) |
|---|---|---|
| 1 | +0.116 | +0.222 |
| 2 | +0.297 | +0.758 |
| 4 | +0.100 | +0.350 |
| 8 | +0.141 | +0.408 |

## 5. How many distinguishable situations exist at all

A free by-product of building the state list, and worth reporting on its own. Under the reference
static interview (`fixedorder:entropy0`, everybody asked the same k questions), the 10,000
users land in this many **distinct states** — not "how many questions are optimal", but how many
distinguishable situations a selector could possibly react to:

| k | distinct STATES the population occupies | distinct loss-optimal q* | modal q* share |
|---|---|---|---|
| 1 | 11 | 4 | 0.896 |
| 2 | 112 | 12 | 0.426 |
| 4 | 1,463 | 31 | 0.353 |
| 8 | 5,767 | 78 | 0.252 |

This is the dispersion collapse measured a second, independent way, and it bounds a DETERMINISTIC
selector exactly: a rule that conditions only on the answers to a fixed order has at most
112 distinguishable inputs at k=2 and so
at most 112 distinguishable behaviours.
That is what `fixedorder:entropy0` (1 interview),
`glm_ucb` (3 interviews) and
`golbandi_tree` (3 interviews) sit far under.

Thompson is NOT bound by it, and that is the point: it conducts
8,006 distinct interviews at k=2 and
10,000 at k=4 — more than the whole population has
distinguishable states — because it escapes the bound by **randomising**, not by learning. A per-user
posterior draw manufactures branching that no amount of answer information could support, which is why
its branching count must never be read as evidence of personalisation. The honest statement is that
Thompson's divergence is a property of its sampling rule; the deterministic selectors' divergence is
bounded above by the state count, and both lose to one order asked of everybody.

## 6. The branching ratio (design sheet section 2)

Of the distinct interviews a selector actually conducts, how many distinct **loss-optimal** questions
do those states correspond to? The interview and state counts need no criterion evaluation and are
reported for every cell; the q* columns need scored states, and criterion scoring was deliberately
run for the PV arm only (see the note under the contract).

| selector | k | distinct interviews | distinct states | states per interview | mean answers | distinct loss-optimal q* | q* per interview | top-q* share | modal q* |
|---|---|---|---|---|---|---|---|---|---|
| fixedorder:entropy0 | 1 | 1 | 11 | 11.000 | 0.437 | 4 | 4.0000 | 0.896 | concept 'vengeance' |
| fixedorder:entropy0 | 2 | 1 | 112 | 112.000 | 0.855 | 12 | 12.0000 | 0.426 | concept 'awesome soundtrack' |
| fixedorder:entropy0 | 4 | 1 | 1,463 | 1463.000 | 1.641 | 31 | 31.0000 | 0.353 | concept 'violent' |
| fixedorder:entropy0 | 8 | 1 | 5,767 | 5767.000 | 3.100 | 78 | 78.0000 | 0.252 | concept 'violent' |
| golbandi_tree | 1 | 1 | 11 | 11.000 | 0.283 | 5 | 5.0000 | 0.835 | concept 'vengeance' |
| golbandi_tree | 2 | 3 | 92 | 30.667 | 0.629 | 11 | 3.6667 | 0.487 | concept 'excellent script' |
| golbandi_tree | 4 | 25 | 1,054 | 42.160 | 1.284 | *not scored* | — | — | — |
| golbandi_tree | 8 | 345 | 5,371 | 15.568 | 2.748 | *not scored* | — | — | — |
| glm_ucb | 1 | 1 | 11 | 11.000 | 0.420 | 4 | 4.0000 | 0.908 | concept 'vengeance' |
| glm_ucb | 2 | 3 | 96 | 32.000 | 0.815 | 11 | 3.6667 | 0.729 | concept 'vengeance' |
| glm_ucb | 4 | 22 | 1,391 | 63.227 | 1.657 | *not scored* | — | — | — |
| glm_ucb | 8 | 314 | 5,452 | 17.363 | 2.971 | *not scored* | — | — | — |
| fixedorder:greedy_items | 1 | 1 | 11 | 11.000 | 0.298 | 5 | 5.0000 | 0.702 | concept 'vengeance' |
| fixedorder:greedy_items | 2 | 1 | 92 | 92.000 | 0.557 | 9 | 9.0000 | 0.549 | concept 'vengeance' |
| fixedorder:greedy_items | 4 | 1 | 1,160 | 1160.000 | 1.294 | *not scored* | — | — | — |
| fixedorder:greedy_items | 8 | 1 | 3,997 | 3997.000 | 2.435 | *not scored* | — | — | — |
| thompson | 1 | 760 | 1,468 | 1.932 | 0.215 | 25 | 0.0329 | 0.852 | concept 'vengeance' |
| thompson | 2 | 8,006 | 8,727 | 1.090 | 0.431 | *not scored* | — | — | — |
| thompson | 4 | 10,000 | 9,999 | 1.000 | 0.859 | *not scored* | — | — | — |
| thompson | 8 | 10,000 | 10,000 | 1.000 | 1.721 | *not scored* | — | — | — |

## Verdict

**The gap tracks PV — and it runs the wrong way.** Of the 48 trend cells (3 selectors × 2
comparators × 2 axes × 4 budgets), **38 separate from zero and every one of them is negative**; 10 do
not separate; **none is positive**. Under the shuffle control **0 of 48** separate. So the
relationship is real, it is driven by `PV` and not by the cut, and its sign is the opposite of the
reading the design sheet listed first: the adaptive selectors do not fail because there is nothing to
adapt to — they fail **hardest exactly where personalising is worth the most**. At k=8, Thompson
gives up nothing to entropy0 in the PV=0 quartile (+0.0004 full, ns) and −0.0285 in the top quartile;
GLM-UCB goes −0.0085 → −0.0211; against the stronger greedy-items comparator every selector's deficit
deepens from Q1 to Q4 by a factor of 1.8–4.8 (`glm_ucb` k8 −0.0193 → −0.0418; `thompson` k8 −0.0104 →
−0.0492), and `golbandi_tree` changes sign outright (k8 +0.0136 → −0.0210). This is the "**methods cannot find it**" branch, and it is a stronger
result than a flat gap would have been, because the exploitable structure is now measured on the same
users rather than argued from a second population.

**The one selector that crosses over only crosses over where PV is zero.** `golbandi_tree` is the
single arm that beats the static order overall on this consumer (+0.0064 full / +0.0052 tail at k=8).
The whole of that advantage sits in Q1: +0.0243 full at k=8 in the PV=0 quartile, decaying
monotonically to −0.0003 in Q4 (k=4: +0.0172 → −0.0068). Its win is therefore not evidence that a
tree exploits personalisation value; it is evidence that a tree is a better *static* opener for the
users for whom no personalisation is available.

**What the criterion surface itself says.** Even at k=8, after 5,767 distinct states over 10,000
users, only **78 distinct questions are ever loss-optimal**, the single modal question is optimal for
25% of users, and a quarter of the cohort has `PV` exactly 0 — personalising is worth literally
nothing to them. The branching ratio makes the Act-1 observation quantitative: `entropy0` conducts
**one** interview and lands on 78 distinct loss-optimal questions, whereas Thompson at k=1 spreads
**760 distinct interviews across just 25 distinct loss-optimal questions** — a ratio of 0.033 optimal
questions per interview, i.e. thirty interviews manufactured for every genuinely distinguishable
decision. §5 states the bound precisely and its limit: it constrains the deterministic selectors
exactly, and Thompson escapes it only by randomising, which is why its branching count is not
evidence of personalisation. The q* column for the adaptive selectors' *own* states at k≥4 was
enrichment and was not part of this run; the interview and state counts there need no criterion
evaluation and are exact.

**The top quartile is high-stakes, not noise.** In Q4 the population's question forfeits **80.5% of
the best question's value at k=2, 57.3% at k=4 and 61.2% at k=8** (own-question gain 0.0398 vs
pop-question gain 0.0154 at k=8) — while in Q1 it forfeits exactly 0.0% at every budget, because the
best question *is* the population's. So the strata are not a slice through measurement noise: they
separate users for whom one order is optimal from users for whom it destroys most of the available
value. The adaptive selectors are furthest behind in the second group.

A further observation from the criterion surface, reported because it is a candidate mechanism rather
than a controlled test: **the loss-optimal question is a *concept*, not an item, for ~100% of users at
every budget** (100.0% at k=1/2/4, 99.0% at k=8; the modal q* is `vengeance`, `awesome soundtrack`,
`violent`, `violent`). The three adaptive selectors in this battery all shop in the item bank. That is
a concrete hypothesis for *why* they cannot convert PV, and it is testable directly.

**The confound is partly real and must be stated.** `PV` correlates with the answer rate — ρ = +0.76
at k=2, +0.35/+0.41 at k=4/8 — and only weakly with profile length (ρ = +0.10 to +0.30). At **k=2 the
stratification is substantially a split on "did you answer the two entropy0 questions"** (Q1 answer
rate 0.05 vs Q4 0.80) and the trend there should not be read as pure PV. At **k=4 and k=8 it is not**:
the answer rate is essentially flat across Q2–Q4 (0.457/0.479/0.483 at k=4) and mean profile length is
**non-monotone and lowest in Q4** (112 in Q4 vs 180 in Q2 at k=4; 131 vs 177 at k=8), so profile size
is demonstrably not doing the work at the budgets that carry the result. That answerability and
personalisation value are entangled at short budgets is itself the chapter's own mechanism
(§3(a) of the design sheet), not a nuisance.

**What Act 2 can now claim.** That the standard openers leave little divergence to exploit — measured,
on ML users, with a loss-coupled criterion. That the residual divergence they do leave is **not
captured**: the methods lose more, not less, as it grows, and no method in this battery converts
personalisation value into NDCG on either axis at any budget. What Act 2 may **not** claim is that
adaptivity is worthless because there is nothing to adapt to; the top quartile at k=8 carries a mean
`PV` of +0.024 NDCG@10 in the criterion's own units and every adaptive arm is furthest behind exactly
there. Act 4 should be redirected at *why* — the answerability-prediction selector of design-sheet
§3(c) is now the obvious next test, since the one variable that most strongly tracks `PV` is the
answer rate.

**Limitations carried forward.** (1) The criterion scores against the tower's own predictive, so `PV`
is self-consistent expected gain, not gain against held-out items; a user whose model-optimal question
differs may still gain nothing measurable. (2) `PV` is read at the `fixedorder:entropy0` states, which
is what makes the strata selector-independent, but it means a selector that would have driven a user
to a higher-PV state is not credited for it. (3) The quartile cells carry 48 CIs per budget (192 in all);
the trend column is the object, the cells are shown only so nothing is hidden. (4) `glm_ucb`
and `thompson` ride an iALS basis constructed without a `random_state`, so their four-decimal values
are not bit-reproducible across processes (`ACT1_PAIRED_CI.md`); every comparison here is within one
process, so the pairing is valid conditional on this run's basis.
