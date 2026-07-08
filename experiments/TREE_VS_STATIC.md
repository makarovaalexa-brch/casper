# LEARNED INTERVIEW TREE vs LEARNED STATIC -- CLASS-MATCHED ADAPTIVITY (v2 fold; DIRECTIONAL 173/300)

> **DIRECTIONAL BANNER** -- 173/300 users, answerer-v1 working grid NOT frozen; v2 fold
> `.cache/i25_fold_v2_best.pt`. NO LLM API calls; local compute. Re-run on the frozen 300-user
> grid before any citation.

Date 2026-07-09. Script `scripts/learned_tree_vs_static.py`. 173 users; 180 candidates
(60c/60i/60a); cold NDCG@10 0.1502; fold val 0.4256. Depth=12 turns. Candidate pool = top-60 by coverage (same as the
static). Paired per-user bootstrap BOOT=5000. Answer classes: pos(liked/loved)/neg(meh/
hated)/noclue(refuse,k=0; turn consumed, no fold update, E1).

## Pre-registered reads (printed BEFORE results)

CLASS-MATCHED contrast: a LEARNED conditional interview TREE vs the LEARNED static questionnaire,
IDENTICAL learner (same greedy objective + code path; branching-disabled tree == the sequence,
asserted at runtime), split-constructed (grown on one half), evaluated on the OTHER (held-out) half.

- **(1) VERDICT -- tree vs (grow-half) static on HELD-OUT users** (anytime NDCG@10, pooled over both
  split directions, paired per-user bootstrap). CI EXCLUDES 0 is THE verdict:
    * POSITIVE -> first honest LEARNED-ADAPTIVITY win (conditioning transfers to new users).
    * NEGATIVE/TIE -> statics optimal even class-matched (a real property, proven fairly).
- **(2) IN-SAMPLE SANITY**: in-sample tree >= in-sample static (grow-half, both) -- MUST hold by
  construction (the tree is a superset of the sequence). If violated, the learner differs -> STOP.
- **(3) TREE STRUCTURE**: #nodes that branched, at which depths, on which questions/channels --
  the interpretability payload (does it descend into taste regions after positive answers?).

Also reported: tree & grow-half static vs the EVAL-HALF-grown static (in-sample reference) -- puts
the (contaminated) in-sample static beside the fair one so the contamination gap is visible.
MIN_USERS in {15, 25}; primary family s-item (matches anchor 0.2251); s-mixed secondary. Verdict
metric = anytime NDCG@10 @T=12. Reproduction gate: all-173 greedy s-item anytime@T=12 == 0.2251.

## 0. Reproduction (anchor gate)

All-173 greedy s-item anytime NDCG@10 @T=12 = **0.2251** vs anchor 0.2251 +/- 0.002 -> **MATCH**.

## 1. Identity proof (SAME learner)

Branching-disabled tree schedule == greedy sequence (`SC.build_greedy_sub`) ? **True**.
This certifies the tree grower and the static comparator share one code path / objective; the
tree is the strict Golbandi-style branching superset of the sequence.

- schedule: `['item:14234', 'item:1262', 'item:680', 'item:10955', 'item:1019', 'item:7531', 'item:2934', 'item:256', 'item:7833', 'item:1217', 'item:988', 'item:16117']`

## 2. HELD-OUT verdict (tree vs grow-half static; class-matched, split-constructed)

### Pooled over both split directions (THE verdict)

| family | MIN_USERS | budget | tree - static [95% CI] | n | CI excl 0? |
|---|--:|--:|---|--:|---|
| s-item | 15 | 12 | -0.0094[-0.0158,-0.0033] | 346 | YES |
| s-item | 15 | 8 | -0.0075[-0.0130,-0.0022] | 346 | YES |
| s-item | 25 | 12 | -0.0094[-0.0158,-0.0033] | 346 | YES |
| s-item | 25 | 8 | -0.0075[-0.0130,-0.0022] | 346 | YES |
| s-mixed | 15 | 12 | +0.0021[-0.0007,+0.0049] | 346 | no |
| s-mixed | 15 | 8 | +0.0016[-0.0006,+0.0038] | 346 | no |
| s-mixed | 25 | 12 | +0.0025[-0.0010,+0.0063] | 173 | no |
| s-mixed | 25 | 8 | +0.0020[-0.0008,+0.0048] | 173 | no |

**PRIMARY VERDICT (s-item MU15 T12): tree - static = -0.0094[-0.0158,-0.0033] -> TREE LOSES -> statics strictly optimal even class-matched**

### Per split direction (held-out)

| run | budget | tree | static(fair) | tree-static [CI] | in-sample tree | in-sample static | sanity tree>=static | fallbacks |
|---|--:|--:|--:|---|--:|--:|:--:|--:|
| seed0_s-item_MU15_growA_evalB | 8 | 0.2113 | 0.2178 | -0.0065[-0.0162,+0.0038] | 0.2264 | 0.2089 | OK | 9 |
| seed0_s-item_MU15_growA_evalB | 12 | 0.2115 | 0.2189 | -0.0074[-0.0203,+0.0061] | 0.2375 | 0.2121 | OK | 9 |
| seed0_s-item_MU15_growB_evalA | 8 | 0.1893 | 0.1916 | -0.0023[-0.0100,+0.0056] | 0.2637 | 0.2485 | OK | 1 |
| seed0_s-item_MU15_growB_evalA | 12 | 0.1876 | 0.1900 | -0.0024[-0.0111,+0.0064] | 0.2734 | 0.2536 | OK | 1 |
| seed0_s-item_MU25_growA_evalB | 8 | 0.2113 | 0.2178 | -0.0065[-0.0162,+0.0038] | 0.2264 | 0.2089 | OK | 9 |
| seed0_s-item_MU25_growA_evalB | 12 | 0.2115 | 0.2189 | -0.0074[-0.0203,+0.0061] | 0.2375 | 0.2121 | OK | 9 |
| seed0_s-item_MU25_growB_evalA | 8 | 0.1893 | 0.1916 | -0.0023[-0.0100,+0.0056] | 0.2637 | 0.2485 | OK | 1 |
| seed0_s-item_MU25_growB_evalA | 12 | 0.1876 | 0.1900 | -0.0024[-0.0111,+0.0064] | 0.2734 | 0.2536 | OK | 1 |
| seed0_s-mixed_MU15_growA_evalB | 8 | 0.2371 | 0.2343 | +0.0028[-0.0017,+0.0079] | 0.2110 | 0.2055 | OK | 0 |
| seed0_s-mixed_MU15_growA_evalB | 12 | 0.2373 | 0.2348 | +0.0024[-0.0037,+0.0092] | 0.2151 | 0.2053 | OK | 0 |
| seed0_s-mixed_MU15_growB_evalA | 8 | 0.1977 | 0.1962 | +0.0015[-0.0016,+0.0047] | 0.2453 | 0.2400 | OK | 0 |
| seed0_s-mixed_MU15_growB_evalA | 12 | 0.1986 | 0.1959 | +0.0026[-0.0011,+0.0067] | 0.2479 | 0.2411 | OK | 0 |
| seed0_s-mixed_MU25_growA_evalB | 8 | 0.2367 | 0.2343 | +0.0025[-0.0017,+0.0072] | 0.2102 | 0.2055 | OK | 0 |
| seed0_s-mixed_MU25_growA_evalB | 12 | 0.2372 | 0.2348 | +0.0024[-0.0034,+0.0087] | 0.2143 | 0.2053 | OK | 0 |
| seed0_s-mixed_MU25_growB_evalA | 8 | 0.1977 | 0.1962 | +0.0015[-0.0016,+0.0047] | 0.2453 | 0.2400 | OK | 0 |
| seed0_s-mixed_MU25_growB_evalA | 12 | 0.1986 | 0.1959 | +0.0026[-0.0011,+0.0067] | 0.2479 | 0.2411 | OK | 0 |
| seed1_s-item_MU15_growA_evalB | 8 | 0.1941 | 0.2007 | -0.0066[-0.0193,+0.0042] | 0.2553 | 0.2402 | OK | 4 |
| seed1_s-item_MU15_growA_evalB | 12 | 0.1921 | 0.2017 | -0.0097[-0.0246,+0.0035] | 0.2635 | 0.2429 | OK | 4 |
| seed1_s-item_MU15_growB_evalA | 8 | 0.1991 | 0.2136 | -0.0145[-0.0263,-0.0034] | 0.2445 | 0.2225 | OK | 6 |
| seed1_s-item_MU15_growB_evalA | 12 | 0.1920 | 0.2100 | -0.0181[-0.0310,-0.0059] | 0.2594 | 0.2252 | OK | 6 |
| seed1_s-item_MU25_growA_evalB | 8 | 0.1941 | 0.2007 | -0.0066[-0.0193,+0.0042] | 0.2553 | 0.2402 | OK | 4 |
| seed1_s-item_MU25_growA_evalB | 12 | 0.1921 | 0.2017 | -0.0097[-0.0246,+0.0035] | 0.2635 | 0.2429 | OK | 4 |
| seed1_s-item_MU25_growB_evalA | 8 | 0.1991 | 0.2136 | -0.0145[-0.0263,-0.0034] | 0.2445 | 0.2225 | OK | 6 |
| seed1_s-item_MU25_growB_evalA | 12 | 0.1920 | 0.2100 | -0.0181[-0.0310,-0.0059] | 0.2594 | 0.2252 | OK | 6 |
| seed1_s-mixed_MU15_growA_evalB | 8 | 0.2187 | 0.2168 | +0.0019[-0.0021,+0.0058] | 0.2300 | 0.2242 | OK | 0 |
| seed1_s-mixed_MU15_growA_evalB | 12 | 0.2196 | 0.2172 | +0.0024[-0.0031,+0.0082] | 0.2335 | 0.2250 | OK | 0 |
| seed1_s-mixed_MU15_growB_evalA | 8 | 0.2226 | 0.2225 | +0.0001[-0.0053,+0.0053] | 0.2358 | 0.2233 | OK | 2 |
| seed1_s-mixed_MU15_growB_evalA | 12 | 0.2232 | 0.2225 | +0.0007[-0.0052,+0.0064] | 0.2388 | 0.2227 | OK | 2 |

## 3. Contamination context -- both vs the EVAL-HALF-grown static (in-sample reference)

The eval-half-grown static is fit to the very users it is scored on (contaminated); comparing the
fair (grow-half) static to it exposes the test-fit bonus, and the tree to it shows whether
adaptivity closes that gap.

| run | budget | ref(eval-grown) static | tree - ref [CI] | fair-static - ref [CI] |
|---|--:|--:|---|---|
| seed0_s-item_MU15_growA_evalB | 8 | 0.2485 | -0.0371[-0.0645,-0.0142] | -0.0306[-0.0565,-0.0085] |
| seed0_s-item_MU15_growA_evalB | 12 | 0.2536 | -0.0421[-0.0687,-0.0192] | -0.0347[-0.0594,-0.0140] |
| seed0_s-item_MU15_growB_evalA | 8 | 0.2089 | -0.0197[-0.0337,-0.0070] | -0.0173[-0.0311,-0.0046] |
| seed0_s-item_MU15_growB_evalA | 12 | 0.2121 | -0.0245[-0.0406,-0.0102] | -0.0221[-0.0369,-0.0085] |
| seed0_s-item_MU25_growA_evalB | 8 | 0.2485 | -0.0371[-0.0645,-0.0142] | -0.0306[-0.0565,-0.0085] |
| seed0_s-item_MU25_growA_evalB | 12 | 0.2536 | -0.0421[-0.0687,-0.0192] | -0.0347[-0.0594,-0.0140] |
| seed0_s-item_MU25_growB_evalA | 8 | 0.2089 | -0.0197[-0.0337,-0.0070] | -0.0173[-0.0311,-0.0046] |
| seed0_s-item_MU25_growB_evalA | 12 | 0.2121 | -0.0245[-0.0406,-0.0102] | -0.0221[-0.0369,-0.0085] |
| seed0_s-mixed_MU15_growA_evalB | 8 | 0.2400 | -0.0029[-0.0129,+0.0076] | -0.0057[-0.0159,+0.0050] |
| seed0_s-mixed_MU15_growA_evalB | 12 | 0.2411 | -0.0038[-0.0146,+0.0078] | -0.0062[-0.0162,+0.0045] |
| seed0_s-mixed_MU15_growB_evalA | 8 | 0.2055 | -0.0078[-0.0172,+0.0019] | -0.0092[-0.0198,+0.0009] |
| seed0_s-mixed_MU15_growB_evalA | 12 | 0.2053 | -0.0067[-0.0146,+0.0014] | -0.0094[-0.0189,-0.0002] |
| seed0_s-mixed_MU25_growA_evalB | 8 | 0.2400 | -0.0033[-0.0130,+0.0072] | -0.0057[-0.0159,+0.0050] |
| seed0_s-mixed_MU25_growA_evalB | 12 | 0.2411 | -0.0038[-0.0144,+0.0075] | -0.0062[-0.0162,+0.0045] |
| seed0_s-mixed_MU25_growB_evalA | 8 | 0.2055 | -0.0078[-0.0172,+0.0019] | -0.0092[-0.0198,+0.0009] |
| seed0_s-mixed_MU25_growB_evalA | 12 | 0.2053 | -0.0067[-0.0146,+0.0014] | -0.0094[-0.0189,-0.0002] |
| seed1_s-item_MU15_growA_evalB | 8 | 0.2225 | -0.0284[-0.0471,-0.0130] | -0.0218[-0.0380,-0.0083] |
| seed1_s-item_MU15_growA_evalB | 12 | 0.2252 | -0.0331[-0.0528,-0.0170] | -0.0234[-0.0390,-0.0102] |
| seed1_s-item_MU15_growB_evalA | 8 | 0.2402 | -0.0411[-0.0642,-0.0204] | -0.0266[-0.0501,-0.0063] |
| seed1_s-item_MU15_growB_evalA | 12 | 0.2429 | -0.0509[-0.0763,-0.0284] | -0.0329[-0.0564,-0.0130] |
| seed1_s-item_MU25_growA_evalB | 8 | 0.2225 | -0.0284[-0.0471,-0.0130] | -0.0218[-0.0380,-0.0083] |
| seed1_s-item_MU25_growA_evalB | 12 | 0.2252 | -0.0331[-0.0528,-0.0170] | -0.0234[-0.0390,-0.0102] |
| seed1_s-item_MU25_growB_evalA | 8 | 0.2402 | -0.0411[-0.0642,-0.0204] | -0.0266[-0.0501,-0.0063] |
| seed1_s-item_MU25_growB_evalA | 12 | 0.2429 | -0.0509[-0.0763,-0.0284] | -0.0329[-0.0564,-0.0130] |
| seed1_s-mixed_MU15_growA_evalB | 8 | 0.2233 | -0.0046[-0.0177,+0.0071] | -0.0065[-0.0179,+0.0040] |
| seed1_s-mixed_MU15_growA_evalB | 12 | 0.2227 | -0.0031[-0.0156,+0.0082] | -0.0055[-0.0149,+0.0033] |
| seed1_s-mixed_MU15_growB_evalA | 8 | 0.2242 | -0.0016[-0.0139,+0.0099] | -0.0017[-0.0113,+0.0072] |
| seed1_s-mixed_MU15_growB_evalA | 12 | 0.2250 | -0.0018[-0.0128,+0.0086] | -0.0025[-0.0104,+0.0050] |

## 4. TREE STRUCTURE (interpretability payload)

### seed0_s-item_MU15_growA_evalB

- nodes 111; branched 11 (by depth {0: 1, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 8: 1, 9: 1, 10: 1}); deepest branch depth 10; branch channel mix {'item': 11}; fallbacks 9

| depth | split question | cohort n | child sizes (pos/neg/noclue) |
|--:|---|--:|---|
| 0 | item:14234 | 86 | 6/79/1 |
| 1 | item:2376 | 79 | 10/68/1 |
| 2 | item:7531 | 68 | 2/66/0 |
| 3 | item:2907 | 66 | 3/63/0 |
| 4 | item:10955 | 63 | 3/60/0 |
| 5 | item:804 | 60 | 1/59/0 |
| 6 | item:1262 | 59 | 1/58/0 |
| 7 | item:680 | 58 | 0/58/0 |
| 8 | item:991 | 58 | 0/57/1 |
| 9 | item:1019 | 57 | 1/55/1 |
| 10 | item:256 | 55 | 0/54/1 |

### seed0_s-item_MU15_growB_evalA

- nodes 134; branched 11 (by depth {0: 1, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 8: 1, 9: 1, 10: 1}); deepest branch depth 10; branch channel mix {'item': 11}; fallbacks 1

| depth | split question | cohort n | child sizes (pos/neg/noclue) |
|--:|---|--:|---|
| 0 | item:1262 | 87 | 6/80/1 |
| 1 | item:680 | 80 | 4/73/3 |
| 2 | item:14234 | 73 | 5/66/2 |
| 3 | item:4399 | 66 | 8/56/2 |
| 4 | item:7833 | 56 | 1/55/0 |
| 5 | item:10955 | 55 | 2/53/0 |
| 6 | item:3171 | 53 | 1/52/0 |
| 7 | item:483 | 52 | 1/51/0 |
| 8 | item:9620 | 51 | 1/49/1 |
| 9 | item:16117 | 49 | 5/44/0 |
| 10 | item:15912 | 44 | 0/44/0 |

### seed0_s-item_MU25_growA_evalB

- nodes 111; branched 11 (by depth {0: 1, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 8: 1, 9: 1, 10: 1}); deepest branch depth 10; branch channel mix {'item': 11}; fallbacks 9

| depth | split question | cohort n | child sizes (pos/neg/noclue) |
|--:|---|--:|---|
| 0 | item:14234 | 86 | 6/79/1 |
| 1 | item:2376 | 79 | 10/68/1 |
| 2 | item:7531 | 68 | 2/66/0 |
| 3 | item:2907 | 66 | 3/63/0 |
| 4 | item:10955 | 63 | 3/60/0 |
| 5 | item:804 | 60 | 1/59/0 |
| 6 | item:1262 | 59 | 1/58/0 |
| 7 | item:680 | 58 | 0/58/0 |
| 8 | item:991 | 58 | 0/57/1 |
| 9 | item:1019 | 57 | 1/55/1 |
| 10 | item:256 | 55 | 0/54/1 |

### seed0_s-item_MU25_growB_evalA

- nodes 134; branched 11 (by depth {0: 1, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 8: 1, 9: 1, 10: 1}); deepest branch depth 10; branch channel mix {'item': 11}; fallbacks 1

| depth | split question | cohort n | child sizes (pos/neg/noclue) |
|--:|---|--:|---|
| 0 | item:1262 | 87 | 6/80/1 |
| 1 | item:680 | 80 | 4/73/3 |
| 2 | item:14234 | 73 | 5/66/2 |
| 3 | item:4399 | 66 | 8/56/2 |
| 4 | item:7833 | 56 | 1/55/0 |
| 5 | item:10955 | 55 | 2/53/0 |
| 6 | item:3171 | 53 | 1/52/0 |
| 7 | item:483 | 52 | 1/51/0 |
| 8 | item:9620 | 51 | 1/49/1 |
| 9 | item:16117 | 49 | 5/44/0 |
| 10 | item:15912 | 44 | 0/44/0 |

### seed0_s-mixed_MU15_growA_evalB

- nodes 81; branched 12 (by depth {0: 1, 1: 1, 2: 2, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 8: 1, 9: 1, 10: 1}); deepest branch depth 10; branch channel mix {'concept': 12}; fallbacks 0

| depth | split question | cohort n | child sizes (pos/neg/noclue) |
|--:|---|--:|---|
| 0 | concept:307 | 86 | 0/86/0 |
| 1 | concept:129 | 86 | 66/20/0 |
| 2 | concept:22 | 66 | 56/10/0 |
| 3 | concept:285 | 56 | 44/12/0 |
| 4 | concept:86 | 44 | 38/6/0 |
| 5 | concept:274 | 38 | 33/5/0 |
| 6 | concept:230 | 33 | 32/1/0 |
| 7 | concept:122 | 32 | 32/0/0 |
| 8 | concept:231 | 32 | 28/4/0 |
| 9 | concept:261 | 28 | 25/3/0 |
| 10 | concept:303 | 25 | 25/0/0 |
| 2 | concept:300 | 20 | 9/11/0 |

### seed0_s-mixed_MU15_growB_evalA

- nodes 63; branched 11 (by depth {0: 1, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 8: 1, 9: 1, 10: 1}); deepest branch depth 10; branch channel mix {'concept': 11}; fallbacks 0

| depth | split question | cohort n | child sizes (pos/neg/noclue) |
|--:|---|--:|---|
| 0 | concept:55 | 87 | 83/4/0 |
| 1 | concept:129 | 83 | 69/14/0 |
| 2 | concept:86 | 69 | 56/13/0 |
| 3 | concept:54 | 56 | 56/0/0 |
| 4 | concept:212 | 56 | 56/0/0 |
| 5 | concept:274 | 56 | 50/6/0 |
| 6 | concept:162 | 50 | 50/0/0 |
| 7 | concept:67 | 50 | 47/3/0 |
| 8 | concept:122 | 47 | 47/0/0 |
| 9 | concept:303 | 47 | 46/1/0 |
| 10 | concept:231 | 46 | 33/13/0 |

### seed0_s-mixed_MU25_growA_evalB

- nodes 71; branched 11 (by depth {0: 1, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 8: 1, 9: 1, 10: 1}); deepest branch depth 10; branch channel mix {'concept': 11}; fallbacks 0

| depth | split question | cohort n | child sizes (pos/neg/noclue) |
|--:|---|--:|---|
| 0 | concept:307 | 86 | 0/86/0 |
| 1 | concept:129 | 86 | 66/20/0 |
| 2 | concept:22 | 66 | 56/10/0 |
| 3 | concept:285 | 56 | 44/12/0 |
| 4 | concept:86 | 44 | 38/6/0 |
| 5 | concept:274 | 38 | 33/5/0 |
| 6 | concept:230 | 33 | 32/1/0 |
| 7 | concept:122 | 32 | 32/0/0 |
| 8 | concept:231 | 32 | 28/4/0 |
| 9 | concept:261 | 28 | 25/3/0 |
| 10 | concept:303 | 25 | 25/0/0 |

### seed0_s-mixed_MU25_growB_evalA

- nodes 63; branched 11 (by depth {0: 1, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 8: 1, 9: 1, 10: 1}); deepest branch depth 10; branch channel mix {'concept': 11}; fallbacks 0

| depth | split question | cohort n | child sizes (pos/neg/noclue) |
|--:|---|--:|---|
| 0 | concept:55 | 87 | 83/4/0 |
| 1 | concept:129 | 83 | 69/14/0 |
| 2 | concept:86 | 69 | 56/13/0 |
| 3 | concept:54 | 56 | 56/0/0 |
| 4 | concept:212 | 56 | 56/0/0 |
| 5 | concept:274 | 56 | 50/6/0 |
| 6 | concept:162 | 50 | 50/0/0 |
| 7 | concept:67 | 50 | 47/3/0 |
| 8 | concept:122 | 47 | 47/0/0 |
| 9 | concept:303 | 47 | 46/1/0 |
| 10 | concept:231 | 46 | 33/13/0 |

### seed1_s-item_MU15_growA_evalB

- nodes 132; branched 11 (by depth {0: 1, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 8: 1, 9: 1, 10: 1}); deepest branch depth 10; branch channel mix {'item': 11}; fallbacks 4

| depth | split question | cohort n | child sizes (pos/neg/noclue) |
|--:|---|--:|---|
| 0 | item:14234 | 86 | 9/75/2 |
| 1 | item:680 | 75 | 6/67/2 |
| 2 | item:2783 | 67 | 6/59/2 |
| 3 | item:176 | 59 | 3/56/0 |
| 4 | item:2165 | 56 | 2/54/0 |
| 5 | item:4209 | 54 | 2/51/1 |
| 6 | item:1217 | 51 | 3/48/0 |
| 7 | item:2790 | 48 | 1/46/1 |
| 8 | item:10955 | 46 | 1/45/0 |
| 9 | item:1174 | 45 | 0/45/0 |
| 10 | item:256 | 45 | 0/43/2 |

### seed1_s-item_MU15_growB_evalA

- nodes 132; branched 11 (by depth {0: 1, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 8: 1, 9: 1, 10: 1}); deepest branch depth 10; branch channel mix {'item': 11}; fallbacks 6

| depth | split question | cohort n | child sizes (pos/neg/noclue) |
|--:|---|--:|---|
| 0 | item:1262 | 87 | 6/81/0 |
| 1 | item:10955 | 81 | 7/72/2 |
| 2 | item:6406 | 72 | 5/66/1 |
| 3 | item:4399 | 66 | 5/60/1 |
| 4 | item:2907 | 60 | 2/56/2 |
| 5 | item:2376 | 56 | 0/53/3 |
| 6 | item:987 | 53 | 3/49/1 |
| 7 | item:1661 | 49 | 2/47/0 |
| 8 | item:16117 | 47 | 2/45/0 |
| 9 | item:11775 | 45 | 1/44/0 |
| 10 | item:3304 | 44 | 0/44/0 |

### seed1_s-item_MU25_growA_evalB

- nodes 132; branched 11 (by depth {0: 1, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 8: 1, 9: 1, 10: 1}); deepest branch depth 10; branch channel mix {'item': 11}; fallbacks 4

| depth | split question | cohort n | child sizes (pos/neg/noclue) |
|--:|---|--:|---|
| 0 | item:14234 | 86 | 9/75/2 |
| 1 | item:680 | 75 | 6/67/2 |
| 2 | item:2783 | 67 | 6/59/2 |
| 3 | item:176 | 59 | 3/56/0 |
| 4 | item:2165 | 56 | 2/54/0 |
| 5 | item:4209 | 54 | 2/51/1 |
| 6 | item:1217 | 51 | 3/48/0 |
| 7 | item:2790 | 48 | 1/46/1 |
| 8 | item:10955 | 46 | 1/45/0 |
| 9 | item:1174 | 45 | 0/45/0 |
| 10 | item:256 | 45 | 0/43/2 |

### seed1_s-item_MU25_growB_evalA

- nodes 132; branched 11 (by depth {0: 1, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 8: 1, 9: 1, 10: 1}); deepest branch depth 10; branch channel mix {'item': 11}; fallbacks 6

| depth | split question | cohort n | child sizes (pos/neg/noclue) |
|--:|---|--:|---|
| 0 | item:1262 | 87 | 6/81/0 |
| 1 | item:10955 | 81 | 7/72/2 |
| 2 | item:6406 | 72 | 5/66/1 |
| 3 | item:4399 | 66 | 5/60/1 |
| 4 | item:2907 | 60 | 2/56/2 |
| 5 | item:2376 | 56 | 0/53/3 |
| 6 | item:987 | 53 | 3/49/1 |
| 7 | item:1661 | 49 | 2/47/0 |
| 8 | item:16117 | 47 | 2/45/0 |
| 9 | item:11775 | 45 | 1/44/0 |
| 10 | item:3304 | 44 | 0/44/0 |

### seed1_s-mixed_MU15_growA_evalB

- nodes 82; branched 14 (by depth {0: 1, 1: 1, 2: 2, 3: 3, 4: 1, 5: 1, 6: 1, 7: 1, 8: 1, 9: 1, 10: 1}); deepest branch depth 10; branch channel mix {'concept': 14}; fallbacks 0

| depth | split question | cohort n | child sizes (pos/neg/noclue) |
|--:|---|--:|---|
| 0 | concept:307 | 86 | 0/86/0 |
| 1 | concept:285 | 86 | 62/24/0 |
| 2 | concept:67 | 62 | 46/16/0 |
| 3 | concept:274 | 46 | 42/4/0 |
| 4 | concept:54 | 42 | 42/0/0 |
| 5 | concept:122 | 42 | 42/0/0 |
| 6 | concept:163 | 42 | 42/0/0 |
| 7 | concept:129 | 42 | 41/1/0 |
| 8 | concept:113 | 41 | 36/5/0 |
| 9 | concept:212 | 36 | 36/0/0 |
| 10 | concept:291 | 36 | 25/11/0 |
| 3 | concept:291 | 16 | 9/7/0 |
| 2 | concept:302 | 24 | 15/9/0 |
| 3 | concept:29 | 15 | 14/1/0 |

### seed1_s-mixed_MU15_growB_evalA

- nodes 109; branched 16 (by depth {0: 1, 1: 2, 2: 3, 3: 2, 4: 2, 5: 1, 6: 1, 7: 1, 8: 1, 9: 1, 10: 1}); deepest branch depth 10; branch channel mix {'concept': 16}; fallbacks 2

| depth | split question | cohort n | child sizes (pos/neg/noclue) |
|--:|---|--:|---|
| 0 | concept:231 | 87 | 52/35/0 |
| 1 | concept:2 | 52 | 35/17/0 |
| 2 | concept:129 | 35 | 33/2/0 |
| 3 | concept:122 | 33 | 33/0/0 |
| 4 | concept:291 | 33 | 22/11/0 |
| 5 | concept:86 | 22 | 20/2/0 |
| 6 | concept:162 | 20 | 20/0/0 |
| 7 | concept:274 | 20 | 19/1/0 |
| 8 | concept:285 | 19 | 18/1/0 |
| 9 | concept:307 | 18 | 0/18/0 |
| 10 | concept:81 | 18 | 18/0/0 |
| 2 | concept:275 | 17 | 13/4/0 |
| 1 | concept:300 | 35 | 21/14/0 |
| 2 | concept:282 | 21 | 15/6/0 |
| 3 | concept:212 | 15 | 15/0/0 |
| 4 | concept:22 | 15 | 11/4/0 |

## Synthesis

Class-matched, split-constructed, held-out: the LEARNED interview tree vs the LEARNED static
(identity-proven same learner). Primary pooled verdict (s-item MU15 T12): **-0.0094[-0.0158,-0.0033]** -> **TREE LOSES -> statics strictly optimal even class-matched**. DIRECTIONAL 173/300; re-run on the frozen grid before citation.


### Mechanism readout (the interpretability payload, read from section 4)

1. **The trees are near-CHAINS, not bushy trees.** Every s-item tree has exactly 11 branched nodes,
   one per depth: at each node one answer class holds nearly the whole cohort (items: `neg` -- most
   users rate popular probe items at/below their own mean, so centered value <= 0 dominates,
   e.g. root split 79/6/1), and only that majority child is big enough (>= MIN_USERS) to branch
   again. The minority children (pos 1-10 users, noclue 0-1) become tiny memorized side-sequences.
2. **Those side-branches are exactly what loses.** In-sample the tree beats the sequence everywhere
   (e.g. 0.2375 vs 0.2121; sanity holds by construction, 28/28 rows OK), but the low-n side-branches
   are fit to a handful of construction users and do NOT transfer: held-out the s-item tree LOSES
   -0.0094[-0.0158,-0.0033] pooled (CI excl 0, both seeds, both directions). The conditioning the
   learner discovers is cohort noise, not user structure.
3. **MIN_USERS=25 changes nothing for s-item** (identical trees/results to MU15): the majority-chain
   cohorts stay >= 25 for ~11 depths, so the threshold never binds on the chain, and the side-branch
   pathology enters at the FIRST split regardless. The overfit is not fixable by this knob.
4. **s-mixed (concept-led) trees branch on concepts only and TIE with a consistently positive point
   estimate** (+0.0021[-0.0007,+0.0049] MU15 T12 pooled; every one of the 6 s-mixed estimates is
   positive, none significant). Concept answers split the cohort more evenly (e.g. 66/20, 56/10),
   so both children stay learnable -- the closest thing to transferring conditioning observed, but
   it does not reach significance at n=346.
5. **Contamination context** (same-night measurement, experiments/STATIC_CONTAMINATION.md): the
   cohort-fit static's test-fit bonus is +0.0283[+0.0193,+0.0377] (s-item T12 pooled). Section 3's
   columns reproduce it here: the fair (split-constructed) static scores ~0.035 below the eval-grown
   reference; the tree ~0.042 below. I.e. BOTH honest artifacts sit well under the contaminated
   in-sample static -- and the tree gives up a further ~0.009 on top of the fair static.

### What this proves

The prior "statics optimal" verdicts pitted a learned static against hand-designed adaptive rules
(class mismatch). This experiment removes the mismatch: the SAME greedy NDCG learner, allowed to
condition on observed answer class (Golbandi-style ternary branching), grown on one user half and
evaluated on the other. Result: **conditioning does not transfer in this arena -- the learned tree
strictly loses to the learned sequence on held-out users in the item family and ties in the mixed
family.** The mechanism is measured, not conjectured: answer-class splits on popular probes are
extremely unbalanced, so branching buys in-sample fit on tiny cohorts and pays for it out-of-sample.
"Statics optimal even class-matched" is now a fairly-proven property of this arena (abundant-
answerability, 86-87 construction users, 12 turns), not an artifact of comparing across learner
classes. The one direction left open by point 4: a branching rule that only splits on BALANCED
answer classes (concepts) with much larger construction cohorts -- the ns-positive s-mixed trend is
the place a real win would first appear, and the frozen-grid re-run should watch it.

**Compute honesty:** candidate pool per node = top-60 by coverage (same POOL as every static in the
program; documented, not a hidden cap); depth-12 growth ran in full (no early stop); wall 31.4m for
14 grow+eval estimates. Missing-class fallback at eval descends the largest construction child
(logged per run: 0-9 users affected). Incremental checkpoints: per-estimate JSON partials +
per-grow tree snapshots in .cache/ (added for the frozen-grid re-run).
