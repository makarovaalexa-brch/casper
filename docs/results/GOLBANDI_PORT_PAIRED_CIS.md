# Paired CIs for the Golbandi tree PORT+CROSS cells (`gtport13_paired`)

Every interval below is a **paired percentile bootstrap over the per-user difference vector**: resample users, recompute the per-user delta, take the 2.5/97.5 percentiles. **10,000 resamples, seed 98765.** Never the unpaired `1.96*sqrt(2)*SE` form (`ARM_N_PHASE1.md` §4b: it overstated intervals ~4.6x and forced a retraction).

Contract: canonical Liang ML-25M, policy-B pool, NO-FLOOR, 10,000 test users. Source vectors `adaptive_table_gtport13_per_user.npz` (n_full=10000, n_tail=9281; the tail axis drops the 719 users with no tail target — a property of the DATA, so rows stay aligned across arms). Machine-readable: `experiments/baselines/interview/gtport13_paired.json`.

Generator `src/baselines/gtport_paired.py`; table renderer `src/baselines/gtport_paired_tables.py`. `SEP` = the 95% interval excludes 0.

## Controls

* **Self-test: PASS.** `golbandi_tree` against ITSELF gives delta 0.0000 with CI [0.0000, 0.0000] at all 8 (budget, axis) cells.
* **Row-alignment canary: PASS.** All 60 cells carry identical user counts.
* **Identity check: EXACT.** max per-user |`treestatic:entropy0` − `entropy0-static`| = 0.0e+00 over every consumer/budget/axis — the matched comparator is the same object per user, not merely to 4 dp in the mean.

* **Greedy admission: PASS.** The greedy order was re-scored inside THIS harness (`src/baselines/gtport_greedy_rescore.py`); its `entropy0` control reproduces the stored gtport13 per-user vector to 0.0e+00 per user, so the two arms are in one contract and can be differenced.

### 1. The CROSSOVER — `golbandi_tree` − `treestatic:entropy0`, consumer `ours`

| comparison | axis | k=1 | k=2 | k=4 | k=8 |
|---|---|---|---|---|---|
| golbandi_tree − entropy0 | full | -0.0078 [-0.0099, -0.0056] **SEP** | -0.0092 [-0.0115, -0.0069] **SEP** | +0.0032 [+0.0006, +0.0059] **SEP** | +0.0064 [+0.0037, +0.0091] **SEP** |
| golbandi_tree − entropy0 | tail | -0.0081 [-0.0106, -0.0057] **SEP** | -0.0050 [-0.0075, -0.0025] **SEP** | +0.0015 [-0.0011, +0.0040] ns | +0.0052 [+0.0024, +0.0079] **SEP** |

### 2. Against the strongest static — consumer `ours`

| comparison | axis | k=1 | k=2 | k=4 | k=8 |
|---|---|---|---|---|---|
| golbandi_tree − greedy items-only | full | -0.0203 [-0.0227, -0.0179] **SEP** | -0.0293 [-0.0321, -0.0264] **SEP** | -0.0203 [-0.0234, -0.0172] **SEP** | -0.0130 [-0.0160, -0.0100] **SEP** |
| golbandi_tree − greedy items-only | tail | -0.0083 [-0.0104, -0.0061] **SEP** | -0.0102 [-0.0130, -0.0075] **SEP** | -0.0053 [-0.0081, -0.0025] **SEP** | +0.0068 [+0.0039, +0.0098] **SEP** |
| entropy0 − greedy items-only | full | -0.0125 [-0.0146, -0.0104] **SEP** | -0.0201 [-0.0226, -0.0175] **SEP** | -0.0235 [-0.0262, -0.0209] **SEP** | -0.0194 [-0.0222, -0.0167] **SEP** |
| entropy0 − greedy items-only | tail | -0.0001 [-0.0023, +0.0021] ns | -0.0052 [-0.0077, -0.0027] **SEP** | -0.0068 [-0.0093, -0.0042] **SEP** | +0.0017 [-0.0009, +0.0043] ns |

### 3. The method as published — consumer `golbandi_leaf`

| comparison | axis | k=1 | k=2 | k=4 | k=8 |
|---|---|---|---|---|---|
| golbandi_tree − HELF | full | +0.0031 [+0.0017, +0.0045] **SEP** | -0.0028 [-0.0049, -0.0007] **SEP** | -0.0023 [-0.0055, +0.0010] ns | -0.0014 [-0.0053, +0.0025] ns |
| golbandi_tree − HELF | tail | +0.0004 [-0.0023, +0.0031] ns | +0.0012 [-0.0010, +0.0035] ns | -0.0002 [-0.0031, +0.0027] ns | -0.0001 [-0.0032, +0.0030] ns |
| golbandi_tree − entropy0 | full | +0.0117 [+0.0098, +0.0135] **SEP** | +0.0149 [+0.0127, +0.0171] **SEP** | +0.0187 [+0.0153, +0.0220] **SEP** | +0.0248 [+0.0210, +0.0287] **SEP** |
| golbandi_tree − entropy0 | tail | +0.0058 [+0.0033, +0.0082] **SEP** | +0.0072 [+0.0049, +0.0095] **SEP** | +0.0042 [+0.0013, +0.0069] **SEP** | +0.0115 [+0.0084, +0.0145] **SEP** |
| golbandi_tree − popularity | full | +0.0164 [+0.0140, +0.0188] **SEP** | +0.0178 [+0.0152, +0.0204] **SEP** | +0.0423 [+0.0388, +0.0458] **SEP** | +0.0218 [+0.0179, +0.0257] **SEP** |
| golbandi_tree − popularity | tail | +0.0074 [+0.0051, +0.0097] **SEP** | +0.0060 [+0.0038, +0.0083] **SEP** | +0.0210 [+0.0181, +0.0240] **SEP** | +0.0096 [+0.0066, +0.0125] **SEP** |

### 4a. The static ladder — consumer `ours`

| comparison | axis | k=1 | k=2 | k=4 | k=8 |
|---|---|---|---|---|---|
| entropy0 − popularity | full | +0.0029 [+0.0012, +0.0046] **SEP** | +0.0058 [+0.0042, +0.0075] **SEP** | +0.0085 [+0.0067, +0.0103] **SEP** | +0.0109 [+0.0090, +0.0129] **SEP** |
| entropy0 − popularity | tail | +0.0033 [+0.0013, +0.0054] **SEP** | +0.0058 [+0.0040, +0.0077] **SEP** | +0.0057 [+0.0041, +0.0073] **SEP** | +0.0106 [+0.0085, +0.0127] **SEP** |
| entropy0 − HELF | full | +0.0032 [+0.0016, +0.0048] **SEP** | +0.0078 [+0.0054, +0.0103] **SEP** | +0.0076 [+0.0046, +0.0106] **SEP** | +0.0160 [+0.0127, +0.0194] **SEP** |
| entropy0 − HELF | tail | +0.0079 [+0.0059, +0.0100] **SEP** | +0.0108 [+0.0081, +0.0137] **SEP** | +0.0048 [+0.0018, +0.0079] **SEP** | +0.0067 [+0.0034, +0.0102] **SEP** |
| popularity − HELF | full | +0.0003 [-0.0012, +0.0017] ns | +0.0020 [-0.0004, +0.0045] ns | -0.0009 [-0.0038, +0.0020] ns | +0.0051 [+0.0019, +0.0083] **SEP** |
| popularity − HELF | tail | +0.0046 [+0.0027, +0.0065] **SEP** | +0.0050 [+0.0023, +0.0077] **SEP** | -0.0009 [-0.0039, +0.0021] ns | -0.0039 [-0.0071, -0.0006] **SEP** |

### 4b. The static ladder — consumer `golbandi_leaf`

| comparison | axis | k=1 | k=2 | k=4 | k=8 |
|---|---|---|---|---|---|
| entropy0 − popularity | full | +0.0048 [+0.0022, +0.0073] **SEP** | +0.0029 [+0.0005, +0.0053] **SEP** | +0.0236 [+0.0204, +0.0268] **SEP** | -0.0030 [-0.0060, -0.0000] **SEP** |
| entropy0 − popularity | tail | +0.0016 [-0.0004, +0.0037] ns | -0.0011 [-0.0030, +0.0007] ns | +0.0169 [+0.0145, +0.0191] **SEP** | -0.0019 [-0.0043, +0.0005] ns |
| entropy0 − HELF | full | -0.0086 [-0.0103, -0.0069] **SEP** | -0.0177 [-0.0203, -0.0151] **SEP** | -0.0210 [-0.0248, -0.0172] **SEP** | -0.0262 [-0.0304, -0.0221] **SEP** |
| entropy0 − HELF | tail | -0.0054 [-0.0077, -0.0031] **SEP** | -0.0060 [-0.0085, -0.0035] **SEP** | -0.0044 [-0.0074, -0.0012] **SEP** | -0.0116 [-0.0149, -0.0082] **SEP** |
| popularity − HELF | full | -0.0133 [-0.0157, -0.0110] **SEP** | -0.0205 [-0.0236, -0.0175] **SEP** | -0.0445 [-0.0483, -0.0408] **SEP** | -0.0232 [-0.0276, -0.0188] **SEP** |
| popularity − HELF | tail | -0.0070 [-0.0094, -0.0046] **SEP** | -0.0048 [-0.0075, -0.0022] **SEP** | -0.0212 [-0.0243, -0.0181] **SEP** | -0.0096 [-0.0129, -0.0064] **SEP** |

### 4c. The static ladder — consumer `golbandi_leaf_lam200`

| comparison | axis | k=1 | k=2 | k=4 | k=8 |
|---|---|---|---|---|---|
| entropy0 − popularity | full | +0.0047 [+0.0022, +0.0073] **SEP** | +0.0029 [+0.0005, +0.0053] **SEP** | +0.0227 [+0.0196, +0.0259] **SEP** | -0.0062 [-0.0088, -0.0036] **SEP** |
| entropy0 − popularity | tail | +0.0016 [-0.0005, +0.0037] ns | -0.0011 [-0.0029, +0.0007] ns | +0.0167 [+0.0144, +0.0190] **SEP** | -0.0026 [-0.0049, -0.0005] **SEP** |
| entropy0 − HELF | full | -0.0086 [-0.0103, -0.0069] **SEP** | -0.0178 [-0.0204, -0.0152] **SEP** | -0.0208 [-0.0244, -0.0171] **SEP** | -0.0103 [-0.0134, -0.0072] **SEP** |
| entropy0 − HELF | tail | -0.0054 [-0.0077, -0.0031] **SEP** | -0.0059 [-0.0084, -0.0034] **SEP** | -0.0043 [-0.0073, -0.0011] **SEP** | -0.0102 [-0.0133, -0.0070] **SEP** |
| popularity − HELF | full | -0.0133 [-0.0156, -0.0109] **SEP** | -0.0207 [-0.0237, -0.0176] **SEP** | -0.0435 [-0.0472, -0.0399] **SEP** | -0.0041 [-0.0075, -0.0007] **SEP** |
| popularity − HELF | tail | -0.0070 [-0.0094, -0.0046] **SEP** | -0.0048 [-0.0075, -0.0022] **SEP** | -0.0210 [-0.0241, -0.0179] **SEP** | -0.0076 [-0.0106, -0.0045] **SEP** |

### 5. Leaf shrinkage — consumer `golbandi_leaf_lam200` − `golbandi_leaf`, same questions

| comparison | axis | k=1 | k=2 | k=4 | k=8 |
|---|---|---|---|---|---|
| golbandi_tree | full | +0.0000 [-0.0000, +0.0001] ns | -0.0000 [-0.0002, +0.0001] ns | -0.0006 [-0.0011, -0.0000] **SEP** | -0.0011 [-0.0035, +0.0013] ns |
| golbandi_tree | tail | -0.0000 [-0.0001, +0.0000] ns | -0.0001 [-0.0001, +0.0000] ns | +0.0000 [-0.0003, +0.0003] ns | -0.0012 [-0.0023, -0.0001] **SEP** |
| treestatic:HELF | full | -0.0000 [-0.0001, +0.0000] ns | +0.0000 [-0.0001, +0.0002] ns | -0.0008 [-0.0013, -0.0003] **SEP** | -0.0038 [-0.0051, -0.0025] **SEP** |
| treestatic:HELF | tail | -0.0000 [-0.0000, +0.0000] ns | -0.0000 [-0.0001, +0.0001] ns | -0.0002 [-0.0005, +0.0001] ns | -0.0018 [-0.0027, -0.0010] **SEP** |
| treestatic:entropy0 | full | -0.0000 [-0.0001, -0.0000] **SEP** | -0.0001 [-0.0002, +0.0000] ns | -0.0006 [-0.0012, -0.0000] **SEP** | +0.0121 [+0.0094, +0.0149] **SEP** |
| treestatic:entropy0 | tail | -0.0000 [-0.0001, +0.0001] ns | +0.0000 [-0.0001, +0.0001] ns | -0.0001 [-0.0004, +0.0001] ns | -0.0004 [-0.0016, +0.0007] ns |
| treestatic:popularity | full | +0.0000 [+0.0000, +0.0001] **SEP** | -0.0001 [-0.0002, +0.0000] ns | +0.0002 [-0.0003, +0.0008] ns | +0.0153 [+0.0128, +0.0179] **SEP** |
| treestatic:popularity | tail | +0.0000 [-0.0000, +0.0000] ns | -0.0000 [-0.0001, +0.0000] ns | -0.0000 [-0.0003, +0.0003] ns | +0.0003 [-0.0008, +0.0013] ns |

## Point estimates behind every delta

| consumer / selector | k=1 | k=2 | k=4 | k=8 |
|---|---|---|---|---|
| golbandi_leaf_lam200 / entropy0-static | 0.1579/0.0347 | 0.1562/0.0414 | 0.1675/0.0668 | 0.1807/0.0673 |
| golbandi_leaf_lam200 / golbandi_tree | 0.1696/0.0405 | 0.1711/0.0485 | 0.1862/0.0711 | 0.1923/0.0779 |
| golbandi_leaf_lam200 / treestatic:entropy0 | 0.1579/0.0347 | 0.1562/0.0414 | 0.1675/0.0668 | 0.1807/0.0673 |
| golbandi_leaf_lam200 / treestatic:helf | 0.1665/0.0402 | 0.1740/0.0473 | 0.1882/0.0711 | 0.1910/0.0774 |
| golbandi_leaf_lam200 / treestatic:popularity | 0.1532/0.0331 | 0.1533/0.0425 | 0.1447/0.0501 | 0.1869/0.0699 |
| golbandi_leaf / entropy0-static | 0.1579/0.0347 | 0.1563/0.0414 | 0.1680/0.0670 | 0.1686/0.0677 |
| golbandi_leaf / golbandi_tree | 0.1696/0.0405 | 0.1712/0.0486 | 0.1867/0.0711 | 0.1934/0.0792 |
| golbandi_leaf / treestatic:entropy0 | 0.1579/0.0347 | 0.1563/0.0414 | 0.1680/0.0670 | 0.1686/0.0677 |
| golbandi_leaf / treestatic:helf | 0.1665/0.0402 | 0.1739/0.0473 | 0.1890/0.0713 | 0.1948/0.0792 |
| golbandi_leaf / treestatic:popularity | 0.1531/0.0331 | 0.1534/0.0425 | 0.1445/0.0501 | 0.1716/0.0696 |
| ours / entropy0-static | 0.1635/0.0433 | 0.1706/0.0528 | 0.1831/0.0636 | 0.2053/0.0792 |
| ours / golbandi_tree | 0.1557/0.0351 | 0.1614/0.0478 | 0.1864/0.0650 | 0.2117/0.0844 |
| ours / treestatic:entropy0 | 0.1635/0.0433 | 0.1706/0.0528 | 0.1831/0.0636 | 0.2053/0.0792 |
| ours / treestatic:helf | 0.1603/0.0354 | 0.1628/0.0420 | 0.1755/0.0588 | 0.1892/0.0725 |
| ours / treestatic:popularity | 0.1606/0.0400 | 0.1647/0.0470 | 0.1746/0.0579 | 0.1943/0.0686 |
| ours / greedy-items-only (re-scored here) | 0.1760/0.0434 | 0.1907/0.0581 | 0.2066/0.0703 | 0.2247/0.0776 |
| ours / entropy0-top16 (re-scored here) | 0.1635/0.0433 | 0.1706/0.0528 | 0.1831/0.0636 | 0.2053/0.0792 |

## What `GOLBANDI_PORT_ON_OUR_RULER.md` may and may not say now

**Survives, with the interval attached.**

* *"The tree crosses over between k=2 and k=4. Against entropy0 it is -.0092 full at k=2 and +.0064
  full / +.0052 tail at k=8."* Both ends separate: -0.0092 [-0.0115, -0.0069] at k=2, +0.0064
  [+0.0037, +0.0091] and +0.0052 [+0.0024, +0.0079] at k=8. The crossover is real, not noise.
* *"the tree takes the tail at k=8 (.0844 vs .0776, +.0068)"* -- the number the chapter would rest on.
  +0.0068 [+0.0039, +0.0098], SEP. It holds.
* *"greedy items-only leads on full at every budget"* -- SEP at all four budgets (-0.0203, -0.0293,
  -0.0203, -0.0130).
* *"essentially tied with HELF"* on `golbandi_leaf` -- now a claim rather than an assertion:
  -0.0014 [-0.0053, +0.0025] full and -0.0001 [-0.0032, +0.0030] tail at k=8, both ns.
* *"the tree beats entropy0 and popularity clearly"* on its own consumer -- SEP at every budget, both
  axes.
* *"the consumer flips which static is strongest: HELF here, entropy0 on our tower"* -- the flip is
  separated in BOTH directions: entropy0 - HELF is +0.0160 [+0.0127, +0.0194] on `ours` and -0.0262
  [-0.0304, -0.0221] on `golbandi_leaf` at k=8 full. The instrument-dependence exhibit stands.

**Does NOT survive as written.**

1. *"On the ranking leaf it is inert: the lambda=200 arm differs from lambda=8 by <=.0015 everywhere."*
   FALSE as a universal. It is true of the `golbandi_tree` arm (-0.0011 full / -0.0012 tail at k=8),
   but on the STATIC arms at k=8 full it reaches +0.0121 [+0.0094, +0.0149] (entropy0) and +0.0153
   [+0.0128, +0.0179] (popularity), both SEP -- an order of magnitude past the stated bound, and in
   the opposite sign to the tree. "Everywhere" must become "on the tree arm"; and note the shrinkage
   moves the COMPARATOR more than the method, which shrinks the tree's k=8 win over entropy0 on that
   consumer from +0.0248 to +0.0116.
2. *"The tree crosses over between k=2 and k=4"* is right on full but over-stated on TAIL: at k=4 the
   tail gap is +0.0015 [-0.0011, +0.0040], ns. On the tail axis the crossover is only established by
   k=8; k=4 is a wash. Write the sentence per axis.
3. *"this is the first break in the pattern"* / *"the first time any adaptive selector has beaten the
   optimised static order on either axis"* -- supported only if "beaten" means one axis at one budget.
   The same k=8 cell is -0.0130 [-0.0160, -0.0100] on full, SEP. A single-axis, single-budget win
   against a comparator the doc itself calls a lower bound is not "the tree beats the optimised
   static"; say "wins the tail at k=8 while losing full at every budget".
4. The identity check reported as *"reproduces ... to 4 dp at every budget"* understates itself: the
   two arms are identical PER USER (max abs diff 0.0e+00). Strengthen it -- 4 dp on a mean is a much
   weaker statement than what the data supports.
5. The cross-run licence -- *"They are admissible because the two runs share an exact anchor:
   entropy0 reads .1635/.0433 ..."* -- is now moot and should be dropped rather than defended. The
   greedy order was re-scored INSIDE this harness (`gtport_greedy_rescore.py`), reproducing
   `.1760/.0434 . .1907/.0581 . .2066/.0703 . .2247/.0776` exactly and matching entropy0 per user at
   0.0e+00. A 4-dp mean anchor never licensed a PAIRED difference in the first place; per-user
   identity does.

**Newly visible, worth stating.** On our tower the static ladder is not a single order: entropy0 beats
both popularity and HELF at every budget on both axes (all SEP), but popularity - HELF flips sign by
axis at k=8 (+0.0051 full SEP, -0.0039 tail SEP). And entropy0 - greedy on tail is ns at k=1
(-0.0001) and k=8 (+0.0017) -- greedy's advantage over entropy0 is a FULL-axis advantage that only
separates on tail in the middle of the range.
