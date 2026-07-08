# STATIC-BASELINE TEST-FIT CONTAMINATION (v2 fold; DIRECTIONAL 173/300)

> **DIRECTIONAL ONLY** -- 173/300 users, answerer-v1 working grid NOT frozen; v2 fold
> `.cache/i25_fold_v2_best.pt`. Re-run on the frozen 300-user grid before any citation.

Date 2026-07-08. Script `scripts/static_contamination.py`. NO LLM API calls; local compute.
173 users; 180 candidates; cold NDCG@10 0.1502; fold val 0.4256. Paired per-user bootstrap BOOT=5000 seed=0. Greedy forward selection is
prefix-consistent, so ONE T=24 build serves budgets T=[8, 12, 24].

## Pre-registered reads (printed BEFORE results)

CONTAMINATION = (in-sample greedy static, built AND evaluated on the eval-half) minus
(out-of-sample greedy static, built on the OTHER half, evaluated on the eval-half). Positive =
the static gains from being fit to its own evaluation cohort. Paired per-user bootstrap over the
eval-half users (E1).

- **CONTAMINATED**: pooled contamination CI EXCLUDES 0 AND point estimate >= 0.005 ->
  every prior static-vs-router contrast needs split-constructed re-runs; the true (fair) static
  scores ~X lower, so each router's reported deficit shrinks by ~X.
- **HYPOTHESIS DEAD**: pooled contamination CI INCLUDES 0 OR point estimate < 0.005 ->
  the greedy statics do NOT meaningfully overfit their cohort; statics-optimal is strengthened
  (this outcome is equally valuable -- it removes a confound from the closed adaptivity verdict).

Verdict uses the POOLED estimate (all 4 per-user delta vectors concatenated). Per-estimate CIs
reported alongside. Reproduction gate: all-173 greedy s-item anytime@T=12 == 0.2251 +/- 0.002.

## 0. Reproduction (anchor gate)

All-173 greedy s-item, anytime NDCG@10 @T=12 = **0.2251** vs anchor 0.2251 +/- 0.002 -> **MATCH**.

## Contamination estimates (per split x eval-half x family)

### Family: s-item

| estimate | budget | in-sample | out-of-sample | contamination [95% CI] | n |
|---|---|--:|--:|---|--:|
| seed0 eval-B | T=8 | 0.2485 | 0.2178 | +0.0306[+0.0085,+0.0565] | 87 |
| seed0 eval-B | T=12 | 0.2536 | 0.2189 | +0.0347[+0.0140,+0.0594] | 87 |
| seed0 eval-B | T=24 | 0.2579 | 0.2154 | +0.0424[+0.0184,+0.0687] | 87 |
| seed0 eval-A | T=8 | 0.2089 | 0.1916 | +0.0173[+0.0046,+0.0311] | 86 |
| seed0 eval-A | T=12 | 0.2121 | 0.1900 | +0.0221[+0.0085,+0.0369] | 86 |
| seed0 eval-A | T=24 | 0.2180 | 0.1860 | +0.0320[+0.0174,+0.0484] | 86 |
| seed1 eval-B | T=8 | 0.2225 | 0.2007 | +0.0218[+0.0083,+0.0380] | 87 |
| seed1 eval-B | T=12 | 0.2252 | 0.2017 | +0.0234[+0.0102,+0.0390] | 87 |
| seed1 eval-B | T=24 | 0.2273 | 0.2009 | +0.0263[+0.0129,+0.0418] | 87 |
| seed1 eval-A | T=8 | 0.2402 | 0.2136 | +0.0266[+0.0063,+0.0501] | 86 |
| seed1 eval-A | T=12 | 0.2429 | 0.2100 | +0.0329[+0.0130,+0.0564] | 86 |
| seed1 eval-A | T=24 | 0.2453 | 0.2090 | +0.0363[+0.0197,+0.0546] | 86 |

**Pooled (all 4 estimates' per-user deltas concatenated):**

| budget | pooled contamination [95% CI] | n | CI excl 0? | >=0.005? |
|---|---|--:|---|---|
| T=8 | +0.0241[+0.0150,+0.0336] | 346 | yes | yes |
| T=12 | +0.0283[+0.0193,+0.0377] | 346 | yes | yes |
| T=24 | +0.0343[+0.0251,+0.0436] | 346 | yes | yes |

### Family: s-mixed

| estimate | budget | in-sample | out-of-sample | contamination [95% CI] | n |
|---|---|--:|--:|---|--:|
| seed0 eval-B | T=8 | 0.2400 | 0.2343 | +0.0057[-0.0050,+0.0159] | 87 |
| seed0 eval-B | T=12 | 0.2411 | 0.2348 | +0.0062[-0.0045,+0.0162] | 87 |
| seed0 eval-B | T=24 | 0.2422 | 0.2354 | +0.0068[-0.0036,+0.0164] | 87 |
| seed0 eval-A | T=8 | 0.2055 | 0.1962 | +0.0092[-0.0009,+0.0198] | 86 |
| seed0 eval-A | T=12 | 0.2053 | 0.1959 | +0.0094[+0.0002,+0.0189] | 86 |
| seed0 eval-A | T=24 | 0.2057 | 0.1954 | +0.0103[+0.0015,+0.0194] | 86 |
| seed1 eval-B | T=8 | 0.2233 | 0.2168 | +0.0065[-0.0040,+0.0179] | 87 |
| seed1 eval-B | T=12 | 0.2227 | 0.2172 | +0.0055[-0.0033,+0.0149] | 87 |
| seed1 eval-B | T=24 | 0.2223 | 0.2183 | +0.0040[-0.0022,+0.0103] | 87 |
| seed1 eval-A | T=8 | 0.2242 | 0.2225 | +0.0017[-0.0072,+0.0113] | 86 |
| seed1 eval-A | T=12 | 0.2250 | 0.2225 | +0.0025[-0.0050,+0.0104] | 86 |
| seed1 eval-A | T=24 | 0.2262 | 0.2202 | +0.0060[+0.0010,+0.0110] | 86 |

**Pooled (all 4 estimates' per-user deltas concatenated):**

| budget | pooled contamination [95% CI] | n | CI excl 0? | >=0.005? |
|---|---|--:|---|---|
| T=8 | +0.0058[+0.0007,+0.0109] | 346 | yes | yes |
| T=12 | +0.0059[+0.0013,+0.0104] | 346 | yes | yes |
| T=24 | +0.0068[+0.0027,+0.0107] | 346 | yes | yes |

## Schedule divergence (A-built vs B-built, first 12 picks)

`differ/12` = how many of A's 12 selected questions are NOT in B's schedule (= 12 - shared);
`pos-diff/12` = positions holding a different question; `sym-diff/24` = total unshared across the
union of both size-12 sets.

| seed | family | differ/12 | pos-diff/12 | shared/12 | sym-diff/24 |
|---|---|--:|--:|--:|--:|
| 0 | s-item | 5/12 | 12/12 | 7/12 | 10/24 |
| 0 | s-mixed | 5/12 | 10/12 | 7/12 | 10/24 |
| 1 | s-item | 7/12 | 11/12 | 5/12 | 14/24 |
| 1 | s-mixed | 8/12 | 12/12 | 4/12 | 16/24 |

### Channel / tier composition of each schedule (first 12)

| seed | family | half | channels | value-tiers (0=lowest..7=highest; -1=untiered) |
|---|---|---|---|---|
| 0 | s-item | A | item:12 | [7, 3, 5, 7, 1, 6, 7, 5, 6, 0, 2, 3] |
| 0 | s-item | B | item:12 | [7, 5, 7, 5, 3, 5, 4, 4, 7, 2, 6, 1] |
| 0 | s-mixed | A | concept:12 | [7, 6, 7, 6, 3, 7, 5, 5, 7, 1, 4, 6] |
| 0 | s-mixed | B | concept:12 | [7, 6, 6, 6, 5, 6, 5, 5, 7, 6, 7, 5] |
| 1 | s-item | A | item:12 | [7, 5, 4, 5, 6, 6, 7, 2, 7, 3, 5, 4] |
| 1 | s-item | B | item:12 | [7, 7, 4, 6, 6, 6, 2, 5, 5, 3, 3, 0] |
| 1 | s-mixed | A | concept:12 | [7, 6, 5, 5, 1, 7, 4, 5, 5, 6, 7, 3] |
| 1 | s-mixed | B | concept:12 | [7, 4, 6, 7, 6, 5, 5, 5, 6, 4, 4, 7] |

## Interpretation table -- prior static-vs-router contrasts (annotation math only, NO re-runs)

Each contrast reported (router - cohort-fit static). If the static carries a test-fit bonus X,
the FAIR (split-constructed) static would score ~X lower, so the router's DE-BIASED delta is
~ (reported delta + X). Positive de-biased delta => the router would win / tie once the baseline
is de-contaminated. X taken from the pooled s-item contamination at the nearest budget:

- X@T8  = +0.0241[+0.0150,+0.0336]
- X@T12 = +0.0283[+0.0193,+0.0377]
- X@T24 = +0.0343[+0.0251,+0.0436]

| prior contrast | fold/arena | reported (router - static) | budget | de-biased ~= reported + X |
|---|---|---|---|---|
| E0 schedule B (adaptive vs greedy static B) | additive-operator fold (VOID) | tie (~0) | T=12 | reported tie (~0) + +0.0283 = **n/a (qualitative)** |
| i25_phase4_fair blind vs s3/s4 | I2.5 fold i25_fold_best (v1) | -0.037..~tie (s3 best) | T=24 | reported -0.037..~tie (s3 best) + +0.0343 = **-0.0027** |
| i25_phase4 a6 vs s3 | I2.5 fold (v1) | -0.0021[-0.0056,+0.0015] | T=24 | reported -0.0021 + +0.0343 = **+0.0322** |
| battery Stage-C r-blind vs s-best | I2.5 fold i25_fold_best (v1) | -0.0060[-0.0155,+0.0020] | T=12 | reported -0.0060 + +0.0283 = **+0.0223** |
| vivid-swap T3 r-vivid vs s-best | I2.5 fold i25_fold_best (v1) | +0.0000[0,0] (declined) | T=12 | reported +0.0000 + +0.0283 = **+0.0283** |
| repair P3 r-value-blind vs s-best | v2 fold i25_fold_v2 (THIS) | -0.0068[-0.0116,-0.0023] | T=12 | reported -0.0068 + +0.0283 = **+0.0215** |
| repair P3 r-value+k vs s-best | v2 fold i25_fold_v2 (THIS) | -0.0126[-0.0225,-0.0049] | T=12 | reported -0.0126 + +0.0283 = **+0.0157** |

NOTE: the annotation adds X to the reported deficit. Contrasts on the v1 (i25_fold_best) or
the VOID additive-operator fold are annotated with the v2-measured X as an ORDER-OF-MAGNITUDE
guide only -- the exact bonus is fold-specific and would need a per-fold split-construction to
pin down. The two repair P3 rows are on the SAME fold as this measurement, so their de-biased
values are directly valid (still DIRECTIONAL 173/300).

