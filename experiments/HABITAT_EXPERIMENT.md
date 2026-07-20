# HABITAT EXPERIMENT -- fuel-in-pool, fuel-trained router, both metrics (v2 fold)

> **DIRECTIONAL ONLY -- 173/300 users, answerer-v1 working grid NOT frozen; v2 fold `.cache/i25_fold_v2_best.pt`. Re-run on the frozen 300-user grid before any citation.**

Date 2026-07-09. Script `scripts/habitat_experiment.py`. NO LLM API calls; local compute. 173 users; fold val 0.4256. Paired per-user bootstrap BOOT=5000. Executes the audit's single corrective experiment (experiments/FRESH_AUDIT.md), removing masks 1 (pool), 2 (fuel-free training), 3 (no taste-policy) at once.

## E7 arm-symmetry table (what each arm may see / learn -- listed before results)

| dimension | specification | symmetry note |
|---|---|---|
| candidate pool | habitat-stratified (20 top-cov + 40 rate-0.2..0.7), population-selected | SAME pool for every arm |
| static construction | greedy on CONSTRUCTION half only (endpoint@50) | router beliefs also construction half only |
| value model V(q) | mean single-answer lift on CONSTRUCTION half (REAL grid) | NOT population sims (Finding 2 fix) |
| knowledge belief | LOUO logistic-MF on CONSTRUCTION half (concept/attr) + population kmap (item) | eval user's own cells never train it (E5) |
| online evidence | eval user's own answered/refused events only | same for router; static is non-adaptive |
| selection value peek | router uses V_pop (population), NEVER the eval user's arena answer value | Finding 9 fixed |
| eval set | eval half only, every user every arm (E1 survivorship) | refusal = no-op turn, cold fallback |
| metrics | anytime@10 + endpoint@10/@50 at T=12/24; PRIMARY endpoint@50 T=24 | same ruler every arm |

## Pre-registered reads (printed BEFORE results)

**PRIMARY (chosen plainly): endpoint NDCG@50 at T=24.** Rationale (audit Finding 5): anytime@10
saturates by turn 4-6 and cannot pay back exploration; endpoint gives an adaptive descent room to
register, and @50 reaches the niche/tail depth where per-user structure lives. All other cells
(anytime@10, endpoint@10/@50 at T=12/24) are reported too, no cherry-pick.

For each router, pooled (router - split-fair static) over the 4 split estimates' concatenated
per-user deltas (seeds {0,1} x eval-half {A,B}), paired per-user bootstrap:
- **(i) CI excludes 0 POSITIVE** = the first fair adaptivity win in a pool that CONTAINS the fuel.
- **(ii) CI includes 0** = honest tie -- reported ONLY WITH its MDE (a tie below the MDE is a
  non-answer, not evidence of no-effect).
- **(iii) CI excludes 0 NEGATIVE** = the router loses even in the habitat pool (reported plainly).

MECHANISM CHECK (pre-registered): per-arm mean per-user REFUSAL RATE in the habitat -- the router
should show LOWER refusals than the static (it routes around no-clue questions). Plus the
taste-descent trace for 3 example users (which habitat questions got picked after which answers).

Anchoring (audit Finding 6): primary comparator = s-mixed-new (split-fair, habitat pool). Also
reported vs s-item-new, and vs the OLD top-coverage-pool static (reference row).

## Pool answer-rate distribution + per-user in-pool refusal rate

| pool | channel | n | habitat | rate min | p25 | median | max | frac in 0.2-0.7 band |
|---|---|--:|--:|--:|--:|--:|--:|--:|
| NEW habitat | concept | 60 | 40 | 0.20 | 0.32 | 0.51 | 1.00 | 0.67 |
| NEW habitat | item | 23 | 3 | 0.39 | 1.00 | 1.00 | 1.00 | 0.13 |
| NEW habitat | attr | 60 | 40 | 0.23 | 0.55 | 0.65 | 1.00 | 0.67 |
| OLD top-cov | concept | 60 | 0 | 1.00 | 1.00 | 1.00 | 1.00 | 0.00 |
| OLD top-cov | item | 60 | 0 | 0.98 | 1.00 | 1.00 | 1.00 | 0.00 |
| OLD top-cov | attr | 60 | 0 | 0.99 | 1.00 | 1.00 | 1.00 | 0.00 |

**Per-user in-pool REFUSAL rate (fraction of pool the user cannot answer):**

| pool | mean | median | p90 | max |
|---|--:|--:|--:|--:|
| NEW habitat | 0.310 | 0.315 | 0.476 | 0.573 |
| OLD top-cov | 0.006 | 0.006 | 0.017 | 0.072 |

The OLD pool's per-user refusal is ~0 (audit Finding 1: answerability is the constant 1, nothing to route on). The NEW pool restores real refusal variance -- the fuel is now in the action space.

