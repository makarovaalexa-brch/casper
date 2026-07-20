# THE FAIR RE-RUNS -- routers vs SPLIT-CONSTRUCTED statics (v2 fold; DIRECTIONAL 173/300)

> **DIRECTIONAL ONLY** -- 173/300 users, answerer-v1 working grid NOT frozen; v2 fold
> `.cache/i25_fold_v2_best.pt`. Re-run on the frozen 300-user grid before any citation.

Date 2026-07-09. Script `scripts/fair_reruns.py`. NO LLM API calls; local compute.
173 users; 180 candidates; cold NDCG@10 0.1502; fold val 0.4256. Paired per-user bootstrap BOOT=5000 seed=0. Greedy prefix-consistent,
one T=24 build serves budgets T=[8, 12, 24].

## Pre-registered reads (printed BEFORE results)

For each router, pooled (router - s-fair) with paired per-user bootstrap over the 4 split
estimates' concatenated per-user deltas (seeds {0,1} x eval-half {A,B}). s-fair = split-built
greedy s-item on the CONSTRUCTION half, evaluated on the EVAL half. Routers warm-start on
s-fair's construction-half schedule; value model V(q|z) fitted on population trU (firewall).

- **(i) CI excl 0 POSITIVE** = THE FIRST FAIR ADAPTIVITY WIN OF THE PROGRAM.
- **(ii) CI incl 0** = honest tie on a fair baseline (still a massive upgrade from 'loses').
- **(iii) CI excl 0 NEGATIVE** = routers genuinely lose even fairly (reported without softening).

Verdict uses the POOLED estimate at T=12. Sanity link: s-fair vs the OLD cohort-fit static
(greedy built on the eval-half itself) on the same eval users -> should reproduce ~-0.028
(= -contamination), linking to STATIC_CONTAMINATION.md. NO LLM calls; DIRECTIONAL 173/300.

## Pooled router-vs-s-fair contrasts (4 estimates concatenated)

| router | budget | pooled (router - s-fair) [95% CI] | n | verdict |
|---|---|---|--:|---|
| r-value-blind | T=8 | -0.0037[-0.0075,-0.0005] | 346 | LOSS |
| r-value-blind | T=12 | -0.0077[-0.0128,-0.0029] | 346 | LOSS |
| r-value-blind | T=24 | -0.0124[-0.0201,-0.0048] | 346 | LOSS |
| r-value+k | T=8 | -0.0054[-0.0095,-0.0018] | 346 | LOSS |
| r-value+k | T=12 | -0.0087[-0.0140,-0.0039] | 346 | LOSS |
| r-value+k | T=24 | -0.0149[-0.0227,-0.0075] | 346 | LOSS |
| r-blind-a6 | T=8 | -0.0022[-0.0043,-0.0005] | 346 | LOSS |
| r-blind-a6 | T=12 | -0.0028[-0.0058,-0.0001] | 346 | LOSS |
| r-blind-a6 | T=24 | -0.0032[-0.0062,-0.0001] | 346 | LOSS |

**Primary verdicts (T=12, pooled):**

- **r-value-blind:** -0.0077[-0.0128,-0.0029] -> FAIR-LOSS (CI excl 0, negative) -- routers lose even fairly
- **r-value+k:** -0.0087[-0.0140,-0.0039] -> FAIR-LOSS (CI excl 0, negative) -- routers lose even fairly
- **r-blind-a6:** -0.0028[-0.0058,-0.0001] -> FAIR-LOSS (CI excl 0, negative) -- routers lose even fairly

## Reference: s-mixed-fair vs s-item-fair (which fair static is stronger)

| budget | s-mixed-fair - s-item-fair [95% CI] | n |
|---|---|--:|
| T=8 | +0.0115[+0.0011,+0.0222] | 346 |
| T=12 | +0.0124[+0.0014,+0.0241] | 346 |
| T=24 | +0.0145[+0.0016,+0.0282] | 346 |

## Sanity link to STATIC_CONTAMINATION.md

s-fair (built on construction half) minus OLD cohort-fit static (built on the eval half itself),
both evaluated on the SAME eval users. This equals -(contamination); expect ~-0.028 at T=12.

| budget | s-fair - old-cohort-fit [95% CI] | n |
|---|---|--:|
| T=8 | -0.0241[-0.0336,-0.0150] | 346 |
| T=12 | -0.0283[-0.0377,-0.0193] | 346 |
| T=24 | -0.0343[-0.0436,-0.0251] | 346 |

## Per-estimate detail (T=12)

| estimate | n_eval | r-value-blind | r-value+k | r-blind-a6 | s-mixed-fair | sanity |
|---|--:|---|---|---|---|---|
| seed0_evalB | 87 | -0.0129[-0.0235,-0.0044] | -0.0133[-0.0245,-0.0038] | -0.0056[-0.0156,+0.0026] | +0.0159[-0.0133,+0.0466] | -0.0347[-0.0594,-0.0140] |
| seed0_evalA | 86 | -0.0021[-0.0110,+0.0058] | -0.0048[-0.0157,+0.0043] | -0.0008[-0.0033,+0.0015] | +0.0059[-0.0131,+0.0264] | -0.0221[-0.0369,-0.0085] |
| seed1_evalB | 87 | -0.0078[-0.0190,+0.0018] | -0.0067[-0.0148,+0.0015] | -0.0018[-0.0046,+0.0007] | +0.0154[-0.0029,+0.0342] | -0.0234[-0.0390,-0.0102] |
| seed1_evalA | 86 | -0.0077[-0.0187,+0.0027] | -0.0100[-0.0220,+0.0001] | -0.0027[-0.0105,+0.0023] | +0.0124[-0.0102,+0.0355] | -0.0329[-0.0564,-0.0130] |

