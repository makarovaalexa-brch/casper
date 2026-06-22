# Policy Optimization Log — beat conc_pop in cold-start (one change at a time)

TARGET to beat (locked ruler: ML-1M, concept-aware enc, ANSWER=geom, realistic cold-start, q8):
- **conc_pop**: FULL 0.315 / TAIL 0.116  (most-frequent concepts; the baseline OURS must match then beat)
- pop_item: 0.307 / 0.085 ; q0: 0.293 / 0.088
- ceilings: conc_gprof (profile) 0.330 / 0.143 ; conc_oracle (test) 0.469 / 0.325

KEY MECHANISM (PART AG/AG2): concept belief saturates at cos~0.83 with u*; ITEMS converge to ~1.0;
**BOTH (items+concepts) breaks the ceiling -> cos 0.887**. So OURS must use the item+concept COMBINATION,
and the right training objective is to **RECONSTRUCT u*** (user embedding), not just held-out item BCE.

## Optimizations tried (one variable each)
| # | change (vs previous) | FULL q8 | TAIL q8 | ans/8 | verdict |
|---|---|---|---|---|---|
| O1 | emit+snap actor, soft-snap, recon-BCE | 0.288 | 0.102 | 6.4 | FAIL (below conc_pop; train/eval mismatch) |
| O2 | + answerability prior (POOL_PRIOR) bias | 0.281 | 0.111 | 7.6 | ~conc_pop (prior-dominated) |
| O3 | scorer + popularity feature | 0.288 | 0.102 | 6.4 | FAIL (below) |
| O4 | + population info-gain feature | 0.288 | 0.102 | 6.4 | FAIL (below) |
| O5 | straight-through (train==eval) | 0.293 | 0.085 | 1.4 | FAIL (collapsed to items) |
| O6 | BC-to-conc_pop floor + BCE finetune | RUN | RUN | | (running) |
| O7 | objective = RECONSTRUCT u* (MSE) | DIVERGED | | | FAIL (MSE explodes w/ straight-through) |
| O7b | RECONSTRUCT u* (1-cos, bounded) from BC floor | ~0.31 | ~0.11 | | flat at concept ceiling (cos stuck 0.81~=conc_pop); finetune didn't improve |
| **conc_mix** | non-learned: interleave popular ITEM + frequent CONCEPT (the COMBINATION) | **0.318** | **0.116** (Rec 0.159) | 5.0 | **>= conc_pop** (full-NDCG +0.003, tail-Rec +0.006) with FEWER answers -> combination validated |

## Queue (to try next, one at a time)
- O8: BC-to-conc_pop floor + **u\*-reconstruction** finetune (combine O6 floor + O7 objective)
- O9: confirm COMBINATION: run OURS with pool = concepts-only vs items-only vs BOTH (show both > each)
- O10: cos(u,u*) reward weighting / add magnitude term; tune TAU
- O11: explore schedule; longer finetune; per-turn intermediate u* reward (dense)
- O12: stronger answerable-item inclusion (so the few fine items get folded)

## Notes
- "match conc_pop must be trivial" (user): O6 BC floor guarantees it; finetune must not regress below.
- gate each: must be >= conc_pop on BOTH full+tail; log every run here.

| O8 | BC floor + u*-cos, LR 5e-4, EXPL 0.3 | ~0.31 | ~0.11 | | stuck at floor |
| O9 | BC floor + REINFORCE (cos reward) | ~conc_pop | | | cos saturates (concept ceiling) |
| O10 | BC floor + REINFORCE (coverage reward) | ~0.31 | ~0.10 | | gameable (inflation) |
| O11 | BC floor + dense REINFORCE (Dcov - PEN*unans) | ~conc_pop | | | flat (BC trap) |
| O12 | **NO-BC** + dense coverage REINFORCE | 0.309 | 0.105 | 0i/all-c | LEARNS (return rises -> BC WAS trapping) but gameable coverage; eval ~conc_pop |
| O13 | NO-BC + RANK-AWARE reward (likes - non-likes) + answerability penalty | 0.291 | 0.075 | 27i/1037c | picks some items, return rises, but TEST WORSE (tail declines) |

## VERDICT (after 13 variants)
Pure cold-start q8: NO learned policy beats conc_pop. Each user-identified bug was real & fixed (crippled EIG, BC trap,
gameable coverage) and it STILL doesn't beat conc_pop -> training reward improves while TEST NDCG doesn't = no realizable
signal left, NOT a bug. Lines up with the MEASURED ceiling (PART AG/AG2): concepts saturate cos 0.83, answerable items
sparse (1.9/8) -> item+concept gain marginal at q8 (conc_mix +0.003). conc_pop is the realizable cold frontier = a CEILING
OF THE SETTING. To beat it needs item-level fineness: (a) WARM/returning user (conc_gprof +23%), (b) higher budget, (c)
SPARSE/large catalogue (frequency weak). NOT more policy tweaking.
