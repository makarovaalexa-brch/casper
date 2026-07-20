# Paper D answerer-band rerun (2026-07-01, review D-B1/B2/B3 response) — 5 canonical seeds, ±std, te[300:], K=8

Simulator NAMES 8 favourites from the KNOWN half by heuristic; fold real (Q,resid) tokens; NDCG@10 full + Cremonesi tail.
Repro: OPENQ=1 HEUR=<h> OPENK=8 EVALSEEDS=1,2,3,7,11 NOBC=1 EP=0 CONTMODE=cont python scripts/paper2/continuous_actor.py

| answerer model (what the user names) | FULL | TAIL |
|---|---|---|
| rating (highest-rated liked)            | 0.4121+/-0.0024 | 0.2097+/-0.0044 |
| align (max u*.Q; info-optimistic)       | 0.4111+/-0.0024 | 0.2083+/-0.0049 |
| distinct (hidden-gem, high-align/low-pop; tail-optimistic) | 0.4065+/-0.0033 | 0.2115+/-0.0036 |
| random5 (uniform among likes; ASSUMPTION-FREE) | 0.3945+/-0.0078 | 0.1883+/-0.0058 |
| popweight ("realistic" = popularity-weighted random draw; AVAILABILITY-HEURISTIC PROXY, unvalidated) | 0.3860+/-0.0040 | 0.1715+/-0.0067 |
| poppop (most-popular liked; PESSIMISTIC) | 0.3791+/-0.0039 | 0.1437+/-0.0045 |
| ref: D1 continuous probe                | 0.378 | 0.178 |
| ref: fair-strict PEBOL / ConTS          | 0.317 / 0.309 | 0.092 / 0.089 |

## Honest reading (for Paper D)
- ROBUST: open recall beats ALL fair-strict discrete probes (PEBOL/ConTS) across the ENTIRE band, both axes.
- vs continuous D1 (0.378/0.178): FULL >= D1 across whole band (even pessimistic poppop 0.379 ~ ties); TAIL beats D1 for
  optimistic (align/distinct/rating 0.208-0.212) AND assumption-free random5 (0.188) but LOSES for popweight (0.172) /
  poppop (0.144). => tail-vs-D1 is BAND-CONDITIONAL (state openly).
- random5 (assumption-free, no popularity model) beats D1 on BOTH axes (0.395/0.188) — cleanest positive; PROMOTE it.
- "framing lever" at K=8: align (favourite) 0.411/0.208 vs distinct (hidden-gem) 0.407/0.212 — hidden-gem +0.003 tail,
  favourite +0.005 full. Small at K=8 (bigger at K=2). REPORT distinct/align as OPTIMISTIC CEILINGS, popweight as
  availability proxy (unvalidated -> human study future). RELABEL popweight (drop "realistic").
- 5-seed numbers (align 0.411) are slightly HIGHER than the paper's quoted 0.405 (3-seed era) -> headline strengthens.
- NO OVERTURN. Std small (0.002-0.008). NEXT: noise-ablation of the recall answerer (softmax temperature) to bound the band.
