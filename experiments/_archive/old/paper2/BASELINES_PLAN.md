# Paper D published-baselines replication plan (2026-06-30, from deep-research agent)

Ruler: ML-1M, frozen V1 set-encoder s=pop_b+Q·u, KNOWN-half elicit / HELD-half target, NDCG@10 full+tail, 8 turns,
GEOMETRIC known-profile answerer (NEVER peeks held target). Any baseline must produce a rankable u under THIS answerer.

## VERDICT per baseline
| baseline | type | needs LLM@eval | reported metric | difficulty | status on our ruler |
|---|---|---|---|---|---|
| Rashid Popularity/Entropy0/HELF + IGCN | probe (rate item) | No | MAE | 1-2 | **MOSTLY DONE** (Paper A/B have entropy/pop selection). Re-report on Paper-D table. |
| Golbandi/fMF adaptive ternary tree | probe (rate item) | No | RMSE | 2-3 | NEW-ish: build depth<=8 tree, fold answers into encoder. Interpretable adaptive floor (fig:tree). |
| NICF (DQN over history) SIGIR'20 | probe (which item) | No | cum P@40/R@40 ML-1M | 3 | **DONE** (Paper B task #52,#59 NICF on te[300:]). Re-report; reward=ΔNDCG. |
| ConTS (Thompson, item+attr) TOIS'21 | probe (attr/item) | No | SR@15/AvgTurns | 3 | **NEW** — implement: Bayesian-linear posterior over u; arms={concepts}∪{items}; geometric answer; NDCG@8. Subsumes EAR/CRM. |
| PEBOL (BO+LLM aspects) RecSys'24 | probe (yes/no aspect) | Yes | MRR@10 /100-shortlist | 4 | **NEW (geometric reduction)** — keep Beta(α,β)-per-item Thompson/UCB; replace LLM-aspect+NLI with concept-embed + geometric entailment w_i=σ(τ·aspect·u_known); α+=w,β+=1-w. STATE the NL-layer substitution. |
| EAR / CRM | probe (attr) | No | SR@15 | 4/2 | SUBSUMED by ConTS (beats both, same family). Drop standalone. |
| GATE ICLR'25 | probe (free-form LLM) | Yes | AUC binary-label | 5/2 | OFF-RULER (no ranking, no rec). CITE as related work only. Non-LLM proxy = our concept-EIG (don't relabel). |
| Funnel 2510.12015 | probe (general->specific LLM) | Yes | BLEU/ROUGE recon | 5/2 | OFF-RULER (text recon). CITE only. |
| Sanner LLM-CRS RecSys'23 | one-shot recall | Yes | NDCG@10 | - | NOT a policy — it's the FULL-named-profile CEILING reference. Already our half-fold ceiling analog. |

## TO IMPLEMENT for Paper D (4 non-LLM + 1 reduced-LLM):
1. Rashid static suite (pop/entropy0/HELF) + IGCN — near-free, leverage Paper B. [probe floor]
2. Golbandi/fMF adaptive ternary tree — interpretable discrete-answer adaptive floor.
3. NICF DQN — leverage Paper B; reward=ΔNDCG@10. [learned-RL comparator]
4. ConTS — Bayesian-linear TS over concepts∪items; geometric answer. [bandit, subsumes EAR/CRM]
5. PEBOL-geometric — Beta-per-item Thompson + geometric entailment over concept channel. [the LLM method, reduced]
GATE+funnel = related-work citations (off-ruler). Sanner = full-profile ceiling row.

## RESULTS — probe baselines on OUR ruler (LITBASE block, geometric reduction, seed-avg{1,2,3} te[300:])
| method | q5 full/tail | q8 full/tail |
|---|---|---|
| ConTS (TS over concept+item arms) | 0.341/0.140 | 0.351/0.151 |
| PEBOL-geometric (Beta-per-item + concept aspects, Thompson acq) | 0.372/0.173 | 0.369/0.168 |
| entropy-over-concepts (Paper B) | ~0.36 | ~0.361/0.140 |
| Paper C D1 (continuous probe) | — | 0.378/0.178 |
| **OPEN-recall asker (ours)** | **0.398/0.190** | **0.405/0.194** |
| **OPEN+CLOSED (ours)** | **0.400/0.193** | **0.405/0.197** |
EVERY published probe elicitor < open recall. PEBOL-geo (0.369/0.168) = strongest probe ~= Paper C D1 (both probes),
~0.035 full / ~0.027 tail BELOW open asker. ConTS weaker (TS burns early turns). PEBOL DEGRADES q5->q8 (concept aspects
saturate). HONEST: geometric reductions — faithful to acquisition/belief mechanics, NL/NLI layer replaced (our answerer
is geometric). Repro: LITBASE=conts|pebol QPTS=5,8 EVALSEEDS=1,2,3. GATE/funnel = related-work cites (off-ruler).

## KEY framing for Paper D table
The probe baselines (pop/entropy/NICF/ConTS/PEBOL/tree) are all CLOSED rating-probes -> they sit in the Paper-C family.
Paper D's OPEN-RECALL asker + the OPEN+CLOSED joint asker are compared against ALL of them on the SAME ruler+answerer.
Sources: PEBOL 2405.00981 (code github.com/D3Mlab/llm-pe); NICF 2007.02095 (github.com/zoulixin93/NICF); ConTS 2005.12979;
GATE 2310.11589; funnel 2510.12015; Sanner 2307.14225; Golbandi WSDM'11; EAR 2002.09102.
CAVEATS: PEBOL per-dataset cells approximate; fMF ML RMSE unverified; EAR numbers differ across papers (don't cross-compare).
