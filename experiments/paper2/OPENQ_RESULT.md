# Paper D P2 (exploratory) — OPEN-RECALL "favourite movie" BEATS D1 (2026-06-30)

"Name 8 favourite movies" (simulator names top-8 from the known half by a heuristic; fold real (Q,resid); NDCG@10
te[300:], seed-avg {1,2,3}), vs D1 (Paper C, 0.378/0.178) on the SAME frozen recommender + ruler:
| answerer heuristic | FULL | TAIL |
|---|---|---|
| rating (highest-rated favourite) | 0.411 | 0.209 |
| align (most u*-aligned) | 0.410 | 0.209 |
| distinct (high-align / low-pop = underrated favourite) | 0.405 | **0.211** |
| random5 (random 5-star) | 0.390 | 0.185 |
| popweight (REALISTIC popularity-biased recall) | **0.387** | 0.170 |
| poppop (names famous favourites = pessimistic) | 0.376 | 0.142 |
| D1 (Paper C ref) | 0.378 | 0.178 |
| half-fold ceiling | ~0.41 | ~0.21 |

## Takeaways
1. Open-recall BEATS the learned continuous policy D1. Realistic answerer (popweight) > D1 on full, ~level tail;
   info-optimistic ~hits the half-fold ceiling (0.41/0.21), well above D1 on both. PUBLISHABLE per the user's criterion
   (heuristic open-recall > our model on our recommender).
2. MECHANISM = bandwidth: naming a movie hands over a whole 64-d item factor + rating; rating a direction gives 1 scalar.
   Open questions are higher-bandwidth elicitation, same 8-question budget.
3. HEAD/TAIL: distinct (underrated favourites) -> best TAIL 0.211; poppop (famous favourites) -> worst tail 0.142. The
   answerer recall assumption is THE lever; realistic sits in the middle.

## Honest caveats
- align/rating are info-optimistic (user names their single most-informative favourite). popweight is the realistic
  honest answerer and still beats D1 on full. Report the answerer-sensitivity band.
- align/rating approach the half-fold ceiling because naming top-aligned items ~= revealing the informative half.
- Comparison is fair: same #questions (8), same encoder/recommender/ruler; open-recall's edge is the richer answer.

## Code / repro
OPENQ block in continuous_actor.py. HEUR=align|rating|popweight|poppop|distinct|random5, OPENK=k.
NOBC=1 EP=0 CONTMODE=cont OPENQ=1 HEUR=<h> OPENK=8 EVALSEEDS=1,2,3 python scripts/paper2/continuous_actor.py

## K-sweep (efficiency) + the QUESTION-FRAMING lever (2026-06-30)
NDCG@10 (full/tail), seed-avg {1,2,3}, vs D1 0.378/0.178:
| K | popweight (plain favourite) | distinct (underrated favourite) | align |
|---|---|---|---|
| 1 | 0.316/0.112 | 0.355/0.174 | 0.337/0.154 |
| 2 | 0.336/0.133 | **0.382/0.193** | 0.379/0.184 |
| 4 | 0.364/0.152 | 0.396/0.202 | 0.400/0.201 |
| 8 | 0.387/0.170 | 0.405/0.211 | 0.410/0.209 |

FINDINGS:
1. EFFICIENCY: distinct@K=2 (0.382/0.193) beats D1 on BOTH axes -> ~2 right questions > D1's 8 direction-probes.
   distinct@K=1 tail 0.174 ~= D1 tail with ONE question.
2. QUESTION-FRAMING is the lever, mapping onto head/tail: plain "favourite" (popweight->popular) is weak on tail
   (needs K=8 to reach D1); "underrated/hidden-gem favourite" (distinct) is far better on tail (0.193@K=2). The WORDING
   of the open question controls recall informativeness -> a controllable design knob. (Novel.)
3. Honest band: distinct is optimistic (user names their MOST distinctive favourite); realistic value between
   popweight and distinct. Direction unambiguous; framing effect large.
NEXT: portfolio of question TYPES (favourite + hidden-gem + hated-movie for negative/tail pruning); the answerer per type.
