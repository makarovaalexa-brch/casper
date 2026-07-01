# Paper D metadata/themes rung — "favourite ACTOR / DIRECTOR / GENRE?" (2026-06-30)

Open-recall beyond items: a named PERSON/GENRE grounds to the **centroid of their catalogue films' factors** (all the
system can infer from a name, user-agnostic). User names the most-salient favourite from their KNOWN-HALF liked films;
fold the centroid token (+POS). Coarser than item recall (averages a whole filmography) but OBJECTIVE + broad-coverage.

Cast/crew: TMDB via links.csv (movieId->tmdbId), scripts/paper2/fetch_credits_ml1m.py ->
.cache/credits_ml1m_actors5.json (top-5 billed actors + directors, full ml-1m catalog). Genre: ml-1m movies.dat (18
genres). METAQ block in continuous_actor.py.

Repro: NOBC=1 EP=0 CONTMODE=cont METAQ=1 METATYPE=actor|director|both|genre METAHEUR=count|rating|align METAK=k
[MIXFAV=n] EVALSEEDS=1,2,3 python scripts/paper2/continuous_actor.py

## GENRE rung (offline, seed-avg {1,2,3}) — themes/mood, coarsest (18 genres, ~344 films/genre)
| heuristic | K | FULL | TAIL |
|---|---|---|---|
| count | 3 | **0.3462** | 0.1083 |
| count | 8 | 0.3273 | 0.1009 |
| align | 3 | 0.3404 | **0.1434** |
| align | 8 | 0.3349 | 0.1172 |

GENRE findings:
1. Below items AND below D1 on FULL — genre is the COARSEST token (centroid of ~344 films averages out everything).
   The honest bottom of the ladder.
2. FEWER genres is BETTER: K=3 > K=8 on both heuristics. Naming all your genres dilutes the belief toward a generic
   centroid; 3 well-chosen genres is the sweet spot. (Counterpart to the item portfolio's K-efficiency.)
3. align-K3 lifts TAIL to 0.143 (best genre tail) by naming the genre whose centroid best matches the user (a
   distinctive-genre framing) — the same head/tail framing lever as items, weaker in absolute terms.

## ACTOR / DIRECTOR rung (seed-avg {1,2,3}) — metadata, mid-coarse (8569 actors, 1972 directors, full ml-1m)
| open question (8 names) | FULL | TAIL |
|---|---|---|
| favourite movie (item, align, ref) | 0.410 | 0.209 |
| **favourite actor (count = REALISTIC)** | **0.369** | **0.154** |
| favourite actor (rating) | 0.377 | 0.160 |
| favourite actor (align = info-fav) | 0.408 | 0.203 |
| **favourite director (count)** | **0.378** | **0.180** |
| favourite actor+director (both, count) | 0.379 | 0.166 |
| D1 (Paper C, ref) | 0.378 | 0.178 |
| **MIX: 4 actors + 4 favourite movies (count)** | **0.409** | **0.203** |

ACTOR/DIRECTOR findings (CONFIRMED story):
1. REALISTIC actor recall (count = the actor in most of your liked films) = 0.369/0.154 — BELOW items and ~D1: an
   actor's filmography centroid is coarse, and the most-FREQUENT actor isn't the most discriminative. Honest operating pt.
2. info-favourable actor (align = the actor whose centroid best matches u*) = 0.408/0.203 — nearly MATCHES favourite-
   movie. When the user names their most-characteristic actor, the centroid is highly informative. = the upper band.
3. DIRECTOR (count) = 0.378/0.180 ≈ D1 EXACTLY, and BEATS actor(count) on both axes. A director's filmography is more
   stylistically COHERENT (auteur signal) than an actor's, so one director-name is worth more than one actor-name even
   under realistic recall. Nice, non-obvious finding.
4. MIX (4 actors + 4 favourite movies) = 0.409/0.203 — best-of-both: a cheap OBJECTIVE opener (actors) + precise ITEM
   anchors recovers near-pure-item performance. Validates the open+closed mix.

LADDER CONFIRMED (NDCG, realistic recall): items (0.39 popweight) > director (0.378) ~ D1 > actor (0.369) > genre (0.346).
Reversed on ease-of-answer / coverage (genre easiest+broadest, item hardest+narrowest) = the paper's openness-vs-precision
axis. Metadata's value is ANSWERABILITY+COVERAGE, not peak NDCG; the MIX gets both.

## FULLY-OPEN "what do you like?" rung (METATYPE=union, seed-avg {1,2,3})
Union answer space: user may name ANY entity type (movie/actor/director/genre); ranked by cosine-alignment to u*
(comparable scale across types) = INFO-OPTIMISTIC fully-open. METAK=8.
| fully-open | FULL | TAIL | type-mix named |
|---|---|---|---|
| union-align (optimistic) | 0.4070 | 0.2063 | person 68%, movie 29%, genre 3% |
| realistic fully-open := popweight favourite-movie | 0.387 | 0.170 | (movie, top-of-mind popular) |

FULLY-OPEN findings (closes the ladder):
1. OPTIMAL fully-open ~= item-level (0.407/0.206): if the user volunteers their single most-characteristic entity,
   maximal openness costs almost nothing vs a designed question. Type-mix: PERSONS dominate optimal recall (68%) — a
   well-aligned actor centroid implicates many on-target films (high coverage per name).
2. BUT fully-open FORFEITS THE FRAMING LEVER: one generic question can't steer head/tail. The REALISTIC "what do you
   like?" = naming a popular top-of-mind MOVIE = popweight-movie 0.387/0.170 — fine on FULL, WEAK on TAIL. A TARGETED
   hidden-gem question reaches 0.193 tail @K=2 (distinct), far above generic-open's 0.170. => generic openness sacrifices
   exactly the tail-steering that is novelty #1. The framing lever REQUIRES a non-fully-open (worded) question.
3. So the ladder has a U-shape in PRACTICE: precise items (designed) best; fully-open optimal ~= items but realistically
   defaults to popular-movie and loses the tail; metadata/genre in between. The paper's payoff: WORD the question.
Repro: NOBC=1 EP=0 CONTMODE=cont METAQ=1 METATYPE=union METAK=8 EVALSEEDS=1,2,3 python scripts/paper2/continuous_actor.py
