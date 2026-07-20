# Askable questions catalog (Paper D) — PARKED list (2026-06-30)

Mix of OPEN (free-recall: user names an entity) and CLOSED (rate a presented thing). Each maps to an embedding token
(entity -> factor / centroid) folded by the frozen Paper-C encoder. We can MIX open + closed in one session.

## OPEN — no extra metadata needed (START HERE)
- **Favourite movie** -> liked item factor Q_j. [head anchor, precise] **<- IMPLEMENT FIRST**
- 2nd / 3rd favourite movie -> more liked item factors. [multi-recall]
- A movie you **disliked/hated** -> low-rated item factor (negative). [pruning, informative]
- A movie you love that's **underrated / a guilty pleasure** -> high-rated + low-popularity item. [TAIL-discriminating]
- A **recent** movie you enjoyed -> recency-weighted high rating. [salience]
- Favourite **genre** -> genre concept centroid. (MovieLens HAS genres natively; map genre -> its movies' centroid or
  the matching genome concept.) [coarse, no metadata]
- A **genre you avoid** -> negative genre concept. [tail pruning]

## OPEN — needs metadata (DEFER to P0)
- Favourite **actor** / **director** -> person -> their films' centroid. (needs ML cast/crew metadata.)

## CLOSED (the Paper B/C paradigm, for MIXING)
- Rate this **concept / genome-tag** (closed concept). [Paper B]
- Rate this **continuous direction** q (the off-manifold query). [Paper C / D1]
- Rate this specific **movie** (closed item, if shown).

## MIX strategies to explore
- Open-anchor -> closed-refine: "favourite movie?" (precise anchor) then rate continuous directions to refine the tail.
- Positive-recall + negative-recall: "favourite" (head) + "hated/avoid" (prune) + "guilty pleasure" (tail).
- Pure open-recall sequence vs pure closed (Paper C) vs hybrid -> the ablation.

## The answerer (simulator) — heuristic recall, ABLATED (no NDCG-opt = no cheating)
Which entity does the user name? (for "favourite movie", among their known-half items):
- align: argmax u*.Q_j (most-aligned) [info-optimistic]
- rating: highest-rated, tie-break by align
- popweight: popularity-weighted sample among >=4-star [REALISTIC default]
- random5: random among 5-star
- poppop: most-popular among >=4-star [pessimistic, low-info]
- distinct: high align / low popularity [TAIL-info; "underrated favourite"]

## Open-question BASELINES from lit (for later)
PEBOL (BO over LLM aspects), GATE (LLM open-ended), 2510 funnel (general->specific). Note: none is a learned policy on
NDCG; most report profile-reconstruction not NDCG. Reuse Paper B baselines for the closed side.
