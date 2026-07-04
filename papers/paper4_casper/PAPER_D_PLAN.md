# Paper D — Open-Question (free-recall) Preference Elicitation — PLAN (2026-06-30)

## Reframe + thesis
Paper C: the policy GUESSES a direction q; the user rates it (cos(u*,q)). Paper D: the agent asks an OPEN question
("what's your favourite movie/actor?"), and the user VOLUNTEERS a named entity, which maps to a precise embedding token.
**Thesis: open free-recall elicits precise, high-information preferences in cold-start that the rating menu cannot —
because the user names their most salient/extreme items instead of being probed — and a learned policy sequences these
open questions to maximise NDCG.** Unlike Paper C (accuracy edge is marginal + off-manifold), open-recall can be a REAL
accuracy lever: "name a movie you love" returns a full liked-item factor with NO guessing.

KEY NUANCE (head vs tail): "favourite movie" tends to be POPULAR (head) -> helps FULL, less TAIL. The TAIL needs
DISTINCTIVE/NEGATIVE recall ("a guilty pleasure / underrated film", "a genre you hate"). So the open-question CATALOG must
span head-anchoring AND tail-discriminating questions, and the policy learns which to ask.

## Q1 — "favourite movie?" answer model (the user has many 5-star ratings)
The simulator must pick WHICH liked movie the user names. This is the load-bearing assumption; ABLATE it:
- (info-optimistic) argmax_j u*.Q_j over rated -> most-aligned/distinctive favourite (HIGH info, upper bound).
- (realistic-salience) popularity-weighted sample among their >=4-star items -> people name famous favourites (Shawshank,
  Inception) = realistic but LOWER info. DEFAULT.
- (recency) most-recent high rating.
- (pessimistic) most-popular among their likes (lowest info).
Report NDCG under each (answer-model sensitivity, like Paper C's answer dependence). The salience<->info tension is the
honest core: realistic recall is popularity-biased = less informative; the policy must compensate (ask for DISTINCTIVE
favourites). "Name THREE favourites" reduces the single-pick variance and is realistic.

## Q2 — the open-question catalog (grounded in the recommender)
Each maps to an embedding token (entity -> factor / centroid). Span head/tail/positive/negative/objective/subjective:
- favourite movie(s) -> liked item factor(s)            [head anchor, precise]
- a movie you HATED -> disliked item factor (negative)   [pruning, informative]
- favourite GENRE -> genre concept                        [coarse, high coverage]
- favourite ACTOR / DIRECTOR -> person -> their films' centroid   [objective, easy to answer; NEED ML metadata]
- a guilty pleasure / an UNDERRATED film -> niche distinctive item   [TAIL-discriminating, high info]
- a genre/thing you can't stand -> negative concept       [TAIL pruning]
- what you watch to relax / be scared / think -> mood/context concept   [subjective, harder]
- favourite era/decade -> temporal concept
LIT TIP (Q4): users answer OBJECTIVE features (movie/actor/director) far more reliably than subjective (mood) -> weight
the catalog toward objective recall. (Springer JIIS 2023 movie NL-assistant.)

## Q3 — how to learn to answer? (heuristic vs NDCG-opt) — CRITICAL distinction
TWO components, opposite treatments:
- The ANSWERER (simulator: which entity the user names) = HEURISTIC recall model (Q1), a FIXED defensible assumption,
  ABLATED. DO NOT NDCG-optimise the answerer -> it would name exactly the items that help the recommender = the
  collusion/cheat we already proved (ANSLEARN uncons -> ceiling). Realistic recall is the honest answerer.
- The ASKER (policy: which open question to ask next) = NDCG-OPTIMISED (differentiable unroll, or REINFORCE since the
  answer is a discrete entity sample). This is legit (the system chooses questions; the user answers honestly).
- The GROUNDING (named entity -> belief token) reuses Paper C's FROZEN encoder; free-text -> SBERT -> nearest entity ->
  factor (PEBOL-style NLI/embedding grounding). No inversion.
ABot returns here LEGITIMATELY as the answerer ONLY IF we model real recall (we lack recall data -> heuristic default).

## Q4 — lit tips
- PEBOL (RecSys'24): NLI between user utterance and item descriptions -> Bayesian belief. Use embedding/NLI to ground
  free-text answers. BO acquisition over a shortlist = the discrete baseline to beat.
- GATE (ICLR'25): LLM open-ended question generation; no learned policy.
- 2510.12015 (Oct'25): funnel general->specific, LLM-prompted selection, profile-recon metric (NOT NDCG) -> the
  heuristic-sequencing baseline.
- Movie NL-assistant (JIIS'23): OBJECTIVE features (actor/director) >> subjective; users prefer naming entities.
- Attribute active learning (1805.09023): attribute-driven AL for cold-start.
- GAP nobody fills: a LEARNED policy (continuous or discrete) over open questions, grounded in the recommender, trained
  on NDCG. That is our differentiator.

## Plan of steps (gated, cheap->expensive)
- **P0 Action space + grounding.** Enumerate open-Q types; build entity->factor maps (movie=Q_j; genre/actor/director=
  film-centroid -> NEED ML actor/director metadata; mood=genome). Reuse Paper C frozen encoder. [no training]
- **P1 Answerer (heuristic recall) + ablation.** Implement the Q1 recall models; sanity-check the named-favourite
  popularity distribution looks realistic. [no training]
- **P2 HEADLINE GATE — does open-recall beat rating-elicitation on NDCG?** Fixed simple policy (e.g., ask: 2 favourites,
  1 favourite genre, 1 hated movie, 1 guilty pleasure) -> fold -> NDCG vs Paper B/C (0.36-0.378) on the SAME ruler,
  under the realistic answerer. If open-recall >= D1, the paper has a number; if not, it's a deployability paper. DECISIVE.
- **P3 Learned asker.** Policy over open-Q types, trained on NDCG (unroll/REINFORCE). Baselines: funnel (broad->specific),
  always-favourite, random-type, entropy-over-types. Does learned sequencing beat heuristics?
- **P4 Answer-model sensitivity.** NDCG under info-optimistic / realistic / pessimistic recall (the honest band).
- **P5 LLM realisation (deployable agent).** LLM renders each open-Q as NL + parses free-text answers -> entity ->
  token. Small human-readability / round-trip check. The deployable demo.
- **P6 Interpretability.** The named open-question decision tree (which questions, in what order, for which users).

## Honest risks
- Salience<->info tension: realistic recall is popularity-biased (low tail info) -> open-recall may help FULL not TAIL.
  Mitigation: distinctive/negative questions for the tail; the policy learns to ask them. This is the science.
- Metadata: actor/director questions need ML cast metadata (available for ML; map to film centroids).
- Simulator realism: free-recall is harder to simulate than rating; heuristic + ablation + (later) small LLM-sim check.
- Collusion: never NDCG-optimise the answerer; keep recall a fixed honest model.

## Continuous tie-in (optional)
Paper D can be pure open-recall (discrete question-type policy) OR the Paper-C continuous policy can emit a direction
rendered as a DIRECTED open question ("name something you like that's a bit [eerie]?"). Recommend P2-P4 as pure
open-recall first (cleanest NDCG test); add the continuous-directed-open-question as the bridge to Paper C if P2 passes.

## ALSO (user note) — Paper C needs a fuller baselines table
Paper C tab:main has CASPER-R/entropy/uent+GRAW/popularity. ADD the rest of Paper B's baselines on the SAME ruler
(NICF deep-RL, the blind LLM askers, lit heuristics) for completeness. PEBOL replication = LATER (optional). Track as a
Paper C task.
