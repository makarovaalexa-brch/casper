# Answerability Study — LOCKED DESIGN + Fable consult questions (2026-07-06)

Status: DESIGN LOCKED, NOT RUN. Full gate/main study awaits Fable consult on the open questions (§7).
Pilot already run (8 users, ML-1M, GPT-5.4-mini): fuel signal present (taste-tracking + popularity sanity).
Arena for v1 = **ML-25M only** (one dataset for starters, per owner). Goodreads later. (Pilot used ML-1M for convenience.)

## 1. PURPOSE
Two jobs, one machinery:
- **Fuel gate:** does LLM-judged answerability vary per-user AND track taste (beyond popularity)? Go/no-go
  on the flagship's ADAPTIVE coarse->granular claim. If flat -> ship the STATIC channel map instead.
- **Main study (if gate passes):** an external (LLM + structural), non-self-authored answer model that
  replaces the circular geometric answerer, so B/C/D headline results are recomputed honestly.

## 2. WHAT THE LLM PRODUCES (both answerability AND value, where we lack ground truth)
Per (user, question): **answerability** {yes/no/maybe + conf} AND, where there is no ground-truth value,
a **predicted answer value** (rating/graded). Answerability is thresholded to BINARY at the interview
decision; conf kept only for fitting.

## 3. ANSWER MODEL PER CHANNEL (non-circular sourcing)
| Channel | Answerability | Answer VALUE (never from the evaluated recommender; never from held-out targets) |
|---|---|---|
| Item — RATED | yes (rated) [+ LLM/structural for counterfactuals] | the real rating + fidelity noise (test-retest σ) |
| Item — UNRATED iconic/popular | LLM judge + structural | **PROBLEM -> §4 solution**: LLM-predicted-from-known-profile (primary) + independent-CF (EASE on known portion, answer-oracle only) cross-check |
| Concept | LLM judge + structural (breadth/coverage) | LLM-rated from known profile (no single ground truth), or data-side aggregate of real ratings over member items — report both |
| Slider/graded | as concept | as concept, graded; the ONLY slider-grade-fidelity channel, labeled optimistic; human study measures real slider fidelity |
| Pair | BOTH items answerable (composes; no separate LLM call) | which of the two the user rated higher = REAL ratings; fidelity = flip-prob from rating gap |
| Open recall | trivially answerable (user names) | named item is a real known-profile item; recall model = popularity-tilted sampling from known ratings (+ framing lever); CACHE per user x framing |
| Abstract continuous | n/a (not askable — that is E's point) | oracle a=cos(u*,q), KEPT ONLY as explicitly-labeled theoretical CEILING. Reported results for continuous come from its REALIZATIONS: pairs (real ratings) + triangulation slider-screens (concept answers). |

## 4. THE UNRATED-ICONIC-ITEM SOLUTION (owner-flagged)
Need answerability AND a guessed rating for popular items the user didn't rate.
- Answerability = LLM + structural, binary.
- Value = **independent of the evaluated recommender**: (primary) LLM predicts the rating from the known
  profile; (cross-check) an independent CF model (EASE/iALS trained on the KNOWN portion only) predicts it.
  Report sensitivity. HARD CONSTRAINTS: value never from the instrument under eval (circularity); never
  uses held-out targets (leak).

## 5. SPLIT / SEED / CACHING (owner corrections applied)
- CLEAN HOLDOUT: answerer sees the KNOWN portion only (profile minus held-out recommendable targets).
  Full-profile-concept-only = PEBOL-style leak -> forbidden.
- ONE FIXED ANSWERER SPLIT: concept AND item answerability are split-dependent (sparse users shift apparent
  engagement) -> run the LLM on ONE canonical split, CACHE, reuse across all recommender-eval seeds.
  (Recommender eval may still use its own seeds for candidate sampling; answerability comes from the fixed split.)
- Judgments are cached (user x question grid) -> one cache serves the whole battery.

## 6. PROMPT DESIGN (realism from world knowledge, not from instruction)
- Rule-free on popularity (do NOT say "famous=answerable" -> would make popularity-monotonicity tautological).
- Frame ratings as a SAMPLE: "Here is a sample of this viewer's ratings; they have seen many more films than
  listed." -> absence != can't-answer; lets the model use world knowledge of iconic films.
- Ask: "Could this viewer give a meaningful answer to X? (not whether they'd like it)."
- Persona-rich: answer as this viewer plausibly would given what a person with this history likely knows.
- JSON out: {q, can_answer, conf, pred_value?}.
- Log response.usage every call -> exact local cost (Costs API needs admin key; do not rely on it).

## 7. OPEN QUESTIONS FOR FABLE CONSULT (settle before full run)
Q-A. Concept answer VALUE: LLM-rated vs data-side aggregate over member items — which as primary, and does
     it change the item-vs-concept headline? (circularity vs definitional-noise tradeoff.)
Q-B. Item answerability at scale: judge-all is infeasible (ML-25M 18k items). Sample-and-FIT a model
     P(answerable | popularity, genre-match, ...) vs restrict item questions to a candidate shortlist? If
     fit — features, sample size, validation. (The fitted model doubles as the structural cross-check.)
Q-C. Unrated-item rating source: LLM-predicted vs independent-CF vs both-with-sensitivity — is LLM-in-the-
     value-channel acceptable, and how to caveat it?
Q-D. Structural answerability model final form: rated OR popularity>τ OR genre-overlap>τ (items);
     breadth>τ OR count-in-concept>=k (concepts) — thresholds, and whether to fit them to the LLM judge or
     set a priori.
Q-E. Fuel-gate statistics: heterogeneity-beyond-popularity test + taste-tracking correlation — exact
     metric, and the pass threshold (r>0.3 proposed but arbitrary; report raw r, or pre-register a cutoff?).
Q-F. Does the LLM gate DE-RISK but not DE-CIRCULARIZE (humans still needed)? Confirm the honest claim
     boundary for the paper.

## 8. COST (measure before full run)
Pilot = 8 calls, cents. Gate ~ (300 users x ~3 batched calls) ~= 900 calls; instrument response.usage to
get the exact per-call number, then multiply. Estimate low single-digit $ on GPT-5.4-mini; CONFIRM by
measurement, not assumption. Main cached grid: judge concepts-all + item-sample once -> low tens of $.

## 9. RUN SCOPE (owner)
v1 = ML-25M only. Gate first (go/no-go). Main study only if gate passes. Goodreads + human study later.
NOT RUN until Fable answers §7.
