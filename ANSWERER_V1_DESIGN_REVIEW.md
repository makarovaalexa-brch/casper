# ANSWERER v1 — design review (Fable, 2026-07-08; NOTHING RUNS until author signs off)

Status: DESIGN UNDER REVIEW. The arena-3 judging run was KILLED at 55/300 users (partial cache
kept, resumable/absorbable). Standing rule reinstated (see §6): no experiment defines its own
environment again; the answerer is built ONCE, validated, FROZEN, versioned.

## 0. What we are building (the original §2 plan, completed properly)
ONE artifact: for every study user and every ALLOWED question, the answerer's response:
  - **answerability**: yes/no/maybe + confidence (LLM-judged, known-half profile shown as a sample)
  - **answer value**: the graded answer the user would give — real rating where rated; LLM-predicted
    where answerable-but-unrated (with its measured error as the fidelity σ)
  - (PROPOSED addition, decide below) **vividness**: "seen it / know it well" vs "only know of it"
    — one extra JSON field, ~zero marginal cost, and it is the fidelity dimension the abundance
    regime showed is load-bearing (vague answers dilute vivid ones).
All cached, versioned, judge-pinned. Every downstream experiment consumes THIS and nothing else.

## 1. DECISION A — the allowed-question universe (the cost driver)
Items dominate cost. Measured rate from the main study: ~600 calls / $3.20 with ~270 questions per
call → ≈ $0.002 per 100 (user,question) judgments. Options:
  A1. Top-1,000 items by ratings count × 300 users = 300k cells ≈ **$6–10**. Covers popular +
      upper-moderate band. Risk: thin in the niche band where per-user structure lives.
  A2. Top-3,000 items × 300 users = 900k cells ≈ **$18–30**. Covers down to the genuinely moderate
      band (ML-25M item #3000 ≈ a few thousand ratings — recognizable-to-some territory).
  A3. Stratified ~1,500 (500 popular + 700 moderate + 300 niche-but-not-obscure, genre/decade
      balanced) × 300 ≈ 450k cells ≈ **$9–15**. Cheaper than A2, denser niche coverage than A1,
      but a designed (not natural) universe — document the strata.
  My recommendation: **A2 if budget allows, else A3.** A1 repeats the popular-bank mistake.
  Already in hand and absorbable: the old 160-bank + 1,716-item judged sample + 55 users × 280
  moderate bank (killed run) — the design should REUSE all cached judgments (same judge/split).
Concepts: 200 breadth-tiered — ALREADY DONE (main study), reuse.
Attributes (decades, genres, maybe top actors/directors if external metadata added): ~50–100
questions × 300 users ≈ **<$2** — cheap, currently missing, include.
Pairs: composed from item judgments (both-answerable ∧ co-rated) — **no extra calls**.
Open recall: EXCLUDED from v1 (needs a recall model, separate design per author instruction).

## 2. DECISION B — what the LLM emits per (user, question)
  B1 (minimum, = original §2): can_answer {yes/no/maybe} + conf + predicted rating for items.
  B2 (proposed): B1 + **vividness** {seen-and-remember / know-of-it / unknown}. One field. Gives
      the fidelity axis per-cell; enables fidelity-aware folding later WITHOUT re-judging.
  My recommendation: **B2** — re-judging 900k cells later to add one field would cost the full
      run again; adding it now is ~free.

## 3. DECISION C — value channel policy (unchanged from the frozen design, restated for sign-off)
  - Rated items: value = REAL rating (+ σ=0.70 test-retest noise knob). Ground truth, headline-safe.
  - Answerable-but-unrated: value = LLM-predicted rating, validated by the masked-item protocol
    (predict masked KNOWN ratings; measured MAE ≈ 0.70 ⇒ that error IS the fidelity σ for this
    channel); independent-CF (EASE) cross-check retained. SENSITIVITY-ONLY for headlines.
  - Concepts: data-side member-aggregate primary; LLM-rated fallback where <k rated members (flagged).

## 4. Validation gates (pre-registered; the answerer is not "done" until all pass)
  V1. Judge self-consistency: re-judged overlap cells vs cached grids (≥40 items × sample users);
      report %agree + kappa (Haiku cross-family spot-check on a 30-user subset, ~$0.50).
  V2. Within-tier validity gap on the NEW universe: rated vs never-rated at matched popularity ×
      genre, per stratum, CIs. (The fuel map — tells us WHERE per-user structure lives, per band.)
  V3. Masked-value MAE on the new universe (LLM + EASE) — the fidelity σ, re-anchored.
  V4. Base-rate sanity per stratum (monotone in popularity; no stratum pinned at 0/1).
  V5. Cost reconciliation from usage sidecar (projected vs actual).

## 5. What this enables (and what it retires)
Enables, all against ONE frozen environment: fair statics; the value×answerability probing agents;
coarse→fine over a REAL granularity range (popular→moderate→niche items now all in-universe with
real judged answerability); the two-regime analysis with clean labels; pair channel; fidelity-aware
folding (if B2). Retires: per-experiment banks, per-experiment answerability rules, pmodel-
synthesized environment cells (the circularity breach), post-hoc arena patches.

## 6. Discipline rules (reinstated, standing)
  R1. No experiment runs unless its STATE entry pre-registers: the question, the possible outcomes,
      and what decision changes per outcome. If no decision changes, it does not run.
  R2. One experiment in flight at a time. No mid-flight scope edits (a redesign = a new entry).
  R3. The environment (answerer) is frozen and versioned; experiments may not modify or extend it
      ad hoc — gaps go back through this design process.
  R4. Author signs off on: any environment change, any LLM spend, any headline claim.

## 7. Sign-off needed from the author (then, and only then, the run)
  [ ] DECISION A: universe = A1 / A2 / A3 (my rec: A2, fallback A3)
  [ ] DECISION B: emit = B1 / B2 (my rec: B2, vividness field)
  [ ] DECISION C: value policy as restated (my rec: yes, unchanged)
  [ ] Budget ceiling: proposed hard cap $30, tripwire-checked after 5 users
  [ ] Reuse of the 55-user partial arena-3 cache (my rec: yes, absorb)
