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

## 1-REVISED (2026-07-08, after author discussion) — THE QUESTION UNIVERSE, mapped channel by channel

Principle (author): the universe = questions a human interviewer could plausibly ask a human about
movies. Locked once, used thesis-wide.

**Dataset:** ML-25M (ratings through Nov 2019 — the recent standard; user base is 2019-era, not
ML-1M's 2000-era). Item universe includes classics (humans know classics); recency concern applies
to the RATER population, which 25M satisfies. Note: ML-32M (2023) exists as a later recency port;
everything (pinned split, grids, RecVAE, EASE, genome) is built on 25M — stay, port later
(Goodreads for domain transfer; optionally ML-32M for recency).

**ITEMS** — probing allowed down to a recognition floor, not into obscurity:
  Universe = top-N items by ratings count. N=3,000 reaches the genuinely moderate band
  (recognizable-to-some; the coarse→fine descent range); N=2,000 is the budget fallback.
  Judged for ALL 300 study users (full grid = any policy runnable, frozen, reproducible).
**CONCEPTS** — genome tags only in v1 (validated membership+relevance = data-side values work).
  The 200 breadth-tiered bank is already judged (reuse); option: extend to all 1,128 tags (+~$3).
  Mined-from-reviews concepts: ML-25M has NO review text — would need external corpora + a new
  validation pipeline. Deferred to v2, flagged as the author's idea worth keeping.
**ATTRIBUTES** — actors, directors, composers (author-requested, currently missing):
  ML-25M carries only genres+year → external join REQUIRED: links.csv → IMDb offline TSV dumps
  (free, no API, reproducible): directors (title.crew), top-billed cast (title.principals),
  composers (title.principals category 'composer'). Universe = entities with enough presence to be
  askable: directors ≥3 films in the item universe, actors ≥5, composers ≥3 → judged battery ~200
  attribute questions (stratified famous→niche). Value = data-side aggregate of the user's ratings
  over the entity's films (as concepts). PREREQUISITE BUILD: the membership map (one script, no LLM).
**PAIRS** — composed from item judgments, zero extra calls. Askable iff both items answerable;
  answer = which rated higher (real where both rated; LLM-compared where not, dual-answerer rule).
**OPEN RECALL** — a fixed menu, each framing usable ONCE per interview (no identical repeats):
  favourite ×1, hidden-gem ×1, and (author to decide) optional "a movie you hated" ×1 — negative
  evidence, humans ask it. Environment: v1 = the validated Paper-D sampler (popularity-tilted from
  known set, framing lever), LLM recall model as v1.1 (+~$2). Out of the probing arena; in the
  answerer artifact.

## 1b. DECISION D (NEW, locks the whole thesis) — THE ANSWER SCALE
What a human actually emits (and therefore what the judge emits, what the fold consumes, what every
experiment uses). Options:
  D1 binary (yes/no) — wastes signal (real answers are graded);
  D2 five-star mimicry (3.5 stars) — humans don't talk like this in interviews;
  D3 (RECOMMENDED) **4-level sentiment + explicit can't-answer + vividness hedge**:
     value ∈ {hated / meh / liked / loved} (centered −1, −⅓, +⅓, +1 for the fold);
     can_answer ∈ {yes / no};
     vividness ∈ {strong (seen it, remember well) / vague (know of it / barely remember)} —
     observable hedging, like a real human ("I think I liked it?"), and the fidelity weight for
     the fold. Raw LLM confidence floats are kept for analysis but NOT exposed to the agent.
  Mapping of real ratings to the scale (for rated items): ≥4.5 loved, 3.5–4 liked, 2.5–3 meh,
  ≤2 hated (documented, fixed). The recommender/fold interface consumes exactly:
  (channel-type, entity, value∈4-level, vividness-weight). LOCKED THESIS-WIDE once signed.

## 1c. DECISION E (NEW) — retire "sensitivity-only"; adopt DUAL-ANSWERER, canonical answers
Author's question ("we do not lock llm rating answers as responses?") — resolved as follows.
Old framing: LLM-predicted values for answerable-but-unrated items were quarantined to side
analyses ("sensitivity-only") because they are model guesses. NEW framing (cleaner, and matches
deployability): a real human answers EVERYTHING they can answer — a simulated user that answers
only where a rating exists is the dishonest one. Therefore:
  - **Canonical answer** = real rating (mapped to the scale) where rated; **the LLM's graded answer
    where answerable-but-unrated** — the LLM IS the simulated human there. Its licence: masked-item
    validation MAE ≈ 0.70 stars ≈ human test-retest inconsistency (Amatriain) — the stand-in errs
    about as much as humans disagree with themselves.
  - **Dual-answerer robustness (pre-registered, replaces 'sensitivity-only'):** every headline runs
    under TWO answerers — Arena-F (full: LLM-valued canonical, PRIMARY) and Arena-R (rated-only:
    hard ground-truth lower bound) — claims stated where both agree; disagreements reported as
    scope. The independent-CF (EASE) value swap remains a one-time robustness table.
  - Human study (later) validates the judge; until then every paper prints the claim boundary.

## 1-ORIGINAL (superseded by 1-REVISED for the universe; kept for the cost table) —
## DECISION A — the allowed-question universe (the cost driver)
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

## 7. Sign-off needed from the author (then, and only then, the run) — REVISED LIST
  [ ] DECISION A (universe size): items top-3,000 (rec) or top-2,000 (budget fallback);
      concepts = 200 judged bank (rec) or extend to all 1,128 (+$3);
      attributes ~200 questions after the IMDb join (rec: yes);
      open-recall menu: favourite+hidden-gem only, or + "hated" (author call).
  [ ] DECISION D (answer scale, locks thesis-wide): D3 4-level + can't-answer + vividness (rec).
  [ ] DECISION E (answers): canonical = real-where-rated, LLM-where-answerable-unrated;
      dual-answerer (Arena-F primary / Arena-R lower bound) replaces "sensitivity-only" (rec: yes).
  [ ] Budget: full est. $25–45 at N=3,000 (items dominate; scale-linear). Hard cap proposal $45,
      tripwire after 5 users (abort if projected > cap). N=2,000 brings est. to ~$18–30.
  [ ] Reuse: absorb all cached judgments incl. the 55-user arena-3 partial (rec: yes).
  [ ] PREREQUISITE (no LLM cost, needed before the run): IMDb TSV join → attribute membership map;
      item-universe list + strata; question templates per channel; the locked scale in one schema
      file that the judge prompt, the cache format, and the fold interface all import.
