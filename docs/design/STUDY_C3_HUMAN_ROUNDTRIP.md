# C3 — human round-trip study design (holdout form; Prolific recruitment)

> The load-bearing form of certification C3 (`GATE_BATTERY_INSTRUMENT.md`). Drafted 2026-07-22; runs LAST
> in the Paper-A sequence (after tower + belief layer + battery). Nothing upstream blocks on it.
> Purpose: the ONLY place a genuinely STATED answer (esp. concept/attribute affinity, which has no
> in-corpus ground truth) meets an OBJECTIVE outcome. Satisfaction-survey designs are rejected.

## Design (holdout round-trip)
1. **Profile acquisition** (before any interview, per participant):
   - Preferred: **imported history** — Letterboxd CSV export (1 click) or IMDb ratings export; require ≥N
     (~40) rated/logged films intersecting the 18,430 catalog.
   - Fallback: **tick-list builder** — 60–80 titles stratified by popularity tier; tick seen, rate ticked.
     Known recall/salience bias → reported, and popularity tier of targets logged.
2. **Holdout:** random half of the profile (stratified by popularity tier) = hidden targets. The interview
   system never sees them.
3. **Interview:** the instrument runs its standard 5–8-question cold interview (closed + concept + open
   free-recall per the deployed policy). Guards: never ask about profile items; anything volunteered that
   overlaps the visible profile folds but is credit-neutrally excluded from targets; refusals allowed and
   logged (they are data — real refusal rates calibrate G4/answerability).
4. **Outcome:** full+tail NDCG@10 of the post-interview ranking against the hidden half. Comparators, same
   participant: (a) popularity-only (floor), (b) random-question interview (the G2 control, now with real
   humans), (c) visible-profile-half fold (ceiling: what knowing their history buys).
5. **Success criterion (pre-registered before launch):** interview arm beats popularity AND the
   random-question control on the held-out half (paired per-participant bootstrap); report the fraction of
   the profile-fold ceiling recovered. Existence claim, NOT effect-size claims (n too small for policy
   comparisons — the simulator keeps that job).

## Recruitment (the "paid service with profile>n prerequisite" = Prolific)
- **Prolific** (prolific.com; UK-based, standard for UK academic studies, ethics-board friendly).
- Stage 1 prescreen (~£0.15–0.25/head, few hundred heads): "Do you have a Letterboxd or IMDb account with
  >40 films rated, and would you export it for a paid study?" + film-consumption frequency.
- Stage 2 main session (qualifiers only): 30–45 min → £8–12/participant at fair UK rates.
- **Budget: n=40 ≈ £400–600 all-in** (+ pilot n=5). Local first-year students remain a free fallback via
  the tick-list arm; the two arms are analyzed separately (import-arm = cleaner targets).
- Ethics: standard information sheet + consent; exported CSVs pseudonymized, film titles only; no
  demographic profiling beyond age-band/film-frequency.

## What it certifies / what it does not
- Certifies: real stated answers (with real coarseness, refusals, intensity) carry recoverable signal
  through the instrument to objectively held-out consumption. Discharges the simulator-only limitation.
- Does NOT certify: effect sizes, policy rankings, population generalization beyond the recruited pool.
- Reported biases: recall/salience skew, head-biased profiles, platform-user selection effect.

## Preconditions before launch
Tower certified (G0 + C1) · belief layer passes per-build battery G1–G9 · C2 answer-model transfer passed ·
interview policy frozen · pre-registration of success criteria + analysis script committed (HARD RULE 10
applies to the analysis code too).
