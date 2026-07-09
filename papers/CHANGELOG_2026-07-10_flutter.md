# CHANGELOG 2026-07-10 — Paper B flutter/equating section (night writing task)

Target file: `casper/papers/paper2_casper/paper2_casper.tex`. Conservative LaTeX edits only.
NO LLM calls, no other files touched. Sources: `experiments/SHUFFLE_PROBE.md`,
`experiments/LLM_DECOMPOSITION.md`, `experiments/DANS_BUILD.md` (v2.1 section),
`experiments/BLIND_VALIDATION.md`.

## EDIT 1 — NEW SUBSECTION `sec:flutter` (inside sec:external, after the coefficient table)
Inserted after Table `tab:extcoef` (end of gate/coefficient material), before `sec:distill`.
Title: "Presentation flutter and the equated trait: how reliable is an LLM judge?"
Contents:
- Shuffle-probe design + finding (know-well rate spread up to 0.66 on identical content: users 0.66/0.54/0.16;
  stars stable MAE 0.12–0.21; within-call ordering ≈0.99). Source SHUFFLE_PROBE.md.
- Score-plus-wobbly-cutoff model (stable latent value, fluctuating per-call decision threshold).
- Equating method paragraph: two-step (per-question-mean removal, then nested variance-components), joint fit
  unidentifiable rationale; cites test-equating/IRT (`kolen04`, `lord80`) and variance components (`searle92`).
  Source DANS_BUILD.md v2.1 Stage A + LLM_DECOMPOSITION.md method.
- Table `tab:flutterdecomp` — variance decomposition, 3 headline channels (54.6/0.6/1.3/4.3/39.3;
  32.0/0.0/2.5/16.2/49.3; 34.3/0.3/4.5/21.7/39.2). Source LLM_DECOMPOSITION.md (b).
- Table `tab:fluttericc` — corrected trait ICC vs old one-way ICC + flutter var, incl. item recognition = 0.000
  ("recognition is not a person dial"). Numbers: concept 0.034→0.043; entity 0.092→0.037; item-knows-well
  0.137→0.073; item-recognition 0.016→0.000. Source LLM_DECOMPOSITION.md (c).
- Two-routes cross-check: equating flutter var 0.046 vs shuffle-probe 0.068, same order of magnitude.
- Scope caveat: 3-user probe + single variance decomposition; formal 30–50-user ×≥3 repeat study flagged as
  pre-registered future work; framed as methods contribution with stated evidence scope.
- General implication: single-read absolute per-subject LLM-judge rates are exposed; within-presentation contrasts
  robust; prefer contrast designs / equate.
GREP: lines 423 (label), 429/444/449, tables at 470/496 — see verification block below.

## EDIT 2 — annotate stale ICC 0.174 (NOT deleted)
`tab:extgate` caption extended: notes the 0.174 heterogeneity ICC is single-read/pre-equating, points to
`sec:flutter` for corrected flutter-free trait ICCs (0.043/0.037/0.073/0.000), states gate passes under both.
GREP: line 381 "single-read, pre-equating".

## EDIT 3 — scope banner "(173/300-user grid; frozen-grid re-run pending)"
Added once as a parenthetical at the head of `sec:flutter` (line 427), worded to cover the grid-derived
decomposition/ICC quantities of the section: "(All quantities in this subsection derive from the 173/300-user
answerer grid; a frozen-grid re-run is pending and these numbers are directional.)" Source BLIND_VALIDATION.md
(unfrozen partial 173/300 grid banner).

## EDIT 4 — sec:distill placeholder replaced with real importance-table numbers
Removed the `% TABLE PLACEHOLDER` comments + `[Placeholder: ...]` bracket. Added:
- Prose citing top LOUO features: era spread, median obscurity, pre-1970 depth, catalogue coverage, franchise share.
- Table `tab:distillfeat` (top-5 features × 3 channels, LOUO fold-mean beta). Source DANS_BUILD.md STANDALONE STUDY.
- Blunt census-CV statement: concept CV R² ≈32% generalises; item ≈3% (weak); attribute ≈−28% (no out-of-sample
  signal, overfits). Source LLM_DECOMPOSITION.md (a) + DANS_BUILD.md DECOMPOSITION.

## BIB additions (manual thebibliography)
`kolen04` (Kolen & Brennan, Test Equating 2004), `lord80` (Lord, IRT 1980), `searle92` (Searle et al., Variance
Components 1992). Lines 984–986.

## COMPILE-SAFETY VERIFICATION
- `\begin{table}`/`\end{table}` = 12/12; `\begin{tabular}`/`\end{tabular}` = 12/12 (balanced).
- Placeholder count 0; `\todo` count 0.
- All new `\label`s (`sec:flutter`, `tab:flutterdecomp`, `tab:fluttericc`, `tab:distillfeat`) defined and `\ref`'d.
- All new `\cite`s (`kolen04`, `lord80`, `searle92`) have matching `\bibitem`s.
- Column counts match preambles: flutterdecomp 6 (lrrrrr), fluttericc 5 (llrrr), distillfeat 4 (lrrr).

## NUMBERS NOT SOURCED / FLAGGED
None invented. Every figure traces to SHUFFLE_PROBE.md, LLM_DECOMPOSITION.md, or DANS_BUILD.md as noted above.
No placeholders left in the tex.
