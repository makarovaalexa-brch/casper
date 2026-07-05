# HANDOFF — Response plan to the adversarial review of 2026-07-05 (for Opus execution)

Source: C:\dev\phd\REVIEW_ADVERSARIAL_2026-07-05.md. Canonical papers: C:\dev\phd\papers\ (sync casper/papers after edits).
NO WORK HAS BEEN STARTED. Execute top-down; each item states DO / WHY / KIND (text|experiment|study) / SIZE.

## Triage verdict (read first)
The review is largely CORRECT on framing and two comparisons, and largely ALREADY-ANSWERED on machinery we built
this week but haven't fully propagated into the papers' framing (P4c fidelity boundary, foreign-geometry
non-circularity, raw-data pairs, prereg). The core response is: (1) one human study, (2) reframe C around the
fidelity boundary as THE contribution, (3) fix D's asymmetric comparison, (4) statistics hygiene everywhere.
It is a framing+validation problem, as the reviewer says — not a results problem.

---

## T1 — THESIS-LEVEL DEFENSE POSITION (text, small, do first — everything else aligns to it)
DO: write a 1-page "measurement position" used by all papers + the viva: contributions are
measurement-theoretic/mechanistic UNDER A STATED ANSWER MODEL; the answer-source ablation (P4C_ANSWER_SOURCES.md:
empirical channel fitted on 1M real ratings, foreign-geometry V1/EASE answers, zero-circularity raw-rating pairs)
bounds the circularity; the human study (T3) closes it. Insert a condensed version into every paper's
methodology/limitations. WHY: review's "one flaw that threatens the whole thesis" — establish the answer ONCE, on
our terms. NOTE: the reviewer's circularity charge is partially pre-answered by P4c (foreign-geometry results
survive; the CONTINUOUS premium indeed does not) — say so explicitly.

## T2 — PAPER C REFRAME (text, medium; paper3_casper.tex)
The reviewer is right that the current title/abstract lead with the regime that P4c refutes realistically.
DO: (a) restructure so the ANSWER-FIDELITY BOUNDARY is the headline contribution: "answer fidelity, not
action-space continuity, governs elicitation; continuity wins iff slider/comparison-grade answers" — the
noiseless battery becomes the mechanism study under a stated oracle channel (label every noiseless table as such);
(b) retitle or subtitle accordingly (keep 'The Best Question Has No Name' only if recast as the un-askability/
fidelity tension — suggest subtitle swap: '...: Answer Fidelity and the Limits of Continuous Preference
Elicitation'); (c) snap-loss presented as regime-dependent (sign flips under noise = a FINDING about
snapping-as-denoising, already in §5.6 — promote it); (d) pair rung: parity language only, CI shown, no
deployability-win claim; (e) demote the RecVAE 'sharpening' phrasing (reviewer: more linear instrument = more
exact circularity — concede in one sentence, the fidelity section carries the honest story).

## T3 — HUMAN STUDY PACKAGE (study prep, medium; USER runs it, Opus prepares everything)
DO: produce a complete run-ready kit: protocol (n≈50; each participant: 8 rendered questions incl. k=3 slider
screens + 4 pair comparisons + 2 open-recall + 2 concept y/n; collect graded answers; fold through I2; downstream
NDCG vs their held ratings — participants must be people with letterboxd/MAL/imdb history or rate 20 seed films
first), materials (question renders from existing artifacts), consent text, power analysis (detectable effect at
n=50), analysis script skeleton, and a one-page runsheet. WHY: review point #1 — highest leverage, defuses
circularity across A–E. SIZE: 1-2 days prep.

## T4 — RESURRECT THE LEARNED ANSWERER (experiment, small-medium)
DO: the removed bot-play/ABot learned answerer (code exists: REFUSAL/ABOT machinery, continuous_actor.py; see
memory 'bot-play removed') is the one NON-ORACLE answer model we own. Run it as a 4th answer source in the P4c
framework (clean + trained-answerer channels; report where the continuous/discrete ordering lands). WHY: review
explicitly flags its removal as suppressed evidence (C L1245 charge). Either it supports the fidelity story or
further bounds it — both fine.

## T5 — PAPER B FIXES (text small + experiment medium; paper2_casper.tex)
(a) TEXT: purge the residual §5 learned-win presentation (+0.012 tail headline) — align with §6 saturation and
the trainseed overturn; CASPER-R = 'matches the heuristic' everywhere; add CI/paired-bootstrap to every table
that still lacks them. (b) EXPERIMENT (review point #5): replace 'answerable iff rated' with a
popularity-conditioned seen-probability model (P(seen|popularity) calibrated on ML-1M co-occurrence or external
watch data; sample answerability from it), re-run the item-vs-concept comparison; report both regimes honestly —
the reviewer predicts the headline narrows; find out. (c) The 4-dataset answerability replication and
concept-necessity remain B's headline — the density-definition charge is partially answered by the LLM-answerer
refusal agreement (83.6%, LLMUSER_PHASE1) — cite it where the definition is introduced.

## T6 — PAPER D FAIR-COMPARISON ARMS (experiment medium + text; paper4_casper.tex)
DO: (a) access-controlled comparison: give every probe baseline k volunteered known-item tokens (matched to what
open-recall extracts) + their probes — isolates selection value from access value; (b) promote the PEBOL+profile
arm (+0.052/+0.076, already run) to the primary baseline table; (c) headline = realistic answerer (popweight)
with its tail loss stated, distinct/info-favourable labeled ORACLE-RECALL upper bounds only; (d) CIs on random5
and all headline rows (3→5 seeds where cheap); (e) name→factor resolution risk stays prominent. WHY: review's
'rigged asymmetry' — the mechanism (volunteering known items) is legitimate but must be separated from privileged
ACCESS in the comparison design.

## T7 — PAPER A FIXES (experiment small + text; paper1_casper/casper_u_chapter.tex)
(a) EXPERIMENT: the missing fair cold comparison — EASE also folds partial profiles; compute EASE's k=1..8
cold-start curve on the canonical arena vs I2's (and V1's). If I2 ≥ EASE cold, the 'warm-start-only tie' charge
dies; report either way. (b) TEXT: scope §12.4 (probes-beat-items) as a geometric upper bound under the oracle
channel + add its empirical-channel counterpart from P4c; reword G4 (drop anything reading as algebraic identity
with WRMF); cross-ref the realizable-EIG evidence properly (it lives in B/C — add pointers, not claims).

## T8 — STATISTICS + CONSISTENCY PASS, ALL PAPERS (text+scripts, medium)
(a) Multiplicity: label every headline claim PREREGISTERED (link the prereg files: PATH1, GOODREADS predictions,
TWOPHASE_E2) vs EXPLORATORY; apply Holm/FDR within each paper's headline family; state the policy in methods.
(b) Metric depth: state the catalogue-fraction rule (K=round(0.0027·N)) ONCE per paper as pre-stated, uniform.
(c) Training-seed variance + paired significance on every headline number (B +0.012 → expected to die: say so;
D random5; E +0.003 agent table → label within-noise). (d) Linear-Gaussian theory: state predictions before
results in each use; cite the prereg where the prediction was registered (Goodreads P3 = confirmed out-of-sample
— our best defense against 'post-hoc theory-fitting').
(e) E abstract: fix stale −0.008 → canonical −0.010/−0.012 (reviewer caught it; flagged in E L54 charge).

## T9 — PAPER E DISPOSITION (text small)
Per the standing plan E merges with D for publication; the review agrees ('one honest negative-result section').
DO: fix the abstract numbers (T8e), add an LLM-judge evaluation of axis-synthesis renders (n≥100, faithfulness/
askability rubric — cheap, kills the 'zero evaluation' charge), keep E as thesis chapter, confirm the merged
D+E publication carries the constructive arc (un-askable → one-screen/pairs → open recall).

## Order & effort
T1 (day 0, text) → T2+T8 (days 1-2, text) → T4+T7a (day 2-3, small experiments) → T5b+T6a (days 3-5, medium
experiments) → T3 kit (parallel, text) → T9. Total ≈ 5-6 working days of Opus + the user-run human study.
Everything commits+pushes per house rules; canonical papers = outer repo; sync casper/papers after.
