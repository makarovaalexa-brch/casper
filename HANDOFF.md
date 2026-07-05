# HANDOFF — single source of truth (2026-07-05, consolidates + replaces HANDOFF_I2.md and HANDOFF_REVIEW2_OPUS.md)

Canonical papers: C:\dev\phd\papers\ (edit here; sync casper/papers after, commit+push both repos).
Backups: GitHub casper repo (all code/results/papers), tag pre-instrument-rebuild-2026-07-04, zips C:\dev\phd_BACKUP_2026-07-04.

## 1. THESIS POSITION after the 2026-07-05 adversarial review (the framing that governs all work)

The review (C:\dev\phd\REVIEW_ADVERSARIAL_2026-07-05.md) is largely RIGHT on framing, largely PRE-ANSWERED on
machinery (P4c answer-source ablation; prereg record) — the papers' framing must catch up with what the
experiments already showed. Reframed thesis = ANALYSIS PAPER WITH A LAW, not method-paper-with-a-hero:

- **The fidelity boundary is the discovery**: elicitation value is governed by answer fidelity, not
  action-space continuity; continuous wins iff slider/comparison-grade answers; under the empirically
  fitted human-like channel, answerable discrete + graded real answers + open recall win. Nobody else
  fits their answer channel on real data — that is the novelty.
- **What survives every answer model** (non-circular, foreign-geometry-proof): answerability (4 datasets),
  concept-necessity at scale, open-recall bandwidth, snapping-as-denoising, rank relationships with a
  pre-registered out-of-sample confirmation (Goodreads P3), the measurement layer (protocol distortion,
  instrument-class visibility, gate suite, EASE-tied instrument).
- **Not "tried and failed"**: we have WHY (noise decomposition), WHERE (the boundary), WHAT WINS INSTEAD,
  and HOW TO MEASURE — the failed version of this PhD is the two-weeks-ago one that knew none of this.
  Viva line: "the field believed continuous elicitation worked because everyone graded it with the same
  oracle; we built the honest grader, found the boundary, and mapped both sides."
- Genre/venues: analysis/reproducibility-class papers (Dacrema precedent). A → reproducibility track;
  C → the fidelity-boundary paper; D → fair-comparison bandwidth paper; B → chapter + material in A/C;
  E → merged into D for publication.

## 2. HUMAN STUDY — value either way (cannot sink the REFRAMED thesis)

Under the old framing the study was sink-risk; under the reframe every outcome is publishable:
(a) humans give decent graded fidelity on sliders/pairs → continuous premium partially real in deployment
→ upgrades C; (b) humans give poor fidelity → the realistic-regime side confirmed, boundary paper MORE
relevant, discrete+open-recall recommendation validated; (c) intermediate → places real interfaces on the
boundary map = completes the headline figure. Residual risks, contained: framing-lever fails on real
memories (demotes one D claim to LLM-validated; D has other legs); catastrophic incoherence of all human
answers (would contradict the LLM-study calibration 71.5→85% + 83.6% answerability agreement; if it
happened it is a field-level finding, not a silent failure). Design accordingly: the study MEASURES
fidelity, it does not defend a claim.

## 3. REVIEW-RESPONSE TASKS T1–T9 (execute top-down; no work started)

- **T1** Thesis-level "measurement position" page; condensed into every paper. (text, small, FIRST)
- **T2** Paper C reframe: fidelity boundary = headline; noiseless battery = mechanism study under stated
  oracle channel (label every noiseless table); snap-loss = regime-dependent finding (denoising); pair rung
  = parity + CI only; demote RecVAE-"sharpening" phrasing; retitle/subtitle ("...: Answer Fidelity and the
  Limits of Continuous Preference Elicitation"). (text, medium)
- **T3** Human-study kit for the user: protocol n≈50 (8 rendered Qs: k=3 sliders, 4 pairs, 2 open-recall,
  2 concept y/n; fold via I2; NDCG vs held ratings), materials from existing render artifacts, consent,
  power analysis, analysis script, runsheet. (prep, medium)
- **T4** Resurrect the removed bot-play/ABot learned answerer as a 4th answer source in the P4c framework
  (answers the "suppressed evidence" charge either way). (experiment, small-medium)
- **T5** Paper B: (a) purge residual §5 learned-win (+0.012) presentation — align with §6 + trainseed
  overturn; CIs on every table; (b) popularity-conditioned seen-probability answerability model, re-run
  item-vs-concept, report both regimes; (c) cite LLM answerability agreement (83.6%) at the definition.
  (text small + experiment medium)
- **T6** Paper D fair comparison: access-controlled arms (baselines get matched volunteered known-item
  tokens), PEBOL+profile as primary baseline, oracle-recall rows labeled bounds, realistic answerer as
  headline (tail loss stated), CIs everywhere. (experiment medium + text)
- **T7** Paper A: (a) EASE cold-start k=1..8 curve vs I2 (kills or confirms the "warm-start-only tie"
  charge); (b) scope §12.4 as geometric upper bound + add its empirical-channel counterpart; reword G4;
  cross-ref realizable-EIG to B/C. (experiment small + text)
- **T8** All papers stats pass: PREREGISTERED vs EXPLORATORY labels (link PATH1/GOODREADS/TWOPHASE_E2
  preregs), Holm/FDR within headline families, catalogue-fraction K rule stated once per paper,
  training-seed variance + paired significance on every headline (expect B +0.012, D +0.007, E +0.003 to
  die — say so); state linear-Gaussian predictions BEFORE results; fix E abstract −0.008 → −0.010/−0.012.
- **T9** Paper E: LLM-judge of axis-synthesis renders (n≥100, faithfulness/askability); merged-D+E
  publication carries the constructive arc; E stays a thesis chapter.
Order: T1 → T2+T8 → T4+T7a → T5b+T6a → T3 (parallel) → T9. ≈5–6 Opus days + user-run study.

## 4. OTHER REMAINING WORK (pre-review backlog, still valid)

- ML-25M I2 port (deferred core-cut item; conventions in experiments/instrument2/PHASE2_PORTS.md).
- Squeeze R2-4 pivoted (task #49): noise-robust policies (train-noisy actors, repeat-top-axis probing,
  decoder-metric belief, fix Kalman magnitude shrinkage) + non-peeking decoder-metric EIG selector toward
  the 0.755/0.878 ceilings (SQUEEZE_R01.md).
- Compile passes (deferred; before ECIR).
- Papers-dir consolidation when file lock clears (retire outer copy OR make it canonical permanently —
  decide; currently outer=canonical, casper/papers=synced backup).
- D+E journal merge (post-ECIR). ECIR assembly deadline 2026-10-02 (C+A). User: 1-2 replication datasets.
- User-owned: human study (T3 kit), supervisor sign-off, Overleaf compile.

## 5. STATE SNAPSHOT (what is DONE and where)

I2 arc complete (Jul 4-5): RecVAE-d512 instrument (ties EASE ML-1M 0.554; Goodreads +0.237=78% of EASE,
21.6x V1), 16/16 gates, additive operator, member-bag concepts, P4a flagship battery (noiseless: all claims
sharpen; snap-loss 4x), P4b cross-domain (prereg 3/4 confirmed), P4c fidelity boundary (the pivot finding),
squeeze R0-1 (actor = realizable myopic ceiling clean; Bayesian repeat-probing wins noisy +0.095).
Papers A-E updated through all of the above incl. sexiness pass (titles, worked-example box, figures).
Key docs: casper/experiments/instrument2/*.md, PHASE4A/4B/P4C/SQUEEZE_R01, INSTRUMENT2_PLAN.md,
REVIEW_HARSH_2026-07-01.md + REVIEW_RESPONSE (first review, fully addressed), PUBLICATION_PLAN_2026-07-01.md.
