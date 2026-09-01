# C6 — the principal disagreement axis is not where the measured gains point

**Status: COMPLETE (train/validation-side; TEST untouched). 2026-09-01.** Code
`src/instrument/c6_eigen_emitter.py` (`3787ad6`→`9ec8abc`), artifacts `experiments/paperC/c6_eigen.json`
(`393054e`). Conclusions Fable's own. Re-scoped mid-run from *candidate method* to **static control**
after the literature pass (`PAPER_C_IMPROVEMENT_ANALYSIS_2026-09-01.md` §7).

**One-line verdict: NO. The measured-best direction does not lie near the cohort's principal
disagreement axis — median |cos| 0.167 [0.132, 0.218], which loses decisively to the state-blind
constant (0.414) and merely ties the cohort MEAN direction the belief state already encodes. The
hypothesis does not dissolve the generalisation problem.**

## The nulls decide it

| comparison | median | paired | reading |
|---|---|---|---|
| v1 vs random direction | 0.0386 | **+0.115 [+0.085, +0.160]**, 56/72 | v1 wins |
| v1 vs the cohort **mean** direction | 0.1694 | −0.022 [−0.059, +0.010] | **tie — the axis adds nothing over the mean already in the belief state** |
| v1 vs the state-blind LOO constant | **0.4142** | **−0.202 [−0.270, −0.167]**, 13/72 | v1 loses decisively |
| v1 vs a random OTHER state's target | 0.1761 | +0.010 [−0.041, +0.061] | **null — v1 is no closer to its own state's target than to another's** |

Not an estimation artefact: v1's split-half reproducibility is **0.977** (profile) / **0.996**
(belief). The axis is precisely estimated; it simply is not the target.

## What survives, and it is worth keeping

1. **The covariance carries signal; the top axis does not.** As a *selector* over the measured
   candidates, v1 loses to a popularity prior (−0.205 [−0.272, −0.145]). But the full quadratic form
   weighted by answerability — `eᵀΣe × P(answer)` — **beats popularity by +0.0270 [+0.0159,
   +0.0390]** (Spearman 0.251 vs 0.228). A closed-form, training-free selector that clears the
   popularity floor is a genuine if modest positive, and it is the honest form of the hypothesis.
2. **The target does sit in a high-variance direction, just not the top one**: variance ratio along
   the target **5.73× isotropic [5.35, 7.07]** against v1's own ceiling of 41.79. The intuition was
   directionally right and quantitatively wrong.
3. **Span-3 capture 0.129 [0.084, 0.183]** against 0.011 for three random directions — 12× chance,
   and yet 87% of the target lies outside the span.

## ★ The decomposition confirms the theory exactly

Splitting each target into the state-blind common direction and its orthogonal state-specific
residual, and asking which one v1 tracks:

| cell | \|cos(v1, **common**)\| | \|cos(v1, **specific**)\| | paired |
|---|---|---|---|
| belief arm, all 72 | **0.369** | 0.129 | **+0.191 [+0.101, +0.251]** |
| belief arm, trained 57 | **0.393** | 0.098 | **+0.234 [+0.143, +0.298]** |
| profile arm, held-out 15 | **0.510** | 0.165 | **+0.348 [+0.264, +0.407]** |

**Where v1 correlates with the target at all, it tracks the STATIC COMMON component, never the
adaptive residual.** That is precisely what the theory predicts — a covariance-functional emitter
cannot be adaptive, because the covariance update carries no dependence on the answer — and it is
the reason this arm is a control rather than a method. The measurement and the theory agree.

## The answerability gap, numbered

Info-Synth derives the information cost of snapping and never reports it. Ours, at the 72 states:

- v1 → nearest **answerable** pool entry: **0.411**; → nearest entry in the full 19,390 bank: 0.502.
- the **target** → its top-1 measured candidate: **0.730**; → its nearest pool entry: **0.819**.
- v1's nearest pool entry is the top-1 measured candidate at **0 of 72** states.

⚠ The gap statistic as literally specified (+0.026 [+0.0002, +0.042]) is small **for the wrong
reason**: v1 is far from *everything* (0.41 to its own nearest answerable entry) while the target
sits at 0.82. Where the mismatch is this large the second-order surrogate `ΔI = ½‖Δz‖²_M` is not the
right quantity and the true MI difference is. **The gap is worth reporting for the TARGET, not for
v1** — that is the version the paper should carry.

## Provenance and one caught defect

Canonical TRAIN snap exact · 72/72 sweeps `argmax(gain)==splitter`, error 0 · blend provenance in two
parts as ruled (EXACT on ingredient identity and weights; TOLERANCE on the assembled vector,
max|diff| 4.77e-07, worst 1−cos 2.2e-16, both banked) · all 144 nodes matching banked `n_users` and
`child_sizes`, 0 users falling off · HARD RULE 10 import-scoped, 0 dirty · nothing trained.

**A third run was spent fixing a real defect the agent found in its own reporting**: the verdict block
named contrasts `beats_x` while testing only `excludes_zero`, which would have reported the two
*losing* contrasts as wins. Caught before any number was read.

**Declared choice:** cohorts are the tree's own fit cohorts (the 10,000 TRAIN users), a stated
departure from C1-final's bank-cohort resolution, because the gains were measured on that cohort
(sweep `n_rows` == node `n_users` at 72/72).
