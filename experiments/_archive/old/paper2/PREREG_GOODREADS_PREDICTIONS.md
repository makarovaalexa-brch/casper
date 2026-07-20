# PRE-REGISTERED PREDICTIONS — Goodreads Mystery slice, phase 2 (written 2026-07-04, BEFORE any phase-2 run)

Basis: the effective-rank law established on ML-1M and ML-25M (concept-channel saturation ≈ concept
effrank; concept-tail advantage requires non-head-aligned concept geometry; continuous-over-discrete
gain lives off the realizable-direction manifold). Goodreads phase-1 measured a geometry OPPOSITE to
MovieLens: shelf concepts effrank 34.65 vs items 30.48 (ML-25M: genome 3.99 vs items 8.35).

If the rank law is a law and not a MovieLens coincidence, phase 2 must show:

1. **Concept-channel saturation LATE or absent within q8.** Marginal value per answered concept
   question should NOT collapse by ~q4 (as it did at ranks 2.25 and 3.99); with effrank ~35, the
   concept channel should still be adding value at q8.
2. **Concepts-vs-items tail verdict FLIPS back or at least improves vs ML-25M.** High-rank,
   non-genome shelf directions are not structurally head-locked, so the concept channel should not
   pay the ML-25M tail penalty (−0.006/−0.007). Directional prediction: conc−item tail ≥ 0.
3. **The continuous-over-discrete gap SHRINKS relative to MovieLens.** The off-manifold prize
   exists where the nameable menu is low-rank. With a rank-35 concept menu, the best discrete
   selector has far more expressible directions, so the continuous actor's margin over
   discrete-graded should be smaller than ML-25M's +0.024/+0.037 (possibly ≈ 0 on full).
   The graded/binary INVERSION should still hold for whatever adaptive gain remains.
4. **Answerability advantage persists** (already confirmed in phase 1: 0.076 vs 0.0005) and the
   selection hierarchy (entropy ≫ random ≥ pop) persists — these are rank-independent claims.

Failure of (1)-(3) in the stated directions = evidence against the rank law's generality; report
either way, no reinterpretation after the fact. Phase-2 protocol otherwise mirrors ML-25M
(pre-stated fraction-matched K, val-selection, never test-peek).

---

**Note (2026-07-03):** arena = **cold cohort** (test/val users with ≤10 kept ratings), chosen on
gate-visibility grounds **before any policy run** (phase-1D re-gate, `gr_cold_regate.py`). Predictions
above unchanged.


---

**Note (2026-07-04):** No valid Goodreads arena materialized — three consecutive health-gate failures
(single-genre +0.010, cold-cohort +0.017, multi-genre composite +0.011/cold −0.000; answered-concept
tokens ~1/8). Predictions 1-3 remain REGISTERED and UNTESTED cross-domain (not falsified, not confirmed);
the composite effrank (concepts 7.41 < items 12.30, genre axis 33.5% of variance) also revises the
phase-1 single-genre effrank (34.65) downward — folksonomy rank is scope-dependent. Next candidate
arena: Steam. Prediction 4 (answerability + selection persistence) CONFIRMED on Goodreads (~7.6x pool / ~750x all-items).

---

**Note (2026-07-05) — PREDICTIONS TESTED. arena = I2 RecVAE-d512 certified composite.**
The predictions were finally run on a VALID, CERTIFIED arena: the **I2 RecVAE-d512 certified composite**
(P2/P3 CERTIFIED, 0.78x EASE headroom) on the phase-1E split (va=500/te=500), restricted top-20k universe,
NDCG@510. Full battery + numbers in `experiments/instrument2/PHASE4B_GOODREADS.md`. Verdicts (no
reinterpretation):
- **Prediction 1 - CONFIRMED.** Concept-only k-curve marginal still **+0.008 full @q8** (does not collapse
  by q4); concept member-bag effrank (I2 latent) = **15.5**.
- **Prediction 2 - REFUTED.** concept-8 tail - item-8 tail = **-0.084** (a *larger* tail penalty than
  ML-25M's -0.007, opposite to the registered >=0). High concept rank did NOT remove the tail penalty -
  evidence against the rank law's generality for the tail claim.
- **Prediction 3 - CONFIRMED.** Continuous adaptive actor - native discrete item-8 fold: **+0.028 (ML-1M)
  -> -0.060 (GR)** - the margin shrank below the ML reference (+0.024/+0.037) and inverted, tracking the
  item effective rank (ML 33 -> GR **160**). Graded/binary inversion still holds (+0.214).
- **Prediction 4 - CONFIRMED (reconfirmed on the certified arena).** 7.98/8 answered concepts; lift-selected
  concepts 0.371 >> naive global-entropy 0.060 (below floor).
