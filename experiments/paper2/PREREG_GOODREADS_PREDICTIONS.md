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
