# Goodreads Dimension-Sweep — is the ~0 MF headroom a dimensionality limit, not an MF limit?

**Date:** 2026-07-04. **Question:** The EASE diagnostic (`GOODREADS_EASE_DIAGNOSTIC.md`) showed that on
the composite arena our d=64 biased-SVD + ridge-fold instrument has ~0 full-profile headroom over MOSTPOP
(−0.0009 @510) while item-item EASE (λ=500) finds +0.3037. That contrasts *model class* (item-item vs
64-d latent-factor) but confounds it with *capacity*: 64 factors may simply be too few to carry
Goodreads' personalization signal. This sweep isolates capacity by retraining the **same biased-SVD recipe**
at d ∈ {64, 128, 256, 512} and measuring ridge-fold headroom on the **identical protocol/universe** as the
EASE diagnostic.

## Config (identical to EASE diagnostic unless noted)

- **Model:** biased-SVD (SGD, `mu + b_u + b_i + p_u·q_i`), recipe fixed at `LAMF=0.05, LR=0.01, EP=15,
  batch=16384, rng(0)` — *only D varies*. Scripts: `scripts/paper2/gr_build_svd_comp_dsweep.py` (build,
  epoch-checkpointed), `scripts/paper2/gr_dsweep_eval.py` (eval). Item factors saved durably to
  `.cache/goodreads/Q_svd_comp_d{D}.npy` (+ `bi_…`, `…_peak.txt`).
- **Arena:** composite, `base_comp.npz`, ni=188,867 items, nu=768,746 users, 46.1M train ratings, mu=3.972.
- **Protocol:** rng(0) 500 val / 500 test users; rng(123) disjoint held-out like-targets (rating≥4, ≥4 likes,
  second half held out); full-profile fold-in (known row minus targets = ridge input); asked/profile items
  excluded from candidates; NDCG@510 primary + NDCG@10.
- **Restricted universe:** top-N=20,000 items by train-like count (76.8% of composite like-mass), identical to
  the EASE diagnostic; MOSTPOP, ridge all scored on this same universe/users/targets.
- **Instrument:** ridge fold-in `u = (FᵀF + λI)⁻¹ Fᵀy`, F = item factors of profile items, y = mean-centred
  residual `r − mu − b_i`; score `log(cnt+1) + q·u` (popularity prior + latent term, same as encoder path).
- **Ridge λ sweep:** {1, 5, 25} on the val cohort per d (optimum may shift with D); test at val-selected λ.
- **Harness fidelity check:** the d=64 rerun of `gr_dsweep_eval.py` reproduces the EASE-diagnostic numbers
  exactly — MOSTPOP @10=0.2045 / @510=0.2921, ridge @510=0.2913, headroom −0.0009 @510. Harness validated.

## Headroom-vs-Dimension curve

| d | train RMSE | val-sel λ | headroom @510 | headroom @10 | MOSTPOP @510 | ridge @510 |
|---|---|---|---|---|---|---|
| 64  | (fill) | 5  | −0.0009 | −0.0051 | 0.2921 | 0.2913 |
| 128 | (fill) | (fill) | (fill) | (fill) | 0.2921 | (fill) |
| 256 | (fill) | (fill) | (fill) | (fill) | 0.2921 | (fill) |
| 512 | (fill) | (fill) | (fill) | (fill) | 0.2921 | (fill) |
| **EASE (λ=500)** anchor | — | — | **+0.3037** | +0.3101 | 0.2921 | 0.5958 |
| **MOSTPOP** anchor | — | — | 0.0 (def) | 0.0 | 0.2921 | — |

Phase-1E gate bar = **+0.025** headroom @510.

## Verdicts

(PENDING — filled after the sweep completes.)
