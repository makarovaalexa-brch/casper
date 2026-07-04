# Goodreads EASE Diagnostic — is the tiny personalization headroom domain-intrinsic or an instrument artifact?

**Date:** 2026-07-04. **Question:** Phase 1/1D/1E measured only ~+0.01 encoder−MOSTPOP full-profile
headroom on Goodreads (three consecutive gate failures) and the phase-1E FINAL VERDICT concluded the
domain is "popularity-dominated / intrinsically ~0.01". This diagnostic stress-tests that conclusion
with a strong, standard, closed-form baseline — **EASE (Steck 2019, WWW: "Embarrassingly Shallow
Autoencoders for Sparse Data")**, B = −P/diag(P) with zero diagonal, P=(XᵀX+λI)⁻¹ — on **the same
protocol as our gate** (rng(0) 500 val / 500 test users, rng(123) disjoint held-out like-targets,
full-profile fold = the user's known row as input, asked/profile items excluded from candidates,
NDCG@K primary + NDCG@10).

Two tracks:
- **VARIANT A (explicit, our signal):** X = binarized explicit likes (rating≥4) from `base_comp.npz`
  (composite arena, K=510). Isolates the **model class** (item-item EASE vs 64-d SVD + fold-in encoder)
  holding the signal fixed.
- **VARIANT B (implicit signal):** X = `is_read=True` interactions parsed from the raw interactions
  `.json.gz`. Run on the **mystery slice only** (parsing all three genres' 5.8 GB of gz was out of the
  foreground budget — stated fallback per the task spec), K=142, alongside an explicit-EASE control on
  the same arena so signal vs model are separable.

**Restricted item universe (memory-forced):** XᵀX over the full 188,867-item composite catalogue is a
285 GB dense matrix — infeasible on this 16 GB box. Both variants therefore restrict to the
**top-N = 20,000 items by train-like count** (20k² float64 Gram+inverse ≈ 3.2+3.2 GB, peak RSS ~11 GB,
fits). N=20,000 covers **76.8%** of composite like-mass (90.0% of mystery like-mass). MOSTPOP, ridge,
and the encoder are **re-evaluated on the same restricted universe, same users, same targets** —
a fair triangle. λ sweep {1, 10, 100, 500} selected on the 500-user val cohort; test numbers reported
at the val-selected λ only.

Scripts: `casper/scripts/paper2/gr_ease_composite.py`, `gr_ease_mystery.py`. Eval-only, no encoder
training, no commits.

---

## VARIANT A — composite arena, explicit likes (K=510, universe N=20,000)

λ sweep on val (EASE NDCG@510): 1 → 0.5857, 10 → 0.5870, 100 → 0.5934, **500 → 0.6024 (selected)**.

**TEST (n=418 users, same protocol as the phase-1E gate):**

| model | NDCG@10 | NDCG@510 | headroom over MOSTPOP @510 |
|---|---|---|---|
| MOSTPOP | 0.2045 | 0.2921 | — |
| ridge (λ=5) | 0.1994 | 0.2913 | −0.0009 |
| ENCODER (`enc_v1_grcomp`) | 0.2145 | 0.3052 | **+0.0130** |
| **EASE (λ=500)** | **0.5146** | **0.5958** | **+0.3037** |

- Encoder headroom on the restricted universe (+0.0130 @510, +0.0099 @10) matches the full-catalogue
  gate (+0.0106 @510) — the restriction does not change our instrument's story.
- **EASE, on the SAME explicit signal, same users, same held-out targets, finds +0.3037 NDCG@510
  (+0.3101 @10) of full-profile personalization headroom — ~29× the encoder's.** The domain's
  explicit signal alone carries an enormous personalization prize; our instrument class captured
  ~4% of it.

## VARIANT B — mystery arena, implicit (is_read) vs explicit, 2×2 (K=142, universe N=20,000)

(RESULTS PENDING — filled after `gr_ease_mystery.py` completes.)

---

## VERDICT

(PENDING — see below after Variant B.)
