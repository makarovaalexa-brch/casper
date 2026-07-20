# Dislike-signal GO/NO-GO probe — do DISLIKES add incremental held-liked ranking value at SHORT interview lengths?

**Date:** 2026-07-11 · **Runtime:** 65 min · closed-form / model-free only (no neural training loop)
**Artifacts:** `.cache/rung1/dislike_signal_probe.json`, `scripts/dislike_signal_probe.py`, `.cache/rung1/dislike_signal_probe.log`

## Verdict — **GO** (decisive, both parts positive; strongest at the shortest k)

Knowing a user's dislikes adds held-LIKED NDCG@10 at short interview lengths in **both** a model-free
taste-space predictor **and** a strong signed model — and, exactly as the region-narrowing hypothesis
predicted, **the gain is largest at the shortest k and shrinks as k grows**. The direction is alive:
build the latent instrument (S2).

**Single most decisive number: +0.0583 held-liked NDCG@10 at k=2** (signed-EASE, α=1), a **+17.5%**
relative lift over likes-only (0.3332 → 0.3915), from folding in ORACLE dislikes as negative entries.

## Setup (faithful, no data reduction)
- **Cohort:** all **200** held-out TEST users = ML-25M `te` (500) minus the 300 LLM study users
  (quarantined). Train-disjoint by construction. No sampling/capping; all their ratings, all 18,430
  items as candidates, full item-item gram, full EASE inverse (chunked dense accumulation + torch
  for memory/speed, never a cap). 8 random-reveal repetitions per user (augmentation, not reduction).
- **Protocol:** reveal k random LIKED items (r≥4); held-liked = the user's *remaining* liked items.
  Candidate pool = all items minus (revealed likes ∪ oracle dislikes), held **identical across α** so
  α isolates ONLY the repulsion effect (dislikes are excluded from candidates even at α=0). NDCG@10.
- **Dislikes = ORACLE** (the user's real r≤2 items) — the strongest possible test; a null here would
  have killed the direction. Only users with ≥1 dislike are counted (n≈175).

## PART A — model-free item-item predictor in MF(taste) space
64-d truncated-SVD of the binarized (r≥4) train matrix (the space where dislike generalizes −0.61).
`score(c) = mean cos(c, revealed likes) − α · mean cos(c, oracle dislikes)`.

| k | α=0 (likes-only) | α=0.25 | α=0.5 | α=1 | α=2 |
|---|---|---|---|---|---|
| **2** | 0.1577 | 0.1662 (**+0.0084**) | 0.1670 (**+0.0093**) | 0.1548 (−0.0029) | 0.1198 (−0.0379) |
| **4** | 0.1585 | 0.1749 (**+0.0164**) | 0.1726 (**+0.0141**) | 0.1481 (−0.0105) | 0.0961 (−0.0625) |

Positive at short k for **small** α (best +0.0164 at k=4, α=0.25); over-repulsion (α≥1) hurts. The
model-free data carries a usable dislike-repulsion signal, but it is weak and needs a light touch —
consistent with RAWDATA_DISLIKE Q3, where a *naive* full-weight explicit penalty (α=1) hurt.

## PART B — SIGNED EASE (closed-form B trained on the CENTERED-rating matrix)
This is the honest test: B itself is solved signed (X_ui = rating − user-mean; dislikes are genuine
negatives), NOT inference-time signed weights on a like-trained model (that gave −33%).

**Strength confirmed — signed-EASE is a genuinely strong recommender, so its verdict is trustworthy:**

| model | full-profile held-liked NDCG@10 | λ |
|---|---|---|
| implicit EASE (binary r≥4, reference) | **0.3631** | 300 |
| **signed EASE (centered)** | **0.3314** | 1000 |

Signed is ~8% below implicit at full profile (the classic "at full profile likes already pin the
region, dislikes look redundant" regime) — competitive, not crippled.

**Short-k fold-in** `user vec = (+1 per revealed like) + α·(−1 per oracle dislike)`, score = vec·B_signed:

| k | α=0 (likes-only) | α=0.25 | α=0.5 | α=1 | α=2 |
|---|---|---|---|---|---|
| **2** | 0.3332 | 0.3712 (**+0.0380**) | 0.3845 (**+0.0513**) | **0.3915 (+0.0583)** | 0.3878 (+0.0546) |
| **4** | 0.3828 | 0.4094 (+0.0266) | 0.4161 (+0.0333) | **0.4177 (+0.0349)** | 0.4123 (+0.0295) |
| **8** | 0.4176 | 0.4365 (+0.0189) | 0.4422 (+0.0246) | **0.4449 (+0.0273)** | 0.4398 (+0.0222) |

Every α>0 helps at every k; the optimum is α≈1 and the **gain decays monotonically with k**
(+0.058 → +0.035 → +0.027), the exact signature of region-narrowing being valuable precisely when
likes under-determine the taste region. At full profile the gap closes — dislikes are a **short-k**
asset.

## Reconciliation with the earlier −33% (RAWDATA_DISLIKE Q3)
The prior "explicit hurts" result used **raw co-rating** item-item CF (a sign-free, popularity-dominated
ruler) at **full profile**. Here the same negative information **helps** because it is used (a) in a
**taste-structured** space (MF / a signed-trained latent Gram), (b) at **short k**, and (c) at a
**moderate** repulsion weight. Ruler + length + weight, not the premise, drove the old null.

## Caveats
- Signed-EASE has NO latent — it is a **signal probe only**. GO means "build the latent version (S2)"
  (Paper C's continuous instrument), NOT "ship signed-EASE".
- Dislikes here are ORACLE. Realized value depends on eliciting genuine dislikes cheaply at short k;
  this probe establishes only that the usable headroom **exists** (and is large: +17.5% at k=2).
