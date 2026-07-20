# DESIGN SHEET — V4 additive concept re-rank head (2026-07-17)  [AWAITING AUTHOR SIGNATURE]

## Question
Does a LEARNED additive re-rank head on the frozen recommender beat the hand-set operator's
+41% cold-concept TAIL (β≈120: TAIL 0.0440→0.0619, FULL +0.0023) — and never drag FULL below intercept?

## Why this, why now (established tonight, 3000 val users, cold, graded)
- Belief-token fold = WRONG operator (blurs off-manifold; −0.037 full even with oracle concept).
- Paper-B operator `score = popb + <Wd[i], u>` (separate additive popularity FLOOR) + WHITENED direction
  + IDF weight + per-user norm → FULL never drops, TAIL peaks +0.0179 (+41%) at β≈120, UNTRAINED.
- The learned head should recover/beat this AND absorb the manual knobs (β, IDF, sign) into parameters.

## Architecture (everything below the head FROZEN)
    score(i) = s_base(i)  +  Δ(i)
    s_base   = frozen tower cold score (== paord empty fold; item path bit-identical, structurally 0.4859 full)
    Δ(i)     = Σ_{c ∈ answered}  φ_sign(g_c) · w_c · align(c, i)
- align(c,i): GENOME RELEVANCE of item i to concept c (graded, from genome-scores.csv), fallback binary
  Mbin where genome missing (entities). Item-side attribute evidence → cannot move off-manifold.
- w_c = softplus(MLP([ log|members_c|, breadth_c, has_genome_c ]))  — FEATURE-parameterized (NOT 1628 free
  scalars; rare concepts would overfit). This learns the IDF-like down-weighting.
- φ_sign: two small maps, one for g>0 (love) one for g<0 (hate) — asymmetric (M&Ms-VAE++). Input the graded g.
- OUTPUT LAYER ZERO-INIT → Δ≡0 at step 0 → score == intercept EXACTLY (monotone-safety, full preserved).

## Exact Ns (NO reduction — HARD RULE #1)
- Train: all usable mm_train users (~150k). Full 18,430-item multinomial softmax per step (NO candidate pool).
- Concepts: all 1,628 (1,128 genome tags + 500 entities). All answered concepts per user (graded).
- Model-selection + reporting: DISJOINT val half (never the training users). 173/300 study users QUARANTINED.

## Supervision
- Concepts-ONLY, items MASKED at input (forces the head to be load-bearing; defeats item redundancy).
- Graded concept answers derived from the FULL profile (answerer duality; item-bias-residualized value),
  the established oracle-answer convention — NOT computed from the held half (leak).
- Loss = the tower's own multinomial CE over held-LIKED items on (s_base + Δ), s_base frozen.
- Optional small L2 on Δ→0 as an annealed safety leash.

## Baselines / action space (same ruler)
- Intercept (β=0, Δ=0): FULL 0.1886 / TAIL 0.0440 (log-pop) — the safety floor.
- Hand-set operator at β≈120: TAIL 0.0619 (the bar to beat).
- Knockout: zero the head → must return to intercept exactly.
- Compose with item channel (k>0 revealed items) → k-curve must not dilute.

## Metric + MDE
- HEADLINE: TAIL NDCG@10, honest bar = pop-WITHIN-attribute-filter (not global full). MDE: beat 0.0619
  by ≥ +0.003 (≈ the hand-set→learned headroom), 95% CI on disjoint half.
- GUARD: FULL NDCG ≥ intercept − 0.002 everywhere (structural via zero-init; verify empirically).
- Supporting: sign-flip specificity (love vs hate), β/scale monotonicity, love/hate asymmetry sane.

## Shortcuts flagged (each with its non-lossy alternative)
- Genome relevance is dense (movie×tag); load full, no thresholding (would drop signal). Fallback Mbin only
  where a concept has no genome row.
- 3000-val cohort for the nightly probe was a DIRECTION read; the headline reruns on the full disjoint half.
- No candidate pools, no user subsampling, no token caps.

## Kill list
- train_wmat.py (belief-token fold) is DEAD — do not resurrect the W(v)-into-belief operator.

## Backup (only if the head caps at the pop-within-genre ceiling)
- Trained whitened-INPUT fold (U1): concepts as zero-init input dims to the encoder, whitened centroids held
  in-subspace, train the encoder fold. Composable with the head, not a rival.

---
SIGN-OFF: __________________________   (author)   — training does not start until signed.
