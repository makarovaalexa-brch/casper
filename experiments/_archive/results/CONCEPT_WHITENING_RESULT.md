# Concepts ARE directions — after popularity-stripping (2026-07-16)

## The arc (a wrong conclusion, then the recovery)

**Negative result (fusj joint fusion+concept run, ordinal recommender paord).** Trained concept embeddings
(member-bag init) folded through the cross-attention fusion. Measured on val users:
- cold-concept TAIL@10: intercept 0.041 -> 0.048 at k=8-16 -> DILUTES to 0.046 at k=32; real ~= random.
- **SIGN-FLIP SPECIFICITY FAILED:** flip one concept LOVED<->HATED, |score change| on member films vs unrelated.
  Ratio ~1.0 (want >>1); only horror 1.9x. The concept negation was GLOBAL, not targeted.

**Wrong conclusion I drew (RETRACTED):** "a genre is not a direction in CF factor space; concepts-as-embeddings
can't be member-specific." I based this on (a) a CEILING test (force z0+alpha*concept_dir, decode, member/unrel
0.6-0.8x <1) and (b) decoder-row coherence: mean pairwise cosine of Wd[members] ~= random pairs (~0.51).

**The error:** both tests decoded through the RAW embedding space, which is dominated by a shared POPULARITY /
common component (the ~0.51 random-pair cosine IS that component). Raw cosine/decoding is swamped by it, MASKING
the genre structure that lives in the residual. The 64-dim SVD (Paper B's space) already gave membership AUC
0.74-0.91 vs the 512-d raw factors' 0.42-0.69 -- the low-rank SVD de-emphasizes popularity magnitude and exposes
taste geometry. That was the clue.

## The recovery (author was right: concepts ARE coherent directions)

Remove the popularity component (CENTER = subtract mean; optionally also remove top-1 PC), then re-measure:

**Membership AUC of centroid-cosine (item_emb whitened):**
```
concept       raw    center   -1PC
horror        0.689  0.943    0.947
sci fi        0.533  0.861    0.913
romance       0.443  0.834    0.849      <- "diffuse" genre, fully recovered
comedy        0.423  0.819    0.861
action        0.463  0.859    0.898
```
Just centering takes AUC 0.44 -> 0.83-0.94.

**Specificity through the recommender's OWN raw decoder** (Wd @ whitened concept direction), member vs unrelated:
```
horror  +0.300/+0.003 (88x)   sci fi +0.232/-0.013 (17x)   comedy +0.158/-0.012 (14x)   romance +0.209/+0.035 (6x)
```
vs the raw-space "failure" of 0.6-0.8x. So a WHITENED concept direction, decoded through the strong
recommender's real decoder, lifts its member films 6-88x over unrelated. **Concepts ARE directions -- in the
popularity-removed subspace.**

## What still needs training (a FOLD problem, not a representation problem)

Naive additive fold z0 + alpha*(whitened dir), decoded via raw Wd, gives a small TAIL lift but TRADES full:
```
k=4 alpha=2.0: FULL 0.137  TAIL 0.043   (intercept FULL 0.179 TAIL 0.039)
k=16 alpha=1.0: FULL 0.168 TAIL 0.042
```
Specificity is perfect (88x); the naive additive MAGNITUDE/BALANCE is wrong (a raw region-shift moves off the
popularity prior the head ranking needs). => the recommender must LEARN to fold whitened concept directions with
the right magnitude/balance. The earlier fusj failure was folding in the RAW (popularity-masked) space, where
the direction was non-specific to begin with.

## THE FIX (recovers Paper B's abstract-embedding result in the STRONG recommender)
- Concepts stay on the INPUT, folded with ratings, unified with items (author's design UNCHANGED).
- The one change: concept embedding = WHITENED member centroid (member-bag, mean + top-1 PC removed), NOT the
  raw member-bag.
- USE vs TRAIN: initialize concepts from the whitened centroid; TRAIN THE FOLD (encoder), keep the concept
  embeddings FIXED or projected-to-whitened-subspace each step -- free training drifts the popularity component
  back in (that's what fusj did). Test freeing them later with a stay-in-subspace regularizer.
- Verify after retrain: sign-flip ratio >>1, k-curve stops diluting, full held ~0.48, real>random.

## NOVELTY (honest): the whitening is NOT novel.
Stripping the dominant/popularity component = ALL-BUT-THE-TOP (Mu & Viswanath, ICLR 2018) / mean-centering -- a
standard embedding-isotropy technique; it is well known that the top principal component of CF and word
embeddings encodes frequency/popularity. So popularity-stripping is a NECESSARY FIX, not a contribution. The
contribution lives in the elicitation architecture around it (unified input fold, negation, continuous/open
channels), not the debiasing.

Probes: inline scratch scripts (whitening AUC, whitened-decode specificity, whitened-fold NDCG). Checkpoints in
.cache/set_mn/ (paord_best, fusj_ep5). See [[fusion-token-context-dependent-value]] and
[[three-channel-continuous-elicitation]].
