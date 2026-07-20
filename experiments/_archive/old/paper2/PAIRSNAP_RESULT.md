# PAIR-SNAP RESULT — realizing D1 queries as item-pair differences does NOT rescue snapping

**Date:** 2026-07-02. **Question:** the concept-snap experiment (SNAPLOSS_RESULT.md) showed naming the query costs
−0.037/−0.040. Is that a limitation of the CONCEPT vocabulary, or of realization itself? Pair differences
d = (e_i − e_j)/||e_i − e_j|| over the full item catalog span a vastly richer realizable set (N(N−1)/2 ≈ 6.9M directions,
a natural "would you rather A than B?" question). If pair-snap ≈ un-snapped, snapping was just a vocabulary problem;
if pair-snap ≈ concept-snap, the off-manifold RESIDUAL itself is load-bearing.
**Verdict: pair-snap lands ON the concept-snap floor (0.337/0.134 ≈ 0.341/0.138), far below un-snapped D1
(0.378/0.178). The snap-loss is NOT a vocabulary artifact — 6.9M pair directions realize the query no better
(cos 0.72) than 761 named concepts (cos 0.70). Off-manifold queries stay load-bearing.**

## Mechanism
Identical to the headline snap-loss harness (COMPARE4 ONLYACTOR, graded, D1 checkpoint), except each emitted query q
is replaced before folding by its best ITEM-PAIR DIFFERENCE:

    (i,j) = argmax_{i≠j} cos(q, e_i − e_j)  over the FULL SVD item factor set Q (N=3706; sign symmetry via both orders)
    d = (e_i − e_j)/||e_i − e_j||;  fold (d·_CN, answer) with the SAME graded geometric answer a = NEG+(POS−NEG)(cos(u*,d)+1)/2

- Search: top-K trick — i from top-64 of s = E·q̂, j from bottom-64, exact cos over the 64×64 candidate pairs.
- Fold magnitude: d is scaled by _CN (mean concept norm), exactly the un-snapped convention (q̂·_CN) → the ONLY thing
  that changes vs un-snapped D1 is the DIRECTION. Concept-snap (ACTSNAP) instead folds Ec[c] at its natural norm.
- Checkpoint: `policy_phase3_d1divw_last.pt` (sha a63fec1e, = LOCKED paperC_divw_LOCKED_2026-06-29). Frozen V1 encoder,
  te[300:] (304 users), q8, seed-avg {1,2,3,7,11}. No training anywhere; eval-only.
- Code: `PAIRSNAP` block in `scripts/paper2/continuous_actor.py` (after COMPARE4); `PAIRBIN=1` = binary variant,
  `PAIRK` = K, `PAIRCHECK` = exact-search verification sample, `PAIRSET=pool` restricts to the 600-item pool (unused).

## Result (seed-avg {1,2,3,7,11}, te[300:], q8)
| condition | FULL | TAIL |
|---|---|---|
| D1 UN-SNAPPED (off-manifold, headline) | 0.3780 ± 0.0032 | 0.1782 ± 0.0065 |
| entropy (binary, canonical) | 0.3618 ± 0.0026 | 0.1393 ± 0.0041 |
| D1 concept-snap (ACTSNAP, SNAPLOSS_RESULT) | 0.3414 | 0.1384 |
| **D1 PAIR-SNAP, graded answer (NEW)** | **0.3373 ± 0.0039** | **0.1344 ± 0.0046** |
| **D1 PAIR-SNAP, binary answer (NEW)** | **0.3398 ± 0.0077** | **0.1265 ± 0.0031** |

- Pair-snap loss vs un-snapped: **−0.041 FULL / −0.044 TAIL** — the same size as the concept snap-loss (−0.037/−0.040),
  despite a ~9,000× larger realization vocabulary. Pair-snap even sits marginally BELOW concept-snap (−0.004, ≈1σ, tie).
- Pair-snap < entropy (0.362/0.139) and < CASPER-R (0.359/0.147): an emit-then-pair-realize deployment would LOSE to the
  discrete SOTA, exactly like emit-then-name.
- BINARY pair variant: FULL statistically unchanged (0.3398 vs 0.3373) but TAIL drops further (0.1265 vs 0.1344,
  ~2σ below graded pair, −0.052 below headline). The 1-bit channel hurts where it always hurts (tail), consistent with
  the D1-binary control (0.341/0.125 in the LOCKED manifest): the pair channel does NOT rescue 1-bit answers either.

## Realization geometry (why it fails)
- Realization cosine cos(q̂, d), all 2432 seed-1 graded queries: **mean 0.7165, median 0.7162**
  (p10 0.6511, p25 0.6862, p75 0.7512, p90 0.7843, min 0.5439).
- Concept-snap cosine on the SAME queries: mean 0.6978, median 0.7074 (matches SNAPLOSS's ~0.71–0.73 reference).
- So the best of ~6.9M pair directions recovers only ~0.02 more cosine than the best of 761 named concepts, and NDCG does
  not move. D1's queries live off EVERY item-spanned manifold (single items, concepts, AND pair chords); the ~0.7-cos
  residual is where the value is. This is the strongest form of the snap-loss claim so far: the un-realizable component
  is not "not yet catalogued", it is genuinely outside the span of catalog-anchored single directions.

## Approximation-gap check (top-K trick vs exact)
Exact argmax over all N(N−1)/2 pairs (chunked full search) on 200 random queries vs the K=64 top/bottom trick:
**mean cos gap 0.0036, max 0.0451, exact pair recovered 130/200**. The gap (≤0.005 mean on a 0.7165 mean cosine) is an
order of magnitude smaller than the 0.28 un-realized residual — the negative result is not a search artifact.

## Honest caveats
1. Pair realization here is GEOMETRIC (fold the normalized difference direction; graded answer along it). A deployed
   "A rather than B?" question would need an answer model for comparative judgments; we deliberately reuse the exact
   SNAPLOSS answer/fold so the ONLY manipulated variable is the direction. Scale is held at the un-snapped q̂·_CN
   convention; concept-snap folds Ec at natural norm — the answer is scale-invariant, only the encoder sees the
   difference, and pair-snap ≈ concept-snap suggests this choice is not driving the result.
2. No no-repeat constraint on pairs (the un-snapped and ACTSNAP rolls handle repetition differently); with 6.9M pairs,
   repeats are rare and both signs are allowed, so this can only FAVOR pair-snap — the negative stands a fortiori.
3. Pairs are unconstrained by answerability/popularity (any catalog item may enter a pair); a realistic pair question
   would restrict to items the user can judge, which could only lower these numbers further.
4. Concept-snap row is the single-run headline number from SNAPLOSS_RESULT (no ±); its harness is byte-identical up to
   the realization step.

## Repro
```
NOBC=1 EP=0 PAIRSNAP=1 PAIRBIN=1 EVALSEEDS=1,2,3,7,11 PAIRCHECK=200 CONTMODE=cont FEATS=ext,ans ANSF=1 \
  ACTORCK=data/movielens/.cache/policy_phase3_d1divw_last.pt python scripts/paper2/continuous_actor.py
# PAIRK=64 (default) top/bottom-K candidates; PAIRSET=pool -> 600-item pool instead of full catalog.
```
