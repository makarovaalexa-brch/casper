# Does the concept fold use WHICH concept it was given? — yes, decisively

**Date:** 2026-07-29 · **Script:** `src/instrument/concept_identity_lift.py` (pre-registered design in
the docstring) · **Output:** `experiments/battery/concept_identity_lift.json` · **Fold:** `cd_s1_l10_best.pt`

## Why it was run
V2 claims the concept channel is a real channel. The existing evidence covered the **direction**
(whitened member AUC 0.94, measured *before* any fold) and one popularity control (+0.0087). Neither
pinned down that the **operator** uses φ_c's identity. A gated fold that read the answer value *y* as a
generic "this user is being positive" nudge would pass both while ignoring the concept entirely — the
reviewer's strongest objection to V2.

## Result (n = 905 users, paired bootstrap)

| arm | fold | member AUC on c's members |
|---|---|---|
| LIKE | (c, +\|v\|) | **0.6256** |
| DISLIKE | (c, −\|v\|) | **0.4159** |
| OTHER | (c′, +\|v\|), c′ ≠ c, also foldable | **0.4978** |

- **SIGN** = like − dislike = **+0.2096**, CI [+0.1986, +0.2208]
- **IDENTITY** = like − other = **+0.1278**, CI [+0.1195, +0.1361]
- Pre-registered MDE 0.02; **PASS** (both positive, both CIs exclude zero).

**The shape is the finding.** A like lifts the concept's members; a dislike pushes them *below chance*;
folding a *different* concept leaves them *at chance* (0.4978). Generic-nudge behaviour would show ≈0 on
IDENTITY. It shows +0.128.

**Two independent consistency checks.** The like arm (0.6256) reproduces `g5_split`'s separately measured
post-fold member AUC (0.630) on the same fold. The sign contrast is consistent with the concept sign-flip
already in the chapter (−0.2012 NDCG).

**Scope.** 905 of 10,000 candidate users qualify — those whose top concept is foldable *and* who have a
second foldable concept for the OTHER arm. Restricting to foldable alternatives is what makes the identity
contrast honest.

## ⚠ The first run was a harness bug, not a null — the pattern to remember
Run 1 reported SIGN = **exactly 0.0000** with a zero-width CI, member AUC 0.51 (chance) in every arm.

`Rung.map_answers` under `val_source="signed"` **discards the value it is handed** and substitutes the
user's stored signed value, so the like and dislike arms were bit-identical (0.511680762690531 in both).
It also returns `[]` for refuse-band concepts, so the fallback folded a 0.0 value and every arm sat at
chance. Had that been believed, we would have reported "the fold ignores answer sign" — false, and
damaging.

**Tells that it was a bug:** a contrast of *exactly* zero to 15 decimal places, a zero-width CI, and a
number (0.51) that disagreed with an existing measurement of the same quantity (0.63). Any one of those
should stop a null being written up. See memory `paper-writing-rules-and-review-round-2026-07`.

## Why we still do not gate on raw post-fold member lift
The additive-union operator scores member AUC **0.820** and craters to **0.0991** NDCG — below the
no-answer intercept. The gated fold scores **0.630** and reaches **0.1759**. Pushing all of a concept's
members up the ranking is genre-filter behaviour and it costs accuracy. The differential contrasts above
ask the question that matters — is the movement concept-specific — without rewarding that failure mode.
