# Deployability TODOs (user-raised, 2026-07-01) — clean the setup so it matches real inference

At inference we know NOTHING about the user except what they told us in the conversation. Two setup shortcuts violate this;
both are SHARED across all methods (so they shouldn't change the RANKING) but they are non-deployable and inflate absolute
NDCG. Fix for Paper E's deployability claim; recalc all methods on the same clean protocol.

## TODO-1: rec-masking is not deployable
Current: `s[list(half)]=-1e9` masks the user's KNOWN-HALF items from recommendations. In deployment we don't have the
user's history to mask. FIX: mask ONLY conversation-revealed items (items the user named this session), not the hidden
half. Shared across methods => ranking ~unchanged; absolute NDCG drops. PARKED (do after the answerability fix).

## TODO-2: answerability filter is BACKWARDS (the important one)
Current DISCRETE eval (CASPER-R, PEBOL, ConTS, entropy) PRE-FILTERS the action set to answerable concepts:
`ac=[c for c in range(NC) if len(citems[c]&half)>=2]` — the QUESTIONER consults the user's actual items to decide what to
ask. Not deployable (questioner shouldn't know the profile) + gives a free "no wasted turns" boost.
DEPLOYABLE PROTOCOL (user-stated): questioner asks BLINDLY; answerer knows own ratings and REFUSES if unanswerable; the
turn is WASTED. => answerability enforced on the ANSWER side (refuse), not the QUESTION side (pre-filter).

WE ALREADY BUILT THIS (Paper C continuous actor): _REFUSE (refuse |cos(u*,q)|<TAUR, token not folded, wasted turn,
straight-through so actor learns per-user-decisive asking), _ABOTREF (refuse when learned answerer confidence low),
rollout_sample reward = D[...] - PEN*(unanswered), unans=(ANS==0). Actor asks over the FULL pool and LEARNS answerability
from the population (no per-user peek). BUT: (a) marked "refusal DEAD" as a LEVER (likely because a trained actor already
asks answerable => refusal adds nothing = a GOOD deployable story, not a failure); (b) the reported DISCRETE baselines
still use the `ac` pre-filter, NOT blind+refuse.

FIX: re-run the discrete methods (CASPER-R, PEBOL-fair, ConTS-fair, entropy) under BLIND-ASK + REFUSE (drop `ac`
pre-filter; user refuses unanswerable via geometric/learned answerability; wasted turn). Report as the deployable numbers.
Ranking expected to hold (shared change); absolute dips. This is the honest Paper-E deployability protocol.

## DONE (Jul 1): strict blind+refuse baselines (TODO-2 resolved for the probes)
Added BLINDREFUSE mode to LITBASE (ask over ALL concepts range(NC); unanswerable pick wastes the turn, no fold/update;
matches CASPER-R). Protocol progression (PEBOL / ConTS, q8, seed-avg{1,2,3}):
  leaky(known-item Beta + prefilter):  PEBOL 0.369/0.168 | ConTS 0.351/0.151
  fair(shared pool, still prefilter):  PEBOL 0.327/0.110 | ConTS 0.306/0.112
  STRICT(shared pool + blind+refuse):  PEBOL 0.317/0.092 | ConTS 0.309/0.089   <- FINAL, matches our methods
All now on the SAME strict protocol as CASPER-R (0.360/0.152), entropy (0.361/0.140), continuous D1 (0.377/0.175), open
recall (0.405/0.195). Probes dominated on both axes; tail gap widened (wasted turns hurt niche discovery). paper4
tab:baselines + "no peeking"/"reading the table" paragraphs updated to strict numbers. Repro: LITBASE=pebol|conts
FAIRCAND=1 BLINDREFUSE=1 QPTS=8. Our methods verified CLEAN (CASPER-R blind+refuse, open recall answerable-by-def,
continuous belief-selection+geometric-answer+ITEMSANS fully-answerable, no leak). TODO-1 (rec-masking) still parked.

## Note on the fairness/leak fix (already done Jul 1)
Separate from these: the PEBOL/ConTS Beta-over-known-half-ITEMS leak is FIXED (fair shared pool: PEBOL 0.369->0.327/0.110,
ConTS 0.351->0.306/0.112; paper4 tab:baselines corrected). The answerability pre-filter (TODO-2) is a DIFFERENT, shared
issue and still open. Both are about "questioner must not consult the hidden profile."
