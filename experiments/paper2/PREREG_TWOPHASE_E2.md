# PRE-REGISTERED GATE — Two-phase E2: realizable exposure-model channel switch (written 2026-07-03, BEFORE any run)

Motivation: the answerability-ORACLE ceiling for 4-concepts→4-rated-items is +0.012 FULL / −0.006 TAIL @q8
(TWOPHASE_ORACLE_RESULT.md). The realizable question: how much of the FULL gain does a learned exposure
model capture, and how much worse does tail get when predicted-answerable items are sometimes unanswerable
(wasted turns)?

## Frozen decisions
- **Exposure model**: supervised P(user has rated item | belief u_t, item) trained on TRAIN users only,
  with partial-reveal states simulated from the entropy-4 prefix (labels free: item in full profile or not).
  Features: u_t·e_i, log-popularity, u_t (or an interaction); small MLP or logistic — implementer's choice,
  fixed before eval. NO RL.
- **Policy**: turns 1-4 = static entropy concepts (unchanged); turns 5-8 = argmax over unasked items of
  P(rated | u_t, i) × item-IG. Unanswerable picks waste the turn (standard B regime).
- **Ruler**: eval seeds {1,2,3,7,11}, te[300:], q8 + q-curve. Comparators: entropy-8 (0.3609/0.1397),
  the oracle ceiling (0.3728/0.1342).
- **PRIMARY metric (named in advance): FULL NDCG@10 @q8 vs entropy-8.** Success = positive at ≥ +0.005
  AND paired per-user bootstrap p<0.05. TAIL reported honestly (expected ≤ entropy; if tail loss > 0.010
  the method is flagged deployment-costly regardless of full win).
- **Also report**: answered-count at turns 5-8 (oracle achieves 4/4; model precision = the mechanism),
  exposure-model AUC/precision@4 on val.
- **Failure**: anything else → record as "oracle does not translate," no iteration without new prereg.
- Model training seed fixed (0); if the primary passes, 2 more training seeds required before any paper claim.

Committed before first run.
