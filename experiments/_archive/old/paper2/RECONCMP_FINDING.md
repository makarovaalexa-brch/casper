# Paper C — CORRECTED: actor reconstructs GREAT (cos 0.922); the wall is cos != NDCG (not underfit, not data)
Counts: nu=6040 users | trU=4827 | trbig=4823 (training set) | te=604 (eval te[300:]=304). "700"=concepts; "400"=a read-off sample.

8-question reconstruction (cos to profile taste u*) + NDCG@10, te[300:]:
  8 real ITEMS            : cos 0.836 | NDCG 0.3832
  8 popular CONCEPTS      : cos 0.811 | NDCG 0.3426
  actor -- BINARY eval(BUG): cos 0.705 | NDCG 0.3573
  actor -- GRADED eval(OK) : cos 0.922 | NDCG 0.3666   <-- matches TRAINING answer model + Paper C continuous-answer

CORRECTION: earlier "actor underfits / 0.71 ceiling" was a BINARY-vs-GRADED EVAL BUG (actor trained on graded answers,
evaluated with binary). With the correct graded answers the actor reconstructs to cos 0.922 -- BEATS the heuristics
(0.811/0.836), as expected for a profile-optimised policy. NOT underfitting; BC-pretrain rationale RETRACTED.

THE REAL WALL: cos 0.922 but NDCG only 0.367. Near-perfect reconstruction of the profile-half taste u* does NOT yield
high NDCG (items8 has LOWER cos 0.836 but HIGHER NDCG 0.383). => the OBJECTIVE is wrong: reconstructing u* (the asked
profile) != ranking the HELD-OUT likes. Lever = NDCG/ranking-aligned objective (e.g. USTARHELD: target held-out likes,
or a soft-NDCG surrogate), NOT data/capacity/encoder.

On the TRAINVAL train<test NDCG gap: recon-cos identical train==test => no overfit (model generalizes); the NDCG
difference is a real PROPERTY OF THE TWO USER SUBSETS (rankability of their held likes + popularity), not model overfit
and not mere noise. (Both TRAINVAL rows were binary-eval; redo graded for clean numbers.)
