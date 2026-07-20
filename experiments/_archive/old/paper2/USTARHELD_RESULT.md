# Paper C — USTARHELD (target held-fold): NEGATIVE (train/test answer mismatch)
USTARHELD=1 (reconstruct held-out-likes taste instead of profile-half u*). Peak ep6 full 0.340 / tail 0.140
(VALTEST seed123) -- WORSE than u*-target baseline 0.346/0.149.
WHY: held-target made the simulated TRAINING answers come from the held taste, but at TEST answers come from the
PROFILE. The held-fold 0.428 ceiling requires KNOWING the held likes -> NOT realizable from profile-based answers.

CORRECTED HIERARCHY (te[300:], RECON3 graded split):
  held-fold 0.428 (needs held)  | full-profile 0.419 (needs full)  | u*=half-fold 0.409 (REALIZABLE ceiling)
  actor 0.369 (cos 0.92 to u*)  | held-target run 0.340 (mismatch)
=> realizable ceiling ~0.41 (reconstruct profile-half taste). Actor at 0.369 has ~0.04 recoverable room, limited by
   QUESTION-SELECTION quality (push 8-question recon cos 0.92->higher), NOT the target/objective. That selection gap is
   exactly what the privileged oracle closes (picks questions using u*). Realizable continuous ~ties discrete; oracle
   headroom needs privileged selection.
