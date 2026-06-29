# Paper C — soft-NDCG (ApproxNDCG listwise, popular hard-negs): NEUTRAL (information wall confirmed)
PURE softNDCG (replace recon), warm from u*-actor: COLLAPSED ep1 0.303/0.066 (games surrogate, wrecks belief).
COMBINED recon + 0.3*softNDCG, warm: STABLE but FLAT. Graded RECON3 best ckpt:
  actor (recon only):       cos 0.920 NDCG 0.3689
  + soft-NDCG (combo):      cos 0.931 NDCG 0.3699   (cos UP, NDCG flat = noise)
=> ranking objective does NOT break the ceiling. Even cos 0.93 -> NDCG 0.370 (vs u* 0.409): NDCG@10 so sensitive that
   7% belief error reshuffles top-10; closing 0.04 needs cos ~0.99 (unreachable with 8 profile-answers).

EXHAUSTED LEVERS (all land at ~0.37 graded / ~tie discrete): objective {recon, bce, softndcg-pure, softndcg-combo};
target {u*, held}; encoder {open-loop rand/info, co-train}; architecture {free, concmix}. ALL confirm the same:
realizable continuous policy is INFORMATION-limited (answers a=u*.q carry only u*-info; held direction beyond u* is
unrecoverable). Oracle headroom (0.39-0.43) needs privileged info. To beat = change the ANSWER MODEL (answerability/bot-play).
