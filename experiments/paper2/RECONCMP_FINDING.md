# Paper C — 0.71 is NOT saturated; the actor UNDERFITS (user was right)
Counts: nu=6040 users | trU=4827 | trbig=4823 | te=604 (eval te[300:]=304) | NC=761 concepts | NI=600 items.
("700" was concepts, not users.)

8-question reconstruction (cos to profile-taste u*) + full NDCG@10, te[300:]:
  8 real ITEMS   : recon-cos 0.836 | NDCG 0.3832
  8 popular CONC : recon-cos 0.811 | NDCG 0.3426
  trained ACTOR  : recon-cos 0.705 | NDCG 0.3573
=> Actor reconstructs WORSE than a trivial heuristic (0.705 < 0.811) => UNDERFITTING, NOT saturated. Room to >=0.81.
=> BUT cos != NDCG: conc8 has higher cos (0.811) yet LOWER NDCG (0.343) than actor (0.705 cos, 0.357 NDCG). The actor
   already beats the popular-concept heuristic on NDCG. items8 (0.383) is the askable-info ceiling but items aren't
   askable in cold-start.

Overfit check (TRAINVAL): recon-cos IDENTICAL train==test (0.712) => NO overfit, NO generalization gap. The earlier
TRAIN(0.291)<TEST(0.361) NDCG gap is a population/sampling artifact (different user subsets), not a model problem.

IMPLICATION (corrected strategy): actor UNDERFITS => (1) BC-PRETRAIN the actor to imitate strong concept/item selection
(reach heuristic ~0.81 cos) THEN unroll-finetune (breaks the 0.71 plateau from below; matches user's pretrain-on-discrete
idea). (2) Finetune objective must be NDCG-aligned (cos!=NDCG). NOT a data or capacity problem.
Probe: RECONCMP=1 block in continuous_actor.py.
