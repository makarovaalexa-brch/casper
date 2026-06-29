# Paper C MILESTONE — continuous policy beats discrete SOTA (seed-avg {1,2,3,7,11}, te[300:], SAME frozen V1 recommender)
4 policies x {binary,graded answers} x full/tail. Only the POLICY + answer model vary; encoder+Ql+popb identical.

| policy            | binary FULL/TAIL    | graded FULL/TAIL    |
|-------------------|---------------------|---------------------|
| continuous actor  | 0.3559/0.1463       | **0.3663/0.1623**   |
| CASPER-R (disc.)  | 0.3523/0.1479       | 0.3281/0.1241       |
| entropy-8 (disc.) | 0.3460/0.1394       | 0.3385/0.1295       |
| popular-8 (disc.) | 0.3465/0.1264       | 0.3246/0.1197       |
(stds ~0.003-0.006)

HEADLINE: in the GRADED (continuous-answer) regime = Paper C's setting, the continuous actor (0.366/0.162) beats
CASPER-R's BEST (binary, 0.352/0.148) by +0.014 full / +0.014 tail (~3 sigma, both significant). The continuous policy
UNIQUELY benefits from graded answers (0.356->0.366 full, 0.146->0.162 tail) while EVERY discrete policy DROPS on graded
(built for binary). Earlier "discrete scores higher" = binary-only + cross-harness + actor evaluated with mismatched
binary answers. Winning model = u*-RECONSTRUCTION actor (policy_phase2_cont_v1_best). Frozen recommender identical across all.

## Objective ablation (continuous actor; seed-avg {1,2,3,7,11}, te[300:])
| training objective        | binary FULL/TAIL | graded FULL/TAIL |
|---------------------------|------------------|------------------|
| distilled-from-discrete   | 0.3545/0.1492    | 0.2928/0.0945 (COLLAPSE) |
| trained-on-u (recon)      | 0.3559/0.1463    | 0.3663/0.1623    |
| trained-on-NDCG (recon+softNDCG) | 0.3546/0.1453 | **0.3695/0.1651** (best) |
- Distilling from discrete INHERITS the binary ceiling & collapses on graded -> continuous win needs NATIVE training.
- u and NDCG both work; NDCG edges u (+0.003, ~1sigma). soft-NDCG DID help (earlier 'neutral' was a binary-eval confound).
BEST continuous = trained-on-NDCG graded 0.370/0.165 vs CASPER-R best (binary) 0.352/0.148 = +0.018 full/+0.017 tail.
