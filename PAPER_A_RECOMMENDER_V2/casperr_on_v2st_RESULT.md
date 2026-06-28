# CASPER-R retrained on V2-ST (set-transformer) — concept-policy ablation
Seed 1, te[300:] (304 users), NDCG@10 FULL/TAIL, q8. Recipe = locked CASPER-R
(BCEP=25 entropy-imitation floor -> OBJ=ustar reconstruction finetune, CONCONLY).
LOADREC=enc_concept_st.pt (V2-ST, MHSA+meanpool, order-invariant).

| checkpoint            | FULL q8 | TAIL q8 |
|-----------------------|---------|---------|
| concept-entropy floor | 0.340   | 0.122   |
| conc_pop              | 0.337   | 0.115   |
| BC floor              | 0.337   | 0.118   |
| ep1                   | 0.333   | 0.116   |
| ep2                   | 0.332   | 0.125   |
| ep3                   | 0.333   | 0.128   |
| ep4                   | 0.331   | 0.122   |
| **ep5 (PEAK)**        | 0.334   | **0.130** |
| ep6                   | 0.324   | 0.122   |

PEAK = ep5 (policy_casperr_st_conc_PEAK_ep5.pt). Finetune lifts tail over the
entropy floor (+0.008) at tiny full cost (-0.006): SAME qualitative shape as V1.
BUT absolute level << V1 CASPER-R (0.361/0.150): V2-ST is a worse ELICITATION
instrument despite being a better FULL-PROFILE recommender (capacity x input-length
tradeoff, confirmed on the trained policy). DECISION: V2-ST -> Paper A recommender;
V1 -> Paper B elicitation (CASPER-R unchanged). Single-seed; gap to V1 >> seed noise.
