# 🔒 LOCKED — Paper C headline checkpoint (continuous actor + divisiveness field)

**Locked 2026-06-29. DO NOT MODIFY. Redundant backup of the Paper C headline model.**
Primary copy: `PAPER_C_CONT_WINNER/policy_cont_actor_DIVW_WINNER.pt`. Live training output:
`data/movielens/.cache/policy_phase3_d1divw_last.pt`. All three are byte-identical (sha1 `a63fec1e`).
This dir + the primary are both committed to git and pushed to the `recommender-improvement` remote → survive disk loss.

## Files (sha1 in SHA1SUMS.txt)
- `policy_cont_actor_DIVW_WINNER_a63fec1e.pt` — **THE headline checkpoint** (= `..._d1divw_last.pt`).
- `policy_phase3_d1divw_ep{6,8,10}.pt` — epoch curve (ep10 is eval-identical to _last; byte-differs: sha `1f3c775e`).
- `cache/` — the EXACT frozen recommender the actor was trained against and must be evaluated against:
  - `enc_concept.pt` — frozen V1 set-encoder (the recommender / fold-in instrument).
  - `Ql_concept.npy`, `Q_svd.npy`, `bi_svd.npy` — item factor matrix (scoring), SVD item factors, item bias (popularity-aware scorer `s = popb + Ql·u`).
  - `Ec_concept.npy`, `ctags_concept.npy` — concept (genome) embeddings + tag ids.
  - `pool_entavg.npy` — POOL_ENT divisiveness prior (CORRECT version; the actor does NOT read it, baselines do).

## What this model is
Continuous-action cold-start elicitation policy. MLP actor π(belief u∈R⁶⁴, turn t/8) → query direction q∈R⁶⁴.
Per turn: emit q, normalize, fold the off-pool point q·_CN into the frozen encoder with a GRADED geometric answer
`a = NEG + (POS−NEG)·(cos(u*,q)+1)/2`, update belief, repeat T=8 turns, recommend by `s = popb + Ql·u_T`.
Trained end-to-end by differentiable unroll (u* enters answer + gradient only ⇒ realizable at test time).

**Headline ingredient (request #3): the continuous divisiveness-field reward.**
`loss = [1 − cos(u_T, u*)] + β·L_softNDCG − DIVW · (1/T) Σ_t D(q_t)`, where
`D(q) = H_b( mean_i σ(u*_i · q̂ / DTAU) )` over 2500 unit train-user tastes = the EXACT continuous entropy
(divisiveness) of a direction; differentiable everywhere off-manifold; reproduces POOL_ENT at concept points.

## Training config (exact)
- Warm-start: from the u-reconstruction variant (`policy_cont_actor_recon_variant.pt`).
- Objective: reconstruction + soft-NDCG (β) + divisiveness reward.  **DIVW=1.0, DTAU=2.0, NUMAT=2500.**
- CONTMODE=cont (off-pool, NO snap), GRADED answers, frozen V1 encoder, T=8, 10 epochs, SKIPVAL (eval `_last`=ep10).
- Code: `scripts/paper2/continuous_actor.py` — field `field_div` (~L262), reward (~L805 `loss=loss-_DIVW*(_DIVH[0]/T).mean()`).

## Result (seed-avg {1,2,3,7,11}, te[300:], q8, graded; COMPARE4 harness — faithful, baselines reproduce canonical)
| policy (graded) | FULL | TAIL |
|---|---|---|
| **D1 = actor + divisiveness field (HEADLINE)** | **0.3780 ± 0.0032** | **0.1782 ± 0.0065** |
| continuity-only actor (recon+softNDCG) | 0.3695 ± 0.0046 | 0.1651 ± 0.0039 |
| CASPER-R (binary, discrete SOTA) | 0.3594 ± 0.0055 | 0.1467 ± 0.0049 |
- vs prev headline +0.0085 full / +0.0131 tail (~2σ); vs discrete SOTA +0.019 / +0.032.
- Monotone epoch curve: ep6 0.3739/0.1701 → ep8 0.3751/0.1758 → ep10 0.3780/0.1782.
- Binary control (graded-specificity): D1 binary 0.341/0.125 (regresses below continuity-only — divisive dirs need graded magnitude).
- Faithfulness: COMPARE4 binary baselines reproduce canonical; continuity actor reproduces 0.3695 vs canonical 0.3696.

## Reproduce the eval
```
NOBC=1 EP=0 COMPARE4=1 ONLYACTOR=1 \
  ACTORCK=data/movielens/.cache/policy_phase3_d1divw_last.pt \
  EVALSEEDS=1,2,3,7,11 CONTMODE=cont FEATS=ext,ans ANSF=1 \
  python scripts/paper2/continuous_actor.py
# full faithfulness table (casper/entropy/popular too): drop ONLYACTOR=1
# gold canonical cross-check: MODES=contactor GANS=1 QPTS=8 EP=0 SEED=<s> ACTORCK=... python scripts/paper2/continuous_policy2_st.py
```
## Reproduce training (D1)
Warm from u-recon variant; DIVW=1.0 DTAU=2.0 NUMAT=2500, CONTMODE=cont GRADED, frozen V1, 10 epochs, save per-epoch.

## Negatives (do not re-try as headline)
- D2 FieldActor (div+pop+rating as scored-candidate features, learned weights, soft-select): 0.351/0.148 < continuity-only.
  ⇒ the win is the divisiveness REWARD on a plain actor, not a candidate-scoring head.
- Popularity / avg-rating fields: built (`field_pop`, `field_rat`) but gave no benefit in any form tried.
