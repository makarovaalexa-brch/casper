# 🏆 PAPER C WINNER — Continuous-Action Preference Elicitation (DO NOT OVERWRITE)

**Date locked:** 2026-06-29 (updated 2026-06-29 w/ DIVW headline). **Branch:** recommender-improvement.

## ⭐ NEW HEADLINE (2026-06-29): D1 = continuous actor + CONTINUOUS DIVISIVENESS-FIELD REWARD
- **`policy_cont_actor_DIVW_WINNER.pt` (sha a63fec1e) = the NEW headline, graded 0.378/0.178.**
  Same continuous actor as below, warm-started from the u-recon variant, then 10 epochs with an AUXILIARY reward
  `loss = base − DIVW·mean_t divisiveness(q_t)`, DIVW=1.0, DTAU=2.0, frozen V1, GRADED. The divisiveness field is the
  EXACT continuous entropy of a direction: `div(q)=H_b(mean_i σ(u*_i·q/τ))` over 2500 train-user tastes — differentiable
  everywhere off-manifold, reproduces POOL_ENT at concept points. It rewards the actor for asking maximally-DIVISIVE
  (population-informative) directions = the entropy heuristic extrapolated into continuous space and shaped into the policy.
  THIS IS REQUEST #3 ("add back entropy/popularity/rating signal, extrapolate in cont space") — and it WORKS.
- **Result (COMPARE4 harness, faithful — baselines reproduce canonical; seed-avg {1,2,3,7,11}, te[300:], q8, graded):**
  | policy (graded) | FULL | TAIL |
  |---|---|---|
  | **D1 = actor + divisiveness reward (NEW HEADLINE)** | **0.3780 ± 0.0032** | **0.1782 ± 0.0065** |
  | continuous actor recon+softNDCG (prev headline) | 0.3695 ± 0.0046 | 0.1651 ± 0.0039 |
  | CASPER-R (binary, discrete SOTA) | 0.3594 ± 0.0055 | 0.1467 ± 0.0049 |
  | entropy (binary) | 0.3618 ± 0.0026 | 0.1393 ± 0.0041 |
  | popular/conc_pop (binary) | 0.3465 ± 0.0027 | 0.1264 ± 0.0046 |
  - **D1 vs prev headline: +0.0085 FULL / +0.0131 TAIL (tail ~2σ).  D1 vs discrete CASPER-R: +0.019 FULL / +0.032 TAIL.**
  - **Epoch curve is MONOTONE (not a lucky last-epoch): ep6 0.3739/0.1701 → ep8 0.3751/0.1758 → ep10 0.3780/0.1782.**
  - Faithfulness: COMPARE4 binary baselines reproduce canonical (casper 0.359≈0.360, entropy 0.362≈0.361, popular 0.347≈0.347);
    prev headline reproduces its canonical 0.3696/0.1650 here as 0.3695/0.1651 → COMPARE4 ≡ canonical for actors.
  - Repro: `NOBC=1 EP=0 COMPARE4=1 ONLYACTOR=1 ACTORCK=.../policy_phase3_d1divw_last.pt EVALSEEDS=1,2,3,7,11 CONTMODE=cont FEATS=ext,ans ANSF=1 python scripts/paper2/continuous_actor.py`
  - D2 (FieldActor: divisiveness+popularity+avg-rating as scored candidate features) LOST: 0.351/0.148 < prev headline → the
    multi-signal scored-candidate architecture underperforms; the win is the divisiveness REWARD on the plain actor, not a
    candidate-scoring head. (Popularity/avg-rating fields built + available but did not help as actor features.)
- Stacked story for the paper: **discrete 0.359/0.147 → +continuity 0.370/0.165 → +continuous divisiveness field 0.378/0.178.**
- GOLD confirmation on the slow canonical `continuous_policy2_st.py` harness: see experiments/paper2/d1_canonical_GOLD_*.log.

## The checkpoint (PREVIOUS headline, now the continuity-only ablation row)
- **`policy_cont_actor_WINNER.pt` = the recon+soft-NDCG combo, graded 0.370/0.165 on the canonical harness.**
  Continuous actor (MLP belief∈R⁶⁴ + turn → query q∈R⁶⁴), FROM SCRATCH (no distillation), differentiable unroll,
  CONTMODE=cont (off-pool, no snap), GRADED answers, frozen V1 encoder; objective = reconstruction (1−cos(u,u*)) + soft-NDCG.
- `policy_cont_actor_recon_variant.pt` — reconstruction-ONLY variant, 0.366/0.162 (within ~1σ; the model the snap-loss/QPROBE
  analyses were run on; qualitatively identical). Reproduce recon: `CONTMODE=cont OBJ=ustar GRADED=1 NOBC=1 VALTEST=1 SELVAL=tail`.
- `cache/` — the EXACT frozen recommender deps: enc_concept.pt (V1, sha `27e6c72d`), Ql_concept.npy, Ec_concept.npy,
  Q_svd.npy, bi_svd.npy, ctags_concept.npy, **pool_entavg.npy (sha `156c072c` = CORRECT; a corrupted `09feb829`
  version depressed baselines during dev — see below)**.

## THE RESULT (harmonized to the canonical Paper B ruler; baselines reproduce paper EXACTLY)
Canonical Paper B harness (`continuous_policy2_st.py`, `contactor` mode), seed-avg {1,2,3,7,11}, te[300:], q8:
| policy | FULL | TAIL |
|---|---|---|
| **continuous actor — recon+softNDCG (HEADLINE)** | **0.3696 ± 0.0045** | **0.1650 ± 0.0040** |
| continuous actor — recon only (analysis variant) | 0.3664 ± 0.0029 | 0.1622 ± 0.0051 |
| CASPER-R (binary, Paper B SOTA) | 0.3602 ± 0.0033 | 0.1522 ± 0.0025 |
| entropy (binary) | 0.3608 ± 0.0016 | 0.1398 ± 0.0040 |
| uent+GRAW (unified entropy, graded, answerable items) | 0.3667 ± 0.0045 | 0.1577 ± 0.0078 |
| conc_pop (binary) | 0.3466 ± 0.0027 | 0.1266 ± 0.0045 |

**WIN (headline combo vs CASPER-R): +0.009 FULL (~2σ) / +0.013 TAIL (~3σ).** uent+GRAW (static entropy + graded + answerable
items, the "item-derived" baseline) reaches 0.367/0.158 on full → learned policy's contribution localizes to the TAIL via
off-manifold action. NOTE: an earlier draft headlined the recon-only 0.366/0.162 by mistake (wrong checkpoint).
Best-vs-best (each policy on its native answer model). On BINARY both tie (actor 0.356 ≈ CASPER-R) → the win is the
GRADED (continuous-answer) regime: **continuous answer + continuous policy together**.

## KEY FINDINGS (durable)
1. **SNAP-LOSS (centerpiece):** snapping each off-manifold query to its nearest answerable concept costs **−0.039 FULL /
   −0.031 TAIL** (un-snapped 0.366/0.162 → snapped 0.328/0.131). Snapped FALLS BELOW CASPER-R (0.360/0.152) ⇒ a
   Wolpertinger/PEBOL-style emit-then-snap LOSES to discrete. **The win requires NOT snapping.** (experiments/paper2/SNAPLOSS_RESULT.md)
2. **What the actor learned (QPROBE):** off-manifold queries (cos 0.45 nearest item, 0.30 nearest concept, 0.00 to u*) =
   interpolations *between* named concepts; NOT a u*-shortcut. Strategy = fixed informative opener (turn0 centroid-cos 1.00)
   then ADAPTIVE per-user queries (turns 1–7 centroid-cos 0.65–0.79). (QPROBE_RESULT.md)
3. **Objective ablation:** distilled-from-discrete COLLAPSES on graded (0.293/0.094 — inherits binary ceiling);
   trained-on-u 0.366/0.162; trained-on-NDCG (recon+softNDCG) best 0.370/0.165. → train NATIVELY, never distill discrete.
4. **Continuity headroom (oracle):** privileged continuous oracle +0.049 tail over discrete oracle, 18% off-catalog picks,
   off-manifold (not item-asking, not u*-shortcut). (PHASE05_HEADROOM_RESULT.md)
5. **Ceilings:** realizable u*=profile-half fold 0.41; full-profile fold 0.42; held-fold 0.43; recommender caps NDCG ~0.43.
6. **Novelty (adversarial lit review):** novel-as-combination; load-bearing individually-novel pillar = off-manifold
   un-snapped query beats discrete (enabled by geometric answer, no NL inversion). Cite defensively: Vendrov/Boutilier
   AAAI 2020, Wolpertinger, ConTS (TOIS 2021), PEBOL (RecSys 2024), Bıyık 2023, DAD (ICML 2021).

## ⚠ BUG LESSON
`pool_entavg.npy` got corrupted (sha 09feb829) during dev → depressed entropy + CASPER-R by ~0.02 (the actor was
unaffected; it doesn't read POOL_ENT). ALWAYS verify baselines reproduce canonical (CASPER-R 0.360/0.152) before claiming.
The CORRECT cache (156c072c) is guarded here and in PAPER_B_WINNER/pool_entavg_CORRECT_156c072c.npy.

## Result docs
experiments/paper2/{HARMONIZED_FIXED_RESULT, SNAPLOSS_RESULT, QPROBE_RESULT, COMPARE4_RESULT, PHASE05_HEADROOM_RESULT,
PHASE2_RESULT, RECON3*, TRAINVAL_DIAGNOSIS}.md. Draft: PAPER_C_DRAFT.md. LaTeX: ../papers/paper3_casper/.
