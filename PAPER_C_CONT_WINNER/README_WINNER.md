# 🏆 PAPER C WINNER — Continuous-Action Preference Elicitation (DO NOT OVERWRITE)

**Date locked:** 2026-06-29. **Branch:** recommender-improvement.

## The checkpoint
- `policy_cont_actor_WINNER.pt` (sha `874163eb`) — the **continuous actor** (MLP belief∈R⁶⁴ + turn → query q∈R⁶⁴).
  Trained FROM SCRATCH (no distillation) by **differentiable unroll** with the reconstruction objective `1−cos(u,u*)`,
  CONTMODE=cont (off-pool fold, no snap), GRADED geometric answers. Frozen V1 encoder.
  Reproduce: `CONTMODE=cont OBJ=ustar GRADED=1 NOBC=1 VALTEST=1 SELVAL=tail SEED=1 python scripts/paper2/continuous_actor.py`
- `policy_cont_actor_ndcg_variant.pt` — the marginally-better variant (recon + soft-NDCG combo): graded 0.370/0.165.
- `cache/` — the EXACT frozen recommender deps: enc_concept.pt (V1, sha `27e6c72d`), Ql_concept.npy, Ec_concept.npy,
  Q_svd.npy, bi_svd.npy, ctags_concept.npy, **pool_entavg.npy (sha `156c072c` = CORRECT; a corrupted `09feb829`
  version depressed baselines during dev — see below)**.

## THE RESULT (harmonized to the canonical Paper B ruler; baselines reproduce paper EXACTLY)
Canonical Paper B harness (`continuous_policy2_st.py`, `contactor` mode), seed-avg {1,2,3,7,11}, te[300:], q8:
| policy | FULL | TAIL |
|---|---|---|
| **continuous actor (graded)** | **0.3664 ± 0.0029** | **0.1622 ± 0.0051** |
| CASPER-R (binary, Paper B SOTA) | 0.3602 ± 0.0033 | 0.1522 ± 0.0025 |
| entropy (binary) | 0.3608 ± 0.0016 | 0.1398 ± 0.0040 |
| conc_pop (binary) | 0.3466 ± 0.0027 | 0.1266 ± 0.0045 |

**WIN: +0.006 FULL (~2σ, marginal; full is popularity-saturated) / +0.010 TAIL (~3σ, significant; Paper B headline metric).**
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
