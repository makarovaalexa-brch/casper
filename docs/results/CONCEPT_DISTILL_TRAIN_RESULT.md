# Concept-fold DISTILLATION training — POC result (2026-07-27)

Single-seed proof-of-concept of `DESIGN_CONCEPT_FOLD_DISTILLATION.md`. Trainer
`src/instrument/concept_distill_train.py` (committed 88144f2 + fix 6270969). Frozen i25 tower
`t2i25_EP4_SNAP.pt` + frozen decoder + gated zero-init fold-to-point `ConceptFoldNet`; teacher-latent
caching (base fold(in_s) + teacher fold(in_s∪members) precomputed once, bit-exact-verified via
`allclose`). Canonical ML-25M Liang split, 10k COLD_SEED cohort, full+tail NDCG@10, intercept
0.12794/0.01923 (snap held), G0 item bit-identity held for every config. Results JSON
`experiments/battery/concept_distill_train.json`.

## Headline 1 — the trained student BEATS the tabular floor in every tier

Capture-rate = (student − intercept)/(teacher − intercept). Step-0 tabular floor in parentheses.
Best deployment fold = **λ=1.0 (`cd_s1_l10`)** (see Headline 3).

| tier | student kc1 (floor) | student kc4 (floor) | teacher ceiling kc4 |
|---|---|---|---|
| overall | 13.5% (11.5%) | 25.4% (**15.3%**) | 0.4005 |
| fine    | 34.9% (21.6%) | 34.6% (**31.8%**) | 0.3165 |
| medium  | 33.5% (21.7%) | 31.4% (**25.7%**) | 0.3346 |
| broad   | 12.1% (10.3%) | 23.4% (**13.5%**) | 0.4039 |

All 8 tier×kc cells clear the floor. Biggest win: **broad@kc4 23.4% vs 13.5% floor (+10 pts)** — the
Step-0 drag tier recovers most, via multi-concept composition (kc>1 joint fold). fine/medium already at
~31–35% (near/at the ~30% design target). No few-shot crater (kc1 all ≥ floor > intercept on the
selection cohort). Deployment concepts-only curve monotone-rising, opener > intercept (λ=1.0).

## Headline 2 — the DISTILLATION teacher is REDUNDANT (claim NULL)

The same-budget no-distill control (λ=0, plain NDCG-NLL on the identical member-drop kc-curriculum)
captures the SAME as the distilled students; the shuffled-teacher canary too:

| config | overall kc4 capture_full | note |
|---|---|---|
| cd_s1_l01 (λ=0.1) | 25.7% | opener gate FAIL |
| cd_s1_l10 (λ=1.0) | 25.4% | opener gate PASS ← deployment fold |
| cd_s1_l100 (λ=10) | 25.0% | opener gate FAIL |
| **cd_ctrl_nodistill (λ=0)** | **25.5%** | no KL term at all |
| cd_canary_shuffle (random teacher) | 25.9% | no leakage; ≈ control |
| cd_s2 (distill-then-sharpen) | 26.1% | best capture, opener gate FAIL |

Distill − no-distill delta is −0.2 to +0.3 pts across all tiers/kc — **within noise**. The fix to the
lossy 1–6% fold comes from the **training RECIPE** (gated fold-to-point architecture + member-drop
kc-curriculum + NDCG-NLL), **NOT from the distillation teacher**. This is exactly what Step-0's
within-cell teacher cos-to-mean = 0.95 predicted: the shortfall was magnitude/ranking dilution, and a
trained NDCG-aware student recovers it directly — the dense KL target adds nothing on top of the rank
loss. Do NOT claim the distillation. The canary confirms no leakage / gate-opening.

## Headline 3 — S1 λ=1.0 is the deployment fold (opener gate)

Capture is flat across λ (25.0–25.7%, noise). The **coarse-to-fine opener gate** (concepts-only kc1 >
intercept) discriminates: only **λ=1.0** passes (curve [0.132, 0.145, 0.160, 0.176], kc1 0.132 >
0.1279). λ=0.1 and the control sit just below intercept at kc1; **S2's phase-2 sharpening destroys the
opener** (kc1 0.1027 ≪ intercept — best capture but not deployable as an opener). So the KL term buys no
capture but does keep the low-kc opener above intercept when weighted (λ=1.0) — a small deployment role.
⇒ downstream chain (D1/D2/D3) uses `cd_s1_l10_best.pt`.

## Method / speedup / controls
- Teacher-latent caching (frozen tower ⇒ deterministic): materialize the seeded kc-curriculum once,
  precompute base + teacher 200-dim latents (disk-cached, `allclose` bit-exact verify, max|Δ|≈1.8e-6 from
  batch padding width — not sampling). Epochs decode cached latents only ⇒ **4.9 min/epoch (was ~27)**.
- Controls PASS: G0 item bit-tie (all configs), q0 snap 0.12794/0.01923, leak known∩held=0,
  no-distill control (a), shuffled-teacher canary (b), fixed SEL answer train==eval.

## Verdict
Plan is a SUCCESS on its primary goal — the concept fold is fixed (floor beaten every tier, broad@kc4
+10). The distillation mechanism itself is the wrong attribution: the recipe, not the teacher, does the
work. Design §5 fallback (richer belief-conditioned input) NOT needed. Downstream re-measure of the
concept-vs-item story, answerer panel, and adaptivity on the fixed λ=1.0 fold = the D1/D2/D3 chain.
