# DESIGN SHEET — Concept FOLD-IN encoder on frozen tower (2026-07-17)  [AWAITING SIGNATURE]

## Question
Does a gated zero-init concept FOLD-IN encoder (fold answers to a POINT in factor space, Paper-B style) on the
FROZEN paord tower beat the linear V4 additive head — i.e. do concepts COMPOUND (non-saturating k-curve) instead
of saturate — while never dropping full NDCG?

## Why (diagnosis, this session)
- V4 additive head `Σ_c w_c·<fixed_dir_c>` scores the UNION of concept directions → redundant tags saturate
  (measured: kc=8 +0.0031, all ~470 +0.028 tail).
- Paper B folds answers to a POINT u; frozen decoder applies → INTERSECTION → COMPOUNDS (measured on u1_best,
  same ndcg10: kc=8 +0.0037, kc=16 +0.027, kc=32 +0.052 tail). Frozen-tower fix is proven (Paper B froze Q/decoder).
- fusj failed because it folded to a belief with NO popb floor (single concept displaced popularity). Add the floor.

## Architecture (item tower + decoder + item-fold ALL FROZEN)
    score(i) = s_base(i) + gate · <Wd_i, u>
    u = AttnPool_c( Enc_token(  whitened_centroid_c ,  answer_embed(g_c) ) )   # fold answers to ONE point
- s_base = frozen cold score (popb / paord empty-fold — pick popb per geo2 evidence it's the stronger cold base).
- Enc = small per-token MLP (in ~ [512 whitened dir ; value/conf embed] → 128 → 128) + softmax attention pool +
  value head → 64-or-512-d point u in decoder factor space. ~50k-100k params (Paper B scale).
- gate scalar, ZERO-INIT → u contributes 0 at step0 → score == intercept EXACTLY (never-drop safety, provable).
- Answers graded (love/like/hate → residual-style value embed); refusals foldable (kept as signal).
- H2 hedge (only if it underperforms): route <Wd_i, u> through a low-rank whitened projection of Wd (all-but-top).

## Exact Ns (NO reduction — HARD RULE #1)
- Train: all usable mm_train users (~150k). Full 18,430 multinomial softmax (no candidate pool).
- Concepts: all 1,628. PCONLY-style mix: ~35% concept-only batches (in-distribution cold concept folds) + rest
  items+concepts. Answered concepts per user folded as a SET (not top-k) so attention learns to weight/discount.
- Selection + report on DISJOINT val half. 173/300 study users QUARANTINED. No LLM calls.

## Loss / training
- Multinomial CE over held-LIKED items on (s_base + gate·<Wd,u>), s_base frozen. Concepts-only masked batches
  force the fold to be load-bearing. Guarded selection: keep best TAIL subject to FULL ≥ intercept − 0.003.

## Baselines / gates (same ruler as V4)
- Intercept (gate=0): must equal popb intercept exactly (structural).
- V4 additive head k-curve (the bar to beat): kc=1 +0.0012 / 3 +0.0018 / 8 +0.0031 tail.
- Paper B u1 k-curve (aspirational, different split): kc=8 +0.0037 / 16 +0.027 / 32 +0.052.
- PASS (Fable prediction): kc=8 concept-only tail ≥ +0.010 (≥3× V4), NON-saturating k-curve.
- Knockout (gate→0 returns to intercept); super-additivity probe (low-overlap loved pair NDCG > sum of singles);
  compose with item channel (k>0 items must not dilute).

## Metric definition (paper2 has TWO tail defs — pick ONE, stated)
- TAIL = NDCG@10 with head (popularity) items masked, our arena `ndcg10(..., tail=True)`. (NOT poprank-500.)

## Shortcuts flagged
- 3000-val nightly probe = direction only; headline on full disjoint half.
- Point dim (64 vs 512): try 512 first (matches Wd); 64 only as the H2 low-rank hedge.

## Kill list
- V4 additive head (`train_v4head.py`) and belief-token fold (`train_wmat.py`) — both wrong/weaker operator class.

---
SIGN-OFF: __________________________  (author)  — training does not start until signed.
