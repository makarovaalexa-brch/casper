# Rung-I encoder — two training-free decisive checks

**Date:** 2026-07-11 · **Checkpoint:** `.cache/rung1/pilot_best.pt` (val NDCG 0.340, value INERT)
**Cohort:** the pilot's held-out val cohort — 1000 users, 2k/1k split, seed 0 (study 173/300 quarantined via the trU firewall).
**No training. No data reduction** (length-bucketed micro-batching only; all 1000 users, no token caps).
Script: `scripts/rung1_checks.py` · raw: `.cache/rung1/checks.json`

---

## CHECK 1 — Learned vs inherited: decompose the 0.340

NDCG@10 for **native-only** `z = enc_items(revealed liked items)` (ρ + tokens bypassed) vs **tokens-only**
`z = fold with native dropped` (keepA=0; member-bags + implicit only) vs **full** `z = native_z + ρ(pool)`.
ρ-contribution = full − native-only.

| context | n | native-only | tokens-only | **full** | ρ-contrib | revealed native-ids | native empty |
|---------|---|-------------|-------------|----------|-----------|--------------------|--------------|
| k=2     | 1000 | 0.1755 | 0.2401 | 0.2367 | **+0.0612** | 0.25 | 76% |
| k=4     | 1000 | 0.1988 | 0.2411 | 0.2508 | **+0.0520** | 0.48 | 61% |
| k=8     | 1000 | 0.2290 | 0.2448 | 0.2552 | **+0.0262** | 0.96 | 38% |
| **full** (clean) | 1000 | **0.5214** | 0.2721 | 0.5026 | **−0.0188** | 36.2 | 0% |
| mixed-default (pilot op-point) | 1000 | 0.3106 | 0.2507 | **0.3343** | **+0.0237** | — | — |

The mixed-default row reconfirms the pilot: full **0.334** (≈ the 0.340 headline), tokens-only **0.271** (the memory's regime-ii ~0.271).

**Verdict — the encoder is largely COASTING on RecVAE geometry.**
- Where native is available it dominates and the learned residual **does not help**: at full profile RecVAE-encoding the revealed likes alone scores **0.521**, and the full learned fold is **0.503** — the residual is **negative (−0.019)**. The encoder cannot beat "just RecVAE-encode the likes."
- The residual only looks positive when native is *empty*. At k=2, 76% of users have **zero** revealed likes (native-ids ≈ 0.25), so native-only collapses to the popularity prior (0.176); the +0.06 "gain" there is the encoder re-encoding **member-bag/implicit tokens** — still frozen-RecVAE geometry, just of concept/attr bags instead of items (tokens-only alone already gets 0.240 ≈ full 0.237).
- Net learned lift over the strongest inherited signal at the operating point is **+0.024** (mixed) and **−0.019** (full profile). The "learning beyond RecVAE" is marginal and vanishes/reverses exactly where the anchor is informative.

---

## CHECK 2 — Is graded VALUE informative for ranking under the frozen RecVAE decoder?

Training-free hard-wired signed value (bypasses the inert γ):
`z = native_z(revealed LIKES) + η · Σ_expl-tok s(value)·memberbag_emb`, `s = {hated:−1, meh:0, liked:+0.5, loved:+1}`
(band thresholds at CENTERED_FOLD midpoints −2/3, 0, +2/3). **Value-blind** = s≡0 = native alone → the delta is a
training-free `value_delta`. Two answer-context distributions; member weights genome×pop, leak-free.

| context | η | blind (native) | signed | **value_delta** | 95% CI | helps? |
|---------|---|----------------|--------|-----------------|--------|--------|
| on-profile / positive | 2 | 0.2321 | 0.2425 | **+0.0104** | [+0.0012, +0.0199] | yes (barely) |
| on-profile / positive | 4 | 0.2321 | 0.2403 | +0.0082 | [−0.0016, +0.0187] | no |
| on-profile / positive | 8 | 0.2321 | 0.2361 | +0.0040 | [−0.0070, +0.0153] | no |
| **dislike-heavy / adversarial** | 2 | 0.3206 | 0.1555 | **−0.1652** | [−0.1810, −0.1499] | **no (hurts hard)** |
| dislike-heavy / adversarial | 4 | 0.3206 | 0.1181 | −0.2026 | [−0.2190, −0.1867] | no |
| dislike-heavy / adversarial | 8 | 0.3206 | 0.0950 | −0.2256 | [−0.2423, −0.2095] | no |

(on-profile: ~6 explicit tok/user, 9% negative. dislike-heavy: ~59 explicit tok/user, 32% negative.)

### Check 2b — fold-level IG2 on dislike-heavy (η=4, n=781 users with held dislikes)
Percentile shift (signed − blind) of held items; negative = pushed **down**.

| held items | pctile shift | 95% CI | direction |
|------------|--------------|--------|-----------|
| held **disliked** | **−0.540** | [−0.566, −0.513] | pushed DOWN ✓ |
| held **liked**    | **−0.489** | [−0.517, −0.459] | ALSO pushed down ✗ |

**Selectivity differential** (liked − disliked shift) = **+0.051** — liked items are dragged down only 0.05
percentile *less* than disliked items.

**Verdict — value is REDUNDANT / UNUSABLE for ranking under the frozen decoder. This is a Rung-II ceiling, not a Rung-I loss bug.**
- The signed term carries a **weakly-correct sign** (disliked regions *are* pushed down more than liked, by 0.05 percentile) — the sign information exists.
- But under the frozen RecVAE decoder that sign is **not separable**: down-weighting a disliked genre/entity/item drags the user's *liked* held items down almost as hard (−0.489 vs −0.540). The decoder geometry conflates the user's liked and disliked neighborhoods — the project's core like/dislike-conflation blocker, now measured at the fold level.
- Consequence: on exactly the contexts where value should matter most (dislike-heavy), a **perfectly-signed, training-free** value term **destroys** NDCG (−0.165 at the gentlest η, worse as η grows). On-profile it buys at most **+0.010** and only at the smallest η — right at the MDE, not robust.
- The pilot's aggregate `value_delta ≈ 0` was NOT purely an MDE mis-specification: even measured on dislike-heavy contexts and hard-wired to the correct sign, value cannot be turned into a ranking gain. **No Rung-I loss fix (counterfactual contrast, KL rebalance, ρ un-bottleneck) can recover an aggregate the frozen decoder geometry forbids.**

---

## Bottom line

- **Check 1 (learned vs inherited):** the encoder rides RecVAE. Learned residual over "RecVAE-encode the revealed likes" is **+0.024** at the operating point and **−0.019** at full profile; the apparent short-interview gain is just re-encoding member-bags (still RecVAE geometry). Learning beyond RecVAE is marginal, and negative where the native anchor is strong.
- **Check 2 (value informativeness):** value is **redundant/unusable** under the frozen decoder. Correctly-signed, training-free value gives ≤ +0.010 on-profile and **−0.165 to −0.226** on dislike-heavy, because the decoder cannot separate "down-rank disliked" from "down-rank the user's liked neighbors" (selectivity 0.05 pctile).

**The single most decisive number:** dislike-heavy `value_delta = −0.165` (best η) — hard-wired, correctly-signed value **hurts** ranking by 0.165 NDCG on the contexts built for it. → **Rung-II ceiling. Escalate; a Rung-I loss fix will not save the aggregate.**
