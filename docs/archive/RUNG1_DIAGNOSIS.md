# RUNG I — value-INERTNESS diagnosis (2026-07-11)

Diagnosis of why the Rung-I answer-encoder pilot went value-inert: zeroing the graded value channel
moved val NDCG by **0.0002** against a **0.01** MDE bar (the 7th fold to hit this wall), and the
intercept gate failed (empty interview → μ ≠ native_z).

**Method (HARD RULE #1 honored):** all micro-experiments use the author-sanctioned **2k train / 1k
val** pilot cohort, the **same `load_split` seed** as the pilot, all users kept, **no caps /
subsamples / top-N / truncation of any kind**. Each config changes exactly ONE knob, 3 epochs each.
Code: `scripts/rung1_diagnose.py` (read-only wrapper; `rung1_encoder.py` untouched). Numbers:
`.cache/rung1/diagnosis.json`, run log `.cache/rung1/diag_run.log`.

---

## Ablation table (value_delta MDE = 0.01)

| Config | value_delta present | value_delta absent | γ spread (L−H) | γ(hated)<0 | IG2 drop | intercept |
|---|---|---|---|---|---|---|
| C0 baseline | −0.0001 [−0.0002, 0.0] | −0.0002 | −0.0001 | no | −0.0000 | 0.114 |
| C1 β=0 (KL off) | −0.0002 | −0.0002 | +0.0049 | no | +0.0000 | 0.243 |
| C2 ordinal ×10 | **+0.0004 [+0.0001,+0.0007]** | +0.0004 | −0.0043 | no | +0.0000 | 0.142 |
| C3 ordinal ×100 | +0.0003 [0.0, +0.0006] | +0.0003 | −0.0047 | no | +0.0001 | 0.180 |
| C4 native always-dropped | +0.0001 | −0.0000 | −0.0046 | no | −0.0000 | 0.105 |
| C5 direct region-aux | +0.0001 | −0.0000 | −0.0026 | no | −0.0000 | 0.090 |
| C6 aux + β=0 | +0.0003 [+0.0001,+0.0007] | −0.0001 | −0.0027 | no | −0.0000 | 0.173 |

**Every** config sits >20× below the MDE, leaves γ(hated) positive (never inverts), and shows a flat
IG2 like/dislike percentile (0.53 vs 0.53, drop 0.000). No single-knob intervention makes value
non-inert.

## Gradient probe — grad-norm reaching the γ **value-slope** (`gamma_head.weight[:,0]`), baseline C0

| term | → γ value-slope | → γ head (full) | → τ head | → ρ_μ last |
|---|---|---|---|---|
| **L_ordinal** (only value-discriminating term) | **2.0e-3** | 9.8e-3 | 1.4e-2 | 0.111 |
| L_consume (likes-only, non-discriminative) | 1.6e-2 | 0.103 | 0.119 | 0.882 |
| L_KL (shrink toward native/prior) | 2.8e-2 | 1.14 | 6.2 | 2.13 |

The ordinal term — the ONLY signal that separates hated from loved — supplies ~7% of the FiLM-head
gradient. KL dominates the γ/τ heads by **100–440×** and consume by ~10×; neither encodes value
polarity.

---

## ROOT CAUSE (ranked)

### 1 — Value is informationally REDUNDANT under the objective (PRIMARY, deep) — HIGH confidence
Given a token's **membership** (which region/item was revealed) plus the consumption anchor, the
**value field carries ~0 marginal information about held-item ranking**, so no reconstruction/ranking
objective assigns γ a value slope. Proof by convergent ablation:
- **Reweighting the ordinal saturates:** ×10 → +0.0004, ×100 → +0.0003 (C2 ≈ C3). Ten-fold more
  weight buys nothing — the held-item objective has an intrinsic ceiling on value info ~25× below MDE.
  So value is **not merely drowned**; the signal on this path is nearly empty.
- **A direct region-ordinal aux fails (C5):** even a loss that explicitly pushes a region's own score
  by the revealed value leaves value_delta +0.0001 and γ spread ~0 — because the aux target (the
  revealed value) equals the user's true taste, which native/membership already encode, so it is
  satisfiable **without** γ.
- **Removing the consumption anchor fails (C4):** native always-dropped still inert → this is not a
  free-ride-on-the-anchor problem.

Mechanistically: the rating-derived emulator makes `value ≈ f(true taste)`, and membership + native
already recover that taste. Training **never** presents value in a *membership-controlled* contrast
(you only ever get a "loved Horror" token *because* the user loves Horror; membership already says
it), so γ is never required to read the value field. This is the same wall the previous six folds hit.

### 2 — Ordinal drowned + KL domination (SECONDARY: real, aggravating, insufficient alone)
The value-discriminating gradient is a 7% minority (table above); KL dominates the FiLM heads 100–440×
and is unstable (raw KL spiked 22→356; 2038 with β=0). This shapes τ by KL/σ pressure rather than
knowledge (hence **τ mis-ordered**: τ(no_clue) 0.899 > τ(know_well) 0.851) and destabilizes training.
But turning KL **off** (C1) leaves value inert — it un-flattens γ a hair (+0.0049, right direction)
and un-shrinks, but does not recruit value. KL is an aggravator and the τ-and-instability culprit, not
the value blocker.

### 3 — ρ_μ zero-init bottleneck delays γ learning (TERTIARY, plumbing)
Both ρ_μ layers are zero-init, so ∂z/∂pool = 0 at t=0 → γ/τ receive **no** gradient until ρ_μ grows.
The easy consumption/native solution is found first; by the time ρ_μ can route pool gradient, value is
already redundant and never gets recruited. Contributes to why even the weak signal isn't used.

### SEPARATE BUG — intercept (definitively diagnosed, trivially fixable)
`empty interview → μ ≠ native_z` because the ρ_μ residual MLP has **bias terms**. Zero-init of the last
layer makes δ=0 **only at t=0**; after training δ = W₂·relu(b₁) + b₂ ≠ 0 even at zero pool. Observed
0.09–0.24 in **every** config. Unrelated to value inertness.

---

## PROPOSED FIX (diagnosis-stage; not a redesign)

1. **Counterfactual value-CONTRAST loss (load-bearing).** For each explicit region/item token,
   materialize the **same entity at ≥2 value levels** {hated, loved} (+meh) as separate forwards with
   **native dropped and other tokens removed**, and supervise the region's own members to rank
   **monotone in value** (up under loved, down under hated) with a margin. Membership is held fixed, so
   **only γ can satisfy it** → forces the γ sign. λ ≈ 0.3–1.0 (comparable to consume). This turns the
   IG2/IG4 gate into a training signal — it defines the value-field *semantics* the natural emulator
   never isolates. Expected: γ(hated) crosses < 0, IG2 drop > 0, IG4 monotone.
2. **KL (secondary, do alongside):** β_max 0.05 → ~0.005 or switch to a capacity / free-bits target so
   KL stops dominating the FiLM gradients; set the KL prior mean to a **fixed 0** (detach native) so the
   consumption channel is not KL-privileged. Fixes the τ mis-ordering and the 356/2038 spikes.
3. **ρ_μ (tertiary):** small-init (not exact-zero) the last layer so γ gets gradient from step 0; keep
   intercept exactness via the token-gate below rather than via zero-init. Optionally a direct γ path
   that bypasses ρ_μ for the contrast loss.
4. **Intercept:** gate the residual by surviving-token presence —
   `μ = native_eff + g · ρ_μ(...)`, `g = (ntok>0)` or `(1 − exp(−ntok))`. Exact at ntok=0 for both the
   empty and native-only cases.

## Confidence & the single biggest risk
- **Root cause (value redundant → objective never recruits γ): HIGH** — four independent ablations
  converge, and the ordinal ×10 ≈ ×100 saturation is the clincher.
- **Fix passes the BEHAVIORAL gates (γ sign / IG2 / IG4): MEDIUM-HIGH** — the counterfactual contrast
  mechanically must move the γ sign.
- **Biggest risk (be faithful):** the **aggregate value_delta ≥ 0.01 MDE may stay unmet even with a
  value-sensitive γ.** Under the rating emulator, sampled interviews are predominantly on-profile /
  positive (value ≈ liked/loved), so replacing value with meh barely changes an already-positive
  ranking — the *dislike* signal, where value matters most, is rare on-profile (regime-absent value_delta
  is also ~0 everywhere). Two honest possibilities then remain: (a) the MDE gate is **mis-specified for
  this emulator** — value should be measured on dislike-heavy / adversarial contexts, or the
  behavioral IG2/IG4 gates should decide (which the design's own gate hierarchy already says are
  PRIMARY); or (b) this is the **frozen-decoder ceiling** the design names as the Rung-II escalation
  branch (dislike expressible only along a few probe directions). Part of the failure may be a genuine
  ceiling, not a bug — that must be stated before spending a full run chasing the aggregate number.
