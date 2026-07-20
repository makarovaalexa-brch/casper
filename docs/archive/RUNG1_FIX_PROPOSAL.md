# RUNG I FIX PROPOSAL — hard-wire the value sign (2026-07-11)
For adversarial Fable review before rebuild. Context: DESIGN_SHEET_RUNG1_ENCODER.md (the design that
went inert), .cache/rung1/diagnosis.json (the 7-config ablation battery), pilot_result.json.

## DIAGNOSIS (proven by exhaustion — 7 configs, all inert)
The value channel is inert because the design makes the network LEARN the value sign. γ is a signed
FiLM head, identity-init (+1), that must travel across zero to negative for "hated." The battery:
| config | value Δ (bar 0.01) | γ(hated) | hated<0 | IG2 flip |
|---|---|---|---|---|
| C0 baseline | −0.0001 | +0.975 | no | 0.000 |
| C1 β=0 | −0.0002 | +0.983 | no | 0.000 |
| C2 ordinal ×10 | +0.0004 | +0.971 | no | 0.000 |
| C3 ordinal ×100 | +0.0003 | +0.966 | no | 0.000 |
| C4 drop-native | +0.0001 | +0.977 | no | 0.000 |
| C5 aux (direct) | +0.0001 | +0.973 | no | 0.000 |
| C6 aux+β=0 | +0.0003 | +0.975 | no | 0.000 |

ROOT CAUSE (grad-probe evidence): the gradient to the value head is PATH-LIMITED, not weight-limited —
`grad[ordinal]→γ_valslope` = 0.002 → 0.0007 → 0.0007 across ×1/×10/×100 (weight-independent); even the
direct aux only reaches 0.005; all drowned by consumption (0.013) and KL (0.05–3.3). γ never leaves
identity. BUT build-sanity PROVED the forward path inverts perfectly when γ is hand-set negative
(hated pool −0.30 < loved +0.03). => LEARNABILITY failure, not expressiveness. Learning the sign from a
tiny path-limited gradient starting at +1 is not reachable in any tested regime.

## THE FIX — signed value multiplier (native polarity), sign NOT learned
Replace the LEARNED signed-γ value gate with a HARD-WIRED signed multiplier from the KNOWN value:
  token contribution to pool  =  s(value) · τ(knowledge) · member_bag_emb
- **s(value)** = the KNOWN centered valence, e.g. {hated −1, meh −⅓ (or 0), liked +⅓, loved +1}. NOT
  learned. "hated Sci-Fi" → −1 · Sci-Fi_emb → the region inverts BY CONSTRUCTION. Grading preserved
  (the scale is graded); only sign DISCOVERY (the thing that failed) is removed.
- **τ(knowledge)** = learned POSITIVE precision scalar per {no_clue<rough<know_well} (softplus). Still
  learned — magnitude/confidence is a real learnable, the SIGN is not.
- Still learned: ρ MLP, concept member-bags, τ. The decoder stays frozen (Rung I).
GROUNDING: this is exactly (a) the frozen-RecVAE valence probe `z0 ± η·d_X` (provably works, symmetric
17.7 vs 17.8 pct suppression/elevation); (b) the pre-VAE instrument's native signed `y∈{−1,+1}`;
(c) build-sanity's hand-set-negative inversion. Three independent confirmations the mechanism works.

## SECONDARY FIXES (real, but not the inertness cause)
- KL instability: β spiked to 356/1652 (Lkl). Cap βmax low + free-bits, or drop KL to a light prior
  regulariser. (C1 showed β=0 alone doesn't fix inertness, but the KL is genuinely unstable.)
- Intercept bug: ρ zero-init holds only at init (post-train dev 0.11–0.24). Gate δ by an evidence
  indicator so empty interview → μ=native_z exactly regardless of trained weights.

## OPEN QUESTIONS FOR FABLE (attack these)
1. Does hard-wiring the sign LOSE anything vs learning it? (Claim: no — we KNOW the sign; grading lives
   in the ±magnitude + τ. Is there a case where a learned sign would help — e.g. an entity whose
   "member-bag direction" points the wrong way in z, so +1·emb should actually DOWN-rank?)
2. Is a fixed s(value) scale right, or should MAGNITUDE be learned (per-level scalar, still signed by
   the known sign — sign fixed, magnitude free)? This keeps the fix's spirit (sign given) while letting
   the model calibrate how far hated pushes vs loved.
3. Does the signed multiplier reintroduce the value–entity COMMUTING risk? (It multiplies the entity
   emb per token before SUM, so sign binds to entity — same guard as the FiLM design. Confirm.)
4. Now that the sign expresses, will value actually IMPROVE held-out NDCG@10, or only move IG2? (The
   re-pilot must show value_delta ≥ MDE AND a real IG2 flip AND G-clean, not just polarity motion.)
5. Concepts/attributes: their member-bag direction may not be as clean as genres (probe: item-level
   lower-SNR). Does hard-wired sign on a noisy concept direction help or hurt? Gate per-channel.
6. Anything that makes this the 8th failure — the single most dangerous residual risk.
