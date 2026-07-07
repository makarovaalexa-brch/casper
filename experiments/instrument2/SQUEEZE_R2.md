# INSTRUMENT 2.0 — "squeeze arc" rung 2: the LINCHPIN noise-channel test (adaptive vs a FAIR noise-adapted static)

Foreground, chunked, no commits. Instrument = certified RecVAE-d512 (`ml1m_recvae_d512_best.pt`,
frozen). Arena `ml1m_arena` (byte-identical V1 splits), eval seeds {1,2,3,7,11}, te[300:] TEST
(304 users), T=8, additive operator `z'=z+eta*a*q` (eta=16). Metric NDCG@10 full + Cremonesi tail.
Empirical channel `.cache/instrument2/p4c_channel.json` (corr(s,rating)=0.516, mean bin entropy 1.887
bits, `a≈1.55s−0.36`, σ≈0.48). Answers scale-matched + per-user centered exactly as in
`p4c_answer_sources.py`. Runner `scripts/instrument2/squeeze_r2.py`, artifact
`.cache/instrument2/squeeze_r2_noise.json`.

## Why this rung exists

P4C established that the noiseless-trained actor **collapses** under the realistic channel
(0.491 → 0.150) and that **training under the channel recovers it to 0.287** (+0.139) — and it read
this recovery as *"adaptivity survives conditional on retraining."* But that recovery was only ever
compared to **clean-era static baselines** (SVD-8 top-8 basis = 0.159), i.e. a static that was NOT
adapted to the deployment channel. The comparison was **unfair**: squeeze_r01 already showed
(RUNG 1d) that the *right* static behavior under noise is to **repeat-probe the top informative axis**
(D-opt(decoder) R=32 schedule `[0,0,1,0,1,0,2,3]` holds 0.248 while the actor sits at 0.153) — a
trick the fixed clean SVD-8 (8 distinct dirs) never gets to use. This rung builds the **strongest fair
noise-adapted static** and asks: **does the actor's adaptive premium survive against it, or is it a
clean-channel-only artifact?**

## Setup

- **Noise-trained actors** `p4c_actor_noisy_s{0,1,2}.pt` (seed-avg; best_val_tail 0.139 / 0.142 / 0.139
  @ ep 15 / 8 / 16), evaluated under the sampled empirical channel (noise×1), scale-matched
  (`native_scale_for_actor`), per-user centered — exactly the eval loop at the bottom of
  `part1_trainnoisy`.
- **Noise-adapted static** = greedy forward-selection of an 8-direction schedule from the
  **decoder-SVD candidate pool** (top-32, the same informative basis squeeze_r01's D-opt(decoder) arm
  uses), each slot chosen to **maximize full NDCG on a NOISY val cohort** (te[:300], sampled empirical
  channel, CRN across candidates) — selection happens **under the deployment channel**, not the clean
  channel. Two variants: **distinct-8** (no repeats) and **repeat-allowed** (may re-probe a selected
  axis to average down channel noise). Winner picked on the noisy val.

## Greedy selection (on the noisy val, te[:300])

| variant | selected schedule (decoder-SVD indices) | val full (final) |
|---|---|---|
| **distinct-8** | `[0, 28, 27, 23, 5, 15, 21, 12]` | **0.190** (peaks at 0.261 with **just dir 0**, then *dilutes* — every added distinct probe injects fresh channel noise) |
| **repeat-allowed** | `[0, 0, 0, 0, 1, 2, 7, 0]` | **0.292** (re-probes the top informative axis **5×**) |

The repeat-allowed greedy **independently rediscovers the squeeze_r01 D-opt(decoder) R=32 behavior** —
hammer the single most-informative direction to average down the 1.9-bit channel noise, then spend a
couple of probes on the next axes. Distinct-8 cannot do this and is strictly dominated. **Val winner =
repeat.**

## TEST — 5-seed, sampled empirical channel (noise×1), scale-matched

| arm | full | tail | note |
|---|---|---|---|
| **noise-adapted static, REPEAT `[0,0,0,0,1,2,7,0]`** | **0.3143** (sd .0076) | **0.1167** | the fair static — repeat-probe top axis |
| noise-trained **actor** (seed-avg s0/s1/s2) | **0.2844** (sd .0039) | 0.1010 | P4C's "recovered" adaptive policy |
| noise-adapted static, distinct-8 | 0.2069 (sd .0074) | 0.0582 | dilution under noise |
| *clean-era SVD-8 (top-8 basis, P4C's comparator)* | 0.1511 (sd .0110) | 0.0794 | the **handicapped** static P4C used |

**Paired bootstrap (304 users, seed-avg per-user full NDCG):**

| margin | Δ full | 95% CI | p(Δ>0) |
|---|---|---|---|
| **actor − noise-adapted static (repeat)** | **−0.0299** | [−0.0403, −0.0202] | **0.000** |
| actor − noise-adapted static (distinct-8) | +0.0775 | [+0.0651, +0.0896] | 1.00 |
| actor − clean-era SVD-8 (P4C comparison) | **+0.1333** | [+0.1177, +0.1489] | 1.00 |

## VERDICT — adaptivity does NOT survive under the realistic channel once the static is also noise-adapted.

**The actor's noisy-channel "recovery" was an artifact of an unfair comparator.** P4C's headline
(train-noisy actor 0.287 ≫ clean-era SVD-8 0.159, +0.128) is reproduced here (+0.133, p=1.0) — but the
clean SVD-8 was a static handicapped into using 8 distinct probes it should never use under noise.
Give the static the **one adaptation the channel actually rewards** — repeat-probe the top informative
axis to average down noise, greedily selected *under the deployment channel* — and the **static wins**:
**0.314 vs 0.284 full (Δ −0.030, CI [−0.040, −0.020], p<0.001) and 0.117 vs 0.101 tail.** The learned
adaptive policy is **significantly beaten by a zero-training fixed schedule of eight probes** (four on
SVD-0, then SVD-1/2/7, then SVD-0 again).

The whole "adaptive edge" of the actor lives in the **clean channel** (P4a: actor 0.491 vs SVD-8 0.471,
+0.019). Under realistic answer noise, the per-turn belief `z_t` the actor conditions on is
noise-corrupted, so its adaptivity becomes feedback instability; the robust move is not "adapt the
direction to the belief" but "**re-measure the same informative direction and average**," which is a
*static, non-adaptive* schedule. Consistent with squeeze_r01's RUNG-1d reading ("robustness is the
schedule, not the fusion / the policy").

**Wording rule for Papers C/D.** The learned-policy / adaptive-continuous claim must be scoped to
**clean answers only**: *adaptivity beats the strongest static under a high-fidelity (noiseless
geometric) answerer; under a realistic empirical answer channel the advantage inverts — a zero-training
noise-adapted static (repeat-probe the top informative axis) beats the noise-trained actor by
+0.030 full (p<0.001).* Do **not** claim the noise-trained actor "recovers adaptivity" — it only
recovers vs a static that was denied the same channel adaptation.

---

## R2b — TIE-BY-CONSTRUCTION arm (is the R2 loss an optimization gap or an expressiveness gap?)

**Motivation.** The noise-adapted static won because it was found by **direct greedy SEARCH** on the
noisy-val objective, whereas the actor was fit by **SGD through the sampled channel** — a weaker
optimizer. But the actor architecture (belief-conditioned MLP → unit direction) *can* emit any fixed
schedule. If we **construct** the actor from the winning schedule, does it at least tie the static?
That isolates optimization gap (SGD is weak) from expressiveness gap (the actor cannot represent the
policy). Runner `squeeze_r2.py r2b`.

**Construction.** BC-warm a fresh `P.Actor` to emit, per turn, the exact winning repeat schedule of
decoder-SVD directions `[0,0,0,0,1,2,7,0]` (belief-independent target). Fidelity check: emitted
direction · target per turn = **[1.0]×8** — the actor reproduces the static schedule *exactly* by
construction. Then two arms: **construct-only** (no fine-tune) and **construct+fine-tune** (20-epoch
noisy-channel unroll fine-tune on top of the construction, val-selected on noisy-val full NDCG).

| arm | VAL (te[:300]) full/tail | TEST 5-seed full/tail |
|---|---|---|
| **tie construct-only** (BC-warm to schedule) | 0.2857 / 0.0932 | **0.3177** (sd .0043) / **0.1168** |
| tie **fine-tuned** (SGD on top) | 0.2980 / 0.0742 | 0.2925 / 0.0974 |
| noise-adapted static-repeat (ref) | 0.2923 / 0.0789 | 0.3143 / 0.1167 |

**Paired bootstrap (304 users, seed-avg per-user full):**

| margin | Δ full | 95% CI | p(Δ>0) |
|---|---|---|---|
| **tie construct-only − static-repeat** | **+0.0035** | [−0.0043, +0.0112] | **0.815** (TIE) |
| tie fine-tuned − static-repeat | −0.0218 | [−0.0340, −0.0098] | 0.000 |
| tie fine-tuned − tie construct-only | −0.0253 | [−0.0386, −0.0122] | 0.000 |

### R2b VERDICT — the actor CAN at least tie the static; R2's loss was an OPTIMIZATION gap, and fine-tuning ADDS NO value (it destroys it).

1. **"At least tie" property CONFIRMED.** Constructed from the winning schedule, the actor emits
   identical directions (cos=1.0) and reaches **TEST 0.318/0.117 ≈ static 0.314/0.117**
   (Δ +0.0035, p=0.82 — a statistical tie, if anything marginally above). On val it ties too
   (0.286 vs 0.292, within the vectorized-vs-per-user noise-draw difference). The R2 defeat of the
   *trained* actor (0.284) was therefore **not an expressiveness limit** — the architecture fully
   represents the noise-robust repeat-probe policy; P4C's `trainnoisy` SGD simply failed to find it.

2. **Fine-tuning does NOT unlock adaptive upside — it is actively HARMFUL.** Starting *exactly at the
   optimum* (the constructed schedule) and fine-tuning under the noisy channel drifts the policy
   **away** from it: TEST 0.318 → 0.293 (Δ −0.025, p<0.001), tail 0.117 → 0.097. The val-selection even
   chose the final epoch (val-full 0.308) that generalized to only 0.293 — SGD's apparent val gain was
   val-overfit noise. **There is no realizable adaptive gain on top of the fixed schedule under this
   channel; gradient descent through the noisy unroll systematically corrupts the robust static
   optimum.** This strengthens R2: the noise-robust optimum *is* the non-adaptive repeat-probe
   schedule, and belief-conditioned adaptivity is not merely unhelpful but a liability under noise.

**Net for Papers C/D.** The learned actor is *expressive enough* to match the strongest noise-adapted
static (state it as an optimization, not capacity, story) — but under a realistic answer channel the
best it can *do* is tie a zero-training fixed schedule, and any attempt to train adaptivity on top of
that schedule loses. Adaptivity's positive value remains **clean-channel only**.

### Durable artifacts
`.cache/instrument2/squeeze_r2_noise.json` (greedy schedules + val curves, R2 5-seed TEST + bootstraps,
and `r2b` block: construction fidelity, val + 5-seed TEST for both tie arms, bootstraps);
`.cache/instrument2/p4c_actor_tie_ft.pt` (fine-tuned tie actor); script
`scripts/instrument2/squeeze_r2.py` (`r2` + `r2b` stages, imports helpers from `p4c_answer_sources` /
`p4a_battery` / `p4a_bootstrap`). Noise-trained actors `p4c_actor_noisy_s{0,1,2}.pt`.
