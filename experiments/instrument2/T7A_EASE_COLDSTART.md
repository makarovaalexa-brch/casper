# T7a — EASE vs I2 cold-start NDCG curve (kills the "warm-start-only tie" charge)

**Reviewer charge.** The EASE ≈ I2 (RecVAE-d512) tie is a *warm-start* (full-profile) artefact:
on ML-1M full-profile both sit at ~0.554 full NDCG. The reviewer suspects I2's parity with the
EASE-class bar only holds when handed the whole profile, and that under cold-start elicitation with
few reveals the instrument would fall apart (or, symmetrically, that the "tie" hides an I2 loss at
low k).

**Test.** For k = 1..8 revealed liked items per user, score the SAME k-item history two ways and
compare cold-start NDCG@10 (full + Cremonesi tail):

- **I2** — fold the k liked items through the frozen RecVAE-d512 encoder → decode. This is exactly
  P4a's `item8_fold` arm (item-fold, **not** the additive actor). Fidelity: at k=8 this reproduces
  the P4a item8_fold reference (**0.4635 vs 0.4625** full). ✓
- **EASE** — cold-start score `B[revealed_k].sum(0)`, with `B` the trU-trained item-item matrix
  (Steck 2019 closed form, identical to `ml1m_bars.ease_*`), λ=1000 selected on the seed-123 val by
  full NDCG.

**Item-selection convention.** Matches P4a's item8_fold arm, which uses
`np.random.default_rng(sd)` random selection of profile likes. For a *nested* k=1..8 curve we draw
one random permutation of each user's profile likes (seeded per eval seed) and take the first k; the
fold saturates on all available likes when a user has < k (mirrors item8_fold only trimming when
len > 8).

**Protocol (frozen invariants).** Instrument `.cache/instrument2/ml1m_recvae_d512_best.pt`; arena
`ml1m_arena` byte-identical splits; eval seeds {1,2,3,7,11}; te[300:] TEST (304 users); NDCG@10
full + Cremonesi tail. Paired per-user bootstrap (I2 − EASE) reuses `p4a_bootstrap.boot` (5000
resamples, per-user seed-averaged).

Anchors: MOSTPOP ~0.31; EASE **full-profile** (warm) ~0.554 full (ties I2 warm); EASE **val** at
λ=1000 = 0.4886 full / 0.3446 tail.

---

## Cold-start curves — 5-seed TEST mean (sd across seeds), 304 users

| k | I2 full | EASE full | Δ full (I2−EASE) | full 95% CI | I2 tail | EASE tail | Δ tail | tail 95% CI |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.2809 | 0.2267 | **+0.0542** | [+0.045, +0.063] | 0.1425 | 0.1371 | +0.0058 | [−0.001, +0.013] |
| 2 | 0.3384 | 0.2772 | **+0.0612** | [+0.051, +0.071] | 0.1793 | 0.1664 | **+0.0135** | [+0.005, +0.022] |
| 3 | 0.3743 | 0.3130 | **+0.0613** | [+0.050, +0.072] | 0.2095 | 0.1874 | **+0.0226** | [+0.014, +0.031] |
| 4 | 0.4021 | 0.3383 | **+0.0638** | [+0.053, +0.074] | 0.2320 | 0.2017 | **+0.0309** | [+0.022, +0.041] |
| 5 | 0.4194 | 0.3576 | **+0.0618** | [+0.051, +0.072] | 0.2423 | 0.2132 | **+0.0298** | [+0.020, +0.040] |
| 6 | 0.4362 | 0.3760 | **+0.0601** | [+0.049, +0.070] | 0.2588 | 0.2279 | **+0.0317** | [+0.022, +0.042] |
| 7 | 0.4491 | 0.3927 | **+0.0564** | [+0.046, +0.067] | 0.2684 | 0.2387 | **+0.0301** | [+0.020, +0.040] |
| 8 | 0.4635 | 0.4104 | **+0.0530** | [+0.043, +0.063] | 0.2785 | 0.2463 | **+0.0327** | [+0.023, +0.043] |

All Δ full: p(Δ>0) = **1.000**. Δ tail: p = 0.95 at k=1 (marginal), **1.000** for all k ≥ 2.
Seed sd is small (≈0.005–0.011 full across the 5 seeds); every full-CI and every k≥2 tail-CI
excludes zero.

---

## VERDICT — the tie is warm-start-ONLY, and it resolves in I2's favour

**I2 significantly beats EASE at EVERY cold-start k = 1..8**, on full NDCG (Δ +0.053 … +0.064,
all p = 1.000, CIs exclude 0) and on tail from k ≥ 2 (Δ up to +0.033, p = 1.000; k=1 tail marginal
p = 0.95). The margin does **not** shrink at low k — on full it is if anything *largest* at low-to-mid
k (peak +0.064 @ k=4) and only narrows slightly as k→8 where both curves climb toward the shared
warm-start ceiling.

So the EASE ≈ I2 parity is **purely a warm-start (full-profile) phenomenon**: the two recommenders
converge to ~0.554 *only* when each is handed the entire profile. The moment you operate in the
actual deployment regime — few revealed items, cold-start elicitation — the RecVAE-d512 instrument
extracts **strictly and significantly more** from the identical k revealed items than the EASE-class
bar does. This **kills** the reviewer's charge: the warm-start tie is not evidence that I2 is only
EASE-equivalent; under cold-start (the regime the whole elicitation thesis lives in) I2 dominates
EASE across the entire k=1..8 curve.

### Durable artifacts
`.cache/instrument2/t7a_ease_coldstart.json` (curves + per-k paired bootstrap). Script
`scripts/instrument2/t7a_ease_curve.py` (imports `ml1m_arena`, `ml1m_bars` EASE, `p4a_battery`
fold/model helpers, `p4a_bootstrap.boot` — no EASE/fold logic duplicated).
