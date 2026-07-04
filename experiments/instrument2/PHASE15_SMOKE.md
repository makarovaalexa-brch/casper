# INSTRUMENT 2.0 — Phase 1.5: Elicitation-Compatibility Smoke Test (RecVAE / ML-20M)

Status: DONE (2026-07-04). EVAL-ONLY, foreground, no training, no commits.
Model: replicated RecVAE best checkpoint `.cache/instrument2/recvae_ml20m_best.pt`
(test NDCG@100 0.4346, Phase-1 verified). Data: ML-20M Liang strong-generalization,
10,000 held-out test users. Runner: `scripts/instrument2/smoke15.py`
(OMP/MKL=3 threads, shared box). Raw: `experiments/instrument2/phase15_smoke_results.json`.

Goal: does the replicated recommender behave sanely as an *elicitation* instrument —
i.e. degrade gracefully to a popularity-like floor with no input, improve inside a small
1–8 reveal budget, and expose a smooth, informative latent action space?

---

## Check 1 — Extreme-cold fold sweep (the key gate preview)  ·  **PASS**

For each of the 10,000 test users, `k` items were randomly sampled from their fold-in set,
fed to the encoder, and full-catalogue NDCG@100 computed on their held-out 20%, with the
`k` folded items masked from scoring (fixed sampling seed per `k` for comparability).

| k | 0 (empty enc.) | 0 (prior mean z=0) | 1 | 2 | 4 | 8 | 16 | 32 | full (80%) |
|---|---|---|---|---|---|---|---|---|---|
| NDCG@100 | **NaN** | 0.1364 | 0.1790 | 0.2159 | 0.2518 | 0.2905 | 0.3334 | 0.3719 | 0.4346 |
| ±SE | — | — | .0015 | .0016 | .0017 | .0018 | .0019 | .0020 | .0021 |

Pure-popularity ranker (train counts): **NDCG@100 = 0.1561**.

- **Monotone, no pathology.** Every step rises; nowhere is `k=1` worse than the floor
  (the red flag the gate asks for is absent). Large, useful gains land squarely inside the
  1–8 elicitation budget: 0.179 → 0.291 (+0.112) over the first 8 reveals, ~64% of the way
  from the floor to the full-profile score by k=8, and k=1 alone already clears popularity.
- **Empty-input = prior mean, ≈ popularity.** A *truly* empty encoder input is **NaN**: the
  RecVAE encoder L2-normalizes `x/‖x‖ = 0/0`. The principled "no information" latent is the
  composite-prior mean `z=0`; `decoder(0)` (the decoder bias) gives NDCG 0.1364 and ranks
  items popularity-like (Spearman 0.670 vs train popularity over the top-2000). So the
  empty-input recommender is a popularity-ish floor, as expected — just *slightly below*
  pure popularity (0.136 vs 0.156), not above it.
- **Implementation note (load-bearing for the pipeline):** elicitation must **seed the state
  with `z = prior mean (0)`**, never hand the encoder an empty interaction vector, or it emits
  NaN. Trivial to handle; flagged so it doesn't silently poison a cold turn-0.

*One-line diagnosis:* healthy monotone k-curve from a popularity-like floor with big gains
inside the 1–8 budget and no pathologies — only caveat is the empty-encoder NaN (seed z=0).

---

## Check 2 — Latent-direction sanity (action-space preview)  ·  **PASS**

**(a) Ranking smoothness along random unit directions `q` (200-d), from `z=0`.**
Spearman rank-correlation of the item ranking at `z+αq` vs at `z`, mean over 8 random `q`:

| α | 0.25 | 0.5 | 1.0 | 2.0 | 4.0 | 8.0 |
|---|---|---|---|---|---|---|
| rank-corr vs base | 0.936 | 0.814 | 0.614 | 0.421 | 0.291 | 0.218 |

Small latent moves → small, coherent ranking shifts; the correlation decays smoothly and
monotonically with step size — no chaotic jumps or discontinuities. The latent is a
well-behaved continuous action space to steer over.

**(b) Do graded geometric answers `a = cos(z*, q)` carry signal?** For 20 users, `z*` =
encoder(full profile). Probes `q` are random unit directions; the simulated answer is
`a = cos(z*, q)`; we reconstruct a direction `ẑ = Σ a_i q_i` (a Monte-Carlo estimate of `z*`)
and decode it. Swept over probe budget:

| n_probes | 8 | 32 | 128 | 512 |
|---|---|---|---|---|
| cos(ẑ, z*) | 0.205 | 0.375 | 0.613 | 0.849 |
| NDCG@100 | 0.0528 | 0.1463 | 0.2527 | 0.3256 |

Floor (`z=0`) = 0.0989; ceiling (`decode(z*)`) = 0.4158 (20-user sample).

- **Signal is real and recoverable.** Both the direction recovery `cos(ẑ,z*)` and downstream
  NDCG climb monotonically with the number of geometric answers, crossing the floor between
  8 and 32 probes and reaching 0.85 cos / 0.33 NDCG (approaching the 0.42 ceiling) at 512.
  The geometric `cos` answer channel demonstrably encodes the user's latent direction.
- **Random probing is budget-inefficient (expected, not a defect).** 8 random 1-D projections
  cannot reconstruct a 200-D direction, so the naive 8-probe prototype (0.053) sits below the
  floor (0.099). This is an information-theoretic property of *random* probes in high-d, not a
  missing signal — and it is precisely the motivation for INSTRUMENT 2.0's learned / adaptive
  probe-selection policy over random directions.

*One-line diagnosis:* smooth navigable latent action space, and geometric cos-answers carry
strong recoverable signal about `z*` — naive random probing just needs a smarter (learned)
direction-selection policy to hit budget.

---

## Check 3 — Input-sparsity calibration of `z`  ·  **PASS**

Statistics of the encoder posterior mean `z` (== eval-time latent) across 10,000 test users:

| input | mean ‖z‖ | sd ‖z‖ | mean per-dim var (across users) |
|---|---|---|---|
| k = 2 | 11.497 | 2.858 | 0.5003 |
| k = full (80%) | 12.839 | 3.281 | 0.6475 |

The encoder does **not** collapse to a degenerate near-prior `z` for tiny inputs: at k=2 the
latent norm is 11.5 — only ~10% smaller than the full-profile 12.8, and far from the prior
mean (‖z‖≈0). Per-dim across-user variance stays substantial (0.50 vs 0.65), i.e. two revealed
items already produce confident, user-*specific* latents that spread across the population.
This is consistent with Check 1's k=2 NDCG (0.216 ≫ 0.136 floor): the low-k latent carries
real per-user structure, not a washed-out prior.

*One-line diagnosis:* no near-prior collapse at low k — the encoder yields confident,
user-distinct latents from 2 items; sparsity-calibrated and elicitation-ready.

---

## Verdicts

| Check | Verdict | Diagnosis |
|---|---|---|
| 1 · Extreme-cold fold sweep | **PASS** | Monotone rise from a popularity-like floor, big gains in the 1–8 budget, no pathology; only caveat = empty encoder input is NaN, so seed z=prior-mean(0). |
| 2 · Latent-direction sanity | **PASS** | Latent action space is smooth (rank-corr 0.94→0.22 with step); geometric cos-answers carry recoverable signal (cos(ẑ,z*) 0.21→0.85, NDCG →0.33); random probing is budget-inefficient → motivates learned probe selection. |
| 3 · Input-sparsity calibration | **PASS** | No degenerate near-prior z at low k (‖z‖ 11.5 @k=2 vs 12.8 full; per-dim var 0.50 vs 0.65); confident user-specific latents from 2 items. |

**Overall: the replicated RecVAE is elicitation-compatible.** Graceful popularity floor,
monotone in-budget gains, a smooth navigable 200-d latent, informative geometric answers, and
no low-input collapse. Two carry-forward notes: (i) seed the elicitation state with `z=0`
(never feed the encoder an empty vector — it returns NaN); (ii) random latent probes waste the
budget, so the value of INSTRUMENT 2.0 is in *learned adaptive* probe/question selection over
this latent, not in the answer channel, which already carries the signal.
