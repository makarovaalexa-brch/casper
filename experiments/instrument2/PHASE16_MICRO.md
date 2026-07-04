# INSTRUMENT 2.0 — Phase 1.6: Elicitation Micro-Battery (RecVAE / ML-20M)

Status: DONE (2026-07-04). EVAL-ONLY (only tiny per-user z-optimizations in Exp B),
foreground, 3 threads (shared with the Phase-1 Mult-VAE run), no commits.
Model: `.cache/instrument2/recvae_ml20m_best.pt` (test NDCG@100 0.4346, Phase-1 verified).
Runners: `scripts/instrument2/prep_concepts.py`, `micro16.py`, `micro16b.py` (follow-up).
Raw: `phase16_micro_results.json`, `phase16_micro_followup.json`.
Conventions from Phase 1.5: seed empty state with z=0; never hand the encoder an
empty interaction vector (NaN).

Concept vocabulary: 200 highest-coverage ML-20M genome tags (genome files at
`data/movielens/genome-*.csv`; ML-25M genome shares movieIds with the ML-20M vocab —
13.66M relevance rows land on 12,110 of the 20,108 vocab items = 60.2% coverage).
Concept bag for a tag = top-M member items by relevance, L1-normalized.
Eval subsample: 2,000 test users (500 for Exp B / dislike-specificity), NDCG@100 on
held-out 20%, fold-in items masked.

---

## Exp A — Concept pseudo-item folding · **FAIL** (naive input-space; z-space dislike is the salvage)

### (i) Fidelity — concept bag ONLY vs z=0 floor

Per-user tag = argmax fold-in affinity (`Σ` relevance over the user's fold-in items);
"strong" = above-median affinity concentration.

| M | concept-only NDCG@100 (all) | (strong) | z=0 floor (all / strong) | Δ vs floor |
|---|---|---|---|---|
| 10 | 0.0872 | 0.0596 | 0.1546 / 0.1181 | **−0.0673** |
| 50 | 0.0746 | 0.0495 | 0.1546 / 0.1181 | **−0.0800** |
| 200 | 0.0527 | 0.0325 | 0.1546 / 0.1181 | **−0.1018** |

Every M is **below** the prior-mean floor, monotonically worse with larger M, and
strong-affinity users are *worse*, not better. Two artifact checks (`micro16b.py`):

- **Generic-tag degeneracy is real but not the whole story.** Raw affinity picks the
  contentless tag "original" for 87% of users (then "imdb top 250", "good"). Re-selecting
  by **lift** (affinity / global tag mass) yields sensible tags (classic, masterpiece,
  romantic comedy, thriller...) and improves fidelity — M=10: 0.0872 → **0.1135**;
  M=50: 0.0746 → 0.0984 — but it **still sits below the 0.1546 floor**.
- Diagnosis: a bag of M pseudo-items with L1 weights is, after the encoder's input
  L2-normalization, a *dilute fake profile of items the user never chose*. The encoder
  treats it as evidence about those specific M items, and the resulting z is a worse bet
  than the popularity-shaped prior. The channel carries some tag signal (lift >> raw,
  small-M >> large-M) but not enough to clear even the no-information floor.

### (ii) Additivity — k=2 items + 2 concept answers vs k=2 items alone (M=50, strength 1.0)

| | items only | + 2 concept bags | Δ |
|---|---|---|---|
| all | 0.2594 | 0.2325 | **−0.0269** |
| strong | — | — | **−0.0534** |

The concept channel does not add — it **subtracts**, and subtracts most for exactly the
users whose tag affinity is strongest. The dilute pseudo-items pollute the two genuine
item reveals inside the shared L2-normalized input.

### (iii) Dislike preview — negative-weight bag

- **Input-space subtraction (clamp at 0) is a structural NO-OP**, not merely weak:
  bag weights (~0.02 after L1-norm) live almost entirely on items *absent* from the
  input, which clamp back to 0; relative input mass changed = **0.00002**. Measured
  demotion of tag members = −0.000 across all 5 tags tested (horror, violence, romance,
  comedy, dark). A non-negative multinomial input cannot express "not X".
- **z-space subtraction WORKS and is specific** (`z ← z − s·‖z‖·unit(z_bag)`): at s=0.25,
  horror-member items drop **−15.4 percentile-rank points** vs **−0.9** for a top-ranked
  non-member control set, with modest collateral NDCG cost (0.2708 → 0.2454); s=0.5
  doubles the demotion (−30.5 pts) at a larger cost (→ 0.2021). Direction is targeted;
  strength trades off against overall accuracy.

**Verdict: FAIL** (input-space concept pseudo-items: fidelity below floor, additivity
negative, dislikes inexpressible). **Design implication for Phase 2/3:** the concept
channel must not enter through the interaction vector — map concepts into the *latent*
(learned tag→z embedding, or use the tag's member-item decoder directions), where the
crude z-subtraction already demotes a disliked tag specifically; and select/weight tags
by lift, never raw affinity (generic-tag degeneracy).

---

## Exp B — Inference derivation sweep · **PASS** (amortized wins; no amortization gap)

500 test users (≥8 fold-in items), NDCG@100 per (k, derivation). Optimization =
MAP over z of the multinomial likelihood of the k observed items with the standard-normal
prior term at the model's own KL weight (γ·k, γ=0.005), Adam lr 0.05. Wall time is per
500-user batch on 3 CPU threads.

| k | amortized (sec) | optim z=0, 100 steps (sec) | hybrid: enc-init + 20 steps (sec) |
|---|---|---|---|
| 1 | **0.2112** (0.2s) | 0.1357 (32.1s) | 0.1485 (6.2s) |
| 2 | **0.2818** (0.3s) | 0.1855 (29.3s) | 0.2111 (6.6s) |
| 4 | **0.3121** (0.2s) | 0.2186 (29.9s) | 0.2413 (6.5s) |
| 8 | **0.3740** (0.2s) | 0.2831 (29.0s) | 0.3018 (6.2s) |

- The **amortization-gap hypothesis is rejected at tiny k**: the encoder pass dominates
  at every k, by large margins (+0.06 to +0.09 NDCG), while being ~130× cheaper than
  100-step optimization.
- Refinement actively *hurts*: 20 likelihood steps from the encoder init move z **away**
  from the good solution (0.2818 → 0.2111 at k=2). The k-item multinomial likelihood is
  a bad objective at tiny k — it rewards concentrating mass on the k observed items,
  destroying the collaborative generalization the trained encoder has amortized. The
  encoder is not approximating this objective poorly; it is solving a *better* one.

**Verdict: PASS.** **Design implication:** use the amortized encoder pass as the belief
update everywhere in Phase 2/3; no per-user optimization loop is needed (or wanted) in
the elicitation inner loop — which also keeps per-turn updates at ~0.4 ms/user.

---

## Exp C — Posterior-variance sanity · **PASS** (native σ is correctly signed, unlike V1)

2,000 test users (≥8 fold-in items). Encoder outputs (μ, σ); mean σ across dims/users:

| k | 1 | 2 | 4 | 8 | full (80%) |
|---|---|---|---|---|---|
| mean σ | 0.5077 | 0.5043 | 0.4954 | 0.4827 | 0.4488 |

(k=0 is encoder-undefined — empty input NaN; composite-prior std reference = 1.0.)

- **(i) Monotone: YES.** Mean σ strictly decreases with evidence at every step,
  1 → full. More reveals → less native uncertainty, as a calibrated posterior should.
- **(ii) Per-user calibration at k=8: Spearman(mean σ_u, NDCG_u) = −0.081 — properly
  NEGATIVE.** Contrast the V1 bootstrap-ensemble uncertainty, which was significantly
  mis-signed at **+0.20** (it measured taste extremity, not confidence — see
  `experiments/paper2/UNCERTAINTY_STOPPING_RESULT.md`). The variational posterior does
  not conflate "distinctive user" with "uncertain estimate".
- Caveat: −0.081 is weak (correct sign ≠ strong signal). Usable as a directional
  input; not, alone, a per-user stopping gate.

**Verdict: PASS.** **Design implication:** RecVAE's native (μ, σ) is the uncertainty
representation to build on in Phase 2/3 (question selection targeting high-σ latent
directions; stopping/confidence features) — retiring the V1 bootstrap-ensemble approach;
but expect to *sharpen* σ (e.g. combine with belief magnitude) before trusting it as a
per-user budget gate.

---

## Verdict summary

| Exp | Verdict | One-line design implication |
|---|---|---|
| A · Concept pseudo-item folding | **FAIL** | Concepts must enter through the LATENT, not the interaction vector: input-space bags rank below the z=0 floor, subtract from real items, and cannot express dislikes (clamp = structural no-op) — while crude z-space subtraction already demotes a disliked tag specifically (−15 pts vs −1 control); select tags by lift. |
| B · Inference derivation | **PASS** | No amortization gap at tiny k — the encoder pass beats 100-step per-user MAP by +0.06–0.09 NDCG at ~130× less compute (likelihood refinement actively hurts); use amortized updates in the elicitation loop. |
| C · Posterior-variance sanity | **PASS** | Native σ is monotone in evidence (0.508→0.449) and per-user calibration is correctly signed (ρ = −0.081 vs V1's mis-signed +0.20) — build Phase-2/3 uncertainty on (μ, σ), but sharpen it before using as a stopping gate (signal is weak). |
