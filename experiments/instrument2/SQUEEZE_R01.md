# INSTRUMENT 2.0 — "squeeze arc" rungs 0-1: oracle sizing + the Bayesian (Kalman + D-optimal) arm

Foreground, chunked, no commits. Instrument = certified RecVAE-d512 (`ml1m_recvae_d512_best.pt`,
frozen). Arena `ml1m_arena` (byte-identical V1 splits), eval seeds {1,2,3,7,11}, te[300:] TEST
(304 users), T=8, additive operator `z'=z+eta*a*q` (eta=16), graded answer `a=cos(z*,q)`,
`z*=enc(profile-half likes)`. Metric NDCG@10 full + Cremonesi tail. Runner
`scripts/instrument2/squeeze_r01.py`, artifact `.cache/instrument2/squeeze_r01.json`.

**Anchors (P4a, 5-seed TEST):** full-profile **0.5541** | actor **0.4907/0.2982** |
SVD-basis-8 **0.4709/0.2916** | item-8 fold **0.4625/0.2796** | MOSTPOP 0.3099/0.0808.

---

## RUNG 0 — ORACLE SIZING (the compass)

Both are **L3 privileged** (they peek at the user's held-out TEST NDCG to choose what to probe/fold;
the *answer* stays the honest graded geometric answer). They are absolute ceilings, not realizable.

| oracle | full | tail | note |
|---|---|---|---|
| **(a) L3 answer-peek DIRECTION oracle** | **0.8782** (sd .0039) | **0.4114** (sd .0148) | greedy per-turn pick of the NDCG-maximizing direction from ~2048 candidates (3706-subsampled item rows + 512 decoder-SVD + 512 random unit dirs); repeats allowed; 8 turns |
| **(b) L3 best-8-item-subset FOLD oracle** | **0.7546** (sd .0046) | **0.3981** (sd .0115) | greedy forward selection of 8 profile-like items to fold through the encoder |
| — full-profile fold (fold ALL likes) | 0.5541 | — | anchor |
| — realizable actor | 0.4907 | 0.2982 | anchor |
| — SVD-basis-8 / item-8 fold | 0.4709 / 0.4625 | 0.2916 / 0.2796 | anchors |

**Compass readings.**
1. **Continuity is nowhere near saturated.** The continuous direction ceiling (0.878) sits **+0.123
   full above** the discrete item-subset ceiling (0.755) and **+0.324 above full-profile** (0.554).
   A continuous query can in principle read taste far better than any item fold — the continuity
   prize the whole program bets on has real privileged headroom on this strong instrument.
2. **Selection >> folding everything.** The best-8-item-subset oracle (0.755) **beats full-profile
   (0.554) by +0.20** — choosing the 8 most discriminative profile items beats folding the entire
   profile (dilution). Elicitation is fundamentally a SELECTION problem (confirms the long-standing
   "true ceiling = oracle best-SUBSET" note).
3. **Realizable methods capture a thin slice.** Actor 0.491 over item-8 0.463 is
   `(0.491−0.463)/(0.878−0.463) ≈ 7%` of the answer-peek direction headroom. The other ~93% is the
   privileged label-peek — **not** a realizable target. The honest realizable gap the actor leaves on
   the table is the **0.491→0.755 selection gap** (item-subset oracle, no continuous magic needed)
   and the **0.755→0.878 continuity gap** (what a non-peeking continuous policy could add).

---

## RUNG 1 — THE BAYESIAN ARM (Kalman belief + D-optimal policy, ZERO training)

**Belief.** Prior `z0=0`, prior covariance `P0` (four designs below); scale `= mean_train||z*|| = 17.68`.
Each probe is a linear observation `y_t = a_t·scale = q_t^T z + noise`, rank-1 Kalman posterior update
(mean+cov). Decode from posterior mean. **D-optimal (= A-optimal here — both maximize `v^T P v`, so both
pick the top eigenvector)** probes the top eigenvector of the running covariance each turn.
Item folds re-anchor the mean to the encoder output (hybrid design, not exercised in the headline —
the headline is pure continuous probes, apples-to-apples with the actor/SVD-8). `R` val-selected per
prior on te[:300].

### (a)+(b) Clean channel — the 2×2 decomposition (update × directions), TEST 5-seed

| directions ↓ / update → | **additive** (operator) | **Kalman** |
|---|---|---|
| **actor** (learned) | **0.4901 / 0.2973** (= anchor ✓) | 0.4486 / 0.2659 |
| **SVD-basis-8** (fixed) | 0.4709 / 0.2916 (= anchor ✓) | 0.4427 / 0.2632 |
| **D-optimal, decoder-metric prior** | 0.4709 / 0.2916 (≡ SVD-8, byte-identical) | **0.4787 / 0.2916** |
| **D-optimal, population-z* prior** | 0.1688 / 0.1574 (≡ PCA-z*-8, P4a's 0.1688 ✓) | 0.1677 / 0.1568 |
| **D-optimal, isotropic prior** | — | 0.1253 / 0.0902 |
| **D-optimal, population-diag prior** | — | 0.1363 / 0.0710 |

**Paired bootstrap (304 users):**
| margin | full Δ | 95% CI | p(Δ>0) |
|---|---|---|---|
| Kalman·D-opt(decoder) − additive·actor | **−0.0114** | [−0.020, −0.003] | 0.005 |
| Kalman·D-opt(decoder) − additive·SVD-8 | **+0.0078** | [+0.004, +0.012] | 1.00 |
| Kalman·actor − additive·actor | −0.0415 | [−0.053, −0.031] | 0.00 |
| Kalman·SVD-8 − additive·SVD-8 | −0.0282 | [−0.041, −0.015] | 0.00 |

**Clean-channel verdict — does D-optimal beat the learned actor? NO.**
- The best Bayesian arm (Kalman·D-optimal, **decoder-metric prior**, 0.4787) **loses to the trained
  actor (0.4901) by −0.011 full (sig, p=.005)** and ties on tail. It does **beat the static SVD-8
  (+0.008, sig)** — a small gain from D-optimal *ordering* + Kalman weighting — but never reaches the
  actor.
- **The Kalman update itself is neutral-to-harmful.** Holding directions fixed, additive→Kalman
  *loses* 0.03–0.04 (actor 0.490→0.449, SVD-8 0.471→0.443, both sig). The Kalman posterior mean is a
  shrinkage estimator that under-inflates the latent magnitude; the additive operator's implicit
  "re-inflate the unit direction to ||z*||≈eta" is doing real work that naive Bayesian fusion discards.
- **D-optimal only works in the DECODER metric.** Under the *natural* Bayesian priors — isotropic
  (0.125) or population-z* covariance (0.168) — greedy optimal design **collapses**, exactly
  reproducing the PCA-z* failure (`additive·D-opt(pop) = 0.1688 =` P4a's PCA-8). Classical optimal
  design in the latent belief space probes the **population-variance / popularity axis**, not the
  item-discriminative coordinates. Informativeness must be injected from the **decoder geometry**
  (decoder-SVD basis) — at which point "D-optimal" is just re-deriving the SVD-8 basis, not adding
  design intelligence.

### (d) Noisy channel — P4C empirical `P(rating|s)` (scale-matched, common random numbers)

The fitted channel `p4c_channel.json` had landed (the full P4C writeup had not). Answers sampled from
the empirical per-bin rating distribution, per-user centering, scale-matched to native-cos RMS
(cmul≈0.68 — isolates information from raw scale), **common random numbers across arms**, `R`
re-selected on a noisy val.

| arm | full | tail |
|---|---|---|
| **Kalman · D-opt(decoder), R=32** | **0.2480** | 0.0944 |
| additive · D-opt(decoder), R=32 | 0.2465 | 0.0954 |
| additive · SVD-8 (8 distinct dirs) | 0.1551 | 0.0789 |
| **additive · actor (learned)** | **0.1530** | 0.0861 |
| Kalman · SVD-8 | 0.1306 | 0.0700 |
| Kalman · actor | 0.0548 | 0.0420 |
| D-opt(population-z* prior), either update | ~0.073 | ~0.068 |

Paired bootstrap: Kalman·D-opt(decoder) − additive·actor **= +0.095 [+0.077, +0.114], p=1.00**;
vs additive·SVD-8 **= +0.093, p=1.00**; Kalman·actor − additive·actor = −0.098, p=0.

**Noisy-channel verdict — under realistic noise the Bayesian/optimal-design arm WINS, but the value is
the direction schedule, not the Kalman fusion.**
- The **trained actor collapses under noise** (0.4901 → 0.1530): its per-turn directions depend on a
  now noise-corrupted belief `z_t` → feedback instability.
- **D-optimal on the decoder metric is far more robust (0.248)**, beating the actor by +0.095 and
  SVD-8 by +0.093. Why: with the noise-appropriate `R=32`, D-optimal's covariance downdate is too weak
  to switch eigenvectors, so it **repeats the top informative axis** — schedule
  `[0,0,1,0,1,0,2,3]` over the SVD indices (vs the clean `R=1` schedule `[0,1,…,7]`). Re-probing the
  most-informative direction **averages down the channel noise** — a genuine, interpretable adaptive
  behavior the fixed SVD-8 (8 distinct dirs) and the actor do not exhibit.
- But **Kalman ≈ additive on that schedule** (0.248 ≈ 0.247): the robustness comes from *the direction
  schedule* (fixed informative basis + repeat-top-axis under noise), **not** from the Bayesian update
  rule. Caveat: this noisy arm re-selects R and shares CRN, but is a robustness *signal* — the
  authoritative noisy-channel analysis remains P4C's.

---

## BOTTOM LINE + RECOMMENDATION FOR RUNGS 2-4

**Compass numbers.** L3 answer-peek DIRECTION oracle **0.878/0.411**; L3 best-8-item-subset FOLD
oracle **0.755/0.398**; vs full-profile 0.554, actor 0.491, SVD-8 0.471, item-8 0.463. The realizable
actor captures ~7% of the (privileged, mostly-unrealizable) direction-oracle headroom; the honest
realizable targets are the **0.49→0.75 selection gap** and the **0.75→0.88 continuity gap**.

**Bayesian-arm verdict.** Zero-training Kalman + D-optimal **does NOT beat the learned actor on the
clean channel** (−0.011 full, tail tie); it only beats the *static* SVD-8 (+0.008). The Kalman update
rule is neutral-to-harmful (shrinkage under-inflates magnitude; additive's re-inflation wins). D-optimal
is only viable in the **decoder metric** — in the natural latent/population metric it reproduces the
PCA-z* collapse (optimal design finds the popularity axis, not the discriminative one). **The one place
the Bayesian arm decisively wins is under a realistic NOISY channel**, where the actor collapses
(0.49→0.15) and D-optimal's "repeat the top informative axis" schedule holds 0.25 (+0.095, p=1.0) —
but that robustness is the *schedule*, not the fusion.

**Recommendation for rungs 2-4.**
1. **Stop chasing clean-channel policy gains.** The actor already sits at the realizable myopic ceiling
   on the clean channel; D-optimal/Kalman can't beat it. Marginal clean-channel policy work is low-ROI.
2. **Pivot the squeeze to ROBUSTNESS (the real, large, realizable prize).** The noisy-channel actor
   collapse vs the fixed-basis/repeat-axis robustness is a 0.10-NDCG, publishable, honest effect.
   Build: (a) **train-noisy actors** (P4C's `trainnoisy` recipe) and check they recover the fixed-basis
   robustness while keeping adaptivity; (b) bake the **"re-probe the top informative axis under noise"**
   behavior into a learned policy (it's what D-optimal-with-large-R does); (c) a **decoder-metric belief**
   (Kalman whose covariance lives in the decoder geometry) rather than latent-space optimal design.
3. **Attack the SELECTION gap.** item-subset oracle (0.755) ≫ full-profile (0.554): a realizable,
   non-peeking best-subset selector (greedy EIG over items/directions **in the decoder metric**) is
   worth a rung — target 0.55→0.75 discrete, then 0.75→0.88 continuous.
4. **Fix the magnitude bug in any belief tracker.** Posterior-mean shrinkage costs NDCG; re-inflate the
   estimate to the natural `||z*||≈eta` (use a scaled/MAP estimate, the additive operator's implicit trick).

### Durable artifacts
`.cache/instrument2/squeeze_r01.json` (oracle_dir, oracle_item, bayes, bayes_noisy);
script `scripts/instrument2/squeeze_r01.py`.
