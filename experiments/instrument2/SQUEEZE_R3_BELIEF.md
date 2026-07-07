# INSTRUMENT 2.0 — "squeeze arc" RUNG 3: decoder-metric Kalman belief + the magnitude-shrinkage fix

Foreground, no commits. Instrument = certified RecVAE-d512 (`ml1m_recvae_d512_best.pt`, frozen).
Arena `ml1m_arena`, eval seeds {1,2,3,7,11}, te[300:] TEST (304 users), T=8, additive operator
`z'=z+16·a·q` (eta=16, z0=0), graded answer `a=cos(z*,q)`, `z*=enc(profile-half likes)`. Metric
NDCG@10 full + Cremonesi tail. Runner `scripts/instrument2/squeeze_r3_belief.py` (imports R1 helpers),
artifact `.cache/instrument2/squeeze_r3_belief.json`.

**What R1 diagnosed (SQUEEZE_R01.md, RUNG 1).** The Kalman posterior mean is a *shrinkage* estimator
that under-inflates the latent magnitude. Because the decode is `S = mu·Wᵀ + bdec`, a shrunk `mu`
lets the popularity bias `bdec` dominate the ranking → additive→Kalman *loses* 0.03–0.04 NDCG.
R1 Recommendation #4: **re-inflate the posterior mean to the natural `||z*||≈eta`** (scaled/MAP
estimate, replicating the additive operator's implicit re-inflation). This rung builds the
decoder-metric belief and **tests that fix head-on.**

**Anchors (R1, 5-seed TEST).** clean: additive·actor **0.4901/0.2973** | additive·SVD-8
**0.4709/0.2916** | Kalman·D-opt(decoder) 0.4787/0.2916. noisy: Kalman·D-opt(decoder) **0.248/0.094**
| additive·actor 0.153/0.086 | (R2 noise-adapted) static_repeat **0.3143/0.1167** | noisy actor
**0.2844/0.1010**.

---

## THE MAGNITUDE FIX (ablation design)

The D-optimal direction schedule depends only on `(P0, R)`, **not** on the decode-time magnitude, so
the fix is a pure decode-time transform: collect the final posterior mean once per belief arm, then
decode under four settings.

| setting | transform | meaning |
|---|---|---|
| **none** | `mu` | R1's raw shrinkage estimator |
| **unit_scale** | `mu → scale · mu/‖mu‖`, scale=17.68 | snap direction to the natural `‖z*‖≈eta` |
| **unit_tuned** | `mu → τ*· mu/‖mu‖`, τ* val-selected | snap to val-best norm |
| **gain_tuned** | `mu → g*· mu`, g* val-selected | MAP-style uniform de-shrink |

`scale = mean_train‖z*‖ = 17.68`. Belief arms: **decoder-metric D-opt** (prior covariance = decoder-SVD
eigenbasis with sv² eigenvalues → D-opt probes the discriminative directions, R1's `make_prior('decoder')`),
**actor-directed Kalman**, **SVD-8 Kalman** (both on the pop_full prior). Val-selected `τ*/g*` on
seed-1 val; `R` reused from R1's val selection (decoder R=1 clean / R=32 noisy; pop_full R=0.25/1).

---

## RUNG 3a — CLEAN channel, TEST 5-seed (full/tail)

| belief arm | none | unit_scale | unit_tuned | gain_tuned |
|---|---|---|---|---|
| **Kalman·D-opt(decoder)** | 0.4787/0.2916 | **0.4792**/0.2917 | 0.4789/0.2915 | 0.4787/0.2916 |
| Kalman·actor | 0.4486/0.2659 | 0.4487/0.2660 | 0.4491/0.2663 | 0.4478/0.2656 |
| Kalman·SVD-8 | 0.4427/0.2633 | 0.4430/0.2637 | 0.4459/0.2649 | 0.4427/0.2633 |
| — additive·actor (anchor) | **0.4901/0.2973** | | | |
| — additive·SVD-8 (anchor) | 0.4709/0.2916 | | | |

Val-selected params: D-opt(decoder) τ*=14, g*=1; actor τ*=12, g*=4; SVD-8 τ*=6, g*=1.

**Paired bootstrap (304 users, full NDCG):**
| margin | Δ full | 95% CI | p(Δ>0) |
|---|---|---|---|
| Kalman·D-opt(decoder), **none** − additive·actor | −0.0114 | [−0.020, −0.003] | 0.005 |
| Kalman·D-opt(decoder), **unit_scale** − additive·actor | −0.0109 | [−0.020, −0.002] | 0.007 |
| Kalman·D-opt(decoder), **none** − additive·SVD-8 | +0.0078 | [+0.004, +0.012] | 1.00 |
| Kalman·D-opt(decoder), **unit_scale** − additive·SVD-8 | +0.0083 | [+0.005, +0.012] | 1.00 |

*(the `none` row reproduces R1's bootstrap byte-for-byte — harness validated.)*

**Clean verdict — the magnitude fix is a NO-OP; R1's shrinkage hypothesis is REFUTED for the decoder-metric belief.**
- Re-inflating the posterior mean moves the decoder-metric arm by **+0.0005** (0.4787→0.4792) — noise.
  It does **not** close the −0.011 gap to the learned additive actor. The corrected belief tracker
  still **loses to additive·actor (0.490)** and only marginally beats static additive·SVD-8 (0.471,
  +0.008).
- **Why the fix does nothing here:** direction-preserving reinflation (`τ·mu/‖mu‖`) and uniform gain
  (`g·mu`) both **preserve the relative weighting across directions/coordinates** — they only change
  the mean's magnitude *relative to the popularity bias*. On the **decoder metric the raw Kalman mean
  is already well-balanced against `bdec`** (val picks g*=1, i.e. "don't inflate"; τ*=14≈scale). There
  is no magnitude-shrinkage to fix in this arm. The additive operator's advantage is therefore **not**
  a global re-inflation trick — it is the **actor's learned adaptive directions / per-direction
  weighting**, which no scalar rescale of the mean can reproduce.
- R1's shrinkage-collapse (iso/pop priors → 0.13–0.17) was a **direction** pathology (D-opt picks the
  popularity axis in the latent metric), not a magnitude one; the decoder metric already cures it, and
  once cured there is no residual magnitude deficit.

## RUNG 3b — NOISY channel spot-check (P4C empirical `P(rating|s)`, scale-matched, CRN)

| belief arm | none | unit_scale | unit_tuned | gain_tuned |
|---|---|---|---|---|
| **Kalman·D-opt(decoder)** | **0.2480**/0.0944 | 0.2449/0.0958 | 0.2457/0.0952 | 0.2463/0.0953 |
| Kalman·SVD-8 | 0.1306/0.0700 | 0.1306/0.0699 | 0.1337/0.0704 | 0.1306/0.0700 |
| Kalman·actor | 0.0548/0.0420 | 0.0547/0.0421 | 0.0564/0.0431 | 0.0548/0.0420 |
| — additive·actor (clean-era) | 0.1530/0.0861 | | | |
| — additive·SVD-8 (clean-era) | 0.1551/0.0789 | | | |
| — **R2 noise-adapted static_repeat** | **0.3143/0.1167** | | | |
| — **R2 noise-trained actor** | **0.2844/0.1010** | | | |

Bootstrap: Kalman·D-opt(decoder) `none` − additive·actor **+0.0950 [+0.077,+0.114], p=1.00**;
`unit_scale` **+0.0919, p=1.00** (fix ≈ no-op again).

**Noisy verdict.** Kalman·D-opt(decoder) reproduces R1's 0.248 exactly and still crushes the *clean-era*
actor (+0.095), but it **loses to R2's fair noise-adapted comparators** — noise-trained actor (0.284)
and the noise-adapted static_repeat basis (0.314). The magnitude fix changes nothing under noise
either (0.2449–0.2480). The noisy-channel robustness prize remains the **direction schedule**
(informative basis + repeat-top-axis), captured better by R2's noise-adapted static/actor than by the
Bayesian fusion.

---

## BOTTOM LINE

**The corrected belief tracker does NOT beat the additive operator, and the magnitude fix is a red
herring.** On the clean channel decoder-metric Kalman (0.479) sits −0.011 below additive·actor (0.490)
and re-inflating the posterior mean (unit_scale/unit_tuned/gain_tuned) moves it by ≤0.0005 — the
val-tuner even selects "no inflation" (g*=1) for the decoder arm. R1's magnitude-shrinkage diagnosis
does **not** hold for the decoder-metric belief: the raw posterior mean is already correctly balanced
against the popularity bias in the decoder geometry, so there is no shrinkage to undo. The residual
additive advantage is the actor's **learned directions**, not a global magnitude trick — confirming
R1's headline recommendation to **stop chasing clean-channel policy gains** and pivot to robustness.
Under noise the decoder-metric belief (0.248) beats the clean actor but loses to R2's noise-adapted
static (0.314) and noisy actor (0.284); the fix is a no-op there too.

**Verdict: NO new winner. Magnitude re-inflation is confirmed neutral (≤+0.001 full, both channels);
the additive re-inflation trick was never the source of the additive operator's edge — the learned
directions are. Rung closed.**

### Durable artifacts
`.cache/instrument2/squeeze_r3_belief.json` (`clean`, `noisy`: arms×{none,unit_scale,unit_tuned,gain_tuned}
+ additive anchors + paired bootstrap); script `scripts/instrument2/squeeze_r3_belief.py`.
