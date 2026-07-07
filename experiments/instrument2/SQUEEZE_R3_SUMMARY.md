# SQUEEZE R3 — consolidated scorecard (2026-07-05/06)

Five headroom-chasing rungs on the ML-1M I2 instrument (`ml1m_recvae_d512_best.pt`, arena `ml1m_arena`,
seeds {1,2,3,7,11}, te[300:] TEST = 304 users, T=8, η=16, graded a=cos(z*,q), NDCG@10 full+tail, paired
per-user bootstrap). Goal: can any *realizable* method reach the R0 compass ceilings (privileged
answer-peek DIRECTION oracle 0.8782, best-8-item-subset FOLD oracle 0.7546) or beat the standing anchors
(clean actor 0.4907; noisy static-repeat 0.3143)?

**Anchors.** full-profile 0.5541 · actor 0.4907/0.2982 · SVD-8 0.4709/0.2916 · item-8 0.4625 ·
noisy: static-repeat `[0,0,0,0,1,2,7,0]` 0.3143/0.117 · noise-trained fixed actor 0.2844/0.101.

## Scorecard — 4 negatives/ties + 1 positive (early-curve efficiency)

| # | rung | channel | result | verdict |
|---|---|---|---|---|
| 16 | non-peeking decoder-metric **EIG selector** | clean | discrete 0.378 / continuous 0.384 — **below every anchor** (−0.093 vs SVD-8, −0.112 vs actor, all p=0) | **FAILS** — the 0.755/0.878 headroom is oracle-only (privileged); a non-peeking selector captures a *negative* fraction of it (BALD/EIG selects for belief-info, not NDCG) |
| 17 | decoder-metric **belief tracker** + Kalman magnitude fix | clean+noisy | decoder-Kalman 0.4792 clean (−0.011 vs actor, +0.008 vs SVD-8); re-inflation ≤+0.0005 | **REFUTED** — magnitude-shrinkage hypothesis dead; raw posterior mean already balanced in decoder geometry; additive's edge is the actor's learned DIRECTIONS |
| 18 | **learned belief head** | clean | 0.4908 vs additive 0.4901, Δ+0.0007, p=0.64 | **TIES** additive — the fixed operator is already essentially optimal; a learned update adds nothing |
| 19 | **sigma-actor** (uncertainty-conditioned) | noisy | 0.2943 (+0.010 vs plain noise-actor, p=0.998) but learns a **distinct sweep 0→7**, still −0.020 below static (p<0.001) | **WRONG STRATEGY** — σ-conditioning steers SGD to diversify-and-stop; linear-Gaussian belief understates quantization noise so σ is miscalibrated; optimization gap stays open |
| 20 | **k-curriculum** (anytime/all-lengths reward) | clean+noisy | see below | **WIN (clean early-curve)**; noisy not rescued |

## Rung 20 detail — the one realizable win

Anytime reward = mean over turns 1..8 of `L_rec_t + 0.3·softNDCG_t` (front-loading), val-selected, 2 trseeds.

**CLEAN — large significant early-curve win, converges at T=8 (Paper-D anytime signature, reproduced on I2):**

| turn | fixed full | anytime full | Δ full | bootstrap |
|---|---|---|---|---|
| 1 | 0.116 | 0.306 | **+0.190** | p=1.00 |
| 2 | 0.139 | 0.358 | **+0.219** | p=1.00 |
| 4 | 0.186 | 0.443 | **+0.257** | p=1.00 |
| 8 | 0.491 | 0.492 | +0.001 | p=0.64 (tie) |

Tail mirrors (turn-4 +0.070; T=8 −0.001). Anytime reaches by turn ~3 (0.421) what the back-loaded fixed
actor needs 6 turns for (0.387). The fixed-horizon actor is back-loaded because it optimizes only turn-8;
the curriculum front-loads it. Deployable property (better at every budget), not a new endpoint.

**NOISY (sampled empirical, noise×1) — curriculum does NOT rescue it:**

T=8 full/tail: anytime **0.269/0.086** · noise-fixed **0.285/0.100** · static-repeat **0.314/0.117**.
- anytime − static-repeat (T=8): −0.045, CI[−0.058,−0.032], p<0.001 (loses at every turn)
- anytime − noise-fixed (T=8): −0.016, p=0.001 (trades late-turn NDCG for early turns)
- anytime IS the strongest *learned* noisy actor at turns 2–5 (repairs the noise-fixed mid-turn dip:
  turn-2 0.261 vs 0.169) — but the zero-training static dominates throughout.
- Noisy VAL (~0.33) over-estimated noisy TEST (~0.27): same val-overfit-to-noise-draw pathology as R2b.
- Pipeline validated: reproduces R2 static-repeat 0.3143 and noise-fixed 0.2844 exactly.

## Bottom line
No realizable method beats the actor's clean endpoint or the noise-adapted static, and none reaches the
privileged 0.755/0.878 ceilings — confirmed five independent ways. The single realizable improvement is
**anytime training → large early-turn efficiency** (converges at T=8; does not help the noisy channel).
Consistent with R2/R2b: **adaptivity's positive value is clean-channel/high-fidelity only; the noise-robust
optimum is the static repeat-probe schedule.** (Repeat-probing itself is over-flattered by the i.i.d.-per-ask
noise model — see the repeat-probe realism caveat; a fixed-vs-resampled-noise decomposition is the decisive
open test.)

## Artifacts
Per-rung docs: `SQUEEZE_R3_EIG.md`, `SQUEEZE_R3_BELIEF.md`, `SQUEEZE_R3_LBELIEF.md` (in-json only),
`SQUEEZE_R3_SIGMA.md`, `SQUEEZE_R3_KCURR.md`. JSONs `.cache/instrument2/squeeze_r3_{eig,belief,lbelief,
sigma,kcurr}.json`. Checkpoints `.cache/instrument2/p4c_actor_{sigma,anytime,anynoisy}_s{0,1}.pt`. Scripts
`scripts/instrument2/squeeze_r3_{eig,belief,sigma,kcurr}.py`. Priority per HANDOFF v2: P8 / post-ECIR.
