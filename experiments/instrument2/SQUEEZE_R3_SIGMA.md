# INSTRUMENT 2.0 — "squeeze arc" RUNG 3-SIGMA: an uncertainty (σ)-conditioned actor — does SGD DISCOVER the repeat-probe strategy?

Foreground, no commits. Instrument = certified RecVAE-d512 (`ml1m_recvae_d512_best.pt`, frozen).
Operator `z'=z+16·a·q`, z0=0. Arena `ml1m_arena`, eval seeds {1,2,3,7,11}, te[300:] TEST (304 users),
T=8. Metric NDCG@10 full + Cremonesi tail. Empirical channel `.cache/instrument2/p4c_channel.json`,
answers scale-matched + per-user centered exactly as `p4c_answer_sources.py`/`squeeze_r2.py`. Runner
`scripts/instrument2/squeeze_r3_sigma.py` (imports P4a/P4c/R1/R2 helpers), artifact
`.cache/instrument2/squeeze_r3_sigma.json`, checkpoints `p4c_actor_sigma_s{0,1}.pt`.

## Why this rung exists

R2 (the linchpin): under the empirical noisy channel a zero-training **noise-adapted static** that
repeat-probes the top informative axis (schedule `[0,0,0,0,1,2,7,0]`, TEST **0.3143**/0.1167) BEATS the
noise-trained **plain actor** (**0.2844**/0.1010, Δ −0.030, p<0.001). R2b: the actor *can* tie the
static (TEST 0.3177/0.1168) but **only if warm-started from the winning repeat schedule** — SGD from the
neutral SVD-8 warm start never found it. That isolated an **optimization gap**: the architecture can
represent repeat-probing, but plain-belief SGD does not discover it.

**Hypothesis of this rung.** The plain actor conditions only on its (noise-corrupted) belief `z_t`; it
has no explicit signal for *"how much uncertainty remains along each informative axis"* — the exact
quantity that says "this axis is still noisy → probe it again." Give it that signal — the per-coordinate
posterior **variance** from a decoder-metric Kalman belief tracker (reduced to the top-K=32 decoder-SVD
subspace, `R=32` = R1's noisy decoder value) as an extra input — and test whether SGD **discovers the
repeat-probe strategy on its own, WITHOUT the tie-by-construction warm start** (BC-warm stays neutral =
SVD-8). σ-feature per axis k: `feat_k = sqrt(C_kk/lam_k) ∈ [0,1]` = fraction of prior std remaining
(1 at t=0 → 0 as axis k is resolved). The reduced-Kalman covariance update is measurement-independent,
so the feature is a legitimate, cheap per-turn input.

## Setup

- **σ-actor** = `SigmaActor(z_t, σ_t, turn)` — MLP input `[z(512), σ(32), turn-onehot(8)]`, otherwise
  identical to `P.Actor` (2×512 SiLU). BC-warmed to the **neutral SVD-8 basis** (the SAME start the R2
  plain actor used — NOT the repeat schedule), then differentiable-unroll fine-tuned 20 epochs under the
  channel's effective law `a=1.552·s−0.362+N(0,0.480)` (noise stop-grad), the identical recipe as
  `part1_trainnoisy`. Trained 2 trseeds {0,1}; both a best-val-**tail** and best-val-**full** checkpoint
  saved. σ path is detached (a read-off feature, not backprop'd).
- **Comparators** recomputed on the same cohorts/CRN via R2 helpers: plain noise-trained actor
  (seed-avg s0/s1/s2), noise-adapted static-repeat (winning `[0,0,0,0,1,2,7,0]`).

## RESULT — TEST 5-seed, sampled empirical channel (noise×1), scale-matched

| arm | full | tail | note |
|---|---|---|---|
| noise-adapted **static-repeat** `[0,0,0,0,1,2,7,0]` | **0.3143** | **0.1167** | R2 winner (repeat top axis) |
| **σ-actor** (tail-sel, seed-avg s0/s1) | **0.2943** (sd .0083) | 0.0976 | uncertainty-conditioned |
| **σ-actor** (full-sel, seed-avg s0/s1) | **0.2943** (sd .0041) | 0.0972 | identical headline |
| plain noise-actor (seed-avg s0/s1/s2) | 0.2844 | 0.1010 | R2 anchor |

Both val-selection protocols give the **same** 0.2943 full. Val curves were strong (σ-actor val full
peaked 0.36 vs the plain actor's ~0.31) but this did **not** translate into a repeat strategy or a
static-beating test score.

**Paired bootstrap (304 users, seed-avg per-user full, 5000 resamples):**

| margin | Δ full | 95% CI | p(Δ>0) |
|---|---|---|---|
| **σ-actor − plain noise-actor** | **+0.0099** | [+0.0032, +0.0168] | **0.998** |
| **σ-actor − noise-adapted static-repeat** | **−0.0200** | [−0.0302, −0.0102] | **0.000** |

### Learned schedule — σ-conditioning induces a DISTINCT ordered sweep, NOT repeat

Per turn, the modal top-8-decoder-SVD axis the σ-actor probes (fraction of users at that axis, mean |cos|):

`t0:ax0(1.00, .93)  t1:ax1(1.00, .93)  t2:ax2(1.00, .90)  t3:ax3(1.00, .87)  t4:ax4(1.00, .68)  t5:ax5(1.00, .85)  t6:ax6(1.00, .77)  t7:ax7(1.00, .75)`

Every user, every turn: the σ-actor walks the top-8 informative axes **0→1→2→…→7, each exactly once**.
This is the **distinct-informative sweep** (a clean, correctly-ordered SVD-8), the polar opposite of the
repeat schedule the channel actually rewards. The uncertainty signal drives a **greedy D-optimal
"go to the most-uncertain-remaining axis"** behavior: once axis 0's σ collapses after one probe, axis 1
is now the most uncertain, so the actor moves on — it never re-probes.

## VERDICT — the optimization gap does NOT close; σ-conditioning helps a little but discovers the WRONG strategy.

1. **σ-conditioning gives a small, significant lift over the plain actor** (+0.0099 full, CI
   [+0.003,+0.017], p=0.998) — cleanly ordering the informative sweep is worth ~1 NDCG point over the
   plain belief-only actor's messier trajectory.
2. **But it does NOT discover repeat-probing and does NOT beat the noise-adapted static.** It still loses
   to static-repeat by −0.0200 (p<0.001) — worse than the R2b *tie-construct* actor (0.3177) that was
   handed the schedule. Giving SGD an uncertainty signal did **not** let it find the strategy; if
   anything it steered SGD *toward* the distinct D-optimal sweep and *away* from repeat.
3. **Why the model-based σ signal misleads.** The reduced Kalman is a **linear-Gaussian** belief: with
   `R=32` a single probe of an axis reduces that axis's posterior variance enough that the σ feature
   reports it **"resolved,"** so the greedy uncertainty-follower moves to the next axis. But the *real*
   channel is **heavy 1.9-bit quantization noise** that a single graded answer does not average out — the
   Kalman **understates the residual uncertainty** because it trusts its own noise model. The strategy the
   empirical channel rewards (re-measure the same axis to average down quantization noise) is precisely
   the one the calibrated-Gaussian σ signal tells the actor is unnecessary. Uncertainty-conditioning
   thus **reinforces** the diversify-then-stop instinct rather than curing it.

**Net for Papers C/D.** The R2/R2b conclusion is **strengthened, not overturned**. An explicit
per-axis-uncertainty input — the most principled ingredient a plain feed-forward actor lacks — was **not
sufficient** for SGD to recover the repeat-probe optimum: the learned policy sweeps eight distinct
informative axes once each and remains **−0.020 below the zero-training noise-adapted static** (p<0.001).
Under a realistic answer channel the robustness prize is the **non-adaptive repeat schedule**, and even a
belief-tracker's covariance cannot coax a learned continuous actor into it — the model-based uncertainty
is miscalibrated to the true (quantization) noise and points the wrong way. Adaptivity's positive value
remains **clean-channel only**.

## CLEAN-channel control (optional)

σ-actor under the noiseless geometric answerer `a=cos(z*,q)`, `R=1`, 5-seed TEST:
**σ-actor ≈ 0.434/0.234** (tail-sel) — below additive·actor **0.4901**/0.2973 and below additive·SVD-8
0.4709. Consistent with the σ-actor having learned an SVD-8-like ordered sweep rather than the P4a
actor's adaptive directions: it neither hurts nor helps on the clean channel, matching R1/R3-belief's
finding that the additive actor's edge is its *learned directions*, not a belief mechanism.

### Durable artifacts
`.cache/instrument2/squeeze_r3_sigma.json` (train val; `eval_tail`/`eval_full` blocks with 5-seed TEST,
paired bootstraps, learned-schedule inspection; `clean` block); checkpoints
`.cache/instrument2/p4c_actor_sigma_s{0,1}.pt` (both tail- and full-selected states); script
`scripts/instrument2/squeeze_r3_sigma.py` (`train`/`eval`/`clean`).
