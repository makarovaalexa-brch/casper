# INSTRUMENT 2.0 -- squeeze arc rung 3: BUDGET-LENGTH CURRICULUM (anytime / all-lengths reward)

Foreground, no commits. Instrument = certified RecVAE-d512 (`ml1m_recvae_d512_best.pt`, frozen). Arena `ml1m_arena`, eval seeds {1,2,3,7,11}, te[300:] TEST (304 users), T=8, operator `z'=z+16*a*q`, z0=0. Metric NDCG@10 full + Cremonesi tail. Runner `scripts/instrument2/squeeze_r3_kcurr.py`, artifact `.cache/instrument2/squeeze_r3_kcurr.json`.

## Why this rung

Paper D's first realizable adaptive win came from an **anytime reward** (mean NDCG over turns 1..T, *all lengths*) -- a **front-loaded** policy that beat fixed-order at early/mid turns and converged at T=8. The I2 fixed-horizon P4a actor optimizes only the **turn-8** reconstruction+softNDCG, so it is heavily **back-loaded** (see table: turn1 0.116, turn4 0.191, then jumps 0.399/0.468/0.490 at turns 6/7/8). This rung retrains the *same* `Actor` with an **anytime loss** = mean over turns 1..T of `(L_rec_t + 0.3*softNDCG_t)` on (1) the clean geometric channel and (2) the empirical noisy channel, and asks whether the curriculum buys early-curve efficiency and/or helps the noisy case.

Actors: 2 clean anytime trseeds `p4c_actor_anytime_s{0,1}.pt` (val-select on anytime val tail), 2 noisy anytime trseeds `p4c_actor_anynoisy_s{0,1}.pt` (val-select on anytime noisy val full). Seed-avg over the 2 train actors, 5 eval seeds. Noise-fixed and noise-adapted static anchors are the 3-actor / greedy artifacts from SQUEEZE_R2.

## CLEAN channel -- k-curve (turns 1..8), 5-seed TEST, 3-actor seed-avg

| turn | fixed full | anytime full | delta full |
|---|---|---|---|
| 1 | 0.1159 | 0.3056 | +0.1897 |
| 2 | 0.1389 | 0.3579 | +0.2190 |
| 3 | 0.1575 | 0.4209 | +0.2635 |
| 4 | 0.1861 | 0.4431 | +0.2570 |
| 5 | 0.2379 | 0.4669 | +0.2291 |
| 6 | 0.3865 | 0.4768 | +0.0902 |
| 7 | 0.4671 | 0.4867 | +0.0195 |
| 8 | 0.4907 | 0.4920 | +0.0013 |

| turn | fixed tail | anytime tail | delta tail |
|---|---|---|---|
| 1 | 0.0677 | 0.0782 | +0.0105 |
| 2 | 0.0966 | 0.1266 | +0.0300 |
| 3 | 0.1194 | 0.1850 | +0.0656 |
| 4 | 0.1541 | 0.2238 | +0.0697 |
| 5 | 0.1868 | 0.2533 | +0.0665 |
| 6 | 0.2327 | 0.2749 | +0.0421 |
| 7 | 0.2710 | 0.2910 | +0.0200 |
| 8 | 0.2982 | 0.2971 | -0.0011 |

**Paired bootstrap (304 users, seed-avg per-user full, anytime - fixed):**

| turn | delta full | 95% CI | p(>0) |
|---|---|---|---|
| 1 | +0.1897 | [+0.1674, +0.2119] | 1.000 |
| 2 | +0.2190 | [+0.1972, +0.2417] | 1.000 |
| 4 | +0.2570 | [+0.2333, +0.2797] | 1.000 |
| 8 | +0.0013 | [-0.0058, +0.0083] | 0.642 |

## NOISY channel (sampled empirical, noise x1) -- k-curve, 5-seed TEST

Anchors from SQUEEZE_R2: noise-adapted static-repeat 0.3143, noise-trained fixed actor 0.2844.

| turn | noise-fixed full | anytime full | static-repeat full |
|---|---|---|---|
| 1 | 0.2702 | 0.2544 | 0.2714 |
| 2 | 0.1689 | 0.2610 | 0.2840 |
| 3 | 0.1976 | 0.2829 | 0.2913 |
| 4 | 0.1997 | 0.2771 | 0.2944 |
| 5 | 0.2306 | 0.2717 | 0.3002 |
| 6 | 0.2701 | 0.2679 | 0.3154 |
| 7 | 0.2855 | 0.2757 | 0.3074 |
| 8 | 0.2851 | 0.2691 | 0.3143 |

**T=8:** anytime 0.2691/0.0863 | noise-fixed 0.2851/0.1000 | static-repeat 0.3143/0.1167

**Paired bootstrap (304 users, seed-avg per-user full):**

| margin | delta full | 95% CI | p(>0) |
|---|---|---|---|
| anytime_minus_noisefixed_t8 | -0.0160 | [-0.0259, -0.0059] | 0.001 |
| anytime_minus_static_repeat_t8 | -0.0452 | [-0.0582, -0.0321] | 0.000 |
| anytime_minus_static_repeat_t4 | -0.0173 | [-0.0266, -0.0083] | 0.000 |
| anytime_minus_static_repeat_t2 | -0.0230 | [-0.0355, -0.0112] | 0.000 |

## VERDICT

- **CLEAN:** the anytime curriculum improves the early/mid curve (turn-4 full delta +0.2570) and converges to the fixed-horizon actor at T=8 (delta +0.0013).
- **NOISY:** the anytime-noisy actor reaches 0.2691 full at T=8, **below** both the noise-trained fixed actor (0.2851) and the R2 noise-adapted static-repeat (0.3143). At early/mid turns the anytime-noisy actor DOES front-load and repair the noise-fixed actor's mid-turn dip (e.g. turn-2 full 0.2610 vs noise-fixed 0.1689), so it is the strongest *learned* noisy actor at turns 2-5 -- but the zero-training noise-adapted static-repeat schedule dominates it at every turn (turn-2 0.2840, T=8 0.3143). The noisy-VAL selection (any-full ~0.33) badly over-estimated the noisy TEST (~0.27): the anytime objective is optimised through a noisy unroll whose favourable val noise-draw does not generalise -- the same val-overfit-to-noise pathology SQUEEZE_R2b documented. Consistent with R2/R2b: adaptivity's positive value is **clean-channel only**; under the realistic answer channel the noise-robust optimum remains the static repeat-probe schedule, and the anytime curriculum does not overturn it.

### One-line takeaway

The budget-length (anytime) curriculum is a **large, significant early-curve win on the clean channel** -- it front-loads the continuous actor so it reaches at turn ~3 what the fixed-horizon actor needs 6 turns for (turn-4 full +0.257, p=1.0), converging to an exact tie at T=8 (+0.001, p=0.64) -- exactly the Paper-D anytime signature; but it **does not rescue the noisy channel** (T=8 0.269 < static 0.314, p<0.001).

### Durable artifacts

`.cache/instrument2/squeeze_r3_kcurr.json` (clean+noisy k-curves, bootstraps); `p4c_actor_anytime_s{0,1}.pt` (clean anytime actors); `p4c_actor_anynoisy_s{0,1}.pt` (noisy anytime actors). Script `scripts/instrument2/squeeze_r3_kcurr.py` (`train_clean|train_noisy|eval|all`).
