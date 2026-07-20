# HANDOFF PROGRESS — 2026-07-05 (review-response experiment batch)

Session focus: execute the CPU-heavy, dependency-free experiments from HANDOFF.md §3 (T1–T9) and
§6 (priority experiment) FIRST, so the token-heavy paper-writing tasks are never done against numbers
that later change. No commits (squeeze arc = no-commit).

## Task tracker (13 tasks)
Priority experiment + T1–T9 + backlog captured, with text tasks BLOCKED by their gating experiments:
- T2 (#9) blocked by R2-noise (#1) + T4 (#2)
- T5a (#10) blocked by T5b (#4)
- T7b (#11) blocked by T7a (#3)
- T8 (#12) blocked by #1 + #4 + #5

## DONE ✅

### R2-noise priority experiment (§6 linchpin) — adaptivity is CLEAN-CHANNEL-ONLY
Under the realistic empirical answer channel (noise×1, 5-seed TEST, scale-matched):
| arm | full | tail |
|---|---|---|
| noise-adapted static, REPEAT `[0,0,0,0,1,2,7,0]` | **0.3143** (sd .0076) | **0.1167** |
| noise-trained actor (seed-avg s0/s1/s2) | 0.2844 (sd .0039) | 0.1010 |
| noise-adapted static, distinct-8 | 0.2069 | 0.0582 |
| clean-era SVD-8 (P4C's comparator) | 0.1511 | 0.0794 |

Paired bootstrap (304 users): actor − static-repeat **Δ=−0.0299, CI [−0.040,−0.020], p(Δ>0)=0.000**
(static wins, significant). **Verdict:** P4C's "adaptivity recovers under retraining" (actor 0.287 ≫
clean SVD-8 0.159) was an UNFAIR-comparator artifact — the clean static was denied the channel
adaptation that matters (repeat the top informative axis to average down noise). Against a fair
zero-training noise-adapted static the actor LOSES. **Papers C/D must scope the learned-policy claim to
"adaptivity under clean answers only."** Artifacts: `.cache/instrument2/squeeze_r2_noise.json`,
`experiments/instrument2/SQUEEZE_R2.md`, `scripts/instrument2/squeeze_r2.py`.

### T7a — EASE cold-start k=1..8 curve vs I2: tie is WARM-START-ONLY
I2 (RecVAE-d512) beats EASE at every k (5-seed TEST, full NDCG@10):
| k | I2 | EASE | Δ full 95% CI |
|---|----|------|---|
| 1 | 0.2809 | 0.2267 | +0.054 [+0.045,+0.063] |
| 2 | 0.3384 | 0.2772 | +0.061 [+0.051,+0.071] |
| 4 | 0.4021 | 0.3383 | +0.064 [+0.053,+0.074] |
| 8 | 0.4635 | 0.4104 | +0.053 [+0.043,+0.063] |

Δ full p=1.0 at every k. **Verdict:** kills the reviewer's "warm-start-only tie" charge — both
converge to ~0.554 only at full profile; under few-reveal elicitation I2 strictly dominates EASE.
Artifacts: `.cache/instrument2/t7a_ease_coldstart.json`, `experiments/instrument2/T7A_EASE_COLDSTART.md`,
`scripts/instrument2/t7a_ease_curve.py`.

### T4 — bot-play/ABot learned answerer as 4th answer source
Recovered from git `b1aaa31` (heteroscedastic MLP `ABot(z*,q,feats[5])→(μ,logσ²)`, Gaussian-NLL on
real ML-1M ratings; retired for policy-side reasons, NOT fidelity — legit 4th source). Re-implemented
on I2 geometry, retrained on 808,940 real ratings: corr(μ,answer)=0.589, calibration gate +0.232 PASS.
| arm | bot-play (ABot μ) | empirical channel ×1 |
|---|---|---|
| continuous actor | **0.407 / 0.222** | 0.150 |
| SVD-8 static | 0.385 / 0.188 | 0.159 |
| concept-8 | 0.406 / 0.221 | 0.325 |
| item-8 | 0.240 / 0.132 | 0.463 (fold) |

**Verdict:** the learned answerer sits on the OPPOSITE side of the fidelity boundary from the sampled
channel. μ = denoised conditional mean (no sampling noise / quantization) → continuous actor SURVIVES
at 0.407 (+0.257 over sampled channel), confirming PART 1's decomposition: the collapse to 0.150 was
SAMPLING NOISE, not model-in-loop. Ordering inverts — item-8 drops to 0.240 (a smooth regressor can't
reproduce idiosyncratic per-item real ratings). Artifacts: `.cache/instrument2/abot_i2.pt`,
`.cache/instrument2/p4c_botplay.json`, PART 4 + verdict V4 appended to `P4C_ANSWER_SOURCES.md`,
`scripts/instrument2/p4c_botplay.py`.

### T5b — popularity-conditioned answerability: concept advantage SURVIVES
Located canonical item-vs-concept exp `scripts/paper2/answerability_concept.py`. New popcond model:
P(recall item j) = σ(α·(log(cnt_j+1) − log(cnt_med+1))) — logistic in log-popularity, P=0.5 at median,
→1 head / →0 tail; item answerable iff recalled, concept iff ≥2 recallable members (items+concepts
treated consistently, so obscure-item concepts lose "free" answerability).
| regime | FULL pop_item→conc_pop | TAIL pop_item→conc_pop |
|---|---|---|
| orig | 0.307→0.315, Δ+0.008 n.s. | 0.085→0.116, Δ+0.031 [+0.004,+0.060] sig |
| popcond α=1.0 | 0.308→0.315, Δ+0.007 n.s. | 0.086→0.116, Δ+0.031 sig |
| popcond α=2.0 | 0.307→0.315, Δ+0.008 n.s. | 0.085→0.116, Δ+0.031 sig |

**Verdict:** concept-channel advantage survives essentially unchanged (tail +0.031 sig in every regime).
Mechanism: the winning policy asks broad concepts whose head-item members stay recallable even when the
tail is crushed — the advantage was never driven by generous answerability of obscure material.
Artifacts: `scripts/paper2/t5b_answerability_popcond.py`,
`data/movielens/.cache/paper2/t5b_answerability.json` (+`_alpha2`), `experiments/paper2/T5B_ANSWERABILITY.md`.

## NEW FOLLOW-UPS (HANDOFF §7, added by user 2026-07-06)
The user confirmed the R2 verdict and added:
- **Pre-authorized wording:** clean-channel adaptive margin (+0.019) is FIDELITY-CONDITIONAL; under noise
  the optimal realizable strategy is repeat-probing the top informative axis ("ask the best question five
  times" — quotable, ours).
- **R2b tie-by-construction arm (#14, RUNNING):** warm actor from the winning repeat schedule
  `[0,0,0,0,1,2,7,0]` + val-select → ties static by construction; test shortfall = pure val→test noise
  (quantifies the optimization gap); upside = adaptive-repeats may add value on top. Resumed the R2 agent
  to run it (extends `squeeze_r2.py`).
- **T8 optimization-gap finding (#15, held, text):** state ONCE across papers that direct
  construction/search beats gradient training of a superset class at these sample sizes (recurring:
  BC-clone<teacher, from-scratch collapses, FieldActor<plain, Gumbel<REINFORCE<fixed, R2 actor<searched
  static); every surviving positive claim already beat its searched-static control.

### R2b — tie-by-construction: DONE, optimization-gap confirmed
BC-warm actor to emit winning schedule `[0,0,0,0,1,2,7,0]` (per-turn cos 1.0×8):
| arm | VAL | TEST 5-seed |
|---|---|---|
| tie construct-only | 0.2857/0.0932 | **0.3177**(sd.0043)/0.1168 |
| tie fine-tuned | 0.2980/0.0742 | 0.2925/0.0974 |
| static-repeat (ref) | 0.2923/0.0789 | 0.3143/0.1167 |

Bootstrap vs static-repeat: construct-only Δ=+0.0035 [−0.004,+0.011] p=0.82 → **TIE**; fine-tuned Δ=−0.022
p<.001. **Verdict:** R2's defeat of the *trained* actor is an OPTIMIZATION gap, not expressiveness — the
architecture fully represents the noise-robust repeat-probe policy; SGD-through-channel just fails to find
it. Fine-tuning from the optimum drifts away (no adaptive gain on top). Feeds T8 (#15) optimization-gap
finding. Artifacts: `r2b` block in `squeeze_r2_noise.json`, `p4c_actor_tie_ft.pt`, R2b section in
`SQUEEZE_R2.md`, `r2b` stage in `squeeze_r2.py`.

### T6 — Paper D fair comparison: DONE, advantage does NOT survive equal access
Access-controlled arms (every arm gets the same `m` matched volunteered known-item tokens, then spends
`T−m` turns on its own policy; equal budget T=8; realistic popweight answerer; BOUND rows labeled). @m=4:
| arm | FULL | TAIL |
|---|---|---|
| OPEN-recall (ours, realistic) | 0.3926 | 0.1776 |
| **PEBOL+profile [PRIMARY]** | 0.3899 | 0.1767 |
| ConTS+profile | 0.3796 | 0.1649 |
| profile-only (control) | 0.3778 | 0.1576 |
| [BOUND] oracle-distinct | 0.4047 | 0.2106 |

open−PEBOL Δ across m-sweep: m=0 +0.018/+0.006 → m=2 +0.013 → **m=4 +0.003/+0.001 (TIE, CIs overlap)** →
m=6 +0.005/**−0.004**. **Verdict:** open-recall's bandwidth edge is an information-access artifact; at equal
known-item access it's a statistical tie (PEBOL edges tail by m=6). Honest tail loss: realistic open tail
0.1776 is −0.033 below the oracle bound (cost of popularity-biased recall). **Paper D reframe:** volunteered
tokens are a big lever FOR EVERY METHOD; at equal budget, naming vs probing is a wash — open-recall's value
is OPERATIONAL (low-friction), not a raw NDCG edge. Artifacts: `data/movielens/.cache/paperD/t6_paperD_fair.json`,
`experiments/paper2/T6_PAPERD_FAIR.md`, `ACCESS` block in `scripts/paper2/continuous_actor.py`.

## ALL 6 REVIEW-RESPONSE EXPERIMENTS COMPLETE
R2-noise, R2b, T4, T5b, T6, T7a all done.

## ML-25M I2 PORT — DONE (Jul 5, 8/8 gates)
RecVAE-d512 full-profile 0.4998/0.3443 (+0.248/+0.294 over MOSTPOP, EASE-class, matches ML-1M), monotone
k-curve, **8/8 gates PASS**. = 2nd replication dataset for A/C. Honest scope note: concept-only fidelity
−0.061 below pop floor (rank-4 genome-centroid concepts) but concepts still ADD on items (G5).
`experiments/instrument2/PHASE2_ML25M_PORT.md`.

## SQUEEZE R3 (user-requested, after the port) — headroom-chasing rungs
R0 compass ceilings: selection 0.755, continuity 0.878 (both privileged). Realizable anchors: actor 0.4907,
SVD-8 0.4709 (clean); static-repeat 0.3143 (noisy).
- ✅ **#17 decoder-metric belief + magnitude fix:** magnitude-shrinkage hypothesis REFUTED — raw posterior
  mean already balanced in decoder geometry; re-inflation ≤+0.0005 (no-op). Decoder-Kalman 0.4792 clean
  (−0.011 vs additive·actor, +0.008 vs SVD-8); noisy 0.248 (= R1, < static). No new winner. The additive
  edge is the actor's learned DIRECTIONS, not re-inflation. `SQUEEZE_R3_BELIEF.md`.
- ✅ **#19 sigma-actor (uncertainty-conditioned):** optimization gap does NOT close. σ-conditioning gives
  +0.010 over plain noise-actor (p=0.998) BUT discovers the WRONG strategy (distinct D-optimal sweep
  0→7, not repeat) → still −0.020 below static-repeat (p<.001). Reason: linear-Gaussian Kalman UNDERSTATES
  residual uncertainty under 5-level quantization → σ miscalibrated, points wrong way. R2/R2b strengthened.
  `SQUEEZE_R3_SIGMA.md`.
- ⏳ **#16 decoder-metric EIG selector (FLAGSHIP)** — running (selection then continuity gap).
- ⏳ **#20 k-curriculum (anytime reward)** — running.
- ⏳ **#18 learned belief head** — running (does a learned UPDATE beat the fixed additive operator?).

## REMAINING after squeeze
Token-heavy paper-writing (T1/T2/T5a/T7b/T8/T9 + #15 opt-gap finding) — numbers frozen, safe to write —
plus T3 human-study kit. The ECIR critical path is the writing (T2/T8 for C+A).

## HELD (deliberately, to avoid redo / CPU oversubscription)
- All text tasks — T1 (#8), T2 (#9, unblocked but held), T5a (#10, unblocked but held), T7b (#11,
  unblocked but held), T8 (#12), T9 (#6), T8-optimization-gap (#15). Token-heavy; wait until gating
  numbers are frozen. **R2 means T2 + T8 must state the FIDELITY-CONDITIONAL / clean-answers-only scoping.**
- ML-25M I2 port (#7) — big dependency-free CPU job; data is local (`data/ml25m`, prior Paper-B port work
  exists). Release when a slot frees.
- T3 human-study kit (#13) — user-run prep; parallelizable later.

## MEMORY updated
New memory `squeeze-r2-adaptivity-clean-only.md` records the R2 overturn (corrects the prior
"train-noisy actor recovers adaptivity" note).
