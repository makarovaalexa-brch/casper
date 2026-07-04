# INSTRUMENT 2.0 — Phase 4b (stretch): the cross-domain policy battery on Goodreads

THE cross-domain question: does the flagship (**continuous + graded + adaptive** elicitation) hold on
**books**, on a certified EASE-class instrument (RecVAE-d512, 0.78× EASE headroom, P2/P3 CERTIFIED)?
This run finally tests the **pre-registered Goodreads predictions**
(`experiments/paper2/PREREG_GOODREADS_PREDICTIONS.md`) on a certified arena.

**Arena note (dated 2026-07-05):** arena = **I2 RecVAE-d512 certified composite** — the phase-1E composite
split (`base_comp.npz`, va=500 / te=500), restricted **top-20k universe** (where EASE-comparability holds;
MOSTPOP/EASE reproduce the phase-1E gate exactly), NDCG **@510 primary + @10**, Cremonesi head-33% tail,
fixed rng(123) held-out disjoint targets (single split — the GR protocol is not seed-averaged, unlike ML-1M).
This is the first VALID Goodreads arena for the predictions: the three phase-1..4 health-gate failures
(single-genre / cold-cohort / composite) are superseded by the P2/P3-CERTIFIED d512 composite instrument.

Foreground, chunked, no commits. Runner `scripts/instrument2/p4b_battery.py` (stages `static`/`train`/`eval`).
Protocol byte-identical to P4a except the GR arena substitutions above.

Fixed operator (P3 winner): additive `z' = z + eta·a·q`, z=0 cold seed, graded answer `a = cos(z*, q)`,
`z* = enc(profile likes in universe)`. **eta = 23** = the measured **GR mean ||z*|| = 23.16** (P4a/ML-1M
used eta=16 ≈ its mean ||z*||=17.2 — the same "re-inflate the unit query to the natural latent magnitude"
scaling rule, re-measured for books). Concept directions = **member-bag encodes of the shelf vocab**
(`concepts_comp.npz`, 1483 shelves with ≥20 in-universe members, popularity-weighted top-50 bag).

---

## Item 1 — STATIC BASELINES (no training), te=500 TEST, restricted 20k universe

| baseline | @510 full | @510 tail | @10 full | note |
|---|---|---|---|---|
| MOSTPOP (popb ranker) | 0.2921 | 0.0659 | 0.2045 | P2/EASE-diagnostic ref 0.2921/0.2045 ✓ (fidelity) |
| z=0 cold floor (decoder bias) | 0.0824 | 0.0286 | 0.0367 | P2 ref 0.082 ✓ |
| global-entropy top-8 concepts (graded) | 0.0601 | 0.0223 | 0.0043 | **fails below floor** — highest-decode-entropy shelves are diffuse junk (christian-fantasy, mm-fantasy, m-m-read…); naive entropy-over-concepts w/o answerability is harmful (same failure mode as ML-1M) |
| **lift-concepts-8 (graded)** = uent+GRAW analogue | **0.3712** | **0.2262** | 0.2646 | per-user top-8 answerable (lift-selected) shelf concepts, member-bag dirs |
| lift-concepts-8 (BINARY answers) | 0.3654 | 0.2176 | 0.2616 | graded − binary = **+0.006 / +0.009** on concepts |
| **decoder-SVD basis-8 (graded)** = P3 W1 static bar | 0.3760 | 0.1877 | 0.2406 | the continuous informative basis |
| decoder-SVD basis-8 (BINARY answers) | 0.2599 | 0.1354 | 0.1275 | graded − binary = **+0.116 / +0.052** — large inversion |
| **k=8 item-fold reference** (discrete) | **0.4837** | **0.3104** | 0.3704 | P2 A.2/B.3 lineage; the STRONG discrete channel |

### Static-bar readings (thesis claims, static-only) — and the ML-1M contrast
- **Discrete items DOMINATE the continuous static basis on GR — a reversal of ML-1M.** On books the
  discrete **item-8 fold (0.4837)** beats the continuous **decoder-SVD basis-8 (0.3760)** by **−0.108 full /
  −0.123 tail**, and beats the discrete **concept-8 (0.3712)** by **+0.112 / +0.084**. On ML-1M I2 the SVD
  basis-8 (0.4709) *beat* item-8 (0.4625, +0.008/+0.012). The item channel's dominance tracks the item
  effective rank: GR item-decoder rows PR = **160** (top-600 pool 78.6) vs ML-1M I2 **33** — books have a
  vastly richer item geometry that the native VAE fold exploits, and a fixed 8-dim continuous basis cannot.
- **Graded >> binary (answer-model inversion) SURVIVES.** On the continuous SVD basis, binarizing collapses
  0.3760 → 0.2599 full (−0.116); on the concept channel graded still edges binary (+0.006). The graded
  geometric answer remains load-bearing.
- **Answerable/informative concept selection matters enormously.** lift-selected concepts (0.3712) beat
  naive global-entropy concepts (0.0601, below floor) by **+0.311** — selection, not the concept channel
  per se, is what makes concepts work (rank-independent claim, prediction 4).

### PREREG-3 rank measurement (I2 latent, d=512; uncentered participation ratio of unit directions)
| geometry | GR I2 (d512) | ML-1M I2 (d512) | phase-1 (64-d Q_svd, prereg) |
|---|---|---|---|
| concept member-bag PR | **15.48** / 512 | 8.43 / 512 | GR composite 7.41 |
| item rows (all) PR | 160.3 | 33.3 | — |
| item rows (top-600 pool) PR | 78.6 | 50.3 | GR composite 12.30 |

The GR nameable-concept menu is **higher-rank than ML-1M's** (15.48 vs 8.43 in the I2 latent), and the GR
item menu is far higher-rank still (160 vs 33). This is the setup the rank law predicts: with a high-rank
concept menu, the discrete selector has many expressible directions, so the continuous actor's off-manifold
margin should SHRINK (prediction 3).

---

## Item 2 — D1-RECIPE ACTOR (differentiable unroll in latent space)

Policy MLP(z_t, turn) → unit q_t (512-d); belief update = additive operator (eta=23); T=8. Objective =
(1−cos(z_T,z*)) + 0.3·softNDCG (teacher = decode(z*) top-20 vs 108 sampled negs). Per the P4a lesson,
actors are **BC-warmed from the static decoder-SVD-8 basis** (from-scratch collapses) then unroll-fine-
tuned (lr 3e-4, 20 epochs, val-best on va tail), 3 TRAIN seeds. Train z* = subsample of 15k train users.

Post-BC-warm val full ≈ 0.373 (≈ the static SVD-8 bar); the fine-tune lifts val full to ≈ 0.43 — so
adaptivity adds ~+0.05 over the static basis on books (larger than ML-1M's +0.02).

| actor variant | @510 full | @510 tail |
|---|---|---|
| **D1-recipe actor (graded, adaptive)** — seed-avg 3 train seeds | **0.4240** (sd .0030) | **0.2167** (sd .0028) |
| per train-seed s0 / s1 / s2 (full) | 0.4249 / 0.4271 / 0.4200 | 0.2154 / 0.2206 / 0.2142 |
| D1-recipe actor (**BINARY** answers sign(cos)) | **0.2100** | **0.0810** |
| best_val_tail per seed (va) | s0 0.2141 / s1 0.2170 / s2 0.2100 | primary = s1 |

**Actor vs the reference channels (seed-avg, @510):**
| comparison | Δ full | Δ tail | reading |
|---|---|---|---|
| actor − discrete **item-8 fold** (0.4837/0.3104) | **−0.060** | **−0.094** | actor LOSES to the native item fold on books |
| actor − discrete **concept-8** (0.3712/0.2262) | **+0.053** | −0.010 | actor beats concept elicitation on full, ties/loses tail |
| actor − continuous **SVD-8 static** (0.3760/0.1877) | **+0.048** | **+0.029** | adaptivity beats its own static basis |
| actor **graded − binary** | **+0.214** | **+0.136** | graded ≫ binary — inversion SURVIVES massively |

**q-curve (primary actor s1, full / tail), turns 1→8:**
| turn | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| full | 0.026 | 0.154 | 0.210 | 0.233 | 0.256 | 0.360 | 0.406 | 0.427 |
| tail | 0.023 | 0.064 | 0.086 | 0.107 | 0.121 | 0.164 | 0.203 | 0.221 |

Monotone; front-loaded onto turns 6–8 (the actor completes the informative span), same shape as ML-1M.

---

## Item 3 — SNAP-LOSS (post-hoc snap each emitted q_t to nearest bank direction)

| condition | @510 full | @510 tail | Δ full | Δ tail |
|---|---|---|---|---|
| unsnapped actor (continuous, s1) | 0.4271 | 0.2206 | — | — |
| snap → nearest **concept** direction (1483 shelves) | 0.2821 | 0.1029 | **−0.145** | **−0.118** |
| snap → nearest **item-decoder-row** direction (20k items) | 0.2843 | 0.0933 | **−0.143** | **−0.127** |

Snapping is **catastrophic on GR (−0.14)** — the **same magnitude as ML-1M I2** (−0.147 concept / −0.158
item). Despite the high concept/item effective rank, the actor's emitted directions are **not** well
approximated by any single discrete concept or item direction: the continuous, off-manifold query is
**load-bearing for the actor** on books too. (This is a distinct claim from "the actor beats the native
item fold" — it does not; but its own queries cannot be discretized without −0.14 loss.)

---

## Item 4 — STATIC-8 CONTROL (isolating adaptivity value)

Greedy forward selection of 8 fixed directions maximizing va full-NDCG, pool = actor per-turn mean
directions + decoder-SVD-8 + PCA-z*-8, applied statically to all TEST users:

| static construction | @510 full | @510 tail |
|---|---|---|
| greedy-static-8 from actor query distribution (val-selected) | 0.3774 | 0.1616 |
| decoder-SVD-8 basis (strongest single static) | 0.3760 | 0.1877 |
| **adaptive actor** | **0.4240** | **0.2167** |

The strongest realizable single static ≈ 0.377 (greedy full / SVD-8 tail). **Adaptive − strongest single
static = +0.047 full / +0.029 tail** — the per-turn directions genuinely depend on the belief z_t, and
adaptivity is NOT reducible to a fixed better basis. Adaptivity is a **larger relative win on books** than
on ML-1M (there +0.020 full), because the fixed continuous basis is weak here (0.376) while the adaptive
actor recovers to 0.424.

---

## Item 5 — PAIRED per-user bootstrap (418 test users, @510 full)

| margin | mean Δ | 95% CI | p(Δ>0) |
|---|---|---|---|
| actor − **per-user strongest static** (max{greedy, SVD-8}) | **+0.0114** | [+0.003, +0.020] | **0.997** |
| actor − discrete **concept-8** (lift, graded) | **+0.0559** | [+0.040, +0.072] | **1.00** |

Adaptivity beats even the **per-user-oracle strongest static** (max of the two static designs picked
per user) **significantly (+0.011, p=0.997)** — a stronger adaptivity result than ML-1M, where the
per-user-max static bound was actually above the actor. The continuous-adaptive actor also beats discrete
concept elicitation significantly (+0.056). (Neither margin rescues the actor vs the native item-8 fold,
which is not a static-query design but a native encoder fold of real items.)

---

## PRE-REGISTERED PREDICTION VERDICTS

### Prediction 1 — concept-channel saturation LATE or absent within q8
Concept-only k-curve (lift-selected, graded, additive; @510 full), marginal per answered concept:

| k | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| full510 | 0.238 | 0.286 | 0.308 | 0.327 | 0.344 | 0.352 | 0.364 | 0.371 |
| Δ (marginal) | — | +0.048 | +0.022 | +0.019 | +0.017 | +0.008 | +0.011 | +0.008 |
| tail510 | 0.159 | 0.184 | 0.195 | 0.205 | 0.216 | 0.219 | 0.223 | 0.226 |

**CONFIRMED.** The marginal does NOT collapse by ~q4: it is still **+0.008 full / +0.004 tail at q8**
(~16% of the k1→k2 marginal), monotone throughout. With concept effrank **15.48**, the channel keeps adding
value across all 8 turns — exactly the rank-law prediction (contrast ML-25M genome rank 3.99, where the
concept channel saturated by ~q4).

### Prediction 2 — concepts-vs-items tail verdict (predicted conc−item tail ≥ 0)
concept-8 tail **0.2262** − item-8 tail **0.3104** = **−0.084**.

**REFUTED.** The registered directional prediction was conc−item tail ≥ 0 (the high-rank shelf directions
"should not pay the ML-25M tail penalty" of −0.007). Instead the tail penalty is **larger** on GR (−0.084),
not ≥0 — and larger than ML-1M I2's −0.045. The strong native item fold pulls tail NDCG well above the
concept channel here. This is **evidence AGAINST the rank law's generality for the tail claim**: high concept
rank did not translate into a non-negative concept-vs-item tail margin. Reported as registered, no
reinterpretation.

### Prediction 3 — continuous-over-discrete gap SHRINKS vs MovieLens
Continuous adaptive actor − strongest **discrete** graded selector, GR vs ML-1M I2:

| margin | ML-1M I2 | GR I2 | measured menu rank (I2 latent PR) |
|---|---|---|---|
| actor − discrete **item-8 fold** (full) | **+0.028** | **−0.060** | item PR: ML 33 → GR **160** |
| actor − discrete **item-8 fold** (tail) | +0.019 | −0.094 | — |
| actor − discrete **concept-8** (full) | +0.046 | +0.053 | concept PR: ML 8.4 → GR **15.5** |
| graded − binary (actor, full) | +0.374 | +0.214 | inversion holds both |

**CONFIRMED.** The continuous actor's margin over the strongest discrete selector (the native item-8
fold) **shrank and inverted**: from **+0.028** on ML-1M to **−0.060** on GR — smaller than the registered
ML reference (+0.024/+0.037) and now ≤0 on full, exactly as predicted for a high-rank nameable menu. The
correlation with the measured rank is clean: the GR item menu is far higher-rank (PR 160 vs ML's 33), so
the discrete item selector has vastly more expressible directions and the continuous actor no longer
out-reconstructs it. The margin over the *concept* channel (rank went 8.4→15.5, a smaller change) barely
moved (+0.046→+0.053), consistent with the smaller rank change. The **graded/binary inversion still holds**
(+0.214), as registered. Note the *snap-loss* (a different measure — whether the actor's own queries can be
discretized) did **not** shrink (−0.14 both datasets); prediction 3 is about the continuous-over-discrete
*selection* margin, which did shrink/invert.

### Prediction 4 — answerability + selection hierarchy persist

### Prediction 4 — answerability + selection hierarchy persist
Answerability (cold te cohort): **7.98 answered concepts / 8**, frac>1 = **0.998** — CONFIRMED (matches P3
W2.4's 8.0/8; vs the V1-era 0.14–1.0). Selection hierarchy: lift-selected/informative concepts (0.3712) ≫
naive global-entropy concepts (0.0601, below floor) = **+0.311** — the answerable/informative selector
dominates, a rank-independent claim. **CONFIRMED.**

---

## CROSS-DOMAIN VERDICT — does continuous + graded + adaptive hold on books?

Per-claim, ML-1M I2 (movies) vs GR I2 (books):

| thesis claim | ML-1M I2 (strong) | GR I2 (books, strong) | verdict on books |
|---|---|---|---|
| **continuous > discrete** | actor 0.491 > item-8 0.463 (+0.028) & > concept-8 0.444 (+0.046) | actor 0.424 **< item-8 0.484 (−0.060)**; > concept-8 0.371 (+0.053) | **PARTIAL / WEAKENS** — beats discrete *concepts*, LOSES to the native discrete *item fold* |
| **graded > binary** | 0.491 vs 0.117 (−0.374) | 0.424 vs 0.210 (−0.214) | **SURVIVES — massively** |
| **snap-loss (continuity load-bearing)** | −0.147 / −0.158 | −0.145 / −0.143 | **SURVIVES — same ~4× magnitude** |
| **adaptivity > static** | +0.020 full (sig); per-user-max static > actor | +0.047 full vs single static; **+0.011 vs per-user-max static (sig, p=.997)** | **SURVIVES — stronger on books** |

### Bottom line
On a **certified EASE-class Goodreads instrument** (RecVAE-d512, 0.78× EASE headroom), the flagship
**continuous + graded + adaptive** elicitation **largely holds, with ONE honest weakening**:

- **graded ≫ binary** (−0.214) and **continuity is load-bearing** (snap-loss −0.14, identical magnitude to
  ML-1M) both **survive unchanged** — binarizing the geometric answer collapses the actor, and the actor's
  emitted queries cannot be snapped to any discrete concept/item bank without catastrophic loss.
- **adaptivity survives and is *stronger* on books**: the adaptive actor beats even the per-user-oracle
  strongest static (+0.011, p=0.997) and its own static basis by +0.047 — because the fixed continuous
  basis is weak on the rich book catalogue while the belief-conditioned actor recovers most of the gap.
- **the one weakening — "continuous > discrete":** on books the continuous adaptive actor (0.424) is
  **outperformed by the native discrete item-8 fold (0.484)**, whereas on ML-1M it beat both discrete
  channels. This is the **pre-registered rank effect** (prediction 3, CONFIRMED): the GR item menu is
  effective-rank ~160 vs ML-1M's ~33, so folding 8 real items reconstructs taste better than 8 continuous
  probes along a fixed-width basis. The continuous actor still beats discrete *concept* elicitation (+0.053).

**Flagship cross-domain verdict:** continuous+graded+adaptive **holds on books** for the graded-answer,
continuity-load-bearing, and adaptivity claims; the "continuous beats discrete" claim **holds vs concepts
but fails vs the native item fold** on the high-item-rank book catalogue — a rank-predicted, honestly
reported limit, not a failure of the mechanism (the actor's queries remain genuinely off-manifold).

### Pre-registered prediction scorecard
| prediction | verdict | key number |
|---|---|---|
| 1 — concept saturation late/absent within q8 | **CONFIRMED** | marginal still +0.008 full @q8; concept effrank 15.5 |
| 2 — concepts-vs-items tail ≥ 0 | **REFUTED** | conc−item tail = **−0.084** (larger penalty than ML-25M's −0.007) |
| 3 — continuous-over-discrete gap shrinks vs MovieLens | **CONFIRMED** | actor−item8 +0.028 (ML) → **−0.060** (GR); item PR 33→160 |
| 4 — answerability + selection hierarchy persist | **CONFIRMED** | 7.98/8 answered; lift 0.371 ≫ global-entropy 0.060 |

### Durable artifacts (.cache/instrument2/)
`p4b_battery.json` (static + eval), `p4b_actor_s{0,1,2}.pt` (3 train seeds, val-best), train/eval logs
`p4b_train_s{1,2}.log`, `p4b_eval.log`. Runner `scripts/instrument2/p4b_battery.py`.
All numbers: te=500 TEST, fixed rng(123) split, restricted 20k universe, NDCG@510 primary; actor
seed-avg over 3 train seeds; per-user bootstrap over 418 test users.
