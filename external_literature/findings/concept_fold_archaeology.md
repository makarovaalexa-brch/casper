# Concept / Attribute Folding — Complete Project Archaeology

Internal-work archaeology (per HARD RULE 4: record so nothing is re-derived). Reconstructs every
stage where concept answers were folded into the recommender, every operator tried, the value/answer
convention at each stage, and the exact evidence with file/commit citations.

**Scope note on demotion:** all pre-Jul-22 interview NUMBERS ran on the retired LLM-answer environment
and are DEMOTED to "credible directions, magnitudes re-establish" (CLAUDE.md "Settled"; memory
`pre-jul22-interview-results-demoted`). Post-Jul-22 numbers use the SEL behavioral answer + real decoder
bias. Both are tabulated below with the convention flagged.

---

## Summary table — every recorded instance concepts HELPED (or the operator class)

| # | Date | Stage / operator | Answer convention | Gain (held-out) | Saturation pt | Caveats | Cite |
|---|------|------------------|-------------------|-----------------|---------------|---------|------|
| 1 | Jun 2026 | Paper B `enc_attr` unified item+genre encoder (ML-1M) | GEOMETRIC (fold-toward-u*), like/dislike, learned Ec | genres > pop: full **+0.011**, tail **+0.009**; concept-EIG tail **+0.084** (~85% of item-EIG +0.099) | coarse "front-loaded, peaks ~q3–5 then plateaus" | genres coarser than items; items > genres | mem `paperB-reconstruction-encoder-breakthrough`; `docs/reference/ATTR_ANSWER_LOG.md:66` |
| 2 | Jun 2026 | Answerable-concept elicitation vs item-asking (ML-1M, T=8) | GEOMETRIC + learned concept channel | conc_pop tail NDCG **0.116 vs item 0.085 (+36%)**; full 0.315 vs 0.307 | — | item unanswerable cold (1.9/8); concept 8/8 | mem `answerability-concept-channel-works` |
| 3 | Jun 2026 | Bounded-weight additive coarse channel (`concept_answer.py`) | graded base-rate-debiased (lift/PMI), additive, BOUNDED weight ~0.4 | genre 0.320→0.393 monotone; C6 0.328→0.331 holds | growing weight → decline past ~q5 | small NDCG (blockbuster-saturated), value on Recall | `docs/reference/ATTR_ANSWER_LOG.md:72-82` |
| 4 | Jul 15 | Cross-attn FUSION token pre-test (concept-hate probe) | ordinal 5-level FiLM + xattn value transform | concept-hate SINKS members below prior (horror +2.96/−0.53) while item-hate stays + | — (probe, not NDCG) | joint `fusj` run later plateaued/DILUTED (tail 0.041→0.048→0.046) | mem `fusion-token-context-dependent-value`; commit `1f3389e`/`8a01921` |
| 5 | Jul 16 | Whitened centroid = a direction (correction) | — (representation fix) | membership AUC 0.44→0.94; specificity 6–88× through decoder | — | whitening not novel (all-but-the-top); naive fold trades full | mem `concepts-are-directions-after-whitening` |
| 6 | Jul 17 | popb-floor additive whitened residual + IDF (`reweight_geo2.py`) | graded, hand-set single β | TAIL **0.0440→0.0619 (+41%)**, FULL +0.0023 (never drops) | β basin 8–120; past ~120 full breaks | UNTRAINED, hand-set; a MANY-answer channel | mem `concept-fold-operator-additive-popb` |
| 7 | Jul 17 | Paper-B u1 fold-to-POINT (direct eval `u1_best.pt`) | graded rating RESIDUAL | kc=32 **+0.052 tail (+95%)**, COMPOUNDS monotone | few-shot HURTS (kc=1 −0.014) | our additive head SATURATES at +0.028 (470 concepts) | mem `concept-fold-additive-union-vs-point-intersection` |
| 8 | Jul 17 | Learned fold-in encoder + LEARNED emb on frozen paord (`train_foldin.py`) | graded residual, attention→point | all-concepts **tail +0.056 / full +0.029** (beats Paper B) | few-shot HURTS (kc=1 full **−0.171**); warm-compose fades K≥5 | trained on all-470 → subsets OOD | mem `foldin-learned-embeddings-beats-paperB` |
| 9 | Jul 18 | Kalman belief-pool (`bpool2.py`), closed-form, NO training | per-level calibrated t (hated −0.40…loved +0.61) | ORACLE-t tail q8 **+0.090** greedy / +0.068 random, MONOTONE never drops | oracle saturates ~q4 | only on popb FLOOR; craters full 0.167→0.097 on real bias | mem `belief-pool-satisfies-elicitation-invariant`, `kalman-incompatible-unified-redesign` |
| 10 | Jul 18 | SEL/VAL R2 decomposition (`bpool_r2.py`) | SEL implicit watch-lift (signed log2), VAL residual rating | SEL R2 **0.376**, VAL 0.042, SEL+VAL **0.398**, LLM 0.017 | — | R2 only (informativeness), not a fold | `scripts/bpool_r2.py` |
| 11 | Jul 19 | pbC = SetEncoder + concept token rows (`train_concepts_ord.py`) | UNIFIED ordinal 5-level FiLM ("signed recipe") | FULL **0.4946** / TAIL **0.3372** — item strength NOT degraded; ep1 random-concept tail 0.0438→0.0618 (+41%) | many-concept tail booster | best concept-CAPABLE ckpt on disk; not a few-shot lever | `docs/reference/RECOMMENDER_TIMELINE.md:47`; commits `1f3389e`,`8a01921`,`b002f20` |
| 12 | Jul 24 | Arm A (additive latent shift, `ConceptMean`) | SEL-graded clip[0.25,1], POSITIVE-only | m=1 +0.0064, m=2 **+0.0108 (peak)**, m=8 **−0.0288** (crater) | ~m=2 | additive-UNION saturates; author ruled UNACCEPTABLE | mem `concept-channel-escalation-2026-07-24`; `experiments/battery/concepts_only_curve.json` |
| 13 | Jul 24 | Fix A diversified selection (\|cos\|<0.5) | same clip[0.25,1] | **0.1537 / 0.0631** monotone to m=8 (+0.0258 full, +0.044 tail vs intercept) | monotone (crater was correlation artifact) | selection fix, not operator | mem `concept-channel-escalation-2026-07-24`; `concepts_only_fixAB.json` |
| 14 | Jul 24 | **Arm C-lite** trained gated fold-to-POINT (`concept_fold.py`, `cfold_best.pt`) | SEL-graded clip[0.25,1], POSITIVE-only | concepts-only **0.1717→0.2091/0.1222 @m8** (62% of items-k8); **item-PARITY @m2**; mixed m2k2 +0.0265 | monotone to m=8 | ONE FAIL: G5 member-AUC 0.617<0.8 (learned "what X-likers watch", not "members of X") | mem `concept-channel-escalation-2026-07-24`; `concepts_only_curve_armC.json` |
| 15 | Jul 24 | Arm C-full (concept tokens INTO tower) | SEL-graded → levels {6..9} | item cost ≈0 (val 0.3442≈0.3435) but concepts m8 0.2037 < C-lite; deployment WORSE | — | tokens-in-tower "bought nothing"; C-lite WINS the ledger | mem `concept-channel-escalation-2026-07-24` (LEDGER VERDICT) |

**Realizable-deployment caveat (the big one):** per-QUESTION with a fixed bank, EVERY rung is ~flat at
intercept (clite 0.130→0.116 by q16) because per-answer curves privilege each user's own top-SEL concepts
(unknowable pre-interview). Concept coarseness ceiling is FUNDAMENTAL: concept belief saturates cos≈0.83
with u* vs items→1.0 (`docs/EXPERIMENTS.md:46`). Success = reach ceiling efficiently + tail-strong +
coarse-to-fine + film-mute coverage, NOT item parity.

---

## Q1 — Paper B era "trained-in concepts" (pbC)

**Architecture.** pbC = the working attention **SetEncoder** (ISAB-style arbitrary-set encoder, interview-
native variable-length fold) with **concept token rows added to the token vocabulary** — a concept is
"another item-table row" folded through the SAME attention pool as items, scored by a MULTINOMIAL decoder
over the 18,430 items only (concepts never enter the item softmax → no collapse). Items + concepts +
emitted embeddings share **ONE 5-level ordinal FiLM** value channel (commit `8a01921` "Collapse to UNIFIED
ordinal grading"). Trained by "freeze recommender, train ONLY concept rows, mixed curriculum" (commit
`b002f20`, "Phase B ordinal trainer + signed recipe"). Reproduced by `train_concepts_ord.py`; eval
`concept_eval.py`; ckpt `pbC_best.pt`.

**Value/answer convention.** GRADED ORDINAL (5-level FiLM: hated/dislike/meh/liked/loved), SIGNED — this
is the fusion line where concept-hate is representable as below-prior (Q2, item 4). It is NOT the implicit
SEL answer (that arrives Jul 18) and NOT the clip[0.25,1] positive-only convention (that arrives Jul 24).
The June predecessor (`enc_attr`) used the GEOMETRIC answer (fold-toward-u*).

**Gains.** pbC preserved item strength: FULL **0.4946** / TAIL **0.3372** (vs item-only paord 0.4859 /
pb2 0.4852) — concept training did NOT degrade the recommender. Concept lift: at ep1, random concept
reveals lifted tail **0.0438 → 0.0618 (+41%)** (commit `1f3389e`).

**Saturation ~4–5 (author's memory confirmed).** The coarse concept/attribute channel is FRONT-LOADED:
"peaks ~q3–5 (genre 0.317→0.351), then plateaus/declines" (`docs/reference/ATTR_ANSWER_LOG.md:66`); the
Kalman oracle "saturates ~q4" (`docs/reference/CAMPAIGN_LOG.md:6`). Mechanism (ATTR_ANSWER_LOG UPDATE 4):
coarse CENTROIDS localize but cannot REFINE; growing weight on the coarse direction eventually demotes
blockbuster held-likes → decline. So concepts are a MANY-answer tail booster / coarse localizer, never a
few-shot high-precision lever on the strong model.

---

## Q2 — the July 2026 arc

**Whitened centroid directions (Jul 16, mem `concepts-are-directions-after-whitening`).** Earlier
"concepts aren't directions" was a raw-space artifact — the shared popularity component (random-pair cos
~0.51) masks genre geometry. FIX = center (subtract mean) + strip top-1 PC (= popularity axis). Then
membership AUC 0.44→0.94 (romance 0.44→0.83); specificity through the raw decoder 6–88× (horror 88×,
sci-fi 17×). Whitening = Mu&Viswanath all-but-the-top, a FIX not a contribution. Code:
`build_concept_dirs` in `belief_layer.py:190` (center → `pca_lowrank` top-1 PC → strip → member-mean →
normalize; returns whitened `d_c`, raw `d_raw`, IDF `w_c = 1/log1p(|members|)`).

**Fold-operator investigations.**
- *belief-token blur (WRONG operator):* folding a concept as a token INTO the single belief z displaces
  the popularity prior → full craters 0.258→0.226 (`reconciled.py:199-210`).
- *additive-UNION vs fold-to-POINT (Jul 17, mem `concept-fold-additive-union-vs-point-intersection`):*
  the additive head `Σ_c w_c·<dir_c>` scores the UNION of directions → redundant prestige tags re-add the
  same acclaim direction → SATURATES (+0.028 tail even with 470 concepts). Paper-B folds answers to a
  POINT u → frozen decoder applies full collaborative structure → INTERSECTION ("animation ∧ dark") →
  COMPOUNDS (+0.052 tail@kc32). Decisive because Paper-B u1 has the SAME ndcg10 as ours.
- *popb-floor operator + IDF (Jul 17, mem `concept-fold-operator-additive-popb`):* `score = popb +
  β·(Wd·whitened_dir)`, a SEPARATE additive popularity FLOOR (empty interview → score=popb exactly →
  cannot fall below intercept). WHITENED is right here (popb carries popularity; raw double-counts and
  craters −0.16 full). + IDF `w_c=1/log(1+|members|)` (kills broad-genre head damage) + per-user norm →
  FULL never drops, TAIL 0.0440→0.0619 (+41%) at β~120. This IDF/whitened/floor recipe survives into
  `ConceptMean` (`belief_layer.py:268`).

**Learned concept embeddings — the +0.056 tail claim (Jul 17, mem `foldin-learned-embeddings-beats-paperB`).**
Concept answers → small attention encoder → a POINT u in decoder factor space → `score = base +
gate·<u,Wd_i>`, gate ZERO-INIT (u=0 → intercept, no harm). Concept embeddings made LEARNED (init whitened,
L2-anchored), tower+decoder FROZEN. All-concepts eval: ep2 TAIL **+0.0564** / FULL +0.0287 (beats linear
head +0.028 AND Paper-B u1 +0.052). BUT gates: few-shot HURTS (kc=1 tail −0.029, full −0.171); warm-compose
fades K≥5. `scripts/train_foldin.py`, ckpt `foldin_A1_best.pt`.

**Fusion-token / FiLM (Jul 15, mem `fusion-token-context-dependent-value`; commits `1f3389e`,`8a01921`).**
Problem: FiLM `γ[level]·emb + β[level]` applies the SAME per-level coefficients to every embedding → can't
encode that "hate" means opposite things for items vs concepts. Model-free probe: item-hate is legitimately
WEAK-POSITIVE (exposure — a film-X hater still watches X's genre), but CONCEPT-hate is genuinely BELOW
population (know-gated Test-0b: concept-haters like the concept's films 0.64–0.94× pop — documentary 0.64,
horror 0.74). Fix = cross-attention FUSION token: `token = γ[v]·emb + β[v] + know[c] + XAttnFuse(q=Wq·emb,
KV={emb, Eval[v], Econf[c]})` with emb IN the KV set + full Wv → Wv can learn ≈−2I → anti-parallel token =
real negation; norm-routing gates it (concept rows norm 0.55 vs item 1.37). Wo zero-init → step-0 == FiLM
base exactly. **Pre-test PASS:** concept-hate SINKS members below prior (horror LOVED +2.96 / HATED −0.53;
documentary +2.15/−0.17) while item-hate stays POSITIVE (+4 to +5.5); mismatch-bind (loved-horror +
hated-romance in one interview) held independently. **BUT the joint `fusj` run plateaued/diluted** (cold-
concept tail 0.041→0.048→0.046; sign-flip specificity failed) — this was the "concepts aren't directions"
scare, resolved by whitening (item 5).

**Kalman / belief-pool (Jul 18–19, mems `belief-pool-satisfies-elicitation-invariant`,
`kalman-incompatible-unified-redesign`).** Closed-form Bayesian update, NO training: belief u~N(mu,Sig),
`score = popb + <mu,Wd>`; each concept answer = linear-Gaussian obs of `<u,d_c>` (d_c = whitened member
centroid). Kalman: `mu += K(t−<d,mu>)`, Sig only SHRINKS. FIRST operator whose NDCG never drops per
truthful answer: ORACLE-t tail q8 +0.090 (greedy) / +0.068 (random), MONOTONE. Per-level calibrated targets:
hated −0.40, meh −0.22, liked +0.27, loved +0.61 (SIGNED). **Killed:** it builds its OWN mu (not the
encoder's z) and only survives on the popb floor — fold-all-concepts with the REAL decoder bias craters
full 0.167→0.097. → pivot to PrecAcc / analytic belief layer (design (ii) decoupled: MEAN = frozen tower
fold + Arm-A linear concept shift; Σ = analytic precision accumulator, mean-path untouched).

---

## Q3 — the SEL/VAL R2 decomposition (`scripts/bpool_r2.py`, Jul 18)

Question: how much of oracle concept affinity `AFF = U·Dc` (U = paord full-profile belief, Dc = whitened
member direction) does each MODEL-FREE answer explain? Sample = 15,000 train users, users with ≥8 items.

**Exact construction (lines 50–56):**
- `p_item = cnt / cnt.sum()`; `pexp = Mbin @ p_item` (per-concept expected mass).
- Per user, over their item set: `n_c` = # rated members of concept c; `e_c = len(items)·pexp` (expected
  members under popularity).
- **SEL (implicit watch-lift, signed):** `SEL[u] = log2((n_c + 0.5)/(e_c + 0.5))` — watch-lift vs
  popularity; **can be negative** (avoidance). No clip here.
- **VAL (explicit shrunk residual rating):** `mr = rsum/n_c` where `resid = rating − item_mean[item]`;
  `VAL[u] = (n_c/(n_c + LAM))·mr`, LAM=3.0 (shrinkage toward 0). Signed.
- **WDPROJ\*** (diagnostic upper-bound, uses model geometry): `(resid·Wcw[items]).sum(0) @ Dc.T`.
- **REFUSAL:** `EXPO < TAU` (TAU=1.5) → excluded from that concept's regression (`pooled_r2` masks
  `~REF`). No exposure = refuse.
- **LLM ordinal:** `cvalT = clip(VL[:,:NC], 0, 3)` on cells where `KN≥1 & VL≥0`.

**pooled_r2 (lines 61–71):** per usable concept (member count ≥20), linear lstsq of AFF on the feature(s)
+ intercept, R² = `1 − var(resid)/var(y)`, averaged weighted by cohort size.

**Numbers (lines 72–77):** SEL **+0.376**, VAL **+0.042**, SEL+VAL **+0.398**, WDPROJ\* +0.069, LLM ordinal
**+0.017**. Conclusions: implicit >> explicit (~9×); item-derived answer viable (23× the LLM); raw watch
counts BEAT the model-geometry projection (0.376 vs 0.069) → non-circular by construction.

**Were the values signed here? YES.** In `bpool_r2` SEL and VAL are both signed and fed RAW into the linear
regression — this is an informativeness (R²) measurement, NOT a fold, so nothing is clipped. The negative
half of SEL is fully present and load-bearing.

---

## Q4 — where the [0.25, 1] clip entered (dropping the negative half)

**The clip did NOT exist in `bpool_r2` (Jul 18)** — SEL there is signed `log2` lift, unclipped, and used as
a regressor.

**The clip first appears when the concepts-only STANDALONE curve was built** (`concepts_only_curve.py`,
added commit `c89d4d2`, ~Jul 24). Two things happen together in `sel_top_concepts` (lines 99–110) and the
documented convention (docstring lines 21–24):

```
lift[counts < 2] = -inf                                  # thin support removed
pos = flatnonzero(isfinite(lr) & (lr > 1.0))             # ONLY positive-lift concepts kept
v = clip( log(lift_j) / log(lift_1), 0.25, 1.0 )         # graded-down to the user's top lift, floored 0.25
```

So the negative half is dropped in TWO ways at once: (a) selection keeps only concepts with `lift > 1`
(positive watch-lift), discarding avoidance/negative-lift concepts entirely; (b) the surviving values are
clipped to [0.25, 1.0] — every folded answer is a (weak) LIKE, floor 0.25, no dislike band. Users whose top
lift ≤ 1 are EXCLUDED from the cohort.

**This same convention propagates to every downstream fold:**
- `concept_fold.py:162` (Arm C-lite) — identical `clip(log(lift)/log(lift_1), 0.25, 1.0)`.
- `train_tower_t2.py:574-578` (`sel_value_to_level`) — "SEL-graded concept value v in [0.25,1] → FiLM level
  {6..9}; Concepts carry no dislike band pre-C3"; smoke-asserted `sel_value_to_level(0.25)==6`.
- `tradeoff_ledger.py:100` — `clip(nan_to_num(val, nan=0.25), 0.25, 1.0)`.

**Consequence + root-cause finding (Jul 25, mem `concept-channel-escalation-2026-07-24` "SIGNED-ANSWER
ARC"; `signed_sel_gate.py:6`):** the clip-up convention is the diagnosed root cause of the deployment-mode
decline (fixed-bank concepts-only 0.1304 → 0.1160, below the 0.1279 intercept by q16) — down a fixed bank
most answers are lift ≤ 1, so accumulated weak-positive "poison" drags NDCG down; implicit DISLIKE (negative
watch-lift) is COMPUTED then DISCARDED. Concepts structurally cannot say "no" although the C-lite operator
is sign-capable (flip test −0.201, CI-clean = a vacuous capability in deployment). An eval-only TRIPLE GATE
(`signed_sel_gate.py`) is pre-registered to test signed-SEL before any retrain (adversarial review:
negative-from-absence risks a volume×popularity artifact). NOTE the pre-existing SIGNED path in
`train_tower_t2.py:820-839`: the ITEM value map keeps dislike levels graded-negative (neutral +0.25); it is
the CONCEPT SEL channel specifically that was forced positive-only.

---

## Q5 — see the Summary table at top (every recorded held-out concept WIN, with convention/gain/saturation/caveat).

Key reading of the table: concepts have helped in FOUR distinct regimes —
(a) as a coarse many-answer TAIL booster (items 6,7,8,11: +0.05–0.06 tail at many concepts),
(b) as a NON-DEGRADING Bayesian channel (item 9, oracle only),
(c) as an ANSWERABILITY channel that beats item-asking cold-start (item 2, +36% tail),
(d) as a trained fold-to-POINT reaching near item-parity at low budget (item 14, C-lite item-parity @m2).
They have NEVER been a realizable FEW-SHOT precision lever on the strong item model without either dropping
full (belief-token/point-shift folds) or losing to an item question, and the coarseness ceiling (cos≈0.83)
is fundamental. All positive magnitudes except items 12–15 predate the Jul-22 answer-env reset and are
demoted to directions.
