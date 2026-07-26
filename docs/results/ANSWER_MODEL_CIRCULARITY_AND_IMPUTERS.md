# The answer model is the load-bearing variable — circularity, and the non-circular imputer panel

> Durable record (2026-07-25). Companion memory: `answer-simulator-circularity-ruling`,
> `attribute-affinity-recipe-selplus`. Research: `external_literature/findings/simulator_circularity_conv_rec.md`,
> `external_literature/findings/attribute_affinity_recipes.md`. Live experiments below.

## 1. The finding (why this is big)

"Answerable concepts beat item-asking" is **not a property of concepts — it is a property of how the
simulated user answers.** Evidence chain:
- Old June-2026 Paper B: with a naive (`mean`) answer, concepts **HURT**; the single change that flipped it
  to **+36% tail** was the **geometric** answer (git: PART X→Y→Z, `scripts/_archive/paper2_old/`).
- Under our current **behavioral SEL** answer, concepts weaken back toward losing to items.
- Therefore the answer simulator is the load-bearing assumption in every concept-elicitation claim. Any
  concept result must state its answer model and defend it.

## 2. Taxonomy of answer models

| id | answer model | independence | status |
|----|--------------|--------------|--------|
| **G** | **Geometric**: like/dislike = whichever fold moves belief closer to `u*` = the recommender's own fold of the known-half profile | **CIRCULAR** (shared latent geometry) | **UNCITABLE** for eval (Fable ruling) — training/shaping signal only |
| **B** | **Behavioral SEL**: signed NPMI watch-lift + shrunk residual rating, from fold-in only | agent-independent (raw logs) | honest baseline; **beatable** |
| **O** | **Oracle-B**: same SEL formula over the user's FULL history (known+held) | agent-independent but PRIVILEGED (uses held-out) | declared **ceiling** only |
| **S** | **SEL⁺**: BM25 signed lift + empirical-Bayes shrinkage to a tag-genome content prior | grade-(i) model-free | lead non-circular upgrade candidate |
| **E** | **ExpoMF**: content-conditioned exposure model over member items (Liang 2016) | grade-(i/ii) separate generative model | exposure-confound fix / robustness |
| **C** | **Content-projection**: tag-genome-projected user profile · concept direction | grade-(i) model-free, content only | coverage-complete, unsigned/positive-leaning |
| **P** | **PITF / TagMF**: separate user×tag factor model (Rendle 2010 / Loepp 2018) | grade-(ii) SOFT (separate params, same CF family) | admissible ONLY under the guard |

**Independence grades:** grade-(i) = no CF parameters anywhere (cleanest: B, O, S, C). grade-(ii) = a
separate learned CF-family model (E partially, P) — soft independence, must pass the guard.

## 3. The circularity ruling (Fable, HIGH conf ~0.9) + literature

Geometric is a **circular-measurement flaw**, not a target leak: `u*` excludes held-out targets, but the
simulator and recommender share one latent geometry, so "did this answer help?" is scored in the model's own
coordinates → concept-vs-item becomes "how well the encoder embeds each channel," not user information. This
is **shared-representation self-preference** (Panickssery/Gao NeurIPS'24) and **target-biased shortcut-taking**
(PEPPER, Kim'24: 0.86 recall on primed targets vs 0.12 residual). RecSim legitimizes a model-defined user
ONLY if agent-independent AND behavior-calibrated — geometric violates both. It is exactly what our own
**C2/G7 firewall** already bans ("recommender-geometry answers appear nowhere in train/eval").
**The user's fair point** (a human DOES know their taste) holds — but that self-knowledge lives in **data
space, not model space**; the legitimate noise-free version is **oracle-B**, not geometric (which sides with
the model when encoder and behavior disagree; stated-vs-behavioral agreement is only ~21%).

## 4. Is SEL good enough? No — it is a beatable baseline

SEL/NPMI is a principled agent-independent **moment baseline**, clearly beatable **non-circularly** on three
weaknesses: (a) exposure/popularity confound, (b) zero coverage of unobserved attributes, (c) heavy-user
count-weighting. The panel above targets all three without touching recommender geometry.

## 5. Validity guard (ships with every imputer)

1. **Representational (CKA/HSIC, Kornblith 2019):** `CKA(answerer geometry, recommender u*) ≤ CKA(SEL, u*) + ε`.
   Behavioral SEL sets the admissible ceiling. Geometric should light up HIGH (the circularity, made visible).
2. **Cross-model differential-benefit:** the answerer must lift RecVAE-class, EASE, and item-kNN
   **indistinguishably** — no preferential lift for the evaluated tower.
3. **PEPPER shortcut-signature control** + **oracle-B as the declared privileged ceiling**.

## 6. Decision procedure

Pick the imputer that **maximizes Spearman agreement with oracle-B subject to the CKA cap.** Empirical, not
assumed — the flagged risk is that a content prior re-imports content-*popularity* through the back door and
adds little over SEL; measure it.

## 7. Experiment design — the grid

**Rows:** answer models {G, B, O, S, E, C, (P)} × {item-asking, concept-asking}.
**Cols:** q ∈ {0,1,2,4,8}, full AND tail NDCG@10, 10k COLD_SEED test users, credit-neutral masking, leak-safe.
**Two recommenders (the 2×N grid):**
- **Strong** — signed C-lite (`cfold_signed_best.pt`) on the frozen i25 tower → `answer_contrast_newrec.json`.
- **Weak** — the faithful Paper B reconstruction-encoder retrained on ML-25M (weak biased-SVD+attn) →
  `pb_results.json`. Tests whether geometric's inflation **shrinks as the recommender strengthens**
  (u* → true taste).

**Key quantities per recommender:** G−B on concept-asking (self-preference inflation), B→O gap (honest
headroom), G vs O (does strong-model geometric converge to the ceiling?), and per-imputer {CKA-to-u*,
oracle-B agreement, does it close the SEL→oracle-B gap under the cap}.

## 8. Status (2026-07-25)

- Weak-recommender (Paper B): two-model interview built (`db92d3a`); encoder in geometric-warmup epochs.
- Records: this file + memory notes; findings in `external_literature/`. Certification of the instrument
  remains parked pending the winner selection (separate track).

### 8.1 PHASE-1 LANDED — strong-recommender G/B/O contrast (`answer_contrast_newrec.json`, 10k users)

Harness `src/instrument/answer_contrast.py` (committed `bc470a4`). Fixed orders identical across arms:
item-ask = popularity, concept-ask = polarization. Controls ALL pass: q0 = intercept 0.1279/0.0192
(canonical-snap), wrong-user shuffle q8 full 0.1252 (< intercept → collapses=True), leak_users=0. B snaps
to the prior suite exactly (concept-ask B ≡ conc-polarization; item-ask B ≡ items-pop).

Full/tail NDCG@10 by budget:

| row | q0 | q1 | q2 | q4 | q8 |
|---|---|---|---|---|---|
| concept-ask **B** | .1279/.0192 | .1323/.0348 | .1328/.0343 | .1433/.0395 | .1441/.0395 |
| concept-ask **O** | .1279/.0192 | .1326/.0350 | .1330/.0345 | .1441/.0408 | .1448/.0404 |
| concept-ask **G** | .1279/.0192 | .1322/.0349 | .1340/.0354 | .1444/.0409 | .1450/.0413 |
| item-ask **B** | .1279/.0192 | .1307/.0266 | .1305/.0315 | .1319/.0343 | .1673/.0504 |
| item-ask **O** (degenerate) | .1279/.0192 | .1113/.0285 | .0976/.0320 | .0761/.0344 | .0992/.0520 |
| item-ask **G** | .1279/.0192 | .1438/.0344 | .1495/.0412 | .1506/.0430 | .1723/.0549 |

**Guards:** CKA(answer-geom, u\*) B 0.351 / O 0.358 / **G 0.406** (G highest = visible circularity, as the
§5 guard predicts; B is the admissible ceiling). Spearman(value, oracle-B) B **0.921** / O 1.0 / G N/A.

**Key quantities (concept-asking, q8, paired bootstrap 95% CI):**
- **G − B = +0.0009 full CI[+0.00000,+0.0019] / +0.0018 tail CI[+0.0009,+0.0027]** — self-preference
  inflation on concepts is TINY (CI-clean but ~1e-3). The circular answer barely helps concepts.
- **O − B = +0.0008 full CI[+0.0003,+0.0012] / +0.0010 tail CI[+0.0004,+0.0016]** — honest headroom is
  also small: behavioral SEL already agrees with the oracle (Spearman 0.921).
- **G − O = +0.0002 full CI[−0.0008,+0.0012] / +0.0008 tail CI[−0.0002,+0.0018]** — straddles 0:
  the strong-model geometric answer CONVERGES to the honest ceiling on concepts.

**READ: on the strong stack the answer model is NOT load-bearing for the CONCEPT channel** — G/B/O concept
curves sit within ~0.002 at every budget; the concept lift over intercept (+0.016 full / +0.020 tail @q8)
is answer-model-invariant. The answer model IS load-bearing for the **ITEM** channel: the circular
geometric answer inflates item-ask by **+0.013/+0.019/+0.019/+0.005 full** at q1/2/4/8 (vs +0.0009 on
concepts), which **erases the honest short-interview concept win** — under G, item-ask beats concept-ask at
EVERY budget, whereas under honest B concept-ask wins q1–q4 (crossover to items at q8). So "concepts beat
items early" is a property of the CONCEPTS, not the simulator; if anything a self-preferencing simulator
HIDES the concept advantage by pumping items. (item-ask O craters because answering held-out target items
consumes them as evidence and they are credit-neutral-masked out of the metric — a degenerate arm, not a
ceiling for items.)

- Phase-2 imputer panel (S/E/C/P) DEFERRED pending author greenlight (after phase-1 + Paper B review).
  `SELPlus` built + committed (`d7dd460`), dormant; E/C/P slot into the same `--arms` interface.

## Concept granularity diagnostic (`concept_granularity.json`, completed 2026-07-26 02:25, 10k users, 1031 concepts, frozen t2i25_EP4 + signed C-lite)

**A1 — what the selector picks:** oracle-selected concepts are BROAD, not fine. Median member count of
oracle-pooled selections = 1320 (polarization/mass banks even broader ~4000) vs 291 across all 1031
concepts. The interview naturally reaches for high-coverage concepts.

**A2 — per-answer cold lift rises with breadth** (single honest B concept, terciles by member count):

| tercile | med members | mean answerable users | lift full@10 | lift tail@10 |
|---|---|---|---|---|
| fine | 90 | 1291 | **−0.0053** | **−0.0061** |
| medium | 295 | 3655 | −0.0005 | +0.0027 |
| broad | 1073 | 7271 | +0.0019 | +0.0074 |

Spearman(lift, member_count) = **+0.49 full / +0.59 tail** → fine-grained concepts HURT per answer; breadth
helps. **Verdict:** on the strong frozen stack the concept channel wants BROAD concepts — a coarseness
ceiling, consistent with the cos~0.83 bound. Fine niche concepts do not compound.

**C-lite vs C-full (concept-ask B, identical answers):** near-identical; the fold-to-point C-full head buys
a hair at low budget (q1 full .1380 vs C-lite .1323) but OVER-COMMITS at q8 (C-full .1378 < C-lite .1441).
Extra fold capacity is not load-bearing for concepts.

## PHASE-2 imputer panel — is SEL beatable non-circularly? NO (`answer_contrast_imputers.json`, completed 2026-07-26 03:01, 10k users, arms B/O/E/P, ~36min)

Ran the two grade-(ii) separate-model imputers against the B/O anchors on concept-asking (strong stack:
frozen t2i25_EP4 tower + signed C-lite). **E = ExpoMF** (content-conditioned exposure, Liang 2016),
**P = PITF/TagMF** (separate user×tag factor model). Same fixed polarization order, credit-neutral, leak=0.

Concept-ask full@10 by budget (q1/2/4/8), q8 tail, q8 mean-answered:

| arm | q1 | q2 | q4 | q8 | tail q8 | ans q8 |
|---|---|---|---|---|---|---|
| **B** behavioral SEL | .1323 | .1328 | .1433 | **.1441** | .0395 | 7.9 |
| **O** oracle-B (privileged) | .1326 | .1330 | .1441 | **.1448** | .0404 | 8.0 |
| **E** ExpoMF | .1277 | .1271 | .1303 | **.1293** | .0203 | 8.0 |
| **P** PITF/TagMF | .1248 | .1243 | .1258 | **.1253** | .0205 | 8.0 |

**Key deltas @q8 (paired bootstrap 95% CI, concept-asking):**
- **O − B = +0.0008 full CI[+0.0003,+0.0012] / +0.0010 tail** — the honest headroom a *better* imputer could
  capture is tiny (re-confirms phase-1: SEL already agrees with the oracle at Spearman 0.921).
- **E − B = −0.0147 full CI[−0.0162,−0.0132] / −0.0192 tail**; **E − O = −0.0155 full** — ExpoMF does NOT
  beat SEL; it sits at/below the q0 intercept (barely moves the belief).
- **P − B = −0.0187 full / −0.0190 tail** — PITF is worse still, dropping *below* the intercept at every budget.

**Guards:** CKA(answer-geom, u\*) E **0.098** / P **0.108** (both far under B's admissible ceiling 0.351 →
cleanly independent of the tower geometry, as intended). But Spearman(value, oracle-B) E **−0.070** /
P **0.158** (vs B 0.921) → their independence comes with near-zero agreement with the true taste ranking.

**VERDICT — SEL is the ceiling among realizable non-circular imputers.** No cheap model-free / separate-model
imputer beats behavioral SEL on the concept channel: the more-independent generative models (ExpoMF, PITF)
are *far worse*, not better. Combined with the tiny O−B gap, behavioral SEL is a strong honest baseline whose
small remaining headroom (+0.0008 full) is not captured by exposure- or tag-factor models. The §4 hypothesis
that SEL is "beatable non-circularly on exposure/coverage/count-weighting" is **not realized by E or P** on
this stack. (Phase-2 S = SEL⁺ and C = content-projection remain to be run for the full panel; E/P alone do
not dislodge SEL.)
