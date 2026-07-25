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

- Strong-recommender contrast: harness built, all arms smoke-passed, G/B/O running; S/E/C/(P) queued.
- Weak-recommender (Paper B): two-model interview built (`db92d3a`); encoder in geometric-warmup epochs.
- Records: this file + memory notes; findings in `external_literature/`. Certification of the instrument
  remains parked pending the winner selection (separate track).
