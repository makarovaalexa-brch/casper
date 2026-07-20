# PLAN — Belief + Shrinking on top of the strong SOTA model (2026-07-12)
Author-directed next big thing: give the recommender an explicit BELIEF (posterior) that ANSWERS SHRINK,
WITHOUT discarding the strong SOTA mechanics (the unified encoder+decoder, warm-started to 0.4961).

## 0. THE ONE DESIGN COMMITMENT (resolves the "linear can't out-score nonlinear" objection)
Do NOT replace the encoder's scoring with a linear posterior. Split the two jobs:
- **The strong unified encoder = the SCORER (kept, untouched).** It produces the point belief μ0 = enc(answers)
  and the item scores decoder(μ0). This is the 0.4961 mechanics; nothing about scoring changes.
- **The belief head = an UNCERTAINTY LAYER on top.** It tracks a covariance Σ around the belief, so the system
  KNOWS how sure it is. Σ is used for: confidence-weighting of answers, question SELECTION, and stopping —
  NOT to take over scoring. So we get "belief + shrinking" and keep SOTA scoring by construction.

## 1. WHERE THE BELIEF LIVES
Posterior over the latent belief θ ∈ R^d (d=512, the encoder's z-space that the frozen decoder scores):
θ ~ N(μ, Σ). Prior: μ0 = enc(∅) (the popularity prior z), Σ0 = τ²·I (wide = "know nothing yet").
Decoder (z→items) FROZEN throughout — item strength can't be touched.

## 2. AN ANSWER IS A MEASUREMENT (the shrinking mechanism)
Each answer (item OR concept — unified) is a measurement of θ along a direction:
- **Direction φ_a** = the encoder's MARGINAL response to answer a: φ_a = enc(a alone) − enc(∅). (For concepts,
  the member-bag deltas e_c are a proven-shaped init.) φ_a lives in the same z-space as θ.
- **Observation**: y_a = g_a (graded value, centered loved..hated) = θ·φ_a + noise.
- **Precision 1/σ_a² = CONFIDENCE** (the principled home for the knowledge axis):
  know_well → small σ (high precision, big shrink); rough_idea/"probably know" → large σ (low precision, small
  shrink, uncertainty RETAINED); no_clue/refusal → no update. THIS is the learnable-confidence mechanism.
- **Bayesian linear (Kalman) update**, closed form, per answer:
  Σ⁻¹ ← Σ⁻¹ + φ_a φ_aᵀ / σ_a²   (COVARIANCE SHRINKS along φ_a — the "narrowing")
  μ   ← Σ ( Σ_prev⁻¹ μ_prev + φ_a g_a / σ_a² )
- **DISLIKE (g_a<0) shrinks AND repels**: φφᵀ is PSD regardless of sign ⇒ Σ still shrinks (we get MORE certain);
  the negative g pulls μ AWAY from φ_a (repel its region). Both-signs-shrink / only-mean-carries-polarity is
  the asymmetry additive fusion structurally cannot express — the defensible novel mechanism.

## 3. HOW IT'S USED (three jobs, none of them "replace the scorer")
1. **Confidence-weighted fusion (helps SCORING under noisy answers).** The posterior mean μ automatically
   down-weights low-precision (rough/guessed) answers via σ_a². The FINAL score is the strong encoder on a
   confidence-consistent belief; concretely score = decoder( enc(answers) )  with the belief head providing
   the per-answer precision the encoder consumes (or a light residual correction decoder(μ) blended small-λ).
   NOTE: this only beats the fixed-CONF point model when answers are NOISY (realistic interviews) — meaningless
   on oracle answers (Fable). So it comes AFTER the realistic-interview regime.
2. **Question SELECTION (value-of-information).** Ask next the concept that most shrinks Σ toward the plausible
   items: argmax_a  ‖X Σ φ_a‖² / (σ_a² + φ_aᵀ Σ φ_a) (D/A-optimal-design criterion, ConUCB Eq.8 form). This is
   the adaptive-elicitation prize (Paper C) — the only place the posterior earns its keep over static ordering.
3. **STOPPING / calibration.** Stop asking when trace(Σ) (remaining uncertainty) is low enough; report belief
   confidence to the user/policy. Answerability head (re-add) gates selection to concepts the user CAN answer.

## 4. WHY SOTA IS NOT DISCARDED (by construction)
- Encoder+decoder FROZEN → the 0.4961 item path is literally unchanged; empty interview → μ0 = prior → intercept
  exactly; full profile → μ → enc(full) → 0.4961.
- The belief head adds parameters (per-answer φ precomputed from the frozen encoder; σ-map; prior τ) — it never
  edits the backbone. At λ=0 residual, the system IS the current working model; the head can only add.
- Fable's linear-Gaussian caveat is respected: the linear posterior does NOT score; the nonlinear encoder does.

## 5. IMPLEMENTATION PHASES (each gated; realistic answers FIRST — required for confidence to mean anything)
- **P-A (prereq): realistic-interview regime.** Train+eval on the distilled ANSWERER (noisy, coarse Likert,
  refusals, knowledge levels, strategy-selected questions) instead of oracle derive_concepts. Guards: retrain
  EASE excluding all eval held; Ce control; 173 transfer. WITHOUT this, confidence/σ is meaningless.
- **P-B: precompute measurement directions** φ_a = enc(a)−enc(∅) for all items+concepts on the frozen encoder
  (one pass). Cache. Fit the σ-map from the knowledge axis (start fixed {know_well,rough,no_clue}→{σlo,σhi,∞},
  then LEARN it if the data supports — the confidence ablation).
- **P-C: belief head as pure inference (no backbone training).** Kalman updates over an interview; score via
  confidence-weighted belief; compare to the fixed-CONF point model ON REALISTIC ANSWERS. Gate: does precision-
  weighting beat flat-CONF on held-liked when answers are noisy? (If not, confidence isn't learnable here.)
- **P-D: question SELECTION.** VOI ordering vs the static engagement/strategy ordering. PRE-REGISTER: adaptive
  must beat static by MDE at k≤8 (project history says this often TIES — E0≈0, R2 clean-channel-only — so this
  is the honest make-or-break for the adaptive claim). + answerability head for realizable selection.
- **P-E: dislike-shrinks-and-repels** behavioral + knockout (negative-half usefulness), the novel-mechanism gate.

## 6. GATES / HONEST CAVEATS
- Strength: frozen decoder ⇒ 0.4961 preserved by construction (assert intercept + full-profile == a0c at λ=0).
- Confidence learnable ONLY under noisy answers (P-A first); ablate flat-CONF vs precision.
- Adaptive SELECTION is the real prize but the risky one (history: adaptivity ties static) — pre-registered.
- Circularity (answerer duality): the measurable de-bias test (concept whose members ⊂ history → ~0 new
  info / Σ-shrink); Ce + 173.
- Novelty: posterior belief-tracking in a STRONG DEEP UNIFIED CF recommender + confidence-as-precision +
  dislike-doubly-informative — no published deep unified-arm ConTS successor (research). Selection/calibration,
  not scoring, is the claim.

## 7. OPEN DECISIONS (for Fable / author)
1. φ_a = enc(a)−enc(∅) (marginal) vs the member-bag e_c delta — which is the better measurement direction?
2. Score = confidence-weighted enc(answers) (encoder consumes σ) vs decoder(μ) blend — which keeps SOTA cleanly?
3. Learn the σ(knowledge) map, or keep fixed? (data-support question — the confidence ablation answers it.)
4. Sequence: is P-A (realistic interviews) a hard prerequisite, or can a minimal belief head be probed on
   oracle answers first as a wiring smoke-test (knowing confidence is meaningless there)?
