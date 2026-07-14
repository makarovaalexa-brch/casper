> **NOTE (2026-07-14):** any statement in this file that adaptivity is worthless, or that a static
> schedule is optimal, is **SUPERSEDED**. See `experiments/ADAPTIVE_PROBE_RESULT.md` (ADAPTIVITY PAYS,
> +47% TAIL) and `CLAUDE.md` HARD RULE #2. The belief layer is to be judged ONLY on whether it makes a
> BETTER RECOMMENDER and a BETTER POLICY — never on theorem-compatibility.

# DESIGN SHEET — Belief / confidence / shrinkage in the set+multinomial encoder (2026-07-13)
Status: DESIGN ONLY. No runs until the author signs. Supersedes PLAN_BELIEF_HEAD.md (dense-era).

## 0. THE CLAIM
The belief layer is NOT a feature bolted on for novelty. It is the principled mechanism that
CURES THREE FAILURES WE ALREADY MEASURED, and it falls almost entirely out of structure the
set-encoder ALREADY has. That is the reason to build it.

| already-measured failure | belief layer's cure |
|---|---|
| Additive concept operator drove concept-only NDCG BELOW the popularity intercept (0.258 -> 0.226). Diagnosed: COLLINEARITY — summing correlated concept vectors amplifies the shared genre/popularity axis. | Precision-weighted (whitened) pooling. The posterior mean is `Sigma_post (Sigma0^-1 z0 + sum_t lambda_t v_t e_t)` — correlated evidence is DISCOUNTED by the precision matrix instead of double-counted. This IS the FM cross-term / ConTS `B^-1`. |
| Forbidden hand-drawn `CONF = [0, 0.5, 1.0]` multiplied into value. | Confidence becomes a LEARNED per-token PRECISION `lambda_t`. It enters as a Bayesian weight, never as a multiplier on value. Value and confidence stay SEPARATE signals — the author's rule, now with a reason. |
| E0: realizable adaptive prize ~= 0 (-0.0003). Theory note: linear-Gaussian => optimal design is NON-ADAPTIVE. | If `lambda_t` depends on the ANSWER (know_well -> high precision, no_clue/refusal -> ~0), then `Sigma_post` DEPENDS ON ANSWERS, and adaptivity is restored *for a principled reason*. See §5 — this is the thesis-level payoff. |

## 1. THE STRUCTURE IS ALREADY THERE
The current encoder already has the two Bayesian objects, unnamed:
- `z0` (the empty-set parameter) IS the prior mean = the popularity intercept.
- tokens pooled into `z` ARE accumulated evidence.
So read it as a conjugate Gaussian update and the semantics we want appear for free.

**Prior**   `z ~ N(z0, Sigma0)`   (z0 = popularity prior; Sigma0 learned, diagonal + low-rank)
**Each answered token** (entity e, value v, knowledge k) = a MEASUREMENT along direction `e`:
`y_t = e_t^T z + noise`, with LEARNED precision `lambda_t = softplus(MLP(entity, value, know, refused))`
**Posterior** (exact, linear-Gaussian):
```
Sigma_post^-1 = Sigma0^-1 + sum_t lambda_t e_t e_t^T          # precision ACCUMULATES
z_post        = Sigma_post (Sigma0^-1 z0 + sum_t lambda_t v_t e_t)
```

What this buys, by construction (not by tuning):
- **SHRINKAGE**: few/weak answers => the sum is small => `z_post -> z0` = popularity. Cold-start
  degrades gracefully to the intercept AUTOMATICALLY. No heuristic, no schedule.
- **INTERCEPT EXACTNESS**: empty set => `z_post = z0` exactly. Gate G-intercept passes by identity.
- **THE CLOUD**: `Sigma_post` is the belief. Its shape says WHICH taste directions remain unknown.
- **WHITENING**: correlated concepts overlap in `e e^T`, so the inverse discounts them. The
  collinearity fix is not an extra step — it is what the posterior mean *is*.

## 2. COMPUTE — cheap exactly where it matters
Full covariance over d=512 is intractable for a 14,040-item profile. It does not need to be:
- **Full profiles (Phase A, n up to 14k)**: DIAGONAL precision-weighted pooling only (elementwise).
  Gives shrinkage + exact intercept at ~zero cost. No matrix inverse.
- **Interviews (Phase B, n <= 20 answers)**: the regime where the cloud actually matters has TINY n.
  Woodbury => invert an (n x n) matrix, n<=20. The FULL structured posterior is essentially FREE
  precisely where we need it. This is the key practical observation.

## 3. HOW IT'S TRAINED (the crux risk: calibration)
An unsupervised "confidence" head will happily collapse to a constant — it must be forced to MEAN
something. Two routes, and they compose:
- (A) **Variational / Mult-VAE-native**: the SOTA class we are already in (Mult-VAE/RecVAE) is
  ALREADY variational — `q(z|x) = N(mu, sigma^2)`, trained by multinomial ELBO. Take the KL to a
  LEARNED prior `N(z0, Sigma0)` (RecVAE uses a composite prior). Shrinkage then also arrives via the
  KL. Standard, published, SOTA-validated.
- (B) **Conjugate/Kalman pooling** (§1) as the AGGREGATOR, giving directional structure the diagonal
  VAE cannot express (needed for EIG).
- **Combine**: (B) as the pooling operator, (A)'s ELBO as the loss that CALIBRATES lambda. Nothing
  else forces lambda to be honest.

## 4. WHAT IT UNLOCKS
- **Question selection = D-optimal design**: pick the question that most reduces `Sigma`. Principled,
  no RL.
- **PAPER C, closed form**: the optimal CONTINUOUS query is the PRINCIPAL EIGENVECTOR of `Sigma` —
  the direction of maximum remaining uncertainty. A continuous actor with no policy gradient at all.
- **The concept-vs-item bandwidth story, finally with math**: items = narrow, high-precision
  measurements; concepts = broad, lower-precision, high-coverage ones. Plot belief entropy vs turn by
  question type — that figure IS the answerability/bandwidth trade the thesis argues about.
- **Refusal as signal**: `no_clue` = a measurement with ~zero precision on value but NONZERO
  information about obscurity (its own learned embedding). No multiplication anywhere.
- **Risk-aware ranking**: score with mean +/- uncertainty (UCB / Thompson) instead of the mean alone.

## 5. THE THEORETICAL PAYOFF (why this is the thesis, not a feature)
Memory/theory: in a LINEAR-GAUSSIAN model, `Sigma_post` depends only on WHICH questions were asked,
not on the ANSWERS => the optimal question sequence is precomputable => ADAPTIVITY IS WORTHLESS.
That is *exactly* what E0 measured (adaptive prize -0.0003) and why every learned policy tied static
entropy for a year.
**Answer-dependent precision breaks the tie.** If `lambda_t = f(value, knowledge, refusal)` — i.e. if
whether-and-how-well the user could answer is itself informative — then `Sigma_post` depends on the
answers, the belief becomes user-specific, and adaptive question selection has something to adapt TO.
So: ANSWERABILITY is not a side-constraint on elicitation; it is THE source of adaptive value.
The belief formalism turns our biggest null result into a derivation.
(Pre-registered honesty: this is a HYPOTHESIS. If the adaptive prize stays ~0 with answer-dependent
precision, we report that, and the E0 no-go stands and gets *stronger*.)

## 6. GATES (pre-registered, must be signed before any run)
- **G-strength [HARD]**: the belief encoder must STILL clear full-profile NDCG@10 >= 0.486. If the
  uncertainty machinery costs accuracy, it is not free and we say so.
- **G-intercept**: empty set -> popularity EXACTLY (should hold by identity; verify numerically).
- **G-collinearity [the decisive one]**: precision-weighted pooling must lift CONCEPT-ONLY NDCG back
  ABOVE the intercept (the additive operator gave 0.226 vs intercept 0.2551). This is the direct test
  of the whitening claim against an already-measured failure.
- **G-calibration**: bin users by predicted belief entropy; realized NDCG must be MONOTONE in it. A
  flat curve means the head is decorative — report it as such, do not ship it.
- **G-lambda-recovers-knowledge**: WITHOUT being told, does mean `lambda` order rough_idea <
  know_well? If yes, confidence is genuinely learned and the forbidden hand-drawn CONF table is dead.
- **G-adaptive**: EIG/D-optimal selection vs the static-entropy ruler, on the REALISTIC answerer
  (not oracle answers). Honest ruler, honest report.

## 7. RISKS
- Uncertainty heads collapse to constants (mitigate: ELBO/proper scoring rule; detect: G-calibration).
- Gaussian-in-z is an approximation; the decoder is a softmax (nonlinear). The belief is a *model*,
  not the truth. Its value is empirical (the gates), not aesthetic.
- Extra params/compute must not dent the Phase A number (G-strength).
- Sigma0 / lambda parameterization is a place hand-drawn constants could sneak back in. They must all
  be LEARNED. No tables.

## 8. STAGING (proposed)
1. Finish Phase A (pb2, FiLM, lam=0) -> a SOTA point-estimate set-encoder. Gate 0.486.
2. Add diagonal precision pooling + learned prior; re-clear G-strength + G-intercept. (Cheap.)
3. G-collinearity test on concepts (the decisive, already-failed experiment, rerun with whitening).
4. Interview regime: full structured posterior (n<=20, Woodbury) + calibration gates.
5. EIG / principal-axis question selection vs static entropy on the realistic answerer.
