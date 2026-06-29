# Bot-play design v2 — validating the answerability assumption (Paper C cherry) — RESEARCH-GROUNDED

**Status:** design only, nothing run. Supersedes v1. Incorporates (i) the answer/answerability factorial, (ii) the
warm-start curriculum, (iii) a literature pass (uncertainty, LLM user-simulators, the Baum-Lang answerability frame).

## Goal
The continuous actor (D1) was trained+evaluated against a GEOMETRIC oracle answerer `a = NEG+(POS-NEG)*(cos(u*,q)+1)/2`,
answerable for ALL directions. The snap-loss shows the gain LIVES off-manifold — exactly where that oracle is least
credible. Bot-play replaces the oracle with a learned, heteroscedastic, possibly-refusing answerer and asks: **does the
off-manifold tail gain survive?** Every outcome is publishable; the factorial below LOCATES which idealization the gain needs.

## The geometric oracle bundles THREE idealizations (they factorize)
1. **Answer shape** — `a` exactly LINEAR in cos(u*,q). Learnable: `mu(u*,q)` trained on real ML-1M rating residuals
   (captures real nonlinear, item-specific response; resid-scale, consistent with the encoder's item tokens).
2. **Answerability/coverage** — EVERY direction answerable, uniform reliability, never refuses. Learnable: `sigma^2(u*,q)`
   / refusal, EXPERIENCE-GROUNDED (proximity of q to the user's experienced item cloud).
3. **Noiselessness** — deterministic answer. Falls out of the heteroscedastic sigma^2 (aleatoric part).

### The 2x2 ablation (user request #1)
{geometric answer, learned mu} x {universal answerability, experience-gated refusal}.
| | universal | refusal (sigma^2>tau) |
|---|---|---|
| geometric answer | current headline 0.378/0.178 | isolates the ANSWERABILITY TAX (perfect answer, gated coverage) |
| learned mu | isolates REALISTIC ANSWER SHAPE | full realism |
Diagnostic: if the gain only dies in the REFUSAL column => the prize is in unanswerable regions => Paper D's LLM renderer
is NECESSARY (renders an off-manifold direction as an answerable NL question). Ties C->D.

## ABot: the answerer
- Input (u*, q, exp). Output (mu, sigma^2). mu=graded answer (resid-scale); 1/sigma^2 = answerability.
- mu trained on REAL ratings: for (user x, rated item j): target=resid[x][j], q=Qhat_j, u*=enc(full profile).
- **sigma^2 must be EPISTEMIC (grow off-manifold), NOT just aleatoric.** Research caveat: a bare heteroscedastic NLL head
  is OVERCONFIDENT off-support; deep evidential regression "systematically underestimates epistemic uncertainty"
  (Amini 2019; UQ-metrics survey 2024; Kendall&Gal 2017). RELIABLE off-manifold uncertainty comes from:
  - (PRIMARY) **experience-distance feature** exp(q|history) = [max_j cos(q,Qhat_j), mean top-5] over the user's rated
    items = deterministic distance-aware uncertainty (cf. DUQ/DDU); directly encodes the answerability thesis.
  - (VALIDATION) a small **deep ensemble** (Lakshminarayanan 2017): disagreement off-manifold = epistemic check that
    sigma^2 tracks OOD-ness, not fitted noise. Use as a sanity baseline, not the production answerer.
- **Experience = the user's FULL history, not the known half.** A user can answer about any film they've watched. ABot's
  exp uses the full rated set; the ASKER still sees only the half-profile belief (realizable boundary intact).
- **VALIDATION GATE (P0, before any use):** (a) sigma^2-vs-|err| calibration corr > 0; (b) sigma^2 off-manifold >
  on-manifold (the whole point); (c) mu fidelity vs the geometric model on held items. If sigma^2 is flat -> ABot is
  circular (a re-skinned oracle) -> STOP, add experience feature / ensemble.

## Curriculum (user request #2) — warm-start, but ASYMMETRIC + ANCHORED
- P0: train ABot on real ratings (mu fidelity + experience-grounded sigma^2). Calibrate sigma^2 ALSO on D1's actual
  off-manifold query distribution (roll out D1, collect q; no ground-truth mu there, so assess sigma^2 via exp/ensemble).
- P2 co-train: warm-start the ASKER from D1 best + ABot from its P0 fit, then iterate.
  **DANGER (collusion):** jointly optimizing asker AND answerer for the SAME NDCG reward => the answerer becomes
  over-confident everywhere / invents a private code => collapses back to the idealized oracle (the "over-cooperative
  simulator" failure named in the LLM-sim reliability critique). AVOID via asymmetry:
  - ABot objective = FIDELITY to real ratings + CALIBRATION (honest user model). NEVER the asker's reward.
  - Asker objective = NDCG under the fixed/anchored ABot.
  - ABot may be RE-CALIBRATED on the asker's shifted query distribution (sigma^2 re-fit) but mu stays anchored to ratings.

## Gated experiment ladder (cheap -> expensive; gate before spending)
- **P0** Train+validate ABot. GATE = calibration + off>on sigma^2 + mu-fidelity. (cheap)
- **P1 (decisive cheap probe)** FROZEN D1 under the 2x2 answerers, seed-avg, tail. The key gate:
  geom+refusal vs geom+universal. If the drop is small -> the gain is ALREADY answerable -> strongest result, STOP.
  If large -> proceed to P2. (cheap; no training)
- **P2** Co-train asker from D1 best vs anchored ABot (REINFORCE or unroll through mu). Does it RECOVER by finding
  answerable-off-manifold directions? Measure: tail NDCG, avg sigma^2 of asked q (should drop), snap-loss still positive
  (still off-manifold). (expensive; only if P1 fails)
- **P3** Refusal/answerability COST vs discrete; risk-coverage curve (selective-prediction framing, Geifman&El-Yaniv 2017).

## Outcome ladder (all publishable)
1. Gain survives geom+refusal (P1) -> off-manifold prize is already answerable -> deployable, strongest.
2. Gain partially survives; recovered by co-training (P2) -> quantified answerability tax + adaptation works.
3. Gain craters under refusal, unrecoverable -> HONEST negative: prize needs users who grade off-manifold OR the LLM
   renderer -> makes Paper D necessary. (Baum-Lang trap confirmed in the continuous setting.)

## Related work (citable framing)
- **Answerability of synthesized queries:** Baum & Lang 1992 (already in bib) — membership queries a learner synthesizes
  are often unanswerable by humans (off-manifold); the 30-yr precedent for our exact concern. Preference-elicitation/
  query-learning backdrop (CMU JMLR 2004).
- **Uncertainty:** Kendall & Gal 2017 (aleatoric vs epistemic); Lakshminarayanan 2017 (deep ensembles); Amini 2019
  (evidential regression) + its known miscalibration critique; Guo 2017 (calibration/ECE). Selective prediction /
  reject option: Geifman & El-Yaniv 2017, Chow 1970.
- **LLM user simulators for CRS (2024-25):** RecUserSim (WWW 2025), UserSimCRS v2, "How Reliable is Your Simulator?"
  (2024) — realism+diversity but documented leakage, over-reliance on history, persona drift, OVER-COOPERATION; NONE
  models answerability. We trade NL-surface realism for answerability-realism + control + calibration; NL realism is
  deferred to Paper D's LLM renderer. This is the gap we fill.

## Honest risks
- Circularity (ABot ~= geometric): #1 risk; mitigate with experience feature + P0 gate.
- u* leakage: ABot uses u* (it IS the user); fine, but the ASKER must see only the belief (boundary intact).
- Collusion in co-training: avoid via asymmetric anchored objective (above).
- mu off-manifold extrapolation: learned mu naturally degrades off-support (toward mean) — that degradation IS part of
  realism, but P1 must separate "answer-shape" effects (universal column) from "coverage" effects (refusal column).

## Code status
ABot class + BOTPLAY block drafted in continuous_actor.py (P0 train+validate, P1 frozen stress-test). NEEDS revision per
v2: (i) experience feature -> FULL history not half; (ii) add the 2x2 answerer modes explicitly; (iii) ensemble-validation
option; (iv) P2 asymmetric-anchored co-train. NOT run.
