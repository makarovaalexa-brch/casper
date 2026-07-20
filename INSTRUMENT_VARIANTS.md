# INSTRUMENT — variants (post deep-research, for Fable extension) 2026-07-11
Requirements: see INSTRUMENT_REQUIREMENTS.md (R1-R8). Scorecard axes: latent z / nonlinear /
dislike+graded / STRONG vs EASE-RecVAE / posterior-for-adaptivity / sparse cold-start fold-in /
attributes / weakness. [V]=verified this session, [R]=recalled/search-summary.

## MAKE-OR-BREAK (settled)
NO single published VAE is BOTH graded/dislike-expressing AND shown competitive with EASE/RecVAE on
top-N. Liang'18 (multinomial >> Gaussian/logistic for top-N) is UNBROKEN [V]. The only graded+signed+
strong+cold-start existence proof is CoFFee — but it's LINEAR (fails R5). The dislike-expressing VAEs
(critiquing line) are never validated on the EASE/RecVAE ruler. => the route is a HYBRID: strong
multinomial CONSUMPTION backbone (R1) + graded/signed VALUE channel + retained latent + posterior
(R2/R3/R5/R8), then re-run the inertness test (IG4/IG5 value-zeroing) that already burned us.

## VARIANTS (ranked by fit to R1-R8)
V1. **TaNP — Task-adaptive Neural Process** (WWW'21, arXiv:2103.06137) [V]. latent z~N(mu,sig) YES /
    nonlinear YES / graded-explicit ratings (MSE, low=dislike) YES, confidence NO / STRONG UNPROVEN vs
    EASE-RecVAE (the risk) / POSTERIOR yes-best-in-class (amortized q(z|support)) / cold-start fold-in
    YES gradient-free few-shot BY DESIGN / attributes YES. ★ best for R5+R8+R4. Gap = R1 strength.
V2. **M&Ms-VAE / M&Ms-VAE+** (RecSys'21 / arXiv:2204.02162) [V]. latent YES / nonlinear YES / dislike
    NATIVE (pos+neg critiquing) YES / STRONG on critiquing/explanation bench, UNPROVEN on top-N ruler /
    posterior yes but selection user-driven not system-adaptive / multi-step fold-in YES / attributes
    FIRST-CLASS (keyphrases=R6). ★ best for R2/R3/R6. Gap = ruler strength + confidence not modeled.
V3. **Hybrid rating⊕consumption VAE** (Gupta&Chakraborty, KDD-DLDay'18, arXiv:1808.01006) [R]. The
    literal multinomial⊕Gaussian-value template. Dated/modest strength (the Liang'18-weak config) =>
    blueprint not drop-in; motivates a STRONGER (RecVAE) backbone under the same idea.
V4. **CoFFee — full-feedback ordinal tensor** (RecSys'16, arXiv:1607.04228) [V]. graded+dislike NATIVE,
    strong-for-cold-start, fold-in YES — but LINEAR (no nonlinearity, no posterior) => the GRADED
    LINEAR BASELINE/control that proves a nonlinear instrument buys adaptivity. Counterpart to FEASE.
V5. **DVMF — Deep Variational MF** (Neurocomputing'19) [R]. latent+posterior+explicit+nonlinear;
    strength on our ruler unknown; not elicitation-designed.
V6. **Bayesian PMF (VI/MCMC)** [R]. linear-Gaussian = THE trap; cite only as "why nonlinearity."

## SELECTION LAYER (on top of the winning scorer — the R5 "show adaptivity" mechanism)
Region-elicitation (UAI'24, arXiv:2406.00973); PABBO amortized preference acquisition (arXiv:2503.00924);
Sanner-group differentiable EVOI; Bayesian preference elicitation w/ LMs (arXiv:2403.05534). Info-gain /
EVOI over a posterior q(z|answers-so-far) is the adaptive-question objective.

## RULE-OUTS
DualVAE/CaD-VAE/MacridVAE (implicit-only, no dislike); Mult-VAE/RecVAE (positive-only, incumbent, sign
inert); FEASE (linear item-item, no latent). Keep RecVAE only as the R1 backbone DONOR in the hybrid.

## RECOMMENDED SHAPE (primary)
HYBRID: RecVAE-class multinomial CONSUMPTION backbone (R1) + TaNP-style amortized POSTERIOR over z
(R5/R8/R4) + M&Ms-VAE-style SIGNED/GRADED value + keyphrase channel (R2/R3/R6). Novel bit = fuse so the
value channel does NOT go inert (tested by IG4/IG5). Least-mod single model = TaNP + strength retrofit
(swap decoder toward multinomial/RecVAE head, keep posterior). Fallback = M&Ms-VAE+.

## OPEN HOLES FOR FABLE
(1) Any ORDINAL-likelihood VAE post-2024 claiming RecVAE parity (found none; Liang'18 unbroken)?
(2) Can TaNP's amortized decoder reach EASE/RecVAE strength WITHOUT losing the posterior?
(3) GRADED CONFIDENCE {no_clue/rough/know_well} is UNADDRESSED across the entire candidate space —
    likely a NOVELTY GAP. Where could it live (per-answer precision in the posterior update)?
