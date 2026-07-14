# NOVELTY CLAIMS — for adversarial destruction (2026-07-14)
Every claim below is to be ATTACKED. If a claim dies, it dies here, not in review.

---

## THE PROPOSED CONTRIBUTION, IN ONE SENTENCE
> **The first elicitation policy that beats a static questionnaire against a SOTA collaborative recommender —
> and the reason the field kept failing is that it optimised INFORMATION instead of VALUE.**

---

## C1 — THE ARCHITECTURE: interview-native AND SOTA-class (the enabler)
**CLAIM.** One latent space in which an ITEM, a CONCEPT, and an ARBITRARY CONTINUOUS DIRECTION are all just
tokens, feeding a SOTA-class multinomial recommender (full-profile NDCG@10 **0.4852**; teacher a0c 0.4961).
Every strong recommender in the field (RecVAE, EASE, a0c/Mult-VAE class) consumes a FIXED DENSE CATALOG
VECTOR and therefore **cannot represent** a 3-answer interview, an open-vocabulary concept, or a continuous
query without folding the whole catalog as zeros. Every interview-native encoder we or the field had was
NON-SOTA. **We are the first with both.**
**WHY IT MATTERS.** It is what makes a continuous, snappable action space possible AT ALL. A dense model
cannot take a concept token or a continuous query. This is the enabler, not the headline.
**ATTACK IT:** has anyone built a SOTA-class recommender that natively consumes an arbitrary SET of
(entity, value) tokens spanning items AND concepts AND continuous directions? (Set-Transformer recommenders?
Any CRS with a unified token space at RecVAE-class accuracy?) If yes, C1 dies.

## C2 — THE EMPIRICAL HOLE IN THE FIELD (the strongest card)
**CLAIM.** **NOBODY has shown an adaptive elicitation policy beating a STATIC one when BOTH use the SAME
STRONG COLLABORATIVE RECOMMENDER.** Three independent lit sweeps found none. Where it WAS tested
symmetrically — **Sepliarskaia, Kiseleva, Radlinski & de Rijke, RecSys 2018** — **STATIC WON (p<0.01)**. The
field's famous adaptive wins (Golbandi WSDM'11; EAR/SCPR/UNICORN; Qrec; PEBOL; MCTS-CRS) are against weak
estimators, weak/asymmetric baselines, or pruning-oracle answer models — **they never run that contrast.**
**SO:** if our policy beats static on a 0.4852 recommender, it is a FIRST — and the hole is in the FIELD, not
just in our project.
**ATTACK IT:** find ONE paper that runs adaptive-vs-static with the same strong CF recommender on both arms.
If it exists, C2 dies.

## C3 — THE MECHANISM / THE BOUNDARY (the spine)
**CLAIM.** Rank candidate questions by **INFORMATION** (answer entropy / answer variance) => the clusters
converge on ONE LIST (Spearman **0.795**) and the adaptive policy **LOSES TO STATIC (-0.0068)**.
Rank them by the **TASK LOSS** (realized NDCG) => the lists become **NEAR-ORTHOGONAL (Spearman 0.062)** and the
policy **WINS (+0.0194 TAIL, +47%, CI [+0.0186,+0.0203], n=74,841)**.
`experiments/ENTROPY_VS_NDCG_RESULT.md` + `experiments/ADAPTIVE_PROBE_RESULT.md`. 150,239 users, no policy,
no optimisation, disjoint selection/eval halves.
**PRIOR ART WE MUST NOT PRETEND AWAY:** Golbandi/Koren/Lempel (WSDM 2011) explicitly *"deviate from entropy/
Gini"* for the **squared loss** and report **6 questions vs 30+** for the variance family (5x), with the
mechanism (*Napoleon Dynamite is maximally controversial and maximally uninformative*). Rashid et al.
(IUI 2002): *"Pure Entropy was the worst… shockingly poor."* TNDP (NeurIPS 2024) shows EIG-driven design is
suboptimal for downstream decisions.
**OUR DELTA:** nobody has isolated it as a **BOUNDARY with BOTH SIDES MEASURED on the same data** — the
information ordering going *backwards* (below static) while the task ordering wins by 47%, with the
near-orthogonality (rho=0.06) quantified. **Is that enough delta over Golbandi 2011?** ATTACK THIS HARDEST.

## C4 — COARSE-TO-FINE, EMERGENT AND MEASURED
**CLAIM.** The best question for EVERYONE is a blockbuster (The Usual Suspects, popularity rank 9/800). The
best question for a GROUP is a NICHE film INSIDE that group's own genre (median rank **456/800**): horror ->
Poltergeist; sci-fi -> Planet of the Apes; animation -> Laputa/Howl's/Totoro (all Ghibli); comedy -> Wayne's
World; drama -> Citizen Kane. **The informative question for a homogeneous group is one that POLARISES it.**
Coarse-to-fine FALLS OUT, and the fine question is **only AVAILABLE to an adaptive policy**.
**PRIOR ART:** the only paper reporting coarse-to-fine in an LLM recommender (Montazeralghaem et al. 2025)
**HARD-CODES it** (ranks tags "general to specific" in the training data) and never compares to a static
ordering. Xia et al. (2026 preprint) observe it in humans and build a router for it.
**ATTACK IT:** has anyone shown coarse-to-fine EMERGING (not imposed) from a task-loss objective?

## C5 — THE POLICY (honest: the WEAKEST novelty claim)
**CLAIM.** A value head `Q(z, e) ~ expected NDCG@10 after asking e`, trained by supervised regression on
REALIZED outcomes, deployed as `argmax_e Q(z,e)` over a unified action space (items ∪ concepts ∪ continuous).
**WHAT IS *NOT* NOVEL — SAY IT FIRST, BEFORE A REVIEWER DOES:**
- Training acquisition on DOWNSTREAM TASK UTILITY instead of information gain **IS TNDP** (Huang et al.,
  NeurIPS 2024, "Decision Utility Gain"). Also ALINE (2025). **The core idea is published.**
- Amortizing acquisition into one forward pass **IS DAD** (Foster et al., ICML 2021) and the whole
  amortized-BOED line.
- Learning a value function and taking the argmax is fitted-Q. Standard.
**=> A reviewer WILL say "this is DUG applied to recommendation." If we frame the paper as "we invented a
decision-theoretic policy", WE GET KILLED.**
**THE ONLY DEFENSIBLE DELTA:** TNDP/ALINE are BED, not recsys; they have no catalog, no collaborative prior,
no answerability, no refusals, no concept/item/continuous unified action space, and no strong recommender to
beat. **The delta is the APPLICATION + the ACTION SPACE + the BOUNDARY, not the estimator.**
**⚠ THE RISK THAT MUST BE CHECKED BEFORE WE WRITE:** if anyone has already ported task-utility-trained
acquisition to conversational recommendation, C5 collapses AND it weakens C2/C3. **THIS IS THE HIGHEST-PRIORITY
LIT CHECK.**

---

## THE FRAME (weak vs strong)
- **WEAK (gets killed):** *"we invented a new elicitation algorithm."*
- **STRONG (defensible):** *"the first elicitation policy that beats a static questionnaire against a SOTA
  recommender — and the field kept failing because it optimised information instead of value."*
The spine is **C3 (the boundary) + C2 (the hole)**, with the policy as the thing that EXPLOITS it and C1 as
the enabler. **C5 is a method, not a contribution.**

## HONEST STATUS OF EACH CLAIM
| | claim | status |
|---|---|---|
| C1 | interview-native + SOTA architecture | **MEASURED** (0.4852). Novelty UNCHECKED. |
| C2 | nobody beat static with a strong recommender | **LIT-SUPPORTED** (3 sweeps, none found). Needs one more adversarial pass. |
| C3 | information LOSES / task-loss WINS; near-orthogonal | **MEASURED** (150k users). Delta over Golbandi 2011 CONTESTED. |
| C4 | coarse-to-fine emerges (niche-within-genre) | **MEASURED**. Novelty plausible (prior work hard-codes it). |
| C5 | the Q policy | **NOT YET BUILT, NOT YET VALIDATED.** Method novelty WEAK by our own admission. |
**NOTHING IS A CONTRIBUTION UNTIL Q BEATS STATIC ON THE RULER.** If it does not, C5 dies and C2 dies with it.
