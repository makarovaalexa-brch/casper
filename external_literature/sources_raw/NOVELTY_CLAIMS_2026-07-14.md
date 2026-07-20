# NOVELTY CLAIMS — for adversarial destruction (2026-07-14)
Every claim below is to be ATTACKED. If a claim dies, it dies here, not in review.

---

## ⚠⚠⚠ C2 IS FALSIFIED AS WORDED. THE REPLACEMENT IS STRONGER. (lit sweep, 2026-07-14)

**THE FALSIFIER — PERE, Nguyen et al., arXiv 2406.00973 (2024)** [venue UNVERIFIED, possibly UAI 2024 — CONFIRM].
Same recommender (LightGCN / biVAE) across ALL arms; metric NDCG@10; and their STATIC arm is literally
**Sepliarskaia's SPQ** (our own best supporting paper, used as their baseline). Adaptive WINS:
| | PEO (static) | c-DPP (sequential, NOT adaptive) | PERE (adaptive) |
|---|---|---|---|
| Amazon-Books | 0.2218 | 0.2901 | **0.2918** |
| Gowalla | 0.3108 | 0.3575 | **0.3616** |
=> **"Nobody has shown adaptive beating static with the same recommender" IS DEAD. DO NOT WRITE IT.**

**BUT THEIR OWN CONTROL SAVES US — AND MAKES THE CLAIM STRONGER.** `c-DPP` is a DIVERSITY selector over the
un-queried items. **It is not adaptive at all.** It recovers **97% of PERE's entire margin over static**
(0.2901 of 0.2918; 0.3575 of 0.3616). PERE's actual response-conditioned machinery contributes **+0.0017 /
+0.0041**. **THE VISIBLE WIN IS THE PIPELINE, NOT THE ADAPTIVITY.**

### ⭐ C2′ — THE REPLACEMENT CLAIM (defensible against every primary source)
> **No prior work has ATTRIBUTED an adaptive elicitation gain TO ADAPTIVITY ITSELF.** Where both arms appear
> in one table with the recommender held FIXED, **either the static arm WINS** (Sepliarskaia RecSys 2018;
> Greedy SLIM 2024 at realistic budgets) **or the adaptive win is NOT ATTRIBUTABLE** — PERE's own
> NON-ADAPTIVE sequential control recovers **97%** of its margin over the static arm.
**Only THREE papers in the corpus put both arms in one table**: Sepliarskaia (static wins), Greedy SLIM
(static wins), PERE (adaptive wins, unattributable). Golbandi's win is ASYMMETRIC (tree node-means vs a
seed-restricted item-item model, RMSE). **This makes the CONFOUND load-bearing rather than the OUTCOME** — and
our job precise: **build the symmetric, ATTRIBUTABLE test nobody has built, and report the honest answer.**
**PRESENT THE NEAR-MISSES IN A TABLE.** A "nobody has" claim defended by HIDING its neighbours dies; defended
by DISPLAYING them, it is the strongest card we hold.

### ⭐ C5′ — THE POLICY GETS A REAL DELTA AFTER ALL: **WE AMORTIZE A PRIVILEGED ORACLE**
Task-utility-trained acquisition is **NOT new in recsys**: Golbandi 2011 (squared loss, explicitly
*"deviate from... entropy or Gini impurity"*); **FacT-CRS (CIKM 2022)** splits on a **BPR ranking loss**;
**UpsRec (WWW 2023)** is a **greedy NDCG selector**; Greedy SLIM (2024). **DROP the criterion novelty.**
**BUT [VERIFIED]: UpsRec is a PRIVILEGED ORACLE** — it computes NDCG using *"the groundtruth items of the user
(in the validation set)"* and **ASSUMES the answer will be "liked."** All 11 of its forward citations were
swept: **NOBODY AMORTIZED IT.**
> **OUR DELTA: a LEARNED, AMORTIZED, ANSWER-MARGINALIZED value head that needs NO GROUND TRUTH at selection
> time and NO PER-CANDIDATE RECOMMENDER REFIT. UpsRec is the privileged greedy oracle we amortize.**
**AND KOREN NAMED OUR GAP HIMSELF (WSDM 2011):** *"In the future we would like to experiment with other cost
functions, especially ones related to the quality of the top-K item ranking."*
**ALSO [VERIFIED]: NO BOED→recsys port exists** (TNDP/DAD/EPIG/ALINE un-ported). That corridor is our framing.
⚠ **SELF PRIOR-ART: our own IJCNN 2024 paper already used "reduction in the recommender's loss" as a reward.**
A reviewer can quote the author against her own "nobody". **Any "first" must explicitly scope our own paper out.**

### ☠ THE HIGHEST-VALUE UNKNOWN — GET THIS PDF BEFORE WRITING A WORD
**Hu & Yu, "Interview process learning for top-N recommendation", RecSys 2013, DOI 10.1145/2507157.2507205.**
Abstract [VERIFIED]: *"our model is able to handle wide ranges of loss functions and can be used in
collaborative ranking task."* **THAT IS GOLBANDI'S STATED FUTURE WORK, DONE IN 2013.** Paywalled.
**IF it does per-node RANKING-METRIC splits, our ranking-target delta collapses to AMORTIZATION ALONE.**
Also to verify: PERE's venue, and re-derive its c-DPP gap ourselves (it is now load-bearing).

---

## THE PROPOSED CONTRIBUTION, IN ONE SENTENCE
> **The first elicitation policy that beats a static questionnaire against a SOTA collaborative recommender —
> and the reason the field kept failing is that it optimised INFORMATION instead of VALUE.**

---

## CORRECTIONS FROM THE ARCHITECTURE LIT SWEEP (2026-07-14) — READ BEFORE WRITING

### THE LANDMINE: NEVER COMPARE OUR NDCG TO SASRec/BERT4Rec's
**There is NO valid head-to-head between the sequential (SASRec/BERT4Rec) and the Mult-VAE/RecVAE lines —
THE PROTOCOLS ARE INCOMPATIBLE.** Sequential = **leave-one-out (ONE held-out target)**; Mult-VAE line =
**strong generalization (80/20 fold-in, MANY targets)**. SASRec's "NDCG@10 ~0.15" and our **0.4852** are
**NOT THE SAME QUANTITY**.
> **If we ever write "SASRec 0.15 vs our 0.485", a knowledgeable reviewer kills the paper on that line.**
**IT CUTS BOTH WAYS: nobody can prove SASRec IS RecVAE-class either — the number does not exist.**
=> **REQUIRED EXPERIMENT: run SASRec (+BERT4Rec/gSASRec) ON OUR RULER** (same catalog, same strong-
generalization fold-in, full catalog, NDCG@10) as a baseline row beside 0.4852. **This comparison appears
NOWHERE in the literature.** It is the ONLY honest way to make C1, and it is itself a contribution.
VERIFIED (RecVAE Table 1, ML-20M, strong generalization): RecVAE R@20 0.414 / NDCG@100 0.442; Mult-VAE
0.395 / 0.426; EASE 0.391 / 0.420.

### TWO STATEMENTS IN C1 ARE FALSE AS ORIGINALLY WRITTEN
1. **"Mult-VAE structurally CANNOT represent a 3-answer interview" — WRONG.** Its input is a SET-INDICATOR
   vector: permutation-invariant, variable-cardinality. A 3-item bag IS a legal input — it is
   **OUT-OF-DISTRIBUTION, NOT INEXPRESSIBLE.** The REAL, narrower gap (genuinely inexpressible in a one-hot
   input basis): **(i) GRADED/SIGNED VALUES per entity**, **(ii) OUT-OF-CATALOG CONCEPTS**, **(iii) ARBITRARY
   CONTINUOUS DIRECTIONS.** Claim ONLY those three.
2. **DEFINITIONAL TRAP: SASRec IS token-based and CAN eat a 3-item interview as a short sequence.** So
   "interview-native = token input" is TRIVIALLY satisfied by SASRec. Interview-native must mean
   **VALUE-CARRYING, MIXED-TYPE, CONTINUOUS-CAPABLE** tokens — where SASRec cannot follow.
3. Say **"we are aware of none"**, never "there are none".

### PRIOR ART WE MUST CITE (we would look ignorant otherwise)
- **TaNP (WWW 2021, arXiv 2103.06137)** — a PERMUTATION-INVARIANT encoder mean-pooling encoded
  **(item, rating) pairs**. THE ancestor of the set-input, value-carrying, cold-start recommender. MUST cite.
  Also **MeLU (KDD 2019)**.
- **M&Ms-VAE (RecSys 2021)** — encodes a user **from KEYPHRASES ALONE**, but via **TWO SEPARATE encoders**
  fused by mixture-of-experts, NOT one shared table; no continuous token; not on ML-20M.
- **P5 (RecSys 2022)** — items/attributes/reviews as text tokens in ONE shared space. Strongest precedent.
- **ConTS (TOIS 2021)** — unifies attributes+items as **BANDIT ARMS (an ACTION space)**, not an encoder INPUT.

### THE ONE LEG THAT SURVIVES CLEAN — LEAD WITH IT
**A CONTINUOUS QUERY AS AN *INPUT TOKEN*: the sweep found NOTHING, anywhere.** Not steering a latent, not
generating one, not a retrieval query — accepting an arbitrary continuous direction as a token in the SAME
entity space as items and concepts. **This is the cleanest, strongest leg of C1.**

### SAFE FORMULATION OF C1
> *"One shared token space in which ITEMS, CONCEPTS and ARBITRARY CONTINUOUS DIRECTIONS are interchangeable
> VALUE-CARRYING inputs to a single set encoder, at RecVAE-class full-profile accuracy."*

### C4 IS DOWNGRADED — DO NOT CLAIM DISCOVERY
Coarse-to-fine is the KNOWN behaviour of adaptive-interview decision trees (Golbandi: popular root,
discriminative deeper) and Rashid already has the popularity-vs-entropy tension. **Pitch as EMPIRICAL
CHARACTERISATION:** nobody MEASURES the popularity inversion (root **9/800** vs within-cluster median
**456/800**), and the **niche within-cluster polariser** statistic appears unreported (confidence MODERATE).
VERIFIED contrast: the Google 2025 paper (arXiv 2510.12015) **HARD-CODES** the funnel — *"We first use an LLM
to rank the tags from general to specific"* — and does not even specify a downstream recommender.
UNVERIFIED, load-bearing: whether Golbandi reports node-depth popularity stats. **READ THE GOLBANDI PDF.**
UNVERIFIED: the "328-person user study" in arXiv 2607.06765 is NOT supported by its abstract. Do not cite it.

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
