---
name: belief-pool-elicitation-novelty-map
description: "Deep-research novelty verdict on the six Kalman belief-pool elicitation claims: Biyik et al. 2023 pre-empts the Gaussian-fold + unified-VoI claims; the DEFENSIBLE novelty is the implicit-vs-stated variance decomposition (Claim 4) and expected-NDCG-under-posterior VoI (Claim 5). Bibliography fix + framing guardrails."
metadata:
  node_type: memory
  type: reference
  originSessionId: a2a12990-a215-4a69-8853-6a98488dcf50
---

Jul 18 2026. Deep-research pass (full-text read of the central threat) on the six elicitation claims from the
Kalman belief pool ([[belief-pool-satisfies-elicitation-invariant]], [[implicit-watch-answer-carries-concept-signal]],
[[ndcg-voi-adaptive-selection-design]]). Verdicts:

**THE CENTRAL THREAT — Biyik, Yao, Chow, Hsu, Haig, Ghavamzadeh, Boutilier, "Preference Elicitation with Soft
Attributes in Interactive Recommendation," 2023 (arXiv:2311.02085, Google/Boutilier).** Full-text confirmed:
multivariate-Gaussian belief over user embedding phi_u, linear utility, **soft attributes = CAV directions in
the item space**, attribute answers folded by **Bayes rule** (their Eq. 7), and **unified item+attribute queries
selected by Entropy/InfoGain/EVOI** (Eq. 13, "BPER" combined acquisition). This IS "concept answer = direction,
fold into a Gaussian belief, pick item-or-attribute by VoI."

- **CLAIM 1 (Gaussian/Kalman fold of attribute answers into frozen CF space) = INCREMENTAL.** Biyik pre-empts the
  concept. Surviving wedge: Biyik EXPLICITLY says his posterior is "generally NOT Gaussian" (ordinal likelihood ->
  sampled/approx) and CO-TRAINS the encoder. Ours = **exact linear-Gaussian conjugacy (scalar obs <u,c>+eps,
  closed-form Kalman) in a truly FROZEN classical 512-d MF space.** Claim ONLY that instantiation. NEVER claim
  "first to treat attribute answers as directions in a CF space + Bayesian fold" -- that is Biyik's.
  Also: Toroghi/Sanner BCIE (SIGIR 2023, Gaussian-conjugate critique fold over frozen factorization);
  Guo & Sanner AISTATS 2010 (canonical linear-Gaussian/TrueSkill elicitation).
- **CLAIM 2 (unified item+attribute VoI in one Gaussian belief) = DONE.** This is exactly Biyik's BPER. DO NOT
  lead with it. Thin wedge: commensurability via EXACT conjugacy (our item & concept obs are both exact
  linear-Gaussian -> one closed-form VoI number; Biyik's is sampled/approx). Tractability nuance, not a capability.
- **CLAIM 3 (non-degradation: Bayes fold makes E[NDCG] non-decreasing in truthful answers) = INCREMENTAL.**
  Abstract principle is I.J. Good 1967 (total evidence) + Blackwell sufficiency. Novel = the RECSYS/RANKING-METRIC
  instantiation + the demo that a MAP/least-squares fold DECREASES E[NDCG] while conjugate-Gaussian does not.
  TWO REVIEWER TRAPS: (a) Good's guarantee needs the recommender to ACT OPTIMALLY under the belief -- an MF
  dot-product ranker does NOT optimize NDCG, so monotonicity is NOT automatic; show the score is a monotone
  functional of mu or prove per-metric. (b) It is EX-ANTE over answers; any single truthful answer can still
  lower realized NDCG -- NEVER state it per-user.
- **CLAIM 4 (implicit watch-lift ~40% vs stated/LLM ordinal ~1.7% of concept-affinity variance) = NOVEL.**
  THE STRONGEST claim. "Implicit beats explicit" (Hu-Koren-Volinsky 2008) and "LLM users unfaithful"
  (Yoon et al. NAACL 2024) are directional priors; NO ONE does this specific variance decomposition of latent
  concept affinity into popularity-corrected watch-lift vs stated/LLM ordinal on the same users (~24x fidelity
  gap). Frame as the DECOMPOSITION/MEASUREMENT (not "stated is weaker"); report as variance-explained of a PROXY
  affinity (state construction; ML-25M/our-concept-set-specific, not a universal constant).
- **CLAIM 5 (expected-NDCG-under-the-posterior VoI for question selection in frozen CF) = NOVEL but narrow.**
  Real empty cell: EVOI-elicitation optimizes UTILITY (no ranking metric); decision-focused learning TRAINS
  predictors, never SELECTS queries; Lin/Zhu/Wang/Caverlee WWW 2023 has a greedy-NDCG attribute selector but
  HEURISTIC (no Bayesian posterior/Gaussian belief). MUST cite Lin et al.; stake claim on "under the Gaussian
  posterior-predictive of a frozen CF recommender," NOT on "NDCG as objective" broadly.
- **CLAIM 6 (adaptive vs static theory) = DONE as theory -> cite as background, never a contribution.** Defensible
  = the EMPIRICAL demo that adaptivity pays where linear-Gaussian static-optimality does NOT apply (our +47% tail;
  value-dependent nonlinear answer model + NDCG objective break Krause-Guestrin data-independence). Consistent
  with HARD RULE #2/#3. NOTE this whole map's Kalman IS linear-Gaussian -> the +47% lives in the NONLINEAR fold
  arena (set-encoder), NOT the Kalman; keep the two arenas distinct in the writeup.

**MANDATORY BIBLIOGRAPHY FIX:** the "20 questions" paper is **Jedynak, Frazier & SZNITMAN**, "Twenty Questions
with Noise: Bayes Optimal Policies for Entropy Loss," J. Applied Probability 49(1):114-136, 2012 -- third author
is SZNITMAN, NOT Zhang. Also: Das & Kempe STOC 2008 is NOT the static=adaptive theorem (it proves submodularity
of variance reduction) -- do not cite it for that. Krause-Guestrin ICML 2007 + Krause-Singh-Guestrin JMLR 2008
are the correct static=adaptive-for-GP-variance anchors. Golbandi-Koren-Lempel WSDM 2011 = recsys adaptive
cold-start anchor.

**★ SECOND deep-research (Jul 18, discover-then-drill + acquisition):** (Q1) the DISCOVER-THEN-DRILL two-phase
schedule (variance-greedy warm-up -> decision-focused drill) = **DONE, NOT NOVEL** = Krause-Guestrin ICML'07
explore-then-exploit + phased-acquisition BO. DEMOTE to a design footnote, never a contribution. (Q2) the
NDCG-posterior VoI = **INCREMENTAL leaning DONE**: Biyik (Gaussian belief + item/attr directions + VoI, EVAL'd by
NDCG) + Lin WWW'23 (expected-NDCG selector but HEURISTIC, no posterior). Surviving sliver: Biyik's VoI INTEGRAND
is expected UTILITY (Eq 13 PEU-EU*, per agent-1 full-text read), NOT rank-discounted NDCG -> "the VoI integrand
IS the ranking metric as a true posterior expectation" is a GENUINE but THIN distinction. Not a headline.
ACTION: get Biyik full PDF, pin the integrand (agent-1 says utility -> thin distinction survives; if NDCG ->
Q2 collapses to DONE). **EMPIRICAL CONFIRMATION the acquisition doesn't earn its keep (`scripts/bpool_dtd.py`,
2500-cache): continuous discover-then-drill VoI TAIL +0.0604 < variance-greedy +0.0669 -- the mu-reading VoI
UNDERPERFORMS the answer-blind variance-greedy.** Reconciliation: the adaptivity PRIZE is REAL (E3 +0.0085,
empirical best-q2 = oracle best-response) but the cheap CONTINUOUS surrogate (borderline A-optimality VoI) does
NOT harvest it -> SELECTOR-QUALITY GAP (E3=ceiling, VoI=weak estimator), not a dead end. **STRATEGIC PIVOT
(recommended to author):** the belief-pool+VoI+implicit-answer is mostly Biyik+Lin+Hu-Koren-Volinsky; move the
HEADLINE to the OPEN free-text + continuous + LLM-verbalisation 3-channel elicitation ([[three-channel-continuous-
elicitation]]) = least pre-empted (Biyik is CLOSED-SET); belief-pool = enabling mechanism, implicit answer =
observation model, adaptivity (E3) = secondary result. DUE DILIGENCE PENDING: sweep open-text-answer belief
folding / generative-elicitation lit (different literature than the two sweeps done).

**BOTTOM LINE for paper positioning:** lead with Claim 4 (implicit-vs-stated decomposition) and Claim 5
(NDCG-posterior VoI); position Claims 1-2 as *tractability/unification via exact conjugacy in a frozen CF space*
(never "concept-directions" or "Bayesian elicitation" -- both taken by Biyik); cite Good 1967 (Claim 3) and
Krause-Guestrin 2007 (Claim 6) as background, claiming only the recsys/NDCG instantiation. Biyik 2023 must be the
primary related-work anchor and be cited on the FIRST page.
