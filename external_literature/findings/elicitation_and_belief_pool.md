# Findings: Elicitation & the Belief Pool

*Golbandi trees, EDDI, Biyik Gaussian belief, VoI, conversational elicitation, adaptive-vs-static.*
Consolidated & deduped from the deep-research memory maps + NOVELTY_CLAIMS + LIT notes. Bib-keys traceable to `../INDEX.md`.

## What we concluded

- **Adaptivity PAYS in CASPER's arena**, contradicting the linear-Gaussian "static is optimal" theorems. On 150k users with NO policy/RL/optimization: one genre question → cluster → per-cluster best follow-up gives TAIL NDCG@10 0.0414 → 0.0608 (**+47%**, CI [+0.0186,+0.0203]). The non-adaptivity theorems (`golbandi2011adaptive` cites Krause-Guestrin ICML'07; Jedynak-Frazier-Sznitman 2012) assume a LINEAR-GAUSSIAN belief + VARIANCE-ONLY objective; CASPER has neither (learned nonlinear fold-in, NDCG task loss). **This is a HARD RULE — do not re-argue "static is optimal."**
- **The information-vs-value boundary is the spine (C3).** Ranking candidate questions by INFORMATION (answer entropy/variance) → clusters converge on ONE list (Spearman 0.795) and the adaptive policy LOSES to static (−0.0068). Ranking by TASK LOSS (realized NDCG) → lists become near-orthogonal (Spearman 0.062) and the policy WINS (+0.0194 tail). Nobody has isolated this as a boundary with BOTH sides measured on the same data.
- **Coarse-to-fine emerges, not imposed (C4).** Best question for everyone is a blockbuster (The Usual Suspects, pop-rank 8-9/800); best question for a cluster is a NICHE genre-congruent film (median rank ~456-471/800): horror→Poltergeist, sci-fi→Planet of the Apes, animation→Ghibli, comedy→Wayne's World, drama→Citizen Kane. The informative question for a homogeneous group is one that POLARISES it → effect is in the TAIL (+47%) not head (+6%). The "fine" question only EXISTS once the coarse answer says where to point it.
- **A closed-form Kalman belief pool satisfies the elicitation invariant** (NDCG never drops per truthful question) — the first monotone-positive per-step fold; but a cheap continuous VoI surrogate does NOT harvest the full adaptive prize (selector-quality gap, not a dead end).
- **Answerable-concept elicitation beats item-asking in cold-start** (Paper B): tail NDCG 0.116 vs 0.085 (+36%) — because cold items are mostly unanswerable while concepts are answerable and fold usefully. See `answerability_and_channels.md`.

## Key external methods

- `golbandi2011adaptive` — RMSE decision-tree cold-start interview; lovers/haters/unknowns split; explicitly deviates from entropy/Gini to squared loss (6 Qs vs 30+); "Napoleon Dynamite = maximally controversial, maximally uninformative." THE recsys adaptive anchor.
- `ma2019eddi` (EDDI) — permutation-invariant partial-VAE set-encoder over an arbitrary observed subset + greedy info-gain acquisition; the myopic discrete baseline CASPER generalizes; right architecture, non-recsys domain.
- `rashid2002getting` / `rashid2008learning` — popularity/entropy/HELF item-selection for new users; "Pure Entropy was the worst"; the popularity-vs-entropy tension.
- `liu2011wisdom` (RBMF), `fonarev2016rectangular` (RMVA), `shi2017local` (Local RBMF), `zhou2011functional` (Functional MF), `kweon2020deep` (DRE) — MF-embedding-as-seeds / interview-tree-in-latent-space elicitation lineage. Established; cite, don't claim.
- **Biyik et al. 2023 (arXiv:2311.02085, NOT in references.bib — must be added)** — multivariate-Gaussian belief over user embedding, soft attributes = CAV directions in item space, Bayes fold (Eq. 7), unified item+attribute queries by Entropy/InfoGain/EVOI ("BPER", Eq. 13). THE central threat.
- Also uncited-but-required: **Toroghi/Sanner BCIE (SIGIR 2023)** Gaussian-conjugate critique fold over frozen factorization; **Guo & Sanner AISTATS 2010** canonical linear-Gaussian/TrueSkill elicitation; **I.J. Good 1967** (total evidence) + Blackwell sufficiency for non-degradation; **Lin/Zhu/Wang/Caverlee WWW 2023** greedy expected-NDCG attribute selector (heuristic, no posterior); **Krause-Guestrin ICML 2007** + **Jedynak-Frazier-Sznitman 2012** static=adaptive-for-Gaussian-variance anchors.
- `vendrov2020gradient`, `viappiani2009regret` — Bayesian/regret preference elicitation background (in bib, not yet deeply summarized).

## What is NOVEL vs pre-empted

**PRE-EMPTED (cite, do not lead with):**
- **Gaussian/Kalman fold of attribute answers into a CF space = Biyik 2023.** NEVER claim "first to treat attribute answers as directions in a CF space + Bayesian fold." Surviving wedge only: EXACT linear-Gaussian conjugacy (scalar obs, closed-form Kalman) in a TRULY FROZEN classical MF space — Biyik's posterior is "generally NOT Gaussian" and he CO-TRAINS the encoder.
- **Unified item+attribute VoI in one Gaussian belief = Biyik's BPER (DONE).** Do not lead with it. Thin wedge: commensurability via exact conjugacy (one closed-form VoI number vs Biyik's sampled/approx).
- **Non-degradation principle = Good 1967 + Blackwell (INCREMENTAL).** Novel only as the recsys/NDCG instantiation + the demo that a MAP/least-squares fold DECREASES E[NDCG] while conjugate-Gaussian does not. TWO TRAPS: (a) Good's guarantee needs the recommender to act optimally under the belief — a dot-product ranker does NOT optimize NDCG, so monotonicity is not automatic; (b) it is EX-ANTE over answers — never state per-user.
- **Discover-then-drill two-phase schedule = Krause-Guestrin explore-then-exploit + phased BO (DONE).** Demote to a design footnote.
- **Adaptive-vs-static theory = DONE.** Cite as background; the defensible piece is the EMPIRICAL demo that adaptivity pays where static-optimality doesn't apply.
- **Coarse-to-fine (C4) = DOWNGRADED, do not claim discovery.** Known in Golbandi (popular root, discriminative deeper) and Rashid (popularity-vs-entropy). The only LLM paper reporting it (Montazeralghaem/Google 2025, arXiv:2510.12015) HARD-CODES it ("rank tags general to specific"). Pitch as EMPIRICAL CHARACTERISATION: nobody MEASURES the popularity inversion (root 9/800 vs within-cluster median ~456/800) or the niche-within-cluster polariser statistic (confidence moderate).

**NOVEL / defensible:**
- **Claim 4 — implicit-vs-stated variance decomposition (THE STRONGEST).** Implicit watch-lift explains ~40% of concept-affinity variance vs stated/LLM ordinal ~1.7% (~24× fidelity gap). "Implicit beats explicit" (Hu-Koren-Volinsky 2008) and "LLM users unfaithful" (Yoon NAACL'24) are directional priors; no one does THIS specific variance decomposition of latent concept affinity on the same users. Frame as the DECOMPOSITION/MEASUREMENT of a proxy affinity (dataset-specific, not a universal constant).
- **Claim 5 — expected-NDCG-under-the-posterior VoI in a frozen CF space (NOVEL but narrow).** Real empty cell: EVOI optimizes UTILITY (no ranking metric); decision-focused learning TRAINS predictors, never SELECTS queries; Lin WWW'23 has greedy-NDCG but heuristic (no Bayesian posterior). MUST cite Lin; stake the claim on "under the Gaussian posterior-predictive of a frozen CF recommender," NOT on "NDCG as objective" broadly.
- **C2 — the empirical hole:** nobody has ATTRIBUTED an adaptive elicitation gain TO ADAPTIVITY ITSELF with the recommender held FIXED (see `../findings` C2′ discussion). Only 3 corpus papers put both arms in one table: Sepliarskaia RecSys'18 (static wins), Greedy SLIM 2024 (static wins at realistic budgets), PERE (adaptive wins but its own NON-ADAPTIVE c-DPP control recovers 97% of the margin → unattributable). Present the near-misses in a TABLE.
- **C3 — the boundary with both sides measured** (info-ordering going backwards below static while task-ordering wins +47%, near-orthogonality rho=0.06 quantified). Delta over Golbandi 2011 is CONTESTED — attack hardest.

## Open questions

- Does the value-trained policy BEAT strong static on the 0.4852 recommender at full T? (Concept-only DKQS collapses to a worse-than-greedy static sequence — a TRAINER failure, not an adaptivity verdict; ITEMS are the load-bearing test.) See `continuous_action_policy.md`.
- Pin Biyik's VoI integrand: agent-1 full-text read says expected UTILITY (Eq 13 PEU−EU*), NOT rank-discounted NDCG → the "VoI integrand IS the ranking metric" distinction survives but is THIN. Get the full PDF to confirm; if it's NDCG, Claim 5 collapses to DONE.
- Does Hu & Yu "Interview process learning for top-N recommendation" (RecSys 2013, paywalled) already do per-node RANKING-metric splits? If yes, the ranking-target delta over Golbandi collapses to AMORTIZATION ALONE. HIGHEST-VALUE unknown.
- Read the Golbandi PDF to confirm it reports node-depth popularity statistics (load-bearing for C4 novelty).

## Bibliography fixes (binding)
- "20 questions" paper = **Jedynak, Frazier & SZNITMAN** (J. Applied Probability 49(1):114-136, 2012) — third author is SZNITMAN, NOT Zhang.
- Das & Kempe STOC 2008 is NOT the static=adaptive theorem (proves submodularity of variance reduction). Use Krause-Guestrin ICML 2007 + Krause-Singh-Guestrin JMLR 2008.
- **Biyik 2023 must be added to references.bib and cited on the FIRST page of related work.**
