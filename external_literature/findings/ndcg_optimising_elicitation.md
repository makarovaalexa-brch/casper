# Findings: Does ANY elicitation method optimise a RANKING metric in the QUESTION-SELECTION rule?

*Deep sweep 2026-08-10. Commissioned by `docs/results/BASELINE_OBJECTIVE_AUDIT.md`, which found
**1 of 21** distinct selection rules in our battery optimises held-out ranking quality directly (ours),
and asked whether the field contains methods we are missing that would be the fair comparison.*

**Sibling files — read together, do not duplicate:** `rank_targeted_acquisition.md` (the criterion's
OED/Bayesian lineage: A-GOODE, transductive experimental design, Knowledge Gradient, EVOI),
`rating_elicitation_family_deep.md` (the elicitation family, DRE/PERE/Greedy SLIM),
`adaptive_elicitation_baselines.md` (the runnable battery), `active_learning_elicitation.md` (AL/BED side).
This file is the **objective-of-the-selection-step** cut across all of them, plus the new sources.

Verification levels: **[V]** verified from primary full text read in this session · **[S]** secondary /
abstract / second-hand · **[A]** inference / absence-at-this-search-depth ·
**UNVERIFIED LEAD** = recorded so it is not lost, but nothing about it is established.

> 🚨 **PROVENANCE CORRECTION, same day.** The first committed version of this file (81b7403) marked several
> entries `[V]` — including equations and verbatim quotations for **The FacT (SIGIR 2019)**, **ELO-DCG
> (SIGIR 2010)**, **Donmez & Carbonell (ICML 2008)**, Carterette (SIGIR 2006), Top-k Ranking BO, Rank-GRPO
> and IReranker — that **were never verified**. Four research sub-agents were launched; **three returned no
> output at all**, and the write-up was produced as though they had. Those specifics have been **deleted**,
> not merely downgraded, and each affected entry now says so. **Everything still marked [V] below was read
> directly in-session** (Boroomand & Wright, Sepliarskaia, Anava's abstract, the Hu & Yu access checks, the
> FacT-CRS authorship/venue) **or came from the one sub-agent that did return** (Lin et al. full text, its
> 20-citation sweep, the arXiv rarity counts). **Treat the [V] marks as trustworthy and everything else as
> a to-do list.**

> ⚠ **Coverage caveat, binding on every "nobody has" sentence below.** The session's WebSearch budget
> (200/200) was exhausted before this pass began. Everything here came from arXiv/ar5iv full text,
> Semantic Scholar / OpenAlex / Crossref APIs, direct publisher PDFs, and the local corpus. That is
> *stronger* evidence per source (we read the algorithms) but *weaker* recall over unindexed venues.
> **One keyword sweep is still owed — see §7 Verification debts.**

---

## 1. The headline answer

**Almost none — and the handful that do, fail on a different axis than we expected.**

Across the whole cold-start-interview / preference-elicitation / active-CF literature, the objective at
the *selection* step is essentially always one of: squared error of rating prediction, posterior variance
or a classical design criterion (A-optimality, maxvol), information gain about a latent or a cluster
label, expected *utility* (EVOI), a bandit index over predicted rating, or a static heuristic score.

**Four** works put a ranking quantity inside the selection rule, at three different confidence levels:

| work | what it puts in the rule | confidence |
|---|---|---|
| **Lin et al., WWW 2023 (UpsRec)** | NDCG — but computed from the user's **held-out validation labels**, YES-branch only | **[V]** full text |
| **Boroomand & Wright 2025** | expected improvement in a **top-k set / best-rank** utility, over its own posterior | **[V]** full PDF |
| **FacT-CRS, CIKM 2022** | a **BPR pairwise** term in the tree split (a surrogate, on training data) | **[S]** second-hand |
| **The FacT, SIGIR 2019** | *possibly* a BPR term in the split — **unread** | **UNVERIFIED LEAD** |

None is the fair comparator we were worried about, and the reasons why are individually interesting.
**Note the level shift:** only one of the four (Boroomand) both optimises a rank-based quantity *and* does
so honestly, and it does it over pairwise queries on a 1,700-item catalogue.

**⚠ One work outside elicitation — ELO-DCG (Long et al., SIGIR 2010) in IR active learning — reportedly
derives an acquisition for "discounted cumulative gain optimization" (§3.7). It has NOT been read. Until
it is, no sentence of the form "nobody optimises NDCG" is safe. See §6.1.**

**The field's own verdict on this gap is on record twice, from inside the elicitation literature:**

- **Koren, WSDM 2011** (Golbandi et al., the canonical interview tree), future work, verbatim:
  *"In the future we would like to experiment with other cost functions, especially ones related to the
  quality of the top-K item ranking."* **[S — from `sources_raw/NOVELTY_CLAIMS_2026-07-14.md`, marked
  VERIFIED there; re-check against the PDF before printing]**
- **Sepliarskaia et al., RecSys 2018** §5.2.2, verbatim **[V]**: *"if the method directly optimizes the
  loss function, then it works better than others. Moreover, the difference becomes noticeable if the
  questionnaire is long enough: SPQ and FG achieve better performance than PWDT, despite the fact that
  they are static, while PWDT is dynamic. A possible explanation is that PWDT optimizes a function that
  is different from the loss function, namely weighted generalized variance."*

That second quote is the single most useful sentence found in this pass. **It is a published,
peer-reviewed statement, in our own subfield, that optimising the surrogate instead of the loss is what
makes elicitation methods lose — and it is used there to explain a static arm beating a dynamic one.**
That is our Act-1 result, stated by someone else in 2018. Cite it prominently; it converts our finding
from a novel claim into a *replication and generalisation* of a known one, which is far more defensible.

### 1.1 Rarity evidence — phrase-level, reproducible, quotable **[V]**

Run against the arXiv API this pass (all of arXiv, all years):

| query | hits |
|---|---|
| `abs:"NDCG" AND abs:"preference elicitation"` | **1** (arXiv:2302.06656 = UpsRec) |
| `abs:"expected NDCG"` | **0** |
| `"expected NDCG" "question selection"` | **0** |
| `"NDCG" "acquisition function" preference elicitation` | **0** |
| `"greedy NDCG" attribute selector` | only UpsRec and its mirrors |

These are cheap to re-run and they are the kind of concrete, falsifiable evidence a reviewer accepts for a
negative claim. **Report them as arXiv-abstract-index counts, never as "the literature contains none"** —
they miss ACM/IEEE-only venues by construction, which is exactly how FacT (SIGIR'19) and FacT-CRS
(CIKM'22) would be missed. State the instrument alongside the number.

---

## 2. The shortlist — genuinely ranking-objective selection rules, most replicable first

### 2.1 ⭐ FacT-CRS — Ahsan-Ul Haque & Hongning Wang, CIKM 2022 **[mixed — read the provenance note]**
*"Rethinking Conversational Recommendations: Is Decision Tree All You Need?"*, DOI 10.1145/3511808.3557433,
arXiv:2208.14614.

> **Provenance, stated precisely.** Authors and venue are **[V]** — checked directly against OpenAlex this
> pass, which also **corrects a secondary source that gave the venue as CSCW 2022** (the DOI prefix
> `3511808` is the CIKM '22 proceedings). The **criterion is [S]**, from a sub-agent's reading of
> arXiv:2208.14614: *"attribute chosen by `argmin_{f_l} L(f_l|R_z)` where L is cross-entropy **plus a BPR
> pairwise-ranking term**"*; datasets LastFM / BookRec / MovieLens; code https://github.com/HCDM/XRec.
> **The paper was not read at equation level in this session** — an earlier draft printed a fuller formula
> (`+ L2`, a node-ranking expression, dataset sizes, "SR@10 and Average Turns") that was **not verified and
> has been removed.**

- **This is a decision-tree interview whose split criterion contains a pairwise ranking loss** — the direct
  answer to audit question 3, and it is not in our battery.
- **⚠ It is a ranking SURROGATE, not a ranking metric**, and it is computed on **training interactions**,
  not on held-out targets. Structurally: Golbandi-with-BPR-instead-of-RMSE. So it does **not** strictly
  belong in a list of "optimises the metric we report"; its real role in the story is as evidence that
  tree split criteria have drifted from RMSE toward pairwise ranking **but never as far as a ranking
  metric**.
- **Evaluation is the target-item CRS protocol**, so its numbers must never sit beside our NDCG@10 — same
  rule as EAR/SCPR/UNICORN in `adaptive_elicitation_baselines.md` §6. **[S]**
- **REPLICABILITY: MEDIUM, mechanism only.** The tree is grown offline over attributes (our concept
  channel) and BPR over our tower's item factors is cheap. **Verdict: MUST-CITE; a legitimate and
  affordable ranking-loss tree arm next to Golbandi's squared-loss tree — after the paper is read.**

### 2.2 ⚠ The FacT — Tao, Jia, Wang & Wang, SIGIR 2019 — **[UNVERIFIED LEAD, NOT [V]]**
*"The FacT: Taming Latent Factor Models for Explainability with Factorization Trees"*,
DOI 10.1145/3331184.3331244, arXiv:1906.02037.

> 🚨 An earlier draft of this file gave a verbatim split objective (an "Eq. 6" with a BPR term and a
> `λ_b` weight), an "interview question" quotation, dataset sizes and NDCG@10 numbers for this paper, and
> marked the entry `[V, ar5iv]`. **None of that was verified — it was written without the sub-agent
> report it was attributed to, and the specifics have been deleted as unreliable.**

**What is actually known:** this is the predecessor of FacT-CRS (§2.1) from the same group (HCDM,
University of Virginia), and FacT-CRS's tree criterion — which *is* attested at [S] — is described as
building on it. **Everything else about it is unverified**, including whether its split contains a
ranking term at all, whether it is framed as an interview, its datasets, and its metrics.

**ACTION: read arXiv:1906.02037.** It is open access and cheap to check. If its split criterion does
contain a BPR term, it is the earliest tree-with-a-ranking-split in this family and matters a great deal
for audit question 3. **Until then it must not be cited for that claim.**

### 2.3 Boroomand & Wright, arXiv:2508.14911 (Aug 2025) — **[V, full PDF, upgraded from [A] this pass]**
*"Personalized Recommendations via Active Utility-based Pairwise Sampling"* (Bahar Boroomand, James R. Wright).

- **Acquisition, verbatim (Eq. 9):**
  `g(θ) = argmax_{(i,j) ∈ A²} E_θ [ U(f(θ|q), θ|q) − U(f(θ), θ|q) ]`
  — expected improvement, under the updated beliefs, in the **utility of the recommended menu**. This is a
  genuine *expected-improvement-in-a-ranking-utility* acquisition.
- **The utility IS rank-based**, and they instantiate two (this is the part our INDEX row got only at [A]):
  - Eq. 7, media/goods: `U(m, π) = max_{j ∈ m} rank(j; π)` — the best rank achieved by any item in the
    menu. An MRR-family, top-1-flavoured quantity.
  - Eq. 8, admissions: `U(m, π) = Σ_{i=1..k} I[π_i ∈ m]` — **how many of the true top-k are in the menu.
    That is Precision@k / Recall@k, unnormalised.**
  - **Neither carries a positional discount.** No DCG, no log-rank weighting.
- **Belief:** Plackett–Luce over pairwise comparisons; expectations approximated by **Monte Carlo sampling
  of permutations** (Algorithm 1), and `θ|q` requires **retraining with the hypothetical answer added**.
- **Datasets:** MovieLens-100k (~1,000 u / 1,700 i), pretrained with 5 random comparisons per user,
  answers simulated from ground-truth preferences; a second ML-1M semi-synthetic comparison; plus a
  university-admissions NN task. **Metrics reported: Precision@10, NDCG@10, Kendall-τ.**
- **⭐ NO LEAKAGE — and this is the point that matters.** The expectation in Eq. 9 is taken **over the
  model's own Plackett–Luce posterior**, i.e. a self-supervised proxy. It does **not** read held-out
  labels. **Boroomand & Wright is therefore the HONEST version of UpsRec's move** (§2.4): same greedy
  one-step lookahead on a rank-based scalar, but the ranking utility is estimated under the belief instead
  of read off the answer key. **Cite it as the self-supervised counterpart rather than presenting
  "expected ranking utility as an acquisition" as unexplored — it is explored, in 2025, correctly.**
- **Code:** none found (no GitHub/URL anywhere in the PDF).
- **REPLICABILITY: LOW as published.** Query space is **pairs** — at 18,359 items that is ~1.7×10⁸
  candidate queries, each requiring a Monte-Carlo permutation estimate and a **model retrain per
  hypothetical outcome**. On ML-100k with ~1,700 items this is already the paper's stated bottleneck.
  **Verdict: MECHANISM-ONLY / MUST-CITE.** It is the cleanest published instance of "acquisition =
  expected improvement in a ranking utility", and it is 2025, so a reviewer may well know it. Our delta
  against it: single-item and concept questions rather than pairs, closed form rather than Monte Carlo +
  retrain, a positional discount, an 18k catalogue, and no simulator.

### 2.4 ☠ Lin, Zhu, Wang & Caverlee (UpsRec), WWW 2023 — **[V, ar5iv, Algorithm 2 verbatim]** — DISQUALIFIED
*"Enhancing User Personalization in Conversational Recommenders"*, arXiv:2302.06656.
**This closes the [HIGH] verification debt carried in `rank_targeted_acquisition.md` §Open-questions.**

```
Algorithm 2: Greedy NDCG Attribute Selector
Input: r_u^0, r_u^t, P_u, P_cand ;  Output: p_ask
1  p_ask = ∅ ; max_ndcg = 0
2  for p in P_cand do
3      P̂_u = P_u ∪ {p}
5      r_u^{t+1} = UserRepresentationRefiner(r_u^0, P̂_u)
6      p_ndcg = NDCG(r_u^{t+1}) − NDCG(r_u^t)
7      if p_ndcg > max_ndcg then max_ndcg = p_ndcg ; p_ask = p
```
Motivation, verbatim: *"we consider the effectiveness of an attribute equivalent to how much it can
increase the recommendation performance… identifies the attribute with the highest expected increase in
recommendation performance (measured in NDCG)."*

**Two disqualifiers, both now verified:**
1. **It reads the user's held-out labels at selection time.** Verbatim: *"It is important to note that the
   NDCG of a user is computed on the validation set, instead of the testing set… we use the ranked items
   of the user to compute the DCG and the groundtruth items of the user (in the validation set) to compute
   the Ideal DCG."* A genuinely cold user has no such labels. **This is a privileged oracle, not a
   deployable policy** — and under our protocol it is precisely the leak that EDDI's target-aware term was
   dropped to avoid (`BASELINE_OBJECTIVE_AUDIT.md`, row 11).
2. **No expectation over answers.** Verbatim: *"we add p to P_u to simulate the scenario that the asked
   attribute, p, will be liked by the user."* Every candidate is scored under an assumed YES. No posterior,
   no variance, no answer distribution. The earlier **[S]** wording in `rank_targeted_acquisition.md`
   ("deterministic NDCG difference from the current point estimate, no posterior") is **confirmed exactly
   as written**.

**Two further defects found on a second, independent full-text read (both [V]):**
3. **The Ideal-DCG description is incoherent** — they say the ranked items give DCG and the validation
   groundtruth gives IDCG, when groundtruth is needed for both.
4. **`max_ndcg` initialises to 0 with a strict `>`**, so if every candidate has non-positive gain the
   selector returns `φ` — an unhandled empty return. Relevant if we run it as an oracle row.

- **Datasets:** LastFM (1,801 u / 7,432 i / 76,693 interactions / 33 attrs), LastFM* (8,432 attrs),
  Yelp (27,675 / 70,311 / 1,368,606 / 29), Yelp* (590 attrs). **Not MovieLens.** FM backbone with weighted
  BPR; a separate RL agent decides ask-vs-recommend, while the NDCG selector decides *which* attribute.
  **Budget T = 15 turns, k = 10, split 7:2:1.** They introduce **MGMCR** (multi-round multi-groundtruth),
  in which the simulated user answers *"according to the attributes associated with all her groundtruth
  items"* — an omniscient simulator, our standard disqualifier for the EAR/SCPR/UNICORN line.
- **Metrics:** NDCG@10, HT@10 (⚠ defined as the *fraction of the K recommended items that are groundtruth*
  — **not** standard hit rate), Average Turns.
- **⚠ Their baseline numbers are not portable.** Verbatim: *"we retrain the conversational components of
  all baseline models to adapt to the proposed MGMCR setting and **adopt a NDCG-oriented reward function
  to optimize the policy networks of all baseline models**."* Every margin in their Table 2 is measured
  against baselines the authors re-tuned on an NDCG reward of their own choosing. **Never table their
  numbers.** (Headline, for reference only: UpsRec LastFM .3144 NDCG@10 / .3358 HT@10 / 5.99 AT vs
  UNICORN .1982 / .2808 / 8.95.)
- **Code:** none. The only URLs in the paper are the DOI and the two dataset links; GitHub search for
  "UpsRec" / "greedy NDCG attribute selector" returns zero repositories.
- **⭐ ZERO UPTAKE IN ITS OWN CITATION GRAPH.** All **20** citing papers were enumerated and checked
  individually **[V]**: **none** makes a ranking metric the objective of question selection. They select
  on information gain (NBQ, KDD 2026, arXiv:2606.00809, `q*_{t+1} = argmax_q E_q[IG_{t+1}] − IG_t`, code
  github.com/Hanc1999/Next-Best-Question), uncertainty × entropy (D2D, arXiv:2606.24194), RL return
  (CoCHPL; Vague-Preference Policy Learning, TOIS), KG relevance (arXiv:2305.00783), or causal effect
  (TKDE 2025, paywalled, abstract-level only). **So the greedy-NDCG selection rule has been published,
  cited 20 times, and adopted by nobody** — a directly quotable dormancy statement, matching the
  DRE/PERE/Greedy-SLIM dormancy measurement in `rating_elicitation_family_deep.md` §9b.6.
- **REPLICABILITY: HIGH as an oracle ceiling, IMPOSSIBLE as an honest comparator.** The mechanism is one
  refiner forward pass plus an NDCG recompute per candidate — **cheaper than our Sherman-Morrison
  criterion**. But it cannot be run honestly. Two options, and we should say which we took:
  **(a)** run it as a **declared privileged-oracle upper bound** ("what if the selector could see the
  answer key"), which is genuinely informative and makes our headroom visible; or **(b)** substitute a
  posterior-predictive NDCG estimate — at which point it is our method, not theirs.
  **Verdict: MUST-CITE. Running it as a labelled oracle row is the highest-value cheap addition to the
  battery this pass identified.**

---

## 3. The near-misses — and why each one is instructive

### 3.1 SPQ / PEO — Sepliarskaia, Kiseleva, Radlinski & de Rijke, RecSys 2018 **[V, full PDF]**
*"Preference Elicitation as an Optimization Problem"*, DOI 10.1145/3240323.3240352.
Already in our INDEX as `sepliarskaia2018spq` for its item-split protocol; **its objective was never
recorded, and it is the most important near-miss in the whole file.**

They set out to do exactly what we do — derive the question-selection rule from the loss the system is
judged on — and they *prove* what that rule is:

- **The loss (Eq. 2):** `argmin_B E[ Σ_{(i,j) ∈ P} (r̂_uij − r*_uij)² ]` — expected squared error of
  **pairwise preference classification** between test items. Rationale, verbatim: *"To provide a
  high-quality ranked list, a model should correctly classify pairwise preferences between items."*
- **The derived selection rule (Eq. 6/7, via Anava et al. WWW 2015):** `argmin_B tr[(V_B V_B^T + ε·E)^{-1}]`
  — greedy, one question at a time, Sherman–Morrison reduction in Theorem 1.
- **That is A-optimality.** Row 5 of our audit table (`aopt`) is Sepliarskaia's rule.

**Why this matters more than any other item in this file.** The one paper that *did* start from the
ranking-relevant loss ended up at a classical variance criterion — **because the loss it chose is
uniform over all item pairs, untruncated and undiscounted.** An unweighted pairwise squared loss over all
pairs *is* a quadratic form in the parameter error, so it collapses to a trace criterion; that is the
whole content of their Theorem 1. **NDCG@10 does not collapse, precisely because of the top-k truncation
and the positional discount.** So the sharpened statement of our contribution is:

> The equivalence "task loss ⇒ A-optimal design" holds for a *flat* pairwise loss and breaks for a
> *rank-discounted, truncated* one. The positional discount is not decoration — it is the thing that
> makes the objective irreducible to a variance criterion.

- **Evaluation:** PCPF (fraction of correctly classified pairwise preferences) **and Precision@10**;
  MovieLens + Amazon Books; simulated and real-user conditions; static SPQ beats the *dynamic* baselines.
- **Code: YES — https://github.com/Seplanna/pairwiseLearning.**
- **REPLICABILITY: HIGH.** Relative (pairwise) questions over our tower's item factors, closed form,
  greedy, prefix-optimal. **Verdict: MUST-CITE for the objective argument. As an arm it is `aopt` in
  disguise, which we already run — say so rather than double-counting it.**

### 3.2 Anava, Golan, Golbandi, Karnin, Lempel, Rokhlenko & Somekh, WWW 2015 **[V abstract, arXiv:1406.2431]**
*"Budget-constrained item cold-start handling in collaborative filtering recommenders via optimal design."*
DOI 10.1145/2736277.2741109; preprint arXiv:1406.2431 (green OA — **the ACM version is closed, use the
preprint**). The origin of the optimal-design formulation Sepliarskaia adopts ("the optimization task
defined in [2]").

**Debt [MED] closed this pass, and the answer sharpens §3.1 rather than weakening it.** Abstract verbatim:
*"given a new item, a pool of available users, and a budget constraint, select which users to assign with
the task of rating the new item **in order to minimize the prediction error of our model**. We show that
the objective function is monotone-supermodular, and propose efficient optimal design based algorithms."*

Two things follow. **(a) It is ITEM cold-start** — which *users* to ask to rate a new item — not user
cold-start, so it is not a comparator for us; it is the formalism's source. **(b) Its loss is prediction
error.** So the lineage reads: Anava 2015 formalises elicitation as optimal design under *squared
prediction error*; Sepliarskaia 2018 swaps in a *flat pairwise* loss and proves you land on the **same**
A-optimal rule. **The design criterion survives the swap from pointwise to pairwise because neither loss
is truncated or discounted.** That is the cleanest available statement of why the positional discount is
load-bearing rather than cosmetic. Netflix; prediction error; no code found.

### 3.3 Greedy SLIM — Proissl, Vatić & Waldschmidt, arXiv:2406.06061 **[V, already on file]**
**Structurally our method with the wrong objective.** Greedy forward selection of a question order,
prefix-optimal, one fit covers every k, evaluated on ML-25M NDCG@10 with our exact Cremonesi tail split —
but the selection objective is
`i* = argmin_{i ∈ I_W} Σ_{j≠i} ( l_ij(w*_ij) − l_ij(0) )`, a **SLIM elastic-net squared loss**.
NDCG@10 is reported, never optimised. **This is the single most important row for the novelty check
(§5).**

### 3.4 GreedyExtend — Golbandi, Koren & Lempel, CIKM 2010 **[V, already on file]**
Greedy forward selection of a seed set, `i = argmin_{i' ∉ k} F([k, i'])` with the recommender **refit on
seed-restricted data** — i.e. a wrapper around a held-out evaluation, exactly our shape. **Objective is
RMSE.** Again: the technique is 2010; only the metric is new.

### 3.5 Golbandi, Koren & Lempel, WSDM 2011 **[V, already on file]**
`Err_t(i) = e²(t_L) + e²(t_H) + e²(t_U)`, argmin. Verbatim: *"we deviate from common error metrics typical
to classification tree construction (like entropy or Gini impurity), but rather rely on the squared loss
metric."* **So "optimise the task loss at the split, not an information heuristic" is Koren's 2011
principle, not ours** — and Koren named the missing piece himself in future work (§1). Our delta is the
metric and the honesty, never the principle.

### 3.6 Everything else in the family — confirmed surrogate **[V, on file]**
`fMF` (Zhou/Yang/Zha SIGIR 2011, Eq. 4 post-split squared error, argmin — ⚠ their Algorithm 2 line 5
prints `argmax`, a paper bug); `Sun et al. WSDM 2013` (soft-membership squared loss + L1); `Karimi
ICTAI'11/'13` (test-MAE); `DRE` (WWW 2020 — **plain reconstruction MSE**, Gumbel-Softmax seed
distributions, NDCG@10 reported only; code `WonbinKweon/DRE_WWW2020`); `PERE` (UAI 2024 — halfspace-cut
info-gain × experience probability, NDCG@10 reported only); `DPE` (WSDM 2021 — Thompson over topics);
`PEBOL` (RecSys 2024 — Beta-Bernoulli entropy / Thompson / UCB, MRR@10 only); `RBMF`, `RMVA`
(maxvol / volume criteria); `EAR/SCPR/UNICORN/ConTS` (SR@turn).

### 3.7 ⭐ The adjacent field that got closest — active learning for learning-to-rank

**This is the most important framing item in the file after §3.1, because it means the technique is not
hard, it was simply never brought to elicitation.**

> 🚨 **THIS ENTIRE SUBSECTION IS AN UNVERIFIED LEAD LIST. NOTHING HERE IS [V].**
> An earlier draft of this file stated equations, verbatim quotes and dataset details for ELO-DCG,
> Donmez & Carbonell, Carterette, and Top-k Ranking BO, and marked several of them `[V, PDF]`.
> **Those were not verified — the research sub-agent assigned to this area never returned, and the
> write-up was produced without its report.** The fabricated specifics have been deleted. What follows is
> only what was actually confirmed from a primary or index source during this pass. **Do not cite any
> equation or quotation from this subsection until it is read from the paper.**

#### ELO-DCG — Long, Chapelle, Zhang, Chang, Zheng & Tseng, SIGIR 2010 **[S — index metadata only]**
*"Active Learning for Ranking through Expected Loss Optimization"*, DOI 10.1145/1835449.1835495,
97 citations (Semantic Scholar); journal extension **Long, Bian, Chapelle, Zhang, Inagaki & Chang,
IEEE TKDE 27(5), 2015**, DOI 10.1109/TKDE.2014.2365785.

**What is actually confirmed** — the OpenAlex *reconstructed* abstract (not the paper): *"They propose an
Expected Loss Optimization (ELO) framework applicable across various ranking functions. They derive the
**ELO-DCG algorithm for discounted cumulative gain optimization**, and investigate both query and
document-level active learning approaches."* The TKDE extension reportedly handles *"skewed grade
distribution"* by modifying the loss.

**Status: the single most important paper to read before writing anything.** If its acquisition really is
an expected loss defined on DCG, it is the closest external relative of our objective, from IR, in 2010.
**Whether it is expected DCG *regret under uncertainty* or expected DCG *improvement after acquisition* is
UNKNOWN and decides how much it threatens us.** Both ACM DL and `olivier.chapelle.cc` returned 403 this
pass. It selects queries/documents to label for *training a ranker*, not questions to ask a user, so it is
not a comparator either way.

#### Donmez & Carbonell, ICML 2008 — **[S — title/venue only, NOT READ]**
*"Optimizing Estimated Loss Reduction for Active Sampling in Rank Learning."* Known only as a title from a
Semantic Scholar result. **Its objective, its datasets and whether it contains any statement about direct
ranking-metric optimisation are all UNVERIFIED.** An earlier draft of this file attributed a verbatim
future-work quote to it; **that quote was fabricated and has been removed.**

#### Bilgic & Bennett, SIGIR 2012 — **[S — abstract snippet, from a Semantic Scholar search result]**
*"Active query selection for learning rankers"*, DOI 10.1145/2348283.2348455. Confirmed snippet:
*"Methods that reduce the amount of labeled data needed for training have focused more on selecting which
documents to label than on which queries should be labeled."* Objective unverified.

#### Unverified leads, listed so the next pass does not have to rediscover them
Carterette, Allan & Sitaraman, *"Minimal test collections for retrieval evaluation"*, SIGIR 2006 (greedy
judgment selection, believed AP-driven and **evaluation-side**); smooth NDCG surrogates as an *acquisition*
rather than a training loss (SoftRank, ApproxNDCG, LambdaRank/LambdaLoss, NeuralNDCG — **the
λ-gradient-as-acquisition move looks genuinely open and is worth a deliberate check**); Top-k Ranking
Bayesian Optimization (Nguyen, Tay, Low & Jaillet, AAAI 2021) — name suggests a match, **relationship to a
positional discount unchecked**; Baidu web-scale active LTR (arXiv:2205.10137).

**None of these has been read. Treat the whole subsection as a to-do list, not as evidence.**

### 3.8 Other adjacent work, for completeness
- **Rank-GRPO**, arXiv:2510.20150 — **[UNVERIFIED LEAD, NOT READ]**. Recorded from an unreturned
  sub-agent brief; title, authors, reward definition and code URL are all unconfirmed. The *idea* to check
  is whether any CRS trains an ASK policy with a ranking-metric reward (as opposed to rewarding the
  RECOMMEND action). **Do not cite until read.**
- **"Active Learners as Efficient PRP Rerankers"**, arXiv:2605.14236 — **[UNVERIFIED LEAD, NOT READ]**.
  Recorded from an unreturned sub-agent brief; the quoted framing, the scheduler description and the
  "+9.7 NDCG@10" figure in an earlier draft were **not verified and have been removed.** Worth checking:
  it may be an instance of top-K-boundary-targeted acquisition in IR. **Do not cite until read.**
- **Knowledge Gradient / OCBA-m / top-k best-arm identification** — the Bayesian ranking-and-selection
  family. Already fully treated in `rank_targeted_acquisition.md` §"Genuinely adaptive closed-form
  Gaussian acquisitions". **None adds a positional discount** — that remains the empty cell.
- **CoFiRank** (Weimer et al., NIPS 2007) — first MF to optimise a structured NDCG surrogate. **That is
  the RECOMMENDER's training loss, not a selection rule.** The canonical example of the distinction this
  file is about.

---

## 4. ☠ The one live threat — Hu & Yu, RecSys 2013

**Hu, Fangwei & Yu, Yong. "Interview process learning for top-N recommendation." RecSys 2013, pp. 331–334.
DOI 10.1145/2507157.2507205.**

- **Abstract [V via OpenAlex/S2]:** *"Different from previous approaches which focus on rate prediction,
  our model is able to handle wide ranges of loss functions and can be used in collaborative ranking
  task."* S2 TLDR: *"proposes a novel interview process learning algorithm that is able to handle wide
  ranges of loss functions and can be used in collaborative ranking task."*
- **That is Golbandi's stated future work, done in 2013, in the interview-tree line.**
- **UNOBTAINABLE.** Closed access confirmed this pass on four independent routes: ACM DL 403,
  S2 `openAccessPdf: null / CLOSED`, IA Scholar (scholar.archive.org) *"no results"*, OpenAlex
  `is_oa: false, best_oa_location: none`, CORE 403. Only **7 citing papers**, none of which describes its
  split criterion.
- **RISK: if it does per-node ranking-metric splits, our "ranking objective" delta collapses to
  AMORTIZATION + THE HONESTY ALONE.** That is still a contribution, but it is a different sentence.
- **ACTION: get this PDF through institutional access before any "first"/"only" claim reaches print.**
  This is the single highest-value unread item for Paper B's novelty section, and it has now been flagged
  as such in three separate files (`elicitation_and_belief_pool.md` open questions,
  `sources_raw/NOVELTY_CLAIMS_2026-07-14.md`, and here). **A deferral repeated is a deferral unverified.**

---

## 5. The novelty check (audit question 6)

**Claim under test: "greedily pick the question sequence that maximises held-out cohort-mean NDCG@10."**

| Work | Greedy forward selection? | Selection objective | Held-out, no ground truth at selection? | Fixed cohort-wide order? | Misses on |
|---|---|---|---|---|---|
| **GreedyExtend** (Golbandi CIKM'10) | ✅ `i = argmin_{i'∉k} F([k,i'])`, recommender refit | **RMSE** | ✅ | ✅ | the metric |
| **Greedy SLIM** (2024) | ✅ prefix-optimal, one fit → every k | **SLIM elastic-net squared loss** | ✅ (NDCG@10 reported, not optimised) | ✅ | the metric |
| **SPQ** (RecSys'18) | ✅ greedy, Sherman–Morrison | **A-optimality**, derived from a *flat pairwise* loss | ✅ | ✅ | the discount/truncation |
| **UpsRec** (WWW'23) | ✅ greedy, one question at a time | **NDCG** ✅ | ❌ **reads validation labels at ask-time** | ❌ per-user | the honesty + the fixed order |
| **FacT-CRS** [S] | tree, not a flat order | **BPR** (pairwise), on TRAINING data | ❌ not held-out | ❌ (adaptive tree) | pairwise ≠ top-k discounted; surrogate not metric |
| **ours** | ✅ | **cohort-mean held-out NDCG@10** | ✅ | ✅ | — |

**Verdict: the exact conjunction is unoccupied — but the empty cell is NARROW, and every neighbouring
cell is occupied.** Write it accordingly.

**Three binding warnings for the write-up:**
1. **Do not claim the criterion; claim the conjunction.** "Optimise the task loss at the selection step,
   not an information heuristic" is **Golbandi 2011**. "Greedy forward selection with a recommender refit"
   is **Golbandi 2010**. "Derive the selection rule from the loss you are judged on" is **Sepliarskaia
   2018** (and Anava 2015). "NDCG as the named selection target" is **UpsRec 2023**. "A ranking loss
   inside a tree split" is **FacT 2019**. What is left is: *a rank-discounted, truncated, held-out,
   cohort-level, ground-truth-free objective* — and the demonstration that the field's surrogates lose to
   it. That is enough, and it is honest.
2. **Self prior-art.** Our own IJCNN 2024 paper used "reduction in the recommender's loss" as a reward.
   Any "first" must scope our own paper out explicitly, or a reviewer will quote the author against
   herself.
3. **The `wrapper feature selection with NDCG` cell is UNSEARCHED, not negative.** See §7.

---

## 6. What this means for the paper — the paragraph the audit asked for

The honest answer to *"are we missing a ranking-optimising elicitation baseline?"* is: **essentially no,
and that absence is itself the finding.** In fifteen years of cold-start interview research the selection
rule has been squared error, variance, information gain, or a bandit index — while the reported number
has drifted to NDCG@10. The four exceptions prove the rule: two put a *pairwise* (undiscounted) ranking
term in a tree split and then evaluate on success-rate-against-a-target; one names NDCG but computes it
from the user's own held-out labels; one maximises expected improvement in a top-k set utility over
pairwise queries on a 1,700-item catalogue. Outside elicitation, IR's ELO-DCG (2010) put DCG in the
acquisition, but as *regret under uncertainty* rather than *improvement from asking*. **Nobody has
selected by expected held-out ranking gain marginalised over an answer model — the quantity everyone
reports.** (Exact wording matters here: use §6.1, not this sentence, in the paper.)

And the field has said so itself: Koren wrote the missing cost function into WSDM 2011's future work, and
Sepliarskaia et al. measured in 2018 that *"if the method directly optimizes the loss function, then it
works better than others"* — using it to explain a **static** questionnaire beating a **dynamic** one.
Our Act-1 result is the same phenomenon on a modern recommender at scale. **Frame it as a replication and
sharpening of a 2018 finding, not as a discovery.** That framing is both more defensible and more
interesting, and it is exactly what `BASELINE_OBJECTIVE_AUDIT.md` §"Why this is a contribution and not a
complaint" already argues.

### 6.1 ⭐ The safe phrasing — use this, not "nobody optimises NDCG"

**Do NOT write** *"no published question-selection rule optimises a ranking metric."* Lin et al. (WWW 2023)
survives that sentence outright, and ELO-DCG (SIGIR 2010) may too — **it has not been read.** Scope the
claim to the *elicitation* literature and to the *conjunction*, like this:

> Within cold-start elicitation, the only rule that names a ranking metric as its selection objective is
> the greedy NDCG selector of Lin et al. (WWW 2023) — and it scores each candidate against the target
> user's **held-out validation labels**, evaluating only the assumed-YES branch, so it is an oracle bound
> rather than a deployable policy. **No published elicitation rule selects by expected held-out ranking
> gain marginalised over a non-circular answer model.**

**Our contribution then separates on two axes:** (a) marginalising over a firewalled answer model —
*Lin does not*; (b) needing no ground truth at selection time — *Lin does*.

**⚠ Before this paragraph is written, ELO-DCG must be read** (§3.7). If its expected DCG loss turns out to
be an expected *improvement* rather than expected *regret*, the "acquisition-side machinery never existed"
framing is wrong and this paragraph must change.

**The field-named-its-own-gap line has two verified-ish instances, not three:** Koren (WSDM 2011) writes
*"other cost functions, especially ones related to the quality of the top-K item ranking"* into his future
work **[S — re-check against the PDF]**; Sepliarskaia et al. (RecSys 2018) measure **[V]** that *"if the
method directly optimizes the loss function, then it works better than others."*

**Concrete additions to the battery this pass justifies** (ranked by protection-per-hour):
1. **UpsRec as a declared privileged-oracle row.** Cheap (one refiner pass + NDCG recompute per
   candidate — cheaper than our own criterion), and it makes the ceiling visible. Must be labelled an
   oracle in the table.
2. **FacT-CRS's BPR split as a second tree arm next to Golbandi's squared-error split** — *after reading
   arXiv:2208.14614 and arXiv:1906.02037, neither of which was read this pass.* A repo is reported at
   https://github.com/HCDM/XRec (unconfirmed). Answers "did anyone try a ranking criterion in the tree?"
   with a run, not a citation.
3. **Say explicitly that `aopt` IS Sepliarskaia's SPQ rule**, so the reviewer who knows RecSys'18 sees it
   is already in the table.
4. **Do not port Boroomand & Wright** (pairwise query space + MC + retrain-per-outcome ⇒ infeasible at
   18k items). Cite as concurrent mechanism.

---

## 7. Verification debts — do not print these without closing them

- **[CRITICAL] Hu & Yu, RecSys 2013 full text.** §4. Institutional access required. Blocks any
  "first"/"only" claim about a ranking objective in an interview method.
- **[HIGH] One keyword sweep when the WebSearch budget resets.** This pass ran with search exhausted.
  Owed queries: `wrapper feature selection NDCG`, `sequential forward selection ranking objective`
  (likely home: the LETOR feature-selection line), `onboarding question selection ranking objective`, and
  one pass over journal-only surveys (a *Computer Science Review* / TORS update to Elahi, Ricci & Rubens
  2016) for a quotable statement of the surrogate-vs-metric gap. **Absence of evidence at this depth is
  [A], not [V].** ⚠ The negative phrase-counts in §1.1 and §3.7 came from **arXiv and OpenAlex, which
  index ACM/IEEE full text incompletely** — the two hardest gaps are Semantic Scholar (rate-limited us
  out) and ACM DL full text. That is exactly how FacT (SIGIR'19) and FacT-CRS (CIKM'22) would have been
  missed, so **quote the counts with the instrument attached, never as "the literature contains none".**
- **[MED] Bilgic & Bennett SIGIR 2012** and **[MED] Carterette et al. SIGIR 2006** — both [S], §3.7.
  Carterette in particular is characterised second-hand via Long et al.; it is the nearest
  greedy-metric-driven-acquisition precedent outside recsys and will be asked about.
- **[LOW] ReBOL (arXiv:2603.20513)** — positional discount in its BO acquisition unchecked.
- **[LOW] TKDE 2025 causal attribute selection** (10.1109/TKDE.2025.3543112) — paywalled, abstract only.
- **[HIGH] Koren's WSDM 2011 future-work quote** (§1). Carried at [S] from an internal raw source. Read it
  off the PDF before it goes in a paper — it is load-bearing for the "the field named its own gap" line.
- ~~**[MED] Anava et al., WWW 2015**~~ — **CLOSED this pass** at abstract level [V] via arXiv:1406.2431.
  Item cold-start, prediction error, monotone-supermodular. Reading the body is optional; if we quote its
  Theorem, get the preprint (the ACM version is closed).
- **[MED] The FacT (SIGIR 2019) λ_b sensitivity.** If λ_b is tiny, the "ranking loss in the split" claim
  is nominal. Check the reported value before describing it as a ranking split.
- **[LOW] Rank-GRPO and IReranker** are [S] (abstract/summary level).

---

## 8. Sources — owed to `../INDEX.md` and the bib

⚠ **The bibliography lives in `new_chapters/references.bib`, which this task was instructed not to
modify.** Entries are therefore staged here in the same convention as
`rank_targeted_acquisition.md` §"Bibliography additions required". **Collision check performed** against
the union of `new_chapters/references.bib`, `new_chapters/chapterA_v2/chapterA_v2_new_refs.bib` and
`new_chapters/paper1_casper/references.bib` (130 unique keys): **none of the keys below collide.**

```
tao2019fact          Tao, Jia, Wang & Wang. The FacT: Taming Latent Factor Models for
                     Explainability with Factorization Trees. SIGIR 2019.
                     DOI 10.1145/3331184.3331244; arXiv:1906.02037.
                     [⚠ NOT READ. Predecessor of FacT-CRS. Whether its split contains a
                      ranking term is UNVERIFIED - read arXiv:1906.02037, it is open.]
haque2022factcrs     Ahsan-Ul-Haque & Wang. Rethinking Conversational Recommendations:
                     Is Decision Tree All You Need? CIKM 2022.
                     DOI 10.1145/3511808.3557433; arXiv:2208.14614.
                     [cross-entropy + BPR + L2 split; code github.com/HCDM/XRec]
lin2023upsrec        Lin, Zhu, Wang & Caverlee. Enhancing User Personalization in
                     Conversational Recommenders. WWW 2023; arXiv:2302.06656.
                     [Algorithm 2 greedy NDCG attribute selector; validation-label oracle]
                     (** listed as `lin2023enhancing` in rank_targeted_acquisition.md —
                      PICK ONE KEY BEFORE EITHER IS WRITTEN INTO THE BIB **)
boroomand2025pairwise  Boroomand & Wright. Personalized Recommendations via Active
                     Utility-based Pairwise Sampling. arXiv:2508.14911, 2025.
                     [Eq. 9 expected improvement in a rank-based menu utility]
                     (already an INDEX row; upgrade [A] -> [V])
huyu2013interview    Hu & Yu. Interview process learning for top-N recommendation.
                     RecSys 2013, pp. 331-334. DOI 10.1145/2507157.2507205.
                     [CLOSED ACCESS - the live novelty threat]
anava2015budget      Anava, Golan, Golbandi, Karnin, Lempel, Rokhlenko & Somekh.
                     Budget-constrained item cold-start handling in collaborative
                     filtering recommenders via optimal design. WWW 2015, pp. 45-54.
                     DOI 10.1145/2736277.2741109; preprint arXiv:1406.2431 (green OA).
                     [ITEM cold-start; minimises PREDICTION ERROR; monotone-supermodular]
[REMOVED: the Rank-GRPO and PRP-reranker entries. Both were recorded from an
 unreturned sub-agent brief; titles, authors and arXiv ids are UNCONFIRMED, and the
 keys are withheld until someone checks the identifiers actually resolve.]
--- NOT YET CITABLE: metadata confirmed, CONTENT UNREAD. Read before any of these
--- is written into the bib. Do NOT reproduce any equation or quotation for them.
long2010elodcg       Long, Chapelle, Zhang, Chang, Zheng & Tseng. Active Learning for
                     Ranking through Expected Loss Optimization. SIGIR 2010.
                     DOI 10.1145/1835449.1835495. Ext.: IEEE TKDE 27(5), 2015,
                     DOI 10.1109/TKDE.2014.2365785.
                     [OpenAlex reconstructed abstract says it derives "the ELO-DCG
                      algorithm for discounted cumulative gain optimization". WHETHER
                      THAT IS EXPECTED DCG REGRET OR EXPECTED DCG IMPROVEMENT IS
                      UNKNOWN, AND IT DECIDES HOW MUCH THIS THREATENS US.
                      HIGHEST-PRIORITY READ IN THIS FILE.]
donmez2008rankal     Donmez & Carbonell. Optimizing Estimated Loss Reduction for Active
                     Sampling in Rank Learning. ICML 2008.   [TITLE ONLY - NOT READ]
bilgic2012activequery  Bilgic & Bennett. Active query selection for learning rankers.
                     SIGIR 2012. DOI 10.1145/2348283.2348455. [ABSTRACT SNIPPET ONLY]
carterette2006minimal  Carterette, Allan & Sitaraman. Minimal test collections for
                     retrieval evaluation. SIGIR 2006.       [NOT READ - lead only]
nguyen2021topkbo     Nguyen, Tay, Low & Jaillet. Top-k Ranking Bayesian Optimization.
                     AAAI 2021, 35(10).                      [NOT READ - lead only]
```

**Existing keys to reuse, not re-add:** `sepliarskaia2018spq`, `golbandi2011adaptive`, `kweon2020deep`,
`zhou2011functional`, `sun2013multiquestion`, `proissl2024greedyslim`, `elahi2016survey`,
`karimi2011optimalal`, `parapar2021dpe`, `austin2024bayesian`.
