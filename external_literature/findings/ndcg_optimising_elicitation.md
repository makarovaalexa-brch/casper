# Findings: Does ANY elicitation method optimise a RANKING metric in the QUESTION-SELECTION rule?

*Deep sweep 2026-08-10. Commissioned by `docs/results/BASELINE_OBJECTIVE_AUDIT.md`, which found
**1 of 21** distinct selection rules in our battery optimises held-out ranking quality directly (ours),
and asked whether the field contains methods we are missing that would be the fair comparison.*

**Sibling files — read together, do not duplicate:** `rank_targeted_acquisition.md` (the criterion's
OED/Bayesian lineage: A-GOODE, transductive experimental design, Knowledge Gradient, EVOI),
`rating_elicitation_family_deep.md` (the elicitation family, DRE/PERE/Greedy SLIM),
`adaptive_elicitation_baselines.md` (the runnable battery), `active_learning_elicitation.md` (AL/BED side).
This file is the **objective-of-the-selection-step** cut across all of them, plus the new sources.

Verification levels: **[V]** verified from primary full text I read · **[S]** secondary/abstract ·
**[A]** inference / absence-at-this-search-depth.

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

Exactly **four** works put a ranking quantity inside the selection rule. Two are trees with a *pairwise*
ranking term, one is a greedy NDCG selector that **reads the user's held-out labels at ask-time**, and one
is a 2025 pairwise-comparison sampler whose utility is a **top-k set** function. None of them is the fair
comparator we were worried about, and the reasons why are individually interesting.

**One work outside elicitation — ELO-DCG (Long et al., SIGIR 2010) in IR active learning — does put DCG in
the acquisition, but as expected *regret under uncertainty*, never as expected *improvement from asking*
(§3.7). Read §6.1 before writing any sentence of the form "nobody optimises NDCG".**

**The field's own verdict on this gap is on record THREE times, spanning 2008–2018:**

- **Donmez & Carbonell, ICML 2008**, conclusion, verbatim **[V]**: *"Possible extensions of this work
  include a study of the risk minimization in terms of **direct optimization of ranking performance
  metrics, such as MAP, NDCG, precision@k**, etc."*
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

### 2.1 ⭐ FacT-CRS — Ahsan-Ul-Haque & Hongning Wang, CIKM 2022 **[V, ar5iv]**
*"Rethinking Conversational Recommendations: Is Decision Tree All You Need?"*, DOI 10.1145/3511808.3557433,
arXiv:2208.14614.

- **Selection objective:** node question chosen by `f_z = argmin_{f_l ∈ F} L(f_l | R_z)` where `L` is
  **cross-entropy + BPR pairwise ranking loss + L2**. Explicitly contrasted in the paper with information
  gain. Node ranking `w_t^I(i) = s_t^T v_i`.
- **This is a decision-tree interview whose split criterion contains a ranking loss** — the direct answer
  to audit question 3, and it is not in our battery.
- **Datasets:** LastFM (1,801 u / 7,432 i / 33 attrs), BookRec, MovieLens (3,000 u / 5,974 i / 35 attrs).
- **Metrics:** SR@10 and Average Turns — the target-item CRS protocol. **No held-out ranking metric.**
- **Code: YES — https://github.com/HCDM/XRec** (live).
- **REPLICABILITY: MEDIUM. The mechanism ports; the evaluation does not.** The tree is grown offline over
  attributes, which is our concept channel, and BPR over our tower's item factors is cheap. But their
  reported numbers are Success-Rate-against-a-known-target and must never sit beside our NDCG@10
  (same rule as EAR/SCPR/UNICORN in `adaptive_elicitation_baselines.md` §6). **Verdict: MUST-CITE; a
  legitimate and affordable new arm if we want a ranking-loss tree next to Golbandi's squared-loss tree.**

### 2.2 ⭐ The FacT — Tao, Jia, Wang & Wang, SIGIR 2019 **[V, ar5iv]**
*"The FacT: Taming Latent Factor Models for Explainability with Factorization Trees"*,
DOI 10.1145/3331184.3331244, arXiv:1906.02037.

- **Split objective (Eq. 6):** `argmin` over feature `f_l` and threshold of
  `Σ_{child ∈ {L,R,E}} [ L(u_child, V, O_child) − λ_b Σ_i B(u_child, V, D_i^o) ]`
  with the ranking term **BPR verbatim**: `B(u_i,V,D_i^o) = Σ_{(j,l) ∈ D_i^o} log σ(u_i^T v_j − u_i^T v_l)`.
- So the criterion is **squared error MINUS a weighted pairwise ranking loss** — a *hybrid*, not a pure
  ranking split, but it is the first tree split in this family with a ranking term at all.
- It **is** an interview tree: *"each node of the user tree in FacT corresponds to an interview question:
  'How do you like this [feature]?'"* — attribute questions, i.e. our concept channel, ternary L/R/E
  branching (E = unknown), the same shape as Golbandi.
- **Datasets:** Amazon (6,285 u / 12,626 i), Yelp (10,719 u / 10,410 i). **Metrics: NDCG@10/20/50/100**
  plus a user study. NDCG@10: Amazon 0.1482 vs MTER 0.1351; Yelp 0.1499 vs 0.1380.
- **Code:** none in the paper; the group's repo is https://github.com/HCDM/XRec.
- **REPLICABILITY: MEDIUM.** Explainability-motivated rather than cold-start-budget-motivated, and the
  split is a hybrid, but this is the closest published relative of "replace Golbandi's squared error with
  a ranking criterion" and it **reports NDCG on the same axis we do**. **Verdict: MUST-CITE.**

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

#### ELO-DCG — Long, Chapelle, Zhang, Chang, Zheng & Tseng, SIGIR 2010 **[V, Chapelle's PDF]**
*"Active Learning for Ranking through Expected Loss Optimization"*, DOI 10.1145/1835449.1835495 (97 cites);
journal extension **Long, Bian, Chapelle, Zhang, Inagaki & Chang, IEEE TKDE 27(5):1180–1191, 2015**,
DOI 10.1109/TKDE.2014.2365785.

**The loss IS DCG, verbatim (Eq. 6):**
`ℓ(π, Y) = max_{π'} DCG(π', Y) − DCG(π, Y)`, with `DCG(π,Y) = Σ_i (2^{y_i} − 1)/log₂(1 + π(i))`.
**Acquisition (Eq. 7, query level; Eq. 9 document level):**
`EL(q) = ∫_Y max_π DCG(π,Y) P(Y|X_q,D) dY − max_π ∫_Y DCG(π,Y) P(Y|X_q,D) dY`,
select the queries with the highest `EL(q)`. Posterior = a **bootstrap ensemble of 8 GBDT rankers**.
Their own framing: *"ELO-DCG optimizes the expected DCG loss that is directly related to the objective
function DCG-10 used to evaluate the ranking quality; on the other hand, VR reduces the score variance
that is not directly related to the objective function."*

**⚠ CRUCIAL CAVEAT — do not overstate this, it is not what it first looks like.** `EL(q)` is
`E[max DCG] − max E[DCG]`: a **Jensen gap**. It is the **expected DCG regret of the current best ranking
under posterior uncertainty** — a *position-discounted generalisation of uncertainty sampling*. It is
**not** an estimate of how much held-out DCG improves *after* the label is acquired and the model
updated; **no belief update or retraining is simulated.** The authors confirm this structurally: ELO
reduces exactly to predictive-variance sampling under squared loss and to plain uncertainty sampling
under 0/1 loss (§3.2). **So the positional discount enters as a WEIGHT ON WHICH UNCERTAINTIES MATTER, not
as a metric-improvement objective.**

Datasets: **proprietary commercial web-search data** (~10k queries, ~500k docs, 5-grade labels, ~500
features) — **not reproducible**. Metric DCG-10 (not NDCG). No code. TKDE version modifies the loss for
skewed grade distributions; the DCG objective is unchanged.

#### Donmez & Carbonell, ICML 2008 **[V, PDF]** — surrogate, and they named our gap
*"Optimizing Estimated Loss Reduction for Active Sampling in Rank Learning"*, ICML '08 pp. 248–255,
DOI 10.1145/1390156.1390188. **The loss is NOT DCG** — it is a **model-change magnitude** on a pairwise
hinge/exponential surrogate: `D(x,w) = Σ_j [1 − z_j ⟨w, x_j − x⟩]_+` (Eq. 6), `g(x,y) = ‖Δw‖` (Eq. 8),
`x* = argmax_{x ∈ U} Σ_y P̂(y|x) g(x,y)` (Eq. 9). Rationale verbatim: *"Adding a new data point to the
labeled set can only change the error on the test set if it changes the current learner."*
LETOR TREC 2003/2004; MAP and NDCG@10 **reported only**; no code.

**⭐ Their conclusion names our gap explicitly, verbatim:** *"Possible extensions of this work include a
study of the risk minimization in terms of **direct optimization of ranking performance metrics, such as
MAP, NDCG, precision@k**, etc."* — and they note up front that ranking metrics *"are harder to directly
optimize than the classical loss functions."* **This is a third independent instance of the field naming
the missing cost function (with Koren 2011 and Sepliarskaia 2018), and the earliest.**
Follow-up: Donmez & Carbonell, ECIR 2009, *"Active Sampling for Rank Learning via Optimizing the Area
Under the ROC Curve"* — optimises **AUC**: rank-based, **no positional discount**. **[S]**

#### The honest summary of this subsection
The acquisition-side machinery for a rank-discounted objective has existed in IR since 2010 — but even
there it is *uncertainty weighted by position*, not *expected metric improvement*, and its authors were
explicit in 2008 that direct ranking-metric optimisation was open. **Nobody, in IR or recsys, selects by
expected held-out ranking gain marginalised over an answer model.**

**Owed, not read this pass [S]:** Bilgic & Bennett, *"Active query selection for learning rankers"*,
SIGIR 2012, DOI 10.1145/2348283.2348455 (verbatim: *"Methods that reduce the amount of labeled data
needed for training have focused more on selecting which documents to label than on which queries should
be labeled."*); Carterette, Allan & Sitaraman, *"Minimal test collections for retrieval evaluation"*,
SIGIR 2006 — greedy document selection to maximally distinguish two systems **in terms of average
precision**. The latter is genuinely metric-driven greedy acquisition, but it is **evaluation-side**
(which judgments to collect to rank *systems*) and uses AP, not a discounted metric. **Worth one sentence
in related work as the nearest non-recommender precedent — characterisation is second-hand via Long et
al.; verify before citing specifics.**

#### Smooth NDCG surrogates as an ACQUISITION function — **none found [A, strong]**
SoftRank (Taylor et al., WSDM 2008), ApproxNDCG (Qin et al. 2010), LambdaRank/LambdaLoss (Burges; Wang et
al., CIKM 2018), NeuralNDCG (arXiv:2102.07831), Chapelle & Wu's smoothed-IR-metric descent — **all are
training losses over a fixed labelled set.** No work uses a LambdaRank-style λ-gradient to decide *which
data to acquire* or *which question to ask*. **This is a genuinely open move and a natural one**, since
`λ_ij` is already a per-pair, position-discounted "how much would fixing this help NDCG" signal. Flag as
an idea, not a gap we have filled.

#### BO / bandits with a positional discount — **still none [V/S]**
**Top-k Ranking Bayesian Optimization** (Nguyen, Tay, Low & Jaillet, AAAI 2021, 35(10)) — despite the
name, "top-k ranking" is the **observation model** (the user returns a top-k ranking as feedback); the
goal is finding the global maximiser and the acquisition is **multinomial predictive entropy search**.
**No positional discount in the utility.** OA at ojs.aaai.org. Knowledge Gradient, OCBA-m and top-k
best-arm identification all treat the top-k set as **unordered** — confirmed, as
`rank_targeted_acquisition.md` assumed. **Unverified lead:** ReBOL, arXiv:2603.20513 (BO acquisition over
document batches for LLM-reranked retrieval) — positional discount unchecked.

#### Active learning in recsys with a ranking criterion — **none [V]**
Elahi, Ricci & Rubens (*Computer Science Review* 20:29–50, 2016): taxonomy is uncertainty /
error-reduction / attention-based / influence-based, evaluation dominated by MAE/RMSE. Karimi, *"Active
Learning for Recommender Systems"*, KI 28:329–332, 2014 **[V, PDF]**: criteria are *"uncertainty in latent
space"* and *"uncertainty in rating space"*; result tables are **MAE and RMSE only**. Baidu's web-scale
active LTR (Qin et al., arXiv:2205.10137) uses **Ranking Entropy** (QBC) + **prediction variance** and
only *reports* ΔDCG@4 — textbook optimise-surrogate / report-metric.

#### Rarity evidence, OpenAlex (title+abstract and full text) **[V]**
`"expected DCG"` → **2** (Long et al. + one unrelated 2014 paper) · `"expected NDCG"` → **0** ·
`"expected DCG gain"` (fulltext) → **0** · `"expected NDCG gain"` (fulltext) → **2, both Lin et al.** ·
`"expected NDCG improvement"` → **0** · `"expected reciprocal rank gain"` → **0** ·
`"NDCG acquisition function"` → **0** · `"greedy NDCG"` → **2, both Lin et al.**
Control passes (`"normalized discounted cumulative gain"` → 1,094), so the nulls are real rather than a
broken query. ⚠ **OpenAlex indexes ACM/IEEE full text incompletely** — state the instrument with the
number.

### 3.8 Other adjacent work, for completeness
- **Rank-GRPO**, arXiv:2510.20150, *"Training LLM-based Conversational Recommender Systems with RL"* —
  rank-level GRPO with rank-wise reward shaping, Recall/NDCG, code
  https://github.com/yaochenzhu/Rank-GRPO. **[S]** **The action is generating a ranked list, not choosing
  a question.** The closest thing to "RL with a ranking reward" in CRS, and it resolves to list
  generation. **Nobody has put that reward on the ASK action.** Cite-and-differentiate.
- **"Active Learners as Efficient PRP Rerankers"**, arXiv:2605.14236v2 (2026), code
  https://github.com/jerecoder/IReranker. **[S]** Reframes LLM pairwise reranking as *"active learning
  from noisy pairwise comparisons, choosing adaptively which pairs to query to maximize top-K quality
  within a budget"*; best scheduler *"adaptively selects which pairs to query, concentrating comparisons
  near the top-K boundary"*; +9.7 NDCG@10 at B=300 on TREC DL2019/20. **Top-k set targeting in the
  acquisition, but no positional discount**, and it is IR reranking of a candidate list, not user
  preference elicitation. Evidence that **rank-aware acquisition is currently being reinvented in IR
  rather than recsys.**
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
| **FacT / FacT-CRS** | tree, not a flat order | **BPR** (pairwise) ✅ | ✅ | ❌ (adaptive tree) | pairwise ≠ top-k discounted |
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

**Do NOT write** *"no published question-selection rule optimises a ranking metric."* Two works survive
that sentence and a reviewer will find them. **Write this instead:**

> Of the published selection rules we audit, only two optimise a ranking quantity at the selection step.
> ELO-DCG (Long et al., SIGIR 2010) maximises the posterior **expected DCG regret** of the current
> ranking — a position-discounted generalisation of uncertainty sampling that never simulates the belief
> update. The greedy NDCG selector of Lin et al. (WWW 2023) does simulate the update, but scores each
> candidate against the target user's **held-out validation labels** and evaluates only the YES branch,
> so it is an oracle bound rather than a deployable policy. **No published rule selects by expected
> held-out ranking gain marginalised over a non-circular answer model.**

**Then separate our contribution on the two axes those two papers each fail:**
**(a)** marginalising over a firewalled answer model — *Lin does not*; **(b)** simulating the actual
posterior / representation update — *Long does not*.

**And add the field-named-its-own-gap line, which now has three independent instances spanning 2008–2018:**
Donmez & Carbonell (ICML 2008) list *"direct optimization of ranking performance metrics, such as MAP,
NDCG, precision@k"* as future work; Koren (WSDM 2011) writes *"other cost functions, especially ones
related to the quality of the top-K item ranking"* into his; Sepliarskaia et al. (RecSys 2018) measure
that *"if the method directly optimizes the loss function, then it works better than others."*
**Eighteen years, three explicit statements of the gap, and it is still open.** That is the paragraph.

**Concrete additions to the battery this pass justifies** (ranked by protection-per-hour):
1. **UpsRec as a declared privileged-oracle row.** Cheap (one refiner pass + NDCG recompute per
   candidate — cheaper than our own criterion), and it makes the ceiling visible. Must be labelled an
   oracle in the table.
2. **FacT-CRS's BPR split as a second tree arm next to Golbandi's squared-error split.** Code is live at
   https://github.com/HCDM/XRec. Answers "did anyone try a ranking criterion in the tree?" with a run,
   not a citation.
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
                     [BPR ranking term inside the interview-tree split; NDCG@10]
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
zhu2025rankgrpo      Rank-GRPO: Training LLM-based Conversational Recommender Systems
                     with Reinforcement Learning. arXiv:2510.20150.
                     [ranking reward on the RECOMMEND action, not the ASK action]
paschmann2026prpactive  Figueiredo Paschmann et al. Active Learners as Efficient PRP
                     Rerankers. arXiv:2605.14236, 2026. [top-K-boundary pair acquisition]
long2010elodcg       Long, Chapelle, Zhang, Chang, Zheng & Tseng. Active Learning for
                     Ranking through Expected Loss Optimization. SIGIR 2010.
                     DOI 10.1145/1835449.1835495. Ext.: Long, Bian, Chapelle, Zhang,
                     Inagaki & Chang, IEEE TKDE 27(5):1180-1191, 2015,
                     DOI 10.1109/TKDE.2014.2365785.
                     [ELO-DCG: loss IS DCG, but = expected DCG REGRET under the
                      posterior, i.e. position-discounted uncertainty sampling]
donmez2008rankal     Donmez & Carbonell. Optimizing Estimated Loss Reduction for Active
                     Sampling in Rank Learning. ICML 2008, pp. 248-255.
                     DOI 10.1145/1390156.1390188.
                     [pairwise-surrogate model change; NAMES our gap as future work]
donmez2009auc        Donmez & Carbonell. Active Sampling for Rank Learning via Optimizing
                     the Area Under the ROC Curve. ECIR 2009. [AUC, no positional discount]
bilgic2012activequery  Bilgic & Bennett. Active query selection for learning rankers.
                     SIGIR 2012. DOI 10.1145/2348283.2348455.  [NOT READ]
carterette2006minimal  Carterette, Allan & Sitaraman. Minimal test collections for
                     retrieval evaluation. SIGIR 2006.
                     [greedy judgment selection driven by average precision;
                      EVALUATION-side, not model-side. Second-hand — verify]
nguyen2021topkbo     Nguyen, Tay, Low & Jaillet. Top-k Ranking Bayesian Optimization.
                     AAAI 2021, 35(10).  [top-k is the OBSERVATION model; acquisition is
                      predictive entropy search; NO positional discount]
```

**Existing keys to reuse, not re-add:** `sepliarskaia2018spq`, `golbandi2011adaptive`, `kweon2020deep`,
`zhou2011functional`, `sun2013multiquestion`, `proissl2024greedyslim`, `elahi2016survey`,
`karimi2011optimalal`, `parapar2021dpe`, `austin2024bayesian`.
