# What each baseline's ORIGINAL PAPER actually REPORTED — the metric audit

Added 2026-08-10. Companion to `docs/results/BASELINE_OBJECTIVE_AUDIT.md`, which audits what each
selection rule *internally optimises*. **This file audits a different and sharper question**, raised by
the author:

> A paper may legitimately optimise a surrogate (squared error, information gain, posterior variance)
> *because* ranking quality is not tractable, and then demonstrate that the surrogate improves a
> ranking metric. If the authors reported NDCG/Recall/Precision, their claim IS "this surrogate
> improves ranking" and holding them to NDCG is entirely fair. If they only ever reported RMSE/MAE,
> we may be comparing against methods whose authors never claimed what we are testing.
>
> **"Did we pick wrong baselines where the authors did not care about NDCG?"**

**Source discipline.** Everything below was already established by earlier verified passes and is
re-stated here in one retrievable place; no new lit search was needed for 12 of 13 entries. Primary
records: `findings/rating_elicitation_family_deep.md` §§0, 2.3–2.7, 3.1–3.8; `findings/bandit_rl_crs_baselines.md`
§§1.1–1.3, 3.3; `findings/llm_elicitation_and_metalearning.md` §1.1; `findings/active_learning_elicitation.md`
§§0.7, 1.1; `findings/answerability_in_elicitation.md` Table (row-per-paper). `[V]` = verified from
primary full text in the cited pass.

---

## 1. Per-method records

### 1.1 Golbandi, Koren & Lempel — adaptive bootstrapping ternary interview tree
**Golbandi, Koren & Lempel, "Adaptive Bootstrapping of Recommender Systems Using Decision Trees",
WSDM '11, pp. 595–604, DOI 10.1145/1935826.1935910.** `[V, full text]`

- **(a) Headline metric: RMSE, and nothing else.** Contract: Netflix (>100M ratings, 480k users,
  17,770 items), test = the Qualifying set. Table 1 = RMSE by tree depth (0.97260 / 0.97172 / 0.97129 /
  0.97115 at depth 5/6/7/8); Fig. 3 = RMSE vs number of items shown for GreedyExtend / Var / HELF.
- **(b) No ranking metric appears in the paper at all** — not primary, not auxiliary.
- **(c) Did the authors claim a ranking improvement? NO — and they explicitly disclaim being a
  recommender.** Verbatim §: *"Compared to other (non-bootstrapping) recommendation algorithms, a
  decision-tree recommender delivers a quite high RMSE. This makes decision-trees undesirable as a
  traditional recommendation system."* The tree is offered as a **bootstrapping instrument**, and its
  claim is RMSE-after-k-questions vs RMSE-after-k-organic-ratings (*"the first 12 ratings of users were
  needed in order to match the accuracy of a depth-6 tree"*).
- **(d) Dataset / budget:** Netflix; the interview budget is the tree depth, **6–8 questions** (headline
  is depth 6 = 6 questions). Budget range overlaps ours; metric does not.
- Predecessor **Golbandi, Koren & Lempel, CIKM '10 (GreedyExtend)**, DOI 10.1145/1871437.1871734: also
  **RMSE only** `[A/S — full text unobtainable; mechanism and numbers via WSDM'11 §3.1.1 + patent
  US8510248B2]`; ">20 shown items to reach RMSE 0.971".

### 1.2 Rashid et al. — IGCN / Entropy0 / HELF (and the 2002 ancestor)
**Rashid, Karypis & Riedl, "Learning Preferences of New Users in Recommender Systems: An Information
Theoretic Approach", SIGKDD Explorations 10(2), 2008, pp. 90–100, DOI 10.1145/1540276.1540302.** `[V, full text]`

- **(a) Headline metrics: MAE, "#movies seen" (answer rate), and Expected Utility.** No NDCG, no
  Recall@k, no Precision@k, no MAP. Offline Fig. 5 = #seen vs #presented per strategy; online Table 5 =
  pages to finish 20 ratings (Entropy0 3.0, Popularity 3.6, HELF 4.6, IGCN 5.5); online MAE via user-kNN
  (IGCN 0.67 best, Popularity 0.73, Entropy0 0.73, HELF 0.84 worst).
- **(b) N/A — no ranking metric anywhere.**
- **(c) NO ranking claim.** The headline verdict is stated on MAE: *"Overall, the Popularity measure
  performed the worst, and the Igcn measure performed the best."* Expected Utility is a rank-weighted
  *utility of a recommendation list*, so it is the nearest thing to a ranking metric in the whole
  classical lineage — but it is a secondary table used to resolve the Popularity/HELF paradox, not the
  headline, and it is not a held-out top-N accuracy metric.
- **(d)** MovieLens ~11,000 users / 9,000 items / ~3M ratings; budgets **15/30/45/60/75 presented items**
  (pages of 15) — i.e. an order of magnitude above our k≤16 in *presented* items. Live study: 381 valid
  subjects over 20 days. **IGCN parameters (c,n,i,l) = (300 clusters, 15/screen, switch after 5 ratings,
  150 neighbours)** — the switch-after-5 fact that produced our `igcn-published-form-is-static-at-our-budgets`
  finding.
- Ancestor **Rashid et al., "Getting to Know You", IUI '02, pp. 127–134** `[V, full text]`: metrics are
  **#movies seen and MAE only**; offline results are plots; budgets 30/45/60/90 presented. Also no
  ranking metric.

### 1.3 Zhou, Yang & Zha — functional matrix factorization (fMF)  *(RULED OUT for running; recorded anyway)*
**Zhou, Yang & Zha, "Functional Matrix Factorizations for Cold-Start Recommendation", SIGIR '11,
pp. 315–324, DOI 10.1145/2009916.2009961.** `[V, full text]`

- **(a) Headline metric: RMSE only.**
- **(b) No ranking metric.**
- **(c) NO ranking claim.** The claim is cold-start RMSE beating Golbandi's tree and TreeU on all three
  datasets, t-test at p=0.05 (no CIs, no paired test described, no seed count).
- **(d)** ML-1M, EachMovie, Netflix. **Budget 3/4/5/6/7 questions (= tree depth) — overlaps our k range
  almost exactly.** ML-1M cold-start RMSE 0.9509 / 0.9480 / 0.9437 / 0.9429 / 0.9410 at q=3..7 vs
  Golbandi Tree 0.9767 / 0.9683 / 0.9615 / 0.9512 / 0.9523. Its warm-start tables use a *different*
  protocol — never order numbers across them.

### 1.4 EDDI (Ma et al., partial VAE / icebreaker line)
**Ma, Tschiatschek, Palla, Hernández-Lobato, Nowozin & Zhang, "EDDI: Efficient Dynamic Discovery of
High-Value Information with Partial VAE", ICML 2019, PMLR 97:4234–4243 (arXiv:1809.11142).**
`[V for datasets — full arXiv PDF grep'd locally 2026-08-06, see `active_learning_elicitation.md` §0.7;
V-prior for the metric names]`

- **(a) Headline metrics: RMSE and negative test log-likelihood as *information curves* (performance vs
  number of features acquired), summarised by area under the information curve (AUIC).**
- **(b) No ranking metric — and none is possible: EDDI is not evaluated on recommendation at all.**
  Verified word counts over the full 22-page PDF: `MovieLens 0 · rating 0 · UCI 28 · MNIST 7 · NHANES 11`.
  Its own contribution list: image inpainting; 6 UCI datasets; **risk assessment in intensive care
  (MIMIC)**; **public health assessment (NHANES)**.
- **(c) NO ranking claim, and no recommendation claim.** The paper's claim is *"cost reduction at the
  same decision quality and improved decision quality at the same cost"* on ML benchmarks and two
  healthcare applications.
- **(d)** UCI / MNIST / MIMIC-III / NHANES; budget = the full feature-acquisition curve, not an interview
  length. ⚠ **This is the single largest contract gap in the whole battery** — the method has never been
  run on a recommender by its own authors. Our EDDI row is a *port of an acquisition function into a
  domain its authors did not enter*, and must be labelled that way.
- Successor **Icebreaker (Gong et al., NeurIPS 2019, arXiv:1908.04537)** does use MovieLens-shaped
  element acquisition, but its target is *model training economy* (which cells to buy to train the
  recommender), not cold-user interview quality `[A]`.

### 1.5 DRE — deep representation elicitation / seed-item reconstruction
**Kweon, Kang, Hwang & Yu, "Deep Rating Elicitation for New Users in Collaborative Filtering",
WWW '20, pp. 2810–2816, DOI 10.1145/3366423.3380042.** `[V, full text + repo inspected]`

- **(a) Headline metrics: Precision@N and NDCG@N, N ∈ {10,20,50,100}. RANKING, PRIMARY.**
- **(b) PRIMARY.** The main results table is P@10/P@20/NDCG@10/NDCG@20.
- **(c) YES.** ML-20M at k=50: DRE P@10 0.5712 / P@20 0.4875 / **NDCG@10 0.6077** / NDCG@20 0.5373 vs
  best baseline RMVA 0.5116 / 0.4382 / 0.5444 / 0.4851 → **+11.6% NDCG@10, p ≤ 0.005**. Best headline
  CiteULike NDCG@10 +31.98%; **weakest Yelp NDCG@10 +1.56%, not significant.**
- **(d)** ML-1M, CiteULike, Yelp, **ML-20M (99,220 u / 20,660 i)**; **binarize at >3.5 — our threshold**;
  80/20 user split. **Primary budget k = 50** — 3× our largest. The k-sweep exists only as Figure 3
  (P@20 vs k); **per-k ML-20M numbers are NOT recoverable from the text**, so any such number is `[S]`.

### 1.6 DPE — topic/preference elicitation bandit
**Parapar & Radlinski, "Diverse User Preference Elicitation with Multi-Armed Bandits", WSDM '21,
DOI 10.1145/3437963.3441786.** `[V, full text; step-0 re-extraction 2026-08-07]`

- **(a) Headline metrics: P@10, P@100, nDCG@10, nDCG@100, serendipity@10/@100, topdiv@10/@100.
  RANKING, PRIMARY** (alongside diversity).
- **(b) PRIMARY** — but ⚠ **the paper reports NO scalar numbers at all.** Table 1 is dataset statistics;
  every result is an unannotated curve (Fig. 1 = ML-20M, 8 metrics × {TestItems, AvailablePreferences};
  Fig. 2 = Amazon, 4 metrics at cut-off 100). **There is nothing to snap to except orderings** (O1–O6,
  recorded in `rating_elicitation_family_deep.md` §3.8-bis).
- **(c) YES** — the claim is a *joint* accuracy+diversity improvement: popularity bias in elicitation
  propagates into recommendation, and diversifying at elicitation time improves both. The strongest
  ordering claim: *"for every cut-off and metric, the Greedy baseline performs much worse than DPE"*
  (AvailablePreferences); reversed at cut-off 10 under TestItems, where *"the Greedy baseline achieves
  the best performance both in terms of accuracy (graded and ungraded) and diversity."*
- **(d)** ML-20M (51,869 u / 26,654 i / 20 categories, users filtered to **>100** ratings — the off-by-one
  we recovered) and Amazon 2014. **Budget 5 → 75 elicited preferences.** Recommender = WSR (ECIR'16).
  nDCG via `trec_eval` with **ratings as graded gain**; relevance for P@k = rating ≥ 4.
- ⚠ **100% answer rate by construction** — users filtered to ≥100 ratings *"given that our evaluation
  elicits up to 75 preferences per user"*; the ask must land inside the user's own rated set. DPE never
  models an unanswerable question.
- ⚠ **Our `dpe_concept` arm is OUR DEVIATION.** DPE's bandit is over topics but **its questions are
  items** (Algorithm 1: `next_topic ← argmax θ_a` … `next_item ← argmax δ_i`). The concept-asking
  variant is buildable and implemented, but must be labelled as ours, not as DPE.

### 1.7 IGCN (as we implemented it)
Same paper as §1.2 — **Rashid, Karypis & Riedl, SIGKDD Explorations 2008**. There is no separate IGCN
paper. **(a) MAE / #seen / Expected Utility. (b) no ranking metric. (c) no ranking claim.
(d) MovieLens, 15–75 presented items, personalised phase engaging only after 5 ratings.**
See `docs/results/SNAPIGCN_NATIVE.md` and memory `igcn-published-form-is-static-at-our-budgets`: at our
budgets and answer rate the published form is 1.9% personalised at k=8 and 0% at k≤4.

### 1.8 ConUCB — conversational contextual bandit
**Zhang, Xie, Wang, Ding, Zheng et al., "Conversational Contextual Bandit: Algorithm and Application",
WWW 2020.** `[V]`

- **(a) Headline metric: cumulative regret, `R(T) = Σ_t (E[r_{a*_t,t}] − E[r_{a_t,t}])`.** Plus regret
  bound `O(d√T log T)`.
- **(b) N/A — no ranking metric of any kind.**
- **(c) NO.** The claim is a regret improvement from conversing (asking key-terms) vs not.
- **(d)** Synthetic (5,000 arms / 500 key-terms / d=50 / 200 users); **Yelp** (1,998 users, 25,160
  restaurants, 1,004 categories as key-terms); **Toutiao** (2,000 users, 1,746,335 articles, 2,384
  keywords). Budget is not an interview length: it is a **conversation frequency schedule** over T
  rounds, `b(t)=5⌊log t⌋` or `b(t)=5⌊t/50⌋`. **No target item; arms are re-drawn each round.**
  No code in the paper.

### 1.9 ConLinUCB variants (-MCR, -BS, and our -BS-static)
**Wang, Liu, Li & Lui, "Efficient Explorative Key-term Selection Strategies for Conversational
Contextual Bandits", AAAI 2023, pp. 10288–10295 (arXiv:2303.00315).** `[V]`

- **(a) Headline metric: cumulative regret.** Bound `O(d√(T log T))`, i.e. an improvement over ConUCB's
  `O(d√T log T)`.
- **(b) N/A — no ranking metric.**
- **(c) NO.**
- **(d)** Synthetic (5,000/500/d=50/200 users); **Last.FM** (2,000 arms / 2,726 key-terms / 500 users);
  **MovieLens** (2,000 arms / 5,585 key-terms / 500 users) — a 2,000-item truncation, not the full
  catalogue. Code: https://github.com/ZhiyongWangWzy/ConLinUCB.
- ⚠ **`conlinucb_bs_static` is OURS, not theirs.** With `b_schedule="always"` and the spanner draw
  seeded on `[seed, turn]` only, it is one static random order over the barycentric spanner, identical
  for all 10,000 users (measured: **1 distinct interview**). It belongs beside `bald`/`aopt` as
  structurally non-adaptive and must not be tabled as a published adaptive arm.
- NB: "MCR" here = ConLinUCB-**MCR** (Maximum Confidence-Radius key-term selection). It is **not** the
  MCR of Lei et al. (Multi-Choice Recommendation, CRS) — do not conflate the two in the chapter.

### 1.10 CLiME
**Liu, Li, Dai & Lui, "Conversational Bandits with Smoothed Key-terms / Adaptive Conversation
Mechanisms" (CLiSK / CLiME / CLiSK-ME), KDD 2025 (arXiv:2505.21393).** `[V]`

- **(a) Headline metric: cumulative regret.** Bounds: CLiSK `O(√(dT log T) + d)`; CLiME / CLiSK-ME
  `O(√(dT log T))`. Their diagnosis of ConUCB is itself a regret statement: `O(d√T log T)` is *"no
  better than not conversing at all"*.
- **(b) N/A — no ranking metric.**
- **(c) NO.**
- **(d)** **ML-25M — but truncated to |A| = 5,000 arms (highest rating/tag count) and N = 200 users
  (most active)**; key-terms = movie genres; d = 50 from truncated SVD; binary reward (rating > 3);
  **T = 6,000 rounds**; baselines LinUCB, Arm-Con, ConUCB, ConLinUCB-BS, ConLinUCB-MCR.
  Code: https://doi.org/10.5281/zenodo.15490021. ⚠ Same dataset name as ours, 0.03% of the users.

### 1.11 Greedy SLIM / QGSLIM
**Proissl, Vatic & Waldschmidt, "Greedy SLIM: A SLIM-Based Approach For Preference Elicitation",
arXiv:2406.06061 [cs.IR], 10 Jun 2024. arXiv-only — dblp lists it under CoRR with no venue.** `[V, full text]`
(QGSLIM = the paper's own name for the Greedy SLIM *questionnaire*; same paper.)

- **(a) Headline metric: NDCG@10, PRIMARY, reported for ALL items AND LONG TAIL, plus P@N and R@N.
  RANKING, and the closest contract match in the entire literature.** Their tail split is verbatim ours:
  *"Following Cremonesi et al. (2010), we refer to the set of most popular items that represent 33% of
  the ratings as short-head items."*
- **(b) PRIMARY.**
- **(c) YES**, and with a budget-dependent crossover stated in words: *"QBandit seems to perform
  slightly better than QGSLIM if the questionnaire consists of very few questions… QGSLIM is able to
  profit from every additional question, which leads to favorable results if the questionnaire consists
  of ten or more questions."*
- **(d)** **ML-25M (162,541 u / 59,047 i / 25,000,095 r) UNFILTERED** + Netflix; explicit 1–5 ratings,
  not binarized; random 90/10 user split; **budgets k ∈ {5,10,15,20}**. ML-25M NDCG@10:

  | k | ALL QGSLIM | ALL QBandit | TAIL QGSLIM | TAIL QBandit |
  |---|---|---|---|---|
  | 5 | **0.3390** | 0.3382 | 0.1163 | **0.1183** |
  | 10 | **0.3594** | 0.3456 | **0.1543** | 0.1465 |
  | 15 | **0.3709** | 0.3221 | **0.1674** | 0.1488 |
  | 20 | **0.3752** | 0.2916 | **0.1779** | 0.1466 |

  (Netflix reverses at k≤10: QBandit 0.3030/0.3251 vs QGSLIM 0.2859/0.3042.)
- ⚠ **Do NOT put these numbers beside ours** — graded gain `2^r − 1` on raw ratings vs our binary
  implicit; 90/10 user split with everything unasked as targets vs Liang 80/20 fold-in/eval; **their
  candidate pool excludes ASKED items, ours excludes RATED items** (see `candidate_masking_protocol.md`;
  their adaptive decline is partly a protocol artefact by their own diagnosis).
- ★ **The most load-bearing sentence in this file for the "is the comparison fair" question:** this group
  *tried* to put the RMSE-lineage methods into an NDCG@10 table and could not — *"For Golbandi et al.
  [2011] and Sepliarskaia et al. [2018] we, unfortunately, obtained poor results. We therefore decided to
  not include them in our official experiments."* Plus: *"the reproducibility issue within the recommender
  systems community is also an issue when it comes to questionnaires. We had to implement all the
  baselines we wanted to consider on our own."*

### 1.12 RBMF (maxvol selector) — dropped by us; recorded
**Liu, Meng, Liu & Yang, "Wisdom of the Better Few: Cold Start Recommendation via Representative Based
Rating Elicitation", RecSys '11, pp. 37–44, DOI 10.1145/2043932.2043943.** `[V, full text]`

- **(a) Headline metrics: Precision@10, MAP, AUC, plus its own Coverage and Diversity. RANKING, PRIMARY.**
  RBMF is the paper that **starts the ranking-metric era in this family** — everything before it is
  RMSE/MAE.
- **(b) PRIMARY.**
- **(c) YES.** ML: at k=5 P@10 RBMF **0.501** vs MostPop 0.432; k=10 0.559 vs 0.500; k=20 0.637 vs 0.530.
  **Honest caveat already recorded:** at k=5 on Netflix RBMF (0.465) barely beats MostPop (0.459), and
  **MostPop wins on AUC**.
- **(d)** Netflix, ML-10M, Last.fm. **Budgets 5 / 10 / 20 — overlapping ours exactly.** Coverage (§4.3)
  and accuracy (§4.4) are reported in separate experiments and never linked. No code.

### 1.13 PEBOL — LLM-based preference elicitation
**Austin\*, Korikov\*, Toroghi & Sanner, "Bayesian Optimization with LLM-Based Acquisition Functions for
Natural Language Preference Elicitation", RecSys 2024, arXiv:2405.00981v2, DOI 10.1145/3640457.3688142.**
`[V, full PDF]` (indexed twice as `austin2024bayesian` / `austin2024pebol` — reconcile the bib-key.)

- **(a) Headline metric: MRR@10 over 10 dialogue turns** (*"equivalent to MAP@10 for the case of a
  single preferred item"*), at noise levels ∈ {0, 0.25, 0.5}. **A RANKING metric, PRIMARY.**
- **(b) PRIMARY.**
- **(c) YES**, on MRR@10. ⚠ Correction to the abstract's "0.27 vs 0.17", which mixes datasets: per-dataset
  at turn 10, PEBOL vs GPT-MonoLLM = **0.27/0.12 Yelp, 0.18/0.09 MovieLens, 0.17/0.11 Recipe-MPR**;
  vs Gemini-MonoLLM = 0.27/0.17, 0.18/0.15, 0.17/0.16.
- **(d)** ⚠ **Catalogue = 100 ITEMS.** Verbatim: *"we have to limit |I| to 100 for fair comparison to
  the MonoLLM baseline"*. Sources: ML-25M (top 10% by rating count, ≥20 tags), Yelp (Philadelphia
  restaurants), Recipe-MPR. Budget = **10 turns**. **Oracle is 100% LLM-simulated**, and the simulated
  user reads the *same* description string the NLI scorer entails against — compounded circularity.
  Our catalogue is **184× larger**.
- Note the split that makes an LLM-free port honest: **stage 1 (selection) is a closed-form bandit rule
  over Beta posteriors (TS/UCB/ER/Greedy) — the LLM is NOT in the selection loop**; stage 2 (phrasing)
  is the only LLM step.

---

## 2. SUMMARY TABLE

Legend: **RANK** = a held-out ranking/top-N metric was reported. **ERR** = rating-prediction error only.
**REGRET** = cumulative regret over interaction rounds, no held-out ranking.

| # | Method | Original paper (year, venue) | (a) Headline metric(s) | (b) Ranking primary? | (c) Claims ranking gain? | (d) Dataset(s) / budget | Class |
|---|---|---|---|---|---|---|---|
| 1 | Golbandi tree | Golbandi, Koren & Lempel, WSDM 2011 | **RMSE** (Table 1, Fig. 3) | — none reported | **NO** — self-disclaims: *"decision-trees undesirable as a traditional recommendation system"* | Netflix / depth 6–8 = 6–8 questions | **ERR** |
| 1b | GreedyExtend | Golbandi et al., CIKM 2010 | **RMSE** | — | NO | Netflix / >20 items | **ERR** |
| 2 | Rashid Entropy0 / HELF / IGCN | Rashid, Karypis & Riedl, SIGKDD Expl. 10(2) 2008 | **MAE**, #movies seen, Expected Utility | — no top-N metric | **NO** — verdict stated on MAE | MovieLens ~11k u / 15–75 presented; live 381 subjects | **ERR** |
| 2b | Rashid Pop·Ent / Item-Item | Rashid et al., IUI 2002 | **MAE**, #movies seen | — | NO | MovieLens 7,335 u / 30–90 presented | **ERR** |
| 3 | fMF | Zhou, Yang & Zha, SIGIR 2011 | **RMSE** | — | NO | ML-1M, EachMovie, Netflix / **q = 3–7** | **ERR** |
| 4 | EDDI | Ma et al., ICML 2019 | **RMSE + neg. test log-likelihood** as information curves (AUIC) | — and **not a recsys paper** | **NO** — "decision quality at cost" | UCI, MNIST, MIMIC, NHANES / feature-acquisition curve | **ERR (non-recsys)** |
| 5 | DRE | Kweon et al., WWW 2020 | **P@N and NDCG@N**, N∈{10,20,50,100} | **YES, primary** | **YES** — +11.6% NDCG@10 over RMVA, p≤0.005 (ML-20M) | ML-1M/CiteULike/Yelp/**ML-20M**, binarized **>3.5** / **k=50** | **RANK** |
| 6 | DPE | Parapar & Radlinski, WSDM 2021 | **nDCG@10/@100, P@10/@100** + serendipity, topdiv | **YES, primary** — but **curves only, zero scalars** | **YES** (joint accuracy+diversity; ordering flips by cut-off/protocol) | ML-20M, Amazon 2014 / **5–75 elicited** | **RANK** |
| 7 | IGCN (as implemented) | = row 2 | **MAE** / #seen / EU | — | NO | as row 2; personalised phase only after 5 ratings | **ERR** |
| 8 | ConUCB | Zhang et al., WWW 2020 | **cumulative regret** `O(d√T log T)` | — | NO | synthetic, Yelp, Toutiao / conversation-frequency schedule over T rounds | **REGRET** |
| 9 | ConLinUCB-MCR / -BS | Wang, Liu, Li & Lui, AAAI 2023 | **cumulative regret** `O(d√(T log T))` | — | NO | synthetic, Last.FM (2k arms), MovieLens (2k arms) / T rounds | **REGRET** |
| 9b | ConLinUCB-BS-**static** | **OURS** — not a published arm | — | — | — | 1 distinct interview across 10,000 users | **OURS** |
| 10 | CLiME | Liu, Li, Dai & Lui, KDD 2025 | **cumulative regret** `O(√(dT log T))` | — | NO | **ML-25M truncated to 5,000 arms / 200 users** / T=6,000 | **REGRET** |
| 11 | Greedy SLIM / QGSLIM | Proissl, Vatic & Waldschmidt, arXiv:2406.06061 (2024, **no venue**) | **NDCG@10 (all + long tail)**, P@N, R@N | **YES, primary** | **YES**, with an explicit budget-dependent crossover vs adaptive QBandit | **ML-25M unfiltered** + Netflix / **k = 5,10,15,20** | **RANK** |
| 12 | RBMF (maxvol) | Liu, Meng, Liu & Yang, RecSys 2011 | **P@10, MAP, AUC** + Coverage, Diversity | **YES, primary** | **YES** — P@10 0.501 vs MostPop 0.432 at k=5 (ML); caveat: MostPop wins AUC on Netflix k=5 | Netflix, ML-10M, Last.fm / **k = 5, 10, 20** | **RANK** |
| 13 | PEBOL | Austin, Korikov, Toroghi & Sanner, RecSys 2024 | **MRR@10** over 10 turns | **YES, primary** | **YES** on MRR@10 | ML-25M/Yelp/Recipe-MPR **sampled to 100 items**, LLM-simulated users / **10 turns** | **RANK** |

**Tally over the 13 distinct source papers:** **5 RANK** (DRE, DPE, Greedy SLIM, RBMF, PEBOL) ·
**5 ERR** (Golbandi WSDM'11, Golbandi CIKM'10, Rashid'02/'08 incl. IGCN, fMF, EDDI) ·
**3 REGRET** (ConUCB, ConLinUCB, CLiME).

**Adjacent rows in the battery, for completeness:**
- `bald` — Houlsby et al. 2011: an AL criterion paper, no recsys ranking metric. **ERR/other.**
- `aopt` — the A-optimal criterion enters via **Sepliarskaia, Kiseleva, Radlinski & de Rijke, RecSys 2018**
  (SPQ/PEO), which **does** report **P@10** (and PCPF) on ML-20M binarized at ≥4, N ≤ 29, figures only.
  So the A-optimal design *was* ranking-validated by its recsys-side authors. **RANK.**
- `popularity` / `pure_entropy` — Rashid lineage, **ERR**.
- `golbandi_leaf` — the leaf model of row 1, **ERR**.

---

## 3. BOTTOM LINE — "did we pick wrong baselines?"

**No — but the battery is not one thing, and the chapter must stop treating it as one.** The honest
answer has three parts, and only the second is uncomfortable.

**(i) The modern half of the battery is a fair fight on our own terms. 5 of 13 source papers report a
ranking metric as their PRIMARY result and explicitly claim a ranking improvement: DRE (NDCG@10), Greedy
SLIM/QGSLIM (NDCG@10, all + tail, on ML-25M, with our exact Cremonesi tail definition), RBMF (P@10/MAP/AUC),
DPE (nDCG@10/@100), PEBOL (MRR@10).** For these, holding them to NDCG@10 is exactly what their authors
asked to be held to. Add SPQ/A-optimal (P@10) and that is 6. **This is the strongest fact in the file
and it should be stated plainly in the chapter**: the comparison against the methods a 2026 reviewer
actually names is fair, and beating them on NDCG is beating them on their own headline.

**(ii) The classical interview lineage is a SURROGATE TRANSFER, and we must say so in print. Golbandi
(WSDM'11 + CIKM'10), Rashid (IUI'02, SIGKDD Expl.'08 — which is where entropy0, HELF, popularity,
pure_entropy AND IGCN all come from), and fMF reported RMSE/MAE and nothing else. None of them claimed
a ranking improvement.** Golbandi goes further and *disclaims* being a recommender at all: *"a
decision-tree recommender delivers a quite high RMSE. This makes decision-trees undesirable as a
traditional recommendation system."* That is a paper offering an **instrument**, exactly as Paper A
offers one — and it is a poor look to table its NDCG without a sentence saying what it optimised and
what it reported. **This is not a reason to drop them.** They are the field's canonical cold-start
elicitation arms, the 2025 tree literature *still* benchmarks against popularity/entropy/HELF/IGCN, and
we must show the classical bank. It is a reason to change one caption. Recommended wording, load-bearing:

> Golbandi (WSDM'11), Rashid (SIGKDD Expl.'08) and fMF (SIGIR'11) reported RMSE/MAE only; the published
> ranks transfer to our protocol as PRIORS, not as certified numbers, and the NDCG column here is ours,
> not theirs.

(That sentence already exists verbatim in `interview_strategies.py:24-26`; it needs to reach the chapter.)

**(iii) The conversational-bandit trio never had a held-out target at all — this is a bigger mismatch
than the RMSE one, and it is currently under-declared.** ConUCB, ConLinUCB(-MCR/-BS) and CLiME report
**cumulative regret over T interaction rounds against re-drawn arm sets**. There is no cold user, no
interview budget, no held-out ranking, and no target item — the quantity is reward *during* the
interaction. Their budgets are T = 6,000 rounds, not k ≤ 16 questions; CLiME's "ML-25M" is 5,000 items
and 200 users. **Our rows for these are ports of a selection rule into a protocol its authors never
used**, and the arms should be named that way in the table ("ConUCB's key-term selection rule, ported").
Two extra honesty items belong with them: `conlinucb_bs_static` is **our** construction and not a
published arm at all, and it conducts **one** interview for all 10,000 users.

**(iv) EDDI is the single largest contract gap and it is not a recsys method.** Verified from the full
PDF: 0 occurrences of "MovieLens", 0 of "rating". EDDI was evaluated on UCI, MNIST, MIMIC-III and NHANES
with RMSE/NLL information curves. Its authors made **no recommendation claim of any kind**. The row is
legitimate as *the principled information-gain acquisition function that an active-learning reviewer
will ask for*, but it must be introduced as an acquisition function imported from active feature
acquisition, never as "the EDDI baseline underperforms".

**What this does NOT do:** it does not soften `BASELINE_OBJECTIVE_AUDIT.md`. That audit's finding —
1 of 21 selection rules optimises held-out ranking — stands, and this file explains *why that is
reasonable*: optimising a tractable surrogate is the correct engineering move. What this file adds is
that **the surrogate→ranking transfer was only ever DEMONSTRATED by 6 of them.** For the other 7 the
transfer is an assumption, and our battery is the first place it is measured. **That is a stronger
framing than "the field optimises the wrong thing":** the field optimises a reasonable surrogate, half
of it never checked that the surrogate transfers to the metric the field now reports, and we checked.

**Concrete actions for Paper B (no `docs/` or `new_chapters/` edits made here):**
1. Split the battery table into three labelled blocks — **ranking-validated** (DRE, Greedy SLIM, RBMF,
   DPE, PEBOL-acquisition, SPQ/A-opt), **rating-error lineage** (Golbandi, Rashid family incl. IGCN /
   entropy0 / HELF / popularity / pure_entropy, fMF), **regret lineage / no held-out target**
   (ConUCB, ConLinUCB-MCR/-BS, CLiME) — plus **imported acquisition functions** (EDDI, BALD).
2. Put the surrogate-transfer sentence in the caption of the second and third blocks.
3. Mark `conlinucb_bs_static` and `dpe_concept` as **our deviations**, not published arms.
4. Cite Greedy SLIM's failed Golbandi/Sepliarskaia reproduction as the published precedent that this
   transfer is hard, and as the reason our own faithful ports are a contribution.
5. Never table Golbandi/fMF/Rashid published *numbers* beside ours; cite the *ordering* only.

## 4. Verification debts left open by this pass
- **EDDI's exact metric wording** is `[V-prior]`, not re-verified from primary text this pass: the PMLR
  PDF (`proceedings.mlr.press/v97/ma19c/ma19c.pdf`) fetched here is **image-only** (10 pages, zero
  extractable text). The *datasets* are `[V]` (arXiv PDF grep'd 2026-08-06) and they are sufficient to
  establish the load-bearing claim (no ranking metric is possible on UCI/MNIST/MIMIC/NHANES). If the
  chapter quotes "AUIC" verbatim, re-extract from arXiv:1809.11142 first.
- **ConUCB's venue** is recorded as WWW 2020 in `bandit_rl_crs_baselines.md` §1.1; some secondary sources
  say SIGIR 2020. Confirm before the bib is frozen. (Web-search budget was exhausted this session.)
- **Rashid 2008's Expected Utility** is the one metric in the ERR block that is rank-sensitive. If a
  reviewer presses "the classical line never reported ranking", the precise statement is *never a
  held-out top-N accuracy metric*; EU is a rank-weighted utility and is the nearest miss.
