> # ⚠ SUPERSEDED IN PART — READ `rating_elicitation_family_deep.md` §0 FIRST (added 2026-08-05)
> A second, exhaustive pass found **eleven corrections to this file**, including three that change
> conclusions: **(a)** the closest contract match in the whole literature — **Greedy SLIM, ML-25M,
> NDCG@10, our exact Cremonesi tail split** — is missing here entirely, and it is the new strongest
> static bar; **(b)** the **"PERE gives 3–10% relative NDCG@10" figure is not in the PERE paper** and is
> refuted (real margins <3% on five of six settings); **(c)** **Sepliarskaia never compares against
> Golbandi**, ties everything below 5 questions, and attributes the tree's loss to its objective rather
> than to staticness. Also: this file's **§8b harness audit is stale** (adaptive selectors now exist),
> and the **Trinity College Dublin MCTS/RLT4Rec line** is absent. Do not cite from this file without
> checking the corrections list.

# Adaptive cold-start elicitation baselines, 2018–2026 — what Paper B Act 1 must beat

*Added 2026-08-04. Question asked: "Golbandi (WSDM'11) alone is too old to argue Act 1. What are the
strongest RECENT, REPLICABLE adaptive/interview cold-start elicitation methods, and which of them fit
our contract?" Verification levels: **[V]** primary text extracted, **[A]** abstract/landing page only,
**[S]** secondary (search summary / third-party). Nothing here is quoted unless it came from [V].*

Read this **with** `rank_targeted_acquisition.md` (what acquisition criteria exist),
`answerability_in_elicitation.md` (why the CRS-RL line's simulator is omniscient),
`unified_embedding_architecture.md` and `paperA_recommender_landscape.md` (ConTS/Biyik/TaNP/MeLU rows).
Those files already settled the CRS-RL lineage; this file adds the **rating-elicitation** lineage
2018–2026 and the contract-fit verdicts.

---

## 0. The headline

**The field's post-Golbandi adaptive elicitation line is much thinner than it looks, and it splits into
two families that do not talk to each other:**

1. **Rating elicitation / seed selection** (Rashid → Golbandi → Zhou fMF → RBMF/RMVA → **DRE WWW'20** →
   **PERE UAI'24** → **pairwise/attribute trees 2025**). This family shares our contract almost exactly:
   cold user, budget of item questions, held-out ranking metric (NDCG@N) on MovieLens. It is where the
   defensible baselines are. Crucially, **its 2020 and 2024 state of the art are DRE (static, one-shot,
   learned) and PERE (adaptive, region-based)** — so the "static vs adaptive" question Act 1 measures is
   *live and unresolved in this family*, not a strawman.
2. **Interactive/conversational RL** (ICF bandits → CRM → EAR → SCPR → ConTS → UNICORN → NICF → RLT4Rec).
   This family looks like our problem but **prices a different thing**: its metric is *cumulative reward
   collected during the interaction* (or success-rate@turn / average turns), not the quality of a
   *subsequent* held-out ranking. Its questions ARE recommendations. **Numbers from this family are not
   orderable against ours** (HARD-RULE / `surprise-means-debug-the-harness`). Their *policies* can be
   ported and re-scored on our metric; their *published numbers* cannot be cited as a comparison.

**The single most useful correction to the author's framing:** the strongest *static* elicitor in the
literature is not popularity or RBMF — it is **DRE (Kweon et al., WWW 2020)**, an end-to-end
Gumbel-Softmax learned seed set, and it *beat* RBMF and RMVA on ML-20M NDCG@10 **[V]**. If Act 1 says
"no uplift over the strongest static", DRE is the static bar a 2026 reviewer will name. It has public
PyTorch code and its dataset is ML-20M. Not running it is the reviewable hole.

---

## 1. The rating-elicitation family (our contract)

### 1.1 DRE — Deep Rating Elicitation, Kweon/Kang/Hwang/Yu, WWW 2020 **[V]**
`kweon2020deep` (already in INDEX, previously only a one-line row). arXiv:2402.16327 is a re-upload of
the WWW'20 paper — **not a 2024 paper**, the correction already recorded in INDEX stands.

- **Mechanism.** Encoder defines *categorical distributions over the whole itemset*; seed items sampled
  via **Gumbel-Softmax** so the selection is differentiable; a 2-layer decoder reconstructs the full
  rating vector from the seeds' feedback. Selection and inference are trained **end-to-end**.
- **Static or adaptive:** **STATIC by design and deliberately so** — "chooses all the seed items at a
  time with consideration of the non-linear interactions between the items", explicitly contrasted with
  greedy incremental selection. This is the paper that argues the *static* side of Act 1's question.
- **Contract:** MovieLens 1M (6,028 u / 3,533 i), CiteULike, Yelp, **MovieLens 20M (99,220 u / 20,660 i)**.
  80/20 user split, 10% of train users held out for validation. Metrics **P@N and NDCG@N, N∈{10,20,50,100}**.
  Primary seed budget **k=50**. Baselines: MOSTPOP, RAN++, POP++, **RBMF**, RBMF++, **RMVA**, RMVA++.
- **Code:** https://github.com/WonbinKweon/DRE_WWW2020 (PyTorch).
- **Fit:** near-perfect. Same dataset family, same metric, same seed-budget framing, and its two named
  baselines (RBMF, RMVA) are things we already have or can get. **What must change:** k=50 is far above
  our k∈{2..16} — retrain the categorical distributions per budget (cheap; it is one softmax head).
  Its decoder is its own recommender — for a clean strategy/recommender separation, take DRE's *selected
  seed set* into our harness and let our tower consume it, and additionally report DRE-with-its-own-decoder
  as its published configuration.
- **Verdict: MUST-INCLUDE (as the strongest STATIC bar).**

### 1.2 PERE — Cold-start Rec by Personalized Embedding Region Elicitation, Nguyen et al., **UAI 2024** **[V-abstract+html]**
arXiv:2406.00973 · https://openreview.net/forum?id=ciOkU5YpvU · PMLR v244 (UAI 2024).

- **Mechanism.** Two phases. (a) **Burn-in**: a **DPP** picks K=50 diverse popular items (non-personalized).
  (b) **Adaptive Q&A**: the user is represented as a **region** in a *frozen* embedding space (not a point);
  each answer cuts the region with a halfspace; the next items are chosen to maximise region reduction
  (information gain), weighted by an estimated *experience probability* (i.e. answerability!). Recommendation
  uses the region's **Chebyshev center**.
- **Static or adaptive:** **ADAPTIVE, per-user, sequential** — and its explicit motivating claim is that
  "a fixed seed set is unlikely optimal for all new users." **This paper is the field's 2024 assertion of
  exactly the thing Act 1 measures.** If Act 1 reports no uplift, PERE is the paper the reviewer will
  hold up. Running it is not optional.
- **Contract:** embeddings from **LightGCN and biVAE**; Gowalla, Amazon-Books; **MovieLens 10M/20M** for
  the real-user experiments. **NDCG@10**, MAP, MRR, HR@k, AUC@k. Budget: 50 burn-in + T=5 rounds × m=10
  = 100 items. Reported **NDCG@10 0.3616 on Amazon-Books (LightGCN)** vs c-DPP 0.3575; abstract claims
  **3–10% relative NDCG@10** over prior elicitation methods **[S]**.
- **Baselines it uses:** DPP, c-DPP, RMV (=RMVA), DPE, PEO, **DRE**.
- **Code:** **none found** (checked arXiv, OpenReview, GitHub search). Reimplementation needed.
- **Fit:** mechanically excellent for us — it operates over a **frozen embedding space**, which is exactly
  our certified-tower setup, and it already contains an answerability term. **What must change:** the
  100-item budget must come down to k∈{2..16} (the burn-in alone is 50 — at our budgets the burn-in *is*
  the whole interview, which is itself a finding worth reporting); needs an LP/Chebyshev-center solver
  (scipy `linprog` is enough); halfspace cuts assume a *pairwise/ordinal* answer, so our binary
  seen+like answers must be encoded as halfspaces (defensible: like ⇒ ⟨u,v_i⟩ ≥ τ).
- **Verdict: MUST-INCLUDE (the strongest ADAPTIVE bar, and the one that dates Golbandi).**

### 1.3 Pairwise & Attribute-Aware Decision-Tree Preference Elicitation, arXiv:2510.27342 (2025) **[V]**
- **Mechanism.** Direct Golbandi successor: ternary like/dislike/unknown tree, extended two ways —
  (a) nodes may ask about **item ATTRIBUTES (genres)** as well as items; (b) nodes present **item PAIRS**
  instead of single items. **Dynamic tree updating**: "the decision tree is retrained using the updated
  known ratings K" each RE iteration.
- **Static or adaptive:** **ADAPTIVE** (tree branching + per-iteration retraining).
- **Contract:** **Yahoo! Music**, filtered to 1,701 users / 4,793 artists / 218 genres. Cold-start = 90%
  of users. **20 elicitation iterations.** **Metric = RMSE only** ("it is vastly employed in the
  literature"). Baselines: popularity, entropy, **HELF**, **IGCN**.
- **Code:** https://github.com/camilleecu/master_thesis/tree/pairwise
- **Fit:** *mechanism* ports cleanly and is directly relevant to our **concept channel** — this is the
  only recent paper that puts attribute questions and item questions in the same tree, i.e. the item-vs-concept
  comparison we run. *Numbers* do not port at all: Yahoo!Music + RMSE + 20 iterations is a different
  contract from ML-25M + NDCG@10 + k≤16. **Never place its RMSE next to our NDCG.**
- **Verdict: NICE — high value for the concept arm, cite prominently as "the 2025 restatement of Golbandi",
  port the mechanism, never the number.** (Also: its baseline set is *literally* our strategy bank —
  popularity/entropy/HELF/IGCN — which is direct external validation that our arm choice is current.)

### 1.4 fMF — Functional Matrix Factorization, Zhou/Yang/Zha, SIGIR 2011 **[K, already indexed]**
`zhou2011functional`. Decision-tree interview whose **leaves are latent vectors**; the cold user is folded
into item-factor space at the leaf. Adaptive. No public code; ~200 lines to reimplement (the tree fit is
alternating: fit leaf profiles by ridge, choose splits by squared loss).
**Verdict: NICE.** It is the "Golbandi with a real latent recommender" control, and it makes the Act 1
argument robust to the objection that Golbandi's leaf model is a shrunk mean. Cheap. Do it if time allows.

### 1.5 Rashid HELF / entropy0 / **IGCN** (SIGKDD Expl. 2008) **[V, already summarised]**
Already wired as strategy arms. **IGCN is the family's only classic *adaptive* member** —
`IG(a_t) = H(C) − Σ_r (|C^r|/|C|) H(C^r)` over **user-cluster labels**, recomputed as answers arrive
(`splitter_and_refusal_novelty.md`). It is still the personalised baseline the 2025 tree paper compares to.
**Verdict: already in — but make sure IGCN is run in its genuinely adaptive form (re-clustering per answer),
not as a precomputed global order.** If our current `entropy0/helf` arms are static orders, the paper is
currently missing its cheapest adaptive baseline.

### 1.6 RBMF / local-RBMF / RMVA / Rashid popularity — static seed prior art
`liu2011wisdom`, `shi2017local`, `fonarev2016rectangular`. All static. RBMF already wired. RMVA is
~50 lines (rectangular maximal-volume on the item factor matrix) and is a **named DRE/PERE baseline** —
adding it costs little and closes the "you skipped the max-volume line" objection.
**Verdict: RMVA = NICE (cheap, named by two of our must-includes).**

---

## 2. The interactive-RL family (different information contract)

### 2.1 NICF — Neural Interactive Collaborative Filtering, Zou et al., SIGIR 2020 **[V, full text extracted]**
- **Mechanism.** Deep Q-learning with **two stacked self-attentive blocks** over the (item,rating)
  interaction history; the exploration policy IS the network; "satisfied recommendations from exploratory
  suggestions" act as exploration bonuses. Meta-learning framing, no explicit state estimate.
- **Adaptive:** yes, strongly (that is the whole paper).
- **Contract — and this is the disqualifier as-published:** datasets **MovieLens 1M (6,040 u / 3,706 i),
  EachMovie (1,623 u / 61,265 i), Netflix**. Metrics are **Cumulative Precision@T, Cumulative Recall@T,
  Cumulative α-NDCG@T** over **T=40 trial recommendations**, with `b_t = 1 if r_{u,i_t} ≥ 4`. That is
  *"how many good items did the agent hand the user while probing"* — **there is no held-out ranking
  after the interview at all.** Reported: cumulative precision@40 relative improvement over best baseline
  **9.43% / 4.59% / 6.65%** on the three datasets; e.g. ML-1M cum-precision@40 NICF **21.5875** vs MeLU
  19.7273 vs GLM-UCB 19.0932.
- **Its baselines:** Random, Pop, MF, BPR, MLP, ε-greedy, **TS**, **GLM-UCB**, **MeLU**.
- **Code:** https://github.com/zoulixin93/NICF and https://github.com/guyulongcs/SIGIR2020_NICF —
  **no README, no requirements, TensorFlow-1-era** (`agents/`, `envs/`, `functionApproximation/`,
  `launch.py`, `model_train.sh`). Wiring cost is real.
- **Fit:** the **policy** is portable — re-purpose the Q-network as a *question selector*, then score the
  post-interview ranking with our tower and our NDCG@10. That is an honest port and worth doing.
  The **published numbers are not orderable against ours.** Say so explicitly in the paper if NICF is cited.
- **Verdict: NICE-if-ported / UNFIT-as-published.** Flag the contract mismatch in text. If effort is tight,
  a **GLM-UCB / Thompson bandit over our frozen tower** captures ~the same adaptive mechanism at 5% of the cost.

### 2.2 Interactive CF bandits — Zhao et al. CIKM'13 (ICF), GLM-UCB, Thompson sampling **[K]**
`zhao2013interactive` already indexed. Adaptive by construction, and **trivial to wire over our frozen
tower**: a Gaussian/linear posterior over `u` given item factors is *already implemented* as
`belief_mf.py`; UCB and Thompson are two extra selection rules on top of that posterior.
**Verdict: MUST-INCLUDE.** This is the cheapest strong adaptive baseline we have access to, it is the
comparison set NICF itself uses, and it makes Act 1's "adaptive doesn't pay" claim vastly harder to
dismiss as "you only tried a 2011 tree."

### 2.3 RLT4Rec, arXiv:2412.07403 (Dec 2024) **[A]**
Sequential transformer RL taking the user's (item,rating) history and outputting the next item to present;
"no need to input a state observation or estimate"; handles new and established users in one framework and
"automatically balances exploration … with exploitation". Same contract problem as NICF (the metric is the
quality of the presented sequence). Datasets/metrics **unverified** — the PDF would not extract.
**Verdict: NICE (cite as the 2024 continuation of the NICF line); do not run unless NICF is run.**

### 2.4 EAR / SCPR / UNICORN / ConTS / CRM / HutCRS / DAHCR — the attribute-CRS line **[V, settled earlier]**
Settled in `answerability_in_elicitation.md` and `paperA_recommender_landscape.md`. Three independent
disqualifiers, all already documented:
1. **Omniscient simulator.** CRM's AMT workers were "presented with the facet values of the target
   restaurant on the side, so that she can correctly answer the questions"; ConTS §5.2.2 — "the user will
   give positive feedback only to item v and attributes P_v". PEPPER (`kim2024targetfree`) shows this is a
   *recognised* validity failure (0.86 recall on known targets vs 0.12 residual). Our behavioural firewall
   forbids it.
2. **Metric.** Success Rate@turn and Average Turns, over a *target item*. No held-out NDCG.
3. **Closed attribute schema**, fixed per dataset.
**Verdict: UNFIT.** Cite them as the adjacent line, state the three reasons in one sentence, and do **not**
put their numbers in any table with ours. This is exactly the "numbers from different information
contracts are not ordered" case.

---

## 3. Belief / VoI elicitation

### 3.1 Bıyık, Yao, Chow, Haig, Hsu, Ghavamzadeh, Boutilier 2023 — Preference Elicitation with Soft Attributes **[A]**
`biyik2023soft`, arXiv:2311.02085. Multivariate-Gaussian belief over a user embedding; queries with
**both items and soft attributes** (CAV directions); EVOI-style selection; models **noise in the CAV
semantics**. Demonstrated on simulated data **and MovieLens 20M**.
**Code:** only the *dataset* is public (https://github.com/google-research-datasets/soft-attributes —
~250k pairwise preferences, 60 soft attributes); **no method code found.**
**Fit:** the closest published thing to our item-vs-concept adaptive belief. **What must change:** its
CAVs are probe-trained per attribute over a closed vocabulary and its encoder is co-trained (fails our R1);
its query type is comparative, not seen/like. Reimplementation is a real project.
**Verdict: NICE — cite as the nearest adaptive item+concept belief; run only if the concept arm needs a
named rival.** (It is already flagged in `paperA_recommender_landscape.md` as the central Paper-A threat.)

### 3.2 PEBOL — Austin et al., RecSys 2024 **[V, already indexed]**
Per-item Beta posteriors, Thompson/UCB acquisition, **LLM extracts aspects + NLI entailment**.
**Two hard blockers for Act 1:** (a) preferences are **100% LLM-simulated** — it is the exact circularity
our behavioural firewall exists to prevent; (b) running it needs LLM API calls, which the project rules
forbid without explicit author approval.
**Verdict: UNFIT for Act 1's primary table.** Its *acquisition* (Beta-Thompson over items) is runnable
LLM-free as an ablation and is worth one row labelled "PEBOL acquisition, behavioural answers"; the
published MRR@10 0.27 vs 0.17 must never sit beside our NDCG.

### 3.3 EDDI (`ma2019eddi`) — already implemented (`src/baselines/eddi_pvae.py`)
Permutation-invariant partial-VAE + myopic info-gain acquisition. Adaptive. **Already ours.** It is the
principled info-gain arm and should be in the Act 1 table by name — a reviewer who knows the AL literature
will look for it before Golbandi.

### 3.4 Adaptive Querying with AI Persona Priors, arXiv:2605.00696 (2026) **[V]**
Not recsys — survey/psychometrics. Finite dictionary of LLM-generated **personas**; closed-form posterior
over persona membership; **greedy one-step-lookahead minimising expected posterior Shannon entropy**.
WorldValuesBench (91 q / 88,459 users), Twin-2K-500. Metrics log loss, Brier, ordinal MSE.
Code: https://github.com/yw3453/adaptive-query-ai-persona-priors.
**Verdict: UNFIT as a baseline (wrong domain, wrong metric), CITE as the 2026 anchor** that greedy
one-step-lookahead posterior-entropy acquisition is still the state of the art criterion in 2026 —
useful for arguing our greedy prefix-optimal selector is not a weak choice.

---

## 4. Meta-learning cold start — adjacent, not elicitation

- **MeLU (KDD'19)** — MAML for cold start, with an **"evidence candidate selection"** module that picks
  distinguishing items. Selection is **global and one-shot** (chosen from the meta-model, not branched
  per user) ⇒ **static**. Public code https://github.com/hoyeoplee/MeLU. It is a *recommender*, not a
  question policy — and NICF already reports it as a baseline (ML-1M cum-precision@40 19.7273).
- **MAMO, TaNP (`lin2021tanp`), CDRNP** — same story: adaptive *recommenders* over a given support set,
  not adaptive *selectors*. **TaNP is already wired.**
**Verdict: the meta-learning line is not an Act 1 elicitation baseline. Cite once, in the recommender
column, and move on.** Confusing "adapts to the user" with "chooses the next question adaptively" is the
error to avoid; only MeLU's evidence selection touches selection at all, and it is static.

---

## 5. Methodological cautions found in this pass

- **Selection bias in preference elicitation** — Gupta et al., arXiv:2405.00554 (2024) **[V]**.
  Applies IPS to elicitation: which items you *ask about* is itself a biased sample, and ignoring it
  degrades downstream recommendation. Yahoo!R3, Coat, synthetic; MAE/MSE/NDCG@3; simulator public at
  https://github.com/shashankg7/Bias-Preference-Elicitation. **Directly relevant to Act 1**: if the
  adaptive arm asks a systematically different item distribution than the static arm, part of any measured
  difference is a *selection* artefact, not an *information* gain. Cite it, and note our fixed
  strategy-independent candidate pool (already in `run_interview_table.py`) is the guard.
- **Lin et al., WWW 2023 "Greedy NDCG Attribute Selector"** (from `rank_targeted_acquisition.md`) remains
  the only paper naming NDCG as the *selection target*, and does so without a posterior. Unchanged.
- **Recency of our strategy bank is externally validated**: the 2025 arXiv tree paper's baseline set is
  popularity / entropy / HELF / IGCN — i.e. Rashid 2008. The field itself still uses these. That is a
  usable sentence: *"the 2025 literature still benchmarks against Rashid 2008; we do too, and add DRE,
  PERE, EDDI and a bandit."*

---

## 6. Ranked verdict table

Ranking is by *how much running it protects Act 1 from a 2026 reviewer*, not by paper quality.

| # | Method | Year/venue | Mechanism (1 line) | Adaptive? | Code | Contract fit | Effort | Verdict |
|---|---|---|---|---|---|---|---|---|
| 1 | **DRE** | WWW 2020 | Gumbel-Softmax categorical seed distributions + reconstruction decoder, trained end-to-end | **STATIC** (by design) | ✔ PyTorch, `WonbinKweon/DRE_WWW2020` | ML-20M, NDCG@10, seed-budget framing, baselines RBMF/RMVA. Retrain per k∈{2..16} (k=50 published) | **M** (2–3 d) | **MUST-INCLUDE** — the strongest static bar |
| 2 | **PERE** | UAI 2024 | User = *region* in a frozen embedding space; DPP burn-in then halfspace-cut info-gain queries; Chebyshev-center rec | **ADAPTIVE** | ✘ none found | ML-10M/20M, NDCG@10, frozen embeddings (= our tower). Budget 100 → must compress to ≤16; needs LP solver; answers→halfspaces | **M-H** (4–6 d) | **MUST-INCLUDE** — the strongest adaptive bar; the paper that dates Golbandi |
| 3 | **ICF bandit (GLM-UCB / Thompson)** | CIKM 2013 + std | UCB / Thompson selection on a Gaussian posterior over the user vector, over frozen item factors | **ADAPTIVE** | ✔ trivial; `belief_mf.py` already has the posterior | Perfect — same tower, same answers, same metric | **L** (<1 d) | **MUST-INCLUDE** — cheapest strong adaptive arm; also NICF's own baseline set |
| 4 | **EDDI** | ICML 2019 | Partial-VAE set encoder + myopic information-gain acquisition | **ADAPTIVE** | ✔ ours (`eddi_pvae.py`) | Already wired | **none** | **MUST-INCLUDE** — the principled info-gain arm; already paid for |
| 5 | **Golbandi ternary tree** | WSDM 2011 | Squared-loss ternary like/dislike/unknown user-splitting tree | **ADAPTIVE** | ✔ ours (`golbandi_leaf.py`) | Already wired; k≤8 depth cap | **none** | **MUST-INCLUDE** — keep, but never alone |
| 6 | **Pairwise/attribute decision trees** | arXiv 2510.27342, 2025 | Golbandi tree + genre(attribute) questions + item pairs + per-iteration retraining | **ADAPTIVE** | ✔ `camilleecu/master_thesis@pairwise` | ✘ numbers (Yahoo!Music, **RMSE**, 20 iters). ✔ mechanism, and it is the only recent item-vs-attribute tree | **M** (2 d, mechanism only) | **NICE-HIGH** — port the mechanism for the concept arm; cite as "Golbandi restated in 2025"; **never table its RMSE** |
| 7 | **IGCN (adaptive form)** | SIGKDD Expl. 2008 | Information gain over user-CLUSTER labels, recomputed as answers arrive | **ADAPTIVE** | ✔ trivial | Perfect; still the personalised baseline the 2025 tree paper uses | **L** (<1 d) | **MUST-INCLUDE if not already adaptive** — check our arm isn't a precomputed static order |
| 8 | **RMVA** | ICDM 2016 | Rectangular maximal-volume seed selection in the item factor space | **STATIC** | ✘ (≈50 lines) | Perfect; named baseline of both DRE and PERE | **L** (<1 d) | **NICE** — cheap, closes the max-volume objection |
| 9 | **fMF** | SIGIR 2011 | Interview tree whose leaves are latent vectors, cold user folded at the leaf | **ADAPTIVE** | ✘ (≈200 lines) | Perfect | **M** (1–2 d) | **NICE** — the "Golbandi with a real latent model" control |
| 10 | **NICF** | SIGIR 2020 | Self-attentive deep Q-network over (item,rating) history; exploration bonus for satisfying probes | **ADAPTIVE** | ✔ but bare (`zoulixin93/NICF`, TF1, no README) | ✘ **as published**: metric is Cumulative Precision/Recall/α-NDCG@T=40 *during* interaction — no post-interview held-out ranking. ✔ if policy is ported and re-scored | **H** (1–2 wk) | **NICE-if-ported / UNFIT-as-published** — flag the contract; ICF bandit is the cheap substitute |
| 11 | **Bıyık soft attributes** | 2023 (arXiv) | Gaussian belief over a user embedding, queried with items AND soft-attribute CAV directions, EVOI | **ADAPTIVE** | ✘ method (dataset only) | Mechanism = our item-vs-concept belief. Co-trained encoder (fails R1), comparative queries, closed CAV vocab | **H** | **NICE** — cite as the nearest adaptive item+concept rival; run only if the concept arm needs a named opponent |
| 12 | **MeLU / MAMO / TaNP / CDRNP** | KDD'19 – WWW'21 | Meta-learned per-user adaptation over a given support set; MeLU adds *global* evidence-candidate selection | **STATIC** selection (adaptive *recommender*) | ✔ MeLU `hoyeoplee/MeLU`; TaNP ours | Recommender column, not selector column | TaNP **none** | **NICE (recommender row only)** — do not present as an elicitation baseline |
| 13 | **RLT4Rec** | arXiv 2412.07403, 2024 | Transformer RL: (item,rating) history → next item, no explicit state | **ADAPTIVE** | ✘/unverified | Same cumulative-reward contract as NICF; datasets/metrics unverified | **H** | **NICE (cite only)** |
| 14 | **PEBOL** | RecSys 2024 | Per-item Beta posteriors + Thompson/UCB, LLM aspect extraction + NLI | **ADAPTIVE** | ✔ | ✘ **LLM-simulated preferences** (circularity) + LLM API forbidden without approval + MRR@10 on small catalogs | **M** (acquisition only) | **UNFIT for primary** — one LLM-free "PEBOL acquisition, behavioural answers" ablation row is legitimate |
| 15 | **EAR / SCPR / UNICORN / ConTS / CRM** | WSDM'20 – TOIS'21 | Attribute-level RL/bandit CRS over a closed schema | **ADAPTIVE** | ✔ all | ✘✘✘ omniscient target-item simulator; Success-Rate@turn / Avg-Turns, no held-out NDCG; closed attribute schema | — | **UNFIT** — cite the line in one sentence with the three reasons; **never table their numbers** |
| 16 | **Adaptive Querying w/ AI Persona Priors** | arXiv 2605.00696, 2026 | Posterior over LLM-persona membership; greedy 1-step-lookahead entropy minimisation | **ADAPTIVE** | ✔ `yw3453/adaptive-query-ai-persona-priors` | ✘ surveys/psychometrics, log-loss/Brier | — | **UNFIT (cite)** — 2026 evidence that greedy 1-step-lookahead entropy is still SOTA criterion |
| 17 | **Selection bias in PE** | arXiv 2405.00554, 2024 | IPS debiasing of which items you ask about | n/a | ✔ `shashankg7/Bias-Preference-Elicitation` | Methodological, not a baseline | — | **CITE** — the confound Act 1's fixed candidate pool already guards |

---

## 7. Shortlist — the 4 that make Act 1 defensible, + 1 cheap insurance

Act 1's claim is *"an adaptive interview buys no uplift over the strongest static one at k≤16."*
A 2026 reviewer kills that claim in one of three ways: **(a) your adaptive method is 2011**;
**(b) your static bar is a heuristic, not the learned SOTA**; **(c) you never tried the obvious
Bayesian/bandit thing.** The shortlist closes exactly those three doors.

1. **DRE (WWW'20)** — closes door (b). Without it, "strongest static" means popularity/RBMF and the claim
   is unfalsifiable in the reviewer's eyes. It is also the only static method whose seed set is *learned
   end-to-end*, so it is the honest opponent: if adaptivity loses to a *learned* static set, that is a
   result; losing to popularity is a harness smell.
2. **PERE (UAI'24)** — closes door (a), decisively. It is the recent, peer-reviewed paper whose entire
   thesis is that personalised adaptive elicitation beats fixed seeds, on MovieLens, on NDCG@10, over a
   frozen embedding space. If Act 1 does not engage PERE, Act 1 has not engaged the field. No public code
   is a cost, not an excuse — the method is a DPP plus halfspace cuts plus an LP.
3. **ICF bandit — GLM-UCB and Thompson over our tower posterior** — closes door (c) for ~a day of work.
   These are NICF's own baselines, they need no new model, and they turn "we tried one adaptive method"
   into "we tried the whole adaptive family."
4. **EDDI (already ours)** — the principled myopic information-gain arm, the one an active-learning
   reviewer looks for first. Zero marginal cost. Report it by name, not as "info-gain".

**Plus insurance (cheap, high leverage):** **IGCN in genuinely adaptive form** (re-cluster per answer) —
verify our current arm is not a precomputed static order. If it is, we currently have *no* cheap adaptive
classic and the 2025 tree paper's own baseline table would embarrass us. Half a day.

**Explicitly park:** NICF, RLT4Rec, Bıyık, MeLU, the whole EAR/SCPR/UNICORN line, PEBOL-with-LLM.
Each is either a different information contract, a forbidden simulator, or a fortnight of work whose
mechanism is already represented by items 2–4.

## 8. Contract-violation flags (do not order these numbers against ours)

| Family | Their metric | Why not orderable |
|---|---|---|
| NICF, RLT4Rec, ICF-as-published | Cumulative Precision/Recall/α-NDCG@T over the interaction | Scores the *probes themselves* as recommendations. We score a *subsequent* held-out ranking. Different quantity. |
| EAR / SCPR / UNICORN / ConTS / CRM | Success Rate@turn, Average Turns | Defined against a known target item, with a simulator that can see the target's attributes. |
| PEBOL | MRR@10 / MAP@10 on ~1–5k-item catalogs, LLM-simulated users | Catalog scale, answer source, and metric all differ. |
| 2025 pairwise/attribute trees | RMSE on Yahoo! Music | Rating prediction, not ranking; different dataset and 20-iteration budget. |
| PERE, DRE | NDCG@10 on ML-10M/20M, k=50–100 | **Closest** — same metric family, but different dataset version, different split, and a budget 3–6× ours. Re-run on our split; do not cite their published numbers as a bar. |

## 8b. HARNESS AUDIT — Act 1 currently has ZERO adaptive question selectors (checked 2026-08-04)

Read before commissioning any of the above. Verified in code, not assumed:

- `src/baselines/interview_strategies.py` — `ARMS = list(LIT_RANK)` =
  `{helf, entropy0, popularity, pure_entropy, random_bank, random_catalog}`. `build_orders()` returns
  **`np.argsort` over a global per-item score**; `ask()` then walks *the same order for every user*
  (the only per-user variation is the RNG for the two random arms). **All six arms are STATIC global orders.**
- `src/baselines/golbandi_leaf.py` — `scores(asked_per_user, answered_per_user, ...)`; it *consumes*
  whatever an arm asked and groups train users by the resulting like/dislike/unknown code
  (`_codes`, `_group_profiles`, `MAX_DEPTH = 8`). **It is a RECOMMENDER, not a question selector.**
  Golbandi's actual adaptive contribution — the squared-loss tree that picks the *next probe item
  conditional on the previous answer* — **is not implemented anywhere in the interview harness.**

**Consequence:** as of today the interview table cannot measure "adaptive vs static" at all — every arm
in it is static, and the one method named as the adaptive anchor is present only in its leaf-model half.
So the first job is not adding a 2024 baseline; it is **making at least one arm genuinely adaptive**.
Cheapest order of work: (1) Golbandi's *tree* as a selector (we already have the leaf model and the
codes — it is the split search that is missing); (2) UCB/Thompson selection on the `belief_mf.py`
posterior; (3) IGCN (re-cluster per answer). Only then do DRE and PERE mean anything as bars.

## 9. Verification debts
- **NICF datasets/metrics/baselines: [V]** (full text extracted). **NICF repo contents: [A]** (file list only,
  no README seen — the TF-version claim is inferred from the 2020 date and file layout, **unverified**).
- **RLT4Rec: [A] abstract only.** Datasets, metrics, budget, code all unverified — PDF would not extract.
- **PERE: [V] for method/datasets/metrics/budget** via the arXiv HTML; the "3–10% relative NDCG@10"
  headline is **[S]** (search summary) — confirm against the paper before quoting.
- **Bıyık 2023: [A]** — method section still unfetched (a standing debt from `text_to_cf_projection.md`).
- **DRE: [V]** via arXiv HTML of the re-upload; confirm the WWW'20 camera-ready matches before citing numbers.
- **2510.27342 is arXiv-only (a master's thesis derivative)** — check for a venue before citing it as
  "the 2025 literature"; if unpublished, cite it as an arXiv preprint and lean on PERE for recency.
