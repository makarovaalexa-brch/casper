# Modern CRS positioning — deep/RL and LLM conversational recommenders vs our contract

*Commissioned 2026-08-12 by the author's question, verbatim: "the whole new deep neural, llm family etc,
are we still missing the strongest baselines? are we comparing wrong? … but should we borrow ideas at
least? i am a bit scared to be alone in this?"*

Feeds **PAPER_B_TODO 1.21** and **PAPER_B_PLAN §3 / B6**. Chapter-ready prose lives in
`docs/results/POSITION_VS_MODERN_CRS.md` (this file is the evidence; that file is the argument, and it
carries the borrow list for the parametric policy, TODO 1.16).

**Method, per HARD RULE 4.** The project's own store was read end-to-end *before* any external search:
`bandit_rl_crs_baselines.md`, `answerability_in_elicitation.md`, `simulator_circularity_conv_rec.md`,
`llm_elicitation_and_metalearning.md`, `ndcg_optimising_elicitation.md`, `rank_targeted_acquisition.md`,
`active_learning_elicitation.md`, `hu_yu_2013_interview_process_learning.md`, plus
`docs/results/BASELINE_OBJECTIVE_AUDIT.md` and `docs/results/HU_YU_2013_REPLICATION_AND_ABLATION.md`.
**Most of Part 1 was already answered there from primary text and is not re-derived here** — it is
consolidated into one five-column audit, with the provenance file named on every row. External passes
were commissioned only for the four things the store explicitly left open (Hu & Yu's citation graph,
ELO-DCG and the adjacent acquisition literatures, the 2025–26 successor sweep, and the baseline/
comparator question (d), which nothing in the store had asked of the deep-RL line directly).

**Verification levels** carry over from the store: **[V]** primary full text read, **[S]** abstract /
index / third-party, **[A]** absence at this search depth (not proof of absence), **NOT READ**.
Nothing is quoted unless its source file marks it [V].

**Our cell, stated once so it can be tested against every row.** A cold user; a budget of k ≤ 16 closed
questions drawn from a fixed bank of items *or* concepts; answers taken from the user's own real
behaviour (held-out ratings for items, behavioural attribute affinity for concepts) with unanswerable
questions **burning** the turn; then the interaction ENDS and a frozen certified recommender ranks the
**full unsampled 18,359-item catalogue**; we report **full and long-tail NDCG@10 on held-out targets**
(Liang strong-generalisation split, 10k validation / 10k test users); and the questions are selected by
**directly maximising that same measured, truncated, position-discounted, held-out quantity**, with no
per-user ground truth read at selection time.

---

## 1. The audit — (a) what the selector optimises · (b) what is reported · (c) where the signal comes from · (d) is there a metric-optimised static comparator · (e) full catalogue or sampled

### 1.1 The deep/RL attribute-asking line (CRM → EAR → SCPR → UNICORN → ConTS)

| System | (a) selector optimises | (b) reported metric | (c) training/eval signal | (d) metric-optimised static comparator? | (e) catalogue | verdict |
|---|---|---|---|---|---|---|
| **CRM** — Sun & Zhang, SIGIR 2018 | REINFORCE policy over {ask facet, recommend}; reward = success minus turn penalty | Average reward, average turns, success rate, "wrong quit rate", "low rank rate" — **no ranking metric** | Scripted agenda over the **target item's facet values**; AMT workers are "presented with the facet values of the target restaurant on the side, so that she can correctly answer the questions" [V]. Belief tracker and FM are separate modules | **No** — but see the MaxEnt@K near-miss below | Yelp; belief-constrained candidate list, item count not recovered | **circular-by-construction (label-based)** |
| **EAR** — Lei et al., WSDM 2020 | Policy network (REINFORCE) over attributes + one recommend action; rewards `r_suc, r_ask, r_quit, r_prev` | SR@t, AT; AUC offline only. **The string "hDCG" does not occur in EAR** [V] | "we treat the v as the ground truth item to seek for and its attributes P_v as the **oracle set** of attributes preferred by the user in this session" [V] | **No.** Full baseline list is exactly three — Max Entropy, Abs Greedy, CRM — preceded by "As our multi-round conversational recommendation scenario is new, there are few suitable baselines" [V] | candidate set filtered by confirmed attributes; SR is hit-of-one-target | **circular-by-construction (label-based)** |
| **SCPR** — Lei et al., **KDD 2020** (⚠ not SIGIR — DOI 10.1145/3394486.3403258 is the KDD '20 proceedings [V]) | DQN over binary {ask, recommend}; attribute chosen by weighted-entropy score on the graph path | SR@t, AT | "when a system asks for an attribute, he will only confirm he likes it if this attribute is included by item v" [V]; "does not care" is **collapsed into reject** | **No.** Baselines verbatim: Max Entropy, Abs Greedy, CRM, EAR | **hard-pruned each turn** — "keeping the items that directly link to all attributes in the updated [set]" [V] | **circular-by-construction (label-based)** |
| **UNICORN** — Deng et al., SIGIR 2021 | Graph DQN over a **unified** action space (items ∪ attributes); Q-value max | SR@t, AT, **hDCG@(T,K)** — introduced here, not in EAR. "Since there is only one target item for each session in MCR, we simply use hDCG@(T,K) for evaluation" [V] | "We regard the item as the ground-truth target item and treat its attribute set as the **oracle set** of attributes preferred by the user" [V]; no limitations section | **No.** Baselines: Max Entropy, Abs Greedy, CRM, EAR, SCPR | **explicitly pruned** — "Instead of enumerating the whole candidate item and attribute set, we adopt preference-based item selection and weighted entropy-based attribute selection strategies to only consider potentially important actions" [V]; top-K_v items, top-K_p attributes. Items: LastFM 7,432 / Yelp 70,311 / E-Commerce 29,428 | **circular-by-construction + negative-sampled ACTION space** |
| **ConTS** — Li et al., TOIS 2021 | Contextual Thompson Sampling over a **unified** item ∪ attribute arm space | SR@t, AT only (also SR@7, SR@10) — **no ranking metric** | "the user will give positive feedback only to item v and attributes P_v in current session" [V] | **No.** Baselines: Abs Greedy, EAR, ConUCB + three self-ablations | genuinely cold-start (user-disjoint 70/15/15), arm pool is the item set, but no full-catalogue ranking metric | **circular-by-construction; closest in SETTING, wrong ruler** |
| **HyperCRS** — Zheng et al., Findings of ACL **2025** | Multi-grained hypergraph policy, DQN Bellman | SR@t, AT, hDCG@(15,10) — **the MCR ruler is unchanged in 2025** | inherits the MCR simulator lineage [S on this point] | **No.** Baselines verbatim: "AbsGreedy, MaxEntropy, CRM, EAR, SCPR, UNICORN, MCMIPL, MHCPL, HutCRS" | pruned to top-`K_p` items and attributes by entropy score [V] | **genuine rival on the TASK, not on our cell** |

**★ The one near-miss on (d), and it is worth knowing about: CRM itself.** Sun & Zhang run a family of
*fixed-length* rule variants and then **select among them by the reported metric** [V]: *"We further
explore its variations that always ask exactly K slots before making a recommendation. We name the
baseline 'MaxEnt Full' for the one that asking all the facets and 'MaxEnt@K' for ones that asking
exactly K facets."* … *"We further run several MaxEnt@K methods … and find that the reward increases as
[K] increases … in our data set, it seems [MaxEnt Full] is the best choice, thus we use it as the
baseline for follow up experiments."* **That is metric-driven selection over a BUDGET, not over an
ORDER** — the facet at each turn is still chosen by max-entropy on the live belief state, so the order
remains adaptive and stimulus-dependent. It is the closest anything in this literature comes, and it is
not the object we are talking about. *(The K values themselves were rendered as MathML and did not
survive text extraction; they are not guessed.)*

**★ Correction to our own prior framing: hDCG is a UNICORN-onward metric, not a family-wide one.**
CRM, EAR, SCPR and ConTS report **no rank-sensitive metric whatsoever** — only success rate and turn
counts. This makes the metric separation stronger, and any chapter sentence attributing hDCG to the base
set must be changed.

**On (c), a distinction we must state precisely and not overstate.** The circularity in this line is
**label-based, not representational**: the simulator answers from the *target item's ground-truth
attribute labels*, not from the recommender's own embedding. That is a different (and in one sense
milder) defect than the geometric self-play our own firewall was built against
(`simulator_circularity_conv_rec.md`). What disqualifies it for us is (i) that **every question is
answerable by construction**, so unanswerability has exactly zero cost and the quantity this chapter
prices cannot exist in their protocol, and (ii) that the candidate pool is **seeded by an attribute of
the target** (`p₀`), so the reported top-k is a re-rank inside a pool the answer key created. Both are
verbatim-sourced in `answerability_in_elicitation.md` and `bandit_rl_crs_baselines.md` §4.
The honest sentence is *"their evaluation prices a different quantity"*, never *"their evaluation is
invalid"* — and their own community published the leakage critique (PEPPER, `kim2024targetfree`;
Interplay, arXiv:2603.18573), which is the fairer thing to cite.

### 1.2 The 2022–2026 successors — the policy changed, the contract never did

Audited in `bandit_rl_crs_baselines.md` §4 (13 systems, per-row [V]/[S] marks there). Consolidated
finding, unchanged by this pass: **not one successor drops the known target item, and not one reports a
post-interview held-out ranking metric over a full catalogue.** The innovation budget went entirely into
the policy — inverse RL (CRIF), multi-interest dueling Q (MCMIPL), interest trees with a "don't know"
class (HutCRS), hierarchical options (DAHCR, CoCHPL), bi-level intrinsic reward (CRSIRL), MCTS lookahead
(SAPIENT), tri-phase simulator separation (TCRS), vague-preference soft filtering (VPMCR).

Three facts from that audit that belong in the chapter because they are about the *protocol*, not about
any one system:
- **The benchmark is saturated.** CoCHPL SR@15 = 0.991 on LastFM at AT 3.75; DAHCR 0.992 on Yelp at
  AT 5.16 [V]. A protocol at its ceiling cannot adjudicate anything.
- **Nobody reports the candidate-pool size.** MCMIPL and VPMCR both report **zero** candidate-set-size
  statistics [V] — VPMCR despite over-filtering being its entire motivation. So the ratio to our
  18,430-item catalogue cannot even be estimated. That absence is the finding.
- **MINICORN uses no RL at all** — a harmonic mean of two uncertainty heuristics — and reports beating
  SCPR by "67.40% and 13.39% in recommendation success rate and conversation turns" on LastFM [V].
  If a two-heuristic rule beats the RL line by that margin, the RL apparatus is not doing the work.
  This is the citation to reach for when asked why we did not build an RL policy.

### 1.3 The conversational-bandit line (the family whose policies DO port)

ConUCB (WWW'20) → ConLinUCB-BS/-MCR (AAAI'23) → Hier-UCB (2022) → CLiSK/CLiME (KDD'25) → FedConPE (2024).
**These are the strongest thing in the modern literature by our own criteria and they are already in our
battery** (`conucb`, `conlinucb_mcr`, `conlinucb_bs`, `conlinucb_bs_static`, `clime` — rows 6–10 of
`BASELINE_OBJECTIVE_AUDIT.md`). (a) key-term posterior width / estimation-error reduction / low-eigen-
direction alignment; (b) **cumulative regret**; (c) **no target item and no simulator** — reward is
replayed real feedback, so this family passes our firewall as-is; (d) no; (e) truncated arm sets
(CLiME's "ML-25M" is 5,000 arms and 200 users at T=6,000 rounds [V]).
**Verdict: genuine rivals on the ACTION SPACE, different task on the METRIC.** We ran them; they lose
(`ACT2_CONCEPT_BANK_PV.md`), and ConUCB's `b(t)=5⌊log t⌋` makes two of them UNINFORMATIVE at k≤2.

### 1.4 The LLM elicitation line (PEBOL and relatives)

Audited from primary text in `llm_elicitation_and_metalearning.md` §1. The disqualifier is the
measurement contract, not the LLM.

| System | (a) selector optimises | (b) reported | (c) answer signal | (d) | (e) catalogue |
|---|---|---|---|---|---|
| **PEBOL** (RecSys 2024) | Beta-posterior bandit index (TS / UCB / max-variance / greedy) — **LLM is not in the selection loop** | **MRR@10** over 10 turns | GPT-3.5 simulates a user who "likes a single item" and is given that item's description [V]; the NLI scorer entails against the **same string** | No | **100 items** [V], capped so a baseline fits a context window |
| **CLARINET** (arXiv:2405.15784) | greedy **rank improvement of the true item** under the retriever | top-1 accuracy, MRR, retrieval success | GPT-3 simulator "has access to the true item" [V] | No | 1,877 documents |
| **OPEN** (arXiv:2403.05534) | EIG over a Bradley–Terry particle filter — LLM only verbalises | TIDA (time-integrated Δ accuracy) | ~40 real users per condition | No | 30 NYT articles |
| **BED-LLM** (ICLR 2026) | argmax expected information gain over LLM-proposed candidates | **mean rating of 10 films** | separate LLM answerer given ground-truth profile | No | not stated [V-negative] |
| **GATE** (ICLR 2025) | none — pure prompting | p(correct), area under Δp-vs-time | **388 real humans**, preregistered | No | no catalogue |
| **COPE** (arXiv:2607.06765) | picks one of **3 strategy types**, not a question | Recall@1/10/50 vs a dialogue target | INSPIRED transcripts | No | 1,783 items |
| **Montazeralghaem et al. 2025** | free-form generation, diffusion-style question ordering | **BLEU / ROUGE** vs a reconstructed profile | LLM simulator given the profile [V] | No | not reported |

**The structural point for the chapter:** *PEBOL on 100 items, CLARINET on 1,877, OPEN on 30, GATE on no
catalogue at all.* There is no shared protocol in this family, which is itself the evidence that a
reviewer cannot demand "the standard benchmark" — there isn't one. PEBOL's own authors decline ReDial
because it is "made up of pre-recorded conversation histories and cannot be used to evaluate active,
cold-start NL-PE systems" [V]; that sentence is ours to quote.

**And the CLARINET result is a gift, not a threat:** its selection objective — maximise the rank of the
true item under the current retriever — *is* our greedy prefix-optimal objective in different notation,
and it **beats EIG and KL acquisition by 17% relative** [V]. That is independent published pre-support
for `prefix-optimal-greedy-is-the-right-object`. Its disqualifier is the target-biased simulator, not
the criterion.

### 1.5 The four genuine near-neighbours — the only rows that can threaten the claim

These are not "the deep/LLM family"; they are the actual rivals, and they were already found. Repeated
here so the positioning section has them in one place.

| Work | Greedy/tree? | Selection objective | Held-out at selection? | Full catalogue? | Misses on |
|---|---|---|---|---|---|
| **Hu & Yu, RecSys 2013** (EIP) | ternary interview tree | Taylor-approximated **training** loss over the tree fit; framework admits a pairwise rank loss, experiments ran **logistic** | ✗ (training loss) | ✗ (10 sampled negatives) | truncation, discount, held-out, full catalogue; **static comparator is top-k popularity (FixQ)** |
| **UpsRec, WWW 2023** | greedy, per question | **NDCG** — named explicitly | ✗ — **reads the user's validation labels at ask time** [V] | ✗ | honesty; per-user, no fixed order; assumed-YES only, no expectation over answers |
| **Greedy SLIM, arXiv 2024** | greedy forward, prefix-optimal | **SLIM elastic-net squared loss** | ✓ | ✓ **ML-25M, no sampling, Cremonesi 33% tail** | the metric (NDCG@10 reported, never optimised); static questionnaire |
| **SPQ, RecSys 2018** | greedy, Sherman–Morrison | **A-optimality**, derived from a *flat pairwise* loss | ✓ | ✗ | the discount and the truncation — the flat loss collapses to a trace criterion, ours does not |

**Greedy SLIM is the single most contract-matching paper in the literature** and it is a CoRR-only
preprint (DBLP-verified, not peer-reviewed). It is a *static* bar with the wrong objective, on our
dataset, with our tail definition — i.e. it strengthens rather than threatens Act 1, and its finding
that **QBandit's NDCG *decreases* past k≈10 on both axes** is an independent external replication of the
bank-quality effect we measure.

---

## 2. ★ The comparator question (d) — stated plainly

**★★ THE EXCEPTION, AND IT IS IN OUR OWN LINEAGE: Golbandi WSDM 2011 vs GreedyExtend (CIKM 2010).**
GreedyExtend builds a static seed set by incrementally adding the item that most reduces error *with the
recommender refit on seed-restricted data* — `i = argmin_{i'∉k} F([k, i'])`, `F` = **RMSE** of a
factorised item-item model. That is a wrapper around a held-out evaluation and it optimises the reported
metric **of its era**. The WSDM'11 adaptive tree beat it: depth-6 tree reaches RMSE 0.97172 after **six**
questions, GreedyExtend needs **>20** [V, via WSDM'11 Fig. 3; GreedyExtend's own full text is [A/S],
closed everywhere]. **So "nobody ever compared adaptivity against a metric-optimised static order" is
FALSE, and the claim must be scoped to the period after the reported metric moved to top-k ranking.**
This is the plan's §0 historical frame, now with the audit behind it, and it is evidence *for* our
diagnosis: in 2011 the split criterion and the reported metric were the same quantity; today they are not.

**Modulo that exception, across every system audited above, in five separate families, we found no paper
that compares its adaptive selector against a static question order which was itself built by maximising
the metric the paper reports.** The deep-RL line was checked baseline-table by baseline-table (§1.1) and
the answer is a clean no, with CRM's MaxEnt@K budget sweep as the only near-miss.

The static arms that exist in this literature are: **top-k popularity** (Hu & Yu's FixQ [V]); **max
entropy / entropy0 / HELF / popularity×entropy** (Rashid's family, all rating-era heuristics [V]);
**Abs Greedy and Max Entropy** in the EAR/SCPR/UNICORN line; **a barycentric spanner** (ConLinUCB-BS —
a principled *static* concept order, but chosen for spanning, not for the reported metric); **DRE's
reconstruction-MSE seed set**; and **Greedy SLIM's QGSLIM**, which is the closest — a static order built
by greedy forward selection against a *SLIM elastic-net squared loss*, then reported on NDCG@10.
QGSLIM is the nearest miss in the entire literature and it misses on exactly one axis: the objective.

That is the empty cell, and it is why our Act 1 result is not pre-empted. **It is also why the result is
fragile in the right way**: the moment someone builds the metric-optimised static order, published
adaptive selectors stop beating it — which is what our own A9 pass measured when the rebuilt comparator
retracted Golbandi's tail-at-eight win (`BATTERY_TABLE_POLICYB_ROWS.md`).

**The field named this gap itself, twice, from inside elicitation** — and this is the framing to lead
with, because it converts our finding from a novelty claim into a replication:
- **Koren (Golbandi et al., WSDM 2011), future work:** "In the future we would like to experiment with
  other cost functions, especially ones related to the quality of the top-K item ranking." **[S — still
  owed a check against the PDF before it is printed.]**
- **Sepliarskaia et al. (RecSys 2018) §5.2.2 [V]:** "if the method directly optimizes the loss function,
  then it works better than others … SPQ and FG achieve better performance than PWDT, despite the fact
  that they are static, while PWDT is dynamic. A possible explanation is that PWDT optimizes a function
  that is different from the loss function, namely weighted generalized variance."

---

## 3. Verdicts, per family

| Family | Verdict |
|---|---|
| CRM / EAR / SCPR / UNICORN / ConTS and all 2022–26 successors | **circular-by-construction + different task.** Cite, do not run. |
| Conversational bandits (ConUCB → ConLinUCB → CLiSK/CLiME) | **Genuine rival on the action space; different task on the metric.** Already run under our contract. |
| ICF bandit lineage (Zhao CIKM'13 → LinUCB/TS/GLM-UCB → PTS, factorUCB, IGCF) | **Different task** (cumulative reward over the recommended sequence). Policies ported; already run. |
| LLM elicitation (PEBOL, CLARINET, OPEN, BED-LLM, GATE, COPE, Montazeralghaem) | **Different task**, and the answerer is LLM-simulated or target-conditioned in every case but GATE/OPEN. Cite-only; PEBOL's LLM-free acquisition is the one legitimate ablation row. |
| Meta-learning cold start (MeLU, MAMO, TaNP, CDRNP) | **Category error** — answer *consumers*, not selectors. Exactly one (MeLU) selects, statically, and independently reproduced our answerability collapse in humans. |
| The four near-neighbours (Hu & Yu, UpsRec, Greedy SLIM, SPQ) | **The real comparators.** Hu & Yu is being replicated and ablated (B1/1.12); Greedy SLIM is a static bar we should port (B-workstream); UpsRec belongs in the battery as a **declared privileged-oracle ceiling**; SPQ is already in the battery as `aopt`. |

---

## 4. Hu & Yu (2013) — the citation picture

*(Part 2(i). External pass, 2026-08-12, index APIs only — WebSearch budget was exhausted for the whole
session, so there is no free-text web coverage and no usable Google Scholar. See §7.)*

**Counts, each with the index named and the date attached (all 2026-08-12):**

| Index | Count | Notes |
|---|---|---|
| **OpenAlex** (`W1979356225`) | **7** | answered; full citing list retrieved |
| **Semantic Scholar** (`DOI:10.1145/2507157.2507205`, CorpusId 17539172) | **7** | answered; **identical set of 7**; later calls 429'd |
| **Crossref** | **3** (`is-referenced-by-count`) | answered; a subset artefact of publisher-deposited reference lists |
| **DBLP** | n/a | key `conf/recsys/HuY13` confirmed; DBLP publishes no citation counts |
| Google Scholar | **UNVERIFIED** | fetches returned mismatched records; one attempt reported "Cited by 14" but attached it to *Joint User Modeling*. **Do not use a Scholar number from this run.** |

**The seven citing works (OpenAlex ∪ S2 = the same seven):**

| # | Work | Year / venue | (a) builds on the interview-tree-with-a-general-loss? | (b) instantiates the **pairwise rank loss**? | (c) any ranking quantity in the split/selection rule? | verification |
|---|---|---|---|---|---|---|
| 1 | Elahi, Ricci & Rubens, *A survey of active learning in CF recommender systems* | 2016, Computer Science Review | No — survey | No | No — NDCG/MAP discussed only as *evaluation*, and lamented as unavailable | **VERIFIED** (accessible pre-print; ⚠ the pre-print contains no visible mention of Hu & Yu — the citation is asserted by both indexes and presumably lives in the published version) |
| 2 | Sanner, Balog, Radlinski, Wedin & Dixon, *LLMs are Competitive Near Cold-start Recommenders* | 2023, RecSys | No | No | No | **VERIFIED** (abstract) |
| 3 | Silva (Veras) et al., *A literature review of recommender systems in the television domain* | 2015, ESWA | almost certainly a passing mention | No | No | **UNVERIFIED — title only**; abstract elided, paywalled |
| 4 | **Sepliarskaia, Kiseleva, Radlinski & de Rijke, *Preference elicitation as an optimization problem* (SPQ)** | 2018, RecSys | No — static questionnaire, explicitly *not* a tree | **No** | **PARTIAL** — see below | **VERIFIED** (full text) |
| 5 | Cao & Yu, *Joint User Modeling across Aligned Heterogeneous Sites* | 2016, RecSys | No | No | No | **UNVERIFIED** — title + one-line snippet (Yong Yu self-citation) |
| 6 | Yang, Mo, Shen & Chen, *Cold-Start Developer Recommendation in Software Crowdsourcing* | 2017, SEKE | No | No | No | **UNVERIFIED** — title + snippet |
| 7 | Sepliarskaia, PhD thesis (contains the SPQ work as a chapter) | 2019, UvA | as #4 | No | as #4 | verified by inheritance from #4 |

**On #4, the only interesting case.** SPQ's selection criterion minimises expected squared error of
*predicted pairwise preferences*, reformulated as the trace objective `argmin_B tr[(V_B V_Bᵀ + εE)⁻¹]`.
So a pairwise-preference quantity *is* in the selection rule — but it is a squared error on pairwise
comparisons collapsing to an A/D-optimal design criterion, **not a rank loss, and neither truncated nor
discounted**. Precision@10 appears only in evaluation. Notably, **SPQ does not cite Hu & Yu for the
general-loss idea** — its surrounding citation context is Golbandi-style decision-tree questionnaires.

### ★ Answer to Part 2(i): has anyone followed up the pairwise-rank-loss variant?

**No. Nobody.** The evidence base for that negative, stated so it can be judged:
- Indexes queried: OpenAlex (answered, 7), Semantic Scholar (answered, 7), Crossref (count only, 3),
  DBLP (no citation data). **Google Scholar and free-text web search were unusable this session.**
- Citing works identified: **7 of 7**. Verified from abstract or full text: **4** (#1, #2, #4, #7 by
  inheritance). Unverified beyond title/snippet: **3** (#3, #5, #6) — none of which is topically an
  interview-tree method paper.
- **Unreconciled tail:** if Google Scholar's cluster really holds ~14, then up to ~7 citers are
  unaccounted for. That number is itself UNVERIFIED and must not be quoted; but the honest statement is
  that the negative rests on the OpenAlex/S2 intersection and could in principle miss a Scholar-only tail.
- **Structural observation worth keeping:** Hu & Yu's general-loss framing left almost no trace —
  **seven citers in thirteen years, not one of them an interview-tree method paper**, and even the
  nearest neighbour (SPQ) reaches the lineage through Golbandi rather than through Hu & Yu's Eq. (1).
  That is the same *dormancy* pattern measured for UpsRec (published, cited 20 times, adopted by nobody)
  and for the DRE/PERE/Greedy-SLIM line. **A published idea in this subfield does not get picked up.**

### The wider Golbandi-lineage sweep (Part 2(i), second half)

Golbandi et al. WSDM 2011 (`W2013221919`) has **cited_by_count = 169** on OpenAlex (2026-08-12).
Filtered full-text search across its citers: `interview` → 16, `ndcg` → 4, `ranking` → 3, `pairwise` → 4.
Candidates opened:
- **Sun et al., WSDM 2013** — split criterion is **squared prediction error** over the three children;
  evaluation RMSE. No ranking metric anywhere. **VERIFIED** (full text).
- **Dureddy & Kaden, arXiv 2018, "Handling Cold-Start CF with Reinforcement Learning"** — DQN reward
  `r_k = 1/RMSE(·)` with discounting `γ^(k−t)` over interview steps. **The discount is temporal, not a
  rank discount.** No NDCG. **VERIFIED** (ar5iv).
- **Volokha & Derevitskii, Procedia CS 2021** — criterion is **content-attribute coverage**, chosen
  because the usual signals are missing. **VERIFIED** (abstract).
- Iovine et al. 2021 (×2), Elahi et al. TIST 2013, Valdiviezo Diaz 2019 thesis — matched "ndcg" among
  Golbandi's citers, but consistent with NDCG as an *evaluation* metric. **UNVERIFIED, not opened.**

**Coverage caveat that binds the negative:** OpenAlex abstract coverage over 169 citers is thin (only 3
matched "ranking"), so these filtered searches are a screen, not an exhaustive scan. The conclusion is
**strong for Hu & Yu** (7/7 citers identified, 4 verified) and **merely well-supported for the wider
Golbandi lineage**.

## 5. Has anyone maximised a truncated, discounted, held-out ranking metric to pick questions? (2013–2026)

*(Part 2(ii), including the adjacent literatures: computerised adaptive testing, Bayesian experimental
design with a ranking utility, learning-to-rank with feature acquisition, active learning for ranking.)*

**What the store already settled, so it is not re-searched:**
- **Computerised adaptive testing** is covered in `active_learning_elicitation.md` §7. CAT's selection
  rules are Fisher information, a-stratification and exposure control; **none carries a ranking utility**,
  and CAT's response model *assumes universal response*, so it has no term for a burned question. The one
  genuinely empty CAT-side cell is that IRT's non-ignorable-missingness models (Holman & Glas 2005) have
  never been put inside an item-selection rule [S/A]. The nearest CAT↔recsys bridge is a 3-dimensional
  MIRT on ML-1M (*Behaviormetrika* 2024) [S].
- **Bayesian experimental design / knowledge gradient / top-k best-arm identification** are covered in
  `rank_targeted_acquisition.md`. **None adds a positional discount** — that is the empty cell.
- **Rarity counts, reproducible, with the instrument attached** (arXiv abstract index, all years):
  `abs:"NDCG" AND abs:"preference elicitation"` → **1** hit (UpsRec); `abs:"expected NDCG"` → **0**;
  `"expected NDCG" "question selection"` → **0**. Report these as *arXiv-abstract-index counts*, never as
  "the literature contains none" — they miss ACM/IEEE-only venues by construction, which is exactly how
  FacT (SIGIR'19) and FacT-CRS (CIKM'22) were missed on an earlier pass.

### 5.1 ★★ FALSIFICATION 1 — a truncated, position-discounted ranking metric HAS been used as an acquisition function. Twice, in IR, in 2010 and 2012.

**This is the most important result of the pass and it forces a wording change.**

**ELO-DCG — Long, Chapelle, Zhang, Chang, Zheng & Tseng, SIGIR 2010. [V, full text read this session**
(obtained from Yi Chang's homepage, `yichang-cs.com/yahoo/sigir10_activelearning.pdf`; ACM/IEEE/
Unpaywall/S2 all confirm CLOSED)**]. This closes the standing `[HIGH]` debt in
`ndcg_optimising_elicitation.md` §3.7/§6.1.**

- Loss, Eq. 6: `ℓ(π, Y) = max_{π'} DCG(π', Y) − DCG(π, Y)` with
  `DCG(π,Y) = Σ_i (2^{y_i} − 1)/log₂(1 + π(i))`.
- Query-level acquisition, Eq. 7:
  `EL(q) = ∫_Y max_π DCG(π,Y)P(Y|X_q,D)dY − max_π ∫_Y DCG(π,Y)P(Y|X_q,D)dY`, rule = **argmax**.
  Posterior approximated by a bootstrap ensemble of N=8 GBDT learners.
- **Truncated? NO.** The DCG inside the criterion sums over *all* documents of the query; truncation
  appears only in the *evaluation* ("we select to use DCG-k … we select k as 10").
- **Position-discounted? YES**, via `1/log₂(1+π(i))`.
- **Regret or improvement? REGRET, unambiguously.** It is the gap between the DCG achievable if `Y` were
  known and the DCG of the best single ranking under the posterior — an EVPI-shaped uncertainty measure.
  **It never models the effect of acquiring the label**: no retraining, no counterfactual, no expectation
  over the post-acquisition model.
- Selects **queries and (query, document) pairs to LABEL for training a ranker**. Never a question to a
  user. Proprietary web-search data, ~10k queries / ~500k documents, GBDT, DCG-10.
- ⭐ **And it states our thesis, in IR, in 2010** [V]: *"ELO-DCG optimizes the expected DCG loss that is
  directly related to the objective function DCG-10 used to evaluate the ranking quality; on the other
  hand, VR reduces the score variance that is not directly related to the objective function. In fact, VR
  performs even worse than random document selection when the size of the selected example is small."*
  **That is a second published instance of the Sepliarskaia finding — optimise the metric, not the
  surrogate — from a different field, eight years earlier. Cite the two together.**
- TKDE 2015 extension [V-partial]: same framework, adds a "Balanced ELO" reweighting for skewed grades;
  no truncation added, still regret.

**★ Bilgic & Bennett, SIGIR 2012, "Active query selection for learning rankers" — THE REAL FALSIFIER.
[V, full text read this session, MSR-hosted PDF].**

- Eq. 1: `max_{q∈C} E_{P(Y|X_q,D)} [ max_π M(π(X_q), y) − M(R(X_q), y) ]` — the ELO regret shape, **with
  `M` instantiated as DCG@10 or NDCG@10.** Their arms are `rndQ-rndD`, `rndQ-topD`, `dcgELOq` ("select
  queries according to ELO with DCG@10 as the selection measure") and `ndcgELOq`.
- **So truncation + positional discount inside a selection rule is OCCUPIED, in SIGIR, in 2012.**
- It remains expected **regret** under a predictive label distribution, it selects **queries whose
  documents get labelled** to train a ranker, and **no user is asked anything**.
- Their own finding, worth knowing: *"using DCG loss (as done in [4]) leads to better performance whether
  DCG or NDCG is used as the final evaluation"* — NDCG's normalisation pulls selection toward queries
  with few relevant documents. Yahoo! LETOR challenge data; SVMRank and Additive Regression.

**⚠ BINDING CONSEQUENCE FOR THE WRITE-UP.** Any sentence of the form *"a truncated, position-discounted
ranking metric has never been used as an acquisition function"* is **FALSE** and must be deleted.
Every such claim must be scoped to **elicitation / questions asked of a user**. The four deltas that
survive against Bilgic & Bennett, and they are clean:
1. **improvement, not regret** — we score what the metric *becomes after the answer arrives*; they score
   how much the current model could lose;
2. **held-out cohort-mean target metric**, not posterior regret on the candidate itself;
3. **a question to a user**, not a training label for a ranker;
4. **full 18k-item catalogue ranking**, not a per-query candidate list.

### 5.2 ★ FALSIFICATION 2 — greedy forward selection with an NDCG wrapper criterion is an ordinary technique in learning-to-rank feature selection

**arXiv:2310.13269, "An Exploratory Study on Simulated Annealing for Feature Selection in
Learning-to-Rank" [V, full text].** Verbatim: *"for each state, i.e., a subset of features among the
available ones, we train the model learnt with only the features that are included in the feature subset
at hand. We then measure the performance of the learnt model using NDCG and MAP."* LambdaMART on
LETOR 3.0 and MSLR-WEB10K; the paper names **forward feature selection** as a standard wrapper search
alongside simulated annealing and local beam search.

**So "greedily grow a subset, refit the ranker, score by NDCG, keep the best" is a published, ordinary
technique** — the same machine as our greedy with "feature" swapped for "question". It is not an
acquisition (nothing unknown is being acquired), there is no user, no answer model, no cohort, and no
prefix-optimality property over a question sequence. **But it removes "greedy forward selection with an
NDCG wrapper criterion" from any list of novel components** — the same demotion Greedy SLIM and
GreedyExtend already force. **Claim the conjunction, never the technique.**
*(UNVERIFIED and wanted: Geng, Liu, Qin & Li, "Feature Selection for Ranking", SIGIR 2007 — reportedly
scores each feature by its own single-feature MAP/NDCG. ACM DL 403, DuckDuckGo CAPTCHA, S2 429.)*

### 5.3 The other adjacent literatures — checked, and empty at this depth

- **Donmez & Carbonell, ICML 2008** [V, full text]. Objective = *estimated loss differential*, where the
  loss is **the ranker's own training objective** (RankSVM hinge, RankBoost exponential), expectation
  over the candidate's unknown label. **Not truncated, not discounted, pairwise.** NDCG@10 and MAP are
  evaluation only. Structurally Golbandi-with-a-pairwise-loss over training documents.
- **Carterette, Allan & Sitaraman, SIGIR 2006** [V, full text]. AP rewritten as a quadratic form in
  Boolean relevance indicators; greedy judgment selection maximising `max{p_i w_i^R, (1−p_i) w_i^N}`.
  **AP is untruncated** (rank-weighted through `1/r` coefficients). The goal is 95% confidence about
  *which of two systems is better* with minimal judging — **evaluation-side, metric-estimation, not
  acquisition for a model.** Not our cell.
- **Computerised adaptive testing / IRT.** Already settled in `active_learning_elicitation.md` §7 (Fisher
  information, a-stratification, exposure control; universal-response assumption; no ranking utility).
  Re-checked this pass: `abs:"computerized adaptive testing" AND abs:"recommendation"` = 4 hits,
  `abs:"adaptive testing" AND abs:"item selection" AND abs:"ranking"` = 1, none with a ranking utility.
  GMOCAT (arXiv:2310.07477) [S] is multi-objective over *quality, diversity and novelty* — no truncated
  or discounted ranking term. **⚠ This negative is [A] and the instrument is weak**: CAT lives in
  *Psychometrika*, *Applied Psychological Measurement* and *JEM*, which arXiv does not index, and no
  free-text web search was available. Do not upgrade it to [V].
- **Bayesian experimental design / BO / ranking-and-selection with a ranking utility.** arXiv abstract
  index, all years: `"experimental design" AND "NDCG"` = **0**; `"acquisition function" AND "NDCG"` =
  **0**; `"expected improvement" AND "NDCG"` = **0**; `"best arm identification" AND "NDCG"` = **0**;
  `"NDCG" AND "preference elicitation"` = **1** (UpsRec, replicating the earlier count).
  **Top-k Ranking Bayesian Optimization (Nguyen, Tay, Low & Jaillet, AAAI 2021, arXiv:2012.10688)** is a
  **false friend** [S, abstract]: its acquisition is **multinomial predictive entropy search**, and
  "top-k ranking" describes the *observation model*, not the utility. **No positional discount.** This
  downgrades the §3.7 lead from "name suggests a match" to "not a match".
  **Could not falsify** "no ranking-and-selection or BO acquisition carries a positional discount" — [A].
- **Smooth NDCG surrogates (SoftRank / ApproxNDCG / LambdaRank / NeuralNDCG) as an ACQUISITION.** None
  found; the three arXiv hits are all training losses. `"acquisition function" AND "NDCG"` = 0.
  **The λ-gradient-as-acquisition move remains unoccupied at this search depth [A].**
  Related lead resolved: **"Active Learners as Efficient PRP Rerankers", arXiv:2605.14236** [S] — NDCG@10
  is the *evaluation*, the abstract states no acquisition function, and **the "+9.7 NDCG@10" figure a
  previous pass recorded does not appear in the abstract and could not be verified.**

### 5.4 The two ranking-split trees, now read

- **★ The FacT — Tao, Jia, Wang & Wang, SIGIR 2019 (arXiv:1906.02037). [V-fetch — read through a
  summariser over ar5iv HTML, NOT raw text; the arXiv PDF would not extract. Re-read the raw source
  before printing any equation, λ value or dataset size.]** **Its split criterion DOES contain a ranking
  term.** Eq. 6 selects the predicate minimising, over the three children (left / right / **unknown**),
  `L(u,V,O) − λ_b Σ B(u,V,D_i^o) + λ_u(‖u_l‖₂+‖u_r‖₂+‖u_e‖₂)`, where `B` is the **pairwise BPR ranking
  loss**. **And it is explicitly framed as an interview:** *"the rules learnt in the user tree naturally
  serve as a set of interview questions to solicit user preferences when a new user comes to the
  system"*, with nodes asking *"How do you like this [feature]?"*. Amazon (6,285 u / 12,626 i) and Yelp
  (10,719 / 10,410); **NDCG@10/20/50/100 reported, never optimised.**
  ⚠ **The `[MED]` debt is NOT closed: λ_b is described only as "a trade-off parameter" with no numeric
  value in the method section.** If λ_b is tiny the ranking-split claim is nominal.
  **This is the earliest confirmed ranking-split interview tree in the family and it upgrades
  `ndcg_optimising_elicitation.md` §2.2 from UNVERIFIED LEAD to a confirmed instance.** It still misses
  our cell: an undiscounted, untruncated pairwise surrogate on **training** observations.
- **FacT-CRS — Ahsan-Ul-Haque & Wang, CIKM 2022 (arXiv:2208.14614). [V-fetch, same caveat.]** Eq. 4:
  `f_z = argmin_{f_l} [ L_CE(s_pos,·) + L_CE(s_neg,·) + λ_BPR·(BPR terms) + λ_s(‖s_pos‖₂+‖s_neg‖₂) ]`,
  computed on **observed training interactions `R_z`, not held-out targets; not truncated, not
  discounted.** LastFM (1,801 u / 7,432 i / 72,040 int / 33 attrs), BookRec (1,891 / 4,343 / 75,640 / 35),
  MovieLens (3,000 / 5,974 / 120,000 / 35). Metrics **SR@T and Average Turns** — the target-item CRS
  protocol. **Confirms the previously-[S] description; the earlier draft's "+L2" is real, its
  "node-ranking expression" was not seen.** Not our cell.

### 5.5 ★ The 2025–2026 falsification sweep — nothing occupies the cell

Instrument, stated so the negative is judgeable. **WebSearch was unavailable for the entire session
(budget 200/200 exhausted before the first query), so everything below is structured-API evidence** —
stronger per source, weaker in recall over venue-only publications. All accessed 2026-08-12.

| Query | API | Hits | Result |
|---|---|---|---|
| `abs:"preference elicitation" AND abs:"NDCG"` | arXiv | **1** | UpsRec (2023), already audited |
| `cold-start interview recommendation` | DBLP | **0** | confirms the Golbandi "interview" vocabulary is dead in indexed titles (partly a query-form artefact — DBLP ANDs prefixes) |
| `abs:"elicitation" AND cat:cs.IR` | arXiv | 142 | 39 triaged from 2025–26 |
| `abs:"cold-start" AND abs:"interactive" AND cat:cs.IR` | arXiv | 301 | first 35 (2026-02→08) triaged: **all item-side cold start** — multimodal embeddings, graphs, diffusion, generative retrieval, semantic IDs. **Zero user-interview papers.** |
| `abs:"cold-start" AND abs:"elicitation"` | arXiv | 30 | 3 relevant, all rejected below |
| `abs:"clarifying question" AND cat:cs.IR` | arXiv | 40 | conversational *search*; no catalogue ranking on held-out items |
| `"experimental design"/"acquisition function"/"expected improvement"/"best arm identification"` + `"NDCG"` | arXiv | **0 each** | see §5.3 |
| forward citations of EAR / SCPR / UNICORN from 2025-01 | OpenAlex | — | see §5.6 |
| `/citations` on arXiv:2405.00981 (PEBOL) | Semantic Scholar | **26** (25 rows returned) | see §5.7 |

**Per-candidate rejections (the near ones):**
- **Gharahighehi, Nakano, Yang, Cu & Vens, "Pairwise and Attribute-Aware Decision Tree-Based Preference
  Elicitation for Cold-Start Recommendation", arXiv:2510.27342 (CoRR only)** — **the nearest miss in the
  whole sweep.** It is Golbandi-lineage (tree, lover/hater/unknown), it asks **attributes (genres) as
  well as items**, and its answers come from real logs ("Users' historical logs were used to simulate the
  rating elicitation scenario"). **It fails because the reported metric is RMSE** — node splits pick "the
  item with the lowest prediction error E(j)", squared error across the three groups. There is no ranking
  metric at all, so the full-catalogue question is moot. Yahoo! Music, 1,701 users / 4,793 artists / 218
  genres after filtering; 20 iterations; 90% of users held out. **[V-fetch, via summariser over arXiv
  HTML — re-read `arxiv.org/html/2510.27342v1` directly before citing its dataset sizes.]**
- **"Adaptive Querying with AI Persona Priors", arXiv:2605.00696, ICML 2026** [V abstract] — targets
  "responses to **held-out items**" under tight query budgets, which is tantalising, but there is no
  catalogue and no ranking (evaluation is probabilistic prediction on synthetic data and
  WorldValuesBench), and the prior is a "finite dictionary of **AI personas**… produced by a large
  language model" — **an LLM-generated response model, i.e. exactly our firewall violation.**
- **D2D, arXiv:2606.24194** [V abstract] — objective is "accuracy in finding target products": a
  pre-named target. Recommendation is interleaved, not terminal.
- **Entropy Guided Diversification…, AAAI Spring Symposia 2026** [S] — "maintains **filtered product
  candidates**" (negative-sampling-equivalent) and uses "simulated users derived from actual reviews".
- **Feedback-Driven Gradual Discovery, RecSys 2025** [S] — outcomes are user-study constructs
  (engagement, genre affinity, sense of control); no catalogue ranking.
- **Ontology-Guided Hierarchical Preference Elicitation, ICAART 2026** [S] — evaluated on "stability and
  interpretability" over product taxonomies.
- **"Investigating Preference Elicitation Strategies in CRS", UMAP 2026** (10.1145/3774935.3803059)
  [V via S2 abstract; ACM DL 403] — a doctoral-consortium-style **proposal**: literature review,
  behavioural archetypes, a planned user study on trust/usability. No metric, no catalogue.
- Boroomand & Wright (arXiv:2508.14911) — already on file; pairwise query space, no positional discount,
  infeasible at 18k items.

**Sweep verdict: zero papers from 2025–2026 satisfy all four criteria** (interaction ends then ranks /
full unsampled catalogue / held-out items with no pre-named target / answers from real user data).
**The structural finding is that the field has bifurcated:** elicitation work went LLM-in-the-loop and
evaluates with target-item accuracy or user studies; cold-start work went item-side and evaluates with
offline ranking. **Nobody is doing budgeted user-side questioning evaluated by held-out full-catalogue
NDCG.** The two cells to put in the paper are the arXiv `"preference elicitation" AND "NDCG"` count
(**1**, from 2023, already audited) and the DBLP `cold-start interview recommendation` count (**0**),
each with the instrument named.

### 5.6 The deep/RL CRS successor sweep, 2025–2026 — what is new

Forward citations of EAR/SCPR/UNICORN from 2025-01-01 via OpenAlex [V for existence/venue/year], plus
arXiv and DBLP. **Almost all 2025–26 CRS output has migrated to the ReDial/LLM-dialogue branch**, which
is a different task. Genuinely new in or adjacent to the attribute-asking MCR cell:

- **HyperCRS** (Zheng et al., **Findings of ACL 2025**) [V, from PDF]. Multi-grained hypergraph policy,
  DQN. **Metrics SR@t, AT, hDCG@(15,10) — the MCR ruler is unchanged in 2025.** Baselines verbatim:
  "AbsGreedy, MaxEntropy, CRM, EAR, SCPR, UNICORN, MCMIPL, MHCPL, HutCRS" — **no metric-optimised static
  order.** Action space pruned to top-`K_p` items and attributes by entropy score.
- **CCR — "A Causal-Based Attribute Selection Strategy for CRS", TKDE 2025** [S, abstract only, IEEE
  paywall, no OA copy per OpenAlex]. **The most methodologically distinct new acquisition rule found:**
  select "the attribute with the highest **Average Treatment Effect** per round", with
  stratification/matching to de-confound. (b)–(e) **unverified.** Worth one citation and a chase.
- **GRSU** (SIGIR 2025, arXiv:2504.20458) [S] — builds its simulator by instruction-tuning an LLM and
  uses "token probability as reward". The simulator is trained independently of the recommender, so this
  is still **not** the shared-representation case.
- **CT-CRS** (arXiv:2508.13209) — **WITHDRAWN by the authors 2025-09-09** ("The tables … need to be
  modified"). Search engines still surface its numbers. **NEVER CITE.**
- Lower-tier, all [S], internals unverified: diversity-enhanced MARL CRS (KAIS 2025); Q&A-driven CRS with
  multiple interest modelling (Neurocomputing 2025); maximum-entropy preference clarification (Data
  Intelligence 2025); multi-view hypergraph disentanglement (JIIS 2026); "Why Not Together?" (WSDM 2025);
  DareCRS (ICPADS 2025); "Action First" (SIGIR 2025).
- **Not our line, do not confuse:** HyFairCRS / "Why Multi-Interest Fairness Matters" (arXiv:2507.02000)
  [V] is ReDial/TG-ReDial/OpenDialKG/DuRecDial with Recall/MRR/NDCG + BLEU — **no simulator, no RL** —
  despite sharing a first author with HyperCRS.

### 5.7 PEBOL's successors — the family moved away from us, not toward us

Semantic Scholar reports **26 citations** for PEBOL (arXiv:2405.00981) as of 2026-08-12 (25 rows
returned by `/citations`; one record unaccounted for, presumed index lag). **OpenAlex reports only 5**
and is a strict subset — **use S2 as the index of record for this paper.**

Of the 2025–2026 citers, **not one reports NDCG over a full catalogue**, and none has an acquisition
rule closer to our contract than PEBOL's own. The nearest are: *Feedback-Driven Gradual Discovery*
(RecSys 2025 — Bayesian active learning, LLM-free, but a user study with no ranking metric);
*Ontology-Guided Hierarchical Preference Elicitation* (ICAART 2026 — Bayesian belief tracking, evaluated
on stability/interpretability); and Pep (already audited). The rest are out of domain (auctions,
chemistry BO, physics instrument design, RLHF reward shaping, digital twins) or are surveys.
**The successors moved *toward* LLM-in-the-loop and *away* from ranking evaluation.**

⚠ **Bibliographic correction found on this pass: our store's DOI for PEBOL was wrong.**
`10.1145/3640457.3688151` resolves to *"Utilizing Non-click Samples via Semi-supervised Learning for
Conversion Rate Prediction"* [V, OpenAlex W4403219679]. **PEBOL's DOI is `10.1145/3640457.3688142`**
[V, S2 record for arXiv:2405.00981; DBLP key `conf/recsys/AustinKTS24`].

---

*(Part 3 — what to borrow for the parametric policy, TODO 1.16 — lives in
`docs/results/POSITION_VS_MODERN_CRS.md` Part II, ranked, with a mandatory "what must NOT come with it"
clause per item and a table of rejected mechanisms with reasons.)*

## 6. Corrections this pass forces on our own records

1. **★ "A truncated, position-discounted ranking metric has never been an acquisition function" is
   FALSE.** Bilgic & Bennett (SIGIR 2012) instantiate `M = DCG@10 / NDCG@10` inside an ELO-style
   acquisition. Scope every such sentence to *elicitation / questions asked of a user*, and lead with the
   improvement-vs-regret distinction instead. (§5.1)
2. **★ "Greedy forward selection with an NDCG wrapper criterion" is an ordinary LTR feature-selection
   technique** (arXiv:2310.13269). Claim the conjunction, never the technique. (§5.2)
3. **★ SCPR is KDD 2020, not SIGIR 2020.** DOI 10.1145/3394486.3403258 is the KDD '20 proceedings
   [V via OpenAlex]. **Fix the bibliography** — `lei2020interactive` currently carries the wrong venue.
4. **★ hDCG is a UNICORN-onward metric, not a family-wide one.** The string "hDCG" **does not appear in
   EAR** [V, ar5iv full text]; it is introduced in UNICORN §5.2.3. **CRM, EAR, SCPR and ConTS report no
   rank-sensitive metric whatsoever** — only Success Rate and Average Turns. This makes the separation
   argument stronger, not weaker, and any chapter sentence attributing hDCG to the base set must change.
5. **The circularity in this line is LABEL-BASED, not representational.** No FM/GNN embedding sits inside
   any of the five simulators. **"The simulator is built from the recommender's embeddings" would be
   false for all five and must not be written.** (§1.1)
6. **PEBOL's DOI in our store is wrong** — see §5.7.
7. **The FacT (SIGIR 2019) is a confirmed ranking-split interview tree** — upgrade
   `ndcg_optimising_elicitation.md` §2.2 from UNVERIFIED LEAD, and add it to the list of things we must
   concede rather than claim. (§5.4)
8. **Top-k Ranking BO (AAAI 2021) is a false friend** — MPES, information-theoretic, no positional
   discount. Downgrade the §3.7 lead. (§5.3)
9. **ELO-DCG's `[HIGH]` debt is CLOSED.** Its integrand is expected DCG **regret**, exactly as
   `ndcg_optimising_elicitation.md` §6 guessed, so the §6.1 safe-phrasing paragraph stands — **but it
   must now also name Bilgic & Bennett**, which §6.1 did not know about.
10. **The "+9.7 NDCG@10" figure for arXiv:2605.14236 does not appear in the abstract and could not be
    verified** — it was already flagged as removed; keep it removed.

---

## 7. What could not be verified, and why

**Access failures (all attempted this pass or a prior one, with the failure mode recorded):**
- **GreedyExtend, CIKM 2010 full text** — ACM 403, S2 `openAccessPdf: CLOSED`. Mechanism reconstructed
  from WSDM'11 §3.1.1 [V], patent US8510248B2 [V] and RMVA's description [V]. **The lead finding of the
  positioning document rests on this reconstruction plus WSDM'11 Fig. 3 [V]. Upgrade via institutional
  access before it is printed as stated.**
- **Koren's WSDM'11 future-work quote** — carried at [S] from an internal raw source; load-bearing.
- **"Towards Efficient Conversational Recommendations: EVOI Meets Bandit Learning", WWW 2025.**
  **Authorship correction: it is NOT Wang et al. — it is Zhuohua Li, Maoli Liu, Xiangxiang Dai &
  John C. S. Lui (CUHK)** [V, OpenAlex + S2; DBLP `conf/www/0001LDL25`], published 2025-04-22,
  **gold OA under CC-BY** — and still unreachable: ACM DL 403'd both landing page and PDF, Unpaywall
  lists only the ACM PDF, no arXiv entry, CUHK pages 404. [S, abstract only] Acquisition is
  "**gradient-based EVOI**, which replaces the complex Bayesian updates in conventional EVOI with
  efficient stochastic gradient descent", plus "smoothed key term contexts", giving **ConTS-EVOI** and
  **ConUCB-EVOI**. **The only quantity in the abstract is regret** ("a √d improvement in their dependence
  on the time horizon T"); datasets unnamed. **Reject on metric, but the rejection rests on an abstract.**
  Cheapest route to close it: institutional proxy or emailing the first author.
- **CCR (TKDE 2025), "Action First" (SIGIR 2025), WSDM 2025 "Why Not Together?", KAIS 2025, Neurocomputing
  2025, UMAP 2026** — IEEE/ACM/Springer/Elsevier 403 to both WebFetch and curl; no OA location.
- **Geng, Liu, Qin & Li, "Feature Selection for Ranking", SIGIR 2007** — ACM 403, DuckDuckGo CAPTCHA,
  S2 429. The key paper for the wrapper/NDCG row.
- **Q&R (KDD 2018)** offline protocol and metrics — PDF would not extract.
- **RLT4Rec** datasets, metrics, session length — HTML 404, PDF would not extract.

**Instrument limits that bind every negative in this file:**
- **WebSearch was unavailable for the entire session (200/200 exhausted before the first query)** across
  all four external passes. There is no free-text web coverage, no author pages, no Google Scholar.
- **Google Scholar could not be used for Hu & Yu** — mismatched records; a "Cited by 14" figure surfaced
  but attached to the wrong paper. **Never quote it.**
- **CAT/IRT** was checked on arXiv only; *Psychometrika*, *APM* and *JEM* are not indexed there. [A].
- **RecSys 2025/2026 proceedings were not enumerated venue-by-venue** — reached only by keyword. A
  "we swept RecSys exhaustively" claim would need a separate proceedings walk.
- **OpenAlex abstract coverage over Golbandi's 169 citers is thin** (3 of 169 matched "ranking"), so that
  screen is not an exhaustive scan.

**Read through a fetch summariser rather than raw text (structure trustworthy, specifics need one direct
re-read before print):** The FacT and FacT-CRS equations and dataset sizes; arXiv:2510.27342's quotes,
dataset sizes and split protocol.

**Unverified specifics deliberately not guessed:** the K values in CRM's MaxEnt@K sweep (MathML, lost on
extraction); `λ_b` in The FacT; HyperCRS's simulator paragraph and per-dataset item counts; the identity
of ICF's `AL` and `Interview` baselines; PEBOL's 26th citer.

**Do-not-cite list (binding):** CT-CRS (arXiv:2508.13209, WITHDRAWN); Pseudo Dyna-Q's DOI (not on arXiv,
DOI 403); any Google Scholar count from this run; the "+9.7 NDCG@10" figure for arXiv:2605.14236.
