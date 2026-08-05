# The rating-elicitation / interview-tree / seed-set family — EXHAUSTIVE pass (2026-08-05)

*Sibling to `adaptive_elicitation_baselines.md` (2026-08-04), which surveyed this family at abstract
depth. This file is PASS 2: origins-to-2026, implementation grade, with the corrections that pass 1 got
wrong. **Read §0 before acting on anything in the 08-04 file.***

Verification tags on every claim: **[V]** = read from full text · **[A]** = abstract / landing page only ·
**[S]** = search-summary or third-party only. Nothing tagged [S] may be presented as verified. Fetch
failures are stated explicitly.

OUR CONTRACT (a baseline fits or is excluded with a stated reason): ML-25M, binary implicit (ratings
>3.5), Liang strong-generalization split (140,768 / 10,000 / 10,000 users), COLD user at interview start,
budget k ∈ {1,2,4,8,16}, questions are ITEMS (answered from real held-out ratings) or GENOME CONCEPTS,
metric full AND tail NDCG@10 on held-out items, no LLM, no simulator.

---

## 0. CORRECTIONS to the 2026-08-04 survey — read first

1. **Biggest miss: GREEDY SLIM (arXiv:2406.06061) runs on ML-25M, reports NDCG@10, and uses our exact
   tail definition.** [V] Closest contract match in the entire literature — closer than DRE (ML-20M,
   k=50) or PERE (ML-10M/20M, k=100). Its long-tail split is verbatim ours: *"Following Cremonesi et al.
   (2010), we refer to the set of most popular items that represent 33% of the ratings as short-head
   items. The remaining items are called long-tail items."* Pass 1 never saw this paper. §2.1.

2. **Greedy SLIM publishes, on our dataset and our metric pair, the exact Act-1 result — and it goes our
   way.** [V] Static Greedy SLIM vs the adaptive QBandit (= Christakopoulou KDD'16, already
   `christakopoulou2016towards` in our INDEX), ML-25M NDCG@10:

   | k | GSLIM (static) | QBandit (adaptive) | GSLIM tail | QBandit tail |
   |---|---|---|---|---|
   | 5 | 0.3390 | 0.3382 | 0.1163 | **0.1183** |
   | 10 | **0.3594** | 0.3456 | **0.1543** | 0.1465 |
   | 15 | **0.3709** | 0.3221 | **0.1674** | 0.1488 |
   | 20 | **0.3752** | 0.2916 | **0.1779** | 0.1466 |

   The adaptive arm ties (full) or wins (tail) at k=5 then **degrades monotonically** while the static arm
   climbs. External, published, same-dataset support for Act 1 — and it independently reproduces our own
   "adaptivity pays only when evidence is scarce" shape. **On Netflix the same table goes the other way
   at k≤10** (QBandit 0.3030/0.3251 vs GSLIM 0.2859/0.3042), which is the honest version and should be
   reported as such. **Do not order these numbers against ours** (explicit ratings, no Liang split, SLIM
   recommender): cite the *pattern*, run the *method*.
   ⚠ Note GSLIM's 0.3752 sits above our certified tower's 0.3496 under a different protocol — check this
   before anyone claims a record.

3. **The "PERE gives 3–10% relative NDCG@10" figure is NOT IN THE PAPER.** [V] Full text contains zero
   `%` characters. Recomputed against c-DPP, its nearest baseline: **+0.59% / +1.15% / +2.38% / +1.25% /
   +3.94%**, and +16.9% only on real-user ML-20M. Pass 1 carried this as [S]; it is now **refuted**.

4. **PERE's own DRE column is ~10× below DRE's published numbers** (0.0382 vs DRE's 0.5712-class
   results) [V] — the `surprise-means-debug-the-harness` signature. **Never cite PERE's DRE row.**

5. **Sepliarskaia does NOT say what we lean on it for.** [V] Authors are **Sepliarskaia, Kiseleva,
   Radlinski, de Rijke** (Kiseleva was missing from our note). **Golbandi is cited but never compared
   against** — the only tree in its experiments is PWDT (Rokach & Kisilevich 2012). The finding is
   narrower and the stated cause is different: trees lose *because they optimize a variance surrogate
   instead of the true loss*, not because they are dynamic. And **everything ties below 5 questions**:
   *"All algorithms provide the same performance for questionnaires with a length less than 5"* (p>0.1).
   At our budgets this paper mostly supports "no measurable difference", not "static wins".
   Also **PEO in PERE's baseline table IS this same paper** — same algorithm, new acronym.

6. **Pass 1's harness audit (§8b) is STALE — verified in code today.** It said Act 1 had ZERO adaptive
   selectors. No longer true: `src/baselines/adaptive_harness.py` (401 lines) implements the sequential
   per-turn selector loop with a byte-identical regression contract against the static path, and
   registers **`glm_ucb`, `thompson`, `eddi_ig`**, backed by `bandit_selectors.py` (GLM-UCB + Thompson
   over the `belief_mf` posterior) and `eddi_selector.py` (EDDI single-order info gain).
   `rbmf_seed.py`, `golbandi_native.py`, `golbandi_node.py` also exist.
   **Still genuinely missing:** the Golbandi adaptive *tree* selector (the split search), IGCN in
   adaptive form, DRE, RMVA, fMF, PERE, Greedy SLIM, A-optimal design.

7. **A whole recent adaptive-interview line was missed: the Trinity College Dublin group.** Shams CBB
   SIGIR'21 → Rajapakse & Leith MCTS RecSys'22 → TORS'23 → RLT4Rec 2024. Pass 1 had RLT4Rec with
   "authorship unverified"; it is **Dilina Chandika Rajapakse and Douglas Leith, TCD** [V, from the PDF].
   UNFIT for us, but for a *stronger* reason than pass 1 gave — §3.

8. **RBMF's author list in our notes was wrong.** **Nathan N. Liu, Xiangrui Meng, Chao Liu, Qiang Yang**,
   RecSys'11 pp. 37–44 [V].

9. **DRE has no duplicate-seed guard.** [V] k independent categoricals over the whole itemset + per-row
   argmax; nothing prevents two rows collapsing onto the same item, silently shrinking the budget. The
   paper never mentions it. Cheap, legitimate criticism to make in print.

10. **fMF's published pseudocode has a sign bug.** [V] Algorithm 2 line 5 prints `argmax_p L_p`,
    contradicting its own Eq. (4) `argmin`. Eq. (4) is correct. You cannot code from the pseudocode.

11. **NOBODY in the classical lineage reports a ranking metric.** [V] Rashid'02, Rashid'08,
    Golbandi'10/'11, Zhou'11, Sun'13 are **RMSE/MAE only** (Rashid'08 adds Expected Utility). Any battery
    we build produces numbers not comparable with theirs. **State this in the chapter rather than let a
    reviewer discover it.** The ranking-metric era in this family starts with RBMF (2011, P@10) and only
    becomes standard with RMVA/DRE/PERE/Greedy SLIM.

---

## 1. Family map, and who is really who

```
Kohrs&Merialdo ICMF'01 (origin of entropy+variance) ─ Rashid IUI'02 ─ Rashid SIGKDD-Expl'08
      (popularity / entropy / pop*ent / logpop*ent / item-item) │ (entropy0 / HELF / IGCN)
                                                                │
  Amatriain SIGIR'09 "Wisdom of the Few" ────────┐              │
  Boutilier UAI'03 (EVOI) ─ Jin&Si UAI'04 ─ Harpale&Yang SIGIR'08 (answerability prior)
  Yu/Bi/Tresp ICML'06 (transductive experimental design, USER side)
                                                 │              │
                                                 │              ├─ Golbandi CIKM'10 (GreedyExtend, STATIC)
                                                 │              │      └─ Golbandi WSDM'11 (ternary tree)
                                                 │              │             ├─ Zhou fMF SIGIR'11
                                                 │              │             ├─ Sun WSDM'13 (l-item screens)
                                                 │              │             ├─ Karimi FDT ICTAI'13, UMUAI'15
                                                 │              │             ├─ Shi LRMF TOIS'17 (local reps)
                                                 │              │             ├─ De Pessemier SciRep'25
                                                 │              │             └─ Gharahighehi arXiv'25 (ATTRIBUTES)
  Liu RBMF RecSys'11 ─ Fonarev RMVA ICDM'16 ──┐
  Anava WWW'15 (A-optimal design, ITEM side) ─┤
                                              ├─ Sepliarskaia SPQ/PEO RecSys'18 (STATIC, A-optimal)
                                              ├─ Kweon DRE WWW'20 (STATIC, learned, Gumbel)
                                              ├─ Proissl Greedy SLIM 2024 (STATIC, SLIM loss) ← ML-25M
                                              └─ Nguyen PERE UAI'24 (ADAPTIVE, region cuts)

  Christakopoulou KDD'16 (QBandit) ─ Parapar&Radlinski DPE WSDM'21 (topic bandit)
  Shams CBB SIGIR'21 ─ Rajapakse MCTS RecSys'22/TORS'23 ─ Rajapakse RLT4Rec'24   ← simulator line
  Houlsby NIPS'12 + BALD arXiv:1112.5745 (GP preference, mutual-information acquisition)
```

**Flag (b) — same algorithm under a new name:**
- **PEO ≡ SPQ ≡ Sepliarskaia et al. RecSys'18.** [V] PERE tables "PEO" as a separate baseline and never
  expands the acronym; its own reference list resolves it to that paper. Do not table them as two systems.
  PERE additionally calls it "active learning", which contradicts the source — SPQ is explicitly static.
- **c-DPP is not an external method at all** — it is PERE's own sequentialised DPP baseline [V]. Since
  c-DPP is the only baseline PERE beats by under 3%, this matters a lot.
- **RMVA is RBMF's maxvol step generalised from square to rectangular.** [V] Same selection principle
  (dominant submatrix of the item factor matrix); RMVA removes the constraint `rank = seed-set size`.
  One family, not two independent bars.
- **`rbmf_seed.py` ≡ `belief_mf.py` with (mu0=0, Sigma0=(1/lam)I, obs_noise=1)** — documented in-module.
- **Anava's A-optimality ≡ Sepliarskaia's SPQ objective ≡ Yu/Bi/Tresp's transductive experimental
  design**, applied to items / question-pairs / users respectively. [V] Three papers, one criterion
  `tr((V_B V_Bᵀ + εI)⁻¹)`. If we run one, say so; do not present three rows as three ideas.
- **Golbandi's tree and Zhou's fMF differ in exactly one thing** — the node output (shrunk rating mean vs
  fitted latent vector). Same ternary split, same squared loss, same sufficient-statistics trick.
- **HELF vs log(pop)·entropy are not the same** [V]: Rashid'08 Table 2 shows the multiplicative version
  correlates **0.99 with rating frequency and 0.17 with entropy** — it *is* popularity. HELF correlates
  0.77 / 0.55. If our `helf` arm was built multiplicatively it is a popularity arm in disguise: **CHECK.**

---

## 2. Origins — implementation grade

### 2.1 Kohrs & Merialdo, ICMF 2001 **[S — FETCH FAILED]**
"Improving Collaborative Filtering for New Users by Smart Object Selection", EURECOM. No DOI, no
retrievable PDF. Attested by Rashid'02 ref [12] [V] and Golbandi'11 ref [10] [V] as **the origin of the
entropy/variance selection idea**. Static. **1–2 agent-hours** (two scalar item statistics).
**Verdict: CITE as the origin; already covered by our variance/entropy arms.**

### 2.2 Rashid et al., IUI 2002 — "Getting to Know You" **[V, full text]**
Al Mamunur Rashid, Istvan Albert, Dan Cosley, Shyong K. Lam, Sean M. McNee, Joseph A. Konstan, John
Riedl. IUI '02, pp. 127–134, ISBN 1-58113-459-2.
- **Six strategies:** Random; MovieLens Classique (1 popular per page of 10, rest random); Popularity;
  **Pure Entropy** (entropy of the five rating frequencies, **missing ratings ignored**); **Pop·Ent**
  (justified as expected bits: "entropy is the number of bits if the user rates the item, popularity is
  the probability that this user will rate it"); **log Pop·Ent** (because "Popularity almost completely
  dominated Pop*Ent… taking the log nearly linearized popularity, making it a better match for
  entropy"); **Item-Item personalized** — the only ADAPTIVE arm: seed from a random draw of the top 200
  by logPop·Ent, then use an item-item recommender to pick items the user is **likely to have seen**,
  recomputed on every submission.
- **Contract:** MovieLens, users with ≥200 ratings → 7,335 u / 4,117 i / >2.7M r. Budget **30/45/60/90
  presented**. Metrics **#movies seen** and **MAE**. Online: 351 new users, 302 completed, 10 days.
- **Numbers:** online mean pages to finish signup **Popularity 1.9, Item-Item 2.3, logPop·Ent 4.0,
  Classique 7.0**. Offline results are **plots only** (Pure Entropy MAE ≈1.08–1.24 vs Pop·Ent ≈0.78–0.80).
- **The "shockingly poor" entropy result, settled [V].** Verbatim: *"Again, Pure Entropy is shockingly
  poor."* Two linked failures: (i) ratability — *"Figure 1 shows a slight increase in entropy for less
  popular movies. Since popularity directly relates to the chance that a new user has seen a movie, this
  strategy presents movies that users are less likely to have seen"*; (ii) accuracy follows — *"with
  fewer rated movies to base predictions on, the MAE for Pure Entropy also suffered."* The kill-shot:
  *"We performed a pilot study… it turned out to be unusable. **Two users had to view several hundred
  movies each before finding ten to rate.**"* Entropy and Random were dropped from the online study.
- **⚠ THIS IS OUR RESULT, 24 YEARS EARLY, AND IT NAMES THE PROBLEM WE SOLVE.** Rashid also found the
  *converse*: Item-Item wins ratability but **loses accuracy** — *"It was hard to believe that the Random
  strategy could get an error rate with eight ratings comparable to the item-item personalized strategy
  with 57 ratings"* — because item-item hones into one cluster (the Star Trek example) and asks about
  items it could already predict. **The axis that maximises answerability is not the axis that maximises
  discrimination, and items cannot decouple them.** That is exactly the sentence our concept-channel
  result answers. Cite Rashid'02 as the statement of the open problem, not just as a strategy bank.
- **2–4 agent-hours** for all six. **Verdict: already in our arm bank; re-cite it properly.**

### 2.3 Rashid, Karypis & Riedl, SIGKDD Explorations 10(2) 2008 **[V, full text]**
DOI 10.1145/1540276.1540302, pp. 90–100.
- **Entropy0** — missing ratings become a **sixth category, value 0**, so every item has the same number
  of votes; then **Guiaşu weighted entropy** to undo the resulting popularity over-reward:
  `Entropy0(a) = −(1/Σ_i w_i) Σ_{i=0..5} p_i w_i lg(p_i)` with **w₀ = 1/2, w₁..w₅ = 1**. (w₀ = 0 recovers
  plain entropy.)
- **HELF** = `2·LF(i)·H(i) / (LF(i) + H(i))` where `LF(i) = lg|R_i| / lg|U|` and `H(i) = H(i)/lg 5`.
  Rationale: *"A nice property of the harmonic mean is that it strongly increases or decreases when both
  of the components increase or decrease."* **Table 2 receipt:** multiplicative combination correlates
  0.17 with entropy / **0.99 with rating frequency**; HELF correlates 0.77 / 0.55. → §1 flag.
- **IGCN** = `IG(a) = H(C) − Σ_r (|C_a^r|/|C|) H(C_a^r)` over **bisecting-k-means user-cluster labels**,
  with the missing-value-weighted variant `IG(a;W) = H(C) − Σ_r [w_r |C_a^r| H(C_a^r)] / E(C;W)`.
- **⚠ IGCN's adaptivity, settled verbatim [V].** *"One issue with information theoretic measures such as
  entropy and its variants discussed so far is that they are **not adaptive to a user's rating
  history**."* Algorithm 3.1 is two-phase: a **non-personalized step** ordering by global IG until the
  user has rated ≥ i items, then a **personalized step** that finds the best l neighbours and
  **re-computes IG using only those l users' ratings**, repeating until the neighbour set stabilises.
  Parameters **(c, n, i, l) = (300 clusters, 15 items/screen, switch after 5 ratings, 150 neighbours)**.
  So IGCN is **re-ranked per BATCH OF 15, not per single answer**, and the adaptivity is a restriction of
  the IG computation to the current 150-NN pool. **Everything else is a static precomputed order** —
  verbatim: *"All of the heuristics except the Igcn can be computed prior to the simulation."*
  **⇒ Pass 1's insurance item stands: if our `entropy0`/`helf` arms are static orders that is CORRECT
  and faithful; but if we bill IGCN as an arm it must do the 150-NN re-computation or it is not IGCN.**
- **Contract:** MovieLens ~11,000 u / 9,000 i / ~3M r, scale 0.5–5.0, 97% sparse. Budget **15/30/45/60/75**
  (pages of 15). Metrics **#seen, MAE, Expected Utility**. Online: 20 days, 381 valid subjects.
- **Numbers:** online pages to finish **Entropy0 3.0, Popularity 3.6, HELF 4.6, IGCN 5.5**; online MAE
  (user-kNN) **IGCN 0.67 best, Popularity 0.73, Entropy0 0.73, HELF 0.84 worst**. Verdict verbatim:
  *"Overall, the Popularity measure performed the worst, and the Igcn measure performed the best."*
  **The Popularity/HELF paradox is resolved and is directly useful to us:** HELF is best at avoiding
  false positives, Popularity worst; Popularity much better on MAE for *positive* actual ratings, HELF
  better for negative — and since positives outnumber negatives 3.6:1, Popularity wins MAE while HELF
  wins EU. IGCN balances both. **This is a published precedent that the metric you pick decides which
  elicitation arm wins** — the same lesson as our own bank result.
- **Historical note [V, from Golbandi'11 §3.1.2]:** Rashid'08 *contemplated* a decision tree and
  **dismissed it as not practically feasible**, citing missing values; IGCN was the approximation.
  Golbandi'11 then did the dismissed thing, with the Unknown branch as the answer.
- **Entropy0/HELF 1–2 agent-hours; IGCN 4–7. Verdict: MUST-INCLUDE IGCN in its genuine two-phase form.**

### 2.4 Golbandi, Koren & Lempel, CIKM 2010 — GreedyExtend **[A/S — full text FETCH FAILED]**
CIKM '10 pp. 1805–1808, DOI 10.1145/1871437.1871734. Full text closed everywhere reachable (ACM 403,
S2 `openAccessPdf: CLOSED`). Mechanism reconstructed from **the same authors' WSDM'11 §3.1.1** [V], the
**Yahoo patent US8510248B2** [V], and the RMVA paper's description [V].
- **GreedyExtend = the STATIC predecessor.** One seed set for everyone. Incrementally add the item that
  most reduces error **when the recommender is restricted to learn only interactions involving seed
  items**: `k ← [k, i]`, `i = argmin_{i'∉k} F([k, i'])`, F = RMSE of a **factorized item-item model**
  fit on seed-restricted data. Complexity **O(m³f²)** — RMVA: *"the brute force manner of GreedyExtend
  implies very high computational costs. Hence, it is hardly scalable."*
- Also introduces **Coverage** (*"emphasizes those items more heavily co-rated with other items"*) and
  **Var** (`√|R(i)| · Var(i)`). **The exact Coverage formula is the one genuine gap in this pass** —
  not reproduced in any secondary source reachable, including the patent. **Verification debt.**
- **The Napoleon Dynamite counterexample lives here:** it tops every contention metric on Netflix yet
  correlates weakly with everything, so it is a bad seed despite winning on popularity AND contention.
  This is the published statement of why our "polar bank" needs a co-rating/coverage term, not just
  discrimination.
- **Numbers [V via WSDM'11 Fig. 3]:** GreedyExtend is the best static alternative; needs **>20 shown
  items to reach RMSE 0.971**, vs >30 for Var and HELF.
- **4–8 agent-hours to code, but compute-bound** — a naive greedy over 18k items with a refit per
  candidate will not finish; needs a candidate-pool restriction (which is itself a HARD RULE 1 decision
  requiring sign-off, or a non-lossy alternative like incremental refit).
- **Verdict: NICE-HIGH.** It is the honest ancestor of *our own* prefix-optimal greedy static bank —
  we should cite it as prior art for that object rather than claim it.

### 2.5 Golbandi, Koren & Lempel, WSDM 2011 — the ternary tree **[V, full text]**
WSDM '11 pp. 595–604, DOI 10.1145/1935826.1935910. Patent US8510248B2.
- **The split criterion, exact, and it is NOT entropy/Gini.** Verbatim: *"we **deviate from common error
  metrics typical to classification tree construction (like entropy or Gini impurity)**, but rather rely
  on the **squared loss metric**."* With `μ(t)_i` the node mean for item i and
  `e²(t)_i = Σ_{u∈S_t∩R(i)} (r_ui − μ(t)_i)²`, `e²(t) = Σ_i e²(t)_i`:
  > **Err_t(i) = e²(t_L) + e²(t_H) + e²(t_U)**,  **splitter(t) = argmin_i Err_t(i)**  (Eq. 1)
  Exhaustive over all n items at every node.
- **Ternary: Lovers / Haters / Unknowns.** 4–5 → like, 1–3 → dislike (dataset mean 3.6), unrated →
  unknown. **The Unknown branch is never computed directly — it is derived by subtraction** from
  sufficient statistics gathered on Lovers and Haters only (Eq. 3), with
  `e²(t) = Σ_j [ sum2(t)_j − (sum(t)_j)²/n(t)_j ]` (Eq. 4). Complexity per level
  `O(Σ_u |R(u)|²)` — *"exactly equals that of the item-item approach."* Splitting is an **in-place 3-way
  counting sort** of a contiguous span in a per-item index; no copying.
- **Bias:** only user bias matters (item bias washes out, since squared error compares values on the same
  item). `b_u = (Σ_{i∈R(u)} r_ui + β·μ)/(|R(u)| + β)`, **β = 7**. *"while biases are important while
  constructing the tree in the training phase, they become immaterial during the online phase."*
- **Depth 6–7**, three stopping rules (depth; no strict error reduction; node rating mass < **ξ =
  200,000**). Table 1: RMSE 0.97260 / 0.97172 / 0.97129 / 0.97115 at depth 5/6/7/8.
- **Leaf profile = per-item hierarchical shrinkage toward the parent**, not a raw mean:
  `r̂_t(i) = (sum(t)_i + λ₁·r̂_p(i)) / (n(t)_i + λ₁)`, **λ₁ = 200**. Chosen over pruning deliberately —
  *"pruning a node as a whole would be unappealing in our sparse data scenario"*, so shrinkage must be
  per item.
- **Serving rank uses a uniqueness gap**, `d̂_t(i) = r̂_t(i) − b·μ_i`, **b = 0.8**.
- **Randomized multi-tree:** splitter picked w.p. ∝ `max(0, e²(t) − Err_t(i))^a`, **a = 10**; 200 trees;
  leaf weight `w_t = (#L·w_L + #H·w_H + #U·w_U)^c` with **w_L = 5, w_H = 1, w_U = 0.02, c = 2**.
- **⚠ THIS INDEPENDENTLY REPRODUCES OUR ARM-N SIGN FINDING.** Verbatim: *"a 'like' vote is considerably
  more useful than a 'dislike' vote, which we find to be quite interesting."* Optimised weights make a
  like worth **5×** a dislike and an unknown worth **0.02** — robust across tree counts. Our "a 'no' is
  worth +0.023 at k=2 and nothing by k=8" has a 2011 precedent that we should cite rather than discover.
- **Headline:** depth-6 tree reaches RMSE 0.97172 **after 6 questions**; GreedyExtend needs **>20**, Var
  and HELF **>30**. Against organic ratings, *"the first 12 ratings of users were needed in order to
  match the accuracy of a depth-6 tree"*. 5 trees: 0.9725 → 0.9656.
- **Their own honest self-assessment, quotable for our Paper A framing:** *"Compared to other
  (non-bootstrapping) recommendation algorithms, a decision-tree recommender delivers a quite high RMSE.
  This makes decision-trees undesirable as a traditional recommendation system."* The tree is an
  **instrument**, not a recommender.
- **Contract:** Netflix >100M r / 480k u / 17,770 i; **RMSE only**; test = Qualifying set.
- **No public code.** **6–12 agent-hours** (~300 lines; the non-obvious piece is the in-place span
  partitioning — a copying implementation will be memory-bound).
- **Verdict: MUST-INCLUDE, and this is the gap that matters most.** We have the leaf model
  (`golbandi_leaf.py`, `golbandi_native.py`) but **not the split search**, i.e. not the adaptive
  contribution. Building it is step one.

### 2.6 Zhou, Yang & Zha, SIGIR 2011 — fMF **[V, full text]**
SIGIR '11 pp. 315–324, DOI 10.1145/2009916.2009961.
- **Core idea:** the user profile is not a free parameter, it is a **function of the answers**,
  `u_i = T(a_i)`, `r̂_ij = v_jᵀ T(a_i)`, minimizing
  `Σ_{(i,j)∈O} (r_ij − v_jᵀ T(a_i))² + λ‖V‖²`.
- **Split (Eq. 4):** partition node users into R_L/R_D/R_U by the answer to candidate p, score p by
  post-split squared error **with each child's optimal profile plugged in**, take the **argmin**.
  ⚠ **Algorithm 2 line 5 prints `argmax` — a paper bug. Eq. (4) is correct.** [V]
- **Leaf fit (Eqs. 5–7), closed form:**
  `u_L = (Σ_{i∈R_L} Σ_{(i,j)∈O} v_j v_jᵀ)⁻¹ (Σ_{i∈R_L} Σ_{(i,j)∈O} r_ij v_j)`.
  **Hierarchical regularization** shrinks the child toward the parent, `λ_h = 0.03`; the closed form is
  **not printed** and must be derived: `u_L = (Σ v_jv_jᵀ + λ_h I)⁻¹ (Σ r_ij v_j + λ_h u_C)`.
- **Item factors (Eq. 2):** per-item ridge with tree outputs as fixed features.
- **Alternating order (Algorithm 1):** init V randomly → **fit the tree with V fixed** → update every
  `v_j` with T fixed → repeat. **Converges in ~5 outer iterations.**
- **Cold-user scoring:** walk root→leaf, the leaf's stored profile IS the user vector, then one matvec.
  No per-user optimization at serve time.
- **Contract:** ML-1M, EachMovie, Netflix. **RMSE only.** **Budget 3/4/5/6/7 questions** (= depth) —
  overlaps ours almost exactly. Users 75/25, each test user's ratings split 75% answer / 25% eval.
- **Numbers (cold-start RMSE, ML-1M):** fMF **0.9509 / 0.9480 / 0.9437 / 0.9429 / 0.9410** at q=3..7 vs
  Golbandi Tree 0.9767 / 0.9683 / 0.9615 / 0.9512 / 0.9523 and TreeU 0.9913→0.9657. fMF > Tree on all
  three datasets, t-test p=0.05, **no CIs, no paired test described, no seed count**.
  ⚠ Its warm-start tables use a *different protocol* — do not order numbers across them.
- **Code:** no official release. Third-party `ameyavilankar/fMf-cold-start-CF` is Python 2, last commit
  2013-05-29, uses SGD instead of the closed forms, updates both user and item vectors inside the
  "alternating" step (a bug), ignores the observed set, no hierarchical regularization —
  **reference for file structure only, not a correct reimplementation.**
- **6–12 agent-hours** (4–6 if only cold-start and reusing our eval harness).
- **Verdict: MUST-INCLUDE.** It is the "Golbandi with a real latent recommender" control, its budget
  range is literally ours, and it makes Act 1 robust to "your tree's leaf model was a shrunk mean".

### 2.7 Sun, Li, Lee, Zhou, Lebanon & Zha, WSDM 2013 — multiple-question trees **[V, full text]**
WSDM '13 pp. 445–454, DOI 10.1145/2433396.2433451. (Ke Zhou is fMF's first author — partly a
self-successor.)
- **Correction to a natural misreading:** a "multiple-question" node is **not** a question about a
  category. It is **l separate per-item questions on one screen** (l ∈ {2,3,4}), each answered
  independently, with a **signed sparse linear combination** routing the user. **No attribute channel
  anywhere** — the paper lists that as future work.
- **Mechanism:** per node learn `w ∈ ℝⁿ` with ≤ l nonzeros (the nonzero items *are* the questions);
  answers `x ∈ {−1,0,+1}ⁿ`; routing `p_i = 1/(1 + c·exp(−x_iᵀw))`, `q_i = 1/(1 + c·exp(+x_iᵀw))`,
  **c = 2**, Unknown = 1 − p − q. So **L when x_iᵀw > 0, D when < 0, U only when the user answers
  "unknown" to every question on the screen** — that is the whole point, since in single-question trees
  the Unknown branch swallowed >80% of users per split.
- **Objective (Eq. 5)** is the soft-membership squared loss + `λ‖w‖₁`, solved by projected scaled
  sub-gradient; λ binary-searched to give exactly 1..l nonzeros, then the best by unpenalised Eq. 5.
  **Initialization is load-bearing** (non-convex): start from the best single item found by **fMF**, so a
  working fMF splitter is a required subroutine.
- **Node profile is not a stored vector** — a node-local **affine map of the whole answer prefix**,
  `T_L(x_i) = Z_L x̄_i`, fit in closed form via Kronecker products (Eq. 7), shrunk toward the zero-padded
  parent (Eq. 8). The intercept dimension recovers the classical constant-profile tree as a special case.
- **⚠ It does NOT benchmark Golbandi as published** — its `w4Base` is Golbandi's multi-tree idea with
  fMF's tree generator swapped in, because *"the tree generation method in [6] always performs worse than
  the one in [27]."*
- **⚠ NO ACCURACY TABLE EXISTS.** All accuracy results are unannotated line plots. Recoverable axis
  ranges bound the whole effect: **ML-1M 0.916–0.928, Netflix 0.910–0.930, EachMovie 1.300–1.345** —
  total spread between best and worst method across the entire interview is ~0.012 / 0.018 / 0.035 RMSE.
  Per-question (the fair axis) the authors concede *"we see no major difference between w1Base and w2,
  w3, w4 when relatively few questions have been asked"*; the advantage is in the tail only, because
  *"in w1Base, the performance cannot improve with trees more than 6−7 layers."*
- **The one real table (Table 3, 76-user study) is the useful one — answer time is SUBLINEAR in l:**
  binary 4.41 / 6.73 / 7.88 / 10.42 s for l = 1/2/3/4 (rating scale 6.93 / 9.59 / 10.14 / 13.86).
  4 questions cost ~2.5× the time of 1. **This is the published evidence for asking in batches**, and it
  is directly relevant if we ever defend a k>1 screen.
- **No public code.** **30–55 agent-hours** (20–30 with an fMF tree in hand). **It cannot be validated
  against the paper — there are no published numbers to match.**
- **Verdict: UNFIT as a table row** (no numbers to snap to, 30–55 h, per-question parity at our budgets).
  **CITE for the batch-answering-time result and for the Unknown-branch-starvation problem.**

### 2.8 McNee, Lam, Konstan & Riedl, UM 2003 **[A — PDF is bitmap-font, unextractable]**
"Interfaces for Eliciting New User Preferences in Recommender Systems", UM'03, LNCS 2702, pp. 178–187,
DOI 10.1007/3-540-44963-9_24. A **user study**, not an algorithm. Three interfaces: system-selected,
user-selected (the user types in titles), and a mixed-initiative hybrid. Findings: both pure interfaces
produce accurate user models; **user-selection produced increased loyalty despite a longer signup**; the
**mixed-initiative interface underperformed** ("less accurate user models with no increase in loyalty").
**Directly relevant to the open-request direction:** this is the published precedent, from the same lab
in 2003, that letting the user name what they want raises loyalty even when it costs accuracy.
**Verdict: CITE (Paper D / open-request framing). Not a baseline.**

---

## 3. The static seed-set / learned-questionnaire branch — implementation grade

### 3.1 Greedy SLIM — Proissl, Vatic & Waldschmidt, arXiv:2406.06061 (2024) **[V, full text]**
**THE contract match. Missed entirely by pass 1.**
- **Citation.** Claudius Proissl, Amel Vatic, Helmut Waldschmidt, "Greedy SLIM: A SLIM-Based Approach For
  Preference Elicitation", arXiv:2406.06061 [cs.IR], 10 Jun 2024. **arXiv-only — dblp lists it under
  CoRR with no venue** [V, checked the dblp author page]. (A search result claiming "PacificVis 2024" is
  wrong; that is a different Proissl paper on road-network rendering.) The authors are algorithms/graph
  researchers, not recsys regulars — weigh accordingly.
- **Mechanism.** SLIM loss `l(W) = ‖X − XW‖²_F + λ_F‖W‖²_F + λ_1‖W‖_1`. Build the n×n item-item matrix W
  **row by row**; the order in which rows are filled **is** the questionnaire. Closed-form optimal weight
  `w*_ij = max( (−λ_1/2 + Σ_u x̂_uj·x_ui) / (λ_F + Σ_u x²_ui), 0 )`; pick
  `i* = argmin_{i∈I_W} Σ_{j≠i} ( l_ij(w*_ij) − l_ij(0) )`; fill row i*, update the residual X̂, repeat.
  No sampling — every candidate is evaluated every round.
- **STATIC**, by explicit statement: *"Our approach describes a static questionnaire but we believe that
  our ideas can be easily extended to the dynamic setting."* Their own definition: *"Static
  questionnaires ask the same questions to all new users whereas the output of dynamic questionnaires
  depends on the already given answers."*
- **Training cost.** ML-25M **19 minutes per row** ⇒ ~6–7 h for 20 questions. The residual matrix does
  not fit in memory and is recomputed each iteration. Hyperparameters ML-25M **λ_1 = 2¹², λ_F = 2¹⁶**
  (grid over powers of two 2⁰–2¹⁹ on a validation split) — enormously larger than SLIM's defaults of 1.
- **Contract.** **ML-25M (162,541 u / 59,047 i / 25,000,095 r), UNFILTERED** — *"We do not filter these
  sets to ensure that we do not accidentally introduce any biases"* — plus Netflix. **Explicit 1–5
  ratings, not binarized.** Random 90/10 user split (≈16,254 test users). Protocol: zero row for the test
  user, ask the questionnaire, copy the true rating from the test set, then rank with SLIM;
  **questionnaire items are excluded from the recommendations** (NOT the same discipline as our harness -- CORRECTED 2026-08-05: they exclude ASKED items, we exclude RATED items; their adaptive decline is a protocol artefact by their own diagnosis, see findings/candidate_masking_protocol.md).
  Unknown = "zero-rating", treated as 0. Metrics **NDCG@10** primary, P@N, R@N, reported **all-items AND
  long-tail** with the Cremonesi 33%-of-ratings head. Budgets **5/10/15/20**.
- **Numbers:** §0.2. User study: 103 participants (*"many of them were somehow linked to the authors"* —
  weak), 10 elicitation ratings, positive-feedback rate GSLIM 77.2% / QBandit 52.6% / RGain 52.0%.
- **Baselines:** QBandit (Christakopoulou et al. 2016, LFM + Thompson sampling), QGreedy (greedy ordering
  over a **pre-trained** W — genuinely different from Greedy SLIM, which builds W greedily), RGain
  (static gain-sum ranking).
- **Code:** https://osf.io/myh2q/ (view-only link in the paper). Not GitHub, unversioned.
- **Fit.** Ports almost perfectly. **What must change:** (i) explicit ratings vs our binary >3.5 — the
  SLIM loss is indifferent, feed it our binary X; (ii) budgets 5–20 vs our 1–16, and **because it is a
  prefix-optimal greedy ONE fit gives every budget** — the same object our own
  `prefix-optimal-greedy-is-the-right-object` note says is right; (iii) its recommender is SLIM itself —
  run both ways, seeds-into-our-tower and seeds-into-SLIM.
- **6–10 agent-hours** (~80 lines; the cost is the 6 h compute and a sparse residual update that does not
  recompute from scratch).
- **Verdict: MUST-INCLUDE. The strongest STATIC bar, displacing DRE.** Same dataset, same metric, same
  tail definition, prefix-optimal so one fit covers every k, and its recommender (SLIM) is in the same
  family as our EASE bar so the comparison is legible.

### 3.2 DRE — Kweon, Kang, Hwang & Yu, WWW 2020 **[V, full text + repo inspected]**
- WWW '20 pp. 2810–2816, DOI 10.1145/3366423.3380042. arXiv:2402.16327 is a re-upload, **not** a 2024
  paper (our standing correction holds).
- **Mechanism, exactly.** Encoder is *only* a trainable matrix `Π ∈ R^{k×m}` of k categorical
  distributions over the whole itemset. Seeds drawn differentiably by Gumbel-Softmax
  `Y_ij = exp((logΠ_ij + g_ij)/τ) / Σ_l exp((logΠ_il + g_il)/τ)`; `z(u) = Y·r(u)`; decoder is a 2-layer
  FC `[k → d → m]`, sigmoid on both layers, `d ∈ {200,300,500}`. Loss is plain MSE reconstruction of the
  full rating vector, end-to-end. **Exponential anneal `τ = T₀·(T_E/T₀)^(e/E)` is the load-bearing
  hyperparameter**: best T₀=10, T_E=0.1 → P@20 0.5396 on ML-1M; with **no** annealing (T_E=T₀) it
  collapses to 0.4502–0.5158. Final seeds `S_i = argmax_j Π_ij`; decoder then re-trained with the encoder
  frozen.
- **STATIC by design and deliberately so** — "chooses all the seed items at a time", explicitly contra
  greedy. This is the paper arguing the static side of Act 1.
- **⚠ Defect: no duplicate-selection guard** — see §0.9.
- **Contract.** ML-1M, CiteULike, Yelp, **ML-20M (99,220 u / 20,660 i / 9,478,902 r)**. **Binarize at
  >3.5 — our threshold.** 80/20 user split, 10% of train for validation. **P@N and NDCG@N,
  N ∈ {10,20,50,100}.** Primary budget **k=50**.
- **Numbers, ML-20M k=50:** DRE P@10 0.5712 / P@20 0.4875 / **NDCG@10 0.6077** / NDCG@20 0.5373 vs best
  baseline RMVA 0.5116 / 0.4382 / 0.5444 / 0.4851 → +11.6% NDCG@10, p ≤ 0.005. Best headline CiteULike
  NDCG@10 +31.98%; **weakest Yelp NDCG@10 +1.56%, not significant**. **The k-sweep exists only as
  Figure 3 (P@20 vs k) — per-k ML-20M numbers are NOT recoverable from the text** [V]; any such number
  is [S]. Training time never reported.
- **Code:** https://github.com/WonbinKweon/DRE_WWW2020 — **Jupyter Notebook**, PyTorch 1.2.0 + SciPy
  1.3.0, 9 stars / 1 fork, **last push 2020-08-27** (README-only). Ships a preprocessed tensor
  `u_implicit_5.zip`, so **raw-MovieLens data prep is NOT included**.
- **3–5 agent-hours.** **Verdict: MUST-INCLUDE** — the strongest *learned* static bar and the one a 2026
  reviewer names; no longer the single strongest static bar, but the one with public code and a venue.

### 3.3 SPQ / PEO — Sepliarskaia, Kiseleva, Radlinski & de Rijke, RecSys 2018 **[V, full text]**
- RecSys '18 pp. 172–180, DOI 10.1145/3240323.3240352. **Code: https://github.com/Seplanna/pairwiseLearning.**
- **Mechanism.** Assumes a pre-trained LFM as ground truth. Questions are **relative (pairwise)**, answers
  ∈ {−1,0,+1}; question vector `v_(i,j) = v_i − v_j`; user vector by linear regression on answers.
  Objective is the true downstream loss, which (citing Anava WWW'15) reduces to the **A-optimal design**
  criterion `argmin_B tr[(V_B V_Bᵀ + εE)⁻¹]`, optimized greedily. Theorem 1 reduces each greedy step to
  "pick a question vector with small component in the span of those already chosen and large orthogonal
  component", approximated by a diameter-of-point-set routine to avoid O(|I|²) pair enumeration.
- **STATIC.** ML-20M filtered to 10M r / 89,169 u / 11,785 i, binarized at ≥4; Amazon books. Metrics
  **PCPF** and **P@10**. Budgets **N ≤ 29**. Conditions *Simulated* (answers generated from the same LFM
  that picks the questions — circular by our firewall) and *Real*. **Figures only, no numeric tables.**
- **What it actually concludes** — §0.5. Also: on Amazon-Real, **Forward Greedy (Anava'15) is
  statistically BETTER than SPQ for N > 25**, gap "<0.5%".
- **Fit.** Its query type is pairwise, ours is seen/like — that is the real mismatch. **What ports is the
  A-optimal criterion**, and it is nearly free: A-optimality is `tr(A_t⁻¹)` and `bandit_selectors.py`
  already maintains `A_t` as the posterior precision. **4–8 agent-hours** (≈30 lines of new acquisition).
- **Verdict: MUST-INCLUDE, as the A-OPTIMAL-DESIGN ACQUISITION, not as SPQ-the-questionnaire.** Citing it
  correctly rather than as "trees lose" is now a correctness issue for us.

### 3.4 RBMF — Liu, Meng, Liu & Yang, RecSys 2011 **[V, full text]**
- RecSys '11 pp. 37–44, DOI 10.1145/2043932.2043943. Ancestor: Amatriain et al., "The Wisdom of the Few",
  SIGIR'09.
- **Mechanism.** Not classical MF — a **CX / column-subset decomposition**: `min ½‖Y − CX‖²_F` subject to
  **C being exactly k actual columns of Y**. Two stages: (1) rank-k SVD with missing values as **zeros**,
  keep `V ∈ R^{n×k}`; (2) **maxvol** (Goreinov 2010) on V — find a dominant k×k submatrix with
  `|V·V(I,:)⁻¹|_∞ ≤ 1` via LU-with-row-pivoting init then repeated swaps on `argmax|b_st|` of
  `B = V·V(p,:)⁻¹` while the max exceeds 1 (<100 iterations in practice). Reconstruction
  `X* = (CᵀC + λI)⁻¹CᵀY`; cold-start fold-in closed form. **Selection is invariant to the latent basis.**
- **Static.** Netflix, ML-10M, Last.fm. P@10, MAP, AUC + its own **Coverage** and **Diversity**.
  **Budgets 5 / 10 / 20 — overlapping ours exactly.** ML: k=5 P@10 RBMF **0.501** vs MostPop 0.432; k=10
  0.559 vs 0.500; k=20 0.637 vs 0.530. **Honest caveat: at k=5 on Netflix RBMF (0.465) barely beats
  MostPop (0.459) and MostPop wins on AUC.**
- No code. **2–4 agent-hours.** (`rbmf_seed.py` is the *fold-in* half only; the maxvol **selector** is
  what is missing.) **Verdict: MUST-INCLUDE.**

### 3.5 RMVA — Fonarev, Mikhalev, Serdyukov, Gusev & Oseledets, ICDM 2016 **[V, full text]**
- ICDM 2016 pp. 141–150, arXiv:1610.04850.
- **The problem it fixes:** square maxvol requires `rank f = seed size L₀`, so a big seed set forces a
  high-rank SVD and overfits. `Rectvol(S) = Π σ_s = sqrt(det(SSᵀ))` removes the constraint.
- **Algorithm.** Rank-f PureSVD `R ≈ PᵀQ`; f pivots by LU/square maxvol → `S = Q(:,k)`, `C = S⁻¹Q`,
  `w_i = ‖c_i‖²`. While `|k| < L₀`: `i = argmax_i w_i`, append, rank-1 update
  `C ← [[C − c_i c_iᵀC/(1+c_iᵀc_i)], [c_iᵀC/(1+c_iᵀc_i)]]`,
  `w_j ← w_j − (c_iᵀc_j)²/(1+c_iᵀc_i)`. Key identity `Rectvol([S,q_i]) = Rectvol(S)·sqrt(1+w_i)`, so
  **maximizing rectangular volume ≡ maximizing the ℓ₂ norm of the representation coefficients**.
  **O(mL₀²)** vs GreedyExtend's O(m³f²). **Theorem 1:** `‖c_i‖₂ ≤ sqrt(f/(L₀+1−f))`, so `L₀ = 2f` makes
  all coefficient norms < 1 — a principled automatic stopping rule for seed-set size.
- **Reconstruction:** the **raw-rating-matrix** variant `C_R = (R(:,k)ᵀR(:,k))⁻¹R(:,k)ᵀR` is significantly
  better than the low-rank one and is used throughout. **No regularization needed** (contra RBMF).
- **Static.** **ML-20M** and Netflix, P@k / R@k, 5-fold, **budgets 5→100 step 5**, rank tuned per budget.
  **Figures only.** Runtime 1.12 s. Its core empirical point — **the optimal rank is always much smaller
  than the seed size** — is directly useful to us.
- **Code:** https://bitbucket.org/muxas/rectmaxvol stated in the paper [V]; **availability unverified [A]**
  (Bitbucket purged Mercurial repos in 2020). Fallback: Mikhalev's `maxvolpy`.
- **3–6 agent-hours. Verdict: MUST-INCLUDE** — named baseline of both DRE and PERE.

### 3.6 LRMF — Shi, Zhao & Shen, ACM TOIS 36(2) art. 22, 2017 **[FETCH FAILED — mechanism [S]]**
DOI 10.1145/3108148. dl.acm 403; S2 reports the PDF closed and the abstract elided; no author PDF.
(Our INDEX's "Shi et al." is right; there is no "Wang" author.)
**[S] mechanism:** global RBMF picks one representative set for everyone; LRMF builds an **interview
decision tree that partitions users into groups and selects LOCAL representative items per group** — the
RBMF↔Golbandi hybrid, and conceptually the closest published thing to what our harness does.
**Verification debt: get this via institutional access before citing.**
**Verdict: potentially MUST, but unciteable until someone reads it.**

### 3.7 Anava et al., WWW 2015 — optimal design (ITEM side) **[V, full text]**
- Oren Anava, Shahar Golan, Nadav Golbandi, Zohar Karnin, Ronny Lempel, Oleg Rokhlenko, Oren Somekh,
  WWW '15 pp. 45–54, DOI **10.1145/2736277.2741109** (note: not `.2741133`), arXiv:1406.2431.
- **Mechanism.** Under whitening, expected MSE reduces to **A-optimality**
  `min_{U_B} σ²·tr((P_B P_Bᵀ)⁻¹) + σ²`; heteroscedastic version via GLS. **No convex relaxation, no
  rounding** — **BGS1/BGS2 = backward greedy** (start from all candidates, repeatedly delete the one whose
  removal least increases the trace), justified by monotone supermodularity with an Il'ev steepness
  guarantee. Supermodularity is proven only under a diagonality assumption the authors themselves call
  *"seeming like a non-realistic assumption."*
- **Static, explicitly:** *"the B users are all selected at once — the publisher does not have the luxury
  to receive some ratings, and then to adaptively select additional users."*
- Netflix, 770 held-out new items, RMSE, **B ∈ {2,…,100}**. **⚠ No numeric table, no annotated figure —
  do not cite a number from this paper; there isn't one.** Significance is asserted in a footnote with no
  test statistic.
- **⚠ THE USER-SIDE ANALOGUE IS SETTLED AND NOT OPEN.** Verbatim conclusion: *"an interesting direction
  would be to study the user cold-start problem using the techniques we presented… the user and the item
  cold-start problem are essentially different, mainly due to the ability to adaptively interview new
  users… **Whereas this enables decision tree type solutions for user cold-start handling (which are
  currently the state-of-the-art)**, such solutions cannot be implemented in the setting considered in
  this paper."* And they concede prior art: **Yu, Bi & Tresp, "Active Learning via Transductive
  Experimental Design", ICML 2006** [V] is literally the user-side version — EachMovie, 5,000/5,000 split,
  ridge from answers to unrated movies, **m = 5,10,…,50**, MAE, candidates restricted to the 100 most
  popular; objective `max_X Tr[K_vx(K_xx+μI)⁻¹K_xv]`, greedy `argmax ‖K_x‖²/(k(x,x)+μ)` with deflation,
  or an **ℓ₁-relaxed alternating optimization**; NP-hard (Thm 4.1); **static**.
- **⚠ Yu/Bi/Tresp's headline finding belongs in our bank, verbatim:** *"A bit surprisingly, choosing the
  most popular movies does not bring advantages over random guessing. This is because that the most
  popular movies are rated highly by nearly all the users and thus give no information about user
  tastes."* That is the popularity-is-not-discrimination result, published in 2006.
- Also **Chamon & Ribeiro, NeurIPS 2017 (arXiv:1711.01501) §5.1** [V] — EachMovie, A-optimality, MAE 2.3
  at k=10 falling below 1.8 by k≥35 vs random 2.8–3.3; supplies α-supermodularity machinery repairing
  Anava's diagonality assumption.
- **4–8 agent-hours** (Sherman–Morrison rank-1 downdates). The user-side transpose is the *same code*.
- **Verdict: MUST-INCLUDE the A-optimal criterion once** (shared with §3.3). Cite Anava for the
  formulation and the guarantee, Yu/Bi/Tresp for the user-side precedent, never for a number.

### 3.8 DPE — Parapar & Radlinski, WSDM 2021 **[V, full text]**
- "Diverse User Preference Elicitation with Multi-Armed Bandits", WSDM '21, DOI 10.1145/3437963.3441786.
- **Mechanism.** Not a bandit over items — **each arm is an item TOPIC** (genre / cuisine / product type;
  items may belong to several). **Thompson sampling with Gaussian priors, one bandit per user**, arms
  initialised identically. Each timestep: sample θ_a per arm, take the max, present one item from that
  topic, use the rating as reward, update.
- **Adaptive**, over a coarse topic space. ML-20M (51,869 u / 26,654 i / 20 categories) and Amazon 2014,
  users filtered to ≥100 ratings, **up to 75 elicited preferences**.
- **Why it matters beyond being a baseline.** (i) Its thesis — *popularity bias in ELICITATION propagates
  into recommendation, and diversifying at elicitation time improves both diversity and accuracy* — is
  the published statement of the effect our own polar-bank result measures. **This is the paper to cite
  there.** (ii) It contributes a protocol argument: **the TestItems protocol is sensitive to missing
  ratings and thereby systematically favours entropy-based elicitation**; they propose
  *AvailablePreferences* instead. **That is a direct methodological threat to our entropy arms and must
  be checked against our harness.**
- **2–3 agent-hours** (Thompson over ~20 genome-derived topics is trivial; the work is the item→topic map,
  which the tag genome gives us free).
- **Verdict: MUST-INCLUDE** — the only elicitation paper whose subject is the *bank*, it is adaptive, it
  is cheap, and it is the concept/attribute-channel analogue we currently have no named rival for.

### 3.9 PERE — Nguyen, Nguyen, Doan & Nguyen, UAI 2024 **[V, full text]**
- UAI 2024 / PMLR v244, arXiv:2406.00973. VinAI / VinUniversity / CUHK.
- **Mechanism, exactly.** Item embeddings normalised into `H = [0,1]^d`, fixed. Answers ∈ {+1,−1,NA}.
  Each (liked i, disliked j) pair gives `‖v_i − u₀‖² ≤ ‖v_j − u₀‖²` ⇒ the **linear cut**
  `2u₀ᵀ(v_j − v_i) ≤ ‖v_j‖² − ‖v_i‖²`; the plausible set is that polytope ∩ H. **NA answers generate no
  cut.** Answerability enters as an **experience probability**
  `p_0i = w_i · sigmoid(1/c_0i − κ₀/sqrt(d − c_0i))` with `c_0i = ‖u₀−v_i‖₂`, `w_i` = normalised
  popularity; the system uses a surrogate with distance to the current Chebyshev center and κ dropped.
  **The Chebyshev center is one LP per round** (Theorem 1):
  `max r s.t. 2u_cᵀ(v_j−v_i) + 2r‖v_j−v_i‖₂ ≤ ‖v_j‖²−‖v_i‖², u_c ∈ H, r ≥ 0`.
  Query rule is a three-case score keyed on the positive/negative shells,
  `min_i (1−p̂_i)[q⁺I⁺ + q⁻I⁻ + q^NA(1−I⁺)(1−I⁻)]`. Centers across rounds combined with a fixed
  linearly-increasing reweighting (∝ K + t·m); recommend by Euclidean distance to the aggregate.
- **Adaptive.** DPP burn-in **K = 50**, then **T = 5 rounds × m = 10**. Embeddings from LightGCN and
  biVAE; Gowalla, Amazon-Books, **ML-10M/20M**. **NDCG@10**, MAP, MRR, HR@k, AUC@k. ~0.27 s/round.
- **⚠ Numbers: §0.3 / §0.4.**
- **⚠ Two structural caveats [V].** (i) Assumption 3 (preference consistency) asserts the user never rates
  a nearer item worse than a farther one — strong; violations tested separately (τ-flip, Table 6).
  (ii) **The main experiments synthesise users from the CF embeddings themselves** ("we consider the
  embedding produced by the collaborative filtering methods … as the 'true' embedding") — the answer model
  and the recommender share a geometry, exactly the circularity `answer-simulator-circularity-ruling`
  forbids. **Only the real-user rows (Table 5) are citable.**
- **Code: none.** No repo, no reproducibility section, nothing on OpenReview [V, searched].
- **10–20 agent-hours.** **What must change for us:** budget 100 → ≤16 (at our budgets the burn-in alone
  would be the entire interview — itself a reportable finding); binary seen+like answers encoded as
  halfspaces (defensible: like ⇒ ⟨u,v_i⟩ ≥ τ).
- **Verdict: MUST-INCLUDE, but DEMOTED from "the strongest adaptive bar".** With the % claim refuted and
  the margins under 3% against its own baseline, PERE is a *necessary citation* (the recent peer-reviewed
  paper whose thesis is Act 1's question) rather than a fearsome bar.

---

## 4. The classical active-learning-for-CF branch — and its consensus

### 4.1 Rubens, Kaplan & Sugiyama — Handbook chapter (2011; 2015 with Elahi) **[V 1st ed. / [A] 2nd]**
Recommender Systems Handbook, Springer 2011 pp. 735–767, DOI 10.1007/978-0-387-85820-3_23; 2nd ed. 2015
pp. 809–846 with **Mehdi Elahi** as second author, DOI 10.1007/978-1-4899-7637-6_24. **[S]: the 3rd ed.
(2022) appears to have no active-learning chapter.**
- **Taxonomy:** instance-based vs model-based × motivation {Uncertainty Reduction (of rating estimates /
  decision boundaries / model parameters); Error Reduction (output-estimate change / test error /
  parameter change / parameter variance); Ensemble-based}, plus a section on conversation-based AL.
  Named criteria with formulas include variance, entropy, least-confidence, confidence interval,
  model-parameter uncertainty, KL-to-optimal-parameters, **Y-Change / output divergence**
  `G(x_a) = −Σ_{x∈X_test} (1/|Y|) Σ_y (f̂_T(x) − f̂_{T∪(x_a,y)}(x))²`, cross-validation-based, parameter
  change, and variance-based `G = C + B + V` naming **A-optimal / D-optimal / transductive experimental
  design**.
- **⚠ The chapter explicitly refuses to report any comparative number, verbatim:** *"Explicit performance
  numbers are not supplied because to our knowledge no such comprehensive comparison in fact exists…
  For these reasons we have opted to omit performances figures of any kind."*
- **⚠ AND THIS IS WHERE THE FOLK CONSENSUS COMES FROM.** §3, verbatim: *"With Traditional AL, users are
  asked to rate a set of preselected items… It may be argued that since these items are selected by
  experts, they capture essential properties for determining a user's preferences. Conceptually this may
  sound promising, but in practice **this often leads towards selecting items that best predict the
  preferences of only an average user**."* **There is no supporting experiment anywhere in the chapter.**
  PERE 2024 repeats the same claim as motivation ("a fixed seed set is unlikely optimal for all new
  users"). **Both are motivating prose, not measurements.** This is the single most useful thing this
  pass found for defending Act 1.
- Also flags **Ratability** ("you cannot properly rate a movie you have not seen") as a first-class
  desideratum from 2011. **Y-Change: 3–5 agent-hours.**

### 4.2 Elahi, Ricci & Rubens — Computer Science Review 20 (2016) 29–50 **[V, full text]**
DOI 10.1016/j.cosrev.2016.05.002. **30 strategies**, taxonomised by *personalization* × *hybridization*
(single vs combined; combined split into static vs adaptive combination). Contains the exact formulas for
Variance, Entropy, **Entropy0**, **GreedyExtend** (`L_u = argmin_{L⊆I_u} F(A(L))`), Representative-based,
Popularity, Co-coverage, log(pop)·entropy, **HELF**, Item–Item, Binary Prediction, Personality-based
Binary Prediction, Influence-based, Impact Analysis, Aspect Model, Highest/Lowest-Predicted, **MinNorm**,
**IGCN**, **DecisionTree**, Influence×Variance, **Non-Myopic**, TreeU, **fMF**, Voting, **Adaptive
Switching**.
- **Which are strongest, verbatim §7.1:** *"Strategies that have shown excellent performance in terms of
  ranking quality (NDCG/MAP), are **Representative-based and Voting** strategies. In terms of precision,
  prediction-based strategies (Highest-predicted, and Binary-predicted) have shown excellent performance.
  In terms of number of ratings acquired, as expected, strategies that consider the popularity of items
  (Popularity and Entropy0) can acquire the largest number of ratings."* With the caveat: *"while the
  adopted evaluation methods share some similarities, still they differ significantly and hence **it is
  difficult to draw a decisive conclusion** and properly generalize the results reported in individual
  works."*
- **⚠ The survey's own justification for personalization is ANSWERABILITY, not informativeness, verbatim:**
  *"**Personalization increases the chance of rating elicitation:** … Personalized strategies tackle this
  issue by maximizing the probability that the selected items are familiar to the user, and hence are
  ratable."* That is a clean citable lineage for a concept channel whose selling point is that concepts
  are answerable by ~97% of users while divisive items are answerable by ~5%.
- Also: *"Partially randomizing can be beneficial… particularly in the early stage."*
- **Open problem listed in 2016 §8: "Sequential vs. batch active learning"** — genuinely sequential
  re-selection was not standard practice as of the field's own survey.
- Non-personalized battery **3–4 agent-hours**; personalized battery **8–12**; IGCN+DecisionTree **10–15**.

### 4.3 Elahi, Ricci & Rubens — ACM TIST 5(1) art. 13, 2014 **[A — full text CLOSED]**
DOI 10.1145/2542182.2542195. **Two claimed flaws in prior evaluation and the fixes, abstract verbatim:**
*"(1) Performance has been evaluated for each user independently (ignoring system-wide improvements).
(2) Active learning strategies have been evaluated in isolation from unsolicited user ratings (natural
acquisition)… we show that **using some common user-centric strategies may actually degrade the overall
performance of a system**."*
**This is a live methodological threat to Act 1** and cheap to guard: our protocol is per-user and
isolated from natural acquisition, i.e. exactly the configuration TIST'14 says can mislead. System-wide
harness ≈ **8–12 agent-hours**. **Verdict: CITE, and state our scope explicitly.**
Companions, all bibliographically verified, none open access: EC-Web 2011 (10.1007/978-3-642-23014-1_14);
ISMIS 2012 Adaptive Switching (10.1007/978-3-642-34624-8_30); AI*IA 2013 personality-based
(10.1007/978-3-319-03524-6_31, needs a TIPI questionnaire → **UNFIT on ML-25M**); EC-Web 2014
(10.1007/978-3-319-10491-1_12).

### 4.4 The Karimi / ISMLL line **[V ×2, [A] ×5]**
All seven verified to exist: CIDM'11 (10.1109/cidm.2011.5949431); IRI'11 (10.1109/iri.2011.6009563);
**ICTAI'11 pp. 1069–1076 (10.1109/ICTAI.2011.182) [V]**; RecSys'12 pp. 317–320 (10.1145/2365952.2366031);
ICTAI'13 FDT (10.1109/ictai.2013.67); LWA'14; UMUAI 25(1):39–64 2015 (10.1007/s11257-014-9153-z);
**LWA'15 CEUR-1458 pp. 171–180 [V]**. **No public code for any of them.**
- **ICTAI'11 criterion (Prop. 1, Eq. 13–14) [V].** Online-updating MF with item factors frozen; minimizing
  test MAE gives
  `m*_u ≈ argmin_{m∈P_u} Σ_{i∈M_u} | 1 − r̂_ui + 2η(r̂_um − R̄_m) Σ_f h_mf h_if |`,
  after approximating the unknown true test rating by 1 (worst case) and the unknown pool rating by the
  training mean. Adaptive in form; the candidate pool is fixed.
- **⚠ The load-bearing ICTAI'11 sentence, verbatim §IV-B:** *"The non-myopic method works well in the
  first queries but it finally converges to the random selection. This convergence also happens for
  active learning in AM [Jin & Si]. **Generally, this evidence holds for active learning methods aiming to
  improve the new user parameters using some heuristics.**"*
- **RecSys'12 abstract verbatim [A]** — the cleanest statement of the trade in the whole corpus:
  *"different from the classic active learning, users (the 'oracle') are not always able to provide an
  answer for queries. The easiest way to solve this problem is to ask most popular items… **But it is
  static and presents the same items to all users regardless of the ratings they have provided so far.**"*
  Their fix is to find similar users in the latent space and take the item most popular among them.
- **ICTAI'13 (FDT) abstract verbatim [A]** — where the adaptive gain actually lands: *"The sequence of
  queries should not be static… **FDT improves the accuracy of rating predictions especially in the last
  queries.**"* Also introduces **Most Popular Sampling (MPS)** — at each tree node only items popular
  *among the users at that node* are split candidates, *"increases the speed of the tree construction
  without harming the accuracy."* **That is a cheap, directly transplantable trick for our Golbandi tree
  build, and it is a published precedent for a candidate-pool restriction.**
- **UMUAI'15 [V via LWA'15]:** *"consider existing users as (hypothetical) new users and solve an active
  learning problem for each of them. In the end, we aggregate all solved problems in order to learn how to
  solve the active learning problem for a real new user."* — **the closest published relative of CASPER's
  clairvoyant-imitation line.** Worth reading before any "first" claim there.
- **LWA'15 [V]:** MF beats the Aspect Model as the AL substrate on accuracy and latency (retrain 3.9 s vs
  44.5 s on ML-100K).
- ICTAI'11 Eq. 14 **3–5 agent-hours**; FDT+MPS **12–18**; UMUAI framework **20–30**.
- **Verdict: NICE. Run the ICTAI'11 criterion if a cheap non-myopic MF arm is wanted; CITE FDT for MPS
  and UMUAI'15 for the imitation precedent.**

### 4.5 Harpale & Yang, SIGIR 2008 **[V, full text]**
DOI 10.1145/1390334.1390352, pp. 91–98. Aspect model; restates Jin & Si as a negated-KL objective, then
**multiplies by an answerability prior**:
`m*_u = argmax_m [KL objective] · P(m|u)`, `P(m|u) = Σ_z P(m|z)P(z|u)`,
`P(m|z) = [Σ_u P(z|u) I(u,m)] / [Σ_u Σ_m' P(z|u) I(u,m')]` with `I(u,m) = 1` iff u **rated** m — an
indicator, not a rating likelihood, *"because we are interested in computing the probability of getting
any rating."* So `P(m|z)` is a **class-conditional exposure prior**: a soft per-segment "have you seen
it" model. Described as "a soft-AND".
- **Their key methodological contribution is one we should adopt language from:** **constrained vs
  unconstrained** evaluation. Constrained (all prior work) restricts candidates to items the user
  actually rated — *"This is unrealistic, since the system will not know beforehand, which items will be
  rated by the user."* New metric **Failures** = solicited ratings not present in the data.
- Mean failures over 10 iterations: Random 9.07/9.33, Bayesian Selection 9.84/9.84, **Personalized BS
  5.995/5.127** (MovieRating / ML-100K). MAE/MSE are figures only. **1–2 agent-hours** given Jin & Si.
- **Verdict: MUST-CITE.** This is the published ancestor of our answerability gate, and its "Failures"
  metric is a ready-made name for the quantity our structural answerability rule controls.

### 4.6 Jin & Si, UAI 2004 **[V]** and Boutilier, Zemel & Marlin, UAI 2003 **[V]**
- **Jin & Si** (UAI 2004 pp. 278–285, **arXiv:1207.4146**): Dirichlet posterior over aspect memberships,
  selection by expected KL over **both** the predicted rating and the model posterior, closed form via
  digamma. **Answerability assumed away and said so:** *"For simplicity, we assume that the active user
  will always be able to rate the items presented… we will leave this issue for future work."*
  **⚠ Its own headline is negative and it is at OUR budget (3 seeds, 5 queries):** *"both the 'model
  entropy method' and the 'prediction entropy method' performs consistently worse than the simple 'random
  method'… As a result, the expected loss can not be computed accurately based on the estimated model."*
  **10–16 agent-hours.**
- **Boutilier et al.** (UAI 2003 pp. 98–106, arXiv:1212.2442; Harpale miscites it as SIGIR): myopic
  **EVOI** over an explicit probabilistic CF model,
  `EVOI(q) = Σ_{r_q} P(R_iq=r_q)·V(P(·|R_iq=r_q)) − V(P)`, with user-independent offline bounds propagated
  through a compact LP to prune targets, plus query prototyping. Prototype selection is **biased toward
  answerability** by ordering movies by rating count. **⚠ Its evaluation is not sequential** — footnote 5:
  *"held out ratings for **a predetermined sequence of items** were added… **the results do not reflect the
  sequential nature of the interaction**."* Metric is improvement in model loss; **figures only**; and
  *"for many users the maximum EVOI was negligible"*. **6–10 agent-hours** on an existing posterior.
- **Verdict: both CITE-only.** They are the EVOI/Bayesian ancestors of our belief-pool line and their
  negative results at k≤5 are load-bearing for Act 1.

### 4.7 Houlsby et al., NIPS 2012 + BALD **[V]**
- **Two separate papers.** Model: Houlsby, Huszár, Ghahramani, Hernández-Lobato, "Collaborative Gaussian
  Processes for Preference Learning", NIPS 2012 pp. 2096–2104. Acquisition: Houlsby, Huszár, Ghahramani,
  **Lengyel**, "Bayesian Active Learning for Classification and Preference Learning", **arXiv:1112.5745**.
- **BALD, exact.** Rewrites the mutual information in *output* space so it needs **O(1)** posterior
  updates instead of O(2n): `BALD(x) = H[P(y|x,D)] − E_{P(g|D)}[H[P(y|x,g)]]`, closed form under a
  Gaussian approximation with `C = √(π log2 / 2)`:
  `BALD(x) = h[Φ(μ_x(σ²_x+1)^{-1/2})] − C(σ²_x+C²)^{-1/2} exp(−μ²_x / (2(σ²_x+C²)))`.
  **The second term is what distinguishes BALD from Maximum Entropy Sampling**, which keeps only the first
  and therefore chases irreducible noise. **Requires a calibrated posterior variance** — a point-estimate
  belief layer cannot use it.
- Synthetic / Sushi / MovieLens / Election / Jura; test error on held-out pairwise judgements; **10
  acquisitions**. Verbatim: *"BALD always outperforms random sampling and usually outperforms or obtains
  equivalent performance to MES… BALD significantly outperforms MES in 9 cases, while MES is better than
  BALD in only 2 cases."* Second quotable finding: *"CP outperforms CPU in all cases except… Synthetic. In
  the real-world datasets, users with similar features do not seem to have similar preferences"* —
  **user side-features hurt on all four real datasets.**
- Code: github.com/neilhoulsby/pref_learning — **R + C, last commit 2012-12-24, 6 commits, no license.**
  Readable as a spec, not runnable in 2026.
- **⚠ [S] but important: no paper was found that uses BALD as the acquisition inside a recommender.** The
  BALD lineage runs through deep AL for classification (BatchBALD, k-BALD, GBALD, ConBatch-BAL) and never
  crosses into elicitation. Before claiming that gap, note (a) the criterion is four lines so the absence
  is probably about calibrated posteriors, not novelty, and (b) **Prediction-Oriented Bayesian Active
  Learning / EPIG (Bickford Smith et al., AISTATS 2023) [S]** argues BALD optimizes *parameter*
  information, which is misaligned with predictive/ranking performance — read that first.
- **BALD on a model that already emits (μ, σ²): 1–2 agent-hours.** Full CP/CPU with EP+VB+FITC: 30–60.
- **Verdict: NICE-HIGH.** One extra acquisition rule over `belief_mf` for ~2 hours, and it is the named
  Bayesian-AL criterion a reviewer from that community looks for.

---

## 5. The interaction-metric / simulator branch — UNFIT, and why the reason is stronger than pass 1 said

### 5.1 The Trinity College Dublin line (new to this pass)
- **Shams, Anderson & Leith, "Cluster-Based Bandits: Fast Cold-Start for Recommender System New Users",
  SIGIR 2021, DOI 10.1145/3404835.3463033** [V citation via dblp + the MCTS reference list; **full text
  FETCH FAILED**, ACM served an error page]. Arms are **user GROUPS**, not items; pulling an arm = asking
  about the item predicted highest for that group. Mechanism [S].
- **Rajapakse & Leith, "Fast and Accurate User Cold-Start Learning Using Monte Carlo Tree Search",
  RecSys 2022, DOI 10.1145/3523227.3546786** [V, full text extracted], extended as **"User Cold-Start
  Learning in Recommender Systems using Monte Carlo Tree Search", ACM TORS 3(1), 2023,
  DOI 10.1145/3618002** [A].
  - **Mechanism.** Users are clustered into |G| groups (**BLC matrix-factorization clustering**, Checco et
    al. TOPS 2017) with per-group per-item rating mean μ(g,v) and variance σ²(g,v) estimated from
    training data. The cold-start task is recast as **identifying which group the new user belongs to**:
    maintain `p(t)_g`, and choose the next item by a **single-player MCTS lookahead** whose rollout draws
    a group ĝ ~ p(t), synthesises ratings, re-estimates p, and **backpropagates reward +1 iff the argmax
    group is correct, else 0**, with UCB-style branch selection. Speedups: **"best distinguisher items are
    enough"** (Netflix 17,770 items → **269 / 759 / 1,224 / 905** distinguishers for 4/8/16/32 groups) and
    **no rollout** (omitting it does not hurt). Cost: thousands of cold-starts/second on one CPU core.
  - **⚠ HARD DISQUALIFIERS, and they are worse than "cumulative reward".** (i) **The metric is GROUP
    ESTIMATION ACCURACY** (Netflix/8: 0.691→0.830 over max_turns; Goodreads/8 0.564→0.703; Jester/8
    0.485→0.551) plus regret. **There is no held-out ranking at all, and no NDCG anywhere.**
    (ii) **MCTS is given the ground-truth group means and variances** — RLT4Rec says so explicitly:
    *"MCTS has the advantage of extra knowledge of the RL task (the number of user groups and the mean and
    variance of the item ratings in each group) that is not available to RLT4Rec."* It is an
    oracle-informed planner. **So pass 1's advice to "port the policy" does not apply here: porting it
    means granting it privileged information our firewall forbids.**
    (iii) The user is a **Gaussian draw from a clustered generative model**, i.e. a model-defined
    simulator of exactly the class `simulator_circularity_conv_rec.md` rules against.
  - Datasets: **Netflix, Goodreads10K, Jester** — no MovieLens. Baselines: optimised CART decision tree
    and the SIGIR'21 cluster-based bandit.
  - **Code: https://github.com/dilina-r/mcts-rec — C++, 1 star, last push 2022-09-21** (Makefile, `mcts/`,
    `data/`, 776-byte README).
- **Rajapakse & Leith, RLT4Rec, arXiv:2412.07403 (Dec 2024)** [V, full text — **authorship now VERIFIED**,
  closing pass 1's debt]. Sequential transformer taking `{(v_i, r_i)}` plus a target rating and emitting
  the next item. Datasets **Netflix, ML-10M, Goodreads10K + two synthetic "polarised" sets**; users again
  simulated from BLC group means; **metric = mean rating of the presented items over iterations**
  (e.g. Netflix MCTS 3.84→4.07 across 5 iterations). Baselines Best*(genie), MCTS, Random-Uniform,
  Random-Popular, CART D-Tree, PRL, Decision Transformer. Code
  **https://github.com/dilina-r/RLT4Rec — Python, 3 stars, last push 2024-12-13.**
- **Verdict for the whole line: UNFIT.** Cite in one sentence with the three reasons (group-identification
  metric, oracle-informed planner, model-defined simulated user). **Never table their numbers.** This
  replaces pass 1's "NICE-if-ported" for MCTS/RLT4Rec.

### 5.2 NICF, EAR/SCPR/UNICORN/ConTS, PEBOL
Unchanged from `adaptive_elicitation_baselines.md` §2 and §3.2 — all UNFIT, for the reasons recorded
there (cumulative-reward metric; omniscient target-attribute simulator; LLM-simulated preferences).
PEBOL's exact mechanism, for the record [V]: Beta-Bernoulli belief updated by NLI entailment scores
(`α += w`, `β += 1−w`), TS/UCB/entropy/greedy acquisition, GPT-3.5 generating the question, 10 turns,
**100-item catalogues**, MAP@10. Austin, Korikov, Toroghi & Sanner, RecSys 2024, arXiv:2405.00981,
DOI 10.1145/3640457.3688142.

---

## 6. Recent work 2021–2026 that pass 1 missed

### 6.1 De Pessemier, Willems & Martens, *Scientific Reports* 15 (2025) **[V]**
DOI 10.1038/s41598-025-09708-2. Golbandi-style binary tree, `E(j) = Σ_t e²(t)`. **3 likes, ≤20 queries** —
genuinely our budget regime.
- ML-1M: DT **P 0.4920 / NDCG 0.5032** vs MostPop 0.3218 / 0.3083.
- **Publiq (sparse, density 0.01%): DT 0.0680 / 0.2585 vs MostPop 0.0789 / 0.3009 — the static baseline
  wins on BOTH metrics.**
- Online with 50 real users: *"online evaluation with 50 real users could not prove that our algorithm
  does indeed have a positive impact"* (0.52 vs 0.49 precision, 0.796 vs 0.793 NDCG, n.s.).
- **10–15 agent-hours. Verdict: MUST-CITE.** It is a 2025, peer-reviewed, *Nature*-family paper reporting
  that an adaptive Golbandi tree fails to beat Most-Popular on a sparse catalogue and fails to separate
  online. It is the single best external cover for Act 1's null.

### 6.2 De Pessemier, Vanhove & Martens, "Batch versus Sequential Active Learning for Recommender
Systems", ORSUM@RecSys 2021 / arXiv:2201.07571 **[V]**
Literally the static-vs-adaptive comparison, **budget = 7 ratings**, 5 active learners × 3 predictors.
- *"sequential mode produces the most accurate recommendations for **dense** data sets"*; *"For most
  active learners, the best predictor turned out to be FunkSVD in combination with sequential mode."*
- But also: *"**Differences between the active learning algorithms are small**"*, *"**There is no clear
  winner** of the active learning algorithms"*, and on BookCrossing *"the results obtained with active
  learning in sequential mode are **not significantly different** than with the active learning in batch
  mode."*
- **Verdict: MUST-CITE.** Together with §6.1 it gives us the moderator: **adaptivity pays on dense data
  and vanishes or reverses on sparse data** — a cleaner, checkable story than "adaptive vs static", and
  directly testable in our harness.

### 6.3 Gharahighehi, Nakano, Yang, Cu & Vens, arXiv:2510.27342 (31 Oct 2025) **[V]**
(Pass 1 had this as "(Cu et al.)" — the full author list is above.) Ternary Golbandi tree,
`E(j) = e²(t_L)+e²(t_H)+e²(t_U)`, extended two ways: **attribute-aware nodes** (ask about genres/artists/
albums, not just tracks) and **pairwise nodes** (present an item pair; branches prefer-first /
prefer-second / indifferent; `pairwise_tree_1` pairs the top-2 lowest-error items, `pairwise_tree_2` pairs
the best with its cosine-nearest neighbour in the top-20). **Tree retrained every iteration.**
Yahoo! Music 1,701 u / 4,793 artists / 218 genres, **RMSE**, 20 iterations, baselines popularity /
variance / entropy0 / **HELF** / IGCN. Code **github.com/camilleecu/master_thesis** branch `pairwise` —
Jupyter Notebook, GPL-3.0, last push **2025-09-13**, 1 star. (Note: the bare repo name 404s on the GitHub
API; use the full URL.)
- **⚠ ITS CENTRAL FINDING IS OURS, INDEPENDENTLY, verbatim:** *"the decision tree-based approach that
  elicits users' preferences on both artists and genre (tree_hybrid) provides more informative input to an
  artist RS compared to the best non-personalized approach, helf0_hybrid. **Starting from iteration 13,
  the decision tree-based model that elicits preferences only on artists outperforms the hybrid
  approach.** This implies that once we have sufficient knowledge about the users, it is more effective to
  elicit preferences on the target item type rather than on item attributes. However, since cold-start
  users are typically reluctant to answer too many questions during RE procedure, the proposed tree_hybrid
  approach appears to perform best for such users."*
  **That is a published, independent, 2025 statement of concepts-win-early / items-win-late, on a
  different dataset, metric and model.** It both corroborates our coarse-to-fine result and **pre-empts
  any claim that the crossover is novel** — we must cite it and claim the *mechanism* (decoupling
  discrimination from answerability, measured on a ranking metric at every budget with CIs), not the
  phenomenon.
- **15–20 agent-hours (mechanism only). Verdict: MUST-CITE, NICE-to-port. Never table its RMSE.**

### 6.4 Zuo, Hu, Yu, Li, Zhao & Joe-Wong, "Hierarchical Conversational Preference Elicitation with Bandit
Feedback", arXiv:2209.06129 **[A]**
Hier-UCB / Hier-LinUCB choose **per turn** between a **key-term (category) question** and an item
recommendation, with a regret bound arguing categories buy sample efficiency.
**The closest formal analogue of our concept-vs-item channel choice**, and the only one with a bound.
**Verdict: MUST-CITE** (Paper B/C). Not a baseline (bandit-regret contract).

### 6.5 Others found and triaged
- **Cantürk & Aydoğan, "Explainable Active Learning for Preference Elicitation", arXiv:2309.00356** [V].
  Uncertainty / diversity / clustered-uncertainty sampling + label spreading + user-corrected feature
  explanations. Recipe domain, MCC, budgets 30–80. Code
  github.com/furkancanturk/explainable_active_learning. **NICE-cite.**
- **Qian, Wang & Zhao, ESWA 2025, DOI 10.1016/j.eswa.2025.128765** [S, ScienceDirect 403]. PLCS —
  personalizes **when the interview stops**, plus a "Variance Via Prediction" metric replacing entropy
  (+5.6% on Coat). **The stopping-rule idea is the obvious reviewer question against a fixed-k protocol.**
  **Verification debt.**
- **Iovine et al., JIIS 2021, DOI 10.1007/s10844-021-00683-4** [S, Springer IdP redirect failed].
  5 AL techniques, **219-user study**; a search paraphrase reports *"non-personalized techniques have a
  positive effect on both the recommendations quality and interaction costs"*. **Do not cite until read —
  highest-value unread item on the static side.**
- **Dureddy & Kaden, "Handling Cold-Start Collaborative Filtering with Reinforcement Learning",
  arXiv:1806.06192 (2018)** [A]. Composite Deep Q-Network learning an interview, framed against decision
  trees. Two-author preprint, no venue, no code, datasets/metrics unverified. **UNFIT (cite at most).**
- **"Information gain based dynamic support set construction for cold-start recommendation" (2023)**,
  **"High Accuracy and Low Regret for User-Cold-Start Using Latent Bandits" (ESANN 2023)**,
  **"Variational Factorization Machines for Preference Elicitation" (IEEE BigData 2022)**,
  **"Reinforcement learning for addressing the cold-user problem" (KBS 2024)**,
  **"Active learning algorithm for alleviating the user cold start problem" (Sci Rep 2025)**,
  **"Cold-Start Personalization via Training-Free Priors from Structured World Models" (arXiv 2026)** —
  all surfaced from the Golbandi citation graph, all **[S] title/venue only**, none verified. Listed so
  the space is documented as swept, not as recommendations.
- **NISS 2024, DOI 10.1145/3659677.3659838** [S] — notable only for running the full Elahi battery
  verbatim in 2024, i.e. more external validation that our arm bank is current.

---

## 7. THE CONSENSUS QUESTION — does adaptive beat a well-chosen static seed set at k ≤ 16?

**No. There is no consensus, the decisive experiment is essentially never run, and at small budgets the
evidence leans the other way.** Three structural facts:

1. **The founding papers never compare against a static seed set at all.** Boutilier, Jin & Si, Harpale
   and Karimi all use **random per-user selection** as the reference. The nearest thing to a static
   baseline in the classic corpus is Boutilier's population-entropy ordering, and *"the entropy-based
   method is virtually indistinguishable from the random method with respect to model loss improvement."*
2. **The "static seed sets are bad" claim traces to an unsupported assertion in the 2011 Handbook
   chapter** (§4.1), repeated as motivation ever since, most recently by PERE 2024. Motivating prose, not
   measurement.
3. **The head-to-head best-static-vs-best-dynamic comparisons that do exist both go to static**
   (SPQ 2018, Greedy SLIM 2024).

**Adaptive side (all [V] unless marked):** Boutilier — EVOI beats random and entropy *"particularly for
small values of |κ|"* (but its evaluation is **not sequential**, footnote 5). Harpale — personalization is
essential, though the benefit measured is **failures 9.84 → 5.13, i.e. answerability, not accuracy**.
Karimi ICTAI'13 [A] — FDT wins *"especially in the last queries"*. De Pessemier ORSUM'21 — sequential
beats batch *on Jester, the dense set*. Houlsby — BALD beats random (random = from that user's 35-pair
pool, not an optimized static set).

**Static / no-difference side — these are the strong ones:**
- Jin & Si §4.2, at 3 seeds and 5 queries: *"both the 'model entropy method' and the 'prediction entropy
  method' performs consistently worse than the simple 'random method'."*
- Harpale §4.6–4.7: *"since Bayesian Selection does not take the probability of getting a rating into
  account, **it performs even worse as compared to plain Random Selection**"*; *"Random Selection performed
  better than Bayesian Selection with statistically moderate evidence."*
- Karimi ICTAI'11 §IV-B: heuristic AL *"finally converges to the random selection… Generally, this
  evidence holds for active learning methods aiming to improve the new user parameters using some
  heuristics."*
- Sepliarskaia 2018: *"All algorithms provide the same performance for questionnaires with a length less
  than 5"*, and SPQ (static) beats the dynamic baselines at longer lengths.
- **Greedy SLIM 2024, ML-25M, NDCG@10, k=20: static 0.3752 vs dynamic 0.2916; tail 0.1779 vs 0.1466.**
- De Pessemier ORSUM'21: *"Differences between the active learning algorithms are small"*, *"There is no
  clear winner"*, BookCrossing sequential-vs-batch *"not significantly different"*.
- De Pessemier Sci Rep 2025: adaptive DT loses to Most-Popular on the sparse Publiq catalogue on both
  precision and NDCG; online with 50 users, *"could not prove that our algorithm does indeed have a
  positive impact"*.
- Yu, Bi & Tresp ICML 2006: *"choosing the most popular movies does not bring advantages over random
  guessing… the most popular movies are rated highly by nearly all the users and thus give no information
  about user tastes."*

**What we can defend, and how to phrase it:**
- **Budget is confounded with everything.** Every clean adaptive win uses a budget far outside k ≤ 16
  (PERE 100; Cantürk 30–80; GSLIM's opponent 20; Karimi's gain is "in the last queries") or a toy
  catalogue (PEBOL, 100 items). Every paper *at* k ≤ 16 reports adaptivity indistinguishable from, or
  worse than, a static baseline.
- **Sparsity is the real moderator.** Two independent De Pessemier papers find the same pattern:
  adaptivity pays on the dense dataset and vanishes or reverses on the sparse one. That is a cleaner and
  more checkable claim than "adaptive vs static", and it is directly testable in our harness.
- **Adaptivity's demonstrated wins are mostly against *random*, not against a *well-chosen* static set.**
- **The founding papers' rescue was never better adaptivity — it was conditioning on answerability.**
  Boutilier biased his prototype set by popularity and listed answerability modelling as future work;
  Jin & Si assumed it away and said so; Harpale's entire contribution is the multiplicative `P(m|u)`; the
  2016 survey justifies personalization on ratability, not informativeness; Rashid'02 killed pure entropy
  on exactly this. **That is a 24-year citable lineage terminating in our concept channel.**
- **Never write "selection doesn't matter" or "static is optimal"** — the standing project rule. The
  defensible sentence is: *at k ≤ 16 on a sparse catalogue, no published adaptive elicitor has been shown
  to beat a well-optimized static questionnaire, and two 2024–2025 papers report the reverse.*

---

## 8. Contract-violation flags — never order these numbers against ours

| Family | Their metric | Why not orderable |
|---|---|---|
| Rashid'02/'08, Golbandi'10/'11, fMF, Sun'13, Karimi, 2025 trees | **RMSE / MAE / Expected Utility** | Rating prediction, not ranking. **The entire classical lineage.** |
| TCD line (CBB, MCTS, RLT4Rec) | Group-estimation accuracy; mean rating of presented items | No held-out ranking; the planner is given ground-truth group statistics; the user is a model-defined draw. |
| NICF | Cumulative Precision/Recall/α-NDCG@T=40 | Scores the probes themselves as recommendations. |
| EAR/SCPR/UNICORN/ConTS/CRM | Success Rate@turn, Average Turns | Known target item + omniscient attribute simulator. |
| PEBOL | MAP@10 / MRR@10 on 100-item catalogues, LLM-simulated users | Catalogue scale, answer source and metric all differ. |
| Sepliarskaia | PCPF and P@10, pairwise questions, figures only | Different question type and metric; no numeric tables exist. |
| Anava 2015 | — | **Contains no numbers at all.** Cite the formulation only. |
| RBMF, RMVA, DRE, PERE, Greedy SLIM | P@k / NDCG@k on ML-10M/20M/25M | **Closest** — but different splits, different binarization, budgets 20–100. Re-run on our split; do not cite their published numbers as a bar. |

---

## 9. Effort ledger (agent-hours) and the recommended build order

| Method | Adaptive? | Effort | Verdict |
|---|---|---|---|
| A-optimal acquisition (Anava/Sepliarskaia/Yu) over `belief_mf` | static-optimal | **4–8** | MUST |
| BALD acquisition over `belief_mf` | adaptive | **1–2** | NICE-HIGH |
| RBMF maxvol selector | static | **2–4** | MUST |
| RMVA | static | **3–6** | MUST |
| DPE topic-Thompson over genome concepts | adaptive | **2–3** | MUST |
| DRE | static | **3–5** | MUST |
| Greedy SLIM | static | **6–10** (+6 h compute) | **MUST — strongest static bar** |
| Golbandi adaptive TREE selector (the missing half) | adaptive | **6–12** | **MUST — the real gap** |
| IGCN, genuine two-phase 150-NN form | adaptive | **4–7** | MUST |
| fMF | adaptive | **6–12** | MUST |
| Karimi ICTAI'11 non-myopic criterion | adaptive | **3–5** | NICE |
| PERE | adaptive | **10–20** | MUST (demoted) |
| GreedyExtend (CIKM'10) | static | **4–8** + heavy compute | NICE-HIGH |
| Elahi non-personalized battery | static | **3–4** | NICE (mostly already ours) |
| Sun WSDM'13 multi-question | adaptive | **30–55** | UNFIT (no numbers to snap to) |
| LRMF | adaptive | unknown | blocked on a fetch |

**Build order.** (1) Golbandi tree selector — it is the named adaptive anchor and we currently have only
its leaf half. (2) A-optimal + BALD acquisitions — two afternoons on machinery we already own, and they
turn "we tried one adaptive method" into "we tried the design-theoretic and the information-theoretic
families." (3) Greedy SLIM + DRE + RMVA + RBMF as the static bar. (4) IGCN and fMF. (5) DPE for the
concept channel. (6) PERE last, and only if the reviewer needs a 2024 name.

---

## 9b. Late additions — 2024–2026 cross-checks and the dormancy measurement

### 9b.1 De Pessemier Sci Rep 2025, mechanism detail (supersedes §6.1's sketch) **[V, via PMC12238261]**
The tree is **TWO-WAY, not ternary**: children are *users who like the item* and *users who cannot rate
it*, justified verbatim — *"by ignoring the negative evaluations, only a small fraction of information is
omitted."* Node prediction is the same shrunk mean as Golbandi (λ=200). Candidates at each node are
restricted by **Most Popular Sampling** (Karimi's ICTAI'13 trick, §4.4). The recommender is treated as a
**black box** via **RecPack** — ItemKNN, **EASE**, SLIM, WMF, TEASER — so the elicitation/recommender
separation is exactly ours. Metrics Precision@8, Recall, F1, **NDCG@8**. Budget: **collect exactly 3
likes, max 20 queries**, chosen as "the estimated patience a real user would display".
**⚠ Their stated cause for the online null is our answerability thesis, verbatim:** *"Real users are not
always able to rate the item selected by the active learning algorithm and therefore cannot provide the
requested information, in contrast to many machine learning scenarios where the labeling of all samples
is possible."* And: *"This observation can be caused by (wrong) assumption that users can accurately rate
an item if they are provided with a description of this item."*
**No code, no data** (IP + participant privacy). **6–10 agent-hours.**
**This is the single strongest external support the concept channel has**, and it retires the reviewer
objection "Golbandi is your most recent adaptive anchor".

### 9b.2 COPE / InPE — Xia, Zhang & Wang, arXiv:2607.06765, **accepted SIGIR 2026** **[A]**
LLM/CRS, so not a baseline. **But its empirical claim is on our exact axis:** *"attribute-based inquiries
are effective in early stages, while item-based strategies become superior as preferences refine."*
Dataset public at https://github.com/juanfacabian/InPE.
**Together with Gharahighehi 2025 (§6.3) this makes the attribute-early / item-late crossover a
TWICE-REPLICATED, 2025–2026 published finding.** Our concept result — concepts winning at *every* budget
under a pure-discrimination bank — is therefore **a contrast with the field, not a confirmation of it**,
and must be framed and defended as such rather than presented as novel.

### 9b.3 "Pep" — Bose et al., arXiv:2602.15012 (Feb 2026) **[A]**
LLM alignment, not recsys ranking, so not a baseline. **One transferable, quantified finding:**
*"in practice learned [RL] policies collapse to static question sequences that ignore user responses"* —
measured as follow-up variation when two users answer differently: **Pep 39–62%, RL policies 0–28%.**
An independent 2026 measurement that RL elicitation policies are effectively static. **CITE in Act 1's
discussion** — it is external support for the framing, from outside our field.

### 9b.4 Bachmann, Sarasua & Bernstein, arXiv:2404.01872 (ECML/PKDD 2024) **[A]**
"Fast and Adaptive Questionnaires for Voting Advice Applications". Wrong domain (Swiss Smartvote), but
**the architecture is ours**: encoder+decoder impute missing answers at any completion stage into a
latent space, plus a **selector module** choosing the next question from the user's latent position.
Best config = IDEAL encoder/decoder + **PosteriorRMSE** selection, **74% accuracy vs <40% for the fixed
condensed questionnaire**. **CITE as an out-of-domain precedent that a posterior-based adaptive selector
beats a fixed short form** — it is the cleanest counterexample to our own null and should be engaged,
not buried.

### 9b.5 Boroomand & Wright, arXiv:2508.14911 (Aug 2025) **[A]**
"Personalized Recommendations via Active Utility-based Pairwise Sampling". Plackett-Luce over pairwise
comparisons, model-agnostic, active sampling picking the query that maximizes **expected improvement in
the utility of the final recommendation** — i.e. **rank-targeted acquisition**. Our
`rank_targeted_acquisition.md` currently says only Lin WWW'23 targets the ranking metric directly;
**this is a second instance and that file needs updating.** No code found. Pairwise queries, so
mechanism-only for us.

### 9b.6 ⚠ THE DORMANCY MEASUREMENT — a usable sentence for the paper
Citation counts enumerated from Semantic Scholar this pass: **DRE (WWW'20) has 6 citing papers in total,
of which only PERE is an elicitation paper. PERE (UAI'24) has 1. Greedy SLIM has 0. The MCTS/TORS paper
has 7, none of them elicitation methods.** The rating-elicitation family is **quantitatively dormant**,
and that is a defensible sentence: *"the post-2020 rating-elicitation literature consists of a handful of
papers that barely cite one another; we run all of them."* It also means our baseline battery can
credibly claim to be exhaustive rather than selective.

### 9b.7 Verification debts still open after this pass
1. **LRMF (TOIS 2017)** — §3.6. Needs institutional access. Highest-value unread *method*.
2. **Golbandi CIKM'10's exact Coverage formula** — §2.4. Not reproduced in any secondary source reachable,
   including the Yahoo patent.
3. **Iovine et al., JIIS 2021** (219-user study, reportedly "non-personalized techniques have a positive
   effect") — highest-value unread item on the *static* side.
4. **Qian, Wang & Zhao, ESWA 2025** (10.1016/j.eswa.2025.128765) — PLCS personalizes *when the interview
   stops*. **The stopping rule is the obvious reviewer question against our fixed-k protocol.**
5. Unverified [S] titles from the Golbandi citation graph: KBS 2024 cold-user RL; JIIS 2023 information-gain
   support-set construction; ESANN 2023 latent bandits; IEEE BigData 2022 Variational FM.
6. **Kohrs & Merialdo ICMF'01** — no retrievable copy anywhere.
**CLOSED this pass:** the MCTS RecSys'22 full text (extracted directly from the ACM PDF — §5.1) and
RLT4Rec's authorship, both of which earlier passes and sub-agents recorded as blocked.

---

## 10. THE TWO NOMINATIONS

**Strongest STATIC bar as of 2026: GREEDY SLIM (Proissl et al. 2024).** On our exact dataset, our exact
metric and our exact tail definition; prefix-optimal so one fit yields every budget; beats an adaptive
bandit by a wide margin at k ≥ 10 on both full and tail. Its weakness is provenance — arXiv-only, authors
outside the field, a weak user study. **Therefore run DRE (WWW 2020) alongside it as the peer-reviewed,
publicly-implemented static bar.** If our claim is "no uplift over the strongest static", these two
together are what makes that claim falsifiable.

**Strongest ADAPTIVE bar as of 2026: the GOLBANDI WSDM'11 TREE, in its genuine split-search form.**
This is not nostalgia. The 2024–2025 literature still benchmarks against it (Gharahighehi 2025 rebuilds
it; De Pessemier Sci Rep 2025 rebuilds it; RLT4Rec uses a CART tree as its baseline); fMF beats it by
0.01–0.02 RMSE with no CIs; Sun'13 beats it only after 6–7 questions and reports no numbers; PERE's
margin over its own baseline is under 3% and its headline claim was fabricated in secondary sources; the
MCTS line beats it only under an oracle-informed simulator on a group-identification metric. **No method
in 15 years has a clean, contract-compatible, replicated win over the ternary squared-loss tree at k ≤ 16.**
Running it properly is both the cheapest and the most defensible adaptive bar we can field — and it is
currently the largest hole in our harness. **Second nomination, if a modern name is required: PERE
(UAI 2024)** — cite it as the field's 2024 assertion of Act 1's thesis, and report its real margins.
