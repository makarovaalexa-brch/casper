# Findings: Evaluating SIGN-AWARE recommenders — is a "held-out dislikes + avoidance metric" protocol already published?

Deep-research pass 2026-07-29. Feeds **Paper A** (the sign-aware instrument on the Liang protocol) and the
metric-bridge obligation. Reads on top of `graded_inputs_for_ranking.md` (the graded-INPUT question — Cremonesi/
HKV/Steck/Frolov/OrdRec line — is settled there and NOT re-derived here) and `paperA_recommender_landscape.md`.

**The design under test:** hold out a fraction of each user's sub-threshold (disliked) items alongside the
held-out likes, as labelled negatives absent from *every* arm's input; report, next to NDCG@10, a
dislike-avoidance metric (held-out-dislike hit-rate in top-K, lower-better, or NDCG with negative gain on those
items). Claimed property: no filter can game it because the items were never observed; only generalisation from
signed input can move it.

**One-line verdict:** the metric concept is **PUBLISHED TWICE under two names** — Frolov & Oseledets's **nDCL**
(RecSys 2016, held-out low-rated items penalised by rank, lower-better) and Sánchez & Bellogín's
**anti-relevance / false-positive metrics** (RecSys 2018; TOIS 2021 fallout / anti-precision / anti-nDCG line) —
so we must **adopt and cite, not christen**. What is genuinely unoccupied: (i) attaching the avoidance probe to
the **Liang strong-generalization binarized protocol** next to the canonical NDCG@10 frontier (every paper on
that frontier, and even the 2022–24 *sign-aware* GNN line, evaluates on held-out **positives only**); (ii) the
explicit **anti-gaming argument** (injected *observed* dislikes are guaranteed non-targets → demotion is a free
lift, so the probe items must be held out); (iii) using the probe to certify that a **frozen tower consuming
signed input generalises** dislike, rather than to compare bespoke negative-feedback models. Known pitfalls that
must be handled: false-positive metrics carry their own **popularity bias** (TOIS 2021), threshold sensitivity,
and observed dislikes are themselves MNAR.

Verification tags: [V] = primary text fetched/read; [A] = abstract/secondary source verified this pass;
[K] = well-established background knowledge, not re-verified from a primary source this pass.

---

## 1. THE CRUX — Frolov & Oseledets 2016 already built almost exactly this protocol (nDCL)

`frolov2016fifty` (RecSys'16, arXiv:1607.04228) is not only the graded-INPUT precedent (see
`graded_inputs_for_ranking.md`); its **evaluation section is the closest published prior art to our design**.
Verified from the full text [V]:

- **Holdout construction:** user-disjoint 80/20 split; for each test user a **fixed number of items (10) is held
  out**, and — the load-bearing sentence — *"we allow both relevant and irrelevant items in the holdout set."*
  Held-out low-rated items are absent from the observation (fold-in) set. This IS "held-out labelled negatives
  no arm has observed."
- **The metric:** **nDCL** (normalized Discounted Cumulative Loss), Eq. 11:
  `DCL = Σ_n (2^{-r_n} − 1) / (−log₂(n+1))` over held-out items with `r_n ≤` a negativity threshold;
  `nDCL = DCL / iDCL` where iDCL is the ideal (negatives ranked last); *"the lower are the values of nDCL the
  better."* Exponential penalty grows as the rating falls. This is precisely "NDCG with negative gain on held-out
  dislikes," normalized to stay bounded.
- **Negativity threshold:** 3 on ML-1M, 3.5 on ML-10M — i.e. the same >3.5 boundary our Liang split uses.
- **The insensitivity argument (their §4), which pre-empts half our motivation:** *"an algorithm that recommends
  3 positively rated and 7 negatively rated items will gain the same evaluation score as an algorithm that
  recommends 3 positively rated and 7 items with unknown ratings."* Standard top-N metrics reward retrieving
  likes and are blind to surfacing known-bad items.
- **Negative-only cold start:** test users whose observation set is 1 or 3 *negatively rated* items only — the
  published template for "what does a dislike answer buy you" (their model recommends from dissimilarity to the
  disliked item; item-indicator baselines cannot).
- **Pitfall they flag:** unrated items in the recommendation list are *ignored*, "not marked as false positive,"
  to avoid over-estimating the false-positive rate — the same unknown-vs-bad ambiguity the anti-metric line
  wrestles with (§2).

**Difference from our design:** Frolov's holdout is a fixed-size mixed set (10 items, any rating) on ML-1M/10M
with weak-generalization-style tensor fold-in; ours holds out a *fraction of the sub-threshold items* on the
**Liang strong-generalization binarized split** where dislikes are additionally absent from all baselines' train
data by construction. The metric idea itself, including lower-is-better rank-discounted penalty on held-out
dislikes, is his. **We must cite nDCL as the metric's origin and present ours as a variant of it on the
canonical protocol** (hit-rate@K of held-out dislikes = the un-discounted special case).

## 2. THE SECOND PRECEDENT — anti-relevance / false-positive metrics (Bellogín–Castells line)

An independent, *named* evaluation framework doing the same thing:

- **`sanchez2018antirelevance`** — Sánchez & Bellogín, "Measuring anti-relevance: a study on when recommendation
  algorithms produce bad suggestions," RecSys 2018 short [V, PDF read via summary]. Held-out test set keeps
  negatively-rated items; items below a rating cutoff are **anti-relevant**; they define **anti-precision**
  (fraction of recommended items that are anti-relevant), **fallout**, and an **anti-nDCG** that penalises
  anti-relevant items by rank. Findings: true-positive and anti-metrics produce **different algorithm
  orderings**; **non-personalized recommenders return fewer bad recommendations than personalized ones** (but
  more unknowns); results are **sensitive to the anti-relevance threshold and cutoff**.
- **`menamaldonado2021fpbias`** — Mena-Maldonado, Cañamares, Castells, Ren, Sanderson, "Popularity Bias in
  False-Positive Metrics for Recommender Systems Evaluation," TOIS 39(3):36, 2021 [A; 43-pp primary not fully
  read]. The false-positive metric family (fallout = anti-recall, anti-precision) is analysed for **its own
  popularity bias**, via the joint distribution of item popularity, average rating, and global relevance. The
  companion earlier study is Mena-Maldonado et al., "Agreement and Disagreement between True and False-Positive
  Metrics in Recommender Systems Evaluation" (SIGIR 2020) [A]. **Binding pitfall for us:** a held-out-dislike
  hit-rate is NOT automatically fair across models — popular items accumulate more observed dislikes (exposure),
  so popularity-avoiding models get a free ride on the anti-metric exactly as popularity-chasing models get a
  free ride on recall. Report the avoidance metric with a popularity-stratified or tail split, mirroring our
  full+tail NDCG discipline, and say why.
- **IR import of negative grades:** **`gienapp2020negndcg`** — Gienapp, Fröbe, Hagen, Potthast, "The Impact of
  Negative Relevance Judgments on NDCG," CIKM 2020 short [A + snippet]. TREC Web Tracks 2010–2014 used a
  **junk/spam grade of −2**; with negative gains **NDCG becomes unbounded**, and standard implementations
  **clamp negative labels to 0, discarding the information** (spam scored like unjudged). Their fix:
  **min–max normalization against the worst possible ranking (nDCG_min)** retains statistical power. Directly
  relevant if we implement "NDCG with negative gain": either clamp (wrong), or normalize against the worst
  ranking as they prescribe, or use the separate lower-better nDCL/hit-rate form (cleaner; our preference).
- **Beyond-accuracy tradition:** Kaminskas & Bridge (TiiS 2016) [K] covers diversity/serendipity/novelty/
  coverage — it does **not** contain a dislike-avoidance objective; no help there beyond citing that
  "beyond-accuracy" ≠ "negative-aware."

## 3. PROTOCOLS THAT RETAIN NEGATIVES — what exists besides the two precedents

- **Unbiased / fully-observed test sets (the cleanest "labelled negatives no model observed"):**
  **Yahoo! R3** (Marlin & Zemel 2009 [K]) — test ratings collected on **randomly assigned** songs, so the test
  set contains MAR positives *and* negatives; **Coat** (Schnabel et al. 2016 [K]) likewise; **KuaiRec** (Gao et
  al., CIKM 2022 [K]) — a **fully-observed** user×item watch-ratio submatrix. These are the field's answer to
  "evaluate on negatives without exposure gaming," and the debiasing line (§5) evaluates on them. They pre-empt
  the *idea* that credible negative evaluation needs negatives outside the model's input; none of them is a
  full-catalogue top-N avoidance protocol on a MovieLens-scale binarized frontier.
- **Sequential/implicit negative-feedback line:** Google's RecSys 2023 paper (§7) trains on dislikes/skips and
  measures **responsiveness**, not held-out-dislike ranking. Music: "Enhancing Sequential Music Recommendation
  with Negative Feedback-informed Contrastive Learning" (arXiv:2409.07367, RecSys-adjacent 2024) reports
  **skip down-ranking scores** — how far items similar to skipped tracks fall [A]. Recommendation *editing*
  ("Better Late Than Never: Formulating and Benchmarking Recommendation Editing," arXiv:2406.04553 [A])
  formalizes post-hoc suppression of known-bad recommendations with editing success/locality metrics — the
  "filtering" arm of the problem, useful as the contrast class for our anti-gaming argument.
- **Graded-holdout explicit top-N:** beyond Frolov and Sánchez–Bellogín, the standard explicit-feedback top-N
  papers (Cremonesi one-plus-random etc.) hold out only high-rated items as targets [K] — negatives discarded,
  same blindness as Liang.

## 4. THE GAP THE PROBE FILLS — even the SIGN-AWARE model line evaluates on positives only

This is the strongest support for running the probe at all:

- **SiReN** (Seo, Jeong, Shin, TNNLS 2022, arXiv:2108.08735) — signed bipartite GNN consuming negative edges —
  evaluates with **P@10 / R@10 / nDCG@10 over held-out positives only** [A].
- **SIGformer** (Chen et al., SIGIR 2024, arXiv:2404.11982) — sign-aware graph transformer; verified from full
  text [V]: *"Two widely-used metrics Recall@K and NDCG@K are employed"*; positives = rating > 3.5 (or KuaiRec
  watch-ratio ≥ 4 / KuaiRand is_click); **no metric measures whether disliked items rank low**. The negatives'
  value is demonstrated only *indirectly*, as a lift in positive-ranking accuracy.
- Same pattern across the negative-sampling survey (Ma et al., arXiv:2409.07237 [A]) and the sign-aware
  successors (SiGR 2025 [A]).

**Reading:** the 2021–2025 sign-aware modelling literature adopted signed *inputs* but not Frolov's signed
*evaluation* — nDCL/anti-metrics never became standard. A paper that pairs the canonical Liang NDCG@10 with a
held-out-dislike avoidance probe is filling a real, documentable protocol gap, and can say so with the SiReN/
SIGformer evaluation sections as evidence.

## 5. THE EVALUATION DEBATES THAT BEAR ON THE DESIGN

- **Sampled metrics:** Krichene & Rendle, KDD 2020 [K] — sampled ranking metrics are inconsistent with
  full-catalogue ranking and can reorder systems. Our probe must be **full-catalogue rank of the held-out
  dislikes** (it is), never "rank the dislike among 100 sampled items."
- **MNAR / propensity:** Steck 2010 (`steck2010training`), Schnabel et al. ICML 2016 IPS
  (`schnabel2016recommendations`, propensity-weighted evaluation on Yahoo R3/Coat), Yang et al. RecSys 2018
  unbiased offline evaluation [K]. Consequence for us: **observed dislikes are MNAR** — users rate 1–2★ mostly
  on *popular, exposed* items. So the held-out-dislike pool over-represents head items; a model that simply
  demotes the head scores well on raw avoidance. Combined with the TOIS-2021 popularity-bias result (§2), the
  mitigation is: (i) report avoidance full **and** popularity-stratified/tail; (ii) report the likes-NDCG next
  to it so head-demotion shows up as an accuracy loss; (iii) optionally an IPS-weighted variant as a robustness
  check, citing Schnabel.
- **Baselines:** Rendle et al. / Dacrema et al. warnings (`dacrema2019progress`) apply unchanged — the avoidance
  probe needs its own floors (random, most-popular, and *least-popular/most-obscure* as the degenerate
  avoidance-maximizer).

## 6. CONVERSATIONAL / CRITIQUING / ELICITATION — how negative feedback is valued there

- **Critiquing & CRS standard:** simulated user with a known target item; metrics = **success rate @ turn
  budget (SR@t), average turns (AT)**, plus variants (SRRR, reward-per-dialogue-length) [A: UserSimCRS v2
  arXiv:2512.04588; KG-CRS lines]. A negative critique/answer is credited **only through faster convergence to
  the positive target** — there is **no dislike-avoidance metric** in the standard CRS protocol.
- **PosNeg critiquing** (`antognini2022posneg`, already in INDEX): evaluates ranking of the target after
  sequences of positive AND negative critiques; finds negative-critiquing gains plateau ~5 turns. Again
  target-retrieval, not avoidance. BCIE (`toroghi2023bcie`) likewise evaluates hit/MRR of a target after
  critiques [K].
- **Marginal value of a "no" answer measured separately from a "yes": NOT FOUND as an established protocol.**
  Searched directly; nothing isolates it. Nearest precedents: (i) **Frolov's negative-only cold-start
  condition** (observation set = 1 or 3 disliked items → can the system still recommend well and avoid
  similars) — the best published template; (ii) Golbandi's ternary trees (`golbandi2011adaptive`), where the
  hater branch exists but its marginal value is never reported separately; (iii) the CRS ablation habit of
  toggling negative-feedback handling on/off (SCPR's rejected-attribute pruning) which measures it only through
  SR@t. **A per-question signed-value decomposition (what a dislike answer buys, at matched question budget) is
  unoccupied** — flag as a defensible measurement for the elicitation papers, framed against Frolov's
  negative-only condition.

## 7. INDUSTRIAL PRACTICE — what is actually reported next to accuracy

- **Google/YouTube** — Wang et al., "Learning from Negative User Feedback and Measuring Responsiveness for
  Sequential Recommenders," RecSys 2023 industry track (arXiv:2308.12256, DOI 10.1145/3604915.3610244) [A].
  Trains a "not-to-recommend" loss on dislikes/skips and — the relevant part — defines a **responsiveness
  metric**: after a simulated/logged dislike, measure the **reduction of similar recommendations** (−60.8% by
  content / −64.1% by creator when dislikes are in both labels and features). This is the industrial
  dislike-metric precedent: *counterfactual responsiveness*, not held-out-dislike ranking; it measures the
  system reacting to an observed dislike (exactly the "filtering" behaviour our probe is designed NOT to
  credit). Cite it as the complementary metric class.
- **TikTok** — Raju et al., "Leveraging Explicit Negative Feedback in Large-Scale Recommendation Systems: A
  Case Study," RecSys 2025 industry (DOI 10.1145/3705328.3748145) [A; ACM page 403'd, details from secondary
  coverage]. Context-aware surveys + in-feed negative signals, denoised, fed into training penalties and
  serving down-rank/filters; guardrails throttle suppression, negatives bucketed by reason. Reported value =
  feed quality / trust / engagement movements — again no offline avoidance metric.
- **Spotify skips** — skip rate/early-skip is a first-class *online* negative signal (grey literature; Brost et
  al. skip-prediction dataset [K]); offline academic use is skip-down-ranking (§3).
- **Verdict:** industry measures (a) online dislike/skip *rates* and (b) post-dislike *responsiveness*; nobody
  found reports a held-out-dislike top-K avoidance number alongside offline accuracy. Consistent with §4.

## 8. NOVEL FOR US vs MUST-CITE

**MUST CITE — pre-empted, do not name a "new metric":**
- The metric concept (rank-discounted penalty on held-out disliked items, lower-better, negatives excluded from
  input) → **`frolov2016fifty` nDCL** (origin, incl. the metric-insensitivity argument and the negativity
  threshold at 3.5) and **`sanchez2018antirelevance`** (anti-precision/anti-nDCG/fallout as a named framework).
- "Standard metrics are blind to surfacing known-bad items" → Frolov §4 (quote above) + Sánchez & Bellogín.
- Negative relevance grades in ranking metrics + the unboundedness/clamping problem → **`gienapp2020negndcg`**
  (TREC junk −2; nDCG_min normalization).
- Anti-metrics have their own popularity bias; threshold sensitivity → **`menamaldonado2021fpbias`** (+ SIGIR'20
  companion), Sánchez & Bellogín.
- Evaluating on negatives the model never saw, without exposure gaming → the MAR test-set line (Yahoo R3
  Marlin & Zemel; Coat `schnabel2016recommendations`; KuaiRec `gao2022kuairec`).
- Industrial dislike metrics → `wang2023negativefeedback` (responsiveness), `raju2025tiktoknegative`.
- Full-rank not sampled → `krichene2020sampled`.

**DEFENSIBLE AS OURS (aware-of-none, verified this pass):**
1. **The conjunction:** a dislike-avoidance probe **attached to the canonical Liang strong-generalization
   binarized protocol**, reported next to the frontier NDCG@10, where sub-threshold items are absent from every
   arm's input *by the protocol's own construction*. Nobody runs nDCL/anti-metrics on this split; the entire
   Liang-frontier and even the sign-aware GNN line (SiReN [A], SIGformer [V]) evaluate positives-only.
2. **The anti-gaming rationale stated as a design principle:** *observed* dislikes are guaranteed non-targets →
   demoting them is a free lift → injected-dislike evaluations reward filtering, not comprehension; therefore
   the probe items must be held out. Frolov's insensitivity argument is adjacent but is about metrics ignoring
   bad items, not about filters gaming an injected-dislike protocol (the recommendation-editing benchmark is
   the literature's filtering arm — cite as the contrast class).
3. **Purpose:** using the probe to certify that a **frozen tower consuming signed input generalises** dislike
   (sign-comprehension of a representation), rather than to win a model-vs-model negative-feedback comparison.
4. **Elicitation angle (§6):** the marginal value of a "no" answer at matched question budget — unoccupied;
   nearest precedent to cite and extend is Frolov's negative-only cold-start condition.

**Protocol recommendations (from the pitfalls found):**
- Prefer the **hit-rate@K of held-out dislikes (lower-better) + an nDCL-form** over "NDCG with negative gains
  mixed into the positive metric" (unboundedness/clamping, Gienapp); keep likes-NDCG and avoidance as two
  numbers.
- Report avoidance **full + popularity-stratified/tail** (anti-metric popularity bias, TOIS 2021; MNAR of
  observed dislikes).
- Include degenerate floors: random, most-popular, and least-popular/head-demotion arms (the avoidance
  metric's "popularity floor").
- State the threshold (=3.5, matching both Liang and Frolov's ML-10M choice) and show threshold sensitivity or
  argue it's fixed by the protocol.
- Decide and state the unrated-items stance (Frolov: ignored, not false positives) — ours is implicit since the
  metric only touches labelled held-out dislikes.

## Bib entries to add (hand to author; do not edit references.bib here)
- `sanchez2018antirelevance` — Javier Sanz-Cruzado? — **correct authors: Pablo Sánchez, Alejandro Bellogín.** "Measuring anti-relevance: a study on when recommendation algorithms produce bad suggestions." RecSys 2018 (short). DOI 10.1145/3240323.3240382.
- `menamaldonado2021fpbias` — Elisa Mena-Maldonado, Rocío Cañamares, Pablo Castells, Yongli Ren, Mark Sanderson. "Popularity Bias in False-Positive Metrics for Recommender Systems Evaluation." ACM TOIS 39(3):36, 2021. (Companion: same authors, "Agreement and Disagreement between True and False-Positive Metrics…", SIGIR 2020.)
- `gienapp2020negndcg` — Lukas Gienapp, Maik Fröbe, Matthias Hagen, Martin Potthast. "The Impact of Negative Relevance Judgments on NDCG." CIKM 2020 (short). DOI 10.1145/3340531.3412123.
- `wang2023negativefeedback` — Yueqi Wang, Yoni Halpern, Shuo Chang, et al. "Learning from Negative User Feedback and Measuring Responsiveness for Sequential Recommenders." RecSys 2023 (industry). DOI 10.1145/3604915.3610244, arXiv:2308.12256.
- `raju2025tiktoknegative` — Madhura Raju, Manisha Sharma, Hongyu Xiong, Bingfeng Deng, et al. "Leveraging Explicit Negative Feedback in Large-Scale Recommendation Systems: A Case Study." RecSys 2025 (industry). DOI 10.1145/3705328.3748145. [A — author list partially verified]
- `seo2022siren` — Changwon Seo, Kyeong-Joong Jeong, Won-Yong Shin. "SiReN: Sign-Aware Recommendation Using Graph Neural Networks." IEEE TNNLS 2022. arXiv:2108.08735. [A — author order to verify]
- `chen2024sigformer` — Sirui Chen et al. "SIGformer: Sign-aware Graph Transformer for Recommendation." SIGIR 2024. arXiv:2404.11982. [V for eval protocol; author list to verify at bib time]
- `krichene2020sampled` — Walid Krichene, Steffen Rendle. "On Sampled Metrics for Item Recommendation." KDD 2020. [K]
- `schnabel2016recommendations` — Tobias Schnabel, Adith Swaminathan, Ashudeep Singh, Navin Chandak, Thorsten Joachims. "Recommendations as Treatments: Debiasing Learning and Evaluation." ICML 2016. [K]
- `marlin2009collaborative` — Benjamin Marlin, Richard Zemel. "Collaborative Prediction and Ranking with Non-Random Missing Data." RecSys 2009. (Yahoo! R3) [K]
- `gao2022kuairec` — Chongming Gao et al. "KuaiRec: A Fully-Observed Dataset and Insights for Evaluating Recommender Systems." CIKM 2022. [K]
- `kaminskas2016diversity` — Marius Kaminskas, Derek Bridge. "Diversity, Serendipity, Novelty, and Coverage: A Survey and Empirical Analysis of Beyond-Accuracy Objectives in Recommender Systems." ACM TiiS 2016. [K; cite only to note beyond-accuracy ≠ negative-aware]
- (Already in bib: `frolov2016fifty`, `steck2010training`, `dacrema2019progress`, `golbandi2011adaptive`, `antognini2022posneg`, `toroghi2023bcie`.)

Optional/secondary (cite only if the relevant sentence survives): music-skip contrastive arXiv:2409.07367 [A];
recommendation editing arXiv:2406.04553 [A]; negative-sampling survey arXiv:2409.07237 [A]; UserSimCRS v2
arXiv:2512.04588 [A].
