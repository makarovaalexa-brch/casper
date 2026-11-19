# Reviewer 1 — ACM TORS review

**Manuscript:** "Elicitation Results Are Instrument-Relative: A Certified Interview-Compatible Recommender for Measuring Cold-Start Preference Elicitation" (36 pp., single author)
**Reviewer role:** methodologist (offline-evaluation hygiene, baseline fidelity, internal consistency, overclaiming)

---

## 1. Summary of the submission

The paper argues that the ranking of classical cold-start question-selection heuristics (HELF, Entropy0, Popularity from Rashid et al.) is not a property of the heuristics but of the heuristic–recommender pair. Holding users, questions, budget (k=8), answer model and candidate pool fixed, it feeds the same eight answers to six "consumers" (the authors' own model, a likes-only RecVAE, and best-effort reconstructions of Golbandi's node model, RBMF, a conjugate-Gaussian belief over a frozen MF, and TaNP) and reports that the consumers disagree about which heuristic wins (Table 3).

From this it motivates four requirements an "instrument" recommender must satisfy (R1 frontier-parity full-profile accuracy; R2 open set-valued input; R3 monotonicity; R4 signed/graded answers), surveys ~25 systems against them (Table 1, Fig. 1), and claims none occupies the conjunction. It then builds one: a frozen RecVAE tower with a Deep-Sets encoder trained over its latent, into which items, Tag-Genome concepts (as whitened centroids with a PMI-style "watch-lift" answer) and free-text phrases (via a linear map from a sentence encoder) enter as (direction, value) pairs. The instrument scores full/tail NDCG@10 0.3496/0.2491 on an ML-25M strong-generalisation split versus RecVAE's 0.3540/0.2497, passes a "verification battery" (Table 4), and is then used to measure answerability (25.2% of the top-100 films vs 53.0% of 1,031 concepts) and to compare item vs concept channels under four bank rankings, finding concepts win under a pure-discrimination ranking and items win otherwise (Table 5).

## 2. Relevance to TORS

High. Evaluation methodology for conversational/cold-start recommendation, reproducibility of classical results, and a reusable evaluation artefact are squarely in scope, and the paper engages explicitly with Ferrari Dacrema et al. [15]. The length (36 pp.) is at the upper end of what the journal format justifies for the amount of new evidence.

## 3. Strengths

1. The core methodological point — that an elicitation-policy result is conditional on the consuming recommender and must be reported with it — is correct, under-appreciated, and worth a TORS paper. The "report the consumer with the strategy / score the whole catalogue / state the answer path" prescriptions (Section 9) are sound.
2. The answer-provenance rule (Section 7.4) is a genuinely useful piece of evaluation hygiene: no recommender-derived quantity and no target-half information may enter a simulated answer. The paper is right that PEBOL-style simulators answer from the target and that bandit simulators answer from the ranking latent.
3. The published-number bridge (Section 7.2) is done properly: EASE 0.4203 vs 0.420 and RecVAE 0.4425 vs 0.442 on the canonical ML-20M split, with Recall@20/50 also matching, and Mult-DAE honestly dropped when it failed to snap. This is exactly what [15] asks for and few papers do it.
4. Paired per-user bootstrap intervals on 10,000 held-out users for every headline contrast, both full and tail NDCG@10 reported throughout, and several negative or unflattering numbers reported rather than hidden (the Golbandi node model tying the instrument on tail; iALS gaining more from ratings than the instrument; the +0.0782 loss of the concept channel under the popularity ranking).
5. The existential controls (wrong-user, shuffle, placebo, duplicate) and the fixed-support decomposition in Appendix A are the right kind of control design.
6. Limitations section (Section 10) is candid on the one-dataset, one-training-run, and self-as-consumer problems.

## 4. Weaknesses (most serious first)

**W1. The headline RQ1 claim — that the classic consumer "reproduces the published order" — is contradicted by the paper's own Table 3.** The abstract (p.1), Introduction (p.2: "Under our reconstruction of the node model of Golbandi et al. [...] the published order reproduces"), the Table 3 caption (p.24: "among these six consumers only the Golbandi node model reproduces it"), the Answer to RQ1 (p.25) and the Conclusion (p.32) all state that the published order helf > entRopy0 > populaRity is reproduced by the Golbandi node model. Table 3 (p.24) gives that consumer .1948 / .1686 / .1716, i.e. **helf > populaRity > entRopy0** — second and third place are swapped — and the "Order" column itself prints "H > P > E0". Section 10 (p.31) even writes it correctly: "the Golbandi node model (helf > populaRity > entRopy0)". So no consumer in the table reproduces the published ordering. The paper's most-repeated sentence is false on its own evidence. Likewise, the "inversion" under "a frontier-accuracy consumer" (p.2) is ambiguous: the instrument gives E0 > P > H, which is not the inverse of H > E0 > P; the only exact inversion (P > E0 > H) is produced by the likes-only RecVAE row, which the authors themselves call "a control, not a rival" (Table 3 caption).

**W2. R1 "pass" is obtained under an input the paper's own protocol denies every other row, and fails under the pre-registered tolerance otherwise.** R1's tolerance is "fixed in advance at within 1.5% relative of the strongest measured baseline on both axes" (p.2). The Table 1 caption states the liang column "admits no graded value" and iALS's liang cell is therefore "the binary build". Yet the instrument's liang cell 0.3496 is computed with "each item carr[ying] its half-star level where the others carry a 1" (p.26). The paper then reports (p.27) that "on the binary information RecVAE consumes, this tower would score 0.3457, some −0.0083 behind it." 0.3457/0.3540 is a 2.35% shortfall — outside the pre-registered 1.5%. So R1 passes only because the instrument is given graded information in a column defined as binary, in direct contradiction of the rule applied to iALS in the same table.

**W3. The R1 ✓/✗ marks in Table 1 do not follow the paper's own R1 definition, so the "bifurcation" is partly an artefact of inconsistent grading.** With the bar at 0.3540 and a 1.5% tolerance, EASE (0.3476, −1.8%; tail −2.2%), EDLAE (0.3433, −3.0%) and Mult-VAE (0.3202, −9.5%) all fail R1, yet all carry ✓ (p.9). Mult-DAE carries ✓ although the paper "drop[s] it from the baseline bank" for failing its bridge. Meanwhile user-kNN at 0.3065 (0.014 behind Mult-VAE) carries ✗. The caption's escape clause — requirements are "read from its published claims" — cannot apply to R1, which the paper insists must be measured on the shared ruler. Under the paper's stated criterion, exactly two rows pass R1: RecVAE and (with graded input, see W2) the instrument.

**W4. The R2–R4 marks in Table 1 and Fig. 1 contradict the gap analysis of Section 4.** Reading Table 1 at high resolution:
- ConTS: R3 ✗ (x=0 in Fig. 1). Section 4 (p.12): "ConTS [29]: R3 plus a unified item/attribute action space."
- PEBOL: R2 ✓, R3 ✗ (x=1). Section 4: "PEBOL [3]: R3 (per-item Beta) plus R2 (LLM-extracted aspects, an open vocabulary)" — should be x=2.
- EDDI: R3 ✗, R4 ✓ (x=1). Section 4: "EDDI / Partial-VAE [37]: R3 (information gain over a belief) plus set input."
- Bıyık: R3 ✗. p.10: "the nearest thing to R2, R3 and R4 together"; Section 4 says it fails R1 and the *open half* of R2 — not R3 — and it maintains exactly the "belief or posterior that its published update rule keeps consistent with a new answer" that the caption defines as the R3 criterion.
- Conjugate belief (frozen MF): R3 ✗, while BCIE, "the closest published analogue of our frozen-tower fold" with the same conjugate update, gets R3 ✓.
- Fig. 1 caption: "every system with interview machinery sits at x ≥ 1" — the figure itself plots ConTS and UNICORN at x=0.
Finally, the instrument gets R3 ✓ although it maintains no belief at all ("Nothing in what follows requires a distribution over u", p.4), so the R3 criterion applied to rivals (structural: a posterior) is not the one applied to the instrument (empirical: a rising curve). A master table whose marks disagree with the text in at least six rows cannot carry the RQ2 claim.

**W5. The instrument-relativity experiment (Table 3) rests on one crippled control and four author-built reconstructions, and its mechanism is a known confound.** Only two of six consumers are "run as designed", and one of those is RecVAE fed at most the likes among ≤1.6 answers with no popularity floor; its helf cell (0.1003) sits *below* the cold intercept (~0.128), i.e. this row's ordering is driven by how a VAE encoder behaves on a near-empty input, not by a frontier recommender "consuming answers". The paper's own explanation of the split (p.24–25) is that helf yields 1.59 graded answers of 8 and only a lovers/haters/unknowns tree can use non-answers — this is precisely the observation Rashid et al. 2002 and Sun et al. 2013 already made, and the paper concedes that "answerability explains why helf divides the table, not the remaining orderings" and that E0 vs P "are within 0.05 answers of each other and still change places". The remaining orderings are decided by gaps of 0.003–0.011 among reconstructions whose fidelity is not demonstrated (see W7). The sentence "The disagreement is not attributable to weak recommenders" (p.25) is not supported: the only strong consumer in the table is the authors' own.

**W6. Several elicitation-corner and classic baselines score implausibly low, which is the failure mode of [15] in reverse: weak reconstructions make the proposed system look necessary.** SASRec at 0.1486 (Most-Popular 0.1345) is at the popularity floor; EDDI/Partial-VAE at 0.2153 is 0.10 below Mult-VAE although Partial-VAE is a Mult-VAE-class amortised encoder with a set input; Item-kNN at 0.2428 trails EASE by 0.105 (on the canonical ML-20M split the analogous gap is ~0.03 at NDCG@100). No tuning grid, neighbourhood size, shrinkage, or training details are given for Item-kNN, SASRec, or EDDI beyond "early-stopped at epoch 46 with epoch 31 validation-selected". A baseline at the popularity floor is a broken baseline until shown otherwise, and the "elicitation corner is measurably weak" reading of Table 1 (p.9) depends on these numbers.

**W7. The "best-effort cores" are never specified.** The paper repeatedly says "every substitution [is] stated in Section 3" (p.23, p.31), but Section 3 contains no list of substitutions for the Golbandi node model, RBMF, the conjugate belief, or TaNP: no tree depth, no leaf regularisation, no number of representatives, no prior/noise variances, no TaNP architecture. Worse, the Golbandi row is labelled "user-kNN [18]" in Table 1 and Appendix A ("user-kNN, measured through one estimator ... 0.3756 on binary co-occurrence") but "Golbandi node model [18]" in Table 3. Golbandi et al. [18] did not propose user-kNN; whether the consumer in Table 3 is a user-kNN or a per-node profile model is not determinable from the text.

**W8. The concept-channel headline is engineered by the choice of ranking, and its absolute effect is tiny.** The abstract leads with "the concept channel wins on both metrics at every budget". That holds only under polaR, a ranking the authors introduce, under which the item bank is answerable 4.8% of the time — a degenerate bank that "forfeits almost every question" (p.28). Under the three other rankings, including the one published heuristic (pop), items win by −0.0782 at m=16. In absolute terms sixteen concept answers move full NDCG@10 from 0.1277 to 0.1498 (+0.022), whereas sixteen item answers on the fixed bank reach 0.2609 (Table 4). The signed-vs-positive-only result (Section 8.3) is +0.0104 at q=16 with the signed curve at 0.1372, i.e. +0.0095 above the intercept after 16 questions. The paper should say plainly that, on this instrument, the concept channel as built is an order of magnitude weaker than the item channel.

**W9. R3 fails for the concept channel and the battery does not record it.** R3 states "a truthful answer must not lower ranking quality beyond noise" (p.3). Section 8.3 reports that with positive-only concept answers — which are truthful — "the positive-only curve sinks below the intercept by the eighth question and stays there". Table 4's R3 row is item-only, on a bank "filtered per user to answerable questions". Section 6.4 reframes the concept-channel violation as "deployment currency", but the battery is claimed to be "one item per requirement" and passes R3 while the paper's own concept channel does not satisfy R3 as defined.

**W10. The R4 sign/neutralisation references are unexplained and possibly leaky.** Table 4 gives references of 0.4298 (sign arm, n=7,563) and 0.4198 (neutralisation arm) full, 0.2999 tail. The caption explains they "sit above the 80/20 fold-in values of Table 1, which withhold a fifth of each profile as targets" — implying the whole profile including targets is folded. If the target half is folded and then scored, that is target leakage; if it is folded and removed from the pool, NDCG would be zero. The numbers instead match the instrument's *native-pool* cell (0.4244/0.3017), which suggests the caption's explanation is wrong. Why the two arms have different references, and why n=7,563 rather than 10,000, is not stated. Note also that the sign-flipped score (0.1345) coincides exactly with the Most-Popular liang cell (0.1345).

**W11. One training seed, one dataset, no family-wise correction, and a tolerance that lands just above the outcome.** The 1.5% R1 tolerance is asserted to be "fixed in advance" but nothing in the manuscript or artefact evidences that; the instrument lands at 1.2%. With one training run, the "statistically indistinguishable on the tail" claim covers user-sampling variance only. Dozens of intervals are reported "without a family-wise correction" (p.21).

**W12. Reproducibility gaps in the concept channel.** The refusal threshold τ, the four band thresholds, the norm cap in Eq. (4), the concept-fold architecture and training data, the sentence encoder behind the text adapter, and the closed-form solver for the linear map are all unstated. The operator ablation that decides the gated fold (Section 6.2, footnote 1) "was run on an earlier tower" and never repeated on the certified one. The repository URL is a placeholder.

## 5. Detailed comments

### 5.1 Internal inconsistencies (numbers, tables, prose)

- p.1/2/24/25/32 vs Table 3: "published order reproduces" under Golbandi; table shows H > P > E0 (see W1).
- p.6 and p.8: graded input pays "+0.0102 at k=2, and nothing separable from k=8 onward"; Fig. 3 and p.27 report +0.0022 at full profile with CI [+0.0008, +0.0036], which is separated.
- p.13: "a random question from an answerable bank *matches* popularity-ranked items at q=1, 2, 4, 8"; p.27: random "*beats* popularity-ranked items at q=1, 2, 4, 8 (for example 0.2358 against 0.2230 full at q=8)". A 0.0128 gap on 10,000 users is almost certainly separated.
- p.13: the "sign contrast (+0.2096, CI [+0.1986, +0.2208])" is offered as what "our own sign contrast measures directly" for "I dislike westerns"; in Table 4 this number is a difference of member-ranking AUCs (0.626 vs 0.416), not an NDCG difference. Units are not stated at p.13.
- Table 1 R1 marks vs the 1.5% tolerance (W3); R2–R4 marks vs Section 4 (W4); Fig. 1 caption vs Fig. 1 plot (ConTS/UNICORN at x=0).
- Table 1 labels the Golbandi row "user-kNN [18]"; Table 3 labels the same core "Golbandi node model [18]"; Appendix A calls it "user-kNN" (W7).
- Table 3, Golbandi tail "−.0000 [−.0032, +.0032]": a paired difference of exactly zero to four decimals with an exactly symmetric interval between two different models on 9,281 users is suspicious; please confirm the two score vectors are not the same file.
- Table 3 caption says the helf RecVAE arm "yields 1.6 answers of 8"; the table's first row says 1.59 for all consumers; fine, but the likes-only restriction means RecVAE sees fewer than 1.59 — state how many.
- Two cold intercepts (0.1279/0.0192 vs 0.1277/0.0267) are explained by pool differences, but no intercept is reported for the *native* pool used in Table 3, so none of the Table 3 cells can be read against "no answer". Given the conjugate-belief row is "flat" at ~0.162 and the RBMF entRopy0/pop cells are ~0.164, the reader needs to know whether these consumers learned anything from eight answers.
- Table 4 R3: "0.1574 at q=2" on the fixed bank (1.97 answered) vs Fig. 3 graded at k=2 answered ≈ 0.18 on the same tower: a 0.02 gap between two arms with ~2 answered items each, and Fig. 3 does not say which bank it uses.
- Table 4 R4 references (0.4298 / 0.4198 / 0.2999) vs Table 1 native cell (0.4244/0.3017) and the caption's "whole rated history is folded" (W10).
- Table 4 "monotone ρ=1.0": Spearman over four budget points is trivially 1.0 and is not evidence.
- p.9 caption: "bold marks the best measured value per column", but the instrument's liang cells are not bold and RecVAE's are; consistent with the numbers, but then the abstract's "frontier-parity" language should be read against a table in which the instrument is best in no column.

### 5.2 Unsupported or overclaimed statements

- Abstract/p.2: "the order reported by Rashid et al. reproduces under our reconstruction of the node model" — false per Table 3 (W1).
- p.2: "Six consumers realise three distinct orderings" — four consumers have a fully separated ordering (RBMF has E0 ≈ P; the conjugate belief is "flat"), and two of the four (instrument, TaNP) agree.
- p.25: "The disagreement is not attributable to weak recommenders" (W5).
- p.3/p.9: "the published literature is bifurcated ... anti-correlated across published systems" — per Table 1, the elicitation corner systems meet at most *one* of R2–R4 each (Bıyık, ICF, BCIE, PEBOL at x=1; ConTS, UNICORN at x=0). "Anti-correlation" between R1 and a count that is 0 or 1 almost everywhere is not a bifurcation; it is that nobody scores well on both R1 and the authors' capability count.
- Abstract: "with every gap that decides one of the three distinct orderings separated" — true only because the non-separated rows were declared not to "decide" an ordering.
- Abstract/p.4/p.29: "the concept channel wins on both metrics at every budget" — under one ranking, against a 4.8%-answerable item bank (W8).
- p.2, p.25: RecVAE-likes-only "scores below our instrument by 0.029 ... a strong full-profile score is not by itself sufficient for a strong short interview" — the comparison is against a model fed ≤1 like with no popularity floor, whose empty-input behaviour is below popularity; the paper concedes on p.25 that this "is not a controlled attribution".
- p.14: "the flat-since-2019 frontier means no deep-net detour is warranted" — asserted, not shown on this ruler (no graph-filter leader except Turbo-CF, which lands "mid-pack", is run).
- p.16: "more than 300× more often than chance (0.167 top-10 overlap against a 0.0005 random baseline)" — 0.167 top-10 overlap is 1.7 items of 10; "300×" is the ratio to an essentially-zero baseline and overstates the practical fidelity of the text adapter.
- p.28: "lifts a frontier-accuracy recommender monotonically above its starting point" for the concept channel — monotonicity of the deployed concept curve is not shown in any table; only endpoints are given.
- "Certified" is used ~40 times for what is a self-check that the authors' reimplementations reproduce published numbers; the word promises third-party verification the paper does not provide.

### 5.3 Missing baselines and controls

- Rashid et al. 2008's own best strategy, IGCN, is omitted; the three chosen heuristics are the subset that supports the story. Also, please state under which recommender and which metric [45] reported "helf > entRopy0 > populaRity"; my recollection is that [45]'s ordering is metric-dependent (MAE vs. coverage/effort) and that [44] already reports entropy failing for answerability reasons.
- No consumer in Table 3 is an untouched published full-profile model fed *all* answers with a popularity floor (e.g. EASE or RecVAE with an unanswered-question default), which is the ordinary way an elicitation paper would evaluate; the RecVAE row is deliberately crippled.
- No native-pool cold intercept per consumer (see 5.1).
- For R1, no seed replicate; for the ablations, no re-run of the operator ablation on the certified tower.
- The concept channel has no item-only comparator at matched *information* (e.g., the concept's member items folded as a weighted bag into the same instrument), which would separate "concept direction" from "many member items at once". Section 7.5 argues this is impossible for EASE/RecVAE, but it is trivially possible inside the instrument itself.
- The text adapter is illustrated on four hand-picked phrases (Table 2) after admitting other phrases "return lists we would not defend"; no quantitative held-out evaluation beyond the 0.167 overlap.

### 5.4 Reproducibility gaps

- Substitutions for the four best-effort consumers (W7).
- Hyperparameter grids and selected values for EASE (λ), Item-kNN, Mult-VAE, RecVAE ("follows the paper configuration" — the paper's configuration was tuned for ML-20M), SASRec, EDDI.
- Concept channel: τ, band thresholds, norm cap, concept-fold architecture/training, IDF weighting details, sentence encoder identity, linear-map solver (W12).
- The set encoder's curriculum mixture proportions, learning rates, epochs, and the checkpoint-selection rule beyond "full NDCG@10 on validation".
- "the absence is asserted in code" (provenance) — no code is available at review time.
- The interview harness's handling of the asked-but-unanswered items for each consumer (the instrument has an exposure branch; what do the other five do?).

### 5.5 Bibliography defects

- [43] Radlinski et al. 2019 has no venue (SIGDIAL 2019).
- [5] Bıyık et al. is cited as arXiv 2023 only; check whether a peer-reviewed version exists and cite it.
- [28] GATE, [30] Bot Play, [2] Antognini 2022, [9], [16], [56]: arXiv-only citations; verify venues.
- [11], [53], [58]: venue strings in lower case / inconsistent style ("22nd ACM SIGKDD international conference", "The 41st international acm sigir conference").
- [50] no pages; [20] no volume/pages; [1] no pages.
- The Golbandi citation [18] is attached to a "user-kNN" row (Table 1), which misattributes the method.
- Running head reads "Makarova et al." for a single-author paper.

### 5.6 Presentation defects

- 36 pages for the evidence presented; the same numbers (0.3496/0.2491; 25.2%/53.0%; 97.2%/4.8%; +0.0102 at k=2; +0.0039) recur five to eight times each. The abstract is ~330 words and reads as a results dump.
- Section 3.2 and Appendix A (what restoring ratings is worth) are a digression from all four RQs.
- "Two intercepts ... never compared across" and "each column is read against its own rows" — the paper repeatedly warns the reader off comparisons it has itself invited by placing the numbers side by side.
- Table 1 uses a "read from published claims" grading for R2–R4 and a measured grading for R1 without saying which rule applies to which column.
- Fig. 1's x-axis is a 0–3 count; plotting a single star at x=3 and calling it "the only measured point at both the frontier bar and full interview compatibility" is a visualisation of the paper's own scoring rule, not of data.
- Section numbering references are stale: "Section 8.3 reports all three" (p.20) vs the three criteria being reported in Section 8.3 as one paragraph; "Section 5.1" cited for the channel-agnostic fold that is defined in Section 5.2 (p.19).

## 6. Questions for the authors

1. Table 3 shows the Golbandi node model at helf > populaRity > entRopy0. On what basis does the abstract, introduction, Table 3 caption and conclusion say it "reproduces the published order" helf > entRopy0 > populaRity?
2. Under the pre-registered 1.5% tolerance, does the instrument pass R1 on the binary input the liang protocol prescribes for every other row (0.3457 vs 0.3540 = −2.35%)? If not, on what grounds is the liang column allowed graded input for one row?
3. Please supply the exact R1/R2/R3/R4 grading rule and regrade Table 1 so that it agrees with Section 4 for ConTS, PEBOL, EDDI, Bıyık and the conjugate belief, and with the 1.5% R1 tolerance for EASE, EDLAE, Mult-VAE and Mult-DAE.
4. What are the native-pool no-answer intercepts for each of the six consumers in Table 3? Do RBMF and the conjugate belief exceed theirs at k=8?
5. Is the "Golbandi node model" a user-kNN (Table 1, Appendix A) or a per-node profile model (Table 3)? Give the full specification: tree depth, routing thresholds for lover/hater/unknown on graded ratings, leaf regularisation, and what it does with the 6.4 unanswered questions.
6. Why do SASRec (0.1486), EDDI (0.2153) and Item-kNN (0.2428) score where they do? Provide tuning grids and, for SASRec, the sequence construction under a random 80/20 fold-in.
7. Table 4, R4: what exactly is folded and what is scored in the 0.4298 and 0.4198 reference arms? Why are they different, why n=7,563, and why do they match the native-pool cell rather than the liang cell?
8. How was the exact −.0000 tail tie between the instrument (entRopy0) and the Golbandi node model (helf) in Table 3 obtained? Confirm the two per-user score vectors differ.
9. What are τ, the four band thresholds, the norm cap and the sentence encoder? What fraction of the 53.0% "answerable" concepts return the neutral band?
10. Was the 1.5% tolerance recorded anywhere before the instrument's number was known?
11. Did [45] report helf > entRopy0 > populaRity on a single metric with a single recommender? Please quote the passage and state the recommender.

## 7. Scores (1–5)

| Criterion | Score |
|---|---|
| Relevance | 5 |
| Originality | 3 |
| Technical soundness | 2 |
| Presentation | 2 |
| Reproducibility | 2 |
| Impact | 3 |
| **Reviewer confidence** | 4 |

## 8. Recommendation

**Major revision.**

The methodological thesis is right and the evaluation hygiene in places (bridge, provenance rule, paired bootstraps, tail reporting) is better than the field's norm, which is why I do not recommend rejection. But the manuscript as submitted cannot be accepted: its most-repeated empirical sentence ("the classic consumer reproduces the published ordering") is contradicted by its own Table 3; its R1 pass depends on giving the instrument graded input in a column the paper defines as binary, and fails the pre-registered tolerance otherwise; its master table's ✓/✗ marks disagree with the text in at least six rows and with the R1 rule in four; the instrument-relativity result rests on one crippled control and four unspecified reconstructions, several of which (and several classic baselines) score implausibly low; and the concept-channel headline is true under one author-chosen ranking against a 4.8%-answerable item bank while the channel's absolute contribution is +0.02 NDCG after sixteen questions. A revision must regrade Table 1 under one stated rule, restate RQ1 honestly (the consumers disagree; none reproduces the published order; the mechanism is answerability handling, which is known), specify and re-tune every reconstruction and classic baseline, report native-pool intercepts, resolve the R4 reference arms, and cut the length by a third.

**The single question a rebuttal must answer:** Under the binary input that the liang protocol imposes on every other row of Table 1, the instrument scores 0.3457 against RecVAE's 0.3540, a 2.35% shortfall outside the pre-registered 1.5% tolerance — so does the instrument satisfy R1 at all, or does the paper's central "frontier-parity" claim rest on an input contract it denies its comparators?
