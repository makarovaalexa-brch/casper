# Reviewer 1 — ACM TORS

**Manuscript:** "Elicitation Results Are Instrument-Relative: A Certified Interview-Compatible Recommender for Measuring Cold-Start Preference Elicitation" (33 pp., review mode with line numbers)

**Reviewer profile:** methodologist; offline-evaluation hygiene, baseline fidelity and overclaiming are the deciding issues for me.

---

## 1. Summary of the submission

The authors argue that published rankings of cold-start question-selection heuristics (HELF, Entropy0, Popularity) are not properties of the heuristics but of the heuristic–recommender pair, and support this with an experiment in which six "consumers" read the same eight answers from the same 10,000 held-out ML-25M users and produce three different (mostly bootstrap-separated) orderings (Table 3). From this they derive four requirements for a recommender that can serve as a "measuring instrument" for elicitation research (R1 frontier-parity full-profile accuracy, R2 open set-valued input, R3 monotonicity, R4 signed/graded answers), survey ~25 systems against these requirements in a master table on a shared ML-25M ruler (Table 1, Figure 1), and claim that the literature is bifurcated: nothing meets all four. They then present such an instrument: a RecVAE-class tower, frozen after training, with a permutation-invariant set encoder over its latent, into which items (as frozen factors with graded values), Tag-Genome concepts (as whitened member centroids with a signed watch-lift "answer"), and free-text phrases (via a linear map from a sentence encoder) fold as (direction, value) pairs. The instrument scores 0.3496/0.2491 full/tail NDCG@10 against a RecVAE realisation at 0.3540/0.2497 (tail CI contains zero; full profile separably −0.0043), and passes a self-defined "verification battery" (Table 4). Finally they use it to measure "answerability" (25.2% of the 100 most popular films vs 53.0% of 1,031 concepts) and to compare item and concept channels under four bank rankings, finding that concepts win under a pure-discrimination ranking and items win under the other three.

## 2. Relevance to TORS

High. Evaluation methodology for conversational/cold-start recommendation, reproducibility of classic baselines, and an artefact (checkpoint plus per-user score vectors) are squarely within TORS's remit. The paper explicitly positions itself in the Ferrari Dacrema et al. tradition, which is the right frame for this journal.

## 3. Strengths

1. The central methodological point, that a strategy comparison should fix and document its consumer and report full-catalogue accuracy, is correct and worth saying loudly in this venue. The experimental design of Section 8.1 (fix users, questions, answers, budget, pool; vary only the consumer) is the right design for the claim.
2. The certification bridge (Section 7.2) is genuinely good practice: reproducing EASE (0.4203 vs 0.420) and RecVAE (0.4425 vs 0.442; Recall@20/50 also matching) on the ML-20M strong-generalisation split before moving to ML-25M, and dropping Mult-DAE and demoting iALS when they miss their anchors. Few submissions do this.
3. Paired per-user bootstrap CIs on 10,000 users for nearly every headline delta; full and tail NDCG@10 reported throughout; validation/test discipline stated (l. 1001–1003).
4. The answer-provenance rule (Section 7.4) is a real contribution to simulator hygiene; the critique of PEBOL/UNICORN/ConTS-style simulators drawing answers from the target or from the model's own latent is well taken.
5. The Section 3.2 decomposition (hyperparameter re-selection vs added support vs restored values, each with a CI) is careful and honest, including the retraction of the previously recorded RBMF −0.070 as a configuration artefact.
6. The limitations section is unusually candid (single dataset, monotonicity empirical, protocol discards dislikes, the instrument is one of its own comparators).

## 4. Weaknesses (most serious first)

**W1. Table 1 block (c) attaches numbers to the wrong systems, and the error propagates to Figure 1 and the prose.** (p. 9, l. 407–410; p. 10, l. 448–451, 458–459; p. 12, Fig. 1; p. 15, l. 734; p. 8, l. 355–356; p. 22, l. 1052–1055)
- Section 3.3 (l. 450–451) says the Golbandi node core scores "0.3065/0.1895" and the RBMF/Functional-MF seed "0.2915/0.2071". Table 1 assigns 0.3065/0.1895 to *RBMF* and 0.2915/0.2071 to *Conjugate belief*, and gives user-kNN [Golbandi] 0.2153/0.1077.
- Section 5.2 (l. 733–735) says the uniform conjugate accumulator "reaches full/tail NDCG@10 0.2267/0.1300 on our ruler … (Table 1)". Table 1 assigns 0.2267/0.1300 to *Biyik soft attributes*.
- Section 3.3 (l. 458–459): "For Biyik and ConTS we print a dash rather than a number". Table 1 prints 0.2267/0.1300 for Biyik.
- Section 3.2 (l. 355–356): user-kNN scores "0.3756 on binary co-occurrence". Table 1 gives 0.3756/0.2377 as *RBMF's* native cell and user-kNN's as 0.2718/0.1383.
- Figure 1 plots user-kNN at ≈0.3065 and "conjugate belief" at ≈0.2267, i.e. it agrees with the prose, not with Table 1.
- The NDCG@100 ordering (l. 1052–1055: user-kNN 0.3958 > RBMF 0.3533 > EDDI 0.3110 > conjugate belief 0.2883) is consistent with the prose labelling, not with Table 1's NDCG@10 labelling (RBMF > conjugate > user-kNN).
- EDDI is described as measured (l. 1044–1045: early-stopped at epoch 46) and has an NDCG@100 number, but no NDCG@10 cell in Table 1.

The only reading that reconciles all of this is that block (c) of Table 1 is shifted up by one row (EDDI ← 0.2153/0.1077, user-kNN ← 0.3065/0.1895, RBMF ← 0.2915/0.2071, conjugate belief ← 0.2267/0.1300, Biyik ← "—"). The master table is the anchor of the taxonomy and of the "bifurcation" argument; a row-shift in it, undetected by the authors, undermines confidence in every other number that was transcribed from run outputs. This must be fixed and every table re-audited against the score vectors.

**W2. The headline "the instrument is the strongest consumer" and the RecVAE-vs-instrument inversion rest on a confound the authors concede but then argue past.** (p. 25, l. 1180–1191, 1205–1209; p. 30, l. 1446–1451; p. 2, l. 78–84)
The instrument is "initialised from that model but subsequently trained on interviews" (l. 1186–1187). Every other consumer in Table 3 is a full-profile model or a best-effort core that has never seen an interview-shaped evidence set. The "Ours − this" column (+0.0293, +0.0105, +0.0409, +0.0422, +0.0311) therefore measures, at least in part, training on the evaluation distribution. The authors admit "this is not a controlled attribution" (l. 1186) and yet (i) the introduction reports the 0.029 gap as showing that "a strong full-profile score is not by itself sufficient for a strong short interview" (l. 78–84), (ii) Section 8.1 says "the instrument's own numbers are the strongest cells, which is the point of R1" (l. 1205), which attributes the advantage to R1 rather than to interview training, and (iii) Section 9 claims the row "closes the objection that the disagreement is an artefact of weak recommenders" (l. 1450). None of these follow. The one frontier-accuracy comparator, "RecVAE, likes-only", is by the authors' own description given "the crudest admissible interface" (l. 1153) and yields 1.6 answers of 8 under HELF; its HELF cell (0.1003) sits *below* the cold intercept (0.1279), i.e. that consumer is doing worse than saying nothing. An inversion produced by a deliberately crippled consumer is not evidence about frontier consumers. What is needed for the claim as stated: (a) the instrument's set encoder trained on full profiles only (no interview-shaped sets), and/or (b) a strong item-only consumer given a fair item-only interface (likes plus the decoder-bias/popularity fallback for empty inputs; EASE with likes; the authors' own iALS-with-confidence at 0.3495 native, which can consume signed ratings and which Section 3.2 shows benefits from them). iALS-confidence is conspicuously absent from Table 3 despite being the best ratings-native baseline in Table 1.

**W3. What is actually shown in Section 8.1 is that consumers differ in how they treat unanswered questions, which is not the claim the title, abstract and conclusion make.** (p. 24–25, l. 1165–1179; p. 1, l. 9–13; p. 30, l. 1458–1462)
The paper's own mechanism paragraph says HELF yields 1.59 graded answers of eight vs 3.10/3.05 for the others, and that "answerability explains why helf divides the table, not the remaining orderings". Entropy0 vs Popularity differences among consumers are on the order of 0.004 and the RBMF and conjugate-belief rows are marked "≈" or "flat". So the robust finding is: a heuristic that asks about unseen films starves a fold-based consumer and feeds a partition-based one. That is a known point (the authors themselves cite Rashid et al., Golbandi et al., Sun et al. 2013 and Elahi et al. for it, l. 512–529). The abstract's "the published ranking … is not a property of the heuristics … with every gap that decides an ordering separated" overstates this: two of six rows contain undecided gaps by the authors' own marking, and the decided inversions are driven by one heuristic's answer count under an unspecified "no popularity floor" contract (see W7).

**W4. The concept-channel results are small in absolute terms and the abstract cherry-picks the one ranking under which concepts win.** (p. 1, l. 25–28; p. 28, l. 1347–1350, 1356–1366; p. 27, l. 1297)
Sixteen concept answers move a cold user from 0.1277/0.0267 to 0.1498/0.0673 (l. 1297), or to 0.1372 in "this gate's arm" (l. 1340) — two different endpoints for the same quantity (see W6). That is +0.01 to +0.02 full NDCG@10 after sixteen questions, against +0.13 for sixteen answerable item questions (Table 4, R3). Calling this "lifts a frontier-accuracy recommender substantially" (l. 1349) is an overstatement. The abstract states that "ranking a question bank by discrimination makes the concept channel beat the item channel on both metrics at every budget" and omits that under three of the four rankings items win "decisively" (l. 1364–1366: −0.0782 full at k=16 under pop). The polaR "win" is a win over an item arm that has collapsed to within 0.008 of the intercept because 95.2% of its questions are unanswerable — it is a statement about the polaR item bank, not about the channels. The body says as much (l. 1376–1379); the abstract and conclusion do not.

**W5. The instrument is a single training run of an undescribed architecture, and the R1 tolerance is asserted to have been fixed in advance without evidence.** (p. 15, l. 715–718; p. 25, l. 1221–1223; p. 3, l. 101–102; p. 22, l. 1049–1051)
"One certified training run of the graded-native set-encoder tower" (l. 1221). No seed variance is reported for the tower or for the RecVAE realisation it is compared with. The RecVAE–EASE gap (+0.0064) is described as "about 1.8 unpaired standard errors … the same order as the tower's own separation from RecVAE" (l. 1050–1051) — but the tower's separation is reported with a *paired* CI that excludes zero, so the two statements are not comparable and the paragraph reads as an attempt to make a separated loss look like noise. A 1.5% tolerance "fixed in advance" (l. 101) that lands 0.3 points above the observed 1.2% shortfall needs a pre-registration reference or should be dropped. Nothing about the tower is specified: latent dimension, set-encoder form, how interview-shaped evidence sets were sampled for training, loss, optimiser, epochs, early-stopping, selection criterion, seed. "Pre-registered configuration" (l. 716) is not a specification. Ferrari Dacrema et al. would not accept a "frontier" claim on one run with no architecture description.

**W6. Numerous internal numerical inconsistencies beyond W1.** (details in Section 5 below) Cold intercept 0.1279/0.0192 (l. 1003) vs 0.1277/0.0267 (l. 1297, 1370); graded premium at k=2 of +0.0144 (l. 263) vs +0.0102 (l. 390, Fig. 3, Table 4); intensity value at full profile +0.0022 (l. 388) vs +0.0039 (l. 1277, Table 4); iALS confidence gain +0.0194 (l. 358) vs +0.0170 (Table 1 footnote); concept-channel endpoint at k=16 of 0.1498 vs 0.1372; sign-flip magnitudes of +0.2096 (R2 row), −0.2953 (R4 row) and −0.2012 (l. 965) with no explanation of which arm each belongs to; R4's two "uncorrupted references" 0.4298 and 0.4198 that are supposedly the same full-profile fold. A submission whose contribution is a *ruler* cannot have this many unreconciled numbers.

**W7. Key protocol objects are undefined.** "No popularity floor" (Table 3 caption, l. 1122, 1145) is never defined; since the intercept is the decoder bias, a consumer with no floor and no answers presumably emits an arbitrary ranking, which is exactly the regime HELF produces. "Sign contrast" and "identity contrast" (Table 4, R2 row; l. 619) are never defined as measurements. "The pre-registered triple gate", "the retrained model" and "three taste criteria" (l. 1336–1337) appear once, undefined. The concept operator ablation (Section 6.2) was "run on an earlier tower" (footnote 2); how many towers exist, and which one produced which number, is not recoverable from the text. The four-band answer thresholds (neutral zone, τ) and the concept-answerability rule ("expected support clears the refusal threshold", l. 1110, vs "rated as few as two of its member films", l. 1388) are inconsistent with each other and unspecified.

**W8. Baseline fidelity and attribution errors in the classical-heuristics setup.** HELF and Entropy0 are not from Rashid et al. [2002] (IUI); to my knowledge they were introduced in Rashid, Karypis and Riedl, "Learning preferences of new users in recommender systems: an information theoretic approach", SIGKDD Explorations 10(2), 2008, which is not cited. Rashid's evaluations used user-based kNN, not a decision-tree node model; the claim that the Golbandi node model is "the model closest to what the heuristics were validated on" (l. 60, 1147–1148) is therefore doubtful, and the user-kNN row would have been the closer consumer. "RBMF [Liu et al. 2011; Zhou et al. 2011]" conflates two different methods (representative-based MF and Functional MF). The "Golbandi node model" consuming heuristic-selected questions is not Golbandi's method (whose contribution is the tree); this is acknowledged as a "best-effort core", but the RQ1 narrative treats it as reproducing "the classic consumer".

**W9. Bibliography defects that suggest the reference list was not checked against sources.** The Bıyık et al. 2023 entry lists "Erdem Biyik, Fan Talati, and Dorsa Sadigh"; the arXiv:2311.02085 paper of that title is by Bıyık, Yao, Chow, Haig, Hsu, Ramachandran and Boutilier (Google), and the body text's "d=50 ALS two-tower" and "164 most frequent tags" match that paper, not a Sadigh-lab paper. The Qin et al. 2020 entry contains stray working notes ("WSDM/KDD line; Google unbiased-LTR", l. 1561–1562). Xia et al. 2026 (arXiv:2607.06765) cannot be checked by this reviewer and is load-bearing for the "direction is established independently" claim (l. 531–532). Several entries carry "Code: https://…" annotations, which is unusual for a reference list.

**W10. Modest originality once the rhetoric is discounted.** The authors are commendably explicit that every component is borrowed (l. 174–181, 670–677, 959–960). What remains is (a) the conjunction, (b) a battery of self-chosen checks, and (c) the RQ1 experiment. The vocabulary ("certified", "instrument", "portable certificate", "verification battery", "deployment currency") does a lot of work that the evidence does not; e.g. R2's "open half" is passed by a concept direction ranking its *own* member items at AUC 0.94 — a near-tautology since the direction is the members' centroid — and by a text-to-latent map whose 0.167 top-10 overlap on ~206 held-out concepts is reported without a CI and without naming the sentence encoder.

## 5. Detailed comments

### 5.1 Internal inconsistencies between numbers, tables and prose
- **Table 1 block (c) row shift** — see W1. Also l. 458–459 ("we print a dash" for Biyik) contradicts the printed cell.
- **Cold intercept:** l. 1003 "0.1279/0.0192"; l. 1297 and l. 1370 "0.1277/0.0267"; l. 906 "0.1279". The tail values differ by 40%. If these are different pools (liang vs native), say so and label every intercept.
- **Graded premium at k=2:** l. 262–263 "+0.0144 at k=2"; l. 390, Fig. 3 caption, Table 4 "+0.0102 at k=2".
- **Intensity at full profile:** l. 388 "+0.0022 (CI [+0.0008, +0.0036])"; l. 1277 and Table 4 "+0.0039 (CI [+0.0025, +0.0053])". Both are described as the identical-support comparison against a positive-only fold.
- **iALS confidence gain:** l. 358 "+0.0194 (CI [+0.0157, +0.0231]) over the best binary build"; Table 1 footnote "+0.0170 full of this cell is the confidence weighting". If the binary comparator differs between the two, name both.
- **Concept channel at k=16:** l. 1297 "0.1498/0.0673"; l. 1340 "0.1372 in this gate's arm"; l. 1371 "0.1498/0.0673". Which is the deployed number?
- **Sign flip:** R2 row "+0.2096 (CI [+0.1986, +0.2208])"; R4 row "0.4298 → 0.1345, −0.2953"; l. 965 "−0.2012" (concept channel). Three sign flips, none defined. Also R4's n=7,563: why not 10,000?
- **R4 references:** 0.4298 (sign arm) vs 0.4198 (neutralisation arm) are both "the user's whole rated history folded" (caption, l. 1259–1260). Same fold, same users, different numbers — explain.
- **Abstract l. 22–23:** "against a RecVAE bar of 0.3540/0.2497 reproduced to the published third decimal" — the ML-25M NDCG@10 bar is not a published number; what was reproduced is ML-20M NDCG@100. Reword.
- **Abstract/intro "every gap that decides an ordering separated":** Table 3 marks RBMF's E0≈P and the conjugate belief as flat; tail rows contain further ≈ marks. Say "every ordering we count as distinct".
- **l. 1194–1196:** "−0.0042 on the full profile and +0.0042 on the tail (a coincidence of rounding …)": from Table 3, 0.1718−0.1760 = −0.0042 and 0.0576−0.0534 = +0.0042; fine, but the parenthetical about "different user subsets, 10,000 and 9,281" is the first mention that tail metrics use 9,281 users. State this in Section 7.1 and in every tail CI.
- **l. 1050:** "1.8 unpaired standard errors" — paired bootstrap is available; report it.
- **Section 9, l. 1416–1418** says sub-threshold ratings are "absent from every model's input, ours included", yet Figure 3 and Section 8.1 fold "real signed star levels" including sub-threshold answers. The interview harness evidently uses a different fold-in than the full-profile ruler; state precisely which ratings enter the interview fold-in, the target half, and the candidate pool in each regime.
- **l. 651–653 vs l. 1307–1310:** consistent, but the "random matches popularity at k=1,2,4,8" claim reports only one number (k=8) and no CIs for the "within noise" claims.

### 5.2 Unsupported or overclaimed statements
- l. 78–84 and l. 1183–1191: the RecVAE-vs-instrument reversal is uncontrolled (W2); the intro presents it as a clean finding.
- l. 1205: "The instrument's own numbers are the strongest cells, which is the point of R1" — non sequitur; see W2.
- l. 1349: "lifts a frontier-accuracy recommender substantially" — +0.01 to +0.02 after 16 questions (W4).
- Abstract l. 26–28 and Conclusion l. 1475–1477: report the criterion-relativity, including that items win under three of four rankings.
- l. 119–121: "R1 and {R2,R3,R4} are anti-correlated across published systems" — this is partly by construction: any system without a full-catalogue ranker is marked ✗ on R1 and placed in the elicitation corner, and R2–R4 marks are read "from its published claims" (Table 1 caption), i.e. are subjective. The "bifurcation" is a classification, not a measurement, for most of block (c).
- l. 887: "implicit outperforms the explicit LLM answer by roughly 22×" — an "LLM-elicited ordinal answer" appears nowhere else in the paper; where did it come from, on which users, under what prompt? Either describe it or remove it.
- l. 758–760: "more than 300× more often than chance (0.167 … against 0.0005)" — no CI, ~206 concepts; and a top-10 overlap of 0.167 means on average fewer than two of ten items agree.
- l. 1373: "stays answerable 97.2% of the time" is a property of the answerability rule (two rated members of a concept with hundreds of members), not a discovered fact; the body concedes this at l. 1391–1393 but the abstract does not.
- Table 4, R2 "AUC 0.94 whitened": the direction is the whitened centroid of the same members whose ranking it is scored on. This is not evidence of an open channel.

### 5.3 Missing baselines or controls
- Table 3: a strong item-only consumer with a fair interface (RecVAE/EASE on likes with the decoder-bias fallback for empty input); iALS-with-confidence consuming the signed ratings (the paper's own best ratings-native baseline); the user-kNN consumer (closest to Rashid's original setting); Golbandi's actual tree rather than its node model driven by external heuristics.
- Table 3: numbers under the "with floor" contract, since the caption asserts orderings are floor-invariant but shows only one contract.
- Section 8.2: a set-encoder tower trained *without* interview-shaped sets, to separate the interface (R2/R4) from the training distribution.
- Section 8.2/8.3: seed replicates of the tower and of the concept fold; CI for the text-adapter held-out overlap.
- Section 8.3: an item bank matched in answerability to the concept bank (e.g. items filtered to ≥ the concept answerability rate) — otherwise the channel comparison is confounded with answerability by design, which is the authors' point but leaves "which channel carries more taste per answered question" unmeasured.
- A second dataset (acknowledged). For a paper whose central claim is generality, one ML-25M split is thin, and ML-25M and ML-20M share most of their users and items, so the "bridge" is not an independent replication.

### 5.4 Reproducibility gaps
- Tower: architecture, latent dimension, set-encoder form (DeepSets? attention?), training-set construction (how evidence subsets were drawn, budgets, channel mix), loss, optimiser, epochs, selection criterion, seed. None given.
- Gated fold (Eq. 4): sizes of g and δ, norm cap value, training data and objective, whether concept answers used in training came from the same fold-in half used at evaluation.
- Text adapter: which sentence encoder; closed-form map (ridge? least squares?); regularisation.
- Concept channel: whitening details (how many PCs stripped; "top principal component" vs "all-but-the-top"), IDF weighting, band thresholds, τ for refusal, α=3 shrinkage justification.
- Heuristics: exact HELF/Entropy0 formulas, the rating scale over which entropy is computed, the question bank size and how it was "fixed in advance", what "no popularity floor" means.
- Baselines: RecVAE hyperparameters on ML-25M; Golbandi node model, RBMF and conjugate-belief implementations (prior, noise, dimension); "best-effort" substitutions are promised "in Section 3" (l. 1444) but Section 3 does not list them.
- Bootstrap: 10,000 resamples stated; confidence level and whether percentile or BCa intervals not stated. No multiplicity correction across the ~60 CIs reported.
- The repository is withheld (fine), but the paper must be reproducible from its text; at present it is not.

### 5.5 Anonymisation or bibliography leaks
- l. 379–381: "selected by a validation sweep that optimised NDCG@100 on a previous dataset" and "recorded previously" (l. 370–374) reveal a prior version of this work; harmless if that work is unpublished, otherwise cite it in third person.
- "Pre-registered" (l. 716, 1336) without a registry pointer is either a leak risk (if the registration is public under the authors' names) or unverifiable; the authors should say which.
- Bıyık et al. 2023 author list appears to be wrong (W9); Qin et al. 2020 entry contains working notes; Rashid 2008 missing (W8).
- Footnote 1 is fine.

### 5.6 Presentation defects
- Figure 1 omits measured rows (Mult-VAE 0.3202, TaNP 0.2645, Turbo-CF 0.2622, SASRec 0.1486, RBMF, EDDI) despite the caption promising that "filled markers have a full-profile number on this ruler"; its labels disagree with Table 1 (W1). The x-axis is a count of subjective ✓ marks and does not warrant a scatter plot.
- Table 1: three fold-in symbols (‡, ¶, ★, §, †) and a ten-line footnote; consider a separate "how measured" column. "bold marks the best measured value per column" but the (d) row is bold in liang where RecVAE is higher.
- Table 3: define "≈", "flat" and "no popularity floor" in the caption; give the cold intercept under this contract so readers can see that RecVAE-HELF and RBMF-HELF are below it.
- Table 4: the "Measured" column mixes at least four different experimental regimes (full-profile corruption, fixed answerable bank, cold concept-only, identical-support re-valuation) in one cell each. Each number needs its regime.
- Figure 3: y-axis label "NDCG@10 (full)"; the "full" x-tick is not an interview length; the two curves are visually indistinguishable at the plotted scale — plot the paired difference with its CI instead.
- Eq. (2): "≈ PMI" for an add-0.5 log-ratio is loose; either derive or say "smoothed PMI". Eq. (3): subscripts garbled in the PDF (α, res).
- Notation: t (evidence size), k (interview length), q (questions) are used for the same quantity; d for latent dimension is never given a value.
- Sections 2.2 and 8.2 restate the same battery; Section 4.1 restates Section 8.2's selection results; the paper could lose 6–8 pages without losing content.
- l. 481: "flat since 2019–2020" — this is asserted, and BSPM/Turbo-CF etc. are then dismissed because they do not report this split; a reviewer cannot tell whether the frontier is flat or merely unmeasured on this split.

## 6. Questions for the authors

1. Please confirm or correct the row assignment of Table 1 block (c) against your score vectors, and re-audit every number in Tables 1, 3, 4 and Figure 1 (W1).
2. Was the instrument's set encoder trained on evidence sets drawn under the same question banks/heuristics used in Table 3 and Section 8.2? What is the Table 3 "Ours − this" column when the tower is trained on full profiles only (W2)?
3. What exactly is "no popularity floor", and what are the Table 3 cells under the floor contract that the caption says leaves orderings unchanged?
4. Where do HELF and Entropy0 come from, and what recommender did their original evaluation use? Why is the Golbandi node model "closest" to it rather than user-kNN (W8)?
5. Why do the two R4 full-profile references (0.4298 and 0.4198) differ, and why n=7,563 for the sign flip?
6. Which of 0.0192 and 0.0267 is the cold tail intercept, and under which pool?
7. Which of +0.0102 and +0.0144 is the graded premium at k=2; which of +0.0022 and +0.0039 is the intensity value at full profile; which of 0.1372 and 0.1498 is the concept-channel endpoint at k=16?
8. What is the tower (architecture, d, training set, seed), and what is the seed-to-seed variance of its full/tail NDCG@10 and of your RecVAE realisation?
9. Where was the 1.5% R1 tolerance and the "pre-registered triple gate" registered, and when?
10. Which sentence encoder produces the free-text directions, and what is the CI on the 0.167 held-out overlap?
11. What is the "LLM-elicited ordinal answer" at l. 886, on which users, and why is it in the paper if the answer path is otherwise LLM-free?
12. Please provide the exact author list of the Bıyık et al. 2023 reference.

## 7. Scores (1–5)

| Criterion | Score |
|---|---|
| Relevance | 5 |
| Originality | 2 |
| Technical soundness | 2 |
| Presentation | 2 |
| Reproducibility | 2 |
| Impact | 3 |
| **Reviewer confidence** | 4 |

## 8. Recommendation

**Recommendation: Major revision.**

The methodological thesis is right and TORS-appropriate, the certification bridge and provenance rule are genuinely good practice, and the Section 8.1 design is the correct one. But the manuscript as submitted cannot be trusted at the level of its own numbers: the master table that anchors the taxonomy has its elicitation-corner rows attached to the wrong systems, at least eight further quantities are reported with two different values in different places, and central protocol objects ("no popularity floor", the sign/identity contrasts, the "triple gate") are undefined. The headline RQ1 claim that the instrument is the strongest consumer and that a frontier consumer inverts the classic ordering rests on (a) a consumer trained on interview-shaped data compared with consumers that were not, and (b) a deliberately crippled RecVAE whose HELF cell falls below the no-answer intercept; the authors concede the confound and then draw the conclusion anyway. The instrument itself is a single run of an undescribed architecture. The abstract selects the one bank ranking under which concepts win. None of this is unfixable, and I would be glad to see a revised version, but the revision must include a full numerical re-audit, the architecture and training details, a fairly interfaced frontier consumer in Table 3, and abstract/conclusion claims cut back to what the body supports. If the Table 1 discrepancy turns out to reflect a deeper bookkeeping problem in the harness rather than a transcription slip, I would recommend rejection.

**The single most damaging question a rebuttal must answer:** *When the set-encoder tower is trained on full profiles only (no interview-shaped evidence sets), and RecVAE/EASE/iALS-confidence are given a fair item-only interface with a fallback for unanswered questions, does the instrument still out-score every consumer in Table 3 and does the "frontier consumer inverts the classic ordering" result survive?* If not, the paper's RQ1 headline is an artefact of training on the evaluation distribution and of a crippled comparator, and the contribution reduces to the (valuable but much smaller) observation that fold-based and partition-based consumers treat unanswered questions differently.
