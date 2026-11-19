# TORS Review — Reviewer 2

**Manuscript:** "Elicitation Results Are Instrument-Relative: A Certified Interview-Compatible Recommender for Measuring Cold-Start Preference Elicitation" (36 pp., single-blind)

**Reviewer profile:** mainstream recommender-systems researcher (conversational and cold-start recommendation; RecSys/CIKM).

---

## 1. Summary

The paper argues that the reported ranking of cold-start question-selection heuristics is a property of the (heuristic, recommender) pair rather than of the heuristic. It re-runs the three Rashid et al. (2008) heuristics (HELF, entropy0, popularity) under one fixed protocol on ML-25M (8 questions, 10,000 held-out users, native candidate pool) and varies only the recommender that consumes the answers: six consumers produce three distinct orderings, with a reconstructed Golbandi node model reproducing the published order and the authors' own model / a likes-only RecVAE reordering it (Table 3, p. 24).

From this it derives four requirements (R1 frontier-parity full-profile accuracy; R2 open, set-valued input whose members may be arbitrary latent directions; R3 monotonicity under truthful answers; R4 signed, graded answers) that a recommender must satisfy to serve as a "measuring instrument" for elicitation research, and surveys about 25 systems against them in a master table (Table 1, p. 9), concluding that the literature is bifurcated: accuracy-corner models are item-only indicator models, elicitation-corner models never report full-catalogue accuracy at the EASE/RecVAE bar.

It then builds an instrument: a frozen RecVAE (reproduced to the third decimal on the ML-20M split) with a deep-sets set encoder over its latent that folds (direction, value) pairs, plus a separately trained gated fold for Tag-Genome concept directions and a closed-form linear text adapter. The instrument scores 0.3496/0.2491 full/tail NDCG@10 versus RecVAE's 0.3540/0.2497 (tail indistinguishable, full 1.2% short, paired bootstrap) and passes a six-item verification battery (Table 4, p. 26). Finally, it measures answerability (25.2% of the top-100 films vs 53.0% of 1,031 concepts) and shows that under a discrimination-ranked bank concepts beat items at every budget, while under answerability-weighted banks items win (Table 5, p. 29).

## 2. Relevance to TORS

High. Cold-start elicitation, conversational recommendation, evaluation methodology and reproducibility are all within TORS scope, and the journal is a better home for a 36-page methodological argument than a conference. The paper cites TORS papers (Göpfert et al., Kostric et al.) and engages with the Ferrari Dacrema et al. reproducibility line that TORS readers care about.

## 3. Strengths

1. **The central finding is real and useful.** Table 3 is a clean experiment: same users, same questions, same answers, same budget, same pool, only the consumer varies, and the deciding gaps are paired-bootstrapped over 10,000 users. That HELF starves fold-based consumers (1.59 graded answers of 8) while feeding a lovers/haters/unknowns partitioner is a crisp, mechanistic explanation (pp. 24–25). The community does carry the Rashid ordering forward as a property of the heuristics; showing that it is not is worth publishing.

2. **The certification bridge is exemplary reproducibility practice.** Reproducing EASE (0.4203 vs 0.420) and RecVAE (0.4425 vs 0.442; Recall@20/50 also matched) on the ML-20M strong-generalisation split before porting to ML-25M, dropping Mult-DAE for failing its snap, and demoting the authors' own iALS to "uncertified floor" for the same reason (p. 21) is exactly the discipline Ferrari Dacrema et al. asked for and few papers practise.

3. **The answer-provenance rule (Section 7.4) is a genuinely good methodological contribution.** Stating in advance, and asserting in code, that no recommender-derived quantity and no target-half information may enter a simulated answer is short, checkable, and separates published simulators (PEBOL, UNICORN, ConTS) sharply. I would like to see this adopted.

4. **Honest reporting of unflattering numbers.** The 1.2% full-profile shortfall is reported as separated rather than hidden in a significance test (p. 26); the graded-input premium is shown to vanish at k >= 8 (Fig. 3); the one-level adaptive probe is +0.0015; iALS gains more from ratings than the instrument does (p. 8). Section 10 is unusually candid, including the confound that the instrument is itself one of the consumers and was trained on interview-shaped sets.

5. **The answerability measurement without a bank (p. 29)** is simple and memorable: 25.2% of the 100 most popular films vs 53.0% of concepts. The discrimination-vs-answerability trade-off for items (POLAR item bank answered 4.8% of the time vs 97.2% for concepts) is a clear structural point.

6. **Full and tail NDCG on the unsampled catalogue for every arm**, including baselines that had to be reconstructed, is the right standard and is applied consistently.

## 4. Weaknesses (most serious first)

### W1. The master table's R2–R4 marks contradict the paper's own gap analysis, and R3 is graded by a different rule for the instrument than for its rivals.

Table 1 (p. 9, verified on the rendered image) gives:
- Bıyık et al. [5]: R2 ✗, R3 ✗, R4 ✓
- EDDI/Partial-VAE [37]: R2 ✗, R3 ✗, R4 ✓
- ConTS [29]: R2 ✗, R3 ✗, R4 ✗
- PEBOL [3]: R2 ✓, R3 ✗, R4 ✗
- BCIE [54]: R3 ✓ only

Section 4 (p. 12) says the opposite for four of these rows: "Bıyık et al. hold part of R2 (items and soft attributes both enter as directions)"; "ConTS [29]: R3 plus a unified item/attribute action space"; "EDDI / Partial-VAE [37]: R3 (information gain over a belief) plus set input"; "PEBOL [3]: R3 (per-item Beta) plus R2". Page 10 calls Bıyık "the nearest thing to R2, R3 and R4 together". The caption of Table 1 states "R3 is marked met only where the system maintains a belief or posterior that its published update rule keeps consistent with a new answer" — Bıyık (Gaussian belief), ConTS (Thompson posterior), EDDI (Partial-VAE posterior) and PEBOL (Beta posteriors) all do exactly that, yet all four carry ✗ on R3. Figure 1's x-axis inherits these marks ("no published system exceeds x=2"), so the gap map is built on marks that the text itself disputes.

Worse, the instrument's own R3 ✓ is earned by a different criterion: Section 2 says "Nothing in what follows requires a distribution over u" (p. 4), the instrument is a point estimator, and its R3 is checked by "cold curve rises on a bank fixed in advance" (Table 4). If rivals need a maintained posterior to get R3 but the instrument only needs a rising curve, the requirement is applied asymmetrically. Either apply the outcome criterion to every runnable rival (the R3 curve can be run on user-kNN, RBMF, the conjugate belief and TaNP, which are already in the interview harness of Table 3) or apply the posterior criterion to the instrument and mark it ✗.

This is fixable, but it must be fixed before the bifurcation claim ("no published system occupies both corners") can be accepted, because the table is the anchor of RQ2 and the paper says so ("The anchor of this section is the per-system master table", p. 7).

### W2. The four requirements are shaped around what the instrument has, and the paper does not commit to what its contribution is.

The paper says (p. 3) "It does not claim a new elicitation policy; it claims the recommender that makes policy measurement possible", yet the title and abstract lead with the RQ1 finding, the longest sections (3, 4, 7.5, App. A: about 11 pages) defend a taxonomy, and the "instrument" as a recommender is a RecVAE that scores 1.2% below RecVAE. Read as a finding paper, Table 3 is the result and 25 of 36 pages are scaffolding. Read as a methodology paper, the battery and provenance rule are the result and the master table is an over-long literature review. Read as a system paper, the system is a set encoder on a frozen VAE whose mechanism precedents the authors themselves list (EDDI, TaNP, Bıyık, BCIE), evaluated as a recommender from a single training run with no seed variance. The paper should choose one framing and move the others to support.

On the requirements themselves:
- R1's tolerance "fixed in advance at within 1.5% relative" (p. 2) sits just above the instrument's 1.2% shortfall. Where and when was this fixed? A pre-registration pointer, or a statement that the tolerance was set before the tower was trained, would help.
- R2's "open half" is defined so that only an architecture with a continuous latent interface can attempt it ("That last clause is the one no fixed-column model can attempt", p. 5). It is legitimate to require this, but the paper should not then count it as a discriminating finding that no fixed-column model passes.
- R4 is a property of the input contract, not the model: user-kNN, RBMF, TaNP and the conjugate belief all get ✓ merely for reading ratings. The requirement does no work in separating systems; the work is done by R1 ∧ R2.
- Nothing in the four requirements covers uncertainty over the user latent, which every acquisition rule in the Bıyık/EDDI/ConTS lineage needs (EVOI, information gain, Thompson sampling). An "instrument for elicitation research" that emits a point and defers uncertainty to "a policy that wants one can be built on top" (p. 4) rules out the acquisition criteria most of the elicitation corner uses. This should be stated as a scope limit, not left implicit.

### W3. The instrument-relativity result is weaker than the abstract claims: four of the six consumers are the authors' reconstructions, the one that "reproduces the classic ordering" is a node model rather than Golbandi's method, and the published ordering was not measured on this task.

Table 3's caption is careful ("best-effort core of the published system"), but the abstract says "the order reported by Rashid et al. reproduces under our reconstruction of the node model of Golbandi et al. and inverts under a frontier-accuracy consumer". Three points:

1. Rashid et al. (2008) evaluated on rating-prediction error with a neighbourhood predictor, not on NDCG@10 over an 18k catalogue. The paper never states what metric or model the published order was obtained under, so "reproduces" and "inverts" are across a change of task and metric as well as consumer. Please state the original protocol and say explicitly that you are testing transfer of the ordering to a top-N task.
2. Only two consumers "are run as designed" (p. 23), and one of those is the authors' instrument. The likes-only RecVAE row is explicitly "a control, not a rival" (p. 24). So the "three distinct orderings" rest on: the instrument, a deliberately crippled control, and four best-effort cores. A sceptical reader will ask whether the reconstructions, rather than the consumers, produce the disagreement. Running one more consumer as published (e.g. a published critiquing VAE with its released code, or EASE fed the like-indicators, which is trivial) would make the point robust.
3. Which consumer "inverts"? The exact inversion of H > E0 > P is P > E0 > H, which is the likes-only RecVAE control; the instrument gives E0 > P > H. Page 2 says the frontier consumer inverts "with entRopy0 leading", i.e. the instrument. The abstract should say which.

The paper acknowledges in Section 10 that the inversion does not depend on its own row (Golbandi vs likes-only RecVAE disagree by themselves). That is the right defence, and it should be moved into Section 8.1 and made the load-bearing statement, because the "Ours − row" column and "the instrument is the strongest consumer on full NDCG@10" (Table 3 caption) otherwise read as the authors grading their own model.

Finally, the paper's own mechanism analysis (pp. 24–25) reduces most of the effect to answerability: HELF yields 1.59 answers of 8 and the consumers differ in whether an unanswered question carries signal. This is precisely what Golbandi et al. designed the three-way split for, and Rashid et al. (2002) already explained popularity's role via seen-probability. The finding is therefore partly a rediscovery, and the residual (E0 vs P changing places between consumers with 3.10 vs 3.05 answers) is the genuinely new part but is the part the paper says it cannot explain ("answerability explains why helf divides the table, not the remaining orderings", p. 25).

### W4. The item-vs-concept comparison is not information-matched, so "concepts win under POLAR" partly measures the answer model, not the channel.

An item answer is one rating. A concept answer (Eq. 2, p. 17) is a log watch-lift computed over all of a user's watched members of the concept in the fold-in half, and a concept is "answerable if the user has rated as few as two of its member films" (p. 29). One concept answer therefore summarises up to hundreds of interactions, drawn from the very fold-in history the item channel can only expose one rating at a time. Under a matched budget of m questions, the concept arm is handed a strictly larger slice of the fold-in data per question. Table 5's +0.0189 full at m=8 may be an information-volume effect. A control that caps the members used to form SEL at one or two (matching an item answer's evidence), or that reports bits of fold-in consumed per arm, is needed before "Concepts decouple discrimination from answerability; items cannot" (p. 28) can be read as a channel property.

### W5. Length is not justified; roughly a third of the manuscript could be cut or moved to an appendix.

Concrete candidates:
- Section 3.2 and Appendix A ("What restoring the ratings is worth", about 2.5 pp.) answer a question none of the four RQs asks, and the section concludes "we make no claim about why" (p. 8).
- Section 4.1 ("Feasibility and necessity of the conjunction", 1 p.) repeats Section 5.2's "Why a learned fold" paragraph and Section 8.2's random-bank numbers.
- Section 6.3 (1 p.) is a chain of quotations establishing that critiquing uses one-hot keyphrases; one paragraph suffices.
- Section 7.5 ("Why 'just add genre columns' is not an alternative") repeats pp. 12–13.
- Figure 1 adds nothing that Table 1 does not state and its x-axis is an ordinal count of contested marks (W1).
- The same numbers are restated many times: 0.3496/0.2491 appears in the abstract, contributions, Section 5.1, Section 5.2, Table 4, Section 8.2 and the conclusion; 25.2%/53.0% at least five times; "+0.0102 at k=2" in Sections 2.2, 3.2, Fig. 3, Table 4, Section 8.2 and Section 10; "true and non-transferable" twice verbatim (pp. 2 and 24).
- Table 4 restates numbers from the text on the facing page.

A 24–26 page version would be stronger.

### W6. Coverage of conversational, LLM-based and user-simulation literature is thin for a paper whose thesis is about how elicitation should be evaluated.

The conversational line is represented by Christakopoulou 2016, Sun & Zhang 2018, EAR, SCPR, UNICORN, He et al. 2023 and the Jannach survey. Missing, and directly relevant to the "simulated user is granted information a cold user does not hold" argument (p. 10, Section 7.4):
- User-simulation evaluation for CRS: Zhang & Balog (KDD 2020) "Evaluating Conversational Recommender Systems via User Simulation"; the Balog & Zhai user-simulation survey line; UserSimCRS (Afzali et al., WSDM 2023); Wang et al. (EMNLP 2023) LLM-based user simulators for CRS (iEvaLM) and the follow-ups auditing LLM simulators for leaking the target item — exactly the provenance failure the paper describes.
- Language-based CRS models with a full-catalogue ranker (ReDial, KBRD, KGSF, CRSLab) are absent; several of them do rank the full catalogue, so the claim on p. 10 that "the consumer is either not a full-catalogue ranker or is not reported as one" needs qualification.
- Bayesian elicitation in latent space with a real recommender: Vendrov et al. (AAAI 2020) "Gradient-based optimization for Bayesian preference elicitation" is the closest predecessor to R2's "continuous direction token" leg and is not cited.
- Meta-learning cold start (MeLU, MAMO) beyond TaNP; Sepliarskaia et al. (RecSys 2018) "Preference elicitation as an optimization problem"; Anava et al. (WWW 2015) budget-constrained cold start; and the interview-tree successors of Zhou/Sun are the item-channel elicitation baselines a cold-start reviewer expects in Table 1.
- Critiquing coverage is fair but stops at 2022; LLM-based critiquing since then should be checked.

The treatment of the cited work is fair in tone; the omissions are the problem.

### W7. Reproducibility of the instrument itself is incomplete, and the code is "to be added".

Section 5.1 gives d=200, width 512, 1.07M parameters and a prose curriculum, but not: learning rates (two are mentioned, neither given), epochs, batch size, curriculum mixture proportions, the norm cap on the accumulated shift in Eq. 4, the refusal threshold τ, the SEL/VAL band thresholds that define like/neutral/dislike, the concept operator's training objective and data, the identity of the "frozen general-purpose sentence encoder" behind Table 2, or p̂_uc in the NPMI expression (p. 18). The operator ablation of Section 6.2 was run "on an earlier tower" (footnote 1, p. 19), so its numbers (0.1759/0.0841 etc.) do not belong to the system evaluated elsewhere. One training run, no seed replicates (acknowledged, p. 32). The repository URL is a placeholder. For a paper whose thesis is "certify your instrument", the instrument should be reproducible from the paper.

### W8. Some verification items are weaker than their "pass" suggests.

- R3's "monotone ρ=1.0" is Spearman over four budget points (q=2,4,8,16); that is not evidence of monotonicity.
- R2's "held-out concept folds from its name alone at >300× chance" is 0.167 top-10 overlap against a 0.0005 random baseline; any random baseline over 18k items is near zero, so 300× is not informative about usefulness. Cosine 0.638 to the true direction is the meaningful number and it is modest.
- The wrong-user control is reported as "−0.0986" without saying relative to what (Table 4).
- Table 4's R4 references (0.4298, 0.4198) are on a different fold-in regime than every other number in the paper, which the caption admits; putting them in the same table invites misreading.
- The published TaNP reaching 0.0060/0.0005 NDCG@10 (p. 7), 27× below Most-Popular, looks like a scoring bug (e.g. ranking predicted ratings over never-exposed items with no popularity term) rather than a property of rating regression; Cremonesi et al. report rating predictors being weak at top-N, not 27× below popularity.

### W9. A practitioner would take away the three reporting rules of Section 9 and little else.

"Report the consumer with the strategy", "score the whole catalogue" and "state the answer path" are sound and I endorse them. But the instrument as built needs the Tag Genome (acknowledged, p. 31), the text adapter "has not been exercised in an interview curve" (p. 32), the adaptive probe is +0.0015, and question choice "matters far less than answerability itself" (p. 27). A practitioner therefore learns (a) ask answerable questions and (b) name your recommender; (a) is Rashid 2002 and (b) is hygiene. The paper should say plainly what is new for practice beyond hygiene, or reposition the contribution as methodological.

## 5. Detailed comments

**Framing and terminology.** The paper leans on a private vocabulary — "ruler", "tower", "bridge", "snap", "deployment currency", "certified", "instrument", "corner", "leg", "arm", "bank", "intercept" — that a RecSys reader has to learn. "Certified" in particular suggests third-party certification; "reproduced" or "verified" is more accurate. Consider a one-paragraph glossary or standard terms.

**Sentences I had to read twice.**
- p. 5: "A system may satisfy R1 while being separably behind the bar, and ours is."
- p. 5: "Two halves, and only the second can fail."
- p. 8 / Table 1 caption: the iALS gain is +0.0170 in the caption and +0.0194 in the text, against two different comparators; the appendix explains. Pick one.
- p. 14: "at initialisation the encoder is the frozen substrate exactly and every departure from it has to be earned by the loss."
- p. 24: "A consumer that folds observations is paid per graded answer, and helf therefore starves it."
- p. 25: "a coincidence of rounding: the two metrics are computed over different candidate sets and different user subsets" — the −0.0042/+0.0042 is not a rounding coincidence; it is two different quantities that happen to share a magnitude.
- p. 27: "it is not free there and worthless here."
- Abstract: "with every gap that decides one of the three distinct orderings separated by a paired per-user bootstrap" — "separated" is used throughout as a technical term that is not defined until p. 21.

**Structure.** Results are answered in RQ order but RQ2 is answered by Sections 3–4, so Section 8.2 opens by pointing backwards. Consider folding the gap analysis into an RQ2 results subsection. Section 2.2's per-requirement checks anticipate numbers (+0.0039, +0.0102) that are only explained in Section 8; either strip the numbers from Section 2 or forward-reference explicitly.

**Tables and figures.**
- Table 1: see W1. "Bold marks the best measured value per column" but only RecVAE is bold; the CASPER row label is also bold, which visually competes.
- Table 1 footnotes run to 17 lines; move the disposition rules into Section 3.3 prose (already partly there) and keep the caption to symbols.
- Figure 1: rhetorical; the x-axis is contested (W1); consider dropping.
- Figure 2: fine, but "policy token: any φ ∈ R^d, emitted directly" is never exercised in the paper.
- Figure 3(b): the y-axis label collides with the panel title; the −0.0014 at k=16 has a CI crossing zero, so say "indistinguishable" in the caption rather than "positive with a clean interval only while evidence is scarce".
- Table 3: excellent table. Add the Rashid protocol/metric as a footnote (W3). State in the caption that bold marks are per row.
- Table 4: put the numeric criteria (tolerance 1.5%, etc.) in the Criterion column so a reader can check pass/fail without the text.
- Table 5: three budgets under POLAR and one under pop; the text mentions four rankings. Show all four rankings at all budgets; it is one small table.

**Notation.** φ, y, μ_t are defined in Section 2; Q_i, β_i in 5.1; W_d in the additive-union score (p. 18) is undefined; p̂_uc undefined; τ undefined; n=7,563 for the sign flip (Table 4) is unexplained (why not 10,000?).

**Running head** says "Makarova et al." for a single-author paper.

**References.** [43] lacks a venue; [28] has broken line formatting; [16] and [56] are arXiv preprints from late 2025/2026 cited for empirical claims — flag them as preprints in the text.

**Minor.** p. 2 "closest in kind to the user-based models the heuristics were validated on" — say what those models were. p. 7: the TaNP paragraph is a page-long footnote and belongs in the appendix. p. 16: "Phrases whose meaning the genome vocabulary does not carry at all ... return lists we would not defend" is good, but Table 2 is then four selected successes; state how many phrases were tried.

## 6. Questions for the authors

1. Please reconcile Table 1's R2/R3 marks for Bıyık, EDDI, ConTS and PEBOL with the Section 4 text, and state one R3 criterion applied identically to the instrument and to every runnable rival. Does the bifurcation claim survive if the posterior-maintaining systems get R3 ✓?
2. Under what metric and predictor did Rashid et al. (2008) obtain HELF > entropy0 > popularity, and why is the Golbandi node model the right stand-in for it?
3. Was the 1.5% R1 tolerance fixed before the set encoder was trained? Is there a record?
4. If SEL is computed from at most one or two watched members per concept (matching an item answer's evidence), does the concept channel still win under POLAR?
5. What is the seed-to-seed variance of the instrument's full-profile NDCG@10? Even three seeds would put the 1.2% shortfall in context.
6. Can the R3 fixed-bank curve be run on the four reconstructed consumers of Table 3, so that R3 is an outcome check for everyone?
7. Why does the published TaNP score 27× below Most-Popular? Have you checked that its scores are not being ranked over items the model never sees in training?
8. Which sentence encoder and which linear fit produce the text adapter, and how many free-text phrases were tried before the four in Table 2?
9. How would an EVOI- or information-gain-based policy be run on this instrument, given that it emits a point and no posterior?
10. When will the code and per-user score vectors be available for review?

## 7. Scores (1–5)

- Relevance: **5**
- Originality: **3** — the RQ1 experiment and the provenance rule are new; the requirements, taxonomy and architecture are assembled from cited precedents, and the answerability mechanism is substantially known.
- Technical soundness: **3** — the statistics and bridge are careful; the master table's marks are internally inconsistent (W1), R3 is applied asymmetrically, the channel comparison is not information-matched (W4), and there is one training run.
- Presentation: **2** — clear sentences but far too long, heavily repetitive, idiosyncratic vocabulary, and the contribution framing is not committed (W2, W5).
- Reproducibility: **3** — baselines are exemplary; the instrument's own training recipe is incomplete and code is not yet released (W7).
- Impact: **3** — the reporting recommendations could change practice; the instrument itself is dataset-bound.
- Confidence: **4** — I work in this area and read the full manuscript including rendered tables; I did not run code.

## 8. Recommendation

**Major revision.**

The paper has a publishable core: a well-controlled demonstration that heuristic orderings do not transfer across consumers, a reproducibility bridge that sets a standard, and a provenance rule I would like to see adopted. But the manuscript in its current form cannot be accepted because its central evidence for the bifurcation claim (Table 1) carries requirement marks that contradict the paper's own gap analysis and applies R3 by a stricter rule to rivals than to the authors' system; because the concept-vs-item comparison confounds channel with answer information volume; and because the paper is roughly a third longer than its content supports while not committing to whether it is a finding, a methodology or a system paper. All of these are addressable without new large-scale experiments: fix and re-derive the table, run the R3 curve on the reconstructed consumers, add an evidence-matched SEL control, cut Sections 3.2/4.1/6.3/7.5 and Appendix A to their essentials, extend related work on user simulation and LLM-based CRS evaluation, and release the code. I would be glad to review a revised version.
