# Review — Reviewer 2 (mainstream RecSys / conversational & cold-start)

Manuscript: "Elicitation Results Are Instrument-Relative: A Certified Interview-Compatible Recommender for Measuring Cold-Start Preference Elicitation" (33 pp., review mode)

## 1. Summary

The paper argues that the reported ranking of cold-start question-selection heuristics (Rashid et al.'s HELF > entropy0 > popularity) is not a property of the heuristics but of the (heuristic, consuming recommender) pair. Holding users, budget (n=8), answer model, question bank and candidate pool fixed on an ML-25M strong-generalisation split, six "consumers" (the authors' set-encoder tower, a likes-only RecVAE, a Golbandi-style node model, RBMF, a conjugate-Gaussian belief over a frozen MF, and TaNP) produce three different orderings (Table 3, p. 24).

From this the authors derive four requirements a recommender must meet to serve as a "measuring instrument" for elicitation research (R1 frontier-parity full-profile accuracy, R2 open set-valued input, R3 monotonicity, R4 signed/graded answers), tabulate ~25 published systems against them (Table 1, p. 9), argue that the literature bifurcates into an "accuracy corner" and an "elicitation corner", and present a system that meets all four: a frozen RecVAE-class tower with a permutation-invariant set encoder over its latent, into which items, Tag-Genome concepts (whitened centroids, signed PMI-style answers, gated fold-to-point operator, Eq. 4) and free-text phrases (a linear adapter from a sentence encoder) fold as (direction, value) pairs. The tower reaches 0.3496/0.2491 full/tail NDCG@10 vs. RecVAE's 0.3540/0.2497, and passes a self-defined verification battery (Table 4, p. 26). Finally the instrument is used to compare item and concept questions under a matched protocol; under a discrimination ranking concepts win at every budget because divisive concepts remain answerable (97.2%) where divisive films do not (4.8%), while under the other three rankings items win.

## 2. Relevance to TORS

High. Cold-start elicitation, conversational recommendation and evaluation methodology are squarely in scope, and a journal is the right venue for a 33-page argument about how such systems should be measured. The relevance question is not whether the topic fits but whether the paper's three candidate contributions (a finding, a methodology, a system) are each carried by the evidence.

## 3. Strengths

1. **The central experiment (Section 8.1, Table 3) is a good idea and is executed with unusual care.** Fixing everything except the consumer and showing three separated orderings over 10,000 held-out users, with paired bootstraps, is a genuinely useful piece of evidence that the field has been treating a pair-property as a strategy-property. The mechanism offered (p. 24, ll. 1165–1179: HELF yields 1.59 graded answers of 8 vs. 3.10 for entropy0, and only a tree consumer is "paid" by an unanswered question) is concrete and testable.
2. **The certification bridge (Section 7.2) is exemplary practice.** Reproducing EASE (0.4203 vs 0.420) and RecVAE (0.4425 vs 0.442, plus Recall@20/50) on the ML-20M split before porting, and dropping Mult-DAE and demoting iALS when they fail to snap, is the kind of hygiene Ferrari Dacrema et al. asked for and few papers deliver.
3. **The answer-provenance rule (Section 7.4)** is a clear, checkable statement of a failure mode (circular simulators) that is real in the CRS literature, and the paper is honest that it buys non-circularity, not ecological validity.
4. **Full and tail NDCG@10 reported everywhere,** with CIs on every headline delta, and an explicit statement that "an interval containing zero is a failure to separate, not a demonstration of equivalence" (Def. 2.1, p. 5).
5. **Candid limitations.** Section 9 admits the single dataset, that sign is "not measurable on this protocol at all" at full profile (ll. 1424–1426), that the instrument is one of its own consumers, and that the RecVAE-vs-instrument interview gap "is not a controlled attribution" (l. 1186).
6. The concept-vs-item comparison under a *fixed* selection criterion (Section 8.3) is the right way to compare channels, and the answerability numbers (25.2% of top-100 films vs 53.0% of 1,031 concepts) are a useful quantification of something usually asserted.

## 4. Weaknesses (most serious first)

### W1. The paper does not commit to what its contribution is, and the three candidates are unequally supported.

The title leads with a finding; the subtitle names a system; the abstract (ll. 16–24) and contribution list (ll. 145–172) give equal billing to a finding (RQ1), a methodology (four requirements + battery + taxonomy, RQ2/RQ3), a system (RQ3) and a second finding (RQ4). The introduction then says "this article is about the instrument rather than about a policy" (l. 69) and "it claims the recommender that makes policy measurement possible" (ll. 142–143). Yet the authors repeatedly concede that the system is a conjunction of borrowed parts ("The components are borrowed; only their conjunction is claimed", l. 959; similar at ll. 174–181, 670–677, 924–925), and the system's own certificate shows it is 1.2% *behind* the bar on full profile with a CI excluding zero (Table 4). Meanwhile the strongest piece of the paper — Section 8.1 — occupies two pages of thirty-three.

A reader who wants the finding gets it buried at p. 23; a reader who wants the system finds no architecture description (see W3); a reader who wants the methodology finds requirements written around the system (see W4). I would ask the authors to pick one spine (my recommendation: the instrument-relativity finding plus the protocol/battery as the methodological contribution, with the system as the enabling artefact) and restructure so that the length is spent where the evidence is.

### W2. The instrument-relativity finding is weaker than the framing, for four reasons.

(a) **Most of the six consumers are the authors' own reconstructions.** Table 1's footnotes and Section 7.3 (ll. 1045–1047) state that "user-kNN, RBMF and the conjugate belief are best-effort cores of the elicitation-corner systems rather than their published systems", TaNP is "objective-ported" (l. 336), and the likes-only RecVAE is "a control, not a rival" (l. 1153). Only the authors' instrument is run as designed. So the headline "the order reported by Rashid et al. reproduces under the node model of Golbandi et al." (abstract, ll. 11–12) is a statement about a best-effort core with substitutions, not about Golbandi's system. Please say so in the abstract and Section 8.1, not only in a footnote and Section 9.

(b) **The mechanism the paper itself identifies is answerability under an offline log, not the recommender.** Lines 1165–1179 explain that HELF asks unanswerable questions and that fold-in consumers are "paid per graded answer". That is a property of the *offline answer model* (an item is answerable iff it appears in the user's fold-in half) as much as of the consumer. Rashid et al. ran online with users who could rate whatever they recognised; the original entropy0 heuristic was introduced precisely because pure entropy selects unseen items (which the paper itself notes at ll. 513–516). The paper needs to separate "the consumer changes the ordering" from "the offline unanswerability rule changes the ordering", e.g. by re-running Table 3 with unanswered questions not spending budget, or on the answerable-filtered bank of R3. The claim that the ordering is "instrument-relative" would then be cleaner. As it stands, ll. 1177–1179 concede that answerability "explains why helf divides the table, not the remaining orderings", and the remaining orderings (entropy0 vs popularity) differ by 0.001–0.004 in several rows (RBMF .1644/.1643; conjugate belief .1614/.1630/.1627 marked "flat"). Three orderings from six consumers, two of which are near-flat, is less striking than the abstract implies.

(c) **The ordering is also metric-relative within a single consumer** (the instrument itself gives E0 > P > H on full and E0 > H > P on tail; likes-only RecVAE flips entropy0/popularity between full and tail, ll. 1193–1197). If the ordering depends on consumer, metric, budget and answer rule, the take-away collapses to "report all your settings", which is correct but not a finding that needs 33 pages.

(d) **The authors' consumer is both a row and the reference, and it was trained on interviews** (ll. 1186–1188). The instrument's "entropy0 wins" cell is therefore the output of a model fitted to the interview distribution it is evaluated on, while the other five consumers were not. Section 9 acknowledges that the inversion survives without the CASPER row; then the CASPER row and the "Ours − this" column of Table 3 add little to the RQ1 claim and read as promotion. I would move "Ours − this" out of Table 3.

### W3. The instrument itself is not described. This is a reproducibility failure, not a presentation nit.

Section 5.1 says the tower is "the graded-native set-encoder T2, trained once in its pre-registered configuration" (ll. 715–716). Nowhere in the paper is the set encoder's architecture given (depth, width, pooling, how (d, v) pairs enter), nor the training objective, nor how "trained on interviews" (l. 1187) was done (which users, which question distributions, how many evidence-set sizes, what loss on the target half), nor how the tower is "initialised from that model" (RecVAE). Section 5.2 says two folds exist, "the tower's own set encoder for item evidence, and a trained gated operator for concept evidence" (ll. 725–726), and only the second gets an equation (Eq. 4), and even that omits the sizes of the gate/MLP, the norm cap, and the training data. The set-encoder training is the single design decision that makes the paper's system claim stand or fall, and "pre-registered configuration" is not a substitute for stating it. Similarly, the text adapter (ll. 754–762) is "a single global linear map ... in closed form" from an unnamed "frozen general-purpose sentence encoder". Hyperparameters, seeds, and variance across training runs ("One certified training run", l. 1221) are absent. A repository URL withheld for review is fine; a paper that cannot be re-implemented from its text is not.

### W4. The four requirements are written around the system, and the pass criteria are tuned to what it achieves.

- R1's tolerance is "fixed in advance at within 1.5% relative" (l. 101) and the instrument lands at 1.2% (Table 4). The pre-registration is unverifiable, and a reader will notice the coincidence. Please give the reader a reason to believe the margin was not chosen after the fact (a date-stamped registration, or a sensitivity statement).
- R2's "open half" is defined so that "That last clause is the one no fixed-column model can attempt" (l. 244). A requirement no other system can even attempt is a description of the architecture, not a requirement derived from the elicitation task. The evidence for openness is thin: a held-out-concept adapter with top-10 overlap 0.167 and cosine 0.638 (ll. 758–760), and four hand-screened phrases in Table 2 with no quantitative evaluation and an admission that phrases outside the genome vocabulary "return lists we would not defend" (l. 773). The free-text channel is never used in any interview curve. Meanwhile Biyik et al. are marked as failing R2 because their vocabulary is 164 tags fixed before the user arrives (ll. 577–579) — but the CASPER concept channel that actually carries results is built over a *fixed* 1,031-tag Genome vocabulary. The distinction is one of degree and of a demonstration that is illustrative only.
- R3 is checked on a bank "filtered per user to answerable questions" (Table 4 caption). That makes monotonicity easy to pass and is not the regime a deployed system faces; Section 6.4 in fact admits that in "deployment currency" the positive-only concept curve "sinks below the intercept by the eighth question" (l. 1339). R3 as certified and R3 as it matters diverge.
- R4's headline evidence is a full-profile sign flip of −0.2953 (Table 4). Negating every rating of a user's whole history and watching NDCG collapse shows that the model reads the value; it does not show that sign is *useful*, and the paper concedes the useful quantity "is not measurable on this protocol at all" (ll. 1424–1426). What is measurable is small: +0.0039 full for intensity at full profile, +0.0102 at n=2 and nothing from n=8 (Fig. 3). Given that 55.7% of a user's history (their dislikes) is discarded by the protocol (l. 1417), the paper is evaluating R4 on a ruler that cannot see most of what R4 is for.

Taken together the battery reads as "the system passes tests the system defined". The paper would be stronger if the requirements were motivated from the elicitation literature's needs *first* (e.g. what a policy study by Christakopoulou et al. 2016, Kostric et al. 2021, or Sepliarskaia et al. 2018 would have needed), with the pass thresholds justified independently of where the system lands.

### W5. The RQ4 channel comparison is not information-matched, and the abstract reports only the favourable ranking.

A concept "answer" is SEL/VAL (Eqs. 2–3) computed from *all* of the user's fold-in ratings on the concept's members (answerable when "as few as two" members are rated, l. 1388, but typically many more). An item answer reveals one rating. One concept answer therefore transmits a summary of potentially dozens of fold-in interactions, which no human user could produce, whereas one item answer transmits one. The "concepts decouple discrimination from answerability" result (ll. 1373–1374) may be largely "one concept answer contains more of the fold-in half than one item answer". The authors treat human answer noise as out of scope (ll. 91–93), but this is not noise; it is a systematic difference in information per answer that the matched protocol does not match. A control that caps the number of member ratings entering SEL (e.g. two, the answerability threshold) would settle it.

Separately, the abstract (ll. 26–28) states "ranking a question bank by discrimination makes the concept channel beat the item channel on both metrics at every budget" without mentioning that under three of the four rankings items win, at n=16 by −0.0782 full under pop (ll. 1365–1366) — four times the largest concept advantage (+0.0189). The body is honest about this (ll. 1376–1379); the abstract is not.

Also, the answerability numbers (25.2% vs 53.0%) are counts over the log under a structural rule and "involve no question bank at all" (l. 1385); they do not need the instrument. Presenting them as "what a certified instrument makes measurable" (RQ4, l. 139) overstates the instrument's role.

### W6. Length is not justified by the evidence density; roughly a third could be cut or moved.

Candidates: Section 3.2 (1.5 pp. of iALS/RBMF hyperparameter and centring decompositions, ll. 351–391, which the authors themselves say they "use ... to decompose and never as a headline"); the TaNP objective digression (ll. 330–349); Section 3.3's per-row dispositions (ll. 443–468); Section 4.1's rehearsal of arguments already made; Section 6.3's critiquing lineage (a paragraph would do); Section 7.5, which restates Section 4's column argument. The paper also repeats itself: "true and non-transferable" (ll. 67 and 1161), "the components are borrowed; only the conjunction is claimed" (four times), the 0.3496/0.2491-vs-0.3540/0.2497 sentence (abstract, contributions, 5.1, 8.2, conclusion). Conversely, the results that matter (R3 curves, the four-ranking channel comparison of 8.3, the deployment-currency concept curves) are reported only as numbers in prose. Two figures and one table for Sections 8.2–8.3 would replace two pages of text and be more readable.

### W7. Related-work coverage of conversational/critiquing/LLM-based CRS is selective.

The critiquing lineage (Section 6.3) is covered well. The CRS coverage is thin for a paper whose title is about conversational elicitation:
- Christakopoulou et al. (KDD 2016, "Towards Conversational Recommender Systems") is the canonical cold-start item-elicitation-with-a-learned-consumer paper and is the obvious precedent for both RQ1 and RQ4; it is not cited. Nor is Christakopoulou et al. 2018 (Q&R).
- Attribute-based CRS beyond UNICORN: EAR (Lei et al., WSDM 2020), CRM (Sun & Zhang, SIGIR 2018), SCPR (Lei et al., KDD 2020) are the systems whose "ask attribute vs. recommend item" decision is exactly the channel question of Section 8.3.
- Usage/answerability-oriented elicitation: Kostric, Balog & Radlinski (RecSys 2021) on usage-related questions, and Radlinski et al. (2019) coached elicitation, both directly on what a cold user can answer.
- User simulation for CRS evaluation: Zhang & Balog (CIKM 2020), Afzali et al. (UserSimCRS), and Wang et al. (2023, iEvaLM) are the literature that the answer-provenance argument of Section 7.4 is really in conversation with; none is cited.
- LLM-based CRS: only He et al. 2023 and PEBOL appear; Chat-REC, LLM-simulated users, and the LLM-as-consumer line that would be the most natural "seventh consumer" for Table 3 are absent. GATE (Li et al. 2023) is about eliciting human preferences for general tasks, not a CRS, and sits oddly in Table 1.
- Surveys: Jannach et al. (ACM CSUR 2021) and Gao et al. (2021) would let the taxonomy of Section 3 be positioned rather than invented.

Fairness: characterising every elicitation-corner system as failing R1 because it "does not report full-profile accuracy" (ll. 127–128) is accurate but somewhat unfair, since those systems were built for a different task and the requirement was defined here. The paper acknowledges this at l. 424 (a footnote symbol), but the prose of Section 3.3 ("either blank or measurably weak", l. 471) does not.

### W8. What should a practitioner do differently?

After 33 pages the actionable advice is two sentences at ll. 1478–1481: report the consumer, use a full-catalogue metric, state the answer path; and study policies on a certified consumer. The first is sound and mostly known. The second requires the authors' artefact, which is tied to ML-25M's Tag Genome for its concept channel and to an illustrative-only text adapter; the paper gives no guidance on porting the instrument to a catalogue without a genome (books, music, e-commerce), nor on what a practitioner with their *own* production recommender should do — the paper even says such a designer "should arguably believe the tree" (ll. 1203–1204), which undercuts the case for the instrument. A short "how to certify your own consumer" section, using the battery, would give the paper the practical pay-off the title promises.

### W9. Internal inconsistencies between text and Table 1.

- Line 356 attributes 0.3756 (binary) vs 0.2822 (ratings) to *user-kNN*; Table 1 lists user-kNN native at .2718/.1383 and *RBMF* native at .3756/.2377.
- Lines 450–451 give "the node recommender that is the Golbandi tree's core at 0.3065/0.1895 and the RBMF/Functional-MF seed at 0.2915/0.2071"; Table 1 lists RBMF at .3065/.1895 and the *conjugate belief* at .2915/.2071, and the Golbandi user-kNN row at .2153/.1077.
- Line 734 says the closed-form conjugate accumulator reaches "0.2267/0.1300 ... (Table 1)"; in Table 1 that number sits on the "Biyik soft attributes" row while "Conjugate belief (frozen MF)" is .2915/.2071. Table 3 then labels its consumer "Conjugate belief (frozen MF), after Biyik et al." Which of these is the same object?
- Section 7.3 lists an NDCG@100 for EDDI (0.3110, l. 1054), implying EDDI was run on the ruler, but I could find no NDCG@10 for EDDI in Table 1.
- Section 8.2 (l. 1297) gives the cold intercept as 0.1277/0.0267; Section 7.1 gives 0.1279/0.0192 and Section 6.2 0.1279. Please reconcile or explain (different pools?).

These must be fixed; as it stands a reader cannot trust which number belongs to which system.

### W10. Statistical reporting.

Dozens of paired CIs are reported with no discussion of multiplicity, and every model-side number comes from a single training run. For the small effects the paper leans on (+0.0015 adaptive probe, +0.0022 graded at full profile, +0.0036 uncertainty selector) seed variance of the tower would plausibly be of the same order. Please report at least the across-seed spread of the tower's full/tail NDCG@10.

## 5. Detailed comments

**Structure.** Sections 2–4 (requirements, taxonomy, gap analysis) span pp. 4–14 before any system is described, and the RQ1 experiment that motivates everything appears at p. 23. Consider: Introduction → RQ1 experiment (the motivation) → requirements → system → battery → RQ4 → related work. The taxonomy would then be related work, where it belongs.

**Internal jargon.** Terms that appear without definition or that read as project-internal: "certified pair" (footnote 2), "T2" (l. 716), "pre-registered triple gate", "retrained model", "clip-up curve", "counterfeit control" (ll. 1336–1341), "deployment currency", "canary" (l. 255), "existential controls/leakage" (ll. 266, 1304), "the ruler". "Certified" itself is used ~30 times; the certification is by the authors against their own criteria, so "verified" or "bridged" would be more accurate.

**Sentences I had to read twice.**
- ll. 7–13 (abstract): a single 70-word sentence with five "holding ... fixed" clauses.
- ll. 200–202: "A system may satisfy R1 while being separably behind the bar, and ours is; the requirement is that the shortfall be small, stated and bounded, not that it be statistically absent." This is R1 redefined to admit the result.
- ll. 237–238: "Two halves, and only the second can fail."
- ll. 323–328 on why native numbers are "uniformly higher for a reason that has nothing to do with capability" — the point (larger exclusion set frees top-10 slots) is right but takes three passes to extract.
- ll. 1193–1196: the "coincidence of rounding ... 10,000 and 9,281" parenthesis; why is the tail computed on 9,281 users, and is that stated anywhere else?
- ll. 1282: "it is not free there and worthless here" — the referents of "there" and "here" are unclear.

**Tables and figures.**
- Table 1 is the paper's most important table and is hard to read: the R1–R4 marks depend on symbol glyphs, the footnote symbols (†, ‡, §, ¶, ◇, ♦) are heavily overloaded, and the caption is 20 lines. Split it: one table for the requirement grid, one for measured numbers.
- Table 3: "Ours − this" mixes an RQ1 table with an instrument-promotion column; move it. Bold "best strategy" is fine; add the graded-answer counts (1.59 / 3.10 / 3.05) as a column since they are the explanation.
- Table 4: the R4 references (0.4298, 0.4198) are full-profile corruptions on a different fold-in than Table 1 — the caption explains this but a reader scanning the table will compare them with 0.3496. Consider reporting the drop only.
- Figure 1: a scatter whose x-axis is a 0–3 count of ticks conveys less than the table it summarises; it could be dropped or reduced to a half-column inset.
- Figure 3 is the only results plot; Sections 8.2–8.3 need at least two more (R3 curves with random/fixed-bank controls; the four-ranking channel comparison).
- Eq. (2)/(3): m_uc, e_uc, r̄_i, ρ_c should be defined in one place; the OCR/layout of ll. 846–851 and 1186–1189, 1226–1228 appears garbled in the PDF (text overlapping); please check the compiled output.

**Notation.** "d" is used for both a latent direction and (implicitly) the latent dimension; Q_i vs Q with no dimensions; the fold f is defined on sets in Section 2 but instantiated as a recurrent update in Eq. (4) — say which is the definition. "helf/entRopy0/populaRity" small-caps rendering makes "helf" look like a typo; write HELF.

**Minor.** l. 29 CCS block has broken glyphs; l. 195 footnote 1 fine. "answerability ... we are renaming a known quantity" (l. 520) — good, then use "rateability" or cite Elahi's term in the abstract too. Abstract says "six-family taxonomy and a master table" — "master table" is internal language. Xia et al. 2026 and Gharahighehi et al. 2025 are arXiv preprints cited for empirical claims about attribute-vs-item ordering; mark them as preprints in the text.

## 6. Questions for the authors

1. What exactly is the set encoder (architecture, pooling, training objective, interview-sampling scheme, initialisation from RecVAE)? How many training runs were performed, and what is the across-seed spread of 0.3496/0.2491?
2. When was the 1.5% R1 tolerance fixed, and can that be evidenced?
3. If unanswered questions do not spend budget (or Table 3 is re-run on the answerable-filtered bank of R3), do the six consumers still produce three orderings? This separates "consumer-relative" from "offline-answerability-relative".
4. In RQ4, how many fold-in ratings enter a typical concept answer (SEL) vs. the one rating in an item answer? Does the polaR concept advantage survive if SEL is computed from at most two member ratings?
5. Which row of Table 1 is the "closed-form conjugate accumulator" of l. 734, and which is the "Conjugate belief ... after Biyik" of Table 3? Are the user-kNN/RBMF numbers at ll. 356 and 450–451 swapped relative to Table 1?
6. Why does the tail metric use 9,281 users (l. 1196), and does that affect any tail CI in Table 3/4?
7. The free-text adapter: which sentence encoder, what is the held-out concept evaluation beyond top-10 overlap and cosine, and why is the channel not exercised in any interview curve if it is what makes R2 "open"?
8. How would a practitioner apply the battery to a catalogue with no Tag Genome? What is the concept vocabulary then?
9. Is EDDI's full/tail NDCG@10 available (its NDCG@100 is), and if so why is it absent from Table 1?
10. Would an LLM-based consumer (e.g. He et al. 2023 given the eight answers as text) be a seventh row of Table 3? Given the title's claim about conversational recommenders, its absence is conspicuous.

## 7. Scores (1–5)

- Relevance: 5
- Originality: 3 — the RQ1 experiment and the provenance rule are original; the system is an acknowledged conjunction of existing parts; the requirements are largely a description of the system.
- Technical soundness: 3 — careful bridging and statistics, but the key consumers are reconstructions, the RQ4 comparison is not information-matched, and the tower itself is undescribed.
- Presentation: 2 — over-long, repetitive, internal jargon, results in prose, and text/table number inconsistencies.
- Reproducibility: 2 — baselines are certified admirably, but the paper's own model cannot be re-implemented from the text.
- Impact: 3 — the "report your consumer" message is valuable; the artefact's portability beyond ML-25M+Genome is unaddressed.
- Confidence: 4

## 8. Recommendation

**Major revision.**

There is a good paper here, and it is mostly Section 8.1 plus the protocol discipline of Sections 7.2 and 7.4. To be publishable in TORS the manuscript needs to (i) commit to one contribution and cut the length by roughly a third (W1, W6); (ii) describe the instrument so it can be reproduced and report seed variance (W3, W10); (iii) disentangle the consumer effect from the offline-answerability effect in Table 3 and state plainly that five of six consumers are best-effort reconstructions (W2); (iv) either motivate the four requirements and their thresholds independently of where the system lands or present them as design goals rather than a certificate (W4); (v) match information per answer in the channel comparison and stop reporting only the favourable ranking in the abstract (W5); (vi) fix the text/table inconsistencies (W9); and (vii) engage with the CRS and user-simulation literature the title invokes (W7). None of this requires new modelling; most of it requires re-running existing experiments under one additional control and rewriting.
