# Review — ACM TORS submission
## "Elicitation Results Are Instrument-Relative: A Certified Interview-Compatible Recommender for Measuring Cold-Start Preference Elicitation"

**Reviewer 3** (industry practitioner-researcher; builds and evaluates recommenders in production)
**Manuscript:** 36 pages, TORS layout, single-blind.

---

## 1. Summary

The paper argues that results in the cold-start preference-elicitation literature ("which questions should we ask a new user?") are not properties of the question-selection strategy alone but of the strategy *and* the recommender that consumes the answers. It supports this with one experiment (Table 3, p. 24): three classical heuristics from Rashid et al. (helf, entropy0, popularity) are run under one protocol on ML-25M with eight questions and 10,000 held-out users, and six different "consumer" recommenders produce three different, statistically separated, orderings.

From this it derives four requirements (R1 frontier-parity accuracy, R2 open set-valued input, R3 monotonicity, R4 signed/graded answers) that a recommender must meet to serve as a "measuring instrument" for elicitation research, surveys about 25 published systems against them (Table 1, Fig. 1) and finds none meets all four. It then builds one: a frozen RecVAE tower with a permutation-invariant set encoder (1.07M trainable parameters) over its latent, into which items (own rating), Tag-Genome concepts (a whitened member centroid paired with a signed log watch-lift) and, illustratively, free-text phrases (via a linear map from a sentence encoder) fold as (direction, value) pairs. The instrument scores 0.3496/0.2491 full/tail NDCG@10 against RecVAE's 0.3540/0.2497 on a certified ML-25M ruler (RecVAE and EASE reproduced on ML-20M to the third decimal), passes a six-item verification battery (Table 4), and is then used to (a) measure answerability (25.2% of the 100 most popular films vs 53.0% of 1,031 concepts) and (b) show that whether the item or the concept channel "wins" depends on the bank-ranking criterion (Table 5).

## 2. Relevance to TORS

High. Evaluation methodology, reproducibility and cold-start elicitation are squarely in scope, and the paper explicitly targets practice ("Three practices follow", Section 9). The confound it names, a strategy comparison silently inheriting the recommender it was measured through, is real and I have seen it bite in A/B programmes. A methodological paper that gets teams to report the consumer, score the whole catalogue and state the answer path would change practice. Whether *this* paper does so in its current form is the question below.

## 3. Strengths

1. **The central experiment is simple, well-controlled and useful.** Table 3 holds questions, users, answers, budget and pool fixed and varies only the consumer. The mechanism offered (p. 24: helf yields 1.59 graded answers of 8 against 3.10/3.05; a tree is routed by "unknown" while a fold gets nothing from it) is a concrete, checkable story that practitioners will recognise.
2. **Certification discipline.** Reproducing EASE (0.4203 vs 0.420) and RecVAE (0.4425 vs 0.442, Recall@20/50 also matching) on the canonical ML-20M split before making any parity claim, and *dropping* Mult-DAE and demoting iALS when they fail the bridge (Section 7.2, p. 21), is exactly the discipline Ferrari Dacrema et al. asked for. It is rare and I would like to see more of it.
3. **The answer-provenance rule (Section 7.4)** is short, cheap, and applied as a checklist to published simulators (PEBOL, UNICORN, ConTS). "No recommender-derived quantity, no target half, no oracle over unobserved preference" is something a team can adopt tomorrow.
4. **Honest hedging where it matters.** The paper states that the R1 shortfall is separated (−0.0043 full, CI excludes zero) rather than hiding it in a significance test; footnote 1 (p. 19) flags that the operator-ablation numbers come from an earlier tower; Section 10 admits one dataset, one training seed, no family-wise correction, and that the instrument's own row in Table 3 confounds its input contract with interview training.
5. **Full and tail reported everywhere**, with the frozen decoder's learned bias rather than a popularity floor. Tail numbers are what a production team actually cares about after an interview.
6. **The "portable certificate" idea (Section 9, "Certifying another consumer")** is the most practice-changing content in the paper: a recipe another team can run on their own recommender.
7. **Deployment-currency evaluation (Sections 6.4, 8.3):** every question spends budget whether or not it is answered. That is the right accounting and most elicitation papers get it wrong.

## 4. Weaknesses (most serious first)

### W1. The headline claim "the published order reproduces under the Golbandi node model" is contradicted by Table 3.
Abstract (p. 1): "the order reported by Rashid et al. reproduces under our reconstruction of the node model of Golbandi et al." Intro (p. 2): "Under our reconstruction of the node model ... the published order reproduces." Table 3 caption (p. 24): "among these six consumers only the Golbandi node model reproduces it." Conclusion (p. 32): "the classic consumer reproducing the published order."

The published order is stated on p. 2 and p. 23 as **helf > entropy0 > popularity**. The Golbandi row in Table 3 reads **.1948 / .1686 / .1716**, i.e. **helf > popularity > entropy0** on full NDCG@10 (and .0792 / .0677 / .0696, again H > P > E0, on tail). Only the winner reproduces; second and third place are swapped. The Order column prints "H > P > E0" with no ≈, so by the paper's own convention the P-vs-E0 gap is separated and the published order is *not* reproduced. This is the most-repeated sentence in the manuscript and it is wrong as stated. Either say "helf wins under the node model, as published" and drop "the order reproduces", or show the gap is unseparated and write "H > P ≈ E0".

Relatedly, "inverts" is used loosely. The strict inversion of H > E0 > P is P > E0 > H, which is the *likes-only RecVAE* row, not the instrument's (E0 > P > H full, E0 > H > P tail). The intro (p. 2) says "Under a frontier-accuracy consumer it inverts, with entropy0 leading on both full and tail"; that is a change of winner, not an inversion. Please use precise language: the results are strong enough without it.

### W2. Table 1's R1–R4 marks are inconsistent with the paper's own definitions and with the gap-analysis text.
- **R1 is defined as a measured tolerance** ("within 1.5% relative of the strongest measured baseline on both axes", p. 2), but Table 1 (p. 9) awards R1 to EDLAE (0.3433, 3.0% below the bar), Mult-VAE (0.3202, 9.5% below) and Mult-DAE (no number at all, "dropped"), while the caption says marks are "read from its published claims". So R1 in the table is a reputational grade, not the measured one the instrument is held to. Pick one rule and apply it to every row.
- **R3 is awarded to exactly one system (BCIE)**, although the caption defines it as "maintains a belief or posterior that its published update rule keeps consistent with a new answer" and Section 4 (p. 12) credits **ConTS** ("R3 plus a unified item/attribute action space"), **EDDI** ("R3 (information gain over a belief)"), **PEBOL** ("R3 (per-item Beta)") and calls Bıyık "the nearest thing to R2, R3 and R4 together" (p. 10). In the table those four all get R3 not-met. **PEBOL** gets R2 met in the table, but Section 4 lists R2 as satisfied and R4 as failed, whereas the table gives R4 not-met and R3 not-met; **Bıyık** gets R2 not-met but p. 12 says it "hold[s] part of R2". Fig. 1's x-axis is computed from these marks, so the "no published system exceeds x=2" claim inherits the inconsistency.
- **TaNP** is marked R4 met with input "b" and is run with "its rating channel binarised" (p. 22). A row cannot both consume binary input and be credited with graded answers in the same table.

The master table is the anchor of Section 3 ("so that the bifurcation is visible in measured numbers rather than asserted"). As it stands a careful reader cannot trust it.

### W3. The answer model gives the concept channel a structural information advantage that the RQ4 comparison does not control for.
An item answer is one recorded rating. A concept answer (Eqs. 2–3, pp. 17–18) is a statistic computed over *all* of the user's fold-in ratings on that concept's member set, potentially hundreds of observations condensed into one "answer", plus a refusal band. So one concept question is worth many item questions of information *by construction of the simulator*, not by any property of asking about concepts. The paper acknowledges that answerability accrues "from hundreds of chances" (p. 29) but treats that as a property of generality rather than of the answer model. A human asked "do you like film noir?" does not compute a PMI over their history; they give a noisy scalar. The paper says as much (p. 2; p. 31: "not a person responding to a rendered question") but then draws a channel-level conclusion ("Concepts decouple discrimination from answerability; items cannot", p. 28) that depends on it. At minimum, report the concept channel under a *degraded* answer model (e.g. the four-band answer with the band taken from a single random member rating, or with answer noise matched to the item channel) and show whether Table 5's direction survives. Also state the refusal threshold τ and the band thresholds: neither is given anywhere, and the 53.0% and 97.2% answerability figures are direct functions of them.

### W4. "The disagreement is not attributable to weak recommenders" (p. 25) is not established.
Of the six consumers, four are "best-effort cores" scoring 0.2267–0.3065 at full profile, far below the frontier. The only frontier-vs-frontier comparison is instrument vs RecVAE-likes-only, and the latter is given "the crudest admissible interface" and, on helf, sees 1.6 answers of 8. So the "frontier consumer inverts it" claim rests on one arm whose interface was deliberately crippled, and the "classic consumer reproduces" claim rests on a reconstruction (and, per W1, does not reproduce). The methodological point survives (orderings are not portable), but the two named sub-claims should be softened.

### W5. Nothing is released at review time, and the engineering cost is not stated.
p. 23: "released at [repository URL to be added at submission]". A TORS reviewer is asked to assess reproducibility with no code, no checkpoint, no per-user score vectors. Beyond the URL, the paper omits: training time and hardware; inference latency per folded answer (a set encoder of width 512 over d=200; is a fold sub-millisecond? can it run inside a chat turn?); the curriculum mixture proportions; learning rates; the norm cap on the concept fold; the refusal threshold τ and band thresholds; the sentence encoder behind the text adapter; the whitening details beyond "top PC removed"; RecVAE's own configuration beyond "follows the paper". One training seed (acknowledged p. 32). For a paper whose thesis is that the instrument must be *reusable*, this is the gap that most limits impact.

### W6. Readability. The manuscript is 36 pages and many sentences need two passes.
Examples:
- p. 2: "Under the strongest full-profile model on our ruler, restricted to the likes among the asked items because a binary indicator input can represent nothing else, popularity leads on the full profile."
- p. 5 (R2): "Two halves, and only the second can fail." The reader does not yet know what the halves are.
- p. 6: "a control may legitimately fall below the no-answer intercept: a stranger's taste is worse than no information, not equivalent to it", a parenthetical inside a list item inside a definition.
- p. 8, Section 3.2 is one 20-line paragraph containing nine numbers and three comparators ("the identical-support re-valuation of Section 8.2, +0.0039, is a different comparison").
- p. 25: "(a coincidence of rounding: the two metrics are computed over different candidate sets and different user subsets, 10,000 and 9,281)".
- p. 27: "but it is not free there and worthless here."

The core argument (instrument-relativity) is restated in near-identical words at least six times (abstract, p. 2 twice, p. 24, p. 25, p. 29, p. 32). The provenance rule is stated in full in Sections 2.2, 7.4, 9 and 11. Table 1's caption is 20 lines and Table 3's is 17. I would cut 8–10 pages without losing a result.

### W7. Three intercepts, three names for the same arm, and undefined system names.
- Three "no-answer intercepts" appear: 0.1279/0.0192 (item harness), 0.1277/0.0267 (concept protocol), and 0.1279 full again in Section 6.2 for "an earlier tower" (only footnote 1 tells you so). A reader comparing Section 6.2's 0.1759 with Section 8.3's 0.1498 will conclude the deployed fold got worse.
- The same consumer is called "user-kNN [18]" (Table 1), "the node recommender that is the Golbandi tree's core" (p. 8), "Golbandi node model" (Table 3) and "Golbandi's node model" (p. 21). "Conjugate belief (frozen MF)", "conjugate accumulator", "belief-MF" (p. 36) and "after Bıyık et al." are one arm.
- "pop" is a heuristic in Table 3 (popularity) and a bank ranking in Section 8.3; "ent0", "E0" and "entropy0" are the same thing.
- Never expanded anywhere: **ICF, UNICORN, GATE, RBMF, EDLAE, iALS-WMF, PureSVD, AUC, M&Ms-VAE, BK-VAE, DLC, LLC, CE-VAE.** UNICORN appears only in Table 1 and Fig. 1 and is otherwise referred to as "unified item-and-attribute conversational work [13]". "Genome vocabulary" (p. 15) and "genome tag" (p. 16) precede the definition of the Tag Genome (p. 21). EDDI, TaNP, PEBOL and BCIE appear in Table 1 (p. 9) before their one-line descriptions (pp. 10–12).

### W8. The "free-text" and "continuous direction" legs are asserted in the abstract but not evaluated.
Abstract: "into which items, out-of-catalogue concepts and free-text phrases enter". Section 5.2 then says the text adapter is "an illustration of what the interface permits" and Section 10 that it "has not been exercised in an interview curve". Table 2 is four hand-picked phrases with no metric. The held-out-concept test (0.167 top-10 overlap, cosine 0.638) is the only quantitative evidence, and 0.167 means fewer than two of the top ten items recovered; "300× chance" is a flattering framing of that. The abstract and contribution list should say "items and concepts, with free text illustrated".

### W9. Small traceability gaps between claims and tables.
- Section 2.2: R3 requires "a single answer from cold already clears the no-answer intercept on both channels". The concept number is given (0.1327/0.0412 vs 0.1277/0.0267, p. 27); the item-channel single-answer number is nowhere.
- R1's tolerance is "on both axes", but the native column (CASPER .4244/.3017 vs RecVAE .4273/.3051) carries no CI and is never tested.
- Section 8.3, first paragraph (signed vs positive-only clip, 0.1372 at q=16) is on an unnamed bank; Table 5 (polar bank) has the concept arm at 0.1498 at m=16. Which bank is "the deployment-currency bank" of Section 6.4? Name it.
- Table 3: "−.0000 [−.0032, +.0032]" should read "0.0000".
- p. 27: "beats the concept's own popularity projection by +0.0087" and Table 4's control margins "+0.0371 / +0.0168" are given without the metric, budget or bank.

### W10. What should a practitioner do differently? The paper answers this for *researchers* (Section 9) but not for a team shipping an onboarding interview.
The single most actionable number for a practitioner, that a frontier model fed only likes loses 0.029 full NDCG@10 at eight questions to a model that can read dislikes and star levels, is one cell of Table 3 and one sentence on p. 2. There is no interview-length curve comparing the instrument against the obvious production baselines (RecVAE-likes-only, EASE-likes-only, item-kNN) at k = 1…16; Fig. 3 plots the instrument against itself. That curve would tell a team whether building a graded, signed ingestion path is worth it, which is the practical question the paper's own numbers already answer.

## 5. Detailed line-by-line comments

- p. 1, abstract: "with every gap that decides one of the three distinct orderings separated". Define "deciding gap"; as written it is unclear whether that is every pairwise gap or only the top-1 gap.
- p. 2: "Six consumers realise three distinct orderings". With one "flat" row and one "≈" row, say "three separated orderings, one partially separated, one flat".
- p. 2: "Frontier parity names a tolerance, fixed in advance at within 1.5% relative". Was this pre-registered anywhere citable? If not, say "fixed before the final run".
- p. 3, R2: "the admissible φ_j are all of R^d", but Section 5.2 states two distinct trained operators (set encoder for items, gated fold for concepts). Which operator consumes a policy-emitted direction? Say so explicitly.
- p. 5, R1: "or within a stated and separated margin". You mean the shortfall is bounded by a CI that excludes zero; say that.
- p. 6: "duplicating an answer changes nothing, since the fold responds to information, not cardinality". For a sum-pooled deep-sets encoder this is a learned property, not a structural one. Table 4 says "duplicate assertion fired", which reads as a failure; rephrase ("duplicate check passed").
- p. 7, Section 3.1: the explanation of why native numbers are higher (+0.028 Most-Popular, +0.066 EASE) is good; move it into the Table 1 caption footnote and shorten the prose.
- p. 7: the TaNP published-objective result (0.0060/0.0005) is an interesting negative and belongs in a footnote or appendix, not the protocol section.
- p. 9, Table 1: caption says "bold marks the best measured value per column" but nothing in the R1 column is bold and the instrument row is bold as a row label; clarify. Consider adding a "% below bar" column so R1 is computed rather than asserted.
- p. 11, Fig. 1: hollow markers overlap; "Bıyık, ICF, BCIE, PEBOL" are unreadable at print size. A jittered strip or a small table would do.
- p. 13: the "+0.2096" sign contrast is stated here before Table 4 (p. 26) with no forward reference.
- p. 14, Section 5.1: "an anchor whose weight is gated by the number of answered items" is the one design detail a reimplementer needs and it is one clause. Give the equation.
- p. 15: "Uniformity therefore costs roughly 0.12 full NDCG". The conjugate-accumulator comparator is a best-effort core with shared hyperparameters swept on "a previous dataset" (p. 36); say that here or the 0.12 reads as a property of closed-form folds in general.
- p. 16, Table 2: give the sentence encoder name and dimension; state how the 0.6 cosine screen threshold was chosen.
- p. 17, Fig. 3(b): the y-axis label "graded − likes-only, full NDCG@10" is clipped in my render; check the final PDF.
- p. 18, Eq. 3: λ = 3, selected how?
- p. 18: "implicit outperforms the explicit LLM answer by roughly 22×". The LLM answer is never described (which model, which prompt, which users); either describe it or drop the comparison.
- p. 19, footnote 1 is load-bearing and belongs in the body: "The operator ablation was run on an earlier tower ... the numbers in this paragraph are that tower's."
- p. 21: "Intervals are reported per cell without a family-wise correction". With roughly forty intervals in the paper, say which conclusions would change under Holm or Bonferroni; my guess is none of the large ones, which is worth one sentence.
- p. 22: the NDCG@100 list is a table pretending to be a sentence.
- p. 23, Section 7.6: "The fixed bank of R3 is ... filtered per user to answerable questions". Then it is not "fixed in advance" per user; Table 4's criterion wording should match.
- p. 24, Table 3: header "ent0" vs text "entropy0"; state the ordering convention once.
- p. 26, Table 4, R4 row: n=7,563 appears with no explanation.
- p. 27: "sixteen reach 0.1498/0.0673" is Table 5's polar concept arm; cite it.
- p. 28: "(it declines, −0.045 at q=16)", from what baseline?
- p. 29: "25.2% of the hundred most popular films" is measured against the fold-in half (80%) of the history, and a user who has seen but not rated a film is counted unanswerable. State that the number is a lower bound on human answerability.
- p. 30: "the ordering a tree induces should be believed by a designer whose production system is a tree". Good, and the most practitioner-relevant sentence in the paper; promote it.
- p. 32: the Acknowledgments placeholder mentions generative-AI disclosure; complete before resubmission.
- pp. 35–36, Appendix A: "a validation sweep that optimised NDCG@100 on a previous dataset". Which dataset? This affects three rows of Table 1.
- References: [43] Radlinski et al. 2019 has no venue; [56] is a 2026 arXiv preprint cited for an empirical finding, flag it as a preprint in the text.

## 6. Questions for the authors

1. Please reconcile W1: on Table 3's own numbers, does the Golbandi node model reproduce helf > entropy0 > popularity, or only "helf wins"? Is the P-vs-E0 gap (.1716 vs .1686) separated?
2. Please regenerate Table 1 with R1 computed from the stated 1.5% tolerance and R2–R4 applied by the caption's rule, and confirm that Fig. 1 and the "no published system exceeds x=2" claim survive.
3. What are τ (the refusal threshold) and the like/neutral/dislike band boundaries? How sensitive are 53.0%, 97.2% and Table 5 to them?
4. Does Table 5's direction survive an answer model in which a concept answer carries no more information than one item answer (e.g. the band from a single random member rating, or Gaussian noise matched to the item channel)?
5. What is the wall-clock cost to (a) train the set encoder, (b) fold one answer at inference, (c) rank 18,359 items after a fold, and on what hardware?
6. Which sentence encoder produces the text-adapter directions, and what is the adapter's dimension?
7. Will the certified checkpoint, the per-user score vectors and the bootstrap harness be released under a licence that permits reuse of the checkpoint as a consumer for other groups' policy studies? That is the reuse the paper is arguing for.
8. Can you provide an interview-length curve (k = 1…16, full and tail) for the instrument against RecVAE-likes-only, EASE-likes-only and item-kNN under the same bank? This is the number a practitioner needs to decide whether to build graded ingestion.

## 7. Scores (1–5)

| Criterion | Score | Note |
|---|---|---|
| Relevance | 5 | Core TORS methodology topic. |
| Originality | 3 | The components are borrowed (and the paper says so); the conjunction, the certification bridge and the provenance rule are the new parts. The instrument-relativity experiment is simple but I have not seen it done this cleanly. |
| Technical soundness | 3 | Careful protocol, but the headline "reproduces" claim contradicts its own table (W1), the master table is internally inconsistent (W2), and the channel comparison rides on an asymmetric answer model (W3). |
| Presentation | 2 | Dense, repetitive, 36 pages; undefined acronyms; three names per arm; load-bearing caveats in footnotes. |
| Reproducibility | 2 | Excellent bridge discipline, but no code or checkpoint at review, one seed, and several hyperparameters (τ, bands, curriculum, learning rates, encoder) missing. |
| Impact | 3 | Could be 4 if the released instrument becomes a shared consumer and the certificate is adopted; today it is an argument plus an unreleased artefact. |
| **Confidence** | **4** | I read the full manuscript including every table and figure; I did not re-derive any result. |

## 8. Recommendation

**Major revision.**

The paper has a real, practice-relevant point and the experimental discipline to back it, and I want to see it published in TORS. But the sentence it repeats most often, that the classic consumer reproduces the published ordering, is falsified by its own Table 3; the master table that anchors the survey argument grades R1 and R3 by different rules than the text states; and the channel comparison (RQ4) draws a conclusion about concepts versus items from an answer model that hands concepts many ratings per question and items one. Those three need to be fixed, not hedged. Alongside them, the manuscript needs a substantial cut for readability, every acronym and arm name defined once and used consistently, the missing hyperparameters and compute costs stated, and the code, checkpoint and score vectors actually available. If the authors do that, and add the practitioner-facing interview curve against production baselines (Q8), I would expect to recommend acceptance.
