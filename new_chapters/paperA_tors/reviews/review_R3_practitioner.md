# TORS Review — Reviewer 3 (industry practitioner-researcher)

**Manuscript:** "Elicitation Results Are Instrument-Relative: A Certified Interview-Compatible Recommender for Measuring Cold-Start Preference Elicitation" (33 pp., review mode with line numbers)

---

## 1. Summary

The paper argues that results in the cold-start question-selection literature are not portable because each strategy is evaluated through whatever recommender consumes the answers. The first experiment (Sec. 8.1, Table 3) fixes users, questions, answers, budget (k=8) and candidate pool on ML-25M and varies only the consuming recommender across six models; Rashid et al.'s classic ordering (helf > entropy0 > popularity) reproduces under a Golbandi-style node model and comes out differently under the authors' model and under a likes-only RecVAE. From this the paper derives four requirements (R1 frontier-parity full-profile accuracy; R2 open, set-valued input; R3 monotonicity; R4 signed and graded answers) that a "measuring instrument" recommender must satisfy, surveys the literature into a six-family taxonomy plus a master table (Table 1) measured on one ML-25M ruler under two protocols, and claims no published system satisfies all four. It then presents such a system: a RecVAE-class tower trained and frozen, with a permutation-invariant set encoder over its latent into which items, Tag-Genome concepts (via whitened centroids and a signed log watch-lift affinity), free-text phrases (via a linear adapter from a sentence encoder) and arbitrary direction tokens fold as (direction, value) pairs. The instrument scores 0.3496/0.2491 full/tail NDCG@10 against RecVAE's 0.3540/0.2497, is certified by a bridge that reproduces published ML-20M numbers, and passes a verification battery (Table 4). Finally it uses the instrument to measure answerability (25.2% of the top-100 films rated vs 53.0% of 1,031 concepts answerable) and to compare item and concept channels under four bank rankings, finding that under a pure-discrimination ranking concepts win at every budget because divisive concepts stay answerable (97.2%) while divisive films do not (4.8%).

## 2. Relevance to TORS

High. Evaluation methodology for conversational/cold-start recommenders, reproducibility of classic results, and a released, certified evaluation substrate are squarely within TORS's scope, and the "report the consumer you measured through" message is the kind of practice-changing methodological point the journal publishes. The paper's length and density are more of a problem for TORS's readership than its topic.

## 3. Strengths

1. **The instrument-relativity experiment (Table 3) is genuinely useful and cheap for others to replicate.** Fixing everything except the consumer and showing three orderings, with paired bootstraps, is the kind of result that changes how I would read any elicitation paper. The mechanism (helf yields 1.59 graded answers of 8 vs ~3.1 for the other two; a tree is paid by "unknown" while a fold is not) is concrete and believable.
2. **Certification discipline.** Reproducing EASE (0.4203 vs 0.420) and RecVAE (0.4425 vs 0.442) on ML-20M before touching ML-25M, dropping Mult-DAE when it fails to snap, and demoting the authors' own iALS as an "uncertified floor" (Sec. 7.2) is exactly the practice Ferrari Dacrema et al. asked for and few papers deliver.
3. **Answer-provenance rule enforced in code (Sec. 7.4).** Stating that no recommender-derived quantity may enter an answer, and naming the simulators in the literature that violate it, is a real methodological contribution and should be lifted out as a checklist.
4. **Honest reporting of negatives and confounds.** The RBMF "-0.070" being a configuration artefact (Sec. 3.2), the tail tie with the Golbandi node model, the admission that tower-vs-RecVAE at k=8 is "not a controlled attribution" (l.1186), and the two-protocol design with its stated reasons all raise trust.
5. **Full and tail NDCG everywhere, paired per-user CIs everywhere, single ruler.** This is the right reporting standard.
6. **The answerability measurement (25.2% vs 53.0%) and the polar-bank collapse (4.8% answered)** are simple, memorable numbers that a practitioner can act on when designing an onboarding flow.

## 4. Weaknesses (most serious first)

### W1. The instrument's training is never described, and the paper contradicts itself about whether the tower is frozen. (Major)
Sec. 5.1 (p.15, l.706-718): "trained once on the full catalogue and then frozen ... The tower is the graded-native set-encoder T2', trained once in its pre-registered configuration." Then p.25, l.1186-1188: "our tower is initialised from that model [RecVAE] but subsequently trained on interviews, so the difference confounds the input contract with that training." So the tower *was* trained on interviews. Nowhere does the paper say: what the set encoder is (architecture, width, depth, pooling), what loss it is trained with, how interview states are sampled during training (which lengths, which channels, graded vs likes), how many epochs, which validation criterion selected the checkpoint, what "pre-registered" refers to, what "T2'" is, what the "certified pair" of footnote 2 (p.19) is, or what "the retrained model" and "the pre-registered triple gate" of p.28 l.1336 are. The gated concept operator Eq. (4) gives symbols but not sizes, the norm cap, its training data or objective. For a paper whose deliverable *is* this artifact, this is disqualifying as it stands: a reader cannot reproduce it, cannot judge whether the 1.2% shortfall is seed noise (one run, l.1221), and cannot tell how much of the k=8 advantage over RecVAE comes from training on interviews (which the authors concede, l.1186) rather than from the interface. Add a "Training the instrument" section with a table of hyperparameters and compute, and reconcile Sec. 5.1 with l.1186.

### W2. Several numbers do not agree with each other across the paper. (Major)
- **Table 3 "Ours - this" tail column is not what the caption says it is.** Caption (l.1150): "compares each system at its own best strategy against ours at ours". Tail, RecVAE likes-only: ours .0792 - its best (ent0 .0576) = +.0216, but the table prints +.0258 = .0792 - .0534 (its *pop* cell). Conjugate belief tail: .0792 - .0338 (helf, its best) = +.0454, table prints +.0492 = .0792 - .0300. The full-profile column checks out for every row. Either the best strategy is chosen on full and reused on tail (then say so) or the cells are wrong.
- **Cold intercept:** 0.1279/0.0192 (Sec. 7.1, p.21 after l.1003) vs 0.1277/0.0267 (p.27 l.1297; p.28 l.1370). The tail value differs by 40%.
- **Graded-vs-binarised premium at k=2:** +0.0144 (Sec. 2.2, l.263) vs +0.0102 (Fig. 3, Table 4, Sec. 3.2 l.390).
- **Full-profile value of the levels:** +0.0022 (CI [+0.0008,+0.0036]) at l.388 vs +0.0039 (CI [+0.0025,+0.0053]) at l.261, l.1277 and Table 4. If these are different comparisons, name them; the sentence at l.386-388 explicitly contrasts "+0.0194 against +0.0022" as if it were the same quantity as the +0.0039 "on identical support".
- **iALS confidence-weighting gain:** +0.0194 (l.358) vs +0.0170 (Table 1 footnote: .3495 - .3325).
- **"NDCG@100 orders the liang column identically" (l.1052):** it does not. At @10, RBMF (.2915) > iALS (.2703) > TaNP > Turbo-CF; at @100, iALS (0.3700) > TaNP (0.3661) > Turbo-CF (0.3569) > RBMF (0.3533).
- **Three different "sign flip" numbers** are quoted without a map: +0.2096 (R2 row of Table 4 and l.619, called "our own sign contrast"), -0.2012 (concept channel, l.965), -0.2953 (R4 row of Table 4). A reader has to work out that these are three different arms. Also the R4 flip reports n=7,563 users with no explanation of why not 10,000.
- **Abstract/intro "every gap that decides an ordering is separated"** (l.12, l.64) vs Table 3 caption "except where marked ~ or flat". RBMF's full ordering is E0 ~ P and the conjugate belief's is flat; the abstract overstates.

### W3. RQ4 -- the headline "concepts beat items at every budget" and the answerability numbers -- has no table and no figure. (Major)
Sec. 8.3 (pp.28-29) is three pages of prose carrying at least 20 numbers (per-budget gaps with CIs under four rankings, intercepts, 25.2%/13.1%/53.0%, 97.2%/4.8%, the counterfeit and permutation controls). The abstract's last sentence and the conclusion rest on these, and the only way to check them is to re-read the paragraphs. Add (a) one table: rows = budget k in {1,2,4,8,16}, columns = item/concept x full/tail under each of the four rankings with the paired gap and CI; (b) one figure of the two curves under polaR and pop against the intercept. Similarly the R3 curves (Table 4 quotes only q=2 and q=16 endpoints) and the "random from answerable bank beats popularity" claim (l.1307) are stated without a figure.

### W4. The answer model is a weak stand-in for real interview answers, and the paper's framing ("truthful, behaviour-grounded", l.88) undersells the gap. (Major for the RQ4 claims, minor for RQ1)
- An item answer is the user's own historical rating *from the fold-in half* -- i.e., from behaviour accumulated over a lifetime on the service, not what a cold user recalls at onboarding. A concept answer is a PMI-style watch-lift over the same history, bucketed into four bands by thresholds the paper never states numerically (tau for refusal, the band edges; only kappa=3 is given). Nobody asked "do you like film noir?" computes a log-lift; the four-band answer is a *label derived from the outcome distribution*, not a response.
- The answerability numbers (25.2% vs 53.0%) are therefore not properties of users but of two authorial rules: "rated" for items and "as few as two rated member films" for concepts (l.1388) -- which itself disagrees with Sec. 6.1's rule (refusal when expected support falls below tau). Under a two-member rule a concept with 500 members is almost automatically answerable for any user with a moderate history. The paper half-concedes this ("What still depends on the setup is any per-budget count", l.1391) but the abstract and conclusion print the numbers bare.
- The 97.2% answerability of the polaR concept bank is partly by construction: polaR = H(bands | answered) already conditions on answerability (l.1358-1359).
- Sec. 9 says (l.1433) "not a person responding to a rendered question" -- good -- but this sentence should be in the abstract or at least in the RQ4 answer, and the paper should state what it *would* take to validate: a small user study on a few hundred users comparing self-reported concept affinity to the SEL band, or at minimum a sensitivity sweep of tau and the band edges with the RQ4 ordering re-reported.
- The "oracle concept affinity" in the informativeness decomposition (l.883-887) is undefined. If it is derived from any model, that decomposition is exactly the circularity Sec. 7.4 forbids; if it is from the held-out half, say so. The "LLM-elicited ordinal answer" is never described (model, prompt, users), and 0.017 R^2 is quoted as if it disposed of LLM answers.

### W5. Readability: the paper is ~33 pages of dense, inverted, qualification-laden prose, and a practitioner will not finish it. (Major for presentation)
The abstract is one 22-line paragraph; the sentence at l.10-12 is 60+ words. Sentences that need two passes, examples: l.201-202 "A system may satisfy R1 while being separably behind the bar, and ours is; the requirement is that the shortfall be small, stated and bounded, not that it be statistically absent." / l.237 "Two halves, and only the second can fail." / l.323-328 (why native numbers are higher) / l.391 and l.438 "The native column's value is that it forecloses an objection rather than that it changes an ordering" -- split by a full-page table / l.979-982 "That choice is what makes R1 free rather than re-earned" / l.1282 "but it is not free there and worthless here" (reads backwards; you mean "it is worth more there, but not worthless here") / l.1336-1340 "Re-running the pre-registered triple gate on the retrained model clears all three taste criteria ... the clip-up curve ... in this gate's arm" -- this reads like an internal lab note pasted in, with four undefined terms. Prior art is credited three separate times in nearly the same words (l.174-181, l.670-677, l.924-960). Sections 3, 4 and 6.3 overlap heavily. I would target 22-24 pages: merge 3.3/4/4.1, move the TaNP rating-objective anecdote (l.330-349) and the RBMF re-tuning forensics (l.369-383) to an appendix, and cut the third prior-art paragraph.

### W6. Cost, engineering and what is released are not stated. (Moderate)
No compute (hardware, wall-clock) for training the tower, the set encoder, the concept operator, the adapter, or the bridge; no parameter counts; no inference latency of the fold (a practitioner wants to know it is one matrix multiply per answer -- say so). "Code, the certified checkpoint and the per-user score vectors ... are available" (l.181) is all we get; state what the repository contains (question banks, the SEL/VAL answer tables, the bootstrap harness, the ML-20M bridge scripts, the battery as a runnable test) and under what license. The sentence encoder used for the text adapter is not named (l.756). "Pre-registered" is asserted (l.716) with nothing to point at.

### W7. The "open vocabulary / free text" leg is illustrated, not evaluated, yet the abstract and contributions say free-text phrases "fold". (Moderate)
Evidence is four hand-picked phrases (Table 2), with the screen rejecting three others, an 80/20 held-out-concept overlap of 0.167 top-10 (l.759), and the admission that phrases the genome does not carry "return lists we would not defend" (l.773). No interview curve is run over text-adapter directions. Either run one (fold the adapter image of the held-out 20% concepts' *names* instead of their centroids through the R3 harness and report the loss) or soften "fold" to "are admissible" in the abstract and contributions.

### W8. Deployment realism of the concept channel is Tag-Genome dependent. (Moderate)
Whitened member centroids need curated membership lists with hundreds of members per concept; the Genome exists for MovieLens and little else. The paper should say plainly that in production the concept channel is (a) a curated taxonomy or (b) the text adapter with its stated limits, and give one paragraph on what a practitioner without a genome would do.

### W9. Single training run, no seed variance. (Minor-moderate)
The R1 shortfall (-0.0043 full) and the entire battery are from "one certified training run" (l.1221). The paired bootstrap covers user sampling, not training variance. Three seeds would settle whether 1.2% is a property of the design.

### W10. The claim "the disagreement is not attributable to weak recommenders" (l.1180) leans on a control the authors call "a control, not a rival" (Table 3 note). (Minor)
Likes-only RecVAE at helf gets 1.6 answers of 8 and no popularity floor; its ordering is dominated by starvation. A frontier consumer given a fair interface (e.g. RecVAE with the likes *and* a popularity floor, or EASE fed the same likes) would be a stronger control for the inversion claim. The caption's own admission (l.1153-1155) that "only its helf cell is materially affected" helps, but the fully-floored ordering should be shown, not asserted.

### What a practitioner should do differently, and what the paper should say about it
The actionable message is sound but scattered: (1) report the consumer any elicitation result was measured through, on a full-catalogue metric; (2) state the answer path and check it reads nothing from the model; (3) rank an item bank by answerability x discrimination, not discrimination alone, because a divisive film is one nobody has seen; (4) prefer broad-property questions at the first two turns. These four lines belong in the conclusion as a numbered list. The paper's implicit ask -- "study your policy on our instrument" -- is not realistic for a team with its own production model; the portable deliverable is the battery plus the provenance rule, and the paper should say so and ship the battery as something one can point at another recommender.

## 5. Detailed line-by-line comments

- **Title/abstract.** The abstract has no paragraph break for 22 lines; split into four sentences: problem, RQ1 result, instrument, RQ4 result. "reproduced to the published third decimal" -- Recall@50 0.5525 vs 0.553 rounds either way; say "to within 0.001".
- **l.56, l.1120:** helf, entRopy0, populaRity are never expanded. Define: helf = harmonic mean of entropy and log-frequency (Rashid et al.); entropy0 = entropy with the "not seen" bucket. The small-caps rendering also breaks mid-word in the PDF ("helf", "entRopy0", "populaRity") -- check the font.
- **Undefined acronyms at first use:** EASE, RecVAE, EDLAE, iALS/WMF, PureSVD, RBMF (expand: representative-based MF), TaNP, MeLU (Table 1, never mentioned in text), EDDI, ICF, ConTS, BCIE, UNICORN, PEBOL, GATE, GF-CF, BSPM, SASRec, BERT4Rec, DLC, LLC, CE-VAE, BK-VAE, M&Ms-VAE, FiLM, EVOI, CAV, PMI/NPMI, ExpoMF, SLIM, CRS, MRR, ALS. TORS readers will know maybe half. A glossary, or an expansion at first use.
- **l.101:** "within 1.5% relative ... fixed in advance" -- where? Point to the pre-registration or drop "fixed in advance".
- **pp.6, 15 rendering:** in my copy the inline math on p.6 (the concept-answer formula around l.273-275) and two passages on p.15 (l.706-709 and l.731-733) are superimposed on the surrounding text and unreadable. Check the compiled PDF.
- **l.204:** "(t=0 yields the learned no-evidence latent)" vs l.1283 "emits a zero latent (|mu_0|=0)". Learned or zero? Both may be true (learned to be zero) -- say it once.
- **l.263:** +0.0144 vs +0.0102 elsewhere (see W2).
- **l.330-349 (TaNP):** two-thirds of a page on a system whose row is a reduction you built. Appendix.
- **l.356:** "0.3756 on binary co-occurrence against 0.2822 on mean-centred ratings" -- 0.2822 does not appear in Table 1 (user-kNN liang .3065, native .3756); label which protocol.
- **Table 1 (p.9).** The caption is 17 lines with six footnote marks. Legible but at the limit. Put "in" (b/g) in a legend row; move the iALS/RBMF/user-kNN footnotes into a short "Notes" paragraph in Sec. 3.3. The R3 column gives a check to user-kNN, RBMF, conjugate belief, Biyik, ICF, EDDI, PEBOL, GATE, LLM-CRS -- monotonicity as a *designed capability* of user-kNN is a stretch; state the rule (a belief/posterior exists => check?) in one line.
- **Table 1:** "TaNP [Lin et al. 2021] (MeLU line)" -- MeLU never cited.
- **Fig. 1 (p.12).** Legible. Hollow markers at x=1 overlap ("Biyik, ICF, BCIE, PEBOL" is one label for four dots at two x positions); the second hollow group sits at x~1.2 on an axis the caption calls an ordinal count. Jitter and label each.
- **Fig. 2 (p.15).** Clear. Consider showing the two distinct folds (tower set encoder vs gated concept operator, l.725-727) since the figure implies one f.
- **Fig. 3 (p.17).** A default-matplotlib plot with a stray title "input encodings", tiny fonts, two lines that overlap for 3 of 5 points, no CI bands, an x tick labelled "full" on a numeric axis, and no tail panel. The caption carries the numbers the plot cannot show. Redraw with CI ribbons, both metrics, log-x.
- **Table 2 (p.17):** fine; "cos" needs "cosine to nearest concept" in the header, and name the sentence encoder.
- **Eq. (2)/(3):** define n-hat, the band thresholds and tau numerically; state how VAL's "each item's own mean" is computed (train users only?). "add-1/2 log watch-lift" -- write "add-one-half".
- **l.883-887:** define "oracle concept affinity"; describe the LLM answer or delete the comparison.
- **l.913-917:** "norm-capped per user" -- value? The numbers 0.1759/0.0841 etc. come from an earlier tower (footnote 2); label them "(earlier tower)" inline so they are not mistaken for the certified ones.
- **Footnote 2 (p.19):** "the certified pair" -- undefined.
- **Sec. 7.1 (p.21):** intercept 0.1279/0.0192 -- reconcile with 0.1277/0.0267.
- **l.1026-1027:** iALS is declared an uncertified floor here, but Sec. 3.2's iALS-with-confidence result (+0.0194) is used as a substantive finding about ratings. Flag it as uncertified there too.
- **l.1043-1046:** "EDDI early-stopped at epoch 46 with epoch 31 validation-selected" -- meaningless without the training description.
- **Sec. 7.6:** state here (not at l.1196) that tail NDCG is computed over 9,281 of the 10,000 users and why.
- **Table 3 (p.24).** Header "Ours - this" -> "Ours (best) - row (best)". "ent0" in header vs "entRopy0" in text. Bold on RBMF's ent0 cell (.1644 vs .1643) marks a 0.0001 difference as "best"; show ~ instead. Tail cells for RecVAE and conjugate belief: see W2.
- **l.1195:** "(a coincidence of rounding ...)" -- a coincidence is not an explanation; just say the two gaps happen to be equal in magnitude.
- **Table 4 (p.26).** Self-contained and legible; the best table in the paper. But the "Measured" cell for R2 packs five experiments and R4 packs six. Split into sub-rows (R4a sign, R4b intensity, R4c graded-vs-binary) with one number each. "n=7,563" unexplained.
- **l.1290-1297:** "a like lifts the concept's members to AUC 0.626, a dislike pushes them to 0.416, and folding a different answerable concept leaves them at chance (0.498)" -- this is the best evidence for R2's open half and it is buried; put it in Table 4's R2 row and drop the "identity contrast +0.1278", whose definition I could not find.
- **l.1336-1344:** define "triple gate", "taste criteria", "clip-up", "this gate's arm", "counterfeit control" before use, or rewrite in the paper's own vocabulary (signed vs positive-only concept answers).
- **l.1358-1359:** the four rankings are defined by inline formulas with undefined symbols (H, P(answered), bands); define them in a display or in Sec. 7.6.
- **l.1385-1386:** "a held-out user has rated 25.2% of them" -- in the fold-in half only, or the whole profile? Does "rated" include sub-threshold ratings, given the liang filter keeps likes only?
- **l.1416-1417:** "55.7% of a test user's rated history, their dislikes, is absent from every model's input" -- but Fig. 3 folds sub-threshold answers in the interview harness. Say explicitly: full-profile rows use liang-filtered likes; interview curves use the full graded history from the fold-in half.
- **References:** Biyik et al. 2023, Gharahighehi et al. 2025 and Antognini & Faltings 2022 are arXiv-only; Xia et al. 2026 (arXiv 2607.06765) -- check. Qin et al. 2020 carries stray text "WSDM/KDD line; Google unbiased-LTR". Liang 2018 venue naming inconsistent with Lin 2021 ("WWW" vs "The Web Conference").
- **Typos/glyphs:** l.150 and l.1455 "R1R2R3R4" (missing conjunction glyph); l.536 same; l.29 CCS bullet renders as a missing glyph; "an-swers" hyphenated inside the Table 3 header cell.

## 6. Questions for the authors

1. What exactly is trained after RecVAE initialisation, on what objective, on which interview distribution, and for how long? Is the *decoder* frozen from RecVAE or retrained? If the decoder is RecVAE's, isn't the empty-interview identity (Spearman 1.0) trivially the RecVAE bias?
2. How much of the +0.0293 (k=8, vs likes-only RecVAE) survives if RecVAE is given the same likes *plus* a popularity floor, or if EASE is used as the consumer with the same likes?
3. What are tau (refusal threshold) and the four-band edges for SEL/VAL, and how do the 53.0% concept answerability and the polaR ordering move as tau varies?
4. What is the "oracle concept affinity" in Sec. 6.1's informativeness decomposition, and does it read the held-out half or any model?
5. Why is R4's sign flip evaluated on n=7,563 users and tail NDCG on 9,281?
6. Table 3 tail column: is the comparator strategy chosen per metric or on full only?
7. Which sentence encoder is the text adapter fitted from, and what happens to the R3 curve when concept directions come from names rather than centroids?
8. Seed variance: what is the spread of full/tail NDCG@10 across three or more training seeds of the instrument?
9. What does the released artifact contain, and does the battery ship as a runnable test a third party can point another recommender at?
10. For a practitioner without a Tag Genome: which channel do you recommend, and what does the instrument lose?

## 7. Scores (1-5)

| Criterion | Score |
|---|---|
| Relevance | 5 |
| Originality | 3 |
| Technical soundness | 3 |
| Presentation | 2 |
| Reproducibility | 2 |
| Impact | 3 |
| **Confidence** | 4 |

Soundness 3 rather than 4 because of the internal number inconsistencies (W2) and the undescribed training (W1); the core RQ1 experiment itself looks sound. Reproducibility 2 because the central artifact's training recipe is absent and the release is a footnote; this rises to 4 if W1 and W6 are addressed and the repository is as described. Originality 3: the components are (as the authors say three times) borrowed; the conjunction, the battery and the instrument-relativity experiment are the new pieces.

## 8. Recommendation

**Major revision.**

The paper has a practice-changing result (Table 3: strategy orderings are a property of the strategy-recommender pair) and an unusually disciplined certification story, and I want to see it in TORS. But in its current form I could not reproduce the instrument (its training is not described and Sec. 5.1 contradicts l.1186 on whether it is trained on interviews), several headline numbers disagree with each other or with their own table caption (Table 3 tail column, the cold intercept, the k=2 premium, the full-profile value of levels), the RQ4 results the abstract ends on exist only as prose, and the manuscript is roughly a third longer than its content needs and written in a way that will lose most practitioners before Section 5. None of this requires new experiments beyond a seed sweep and, optionally, a tau sensitivity and a fairer frontier-consumer control; all of it requires a careful rewrite. I would be glad to review a revision that (i) adds a training section and a hyperparameter/compute table, (ii) reconciles every number listed under W2, (iii) adds a table and a figure for Sec. 8.3, (iv) states the answer model's limits in the abstract and the RQ4 answer, and (v) cuts the paper to ~24 pages with every acronym and arm defined at first use.
