# ECIR 2027 — Review (Reviewer 2)

**Paper:** Objective Mismatch in Adaptive Preference Elicitation for Cold-Start Recommendation

## Summary of claims

The paper argues that the cold-start preference-elicitation literature selects questions by criteria (rating variance, information gain, posterior variance, A-optimality, bandit indices) designed for RMSE-era objectives, while the field now reports truncated, position-discounted ranking metrics — and that nobody has checked whether the transfer holds.

Three claims follow. (RQ1) Under a single measurement contract on ML-25M — strong-generalisation split, a frozen two-tower "certified" recommender, answers read from users' own held-out ratings, full-catalogue ranking, unanswerable questions charged — thirteen re-implemented published selectors plus classic heuristics all lose, at every budget from one to sixteen questions, to a *fixed* question order built by greedily maximising held-out NDCG@10 on a validation cohort. (RQ2) Scored against exact measured NDCG@10 gains at the 22 nodes of a grown tree, three published split criteria correlate at +0.010, −0.060 and −0.104 with the metric and recover only 11–34% of what a uniform random pick forgoes. (RQ3) Putting the measured metric itself into Golbandi's ternary interview tree — one variable changed, everything else held fixed — is the only one of six criteria that lifts the adaptive interview above the static order (+0.01035 full / +0.00705 tail at K=16, paired bootstrap intervals excluding zero); and enlarging the question bank with Tag-Genome "concept" questions, with no conditioning at all, buys more (+0.01430 / +0.02199), because sixteen concept questions collect 12.4 answers where sixteen film questions collect 3.1.

The framing is explicitly "objective mismatch, not a verdict on adaptivity", since a static order is an adaptive policy ignoring its inputs.

## Strengths

1. **The question is the right one and the answer is non-obvious.** "Do elicitation selectors optimise the metric they are reported on?" is a question the community has assumed away, and the paper is right that the recent ML-25M benchmark [29] could not obtain working results for these methods. Section 3's contract — no simulator, no LLM in the measurement path, full catalogue, unanswerable questions charged, answers from the user's own held-out ratings with targets asserted absent — is stricter than most of what appears at ECIR, and the reasoning for each choice is given rather than asserted.
2. **Table 4 is a real ablation, not a decoration.** Same tree code, branch rule, minimum leaf size, pool, mask, cohorts and budget, with only the scalar changed, plus two shuffle controls including a min-leaf-respecting one. The finding that the separating axis is *measured vs. surrogate* rather than *ranking vs. rating* (pairwise rank loss and rating variance indistinguishable, +0.00020 [−0.00331, +0.00371]) is exactly what an ablation should be able to produce, and the authors flag that it contradicts their pre-registration. So is the root-level diagnosis that the *measured* pairwise loss tracks NDCG at Spearman +0.784 while its Taylor-approximated training version correlates at −0.149.
3. **Honesty about costs.** "Five qualifiers travel with +0.01035" — validation overstates it 2.7×, six of fifteen leaves lose on test covering 39.3% of the cohort, ~14% of users answer nothing, the pre-registered routing congruence test is a null so no routing claim is made, and the tree costs 37,191 s against 11 s. Most submissions would bury three of those five.
4. **The mechanism decomposition of the concept result** (a film answer worth 3.98× a concept answer; the concept channel collecting 3.98× as many; the matched-answer comparison flipping to +0.02734 for films) is the most instructive paragraph in the paper and turns a headline number into an explanation.
5. **The consumer study (Table 5)** is a good instinct: the *same banked interviews* handed to different consumers, with the one reversal (leaf model, −0.01375 at K=8) reported rather than hidden.

## Weaknesses, most severe first

1. **RQ2 evaluates the surrogates at states chosen by the winning criterion.** The 22 nodes are "all 22 nodes of the grown tree" — the tree grown by *CASPER's own* criterion. Each published criterion is then judged on how well it ranks candidates at states it would never itself have visited. That is a real selection effect sitting on the paper's cleanest-looking diagnostic, and the sentence "They agree with the metric where the cohort is largest and stop agreeing where an adaptive method must earn its keep" is doing interpretive work the design cannot support. *Fix:* repeat the regret table at the nodes of the variance-grown and pairwise-grown trees (Table 4 already produced those trees) and report the diagonal alongside the off-diagonal.
2. **The criterion is the metric computed through the recommender that then scores it.** The paper says so — "CASPER's criterion is the metric computed through the recommender that then scores it" — but under-weights the consequence: CASPER's split gain is measured *through the certified tower*, and it is on the certified tower that it wins, while on Golbandi's leaf model the comparison reverses at K=8. "Read Table 5 down its columns and that reversal is an argument for certifying the consumer" is a defensible reading; so is "part of the gain is a fit to one scorer". With one dataset, one split, one seed and one instrument the paper cannot separate them. *Fix:* grow one tree whose criterion is measured through a *different* consumer (RecVAE is already implemented) and check whether it too beats its own static order.
3. **"Thirteen published methods, every one loses" is stronger than the ports support.** Only five of thirteen have a native reproduction gate; EDDI fails its gate and is replaced by a reduced latent-only acquisition marked †; Greedy SLIM "passed only in part and is not run". The remaining eight — the heuristics and all the bandits — are ported with no evidence they reproduce their own papers. For an audit paper the ports *are* the load-bearing artefact, and the abstract's phrasing carries none of this. *Fix:* say "thirteen re-implemented, five natively validated" in the abstract, and add an appendix table of native-protocol numbers against published ones.
4. **The RQ1 fitting asymmetry is acknowledged but only partly bounded.** "The comparator is given something no ported arm is given: a search, on the reported metric, over a validation cohort." The bandit re-tuning (+0.0058 and +0.0018 against a 0.037–0.044 deficit) bounds it for methods with one free constant, and says nothing about IGCN, DRE, DPE or the trees, several of which have structural choices a metric-driven search could plausibly move much further. The authors' own escape hatch — "a reader who discounts RQ1 on fairness grounds should read that table as the load-bearing comparison" — is honest, but concedes that the audit is a diagnosis and the contribution is Table 4. *Fix:* give one or two non-bandit ports a comparable validation search, or demote RQ1 explicitly to a descriptive audit in the abstract.
5. **The concept-channel headline rests entirely on an unvalidated answerability model.** Item answerability is "having rated the item"; concept answerability is the signed exposure statistic with the ec<1.5 refusal rule. Both are modelling choices, and by the paper's own mechanism analysis the +0.01430 is almost entirely an answer-rate effect. If rated-implies-answerable understates item answerability — users rate films they could not discuss, and can answer about films they never rated — the comparison moves. Section 7 concedes the assumption is "load-bearing for two mechanisms above" but runs no sensitivity analysis. *Fix:* sweep the ec threshold, and for items try a popularity-based "would plausibly recognise it" relaxation, reporting how +0.01430 moves.
6. **"Sixteen conditioned questions" is not what Table 3 measures.** "Beyond K=5 the tree's depth is exhausted and the remaining questions are chosen inside the reached leaf by the same measured-NDCG search." The K=16 headline is therefore five conditioned questions plus a leaf-conditioned static tail, and the gain is flat from K=8 (+0.01062) to K=16 (+0.01035). The abstract's "+0.01035 full NDCG@10 at sixteen questions" invites a reading of sixteen adaptive decisions. *Fix:* state the 5+tail structure in the abstract, or move the headline to K=8.
7. **Statistical hygiene at this table size.** No family-wise correction across a battery the paper itself counts as 138 cells plus a 56-interval matrix, justified by "directional consistency across whole tables" and pre-specification of three cells. That is a defensible position, but with effects of +0.010 and intervals like [+0.00739, +0.01323] a reader will want the corrected numbers reported as well as argued away.

## Questions for the authors

- Does the CASPER tree still beat its static order when the split criterion is measured through RecVAE and scored on RecVAE, i.e. with the certified tower out of the loop entirely?
- What are the native-protocol reproduction numbers for the eight ungated ports? Even loose agreement would raise confidence in Table 1 substantially.
- Table 5: RecVAE's static arm reads .0000/.0000 at K=2. Is that arm meaningfully alive at any budget, and if not, why is +0.04122 quoted at all, even "for direction only"?
- How sensitive are the answer-rate mechanism and the +0.01430 to the ec<1.5 refusal threshold?
- Golbandi's tree has no K=16 row because "3^k answer patterns fragment its training cohort beyond depth eight". Since CASPER is itself depth-5-plus-leaf-tail, would a depth-limited Golbandi variant not be the fairer port?

## Minor points and presentation

- The prose is dense to the point of being hard work. "That tie is two effects cancelling: lift over the answer-free reference factorises into answers collected and lift per answer" is precise but arrives with no breathing room. One plain-language sentence per subsection would help a conference audience considerably.
- Table 1 is very cramped, and the five omitted arms should be listed with their K=16 numbers in an appendix rather than characterised as "degenerate or duplicate".
- Figure 1 is the only figure and carries a lot; a second panel for the tail axis, or for answers collected against budget, would earn its space more than some of the prose it would displace.
- Reference [3] is the anonymous companion instrument report. Section 3 summarises the four certification checks adequately, but the submission leans on that report for a load-bearing artefact; a short appendix restating the checks and the parity numbers would make the paper self-contained.
- "ans@16" is defined only in a caption; the § and † markers are defined in the caption but reused in the body pages later without reminder.
- The subsection heading "What the comparison is, and is not" is good practice — please keep it.

## Scores

| Criterion | Score | Justification |
|---|---|---|
| Relevance | 5 | Cold start, elicitation, evaluation methodology and reproducibility are squarely ECIR core topics. |
| Originality | 4 | The conjunction — measured held-out NDCG@10 *inside* the split criterion, tested against a static order maximising the same quantity — is genuinely unoccupied, and the Golbandi-2011 qualification shows the authors know where the novelty actually sits. |
| Technical soundness | 3 | Contract and Table 4 ablation are strong; the criterion/consumer circularity, the criterion-chosen states in RQ2, and the eight ungated ports keep this off a 4. |
| Quality of presentation | 3 | Well organised and unusually self-critical, but the prose density and cramped tables will cost readers. |
| Reproducibility | 4 | Split, seed, cohort sizes, bank construction, tuning grids, stopping rule, pool definition and bootstrap procedure are all specified, and with the complete artefact a competent group could rebuild this; short of 5 only because the instrument is described by reference to an anonymous companion report. |
| Potential impact | 4 | "Measure your surrogate against the metric you report" and the answer-rate mechanism should change how elicitation studies are designed, even though the absolute gains are small. |

## Recommendation

**Weak accept**, confidence **4**.

A careful, self-critical audit with one clean mechanism result at its centre. The weaknesses are real, but 1–3 are addressable by re-running analyses on artefacts the authors already hold, and none of them touches the best-supported claim: with everything else held fixed, only the measured criterion flips the sign of the conditioning gain.

## The one sentence I would send the authors

Table 4 is your paper — lead with it, qualify the thirteen-method audit in the abstract to match what the ports actually validate, and show me the criterion-regret table at the surrogates' *own* tree nodes.

## Would I be happy to see this at ECIR, and would I learn from the talk?

Yes to both. The talk has a clear, contrarian, one-sentence takeaway that most of the room does not currently believe, one honest ablation that supports it, and a genuinely surprising secondary finding — that changing the question bank beats changing the selection rule, for an answer-rate reason nobody in this literature measures. I would expect a lively Q&A about the fitting asymmetry, which is a point in the paper's favour rather than against it.
