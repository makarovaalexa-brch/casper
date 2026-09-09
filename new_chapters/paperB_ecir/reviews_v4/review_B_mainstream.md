# ECIR 2027 — Review (Reviewer 2, full paper track)

**Paper:** *Objective Mismatch in Adaptive Preference Elicitation for Cold-Start Recommendation*

## Summary

The paper argues that cold-start elicitation methods select questions by optimising surrogates (rating variance, information gain, posterior variance, A-optimality, bandit indices) that no longer match the metric the field reports, truncated top-k NDCG. Thirteen published selectors (twenty-three arms) plus classic heuristics are re-implemented under one "measurement contract": ML-25M on the Liang strong-generalisation split, one frozen two-tower recommender, answers read from users' own held-out ratings, full-catalogue ranking without sampling, unanswerable questions charged. Against a static comparator built by greedy prefix-optimal maximisation of held-out NDCG@10, every published selector loses at every budget from one to sixteen questions on 10,000 test users (RQ1). Three published split criteria scored against exact measured NDCG@10 gains at the 22 nodes of a grown tree correlate near zero, two negatively, recovering only 11–34% of what a random pick forgoes (RQ2). Substituting the measured metric into Golbandi's interview tree ("CASPER") reverses the comparison by +0.01035 full / +0.00705 tail at sixteen questions, while adding Tag-Genome concept questions with no conditioning buys more (+0.01430 / +0.02199) (RQ3). The finding is framed carefully as an objective mismatch, not a verdict on adaptivity. Replication scale and candour are the paper's best features; the headline nonetheless rests on a comparator no ported method could match, and reproducibility is unverifiable as submitted.

## Scores

| Criterion | Score | Justification |
|---|---|---|
| Relevance | 5 | Cold-start elicitation and evaluation are squarely in ECIR's remit. |
| Originality | 3 | Scale and the measured-criterion tree are new, but the core observation is credited to Sepliarskaia et al. 2018 by the authors themselves (Sec. 2). |
| Technical soundness | 3 | Careful protocol, but the headline comparison is asymmetric by construction and Table 5's RecVAE block looks wrong. |
| Quality of presentation | 3 | Precise but very dense; Table 1 and Fig. 1 hard to read, several terms undefined. |
| Reproducibility | 2 | Repository withheld; instrument is an anonymous unpublished report [3]; one dataset, one split, one seed. |
| Potential impact | 4 | If the measurement holds it should change how elicitation selectors are designed. |

**Overall recommendation: Borderline. Confidence: 4.**

## Strengths

- Exceptional breadth: thirteen methods across four families (interview trees, learned static, information-theoretic, conversational bandits), twenty-three arms, one contract, 10,000 held-out users.
- The contract (Sec. 3) is unusually precise: no sampled metrics (correctly citing Krichene & Rendle), no simulator or LLM in the measurement path, unanswerable questions charged rather than free.
- Table 2 is the paper's most original artefact: published criteria scored against *exact measured* gains at states a real tree visits, with a random control. It converts an assertion into a measurement.
- Table 4 is the right one-variable ablation with two well-chosen shuffle controls; that the *approximation* rather than the loss shape does the damage (Spearman +0.784 measured vs −0.149 Taylor-approximated, p. 10) is informative.
- The paper argues against its own headline throughout, and gate failures are reported rather than hidden.

## Weaknesses (most serious first)

- **The RQ1 comparator is not commensurable with the arms it beats, making the headline near-tautological.** The static order is built by lazy-greedy maximisation of held-out NDCG@10 *on the same recommender it is scored with* (Sec. 3, p. 5): it consumes a 10,000-user cohort's held-out targets at search time. No ported selector sees that signal, as Sec. 7 concedes ("the comparator is searched on the reported metric on the scoring recommender, which no ported arm can be"). "Every published method loses" is then largely a statement about what the comparator was given. The real contributions are RQ2/RQ3; RQ1 is oversold in title, abstract and contribution list.
- **Table 5 contains numbers that look broken, and a headline claim is computed from them.** RecVAE's static arm reads .0000/.0000 at K=2 and .0662/.0130 at K=4, below the answer-free popularity reference of .1626/.0262 in the same caption. A consumer scoring zero NDCG@10 after two answers is not a working fold-in, yet the +0.04122 RecVAE gain (p. 11) comes from this block and supports "CASPER beats the static order on three of the four".
- **Reproducibility cannot be assessed.** The repository is withheld (footnote 1, p. 2) with a request that reviewers disregard its absence — which I cannot do when reproducibility is scored and the paper's value *is* a measurement. The instrument every number depends on is an anonymous unpublished report [3]; its certification is asserted and its parity bar is "in-house". One dataset, one split, one seed (98765).
- **The most relevant baseline is incomplete at the headline budget:** Golbandi's tree, the machinery CASPER modifies, has no K=16 row in Table 1, and the leaf-model consumer in Table 5 stops at K=8. Relatedly, EDDI failed its native gate and runs only as "a reduced latent-only acquisition" (†, p. 5), so "thirteen published methods" overstates the battery.
- **The item-vs-concept comparison is not like-for-like.** Concept answers are a derived log-ratio statistic over the user's watching, not elicited responses; that such a statistic is "answerable" 12.36/16 times where ratings exist 3.11/16 is close to definitional, and the +0.01430 headline leans on it.
- **No multiple-comparison correction** over 138 cells (p. 5) or the 56-interval matrix (Table 4); pre-registration is claimed for three cells but no artefact is offered.
- **Density.** Table 1 packs seventeen arms, dual metrics, four typographic markers and a nine-line caption; Fig. 1 plots eight series with no described marker scheme. Both risk illegibility at print size.

## Detailed comments by section

**Abstract/Introduction.** A wall of numbers that will not orient a non-specialist; "measurement contract", "frontier recommender", "charged" and "answer consumer" are used before definition. Lead with the mismatch idea in plain language. The RQs are crisply stated.

**Related Work.** Strong and honest, but the Sepliarskaia paragraph and the Golbandi-2010 qualification sharpen the novelty question: if 2018 already showed a static questionnaire beating a mis-objectived dynamic one, RQ1 is a replication in a new metric regime, and the contributions should say so, not only Sec. 2. The "unoccupied conjunction" paragraph (p. 3) is the clearest novelty statement in the paper and belongs in the introduction.

**Measurement Contract.** Mostly exemplary. Gaps: is the concept statistic computed over the fold-in half or the whole profile ("over their profile" is ambiguous, and target-half leakage would contaminate the metric)? The e_c < 1.5 exposure threshold is unmotivated, and GLM-UCB's retuned constant (4.0 against published) deserves a sensitivity curve.

**RQ1.** Answered on its own terms, subject to the comparator objection. The auxiliary findings are more interesting than the headline and underplayed: twelve of twenty-three arms turn out to be value-free; Thompson runs 10,000 distinct interviews and still loses; but GLM-UCB collects 5.24 answers against the static order's 5.52 and also loses, so the answerability mechanism is only partial. The leaf-model reversal is flagged honestly but never explained.

**RQ2.** The best evidence in the paper, but the 22 states are self-selected: they are the states the authors' own criterion chose, so competing criteria are scored off-policy at nodes they would never reach. Report the same regret on a variance-grown tree. The "how much is there to adapt to" paragraph is the least clear passage in the paper — U(q | z), V and "perplexity of the argmax distribution" appear without derivation, "one standard opener" is unspecified, and the "real free-text request" producing 442 optimal questions has no stated source or encoding and may sit outside the contract. Since it implicitly argues adaptivity could pay with a better opener, it needs proper treatment or removal.

**RQ3.** The single-variable design is right and the exact K=1 gate is a nice touch. The tail interval at K=2 excludes zero *against* the tree, so the reversal holds from K=4 — the body says so, the abstract does not. The cost gap (37,191 s vs 11 s) deserves more than a clause, and the criterion needs a labelled cohort with held-out targets: a deployment constraint, not just a compute one. Sec. 7 is candid (rebuilding the comparator cut the gain from +0.01371 to +0.01035), but naming the asymmetry does not repair it.

## Questions for the authors

1. Can you build a comparator that does not see held-out targets at search time (e.g. greedy on training-user estimates) and report RQ1 against it? Does any published selector then win?
2. Why does RecVAE's static arm read .0000/.0000 at K=2, below the answer-free reference? Does +0.04122 survive a fix?
3. Why is there no K=16 row for the Golbandi tree in Table 1?
4. Is n_c computed over the fold-in half only?
5. Can Table 2 be repeated on-policy, at states grown by each criterion's own tree?
6. What are the "standard opener" and the "real free-text request" in Sec. 5, and are they inside the contract?
7. Does the ordering hold on a second dataset and a second seed?
8. Do any conclusions change under family-wise correction over the 138-cell and 56-interval batteries?
9. How would a practitioner compute the CASPER criterion without a labelled cohort and ~10 CPU-hours per tree?

## Minor issues

- Fig. 1: no described colour/marker scheme; confirm greyscale legibility. Table 1 would be clearer split into static and adaptive arms.
- Define "frontier recommender", "answer consumer", "charged" and "gate" at first use.
- The abstract quotes +0.01035 without noting the reversal holds only from K=4; contribution 1 says "thirteen published methods" while one is reduced and one is not run.
- "Code and the per-user score vectors behind every table are available" (p. 2) is contradicted by footnote 1 on the same page.

## What would change my recommendation

I would move to **weak accept** if the authors (a) explain or correct the Table 5 RecVAE block, since a headline consumer-generalisation number rests on scores below the answer-free floor; (b) reframe RQ1 as a measurement of what a metric-optimised order can reach, given their own admission that no ported arm can be searched the same way, shifting the paper's weight onto RQ2 and RQ3 where the evidence is strong and new; and (c) commit to an anonymised artefact at review time, or give enough detail on instrument [3] to rebuild the contract from the paper alone. I would move to **accept** if the ordering also held across seeds or on a second dataset and the Golbandi tree were completed to K=16. I would move to **weak reject** if the RecVAE anomaly proves to be a fold-in bug that also touches the certified tower, or if concept answers draw on the target half of the profile — either would put the headline gains themselves in doubt, not merely their framing.
