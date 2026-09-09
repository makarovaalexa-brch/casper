# ECIR 2027 — Review (Reviewer 3, full paper track)

**Paper:** Objective Mismatch in Adaptive Preference Elicitation for Cold-Start Recommendation

## Summary

The paper re-implements thirteen published cold-start question-selection methods (twenty-three arms including classic heuristics) under one measurement contract on one frozen "certified" recommender, on a strong-generalisation split of ML-25M with 10,000 held-out test users. Answers come from users' own recorded ratings, the full catalogue is ranked without sampling, and unanswerable questions are charged. The comparator is not a heuristic but a static order built by lazy-greedy maximisation of the reported metric (full NDCG@10) on a disjoint validation cohort, and every published selector loses to it at every budget from 1 to 16. The diagnosis: at the 22 states an interview tree visits, three published split criteria correlate at +0.010, -0.060 and -0.104 with exact measured NDCG@10 gains and recover only 11-34% of what a random pick forgoes. Two fixes are then priced on the same contract: putting the measured metric into Golbandi's tree split (CASPER) reverses the comparison by +0.01035 at K=16, and merely enlarging the bank with Tag-Genome concept questions, with no conditioning at all, is worth more (+0.01430). Qualifiers, nulls and costs are reported unusually honestly.

## Scores

| Criterion | Score | Justification |
|---|---|---|
| Relevance | 5 | Cold-start elicitation under an honest contract is squarely ECIR. |
| Originality | 3 | Scale and discipline are new; the mechanism is unsurprising, and Sepliarskaia et al. 2018 is conceded to show the RQ1 shape. |
| Technical soundness | 4 | Protocol, controls and ablations are strong; one consumer arm looks broken and two definitional asymmetries go unaddressed. |
| Quality of presentation | 3 | Well structured, but compressed to the point of needing re-reading; load-bearing terms undefined. |
| Reproducibility | 2 | Repository withheld; the instrument every number depends on is an anonymous tech report with no architecture given here. |
| Potential impact | 4 | "Change what you ask about, not how you decide" and "certify your consumer" are directly actionable. |

**Overall recommendation: Weak accept. Confidence: 4/5.**

## Strengths

- The measurement contract (Section 3) is the best part of the paper: no sampled metrics, no simulator in the measurement path, unanswerable questions charged, fitted on validation and scored once on a disjoint test cohort.
- Ports are *gated*: five of thirteen reproduce their published construction natively, and gate failures are reported rather than quietly scored (EDDI reduced and daggered; Greedy SLIM not run). Few benchmark papers are this disciplined about which re-implementations they trust.
- Table 4 is a real one-variable ablation with two shuffle controls, including the min-leaf-respecting one, and bandit constants are re-tuned on validation. That the separating axis is *measured vs surrogate*, not ranking vs rating, is the most useful single result here.
- Honest reporting: validation overstates the gain 2.7x, six of fifteen leaves lose on test, ~14% of users answer nothing, the routing claim is a registered null and is withdrawn, and the cost is stated, not hidden.

## Weaknesses (most serious first)

1. **Reproducibility is not assessable (footnote 1, p.2; Section 3, p.4).** The repository is withheld with a request that reviewers "disregard its absence"; I cannot. Worse, the frozen recommender every number depends on is "[3] Anonymous, technical report, under review" — two NDCG numbers and four bullet requirements, no architecture, objective or hyper-parameters. A practitioner cannot rebuild the instrument, and therefore cannot rebuild any result, including the comparator. For a paper whose contribution is "we measured this properly", this is central.
2. **The RecVAE static arm reads .0000/.0000 at K=2 (Table 5, p.12)** — below the answer-free reference .1626/.0262, i.e. worse than doing nothing. That looks like a fold-in bug, not a finding, yet +0.04122 on RecVAE is quoted in Section 6 as one of three consumer corroborations.
3. **Budget and tree depth do not add up (Section 6, p.9; Table 3).** The tree is depth 5, 15 leaves, 22 nodes, but users are charged sixteen questions, and questions 6-16 are never accounted for. The headline +0.01035 is a K=16 number, so most questions in the winning arm may not come from the mechanism the paper credits — and Table 1 gives Golbandi's tree no K=16 cell at all.
4. **The two channels use different answerability definitions (Section 3, p.4; Section 6, p.10).** An item is answerable iff the user *rated* it; a concept is answerable from an implicit-watch statistic over the whole profile. Of course concepts collect 12.36 answers to films' 3.11 — one channel reads the full watch history, the other only explicit ratings. The +0.01430 bank result is the paper's headline practical advice and rests on that asymmetry.
5. **One dataset, one instrument.** Everything is ML-25M with the Tag Genome; robustness varies the consumer but not the data or domain, and the paper never says what a practitioner without a curated tag vocabulary should do.

## Detailed comments by section

**Abstract and introduction.** The headline survives one pass, and framing the result as objective mismatch rather than a verdict on adaptivity (p.2) is the right disclaimer. But gains appear only as deltas: +0.01035 means little until Table 3 supplies 0.25900 -> 0.26936, about 4% relative. Put a base value in the abstract.

Sentences I read twice, verbatim:
- "three published split criteria recover only 11–34% of the gain a uniformly random choice leaves unrealised" (Abstract) — a triple negative.
- "on four consumers of the answers" (Abstract) — "consumer" is not introduced.
- "every sequence is selected and scored under the protocol pool, which removes every rated fold-in item (141.2 per user), under which the empty interview on the test cohort reads 0.1586/0.0282" (p.5) — chained "under which".
- "a model that received no answer is scored on what it predicts rather than overwritten by a popularity row" (p.4).
- "downstream of a closed cold opener there are roughly five materially different questions to choose between" (p.8) — "closed cold opener" never defined.
- "The comparator is searched on the reported metric on the scoring recommender, which no ported arm can be" (p.11).

**Related work and contract.** Related work is fair, and the "unoccupied conjunction" paragraph is the clearest novelty statement here — it belongs in the introduction. The contract is reproducible *in recipe* (split, seed, cohorts, metric, bootstrap, bank of 500 items and 500 concepts, lazy greedy), but how concepts are "answerability-ranked" is never said and the instrument is opaque: I could rebuild the comparator's procedure, not its numbers.

**RQ1 and RQ2.** RQ1 is the strongest section: "the loss is not a failure to branch" (Thompson runs 10,000 distinct interviews and loses; GLM-UCB matches the static answer rate and loses) closes off the obvious rebuttal, and that twelve of twenty-three arms are value-free is useful field hygiene. The shifting counts (thirteen methods / twenty-three arms / twenty-one rules / 138 cells) are hard to track. Table 2 convinces, but "Placing users by a real free-text request instead spreads the optimum to 442 questions" (p.8) arrives from nowhere: where does free text come from in a contract that bars language models from the measurement path?

**RQ3 and limitations.** The one-variable design is right, and the logistic arm walking downhill (-0.00896 to -0.08627 while opening on the *correct* first question) is memorable. The cost is under-analysed for a practitioner: how does 37,191 s scale with bank size and catalogue, and does a cheaper surrogate recover most of it? Limitations are candid: disclosing that rebuilding the comparator under the scoring contract cut the gain from +0.01371 to +0.01035 is what most papers omit.

## Questions for the authors

1. How are questions 6-16 chosen for a depth-5 tree, and what share of the K=16 gain comes from nodes the measured criterion actually chose?
2. Why does the RecVAE static arm score .0000/.0000 at K=2, and is +0.04122 robust to fixing it?
3. If item questions used the same implicit-watch answerability rule as concepts, how much of the +0.01430 survives?
4. Where do the "real free-text requests" come from, for how many users, and how is a user placed by one?
5. How are the 500 concepts answerability-ranked, and on which cohort? And how do six arms yield Table 4's "56 non-nested intervals"?
6. Did you try the *measured* pairwise loss (Spearman +0.784) as a cheaper split criterion at scale?

## Minor issues

- Define the instrument's four certification requirements inline rather than by reference to [3].
- One figure against five dense tables; a plot of answers collected per channel by budget would carry the concept argument better than prose.
- Table 1's Golbandi K=16 em-dash is unexplained at the point of use.

## What would change my recommendation

Two things would move this to a clear accept. First, an anonymised artefact containing the frozen instrument — weights or a full recipe — plus the greedy search and tree code; without it the contract is an assertion, and a reproducibility-framed paper cannot ask reviewers to take the measuring device on trust. Second, straight answers on depth-5-versus-K=16 and on the RecVAE zero row, which between them touch the headline number and one of its three corroborations; an ablation asking film questions under the implicit-watch answerability rule would additionally firm up the paper's most useful takeaway. Conversely, if most of the K=16 interview turns out not to be tree-chosen, I drop to borderline: "one variable changed inside the field's own tree" would then not describe the arm producing the headline.
