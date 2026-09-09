# ECIR 2027 — Full Paper Review (Reviewer 1)

**Paper:** *Objective Mismatch in Adaptive Preference Elicitation for Cold-Start Recommendation*

## Summary

The paper argues that published cold-start elicitation selectors optimise objectives other than the metric they are now judged on, and that this mismatch — not adaptivity itself — explains their weak measured performance. Thirteen published methods (twenty-three arms) plus classic heuristics are re-implemented under one "measurement contract" on ML-25M: a frozen two-tower set encoder as scoring recommender, answers read from users' own held-out ratings, full-catalogue ranking without sampling, unanswerable questions charged, 10,000 held-out test users. The comparator is a static order built by lazy-greedy maximisation of held-out full NDCG@10 over a 1,000-question bank; every published selector loses to it at every budget (RQ1). Three published split criteria are then scored against exact measured NDCG@10 gains at the 22 nodes of a grown tree, giving near-zero or negative rank correlations and 11–34% regret recovery (RQ2). Finally, substituting measured NDCG@10 into Golbandi's ternary interview tree ("CASPER") reverses the comparison (+0.01035 full at K=16), and enlarging the bank with Tag Genome concept questions buys more (+0.01430) with no conditioning at all (RQ3). Section 7 lists four limitations, including the comparator's search-on-the-metric advantage.

## Scores

| Criterion | Score | Justification |
|---|---|---|
| Relevance to ECIR | 5 | Cold-start elicitation, reproducibility and evaluation methodology are squarely in scope. |
| Originality | 3 | The battery's breadth and the criterion-regret measurement are new; the "static beats dynamic" headline is conceded to replicate Sepliarskaia et al. (2018), and metric-in-the-split is a natural fix. |
| Technical soundness | 2 | The central comparison is asymmetric by construction, and two load-bearing modelling choices (answerability inference; item-vs-concept answer equivalence) are neither controlled nor probed. |
| Quality of presentation | 3 | Dense but organised; several tables under-specified, Table 1's notation opaque. |
| Reproducibility | 2 | Code withheld with a request that reviewers disregard its absence; the instrument is an anonymous unpublished report; one dataset, one split, one seed. |
| Potential impact | 3 | The methodological warning is valuable; the positive results are small, expensive and instrument-specific. |

**Overall recommendation:** **Weak reject**  **Reviewer confidence:** 4

## Strengths

- Unusual and useful breadth: interview trees, optimal design, active learning, conversational bandits and learned-static elicitation under one stated protocol.
- The contract is well chosen: full-catalogue ranking (citing Krichene & Rendle), unanswerable questions charged, no simulator or LM in the measurement path, disjoint fit/score cohorts.
- Section 5's criterion-regret table is the paper's best idea — what a surrogate forgoes against exact measured gains, with a random-pick control it must beat.
- Table 4 is a proper one-variable ablation, and the paper is candid about unregistered readings, validation overstating the gain ~2.7×, 6/15 leaves losing on test, and the routing congruence test being a null.
- The authors resist the obvious over-claim: "no measurement can place adaptivity below staticness, only particular methods." That belongs in the abstract.

## Weaknesses (most serious first)

1. **The headline comparison is asymmetric by construction, and close to tautological (Sec. 3 "The comparator"; Sec. 4; conceded Sec. 7).** The static order is the greedy maximiser of exactly the reported metric, computed *through the same frozen recommender that then scores it*. No ported arm has that access. Defining the comparator this way makes RQ1 largely definitional. The symmetric experiment — the same fitting budget on the same objective for the published families, as Table 4 does *within* the tree — is never run for the bandits, DRE, IGCN or the heuristics.
2. **Item and concept "questions" are not commensurable, so the +0.01430 claim may be an artefact (Sec. 3 "Answers and interface"; Sec. 6).** An item answer is one half-star on one film; a concept answer is `log2((n_c+½)/(e_c+½))`, a statistic over the user's *entire* profile, computable without the user saying anything. Sixteen concept answers transmit far more history than sixteen item answers. The paper attributes the whole advantage to answerability (3.11 vs 12.36 answers) and never separates answerability from per-answer information content. The 500/500 bank is a further confound: the concept side is ranked by exactly the property that later explains its win.
3. **Possible target-set leakage (Sec. 3).** Answers are read from the *held-out* ratings over which NDCG@10 is computed. If an asked-and-answered item stays in the target set, folding it in should place it near rank 1 and mechanically inflate the metric. The paper says the *candidate pool* removes fold-in items (141.2/user) but is silent on answered target items. This must be stated; it changes the reading of Table 1.
4. **The instrument is unreviewable (Sec. 3; ref. [3]).** Everything is scored through an encoder "certified" in an anonymous report under review; I cannot assess frontier parity, the empty-interview identity, or the fold-in interface. Its parity is also *below* the in-house RecVAE bar on both axes (0.3496/0.2491 vs 0.3540/0.2497), undercutting "a recommender nobody can dismiss as weak".
5. **Consumer transfer contradicts the RQ3 claim (Table 5).** +0.01035 on the certified tower, +0.04122 on RecVAE, +0.00161 (nothing) on the accumulator, and *negative* on the leaf model. That is not "it holds under other consumers"; it is evidence the gain is instrument-specific — expected, since the split criterion *is* the tower's own metric. The abstract's "on four consumers" overstates it. The RecVAE static arm also reads `.0000/.0000` at K=2, which looks like a broken consumer, and +0.04122 comes from that block.
6. **Statistics (Sec. 3).** The bootstrap resamples users only, not the fitted order or grown tree (single draws each). With 138 cells in RQ1 and 56 intervals in Table 4 and no family-wise correction, "directional consistency" is an argument, not an error rate. Pre-registration of three cells is asserted but unverifiable in a blind submission.
7. **Port validity (Sec. 3 "Ports and tuning").** Only 5 of 13 have a native gate; EDDI failed its (3.402 vs 3.236 for random) and appears as an unpublished reduced variant marked †; Greedy SLIM is excluded. Several ports never clear the answer-free popularity reference (A-optimal .1740 and ConUCB .1874 at K=16, vs .1626), which is as consistent with a weak port as a weak method.
8. **RQ2 is off-policy (Sec. 5).** The 22 nodes are states visited by *the authors' own* metric-optimised tree; a criterion should be judged at the states its own tree reaches. The text notes this ("at the states this tree visits") but the answer to RQ2 generalises past it.
9. **Single dataset, split and seed, and cost buried (Sec. 3, 6).** Seed 98765, one Liang split, ML-25M only, for a thesis about the state of a field; and 37,191 s per tree against 11 s for the Taylor construction — ~3,400× for +0.01035 — belongs in the framing, not one late clause.

## Detailed comments by section

**Abstract/Intro.** "Every published method loses ... to that fixed order, which is chosen by greedily maximising the reported metric" is honest about mechanism but never says no ported method was allowed to do the same. Put the asymmetry clause where the claim is made.

**Sec. 2.** The most honest passage in the paper: "their 2018 paper already shows static questionnaires beating a dynamic one that optimises a function other than the loss, so RQ1 replicates that observation." Contribution bullet 1 should then say "replication". The Golbandi-2010 qualification ("when the split criterion essentially was the reported metric") anticipates the paper's own fix and deserves prominence.

**Sec. 3.** Not overwriting no-answer models with a popularity row is good practice, but the empty interview reads 0.1586 against an answer-free popularity reference of .1626 — the instrument at zero questions is *below* popularity; discuss what that implies for the battery. The `e_c < 1.5` refusal threshold is unmotivated: how sensitive is 12.36/16 to it?

**Sec. 4.** "Thompson sampling conducts a distinct interview for every one of the 10,000 users ... and still loses to one order asked of everybody" is a strong diagnostic, as is the finding that twelve of twenty-three arms are value-free. The Golbandi row has no K=16 entry and no explanation, awkward given Section 6 builds on that tree.

**Sec. 5.** "Placing users by a real free-text request instead spreads the optimum to 442 questions (effective 95.13)" introduces an undescribed object. Where do the requests come from, how many users, how were they mapped to states, and is a language model in that path — which the contract elsewhere forbids? As written this is unsupported and doing real rhetorical work.

**Sec. 6.** "Nothing is approximated and no ground truth is read at selection time, since the criterion is a cohort-level quantity on the validation cohort." Cohort-level or not, it reads validation users' held-out targets; legitimate, since test is disjoint, but the phrasing misleads. The best result in the paper is buried here: the *measured* pairwise rank loss tracks NDCG@10 at Spearman +0.784 while its Taylor-approximated training-loss version correlates at −0.149. Give it more room than the ablation around it.

**Sec. 7.** Frank, and the disclosure that rebuilding the comparator cut the gain from +0.01371 to +0.01035 is exactly what reviewers want. But footnote 1's request that reviewers "disregard its absence when assessing reproducibility" is not something a reviewer can grant; an anonymised artefact is standard at ECIR.

## Questions for the authors

1. Are asked-and-answered items removed from the held-out target set before NDCG@10 is computed? If not, how large is the inflation for item-only arms?
2. What happens to RQ1 if each published selector uses the same 1,000-question bank with parameters fitted on validation against full NDCG@10?
3. Can you separate answerability from per-answer information content in the concept channel (quantise the concept statistic, or give item questions an aggregate answer)?
4. Why does the RecVAE static arm read .0000/.0000 at K=2, and is +0.04122 robust to fixing it?
5. What is the variance of +0.01035 across re-fits (different validation cohorts, seeds)? The user bootstrap does not cover fit variance.
6. Why is there no K=16 entry for the Golbandi tree in Table 1?
7. Does RQ2 survive if each criterion is evaluated at the states of its *own* grown tree?
8. What is the provenance and protocol of the free-text-request analysis, and does it involve a language model?
9. On what basis is "frontier parity" claimed when the instrument scores below the RecVAE bar on both axes?
10. Does Table 1's ordering replicate on a second dataset or split?

## Minor issues

- Table 1: `ans@16` "1.5i/9.2c" is never defined; the caption's "degenerate or duplicate" is asserted without evidence (move those arms to an appendix); IGCN is labelled "adaptive" while Sec. 4 says it is static at K≤4; the 138-cell count is uncheckable with five arms omitted.
- Table 2: "top-10 recall" is undefined. Table 3: if K=1 is a contract gate, mark it as one rather than scoring it. Table 5: the fourth consumer is omitted but quoted in the text.
- Fig. 1: a log x-axis with five ticks makes crossings hard to read; consider plotting gain-over-static. Inconsistent decimals (`.2447` vs `0.26936`). Ref. [3] is a load-bearing anonymous self-citation — state whether it will be public at camera-ready.

## What would change my recommendation

Section 5's diagnosis and Table 4's ablation are publishable and are, in my view, the real contribution: measuring how far a surrogate sits from the reported metric, with a random-pick control, is a method others can reuse. What blocks acceptance is that RQ1 rests on an asymmetry the paper acknowledges but does not repair, and that both positive results have plausible alternative explanations left untested. I would move to **weak accept** if the authors (i) state unambiguously whether answered items are excluded from the target set, correcting the tables if not; (ii) add the symmetric comparison with at least the three strongest published selectors fitted on validation against the reported metric; (iii) add one control separating answerability from per-answer information content in the concept channel; and (iv) release an anonymised artefact including the instrument. Reframing around the criterion-regret and ablation evidence, with RQ1 presented explicitly as a replication of Sepliarskaia et al. under a modern metric and the comparator's advantage stated up front, would strengthen it further. A second dataset on top of that would move me to **accept**.
