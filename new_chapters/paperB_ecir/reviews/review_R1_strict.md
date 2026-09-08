# ECIR 2027 — Full Paper Review (Reviewer 1)

**Paper:** *Objective Mismatch in Adaptive Preference Elicitation for Cold-Start Recommendation*

## Summary

The paper argues that cold-start elicitation methods optimise surrogate objectives (rating variance, information gain, posterior variance, A-optimality, bandit indices) while the field reports truncated top-k ranking, and that this mismatch — not adaptivity itself — explains their behaviour. Twenty-one published rules are re-implemented under one contract on MovieLens-25M, with a frozen "certified" two-tower set encoder as consumer, answers read from users' own ratings, and unanswerable questions charged as forfeited turns. Against a static comparator built by greedy maximisation of held-out full NDCG@10 on that same recommender, every rule loses at every budget from K=1 to K=16 on 10,000 test users, and not for want of branching. Section 5 quantifies the mismatch: at the 22 nodes of a grown tree, three published criteria correlate at +0.010, −0.060 and −0.104 with exact measured NDCG@10 gains, forgoing 66–89% of what a random pick forgoes. Section 6 introduces CASPER — Golbandi's ternary interview tree with the split criterion replaced by measured held-out NDCG@10 — which reverses the comparison (+0.01035 full, +0.00705 tail at K=16). Finally, enlarging the bank with Tag Genome concept questions, with no conditioning, is worth more (+0.01430 full, +0.02199 tail), because concepts are answerable where films are not (12.36 vs 3.11 answers in sixteen turns).

## Scores

| Criterion | Score | Justification |
|---|---|---|
| Relevance to ECIR | 5 | Cold-start elicitation, replication and evaluation methodology are squarely in scope. |
| Originality | 3 | Well-executed framing, but the authors concede RQ1 replicates Sepliarskaia et al. (2018); the new element is one criterion substitution. |
| Technical soundness | 2 | The central comparison is asymmetric (an order fitted on the evaluation objective and the evaluation model vs untuned ports), and the one alternative consumer tested reverses the headline. |
| Quality of presentation | 3 | Dense, numerate prose, but crowded tables, under-specified setup, inconsistent arm counts. |
| Reproducibility | 2 | Instrument is an unpublished anonymous report; no tuning, seed or repetition protocol for 21 re-implementations; five arms omitted unnamed. |
| Potential impact | 3 | The mismatch diagnosis is useful; the prescription encourages metric-targeted selection whose transfer is untested. |

**Overall recommendation: Weak reject.** **Reviewer confidence: 4.**

## Strengths

- The contract (Sec. 3) is explicit and disciplined: full-catalogue ranking with no sampling, behavioural answers rather than a simulator, unanswerable questions charged, disjoint fitting/scoring cohorts, and an exactly-zero K=1 gate (Table 3).
- The scale of re-implementation (21 rules, 10,000 users, 138 cells) is a service to the community, and Table 2 — criteria scored against *exact measured* gains at states a real tree visits — is the most convincing contribution.
- The authors are candid about awkward findings: the Golbandi-leaf reversal, the Drama null, the routing null, 6 of 15 leaves losing on test, validation overstating the gain ~2.7×.

## Weaknesses (most serious first)

1. **RQ1's comparison is asymmetric by construction (Sec. 3 "The comparator"; Sec. 4).** The comparator is built by greedy maximisation of held-out NDCG@10 *on the very recommender it is scored with*, from a 10,000-user fitting cohort, while baselines are ported "as written" and, per Sec. 7, "one free constant, the exploration weight, was never tuned". No tuning budget, grid or selection metric is reported for any baseline. An untuned bandit is not a measurement of that bandit; the headline conflates objective mismatch with fitting asymmetry, and the paper never bounds how much of the .0281 gap is the latter.
2. **One dataset, one non-public instrument, and the consumer that reverses the result is left unexplored (Sec. 3; Sec. 4 "Robustness").** The consumer is an anonymous report "under review" [3], so every number is conditional on its inductive bias. Under Golbandi's leaf model, Thompson sampling *beats* the static order at every runnable budget (+.0100 to +.0281) — given three sentences. That is evidence the headline is partly a property of the chosen consumer, and it deserves a subsection.
3. **CASPER is fitted on the same metric and the same frozen scorer used to evaluate it (Sec. 6).** The consumer-robustness sweep covers the RQ1 arms but not the paper's own method, so "build the selector around the quantity that will be reported and around nothing else" is indistinguishable from "matching the scorer helps against that scorer". The effect is modest (≈4% relative, one split, one seed), negative at K=2 on the tail, and unidentified in mechanism — leaves share no free picks, yet the pre-registered routing test is a null.
4. **Multiplicity is never addressed (Secs. 4, 6).** Stars are attached across 138 cells, a 56-interval matrix and per-K tables from paired bootstraps, with no correction, no seed variability, no repetition over splits. For 0.005–0.010 NDCG gaps this matters.
5. **The item/concept comparison may be an artefact of the answerability construction (Sec. 3 "Answers and interface"; Sec. 6).** Item answers come from held-out ratings; concept answers from "the signed implicit-watch statistic over their profile", with neither formula nor data half stated. If the fold-in half, the channel partly re-encodes what the model already holds; if the target half, it touches scored data. The 12.36-vs-3.11 answer gap, and the paper's strongest practical claim, hangs on this.
6. **Reproducibility falls short of replication norms (throughout).** No dataset table, preprocessing statement, tuning section, repetitions, or hardware beyond one wall-clock figure; per-arm native validation is asserted, never shown; Table 1 omits "five further degenerate or duplicate arms" without naming them.
7. **Missing comparator.** Greedy SLIM [29] is named in Sec. 2 as "the closest protocol sibling" — same dataset, metric and tail definition — and is not run.

## Detailed comments by section

**Abstract/Intro.** "Every published rule loses, at every budget" is unqualified, yet Table 1 leaves the Golbandi tree's K=16 cell empty and Sec. 4 reports four cells that "separate neither way". Conversely, "a fixed order is an adaptive policy that ignores its inputs" is exactly right and belongs in the abstract, which currently reads as the stronger claim.

**Sec. 2.** The admission that RQ1 "is a replication of that rather than a discovery" is correct, but undercuts a contributions bullet still leading with "the first replication". References [9], [14] and [27] carry author-facing notes ("FULL AUTHOR LIST UNVERIFIED – replace the and others before print") that must not appear in a submission.

**Sec. 3.** "certified against four requirements" and "a recommender nobody can dismiss as weak" describe a self-citation a reviewer cannot check; with one split and seed 98765, every number is a single draw and no sensitivity analysis is offered.

**Sec. 4.** Table 1 is dense and carries no confidence intervals. "Twelve of the twenty-three arms are value-free" contradicts "twenty-one distinct selection rules" two paragraphs later. The genre-conditioning result routes "groups with fewer than 200 validation users" back to the global order — a rule chosen with group sizes in view — and Drama, 54.8% of users, is a null; label it exploratory.

**Sec. 5.** The best evidence here, with two caveats. The criteria are scored only at the 22 states of the *winning* criterion's own tree, so surrogates are judged off their own support. And the dispersion analysis scores against the recommender's own predictive distribution — what the model believes is worth asking — yet the RQ2 answer ("roughly five materially different questions") is stated without that hedge, although it bounds adaptivity's headroom.

**Sec. 6.** The one-variable ablation and the min-leaf-respecting shuffle are the right controls. That the *measured* pairwise rank loss tracks NDCG@10 at ρ=+0.784 while its Taylor-approximated version correlates at −0.149 is the most valuable sentence in the paper: it reframes the story as "wrong approximation" rather than "wrong objective", which sits uneasily with the title. Cost is under-discussed: 37,191 s per tree against 11 s, for +0.01 NDCG, with no cheaper measured proxy attempted.

**Sec. 7.** Welcome candour — but that rebuilding the comparator cut the gain from +0.01371 to +0.01035 full and +0.02092 to +0.00705 tail is evidence of fragility, not a mitigation.

## Questions for the authors

1. Is it 21 or 23 distinct rules, and what are the five omitted arms?
2. What tuning did each ported baseline receive — specifically the exploration constants of GLM-UCB, Thompson, ConUCB and CLiME — and what happens to RQ1 if they are tuned on the validation cohort?
3. Exactly how is a concept answer computed, and from which half of the user's data?
4. Does CASPER's advantage survive a consumer change, in particular the Golbandi leaf model that reverses the RQ1 ordering?
5. Do the surrogates in Table 2 look equally uninformative at the states their *own* trees visit?
6. Given the routing null and zero leaf overlap, what mechanism produces the +0.01035?
7. What is the variance across seeds/splits, and does anything change under multiplicity control?
8. Why was Greedy SLIM [29] not run?

## Minor issues

- Bibliography notes in [9], [14], [27] must be removed.
- Table 1: empty K=16 cell unexplained; "ans@16" undefined at first use; "1.5i/9.2c" unexplained.
- Table 4: "—" entries unexplained; the caption's 56-interval claim would read better as a row footnote.
- Inconsistent decimals within two pages (".2447", "0.24295", "+0.0283"); standardise to four.
- Fig. 2: eight series is too many for a greyscale legend, and the log axis is named only in the caption's last words.
- Word-numeral counts disagree ("twenty-one", "twenty-three", "thirteen published methods").
- No acknowledgements/disclosure paragraph; check the reference page allowance (body 12pp, references to p.16).

## What would change my recommendation

I would move towards acceptance if RQ1's asymmetry were removed and RQ3's generality shown: (a) give every ported baseline a documented tuning budget on the validation cohort — at minimum the exploration constants — and re-report Table 1 with how much of the gap survives; (b) run CASPER itself, not only the RQ1 arms, under at least two alternative consumers including the Golbandi leaf model that reverses the ordering, so a reader can tell whether "put the reported metric in the split" is transferable or an alignment with one frozen scorer; (c) score the surrogates at their own trees' states; (d) define the concept-answer statistic and show the item-vs-concept result is not driven by a channel whose answerability derives from data the encoder already holds; (e) report seed/split variability with multiplicity control. Even partial versions of (a) and (b) — the two that decide whether the headline is about the literature or about this instrument — would move me to weak accept, since the Table 2 diagnostic and the measured-versus-approximated finding are worth publishing.
