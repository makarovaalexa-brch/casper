# Meta-review — Area Chair

**Submission:** *Objective Mismatch in Adaptive Preference Elicitation for Cold-Start Recommendation*

## What the paper contributes, and to whom it matters

Three things, in descending order of durability.

First, a **replication battery**: thirteen published cold-start elicitation selectors (twenty-three arms) re-implemented and run under a single, unusually explicit measurement contract — one frozen recommender, behavioural answers read from each user's own held-out ratings, the full 18,359-item catalogue ranked with no negative sampling, unanswerable questions charged rather than free, 10,000 held-out test users, fitting on a disjoint validation cohort. Every published arm loses, at every budget, to a static order built by greedily maximising the reported metric.

Second, a **diagnosis** (Table 2): at the 22 states an adaptive tree actually visits, three published split criteria correlate with exact measured NDCG@10 gains at +0.010, −0.060 and −0.104, and recover only 11–34% of what a uniformly random pick leaves unrealised. This is the paper's intellectual core — the mismatch is measured against ground truth, not argued.

Third, a **one-variable intervention** (Tables 3–4): put measured held-out NDCG@10 into Golbandi's split criterion, hold the tree code, branch rule, leaf minimum, pool, cohorts, budget and recommender fixed, and the interview rises above the static order (+0.01035 full, +0.00705 tail at K=16); five alternative criteria fitted by the identical search do not. Alongside it, the honest counter-result: enlarging the question bank with concept questions, with *no* conditioning at all, buys more (+0.01430 full, +0.02199 tail).

This matters to the elicitation and conversational-recommendation communities, and to the ECIR reproducibility constituency in particular. It is the kind of paper that changes what the next author measures.

## Strongest and weakest

**Strongest:** Table 4. The ablation removes exactly the fairness objection that the RQ1 audit invites — every criterion, theirs included, fitted by the same search on the same data at the same budget — and the sign of the conditioning gain flips on the scored scalar alone. The supporting analysis is better still: the measured pairwise rank loss tracks NDCG@10 at Spearman +0.784 while its Taylor-approximated training-loss form correlates at −0.149, which localises the failure in the *approximation* rather than the *shape of the loss*, and refutes the paper's own pre-registered expectation. The two shuffle controls, the min-leaf-respecting one being the fair control, and the observation that the shuffle grows *deeper and more varied* while losing by −0.048, are exactly the controls I would have asked for.

**Weakest:** the evidence base is one dataset, one split, one seed, and above all **one instrument**, and the central criterion is the metric computed through the very recommender that then scores it. The authors see this and answer with Table 5 — but Table 5 is also where the answer is weakest: on Golbandi's leaf model the effect *reverses* at K=8 (−0.01375, interval excluding zero), and the RecVAE column is conceded to count for direction only. The honest reading of the transfer evidence is two clean consumers, one reversal, and one arm the authors themselves discount. The criterion-regret table, the paper's most quotable result, has no transfer evidence at all; Section 7 says so.

## Over- and under-claimed

**Over-claimed:** the abstract's "every one loses, at every budget" is technically true and rhetorically larger than the design supports. The comparator is given a metric-driven search over a validation cohort that no ported arm receives. Section 3's "What the comparison is, and is not" is a model of how to bound such an asymmetry — the bandit re-tuning experiment (+0.0058 and +0.0018 against a 0.037–0.044 deficit) is a real bound — but that qualification lives on page 5 while the abstract carries the headline. The qualification belongs where the claim is made.

**Also over-claimed, more subtly:** Table 2 evaluates three surrogates at the 22 nodes of a tree *grown under the measured criterion*. Those are not the states a variance-grown or pairwise-grown tree would visit; the criteria are scored off their own trajectories. This does not overturn the finding — a criterion uninformative at plausible interview states is still a problem — but "the states an adaptive interview visits" is doing work the experiment does not license, and the fix is cheap.

**Under-claimed:** the answer-rate decomposition on page 11 is buried. That a film answer is worth 3.98× a concept answer while the concept channel collects 3.98× as many, so the channels tie on full and separate on tail, with two concept questions leaving 11 of 10,000 users mute against 5,779 for two film questions — that is a clean, portable, mechanism-level finding about *what to ask about*, and it is stronger than its half-page. Likewise, that twelve of twenty-three arms carrying the "adaptive" label are value-free is a standalone service to the literature, dispatched in four lines.

## REQUIRED before camera-ready

1. **Recompute Table 2 at states each criterion's own tree reaches** — or, if compute forbids, add an explicit paragraph stating that the node set is the measured-criterion tree's and that the comparison is therefore off-trajectory for the surrogates. One or the other; the current framing is not sustainable as written.
2. **Make the instrument self-contained.** The certification of the frozen recommender rests on anonymous reference [3], a technical report under review. Add an appendix giving the four certification requirements, the parity numbers and the empty-interview identity check in full, so the paper stands without an unreviewable citation.
3. **Move the fitting asymmetry into the abstract**, in one clause, and state there that Table 4 is the load-bearing comparison.
4. **State the Golbandi-leaf reversal wherever the consumer claim is made** (Section 7 and the conclusion), not only in Section 6's body text. "Three of four" should not be compressed to a clean win.
5. **Enumerate the pre-registered cells explicitly** and state the no-correction position in the same place — Section 3 asserts three cells were specified before the test cohort was constructed; name them.
6. **Supply the anonymised artefact mirror** as the footnote promises.

## ADVISED

7. Give Table 4 a figure; it is the paper's best result and currently the least visual.
8. Report the construction cost trade-off (37,191 s vs 11 s) as a subsection, not a clause — it is the practical barrier to adoption, and a paragraph on amortisation would help adopters.
9. Clarify the arithmetic of "thirteen published methods" against five natively gated, one reduced (†) and Greedy SLIM not run. A one-line table of gate outcomes removes a reviewer's easiest objection.
10. If a second seed or split is cheap on the existing pipeline, report Table 3 under it. One seed for a +0.01035 headline invites the question.

## Scores

| Criterion | Score |
|---|---|
| Relevance | 5 |
| Originality | 4 |
| Technical soundness | 4 |
| Quality of presentation | 3 |
| Reproducibility | 4 |
| Potential impact | 4 |

Presentation at 3 is compression: the paper is superb per-sentence and punishing per-page, with load-bearing numbers stranded in body text and one figure carrying five tables' worth of narrative. Reproducibility at 4, not 5, is not about the artefact — which I am instructed to treat as complete, and which the protocol description (seed, split, pool sizes, per-user fold, bootstrap procedure, tuning grids) amply supports — but about the instrument's dependence on [3].

## Decision

**Weak accept. Confidence: 4/5.**

*To the authors:* This is a careful, self-critical audit whose central ablation is the cleanest one-variable experiment I have read on this problem this year, and I am recommending acceptance on the strength of Table 4 and the criterion-regret diagnosis; the required changes concern the states at which the surrogates are judged, the self-containedness of the instrument, and moving your own qualifications up to sit beside the claims they qualify.

## What would make this an unambiguous Accept

One experiment: **replicate Tables 2, 3 and 4 on a second dataset with a second independently certified recommender** — the criterion-regret correlations, the six-criterion ablation and the conditioning gain. If the surrogates are near-uninformative and the measured criterion still flips the sign somewhere other than ML-25M and this tower, the objective-mismatch thesis stops being a property of one instrument and becomes a property of the field. That is a follow-up paper, not a camera-ready revision — the tree costs ten hours to build and the second recommender needs its own certification. Achievable before camera-ready, by contrast: the re-run of Table 2 at each criterion's own nodes (item 1), which is the same machinery on states the pipeline already produces.

## Acceptance probability as it stands

At a venue taking roughly the top 25%: **55%.** The reproducibility framing and the quality of the controls are strongly in its favour; the risks are a reviewer who reads the fitting asymmetry as fatal to RQ1 despite Section 3, a reviewer who finds +0.01035 too small to headline when the paper's own bank result beats it, and the density of the presentation. It is above the line for me, but not by so much that a single hostile reviewer could not put it below one.
