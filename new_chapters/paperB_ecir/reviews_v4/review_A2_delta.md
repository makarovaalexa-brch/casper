# ECIR 2027 — Second-stage review (Reviewer 1, delta on the revision)

**Paper:** *Objective Mismatch in Adaptive Preference Elicitation for Cold-Start Recommendation*
**Previous recommendation:** Weak reject (confidence 4)

I diffed v4 against v5 in full. The revision is a careful pass: the same experimental corpus with no new runs, but the disclosure that could have sunk the paper has been made, and the framing now matches the evidence in several places where it did not. Per the chairs' instruction I score reproducibility as if the anonymised repository is present, judging only the adequacy of the paper's own description.

## Original weaknesses, one row each

| # | What I objected to | What the revision does (quoted) | Verdict |
|---|---|---|---|
| 1 | RQ1 comparison asymmetric by construction; symmetric fitting never run for the published families | New Sec. 3 subsection: "The comparator is given something no ported arm is given: a search, on the reported metric, over a validation cohort. That asymmetry is the substance of the contrast and also its principal limit… it is a fitting advantage." Bounded with v4's bandit tuning ("+0.0058 and +0.0018 against a deficit of 0.037–0.044"), then redirected: "A reader who discounts RQ1 on fairness grounds should read that table [Table 4] as the load-bearing comparison." | **Partly resolved** — the asymmetry is now stated where the claim is made, and the symmetric ablation is nominated as load-bearing: the reframing I asked for. But no new symmetric arm was run, and the quoted numbers cover two bandits, not DRE, IGCN or the heuristics. |
| 2 | Item and concept answers not commensurable; +0.01430 may be per-answer information, not answerability | Limitations concedes: "Nor are the channels commensurable per question: a concept answer summarises many films where an item answer carries one, so that channel's advantage is answerability together with breadth, not answerability alone." | **Partly resolved** — the over-claim is withdrawn, which earns real credit. No control was run; Sec. 6 still asserts "The mechanism is answerability", so body and limitations disagree, and the 500-popularity/500-answerability bank confound is untouched. |
| 3 | Possible target-set leakage from answering held-out items | New in Sec. 3: "Answers are read from a per-user lookup built with that user's held-out targets removed and asserted at run time to contain none of them, and an answered item is outside the candidate pool, so no answer can place its own target in the ranking." | **Resolved** — unambiguous, names a run-time assertion, tables need no correction. My largest single risk, closed. |
| 4 | Instrument unreviewable; "frontier parity" claimed while below the RecVAE bar | "against a RecVAE bar [33] of 0.3540/0.2497 rebuilt here, indistinguishable on the tail (paired interval spanning zero) and 1.2% short on the full profile"; the abstract now says "one strong recommender" (was "frontier"). | **Partly resolved** — the parity claim is quantified and honest. Ref. [3] is still an anonymous unpublished report, and RQ1's own statement still says "one frontier recommender". |
| 5 | Consumer transfer contradicts the RQ3 claim; RecVAE `.0000/.0000` cell | "RecVAE's +0.04122 is measured against a static arm with almost no usable input and counts for direction only"; the Table 5 caption adds that "RecVAE ingests liked items only, and its low-budget cells are a property of that input contract rather than of the interview." | **Resolved on the mechanism, not the claim** — the broken-looking cell is explained and its number demoted. But the abstract still reads "on four consumers of the answers" when the honest count is three of four, one direction-only. Fix the abstract. |
| 6 | Statistics: no fit variance in the bootstrap, 138+56 intervals, no correction | Unchanged, word for word. | **Not addressed** |
| 7 | Port validity; several arms below the answer-free popularity reference | Unchanged, except the abstract's welcome "thirteen of them, one in reduced form". | **Not addressed** (honesty improved in the abstract) |
| 8 | RQ2 is off-policy — criteria judged only at *this* tree's states | Body unchanged ("at the states this tree visits"), but the Answer to RQ2 is narrowed to "after two questions there are roughly five materially different questions". | **Partly resolved** — the generalisation past the state distribution is trimmed; the off-policy design stands. |
| 9 | One dataset/split/seed; 3,400× cost buried | Unchanged. | **Not addressed** |
| Minor | Golbandi's missing k=16 | Caption now explains it: "3^k answer patterns fragment its training cohort beyond depth eight". | **Resolved** |
| Minor | Undescribed free-text-request analysis (442 questions, effective 95.13) | Deleted outright. | **Resolved by removal** |
| Minor | `ans@16`, `1.5i/9.2c`, "top-10 recall", IGCN labelled adaptive, inconsistent decimals | Unchanged. | **Not addressed** |

## New in the revision, and what got weaker

- **A material new disclosure in Sec. 6**: the sixteen questions are "the splitters on that user's path, then a tail chosen inside their leaf by the same measured-NDCG search over the same pool". The tree contributes five conditioned questions; eleven come from a per-leaf greedy search. Disclosing this is right and I credit it, but it changes how Table 3 reads — +0.01035 at K=16 is depth-5 conditioning plus a leaf-conditioned tail, not sixteen adaptive questions. Say so where the number is quoted, not only in the construction paragraph.
- **Figure 1 was deleted** to buy space. My objection was to its log axis, not its existence; five dense tables and no picture is harder to read, and the oracle ceiling is now invisible. Cut two omitted-arm rows instead.
- **Weakened claims**, correctly but worth noting: the RQ2 dispersion result loses its free-text span; the leaf-model reversal loses its numbers ("+.0100⋆ to +.0281⋆ full" is gone); RecVAE's +0.04122 is demoted. Sections 7 and 8 merge into "Limitations and Conclusion", which costs the argument nothing.

## Final recommendation

**Borderline** (confidence 4). The sentence I would send the authors: *The leakage question is now answered cleanly and the RQ1 asymmetry is finally stated where the claim is made — but the two experiments I asked for, a symmetric fit for the strongest published selectors and a control separating answerability from per-answer breadth, were conceded in prose rather than run, and that is what still separates this from a clear accept.*

**Did my recommendation move?** Yes — from Weak reject to Borderline, on the leakage disclosure, the honest parity statement, the RecVAE demotion, the leaf-tail disclosure, and the reframing of Table 4 as load-bearing. It did not reach weak accept because two of my four conditions were met by writing rather than by doing. To move me further the authors must **run**, not phrase: (a) refit three published selectors — DRE, one bandit, IGCN — on the same validation cohort against full NDCG@10 over the same bank, and report whether RQ1 survives; (b) one control quantising the concept answer to item-like information content, splitting the +0.01430 into answerability and breadth. Both reuse the existing harness, cohorts and banked interviews; neither needs new data or a new model, so both fit inside a revision rather than requiring a different paper. A second dataset would still be needed for accept-with-confidence, and *that* is a different paper's worth of compute.

## Updated scores

| Criterion | v4 | v5 | Note |
|---|---|---|---|
| Relevance | 5 | 5 | unchanged |
| Originality | 3 | 3 | no new contribution; scope honestly narrowed |
| Technical soundness | 2 | **3** | leakage closed with a run-time assertion; parity quantified; RecVAE cell explained and demoted; channel breadth conceded. Held below 4 by the unrun symmetric fit and the unrun channel control |
| Quality of presentation | 3 | 3 | caption fixes and removal of the unsupported free-text passage, offset by the deleted figure and the still-undefined `ans@16` and "top-10 recall" |
| Reproducibility | 2 | **4** | with the repository stipulated present, the paper's own protocol description is now sufficient: answer-lookup construction, fitting cohort, leaf-tail selection and the RecVAE input contract are all stated. Not 5: one dataset, one split, one seed, and the bootstrap still carries no fit variance |
| Potential impact | 3 | 3 | the methodological warning still travels further than the +0.01 gains |

## Is the residual objection this paper's flaw, or the question's?

Partly the question's. The comparator asymmetry is *inherent*: the moment you ask whether published selectors optimise the metric they are now judged on, the only sharp instrument is a comparator that does optimise it, and any such comparator gets a fitting advantage no ported method was designed to take. Every paper asking this question inherits that, and the revision handles it about as well as prose can — bounding it, refusing the stronger reading at scoring time, and nominating Table 4, where the asymmetry genuinely vanishes, as the load-bearing evidence. The rest is not inherent. Whether well-fitted published selectors close the gap is an empirical question this harness can answer in an afternoon; whether the concept channel wins on answerability or on breadth is a one-variable control on data already on disk; the single dataset, split and seed are resource choices, not properties of the question. The objections that remain are unrun experiments rather than unanswerable ones — which is precisely why this sits at borderline rather than at reject.
