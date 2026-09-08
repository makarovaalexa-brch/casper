# ECIR 2027 — Full Paper Review (Reviewer 3)

**Paper:** Objective Mismatch in Adaptive Preference Elicitation for Cold-Start Recommendation

## Summary

The paper re-implements twenty-one published cold-start question-selection rules under one measurement contract on one frozen recommender (a two-tower set encoder at near-RecVAE parity): ML-25M, 10,000 held-out test users, full-catalogue NDCG@10 (full and tail), answers read from each user's own held-out ratings, unanswerable questions charged. RQ1's headline is negative: every published selector, adaptive or not, loses at every budget from one to sixteen questions to a fixed order built by greedily maximising the reported metric on the same recommender. RQ2 diagnoses why: three published split criteria correlate at +0.010, −0.060 and −0.104 with exact measured NDCG@10 gains at the 22 nodes of a grown tree, forgoing 66–89% of what a random pick forgoes. RQ3 changes one variable — the scalar a candidate is scored by inside Golbandi's ternary tree — and the comparison reverses: +0.01035 full and +0.00705 tail at sixteen questions, where the surrogates lose by three times that margin. A larger separate finding is that enlarging the bank with tag-level "concept" questions, with no conditioning at all, is worth more (+0.01430 full, +0.02199 tail), because sixteen concept questions collect 12.36 answers where sixteen film questions collect 3.11. Qualifiers are honest: validation overstates the tree gain 2.7×, six of fifteen leaves lose on test, the routing test is a null, and one alternative consumer flips the ordering.

## Scores

| Criterion | Score | Justification |
|---|---|---|
| Relevance | 5 | Cold-start elicitation, evaluation contracts and reproducibility are squarely ECIR business. |
| Originality | 4 | The single-contract replication at this scale and the measured-gain criterion audit are new; the "static beats dynamic" observation is acknowledged as a replication of Sepliarskaia et al. |
| Technical soundness | 3 | Controls are careful, but RQ1 rests on twenty-one re-implementations whose tuning is undocumented and on a comparator no published number validates. |
| Quality of presentation | 3 | Organised but dense; several sentences need two passes and the largest practical finding is buried on page 11. |
| Reproducibility | 2 | The frozen instrument exists only as a cited anonymous tech report; no baseline hyper-parameters, tuning budget, implementation notes or dataset table. |
| Potential impact | 4 | "Score the question by the metric you will report, and enlarge what you ask about" is directly actionable. |

**Overall recommendation:** Weak accept. **Confidence:** 4.

## Strengths

- The measurement contract is the best part of the paper: full-catalogue ranking with no sampling, behavioural answers with no simulator in the measurement path, unanswerable questions charged, and an exact contract gate (K=1 per-user difference 0.0 for all 20,000 users) proving tree and comparator start from the same place.
- Honest negative reporting: the K=2 tail interval excluding zero *against* the method, the Drama null, the 2.7× validation inflation, six losing leaves covering 39.3% of users, the routing null, and the fifth-consumer reversal where Thompson sampling wins. This raised my trust in everything else.
- RQ2's audit is the most transferable artefact here: 22 nodes × ~1,000 candidates of exact measured gain, three criteria, a random-pick control. That the criteria agree with the metric at the root and stop agreeing deeper down is a real insight.
- The Table 4 ablation, with two shuffle controls (one min-leaf-respecting), kills the "the tree just got deeper" story; the answerability mechanism (3.11 vs 12.36 answers collected) is concrete and triangulated three ways.

## Weaknesses

1. **The baselines are undocumented as implementations (Sec. 3–4, pp. 4–7).** Twenty-one rules are re-implemented and all twenty-one lose, yet nothing states how each was ported, what hyper-parameters were searched, or whether ranges covered the original authors'. The paper cites "Are we really making much progress?" and is exposed to exactly that criticism: a strong home-built comparator against baselines tuned by their re-implementer. The Sec. 2 gate ("beating random and popularity") is a floor, not a tuning protocol.
2. **Instrument and comparator are unverifiable from this submission (Sec. 3, p. 4; ref. [3]).** Every number depends on a frozen encoder whose certification lives in an anonymous under-review report, and the comparator has, by the authors' own admission, no published validation. There is no dataset table, no implementation subsection, no named toolkit, and one redacted footnote.
3. **CASPER's practical economics are stated once and never confronted (Sec. 6, p. 11).** 37,191 seconds per tree against eleven for the Taylor construction, for +0.01035 NDCG@10, on a tree whose gain shrinks 2.7× from validation to test and whose leaves lose for 39.3% of users. I want a paragraph on when that trade is worth making.
4. **The biggest practical result is buried (Sec. 6, p. 11).** "Changing the bank yields more than conditioning does" arrives in the second-to-last results paragraph, after the method the paper is named for. The abstract mentions it, but the architecture buries what a practitioner would act on first.
5. **Arm counts are inconsistent:** "twenty-one published selection rules" (abstract, conclusion), "twelve of the twenty-three arms" and "six of the thirteen published methods" (p. 7), and a Table 1 captioned "every selector measured to date" that omits five.

## Detailed comments by section

**Abstract and introduction.** One pass is enough to learn what the paper found — not the norm, and I credit it. Two problems. "The three published split criteria forgo 66–89% of what a random choice forgoes" stopped me: it first reads as if the criteria beat random, and takes a second pass to see it is damning. Consider "recover only 11–34% of the available gain". Second, the abstract never says the concept-bank result needs *no conditioning at all* — the sentence that changes practice. The scoping line "a fixed order is an adaptive policy that ignores its inputs… no measurement can place adaptivity below staticness, only particular methods" is exactly right and should be repeated in the conclusion.

**Related work.** Well targeted, and the Sepliarskaia and Golbandi-2010 qualifications are self-limiting scholarship of a kind reviewers should reward. One sentence I read three times: "What separates the family is the quantity, cumulative regret over T rounds against post-interview held-out ranking here, so a ported policy is gated on reproducing the selection rule as written…". Three ideas, one sentence, "here" far from its referent.

**Measurement contract (Sec. 3).** The section a practitioner would reproduce from, and it is a page and a half where the venue norm is two to three. Missing: dataset table; binarisation threshold if any; runs/seeds; toolkit; per-baseline provenance ("official code" vs. "ours"). "The combined sequence being a bound on what both channels can buy" (p. 5) is an optimality claim with no argument.

**RQ1 (Sec. 4).** The distinct-prefix counts (Thompson: a distinct interview per user from Q4; GLM-UCB: 5,739 by Q16) are the most persuasive evidence here, pre-empting the objection that the adaptive arms simply failed to branch; the answered-question column is the mechanism and deserves promotion. Table 1's `ans@16` mixes units ("1.5i/9.2c" needs a gloss), the Golbandi row is missing at K=16 with no note, and Fig. 2's eight series will not survive greyscale printing.

**RQ2–RQ3 (Secs. 5–6).** Both strong. "These are not weak but aligned surrogates" is a parsing trap — I read "not weak, but aligned", the opposite of the intent; hyphenate. Likewise "headroom runs the other way": than *what* is not recoverable in one pass. Table 4 is the table I would show a colleague, and the logistic arm degrading until it falls below its own one-question value is a memorable lesson — agreeing with the metric on the easiest decision buys nothing. This is where the missing cost discussion bites and where the bank result needed its own heading.

**Limitations (Sec. 7).** Blunt and welcome, but the answerability assumption is not neutral for the headline: "has rated it" as a proxy for "could have answered" plausibly under-counts answerability for popular films — exactly where the item-based selectors concentrate.

## Questions for the authors

1. For each of the twenty-one rules: official code, author artefact, or your re-implementation? What search was run, on what budget, selected on which metric?
2. Can the frozen instrument be described sufficiently here (architecture, training, parity) that results do not depend on an unavailable report?
3. What are the denominators — 21 rules, 23 arms, 13 published methods, 5 omitted?
4. Under Golbandi's leaf model Thompson beats the static order at every runnable budget. Is the RQ1 null a property of elicitation, or of neural fold-in consumers?
5. Would CASPER need a per-cohort refit in deployment, and what is 37,191 s per tree worth against +0.01035?
6. Does the rating-based answerability proxy bias the item-versus-concept comparison, given that rating propensity tracks popularity?
7. Why is the Golbandi tree missing at K=16 in Table 1?

## Minor issues

- Table 1's caption promises "every selector measured to date" then omits five arms.
- Bold-best without underline-second-best, unlike the venue norm; Fig. 2 needs greyscale-safe markers and fewer series.
- References [9], [14], [27] carry visible internal provenance notes, including "FULL AUTHOR LIST UNVERIFIED – replace the and others before print"; [14] as it stands is not citable.
- "entropy0" is subscripted in the table, unsubscripted in prose; the redacted code footnote should be an anonymised repository link at submission.

## What would change my recommendation

An accept is within reach and needs no new experiments. Give me (i) a per-arm implementation and tuning table — provenance, hyper-parameters, search budget, native-validation evidence — because the strongest claim currently has the weakest paper trail; (ii) enough of the frozen instrument in this document that a reader does not depend on an unavailable report, plus an anonymised artefact link; and (iii) a paragraph pricing CASPER against its surrogates in wall-clock and refit frequency. Reconciling the arm counts and giving the concept-bank result its own subsection would take presentation to 4. Conversely, if ported baselines received no tuning beyond a random/popularity floor while the comparator was searched exhaustively on the very recommender it is scored on, I drop to borderline: the headline null would then be about implementation effort rather than objectives.
