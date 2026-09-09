# Meta-review, round 2 (Area Chair) — final recommendation

**Paper:** *Objective Mismatch in Adaptive Preference Elicitation for Cold-Start Recommendation*
**Prior meta-review:** Weak accept, confidence high, with two required and one advised change.

I diffed the revision page by page. The numbers are unchanged: no new experiment was run. Per the chairs'
instruction I treat the withheld repository URL as a blinding artefact and score reproducibility on the
paper's own description, which is good in recipe.

With the artefact assumed present, the reproducibility score three reviewers put at 2 is no longer the
constraint, and the disagreement reduces to comparator fairness.

## Required change 1 — the comparator-fairness asymmetry, with a number, before Table 1

**What it does.** A new named paragraph in Section 3, immediately before Table 1: *"What the comparison is,
and is not. The comparator is given something no ported arm is given: a search, on the reported metric, over
a validation cohort. That asymmetry is the substance of the contrast and also its principal limit … it is a
fitting advantage … And the ablation of Table 4 removes the asymmetry entirely … A reader who discounts RQ1
on fairness grounds should read that table as the load-bearing comparison."* The comparator paragraph also
now names the search cohort and adds *"No test user's targets enter any search."*

**Verdict: met in framing, not in evidence.** The asymmetry is stated where the claim is made, with numbers,
and the reader is told which table to weight — the disclosure I demanded, and the pointer to Table 4 is the
right argument. But the other half — *one arm run with a matched metric-driven search over its own free
parameters* — was not run; the bandit constants were already tuned in v4, and re-quoting them elsewhere is
no new experiment. The +0.01371 → +0.01035 rebuild is still only in Limitations.

## Required change 2 — reframe the abstract so the audit is setup and CASPER is the claim

**What it does.** Almost nothing. The abstract still opens *"Every published method loses … to that fixed
order."* Two edits: *"thirteen of them, one in reduced form"* and *"one strong recommender"* for *"frontier
recommender."* The pre-registered routing null still does not appear.

**Verdict: not met.** The cheapest of the three — no compute, no page budget — and the one declined.
Abstract, title and conclusion still lead with the half Reviewer 1 calls near-tautological and that the
authors concede replicates Sepliarskaia et al. Worse, *"on four consumers"* survives while the revised body
now says RecVAE's +0.04122 *"counts for direction only"* and the leaf model reverses at eight: the abstract
is contradicted by the authors' own text.

## Advised change 3 — one table pairing every headline gain with cost and reach

**What it does.** Nothing. The "What it costs" paragraph is verbatim identical (validation overstates 2.7×,
six of fifteen leaves losing over 39.3% of the cohort, ~14% answering nothing), and 37,191 s vs 11 s remains
a trailing clause two pages earlier. **Not met** — advisory, so lightly weighted, but a practitioner still
cannot read the price of +0.01035 off one place.

## What the revision did unprompted, and it matters

Three of the other reviewers' most dangerous objections are closed. (i) Target leakage: *"Answers are read
from a per-user lookup built with that user's held-out targets removed … and an answered item is outside the
candidate pool, so no answer can place its own target in the ranking."* That was Reviewer 1's weakness 3 and
could have sunk Table 1. (ii) The depth-5-versus-K=16 accounting Reviewer 3 said would drop them to
borderline: the sixteen are now *"the splitters on that user's path, then a tail chosen inside their leaf by
the same measured-NDCG search over the same pool, so all sixteen are selected by the criterion under
test."* (iii) Golbandi's missing K=16 cell is explained (3^k patterns fragment the cohort beyond depth
eight), the RecVAE zero cells are attributed to an input contract with no popularity floor, the instrument
claim is softened to a measured gap (*"indistinguishable on the tail … and 1.2% short on the full
profile"*), the free-text/442-questions passage is deleted, and a commensurability limitation is added:
*"that channel's advantage is answerability together with breadth, not answerability alone."* One
regression: Figure 1, the only figure, was deleted to make room. Restore it.

## Decision

**Weak accept. Confidence: high.**

*To the authors:* The revision closes the leakage and budget-accounting questions that could have sunk the
measurement and retires two over-claims, but it declines the abstract reframe I required; I am recommending
acceptance on the strength of Sections 5 and 6, and the abstract must be rewritten to match them.

**Did it move me? No — and not because the paper failed to improve.** It improved materially: I was at real
risk of dropping to Borderline had the leakage question come back unanswered or the K=16 interview turned
out not to be criterion-selected. The revision removed downside; it did not add upside. My weak accept was
already the level the substance supports, and the substance is unchanged — the same single-dataset,
single-instrument, +0.010–0.014-on-0.259 result, won by a comparator whose fitting advantage is now labelled
rather than repaired. Nothing that would have moved me up was attempted.

## Adjudicating the comparator-fairness objection

I expect Reviewer 1 to hold Weak reject. I concede the objection and still come down for acceptance.

1. It is correct about RQ1 and no longer disputed, but it is a *fitting* asymmetry: at scoring time every
arm meets the same unseen test users under the same budget and answer contract.
2. It does not reach RQ2 or RQ3. Table 4 *is* the symmetric experiment Reviewer 1 asks for — six criteria,
same tree code, branch rule, leaf size, pool, mask, cohort and budget, all fitted by the same search on the
same data — and only the measured metric puts the interview above the static order.
3. The tautology reading predicts the wrong thing. If "optimise X, win on X" were the mechanism, the
*measured* pairwise rank loss would not track NDCG@10 at +0.784 while its Taylor-approximated form
correlates at −0.149. The separating axis is measured-versus-approximated, which nothing definitional
predicts.
4. For the non-tree families the symmetric experiment is largely unavailable in principle: the bandits' free
surface is one constant, which was tuned; refitting DRE's or EDDI's *objective* on validation NDCG@10 does
not produce a fairer port, it produces CASPER.

Ruling: the objection downgrades RQ1 from headline to setup — precisely my required change 2 — and is a
condition on the camera-ready, not grounds to reject a paper whose load-bearing evidence sits in the two
sections it does not reach. I would ask Reviewer 1 whether Table 4 as the primary comparison changes their
score, and expect Reviewers 2 and 3 to move up.

## What would make this an unambiguous Accept

**Before camera-ready, and I would ask for it.** Re-run the bank comparison with item questions scored under
the *same* implicit-watch answerability predicate the concept channel uses, instead of "rated" — one
predicate changed in an existing pipeline, no new model, no regrown tree, a greedy re-search over the item
bank — which separates answerability from per-answer breadth. The paper's most actionable advice ("change
what you ask about, not how you decide") rests on an asymmetry now conceded in Limitations but not in the
claim. Either outcome is publishable: if most of +0.01430 survives, the advice is established; if not, the
honest number is the contribution. **Needs a follow-up paper:** a second dataset and question bank for the
criterion-regret table, which requires a second certified instrument.

## Over- and under-claiming as it now stands

*Over.* The abstract's "on four consumers" is contradicted by the body's own demotion of RecVAE; "every
published method loses" still travels without its asymmetry clause in abstract and conclusion; and the
abstract's answerability-mechanism sentence over-states relative to the new commensurability caveat.

*Under.* Unchanged and now conspicuous: the ranking-vs-rating null (+0.00020, the only null in the matrix),
the Taylor diagnosis, and the depth-shuffle control are still clauses inside long paragraphs, and the new
leakage guarantee is a real soundness result buried in one sentence.

## Acceptance probability at a venue taking the top 25%

**About 55%.** The measurement is one nobody else has assembled and the one-variable ablation is the
cleanest experiment in this literature; against that the delta is small, single-instrument, and attached to
a headline the authors have been asked three times to reframe and have not. Its fate turns on whether the
committee reads Table 4 or the title.
