# Panel on v7 — fresh, PDF only, repository stipulated real (2026-09-09)

**Conditions.** Four independent Opus reviewers, each given **only** `submission_v7.pdf`, in a
directory isolated from the repository. No reviewer saw an earlier version, an earlier review, or any
indication that revisions had happened. Each was told by the "programme chairs" that the artefact
link in the footnote is real, resolves, and is complete, but that the review environment has no
internet access, so the paper must be judged as if the artefact had been inspected and found
complete, and its unavailability must not affect any score in either direction.

## Verdicts

| reviewer | recommendation | confidence |
|---|---|---|
| R1, methodology hawk | **Borderline** — states weak accept on answers to two questions | 4 |
| R2, mainstream ECIR/SIGIR | **Weak accept** | 4 |
| R3, industry practitioner | **Weak accept** | 4 |
| Area chair | **Weak accept** | 4 |

**Three weak accepts and one borderline that is one rebuttal from weak accept.** Area chair puts
acceptance at **55%** at a venue taking the top quarter.

## Scores

| criterion | R1 | R2 | R3 | AC |
|---|---|---|---|---|
| Relevance | 5 | 5 | 5 | 5 |
| Originality | 4 | 4 | 4 | 4 |
| Technical soundness | 3 | 3 | 4 | 4 |
| Quality of presentation | 4 | 3 | 3 | 3 |
| **Reproducibility** | **4** | **4** | **4** | **4** |
| Potential impact | 3 | 4 | 3 | 4 |

Reproducibility is unanimous at 4 and no longer the constraint. Every reviewer withheld the fifth
point for the same reason: one dataset, one split, one seed, one instrument.

## What three or four of them independently say

1. **RQ2 is scored off-trajectory** (R1, R2, AC — the AC's first required change, and R2's most severe
   weakness). The published surrogates are evaluated only at the 22 nodes of the tree grown by *our*
   criterion, so the cleanest diagnostic in the paper carries a selection effect. **The fix they want
   is the diagonal: grow each criterion's own tree and score each at its own nodes.** The AC will
   accept an explicit flag instead of the run, but the run is what removes the objection.
2. **The criterion is measured through the recommender that then grades it**, and the transfer
   evidence is partial: RecVAE direction-only, the accumulator a sixth of the headline, the leaf
   model *reversing* at K=8 (R1, R2, R3). R1 specifically objects that reading the reversal as "an
   argument for certifying the consumer" converts a failed generalisation test into a recommendation.
3. **"Thirteen published methods, every one loses" outruns the ports** (R1, R2, AC): five have native
   gates, EDDI fails its gate and runs reduced, Greedy SLIM — the paper's own "closest protocol
   sibling" — is not run, and the abstract carries none of that.
4. **The instrument is documented only through anonymous reference [3]** (R1, AC required change 2).
   Both want a self-contained half-page with the four certification requirements and the parity
   numbers, so no claim rests on an unreviewable report.
5. **The fitting asymmetry belongs in the abstract**, with Table 4 named there as the load-bearing
   comparison (AC required change 3). Also: state the leaf-model reversal wherever the consumer claim
   is made, including the abstract (AC required change 4).
6. **The K=16 headline is five conditioned questions plus a leaf-conditioned tail** (R1, R2). R1
   sharpens it: fifteen leaf-specific validation-fitted tails against one global order is not
   capacity-matched, and the 2.7× validation overstatement and 6-of-15 losing leaves are its
   signature.

## Where they disagree, and it matters

**R3 (practitioner) splits the two contributions and would ship only one.** The tree: no — about 4%
relative NDCG@10 at a sixteen-question budget nobody ships, negative on the tail at K=2, a coin flip
for two users in five, ten hours to build and repaid at every retrain. The concept bank: yes —
first-two-question refusals falling from 5,779 in 10,000 to 11 in 10,000 is a real and expensive
funnel problem. R3 also notes the paper's own consumer result points away from shipping the selector.

**R3's first weakness is new and not raised by the others:** the concept answer is a behavioural
statistic that a live onboarding UI cannot reproduce, since a real user self-reports and is subject
to acquiescence bias. The paper concedes the gap is unmeasured, but it is load-bearing for the
paper's strongest practical advice. R3's suggested fix is cheap and does not need people: **a
sensitivity analysis degrading concept answers with plausible noise, showing where +0.01430 goes to
zero.**

## Area chair's required changes, verbatim in substance

1. Recompute Table 2 at nodes each criterion's own tree reaches, **or** explicitly flag the
   off-trajectory scoring.
2. Make the instrument self-contained in an appendix: four certification requirements plus parity.
3. Move the comparator's fitting asymmetry into the abstract and name Table 4 as load-bearing.
4. State the Golbandi-leaf reversal wherever the consumer claim is made, not only in body text.
5. Name the three pre-registered cells and state the no-family-wise-correction position beside them.
6. Supply the anonymised artefact mirror per the footnote.

**Advised, not required:** a figure for Table 4; a subsection on the 37,191 s versus 11 s build cost;
a gate-outcome table separating the thirteen methods into five natively validated, one reduced, one
not run; a second seed for Table 3 if cheap.

**Unambiguous accept** would need Tables 2–4 replicated on a second dataset with a second certified
recommender. The AC calls that a follow-up paper, not a camera-ready fix.

## Reading

The panel has converged. Every reviewer is now at or above borderline, reproducibility is settled,
and the remaining objections are a short list that three or four of them share. Items 3, 4, 5 of the
AC's list plus the abstract changes are **writing**, doable in an afternoon. Item 1 is the only one
that is a **run**, and it is the one the panel most wants. Item 2 is writing but costs page budget,
which is already at the 12-page limit.
