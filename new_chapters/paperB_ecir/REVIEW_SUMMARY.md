# ECIR 2027 draft — review panel on v1.1, and what v2 did about it (2026-09-08)

Four independent reviews of `paperB_ecir` v1.1 (commit `34feb51`): three simulated ECIR reviewers of
increasing leniency and one supervisor/co-author read. Full texts in `reviews/`.

## Verdicts

| reviewer | recommendation | confidence | relevance | originality | soundness | presentation | reproducibility | impact |
|---|---|---|---|---|---|---|---|---|
| R1 strict methodology | **Weak reject** | 4 | 5 | 3 | 2 | 3 | 2 | 3 |
| R2 mainstream RecSys | **Borderline** (weak accept if baseline fidelity shown) | 4 | 5 | 3 | 3 | 3 | 2 | 3 |
| R3 practitioner | **Weak accept** | 4 | 5 | 4 | 3 | 3 | 2 | 4 |
| Supervisor / co-author | **Borderline now; weak accept after revision, pushable to accept** | — | | | | | | |

Consensus: relevance is not in doubt; the evidence base (23 arms on one contract, 10,000 held-out
users, one-variable ablation with shuffle controls) is unusually strong for ECIR; what costs the paper
is reproducibility and the fairness of the RQ1 comparison, plus presentation defects.

## What every reviewer raised

1. **Bibliography leaked working notes** into the printed PDF ("FULL AUTHOR LIST UNVERIFIED", "record was
   wrong twice", "not peer reviewed", the GreedyExtend provenance note). — **Fixed in v2** (four `note`
   fields removed; 41 entries, 0 leaked notes).
2. **Arm counts inconsistent**: "twenty-one published rules" (which included our own comparator), 23 rows,
   13 published methods, "every selector measured" in a caption that omits five. — **Fixed in v2**: abstract,
   contributions, RQ1 audit and conclusion now say thirteen published methods / twenty-three arms /
   twenty-one distinct rules, of which the one that optimises the metric is ours; Table 1's caption names
   the five omitted arms and states that separation counts are over all twenty-three.
3. **No tuning or porting protocol for the baselines** (R1, R2, R3, supervisor): the comparator is searched
   on the metric on the scoring recommender, the ported arms run as published. — **Addressed in v2** by a
   new "Ports and tuning" paragraph in Section 3 (published hyper-parameters; native-validation gate; which
   five methods have a native check; EDDI's gate failure with its numbers; Greedy SLIM not run; the
   exploration weight fixed, not tuned) and a limitation sentence stating the asymmetry and pointing to the
   ablation as the comparison that does not carry it. Not resolvable without new runs: tuning the ported
   arms on our data.
4. **Instrument unverifiable / single dataset** (all): the recommender's certification lives in an
   anonymous technical report. — Inherent to double-blind; v2 keeps the one-paragraph description. Not
   resolvable before submission except by de-anonymised preprint.
5. **CASPER's own transfer to a second consumer untested** (R1, supervisor): RQ1 was checked on four
   consumers and reverses on a fifth; the tree was not. — **Stated in v2 Limitations** with the RecVAE
   re-score named as first future work. Resolvable with one machine-night if the author wants it before
   5 October.
6. **Concept-answer statistic undefined**, so the concept-vs-item result could be an answerability
   artefact (R1, R2). — **Fixed in v2**: the SEL formula and the exposure/refusal rule are stated in Section 3,
   and concept questions are explained as answerable generalisations in the abstract, Section 3 and RQ3.
7. **Multiplicity** across 138 / 56 cells (R1, supervisor). — **Fixed in v2**: one sentence in the protocol
   paragraph, taken from the chapter (no family-wise correction; claims rest on directional consistency; the
   three single-interval claims were pre-specified).
8. **Unsupported "14 %" in the conclusion** (supervisor). — **Fixed in v2**: the supporting counts (1,785 answer
   neither opener, 1,438 of them not the third) are now in the "What it costs" paragraph.
9. **Readability** (R3, supervisor): "forgo 66–89 % of what a random choice forgoes" reads backwards;
   "not weak but aligned surrogates"; "What separates the family is the quantity". — **Rewritten in v2**
   ("recover only 11–34 % of the gain a random choice leaves unrealised"; "not merely weak, two of them
   point the wrong way"; the bandit sentence restated).
10. **Framing** (supervisor, R3): present two priced fixes rather than one named method plus an objection
    that beats it; the bank result was buried. — **Adopted in v2**: abstract and contribution 3 now say
    "two fixes, each priced on the same contract", CASPER first, bank change second with both numbers.
    CASPER keeps its name and the sentence on why (author instruction).
11. **EDDI native-validation paragraph lost to the cut** (supervisor). — Restored in compressed form inside
    "Ports and tuning".
12. **Cost of the method** (R3): 37,191 s per tree vs 11 s. — Already stated; kept.

## What was cut to pay for the additions (body is 12 pages again)
The genre-conditioned greedy paragraph (its +0.00862 survives nowhere in the paper now), the DRE-vs-entropy
sentence, the bit-identity sentences on the comparator and sequential loop, the finite-difference check
sentence, the root-node anecdote shortened, the RL related-work aside shortened, the dispersion headroom
sentence, the EDDI self-consistent-cell numbers, one limitation sentence on the dispersion criterion.

## Open items only the author can decide
- Run the RecVAE re-score of the tree before submission (one night; closes the strongest R1 objection).
- Whether to post a de-anonymised preprint of the instrument report so the certification is checkable.
- Whether to tune the ported bandits on validation users (a campaign, not a night); otherwise the
  asymmetry stays a stated limitation.
- Reproducibility track as fallback if the fidelity evidence is judged insufficient for the full track.
