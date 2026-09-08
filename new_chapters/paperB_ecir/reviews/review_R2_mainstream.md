# ECIR 2027 — Full Paper Review (Reviewer 2)

**Paper:** Objective Mismatch in Adaptive Preference Elicitation for Cold-Start Recommendation

## Summary

The paper asks whether adaptive question selection for cold-start users pays under the metric the field
now reports. The authors define one "measurement contract" — the Liang strong-generalisation split of
ML-25M (140,768/10,000/10,000 users, 18,359 items), answers read from users' own held-out ratings,
unanswerable questions charged as forfeited turns, unsampled full-catalogue ranking, and a frozen
in-house two-tower set encoder as consumer — and re-implement 21 published selection rules inside it.
Their comparator is a greedy prefix-optimal *static* order built by directly maximising held-out NDCG@10
on the same recommender. Every published selector loses to it at every budget K=1..16 on 10,000 test
users, and not for want of branching (RQ1). RQ2 diagnoses the cause: at the 22 nodes of a grown interview
tree, three published split criteria correlate at +0.010, −0.060 and −0.104 with exactly measured
NDCG@10 gains, forgoing 66–89% of what a random pick forgoes. RQ3 puts the measured metric into
Golbandi's ternary tree ("CASPER"), everything else fixed, and the sign flips: +0.01035 full / +0.00705
tail at K=16, backed by a six-arm one-variable ablation and two shuffle controls. Separately, enlarging
the bank with Tag-Genome concept questions, with no conditioning at all, buys more (+0.01430 full).

## Scores

| Criterion | Score | Justification |
|---|---|---|
| Relevance | 5 | Cold-start elicitation and evaluation validity are core ECIR topics. |
| Originality | 3 | Sepliarskaia (RecSys'18) already reported static beating dynamic for this reason; novelty is scale plus the metric-in-the-split fix. |
| Technical soundness | 3 | Controls and ablation are careful, but all rests on 21 untuned re-implementations and an unpublished recommender. |
| Quality of presentation | 3 | Table 1 barely legible, no datasets table, bibliography carries internal editorial notes. |
| Reproducibility | 2 | Instrument is an anonymous "under review" report [3]; no tuning or seeds statement. |
| Potential impact | 3 | Useful corrective, limited by one dataset and one instrument. |

**Overall recommendation: Borderline** (leaning weak accept if baseline fidelity is established).
**Confidence: 4.**

## Strengths

- Important question, honestly framed: the paper explicitly refuses "adaptivity does not pay" and argues
  the narrower objective-mismatch claim (§1). Replication scale (21 rules, 10,000 test users, unsampled
  catalogue, Krichene & Rendle cited for why) exceeds the norm in this literature.
- §5 measures rather than asserts misalignment, with per-node ground truth and a random-pick control. The
  root-node exception — the logistic criterion picks the metric's argmax at the root, then walks the
  interview downhill — is genuinely illuminating.
- §6's ablation is exemplary: one variable changed, same tree code and pool, a min-leaf-respecting
  shuffle control, and an exactness gate (K=1 difference exactly 0.0 for all 20,000 users).
- Self-damaging results are reported, not buried: validation overstates the gain ~2.7×, 6/15 leaves lose
  on test, the routing test is a null, and "bank beats conditioning" undercuts the authors' own method.

## Weaknesses (most serious first)

1. **No tuning protocol for any baseline (§3–§4).** The headline is that 21 published methods all lose;
   that claim is only as strong as the re-implementations. No hyper-parameter search, budget, ranges or
   selection metric is stated for any ported arm — well short of the ECIR norm of tuning all baselines to
   their best. "Validated natively and ported" (Table 1) and the "† failing native validation" marker are
   the only provenance offered.
2. **The instrument is unavailable (§3, ref. [3]).** Every number comes from a frozen in-house encoder
   documented only in an anonymous report under review. Its certification cannot be checked, and 1.2%
   short of an *in-house* RecVAE bar is not parity with a published reproducible number. The main table
   should be reproduced on a public consumer (RecVAE/EASE) as a primary result, not a robustness aside —
   especially since §4 already reports one consumer (Golbandi's leaf model) that *reverses* the verdict.
3. **Single dataset, no datasets table (§3).** ML-25M only, split sizes in prose. The generality of "the
   published criteria are uninformative" is untested outside MovieLens.
4. **The answerability rule may be doing much of the work (§3, §7).** "Rated ⇒ answerable" is MNAR by
   construction: selectors that pick informative but obscure items forfeit a turn precisely because few
   users rated them — which is the paper's own stated mechanism. No sensitivity analysis (rated-or-popular
   rule, or a coverage-restricted bank applied equally to all arms) is run.
5. **Concepts vs items is not a matched per-question comparison (§3, §6).** A concept answer aggregates
   the user's whole history; an item answer is one rating. The +0.01430 result may reflect information per
   question rather than answerability, yet is attributed wholly to answerability (3.11 vs 12.36 answers).
6. **Counting inconsistencies.** The abstract says "twenty-one published selection rules", but §4 says one
   of the twenty-one optimises the reported metric "and it is the static greedy order" — the authors' own
   arm. Elsewhere, "twelve of the twenty-three arms". Table 1 lists 17 arms plus an oracle and omits "five
   further degenerate or duplicate arms… for space"; a replication table should not omit arms.
7. **A conditioning result sits under RQ1 (§4, "One arm clears the bar").** The genre-routing experiment
   answers RQ3, has an acknowledged null on 54.8% of the cohort (Drama), and charges nothing for the genre
   answer, so it is an upper bound presented next to real numbers.

## Detailed comments by section

**Abstract / §1.** RQs are enumerated and each gets an explicit answer sentence — good. RQ3 bundles two
distinct questions (metric-in-split; bank vs conditioning); split them. Correct the "twenty-one published
rules" count.

**§2.** Well informed and appropriately blunt about [32] and [29]. But it is a wall of dense prose, and the
key gap sentence ("Unoccupied is the conjunction…") is a 60-word list of conditions a non-specialist ECIR
reader will not parse.

**§3.** As the setup section this misses several community norms: no datasets table; no tuning statement;
no seeds/repetitions statement (are the bandit arms single deterministic runs?). The bootstrap procedure
is well specified and used consistently — good. Terms used before definition: "protocol pool", "fold-in",
"unseen-exposure token", "answerability-ranked", "whitened centroid". Fig. 1 largely restates the text;
that space would be better spent on the datasets table.

**§4.** The "not a failure to branch" evidence (Thompson runs a distinct interview per user from Q4 on and
still loses) is convincing. Table 1 is very dense — two metrics per cell, slash-separated, at 7–8pt — and
cannot be scanned down a single metric; consider separate full/tail column blocks. The em-dash for the
Golbandi tree at K=16 is unexplained. RQ1 is answered, for the arms as implemented.

**§5.** Strongest section, two caveats. The dispersion analysis uses V = the recommender's own predicted
top-10 mass, i.e. the model's belief about its own gain; the paper flags this as necessary-not-sufficient,
then uses "effective 5.04 distinct optimal questions" rhetorically as measured headroom. And the "4,431
users placed by a real free-text request" arrives with no provenance — who collected these, and are those
users inside the test split? Add a sentence in §3 or drop it.

**§6.** Well executed; Tables 3–4 are readable and the K=2 tail regression is reported rather than hidden.
But the comparator is unvalidated against any published number (§7), and rebuilding it under the scoring
contract already cut the gain from +0.01371 to +0.01035 full and +0.02092 to +0.00705 tail — the headline
effect is the same order as the authors' own last comparator-strength fix. With 6/15 leaves losing on test
and validation overstating ~2.7×, +0.01035 is a small effect resting on a comparator with no external
anchor. The bank-vs-conditioning paragraph deserves its own RQ.

**§7–§8.** Candid, specific limitations; the conclusion repeats the numbers as it should.

## Questions for the authors

1. For each ported selector: what search, ranges, budget and selection metric were used for tuning?
2. What native-validation target did each ported method have to hit, which arms failed, and what were the
   five omitted arms?
3. Can RQ1 be reproduced with a fully public consumer (RecVAE or EASE) as the primary instrument?
4. How sensitive is RQ1 to the answerability rule (e.g. rated-or-popular, or a coverage-restricted bank
   applied identically to all arms)?
5. Can you report gain per *answered* question, to separate concept answerability from the larger
   information content of an aggregated concept answer?
6. Where do the 4,431 free-text requests in §5 come from, and are those users disjoint from test?
7. Why is there no K=16 entry for the Golbandi tree in Table 1?
8. Any evidence outside MovieLens that the published criteria are similarly uninformative?

## Minor issues

- **The bibliography contains internal working notes that must not appear in a submission**: [9] "Full
  text closed on every route tried…", [14] "FULL AUTHOR LIST UNVERIFIED – replace the and others before
  print", [27] "The earlier [S] record was wrong twice…". Also [14] uses "et al." rather than the full
  LNCS author list.
- Reconcile "twenty-one" vs "twenty-three" arm counts and use one consistently.
- Fig. 2: eight overlapping series on a log axis; add per-series markers or move two arms to an appendix.
- State once whether any multiple-comparison adjustment was applied across the 138 cells.

## What would change my recommendation

I would move to **weak accept / accept** if the authors (a) document a real tuning protocol and a native
validation target for every ported selector, so "every published method loses" is a claim about the
methods rather than the re-implementations; (b) reproduce RQ1 on at least one fully public recommender as
a primary result, so nothing depends on the unavailable instrument [3]; and (c) add a sensitivity
analysis on the answerability rule. I would move to **weak reject** if tuning proves minimal or absent and
the ported implementations cannot be shown to reproduce their sources' behaviour on any anchor — the
central negative claim would then not be supported at the strength it is stated. A second dataset would
raise impact but is not, for me, a condition of acceptance given the depth of this study.
