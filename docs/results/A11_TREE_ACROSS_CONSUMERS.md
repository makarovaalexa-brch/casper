# A11 — the measured-metric tree consumed by other recommenders (2026-09-09)

**Status: COMPLETE (a re-consumption of banked TEST interviews; no new selection, no refit).** Design
`docs/design/DS_A11_TREE_ON_OTHER_CONSUMERS.md`. Code `src/baselines/adaptive_harness.py`
(`replay:` and `fixedorder_json:` specs), committed before launch (`3f160fc`). Artifacts
`experiments/baselines/interview/adaptive_table_a11_tree_consumers.json` + `_per_user.npz`,
`a11_paired_intervals.json`. Conclusions Fable's own.

**Why it exists.** The strict reviewer's sharpest point: CASPER's split criterion is the metric computed
through the same frozen recommender that then scores it, and the paper checked the RQ1 ordering on four
consumers but never checked its own method on any. So "put the reported metric in the split" could not be
distinguished from "align with one scorer".

**One-line verdict: the tree's advantage is not a property of the recommender that scored it. Replayed
verbatim through three other consumers, the tree beats the same static order on two of them at every budget,
and reverses on exactly one — Golbandi's leaf model, the same consumer on which the published selectors'
ordering reverses in the paper's own robustness check. Every interval separates.**

## The gate: PASSED
Replayed through the certified tower, both arms reproduce the chapter to five decimals at every budget:
tree 0.26936 / 0.13074 and static 0.25900 / 0.12370 at K=16, difference +0.01035 full / +0.00705 tail,
and +0.00528 / +0.00891 / +0.01062 full at K=2 / 4 / 8. The replay is the banked interview, not a re-run.

## Tree minus static order, full NDCG@10, 10,000 test users
Paired percentile bootstrap, 10,000 resamples clustered over users; `*` = interval excludes zero.

| consumer | K=2 | K=4 | K=8 | K=16 |
|---|---|---|---|---|
| **certified tower** (the paper's) | +0.00528* | +0.00891* | +0.01062* | **+0.01035*** [+0.00737, +0.01330] |
| **RecVAE** | +0.04281* | +0.03534* | +0.01034* | **+0.04122*** [+0.03709, +0.04531] |
| **conjugate accumulator** | +0.00049* | +0.00071* | +0.00128* | **+0.00161*** [+0.00101, +0.00220] |
| **Golbandi leaf model** | +0.01195* | +0.01118* | **−0.01375*** [−0.01649, −0.01094] | — |

Tail behaves the same way: positive and separating on the tower (+0.00705), RecVAE (+0.03554) and the
conjugate accumulator (+0.00938); negative and separating on the leaf model at K=8 (−0.01487). The tower's
K=2 tail is the single known exception in the chapter (−0.00262) and reproduces here.

## Reading, with the caveats attached
1. **Three of four consumers agree with the tower.** The claim "changing the split criterion to the measured
   metric beats the static order" survives a change of consumer, which is what the reviewer asked for.
2. **RecVAE's margin is larger but measured in a starved regime.** RecVAE ingests a binary liked-items vector
   only, and the battery runs with the popularity floor off, so a model given no answer is scored on what it
   actually predicts: RecVAE reads 0.0000 at K=1 and 0.1357 at K=16 against the tower's 0.2590. The direction
   is informative; the magnitude is not comparable across consumers and must not be quoted as though it were.
3. **The reversal is on the expected consumer.** Golbandi's leaf model is the one that pays in *unknowns*
   (it matches a user to train users by like/dislike/unknown pattern), and it is the consumer on which
   Thompson sampling beats the static order in the chapter's Section-1 robustness sweep. The same consumer
   reversing in both directions is instrument-dependence appearing twice, not an inconsistency.
4. **No K=16 row for the leaf model**: the harness skips it beyond depth 8 because 3^k answer patterns
   fragment the training cohort. That is a property of the published consumer, stated, not a gap in the run.

## Two consumers could not be run, and it is a checkpoint problem, not a choice
`tanp` and `eddi` both read their epoch off a cached checkpoint before fitting, and neither
`.cache/baselines/tanp_ml25m_liang.pt` nor `eddi_ml25m_liang.pt` exists on this box. Producing them is a
training run, not a re-score. **Report as unavailable; do not imply the set was chosen.**

## What goes in the paper
One sentence in RQ3 or Limitations, replacing "CASPER's own transfer to a second consumer is not measured
here": replayed through three further consumers, the tree's advantage holds on RecVAE (+0.041) and the
conjugate accumulator (+0.002) and reverses on Golbandi's leaf model (−0.014 at K=8), the same consumer on
which the published selectors' ordering reverses, with the starved-regime caveat for RecVAE.
