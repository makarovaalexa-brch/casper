# ECIR 2027 submission — to-do (author-ordered, 2026-09-08)

Deadlines: **abstract 21 Sep 2026**, **paper 5 Oct 2026** (11:59pm GMT), EasyChair. Body ≤ 12 pp.

## 1. Requests-section arms on TEST — DONE 2026-09-08 (`docs/results/A9_OPENER_ARMS_ON_TEST.md`; pooled opener − cold +0.01041 [+0.00583, +0.01520]* on test; gate reproduced the 29 Aug file to 0). Chapter `tab:volunteered` still to be moved to test on the author's instruction.
State of the artifacts, checked against `chapterB_v2.tex`:
- **Already on test** (`tab:policy`, one look, 29 Aug): policy, cold tree, static order for the whole mix
  and the volunteered / not-volunteered splits, K=15; and per group (b): routing, policy, **own opener tree**,
  cold, policy − cold with intervals.
- **Still validation-only** (`tab:volunteered`): per-group static, and the **opener − cold paired interval**
  per group and pooled (static < cold < opener).
- So the job is a **re-score of banked test interviews**, not a refit: compute per-group static on test and
  the opener − cold and cold − static paired bootstrap contrasts (10,000 resamples, seed 98765) from the
  test scores already produced for `tab:policy`. Same test look, reused not refitted (rule: TEST material
  is reused, never refitted). Then `tab:volunteered` moves to test in the chapter.
- Design sheet before running (one page: inputs = banked test per-user score vectors; outputs = the two
  contrast columns; gate = pooled policy − cold reproduces +.00343 exactly). HARD RULE 10: commit first.
- Not on the ECIR paper's critical path (the requests section is excluded from this paper), but it is on
  the chapter's.

## 2. Bandit constants — DONE 2026-09-09 (`docs/results/A10_BANDIT_TUNING.md`; chosen on validation, interior optima at alpha 4.0 and scale 0.5; worth +0.0058/+0.0018 against a 0.037-0.044 deficit; in the paper's ports paragraph, RQ1 and Limitations)
- GLM-UCB / Thompson / ConUCB exploration weight: 3–5 values, chosen on **validation** full NDCG@10,
  best value re-run on test under the contract. Everything else as published.
- Half a day of compute per arm at 10,000 users; one afternoon total if run sequentially. Check free RAM;
  never beside another heavy job.
- Paper: one sentence in "Ports and tuning" and one row-note in Table 1.

## 3. Tree on other consumers — DONE 2026-09-09 (`docs/results/A11_TREE_ACROSS_CONSUMERS.md`; holds on 3 of 4, reverses on Golbandi's leaf model; absolutes now in the chapter as tab:consumers and in the paper as Table 5)
- Re-score the tree's **banked test interviews** through the RQ1 robustness consumers: RecVAE (likes only,
  concept answers dropped, conservative), TaNP, Golbandi leaf model. No refitting.
- Cost: one machine-night; the RQ1 consumer-sweep harness exists.
- Paper: one sentence in RQ3 or Limitations with the three numbers.

## 4. Code link — v5 carries a withheld-URL footnote (v2's personal GitHub URL removed in v4)
- **Before submission**: replace the footnote with an anonymised mirror (anonymous.4open.science or a fresh account); a
  personal GitHub URL breaks double-blind. Recorded in `docs/STATE.md`.
- Before publication: clean the repo (see STATE note).

## 5. Closed / decided
- Instrument verifiability: code link suffices; a preprint of the instrument report does not block TORS
  (ACM accepts prior arXiv preprints). Optional.
- Single dataset: keep. Justification in the setup paragraph, corrected in v4 after the author caught the
  false claim: ML-20M and ml-latest also carry the Tag Genome, so the paper says ML-25M is the most recent
  *stable* MovieLens release carrying both ratings and the Tag Genome. **Done.**
- Baseline provenance, tuning statement, bank result placement, arm counts, bibliography notes, 14 %,
  multiplicity, readability items: done in v2.

## 6. Before submitting
- Abstract into EasyChair by 21 Sep.
- Final read for anonymity ("CASPER" not publicly attached to the author before notification).
- Re-run the numeric cross-check script against the chapter after every edit; page budget ≤ 12.

## 7. Delta review, 2026-09-09 (`reviews_v4/review_A2_delta.md`, `review_D2_final.md`)
Both reviewers were given v4 and v5 together and instructed to treat the anonymised repository as
present and complete, so reproducibility was scored on the paper's own description.
- Strict reviewer: **Weak reject -> Borderline**. Technical soundness 2 -> 3, reproducibility 2 -> 4.
- Area chair: **Weak accept, unmoved** — "my weak accept was already the level the substance supports".
  Acceptance probability at a 25%-acceptance venue: about 55%.
- Both asked for the same two things, done in v6: reframe the abstract so CASPER is the claim, and
  restore Figure 1.

### Done since
**1. Answerability versus breadth — DONE 2026-09-09** (`docs/results/A12_ANSWERABILITY_VS_BREADTH.md`).
The control as the reviewers described it is **not constructible**: ML-25M ships no watch log, so
"watched" and "rated" are the same event for a film and the item predicate is already the widest
available. Run in its place: lift factorises into answers x lift-per-answer. A film answer is worth
3.98x a concept answer and concepts collect 3.98x as many, which is why the single channels tie at
K=16. At matched answers the film channel wins every comparison (+0.02734* at 1.79 vs 1.75). So
breadth runs the WRONG way and the gain is the answer rate. In the paper as v7; the v5 concession
that contradicted it is withdrawn.

### Still open
1. ~~**Answerability versus breadth.**~~ DONE, see above. Original brief kept for the record: Re-run the bank comparison with item questions scored under the
   concept channel's implicit-watch answerability predicate instead of "rated", and report how much of
   the $+0.01430$ survives. One predicate changed, existing pipeline, no new model, no regrown tree.
   The chair would ask for this before camera-ready; the strict reviewer names it as one of two
   conditions for moving past borderline. **Either outcome is publishable.**
2. **A symmetric fit for the published selectors.** Refit DRE, one bandit and IGCN on the validation
   cohort against full NDCG@10 over the same bank, and report whether RQ1 survives. The strict
   reviewer's other condition. The chair argues it is unavailable in principle for the non-tree
   families, since refitting their objective on validation NDCG@10 produces CASPER rather than a
   fairer port — so this is a judgement call, not an obligation.
3. Second dataset: both agree it is a follow-up paper, not a revision.

## 8. Criterion-regret DIAGONAL — PARKED, design sheet `docs/design/DS_A13_CRITERION_REGRET_DIAGONAL.md`
The v7 panel's most-shared objection and the area chair's first **required** change. Table 2 scores
the published surrogates only at the 22 nodes of the tree grown by our own criterion, so they are
judged at states they did not choose. The diagonal grows each criterion its own tree (**already
banked from the Table 4 ablation**) and computes the exact measured gains at *its* nodes.

- **v8 already carries the fallback the chair explicitly permits**: the table and caption now say the
  numbers are off-trajectory and point at Table 4. Disclosure closed; objection not.
- **Cost unbenched.** Naive extrapolation from the 37,191 s tree build gives ~31 h for three
  criteria, but that figure includes the search and CELF heap this job does not need. **Bench one
  node first**, then decide between three criteria and two.
- ⚠ Nothing to do with anchor or cluster teachers. Cold Golbandi ternary interview trees on the
  certified tower, the same primitives the Table 4 ablation imports.
- **Not a blocker for submission.** Improvement only, and must not delay 21 Sep / 5 Oct.

## 9. Second dataset — follow-up paper, NOT a camera-ready fix
The area chair's answer to "what would make this an unambiguous accept" was Tables 2--4 replicated on
a second dataset with a second certified recommender. Costing in
`docs/design/DS_A14_SECOND_DATASET_COST.md`: **~62 h of recorded compute, two to four weeks elapsed**
on this box, and the chair called it a follow-up paper themselves.

**Goodreads is the candidate.** The first version of that costing wrongly said the Tag Genome has no
substitute. It does. The concept channel consumes a **binary membership matrix**, not graded
relevance (`belief_layer.py:632` thresholds at 0.5; `strategy_channel_suite.py:1422` builds with
`np.ones`), so any mined folksonomy is a drop-in. `scripts/_archive/paper2_old/gr_books.py` already
mines Goodreads `popular_shelves` into exactly that shape, with a >=30-item floor mirroring ML-25M,
and `scripts/instrument2/gr_recvae.py` already trained a Goodreads RecVAE reading **0.4340/0.2507**
full/tail against a 0.2045/0.0090 most-popular floor -- the certification bar, already measured.

⚠ Before anything else: the old Goodreads arena was truncated to **20,000 of 188,867 items and 418
test users**. Neither may be inherited under HARD RULE 1, and inheriting them would undercut the
catalogue-scale claim that is one of the paper's selling points. The data is also no longer on the
box and the code speaks the retired pre-Jul-22 contract.
