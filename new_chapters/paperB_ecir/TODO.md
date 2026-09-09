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
