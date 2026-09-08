# ECIR 2027 submission — to-do (author-ordered, 2026-09-08)

Deadlines: **abstract 21 Sep 2026**, **paper 5 Oct 2026** (11:59pm GMT), EasyChair. Body ≤ 12 pp.

## 1. Requests-section arms on TEST (author ruling: "we must do it now")
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

## 2. Exploration-weight sweep for the ported bandits (closes the strict reviewer's "asymmetry" point)
- GLM-UCB / Thompson / ConUCB exploration weight: 3–5 values, chosen on **validation** full NDCG@10,
  best value re-run on test under the contract. Everything else as published.
- Half a day of compute per arm at 10,000 users; one afternoon total if run sequentially. Check free RAM;
  never beside another heavy job.
- Paper: one sentence in "Ports and tuning" and one row-note in Table 1.

## 3. The measured-metric tree on other consumers (closes "CASPER never tested for transfer")
- Re-score the tree's **banked test interviews** through the RQ1 robustness consumers: RecVAE (likes only,
  concept answers dropped, conservative), TaNP, Golbandi leaf model. No refitting.
- Cost: one machine-night; the RQ1 consumer-sweep harness exists.
- Paper: one sentence in RQ3 or Limitations with the three numbers.

## 4. Code link — DONE in v2 footnote: `https://github.com/makarovaalexa-brch/casper`
- **Before submission**: replace with an anonymised mirror (anonymous.4open.science or a fresh account); a
  personal GitHub URL breaks double-blind. Recorded in `docs/STATE.md`.
- Before publication: clean the repo (see STATE note).

## 5. Closed / decided
- Instrument verifiability: code link suffices; a preprint of the instrument report does not block TORS
  (ACM accepts prior arXiv preprints). Optional.
- Single dataset: keep; add the one-sentence justification (ML-25M is the only MovieLens release with both
  ratings and the Tag Genome) to the setup paragraph. **Small edit, pending.**
- Baseline provenance, tuning statement, bank result placement, arm counts, bibliography notes, 14 %,
  multiplicity, readability items: done in v2.

## 6. Before submitting
- Abstract into EasyChair by 21 Sep.
- Final read for anonymity ("CASPER" not publicly attached to the author before notification).
- Re-run the numeric cross-check script against the chapter after every edit; page budget ≤ 12.
