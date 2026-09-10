# DS-A17 — porting the instrument to Goodreads: arena, split, concepts, tower, gates

**Status: IN PROGRESS.** Author instruction, verbatim: *"commit everything we have, do not edit the
paper or the chapter, and replicate the process exactly without corner cutting. if you start getting
netaive results stop and tell me. Do not replicate chapter A full battery, just the instrument itself
and its gates"*.

## Why Goodreads, after two rejected candidates

| candidate | verdict | evidence |
|---|---|---|
| goodbooks-10k | **rejected on citability** — a GitHub/Kaggle release with no paper | passed on numbers: concept:item answer ratio 4.09 vs ML-25M's 3.98 |
| Goodreads UCSD, genre subsets | **rejected on depth** | poetry subset: median **2** ratings per user, because a genre slice cuts each history to one corner |
| Netflix Prize | **rejected on provenance** | formally withdrawn after a privacy lawsuit and FTC investigation; no licensed distribution; behaviour from 1998--2005 |
| Amazon 2023 / Yelp | **rejected on depth** | 1.6--2.7 reviews per user; an interview collects no answers |
| **Goodreads UCSD, full book graph** | **SELECTED** | 228,648,342 interactions, 104,551,549 rated, 816,371 users, mean 128.1 ratings each |

## The two calibrated reductions, both author-approved and both stated

Neither is chosen for convenience; both are set by *matching the reference dataset*, which is what
makes the replication comparable rather than merely also-true.

1. **Arena size.** The 18,430 most-rated books, matching ML-25M's 18,430-film arena. Matching
   ML-25M's *rule* (>= 20 ratings) leaves 434,531 books, untrainable here; matching its *coverage*
   (99.2% of rating events) needs >1.1M. Size is the only one that fits. **Keeps 56.7% of rating
   events** — Goodreads is a raw crawl with a far longer tail than a curated MovieLens catalogue, and
   that asymmetry goes in the write-up.
2. **Cohort size.** The raw file is emitted for 160,768 sampled users so `liang_split` yields
   140,768 / 10,000 / 10,000. Using all 683,636 eligible users would give a 4.7x larger train cohort
   — a better-resourced instrument, not a fairer one, and any cross-dataset difference would be
   uninterpretable.

## The chain, and why it is the project's own scripts

**The decisive design fact**: `train_tower_t2.py` does not read the split alone. It **re-derives the
user partition from the raw ratings file** so graded half-star profiles align with binarised likes
(`reproduce_partition`; catalogue assertion at `:423-427`). A hand-rolled split therefore drifts from
what the pipeline expects however faithful its recipe looks. My first attempt (`a18_goodreads_split.py`)
did exactly that and is **superseded**. The corrected chain emits Goodreads in MovieLens *raw* shape
and runs the project's own scripts unmodified.

| # | stage | script | status |
|---|---|---|---|
| 1 | arena, graded store, shelf vocabulary | `a17_goodreads_arena.py` | DONE — 18,430 books, 59.3M rated events, 974,142 shelf rows |
| 2 | concept bank | `a19_goodreads_concepts.py` | DONE — 2,408 concepts, per-book density median **37** vs ML-25M's 37 |
| 3 | raw-shape emission | `a21_goodreads_rawform.py` | DONE — `ratings.csv` + `genome-scores.csv` |
| 4 | the Liang split | **`scripts/baselines/liang_split.py --data goodreads`** | DONE — 18,430 items, 140,768/10,000/10,000, catalogue assertion passed |
| 5 | RecVAE teacher | `a20_goodreads_recvae.py` -> canonical `src/baselines/recvae.py` | RUNNING |
| 6 | i26 tower | `src/instrument/train_i26.py` under `CASPER_DATASET=goodreads` | pending |
| 7 | concept channel | `concept_train_v2.py` | pending |
| 8 | **the gates** | `run_battery_phaseA.py` + `concept_identity_lift.py` | pending |

## Adaptations, each stated rather than silent

- **Rating scale.** Goodreads integers 1--5 are left unchanged. Liang's `rating > 3.5` means `>= 4`
  on both scales. The tower's level table is 10-wide for half-stars; Goodreads occupies levels
  1,3,5,7,9 of that same table and `LIKE_MIN_LEVEL = 7` still means star >= 4. The like boundary is
  preserved; the even levels are simply never observed.
- **Concept relevance.** 1.0 per membership pair. The channel thresholds at >= 0.5 then takes
  >= 30 members, so binary membership is exactly what it consumes.
- **Concept junk filter.** Format ("e-book", "audible", "nook", "calibre"), structural
  ("part-of-a-series"), process ("didn-t-finish", "recommended") and list-membership ("1001-books")
  shelves are dropped. These are highly answerable and carry no taste signal — the way a concept
  channel's answer rate gets inflated with vacuous questions. Dropping them is **conservative for our
  own claim**: it can only lower the measured concept answer rate.
- **`warm_lr_scale`.** The i26c watchdog's comment records `0.01`; the certified run's own training
  log records `0.1` at every resume. **The log is primary evidence and is what we follow.**
- **Dataset switch.** `CASPER_DATASET` selects five constants in `train_tower_t2.py`, defaulting to
  `ml-25m`. Verified both ways: unset gives byte-identical ML-25M paths.

## The gates — what "the instrument and its gates" means here

Reproduce the **published certificate**, not the superseded spec. The chapter's table retired the
G-names: **G0 -> R1, G5 -> R2, G9 -> R3, G3a+G3b -> R4, G6 -> Controls, C2 -> Provenance.**

- `python src/instrument/run_battery_phaseA.py --snapshot <tower.pt>` runs **g3a, g3b, g6, g9, g5**
  in that order (`ORDER` at `:789`). This is the whole of it.
- **Phase B is skipped.** G1, G4, G8 and G2-formal are **moot for the certified pair**: the
  belief/covariance layer was deleted by author ruling 2026-08-02, so there is no Sigma in the
  shipped fold. G10 was never run.
- **G6 must run with `--force`.** Its cached ML-25M JSON is stale in both value and schema — it
  predates the current one-sided rule (`run_battery_phaseA.py:499-506`).
- **R2 needs a fresh `concept_identity_lift.py` run** against the actually-certified concept fold.
  The published R2 row was measured on `cd_s1_l10_best.pt` over 905 users, six days before the
  certified checkpoint existed, and must not be inherited.
- **C-lite is excluded**: it ran on the i25 ep4 tower, not the certified pair.

## Acceptance and STOP conditions, registered before the numbers

The author's instruction is to **stop and report on negative results**, not to push through. Stop if:

1. **RecVAE fails to reach a sensible level for the dataset.** A weak consumer makes every downstream
   elicitation number uninterpretable. There is no published Goodreads bar under this protocol, so
   the bar is dataset-local; ML-25M's RecVAE reached 0.3540/0.2497 test.
2. **The tower fails R1 (strength parity)** — it must not be materially below its own RecVAE teacher.
3. **Any gate in the Phase-A battery fails** on the certified snapshot.
4. **The empty-interview identity breaks**, or any leak assertion trips.
5. **Concept answerability collapses** relative to the probe's prediction.

## Hard rules in force

HARD RULE 1: two reductions, both stated above, both approved. HARD RULE 3: gates listed here before
the tower runs. HARD RULE 10: every script committed before launch. HARD RULE 11: RecVAE checkpoints
per epoch and resumes; one heavy job at a time; free RAM checked at each launch.
**No paper or chapter file is touched by this work.**
