# REPO AUDIT — 2026-07-20 (adversarial)

> Hard-nosed audit of the post-reorg repo. Skeptical by remit. Findings ranked
> blocking / should-fix / nit within each of the 6 categories.

## Verdict (one paragraph)
The reorganization is **structurally sound but not clean-finished**. The north-star docs
(`CLAUDE.md`, `VISION.md`, `STATE.md`, `EXPERIMENTS.md`, `README.md`, `PUBLICATION_PLAN.md`)
are internally consistent, their pointers almost all resolve, `docs/reference`↔`archive`↔`design`
have no duplicate basenames, and — importantly — **all 65 live scripts import cleanly**: nothing
references the removed `src/casper` package or any archived module. So the core is trustworthy.
But the reorg left a trail of loose ends: one broken source pointer in `EXPERIMENTS.md`; an
**out-of-repo `MEMORY.md` (which auto-loads every session) still pointing at ~7 pre-reorg paths and
several relocated/absent memory notes**; a stale `setup.py` and an entirely undocumented, orphaned
`project_docs/` directory full of V0-era junk; and **five cache files tracked in git despite matching
`.gitignore`**. None of it blocks work, but a fresh reader (or a fresh session bootstrapping from
`MEMORY.md`) will be misdirected. Grade: **B-** — good bones, sloppy edges.

Counts: **blocking 0 · should-fix 7 · nit 5**.

---

## 1. Dangling / broken references

**[should-fix] `EXPERIMENTS.md` cites a source that doesn't exist at the stated path.**
- Where: `docs/EXPERIMENTS.md:16` — RS-validity row, source `archive/POLICY_LOG.md`.
- Why: `POLICY_LOG.md` lives in `docs/reference/POLICY_LOG.md`, NOT `docs/archive/`. Two later rows
  (lines 41, 45) cite it correctly as `reference/POLICY_LOG.md`, so the file is inconsistently
  pointed to within the same doc. Verified: `docs/archive/POLICY_LOG.md` does not exist; it is the
  ONLY broken source pointer of the 31 in the file (all script refs and the other 30 sources resolve).
- Fix: change `archive/POLICY_LOG.md` → `reference/POLICY_LOG.md`.

**[should-fix] `MEMORY.md` (auto-loads every session) points at ~7 pre-reorg repo paths.**
- Where: `C:\Users\Tech\.claude\projects\C--dev-phd\memory\MEMORY.md` (outside the repo, but explicitly
  in scope — it is the session bootstrap). Confirmed-stale pointers:
  - `casper/HANDOFF.md` → now `docs/archive/HANDOFF.md`
  - `casper/PLAN_NEXT_2026-07-14.md` → now `docs/archive/PLAN_NEXT_2026-07-14.md`
  - `casper/PROGRAM_STATE_2026-07-10.md` → now `docs/archive/PROGRAM_STATE_2026-07-10.md`
  - `casper/src/casper/` → **removed** (see `docs/reference/ARCHITECTURE_V0_NOTES.md`)
  - `lit_review/` → renamed `external_literature/`
  - `experiments/CONCEPT_WHITENING_RESULT.md` → now `experiments/_archive/results/CONCEPT_WHITENING_RESULT.md`
  - `scripts/reweight_geo2.py` → now `scripts/_archive/experiments/reweight_geo2.py`
- Why: every new session loads the first ~25 KB of `MEMORY.md`; these stale pointers will send agents to
  dead paths. The "Project facts" and "Current" blocks are the worst offenders.
- Fix: rewrite the stale pointers to their new homes; point "Live state doc" at `docs/STATE.md`.

**[should-fix] `MEMORY.md` links to memory-notes that were moved into the repo, and to notes that no
longer exist in the memory dir.**
- Where: same `MEMORY.md`. Four notes were relocated into
  `external_literature/sources_raw/memory_maps/` (`belief-pool-elicitation-novelty-map`,
  `unified-embedding-novelty-map`, `lit-verification-and-paper-plan`, `paperC-continuous-deep-research`)
  but `MEMORY.md` still links them as local `(name.md)` notes → dangling.
- Also referenced but ABSENT from the memory dir: `read-own-results-before-theorizing`,
  `three-channel-continuous-elicitation`, and the six HARD-RULE notes
  (`never-reduce-data-without-confirmation`, `design-sheets-before-execution`,
  `stop-selling-negatives-fix-the-signal`, `save-all-checkpoints`, `fable-oversees-opus-executes`,
  `be-patient-with-training-dips`). *(Caveat: the absent-note set may predate this reorg; the four
  relocated-into-repo links are unambiguously reorg-caused.)*
- Fix: repoint the 4 relocated links at their `external_literature/` paths; verify/restore or delink the
  absent notes.

**Resolved / fine:** `CLAUDE.md`, `VISION.md`, `README.md`, `PUBLICATION_PLAN.md`, `STATE.md` pointers all
resolve. `README.md:16` `src/casper` is an intentional tombstone → `ARCHITECTURE_V0_NOTES.md` (which
exists and is a proper one-pager). `HARSH_REVIEW_POINTS.md` has no unresolved `.md`/`.py` pointers.

## 2. Contradictions between the north-star docs

**[should-fix] Which model is "canonical" is ambiguous across STATE and EXPERIMENTS.**
- Where: `docs/STATE.md:9` lists `set_mn/pbC_best.pt` (0.4946 / 0.3372) as the "current recommender…
  frozen mean for the current direction"; `docs/EXPERIMENTS.md:21` + `reference/INSTRUMENT_REVIEW.md`
  present RecVAE-d512 I2 (ML-25M 0.4998 / 0.3443, 8/8 gates) as the "strong-CF ruler / instrument".
- Why: two different "best" numbers/models with no doc stating which is THE canonical ruler for Paper A
  vs the belief work. A reader cannot tell whether the instrument is pbC or RecVAE. (They plausibly serve
  different roles, but nothing says so.)
- Fix: one line in `STATE.md` disambiguating: RecVAE-I2 = Paper-A instrument/ruler; pbC = concept-capable
  set-encoder frozen for the belief direction.

**[nit] Same checkpoint, two full-NDCG numbers.** `STATE.md:11` `a0c_best.pt` = 0.4961; `EXPERIMENTS.md:26`
SignedAE (a0c) "full 0.4954". Reconcile to one.

**[nit] `EXPERIMENTS.md` header is stale vs its own body.** `EXPERIMENTS.md:6` says "Chapters filled so
far: A + recommender-core. B/C/D/E/apparatus pending their review pass" — but the B/C/D/E/Apparatus
tables below are all fully populated, most citing promoted `reference/` sources. Update the header.

## 3. Stale content

**[should-fix] `setup.py` describes a project that no longer exists.**
- Where: `setup.py:70-78` `validate_project_structure()` creates `data/reddit`, `notebooks`, `tests`
  (Reddit ingestion is V0 / old-Paper-1; there is no `notebooks`/`tests` workflow now). `setup.py:47-61`
  `setup_environment()` copies from `.env.example`, which does **not** exist (setup would print an error).
  `setup.py:168` advertises a `poetry run jupyter notebook` flow.
- Why: the one "clone-and-init" entry point misdescribes the repo; a new machine would create phantom dirs.
- Fix: drop `data/reddit`/`notebooks`/`tests`, or refresh to the real ML-25M layout; ship an `.env.example`
  or remove that branch.

**[should-fix] `project_docs/TODO.md` and siblings are wholly V0-era.** `project_docs/TODO.md:3` "Paper 1
Deadline: UMAP 2026 (Jan 22 abstract)" — long past; body is DDPG/Reddit/continuous-RL/"Paper 1" content
superseded by `ARCHITECTURE_V0_NOTES.md`. (Folded into the orphan-dir finding in §4.)

## 4. Structural problems

**[should-fix] `project_docs/` is an undocumented top-level directory, orphaned and stale.**
- Where: `project_docs/` (tracked): `1.sh`, `1st/2nd/3rd run graphs.PNG`, `API_COST_ESTIMATE.md`,
  `CSMAI_19_MovieLens_Dataset_+_attributes.ipynb`, `TODO.md`, `TRAINING_ESTIMATES.md`.
- Why: it is NOT in the intended structure and is referenced by **no** live doc (grep of
  `CLAUDE.md`/`README.md`/`docs/` returns nothing). Its content is V0/old-Paper-1. It sits at the repo root
  masquerading as live.
- Fix: move under `docs/archive/` (or `previous_work/`), or delete; if kept, add a one-line tombstone.

**[should-fix] Five cache files are git-tracked despite matching `.gitignore`.**
- Where: `data/movielens/.cache/{Ql_unified.npy, evalcks_entdistill.csv, evalcks_st_grid.csv,
  peak_entdistill.txt, qcurve_paperC.csv}`. `git check-ignore` confirms they match the ignore rules
  (`data/movielens/`, `.cache/`) yet `git ls-files` lists them → committed before the rule / force-added.
- Why: directly contradicts the stated layout ("`data/` … `.cache/` — git-ignored") and the "raw outputs
  stay where produced, never re-copied" rule; bloats the repo with derived artifacts.
- Fix: `git rm --cached` these five; confirm nothing else under `data/`/`.cache/` is tracked.

**[nit] ~29 `*.log` files + two partial dumps are tracked despite `*.log` ignore + being junk.**
- Where: `experiments/SIM_RESEARCH_RAW_PARTIAL.txt`, `experiments/tree_louo.json.partial`, and ~29
  `experiments/_archive/old/paper2/*.log`. `.gitignore:78` ignores `*.log`; these predate it. Low priority
  (all archive), but inconsistent with the ignore policy.
- Fix: `git rm --cached` the logs and `.partial` scratch, or accept and note them as archived.

**[nit] `config/` at root is undocumented.** `config/config.yaml` + `config/config_quick_test.yaml` are
tracked and used, but the intended-structure list names only `answerer_schema.json` among root configs.
Harmless; add `config/` to the layout note in `README.md` for completeness.

## 5. Rule / protocol violations

**[should-fix] `MEMORY.md` "Project facts" violates the current recording protocol's homes.** It still
names `casper/src/casper/` as Source, `lit_review/` as lit, and `casper/PROGRAM_STATE_2026-07-10.md` /
`casper/HANDOFF.md` as the live state/handoff — but `CLAUDE.md:13` now mandates `docs/STATE.md` as the
single status doc and those two files are archived. (Same root cause as §1; called out here because it is
a protocol-home mismatch, not just a dead link.) Fix: update Project-facts to the new homes.

**Fine:** `CLAUDE.md` recording-protocol targets all resolve (`docs/STATE.md`, `docs/VISION.md`,
`docs/PUBLICATION_PLAN.md`, `docs/HARSH_REVIEW_POINTS.md`, memory notes). `CLAUDE.md:30,34` inline memory
pointers (`popb-vs-decoder-bias-full-ndcg-bug`, `ADAPTIVE_PROBE_RESULT.md`) resolve.

## 6. Code sanity (static)

**FINE — no action.** Static import scan of all 65 live `scripts/*.py` (+ `scripts/instrument2/`):
- No live script imports `casper`/`src.casper` (the removed package).
- No live script imports any archived module (`mf_foldin`, `golbandi_tree`, `continuous_actor`, `env`,
  `reweight_geo2`, `i25_fold_master`, `tree_louo`, …).
- The live cross-import graph closes over live modules only: `set_mn`, `signed_latent`, `arena_core`,
  `arena_policies`, `reconciled`, `i25_lib`, `i25_fold_v31`, `i25_fold_v3_sampler`, `dans_build`,
  `dans_stages`, `llm_answerability_gate`, `adaptivity_battery_v1`, `battery_stage_b`, `i25_phase4`,
  `i25_phase4_fair`. The `train_precacc → set_mn / signed_latent / arena_core / train_varhead` chain
  (the `STATE.md` next-action) resolves.
- `scripts/train_precacc.py` and `scripts/bpool2.py` (cited in `STATE.md` / `EXPERIMENTS.md`) both exist
  and are live.

---

## What's actually fine (calibration)
- North-star set is coherent and the pointer web is ~97% intact (1 broken pointer in-repo).
- `docs/reference` / `docs/archive` / `docs/design` have **zero** duplicate basenames — clean separation.
- Script archiving is disciplined: `_archive/{paper1_v1, paper2_old, frozen_instrument_saga, experiments}`
  is well-segregated and nothing live depends on it.
- `ARCHITECTURE_V0_NOTES.md` is a genuinely good tombstone for the removed package.
- Headline numbers are consistent where it matters (pbC 0.4946/0.3372 agrees between `STATE.md` and
  `MEMORY.md`; adaptivity/Kalman/PrecAcc narrative is aligned across `STATE`, `EXPERIMENTS`, `MEMORY`).
- `MEMORY.md` is 14.7 KB — comfortably under the 25 KB auto-load budget.
