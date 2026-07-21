# CASPER

Cold-start conversational recommender — a short **adaptive interview** that ranks a large catalog from a
handful of answers, questions eventually rendered by an LLM. (ML-25M.)

## Orientation
- **Rules** → `CLAUDE.md`
- **End-state / vision** → `docs/VISION.md`
- **Current status** → `docs/STATE.md`
- **Experiment record** → `docs/EXPERIMENTS.md`
- **External literature** → `external_literature/`
- **Papers (A–E)** → `new_chapters/` · **prior PhD output** → `previous_work/`

## Layout
- `src/` — **paper-supporting code only**: everything behind a number reported in a thesis chapter
  (`src/baselines/` = baseline implementations + the ML-25M-ruler runner). Rule: no discarded/false-path
  code ever lands here — working lines and explicit superseded-baselines only.
- `scripts/` — the lab: live research modules (`set_mn`, `arena_core`, `signed_latent`, `reconciled`, …),
  certification apparatus (`scripts/baselines/` = ML-20M snap harness), forensics (`scripts/_verify/`),
  and `scripts/_archive/` for dead ends.
- `data/` — ML-25M + ML-20M (git-ignored) · `.cache/` — checkpoints + derived caches (git-ignored)
