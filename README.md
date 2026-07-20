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
- `scripts/` — research code (the live modules: `set_mn`, `arena_core`, `signed_latent`, `reconciled`, …)
- `src/casper/` — *(removed 2026-07; V0 architecture, see `docs/reference/ARCHITECTURE_V0_NOTES.md`)*
- `data/` — ML-25M · `.cache/` — checkpoints + derived caches (git-ignored)
