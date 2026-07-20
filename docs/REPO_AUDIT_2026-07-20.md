# REPO AUDIT — 2026-07-20 (adversarial, SECOND PASS)

> Second adversarial pass after the first-round fixes + the `simulator_ui/` removal.
> Verifies the claimed fixes AND hunts what the first pass missed. Findings ranked
> blocking / should-fix / nit, grouped by category. No files were modified except this report.

## Verdict (one paragraph)
The first-round fixes **mostly landed and verified clean** — `setup.py` gone, `project_docs/`
relocated into `docs/archive/`, zero tracked files under `data/`/`.cache/`, the `EXPERIMENTS.md`
`POLICY_LOG` pointer corrected, `MEMORY.md` rewritten to the new paths with **every note-link now
resolving**, and `STATE.md` given the pbC-vs-RecVAE canonical-model disambiguation the first pass
asked for. But the first pass **let the biggest V0 leftover walk**: `config/config.yaml` and
`config/config_quick_test.yaml` are **pure V0** — they configure a two-tower recommender, an
actor-critic RL agent, a GPT `question_generator`, a `preference_extractor`, a `user_simulator`, an
`embedding_space`, and Reddit training data: *exactly* the module set the V0 tombstone says was
removed. The first report literally called `config/` "used" and "harmless (nit)"; it is neither —
no live script reads it, it was last touched 2026-06-11, and `MEMORY.md` still advertises it as THE
project config. `pyproject.toml` carries the same V0 residue (dead `flask`/`flask-cors` from the
deleted server, `jupyter`/`notebook` with no workflow) **and is package-mode-broken** (`name="casper"`,
no `packages=`, and neither `./casper` nor `./src/casper` exists → `poetry install` fails). Finally,
**this very audit report was stale** — it described already-fixed issues as open and missed the
V0 configs; this pass overwrites it. Nothing blocks running an experiment today, but a fresh clone
can't `poetry install`, and a fresh session is told to read a dead V0 config. Grade: **B** — the
core doc web is now trustworthy; the build/config perimeter is still V0-contaminated.

Counts: **blocking 0 · should-fix 4 · nit 5**.

Re-verified CLEAN (first-pass fixes that actually landed):
`EXPERIMENTS.md:16` → `reference/POLICY_LOG.md` (all 30 source paths + `scripts/bpool2.py` resolve);
`MEMORY.md` repointed (`src/casper`→`scripts/`, `lit_review/`→`external_literature/`,
PROGRAM_STATE/HANDOFF→`docs/STATE.md`, PLAN_NEXT→`docs/archive/`, reweight_geo2→`scripts/_archive/`,
CONCEPT_WHITENING→`experiments/_archive/results/`) with **all note-links present on disk**;
**no** tracked files under `data/` or `.cache/`; `setup.py` gone; `project_docs/` gone (files now in
`docs/archive/`); `STATE.md:14` disambiguates pbC (interview recommender) vs RecVAE-I2 (Paper-A ruler);
`simulator_ui/` gone and **no live script imports `casper`/`src.casper` or any removed module**.

---

## 1. V0 / stale-provenance leftovers

**[should-fix] `config/config.yaml` is a pure-V0 config presented as current — the first pass missed it.**
- Where: `config/config.yaml` (tracked, last commit 2026-06-11 "V1-V4 training history").
- What: configures the entire removed V0 stack — `models.embedding_space` (SBERT entities),
  `models.actor_critic` (RL policy), `models.question_generator` (gpt-4o-mini asker),
  `models.preference_extractor`, `models.recommender` = **two-tower**, `models.user_simulator`,
  `paths.reddit_dir`, and `training_data.supervised.source: reddit`. This is one-to-one the module
  list in `docs/reference/ARCHITECTURE_V0_NOTES.md` ("what was removed").
- Why it matters: no live code reads it (`grep` for `config.yaml`/`load_config`/`OmegaConf`/`yaml.load`
  hits only `docs/archive/*` and this report). It survives as a live-looking root config describing an
  architecture that was deleted — precisely the class of thing this audit exists to kill. The first
  report called it "used" and filed `config/` as a harmless nit; both are wrong.
- Fix: delete `config/config.yaml` (recoverable from git), or move to `docs/archive/` with a one-line
  tombstone pointing at `ARCHITECTURE_V0_NOTES.md`. Update `ARCHITECTURE_V0_NOTES.md` to name the config
  among the V0 artifacts.

**[should-fix] `config/config_quick_test.yaml` is the same V0 config (quick-test variant).**
- Where: `config/config_quick_test.yaml` (tracked, same 2026-06-11 commit).
- What: identical V0 shape (embedding_space / actor_critic / question_generator / two-tower recommender /
  user_simulator / `source: reddit`), just with gpt-5-nano and 1-epoch/1-episode minima.
- Fix: remove or archive alongside `config.yaml`.

**[should-fix] `pyproject.toml` carries V0-only dependencies and is package-mode-broken.**
- Where: `pyproject.toml`.
- What: (a) `flask` + `flask-cors` have **zero live importers** — the only thing that used a Flask
  server was the just-deleted `simulator_ui/` (`grep` for `Flask(`/`app.run(`/`render_template` returns
  nothing outside `_archive`). (b) `jupyter` + `notebook` back a workflow that no longer exists (the
  first pass already removed `setup.py`'s `jupyter notebook` flow; the only `.ipynb` is archived under
  `docs/archive/`). (c) **Package-mode breakage:** `name = "casper"` with no `packages = [...]` and no
  `./casper` or `./src/casper` directory (V0 package removed) → Poetry's default package mode makes
  `poetry install` fail to find the package on a fresh clone.
- Note (not a bug): `openai` IS still justified — live `scripts/experiment_llm_recommender.py` and
  `scripts/llm_answerability_gate.py` import it. Verify `langchain`/`langchain-openai` before pruning
  (no live importer found, but confirm).
- Fix: add `package-mode = false` under `[tool.poetry]` (correct for a scripts-only research repo), and
  drop `flask`, `flask-cors`, and unused `langchain*`; keep `openai`.

## 2. Dangling references

**[should-fix] `MEMORY.md` still advertises the dead V0 config as THE project config.**
- Where: `C:\Users\Tech\.claude\projects\C--dev-phd\memory\MEMORY.md:60` — "Source `casper/scripts/`,
  config `casper/config/config.yaml`, lit `casper/external_literature/`." The `src/casper`→`scripts/`
  and `lit_review`→`external_literature` fixes landed, but the `config/config.yaml` pointer now sends
  every new session at the V0 file in §1. (Auto-loads every session.)
- Fix: drop the config clause (there is no live config), or repoint once the config is archived.

**[nit] `answerer_schema.json` uses stale/prefixless doc paths.**
- Where: `answerer_schema.json` — `source_design: "casper/ANSWERER_V1_DESIGN_REVIEW.md"` and
  `status: "…(see answerer_schema.md)"`. Both files now live under `docs/reference/`
  (`ANSWERER_V1_DESIGN_REVIEW.md`, `answerer_schema.md`); the `casper/`-prefixed / bare names don't
  resolve from repo root.
- Fix: rewrite to `docs/reference/ANSWERER_V1_DESIGN_REVIEW.md` and `docs/reference/answerer_schema.md`.

## 3. Contradictions / stale numbers

**[nit] `a0c_best.pt` full-NDCG still disagrees across docs.** `STATE.md:11` = 0.4961;
`EXPERIMENTS.md:26` (SignedAE a0c) = "full 0.4954". First pass flagged this; unfixed. Reconcile to one.

**Re-verified consistent:** pbC 0.4946/0.3372 agrees (`STATE.md:9` ↔ `MEMORY.md:8`); the canonical-model
ambiguity the first pass raised is now RESOLVED at `STATE.md:14`; the Kalman-incompatible → PrecAcc
narrative is aligned across STATE / EXPERIMENTS / MEMORY; PUBLICATION_PLAN and VISION are current
(VISION §5 correctly records the `simulator_ui/` removal).

## 4. Structure

**[should-fix] This audit report was itself stale until now.** `docs/REPO_AUDIT_2026-07-20.md` (the prior
version) listed `setup.py`, `project_docs/`, the 5 cache files, the `MEMORY.md` paths, and the
`EXPERIMENTS.md` pointer as open should-fix items **after they were fixed**, and its §6 "code sanity FINE"
verdict **missed the V0 configs entirely**. A reader would chase resolved findings and trust a clean bill
the perimeter didn't earn. Fixed by this overwrite; keep the report regenerated after each fix round.

**[nit] ~28 `*.log` + one `.partial` still tracked despite the `*.log` ignore.**
`experiments/_archive/old/paper2/*.log` (28) and `experiments/tree_louo.json.partial`. All under
`_archive`; predates the ignore rule. First-pass nit, unfixed. `git rm --cached` or accept-and-note.

**[nit] README layout note omits `config/`.** `README.md:14-18` lists `scripts/ src/casper data/ .cache/`
but not `config/`. The right resolution is deleting the V0 configs (§1); if any config survives, document
it — otherwise leave README as-is once `config/` is gone.

## 5. Config / README drift

**[nit] `config/config.yaml` paths point at non-existent V0 data dirs.** `paths.reddit_dir`,
`paths.movielens_dir`, `paths.processed_dir` describe a V0 `data/{reddit,movielens,processed}` layout
that drives nothing (folded into §1 — the file should go, not be repaired).

**Fine:** README's `src/casper` tombstone (`README.md:16` → `ARCHITECTURE_V0_NOTES.md`) is correct and
that note is a genuinely good one-pager; the `.gitignore` `.env*` rules match no tracked file
(no `.env`/`.env.example` present).

---

## Top actions (ranked)
1. Delete/archive `config/config.yaml` + `config/config_quick_test.yaml` (V0); drop the config pointer
   in `MEMORY.md:60`; name the config in `ARCHITECTURE_V0_NOTES.md`.
2. `pyproject.toml`: `package-mode = false`; prune `flask`/`flask-cors`/unused `langchain*`; keep `openai`.
3. Fix `answerer_schema.json` doc paths; reconcile the a0c 0.4961/0.4954 number.
4. `git rm --cached` the archived `*.log`/`.partial` (optional, consistency only).
