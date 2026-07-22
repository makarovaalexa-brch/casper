# CASPER — project rules (read before any experiment or agent task)

Auto-loaded every session — kept terse on purpose. Each rule is the directive; detail/evidence is a `→ pointer`.
Orientation: end-state → `docs/VISION.md` · current status → `docs/STATE.md` · mistakes checklist → `docs/HARSH_REVIEW_POINTS.md`.

## Working rules
- **Design sheet before any run** — pre-register the question, exact Ns, full action space, baseline symmetry, metric + MDE, and every shortcut with its non-lossy alternative. If no decision changes on the outcome, it doesn't run.
- **No LLM API calls without explicit author approval.**
- **Fable oversees, Opus/Sonnet execute** — Fable = strategy/design/review/final-verdict only; Opus for research & hard coding, Sonnet for mechanical work (abstracts, tabulation, refactors, plots). Fable never reads the repo broadly or does groundwork itself — if a Fable task looks token-heavy, stop and ask the author first.

## Recording protocol — where findings go (no ledger; record before moving on)
- **A finding / lesson** → a **memory note** (`…/memory/<slug>.md`, one fact per file, frontmatter `type: project|feedback|user|reference`) + a one-line pointer in `MEMORY.md`. **Update** an existing note rather than duplicate; **delete** notes that turn out wrong. Keep `MEMORY.md` lean — only its first ~25 KB auto-loads.
- **Current status / numbers / next step** → overwrite `docs/STATE.md`.
- **End-state change** → `docs/VISION.md` · **paper/venue status** → `docs/PUBLICATION_PLAN.md` · **a review criticism** → `docs/HARSH_REVIEW_POINTS.md`.
- **Raw run outputs** (logs, metrics, checkpoints) stay where produced (`experiments/`, `.cache/`) — never re-copied into a doc.
- Nothing important lives only in chat: after a substantive result, record it in the right home the same session.

## HARD RULES
1. **No data truncation without explicit confirmation.** Never sample/cap/subsample/truncate/top-N/coverage-sample ANY data (users, items, questions, answers, tokens, features, candidates) without sign-off obtained first — silent reductions have repeatedly corrupted results. Use non-lossy alternatives (bucketing, streaming, sparse ops, per-item folding); if a reduction seems unavoidable, STOP and ask. Restate in every subagent brief.
2. **Start simple; verify against published numbers.** Begin with the simplest baseline and the published/named baselines, and **match their reported numbers exactly** before building anything on top — only trust a delta once its baseline snaps. Run the actual named prior; never justify skipping it, or a claim, with a bound measured under different settings.
3. **Define and check gates before moving forward.** Every build states its acceptance gates up front (e.g. the recommender's G0 strength-preserved / G1 usable-posterior / G2 elicitation-curve) and must pass them before the next step. No gate → no progress.
4. **Summarise all lit + deep-research so nothing is re-run.** Every external source and verdict lives in `external_literature/` (INDEX + bib + `findings/` by topic/paper); read it before any lit search. **After ANY deep research, commit its findings there immediately** — new sources → INDEX + bib, conclusions → the relevant `findings/` file, structured for retrieval so it never needs redoing. Read our own prior results before theorizing.
5. **Sanity checks / controls on every quantitative test.** Include an existential/shuffle control, a canonical-snap, and a leak check. A failed control means a broken instrument, not a real effect.
6. **No jumping to conclusions.** Failure to prove an idea by experiment defaults to a **weak method**, not that the idea is impossible/undoable. Fix the signal; never market a negative as a contribution.
7. **Save Fable tokens.** Hand Fable a self-contained brief (design sheet + the specific numbers), never let it read the whole repo; batch decisions; ask for a typed verdict, not open chat.
8. **Fixed tests, incremental steps, locked gains.** One fixed ruler / metric / user-count, stated once and reused; replicate before inventing; move forward incrementally without diverging or backtracking. **Each stage's key achievement stays locked behind a regression check** — a later step may not silently degrade what a prior step proved. Be patient with training dips (track the return curve; don't early-stop on a val dip).
9. **Keep the best checkpoint only.** Select on a disjoint val and record a durable peak; no train-then-eval throwaway, and no hoarding every epoch.
10. **Commit the training code BEFORE launching any run.** A checkpoint is only as good as the committed forward that scores it — pbC's recorded 0.4946 became unreproducible because its forward lived only in the working tree. No run starts on a dirty tree. *→ memory `anchor-provenance-hole-pbc-recvae`.*

## Metric & reporting (domain)
- **Canonical full-profile ruler (2026-07-21/22, author-approved): the Liang-recipe strong-generalization split applied to ML-25M, catalog-restricted** — items pre-filtered to the project catalog (movies ≥20 total ratings = the arena 18,430; final train vocab **18,359**); then Liang user-side verbatim: ratings >3.5, min_uc=5, **train 140,768 / val 10,000 / test 10,000 users**, per-user 80/20 fold-in/target, seed 98765; built by `scripts/baselines/liang_split.py --data ml-25m` → `data/ml-25m/proc/`; runner `src/baselines/run_ml25m_liang.py`. Primary metric: full AND tail NDCG@10 (tail = top-33%-train-mass head mask, 190 head items); NDCG@100 secondary. The old 500/500 arena split is for elicitation-line continuity ONLY — never for G0/full-profile claims. Study-user quarantine: dropped (author ruling, Jul 21).
- **Always report full AND tail NDCG@10**, scored with the recommender's **learned decoder bias**, not a log-count `popb` floor (corr ~0.79; popb craters full-NDCG). A reconstruction must reproduce ~0.486 full-profile before any conclusion; cold-start ≈ 0.19 — headroom is real, never call full "solved". *→ memory `popb-vs-decoder-bias-full-ndcg-bug`.*
- **Check every experiment/claim against `docs/HARSH_REVIEW_POINTS.md`**; if it matches a ✗, STOP. Add new criticism there.

## Settled — do not relitigate
- **All pre-Jul-22 interview-line NUMBERS are demoted** (they ran on the retired LLM answer environment) — incl. B answerable-concepts, C ladder, D open-recall, and the +47% adaptivity magnitude. Directions = credible priors; magnitudes re-establish on the new stack before being cited. *→ memory `pre-jul22-interview-results-demoted`.*
- **LLM-derived answer/answerability artifacts are NOT usable for new work** (Jul-22 audit: popularity-dominated, 3/4 gates null-passing, distilled value ≈ EASE). No new model may train on, gate on, or cite the distilled grids (`mm_*`, answerability pmodel). Replacements: values = real ratings (items) / SEL behavioral (concepts); answerability structure = the transparent structural rule (rated∨pop>τ; ≥k rated members). The grids stay ONLY to reproduce pre-Jul-22 B/C/D numbers, labeled "under an LLM-derived answer model". *→ memory `answerability-audit-2026-07-22`.*
- **Adaptivity-pays: direction settled, magnitude demoted.** Never argue "static is optimal" — the theorem assumes linear-Gaussian + variance-only, which we have neither of. But the +47% NUMBER ran on the retired answer environment — re-establish on the clean stack before citing it. *→ `experiments/ADAPTIVE_PROBE_RESULT.md`, memory `pre-jul22-interview-results-demoted`.*
