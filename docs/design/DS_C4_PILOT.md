# DS_C4_PILOT — render the policy's own emitted points, fold them back, measure faithfulness

**Status: PRE-REGISTERED 2026-08-25, author approved ("please proceed yes").** Completes the loop
policy → point → English → fold, ANSWER-FREE throughout: no simulated answer to any rendered
question, no NDCG claim anywhere (the circularity ruling's fatal use stays closed; this measures
whether the TEXT is a faithful carrier of the POINT, never whether asking it would help).
Owners: `PAPER_C_DESIGN.md` (C4 as amended 2026-08-24), `docs/results/C3B_ADAPTER.md` (the trained
adapters), `docs/results/C1_PILOT.md` (the points).

## Objects

- **Points**: the C1 pilot's registered 42 states — the m=5 blend target AND the full/seed1
  emission per state (84 vectors, bias-free k(φ), from `c1_pilot_per_state.npz`; the emission sits
  at cos ≈ 0.99 to its target, declared). Plus **42 spectrum-matched random points** (the repaired
  mean-removed-space rescale, seeded) as the point-level null.
- **Renderers**: adapter **A** (true, `c3b_true_best.pt`) and twin **A′** (scrambled) — split-hash
  asserts as in stage-2. Adapter input = the manifest's mean-removal + global scale applied to each
  point. **A policy point has no channel**, so every point is rendered under BOTH channel one-hots
  (item-hot / concept-hot); both rows reported in full, never pooled, no post-hoc primary.
- **Prior-only** text per channel (zeroed latent + one-hot), from stage-2's construction.

## The reference effect (what "faithful" is measured against)

A k(φ) vector cannot be folded into the tower directly, so the state's reference direction is
**R_s = Σ w_j · tower(q_j)** over the state's m=5 measured ingredients (item → its centred,
popularity-stripped decoder row; concept → its `build_concept_dirs` direction) — the exact
construction the C3B bridge uses for targets, applied to the blend's own ingredients. R_s serves as
the reference for BOTH the target and the emission at that state (declared; the emission has no
ingredient decomposition of its own).

## The measure

Rendered text → the C3B tower bridge (SBERT → concepts → cv3 fold displacement, popularity-stripped)
→ **cos(u_text, R_s)**, established conventions. THE FOLD IS THE FOLD-BACK: this displacement is
exactly what entering the rendered question's content into the belief contributes; its alignment
with R_s is the answer-free faithfulness number. Every statistic: paired percentile bootstrap over
the 42 states, 10k resamples, seed 98765, CI reported.

## Registered readings (directional — the C3B C2 lesson applied)

1. **Faithfulness**: A(point) − A′(point) gap per state, mean > 0 with CI excluding zero, per
   point-kind × channel-hot cell. This is the carrying criterion.
2. **Point-specificity**: A(real point) − A(random point) > 0 with CI excluding zero. (A renders
   ANY vector; this asks whether real points' renders align better with their R_s than random
   points' renders do with theirs — computed against each random point's own nearest-state R.)
   Prediction registered as uncertain: C0 showed even random directions carry bank-aligned
   mean-removed structure; a null here is reportable, not a failure.
3. **The catalogue comparator, report-only, no bar**: the state's nearest bank question's GOLD
   phrase through the same bridge vs R_s — how well the best on-catalogue text carries the state's
   direction. A(point) above it would be remarkable; below it is expected and claims nothing (the
   in-between claim is about existence, not superiority).
4. **Prior floor**: prior-only text vs R_s — must sit at ≈0; A′ may sit anywhere at or below the
   prior + A row (directionless equivalence NOT used).
5. **The demo table** (the author's "what do they translate to"): all 42 states × {target,
   emission} × both channel-hots — the rendered English verbatim, with the state's interview path,
   in the results doc. Qualitative, no grading, explicitly not evidence.

## What may NOT be claimed

No NDCG, no "asking it would help", no answer simulated to any rendered text, no human-plausibility
grading (machine-only synonyms: C3B), no pooling across the declared cells, no post-hoc primary.
n = 42 nested, structurally-selected states (the C1 caveats travel); this is a PILOT of C4's
machinery on the only emitted points that exist pre-retrain.

## Mechanics

Code `src/instrument/c4_pilot.py` importing the C3B bridge/adapters (never reimplementing);
commit before run; threads ≤ 4; minutes of compute (≈250 generations + SBERT over short strings);
outputs `experiments/paperC/c4_pilot.json` + per-state npz + `docs/results/C4_PILOT.md` (conclusions
Fable's own). Validation-side only; no test users; no LLM API.
