# Paper B — the single TODO (state of the whole paper, one page)

Maintained 2026-08-10. This is the one place to see the paper's state; per-stream task lists and the
per-act design sheets hang off it. Draft: `new_chapters/chapterB_v2/chapterB_v2.tex` (14 pp, builds
clean, 8 red REDO + 7 blue PLAN markers so nothing unfinished is invisible).

---

## THE CLAIM, as the evidence now supports it

Adaptive question selection does not pay in cold-start interviews **and the reason is not the one we
first proposed.** It is not that there is nothing to adapt to. Where the criterion says personalising
is worth the most, the adaptive methods are furthest behind — measured on one population, 38 of 48
trend cells separated, all negative, shuffle control clean. So the finding is *methods cannot find
the structure that exists*, which is stronger than a flat null and different from what Acts 1–2 said
a week ago.

---

## ACT 1 — the battery. **EXPERIMENTALLY COMPLETE.**

21 selectors across six families, one contract (certified pair, policy-B pool, NO-FLOOR, behavioural
answers, 10,000 test users), paired intervals throughout. Six native snaps: EDDI **FAILED** (reported
as failed), DRE / Golbandi / IGCN **PASSED**, DPE setup passed, Greedy SLIM **PARTIAL + the ablation
finding**. Floored arm done. Oracle ceiling measured.

| # | item | state |
|---|---|---|
| 1.1 | Revise the Act-1 conditional — **the tree's crossover is a PV=0 effect** (see Act 2) | **OPEN, blocking** |
| 1.2 | Concept-bank test: give an adaptive selector the concept bank, see if it can then convert PV | **OPEN — the key experiment** |
| 1.3 | EDDI full Eq. 9 (both terms); current row is relabelled, not fixed | open |
| 1.4 | CLiME width-rule run (measured negative: zero concept questions under a pure uncertainty trigger) | open |
| 1.5 | Re-run DRE + DPE with in-run `fixedorder:` comparators (currently cross-run, admitted on a 0.0e+00 control) | open, cosmetic |
| 1.6 | ConUCB Eq. 8 caching fix (value-free but uncached; 451× redundant) | open, low |
| — | fMF | **RULED OUT by the author** |
| — | RBMF, RMVA, PERE, GreedyExtend, MeLU, PEBOL, a-stratification, SEL-BALD | dropped, reasons recorded |

## ACT 2 — the mechanism. **MEASURED, AND IT CHANGED THE STORY.**

Dispersion collapse measured two independent ways: effective distinct optimal questions
(1.00 → 2.88 → 5.04 on ML; 95.13 on real requests) **and** distinct states (11 / 112 / 1,463 / 5,767
at k=1/2/4/8). Quartile stratification done on 10,000 users.

| # | item | state |
|---|---|---|
| 2.1 | Fold the quartile result into the chapter — it changes Act 2's claim **and** Act 1's | **OPEN, blocking** |
| 2.2 | State the confound in-chapter: PV ↔ answer rate ρ=+0.76 at k=2; clean at k=4/8 | open |
| 2.3 | Answerability-prediction selector → decompose the oracle gap into predictable vs unknowable | open, high value |
| 2.4 | Test "popularity is the answerability prior": correlate each selector's answer rate against the popularity of what it asks | open, cheap |

## ACT 3 — the intervention (open the opener). **NOT STARTED.**

Emulate an open opener on ML-25M (favourite film / favourite genre / distinctiveness-framed), with
the question distribution mined from the 4,431 real requests; users, answers and scoring stay
MovieLens. Difference-in-differences against the closed opener. 9-point threat model already written
(`PAPER_B_DESIGN.md`), including the dispersion-reproduction validation gate that must pass first.

## ACT 4 — coarse-to-fine. **PARTLY MEASURED, NEEDS REDIRECTING.**

Greedy emits `c c c i c c i i i i c c i i i i` with no curriculum specified; the channel flips with
placement. **New lead from Act 2:** the loss-optimal question is a *concept* for ~100% of users at
every budget while all three adaptive selectors shop in the item bank — a concrete hypothesis for
why they cannot convert PV, and testable (1.2).

| # | item | state |
|---|---|---|
| 4.1 | `--objective tail` greedy sequence (required before any tail-optimised claim) | open |
| 4.2 | Face-validity exhibits: hand-picked trajectories + genre-congruence vs matched-divisiveness control | open |
| 4.3 | `--conc_bank polar` in the greedy suite (needs the `H(bands\|answered)` estimator ported) | open |

## ACT 5 — Reddit external validity. **ARTIFACTS BUILT, CELLS NOT RUN.**

4,431 linked threads, slot parse done (94% genuine requests, median 5 slots), linking precision
0.960 vs a 500-mention gold standard. Withheld-slot replay (both variants) not run.

---

## CROSS-CUTTING

* **Writing**: related-work section (survey exists in `external_literature/`, bib entries not
  written); the concept-fold result must be stated narrowly (it compares *initialisation recipes*,
  not our trained embedding — author correction 2026-08-10).
* **Protocol paragraph**: Greedy SLIM is now a published case study — a real adaptive-degradation
  curve manufactured by target-set shrinkage, which our policy B removes. Written in.
* **Bib hazard**: adding a key that collides with `references.bib` once aborted Paper A's whole
  bibliography (366 citations → `[?]`). Collision-check before adding.
* **Reproducibility debt**: `glm_ucb`/`thompson` ride an iALS basis that is not bit-reproducible
  across BLAS thread counts; the basis is now banked, but pre-fix cells must not be compared to
  post-fix cells at 4 dp.

## ORDER I WOULD WORK IN

1. **1.1 + 2.1** — the chapter currently states a conditional the data no longer supports. Writing, no compute.
2. **1.2** — the concept-bank test. One experiment, and it is the direct test of the only mechanism hypothesis we have for the central finding.
3. **2.3 / 2.4** — the answerability decomposition; cheap and it closes the chapter's own loop.
4. **4.1, 4.2** — tail-objective greedy and face validity.
5. **Act 3** — the intervention, once the mechanism is settled.
