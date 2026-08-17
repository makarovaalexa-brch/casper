# Overnight handoff — night of 2026-08-18

**AUTHOR SAID (verbatim, before sleeping): *"is it running? i am going to bed, plz keep running and
distilling and all of it"***

Everything below this line is MINE (working notes). Governing plan: `docs/design/PAPER_B_POLICY_DESIGN.md`
Part M (+M.1–M.4) and `docs/design/DS_M3_FOREST_TREES.md`. Read those before acting.

## Running now — ONE heavy job, sequential driver

`bash src/instrument/run_m3_v2.sh` → `experiments/battery/m3v2.out`

1. **`coldtree_v2`** — the incumbent regrown on all 10,000 validation users under **stopping rule v2**.
   Started 00:00:38. Root seed pass measured at **8.73 s/eval**, so the root alone is ~2.4 h;
   expect **12–20 h** for the whole tree (v2 adds a tail search per splitting node and should reach
   greater depth than v1's 5).
2. **Six teachers** `m3v2_{arthouse,dialogue,marvel,gore,teenagers,inspiring}` — same recipe, cohort
   restricted to validation users who STATE the anchor, anchor folded free at the root and masked.
   ~20 min each under v1; v2 will be longer if they now branch.

Everything is resumable: contract-signed partials + per-node CELF heaps. A kill costs at most the
lazy re-evals since the last pick. **The v2 contract string differs from v1, so v1 partials cannot
resume — that is deliberate.**

## Why both sides are being regrown

Rule v1 took `min()` over ALL THREE children, so one small branch vetoed a whole split. For the
teachers it vetoed splits whose two real children held 334 and 396 users — in one case because the
did-not-answer branch was **empty**. All four teachers grown under v1 came out at **depth 0**, i.e.
conditioned static orders, which is exactly the structure the author rejected in the genre teachers.

**v2:** split iff ≥2 children are non-empty and ≥1 reaches `min_leaf`; children below the floor become
leaves that **inherit the parent's tail**, fitted on the parent's full cohort. Nothing is ever fitted
below the floor — which is what the 200 threshold's evidence (genre split, 2026-08-11) was actually
about.

**The cold tree must be regrown too, or the comparison is rigged**: giving teachers a better stopping
rule than the incumbent would manufacture a win. Both sides, same rule, same recipe.

★ Free by-product: this settles the open question in memory note
`tree-depth-is-the-stopping-rule-not-the-data` — v1's depth-5 stop was 15 floor stops with positive
gain in hand, and nobody knew whether deeper would pay. **v2 measures it.**

## Sanity gate — PASSED before launch (3 attempts, 2 real defects caught)

`m3_sanity_SANITY.json`. Heap-resume self-test **bit-identical** (cold vs resume-and-extend), K=1
cohort-weighted difference **exactly +0.00000** against both combined and concepts-only comparators
(the nesting guarantee survives v2), depth 2 / 13 nodes / 9 leaves on a 3-question budget.
Two defects it caught, both mine, both the same class — inherited-child records missing fields that
searched nodes carry (`sec_per_eval`, then `genre_face_validity`). Fixed at source.

## NEXT, once the driver reports ALL DONE

1. **Read the shapes first.** `[shape] depth reached …` for the cold tree and each teacher. The two
   numbers that matter: does the cold tree now go deeper than 5, and do the teachers branch at all?
   If teachers are still depth 0, the forest is conditioned orders and the chapter must say so.
2. **M4 — distil** the cold tree + six teachers into ONE policy (`forest_policy.py`, L2a machinery,
   warm-start E1). Gates: every teacher's fidelity ≥0.999 at its own states; **cold-tree fidelity
   unchanged** (zero interference re-proven at N=7 having been shown at N=2); shuffle control.
3. **The screen's missing control** (cheap, minutes): for each anchor's selected question, evaluate it
   against the tree root `(1,114)` at the **cold** state on all 10,000 validation users. If it wins
   there too, the anchor-specific gain is not conditioning. Expected clean — `(1,114)` is the
   policy-B cold argmax, confirmed as the first pick of `greedy_static_policyB_combined.json` — but
   it is an argument, not a measurement, until run.
4. **M5 — the single TEST look. DO NOT TAKE IT UNATTENDED.** It is pre-registered in
   `DS_M3_FOREST_TREES.md` (per-anchor + assembled mix + with/without openers + shuffled-routing
   control, all at **K ∈ {1,2,4,8,16}** per the author's instruction). It runs ONCE and it is the
   author's call when.

## Standing constraints (bind overnight)

Validation only until M5. Certified pair `t2i26c_ep31` + `concept_cv3_ep08`; policy-B mask/pool;
canonical snap `0.15686538118354684` asserted before any number is trusted; paired percentile
bootstrap 10,000 resamples seed 98765, never the unpaired form; `--threads 6`; **one heavy job at a
time**; commit before launching; persist per cell; a surprising number is a harness bug until proven
otherwise; no LLM API calls; never the Reddit API.

⚠ **The m3 (v1) teacher artifacts are superseded** — their partials and heaps were deleted, and their
reported "gain vs comparator" fields were against the policy-A static and were never citable anyway.
