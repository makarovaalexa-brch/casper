# DS_C1_FINAL — train the policy to EMIT off-catalogue directions, on Paper B's cold tree

**Status: PRE-REGISTERED 2026-08-30. Author approved the run and settled the teacher** (*"plz train
on current tree from paper b, cold one i guess?"*) ⇒ teacher = **`coldtree_v2_train10000.json`**
(144 nodes, depth 11, L=15, fitted on 10,000 TRAIN users — Paper B's Act-5 artifact).
Scoping: `C1_FINAL_SCOPING.md`. Predecessor + caveats: `docs/results/C1_PILOT.md`.
Downstream consumers: `DS_C3B_ADAPTER.md` (renderer), `DS_C4_PILOT.md` (the loop).

## Why this run exists

The pilot proved the head can be aimed at an off-grid point at 42 states, but it was not a policy:
it collapsed everywhere else (163 → 4–6 distinct questions, snap agreement 0.0000), its emissions
were 38× out of scale and unrenderable, and it could not test generalisation. C1-final fixes all
three, because the two things that made it expensive turned out to be free.

## Inputs (all banked; no measurement campaign, no bank rebuild)

| what | artifact | note |
|---|---|---|
| teacher | `coldtree_v2_train10000.json` | 72 searched nodes + 72 below-floor inheritors |
| gain tables | `coldtree_v2_train10000_sweeps/` | **72 FULL exact whole-pool tables, 0 partial** — verified argmax/max against `splitter`/`splitter_gain` to <1e-12, all under `pool_sha1 ed67fe28…` |
| trajectory bank | `m6/traj_dj/cold_*.npz` | 85,289 users × 15 × 200, built 2026-08-29 (`run_disjoint_full.py`); 346 labels = the tree's 346 questions; escape 15/85,289 |
| warm start | `m7dj_policy.pt` | 526,542 params, identical head keys to E1 |
| pipeline | `m6c_distill_gen.py` + `tree_route.py` | `policy_traj.py` is pinned to `ndcgtree*` and must NOT be used |

## The target

At each of the 72 searched nodes: `target_s = Σ_{j≤5} w_j e_{q_j}`, `w_j ∝ max(gain_j, 0)`
normalised, over that node's own exact table. Everywhere else (tail positions, inheritor nodes —
870 slots) the target stays the teacher's **one-hot** question. Declared: this is a *mixed* target
set, and only the 72 node-states are blend states.

**Loss** = `α · (1 − cos(ẑ, target))` at blend states + the incumbent distillation CE elsewhere,
with `α` fixed at 1.0 (not tuned). **Plus a scale term**: `λ · (‖ẑ_mr‖/‖target_mr‖ − 1)²`,
λ = 0.1, mean-removed norms — registered because the pilot's cosine-only loss left the norm
unconstrained and it drifted 38×, which is what made its emissions unrenderable.

## Held-out STATES — the test the pilot could not run

**Primary split: by root subtree — hold out the entire `root.like.*` subtree** (15 searched states
held out, 57 trained on; prefixes disjoint, so no nesting leak). **Secondary, reported: depth ≥ 5**
(30 held out / 42 trained — and 42 is exactly the pilot's population size, so the comparison is
like-for-like). Random state splits are FORBIDDEN here: nesting leaks.

## Registered readings (paired bootstrap 10k, seed 98765; unit declared: states, and user-rows
within state where a per-user quantity is scored)

1. **GENERALISATION (the headline).** At held-out states, `cos(ẑ, target_s)` must exceed
   (a) `cos(ẑ, target_{s'})` for a seeded within-split random other held-out state, and
   (b) the warm-start policy's own `cos` at those states, both with CI excluding zero.
   *This is the claim the pilot could not make and the reason for the split.*
2. **FIT** at trained blend states: `cos(ẑ, target)` — reported, not a gate (the pilot showed it is
   easy).
3. **★ NO-COLLAPSE GATE (hard, blocking).** At 1,000 ordinary non-blend states: distinct emitted
   questions ≥ 0.75 × the warm-start policy's count, AND snap agreement with the warm start
   ≥ 0.90. **If either fails the run is a failure regardless of readings 1–2** — the pilot's
   collapse is the specific defect this run exists to avoid.
4. **RENDERABILITY GATE.** Median `‖ẑ_mr‖ / ‖target_mr‖` ∈ [0.5, 2.0]. Below/above ⇒ emissions go
   to the renderer only after a declared rescale, as in `C4_PILOT.md`'s ruling 2.
5. Off-grid geometry, as C0/C1: `cos₁` and margin of the emissions, mean-removed, against the
   random floor.

## Declared caveats (stated now, not discovered later)

- **Population**: the cold bank is a selected **60.6% of train** (users endorsing no anchor
  concept) — a by-design consequence of Paper B's forest split. Every C1-final number is
  conditioned on that population and says so.
- **Ingredient instability is NOT discharged by this tree.** Node cohorts are min 200 / median
  356, 43 of 72 below 500 — comparable to the pilot's, not larger; the repair pass's prescription
  ("build targets on larger cohorts") is therefore **unmet**, and the only fix is a 192-hour
  re-sweep we are not doing. **Claim wording, binding: we train toward a measured direction; we do
  NOT claim the five ingredients are the right five.** Split-half direction stability is reported
  if per-user sidecars exist for this tree, else the pilot's 0.656 (mean-removed) is carried as a
  prior and labelled as such.
- Stage 0's premise gate will not fire (top-1 blend weight median **0.225** over all 72, none
  above 0.5) — the blends are genuinely off-grid, as in the pilot.
- The 870 tail slots stay one-hot: this policy emits novel directions at node states only.
- `m7dj`'s phi bank is **1,004** entries, not 1,000 — assert, do not assume.
- The canonical snap constant for this split is the TRAIN value **0.15831**, not the validation
  0.15686538 the C-series has been asserting.

## Mechanics

`src/instrument/c1_final.py`, importing `c1_pilot` (target construction), `c0_geometry` (bank +
geometry), `m6c_distill_gen`/`tree_route` (bank + routing); never reimplementing. Warm start from
`m7dj_policy.pt`. Commit code before the run; threads ≤4; **reuse the existing bank — a rebuild is
~1.42 GB against 2.46 GB free and is not authorised**; atomic writes; incremental per-epoch flush.
Outputs `experiments/paperC/c1_final.json` + per-state npz; write-up `docs/results/C1_FINAL.md`
(Fable's conclusions). Validation/train-side only; the 10,000 canonical TEST users are untouched.
