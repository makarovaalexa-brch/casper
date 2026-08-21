# C1 PILOT — can a gain-weighted blend target pull the emitted vector off the question grid?

**Status: C1 PILOT COMPLETE (validation only; no test look taken). 2026-08-21.**
Design sheet: `docs/design/DS_C1_PILOT.md` (pre-registered 2026-08-21, binding; author approved a
pilot/PoC the same date). Data provenance: `docs/design/C1_GAIN_TABLE_AUDIT.md`. Baseline:
`docs/results/C0_GEOMETRY.md`. Code: `src/instrument/c1_pilot.py`, committed **before** the run at
`a7ffc52c031bbf2007a08d4876de41591319b309` (HARD RULE 10). Checkpoint fine-tuned from:
`experiments/battery/policy_main_equivariant_both.pt`, epoch 2, sha256 `9c4c61e7…6003058` — the
sha is asserted against `c0_geometry.json` at run time, so this is the same object C0 measured.
Wall clock **4.4 minutes** (42 states, 8 fine-tunes, all geometry).

**One-line verdict: Stage 0 does NOT stop the pilot — the blends are genuinely off the grid (median
target-cos₁ 0.931, median margin ratio 0.215) — and Stage 1 fits them. The full-policy arm passes all
three registered criteria (SUCCESS); the head-only arm passes two of three, missing criterion 2's
80% same-nearest-question bar by a single state (33/42 = 0.7857). But the two controls are what the
reading turns on: a policy trained on PERMUTED targets scores 0.7857 on the same criterion and 0.988
on criterion 1, and the fine-tuned policy's behaviour at ordinary states collapses completely
(4–6 distinct questions over 1,000 states against the banked E1's 163; snap agreement 0.0000).**

So the sentence the sheet licenses on success — *"the architecture can be trained to emit stable
off-catalogue points at states with measured blend targets"* — is earned, and nothing beyond it is.
The pilot did not show that the blend TARGET is what the policy learned; it showed that a 42-target
regression onto arbitrary points in e_q-space is easy for this head.

---

## What was run

| | |
|---|---|
| states | the **42** with full exact ~1,000-candidate gain tables: **35** `policy_labels_deep_partial` sweeps whose `nid` and cohort size match a `coldtree_v2` node + **7** single-pick (`len(seq)==1`) `ndcgtree_heaps` nodes. The entire exact-coverage population, not a sample (HARD RULE 1) |
| secondary | the **7** deep sweeps that map to no `coldtree_v2` node (children of nodes the tree left as leaves). Exact and paid-for, outside the registered 42 — Stage 0 only, never trained on, never in a registered cell |
| users | 36,197 user-rows over the registered 42 (cohorts 9–10,000; prefix lengths 0–6; 994–1,000 measured in-pool candidates per state), 37,197 including the secondary 7 |
| target | `target = Σ_{j≤m} w_j e_{q_j}`, `w_j ∝ max(gain_j, 0)`, normalised; **m=5 primary**, m∈{3,10} reported. `gain` = `vals[q][0]`, the NDCG-full marginal gain the tree optimises |
| space | C0's homogeneous coordinates `ẑ = [u(h), 1]`, `e_q = [k(φ_q), b(φ_q)]` (primary) and the bias-free `k(φ_q)` (secondary) |
| geometry | `c0_geometry.geometry()` / `bilinear_cos()` / `dist()` **imported and called**, not reimplemented; the 42 states are placed into the rollout's `(user, position)` row layout so the same function, with the same already-asked exclusions, runs unmodified |
| gates | canonical m=0 snap `0.15686538118354684` exact; `branch_of_per_user == branch_of`; leak check 0/10,000; the precomputed encoder path asserted **bit-identical** (max abs diff **0**) to `pol.enc` + `pol.head.q` |
| firewall | validation only; the 10,000 test users are not built, not loaded, not touched |

### ★ The analytic ceiling (computed before the optimiser ran)

`ẑ`'s last coordinate is pinned to 1, so for a target `t = [t_u, t_b]` no setting of `u(h)` can beat
`cos = 1` when `t_b ≥ 0`, or `|t_u|/|t|` when `t_b < 0`. Median target bias coordinate is **−1.87**,
but `|t_u|` dwarfs it: the ceiling is **0.9964 median, 0.9920 minimum, ≥0.90 at 42/42 states**. The
registered `cos ≥ 0.90` was therefore reachable, and that was known before any training.

---

## Stage 0 — the free gate. The targets ARE off the grid

Registered stop rule: **STOP if median target-cos₁ > 0.99.** It did not fire.

| m | set | median target-cos₁ | IQR | min/max | median margin `m` | bias-free cos₁ | ≥3 in 5% band | median w₁ | random floor cos₁ / m |
|---|---|---|---|---|---|---|---|---|---|
| **5** | **registered 42** | **0.9313** | [0.9168, 0.9494] | 0.8915 / 0.9628 | **0.2150** | 0.9360 | **1.000** | 0.228 | 0.0734 / 0.258 |
| 5 | secondary 7 | 0.9198 | [0.9153, 0.9369] | 0.9051 / 0.9819 | 0.3237 | 0.9263 | 1.000 | 0.215 | 0.0726 / 0.487 |
| 3 | registered 42 | 0.9243 | [0.9141, 0.9448] | 0.8809 / 0.9629 | 0.2405 | 0.9285 | 1.000 | 0.356 | 0.0734 / 0.261 |
| 10 | registered 42 | 0.9316 | [0.9198, 0.9468] | 0.9009 / 0.9700 | 0.2631 | 0.9368 | 1.000 | 0.128 | 0.0734 / 0.255 |

The C0 random floor (median cos₁ **+0.0642**, m **0.2668**, from `c0_geometry.json`) is carried into
every table above as the per-state floor recomputed by the same call (0.073 / 0.26).

**Three things this says, and they do not all point the same way.**

1. **The blends do not collapse onto `e₍₁₎`.** Median w₁ is 0.228 at m=5 — the measured gain profile
   at these states is close to flat over the top few candidates, which is why m=3, 5 and 10 give
   nearly the same geometry. The gate's premise-failure case is not what happened.
2. **They are "between candidates" by C0's own reading rule** — median margin ratio 0.215 (C0's
   `between_candidates` trigger is m ≤ 0.60), and **every one of the 42 states has ≥3 bank entries
   inside the 5%-of-|s₁| band**, against 20.8% for E1's own emissions.
3. **But cos₁ = 0.93 is high, and the question it is 0.93 close to is never one of its ingredients.**
   `target_snap_equals_top_gain_candidate = 0.0000` at every state and every m: the argmax over the
   full 19,390-question bank for a blend of five measured candidates is some *other* bank entry
   entirely. The bank is anisotropic enough (C0: a large shared component in `k`/`b`) that an
   arbitrary point in it sits at cosine ~0.93 to something. "Off the grid" here means *crowded and
   low-margin*, not *far from everything*.

**Stage 0 verdict: PROCEED.** (Recorded in `c1_pilot.json:stage0.gate`.)

---

## Stage 1 — the fine-tunes

Loss `1 − cos(ẑ, target)` at the 42 states, each state weighted equally regardless of cohort size;
Adam, lr 1e-3, 40 epochs, batch 4,096 user-rows; seeds 1, 2, 3 per arm. `head.k` / `head.b` receive
no gradient from this loss **by construction** (the target is a constant built from them), so the
bank the geometry is read against cannot move in either arm — that is a property of the loss, not a
freezing decision. Arm A trains `head.q` only (the encoder is frozen, so `h` is cached); arm B trains
the encoder as well.

Starting point, for scale: the banked E1 emits a vector at median cos **−0.2893** to the blend target
it is about to be trained on (IQR [−0.370, −0.189]).

| run | final loss | median cos(ẑ,target) | min | bias-free cos | \|Δcos₁\| vs target (med) | same nearest q | emitted cos₁ | emitted margin |
|---|---|---|---|---|---|---|---|---|
| head_only/seed1 | 0.01093 | 0.98930 | 0.9725 | 0.99496 | 0.01095 | 0.7857 | 0.9306 | 0.340 |
| head_only/seed2 | 0.01094 | 0.98932 | 0.9725 | 0.99496 | 0.01096 | 0.7857 | 0.9304 | 0.341 |
| head_only/seed3 | 0.01096 | 0.98930 | 0.9725 | 0.99488 | 0.01130 | 0.7857 | 0.9308 | 0.341 |
| full/seed1 | 0.00693 | 0.99352 | 0.9881 | 0.99778 | 0.00854 | 0.8095 | 0.9248 | 0.311 |
| full/seed2 | 0.00679 | 0.99357 | 0.9889 | 0.99784 | 0.00703 | 0.8095 | 0.9244 | 0.305 |
| full/seed3 | 0.00687 | 0.99353 | 0.9890 | 0.99789 | 0.00717 | 0.8095 | 0.9235 | 0.312 |
| **head_only SHUFFLED** | 0.01353 | **0.98809** | 0.9659 | 0.99235 | 0.01509 | **0.7857** | 0.9357 | 0.362 |
| **full SHUFFLED** | 0.00938 | **0.99104** | 0.9722 | 0.99626 | 0.01307 | **0.7857** | 0.9302 | 0.316 |

`cos ≥ 0.90` holds at **42/42 states in every run**. Within-state agreement (per-user ẑ against the
cohort-mean ẑ) is **0.99999** everywhere: the cohort's users emit essentially one vector per state,
so "the state's emitted point" is a well-defined object and not an artefact of averaging.

### The registered criteria

| criterion | bar | head-only (seed-avg) | full (seed-avg) |
|---|---|---|---|
| 1 — fits its target | mean cos ≥ 0.90 | **0.98908 PASS** | **0.99320 PASS** |
| 2a — off-grid signature | median \|cos₁(ẑ) − cos₁(target)\| ≤ 0.05 | **0.01107 PASS** | **0.00758 PASS** |
| 2b — same nearest bank question | ≥ 0.80 of states | **0.7857 (33/42) FAIL** | **0.8095 (34/42) PASS** |
| 3 — seed stability | pairwise cos ≥ 0.95 | **0.99999 PASS** | **0.99977 PASS** |
| **registered cell** | all three | **criterion 2 misses with 1 passing** | **SUCCESS (all three)** |

Criterion 3 is passed with room to spare but is a weak test **as implemented**: all three seeds start
from the same banked checkpoint and the only stochasticity is minibatch ordering, so 0.9999 is close
to what one should expect. It rules out optimisation instability, not much else.

### Controls

**Shuffle (targets deranged across the 42 states — no state keeps its own).** In both arms the true
assignment fits better than the shuffled one (head-only 0.98930 vs 0.98809; full 0.99354 vs 0.99104),
so the registered control passes and a state-independent collapse is ruled out. **That is all it
rules out.** The shuffled runs still reach cos ≈ 0.99, i.e. 42 *arbitrary* points are as fittable as
the 42 *measured* ones — which is the sheet's own warning (42 examples, 526k parameters) showing up
in the control rather than in the loss. Worse for the reading of criterion 2: the shuffled arms score
**0.7857 on same-nearest-question**, identical to head-only's true-target score and 0.024 below full's.
**Criterion 2 as registered barely distinguishes the correct target from a permuted one**, because
what it really measures is "did the emission land on a blend-shaped point", and any blend of five
bank entries at any of these states has the same shape. This is recorded here rather than argued
away: the criterion did what it was written to do (rule out a snap back onto `e₍₁₎`), but it is not
evidence that the *measured gains* were learned.

**C0 random floor** carried into every Stage 0 table (median cos₁ +0.064, m 0.267).

**Collateral (reported, not gated).** Snap agreement with the banked E1 at 1,000 random ordinary
states (the validation trajectory bank's own `(user, position)` states, both policies scoring the
same 1,000-candidate pool):

| | banked E1 | fine-tuned |
|---|---|---|
| agreement | — | **0.0000 in all six runs** |
| distinct questions over the 1,000 states | 163 | **6** (head-only) / **4** (full) |

The fine-tune does not merely drift; away from the 42 states the policy collapses to a handful of
questions and agrees with its parent nowhere. The sheet deliberately did not gate this — the pilot's
claim is capability, not a deployable policy — but it fixes the size of the problem any later C1 has
to solve: a replay/regularisation term is not a nicety, it is the whole difference between this
result and a policy.

---

## What may and may not be said

**May.** *The architecture can be trained to emit stable, well-defined off-catalogue points at states
with measured blend targets, and those points carry the same off-grid signature as the targets
(cos₁ ≈ 0.93 vs 0.93, margin 0.31 vs 0.22, ≥3 near-ties at 100% of states).* The registered
kill-gate of `PAPER_C_DESIGN.md` did not fire: C1 is not dead on the "can it place a point" question.

**May not.** No NDCG number, no generalisation, no "novel question", and — added by the controls —
**no claim that the measured gain profile is what the policy encoded**, since a permuted assignment
fits nearly as well and scores the same on criterion 2. Nor is the fine-tuned object a policy: at
ordinary states it emits four to six distinct questions in total.

**The named cheat did not fire, in the sense it was named.** The sheet's worry was a high target-cos
bought through the common-mode/bias component. The bias-free reading is *higher*, not lower
(0.995 / 0.998 against 0.989 / 0.993 homogeneous), and the targets' own bias-free cos₁ (0.936) tracks
their homogeneous cos₁ (0.931). The common-mode component shows up somewhere else instead: in the
fact that an arbitrary point of e_q-space is already at cosine 0.93 to *some* bank entry, which is
why "cos₁ ≈ 0.93" cannot be read as "nearly on a question".

## Owed next, if C1 continues

1. A target the controls can distinguish — the shuffle control fitting to 0.99 means 42 states are
   too few to separate "learned the gains" from "memorised 42 points". More states (the audit's
   targeted 14,123-evaluation repair buys 55 more) or a held-out state split.
2. Replay/regularisation, and a rerun of the collateral measurement as a gate rather than a report.
3. Criterion 2 needs a version that a permuted target fails.

## Artifacts

| what | where |
|---|---|
| Stage 0 tables (all m, both sets), per-run Stage 1 results, registered reading, controls, gates, git rev, checkpoint sha | `experiments/paperC/c1_pilot.json` |
| per-state targets (m=3/5/10), their cos₁ / margin / snap / analytic ceiling, per-state emitted vectors and per-state fit cosines for every run | `experiments/paperC/c1_pilot_per_state.npz` |
| run log | `experiments/paperC/c1.out` |
| code | `src/instrument/c1_pilot.py` (commit `a7ffc52`) |
