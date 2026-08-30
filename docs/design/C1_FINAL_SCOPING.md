# C1-FINAL SCOPING — costing and de-risking the emitting policy on the CURRENT cold tree

**Read-only investigation, 2026-08-30. Nothing was run, nothing was retrained, no test data was
named.** Every number below is read out of files already on disk or quoted from a recorded runtime.
Working tree is dirty (docs only) at `d3bc837`; nothing was committed.

**The decision this costs (author, 2026-08-30):** C1-final trains the emitting policy on **Paper B's
current COLD tree**, `experiments/battery/coldtree_v2_train10000.json` — the train-fitted Act-5
artefact (144 nodes, depth 11, `total_L` 15, `min_leaf` 200, fitted on 10,000 TRAIN users,
`seconds` 243,006 = 67.5 h) — **not** the 22-node `ndcgtree16` E1 was distilled from, and not the
164-node validation-grown `coldtree_v2` the C1 pilot used.

---

## ★ BOTTOM LINE — the two blockers the gain-table audit left open are BOTH already closed

| the audit's blocker (2026-08-21, on `coldtree_v2`) | status on `coldtree_v2_train10000` |
|---|---|
| only **42 of 164** node states have a full exact gain table; 55 partial, 67 unmeasured | **72 of 72 searched nodes have a FULL exact whole-pool table.** 0 partial. The 72 below-floor inheritors are unmeasured (inherent, unchanged) |
| repair costs ~14,123 evaluations / ~1.75–2.3 model-hours | **£0. The repair is already paid.** The commit-`8b3024f` `save_sweep` fix was live when this tree was grown, so the sweeps were banked as the tree was built |
| **"E1 was not trained on this tree — a new `policy_traj` bank is needed"** — *"still open and still the bigger issue"* | **A trajectory bank on THIS tree already exists**: `experiments/battery/m6/traj_dj/cold_*.npz`, 85,289 train users × 15 positions, 852 MB, built 2026-08-29 by `run_disjoint_full.py` (`COLD = "coldtree_v2_train10000.json"`). Verified: its 346 distinct label indices are **exactly** the tree's 346 distinct questions |

**So C1-final's marginal compute for data is ZERO if the existing bank's cohort is accepted.** What
remains to be decided is (i) whether the 85,289-user cold bank (a *selected* subpopulation — users
who endorse none of the six group anchors) is acceptable, (ii) whether targets built on the tree's
own **small fit cohorts** (median 356 users) carry the word "measured", and (iii) disk — C: has
**2.46 GB free**.

---

## 1. Gain coverage of the CURRENT cold tree

`coldtree_v2_train10000.json`: **144 nodes** = 72 searched (49 internal + 23 searched leaves) + 72
below-floor inheritors (all leaves). 95 leaves. Depth histogram 1/3/9/20/26/29/17/12/9/6/6/6 at
depths 0…11.

### 1.1 The three-way split (same classification the audit used)

| coverage class | nodes | share | source |
|---|---|---|---|
| **FULL exact table** — every in-pool candidate exact at the node's own state | **72** | 50.0% | `coldtree_v2_train10000_sweeps/node_<nid>.npz`, 72 files, **838 KB total** |
| **PARTIAL — rank-1 exact, ranks 2…D+1 destroyed** | **0** | 0% | — |
| **NO own measurement** — below-floor children that never ran a search | **72** | 50.0% | — |

**All 72 searched nodes can supply a top-5 blend target today.** That is 72 registered states against
the pilot's 42, and — the important difference — they are the tree's *own* node states, not 35
below-floor sweeps plus 7 lucky single-pick heaps.

### 1.2 Verification performed here (read-only)

* Sweep file set == searched node set, exactly: 0 searched nodes without a sweep, 0 sweeps without a node.
* **72/72: `argmax(gain)` equals the node's banked `splitter`, AND `max(gain)` equals `splitter_gain`
  to < 1e-12.** (Root: `base` 0.15830706760250668, `max gain` 0.020216505387627565 == `splitter_gain`.)
* Candidate counts are the whole available pool, `1000 − len(prefix)`: 1000×1, 999×3, 998×9, 997×13,
  996×16, 995×11, 994×6, 993×4, 992×3, 991×2, 990×2, 989×2.
* Every sweep `sig` (v2) carries `tower` `t2i26c_ep31.pt`, `conc_ckpt` `concept_cv3_ep08.pt`,
  `objective` `"full"`, `mask_pool.policy` `"B"`, `split` `"train"`, `n_pool`, **`pool_sha1`
  `ed67fe28789be5214304370c25987cadb1127e25` — byte-identical to the validation pool sha1 the C1
  pilot used** — plus `n_rows` and `rows_sha1`. `n_rows` matches the node's banked `n_users` at
  **72/72**, so the cohort is pinned by hash, not by size (memory
  `cross-population-index-match-is-silent`).

### 1.3 The "D distribution" — now a counterfactual, not a cost

The heap sidecars (`coldtree_v2_train10000_heaps/`, 72 files, 2.9 MB) still carry the CELF decay:
**65 of 72 have level>0 entries; 15,434 stale entries in total** (49 internal nodes 9,212, max 780;
23 searched leaves 6,222). Had the `save_sweep` fix not landed, this tree would have needed a
**15,434-evaluation repair** — the direct analogue of the val tree's 14,123. It does not, because
the `.npz` sweeps are written once after the seed pass and never rewritten.

### 1.4 What is still NOT covered

* **The 72 below-floor inheritors.** Cohorts 1–193 users (median 38, sum 4,193); none reaches
  `min_leaf` 200. Nothing was ever measured at their states. The tree itself fits nothing there
  (they inherit the parent tail), so the honest default is to let C1 inherit the parent's blend too.
* **Tail states.** `total_L` 15 with 95 leaves gives **870 tail-state slots** (positions
  `depth`…14). The sweeps are one per node at the node's own state only. From the heaps, a tail state
  at level *k* has, exactly as in the audit, only its own pick plus the entries that ended at level
  *k*: over the 526 populated tail levels, **exact candidates per tail state: min 2, p25 6, median
  15, p75 37, max 339, mean 30.3** of ~985 — and they are enriched for LOW gains, so they are not a
  top-m. **For tail states the only guaranteed-exact quantity is still the argmax, i.e. the one-hot.**

---

## 2. The repair cost for THIS tree

### 2.1 Top-m at the partial nodes: **there are none. Zero evaluations, zero hours.**

The audit's headline repair (14,123 evals ≈ 1.75 h fitted / ≈ 2.3 h at the observed slippage)
**does not transfer** — not because the arithmetic changes but because the class it repairs is empty
here. The counterfactual for this tree would have been **15,434 evaluations ≈ 2.00 h fitted /
≈ 3.29 h at this tree's own observed rate**; that work was avoided at the source.

### 2.2 Cost anchors, with provenance

| anchor | value | source |
|---|---|---|
| fitted model | `sec_per_eval(n) = 0.0858 + 0.8157·(n/1000)`, R² 0.985 over 22 real nodes | `policy_labels.py:97–105` |
| … at n = 10,000 | **8.24 s/eval** | same |
| this tree's own recorded rate at n = 10,000 | **13.543 s/eval** | `coldtree_v2_train10000.json:baseline_sec_per_eval_10k` |
| ⇒ **observed / fitted = ×1.64** on the train split | | derived here |
| the one recorded validation sweep | 42 states, ~41.8k evals → **252.3 min**, slippage ×1.29 | `policy_labels_deep.json:minutes` |
| whole 144-node train build | **243,005.7 s = 67.5 h**; 90,857 heap evaluations ⇒ 2.67 s/eval averaged over a size-varying cohort | tree artefact + heaps |

⚠ **Standing rule: cost estimates on this box are not properties of a method** — over-projected 4×,
under-projected 2.6×. Use ×1.64 as the *measured* train-split correction, not as a guarantee.

### 2.3 Optional measurement campaigns, if C1-final wants more than the 72 states

| campaign | evaluations | fitted hours | ×1.64 |
|---|---|---|---|
| repair the partial nodes (**empty class**) | 0 | 0 | 0 |
| sweep the 72 below-floor inheritors at their **fit** cohorts (1–193 users) | 71,631 | **2.65 h** | **4.36 h** |
| …same, at their **bank** cohorts (7–2,047 users, see §5) | 71,631 | 10.09 h | 16.58 h |
| re-sweep the 72 node states on the **85,289-user bank cohort** (bigger, more stable targets) | 71,673 | 116.9 h | **192 h** |
| all 870 tail states, full pool, at leaf fit cohorts | 861,620 | 42.5 h | **69.8 h** |

**Recommendation: run none of them for C1-final.** The 72 free node states are already 1.7× the
pilot's registered population; the inheritor sweep (4.4 h) is the only one with a defensible
cost/benefit, and even that buys states with 1–193 users where the tree itself declines to fit.

---

## 3. The trajectory bank rebuild

### 3.1 `policy_traj.py` is the WRONG tool for this tree — it is E1-era and hard-wired

`src/instrument/policy_traj.py` cannot read this tree without edits:

* `TREE_JSON`/`TREE16_JSON` are module constants pinned to `ndcgtree.json` + `ndcgtree16.json`
  (lines 65–66); `load_tree()` asserts `total_L == 16` and `split == "validation"` (lines 84–87).
  This tree is `total_L` 15, `split` "train".
* `build_trajectories()` reads `t16["leaf_table"][i]["own_sequence"]` (line 166). **This tree's
  `leaf_table` has no `own_sequence` field** — its entries are
  `nid, depth, n_users, path, branch_from, stop_reason, sequence, diff_full@10, diff_tail@10,
  sec_per_eval, n_evals, genre_face_validity`.
* Its `descend()` assumes every ternary child exists. It does not here: the v2 stopping rule creates
  no node for an empty branch, and a train user can land there (see §3.3).

**The current, correct generator is `src/instrument/m6c_distill_gen.py`** (or `m4_distill_gen.py`),
which routes through `src/instrument/tree_route.route(...)` — the single continuation rule
(ancestor tail → global static) added after `fitted-tree-escapes-its-own-support`.

### 3.2 The bank ALREADY EXISTS for this tree

`experiments/battery/m6/traj_dj/cold_*.npz`, 8 slice files, **852 MB**, written 2026-08-29 15:14–15:29.

| field | shape / dtype | note |
|---|---|---|
| `rows` | (n,) int32 | canonical train uid |
| `label_idx` | (n, 15) int16 | pool index of the tree's question at each position |
| `outcome` | (n, 15) int8 | like / dislike / noanswer, per user |
| `z` | (n, 15, 200) float32 | latent after folding opener + positions 0…t−1 |
| `valid` | (n, 15) bool | false after a continuation-rule fallback |
| `anchor`, `half`, `opener`, `n_fallback` | scalars / (n,) | `anchor = −1`, `half = "A"`, `opener` empty for cold |

Contract evidence gathered here:
* **346 distinct label indices in the bank == 346 distinct questions across the tree's leaf
  sequences.** (The `m4/traj/cold_*` bank has 383 — that is the *old* `coldtree_v2`, 164 nodes.)
* Descending the tree with the banked `outcome` array visits **144 of 144 nodes**, zero
  zero-occupancy nodes.
* **Recorded runtime: 15:12:20 → 15:29:10 = ~17 minutes** for all 8 slices × 7 trees (`m6c_gen_dj.out`).
  A cold-only regenerate is a fraction of that; a full-cohort cold regenerate is ~20–25 min.
* Recorded distillation runtime downstream: `m4b_policy.py`, 6 epochs, 1.8 M examples/epoch,
  **15:29:13 → 16:15:47 = 46.5 minutes** (`m7dj_train.out`), ep5 mean dev top-1 0.9910.

For contrast, the original E1 bank (`experiments/battery/policy_traj/`, **1.9 GB**, `z_train.npy`
alone 1.80 GB) covered 140,768 users × 16 positions × 200 dims on the 22-node tree.

### 3.3 The one caveat on reuse: the cold bank is a SUBPOPULATION

`m6c_distill_gen.py:163` sets the cold arm's rows to `rows_loc[~in_any]` — train users who endorse
**none** of the six anchor concepts. That is **85,289 of 140,768 (60.6%)**, and it is a
by-design property of the forest split (`m4b_policy --no_cold_all`), stated in the training log:
*"cold bank 85,289 users (a SUBSET by design: endorses no anchor)"*.

Whether this matters for C1 depends on the claim. It gives **1,279,335 (user, position) examples**
and node occupancy of **min 356 / p25 2,425 / median 3,393** at the 72 searched nodes and
**min 7 / median 382 / max 2,047** at the 72 inheritors — i.e. every state C1 needs is populated,
with ~10× the fit cohort. But it is a taste-selected 60% of train, so a C1 result read as "the
policy's emission at ordinary states" inherits that selection.

**Regenerating the cold arm over all 140,768 users** is a one-line change (drop the `~in_any` filter)
plus ~20–25 min and **~1.42 GB of disk**, which C: cannot comfortably absorb (§4).

Escape rate on the banked cohort: **15 users of 85,289 (0.018%)**, all at one site,
`root.dislike.like.dislike.noanswer` — a branch the fit cohort never populated. The continuation rule
in `tree_route.py` already handles it and `n_fallback` was 0 in the banked slices.

### 3.4 Warm start — **nothing blocks it architecturally**

`experiments/battery/m4/m7dj_policy.pt` (the Act-5 forest policy, epoch 5) and
`experiments/battery/policy_main_equivariant_both.pt` (E1, epoch 2, the C1 pilot's parent) are
**structurally identical**: `arch="equivariant"`, `state="both"`, `d_phi=205`, `n_pool=1000`,
`d_z=200`, **526,542 parameters**, same `state_dict` key set including
`head.q.{0,2}`, `head.k.{0,2}`, `head.b` — the exact tensors `c0_geometry.geometry()` /
`bank_vectors()` read. So the C1 pilot's geometry machinery transfers unchanged.

Four small, real differences to handle in the harness:

1. **Key name**: E1 stores the weights under `"model"`, m7dj under `"sd"`.
2. **Normalisation**: E1 stores `norm_mu` / `norm_sd`; m7dj stores a single `norm` of shape (2, 1, 205).
3. **Extended phi bank**: m7dj carries `ext` of length **1004** — the 1,000-candidate pool plus four
   anchor concepts `[1,68] [1,581] [1,417] [1,922]` — with `a_phi` / `a_pool` per group
   (`anchor_encoding = "none"`, `drop_p = 0.5`). C1 must decide explicitly whether it scores against
   1,000 or 1,004, and against the 19,390-entry full bank (`c0_geometry.N_BANK`) for the snap.
4. **`c1_pilot.py` asserts the C0 checkpoint sha256** `9c4c61e7…6003058` against `c0_geometry.json`,
   and `c0_geometry.py` asserts the **validation** m=0 intercepts `VAL_M0_FULL = 0.15686538…` /
   `VAL_M0_TAIL = 0.02747733…`. Both are E1/validation objects. Using m7dj on train states needs
   **C0 re-run on the new checkpoint** (recorded C0 wall clock: **17.6 minutes**) and a **train-side
   canonical snap** — the train m=0 full is **0.15830706760250668**, banked as `base` in
   `sweeps/node_root.npz`. Do not silently carry the validation constant across.

---

## 4. Disk and box

| | |
|---|---|
| **C: free** | **2.46 GB** of 510.8 GB — the binding constraint |
| existing E1 trajectory bank `experiments/battery/policy_traj/` | **1.9 GB** (`z_train.npy` 1.80 GB, `z_val.npy` 128 MB) |
| existing cold bank on THIS tree `m6/traj_dj/cold_*.npz` | **852 MB** (whole `traj_dj` 1.46 GB) |
| other banks | `m4/traj` 2.2 GB · `m6/traj` 1.6 GB · `m6/traj_forest` 1.6 GB · `m6/traj_fair` 902 MB · `m6/traj2k` 906 MB · `m6/traj_dj` 1.4 GB |
| the 72 gain sweeps | **838 KB** (~11.6 KB/node) |
| `experiments/paperC/` | 58 MB |

**Does a second bank fit?**

* **Reusing `m6/traj_dj/cold_*` costs 0 bytes.** This is the only option that is unambiguously safe today.
* A **full-cohort (140,768-user) cold bank** would be ≈ **1.42 GB**, leaving ~1.0 GB free. That is
  inside HARD RULE 11's spirit but not its comfort zone: `policy_traj.py` itself refuses to start
  unless `free > need × 1.05`, and a long run that fills the disk loses everything.
* An **E1-style dense `z_train.npy` at 140,768 × 15 × 200 float32 = 1.69 GB uncompressed** does
  **not** fit. (The m6c banks are `savez_compressed` per slice, which is why 85,289 users cost
  852 MB rather than 1.02 GB.)

**Free space first if the full cohort is wanted.** Obvious candidates, all superseded and none read
by the current pipeline as far as this scoping checked: `m6/traj` (1.6 GB, pre-disjoint),
`m6/traj_forest` (1.6 GB), `m6/traj_fair` (902 MB), `m4/traj` (2.2 GB, the old val cold tree).
⚠ Per memory `anchor-provenance-hole-pbc-recvae`: **grep for the path before deleting** — byte
redundancy is not proof a path is unreferenced.

---

## 5. Held-out STATES (not users) — what the tree's shape permits

The pilot's own "owed next" list names this: *"More states … or a held-out state split"*, because
42 states with 526k parameters let a **shuffled** target fit to cos 0.99. With 72 states the
memorisation risk is smaller but not gone, so a state-level holdout is the control that separates
"learned the gain geometry" from "memorised N points".

All counts below are over the 144 nodes; **fit-users** = the tree's own cohort sizes,
**bank-users** = occupancy in the existing 85,289-user trajectory bank.

### (a) By root subtree — the cleanest structural holdout

| subtree | nodes | of which searched | fit-users | bank-users |
|---|---|---|---|---|
| `root` | 1 | 1 | 10,000 | 85,289 |
| `root.like.*` | 28 | **15** | 10,091 | 43,343 |
| `root.dislike.*` | 73 | **36** | 27,721 | 264,885 |
| `root.noanswer.*` | 42 | **20** | 14,353 | 153,479 |

Holding out `root.like.*` leaves **57 searched training states / 15 held out (20.8%)**; holding out
`root.noanswer.*` gives 52 / 20 (27.8%). ★ This is the strongest option: the held-out states share no
prefix with any training state beyond the root question, so it is a genuine transfer test — and it
directly answers the pilot's structural-selection critique (the pilot's 42 were **27/42 under
`root.dislike`**, a ~2:1 over-representation). Here `root.dislike` is 36 of 72 searched nodes
(50%) — still heavy, but it is the tree's real mass, not a coverage artefact.

### (b) By depth — a graded extrapolation test

| held out | searched nodes held out | searched nodes kept | fit-users held out | bank-users held out |
|---|---|---|---|---|
| depth ≥ 3 | 59 | 13 | 27,972 | 253,955 |
| depth ≥ 4 | 46 | 26 | 19,248 | 179,513 |
| **depth ≥ 5** | **30** | **42** | 11,843 | 115,952 |
| depth ≥ 6 | 19 | 53 | 7,255 | 73,146 |
| depth ≥ 7 | 13 | 59 | 4,693 | 47,917 |
| depth ≥ 9 | 6 | 66 | 1,819 | 19,845 |

**depth ≥ 5 is the natural cut**: a 42/30 split, and 42 is exactly the pilot's registered population,
so the training arm is directly comparable. ⚠ Depth is confounded with cohort size (deep node
cohorts are small, hence noisier targets) — a failure at deep states would be ambiguous between
"cannot extrapolate" and "the target is noise".

### (c) By last branch — holds prefix depth roughly constant

| last branch | nodes | searched | fit-users | bank-users |
|---|---|---|---|---|
| `like` | 49 | 16 | 11,438 | 72,250 |
| `dislike` | 49 | 26 | 20,298 | 194,600 |
| `noanswer` | 45 | 29 | 20,429 | 194,857 |

Holding out all `*.like` node states costs 16 of 72 (22.2%) and keeps the depth distribution roughly
matched. Weaker than (a) as a transfer test — an ancestor of a held-out state is usually still in the
training set — but the fairest for isolating "does the emission depend on the last answer".

### (d) Searched vs inherited — a capability test, not a holdout

72 searched (fit-users 57,972 / bank-users 509,822) vs 72 inheritors (4,193 / 37,174). The inheritors
have **no measured target at all**, so they cannot be scored against a blend; they are usable only as
a *behavioural* probe ("does the emission at an unmeasured below-floor state resemble its parent's
blend?"). Worth reporting, never a registered cell.

### (e) A seeded random state split

72 states support a 48/24 or 36/36 draw. Cheapest to implement, but with nested cohorts (a node's
users are a subset of its ancestor's) a random split leaks: 5 of the pilot's 42 states were ancestors
of 20+ others and the same nesting holds here. **Prefer (a) or (b).**

**Recommendation:** register **(a) `root.like.*` held out** as primary (57/15, disjoint prefixes) and
**(b) depth ≥ 5** as the secondary graded reading. Report (d) descriptively.

---

## 6. Risks and unknowns to be ruled on BEFORE any run

1. ⛔ **Disk. 2.46 GB free.** No run starts until this is settled. Reusing the existing bank needs
   0 bytes; a full-cohort rebuild needs ~1.42 GB and leaves ~1.0 GB. Decide, and if freeing, grep
   before deleting.
2. ⛔ **Which cohort is the bank?** The existing cold bank is **85,289 users who endorse no anchor** —
   a selected 60.6% of train. Accept it (free, contract-verified, all 144 nodes populated) or
   regenerate over 140,768 (~25 min + 1.42 GB). This changes what the C1 claim is *about* and must be
   fixed before the fit, not after.
3. ⛔ **Target cohort size — the pilot's own unrepaired critique.** The 2026-08-24 repair pass found
   the pilot's top-5 **ingredients** are majority resampling noise (bootstrap overlap 1.905 of 5; all
   five survive a majority of resamples at **0 of 35** states; split-half mean-removed direction
   cosine 0.656) and prescribed *"targets built on larger cohorts (the retrained tree) before the word
   'measured' carries weight"*. **This tree does not fix that**: its node cohorts are **min 200,
   p25 269, median 356, max 10,000, with 43 of 72 below 500 users** — comparable to, not larger than,
   the pilot's. The only fix is the 192-hour re-sweep on the bank cohort (§2.3). ★ **Either pay it,
   restrict the registered states to the shallow high-n nodes, or write the instability into the
   claim.** Do not let "the retrained tree" be read as having discharged this.
4. ⚠ **The Stage-0 premise gate will not fire — verified for free.** Over all 72 sweeps the top-1
   share of the top-5 weight is **median 0.225** (min 0.207, max 0.327; **0 of 72 states above 0.5**),
   against the pilot's 0.228, and the median `gain₅/gain₁` ratio is 0.805. The gain profiles are as
   flat here as there, so the blend will not collapse onto `e₍₁₎`. Median positive-gain candidates per
   state 484; top-1 gain min 0.00384 / median 0.01089 / max 0.04780.
5. ⚠ **Structural differences from the tree the pilot used, in one list.**
   `total_L` **15** not 16 · **144** nodes not 164 · fitted on **TRAIN** not validation · leaf records
   use **`sequence`**, not `own_sequence` · the **72 targets are node states** (the pilot's were 35
   *below-floor* sweeps + 7 single-pick heaps, i.e. mostly states the tree declined to fit) · the
   train m=0 intercept is **0.15831**, not the validation canonical 0.15687 · one branch
   (`root.dislike.like.dislike.noanswer`) has **no node** and 15 bank users escape into the
   continuation rule.
6. ⚠ **C0 has not been measured on the target checkpoint.** `c1_pilot.py` hard-asserts E1's sha256
   against `c0_geometry.json`. If C1-final warm-starts from `m7dj_policy.pt` (recommended — it is the
   Act-5 policy and architecturally identical), **C0 must be re-run first** (17.6 min recorded) and
   the `VAL_M0_*` snap replaced with a train-side snap. Reusing E1's geometry constants against a
   different checkpoint would be exactly the "numbers from different information contracts" failure.
7. ⚠ **The phi bank is 1,004 not 1,000 in m7dj** (`ext` = pool + 4 anchor concepts, `a_phi`/`a_pool`
   per group). Whether the blend, the snap and the "off-grid" reading operate over 1,000, 1,004 or the
   full 19,390 must be declared in the design sheet, not discovered at run time.
8. ⚠ **Tail states remain one-hot.** 870 of the tree's 1,014 (144 node + 870 tail) supervision slots
   have no blend target and would train against the existing one-hot. A C1 that blends at node states
   and one-hots at tail states is a *mixed* objective; say so, or restrict the loss to node states and
   accept the coverage.
9. ⚠ **Collateral collapse is the known failure mode and it is not gated.** The pilot's fine-tune
   dropped from 163 distinct questions to **4–6** at ordinary states with **0.0000** snap agreement.
   The pilot's own note: *"a replay/regularisation term is not a nicety, it is the whole difference
   between this result and a policy."* Any C1-final must gate this, not report it — and the replay
   data is free, since `traj_dj/cold_*` is exactly the teacher-forced supervision.
10. ⚠ **Criterion 2b needs replacing.** As registered it could not separate true from *permuted*
    targets (shuffle scored the same 33/42). The one statistic that separated was the paired sign test
    on true-vs-deranged fit (full arm 30/42, p = 0.0079). 72 states raise the power but do not fix a
    criterion with no separating power; design the falsifier before the run.
11. ⚠ **Firewall.** Every artefact touched here is `test_touched: false`. The Paper B test look was
    spent on 2026-08-29; **C1-final is a Paper C object and must not reuse it**. The 10,000 test users
    stay unbuilt.

---

## 7. Files this scoping read (absolute paths)

* `C:\dev\phd\casper\experiments\battery\coldtree_v2_train10000.json` — the tree
* `C:\dev\phd\casper\experiments\battery\coldtree_v2_train10000_sweeps\` — 72 full gain tables, 838 KB
* `C:\dev\phd\casper\experiments\battery\coldtree_v2_train10000_heaps\` — 72 CELF sidecars, 2.9 MB
* `C:\dev\phd\casper\experiments\battery\m6\traj_dj\cold_*.npz` — the trajectory bank on this tree
* `C:\dev\phd\casper\experiments\battery\m4\m7dj_policy.pt` — the Act-5 forest policy (warm start)
* `C:\dev\phd\casper\experiments\battery\policy_main_equivariant_both.pt` — E1 (the pilot's parent)
* `C:\dev\phd\casper\src\instrument\m6c_distill_gen.py` — the current bank generator
* `C:\dev\phd\casper\src\instrument\tree_route.py` — the single continuation rule
* `C:\dev\phd\casper\src\instrument\policy_traj.py` — E1-era, pinned to `ndcgtree*.json`, do not reuse as-is
* `C:\dev\phd\casper\src\instrument\c1_pilot.py`, `c0_geometry.py`, `policy_net.py`, `m4b_policy.py`
* `C:\dev\phd\casper\experiments\battery\m6c_gen_dj.out` (17 min) · `m7dj_train.out` (46.5 min)
