# C1 GAIN-TABLE AUDIT — can the blend targets be built from banked artifacts?

**Read-only investigation, 2026-08-21. Nothing was run, nothing was retrained, no test data was
named.** Every number below is read out of files already on disk, or quoted from a recorded runtime.

**Question:** `docs/design/PAPER_C_DESIGN.md` §C1 asserts *"Per-state measured gains already exist
(`policy_labels.py`)"*, and proposes to replace E1's one-hot target with a gain-weighted blend of the
**top-m measured candidates at each state**. Does the bank hold per-candidate measured gains at every
node of the frozen tree, sufficient to build that target without re-measurement?

---

## ★ BOTTOM LINE — **PARTIALLY**, and the assertion in the design sheet is *half* true

`policy_labels.py` exists (`src/instrument/policy_labels.py`) and **can** produce exact 1,000-candidate
gain vectors — but only in `--mode sweep`, which pays for them. The mode that actually built E1's
labels is `--mode banked`, and **banked mode emits the argmax only** — it is a one-hot extractor by
construction ("Scoring cost: ZERO"). So the sentence *"the per-state measured candidate gains are
exactly what the labelling harness already produces"* is **false for the artifact E1 was trained on**
and **true only for the 42 states that were separately swept**.

For the 164-node `coldtree_v2`:

| coverage class | nodes | share |
|---|---|---|
| **FULL exact table** — all ~1,000 candidates exact at the node's own state | **42** | 25.6% |
| **PARTIAL** — rank-1 exact, ranks 2…D+1 **destroyed**, ranks D+2…1000 exact (median D = 192) | **55** | 33.5% |
| **NO own measurement** — below-floor children that never ran a search | **67** | 40.9% |

**A top-m blend with m ≥ 2 is buildable, unmodified, at 42 of 164 nodes.** At the other 122 it is not:
at the 55 partial nodes the missing entries are *precisely the top of the ranking* (proof below), and
at the 67 inherited nodes nothing was ever measured.

**But the repair is cheap and targeted, not a campaign.** The destroyed candidates are individually
*identifiable* from the heap sidecars (they are exactly the entries with `level > 0`), so the fix is
14,123 evaluations, not 55,000 — **≈1.75 model-hours** by the project's own fitted cost model
(≈2.3 h after the 1.29× slippage the one recorded sweep actually showed). See the caveat on cost at
the end.

⚠ **Separate and possibly bigger issue: E1 was not trained on `coldtree_v2`.**
`policy_traj/manifest.json` records `"tree": "ndcgtree.json+ndcgtree16.json"`, `"tree_nodes": 22`,
`total_L: 16`. `coldtree_v2` (164 nodes, `total_L` 15, built 2026-08-18) has **no trajectory bank**.
If C1 is to distil `coldtree_v2`, a new `policy_traj` bank is needed regardless of gain tables. If C1
stays on E1's actual teacher (the 22-node tree), coverage is **7 full tables out of 202 label states**
— far worse. Choose the teacher before costing C1.

---

## 1. Where the per-candidate gains live, and the exact-vs-stale rule

Two mechanisms bank per-candidate gains. They are not equivalent.

### (a) CELF heap sidecars — near-complete, but the top of the ranking is eaten

`src/instrument/ndcg_tree.py` calls `greedy_certified`
(`src/instrument/strategy_channel_suite.py:868`) once per node with `heap_path=<dir>/node_<nid>.json`.
The seed pass evaluates **every** pool candidate at the node's state and pushes
`(-(v − base), q, 0)`; the sidecar is rewritten after the seed pass **and after every pick**.

The third tuple element is the level at which that entry was last evaluated. **An entry is an exact
node-state gain if and only if `level == 0`.** `policy_p1b.py:130` asserts exactly this:

```python
assert int(lvl) == 0, f"{nid}: a heap entry is not at level 0 -- not an exact gain"
```

**The destroyed entries are exactly the top of the node-state ranking — this is provable, not
empirical.** When a candidate `c` is popped for the *first* time during the tail greedy, its heap key
is still its node-state gain `g₀(c)`, and `heappop` returns the maximum, so `g₀(c) ≥ g₀(x)` for every
`x` still in the heap — including every entry that ends the run at `level == 0`. Re-evaluation
overwrites the key. Therefore, at a node whose tail ran and re-evaluated **D** distinct candidates:

> **rank 1 is exact** (it is `splitter_gain`, popped at `level == 0` and recorded), **ranks 2 … D+1
> are irrecoverably lost**, and **ranks D+2 … 1000 are exact.**

Since `D ≥ 13` at every one of the 55 partial nodes (median 192, max 826), **not even the top-2 is
available at any node that grew a tail.** m = 3, m = 5 and m = 10 are all equally unavailable there —
the coverage number does not depend on m.

The one node type that escapes: `needs_tail == False` (all non-empty children above the min-leaf
floor) runs `L = 1` only, so the sidecar holds the untouched seed heap — a complete exact table. That
is the same fact `policy_labels.py` records in its header:

> *"PERSIST THE FULL SWEEP VECTOR AT STEP 0, BEFORE THE FIRST POP — CELF's pops overwrite their own
> step-0 record, which is why only 7 banked states carry a complete level-0 vector."*

### (b) Deep sweeps — genuine full 1,000-candidate exact vectors

`policy_labels.py --mode sweep --states deep` measured **42 states, ~996 in-pool candidates each,
nothing sampled** (HARD RULE 1). Output: `experiments/battery/policy_labels_deep_partial/<sid>.json`
plus a per-user `<sid>.f32`, consolidated into `policy_labels_deep.json` (`minutes: 252.3`).

**These states are contract-identical to `coldtree_v2` nodes.** Same tower `t2i26c_ep31.pt`, same
`concept_cv3_ep08.pt`, same policy-B mask fingerprint `c282962b49a2d67ef47ff07544519f20`, same
`pool_sha1 ed67fe28…`. **35 of the 42 map onto a `coldtree_v2` node id with an identical cohort size,
and at the 15 of those that are searched nodes the sweep's NDCG-full argmax *and its value* are
bit-identical to the tree's banked `splitter` / `splitter_gain` — 15/15, zero mismatches.** That is a
hard cross-verification of the contract, not an assumption.

The 7 non-matching sweeps are children of nodes that `coldtree_v2` left as leaves (e.g.
`root.noanswer.dislike.*`); they are paid-for and correct, simply not `coldtree_v2` states.

---

## 2. Per-node coverage — the numbers

### 2.1 `coldtree_v2.json` (164 nodes = 77 searched + 87 below-floor inheritors)

Every one of the 77 searched nodes **does** have a heap sidecar. They are split across two
directories because `coldtree_v2` is a *resume* of the earlier 22-node build (its root splitter
`[1, 114]` and root gain `0.020490556975873986` are bit-identical to `ndcgtree.json`):

* `experiments/battery/coldtree_v2_heaps/` — **55** files, `sig.v = 2`, written 2026-08-18.
* `experiments/battery/ndcgtree_heaps/` — **22** files, `sig.v = 1`, written 2026-08-12; these are the
  22 nodes `coldtree_v2` reused ("already complete in the sidecar"). Bit-identical gains, but the v1
  signature omits `tower`/`conc_ckpt`/`mask_pool`, so it is a **weaker contract** — verify against
  `nodes[nid].splitter_gain` before use (all 7 that matter pass; see below).

| source | nodes | exact candidates at the node's own state |
|---|---|---|
| deep sweep (`policy_labels_deep_partial`) | **35** | 994–997 of ~997 (the whole in-pool set) |
| single-pick heaps, `D = 0` | **7** | 1,000 / 999 / 998 / 997 (whole pool) |
| — `root`, `root.like`, `root.dislike`, `root.noanswer`, `root.dislike.dislike`, `root.dislike.dislike.dislike`, `root.dislike.dislike.dislike.noanswer` | | (all from `ndcgtree_heaps`, all `sig.v = 1`) |
| **FULL, union (disjoint)** | **42** | |
| tail-contaminated heaps, `D > 0` | **55** | rank 1 + ranks D+2…1000; **ranks 2…D+1 lost** |
| below-floor inheritors, never searched | **67** | none |

`D` over the 55 partial nodes: **min 13, median 192, max 826.** The two worst are
`root.noanswer.noanswer.like` (D = 651) and `root.like.like.noanswer.dislike` (D = 569).

Of the 87 below-floor inheritors, **20 are covered by the deep sweep** (they were the below-floor
children the sweep was designed to measure) and 67 are not. Their cohorts run from **1 to 145 users**
— arguably C1 should mirror the tree and let them inherit the parent's blend, since the tree itself
fits nothing below the floor.

### 2.2 Tail states (a node's 2nd…kth question) — essentially uncovered

E1's supervision is not only node states; a leaf carries a fitted tail, one label per position. For a
tail state at level k, the exactly-known candidates are `{the pick at level k}` ∪ `{heap entries whose
final level == k}`. Across the 54 tail-growing nodes in `coldtree_v2_heaps` (463 tail-state slots):

> **exact candidates per tail state: min 0, p25 3, median 12, p75 36, max 356, mean 27.1** — out of
> ~985. And they are *not* known to be the top-m; the entries re-evaluated at deeper levels dominated
> them at level k.

**For tail states, the only guaranteed-exact quantity is the argmax — which is the one-hot E1 already
has.** A blend at tail states needs a sweep.

### 2.3 The six `m3v2_*` teacher trees (secondary)

All 32 searched nodes across the six teachers have heap sidecars; **only one has `D = 0`** (one
`m3v2_dialogue` node). Minimum `D` per teacher: arthouse 293, dialogue 244, gore 279, inspiring 95,
marvel 232, teenagers 162.

| teacher | nodes | leaves | searched | heaps | FULL exact | partial |
|---|---|---|---|---|---|---|
| arthouse | 7 | 5 | 3 | 3 | 0 | 3 |
| dialogue | 9 | 6 | 5 | 5 | **1** | 4 |
| gore | 9 | 6 | 5 | 5 | 0 | 5 |
| inspiring | 21 | 14 | 11 | 11 | 0 | 11 |
| marvel | 4 | 3 | 3 | 3 | 0 | 3 |
| teenagers | 10 | 7 | 5 | 5 | 0 | 5 |

**Extending C1 to the forest policy requires a measurement campaign — the bank gives it essentially
nothing** (1 blendable node of 60). No deep sweep was ever run on the teacher cohorts.

---

## 3. What re-measurement would cost

**The script already exists and needs no new code:**

```
python src/instrument/policy_labels.py --mode sweep --split validation --states {root|tree|deep}
```

`--states` currently offers `root`, `tree`, `deep` (`policy_labels.py:216–247`). A `coldtree_v2` state
set would be a new selector — a small addition, but the sweep loop, the per-candidate partial, the
`.f32` per-user sidecar and the resume contract are all in place.

**Recorded cost anchors (do NOT treat these as properties of the method — the standing project rule is
that cost estimates on this box are unreliable; this box has over-projected 4× and under-projected
2.6×):**

* **Fitted model**, `policy_labels.py:97–105`: `sec_per_eval(n) = 0.0858 + 0.8157·(n/1000)`, fitted
  over all 22 real tree nodes, R² = 0.985, max residual 0.79 s. Anchor: **8.48 s/eval at n = 10,000,
  i.e. 141.8 min for one 1,000-candidate sweep at the root.**
* **The one recorded sweep**: 42 states, cohorts partitioning the 10,000 validation users, ~41.8k
  evaluations → **252.3 minutes** (`policy_labels_deep.json:minutes`). The fitted model predicts
  196 min for the same work, so the observed slippage is **×1.29**.
* `coldtree_v2.json` records `baseline_sec_per_eval_10k: 13.543` and `seconds: 24944.1` for the whole
  164-node build.

Applying the fitted model (and, in brackets, ×1.29):

| repair | evaluations | model-hours |
|---|---|---|
| **targeted**: re-measure only the D destroyed candidates at the 55 partial nodes | 14,123 | **1.75 h** (≈2.3 h) |
| full re-sweep of the 55 partial nodes | 55,000 | 6.6 h (≈8.5 h) |
| full sweep of the 67 below-floor inheritors (cohorts 1–145) | 67,000 | 2.2 h (≈2.8 h) |
| everything missing at node states (122 nodes) | 122,000 | 8.8 h (≈11.4 h) |
| tail states (463 slots) — not costed; adds ~463 × 1,000 evals | ~463,000 | large |

**The targeted repair is the recommendation.** The heap sidecar names the destroyed candidates
exactly (`[negg, [chan, id], level]` with `level > 0`), so a repair pass evaluates a known 14,123
(state, candidate) pairs and writes them beside the existing exact ones. It reuses `genv_mean` at
`prefix = node.path`; nothing about the contract changes.

---

## 4. The data contract C1's target builder would read

### 4.1 Pool index (needed to turn `(chan, id)` into a vocabulary slot)

`ndcg_tree.build_pool(E, 500, 500)` → `pool = item_pool + conc_pool`, **items first, then concepts**.
`chan = 0` is an item (ranked by train popularity `E["ctx"].cnt`), `chan = 1` is a concept (ranked by
`answer_rate`). Order is load-bearing (CELF tie-breaks) and is fingerprinted as
`pool_sha1 = ed67fe28789be5214304370c25987cadb1127e25`. Build the index exactly as
`policy_p1b.py:202` does:

```python
pool, _it, _cc = build_pool(E, 500, 500)
pool_index = {(int(a), int(b)): i for i, (a, b) in enumerate(pool)}   # 1000 entries
```

### 4.2 Full exact tables — `policy_labels_deep_partial/<sid>.json` (42 files + `_sig.json`)

Filename: `sid.replace("@", "_at_") + ".json"`, where `sid = f"{nid}@{pos}"`.

| key | type / shape | meaning |
|---|---|---|
| `sid`, `nid`, `pos`, `kind` | str, str, int, str | `kind` is `below_floor` for all 42 |
| `prefix` | list[[chan, id]], length = `pos` | the asked history defining the state |
| `rows` | list[int], length `n_users` | canonical **validation** uids |
| `n_users` | int | 1 – 1,438 |
| `base` | float | cohort-mean NDCG@10 at the state, before the candidate |
| **`vals`** | dict `"{chan}_{id}" -> [f, f, f, f]` | **`["ndcg_full", "ndcg_tail", "dcg_full", "dcg_tail"]` marginal gains.** 994–997 keys. `vals[q][0]` is the objective the tree optimises |
| `argmax_ndcg_full`, `argmax_dcg_full`, `argmax_agrees` | `"{chan}_{id}"`, same, bool | banked argmaxes under both normalisations |
| `idcg_full_mean`, `spearman_ndcg_vs_dcg` | float | |
| `peruser_file`, `peruser_order`, `peruser_layout`, `peruser_dtype` | str, list[str], `["full","tail"]`, `"float32"` | sidecar `.f32`: record *i* = `peruser_order[i]`, `n_users` FULL then `n_users` TAIL; **record 0 is `"__base__"`** |
| `done` | bool | only `done` states are usable (42 of 43 files; `_sig.json` is the contract) |

Contract file `_sig.json` → `sig`: `{v:1, split:"validation", objective:"full_and_tail_ndcg_and_dcg",
tower:"t2i26c_ep31.pt", conc_ckpt:"concept_cv3_ep08.pt", mask_pool:{policy:"B", fp:"c282962b…"},
n_pool:1000, pool_sha1:"ed67fe28…", states:"deep", state_sha1:"3f94ee3e…"}`.

**C1's blend at these states:** take `vals[q][0]` for all q, sort descending, take top-m, softmax or
normalise the gains as weights, blend the corresponding pool embeddings. Nothing else is needed.

### 4.3 Heap sidecars — `coldtree_v2_heaps/node_<nid>.json` (55) and `ndcgtree_heaps/node_<nid>.json` (22)

| key | type | meaning |
|---|---|---|
| `sig` | dict | v2: `{v, split, objective, tower, conc_ckpt, mask_pool, prefix, n_pool, pool_sha1, n_rows, rows_sha1}`. v1 lacks `tower`/`conc_ckpt`/`mask_pool` |
| `tag` | str | e.g. `"<nid>/split"` or `"<nid>/tail"` |
| `base`, `cur`, `nev` | float, float, int | cohort mean at the state; after picks; evaluations paid |
| `seq` | list[[chan, id]] | the greedy picks made from this state, in order |
| `gains` | list[float] | marginal gain of each pick. **`gains[0]` is exact at the node state** |
| **`heap`** | list of `[-gain, [chan, id], level]` | **`level == 0` ⇒ `-gain` is the exact node-state marginal gain. `level > 0` ⇒ the value is a gain at a DEEPER state and the node-state value is lost.** |

**Reader rule (copy `policy_p1b.load_onmanifold`):** accept a table only if every entry is
`level == 0`; then `gains = {pool_index[seq[0]]: gains[0]} ∪ {pool_index[q]: -negg}` and assert
`len(gains) == sig["n_pool"]` and `argmax == pool_index[seq[0]]`. Cohort rebuild from the trajectory
bank is asserted against `sig["n_rows"]`.

### 4.4 Tree structure — `experiments/battery/coldtree_v2.json`

`nodes[nid]` carries `path` (list[[chan,id]] — **note: for an inheriting child this is the *parent's*
path, it does not include the parent splitter**), `steps`, `gains`, `splitter`, `splitter_gain`,
`tail`, `tail_gains`, `n_users`, `depth`, `child_sizes`, `children_inheriting`, `inherited_from`,
`split`. Top level: `n_nodes: 164`, `n_leaves: 109`, `depth_reached: 11`, `total_L: 15`,
`min_leaf: 200`, `n_pool: 1000`, `pool_policy: "B"`, `sig`, `pool_contract`.

### 4.5 Also on disk, and probably not useful for C1

* `experiments/battery/value_sample_partial/prefix_*.json` — 19 files, 138 random-prefix states,
  **50 candidates each (≈22 in pool)**. A "top-m" drawn from 22 sampled candidates is not the pool's
  top-m; P1b carried this as an explicit teacher-honesty caveat. Use for auxiliary states only, never
  pooled with the full sweeps.
* `experiments/battery/p1b_nevertaught/`, `policy_labels_deep.json` — consolidated readouts of the
  above.
* `ndcgtree16_heaps/` (15 leaf heaps, `D ≥ 145`), `greedy_static_policyB_*_heaps/`,
  `policy_deep_heaps/` — same exact/stale rule applies; none carry a full table.

---

## 5. Recommendation

1. **Decide the teacher first.** `coldtree_v2` has no trajectory bank; E1's teacher is the 22-node
   `ndcgtree16` and has only 7 full tables of 202 label states. This choice dominates the cost.
2. If `coldtree_v2`: **42 of 164 node states are blendable today, for free.** A pilot C1 on those 42
   is a real experiment and costs nothing — and it is the honest way to find out whether the blend
   moves emissions off-catalogue before paying for anything.
3. The remaining 55 node states cost a **targeted 14,123-evaluation repair (~1.75 model-hours fitted,
   ~2.3 h at the one observed slippage factor)** because the missing candidates are individually named
   in the heaps. This is the cheapest correct path; a blanket re-sweep costs 3.8× more.
4. **Tail states and the six teacher trees need a real sweep** — the bank gives them almost nothing.
5. Amend `PAPER_C_DESIGN.md` §C1: `policy_labels.py --mode banked` produces the **argmax only**. The
   gain vectors exist for 42 swept states, not "already, per state".
6. Carry forward the rule this audit re-confirms, already written into `policy_labels.py`:
   **persist the finest-grained quantity the computation naturally produces.** A one-line
   `_save(heap, …)` immediately after the seed pass, *never* overwritten, would have made this whole
   question free.

---

# ✅ RESOLVED AT THE SOURCE — 2026-08-21 (commit `8b3024f`)

**The repair campaign is cancelled, because the destruction no longer happens.**

Both growers now persist the complete per-candidate gain table at every node they search, written
**immediately after the CELF seed pass and before the first pop** — the one moment at which every
heap entry is still at `level == 0`:

| grower | function | used by | sweep path |
|---|---|---|---|
| `strategy_channel_suite.py` | `greedy_certified(..., sweep_path=)` | `ndcg_tree` (cold tree) | `<label>_sweeps/node_<nid>.npz` |
| `group_greedy_bands.py` | `celf_per_user_prefix(..., sweep_path=)` | `m6k_cluster_tree` (group trees) | `m6/sweeps/sweep_<tag>_<nid>.npz` |

Each `.npz` holds `cand` (n×2 int32), `gain` (n float64, exact at the node's own state), `base`,
`n_evals`, `objective`, `prefix` and the contract `sig`. It is written **once and never rewritten**,
so unlike the heap sidecar it cannot decay as the tail greedy proceeds.

Two asserts make silent corruption impossible: the table must cover the **whole pool**, and **every
entry must be at level 0** at write time (i.e. `save_sweep` cannot have run after a pop). A resume
that finds a heap with picks already taken and no banked sweep now logs a loud warning that the
node's ranking is unrecoverable, instead of quietly producing nothing.

**Cost: ~16 KB per node and ZERO extra evaluations** — the numbers were already computed and were
simply being thrown away. A 164-node tree costs ~2.6 MB.

**Verified** by `ndcg_tree --sanity` on both splits:
- `[heapcheck] PASS` — identical sequence and **bit-identical gains** vs the pre-change path; the
  resume-and-extend and different-contract paths are unchanged.
- **sweep-max == `splitter_gain` at 7 of 7 searched nodes, 0 mismatch, 0 nodes missing a sweep.**

**What this does NOT fix.** Below-floor children that never run a search still have no table — the
audit's 67-node class. That is inherent, not a regression: nothing was ever measured at their state,
and manufacturing one costs a full seed pass per node. Any C1 blend must either restrict itself to
searched nodes or pay for those explicitly.

**Status of the two blockers this audit raised:**
1. *Gain tables* — solved prospectively. Every tree grown from now on ships full C1 targets. The
   frozen `coldtree_v2` still has only 42 usable nodes; it is not repaired and does not need to be,
   because it is being **regrown on TRAIN** (`coldtree_v2_train10000`, launched 2026-08-21) so that
   it and the opener-fitted group trees share a fitting population — see memory
   `train-user-contrast-does-not-cancel-inflation`. **That regrow carries the sweeps.**
2. *E1 was not trained on `coldtree_v2`* — **still open and still the bigger issue.** Choose the
   teacher before costing C1; a new `policy_traj` bank is needed either way.
