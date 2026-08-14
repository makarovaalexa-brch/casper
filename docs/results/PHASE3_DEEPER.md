# PHASE 3 — ASKING ONE MORE QUESTION BELOW THE TREE'S FLOOR STOPS

**Status: RUNNING. This file holds everything that is already established — the design, the gates that
have passed, the defects found, and the numbers pre-registered BEFORE the run. The results tables are
marked PENDING and are not to be read as results until they are filled.** Started 2026-08-14.

Design sheet: `docs/design/PAPER_B_POLICY_DESIGN.md`, **Part G.B is binding**. Code:
`src/instrument/policy_deep.py`, `policy_labels.py`, `policy_net.py`. Chain:
`experiments/battery/run_phase3.ps1`.

> **AUTHOR SAID (2026-08-13), on why the policy is wanted at all:** *"policy is nice becuase it is
> generalisable and scalable … so it is supposed to be able to learn from fewer examples … and most
> importantly, keep adapting fter 5 questions."* Phase 3 is the stage that delivers that last clause.
>
> **AUTHOR SAID (2026-08-13, before going to sleep):** *"ok, i am very confused now… this whole design
> is confusing, but i am going to sleep, you proceed as planned and we reassess in the morning."*

---

## What this is asking, in one paragraph

The NDCG tree stopped at depth 5. `TREE_DEPTH_AND_MINLEAF.md` established that this is a property of
the **stopping rule**, not of the data: all 15 stops were floor stops with a strictly positive gain
still in hand (+0.00672 … +0.04251), **not one** stopped because no question helped, and 8,398 users
(84%) sit below a node that had more to ask. Phase 3 asks what happens if you ask one more conditioned
question below every one of those stops — and it asks it **twice**, once through the parametric policy
and once through the tree itself with the floor lowered. The second one is the control that decides
whether the first one means anything.

---

## ★ THE COMPARATOR ASYMMETRY — pre-registered 2026-08-14, before any number existed

A fully re-grown deeper level would re-run the greedy tail from each of the 42 new child states: ~10
further greedy levels × 42 states × ~995 candidates ≈ **33 h on this box**. That was not affordable in
this window and **was not bought**. So the deeper tree keeps its parent leaf's greedy tail:

```
tree    :  path(d) , splitter , s_{d+1} , s_{d+2} , … , s_15                      (16 questions)
deeper  :  path(d) , splitter , g_branch , [same tail, minus g if it is in it,
                                            otherwise minus its LAST pick]        (16 questions)
```

Budget-neutral, branch-conditioned at position d+1, and the dropped question is the greedy tail's
weakest pick (greedy is prefix-optimal). **It is a real deeper tree in the sense that matters — the
question at d+1 is chosen from the child's own measured NDCG — but it is NOT a fully re-greedied one,
and no sentence here says otherwise.**

**The consequence, which belongs in the headline and not a footnote.** The control may differ from the
frozen tree by **exactly one question per user**. The fine-tuned policy, trained on the *same* deep
labels, may change its pick at **any** position past the floor stop. **The two arms do not have equal
freedom.** A policy win is therefore readable two ways — "the policy exploits these labels better" or
"the control was structurally handicapped". This project has been burned by comparator handicap twice
(Hu & Yu's headline was one; our own policy-A contamination was the same shape), so it is written down
before the numbers exist.

**The measurement that decides which reading is right** is how many positions the policy *actually*
changes per user. `policy_net.py` reports the full histogram, the median, the mean among deviators,
the first- and last-deviation histograms and the per-position share of users differing.

- If the policy also moves ~1 position → the asymmetry is theoretical, the comparison is clean, say so.
- If it moves many → the comparison measures the policy's extra **freedom** as much as its extra
  **skill**, and the write-up must say that too.

**The only claimable statement, if anything lands:** *a tree can apply a deep label at one node; the
policy can generalise it across states.* **Never a bare "the policy beats a deeper tree."**

---

## ★ THE OVERFITTING EXPOSURE — measured from banked artifacts BEFORE any scoring

Stating this before the run is what makes a null interpretable afterwards. If the deep labels turn out
to be noise, **these numbers are the explanation, not a post-hoc excuse.**

| | |
|---|---|
| below-floor states | **42**, partitioning the 10,000-user cohort exactly |
| state sizes | min **9**, p25 55, median **154**, p75 311, max 1,438 |
| states under the 200-user floor | **27 of 42**, covering **2,571 users (25.7%)** |
| states under 50 users | **10**, covering 303 users |

An argmax over ~995 candidates measured on 9 users is noise, and the floor existed to prevent exactly
that. `policy_deep.py --mode control` **recomputes every one of these figures live and refuses to
proceed if any moved** — a doc and its artifact must read the same field.

---

## The comparators, and the one that actually matters

| comparator | validation | test |
|---|---|---|
| `greedy_static_policyB_combined` (K=16) | 0.25561016 | — |
| the frozen tree, over that static | **+0.02794** | +0.01035 |
| **bank-alone** (combined pool − items-only pool, equal budget) | **+0.01280 full / +0.02255 tail** | +0.01430 / +0.02199 |

**F.4's bar is the BANK, not the tree**, and the two numbers above are on *different cohorts*. A
validation gain is never compared to a test bar directly. Using the tree's own measured val→test
shrinkage on full (**2.70×**), clearing bank-alone on test needs roughly **+0.0386 validation gain over
static** — about **+0.011 above E1's +0.02794**, which is exactly the pre-registered G-E2-pre no-go
bar. The two bars agree, which is reassuring about both.

**The spanning regression gate (design sheet G, HARD RULE 8):** no stage may score below the previous
stage's validation number. Phase 3 must not fall below **E1's 0.283549718 full (+0.02794)**.

---

## Gates passed so far

| gate | result |
|---|---|
| **canonical snap** (m=0 validation intercept) | pending in the control; asserted `== 0.15686538` before anything is scored |
| **G-L0** pool contract | ✅ **PASS** — `{"policy":"B","n_rows":10000,"n_items":18359,"nnz":1356475,"per_user":135.6475,"fp":"c282962b49a2d67ef47ff07544519f20"}`, identical to the banked build |
| pool identity | ✅ 500 items + 500 concepts = 1,000 candidates, same ids in the same order, **no truncation** |
| leak | ✅ 0/10,000 validation users have a held-out target inside the routing lookup |
| **G-L3** ternary partition recompute | ✅ enforced per parent node inside state enumeration |
| **firewall** | ✅ `--split` accepts `validation` only; every artifact stamps `test_touched=false`; **no test look has been taken** |

---

## Three defects found by running this, and the rules they leave behind

None of these was visible from reading the code. All three were found in the first ten minutes of
actually executing it.

### 1. `policy_labels.py` had never been run, and its first gate refused

`ndcgtree.json` **predates the `pool_contract` field and carries none**, so G-L0 fired its own rule —
*a missing field is a refusal, never a pass* — and stopped the run in 52 seconds. The tempting fix was
to skip the gate when the field is absent, which is precisely the failure the rule exists to prevent.
**The gate was not relaxed; it was routed** to `policy_traj/manifest.json`, which banks
`val_pool_contract` and `pool_sha1` and was produced by applying this same frozen tree and reproducing
all 7 banked ternary partitions exactly — a strictly stronger witness. The pool is additionally
asserted candidate-for-candidate, by sha1, and against every pool field `ndcgtree.json`'s own `sig`
can witness.

> **Rule:** a gate that cannot find its reference is not a gate to skip; it is a reference to replace
> with a stronger one.

### 2. ★ PERSIST THE FULL SWEEP VECTOR AT STEP 0, BEFORE THE FIRST POP

CELF pops the candidate it selects and re-pushes the rest stamped at a higher level, so **every run
that made a sequence of picks overwrote its own step-0 record**:

| run | level-0 entries retained | picks |
|---|---|---|
| `greedy_static_policyB_combined_heap` | **274** of ~1,000 | 16 |
| `greedy_static_policyB_items` | 259 | 16 |
| `greedy_static_policyB_concepts` | **10** | 16 |
| `ndcgtree16` leaf sidecars | 546 and below | 12–16 |
| the 7 **internal** tree nodes (exactly one pick each) | 996–1,000 — **complete** | 1 |

Those runs *did* evaluate ~1,000 candidates at their step-0 state. We paid for it in CPU and the
artifact did not preserve it. **`--mode sweep` has no CELF loop** — it is exhaustive and flushes after
every candidate — so phase 3's 42 vectors were never at risk; what was missing was the *consumable
form*. `--mode labels` now writes one `lift_head.level0()`-compatible sidecar per state into
`experiments/battery/policy_deep_heaps/`, carrying the complete level-0 vector plus all four
normalisation readings. **Complete, unbiased vectors available to the value head go from 7 states to
49, at zero extra evaluations** — which matters because the ruling of 2026-08-14 is that the value
head's failure was *label starvation, not a representational limit*.

### 3. "Persist everything, always" degraded with progress and became the crash

The single `*_partial.json` was serialised **in full** after every candidate. At state 1 that is 62 KB;
by state 42 it would rewrite ~42,000 gain vectors (~2 MB) about 1.4 times a second. The sweep died at
candidate 600/997 with `PermissionError: [WinError 5]` out of `os.replace(tmp → partial)` — on Windows
a rename fails if any process holds the target for an instant, and a permanently-hot file is a standing
AV target. It is now **one partial file per state**, plus a retrying rename. Cadence, contract check
and durability are unchanged, and a legacy single-file partial is **migrated, not discarded** (607
measured gains recovered on restart).

> **Rule:** a durability mechanism whose cost grows with progress will eventually destroy the run it
> protects. Bound the write, not just the interval.

**Also fixed:** the chain cascaded — a step-1 refusal was followed by steps 2–5 each failing on the
artifact the previous step never produced, burying the real error five logs deep and starting a
fine-tune that could not have worked. A failed step now aborts the chain.

---

## Cost — corrected, and the correction is itself the finding

The design sheet projected **3.3 h** for the 42-state sweep, from the measured fit
`s/eval = 0.0858 + 0.8157·(n/1000)` (R² = 0.985 over all 22 real tree nodes). `--cost_only` reproduced
it exactly: **42 states, 41,834 evaluations, 3.3 h**.

**Measured reality: 0.70–0.74 s/eval on a 223-user state against 0.27 predicted — a stable 2.6×.**

The fit was calibrated on the tree's nodes, which are mostly **large**; phase 3's states have a median
of **154 users**, so almost all of their cost is the per-evaluation *fixed overhead* — exactly the term
a large-node fit constrains worst. **The realistic figure is 8–9 h, not 3.3 h.** Threads were left at
6: CPU sits at ~73%, so threads are not the binding constraint, and tuning a setting mid-run to improve
a quoted cost is not a measurement.

> **Rule:** a cost model fitted on large cohorts underestimates a sweep over many small ones, because
> the intercept is the term it constrains worst. Costs are properties of this box on this night, never
> of the method.

---

## RESULTS — PENDING

Nothing below is filled in yet. Do not cite this section until it is.

- [ ] the 42 below-floor argmaxes: gains, channel mix, how many were already in the parent's tail
- [ ] **order-invariance at K=16** — decides what a fixed-budget comparison can see at all
- [ ] the deeper tree vs the frozen tree, paired percentile bootstrap (10,000 resamples, seed 98765)
- [ ] the deeper tree vs its **shuffled-branch-label** control (HARD RULE 5)
- [ ] the fine-tuned policy: the **spanning regression gate** against E1's +0.02794
- [ ] **★ the headline: policy vs deeper tree**, read only after the positions-changed histogram
- [ ] policy vs bank-alone, on both bars
- [ ] where the policy first deviates from the tree, and where it stops changing (E1: `frac_never_deviates = 1.0000`)
- [ ] the deep-label shuffle control on the policy
- [ ] anti-collapse: distinct interviews, max share, Pep divergence vs the tree's own

---

## Firewall

The 10,000 canonical TEST users are never built, never loaded, never touched anywhere in phase 3.
`--split` accepts `validation` only in every file involved. Every artifact carries
`test_touched=false`. **No test look has been taken.** The tree is never regrown — it is loaded frozen,
and growing one level *below* its existing stops is what this phase is.
