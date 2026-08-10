# Design sheet: does an adaptive selector convert PV *once it can ask concepts*?

TODO item 1.2. Written 2026-08-10 (Fable), off the reconnaissance in this session. This is the direct
test of the only mechanism hypothesis the chapter has for its central finding, and it must run before
Act 3.

---

## 1. THE QUESTION, AND WHY IT IS THE RIGHT ONE

Act 2 established, on 10,000 ML test users, that the adaptive-minus-static gap **tracks personalisation
value and runs the wrong way**: 38 of 48 trend cells separate, all negative, 0/48 under shuffle. The
adaptive selectors lose hardest exactly where personalising is worth most. That is the "methods cannot
find it" branch.

The one mechanism hypothesis on the table comes from the criterion surface itself:

> **The loss-optimal question is a CONCEPT, not an item, for ~100% of users at every budget**
> (100.0% at k=1/2/4, 99.0% at k=8; modal q\* = `vengeance`, `awesome soundtrack`, `violent`, `violent`).
> All three adaptive selectors in that battery (`glm_ucb`, `thompson`, `golbandi_tree`) shop in the
> **item** bank.

If that is the binding constraint, an adaptive selector that *can* ask concepts should convert PV — its
Q4−Q1 trend should stop being negative. If it does not, the failure is deeper than the action space, and
the chapter's mechanism paragraph must be rewritten as a stronger and more uncomfortable claim.

**Either outcome is publishable and both change the text.** That satisfies the "if no decision changes on
the outcome, it doesn't run" rule.

## 2. WHAT ALREADY EXISTS (do not rebuild)

Verified in code this session, with citations:

| asset | where | note |
|---|---|---|
| Two-type action space, ids `[n_items, n_items+C)` | `adaptive_harness.py:369-374`, `:336` | `n_items`=18,359, `C`=1,031 |
| **Concepts-only** adaptive selector | `conlinucb_bs_static` = `ConLinUCBBSSelector(key_term_only=True)`, `adaptive_harness.py:777` | forces `b_schedule="always"` |
| Concept-asking adaptive selectors | `conlinucb_mcr`, `conucb`, `dpe_concept` | already run in Act 1 |
| Concept vectors in the bandit basis | `keyterm_features()`, `conversational_bandit_selectors.py:259-297` | `Xtilde` is (C x f) in **Y's own basis** |
| Unified argmax over both halves | `_pick_mixed`, `:622-654` | with per-user asked mask on both halves |
| Concept-channel static optimum | `fixedorder:greedy_concepts`, `GREEDY_FIXED_ORDERS` `:112-114` | stored order, 16 steps, already run |
| Behavioural concept answers | `CB.concept_answer`, `:459-463` | unanswerable concept **burns the turn**, recorded as refusal |

**`glm_ucb`/`thompson` are structurally item-only** — the arm *is* a row of `belief.Y`
(`bandit_selectors.py:240`, `:334`, `:572`), and no concept vector exists in that module. `golbandi_tree`
is item-only more deeply still: split candidates must be columns of the rating matrix R
(`golbandi_tree_selector.py:159-160`, `:212-241`), and a concept has no column in R.

**Consequence for scope.** We do NOT extend the item-bank bandits to concepts in this experiment. That
would be a new selector, needing its own snap and its own defence, and it confounds "can an adaptive
method use concepts?" with "is our bolted-on concept arm any good?". We instead stratify the
**published, already-benchmarked, already-snapped** concept-asking selectors. Extending `glm_ucb` to the
unified space via `_pick_mixed` is a clean follow-up **if and only if** the published concept-askers also
fail to convert PV (see §7).

## 3. THE BLOCKER, AND THE ONLY BUILD WORK

`scripts/act2_pv_quartiles.py` **refuses concept-asking cells outright**:

```
:184-186  assert all(len(t) == 0 for t in d["conc_state"]["asked"]), \
              f"{spec}/k{k} spent turns on concepts; the state builder here is item-only"
```

`states_of` (`:189-198`) emits `(item sids, half-star levels)` and the concept slot is fed the empty
triple `empty_c = ([], [], [])` (`:252`, `:259`). The **criterion machinery already supports concept
tokens** — `L1.canon(cc[0]+[c], ...)` at `:371` — so this is a state-builder gap, not a criterion gap.

**Build work, and it is the whole of it:** carry concept tokens through `read_interview` / `states_of`
into the `(item, concept)` state pair, and drop the assert. The concept token triple is already recorded
by the harness in the exact shape needed: `conc_state["tokens"][u] = (c, value, refused_flag)`
(`adaptive_harness.py:381-389`).

## 4. DESIGN

**Arms** (one harness process, fresh label `act2pvc`, so every difference is in-run):

| role | spec | why |
|---|---|---|
| concepts-only adaptive | `conlinucb_bs_static` | the cleanest test — it *only* asks concepts |
| mixed adaptive | `conlinucb_mcr` | asks both; the realistic conversational setting |
| second family | `conucb` | ConUCB Eq. 4 lineage, different acquisition |
| cheap concept adaptive | `dpe_concept` | ~0.2 m at k=8; a fourth family for free |
| **PV reference static** | `fixedorder:entropy0` | **must be bit-identical to the act2pv run** — see §5 |
| concept-channel static | `fixedorder:greedy_concepts` | the RIGHT comparator for a concept asker |
| item-channel static | `fixedorder:greedy_items` | carried so the new table is commensurable with the old |

**Budgets** k ∈ {1,2,4,8}. **Metrics** full AND tail NDCG@10, learned decoder bias, NO-FLOOR, policy-B
candidate pool, all 10,000 canonical TEST users, behavioural answers.

**Statistics** paired percentile bootstrap over per-user difference vectors, 10,000 resamples, seed
98765. **Never** the unpaired `1.96*sqrt(2)*SE` form. Trend = Q4−Q1, two-sample bootstrap (disjoint
strata), reported per comparison as the primary object; the 96 cell CIs are shown so nothing is hidden
but are not the object.

**Strata** the SAME PV quartiles as `ACT2_PV_QUARTILES.md`, read at `fixedorder:entropy0` states. PV is
already fully scored for that arm; reusing it is what makes the strata selector-independent and
comparable across the two tables.

**Conventions, non-negotiable.** `--concept_neg_cap 0` (the certified-era convention `concept_cv3_ep08`
was actually trained under, `adaptive_harness.py:1027-1030`); certified pair `t2i26c_ep31.pt` +
`concept_cv3_ep08.pt`; `--keyterm_repr` set to whatever the carried-forward Act-1 cells used, recorded
explicitly in the output meta. **Never difference across conventions** — the same 16 greedy concepts
score .2473/.1154 under `0` and .2298/.0977 under `1`.

## 5. CONTROLS AND GATES (all must pass, and be printed)

1. **Self-difference control.** An arm differenced against itself inside every quartile must give
   max|delta| + |CI| **exactly 0.0e+00**.
2. **★ entropy0 REPRODUCTION GATE.** The `fixedorder:entropy0` interviews in the new `act2pvc` run must
   be **byte-identical** to those in `act2pv` (it is deterministic, so under a matching contract they
   must be). Assert on the checkpoint JSON. If they differ, the PV strata do not transfer and the run is
   invalid — **stop and report, do not reconcile by hand.** This is the gate that makes reusing the
   already-scored PV arm legitimate rather than an assumption.
3. **Shuffle control.** Permute PV across users, re-cut, re-run every trend. All must go non-significant.
4. **Answer-rate reporting per quartile.** ρ(PV, answer rate) is **+0.76 at k=2**, so the k=2 stratification
   is substantially a split on "did you answer". Report the rates; do not read k=2 as pure PV. k=4/8 carry
   the result (rates flat across Q2–Q4, profile length non-monotone and *lowest* in Q4).
5. **Channel accounting.** Per selector per budget, report the **fraction of asked turns that were
   concepts** and the **refusal rate**. A "concept-asking" selector that asks few concepts (the
   ConLinUCB schedule is known to fire twice in sixteen) has not tested the hypothesis, and the table
   must make that visible rather than let a null ride on an unfired schedule.

## 6. COST

Interview cost is known from the run JSONs at 10,000 users: `conlinucb_mcr` 0.10/0.03/3.04/12.93 m at
k=1/2/4/8; `conlinucb_bs_static` 0.02/9.40/34.89/16.12; `conucb` 0.01/0.03/1.34/6.48; `dpe_concept`
0.08/0.04/0.08/0.20; the three static orders are ~0.05 m each. **Total ≈ 1.5–2 h**, plus ~0.25 m per
(cell, consumer) scoring.

**No new criterion scoring is required for the primary result** — PV comes from the already-scored
entropy0 states, and the quartile gap needs only per-user NDCG vectors. The §6 branching q\* columns for
these new cells would need scoring and are **explicitly out of scope**, consistent with stopping the
frontier at 11/20 cells.

Run it as ONE process (HARD RULE 11(d): never two heavy jobs on the box) with per-cell persistence
(11(a)) and a unique output label (11(e)).

## 7. HOW TO READ IT, STATED BEFORE THE RUN

Not a pre-registration — the author has ruled Paper B does not pre-register (2026-08-03, reaffirmed
2026-08-10); the exploratory/confirmatory ledger in `PAPER_B_EVIDENCE_POLICY.md` is the substitute. These
are written down so the reading is not retrofitted.

* **Trend becomes ≥ 0 / non-negative for the concept askers** → the action space *was* the binding
  constraint. The chapter gains a constructive result: the failure is a **bank** problem, fixable, and
  Act 4's coarse-to-fine story becomes the remedy rather than a curiosity. This is the strongest
  available outcome and it changes the paper's contribution from a null to a mechanism-plus-fix.
* **Trend stays negative** → the action space is NOT the constraint, and the mechanism paragraph in
  `ACT2_PV_QUARTILES.md` ("a concrete hypothesis for why they cannot convert PV") must be **retracted in
  the same document, not quietly dropped**. The finding hardens to: adaptive selection fails to convert
  personalisation value *even when the optimal action type is available to it*. Then, and only then, the
  `_pick_mixed` extension of `glm_ucb` becomes worth building, because at that point the question is
  whether ANY acquisition rule converts PV.
* **Schedule doesn't fire** (control 5 shows few concept asks) → the cell is uninformative and must be
  reported as such, not as a null. `conlinucb_bs_static` is the guard against this, since it can only ask
  concepts.

**A null here is not citable as a contribution on its own** (author ruling: "our nulls are not citable at
all"). It is a mechanism test whose job is to tell us which sentence in the chapter is true.

## 8. WHAT THIS DOES NOT TEST

* It does not test whether a *better* concept bandit exists — only whether the published ones convert PV.
* PV is read at entropy0 states, so a selector that would have driven a user to a higher-PV state gets no
  credit for it (limitation (2) of the parent result, inherited).
* The criterion scores against the tower's own predictive, so PV is self-consistent expected gain, not
  gain against held-out items (limitation (1), inherited).
