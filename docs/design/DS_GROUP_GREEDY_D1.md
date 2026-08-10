# DS — Depth-1 group-conditioned greedy (label `groupgreedy_d1`)

**Date** 2026-08-10 · **Code** `src/instrument/group_greedy_d1.py` (wrapper) + `prefix=`/`on_pick=` on
`strategy_channel_suite.greedy_certified` · **Cohort** validation only

## Question

Every adaptive-vs-static comparison in the Act-1 battery confounds two things. Our greedy static order
is the **only** arm that optimises the quantity we report (cohort-mean held-out NDCG@10) — and it is
also the **only static** arm (`docs/results/BASELINE_OBJECTIVE_AUDIT.md`, table rows 21–23). So
"adaptive loses to static" may be "these adaptive methods optimise a surrogate".

This run holds the objective, the optimiser, the pool and the cohort fixed and varies **only whether
the selector may condition on an answer** — at exactly **one** branch point.

## Design

| | |
|---|---|
| Cohort | the canonical **10,000 VALIDATION** users (rows with a non-empty target). **Forced:** `data/ml-25m/proc/` has no `train_tr`/`train_te`, so train users have no fold-in/target split, and the tower was trained on them. Selection has always run on validation. |
| q1 | **read**, never re-derived, from `experiments/battery/greedy_static_i26c.json` → arm `items-only` → step 0 = **(channel 0, item 83)** |
| Branch | 3 groups by the answer to q1, using the harness's own behavioural rule off the graded fold-in lookup `sh["lvl_lookup"]` (targets already excluded): **absent → DID-NOT-ANSWER**; **level ≥ `LIKE_MIN_LEVEL`=7 (≥ 4.0 stars) → LIKE**; **below → DISLIKE** |
| Inside each group | `greedy_certified`, **prefix = [q1]**, **L = 15** further steps, pool = the **same `_ranked_pool("pop", …, 500)`** call the global build makes, minus q1 (499 candidates), objective **`full`** |
| Output per group | `[q1] + its own 15`, plus `n_evals` and `sec_per_eval` |
| Comparison (validation only) | `genv_mean` of (a) the group's own order and (b) the **global 16-step order** on that same group, at budgets 1/2/4/8/16 |
| Not done tonight | **nothing is evaluated on TEST.** The test cohort is not built, not loaded, not touched. |

## Baseline symmetry

Group order and global order are scored by the same function, on the same users, in the same env, on
the same pool policy (B), at the same budgets, in the same process. The only difference between them
is whether the 15 steps after q1 were chosen knowing the answer to q1.

## Sanity check (the built-in control)

On the **selection** cohort a group's own order must be **≥** the global order on that group's users:
the global order lives inside the search space the group's greedy explored. A violation means the
search, the cohort masking or the memo is broken — the script logs it as a screaming `SANITY *** FAIL
***` and marks `sanity_pass: false` in the JSON. It is not evidence about conditioning.

Second control, cheap and mechanical: the memo in `genv_mean` is keyed on the **sequence only**, so it
is cleared (with `n_evals`/`t_evals`) every time the active cohort changes. Without that, group B
would read group A's numbers.

## No truncation (HARD RULE 1)

All 10,000 validation rows are partitioned; the script asserts the three groups are disjoint and sum
to the cohort. A small group is **reported**, never merged and never dropped. `--sanity` mode does
truncate (12-item pool, L=2, one group) — it is labelled `_SANITY.json`, `citable: false`, and exists
only to prove the wrapper runs end to end before a 10-hour launch.

## Cost — this measurement is itself a deliverable

`n_evals` is set by pool size and L, **not** by cohort size (`BASELINE_OBJECTIVE_AUDIT.md` §(b)).
Whether **per-eval** cost falls with cohort size is what the audit called UNVERIFIED and what the
proposed "cost smoke" was for. This run measures it directly, three times, at three different cohort
sizes, against the recorded **13.543 s/eval at 10,000 users**.

* **optimistic** (per-eval linear in users): Σ over groups ≈ one 10k items-only build ≈ **3.2 h**
* **pessimistic** (fixed overhead dominates): ≈ 3 × 3.2 h ≈ **9.6 h**

Either fits one night. If a ~3,300-user group runs at ~1/3 of 13.543 s, per-eval scales with cohort
size and every future group-split design is cheap.

## What changes on the outcome

* Group orders **diverge** from the global order and beat it on their own users → conditioning has
  value that the battery's adaptive arms failed to capture; the Act-1 null is about the arms, not
  about conditioning. Next step: authorise a TEST evaluation.
* Group orders are **near-identical** to the global order → the depth-1 branch buys nothing even when
  the brancher optimises the reported metric directly, which is a much stronger statement of the
  Act-1 null than the battery can currently make.

## Persistence & launch

HARD RULE 11(a): `experiments/battery/groupgreedy_d1_partial.json` is rewritten after **every pick**;
a kill costs one pick. Unique label `groupgreedy_d1`; no other run shares these paths. Driven by a
Windows **scheduled task** `casper_groupgreedy_d1` (background launchers on this box have been killed
by console interrupts delivered to the process group — the `.err` files hold exactly two bytes, `^C`).

Delete with: `schtasks /Delete /TN casper_groupgreedy_d1 /F`
