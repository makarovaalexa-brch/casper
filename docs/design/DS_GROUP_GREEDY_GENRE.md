# DS — Genre-split group-conditioned greedy (label `groupgreedy_genre`)

**Date** 2026-08-11 · **Code** `src/instrument/group_greedy_genre.py` (adapts `group_greedy_d1.py`)
· **Cohort** validation only · **Predecessor** `DS_GROUP_GREEDY_D1.md`

## Question

`groupgreedy_d1` branched on the answer to ONE popular item (item 83, the global greedy's own first
pick) and found conditioning gains of **+0.0129** (like, n=2,099) and **+0.0237** (dislike, n=856) —
but **+0.0019** in the **70.45%** of the cohort (n=7,045) that *could not answer the question at
all*. The depth-1 branch was therefore limited by **answerability**, not by conditioning.

This run keeps everything else fixed and swaps the branch question for one nearly everyone can
answer: **"what is your favourite genre?"** — asked open (no closed vocabulary shown), answered
faithfully from the user's own fold-in profile.

## Design

| | |
|---|---|
| Cohort | the canonical **10,000 VALIDATION** users (rows with a non-empty target). **Forced:** `data/ml-25m/proc/` has no `train_tr`/`train_te`, so train users have no fold-in/target split, and the tower was trained on them. |
| Branch question | **open**: favourite genre. Simulated, never asked of a model. |
| Branch answer | the most frequent MovieLens genre over the user's **LIKED FOLD-IN items** (`sh["lvl_lookup"]` level ≥ `LIKE_MIN_LEVEL`=7, i.e. ≥ 4.0 stars). **Ties broken ALPHABETICALLY.** No liked item, or no liked item carrying a genre → explicit group **`none`**. |
| Honesty of the answer | `lvl_lookup` is built from `ctx.allb`, which `build_real_ctx` constructs with `keepm = ~g["sid"].isin(te_sets[r])` — held-out targets removed. The script **asserts this at runtime** (`_leak_assert`: zero users may have a target inside the grouping lookup) rather than trusting the comment. |
| Genre source | `data/movielens/movies.csv` (the ML-25M raw movies file; `proc/meta.json` names `data/movielens/ratings.csv` as the raw source). Item ids are row indices into `data/ml-25m/proc/unique_sid.txt`; the mapping is **asserted equal to `ctx.show2id`** over all 18,359 items. Coverage of the catalogue is reported in the output. |
| Inside each group | `greedy_certified`, **no prefix**, **L = 15** ITEM steps, pool = the **same `_ranked_pool("pop", …, 500)`** call the global build makes (all 500 — nothing is removed, the genre question is not an item), objective **`full`** |
| Comparator | the frozen global 16-step items-only order, `experiments/battery/greedy_static_i26c.json` → arm `items-only` |
| Not done | **nothing is evaluated on TEST.** The test cohort is not built, not loaded, not touched. |

## What the genre answer does and does NOT do

It conditions the **selector** only. It is **not folded into the recommender**: the pool is
items-only, the scored sequence contains items only, and no genre/concept token enters the fold. So
the genre turn buys the interview nothing except a better ordering of the remaining items — and it
still costs a turn. That is exactly what makes the budget accounting the load-bearing part.

## Budget accounting

A group's interview is **[genre question] + 15 items = 16 turns**.

* **HEADLINE — matched TOTAL budget T** (`curve_total_budget`): the group's first **T−1 items**
  against the global order's first **T items**, for T ∈ {1, 2, 4, 8, 16}. The genre turn is charged.
  T=1 is the degenerate cell (group has asked nothing but the genre question).
* **CONTAINMENT — matched ITEM count K** (`curve_item_matched`): K group items vs the global order's
  first K items.

**The sanity gate runs on the CONTAINMENT comparison only.** Its rationale — "the global order is
inside the search space the group's greedy explored" — is true at matched *item count* and **false**
at matched *total budget*, where the global order has one item more than the group could ever pick.
A shortfall at matched total budget is therefore a **finding** (the genre turn did not pay for
itself), not a bug. Both numbers are computed, logged and stored; only the item-matched one sets
`sanity_pass` and can trigger the screaming `SANITY *** FAIL ***`.

Caveat recorded up front: CELF is step-wise-optimal, not sequence-optimal, so on a **small** group a
containment failure can be genuine greedy myopia rather than a broken harness. The failure message
says so and names the group size.

## No truncation, no merging (HARD RULE 1)

All 10,000 validation rows are partitioned; disjointness and cover are asserted. Genre groups are
heavily skewed. **No group is merged and no group is dropped**, however small. Any group with fewer
than **200** users is flagged in the log and carries `small_group: true` in the JSON as
*statistically untrustworthy*. If merging turns out to be needed that is an author decision, not a
runtime one.

## Cost

`n_evals` is set by pool size and L, not by cohort size; per-eval cost scales roughly with cohort
size (measured: 1.62 s at 856 users, 4.85 at 2,099, 12.78 at 7,045, 13.54 at 10,000). Because the
group sizes sum to 10,000, Σ(evals × s/eval) ≈ **one global build**, plus a per-group floor from the
500-candidate seeding pass. Expected ≈ **3–6 h**. `sec_per_eval` is logged per group and stored.

## Persistence & launch

HARD RULE 11(a): `experiments/battery/groupgreedy_genre_partial.json` is rewritten after **every
pick**; a kill costs one pick. Contract-signed resume (`sig`), refusing to resume across a changed
contract. Unique label `groupgreedy_genre`; no other run shares these paths.

Driven by a Windows **scheduled task** `casper_groupgreedy_genre` (background launchers on this box
have been killed by console interrupts delivered to the process group). `schtasks /Create` defaults
to `DisallowStartIfOnBatteries=$true` and will sit **Queued** while reporting Last Result 0 — both
battery flags are cleared with `Set-ScheduledTask` immediately after creation.

Delete with: `schtasks /Delete /TN casper_groupgreedy_genre /F`
