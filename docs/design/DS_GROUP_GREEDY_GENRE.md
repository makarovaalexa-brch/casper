# DS — Genre-split group-conditioned greedy, 16 vs 16 (`groupgreedy_genre16`)

**Date** 2026-08-11 · **Code** stage 1 `src/instrument/group_greedy_genre.py`, stage 2
`src/instrument/group_greedy_genre_test.py` · **Predecessor** `DS_GROUP_GREEDY_D1.md`

**Author instruction, verbatim (design):** *"can you not fold in the genre question? but instead
partition by genre, evaluate the static order vs adaptive order found per group? so 16 vs 16, would
just show, hopefully, that adaptive order is indeed optimal"*

**Author instruction, verbatim (endpoint):** *"only result on test users is interesting"*

## Question

`groupgreedy_d1` branched on the answer to ONE popular item and found conditioning gains of +0.0129
(like, n=2,099) and +0.0237 (dislike, n=856) — but **+0.0019** in the **70.45%** (n=7,045) that
*could not answer the question at all*. The branch was limited by **answerability**, not by
conditioning. Here the branch is **favourite genre**, which every validation user can answer.

## Stage 1 — build (VALIDATION)

| | |
|---|---|
| Cohort | the canonical **10,000 VALIDATION** users. **Forced:** `data/ml-25m/proc/` has no `train_tr`/`train_te`, so train users have no fold-in/target split, and the tower was trained on them. |
| Branch answer | most frequent MovieLens genre over the user's **LIKED FOLD-IN** items (`lvl_lookup` level ≥ `LIKE_MIN_LEVEL`=7, i.e. ≥ 4.0 stars), **ties ALPHABETICAL**; none → explicit `none` group. |
| Honesty | `lvl_lookup` comes from `ctx.allb`, built with `keepm = ~g["sid"].isin(te_sets[r])`. **Asserted at runtime** (`leak_assert`), not assumed: 0/10,000 users may have a target inside the grouping lookup. |
| Genre source | `data/movielens/movies.csv` × `data/ml-25m/proc/unique_sid.txt`; sid map **asserted equal to `ctx.show2id`**. Coverage reported (18,235/18,359 = 99.32%). |
| Per group | `greedy_certified`, **no prefix**, **L = 16 ITEM steps**, pool = the same `_ranked_pool("pop", …, 500)` call the global build makes, objective **`full`**. |
| Comparison | **ITEM-FOR-ITEM**: the group's first K items vs the global 16-step items-only order's first K items, K ∈ {1,2,4,8,16}. **There is no total-budget curve.** |

### The generous assumption — stated prominently, by instruction

**The favourite-genre answer is FREE SIDE INFORMATION.** It costs no turn and is not folded into the
recommender, so the comparison is **16 items vs 16 items**. A deployed system would have to ask for
it and would pay a turn. This is **generous to the conditioned arm by construction**, and every gain
reported anywhere in this design is therefore an **upper bound**. It is repeated verbatim in the log
header, in `generous_assumption` in both output JSONs, and here.

### What the sanity gate does and does not mean

At 16-vs-16 on the **selection** cohort the group's own order is **guaranteed** ≥ the global order,
because the global order lies inside every group's search space. So **`sanity_pass` is a pure harness
check and a win is NOT evidence that conditioning pays.** The informative quantities are:

1. **Magnitude** of the per-group and cohort-weighted gain — the ceiling on conditioning here.
2. **Gain vs group size** — the overfitting signal. Large gains on small groups beside ~zero on the
   big ones is fitting noise, not structure. The end-of-run table sorts by n for exactly this read.
3. **Order overlap with the global order** — set intersection and positions differing, per K. If the
   conditioned orders are near-identical to the global one, that is the cleanest possible statement
   that there was nothing to condition on.

### No truncation, no merging (HARD RULE 1)

All 10,000 rows partitioned; disjointness and cover asserted. **18 groups**, none merged, none
dropped: Drama 5,481 · Action 1,582 · Comedy 1,475 · Adventure 534 · Thriller 451 · Sci-Fi 161 ·
Romance 88 · Crime 79 · Horror 51 · Animation 44 · Children 29 · Fantasy 11 · Documentary 3 ·
Musical 3 · Mystery 3 · War 2 · Western 2 · Film-Noir 1. **13/18 below 200 users**, flagged
`small_group: true`. The `none` group is empty — the question has a 100% answer rate.

---

## Stage 2 — evaluate (TEST). **PRE-SPECIFIED, COMMITTED BEFORE ANY TEST NUMBER EXISTS.**

One-shot. Nothing below may be changed after a test number is seen — not the threshold, the grouping
rule, the pool, the budget or the number of resamples. If something looks wrong it is reported as it
stands, with a note on what would have been done differently.

| | |
|---|---|
| Cohort | the **10,000 canonical TEST** users, disjoint from validation. |
| Grouping | the **same rule**, from each test user's **own fold-in** ratings. `partition_by_genre` is imported from stage 1, never re-implemented. `leak_assert` re-run on the test side: 0/10,000. |
| Orders | the stage-1 **validation-built** 16-item orders, read from `groupgreedy_genre16.json`. Never rebuilt. Stage 2 asserts every group finished. |
| Contract | full **and** tail NDCG@10, learned decoder bias, **no floor**, policy-B candidate pool, canonical head mask — `genv_eval`, the same function stage 1 used. |
| Comparator | the global 16-step items-only order on the same test users, differenced **per user, paired**. |

### Primary endpoint

**Cohort-weighted group-order minus global-order, FULL NDCG@10 at K=16, all 10,000 test users,
paired CI.** (`arms.all_groups.16.paired_ci_full`.)

### Inference

**Paired percentile bootstrap over the per-user difference vector, 10,000 resamples, seed 98765.**
Never the unpaired `1.96*sqrt(2)*SE` form — that signature is what made the earlier "tie" claim wrong
(memory `paired-test-kills-the-tie-claim`). Reported at K ∈ {1,2,4,8,16}, full and tail.

### The two arms — both pre-specified, both reported

* **ALL-GROUPS** — every test user gets their own group's order, including groups whose order was
  fitted on a handful of validation users.
* **FALLBACK-200** — test users whose genre group had **fewer than 200 VALIDATION users** get the
  **global** order instead. This is the deployable policy: you do not ship an order fitted on 3
  people. **200 is fixed now, in advance**, and is the same threshold stage 1 already flags with. It
  is not a tuned parameter and must never become one.

### Also reported, all decided now

* Test-side genre distribution beside the validation one; empty test groups named, never merged.
* Per-group **test** gain against **validation** group size.
* Order-overlap statistics vs the global order.
* **Validation-to-test shrinkage**: each group's validation gain beside its test gain. This is the
  overfitting measurement and is the most likely thing to carry the finding.

## Cost

`n_evals` is set by pool size and L; per-eval cost is ≈ proportional to cohort size (measured
0.891 s at n=451, 10.42 s at n=5,481 → `s/eval ≈ 1.9e-3·n`). Group sizes sum to 10,000, so stage 1
≈ one global build, **≈ 4 h**. Stage 2 is scoring only — 5 budgets × (1 global + 18 group) passes,
**≈ 15 min**.

## Persistence & launch

HARD RULE 11(a): `*_partial.json` rewritten after every pick; contract-signed resume refusing a
changed contract. Per-user vectors banked to `*_per_user.npz` so every CI is paired and re-derivable.
Unique labels `groupgreedy_genre16` / `groupgreedy_genre16_TEST` — no shared output paths. The
abandoned 15-step run's outputs were renamed `groupgreedy_genre_ABANDONED_15step*`.

Windows scheduled task `casper_groupgreedy_genre16`. `schtasks /Create` defaults to
`DisallowStartIfOnBatteries=$true` and will sit **Queued** while reporting Last Result 0 — both
battery flags are cleared with `Set-ScheduledTask` **before** `schtasks /Run`.

Delete with: `schtasks /Delete /TN casper_groupgreedy_genre16 /F`
