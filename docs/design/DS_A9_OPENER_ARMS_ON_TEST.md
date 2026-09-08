# DS-A9 — opener-state arms on the reserved TEST cohort (re-score, not refit)

**Author ruling (2026-09-08):** *"the ones we fitted on train should have been then evaluated on test. we
must do it now. this should be number one on to-do list."*

## What already exists on test (29 Aug test look, `m6generalise_B_test.json`, `m6mix_B_test.json`)
Per group: policy, opener tree (train-2000, disjoint), cold tree, policy − cold [CI], policy − tree [CI].
Mix: policy, cold, static (validation-fitted incumbent), policy − cold, policy − static.

## What `tab:volunteered` (validation, `m6t_order_check.json`) has that test lacks
Per group and pooled: **static_train** (the train-fitted 15-question order), and the contrasts
**opener − cold**, **cold − static_train**, **opener − static_train**, each with a paired bootstrap interval.

## Design
- Extend `m6d_eval.py` generalise mode with an optional `--static_train` arm and the three contrasts above
  (plus policy − static_train), and bank **per-user score vectors** to a `.npz` so no further look is needed.
- Same test material as 29 Aug, reused not refitted: `--split test --load_tabs m7_tabs_val.npz --disjoint
  m7b_disjoint.npz --tree_suffix _train2000 --cold_tree coldtree_v2_train10000.json --half B`, policy
  `m4/m7dj_policy.pt`. Users and their own-data group assignment come from test; every fitted quantity is banked.
- **Gate (must hold before any new number is read):** per-group policy, tree and cold means at K=1, 4, 15
  reproduce `m6generalise_B_test.json` to 1e-9; test m=0 snap within the plausibility band.
- Statistic: paired percentile bootstrap over the per-user difference, 10,000 resamples, seed 98765; each test user
  is in at most one group (disjoint partition), so user-level pairing is exact; pooled = concatenation of groups.
- No data reduction; all warm test users in the six groups; no sampling.
- Output: `experiments/battery/m6/m6generalise_B_test.json` (rewritten with the extra keys, old file kept as
  `m6generalise_B_test_2026-08-29.json`) and `m6generalise_B_test_peruser.npz`.
- Cost: one pass of the 29 Aug job plus one extra static arm; expected well under an hour. Free RAM 4.2 GB, no
  other Python job running (checked 2026-09-08). HARD RULE 10: code committed before launch.

## What decision it changes
`tab:volunteered` moves from validation to test in the chapter, with the same three contrasts; the paper's
protocol sentence for the requests section becomes "fitted on train, scored once on test". Not on the ECIR
paper's path (requests section excluded) but on the chapter's.

## Declared caveat
This is a second reading of the same test look (same users, same banked material, one additional arm). It is
recorded as such; the test cohort is not refitted on and no arm is changed after seeing it.
