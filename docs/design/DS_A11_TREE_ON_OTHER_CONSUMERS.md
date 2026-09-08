# DS-A11 — the measured-metric tree consumed by other recommenders (transfer check)

**Author ruling (2026-09-08):** item 3 of the ECIR to-do; answers "CASPER is fitted on the same scorer it is
evaluated with and never tested for transfer".

## Design
- The tree `ndcgtree16` was grown on validation and scored once on test; its per-user TEST interview is banked:
  each test user's leaf (`ndcgtree16test_policyB_per_user.npz`, `test|<leaf>|rows`) and each leaf's own
  16-question sequence (`ndcgtree16test.json`, `leaf_table[].own_sequence`). Nothing is regrown.
- New selector spec `replay:ndcgtree16test` in `adaptive_harness.py` replays that sequence per user; new spec
  `fixedorder_json:policyB_combined` replays the policy-B static comparator (the order the tree is measured
  against). Both are stored orders: nothing is selected, the answer convention comes from the cell
  (`--concept_neg_cap 0`, the certified-era convention `build_greedy_env` scores under).
- Consumers: `ours` (gate), `recvae_likes`, `tanp`, `golbandi_leaf`, `belief_mf`, `eddi`. Non-`ours` consumers
  hear ITEM answers only (the harness folds concept answers only through the certified concept channel), so
  both arms lose their concept answers symmetrically there; this is stated with the result.
- Budgets 1,2,4,8,16; paired per-user vectors (`--per_user`) so tree − static gets a paired bootstrap per
  consumer. Cohort: test (the same single look; a re-consumption of banked interviews, no new selection).

## Gate
On `ours` at K=16 the replayed tree and the replayed static order must reproduce `ndcgtree16test_policyB`:
tree 0.26936 / 0.13074, static 0.25900 / 0.12370 (full / tail). If the harness's consumer path differs from
`genv_eval` (different fold implementation), the gate reports the discrepancy and the cross-consumer contrast
is still read, since every consumer is scored through one harness; but the discrepancy is written down first.

## What decision it changes
If tree − static holds on RecVAE and TaNP, RQ3's result is not a property of the scoring recommender and the
paper says so in one sentence with three numbers. If it reverses on a strong consumer, the paper's claim narrows
to the certified consumer and the Limitations paragraph says so. Cost: consumer loading plus scoring only;
minutes to an hour. Code committed before launch (HARD RULE 10).
