# Archive — superseded avenues (kept on disk, not deleted)

These are abandoned/superseded work, moved here on 2026-06-17 to keep the live tree clean. Nothing deleted.
Live results are in `../RESULTS.md`; plan in `../CASPER_U_PLAN.md`.

## `CASPER_EXPERIMENTS_E1-E14_archived.md`
Full log of experiments E1–E14. E1–E13 = the **frozen-instrument RL saga**: an extensive effort to make an RL
elicitation policy beat popularity *against a frozen recommender ("instrument")*. CONCLUSION (now understood):
that negative result was an **artifact of freezing the recommender** + a non-standard NDCG + pure-personalization
(vs popularity+personalization). E14 = the breakthrough that reversed it (see RESULTS.md R1). Kept for the
methodology/diagnostics thesis chapter, but NOT the go-forward design.

## `frozen_instrument_saga/` (15 scripts)
The RL-on-frozen-instrument experiment scripts (casper_e8…e13, casper_rloo/belief/recon/bc_rl, oracle_ceiling_ml100k,
heuristic_elicit/recon) and the early/unfaithful DRE attempt (dre_ml100k.py, superseded by ../scripts/paper1/dre_faithful.py).
Superseded because the whole approach measured policies against a frozen ranker, which structurally cannot use
elicited information. The go-forward design co-trains / folds-in (CASPER-U).

## Still LIVE (NOT archived) in `../scripts/paper1/`
- `mf_foldin.py` — canonical reproduced baseline (representative-MF beats popularity)
- `golbandi_tree.py` — Golbandi adaptive-tree replication (RMSE)
- `dre_faithful.py` — faithful DRE re-implementation (baseline reproduced; neural gain did not)
- `instrument_v2.py` — set-encoder instrument base (dataset-flexible: env DATASET=ml-100k|ml-1m)
