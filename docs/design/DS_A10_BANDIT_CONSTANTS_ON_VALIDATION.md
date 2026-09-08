# DS-A10 — choose the bandits' free constants on VALIDATION, then re-run the chosen value on TEST

**Author ruling (2026-09-08):** item 2 of the ECIR to-do; answers the strict reviewer's "comparator searched
on the metric, ported arms untuned".

## The asymmetry, stated
The static comparator is a greedy search on held-out full NDCG@10 over validation users. The ported adaptive
arms run with their published constants: GLM-UCB's exploration weight `alpha=1.0` (one posterior s.d.),
Thompson's exact posterior draw, ConUCB's `lam=0.5`, `lam_til=1.0`, CLiME/ConLinUCB's `conv_beta`. None was
chosen on our data. The heuristics (entropy₀, popularity, HELF) and the tree learners have no such constant.

## Design
- Harness: `src/baselines/adaptive_harness.py --cohort validation` (new; `arm_n.load_arm_n(cohort=...)`
  serves the 10,000 validation users through the identical contract: same pool rule, same floor, same graded
  fold-in, same consumer). Output label forced to `VAL_*`; never citable.
- Selection metric: full NDCG@10 at K=16 on the certified consumer (`--only ours`), the same objective the
  comparator was searched on. Tail reported, never selected on.
- Grids (one constant per arm, the published value included):
  - GLM-UCB `--ucb_alpha` ∈ {0.25, 0.5, 1.0, 2.0, 4.0}
  - Thompson `--ts_scale` ∈ {0.5, 1.0, 2.0} (posterior temperature; 1.0 = published exact draw)
  - ConUCB `--conucb_lam` ∈ {0.1, 0.5, 2.0}; CLiME `--conv_beta` ∈ {published, ×0.5, ×2}
- Then ONE run per arm on `--cohort test` with the chosen value, standard budgets 1,2,4,8,16, all consumers,
  label `adaptive_table_tuned_<arm>`; compared against the same static comparator.
- Everything else as published; nothing else is touched. No test number is read before the choice is made.
- Cost: GLM-UCB ≈ 30 s per turn at K=16 on 10,000 users (log evidence: 10.7 s at turn 8), so ≈ 5–10 min per
  grid point; whole validation grid ≈ 1.5 h sequentially; test re-runs ≈ 40 min. Run detached, monitored.
- HARD RULE 10: code committed before launch. HARD RULE 11: the harness banks per (selector, budget) cell and
  checkpoints interviews per turn.

## Gate
The published-value cell on validation must reproduce the banked validation-era numbers where they exist
(none are banked for these arms on validation, so the gate is the test re-run: `--ucb_alpha 1.0` on test must
reproduce Table 1's GLM-UCB row bit-for-bit, `.1606/.0400 … .2161/.0833`).

## What decision it changes
If the chosen constant moves an arm above the static comparator at any budget, RQ1's headline changes and
the paper says so. If not, the paper states the constants were chosen on validation and the asymmetry
objection is closed. Either way the sentence in "Ports and tuning" and the Limitations paragraph change.
