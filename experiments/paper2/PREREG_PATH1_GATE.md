# PRE-REGISTERED GATE — Path 1: anytime-reward CASPER-R retrain (written 2026-07-02, BEFORE any run)

Motivation: every prior Paper-B attempt optimized the q8 endpoint, where P0-a/P0-b (P0_DIAGNOSTICS_RESULT.md)
prove the budget saturates (answers 6-8 worth <=+0.001; answerability-oracle q8 delta ~0). The only
training-seed-robust policy win in the project (Paper D POLOPEN) used a mean-NDCG-over-turns reward.
This experiment ports that reward to the Paper B closed-concept harness.

## Frozen decisions (may not be changed after seeing any result)

- **Recipe**: CASPER-R architecture + BC-entropy floor, reward = MEAN of NDCG@10 over turns 1..8
  (anytime), replacing the endpoint objective. All other hyperparams = locked winner recipe.
- **Training seeds**: TRSEED {0,1,2}. Eval seeds {1,2,3,7,11}. Ruler: ML-1M te[300:], 304 users.
- **Checkpoint selection**: best-VAL (val split, SELVAL=tail), never test. USEBEST eval.
- **Primary metric (named in advance)**: TAIL NDCG@10 AUC over turns 1-8 vs static entropy.
  **Secondary**: tail@q4. Endpoint q8 is reported but NOT a success criterion (known saturated).
- **Success**: delta positive for ALL 3 training seeds on the primary metric AND paired per-user
  bootstrap p<0.05 (pooled across training seeds, per-user pairing vs entropy on the same seeds).
- **Failure**: anything else. On failure, Paper B ships as answerability + saturation mechanism +
  "policy matches heuristic" with NO learned-win claim; no further recipe iteration without a new
  pre-registration.
- **Expectation (calibrated)**: capturable headroom from P0-a is ~+0.009-0.014 tail in q1-q4;
  a win of that size is the realistic best case.

Committed before first run. Any deviation must be recorded here with a dated note BEFORE eval.
