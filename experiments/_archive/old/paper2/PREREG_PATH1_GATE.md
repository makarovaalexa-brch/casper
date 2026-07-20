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

## Implementation note (2026-07-02, recorded BEFORE any training run or eval)

- **Reward realization (the "pick ONE" decision)**: `OBJ=reinforce REW=anytime` in continuous_policy2.py —
  the existing REINFORCE machinery with a new reward branch: per-turn reward = ABSOLUTE
  (WF*fullNDCG@10 + WT*tailNDCG@10)/T with WF=WT=1 (the exact Paper-D POLOPEN CURVEREW weighting),
  instead of the delta-over-previous-turn used by REW=ndcg. With return-to-go credit assignment the
  episode return G0 = mean over turns 1..8 of (full+tail) NDCG@10 = the pre-registered objective
  (return-to-go is an unbiased, lower-variance decomposition of the same terminal mean reward).
  PEN=0 (pure mean-NDCG reward, no answerability penalty — the penalty is not part of the registered
  reward). No critic, no entropy bonus, TAU/FTLR/HID at winner defaults.
- **Checkpoint selection**: CPU chunking (BCFROM + RESUME, EP=1 per chunk) exactly as in the clean
  trseed rerun (TRAINSEED_RESULT.md): each chunk runs the in-script val_ndcg (val = te[:300], split
  rng 123), epoch checkpoints copied to durable _gepN names, and the best-VAL checkpoint is selected
  MANUALLY across chunks by the SELVAL=tail criterion (identical to SELVAL=tail/USEBEST; per-chunk
  best-tracking resets at chunk boundaries so cross-chunk selection is by the recorded val numbers).
  Test users never touched for selection. Chunk boundaries reset Adam state + rng stream (same caveat
  as the clean trseed rerun). RL epochs per seed: 5 (>= the 3 used in the clean rerun; fixed in advance).
- **Primary-metric computation**: AUC over turns 1-8 = mean of per-user tail NDCG@10 at q=1..8
  (QPTS=1,2,3,4,5,6,7,8; the registered q-curve points {1,2,4,8} are a subset and are reported).
  Paired per-user bootstrap: per user, delta(policy - entropy) averaged over the 5 eval seeds and the
  3 training seeds; 10k bootstrap resamples over users; two-sided p (PAIRBOOT-style).

## OUTCOME (2026-07-03): FAIL — pre-registered NULL. ts0 primary ΔAUC(tail)=+0.0003 (p=0.85 paired), secondary tail@q4=−0.0034; campaign stopped (ts1 abandoned after ep2, ts2 not trained). See PATH1_ANYTIME_RESULT.md. Paper B ships with NO learned-win claim per the gate.
