# TRAINING-SEED robustness rerun (2026-07-02, review S1/B-B3/B-B4/C-B5) — CRITICAL, possible B overturn

## Background / method
Reviewer flagged: no training-seed discipline; the "6 seeds" in prior tables were EVAL-split seeds only (training was
hardcoded `torch.manual_seed(0)` at line 9 of continuous_actor.py + continuous_policy2.py). PATCHED both scripts to read
`TRSEED` env (line 9 rng+torch seed; continuous_actor.py:1823 actor training rng). Data splits stay deterministic (range-
based te[:300]/te[300:]); TRSEED varies network init + training-sampling only. Default TRSEED=0 reproduces the winner.

Then RETRAINED across training seeds (fresh TAGs, winners untouched): CASPER-R (continuous_policy2.py, BCTGT=entropy
OBJ=ustar SELVAL=tail USEBEST=1) TRSEED=0,1,2; D1 (continuous_actor.py, DIVW=1.0 DTAU=2.0 GRADED) TRSEED=0,1,2.
Eval: 5 canonical eval seeds {1,2,3,7,11}, te[300:], q-curve 0/2/4/8.

## CASPER-R (Paper B) — learned-win does NOT survive training-seed variation
TAIL delta (CASPER-R - entropy heuristic), by TRAINING seed:
| TRSEED | q2 | q4 | q8 |
|---|---|---|---|
| 0 (=winner's seed) | **+0.008** | +0.005 | +0.001 |
| 1 (diff init)      | **-0.007** | -0.001 | -0.003 |
| 2                  | -0.005 | +0.006 | -0.002 |
| **mean (3 seeds)** | **-0.001** | **+0.003** | **-0.001** |
Also: q8 tail ts0 +0.001, ts1 -0.003. FULL metric ~ties throughout both seeds.

=> The ts0 advantage (front-loading +0.008@q2 AND the +0.011 test-selected q8 tail) is a FAVOURABLE TRAINING SEED (the
same seed the locked winner was trained on). With a different init CASPER-R is BELOW entropy at every q. Averaged over
training seeds, **CASPER-R ~= entropy** — the learned-policy advantage (q8 win AND front-loading) is NOT robust.
Prior finding (val-selection, one training seed) showed q8 win +0.011->+0.001; this shows even the front-loading is
seed-fragile. Robust B content = ANSWERABILITY (concepts>>items, training-free, solid) + CASPER-R MATCHES the heuristic
(not beats). Learned-win claim should be dropped. (Consistent with user's point-1 reframe; goes further.)

## D1 (Paper C flagship) — ROBUST across training seeds (C STANDS)
D1 graded FULL/TAIL: ts0 0.3799/0.1828, ts1 0.3733/0.1713, ts2 0.3759/0.1753 => MEAN 0.376+/-0.003 / 0.176+/-0.006.
Stable. Even with BOTH sides training-seed-averaged, continuous beats discrete (vs graded uent+GRAW 0.367/0.158 =>
+0.009/+0.018; vs CASPER-R/entropy ~0.36/~0.145 => larger). C's continuous-beats-discrete margin + snap-loss stand.
CONCLUSION: C = robust flagship; B = answerability paper (learned-win dropped). ts0=TRSEED0 reproduces winner exactly
(patch validated). STILL TODO: C-B4 graded-discrete control eval (policy_gradeddisc_rerun); paired-user bootstrap harness.

## Provenance
Code: TRSEED patch in continuous_actor.py + continuous_policy2.py (line 9, +actor rng 1823). Logs:
experiments/paper2/trseed_rerun.log; CSVs data/movielens/.cache/evalcks_ts{0,1,2}.csv; checkpoints
policy_casperR_ts{0,1,2}_best.pt, policy_d1divw_ts{0,1,2}_last.pt (fresh TAGs; WINNERS entdistill_ep4 /
policy_phase3_d1divw_last.pt / PAPER_*_WINNER untouched). Batch watcher bksr650e5.
NOT YET FULL: ts2 CASPER-R + all D1 seeds. Paired per-user bootstrap harness still TODO (the correct significance test).
