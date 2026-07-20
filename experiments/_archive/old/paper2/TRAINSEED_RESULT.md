> **⚠ 2026-07-02 LATER: ts1/ts2 rows below are INVALID — split-leak bug.** TRSEED also seeded the
> train/test split shuffle, so TRSEED!=0 runs trained on ~80% of canonical test users (leakage).
> TRSEED=0 rows (the winners) are unaffected (byte-identical). CASPER-R conclusion survives a fortiori
> (ts1 lost even WITH leakage inflation). **D1 robustness RE-VERIFIED CLEAN 2026-07-02 — see the CLEAN RERUN section directly below.**
> Fix: split pinned to rng(0) in continuous_actor.py + continuous_policy2.py. See PAIRTRAIN_RESULT.md.

# ✅ CLEAN RERUN on the FIXED split (2026-07-02, code 84cd358) — D1 robustness RE-ESTABLISHED

Retrained D1 at TRSEED=1,2 on the fixed split (split pinned to rng(0); TRSEED varies torch init + training sampling
only). Exact winner recipe: warm-start INIT=PAPER_C_CONT_WINNER/policy_cont_actor_recon_variant.pt, DIVW=1.0 DTAU=2.0
NUMAT=2500, CONTMODE=cont GRADED=1 OBJ=ustar NOBC=1 FEATS=ext,ans, frozen V1, EP=10, SKIPVAL, eval `_last`(=ep10).
(Foreground chunked 4+3+3 epochs via RESUME — optimizer state + rng stream reset at the 2 chunk boundaries; a
legitimate independent random draw for a robustness check, noted for the record.)

**Contamination sanity check PASSED** (graded val_ndcg on te[:300], ANSLEARN=geom harness, ep1 checkpoints):
ts1 ep1 val 0.3311/0.1122, ts2 ep1 val 0.3279/0.1176 — normal cold-start territory (clean ts0 ep1 ref 0.3053/0.0959),
NOT the ~0.36/0.147 leak signature the contaminated runs showed. Training return curves normal
(ts1 0.9204→0.9380, ts2 0.9183→0.9372 over 10 epochs).

## D1 (Paper C flagship) — clean numbers (COMPARE4 ONLYACTOR, seed-avg {1,2,3,7,11}, te[300:], q8, GRADED)
| TRSEED | FULL | TAIL | ckpt |
|---|---|---|---|
| 0 (= locked winner, unaffected by the leak) | 0.3780 ± 0.0032 | 0.1782 ± 0.0065 | policy_phase3_d1divw_last.pt (a63fec1e) |
| 1 (CLEAN split) | 0.3764 ± 0.0043 | 0.1785 ± 0.0054 | policy_d1divwCL_ts1_FINAL_ep10.pt (e0b14e48) |
| 2 (CLEAN split) | 0.3756 ± 0.0050 | 0.1748 ± 0.0049 | policy_d1divwCL_ts2_FINAL_ep10.pt (bbf0dd5b) |
| **mean ± std across 3 training seeds** | **0.3767 ± 0.0012** | **0.1772 ± 0.0021** | |

vs anchors: uent+GRAW 0.367/0.158 → **+0.010 FULL / +0.019 TAIL**; CASPER-R 0.360/0.152; entropy 0.361/0.140.
**VERDICT: D1's continuous-vs-discrete margin SURVIVES clean training-seed variation — every trseed's tail
(0.1782/0.1785/0.1748) clears the best graded static baseline 0.158 by ≥ +0.017; spread is TIGHTER than the
invalidated run (±0.002 vs ±0.006). Paper C's flagship robustness claim stands, now on uncontaminated splits.**
(Binary control unchanged in kind: ts1 0.3455/0.1305, ts2 0.3474/0.1380 — graded regime still required.)

## CASPER-R clean rerun — ts1 CLEAN CONFIRMS ≈entropy; ts2 DEFERRED (a fortiori)
Recipe: continuous_policy2.py, BCTGT=entropy OBJ=ustar SELVAL=tail USEBEST=1 FEATS=ext,ans, TRSEED=1 on the fixed
split; BC floor + 3 RL epochs (foreground-chunked EP=1 via BCFROM/RESUME), val-best selected manually across chunks
(gep1 val 0.313/0.106, gep2 0.314/0.105, gep3 0.312/0.103 → best=gep1 by SELVAL=tail; peak_casperRCL_ts1.txt).
Ep1 val sanity PASSED: clean ts1 0.313/0.106 ≈ clean-split ts0 0.3148/0.1103; the contaminated ts1 had shown
0.3544/0.1316 (the leak signature, visible in the old peak_casperR_ts1.txt).

Canonical EVALCKS (seed-avg {1,2,3,7,11}, te[300:], binary; evalcks_CLts1.csv), casperRCL_ts1_best vs entropy:
| | q2 | q4 | q8 |
|---|---|---|---|
| entropy FULL/TAIL | 0.3484/0.1202 | 0.3588/0.1352 | 0.3608/0.1398 |
| CASPER-R ts1-clean FULL/TAIL | 0.3482/0.1284 | 0.3588/0.1396 | 0.3622/0.1362 |
| TAIL delta | +0.008 | +0.004 | −0.004 |
=> **≈entropy on the clean split too** (mixed small deltas, no robust learned win) — the prior conclusion is
CONFIRMED (ts1 was the originally-negative seed and it lost/tied even WITH leak inflation, so the conclusion was
a fortiori safe). **ts2 deferred; conclusion a fortiori + ts1-clean confirmed** (CPU reprioritized to the
entity-ladder control). B stays the answerability paper; the learned-win claim stays dropped.

Checkpoints: policy_d1divwCL_ts{1,2}_{gep1..gep10,FINAL_ep10,_last}.pt, policy_casperRCL_ts1_{bc,gep1..gep3,best}.pt
(best sha c071ef25; winners untouched; durable copies + SHA1SUMS in experiments/checkpoints/trainseed_clean_2026-07-02/).
Logs: experiments/paper2/d1clean_ts{1,2}_chunk{A,B,C}.log, d1clean_ts{1,2}_eval.log, casperRclean_ts1_{bc,ep1,ep2,ep3,eval}.log.

---

# ⚠ INVALIDATED SECTION BELOW (kept for the record) — original rerun on the LEAKY split
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
