# PATH1 RESULT — anytime-reward CASPER-R retrain: pre-registered NULL (gate FAIL)

**Date:** 2026-07-03. Pre-registration: `PREREG_PATH1_GATE.md` (frozen 2026-07-02 before any run, incl. the dated
implementation note). Code: `scripts/paper2/continuous_policy2.py` @ eeb6bd5 + the env-gated `REW=anytime` branch
(non-default; canonical paths untouched). Ruler: ML-1M, te[300:] (304 users), eval seeds {1,2,3,7,11}, NDCG@10
FULL + Cremonesi TAIL, frozen V1 encoder, binary answers (Paper B convention).

## Config (as registered)

- **Recipe**: CASPER-R scorer architecture, BC-entropy floor (BCTGT=entropy, BCEP=25, cached demos), then
  REINFORCE refinement with `OBJ=reinforce REW=anytime PEN=0`: per-turn reward = ABSOLUTE
  (WF·fullNDCG@10 + WT·tailNDCG@10)/T, WF=WT=1 — the exact Paper-D POLOPEN CURVEREW port; with return-to-go
  credit the episode return = **mean NDCG@10 over turns 1..8** (the registered anytime objective).
  FEATS=ext,ans, TAU=0.3, FTLR=5e-4, HID=128, no critic, no entropy bonus, no FIXQ1.
- **Training**: TRSEED=0, split-leak-fixed code (split pinned to rng(0)). 5 RL epochs, CPU-chunked
  (BCFROM + RESUME, EP=1/chunk; Adam state + rng stream reset at chunk boundaries — same caveat as the
  clean trseed rerun). Every epoch checkpointed (`policy_path1any_ts0_gep1..5.pt`).
- **Selection**: best-VAL, SELVAL=tail, val = te[:300] (split rng 123), never test. Chunk-level 3-decimal vals:
  gep1 0.109, gep2 0.105, gep3 0.102, gep4 0.108, gep5 0.109; full-precision re-eval of the tied pair
  (EVALCKS, seed 123, te[:300], q8 tail): gep1 **0.1094** vs gep5 0.1089 → **BESTVAL = gep1**
  (`policy_path1any_ts0_BESTVAL.pt`; `peak_path1any_ts0_manual.txt`). Training returns (mean anytime reward)
  0.3332, 0.3345, 0.3327, 0.3312, 0.3326 — flat: REINFORCE does not improve its own objective past ep1.
- **Comparator**: canonical static `entropy` heuristic on the identical users/splits/seeds — per-user dump
  reused from P0-a (`ansdump_entropy.csv` / `ansoracle_grid.csv`; entropy is training-free, so identical under
  any TRSEED). Ruler verified: entropy q8 = 0.3609/0.1397 = canonical 0.361/0.140.

## ts0 q-curve (seed-avg over {1,2,3,7,11}, mean±std across seeds)

| q | entropy FULL | anytime FULL | ΔFULL | entropy TAIL | anytime TAIL | ΔTAIL |
|---|---|---|---|---|---|---|
| 1 | 0.3193 ± 0.0047 | 0.3409 ± 0.0070 | **+0.0216** | 0.1034 ± 0.0038 | 0.1092 ± 0.0047 | +0.0058 |
| 2 | 0.3482 ± 0.0073 | 0.3569 ± 0.0053 | +0.0087 | 0.1201 ± 0.0047 | 0.1296 ± 0.0021 | **+0.0095** |
| 3 | 0.3507 ± 0.0044 | 0.3581 ± 0.0032 | +0.0074 | 0.1235 ± 0.0057 | 0.1337 ± 0.0043 | **+0.0102** |
| 4 | 0.3588 ± 0.0037 | 0.3594 ± 0.0032 | +0.0006 | 0.1353 ± 0.0038 | 0.1320 ± 0.0035 | −0.0034 |
| 5 | 0.3578 ± 0.0035 | 0.3623 ± 0.0032 | +0.0045 | 0.1352 ± 0.0047 | 0.1335 ± 0.0026 | −0.0017 |
| 6 | 0.3597 ± 0.0016 | 0.3602 ± 0.0026 | +0.0005 | 0.1394 ± 0.0037 | 0.1342 ± 0.0031 | −0.0052 |
| 7 | 0.3610 ± 0.0015 | 0.3580 ± 0.0027 | −0.0030 | 0.1402 ± 0.0036 | 0.1322 ± 0.0038 | −0.0080 |
| 8 | 0.3609 ± 0.0014 | 0.3598 ± 0.0031 | −0.0011 | 0.1397 ± 0.0039 | 0.1351 ± 0.0031 | −0.0046 |

(q=0 identical 0.3099/0.0808 by construction. Registered q-points {1,2,4,8} are the corresponding rows.)

## Pre-registered metrics

- **PRIMARY — TAIL NDCG@10 AUC over turns 1-8 vs entropy**: entropy 0.1296, anytime-ts0 0.1299,
  **ΔAUC = +0.0003** (paired per-user: +0.0004). Pooled paired per-user bootstrap (304 users × 5 eval seeds,
  10k resamples, two-sided): **95% CI [−0.0041, +0.0047], p = 0.85** — a statistical ZERO.
- **SECONDARY — tail@q4**: **−0.0034** (negative).
- q8 (reported, not a criterion): tail −0.0046, full −0.0011.
- FULL AUC (not registered, for the record): +0.0049, paired bootstrap CI [+0.0002, +0.0097], p = 0.04 —
  a small head-dominated front-loading gain (q1 full +0.022) that does not transfer to the tail metric that
  Paper B headlines.

## VERDICT: **FAIL** (pre-registered success criterion not met)

Success required ΔAUC(tail) > 0 for ALL 3 training seeds AND pooled paired bootstrap p < 0.05. ts0 — the
first and historically most favourable seed (the locked winner's seed, cf. TRAINSEED_RESULT.md) — is
+0.0003, p=0.85, with the secondary metric negative. The pooled-significance arm of the gate is therefore
already unreachable and the campaign was STOPPED per the gate's no-iteration clause: **ts1 was trained
(BC floor + RL ep1-2, val 0.305/0.106 @ep1) but abandoned unevaluated; ts2 was never trained.** No further
recipe iteration without a new pre-registration.

**Consequence (as pre-registered)**: Paper B ships as answerability + saturation mechanism + "policy matches
heuristic", with NO learned-win claim.

## Mechanism (why the anytime reward nets to zero) — the P0 saturation story, closed

Answered-so-far per turn (FULL runs, seed-avg):

| turn | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| entropy | 0.38 | 0.95 | 1.45 | 2.12 | 2.31 | 3.31 | 4.28 | 5.16 |
| anytime ts0 | 0.56 | 1.22 | 2.18 | 3.10 | 3.89 | 4.78 | 5.70 | **6.50** |

The anytime reward did exactly what it pays for: the policy asks more-ANSWERABLE questions at every turn
(+1.0-1.6 answered by mid-conversation, 6.50 vs 5.16 at q8) and converts that into genuine front-loading —
tail +0.006/+0.010/+0.010 at q1-q3, matching the P0-a answerability-oracle headroom (+0.008..+0.014 in that
range) almost exactly. But per P0-b the marginal value of answers 5-8 is ≤+0.001, and the answerable-but-
less-divisive concepts it substitutes cost late-turn informativeness: q4-q8 tail goes −0.003..−0.008. The
early gain and the late loss cancel to ΔAUC ≈ 0. This is the same efficiency-not-endpoint bound P0-a
established for ANY answerability-aware selection — realized by a learned policy, and insufficient even on
the AUC (anytime) yardstick chosen to be maximally favourable to front-loading.

## Provenance

- Code: `REW=anytime` branch + `WF/WT` weights + delta-vs-absolute reward switch in `rollout_sample`;
  EVALBASE empty-skip in EVALCKS (all env-gated, non-default). Analysis: `scripts/paper2/path1_auc_boot.py`.
  NOT committed (per task instruction). Locked/winner checkpoints untouched.
- Checkpoints: `policy_path1any_ts0_{bc,gep1..gep5,BESTVAL,best,last}.pt`, `policy_path1any_ts1_{bc,gep1,gep2}.pt`
  (abandoned), `peak_path1any_ts0_manual.txt`, `policy_runs.tsv` row `path1any_ts0`.
- Logs: `experiments/paper2/path1any_ts0_{bc,rl1..rl5,valsel,test}.log`, `path1any_ts1_{bc,rl1,rl2}.log`.
- Data: `data/movielens/.cache/path1_test_ts0.csv` (grid), `experiments/paper2/path1_ansdump_ts0.csv`
  (per-user dump), comparator `experiments/paper2/ansdump_entropy.csv` + `ansoracle_grid.csv` (P0-a).
- Correction for the record: an intermediate coordinator readout quoted entropy tail-AUC 0.1322 → Δ −0.0023;
  the correct entropy tail-AUC(1-8) on this grid is 0.1296 → Δ +0.0003, p=0.85. Same verdict (FAIL), the
  delta is a statistical zero rather than negative.
