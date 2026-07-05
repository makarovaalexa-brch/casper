# INSTRUMENT 2.0 — Live Schedule, State & Handoff (updated 2026-07-04 late)

Living doc — updated at every phase gate, committed+pushed each time. Resume point for any session.

## Dated schedule (core cut; user available until Wed Jul 9)

| date | phase | state | gate / deliverable |
|---|---|---|---|
| Jul 4 | P1 replicate RecVAE ML-20M | **DONE** — test NDCG@100 0.4346 (pub 0.442, kill 0.43) | PASS |
| Jul 4 | P1.5 elicitation smoke | **DONE** — k-curve monotone (0.136→0.290 @k8), latent smooth, no sparse collapse | PASS ×3; seed z=0 rule |
| Jul 4 | P1.6 micro-battery | **DONE** — input-space concept fold FAIL → z-space channel; NO amortization gap (encoder-only); native σ correctly signed (−0.081, weak) | design locked |
| Jul 5 | P2 ports | **DONE — BOTH PASS** — ML-1M d512 0.5541 ties EASE (0.5549), +0.147 over V1; Goodreads +0.237 = 0.78×EASE = 21.6×V1 (lower bound, 40k-user subsample; lever = more train users) | PASS; d=512 both; z=0 seed mandatory on ML-1M (cold floor below MOSTPOP) |
| Jul 5-6 | P3 CASPER-ize: z-space concept/dislike channel (benchmark both polarity designs), k=1..20 gates G1-G8 on both datasets | **DONE — CERTIFIED** — W1 winner = additive geometric-query operator (0.4688 BEATS item-8 fold 0.4603 at 8 turns); concept channel = member-bag encode (+0.15 vs floor both datasets, reverses 1.6); dislike = operator a<0 (spec −33.7) + preservation-exact two-channel variant; **gates 8/8 ML-1M AND 8/8 Goodreads** | PASS both (see experiments/instrument2/PHASE3_GATES.md) |
| Jul 5 | P4a essential battery | **DONE — FLAGSHIP SURVIVES**: actor 0.4907/0.2982; cont>disc +0.046/+0.062 sig; graded>>binary (0.491 vs 0.117); snap-loss 4x larger; adaptivity +0.019 full sig / tail tie; from-scratch collapses -> BC-warm from decoder-SVD basis; div-field inert on I2 | ALL CLAIMS SURVIVE |
| Jul 5 | P4c answer-source ablation (empirical-noise channel fitted on real ML-1M ratings; cross-representation V1/EASE foreign-geometry answers; RAW-DATA pair answers = zero-circularity) | **DONE — FIDELITY BOUNDARY**: under the empirical channel (corr(s,rating)=0.516, 1.9-bit, 5-level noisy) the P4a ordering **INVERTS** — continuous actor collapses 0.491→0.150 (train-noisy recovers 0.287, still loses), while discrete concept-8 0.325 / item-real 0.394 / item-fold 0.463 stay robust. NON-CIRCULAR: foreign geometry V1 0.392/0.396, EASE 0.439; raw pairs 0.223 (cov 1.0) but 0.003 cov unrestricted (answerability binds). Snap-loss REVERSES (item-snap +0.051 = catalog anchors denoise); graded>>binary evaporates to +0.005 on actor. Continuous premium = an ANSWER-FIDELITY phenomenon; two-regime honest conclusion woven into papers C(new subsec sec:i2fidelity + abstract + deployment rule)/B(P4c vindication para)/D(noise-robust backbone sentence). | Holes 1+2 CLOSED w/o humans; source experiments/instrument2/P4C_ANSWER_SOURCES.md; artifacts .cache/instrument2/p4c_*.json + p4c_actor_noisy_s0.pt |
| Jul 5 | P4b stretch: Goodreads cross-domain battery + PREREG predictions | **DONE** — arena = I2 RecVAE-d512 certified composite (phase-1E 500/500, 20k univ, @510). Actor 0.4240/0.2167; graded>>binary (−0.214); snap-loss −0.14 (=ML); adaptivity +0.011 vs per-user-max static (p=.997, STRONGER than ML). BUT continuous actor **LOSES to native item-8 fold 0.4837** (item PR 160). PREREG: **P1 CONFIRMED** (concept marginal +0.008@q8, effrank 15.5), **P2 REFUTED** (conc−item tail −0.084, not ≥0), **P3 CONFIRMED** (cont−disc gap +0.028→−0.060, tracks item rank 33→160), **P4 CONFIRMED** (7.98/8, lift≫entropy) | see experiments/instrument2/PHASE4B_GOODREADS.md |
| Jul 5 | Consolidate | **CORE COMPLETE 4 DAYS EARLY** — all five phases done, papers A/B/C/D updated+synced+pushed | DONE |
| post-Wed (deferred) | ML-25M port; full 5-paper I2 update; remaining ladder rungs; compile passes; D+E merge; ECIR assembly (Oct 2) | handoff | tasks #39-#44 |

## Paper-update rule (user instruction)
As soon as gates pass sufficiently to show the thesis likely works on I2 (= P3 gates + first P4a
signal), START adding I2 results to papers (C first) — do not wait for the full battery.

## Binding design rules (from P1.5/P1.6 — do not relitigate)
- Empty state = z=0 (encoder NaNs on empty input). Belief update = amortized encoder pass ONLY.
- Concepts + dislikes enter via a LEARNED tag→z latent channel (input-space pseudo-items FAILED:
  below-floor fidelity, negative additivity, dislikes structurally impossible in nonneg multinomial).
  Crude z-subtraction works (horror −15.4pts specific) = the channel's existence proof.
- Uncertainty = native posterior (μ,σ) — correctly signed; too weak alone for stopping gates.
- Tag selection by LIFT, not raw affinity ("original"-tag degeneracy).

## Key artifacts
scripts/instrument2/*, .cache/instrument2/* (ml20m recvae/multvae ckpts), experiments/instrument2/
(PHASE1_REPLICATION, PHASE15_SMOKE, PHASE16_MICRO, PHASE2_PORTS[in progress]); plan INSTRUMENT2_PLAN.md;
research INSTRUMENT2_RESEARCH_RAW.txt; rollback tag pre-instrument-rebuild-2026-07-04 (pushed);
papers backed up casper/papers/ + C:\dev\phd_BACKUP_2026-07-04. Task IDs: #37-#45.

## Open risks (ranked)
1. Flagship may shrink on strong instrument (FTREC precedent) — P4a answers it; honest either way.
2. Goodreads VAE may fall structurally short of EASE (MSD pattern) — candidate B = ELSA-style low-rank.
3. Polarity channel design proven (P3): operator a<0 gives strong specific demotion (spec −33.7,
   beats 1.6's −14.5); two-channel encoder is preservation-exact but weak (−4.3) = fallback only.
4. ~~Geometric answer model: P4c answer-source ablation is the non-human closer.~~ **RESOLVED → the fidelity-boundary finding.** P4c ran: the geometric-answer premium is real on high-fidelity/clean channels but is an ANSWER-FIDELITY phenomenon — under a real-ratings-calibrated noisy 5-level channel the continuous/discrete ordering inverts and answerable discrete + graded real answers win (robust, non-circular). Woven honestly as a two-regime scoping result (papers C/B/D + this handoff). Remaining sub-risk: closing the continuous policy's noisy-channel gap is a policy-ROBUSTNESS problem (train-noisy 0.287) = future work, not action-space. Human study now covers only framing-lever + slider usability (slider/comparison-grade fidelity is the deployment precondition for the continuous channel).

NEW NOTE (squeeze): the R1 Bayesian arm MUST be evaluated under **BOTH** answer channels (noiseless P4a AND the empirical noisy channel). Post-P4c, the **noisy channel is the more important** of the two — a Bayesian-design win on the clean channel that does not hold under the fitted noisy channel is not deployment-relevant.


## LOCKED PLAN + BUDGET (2026-07-05, user at 67% weekly usage until Tue)

Priority order if budget tightens (cut from the bottom):
1. P4c answer-source ablation (RUNNING) — Opus ~150-200k tok, ~2-4h CPU
2. Sexiness pass (RUNNING) — Opus ~120-180k tok, ~0 CPU, cents of API
3. Squeeze R0-1 oracle+Bayesian arm (RUNNING) — Opus ~120-180k tok, ~1-3h CPU
4. Squeeze R2-4 (gated on R0-1 verdict) — Opus ~200-300k tok, ~4-8h CPU
5. Final consolidation: results->papers weave + handoff finalization — Opus ~100k + Fable coordination

Est. total remaining: ~0.7-1.0M Opus tokens + ~40-60k Fable tokens + ~8-15h CPU wall-clock
(vs ~3M-equivalent remaining in the weekly budget -> comfortable, ~2x margin).
Token rules in force: no more deep-research workflows; compact agent reports; coordinator polls
files rather than resuming agents; batch writing tasks; Fable only at gates + strategy.
User plan post-Tue: 1-2 replication datasets, finalize drafts, compile.


## SQUEEZE ARC R0-1 (Jul 6, DONE) + FINAL STATE
- Compass: privileged direction ceiling 0.878/0.411; item-subset 0.755 > full-profile 0.554; actor ~7% of headroom.
- Clean channel: actor = realizable myopic ceiling (D-optimal -0.011; Kalman shrinkage hurts; D-opt needs decoder metric).
- NOISY channel: Kalman-D-optimal repeat-probing WINS +0.095 (0.248 vs actor 0.150) -> noise robustness = the realizable prize. Woven into C's fidelity subsection.
- R2-4 PIVOTED to handoff (task #49): noise-robust policies + non-peeking decoder-metric EIG selector toward the 0.755/0.878 ceilings.
ALL CORE + PRIORITY WORK COMPLETE. Remaining for user: 1-2 replication datasets (ML-25M I2 port spec via PHASE2_PORTS conventions), draft finalization, compile passes (#43), ECIR assembly (#13), human micro-study (#12), D+E merge (#44), squeeze R2-4 (#49).
