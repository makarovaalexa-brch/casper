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
| post-Wed (end of queue, per user) | P4c answer-source ablation (empirical-noise channel fitted on real ML-1M ratings; cross-representation 2x2 V1/EASE answers x RecVAE/V1 rec; RAW-DATA pair answers = zero-circularity — elevates pair-native) | pending P4a | Holes 1+2 closed w/o humans if findings hold (task #46) |
| Jul 5 | P4b stretch: Goodreads cross-domain battery + PREREG predictions | **DONE** — arena = I2 RecVAE-d512 certified composite (phase-1E 500/500, 20k univ, @510). Actor 0.4240/0.2167; graded>>binary (−0.214); snap-loss −0.14 (=ML); adaptivity +0.011 vs per-user-max static (p=.997, STRONGER than ML). BUT continuous actor **LOSES to native item-8 fold 0.4837** (item PR 160). PREREG: **P1 CONFIRMED** (concept marginal +0.008@q8, effrank 15.5), **P2 REFUTED** (conc−item tail −0.084, not ≥0), **P3 CONFIRMED** (cont−disc gap +0.028→−0.060, tracks item rank 33→160), **P4 CONFIRMED** (7.98/8, lift≫entropy) | see experiments/instrument2/PHASE4B_GOODREADS.md |
| Jul 9 | Consolidate: Paper C core numbers updated IF enough gates passed; this doc finalized | — | clean handoff |
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
4. Geometric answer model: P4c answer-source ablation is the non-human closer (empirical channel + cross-representation + raw-data pairs); human study then covers only framing-lever + slider usability.
