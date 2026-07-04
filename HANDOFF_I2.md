# INSTRUMENT 2.0 — Live Schedule, State & Handoff (updated 2026-07-04 late)

Living doc — updated at every phase gate, committed+pushed each time. Resume point for any session.

## Dated schedule (core cut; user available until Wed Jul 9)

| date | phase | state | gate / deliverable |
|---|---|---|---|
| Jul 4 | P1 replicate RecVAE ML-20M | **DONE** — test NDCG@100 0.4346 (pub 0.442, kill 0.43) | PASS |
| Jul 4 | P1.5 elicitation smoke | **DONE** — k-curve monotone (0.136→0.290 @k8), latent smooth, no sparse collapse | PASS ×3; seed z=0 rule |
| Jul 4 | P1.6 micro-battery | **DONE** — input-space concept fold FAIL → z-space channel; NO amortization gap (encoder-only); native σ correctly signed (−0.081, weak) | design locked |
| Jul 5 | P2 ports: ML-1M + Goodreads, d-discovery {64..512}, EASE/V1 bars | **RUNNING** (agent launched Jul 4 eve) | ML-1M: RecVAE ≥ EASE and ≥ V1. Goodreads: headroom ≥0.8×EASE (≥+0.15 = success vs V1's +0.011) |
| Jul 5-6 | P3 CASPER-ize: z-space concept/dislike channel (benchmark both polarity designs), k=1..20 gates G1-G8 on both datasets | pending P2 | gates PASS both datasets (Goodreads REQUIRED) |
| Jul 6-8 | P4a essential battery @ML-1M: static-graded vs D1-recipe actor (3 trseeds), snap-loss, static-8 | pending P3 | THE question: flagship survives strong instrument? |
| Jul 8-9 | P4c answer-source ablation (empirical-noise channel fitted on real ML-1M ratings; cross-representation 2x2 V1/EASE answers x RecVAE/V1 rec; RAW-DATA pair answers = zero-circularity — elevates pair-native) | pending P4a | Holes 1+2 closed w/o humans if findings hold |
| Jul 8-9 | P4b stretch: Goodreads cross-domain battery + PREREG predictions | pending P3+time | rank-law out-of-sample test |
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
3. Polarity channel design unproven (P3 benchmark, two designs).
4. Geometric answer model: P4c answer-source ablation is the non-human closer (empirical channel + cross-representation + raw-data pairs); human study then covers only framing-lever + slider usability.
