# INSTRUMENT 2.0 — Rebuild Plan (2026-07-04, DRAFT FOR USER APPROVAL — nothing trains before sign-off)

Trigger: the V1 instrument (d=64 fold-in over frozen SVD factors) is not competitive with top models
(Goodreads: V1 +0.013 headroom vs EASE +0.30 on identical arena; ML-1M: below WRMF under a self-chosen
tolerance). All thesis results currently sit on this ruler. Full deep-research report (104 agents,
adversarially verified): experiments/paper2/INSTRUMENT2_RESEARCH_RAW.txt.

## The recipe from the literature (no wheel reinvention)

**RecVAE** (Shenbin et al., WSDM 2020) — the verified best fit for ALL four requirements:
- Compact amortized latent (d=200-512): ONE encoder pass folds any partial profile → differentiable
  user vector = exactly the action space the elicitation policies need.
- SOTA-class under the rigorous non-sampled Liang protocol: **ML-20M NDCG@100 0.442 vs EASE 0.420,
  Mult-VAE 0.426, tuned iALS 0.425, SLIM 0.401**; ties EASE on Netflix.
- Load-bearing recipe details (verified, must copy faithfully): composite prior (N(0,I) mixed with
  previous-epoch posterior), per-user beta = gamma*|X_u| (gamma≈0.005), alternating 3:1
  encoder/decoder updates, denoising (mu=0.5) on encoder only. Multinomial likelihood.
- CPU-feasible; **SANSA/ELSA** (sparse/low-rank EASE) as the CPU-cheap linear anchors
  (SANSA: Amazon-Books in 49s / 9GB).

**Verified caveat (drives Phase 2 design):** the winner is dataset-dependent — on sparse huge-catalogue
data (MSD) linear EASE-class beats the whole VAE lineage by ~19%. Goodreads may be MSD-like. So the
acceptance bar is per-dataset against TUNED EASE/SANSA/iALS, and ELSA (low-rank linear, still gives a
compact-ish fold) is candidate B if the VAE loses on Goodreads.

**Verified open novelty (the gift):** no published work combines a latent VAE recommender with
question-asking/elicitation in its latent space. The rebuild is not damage control — it upgrades the
thesis contribution to "continuous elicitation in the latent space of a SOTA-class recommender."

## Phases (each gated; kill-criteria explicit)

**Phase 0 — diagnostics wrap (running now, no new compute):** d-sweep + implicit-EASE finish →
position Goodreads on the ML↔MSD spectrum; informs d and candidate-B likelihood.

**Phase 1 — REPLICATE RecVAE on its home benchmark (~3-4 days).**
Faithful reimplementation (or author-code port) on ML-20M, Liang protocol. TARGET: NDCG@100 ≥ 0.435
(published 0.442). Also replicate Mult-VAE (0.426) as the simpler control. KILL: if <0.43 after 3
debugging passes, stop and report before proceeding.

**Phase 2 — port to OUR arenas + honest bars (~3-4 days).**
Datasets: ML-1M (canonical protocol), ML-25M (full), Goodreads composite. On EACH: tuned EASE + SANSA
(+ iALS if cheap) as the bar. ACCEPTANCE: RecVAE ≥ EASE on ML-1M/25M; on Goodreads ≥ 0.8×EASE-headroom
or better (if structurally short per the MSD pattern → switch to candidate B: ELSA-style low-rank
linear with an amortized fold head; decision documented, not fudged).

**Phase 3 — CASPER-ize + gates (~4-6 days).**
(a) Explicit-signal handling: Mult-VAE lineage is implicit-binary; our answers are graded likes AND
dislikes → two-channel input (x_like, x_dislike) or signed weighting — DESIGN DECISION, options
benchmarked, polarity kept structural (the January lesson). (b) Concept/tag channel via the proven
learned-channel recipe (item-preservation gate). (c) k=1..N fold sweep (published protocol only tests
80/20 — our G-gates test k=1..20). (d) FULL acceptance suite (G1-G8 incl. no-harm calibration) on
ML-1M + ML-25M + Goodreads. REQUIREMENT: gates pass on all three (Goodreads pass is now a requirement,
not an aspiration).

**Phase 4 — re-anchor the thesis (~1-1.5 weeks).**
Canonical policy battery on the new instrument: ML-1M first (entropy / uent+GRAW analogue / CASPER-R
class / D1 recipe / snap-loss / static-8 / ladder core), then Goodreads phase-2 (the cross-domain
battery + pre-registered predictions, finally on a valid arena). HONEST RISK, stated up front: some
findings may shrink on a stronger instrument (the FTREC/V2-ST precedent: stronger recommenders absorb
policy edges). Better to know in July than at ECIR rebuttal. Papers update from the tagged baseline
(pre-instrument-rebuild-2026-07-04 = full rollback insurance, pushed to GitHub).

## Timeline & decision points
Total ≈ 3-4 weeks → policy re-anchor complete early-to-mid August; ECIR (Oct 2) remains feasible.
USER DECISIONS NEEDED: (1) approve plan; (2) d target (research supports 200-512; d-sweep will refine
— default 256); (3) Phase-3 dislike-handling: benchmark both options or pick one a priori?
