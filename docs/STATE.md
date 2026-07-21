# STATE — where the project is now

> The single current-status doc. **Overwrite as things change.** Pointed to from `MEMORY.md`.
> End-state is in `VISION.md`. Last updated: **2026-07-20**.

## Current recommender (canonical — full / tail NDCG@10, scored with the LEARNED decoder bias, not popb)
| ckpt (`.cache/…`) | role | full / tail |
|---|---|---|
| `set_mn/pbC_best.pt` | concept-capable set-encoder — **frozen mean for the current direction** | **0.4946 / 0.3372** |
| `set_mn/paord_best.pt`, `pb2_best.pt` | item-only base encoders | ~0.486 / ~0.485 |
| `signed_latent/a0c_best.pt` | dense teacher / oracle ceiling | 0.4961 |
| cold-start (popb only) | — | ≈ 0.19 full → **huge headroom** |

_Canonical current recommender = **pbC set-encoder** (interview-native). The RecVAE-d512 instrument (0.4998/0.3443, `EXPERIMENTS.md` recommender-core) is a **separate Paper-A ruler**, not the interview recommender._

## Current direction
- **Paper B lead = uncertainty-shrinkage / belief-distribution** — the belief-pool elicitation invariant
  (closed-form update, Σ only shrinks, NDCG never drops per question) → Kalman-incompatible-with-canonical
  → the PrecAcc redesign. Design lineage in `docs/design/`; experiments in `docs/EXPERIMENTS.md §B (★ LEAD)`.
- **Unified multi-channel belief recommender = PrecAcc** (`scripts/train_precacc.py`): frozen pbC mean +
  analytic precision-accumulator covariance. **Coded, adversarially reviewed, NOT yet run.**
- Design: `docs/design/DESIGN_PRECACC.md`. Planned first run: `--conc_dirs conc_dirs_meanshift.npy
  --max_users 30000 --max_atoms 40` → sign-proof G, α's crediting concepts, G1c/G2 curves (full+tail).

## Settled — do NOT relitigate (see memory)
- Kalman belief-pool **incompatible** with the canonical recommender (craters full 0.167→0.097).
- Metric bug fixed: score with the learned bias, not popb. **Always report FULL and TAIL.**
- Adaptivity **proven** (+47% tail from one genre question — HARD RULE #2).
- Concepts **lose to items** as an interview router; a concept answer ≈ a coarsened item-watch signal.

## Baseline campaign (2026-07-21, running)
- **ML-20M Liang snap: EASE PASS +0.0003 (0.4203 vs 0.420) AND RecVAE PASS +0.0005 (0.4425 vs 0.442;
  R@20 0.4144 vs 0.414, R@50 0.5525 vs 0.553)** — split+metric+implementation fidelity certified
  (split stats exact: 9,990,682 / 116,677 / 20,108). iALS 0.3580, EDLAE 0.4167, kNN 0.2995 (advisory),
  Pop 0.1906. Mult-VAE/DAE queued (one command: `run_snap_ml20m.py --only dae,multvae`, ~16 h CPU).
- **ML-25M G0 (same canonical ruler, verified line-for-line): EDLAE 0.5230/0.3464 is the NEW BAR,
  EASE 0.5078/0.3394 — both ABOVE the in-house anchors** (pbC 0.4946/0.3372, RecVAE-d512 0.4998/0.3443).
  R1 obligation is now "close ~2.8pt to EDLAE", not "tie RecVAE". kNN 0.4272/0.2208, iALS 0.4272/0.3172,
  Pop 0.2851/0.0589 (pop gap vs MOSTPOP 0.2522 diagnosed benign: like-count vs all-band popb).
- **⚠ ANCHOR FORENSIC VERDICT (Jul 21): pbC/paord belief-pool numbers are IRREPRODUCIBLE from git** — the
  training-time forward was never committed (lived in the Jul-15 working tree, overwritten same day); pbC
  measures 0.2428/0.1505 under every recoverable pin. **pb2 (attn pool) is the ONLY reproducing in-house
  anchor: 0.4917 full / 0.3071 tail** (recorded 0.4852/0.3295; tail −0.022 unexplained). RecVAE-d512
  rebuildable ~7h via the snap-certified recvae.py. **PrecAcc decision needed:** retrain belief-pool encoder
  under committed code, or re-base the frozen mean on pb2. New HARD RULE 10: commit training code before any
  run. `scripts/_verify/verify_pbc.py` reproduces the forensics.

## Paper A restart (2026-07-20)
- Lit re-review DONE: `external_literature/findings/paperA_recommender_landscape.md` (SOTA, taxonomy, gap
  verdict, baseline bank) + per-paper records in `external_literature/papers/`. Gap CONFIRMED (no system holds
  R1–R5; Biyik 2023 = central threat). Design verdict → `docs/design/PAPERA_DESIGN_VERDICT_2026-07-20.md`:
  **frozen shallow tower + PrecAcc conjugacy IS the Paper-A instrument**; obligations = metric bridge
  (NDCG@10/ML-25M vs published NDCG@100/ML-20M) + c4 baseline tiers before any "ties SOTA" claim.

## Running now
- Nothing.

## Next action (proposed — awaiting go)
Per `VISION.md` sequencing (closed probes first): land the unified recommender's uncertainty layer, then
the discrete adaptive policy. Concretely — run the PrecAcc sanctioned-subsample experiment (frozen pbC mean
+ analytic covariance) to clear gates G0/G1/G2 on full+tail, then build the adaptive policy on top.
