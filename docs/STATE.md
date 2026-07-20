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

## Running now
- Nothing.

## Next action (proposed — awaiting go)
Per `VISION.md` sequencing (closed probes first): land the unified recommender's uncertainty layer, then
the discrete adaptive policy. Concretely — run the PrecAcc sanctioned-subsample experiment (frozen pbC mean
+ analytic covariance) to clear gates G0/G1/G2 on full+tail, then build the adaptive policy on top.
