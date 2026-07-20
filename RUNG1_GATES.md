# RUNG I — PILOT KILL-SWITCH RESULT: **STOP** (value channel INERT)

Faithful execution of `DESIGN_SHEET_RUNG1_ENCODER.md`. Encoder built at `scripts/rung1_encoder.py`
(reuses `i25_lib` / `i25_fold_v3_sampler`; frozen RecVAE-d512 decoder; every load-bearing mechanism
wired — see `.cache/rung1/WORKING_NOTES.md`). **Phase 0 PASSED; Phase 1 PILOT = STOP. Per the design's
STOP rule and the brief, the full train was NOT run.** Honest report; a failure is reported as a failure.

## Phase 0 — build + unit sanity: ALL PASS (`.cache/rung1/build_sanity.json`)
| check | result |
|---|---|
| forward runs on toy batch | PASS |
| intercept exact at init (empty → μ=native, max-dev) | 0.0 |
| native intercept (empty tokens, non-empty anchor) max-dev | 0.0 |
| NaN screen drops non-finite emb + finite loss (F6) | PASS |
| γ(meh) ≈ 1 (identity anchor) | 1.000 |
| τ init ≈ 1 | 1.000 |
| **signed-γ CAN invert** (hand-set −slope: hated pool score −0.299 < loved +0.027) | PASS |
| one training step, finite loss + channel-dropout | PASS |

The architecture is sound and the inversion *mechanism* is provably capable (Phase 0 §5). The failure
below is that **training did not drive the value head to use it.**

## Phase 1 — PILOT (2k train / 1k val users, 8 epochs, ~9.5 min; `.cache/rung1/pilot_result.json`)
Best val NDCG@10 = **0.3403** (native/consumption carries this, as in i25). Kill-switch gates:

| PILOT GO gate | value | verdict |
|---|---|:--:|
| intercept dev 0 (post-train) | dev **0.16** | **FAIL** (bug, see below) |
| no NaN | clean | PASS |
| **value-zeroing ΔNDCG@10 ≥ 0.005, CI>0** | **+0.0002** CI[−0.0001,+0.0006] | **FAIL — INERT** |
| σ shrinks with length | corr −0.995 | PASS |
| σ↔error within-length partial-corr > 0 | +0.037 | PASS (weak) |
| γ(meh) ≈ identity | +0.939 | PASS |
| γ(hated) sign | +0.938 vs loved +0.942 | not inverting (mean; see IG2) |

**Verdict: STOP.** The centerpiece — value non-inertness — fails at exactly the ΔNDCG≈0.000 wall the
prior six folds hit.

## Diagnostics (why STOP is real, not undertraining) — `.cache/rung1/pilot_diagnostic.json`
Ran the full **G-value-NONINERT** protocol (both regimes × {short,long,full}) on the pilot checkpoint —
the mechanism trains value to carry signal *when consumption is dropped*, so the consumption-absent
regime is the fair test. It does **not** rescue value:

| context | Δ (consumption present) | Δ (consumption absent) |
|---|---|---|
| short | +0.0000 CI[−0,+0] | −0.0000 CI[−0.0001,+0] |
| long | +0.0002 CI[+0,+0.0005] | +0.0003 CI[−0,+0.0007] |
| full | −0.0002 CI[−0.0008,+0.0004] | −0.0011 CI[−0.0022,−0.0001] |

**IG2 polarity-flip (the behavioral value test) — completely inert** (`.cache/rung1/pilot_ig2.json`):
"like genre X" vs "dislike genre X" give **identical** region percentiles for *every* genre
(target 0.90→0.10; actual **0.524 → 0.524**, drop 0.0000):

| genre idx | love-pctile | hate-pctile | drop |
|---|---|---|---|
| 0 | 0.549 | 0.549 | 0.0000 |
| 1 | 0.579 | 0.579 | 0.0000 |
| 2 | 0.495 | 0.495 | −0.0001 |
| 3 | 0.516 | 0.516 | −0.0001 |
| 5 | 0.545 | 0.545 | 0.0000 |

## What DID work (the encoder is not broken — only the value channel is inert)
- **G-GoT PASS**: implicit consumption pull — entity +0.0127 CI[0.0052,0.0206], item +0.2308
  CI[0.2163,0.2467]. Consumption-as-taste carries through ρ.
- **σ posterior sharpens** monotonically with length (corr −0.995): 0.910→0.879 over budget 1→32.
- **Regime-(ii) research question**: consumption-absent full NDCG = **0.271** — above intercept (0.150)
  and above static (0.226). But this stands on the **implicit / member-bag geometry**, not the graded
  value field (value-zeroing there is still ≈0).
- **G-clean-ish**: full-profile fold NDCG 0.505 (native carries it) — degrades to i25 behavior, as the
  rollback ladder anticipated ("no worse than the proven fold; extras add nothing").

## Interpretation & the two failure classes (design §0 decision rule)
1. **Value inertness (fatal, centerpiece).** The FiLM value-entity binding + channel-dropout + ordinal
   loss + confidence-precision did **not** produce a strong enough value gradient. The likely cause:
   L_ordinal targets are HELD items outside the interview, so the value field of interview tokens only
   reaches held-item ranks *indirectly through the frozen decoder geometry*; meanwhile the held-LIKES
   reconstruction (L_consume) is fully satisfiable by native_z + implicit membership, so the value-gain
   γ head receives ~no gradient and stays at its identity init (γ(hated)≈γ(loved) behaviorally). This
   reads as an **our-own-bug inertness** (design says fix-in-place, not escalate to Rung II) — but the
   fix is non-trivial and is an overseer/design decision, so per the brief I HALTED rather than iterate.
2. **Intercept not preserved post-training (separate, fixable).** ρ's zero-init last layer gives an
   exact empty→native intercept only *at init*; after training ρ([0,0,0])≠0. Fix = gate δ by an
   evidence indicator (δ ← δ·1[ntok>0 ∨ native present]) so empty→prior stays exact. Independent of the
   inertness; noted for the rebuild.

## Recommended next step (for the overseer, not executed)
Give the value field a **direct** gradient rather than the indirect held-item one, e.g. an auxiliary
per-token value-reconstruction / region-ordinal term that forces γ to invert a *revealed* region's own
members hated→loved (an in-interview supervision signal), and/or raise the ordinal weight and drop KL
pressure. Re-pilot the same 2k kill-switch; only escalate to Rung II (joint decoder retrain) if value
stays inert after value gets a direct gradient — i.e. once "our own bug" is ruled out.

**No full train run. No gates on a full model. QUARANTINE intact (173/300 never touched). No data caps.**
