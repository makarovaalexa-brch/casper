# STATE — where the project is now

> The single current-status doc. **Overwrite as things change.** Pointed to from `MEMORY.md`.
> End-state is in `VISION.md`. Last updated: **2026-07-25 (signed-concepts GO)**.

## The concept escalation arc (Jul 24-25) — where it stands
Author requirement: concepts must carry whole interviews (G5-E). Escalation ladder ran end-to-end on
the frozen ep4 i25 snapshot (val 0.3435 full; canonical 10k COLD_SEED cohort; intercept 0.1279/0.0192):

| rung | concepts-only m8 (full/tail) | verdict |
|---|---|---|
| ArmA additive (untrained, whitened β=1) | 0.0991/0.0455 top-SEL (craters) | additive-union saturates |
| FixA div-selection (\|cos\|<0.5, eval-only) | 0.1537/0.0631 monotone | crater = correlation artifact; FixB Bayes KILLED (below intercept) |
| **C-lite trained fold** (`cfold_best.pt`, 464k params) | **0.2091/0.1222** monotone, redundancy-robust (top-SEL m8 0.2219 > m4) | item-parity at m2 (0.1922 vs 0.1965); mixed m2k2 +0.0265; **G5 split**: member-AUC 0.617 FAIL / pop-projection control PASS (+0.0494) — learned taste-region movement, not member-shape |
| **C-full tokens-in-tower** (`cfull_best.pt`) | see ledger | **item cost ~zero** (full ≥ native-init; coldk8 above item-only tower by ep2) |

**Ledger (committed 2eb1b51, `experiments/battery/tradeoff_ledger.json`)**: the author decision
table — per-answer curves, redundancy rows, split-G5 columns, deployment per-question section.
**Deployment finding**: realizable fixed-bank concepts-only DECLINES under the clip-up value
convention (clite: 0.1304→0.1160 by q16, below intercept by q8).

## Signed-SEL triple gate (Jul 25, eval-only, `signed_sel_gate.json`, both runs committed)
Clip-up poison confirmed; the **bpool_r2 SEL+VAL signed port beats clip-up at every q**
(+0.0093 @q16 CI-clean, 0.1317→0.1254, never below intercept); popularity-counterfeit reproduces
only 12% (taste-real, no hard kill); value-permutation collapses the gain (values carry it);
**volume leak found** (ridge R² latent→log-volume 0.025→0.260) → per-user negative-channel
normalization mandatory. Verdict INCONCLUSIVE per the OOD-asymmetry ruling (module trained on
[0.25,1] likes only) → **retrain = the fair test. AUTHOR GO given.**

## What trains tonight (design: `docs/design/DESIGN_SIGNED_CONCEPTS.md`, pre-registered)
Four-band signed SEL+VAL/NPMI answers (shared module `signed_answers.py`, replaces the clip
everywhere), C_NEG negative-channel volume cap. Sequential, session-independent, queue-runner +
watchdog: (T1) signed C-lite, m~U{1..16} curriculum, ~2h → (T2) signed C-full (--concept_tokens,
signed levels incl. graded dislikes), ~3-5h → (T3) acceptance batch: deployment ≥ signed-eval curve
(≥0.1317@q2, never below intercept), counterfeit ~0%, leak R² ≤~0.05, redundancy holds, per-answer
m≤8 not degraded, split-G5 reported, ledger rows signed-clite/signed-cfull.

## Parked / pending
- **Certification retrain (t2final) PARKED** until the author picks the final winner
  (relaunch cmd recorded: `experiments/baselines/t2final_RELAUNCH_CMD.txt`; killed at ep4-b500,
  ckpts preserved).
- S4 filler (DAE/MultVAE ML-20M snap) running; demoted to idle priority during the retrains.
- C-lite gates (committed 70cbe78): flip −0.2012 PASS, wrong-user PASS, dup ×2 PASS / ×3 −0.005
  (slight over-count at ×3 — re-gate on the signed retrain's extended curriculum).
- Strategy ladder (Jul 24): HELF lit-rank-1 replicated CI-clean, 0 inversion flags; bib add flagged
  (golbandi2010 absent from INDEX).
- Belief layer (design (ii)) + Phase B battery built & smoked; belief shakedown fit exists
  (`belief_i25.pt`, sign proof G=2291); Phase B runs after the concept line settles.

## Older context (pre-Jul-24; see git history of this file for the full pre-reset picture)
- Canonical ruler = Liang ML-25M (CLAUDE.md); EDLAE 0.5230/0.3464 = the external bar; ep4 i25 tower
  G0-strength tie (0.3536 vs 0.3540 test).
- Phase A battery PASSED on the ep4 snapshot (G3a/G3b/G9 clean; G6 re-specified PASS; G5 → the arc
  above). Pre-Jul-22 interview-line numbers remain demoted (memory).

## Standing cautions
- Session-managed background tasks are reaped at ~1h — long runs launch via Start-Process
  (session-independent) with split stdout/stderr; watchdog via schtasks.
- The G5 split (member-AUC vs pop-projection) is an open author ruling: trained concept operators
  learn "what X-likers watch", not "members of X" — C3 human round-trip is where stated-attribute
  semantics ultimately get tested.
