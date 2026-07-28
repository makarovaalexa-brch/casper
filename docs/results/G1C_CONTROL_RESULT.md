# RESULT — the G1c control: the calibration gate was measuring the wrong thing

> Ran 2026-07-28, ~3 min (`src/instrument/g1c_control.py` → `experiments/battery/g1c_control.json`).
> All 10,000 eligible val rows, no subsample. Mirrors the G1c computation in `run_battery_phaseB.py`
> verbatim — same rows, same k=8 item prefix, same seed, same held-out NLL target — and swaps only the
> predictor. Fitted alphas loaded (`belief_i25.pt`).

## The question

G1c scored Spearman(belief-σ, held-item NLL) = **0.2393** against a bar of 0.25, and we were about to
spend time improving it (longer fit, calibration-targeted objective, delta-method push-through the
decoder). The control asks the prior question: **how well does a trivial scalar do on the same target?**

## The numbers

| predictor | Spearman ρ vs held-out NLL |
|---|---|
| **belief σ** (the gate) | **0.2393** |
| mean log-popularity of the user's targets | **−0.6964** |
| mean decoder bias of the user's targets | −0.5673 |
| profile size | 0.5195 |
| number of held-out targets | 0.4992 |
| answers folded | undefined (constant: k=8 for every user) |

## The verdict — DROP G1c, do not improve it

**Simply reading off how popular a user's held-out items are predicts our error nearly three times
better than the belief covariance does** (|ρ| 0.70 vs 0.24). Profile size alone more than doubles it.

This means G1c, as specified, is **not measuring the covariance's contribution at all**. It is largely
measuring *how hard this user's targets happen to be* — and popularity answers that question far better
than any uncertainty estimate. A gate that a popularity lookup beats by 3× is not a test of the belief
layer.

**Consequence: the planned calibration work is cancelled.** Refitting 8 scalars for more epochs, or
against a calibration objective, would have been optimising a badly-posed measurement. The 2-epoch
hardcoded default turned out not to be the problem.

## What this does NOT touch

R4 rests on the covariance having correct *structure*, and that evidence is unaffected:
- **G1a** — total posterior variance falls monotonically over answers, 1881.9 → 1805.1.
- **G1b** — shrinkage is **11.5×** larger along the queried direction than elsewhere, against a 2× bar.

Those measure what R4 actually claims. The magnitude-calibration question was an addition of ours and it
was posed badly.

## If anyone ever wants a real calibration test

The honest version controls for target difficulty: does σ predict held-out error *after* popularity and
profile size are partialled out (partial correlation, or σ against the residual)? That is a different and
much harder test, and nothing in this chapter depends on the answer, so we do not run it.

## Bookkeeping

`n_answers_folded` is constant by construction — the harness folds a k=8 prefix for every user — so its ρ
is undefined. The script's original verdict line let that NaN win the `max()`; fixed, and the JSON
verdict recomputed.
