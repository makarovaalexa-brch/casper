# #3 RESULT — Continuous divisiveness-field reward beats the continuity-only headline

**Date:** 2026-06-29. **Request #3:** "add back entropy/popularity/rating signal, extrapolate in continuous space, add to
the actor; not trivial, valuable signal, boost performance; keep the current best." **Verdict: CONFIRMED — divisiveness
helps; popularity/avg-rating as candidate features do not.**

## The mechanism (D1)
The continuous actor emits a query direction q∈R⁶⁴ per turn. We add an AUXILIARY training reward that pushes those
directions toward maximally-DIVISIVE (population-informative) regions:

    loss = base_objective(recon + soft-NDCG)  −  DIVW · mean_t  div(q_t)

where `div(q)` is the EXACT continuous entropy of a direction, computed from the train-user taste distribution:

    div(q) = H_b( mean_i  σ( u*_i · q / DTAU ) ),    over 2500 unit train-user tastes u*_i

- H_b = binary entropy. This is smooth + differentiable everywhere off-manifold and reproduces the discrete POOL_ENT
  divisiveness at the catalogued concept points — i.e. it is the entropy heuristic (the strongest Paper-B discrete
  baseline's criterion) extrapolated into the continuous action space and shaped directly into the policy.
- Config: DIVW=1.0, DTAU=2.0, warm-start from the u-recon variant, 10 epochs, frozen V1 encoder, GRADED geometric answers.
- Code: `scripts/paper2/continuous_actor.py` — field at L262 `field_div`, reward at L805 `loss=loss-_DIVW*(_DIVH[0]/T).mean()`.
- Checkpoint: `data/movielens/.cache/policy_phase3_d1divw_last.pt` (= ep10); parked `PAPER_C_CONT_WINNER/policy_cont_actor_DIVW_WINNER.pt` (sha a63fec1e).

## Result (COMPARE4 harness — faithful to canonical; seed-avg {1,2,3,7,11}, te[300:], q8)
| policy (graded answers) | FULL | TAIL |
|---|---|---|
| **D1 = actor + divisiveness reward (NEW HEADLINE)** | **0.3780 ± 0.0032** | **0.1782 ± 0.0065** |
| continuous actor recon+softNDCG (prev headline) | 0.3695 ± 0.0046 | 0.1651 ± 0.0039 |
| CASPER-R (binary, discrete SOTA — Paper B) | 0.3594 ± 0.0055 | 0.1467 ± 0.0049 |
| entropy (binary) | 0.3618 ± 0.0026 | 0.1393 ± 0.0041 |
| popular / conc_pop (binary) | 0.3465 ± 0.0027 | 0.1264 ± 0.0046 |
| casper graded | 0.3432 ± 0.0023 | 0.1382 ± 0.0034 |
| entropy graded | 0.3529 ± 0.0038 | 0.1328 ± 0.0065 |
| popular graded | 0.3246 ± 0.0023 | 0.1197 ± 0.0047 |

- **D1 vs prev continuity headline: +0.0085 FULL / +0.0131 TAIL** (tail ≈ 2σ, the Paper-B headline metric).
- **D1 vs discrete CASPER-R: +0.019 FULL / +0.032 TAIL.**
- Stacked contributions: **discrete 0.359/0.147 → +continuity 0.370/0.165 → +continuous divisiveness field 0.378/0.178.**

## Why this is trustworthy (anti-self-deception checks)
1. **Faithful harness.** COMPARE4 binary baselines reproduce the canonical Paper-B ruler: casper 0.3594≈0.360, entropy
   0.3618≈0.361, popular 0.3465≈0.347. The previous headline actor reproduces its canonical 0.3696/0.1650 here as
   0.3695/0.1651 → COMPARE4 ≡ canonical for the continuous actor. (Slow `continuous_policy2_st.py` gold-check launched
   separately: experiments/paper2/d1_canonical_GOLD_*.log.)
2. **Not a lucky last-epoch.** Monotone epoch curve: ep6 0.3739/0.1701 → ep8 0.3751/0.1758 → ep10 0.3780/0.1782. No val
   selection on the test set (SKIPVAL; the checkpoint is fixed, seeds vary only the eval split). Monotonicity even hints at
   headroom past ep10.
3. **Same answer model for everyone.** All policies use the identical GRADED geometric answer in COMPARE4; D1 beats them all.
   Divisiveness is computed over TRAIN users only (independent of the test u*), so it is genuine optimal-design signal, not
   test-leakage.
4. **Mechanism is on-thesis, not a hack.** It is the entropy/divisiveness heuristic made continuous and differentiable and
   folded into the policy gradient — precisely the "extrapolate the discrete signal into continuous space" ask.

## What did NOT work (honest negatives)
- **D2 — FieldActor (continuous CASPER-R):** propose a region → K candidates → score by [divisiveness, popularity,
  avg-rating, belief-align] with learned weights → soft-select. Graded 0.3510/0.1481 — BELOW the prev continuity headline.
  The multi-signal candidate-scoring architecture underperforms the plain actor + divisiveness reward. ⇒ the win is the
  divisiveness REWARD shaping a plain actor's directions, NOT a popularity/rating candidate-scoring head.
- Popularity field `field_pop(q)=Σ softmax(q·Q̂/PTAU)·popz` and avg-rating field `field_rat` were built and available
  (L265–266) but only entered D2 (which lost). Divisiveness alone, as a reward, is the load-bearing signal.

## Repro
```
# D1 (headline):
NOBC=1 EP=0 COMPARE4=1 ONLYACTOR=1 ACTORCK=data/movielens/.cache/policy_phase3_d1divw_last.pt \
  EVALSEEDS=1,2,3,7,11 CONTMODE=cont FEATS=ext,ans ANSF=1 python scripts/paper2/continuous_actor.py
# full 4-policy faithfulness table: drop ONLYACTOR=1.
# epoch curve: loop ACTORCK over policy_phase3_d1divw_ep{6,8,10}.pt
# retrain D1: DIVW=1.0 DTAU=2.0 (warm from u-recon variant), 10 epochs, CONTMODE=cont GRADED, frozen V1.
```
