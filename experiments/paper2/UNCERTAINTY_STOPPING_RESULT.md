# Experiment #8 — Uncertainty ensemble + adaptive stopping

**Date:** 2026-07-04 · **Harness:** `scripts/paper2/continuous_actor.py` env-gate `UNCSTOP=1`
(canonical ML-1M, frozen V1 encoder `enc_concept.pt`, `te[300:]` = 304 test users,
eval seeds {1,2,3,7,11} → 1520 user×seed instances/policy, graded geometric answers).
D1 actor = `policy_phase3_d1divw_last.pt`; entropy baseline = top-8 most-divisive answerable concepts.
Data dump: `uncertainty_stopping_data.json`. **No commits; no retraining.**

Reproduce:
```
UNCSTOP=1 EVALSEEDS=1,2,3,7,11 UNCPOLS=entropy,actor python continuous_actor.py
```

## What was built (PART 1)
The belief `u` is a deterministic fold and cannot express *how well it knows a user*. I added a cheap
uncertainty representation: **K=8 fold variants per user per turn via answer-subset bootstrap**
(resample the answered `(entity, answer)` tokens with replacement; leave-one-out if <3 tokens; q1 = max
sentinel). Per-turn uncertainty = **mean pairwise (1−cos)** among the K folds. Secondary variant =
mean pairwise (1−Jaccard) of the K top-50 recommendation lists. No retraining — pure post-hoc.

**Ensemble is NOT degenerate.** Uncertainty spans deciles 0.003→0.071 (entropy) / 0.014→0.113 (actor)
and decreases monotonically with turns (entropy 0.296→0.024; actor 0.556→0.102→0.049). So the single
prescribed fallback (dropout-style token masking) was **not needed**.

Sanity: D1 actor reproduces its locked anchor exactly — fixed q8 FULL **0.3780**, TAIL 0.178.

## PART 2 — Calibration ("knows when it doesn't know you") — **DOES NOT SURVIVE (inverted)**

Spearman(uncertainty@q8, NDCG@10):

| policy | ρ full | p | ρ tail | ρ full (Jaccard var.) |
|---|---|---|---|---|
| entropy | **+0.038** | 0.14 (ns) | +0.046 | +0.006 |
| D1 actor | **+0.203** | 1.5e-15 | +0.045 | −0.048 |

The relationship is **significant for the actor but in the WRONG direction**: *higher* uncertainty →
*higher* NDCG. Reliability deciles make it unambiguous (actor, low→high uncertainty decile):

```
unc 0.014 → full 0.282     unc 0.054 → full 0.402
unc 0.032 → full 0.357     unc 0.079 → full 0.435
unc 0.046 → full 0.396     unc 0.113 → full 0.465   (lowest-unc decile is the WORST recs)
```

**Diagnosis (honest).** Mean-pairwise-(1−cos) of answer-subset folds measures answer *variance*, i.e.
taste *extremity*, not estimate confidence. A bland near-origin user gives mid-range graded answers →
all bootstrap folds land near zero and near each other → *low* uncertainty, but the belief has no
magnitude → popularity-like recs → *low* NDCG. A distinctive user gives spread answers → folds diverge
→ *high* uncertainty, but a strong belief → *high* NDCG. So this naive belief-ensemble spread conflates
"informative user" with "uncertain estimate" and **cannot be used as an "I don't know you" gate.** A
usable confidence signal would need to normalize by belief magnitude / signal strength.

## PART 3 — Adaptive stopping ("buys per-user budgets") — **barely survives on the actor; fails on entropy**

Stop when uncertainty < τ (τ swept, 40 points), else continue to q8. Frontier = mean questions vs mean
NDCG, on the SAME users. Controls: **(a) random-stopping matched to the same mean length** (shuffle the
adaptive stop-lengths across users, 25 shuffles); **(b) oracle-stopping L3** = each user's per-user
argmax-NDCG turn (peeks — upper bound).

Fixed-length NDCG@10 FULL (the curve adaptive must beat):
```
entropy   q1 .342  q2 .357  q3 .353  q4 .351  q6 .353  q7 .353  q8 .353   (FLAT — plateaus at q2)
actor     q1 .292  q2 .331  q3 .359  q4 .370  q5 .379  q6 .383  q7 .384  q8 .378   (rises, dips q7→q8)
```

Matched-accuracy operating points:

| policy | match target | adaptive mean q | adaptive FULL | random-same-len FULL | Δ vs random |
|---|---|---|---|---|---|
| entropy | fixed-q8 (0.3529) | 2.00 | 0.3565 | 0.3565 | **+0.0000** |
| entropy | fixed-q4 (0.3509) | 2.00 | 0.3565 | 0.3565 | **+0.0000** |
| D1 actor | fixed-q8 (0.3780) | **6.80** | 0.3790 | 0.3774 | **+0.0016** |
| D1 actor | fixed-q4 (0.3699) | **4.15** | 0.3710 | 0.3694 | **+0.0016** |

Oracle L3 (peeks — headroom only): entropy 0.4008 @ 2.67q; actor **0.4298 @ 3.71q**.

**Entropy (the deployable, no-policy setting):** NDCG is flat after q2, so "adaptive stopping matches q8
at 2.0 questions" is real *savings* — but adaptive **exactly equals random-same-length** (0.3565 = 0.3565).
The saving is a **pure length artifact** (2 questions is enough for this heuristic); the uncertainty
signal contributes **nothing** over random. Fails the control.

**D1 actor:** adaptive stopping saves **~1.2 questions** at matched fixed-q8 accuracy (6.80 vs 8) and
**beats random-same-length by +0.0016 FULL / +0.0023 TAIL**, and similarly at the q4 operating point
(+0.0016). So the uncertainty signal does a *little* real per-user timing work — it stops already-plateaued
users before the q7→q8 over-elicitation dip more precisely than random. But the margin is **tiny
(~0.0016, no clear significance)** and realizable adaptive (0.379) captures only a sliver of the oracle
headroom (0.430).

**Decisive control — best fixed budget dominates adaptive.** The matched-q8 framing flatters adaptive.
The actor's own fixed curve *peaks early* — q6 = **0.3832 @ 6.0q**, q7 = 0.3837 @ 7.0q — both above
adaptive's best (0.3790 @ 6.8q). So a trivial **static "stop everyone at q6"** beats per-user adaptive
stopping on **both** axes (fewer questions *and* higher NDCG). Adaptive stopping's only real merit is
"cheaper than the naive q8"; it is **dominated by a single well-chosen static cutoff.** (This is also why
no adaptive τ could match the fixed-q6 target — it never reaches 0.3832.)

## Verdict — does "adaptivity buys per-user BUDGETS" survive its controls?

**NO — it does not survive the strong control.**

- **Calibration payoff: NO.** Belief-ensemble uncertainty is not a confidence signal — it's a taste-extremity
  proxy and correlates with NDCG in the *wrong* direction. The "knows when it doesn't know you" result
  does not hold with this representation.
- **Stopping payoff: NO against the right static baseline.** Adaptive stopping beats the matched-length
  *random* control by a whisker (+~0.0016 NDCG) and beats the *naive q8* by ~1.2 questions — but it is
  **dominated by the best fixed budget (static stop-at-q6: fewer questions AND higher NDCG).** A single
  global cutoff, chosen once offline, does everything per-user adaptive stopping does and more. For the
  **deployable entropy baseline** adaptive stopping is **indistinguishable from random** (pure length
  artifact). So the per-user-budget effect that "no static design can imitate" is, on this harness, imitated
  and beaten by a trivial static design.
- **Architecturally:** the missing piece (an uncertainty representation over the belief) is cheap and
  buildable with zero retraining, and belief uncertainty does fall with turns with genuine cross-user
  heterogeneity in *rate* (decrease-rate std 0.20–0.25, IQR e.g. actor [0.37, 0.66]) — the raw material
  stopping would exploit. But the naive bootstrap-spread metric turns that raw material into the wrong
  signal for calibration and only a whisper of signal for stopping. To make this channel pay off, the
  uncertainty score must be **magnitude-normalized / posterior-relative**, not raw fold spread.

**Bottom line for the thesis:** on this harness, adaptivity does **not** buy per-user budgets. The
uncertainty representation is cheap and buildable, but (i) as a confidence gate it is mis-signed, and
(ii) as a stopping trigger it is beaten by a single static cutoff (stop-at-q6) that is both shorter and
more accurate. The one defensible, narrow statement: for the *learned* actor, per-user stopping is
slightly better than random-at-the-same-length — a real but negligible effect swamped by the
over-elicitation dip that a fixed early cutoff already removes. Recommend **not** advancing
"adaptivity buys per-user budgets" as a headline claim; if pursued, the uncertainty score must be
re-defined to be magnitude-normalized, and the comparator must be the best fixed budget, not q8.
