# C7 — local transfer fails, and D2 was measuring interpolation

**Status: COMPLETE (train/validation-side; TEST untouched). 2026-09-01.** Brief follows
`docs/design/PAPER_C_IMPROVEMENT_ANALYSIS_2026-09-01.md` §6 (which this result corrects). Code
`src/instrument/c7_knn_profile.py` (`f8f14b2` pre-run), artifacts `experiments/paperC/c7_knn.json`
(`ad92974`). Conclusions Fable's own. 12.1 min.

**One-line verdict: NO. Local kNN profile transfer does not beat the state-blind floor at the
emission level on ANY split, and on the registered primary split it is significantly WORSE than
ignoring the state entirely. The pre-registered arm scores profile Spearman +0.1218 against a
state-blind floor of +0.3103 — paired −0.1884 [−0.2332, −0.1421]. All 12 arms lose.**

## Why D2 looked promising and was not — the correction

D2 reported that the **nearest of all 71 other states** predicts a profile at Spearman +0.617 vs
+0.150 for a random one, and I promoted local transfer to the top of the improvement menu on that
basis. C7 shows what that number was:

- median tree distance to the nearest of all others, **within train: 1** — i.e. the "neighbour" is
  usually the state's own **parent or sibling**;
- median distance to the nearest **training** state for the primary held-out set: **3.80** (max 8);
- accuracy tracks that distance hard: **r = −0.898 [−0.958, −0.819]**.

**D2 measured interpolation among measured states and I read it as generalisation.** Under a split
that withholds a contiguous subtree — which is the deployment-relevant case and the one C1-final
registered — the informative neighbour is gone by construction. The LOO-within-train figure (+0.69)
reproduces D2 and **overstates the held-out figure by 5.7×**.

## The table (every number with its floor, per the standing rule)

| arm | primary split | vs floor +0.3103 | secondary split | vs floor +0.2598 |
|---|---|---|---|---|
| pre-registered `both/k10` | +0.1218 | **−0.1884 [−0.2332, −0.1421]** | +0.4985 | **+0.2387 [+0.1583, +0.3177]** |
| best primary arm (`tree/k1`) | +0.1571 | −0.1531 [−0.200, −0.104] | +0.4878 | +0.228 |
| popularity prior | +0.2058 | −0.105 | +0.0691 | −0.191 |
| chance (permuted) | +0.0093 | −0.301 | −0.0086 | −0.268 |
| oracle (true profile) | +1.000 | +0.690 | +1.000 | +0.740 |

The secondary (depth ≥5) split **does** show a real profile-level win — but it **does not survive
translation into an emission** (−0.037, CI includes zero), and the state-blind *constant target*
still beats every kNN arm there (mean-removed +0.6200 vs +0.5301; paired +0.1223 [+0.076, +0.167]).
**On no split, at the emission level, does any local predictor beat the dumbest admissible one.**

## ★ THE CORRECTION THAT MATTERS MOST — the residual is NOT small

I wrote earlier today that the targets are "one dominant direction plus a **small** state-specific
residual". The banked energy decomposition says otherwise:

| of a target's MEAN-REMOVED energy | fraction |
|---|---|
| common component (LOO mean of the other 71) | **0.4085 [0.3730, 0.4442]** |
| **state-specific residual** | **0.5915 [0.5558, 0.6270]** |

**Mean-removed, the state-specific part is the MAJORITY of the energy.** What makes raw cosines read
0.95 is the *bank's* common mode, not a shared target direction. So the honest statement is not "the
residual is small" — it is:

> **The state-specific component of the measured-best direction is substantial (59% of mean-removed
> energy) and is not predictable from the interview state by any method we tried: global linear
> probes, a profile parameterisation, and local neighbour transfer all fail to beat a state-blind
> constant at the emission level.**

That is a stronger and more interesting claim than the one it replaces, and it also weakens the tidy
analogy I drew to Paper B's "the bank beats conditioning": there, conditioning *did not pay*; here,
the conditioning information demonstrably **exists** and we **cannot extract it**. Related, not
identical — and the chapter must say the second thing, not the first.

## What is now closed

Three independent routes to state-conditioned emission have been tested against a proper floor and
all three fail on the registered split: **global linear** (D4), **profile parameterisation** (D5),
**local transfer** (C7). Combined with D3 (a constant beats the trained policy by +0.89 at held-out
states), the question "can the emitter be made to generalise across a withheld subtree?" is answered
**no, by every method available to us**, and further architecture work is not indicated under the
capstone steer. The emission claim stands where it always stood: **at states where the question
landscape was measured.**

## Methodological notes worth keeping

- The agent used C5's *banked* profiles and targets as **gates rather than inputs**, rebuilding them
  with C5's own loaders so every floor reproduces C5's numbers bit-for-bit (D3's LOO to 2.2e-16)
  instead of at a 1e-5 tolerance. That is the right instinct and should be the default.
- Latent distance shows no degradation correlation on the primary split (−0.05) **not** because
  distance is irrelevant but because all 15 held-out states are uniformly far (0.400 vs 0.068
  within-train) — no variance left to correlate. Reported rather than read as a null.
- Box: first launch found 0.18 GB free against a 7 GB eigen job; the agent **waited** rather than
  co-running. Correct call.
