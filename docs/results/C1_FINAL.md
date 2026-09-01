# C1-FINAL — a policy that emits off-catalogue directions without collapsing, and does not extrapolate

**Status: COMPLETE (train/validation-side; the 10,000 canonical TEST users untouched). 2026-08-30.**
Design `docs/design/DS_C1_FINAL.md` (pre-registered; author settled the teacher: Paper B's cold
tree). Scoping `C1_FINAL_SCOPING.md`. Code `src/instrument/c1_final.py` (committed pre-run
`9f06967`, reading-1 null `a52c964`); artifacts `experiments/paperC/c1_final.json` + per-state npz
(`e6d7389`). Conclusions Fable's own. 30.0 min + 3.8 min dry pass; the existing 85,289-user bank
was reused and nothing was rebuilt.

**One-line verdict: both blocking gates PASS — the pilot's collapse is fixed and this is a policy,
not 42 constants: one object reproduces the tree's ordinary interview (245/245 distinct questions,
snap agreement 0.995) AND emits off-catalogue directions at cos 0.970 to the measured-gain blend at
every state it was trained on. The registered generalisation reading FAILS: at withheld states the
emission lands on the wrong side of zero (−0.119 / −0.011 against 0.970 / 0.975 at trained states),
and the one statistic that appeared to support extrapolation is matched or beaten by the UNTRAINED
warm start. The policy emits where it is taught to emit; the gain geometry does not extrapolate.**

## Gates (both cells, both blocking) — PASS

| gate | primary (`root.like.*` out, 15/57) | secondary (depth ≥5, 30/42) |
|---|---|---|
| **3 no-collapse** (≥0.75× distinct, ≥0.90 agreement) | distinct **245/245** (1.000) · agreement **0.9950** | 244/245 (0.996) · **0.9960** |
| **4 renderability** (median norm ratio ∈ [0.5, 2.0]) | **1.0629** all-72 · **1.0449** trained · 3.4798 held-out | **1.1056** all-72 · **1.0541** trained · 3.4213 held-out |

**[Aggregation clarified 2026-09-01]** Three medians circulate and the artifact labels all three;
the docs did not. The registered gate is the **all-72** median (1.0629 / 1.1056). The number the
demo relies on — because rendering happens only at trained states — is the **trained-state** median
(1.0449 / 1.0541). The held-out median (≈3.42–3.48) is unchanged from the warm start's 3.4798,
which is the finding: the scale term does not transfer.

**The pilot's fatal defect is gone.** It fell to 4–6 distinct questions at 0.0000 agreement; here
the warm start's full behaviour survives, and the replay CE (ending 0.0062 / 0.0040 per example) is
the difference. ⚠ **Gate 4 passes on a split that must be read**: trained states sit at 1.045/1.054,
**held-out states at 3.480 — unchanged from the warm start**. The scale term is learned per trained
state and does not transfer, so held-out emissions remain out of scale and would still render as
empty strings (`C4_PILOT.md`'s ruling 2 still binds them).

## The headline reading — generalisation is not demonstrated

| | primary | secondary |
|---|---|---|
| cos(ẑ, own target), **trained** states | **0.96958** (min 0.932, 57/57 ≥0.90) | **0.97455** (42/42) |
| cos(ẑ, own target), **held-out** states | **−0.11948** | **−0.01069** |
| (a) own − random other held-out target | +0.05926 [+0.04269, +0.07539] | +0.10114 [+0.08436, +0.11798] |
| **(a) the same statistic on the UNTRAINED warm start** | **+0.08432 [+0.04738, +0.12680]** | **+0.10441 [+0.08950, +0.11953]** |
| (b) own − warm start's cos at the same states | +0.25954 [+0.09535, +0.45041] | +0.15696 [+0.06792, +0.25538] |

**★ The methodological point, and it is the reason this doc says what it says.** Reading 1a was
registered without a null. The executing agent added one — the warm start's own own-minus-other
margin at the same states under the same seeded pairing — and it is *larger* (primary) or
statistically indistinguishable (secondary). Neighbouring states in one tree carry correlated
targets, so "own beats a random other" is positive **before any training**. Reported as registered,
1a would have read as evidence of extrapolation; it is an artefact of target correlation. The null
is banked as `reading1_warm_start_reference`. **This is the third time in this workstream that a
control decided a headline** (C3B's directionless C2, C4B's reversed twin, now this) — the standing
lesson holds: apply every registered rule to its own null at design time.

Arm (b) does fire: training moved the emission toward the withheld target. It did not move it far
enough to cross zero. **Plain statement: the policy fits the states it is shown and does not infer
the gain geometry of a withheld subtree or of deeper states.**

## What IS established

1. **One object, two behaviours.** The same policy runs the tree's ordinary interview essentially
   unchanged and emits measured-gain blends at its node states. The pilot could not do this; it is
   the Paper-B "one policy holds the forest" shape at the emission level.
2. **Fit is excellent and well-defined**: 0.970/0.975 median, every trained state ≥0.90,
   within-state per-user agreement 0.99991 — the emitted point at a state is one object, not an
   average.
3. **The emissions are genuinely off the grid**: median target-cos₁ 0.894, median top-1 blend
   weight **0.2254** (0 of 72 above 0.5), against a random floor of 0.121. Stage 0's premise gate
   did not fire, as the scoping predicted.
4. **Contract integrity**: the TRAIN snap 0.15830706760250668 (not the validation constant), the
   1,004-entry phi bank asserted, 72/72 sweeps matching `splitter`/`splitter_gain` at max error 0,
   and the descent reproducing **all 1,279,335 banked labels with 0 mismatches**.

## What may NOT be claimed

That the policy emits sensible novel directions at arbitrary interview states · anything from
held-out-state emissions (they are both off-target and out of scale) · that the blend's five
ingredients are the right five (unmet: 43 of 72 cohorts below 500; split-half stability is **not
computable on this tree** — no per-user gain sidecar exists — so the pilot's 0.656 mean-removed is
carried as a labelled prior) · any test-set claim.

## The echo worth one chapter sentence

This is the same shape as Paper B's P1/P1b: *the policy's behaviour is first-rate exactly where it
was supervised and no better than the untrained baseline outside it.* At the emission level the
finding recurs, now with the target being a measured direction rather than a labelled question.
That is a coherent thesis-level regularity, not an isolated disappointment.

## Reviewed deviations (Fable's ruling on each)

`--allow_dirty` — **accepted**: the four dirty files are another workstream's and could not be
staged; the exact list is banked in the git block, and C4C's import-scoped guard confirmed such
files do not enter these runs. `head.k`/`head.b` frozen — **correct and necessary** (the target is
built from them; letting replay move them would measure two banks). Held-out states receiving no
loss at all — **correct**: supervising them with the teacher's one-hot would pull emissions back
onto the grid at exactly the states reading 1 scores. 3 epochs / batch 512 / lr 3e-4, chosen once
and never swept — **accepted and declared** (`hyperparameters_not_tuned`); the per-epoch curve
0.850→0.971→0.977 shows the fit converged, so epochs are not the binding constraint on reading 1.

## If generalisation is ever wanted

The cheapest lever is more emission states, not more epochs: the scoping prices an inheritor sweep
at **4.4 h** to roughly double the 72. Whether that buys extrapolation is untested and, under the
2026-08-30 author steer (capstone, not exhaustive), optional.

---

## ★★★ CORRECTION 2026-09-01 — THE FIT NUMBER IS MOSTLY CONE MEMBERSHIP

`experiments/paperC/c5_diagnostics.json` (D1/D3/D4, commits `cf49372`/`7f9b420`) supplies the
trivial-baseline floor this document never reported, and it changes how every cosine here reads.

**The state-blind baseline.** Take the mean of the OTHER 71 blend targets — a single constant vector
that ignores the state entirely — and score it against each state's own target:

| | state-blind constant | this policy | paired difference |
|---|---|---|---|
| **trained** states (raw cos) | **0.9531** | 0.9696 / 0.9745 | policy **+0.015** |
| held-out, primary split (raw) | **0.9502** | **0.0609** (median −0.1195) | **constant +0.8893 [+0.6555, +1.0905]**, wins **14/15** |
| held-out, secondary split (raw) | **0.9623** | 0.0351 | **constant +0.9271 [+0.8254, +1.0194]**, wins **30/30** |

**Three things follow, and they are not softened.**

1. **"Fit 0.970" bought about +0.015 of raw cosine over a constant.** The registered metric had a
   ~0.95 floor because the targets are a tight cone (D1: median raw pairwise cosine 0.920), and no
   floor was computed before the result was read. The fit is real but nearly all of it is the cone.
2. **At held-out states the trained policy is far WORSE than doing nothing.** A constant scores 0.95;
   the policy scores 0.06. Training pushed it off the cone precisely where it had no supervision.
   The original wording — "does not extrapolate" — was too kind: it is *actively worse than the
   state-blind guess* there.
3. **The state-specific residual is not recoverable from these inputs.** Ridge probes over the latent,
   the warm-start emission, and structural features all fail to beat the state-blind baseline on
   either split (best +0.016, CI includes zero), and a two-feature `size+depth` probe matches the
   200-d latent. So this is *both* "model at fault" (a constant beats a 526k-parameter fine-tune) and
   "inputs insufficient" (nothing here recovers the residual).

**What SURVIVES unchanged.** The two blocking gates — no-collapse (245/245 distinct, agreement 0.995)
and renderability at trained states — are behavioural, not cosine-based, and stand. The claim *"one
object runs the ordinary interview and emits an off-catalogue point at measured states"* stands. What
does not stand is any reading of 0.970 as evidence that the policy learned the state→direction map.

**What must change downstream:** every cosine in the C-series needs its trivial floor printed beside
it (the C4 pilot's +0.286 point-specificity and C3B's ceiling-relative numbers already have proper
nulls; C1's did not). The chapter's C1 section must carry the +0.015-over-constant number, not the
0.970 alone.

## ★ The finding underneath, which is worth more than the number it cost

The measured-best directions at different interview states **share a dominant common component**:
raw cosine 0.95 to a state-blind constant, 0.63 mean-removed, while the top-1 candidates they are
built from differ at 0.247. In plain terms — *there is largely one direction worth asking about, and
conditioning on the interview state moves a modest residual off it.*

That is **Paper B's "the bank beats conditioning" (`policyB-rebuild-shrinks-conditioning-bank-wins`,
`concept-bank-does-not-rescue-adaptive-pv`), arrived at independently, in the geometry of the
question space rather than in an NDCG table.** Two chapters, two methods, one fact. It is a better
thing for the thesis to be able to say than a larger fit would have been, and Paper C should say it.
