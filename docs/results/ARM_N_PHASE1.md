# Arm N, phase 1: the harness, the sign result, and the Golbandi test

**2026-07-30.** Protocol spec: `PROTOCOL_DISLIKE_DISCARD.md` §11. Task #66.
Author's stated priority: *"i fear that golbandi outperforms ours on N arm... so maybe start with
re-evaling ours and running golbandi?"* — so phase 1 is exactly those two, plus the gate.

---

## 1. The harness, and what it verified

`metrics.evaluate` now takes `mask_X` (default `None` == `data_tr`, so all 40+ existing call sites are
byte-identical). Arm N always passes the protocol pool, so **model input and candidate pool vary
independently** — the coupling that produced the void "Golbandi 0.3894" cannot recur.

`arm_n.py` builds the protocol object once and asserts the invariants:

| quantity | value |
|---|---|
| catalogue / test users / head items | 18,359 / 10,000 / 190 |
| arm-A exclusion (`te_tr`) | 625,946 nnz = 62.6 items/user |
| **arm-N pool** (all rated, likes ∪ dislikes) | **1,411,761 nnz = 141.2 items/user** |
| binary fold-in items absent from the graded rebuild | **0** — `te_tr ⊂ g_te_tr`, never checked before |
| held-out targets inside the pool mask | **0** (leak assert) |

### The canary gate — PASSED (6.6 m)

Most-Popular and EASE are binary-native, so their arm-N *input is identical* to arm A and only the pool
can move them.

| model | arm A full / tail | arm N full / tail | Δ full |
|---|---|---|---|
| Most-Popular | 0.1345 / 0.0226 | 0.1626 / 0.0262 | +0.0280 |
| EASE | 0.3476 / 0.2441 | 0.4135 / 0.2941 | +0.0658 |

Both arm-A rows **snap exactly** to the recorded canonical numbers ⇒ the `mask_X` refactor is a
certified no-op. Order preserved.

**The uneven rise is a real protocol property and must be reported, not smoothed over.** EASE gains
2.35× what Most-Popular gains. The mechanism: a strong personalised model spends top-10 slots on items
the user has already rated (they are, after all, exactly the kind of thing that user watches), and the
arm-N pool hands those slots back. Most-Popular's top-10 is generic and was never competing for them.

**So arm N systematically favours personalisation over popularity.** This is defensible — it is what a
served system does — but it means arm N is *not* a neutral re-scaling of arm A, and any A→N reordering
must be read against this baseline effect before being called a capability difference.

## 2. The tower under the native contract

Snapshot `t2final_best.pt`. **Arm-A reproduction gate: 0.3482 / 0.2462 — exact snap to the chapter.**

| row | input | pool | full@10 | tail@10 | NDCG@100 |
|---|---|---|---|---|---|
| `A_full` | likes only, graded levels | arm A | 0.3482 | 0.2462 | 0.4486 |
| `A_input_N_pool` | likes only, graded levels | arm N | 0.4203 | 0.2985 | 0.5036 |
| `N_full` | **all bands, signed** | arm N | 0.4179 | 0.2955 | 0.5023 |

Two readings, both important:

1. **The pool alone is worth +0.0721 full / +0.0523 tail** to the tower — the same personalisation
   effect the canary exposed, and larger than for EASE.
2. **At full profile, folding the dislikes is worth −0.0024.** Sign adds nothing when 62 likes are
   already on the table. This is the exact analogue of the −0.0002 intensity result, and it points the
   same way: *the value of a signed answer is a scarce-evidence phenomenon.*

### The budget rows, and the confound in them

`arm_n_tower.py` also ran k ∈ {2,4,8,16} drawing the k answers from the full rated history (arm N) vs
from the likes (arm A), both under the arm-N pool:

| k | A (k likes) | N (k rated) | Δ |
|---|---|---|---|
| 2 | 0.2464 | 0.2289 | −0.0175 |
| 4 | 0.2849 | 0.2540 | −0.0309 |
| 8 | 0.3178 | 0.2770 | −0.0408 |
| 16 | 0.3586 | 0.3124 | −0.0462 |

**Do not read these as "sign hurts".** The two arms ask *different questions*: arm N spends ~56% of its
budget on items the user disliked, arm A spends all of it on likes. The rows say a like is a more
informative *answer* than a dislike — a statement about question **selection** (Paper B), not about
answer expressiveness. The growing gap with k is just the compounding of that per-question difference.

## 3. The clean sign test — SAME questions, different answer contract

`arm_n_sign_probe.py`. Draw k items once per user from their full rated history (fixed seed); that
question set is **shared**. Then vary only what the user is allowed to say back:

- **A** — the Liang contract: a sub-3.5 answer does not exist, so the question comes back unusable and
  is not folded (`usable` column shows how many of the k survive).
- **N** — the native contract: every answer is folded at its real level, so a dislike enters as a
  negative observation (the tower's γ is negative for levels 0–4).

Pool is the arm-N pool in both, so demoting the *observed* dislikes wins nothing: they are already out
of everyone's candidate list. **Any gap is off-support generalisation.**

| k asked | usable in A | A full / tail | N full / tail | **gap full** | **gap tail** |
|---|---|---|---|---|---|
| 1 | 0.5 | 0.1833 / 0.0586 | 0.2002 / 0.0803 | **+0.0169** | **+0.0217** |
| 2 | 1.1 | 0.2060 / 0.0849 | 0.2289 / 0.1062 | **+0.0229** | **+0.0212** |
| 4 | 2.1 | 0.2389 / 0.1193 | 0.2540 / 0.1318 | **+0.0151** | **+0.0125** |
| 8 | 4.2 | 0.2779 / 0.1572 | 0.2770 / 0.1603 | −0.0009 | +0.0031 |
| 16 | 8.5 | 0.3142 / 0.1953 | 0.3124 / 0.1951 | −0.0018 | −0.0003 |
| 32 | 15.8 | 0.3576 / 0.2326 | 0.3580 / 0.2329 | +0.0005 | +0.0002 |

**This is the R5 evidence, and it is now about SIGN, not intensity.** The chapter previously had to
concede that sign was unmeasurable on this ruler (the −0.0002 number bounds intensity within likes
only). It is measurable here, and it says:

> A "no" is worth **+0.023 full / +0.021 tail NDCG@10 at a two-question budget**, still **+0.015** at
> four, and **nothing at all** from eight questions onward.

The shape matches the intensity curve exactly (+0.0147 at k=2 → −0.0002 at full profile), which is the
strongest possible form of the argument: *two independent channels of the same capability both pay only
in the scarce-evidence regime.* That is not a limitation to confess — it is the argument for why an
**interview** instrument is the right place to exercise R5, and why a full-profile recommender would
never surface it.

Note the tail gap exceeds the full gap at k ≤ 2 (+0.0217 vs +0.0169 at k=1): dislikes disambiguate
hardest where popularity carries least.

## 4. The author's fear, resolved — and a naming correction

**Golbandi has no full-profile mode by construction.** The WSDM'11 contribution is an *elicitation
policy*: ask ≤ k questions, route by like / dislike / unknown, recommend the leaf's shrunk group mean.
At full profile the tree is meaningless. So the master-table row we have been printing under his name
is **our construction**, not his — `golbandi_node.py` says so in its own header ("NOT the tree /
elicitation ... the full-profile ceiling of the node model").

**Author ruling, 2026-07-30: relabel it `user-kNN` and stop printing it under Golbandi's name.** Same
class of error as printing belief-MF under Bıyık's name, already fixed once. Correct attribution in
prose: *"user-kNN, the full-profile limit of Golbandi's node-mean recommender"* — cite them for the node
model, own the reduction. Do **not** write "Golbandi used user-kNN"; their node model is a group mean.

**Golbandi appears in exactly one table: the interview table.** Never the full-profile one.

### The ratings-native user-kNN

The void 29-Jul graded run (0.2641, *below* its own binary 0.3065) fed raw ratings into a cosine built
for binary rows, so a 1.0-star rating pulled the user *toward* that neighbour. That was a bug.

`golbandi_native.py` is the proper neighbourhood formulation: row-centred ratings (a dislike is
**negative**), adjusted cosine, top-k with negative similarities clipped, shrunk by λ on thin support.
Val sweep k ∈ {100, 300} × λ ∈ {0, 8, 25}:

| k | λ | val full | val tail |
|---|---|---|---|
| 100 | 0 | 0.0034 | 0.0035 |
| 100 | 8 | 0.2503 | 0.1162 |
| **100** | **25** | **0.2663** | **0.1302** |
| 300 | 8 | 0.2377 | 0.1056 |
| 300 | 25 | 0.2589 | 0.1259 |

λ=25 won at the edge of the first grid, and a selected value on the boundary means an under-tuned
baseline — so the grid was extended and re-selected on val (a λ=50 *test* run made before this was
discarded: selecting on test is not selection):

| λ (k=100) | 25 | 50 | 100 | **200** |
|---|---|---|---|---|
| val full | 0.2663 | 0.2713 | 0.2737 | **0.2754** |
| increment | — | +0.0050 | +0.0024 | +0.0017 |

Increments halve per doubling, so the curve is asymptotic near ~0.278 and λ=200 is a properly converged
selection rather than a boundary artefact.

**ARM-N TEST (k=100, λ=200): 0.2808 / 0.1391.**

Bug found and fixed mid-run: at λ=0 an item no neighbour rated scores 0/0 = NaN, which sorts
unpredictably rather than ranking last. λ=0's 0.0034 is the unshrunk estimator being degenerate at this
catalogue size, exactly why shrinkage is in the published design.

### The finding that matters more than the fear

**The ratings-native user-kNN (0.2725) lands BELOW where its binary twin will land** (binary arm A is
0.3065; the pool lift for a model of that strength is ≈ +0.05, so ≈ 0.35). *Giving a ratings-native
model its ratings back made it worse at ranking.*

This is **Cremonesi, Koren & Turrin (RecSys 2010)**: rating-prediction and top-N ranking are different
objectives, and the implicit/binary formulation is legitimately strong for top-N. A centred
neighbourhood model ranks by "liked this more than their own average" — the right quantity for RMSE,
the wrong one for "what should we show next".

**Consequence for the argument, and it sharpens it.** Two claims must stay separate:

1. The discard costs Golbandi **its dislike branch** — an *elicitation* loss. Real, demonstrable in the
   interview table, and on half-star data his `< 4` boundary is *exactly* Liang's `> 3.5` cut, so the
   protocol deletes precisely the branch his tree routes on.
2. It does **not** follow that ratings-native input makes a **recommender** stronger. Here it does the
   opposite, for a published reason.

Claim 1 is the one we can demonstrate; claim 2 would have been easy to overclaim. Keeping them apart is
what makes "every baseline is reproduced faithfully on its own contract" a defensible sentence.

## 4b. ★★★ THE TIE CLAIM DOES NOT SURVIVE A PAIRED TEST

**Found 2026-07-30 while computing the arm-N comparison. This reaches back into the chapter.**

The chapter reports our tower as tying RecVAE on arm A: difference −0.0057, "CI 0.0069". That interval
is exactly `1.96 × √2 × 0.0025` — i.e. an **unpaired** SE-of-difference. Both models are scored on the
*same 10,000 users*, so the comparison is paired and the unpaired interval is far too wide.

Re-tested with a percentile bootstrap over the per-user differences (`arm_n_paired.py`, 10,000
resamples):

| arm | metric | ours − RecVAE | 95% CI (paired) | verdict |
|---|---|---|---|---|
| **A** | full@10 | −0.0057 | [−0.0072, −0.0042] | **SEPARATED** |
| **A** | tail@10 | −0.0035 | [−0.0049, −0.0021] | **SEPARATED** |
| **N** | full@10 | −0.0094 | [−0.0108, −0.0079] | **SEPARATED** |
| **N** | tail@10 | −0.0096 | [−0.0111, −0.0081] | **SEPARATED** |

The paired CI half-width is ±0.0015, **4.6× tighter** than the unpaired ±0.0069 the chapter used. The
sign of the difference is consistent across arms, both metrics, and every resample.

**We do not tie RecVAE. We are reliably just below it.**

### What may be claimed instead

The gap is small and quantifiable: **1.6% relative on arm A** (0.3482 vs 0.3540), 2.2% on arm N. And
the context is favourable, because the tower is *built on a frozen RecVAE decoder*: we are within 1.6%
of the model we sit on top of, while adding set-structured input, signed folding, and the interview
capability none of the rivals have.

Honest formulations, in decreasing strength:
- "within 0.006 NDCG@10 (1.6%) of the strongest baseline on its own protocol"
- "near-parity"; **never** unqualified "parity", "ties", or "matches"
- The frozen-tower design *chooses* to give up a little full-profile accuracy for R2–R5. That is now a
  measured price, not a hand-wave — and a 1.6% price for the whole capability set is a good trade to
  argue explicitly.

### Actions

1. Definition 1 is currently titled *"Parity on a shared protocol"* — the word no longer fits. Retitle.
2. Every "ties SOTA" / "matches RecVAE" / "parity" phrasing must be requantified. (A prior pass already
   replaced "ties SOTA" with "parity"; that was the right instinct with the wrong destination.)
3. Report the paired CI everywhere a difference is claimed. The unpaired interval must not reappear.
4. Re-check any other CI in the chapter computed as `1.96 × √2 × SE`.

**This is exactly what a reviewer would have found.** Finding it ourselves converts a fatal objection
into a measured design trade-off.

## 5. Standing interpretation, now that the ordering is visible

1. **Arm A is untouched.** Parity is certified there, against published numbers.
2. **The A→N ordering is close to unchanged**, because Δ tracks personalisation rather than capability.
   RecVAE (arm A 0.3540 vs our 0.3482, a reported tie at −0.0057, CI 0.0069) will most likely stay just
   above us in arm N. Plan no claim that needs us in first place.
3. The chapter's claim lives in the **scarce-evidence regime** (§3), not in any full-profile row.

## 6. Files

- `src/baselines/arm_n.py` — protocol object, cached graded matrices, invariants
- `src/baselines/arm_n_canary.py` — the gate (run first, always)
- `src/baselines/arm_n_tower.py` — tower, both contracts, budgets
- `src/baselines/arm_n_sign_probe.py` — **the R5 number**
- `src/baselines/golbandi_native.py` — faithful ratings-native Golbandi
- `src/baselines/run_arm_n.py` — phase 2: all binary-native baselines, re-eval only
- Outputs: `experiments/baselines/arm_n/*.json`, `*.log`
