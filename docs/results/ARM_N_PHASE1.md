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

## 4. Golbandi — the author's fear

**Status: sweep running.** First configuration is already informative.

The void 29-Jul graded run (0.2641, *below* its own binary 0.3065) fed raw ratings into a cosine built
for binary rows, so a 1.0-star rating pulled the user *toward* that neighbour and the node mean summed
rating magnitudes upward. That was a bug, not evidence about ratings.

`golbandi_native.py` is the published model: row-centred ratings (a dislike is **negative**), adjusted
cosine, top-k with negative similarities clipped, shrunk by λ on thin support — genuinely sign-aware in
a way no binary neighbour mean can be. Val sweep over k ∈ {100, 300} × λ ∈ {0, 8, 25}.

- Bug found and fixed mid-run: at λ=0 an item no neighbour rated scores 0/0 = NaN, which sorts
  unpredictably rather than ranking last.
- `k=100, λ=0`: val full **0.0034**. The unshrunk estimator is degenerate at this catalogue size,
  exactly as expected — the top-10 fills with items one enthusiastic neighbour rated. This is why the
  shrinkage term is in the published design and why λ is swept rather than assumed.

## 5. Standing interpretation of the fear (written before the Golbandi number lands)

If the ratings-native Golbandi *does* beat the tower at full profile in arm N, the paper is not in
trouble, provided we hold the line on what it claims:

1. **Arm A is untouched.** Parity is certified there, against published numbers.
2. A ratings-native model winning on the protocol *built to stop punishing ratings-native models* is
   the finding the author asked for, not a refutation. It is the strongest possible evidence that the
   canonical protocol distorts the league table.
3. The chapter's claim lives in the **scarce-evidence regime**. §3 shows sign pays at k ≤ 4 and vanishes
   by k = 8; the full-profile row is the least load-bearing number in the whole table.

The result would only be dangerous if a native Golbandi also won the **short-interview** curve — and
that requires the tree, which does not exist yet on this ruler (`golbandi_node` is explicitly the
full-profile kNN ceiling with the elicitation removed). That build is phase 3.

## 6. Files

- `src/baselines/arm_n.py` — protocol object, cached graded matrices, invariants
- `src/baselines/arm_n_canary.py` — the gate (run first, always)
- `src/baselines/arm_n_tower.py` — tower, both contracts, budgets
- `src/baselines/arm_n_sign_probe.py` — **the R5 number**
- `src/baselines/golbandi_native.py` — faithful ratings-native Golbandi
- `src/baselines/run_arm_n.py` — phase 2: all binary-native baselines, re-eval only
- Outputs: `experiments/baselines/arm_n/*.json`, `*.log`
