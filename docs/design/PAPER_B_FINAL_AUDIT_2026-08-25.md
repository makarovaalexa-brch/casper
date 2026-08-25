# Paper B — full audit and finishing plan

**2026-08-25.** Read-only audit of every artifact, the chapter, and the test-look ledger, plus the
decisions that follow. Supersedes the action list in `PAPER_B_ACTION_ITEMS.md`.

---

## ★ HEADLINE: the paper is further along than the last two weeks suggested

**Acts 1–4 are already measured on the 10,000 disjoint TEST users and frozen.** The opener line
(Act 5) is the only part still on validation. The last fortnight's work — opener trees, distillation,
the home-advantage measurement — is all *new material for Act 5*, none of it disturbs a settled act.

| act | content | cohort | status |
|---|---|---|---|
| 1 | adaptive selectors vs a static order, 30 cells | **10,000 TEST** (`tab:act1`, line 493) | ✅ frozen |
| 2 | surrogate-vs-metric diagnosis | validation (a measurement, not a contrast) | ✅ fine |
| 3 | the tree; split-criterion ablation; genre arm | **TEST**, `+0.01035 / +0.00705` @K=16 (`CONDITIONED_ARMS_ON_TEST.md`) | ✅ frozen |
| 4 | answerability orders the channels | validation, descriptive | ✅ fine |
| 5 | volunteered information | **validation only** (`tab:volunteered`) | ⛔ needs the look |

**The test look has already been spent once** (2026-08-12, Act 1 + Act 3). The protocol was
*freeze every sequence on validation, then construct test* — chapter line 215: "the test cohort was
not constructed until all sequences were frozen." **One look remains, and it belongs to Act 5.**

---

## 1. AUDIT FINDINGS

### 1.1 ⛔ A claim I made yesterday is wrong and must be retracted

I told the author that making both teachers train-fitted is "what turned routing into VALUE". **It is
not.** Put the two distillation runs side by side:

```
m6a2k  route root 0.906 whole 0.860 | policy 0.28398  tree 0.28366  cold 0.28685 | vs tree +0.00032
m6fair route root 0.905 whole 0.898 | policy 0.28310  tree 0.28366  cold 0.27311 | vs tree -0.00057
```

**The tree scores 0.28366 in both — identical. The policy tracks it in both (±0.001).** The only
thing that changed is which cold tree the comparison used (0.28685 → 0.27311). Distillation was
never broken and is not newly fixed: it has always reproduced its teacher. The honest statement is
*distillation is solved infrastructure*, which is what `W9` already says. `route_whole` did improve
(0.860 → 0.898) and that may be the consistent pair, but **no value conclusion rests on it.**

### 1.2 Two trees are in play and the chapter does not distinguish them

* `ndcgtree16` — 22 nodes, 15 leaves, depth 5, **K=16**. This is **Act 3's tree** and the one on test.
* `coldtree_v2` — 164 nodes, 109 leaves, depth 11, **K=15**, built 2026-08-18. Never tested. This is
  the **incumbent used throughout the Act 5 opener work**.

A reviewer will ask why there are two and why the budget differs (16 vs 15). Needs one explicit
sentence, not silence. The favourable framing is true: Act 5's incumbent is the *stronger* of the
two (164 nodes vs 22), so beating it is the more conservative comparison.

### 1.3 Act 5's payoff table is at the wrong states for Act 5's own methodology

`tab:volunteered` measures each group's **anchor-state** teacher (`m3v2_*`, one attribute folded).
But Act 5 now contains `sec:opener_synthesis`, which argues that real requests are Reddit-shaped
openers — films 63%, genre 40%, tone 21%. And the banked result is that **anchor teachers do not
survive the move to opener states**: teacher−cold goes `+0.00897` at the anchor to
`−0.0020 / +0.0008 / −0.0014 / −0.0024` at one film / two films / a concept / a bundle.

**So the chapter currently motivates realistic openers and then pays off at anchor states.** This is
the single largest internal inconsistency and it is what the opener-fitted trees exist to fix.

### 1.4 The home-advantage number is a result, not just a nuisance

`cold_val − cold_train` on 10,000 validation users at K=15 = **+0.02881** [+0.02597, +0.03183].
My earlier indirect estimate (+0.00782, from differencing `cold − static` across populations) was
**3.7× too small**; that estimation method should never be reused. This belongs in Limitations as a
methodological finding: *interview-tree comparisons must control the fitting population*, and it is
the reason the design fits and reports on disjoint, symmetric cohorts.

### 1.5 Act 1 carries an unstated asymmetry

The static order was greedily **selected on validation**; the adaptive selectors (EDDI, entropy,
Thompson, LinUCB…) are computed at inference and are **fitted on nothing**. On validation that would
favour static. `tab:act1` is on **test**, where static is out of sample, so the claim survives — but
the chapter should say so explicitly, because a reader who has just read §1.4 will ask.

### 1.6 Numbers that must not appear anywhere

* `+0.01762` — the retracted arthouse "held-out" figure (measured on a memorised ruler, tree at home).
* Every held-out figure from the M6m train-user programme (pooled `+0.00278`) — same defect.
* `+0.00782` as the home advantage — superseded by the measured `+0.02881`.
* `+0.229 / +0.0115 / +0.0153` from K.6 — already flagged, audited-wrong, never quote.

### 1.7 Minor, to verify before submission

`greedy_static_policyB_combined.json` carries a stale `ruler_note` saying the order "was selected
under policy A", while its own `supersedes` field and `CONDITIONED_ARMS_ON_TEST.md` both state all
three static orders are policy-B-selected. Confirm which is true so the chapter does not repeat a
retired caveat.

---

## 2. THE DESIGN DECISION: fit on VALIDATION, not train

The author asked: *"grow all trees on train and distill on train too, right?"* **My recommendation is
no — and this is the one place I would push back hardest.**

**Why the train excursion happened.** Validation group cohorts are small (587–1,822), so I moved the
group trees to train for more fitting data. That, and only that, created the asymmetry — cold and
static stayed on validation while the group trees moved. The confound was mine, not the design's.

**Why validation is right:**

1. **It is already symmetric.** Every other arm — cold, static, anchors, ablation — is
   validation-fitted and test-reported. Nothing needs to move.
2. **The validation ruler is the honest one.** The tower was *trained* on train users, so a tree
   fitted there optimises measured NDCG under conditions that do not hold at test. Fitting on
   validation optimises under test-like conditions.
3. **The evidence points that way.** `cold_train` reaches only **+0.0074 over static** on validation
   where `cold_val` reaches **+0.0362** — about a fifth of the conditioning value. Confounded with
   home advantage, but the direction is what a degraded fitting objective predicts.
4. **On test the remaining bias runs in our favour.** Group trees fitted on 587–1,822 users against
   an incumbent fitted on 10,000 are *under*-powered, so a win is conservative. Contrast the current
   chapter, which must call `tab:volunteered` an *upper bound* because both arms are scored on the
   users they were fitted on. The test look flips that caveat from apologetic to conservative.
5. **Cost.** ~20 h on validation versus 40–120 h on train.
6. **It is the pre-registered protocol.** Changing the fitting population after seeing results is
   exactly what a reviewer looks for.

**The honest uncertainty.** We have *not* proven that `cold_val` transfers better than `cold_train` to
a third population; home advantage and ruler quality are confounded in every number we hold. If the
author wants that settled before committing, the cheap decisive experiment is **§3.0** below.

**What the train work becomes:** the m6r home-advantage measurement (§1.4) and the
`coldtree_v2_train10000` artifact stay as *methodological evidence justifying the design*. They are
not paper arms. Nothing is wasted.

---

## 3. WHAT MUST BE RERUN

### 3.0 OPTIONAL de-risker (~8 h) — split-half on validation

Fit a cold tree on 5,000 validation users, score the other 5,000. Same ruler on both sides, so it
isolates **pure selection overfitting** from ruler quality — the confound §2 cannot resolve. It also
predicts how much of Act 5's validation effect will survive the test look. Worth it if the author
wants the look de-risked; skippable if we accept the conservative framing.

### 3.1 REQUIRED — six opener-fitted group trees on VALIDATION (~20 h)

| group | n | note |
|---|---|---|
| arthouse | 587 | **artifact lost** (overwritten 2026-08-20) |
| dialogue | 730 | existing `m6k_dialogue.json` is a 4-node depth-1 stub |
| marvel | 734 | 3 nodes |
| gore | 747 | 3 nodes |
| teenagers | 786 | 4 nodes |
| inspiring | 1,822 | 3 nodes |

All five survivors predate the author's `min_leaf`-not-depth ruling, so **all six need refitting**
under the current rule: `--split validation --min_leaf 200 --depth 6` (a cap high enough not to bind
at these cohort sizes). Gain tables now bank automatically, so C1's targets come free.

### 3.2 REQUIRED — distillation over the validation forest (~3 h)

`m6c --cold_tree coldtree_v2.json` + the six new group trees → `m4b` → `m6d`. Trajectory states come
from train users, which is correct and unchanged: that is imitation of a frozen teacher, not fitting.
Per §1.1 this is infrastructure and is expected to reproduce its teachers; it is not a result.

### 3.3 REQUIRED — validation-side screen, then freeze (~2 h)

Which groups clear the pre-registered screen; pooled effect across groups; then **freeze every
sequence** before the test cohort is built, exactly as Act 1/3 did.

### 3.4 NOT rerun

Cold tree, static orders, anchor teachers, split-criterion ablation, Acts 1–4. All frozen, all
test-confirmed where it matters.

---

## 4. THE TEST LOOK — pre-register before running

**Arms** (all validation-fitted, all frozen): opener-fitted group tree · `coldtree_v2` · static ·
anchor teacher. **States:** per-user Reddit-shaped openers, deterministic in the user id.
**Budgets:** K ∈ {1, 4, 15}, **primary K=15**. **Metric:** full AND tail NDCG@10.

**Primary contrast:** group tree − cold tree, **pooled across the six groups** (n ≈ 5,400), paired
percentile bootstrap, 10,000 resamples, seed 98765.

**Secondary, pre-declared:** group − static; per-group breakdown; the policy vs its teachers
(fidelity only — per §1.1 the policy cannot exceed its teacher and must not be sold as if it could).

**Retreat lines, fixed now:**

* Pooled group−cold CI excludes zero → Act 5 is a **result**; the opener line is the payoff act.
* Pooled CI includes zero but group−static clears → Act 5 claims **conditioning beats a static
  order at realistic opener states**, which is the conservative claim and still novel.
* Neither clears → Act 5 goes **descriptive** (reach, what people volunteer, the era axis, the
  answerability asymmetry), the paper ships on the Act 3 ablation, and the opener line is written up
  as a characterised negative. This is a shipping paper in all three branches.

**Power note:** a group holds 3–7% of 10,000 users, so a per-group interval is ~4× wider than a
cohort-scale one. Pooling is what buys the power; per-group cells are descriptive.

---

## 5. WRITING TO-DO

| # | item | depends on |
|---|---|---|
| **W1** | Resolve §1.3: replace/supplement `tab:volunteered` with opener-state results | 3.1 |
| **W2** | One sentence distinguishing `ndcgtree16` (Act 3, K=16) from `coldtree_v2` (Act 5, K=15); note the latter is the stronger incumbent | — |
| **W3** | Add the home-advantage result (+0.02881) to Limitations as a methodological finding | — |
| **W4** | State Act 1's fitted/unfitted asymmetry explicitly, and that `tab:act1` is on test | — |
| **W5** | Rewrite the `tab:volunteered` scope paragraph: on test the bias flips from *upper bound* to *conservative* | 3.3 |
| **W6** | Elevate the ablation to **prescription** in abstract + intro: optimise the reported metric, and only that | — |
| **W7** | **Run GreedyExtend** (`golbandi2010bootstrapping`) — the missing external *static* anchor; kills "your comparator is self-built" | few h |
| **W8** | Adopt the "the objective is the bug" spine | — |
| **W9** | Distillation = infrastructure, never a contribution (§1.1 makes this unambiguous) | — |
| **W10** | Fold §4's pre-registration into the chapter before the run | — |
| **W11** | State the ablation's held-fixed design early; do **not** claim we "repaired a published method" | — |
| **W12** | Two sentences on the policy in Discussion: representable, routes unseen films, does not exceed its teacher | — |
| **W13** | Purge every number in §1.6 from chapter, docs and slides | — |
| ✔ | `sec:circularity`, `sec:opener_synthesis`, `tab:opener_match` | done |

---

## 6. SEQUENCE AND COST

```
0. [decision]  validation-fit confirmed?                          author
1. [optional]  split-half de-risker                               ~8 h
2. [run]       six opener trees on validation                     ~20 h   (2 nights)
3. [run]       distillation over the validation forest            ~3 h
4. [analyse]   validation screen; freeze all sequences            ~2 h
5. [write]     W2, W3, W4, W6, W8, W9, W11 (independent of runs)  parallel with 2-3
6. [run]       W7 GreedyExtend                                    few h
7. [PREREG]    §4 written into the chapter, reviewed              author sign-off
8. [TEST]      the one remaining look                             ~4 h
9. [write]     W1, W5, W10, W12, W13 + abstract/intro             --
```

**Critical path ≈ 30 h of compute plus writing.** Steps 5 and 6 run in parallel with 2–3. The test
look is the last irreversible act and nothing after step 7 may change a fitted object.

## 7. OPEN QUESTIONS FOR THE AUTHOR

1. **Validation-fit confirmed?** (§2) — everything downstream depends on it.
2. **Run the split-half de-risker** (§3.0), or accept the conservative framing and go straight to the trees?
3. **Primary budget K=15** and **pooled across six groups** as the primary contrast — confirm.
4. **ECIR deadline 2 Oct 2026.** ~30 h compute + writing fits, but only if the test look happens
   once. No second look.
