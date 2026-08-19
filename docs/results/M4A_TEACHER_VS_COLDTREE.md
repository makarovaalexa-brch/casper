# M4a — do the teachers beat the cold tree at opener states?

**Code** `src/instrument/m4_compare.py` · **artifact** `experiments/battery/m4/m4a_compare.json` ·
run 2026-08-19. **VALIDATION ONLY — the single TEST look remains unspent.**

## The question, and why this framing

M3's teacher numbers were measured against the **global static order with no anchor**, so a large part
of every gain was the free anchor's own contribution (priced at **+0.056** at K=1 for `arthouse` by
the seeded sanity gate). That inflates every teacher and answers nothing about conditioning.

This run neutralises it: **the anchor is folded into BOTH arms**. Per anchor cohort (validation users
who endorse it under the env's SEL rule), the comparison is

- **cold** — anchor folded, then the COLD TREE (depth 11, 164 nodes) routed by the user's own answers;
- **teacher** — anchor folded, then that anchor's own teacher, routed the same way;
- **static** — anchor folded, then the policy-B combined order (context only).

Budgets K ∈ {1,2,4,8,15} count ASKED questions; the anchor is free everywhere. Paired percentile
bootstrap, 10,000 resamples, seed 98765, over per-user differences. NaN tail pairs are **dropped,
never imputed**.

## ★ THE RESULT: 17 of 30 cells separate from zero, and every one is positive

Zero cells separate negative. 29 of 30 point estimates are positive.

| teacher | n | depth | **K=15 full** | 95% CI | K=15 tail | cells sep. |
|---|---|---|---|---|---|---|
| dialogue | 730 | 2 | **+0.01625** | [+0.00715, +0.02535] | −0.00860 | 4 / 5 |
| teenagers | 786 | 3 | **+0.01427** | [+0.00393, +0.02483] | **+0.02339** | 4 / 5 |
| marvel | 734 | 1 | **+0.01057** | [+0.00013, +0.02105] | +0.00554 | 4 / 5 |
| arthouse | 587 | 2 | +0.00584 | [−0.00539, +0.01749] | +0.01071 | 2 / 5 |
| gore | 747 | 3 | +0.00556 | [−0.00469, +0.01573] | −0.01408 | 2 / 5 |
| inspiring | 1,822 | 4 | +0.00513 | [−0.00091, +0.01119] | +0.00135 | 1 / 5 |

**For scale:** the cold tree's ENTIRE conditioning gain over the static order is **+0.0399** on
validation. `dialogue` adds a further **+0.016** for its own users — roughly 40% again, on top of an
incumbent that is already three levels deeper than the published one.

### Per-budget grid (teacher − cold, full NDCG@10; ★ = CI excludes zero)

| anchor | K=1 | K=2 | K=4 | K=8 | K=15 |
|---|---|---|---|---|---|
| arthouse | +0.0043 | +0.0052 | **+0.0111★** | **+0.0098★** | +0.0058 |
| dialogue | +0.0044 | **+0.0134★** | **+0.0170★** | **+0.0137★** | **+0.0163★** |
| marvel | **+0.0219★** | **+0.0125★** | **+0.0133★** | +0.0077 | **+0.0106★** |
| gore | +0.0028 | +0.0040 | **+0.0095★** | **+0.0114★** | +0.0056 |
| teenagers | **+0.0123★** | **+0.0167★** | +0.0084 | **+0.0154★** | **+0.0143★** |
| inspiring | **+0.0087★** | +0.0012 | −0.0004 | +0.0003 | +0.0051 |

## What this does and does not establish

**Does:** at states where a user has volunteered an anchor, a dedicated teacher asks better questions
than the incumbent cold tree — and the advantage is not the anchor, because the anchor is in both
arms. The M2 screen's front-loaded finding extends: the advantage is not confined to the first
question, and for `dialogue`, `teenagers` and `marvel` it survives the full 15-question budget.

**Does NOT:** establish generalisation. ⚠⚠ **AND THE COMPARISON IS BIASED TOWARDS THE TEACHER**
(author, 2026-08-19). An earlier version of this file claimed both arms were "equally in-sample, so
the comparison is fair". **That was wrong.** The teacher was fitted on *precisely* the cohort it is
scored on; the cold tree was fitted on all 10,000 and merely happens to cover that cohort. The
specialised arm has an advantage the incumbent does not. **These numbers are an UPPER BOUND on what
conditioning is worth, not an estimate of it.**

Two things keep them informative anyway. **The bias is not decisive** — the genre orders of
2026-08-11 were also fitted per cohort on their own users and still LOST to the tree at those states
(−0.026/−0.033/−0.031), so specialisation does not by itself manufacture a win. And **the direction
of the shrinkage is known**: v1's cold tree went +0.0314 val → +0.01035 test, a 3× loss. Expect these
effects to shrink materially rather than reverse. The M5 test look is what converts this from an
upper bound into a measurement; **every number here is provisional until then.**

## ★ Anchor specificity beats tree depth, confirmed a second way

`inspiring` — the deepest teacher (4 levels, 21 nodes) on the largest cohort (1,822) — separates only
at **K=1** and is flat (−0.0004 to +0.0051) at every later budget. Its opening question helps; four
levels of subsequent branching add nothing. `marvel`, at depth 1, separates at four budgets out of
five.

**So depth does not predict payoff. How SPECIFIC the volunteered fact is does.** `inspiring` is a
broad, heterogeneous group; `marvel` and `teenagers` are narrow. This is the same conclusion M3
reached from a different measurement, and it retires the "depth predicts persistence" reading that
was floated and withdrawn on 2026-08-18.

## ⚠ Two teachers win on the head and lose on the tail

`dialogue` (−0.0086) and `gore` (−0.0141) have **negative tail** deltas at K=15 despite positive full
NDCG. They are winning on head items and losing slightly on the tail — the opposite of what the
chapter usually wants, and it must be stated rather than reported as a full-NDCG-only headline.
`teenagers` is the counterexample (+0.0234 tail, its best cell).

## Defects fixed in this run

0. **Routing outside the fitted support** (found later, 2026-08-19, when the trees were applied to
   train users) — a branch that was empty in the fitting cohort has no child and no fitted tail, and
   a held-out user routed there got nothing. Fixed by one shared continuation rule in
   `src/instrument/tree_route.py`; **verified bit-identical on all seven validation cohorts with zero
   fallbacks**, so the numbers in this file are unaffected. See `M3_FOREST_TREES.md`.
1. **Duplicate questions in a routed path** — a question can recur along a root-to-leaf path once a
   below-floor child inherits its parent's tail; item sequences are now de-duplicated as concept
   sequences already were.
2. **NaN tail handling** — tail NDCG is undefined for users with no tail target; those PAIRS are
   dropped from the bootstrap rather than imputed or mean-filled.

## Next

Arm C — the distilled policy — needs a build: `policy_traj.py` is hardwired to the old tree and a
16-question budget, and `forest_policy.py` distils static ORDERS, not trees. E1 established that a
distilled policy reproduces its teacher at best and never exceeds it, so arm C is expected at or
below arm B. The M5 test look stays pre-registered and unspent.
