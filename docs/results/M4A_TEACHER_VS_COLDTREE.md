# M4a — do the teachers beat the cold tree at opener states?

**Code** `src/instrument/m4_compare.py` + `src/instrument/tree_route.py` · **artifact**
`experiments/battery/m4/m4a_compare.json` · **re-run 2026-08-19** after a routing defect (below);
the superseded run is kept as `m4a_compare_BUGGY_ROUTER.json`.
**VALIDATION ONLY — the single TEST look remains unspent.**

## ⚠ First: the routing defect that invalidated the original run

The first M4a run reconstructed each user's interview by **accumulating splitters while descending
the tree and then appending the leaf's `steps`**. That is not how the tree defines an interview. The
builder's own definition (`ndcg_tree.py`) is

```
own = node["path"] + node["steps"]
```

where `path` holds the ancestors' splitters — and, for a seeded teacher, the free anchor as its first
element. For a **below-floor child** the builder deliberately sets `path` to the *parent's* path
(excluding the parent's splitter) and `steps` to the *parent's tail*, whose first element **is** that
splitter. Reconstructing the sequence therefore supplied the splitter twice.

**Consequence:** 18.5–45.3% of users per tree were asked one question twice and received **14
distinct questions on a budget of 15**. 87 of the cold tree's 164 nodes are below-floor children. It
affected both arms, at different rates, so it distorted the comparison as well as the levels.

**The trees were never at fault** — checked directly, **zero** leaves in any of the seven trees
contain a duplicate in their own `path + steps` (0 of 5 for `arthouse`, 0 of 109 for `cold`). A
repeated question is never optimal and the trees never prescribe one.

Routing now emits the tree's own `path + steps` and never rebuilds it, in one shared module used by
validation, train and the reserved test look. Verified after the fix: **15.00 distinct questions for
every tree, zero repeats, zero fallbacks** on validation.

*How it hid:* the scoring path de-duplicates before folding (`dict.fromkeys`), so a repeated question
silently became a wasted turn rather than an error, and the original run recorded "duplicate
questions in a routed path" as an expected quirk of tail inheritance instead of a defect.

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
never imputed**. Zero users escaped any tree's fitted support on validation.

## ★ THE RESULT: 16 of 30 cells separate from zero, and all 30 point estimates are positive

Zero cells separate negative.

| teacher | n | depth | cold | teacher | **K=15 full** | 95% CI | K=15 tail | cells sep. |
|---|---|---|---|---|---|---|---|---|
| dialogue | 730 | 2 | .26437 | .27927 | **+0.01490** | [+0.00578, +0.02395] | −0.00890 | 4 / 5 |
| teenagers | 786 | 3 | .28633 | .29923 | **+0.01289** | [+0.00247, +0.02358] | **+0.02233** | 5 / 5 |
| marvel | 734 | 1 | .42452 | .43451 | +0.00999 | [−0.00055, +0.02060] | +0.00793 | 3 / 5 |
| gore | 747 | 3 | .36198 | .36767 | +0.00569 | [−0.00453, +0.01600] | −0.01358 | 2 / 5 |
| arthouse | 587 | 2 | .30144 | .30664 | +0.00520 | [−0.00593, +0.01658] | +0.01120 | 1 / 5 |
| inspiring | 1,822 | 4 | .26911 | .27425 | +0.00514 | [−0.00097, +0.01124] | +0.00136 | 1 / 5 |

### Per-budget grid (teacher − cold, full NDCG@10; **bold** = CI excludes zero)

| anchor | K=1 | K=2 | K=4 | K=8 | K=15 |
|---|---|---|---|---|---|
| arthouse | +0.0043 | +0.0068 | **+0.0147** | +0.0084 | +0.0052 |
| dialogue | +0.0044 | **+0.0134** | **+0.0164** | **+0.0133** | **+0.0149** |
| marvel | **+0.0219** | **+0.0184** | **+0.0139** | +0.0082 | +0.0100 |
| gore | +0.0028 | +0.0053 | **+0.0108** | **+0.0110** | +0.0057 |
| teenagers | **+0.0123** | **+0.0167** | **+0.0103** | **+0.0147** | **+0.0129** |
| inspiring | **+0.0087** | +0.0012 | +0.0016 | +0.0006 | +0.0051 |

### What the fix changed

Levels rose in both arms (each gained a genuine question) and the differences shrank slightly.
Separating cells 17/30 → **16/30**; positive point estimates 29/30 → **30/30**. `marvel`'s K=15 cell
lost significance (+0.01057 [+0.00013, +0.02105] → +0.00999 [−0.00055, +0.02060]). **No conclusion
reverses**, which is the reassuring part, but the individual numbers moved and the old ones must not
be quoted.

## What this does and does not establish

**Does:** at states where a user has volunteered an anchor, a dedicated teacher asks better questions
than the incumbent cold tree — and the advantage is not the anchor, because the anchor is in both
arms. The advantage is not confined to the first question; for `dialogue` and `teenagers` it survives
the full 15-question budget.

**Does NOT:** establish generalisation. ⚠⚠ **AND THE COMPARISON IS BIASED TOWARDS THE TEACHER**
(author, 2026-08-19). The teacher was fitted on *precisely* the cohort it is scored on; the cold tree
was fitted on all 10,000 and merely happens to cover that cohort. The specialised arm has an
advantage the incumbent does not. **These numbers are an UPPER BOUND on what conditioning is worth,
not an estimate of it.**

Two things keep them informative anyway. **The bias is not decisive** — the genre orders of
2026-08-11 were also fitted per cohort on their own users and still LOST to the tree at those states
(−0.026/−0.033/−0.031), so specialisation does not by itself manufacture a win. And **the direction
of the shrinkage is known**: v1's cold tree went +0.0314 val → +0.01035 test, a 3× loss. Expect these
effects to shrink materially rather than reverse. The M5 test look is what converts this from an
upper bound into a measurement; **every number here is provisional until then.**

## ★ Anchor specificity beats tree depth, confirmed a second way

`inspiring` — the deepest teacher (4 levels, 21 nodes) on the largest cohort (1,822) — separates only
at **K=1** and is flat (+0.0006 to +0.0051) at every later budget. Its opening question helps; four
levels of subsequent branching add nothing. `marvel`, at depth 1, separates at three budgets of five
and posts the single largest cell in the grid (+0.0219 at K=1).

**So depth does not predict payoff. How SPECIFIC the volunteered fact is does.** `inspiring` is a
broad, heterogeneous group; `marvel` and `teenagers` are narrow. Same conclusion M3 reached from a
different measurement, and it retires the "depth predicts persistence" reading floated and withdrawn
on 2026-08-18.

## ⚠ Two teachers win on the head and lose on the tail

`dialogue` (−0.0089) and `gore` (−0.0136) have **negative tail** deltas at K=15 despite positive full
NDCG. They win on head items and lose slightly on the tail — the opposite of what the chapter usually
wants, and it must be stated rather than reported as a full-NDCG-only headline. `teenagers` is the
counterexample (+0.0223 tail, its best cell).

## Defects fixed across the two runs

0. **Routing outside the fitted support** — a branch empty in the fitting cohort has no child and no
   fitted tail, and a held-out user routed there got nothing. Would have crashed the TEST look. Fixed
   by one continuation rule in `tree_route.py`.
1. **★ The splitter supplied twice** — the defect above; the reason this run exists.
2. **NaN tail handling** — tail NDCG is undefined for users with no tail target; those PAIRS are
   dropped from the bootstrap rather than imputed or mean-filled.

## Next

M4b arm C — the distilled policy — is regenerating on corrected labels. Its first build was trained
on sequences containing the duplicated question, so it learned to reproduce a sequence the tree never
prescribes; every deviation it made from its teacher was explained exactly by the teacher's repeat
(`frac_never_deviates` matched 1 − repeat-rate to within 0.002 in all six cohorts). The M5 test look
stays pre-registered and unspent.
