# Golbandi tree PORT+CROSS: the first published adaptive method to beat a static comparator here

Run 2026-08-06. Selector `src/baselines/golbandi_tree_selector.py` (built and harness-verified
during SNAP-5, `docs/results/SNAP5_GOLBANDI_NATIVE.md`, which PASSED). Tree grown once on all
140,768 train users, exhaustive over all 18,359 candidates per node, depth 8 (6.5 h) →
`.cache/snap5/tree_ml25m_liang_d8.json`; cells scored from that checkpoint.
Output `experiments/baselines/interview/adaptive_table_gtport13.json`.
Contract: canonical Liang ML-25M, **policy-B candidate pool**, NO-FLOOR, behavioural answers,
all 10,000 test users, k ∈ {1,2,4,8} (`golbandi_leaf.MAX_DEPTH`).

## Cross-run comparability, verified rather than assumed

The greedy rows below come from `MASK_CONTRACT_DIAGNOSTIC.md`, a different run. They are admissible
because the two runs share an **exact anchor**: entropy₀ reads `.1635/.0433 · .1706/.0528 ·
.1831/.0636 · .2053/.0792` at k=1,2,4,8 in **both**. Four budgets, both axes, identical to 4 dp.

## (X) The Act-1 cell — consumer = certified tower i26c ep31

| selector | k=1 | k=2 | k=4 | k=8 | answers@8 |
|---|---|---|---|---|---|
| greedy items-only (strongest static)¹ | **.1760**/.0434 | **.1907/.0581** | **.2066/.0703** | **.2247**/.0776 | 4.20 |
| **golbandi_tree** | .1557/.0351 | .1614/.0478 | .1864/.0650 | .2117/**.0844** | 2.75 |
| treestatic:entropy₀ | .1635/**.0433** | .1706/.0528 | .1831/.0636 | .2053/.0792 | 3.10 |
| treestatic:popularity | .1606/.0400 | .1647/.0470 | .1746/.0579 | .1943/.0686 | 3.05 |
| treestatic:HELF | .1603/.0354 | .1628/.0420 | .1755/.0588 | .1892/.0725 | 1.59 |

¹ lower bound: the greedy sequence was *selected* under the narrower mask and scored under this
one, so the true strongest static is at least this good — which cuts against the tree.

**The tree crosses over between k=2 and k=4.** Against entropy₀ it is −.0092 full at k=2 and
**+.0064 full / +.0052 tail at k=8**. Every other adaptive selector in this battery loses at
*every* budget, so this is the first break in the pattern — and it is the regime a depth-8 tree
needs, since a tree cannot partition anything until it has depth.

**Against the strongest static the result splits by axis**: greedy items-only leads on full at
every budget (+.0130 at k=8), but **the tree takes the tail at k=8** (.0844 vs .0776, +.0068) —
the first time any adaptive selector has beaten the optimised static order on either axis.

## (P) The method as published — consumer = golbandi_leaf

| selector | k=1 | k=2 | k=4 | k=8 |
|---|---|---|---|---|
| golbandi_tree | **.1696/.0405** | .1712/.0486 | .1867/.0711 | .1934/.0792 |
| treestatic:HELF | .1665/.0402 | **.1739/.0473** | **.1890**/.0713 | **.1948**/.0792 |
| treestatic:popularity | .1531/.0331 | .1534/.0425 | .1445/.0501 | .1716/.0696 |
| treestatic:entropy₀ | .1579/.0347 | .1563/.0414 | .1680/.0670 | .1686/.0677 |

With its own consumer the tree beats entropy₀ and popularity clearly but is **essentially tied with
HELF** (.1934 vs .1948 full, .0792 both on tail). Note the consumer flips which static is strongest:
HELF here, entropy₀ on our tower — Chapter A's instrument-dependence, again.

## Three checks that make the cells trustworthy

1. **Identity check PASSES exactly.** `treestatic:entropy₀` reproduces the independent
   `entropy0-static` arm to 4 dp at every budget on our tower (.2053/.0792 at k=8). The matched
   comparator running through the tree's own machinery is therefore honest, not advantaged.
2. **Fallback rate at k=8 = 11.2%**, against 11.9% predicted from the ML-100k native snap — the
   leaf-exhaustion mechanism transferred at 190× the users. Occupancy: every train user still has a
   question through depth 3, 71.6% at depth 6, 44.3% at depth 7, **0% at depth 8** (nodes peak at
   417 at depth 6 then fall to 210 with no internal nodes). **The tree runs out of splits before it
   runs out of budget**, so part of the k=8 cell measures the popularity fallback, not the tree.
3. **The tree wins on our tower while collecting FEWER answers** (2.75 vs entropy₀'s 3.10), which
   cuts against answerability as the explanation for this method — consistent with the native
   snap, where its questions were answered at 24–38% vs 36–44% for popularity, a modest cost.

## What did NOT transfer

**Leaf shrinkage.** The native snap found λ₁ load-bearing on RMSE (at λ₁=8 the interview actively
*hurts* past depth 1). On the ranking leaf it is inert: the λ=200 arm differs from λ=8 by ≤.0015
everywhere (k=8: .1923/.0779 vs .1934/.0792). Report the RMSE finding as an RMSE finding.

## Owed before citing

* **Paired CIs on every gap here.** The headline tail difference is +.0068; nothing in this table
  has an interval yet.
* k ≤ 8 only, by construction. The k=16 column of the battery table stays empty for this row.
* The 11.2% fallback should be reported *with* the k=8 cell wherever it appears.
