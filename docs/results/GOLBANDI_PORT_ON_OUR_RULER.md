# Golbandi tree PORT+CROSS: the first published adaptive method to beat a static comparator here

Run 2026-08-06. Selector `src/baselines/golbandi_tree_selector.py` (built and harness-verified
during SNAP-5, `docs/results/SNAP5_GOLBANDI_NATIVE.md`, which PASSED). Tree grown once on all
140,768 train users, exhaustive over all 18,359 candidates per node, depth 8 (6.5 h) →
`.cache/snap5/tree_ml25m_liang_d8.json`; cells scored from that checkpoint.
Output `experiments/baselines/interview/adaptive_table_gtport13.json`.
Contract: canonical Liang ML-25M, **policy-B candidate pool**, NO-FLOOR, behavioural answers,
all 10,000 test users, k ∈ {1,2,4,8} (`golbandi_leaf.MAX_DEPTH`).

## The greedy row is scored IN THIS HARNESS (superseding the anchor argument)

An earlier version of this doc admitted the greedy rows from `MASK_CONTRACT_DIAGNOSTIC.md` on the
grounds that both runs share a 4-dp entropy₀ anchor. **A mean anchor never licensed a PAIRED
difference**, so the fixed greedy items-only order was re-scored inside this harness (same
`run_interview` + `score_arm`, same pool, ~4 min). It reproduces the diagnostic exactly
(.1760/.0434 · .1907/.0581 · .2066/.0703 · .2247/.0776) and its entropy₀ control matches the stored
gtport13 vectors **per user at 0.0e+00**. All differences below are therefore genuinely paired.
CIs: `docs/results/GOLBANDI_PORT_PAIRED_CIS.md` (10,000 resamples, seed 98765, self-test exact).

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

**The tree crosses over on FULL between k=2 and k=4, and the crossover is real at both ends.**
Against entropy₀: −.0078 [−.0099,−.0056] at k=1 and −.0092 [−.0115,−.0069] at k=2, both separated
NEGATIVE; +.0064 [+.0037,+.0091] at k=8, separated POSITIVE. Every other adaptive selector in this
battery loses at *every* budget, so this is the first break in the pattern — and it is the regime a
depth-8 tree needs, since a tree cannot partition anything until it has depth.
**On TAIL the crossover is not established until k=8**: the k=4 tail gap is +.0015
[−.0011,+.0040], **ns**; k=8 is +.0052 [+.0024,+.0079], separated. State the crossover per axis.

**Against the strongest static the result splits by axis, and the honest phrasing matters**:
greedy items-only leads on full at **every** budget, all separated (−.0203, −.0293, −.0203,
−.0130 at k=1,2,4,8), while **the tree takes the tail at k=8** (.0844 vs .0776,
**+.0068 [+.0039,+.0098], separated**). So: *the tree wins the tail at k=8 while losing full at
every budget* — not "beats the optimised static order". It is the first adaptive win over the
optimised static on either axis in this battery, on one axis at one budget, against a comparator
this doc already labels a lower bound.

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

1. **Identity check PASSES exactly — per user, not merely to 4 dp.** `treestatic:entropy₀` and the
   independent `entropy0-static` arm are identical **per user (0.0e+00)** at every budget on our
   tower. The matched comparator running through the tree's own machinery is therefore honest, not
   advantaged.
2. **Fallback rate at k=8 = 11.2%**, against 11.9% predicted from the ML-100k native snap — the
   leaf-exhaustion mechanism transferred at 190× the users. Occupancy: every train user still has a
   question through depth 3, 71.6% at depth 6, 44.3% at depth 7, **0% at depth 8** (nodes peak at
   417 at depth 6 then fall to 210 with no internal nodes). **The tree runs out of splits before it
   runs out of budget**, so part of the k=8 cell measures the popularity fallback, not the tree.
3. **The tree wins on our tower while collecting FEWER answers** (2.75 vs entropy₀'s 3.10), which
   cuts against answerability as the explanation for this method — consistent with the native
   snap, where its questions were answered at 24–38% vs 36–44% for popularity, a modest cost.

## Leaf shrinkage — CORRECTED, and it moves the COMPARATOR more than the method

An earlier version of this doc said λ₁ "is inert on the ranking leaf: the λ=200 arm differs from
λ=8 by ≤.0015 everywhere". **That is false as a universal, and the intervals say so.** It holds for
the *tree* arm (−.0011 full / −.0012 tail at k=8). But on the **static** arms at k=8 full the same
shrinkage change moves things by **+.0121 [+.0094,+.0149]** (entropy₀) and **+.0153
[+.0128,+.0179]** (popularity), both separated — an order of magnitude past the stated bound and in
the **opposite sign** to the tree.

The consequence is direct: on the `golbandi_leaf` consumer, raising λ cuts the tree's k=8 advantage
over entropy₀ from **+.0248 to +.0116**. A tuning constant nobody would think to report moves the
comparator more than it moves the method under test. Any published-method cell must therefore state
its λ, and the native snap's RMSE finding must be reported as an *RMSE* finding, not carried over.

## The consumer flips which static is strongest — tested, not asserted

entropy₀ − HELF at k=8 full is **+.0160 [+.0127,+.0194]** on our tower and **−.0262
[−.0304,−.0221]** on `golbandi_leaf` — separated in **both** directions. Chapter A's
instrument-dependence, reproduced on the selection side with intervals.
Newly visible: the static ladder on our tower is not a single order either — popularity − HELF
**flips sign by axis** at k=8 (+.0051 full, −.0039 tail, both separated), and entropy₀'s tail
advantage over greedy is ns at both k=1 and k=8.

Against its own consumer the tree separates from entropy₀ and popularity everywhere, and is a
genuine **tie with HELF**: k=8 full −.0014 [−.0053,+.0025] ns, tail −.0001 [−.0032,+.0030] ns.

## Owed before citing

* k ≤ 8 only, by construction. The k=16 column of the battery table stays empty for this row.
* The 11.2% fallback should be reported *with* the k=8 cell wherever it appears.
* Every published-method cell must state its **λ** (see the shrinkage section — it moves the
  comparator more than the method).
