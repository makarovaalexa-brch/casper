# RESULT — Best-static greedy question sequences: items vs concepts vs combined

> Ran 2026-07-27/28, 568.6 min (PID 2652). `src/instrument/strategy_channel_suite.py --greedy
> --n_items 500 --n_concepts 500 --eval_test --sclite_ckpt cd_s1_l10_best.pt`.
> Output: `experiments/battery/greedy_static.json` + `.png`. Fold = the FIXED Jul-27 deployment fold
> (`cd_s1_l10_best.pt`, λ=1.0), tower = `t2i25_EP4_SNAP.pt`.
>
> **Design (author-specified):** the strategy-free concept-vs-item comparison. Greedily select the best
> STATIC question sequence for NDCG at every step (lazy CELF) from three banks — items-only,
> concepts-only, combined. **Built on all 10k VAL users, evaluated on the disjoint 10k TEST cohort.**
> Pools capped at 500 items / 500 concepts (top-answerable). Refusals NOT folded. Intercept 0.1279/0.0192.

## The curves (TEST, full@10 / tail@10)

| bank | q1 | q2 | q4 | q8 | q16 | answered@16 |
|---|---|---|---|---|---|---|
| items-only | .1405/.0275 | .1524/.0387 | .1656/.0445 | **.1863**/.0538 | **.2098**/.0679 | 3.63 i |
| concepts-only | **.1436**/**.0323** | .1504/**.0434** | .1647/**.0551** | .1776/**.0698** | .1808/.0758 | **11.04 c** |
| combined | **.1436**/**.0323** | .1504/**.0434** | .1647/**.0551** | .1817/.0616 | .2049/**.0776** | 2.75 i + 4.25 c |

## THE HEADLINE — coarse-to-fine emerges, unforced

The combined arm's emergent order is:

```
c c c c i i i i c i i i i i i i
```

**Four concepts, then items, with one concept re-entering at position 9.** Nothing in the objective
encodes "ask broad things first" — this is pure per-step NDCG maximisation over a mixed bank, and it
discovers coarse-to-fine on its own. This is the natural-sequence result the design was built to find.

## Readings (each checked against the JSON)

1. **Concepts own the opening on BOTH metrics.** At q1 concepts lead full (.1436 vs .1405) and tail
   (.0323 vs .0275). Combined is *identical* to concepts-only through q4 because greedy picks concepts
   first — the combined bank freely chose the concept channel when it had every item available.
2. **Concepts dominate the TAIL at every budget** (.0758 vs .0679 @q16), and **combined is the best tail
   arm overall** (.0776 @q16 = **+0.0097 / +14% over items-only**).
3. **Items win FULL at long budgets** (.2098 vs .1808 concepts @q16).
4. **ANSWERABILITY, the stark number:** items are answered **3.63 of 16**; concepts **11.04 of 16** —
   a 3× difference on the same budget.

## The one that needs care: combined < items on full@16

Combined (.2049) trails items-only (.2098) on full at q16 by −0.0049, even though the combined bank
*contains* the items bank. **This is NOT overfitting.** The build-vs-eval gaps are all NEGATIVE — the
test cohort scores *higher* than the build cohort in every arm:

| arm | eval full@16 | build full@16 | gap |
|---|---|---|---|
| items-only | 0.2098 | 0.2051 | −0.0047 |
| concepts-only | 0.1808 | 0.1806 | −0.0002 |
| combined | 0.2049 | 0.2024 | −0.0025 |

The real cause is **greedy myopia**: lazy-greedy maximises each *prefix*, not the endpoint. Having found
that concepts are the best opening (true — see reading 1), it commits four concept slots, and by q16 that
prefix cannot be undone. NDCG-over-a-sequence is not guaranteed submodular, so the greedy path is locally
optimal and globally slightly behind pure items at the longest budget.

**Consequence: the combined−items gap is NOT a clean readout of "what concepts add" at long budgets.**
It is confounded by the optimiser. Where it IS clean: at q1–q4 (combined chose concepts over every
available item) and on the tail throughout (combined is the best arm).

## What this does and does not support

- **Supports:** concepts are a genuine short-interview and tail instrument; a mixed bank naturally
  produces coarse-to-fine; the concept channel is answered ~3× more often than items.
- **Complicates the "answerability is ~95% of the headroom" framing:** concepts get 3× the answers and
  still LOSE on full (.1808 vs .2098). So on full-catalogue accuracy, **item information density beats
  answerability**; the answerability dividend cashes in on the TAIL, not on full. Any decomposition
  claim must be split by metric rather than stated globally.
- **Does not support** a claim that combined dominates items — it does not, on full, at this budget,
  under this optimiser.

## Follow-ups this implies

1. A **non-myopic** sequence (beam search, or greedy-with-swaps / local search at fixed budget 16) to
   remove the optimiser confound before any combined-vs-items claim is made at long budgets.
2. Endpoint-targeted greedy (optimise q16 directly rather than every prefix) as the cheaper alternative.
3. The answerability decomposition (task #45) must report full and tail separately.
