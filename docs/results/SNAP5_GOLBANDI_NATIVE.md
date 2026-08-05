# SNAP-5: Golbandi ternary tree — native snap PASSES, and there is no inversion

Run 2026-08-05. Target: Golbandi, Koren & Lempel, *"Adaptive Bootstrapping of Recommender Systems
Using Decision Trees"*, WSDM'11 pp. 595–604, DOI 10.1145/1935826.1935910.
Artifacts: `experiments/results/snap5/` (JSON with per-user paired arrays + four .txt tables),
trees cached in `.cache/snap5/`. Code: `src/baselines/golbandi_tree_selector.py`,
`scripts/baselines/golbandi_ml100k_snap.py`.

## The published target [V]

Extracted from the repo's own full-text pass (`external_literature/findings/
rating_elicitation_family_deep.md` §2.5); the PDF mirror is HTTP 410.

* **Split criterion (Eq. 1).** They "deviate from common error metrics typical to classification
  tree construction (like entropy or Gini impurity), but rather rely on the squared loss metric":
  `Err_t(i) = e²(t_L) + e²(t_H) + e²(t_U)`, `splitter(t) = argmin_i Err_t(i)`, **exhaustive over all
  n items at every node**. The unknown branch is derived by subtraction (Eq. 3).
* **Ternary cut:** 4–5 lover, 1–3 hater (dataset mean 3.6), unrated unknown.
* **Constants:** user-bias shrinkage β=7; leaf = per-item hierarchical shrinkage **toward the
  parent**, **λ₁=200**, chosen over pruning explicitly ("pruning a node as a whole would be
  unappealing in our sparse data scenario"); node stop ξ=200,000 rating mass.
* **Netflix, RMSE only.** Table 1: 0.97260 / 0.97172 / 0.97129 / 0.97115 at depth 5/6/7/8.
  Headline (Fig. 3): depth-6 reaches 0.97172 in **6** questions; GreedyExtend needs >20, Var and
  HELF >30; organic ratings need the first 12.
* Their own verdict, worth quoting in the chapter: *"a decision-tree recommender delivers a quite
  high RMSE. This makes decision-trees undesirable as a traditional recommendation system."* The
  tree is an **instrument**, not a recommender — the same framing as Chapter A.
* ★ **[V] "a 'like' vote is considerably more useful than a 'dislike' vote"** (leaf weights
  w_L=5, w_H=1, w_U=0.02). **A 2011 precedent for our Arm-N sign result — cite it, do not
  rediscover it.**

**Netflix is unavailable, so a numeric snap to Table 1 is impossible and was not attempted.** What
follows is a **shape snap on ML-100k** and must always be labelled as such.

## Why the archived precedent is not citable

`scripts/_archive/paper1_v1/golbandi_tree.py` still runs and reproduces its recorded numbers
exactly (depth 6: adaptive 0.9804 vs static 0.9885). But it has **an answer leak** — the interview
was answered from the cold user's *whole* profile, including the ratings it then scored RMSE on —
and **a candidate-pool truncation** (`POOL=80` vs the paper's exhaustive search). Both inflate the
adaptive arm. The rebuild below fixes both.

## The rebuilt snap

200 held-out cold users; each user's ratings split 70/30 into a fold-in side (the only source of
answers) and a **disjoint** eval side; exhaustive search over all 1,682 items; β=7 and
parent-shrinkage as published; ξ rescaled from 200,000/100M = 0.2% of corpus mass; **three** static
comparators (popularity, Var, HELF) driven through the *same* tree machinery with the question
fixed per depth — identical node model, routing, shrinkage and question count, only the selection
differs; 3 seeds; paired-over-users bootstrap.

**RMSE, seed-averaged, λ₁=64 (sweep optimum). Lower is better. \* = paired CI excludes zero.**

| depth | ADAPTIVE | HELF | popularity | Var | Δ vs HELF | Δ vs pop | Δ vs Var |
|--:|--:|--:|--:|--:|--:|--:|--:|
| 0 | 1.0174 | 1.0174 | 1.0174 | 1.0174 | — | — | — |
| 1 | **1.0123** | 1.0153 | 1.0162 | 1.0165 | −.0031\* | −.0038\* | −.0042\* |
| 2 | **1.0109** | 1.0136 | 1.0162 | 1.0155 | −.0028 | −.0054\* | −.0047\* |
| 3 | **1.0106** | 1.0135 | 1.0166 | 1.0151 | −.0030 | −.0061\* | −.0046\* |
| 4 | **1.0106** | 1.0130 | 1.0171 | 1.0150 | −.0026 | −.0067\* | −.0046\* |
| 6 | **1.0106** | 1.0133 | 1.0172 | 1.0149 | −.0028 | −.0067\* | −.0044\* |
| 8 | **1.0107** | 1.0136 | 1.0173 | 1.0149 | −.0030 | −.0068\* | −.0044\* |

### ★ Shrinkage decides whether depth helps at all

| λ₁ | d1 | d2 | d3 | d4 | d6 | d8 | separates from all three statics at every depth? |
|--:|--:|--:|--:|--:|--:|--:|:--|
| 8 (archived) | 1.0143 | 1.0192 | 1.0270 | 1.0344 | 1.0435 | 1.0458 | no |
| 32 | 1.0119 | 1.0114 | 1.0122 | 1.0129 | 1.0139 | 1.0142 | no |
| 64 | 1.0123 | 1.0109 | 1.0106 | 1.0106 | 1.0106 | 1.0107 | no (HELF ns from d2) |
| 128 | 1.0135 | 1.0120 | 1.0114 | 1.0111 | 1.0109 | 1.0109 | **YES** |
| **200 (paper)** | 1.0144 | 1.0130 | 1.0124 | 1.0121 | 1.0119 | 1.0119 | **YES** |

At the archived λ₁=8 the interview actively **hurts** past depth 1 (1.0143 → 1.0458). At the
paper's own λ₁=200 it helps monotonically and beats every static at every depth with CIs excluding
zero. **VERDICT: SNAP PASSES on the paper's own constant.**

## ★★★ THE INVERSION VERDICT: do not report an inversion

**On RMSE — the paper's metric — there is none.** The adaptive tree is never significantly worse
than any static order at any depth, any shrinkage, any seed; at the paper's λ₁ it is significantly
better everywhere.

**On ranking it is a TIE, not an inversion.** With the like-rate leaf (`golbandi_leaf`'s documented
ranking adaptation — Golbandi's mean-rating leaf is unusable for top-N), the adaptive tree's
NDCG@10 point estimate leads every static at every depth ≥2, but **no paired CI excludes zero
against HELF or Var** (λ₁=64, d8: +0.0030 [−0.0102, +0.0161] vs HELF; +0.0028 [−0.0100, +0.0162]
vs Var). It separates only against popularity. At λ₁=400 the best static overtakes it at d8.

**The honest statement is: the metric decides.** On squared loss the split search buys a real,
separated gain; on ranking it is indistinguishable from a strong static order. That is exactly the
shape Sepliarskaia RecSys'18 attributes to the tree optimising a variance surrogate rather than the
task loss, and it is consistent with our own `TREE_VS_STATIC.md` (−0.0094 [−0.0158, −0.0033] on
NDCG) — that experiment used a *learned-NDCG* comparator, a stronger static than any of these
three. **The old ML-1M arena number (.343/.124 vs .352/.129) is retired: it measured a different
object.** PAPER_B_DESIGN's open item "the inversion must vanish or be explained before Act 1
prints" is hereby CLOSED — it vanishes, and the explanation is the metric.

## Leaf fragmentation — the mechanism behind MAX_DEPTH

743 train users, ternary, adaptive tree, seed-averaged:

| depth | #nodes | users/node mean | median | #nodes <10u |
|--:|--:|--:|--:|--:|
| 1 | 3 | 247.7 | 178 | 0 |
| 2 | 9 | 82.6 | 65 | 0 |
| 3 | 27 | 27.5 | 23 | 5 |
| 4 | 80 | 9.3 | 6 | 57 |
| 6 | 383 | 1.7 | 1 | 377 |

A cold user's node holds 330 → 2.7 train users across depths 1–8, and the fraction hitting a leaf
before the budget is spent goes **0% through depth 5 → 6% at d6 → 37% at d7 → 75% at d8**
(harness fallback rate at k=8: 11.9% of asks). Past depth 3 the median leaf fits 1–2 users, which
is why λ₁ is load-bearing and why `golbandi_leaf.MAX_DEPTH=8` exists.

**A minimum-child-size floor is not the fix**: `--min_child_users 25` stops the tree at depth 3
(10 nodes, none still splitting) — it converts the pathology into a truncated budget. It does
flatten the λ₁=8 degradation, confirming fragmentation as the cause.

**★ Answerability is NOT the mechanism here** (contrary to prior expectation): the exhaustive
search picks items at median popularity rank 5–20 (vs 0–7 for the popularity order), answered by
24–38% of cold users vs 36–44% for popularity — a real but modest cost, not a collapse. The
answerability story belongs to the *item-vs-concept channel*, not to tree selection.

## The selector (step P is now wiring, not a rewrite)

`src/baselines/golbandi_tree_selector.py` — `GolbandiTree` implements Eq. 1 exhaustively over all
candidates via six sparse products per candidate chunk, unknown branch by Eq. 3 subtraction;
`GolbandiTreeSelector` implements the `adaptive_harness.Selector` contract (grows once on train
users to the largest budget — a depth-d prefix *is* the depth-d tree; an asked-but-unrated item
takes the UNKNOWN branch; leaf-before-budget falls back to the next unasked popularity item);
`StaticOrderTreeSelector` is the symmetric comparator through the same code path. Verified against
the real harness loop (shape, range, no-re-ask assert): 8 turns, 200 users, 91 distinct prefixes.
**Pair it with the `golbandi_leaf` consumer for the method as published**; any other consumer is
the selector on a different node model and must be labelled so.
