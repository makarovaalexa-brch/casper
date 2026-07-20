# LEARNED TREE vs LEARNED STATIC -- LEAVE-ONE-USER-OUT / K-FOLD (concept family; DIRECTIONAL 173/300)

> **DIRECTIONAL BANNER** -- 173/300 users, answerer-v1 working grid NOT frozen; v2 fold
> `.cache/i25_fold_v2_best.pt`. NO LLM API calls; local compute. Re-run on the frozen 300-user
> grid before any citation.

Date 2026-07-09. Script `scripts/tree_louo.py`. 173 users; 180 candidates
(60c/60i/60a); cold NDCG@10 0.1502; fold val 0.4256. Depth=24 turns. Candidate pool = top-60 by coverage (same as
the static). Paired per-user bootstrap BOOT=5000. FOCUS = s-mixed (concept) family; item
family = cheap confirmation only. Answer classes: pos(liked/loved)/neg(meh/hated)/noclue(refuse,k=0; E1).

## Pre-registered reads (printed BEFORE results)

CLASS-MATCHED contrast at POWER: the LEARNED conditional interview TREE vs the LEARNED static
questionnaire, IDENTICAL learner (same greedy NDCG objective + code path; branching-disabled tree
== the sequence, asserted at runtime), grown by CROSS-CONSTRUCTION (K-fold: every eval user scored
by an artifact grown on the OTHER K-1 folds = 156-172 construction users, ~2x the 86 of the
half-split TREE_VS_STATIC run). FOCUS family = s-mixed (concept). Pooled paired per-user delta
(tree - static) over ALL 173 users (each user an eval user exactly once), bootstrap CI, at
T=8/12/24.

THRESHOLDS (decided before any number is seen):
- **CI EXCLUDES 0, POSITIVE** -> the program's FIRST FAIR, ADEQUATELY-POWERED ADAPTIVITY WIN:
  the concept-tree's 6/6-positive-ns trend becomes detectable once the estimator is fed 2x data.
  Stated plainly, no softening.
- **CI INCLUDES 0** -> the trend does NOT strengthen with 2x construction data. Evidence the true
  advantage is below the resolvable band; we QUANTIFY THE BOUND as the CI upper limit (report it).
- **CI EXCLUDES 0, NEGATIVE** -> reported as-is (the tree loses even fed 2x data).

SECONDARY (E7 -- both arms get the same extra food): the K-fold static (156-172 construction) vs
the OLD half-split static (86 construction), paired per-user over 173 -> does the STATIC also
improve with more data? Neither arm may be starved relative to the other.

MECHANISM: branch counts / depths / channel mix at 156-172 users vs the old 86; which concept is
asked after a positive vs a negative answer (coarse->fine visibility).

GATES: (0) all-173 greedy s-item anytime@T=12 == 0.2251 +/- 0.002 (anchor). (1) branch-disabled
tree schedule == SC.build_greedy_sub sequence (SAME learner). (E2) in-sample tree >= in-sample
static by construction.

## E7 ARM-SYMMETRY TABLE (what each arm sees / learns from / is constructed on)

| property | LEARNED TREE | LEARNED STATIC (greedy sequence) | symmetric? |
|---|---|---|---|
| construction cohort | the eval user's complement (K-1 folds, 156-172 users) | SAME complement | YES |
| objective | greedy cohort-mean NDCG@10 gain | greedy cohort-mean NDCG@10 gain | YES |
| code path | LT.greedy_pick (per-node body) | SC.build_greedy_sub (per-pos body) -- identical | YES |
| candidate pool | top-60 concepts by coverage | top-60 concepts by coverage | YES |
| instrument / fold | v2 fold i25_fold_v2_best | v2 fold i25_fold_v2_best | YES |
| eval cohort | held-out fold (never in construction) | SAME held-out fold | YES |
| privileged info / value model | NONE | NONE | YES |
| refusal handling | no-op turn, cold fallback (E1) | no-op turn, cold fallback (E1) | YES |
| **conditioning on observed answer class** | **YES (3-way Golbandi branch, >=MIN_USERS)** | **NO (fixed order)** | **the ONLY asymmetry -- the treatment under test** |

> The tree is a strict superset of the static (branch-disabled tree == the sequence). The single
> asymmetry is the adaptivity itself. Any held-out tree advantage is therefore attributable to
> answer-class conditioning and nothing else (E7 satisfied).

## 0. Reproduction (anchor gate)

All-173 greedy s-item anytime NDCG@10 @T=12 = **0.2251** vs anchor 0.2251 +/- 0.002 -> **MATCH**.

## 1. Identity proof (SAME learner)

Branching-disabled tree schedule == greedy sequence (`SC.build_greedy_sub`) ? **True**.
Certifies the tree grower and the static comparator share one code path / objective.

## 2. Protocol selection (LOUO vs K-fold)

One concept grow (depth 24, 155 users) = **224.1s**. Full LOUO (K=173, 346 grows) extrapolates to **~1434 min (23.9 h)**.

**PROTOCOL USED: 10-fold cross-construction** (forced via --k 10). Construction cohort ~= **155 users** per eval user (vs 86 in the half-split TREE_VS_STATIC run; vs 172 for full LOUO). Every one of the 173 users is an eval user in exactly one fold -> pooled n=173.

