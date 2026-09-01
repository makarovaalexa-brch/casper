# Paper B — audit and finishing plan

**2026-08-25.** Full artifact audit, the test-look ledger, and the run plan. Supersedes
`PAPER_B_ACTION_ITEMS.md`. Author's design ruling incorporated (§2).

---

## ★ STATE: Acts 1–4 are already on TEST and frozen

The test look was spent 2026-08-12. The protocol was *freeze every sequence on validation, then
construct test* (chapter line 215). Act 5 is the only part that is not measured.

| act | content | cohort | status |
|---|---|---|---|
| 1 | adaptive selectors vs a static order, 30 cells | **10,000 TEST** (`tab:act1`) | ✅ frozen |
| 2 | surrogate-vs-metric diagnosis | validation (a measurement, not a contrast) | ✅ |
| 3 | the tree; split-criterion ablation; genre arm | **TEST**, `+0.01035 / +0.00705` @K=16 | ✅ frozen |
| 4 | answerability orders the channels | validation, descriptive | ✅ |
| 5 | volunteered information | validation, **anchor states** | ⛔ rebuild |

**Act 3's static comparison already won cleanly on test.** That is what licenses §2: refitting the
comparators on train can only *disadvantage* them relative to that, and the ordering is not at risk.

---

## 1. AUDIT FINDINGS

### 1.1 ⛔ Retraction: "the consistent pair fixed distillation" is wrong

```
m6a2k   policy 0.28398  tree 0.28366  cold 0.28685  | vs tree +0.00032
m6fair  policy 0.28310  tree 0.28366  cold 0.27311  | vs tree -0.00057
```

**The teacher is identical in both (0.28366) and the policy tracks it in both.** Only the comparator
moved. Distillation was never broken and is not newly fixed — it has always reproduced its teacher to
±0.001. **Distillation is solved infrastructure, never a contribution** (`W9`). General lesson: when
a contrast moves, check whether the ARM moved or the COMPARATOR did.

### 1.2 Two trees are in play

* `ndcgtree16` — 22 nodes, 15 leaves, depth 5, **K=16**. Act 3's tree, on test.
* `coldtree_v2` — 164 nodes, depth 11, **K=15**, built 2026-08-18. Act 5's incumbent. Never tested.

Needs one sentence in the chapter; the favourable framing is true (Act 5's incumbent is the stronger
of the two, so beating it is the conservative comparison). Do not belabour it.

### 1.3 ★ Act 5 pays off at the wrong states — this is the real defect

`tab:volunteered` measures the OLD teachers (`m3v2_*`), each fitted at ONE state with its group
concept folded and nothing else. But Act 5 now
argues that real requests are Reddit-shaped openers (films 63%, genre 40%, tone 21%), and the banked
result is that anchor teachers **do not survive the move**: `+0.00897` at the anchor →
`−0.0020 / +0.0008 / −0.0014 / −0.0024` at one film / two films / a concept / a bundle.

The chapter motivates realistic openers and then pays off at anchor states. **Everything in §3 exists
to close that gap.**

### 1.4 Numbers that must never appear

* `+0.01762` — retracted arthouse "held-out" figure (memorised ruler, tree at home).
* All M6m train-user held-out figures (pooled `+0.00278`) — same defect.
* `+0.00782` as a home advantage — superseded by the measured `+0.02881`.
* `+0.229 / +0.0115 / +0.0153` (K.6) — audited wrong.

### 1.5 To verify before submission

`greedy_static_policyB_combined.json` carries a stale `ruler_note` claiming policy-A selection while
its own `supersedes` field and `CONDITIONED_ARMS_ON_TEST.md` say all three static orders are
policy-B-selected. Confirm, so the chapter does not repeat a retired caveat.

---

## 2. DESIGN (author's ruling, 2026-08-25)

**Fit every tree and the static order on TRAIN. Evaluate on VALIDATION. Do not spend the test look.**

Rationale, recorded once and **not rehearsed in the paper**: validation holds only 587–1,822 users
per group, which under `min_leaf` 200 yields a stub, not a teacher. Fitting everything on train gives
real teachers and puts every arm on the same footing, with validation clean for all of them.

The residual — a train-fitted arm optimises against a ruler the tower memorised — applies to **every
arm equally** and can only cost absolute magnitude, not order. Act 3 already showed the tree beating
static on test, so the ordering is established; this design can only disadvantage the comparators
relative to that.

> **WRITING RULE (author, 2026-08-25): none of this goes in the paper.** The protocol is stated flat
> — *all interviews were fitted on the 140,768 training users and evaluated on the 10,000 held-out
> validation users* — and nothing more. No confound narrative, no home-advantage essay, no account of
> how we got here. No actual author writes that. Cf. `papers-present-the-working-version`.

### The target result

```
static  <  cold  <  opener-fitted        (at opener states, per group and pooled)
```

and then the payoff, which is the deployable claim:

> One distilled policy tracks each opener tree closely, routes to the right tree from the STATE
> alone, extrapolates to films held out of its training openers, and therefore wins on a MIX of cold
> users and opener users drawn from every cluster — a single universal interview strategy, no router.

**The mix arithmetic, corrected.** `[mix] 587 in a group (5.9%)` in the banked logs is a
SINGLE-GROUP run (`[groups] ['arthouse']`) and must not be read as the population split. With all six
groups the training banks give **cold 85,289 / 140,768 = 60.6%**, so **39.4% of users behaviourally
endorse at least one of the six group-defining concepts**; validation agrees (5,406 memberships over 10,000 users at 1.37 groups per grouped
user ⇒ ~3,941 ⇒ ~39.4%). So the expected mix gain is ≈ `0.394 × 0.011` ≈ **+0.0043**, not the
+0.0007 a 6% share would imply.

Three things to keep straight when writing it:
1. **Membership is behavioural, not assigned** — `foldable(u,g) > 0 AND V(u,g) > 0`, i.e. the user can
   answer that concept and their signed value from real ratings is positive.
2. **The 39/61 split is a property of choosing SIX concepts**, not of people. A larger vocabulary
   reaches more users. One honest clause: our six-concept vocabulary reaches 39% of the population.
3. **Grouped users average 1.37 groups each**; the mix assigns multi-group users by nearest centroid,
   which is our tie-break. And a user who volunteers something outside the six is counted cold here —
   that understates reachable users, it does not overstate them.


### 2.1 TERMINOLOGY — "anchor" is retired

The word does two jobs in the code and only one is still live:

* **Anchor as a teacher STATE — DEAD.** Fitting a tree with one attribute folded and nothing else is
  the `m3v2` / `ndcg_tree --cohort_concept` path. It is what `m6k` exists to replace
  (`m6k_cluster_tree.py:18`), because those teachers do not survive the move to realistic openers.
* **The group-defining CONCEPT — live.** `ANCHORS[g]` is now only a lookup for the concept id that
  selects the cohort: `cohort = rows_all[endorse[rows_all, cid]]`. The correct name for the result is
  an **endorsement cohort** — users who behaviourally endorse one concept. Cf.
  `cluster-vs-endorsement-cohort-and-the-era-hole`.

**In writing and in conversation, say "group-defining concept" and "endorsement cohort". Never
"anchor".** It is a legacy variable name, not a design element. (`m6c_distill_gen.py:209` still
stores `anchor=` in each bank; under `--anchor_encoding none` the policy never reads it — the field
is inert and must not appear in any description of how the policy routes.)

---

## 3. SAMPLE SIZES AND RUNS

Measured costs (`m6q`, assembled path): 2,037 users + openers = **1.97 s/eval**; 10,000 users without
openers = **6.78 s/eval**. `arthouse@2000` actually cost ~4 h of real compute (its 13.4 h wall was
4.1 h battery stall + 2.6 h redundant re-seeding, the latter now fixed) ⇒ **~3 h per group at N=2,000**.

| arm | fit split | **N** | status | cost |
|---|---|---|---|---|
| cold tree | train | **10,000** | ✅ `coldtree_v2_train10000` (144 nodes, depth 11) | done |
| static order | train | **10,000** | ⛔ refit — resume `coldtree_v2_train10000_heaps/node_root.json`, which *is* its seed pass | ~1.5 h |
| opener trees ×6 | train | **2,000** | arthouse ✅ (27 nodes, depth 6); five remaining | ~15 h |

**Why N=2,000 for the group trees** (train availability is 8,263–25,648, so this is a choice not a
limit):

1. **It reuses the arthouse tree**, which is already fitted and already produced `+0.01112`.
2. **Equal fitting budget per group** — otherwise cross-group differences confound cohort size.
3. **Significance comes from the EVAL cohort, not the fitting N.** Per-group validation n is
   587–1,822; pooled ≈ 5,400. At the observed +0.011 with pooled SE ≈ 0.002 that clears zero
   comfortably. Spending 37 h to reach N=4,000 buys effect size we do not need.
4. **N=2,000 against cold's 10,000 runs against us** — conservative, and worth one clause in the paper.

Alternative if the pooled result comes out marginal: rerun the weakest groups at N=4,000 (~7 h each).
Cheap-first is the right order; going straight to 4,000 costs 37 h up front.

### Run sequence

```
1. static order on train @10,000 (resume the root heap)                    ~1.5 h
2. five opener trees on train @2,000  (dialogue, marvel, gore,
   teenagers, inspiring) -- min_leaf 200, depth 6, sweeps banked           ~15 h
3. m6c gen: cold_train + 6 opener trees, half A                            ~1 h
4. m4b distil: 6 epochs, anchor_encoding=none, expect_cover back to 0.95
   (all six groups banked -> coverage returns to ~100%)                    ~1.5 h
5. m6d --mode generalise --half B     (unseen films)                       ~20 m
6. m6d --mode mix                     (THE payoff: cold + all clusters)    ~30 m
7. m6r extended: static_train / cold_train / opener, per group AND pooled  ~1 h
                                                                    total ~21 h
```

Step 7 needs `m6r_fair_compare.py` extended: add `static_train` as an arm, loop all six groups, and
add the pooled contrast. Steps 1–2 are the only long ones and can run overnight back-to-back.

### Not rerun

Cold tree (done), anchor teachers, split-criterion ablation, Acts 1–4. All frozen.

---

## 4. WRITING TO-DO

| # | item | depends on |
|---|---|---|
| **W1** | ★ Replace `tab:volunteered` with **opener-state** results: static / cold / opener, per group + pooled | 3.7 |
| **W2** | Add the mix result as Act 5's payoff — one policy, no router, wins across cold + all clusters | 3.6 |
| **W3** | One sentence distinguishing `ndcgtree16` (Act 3, K=16) from `coldtree_v2` (Act 5, K=15) | — |
| **W4** | State the protocol flat: fitted on train, evaluated on validation. **One sentence. No narrative.** | — |
| **W5** | One clause noting group trees used 2,000 fitting users against the incumbent's 10,000 (conservative) | — |
| **W6** | Elevate the ablation to **prescription** in abstract + intro: optimise the reported metric, and only that | — |
| **W7** | **Run GreedyExtend** (`golbandi2010bootstrapping`) — the missing external *static* anchor. **Author ruling 2026-08-29: "run it just for the sake of it" — completeness, not a headline arm.** ⚠ Its objective is **RMSE**, items-only; ours is greedy on **NDCG@10** over 500 items + 500 concepts. Same algorithm, different objective/action space, so **do NOT rename our static to GreedyExtend** — ours optimises the reported metric and is therefore the HARDER static baseline; W7 should widen the adaptive margin, not threaten it. ⚠ CIKM'10 full text unobtainable — any implementation is a reconstruction from WSDM'11 §3.1.1 + patent US8510248B2 and must be labelled as one. ⚠ **If it is ever to appear on TEST it must run BEFORE the confirmatory look** | few h |
| **W13** | **MovieLens user-segmentation literature** — never searched (`INDEX.md` has zero hits for segment/user cluster/taste group). Author asked on 2026-08-28 and it was skipped; the clustering was built from scratch instead. Needed for related work and for "why these six groups", NOT for the result. Commit to `external_literature/` per hard rule 4 | ~1 h |
| **W8** | Adopt the "the objective is the bug" spine | — |
| **W9** | Distillation = infrastructure, never a contribution (§1.1) | — |
| **W10** | Rewrite the Act 5 scope paragraph — the *upper bound* caveat dies once both arms are fitted off-cohort | 3.7 |
| **W11** | State the ablation's held-fixed design early; do **not** claim we "repaired a published method" | — |
| **W12** | Purge every number in §1.4 from chapter, docs, slides | — |
| ~~W-old~~ | ~~home-advantage essay in Limitations~~ — **struck by the writing rule in §2** | — |
| ✔ | `sec:circularity`, `sec:opener_synthesis`, `tab:opener_match` | done |

---

## 5. SEQUENCE

```
0. [decide]  N=2,000 per group confirmed?                          author
1. [run]     static on train, five opener trees        ~17 h   (2 nights)
2. [write]   W3, W4, W6, W8, W9, W11  -- independent of runs      parallel
3. [run]     distil + generalise + mix + comparison    ~4 h
4. [run]     W7 GreedyExtend                           few h
5. [write]   W1, W2, W5, W10, W12 + abstract/intro
```

**~21 h compute + writing. The test look stays unspent.** ECIR deadline 2 Oct 2026.

## 6. FOR THE AUTHOR

1. **N=2,000 per group** — confirm, or go to 4,000 up front (37 h instead of 15 h).
2. **Static refit at N=10,000** to match the cold tree — confirm (it is the incumbent; matching it
   rather than the group trees keeps the comparison conservative).
3. Anything else that must stay out of the paper beyond §2's writing rule?
