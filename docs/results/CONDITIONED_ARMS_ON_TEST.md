# Conditioned selection on held-out users: the tree, the genre split, and the static orders

Written 2026-08-12. **REVISED 2026-08-12 17:45 — the tree is now measured against the REBUILT
POLICY-B comparator (A2). §7's contamination caveat is DISCHARGED FOR THE TREE; it still stands for
the genre arm.** Every number below is on the **10,000 canonical TEST users** unless marked
otherwise, scored with the certified pair (`t2i26c_ep31.pt` + `concept_cv3_ep08.pt`), NO-FLOOR,
policy-B candidate pool, learned decoder bias, behavioural answers. Paired percentile bootstrap over
per-user differences, 10,000 resamples, seed 98765 — never the unpaired form.

Sources: `experiments/battery/ndcgtree16test_policyB.json` (the current tree numbers),
`greedy_static_policyB_combined.json` (the rebuilt comparator), `ndcgtree16.json` (validation),
`ndcgtree16test.json` (the superseded policy-A measurement, kept for the before/after),
`groupgreedy_genre15_TEST.json`, `ndcgtree.json`, `docs/results/MASK_POLICY_B_MIGRATION.md`.

---

## 1. The table

| arm | pool | Q | full@10 | tail@10 | gain vs its own static |
|---|---|---|---|---|---|
| **ask nothing (m=0)** | — | 0 | 0.15855 | 0.02819 | — |
| static items-only (policy-A selected) | items | 15 | 0.24356 | 0.09689 | — |
| static combined (policy-A selected, superseded) | combined | 16 | 0.25565 | 0.10982 | — |
| **static combined, REBUILT under policy B** | combined | 16 | **0.25900** | **0.12370** | — |
| genre split, fallback-200 | items | 15 | 0.25173\* | 0.10580\* | +0.00817 / +0.00891 |
| genre split, all-groups | items | 15 | 0.25311\* | 0.10760\* | +0.00955 / +0.01071 |
| **NDCG tree** | combined | 16 | **0.26936** | **0.13074** | **+0.01035 / +0.00705** |

\* derived as comparator + difference; the genre artifact stores differences and the comparator's absolute.
The genre rows are still against a **policy-A-selected items order** — A3 rebuilds it.

The m=0 intercept reproduces `MASK_POLICY_B_MIGRATION.md`'s banked policy-B test figure
(0.15855 / 0.02819) **exactly** — an independent check that the ruler is right, against a number
recorded long before this run.

**The tree's own numbers did not move by a digit** (0.26936 / 0.13074 at K=16, identical to the
policy-A measurement): the tree is not regrown, not modified, and its per-user vectors were
recomputed and asserted **bit-identical** to the banked run, as were the routed row ids. Only the
comparator changed, and the rebuilt comparator is **better** — 0.25900 vs 0.25565 full, 0.12370 vs
0.10982 tail.

### Gains by budget — the tree, before and after the clean comparator

| K | full, **vs rebuilt policy-B** | 95% CI | tail, **vs rebuilt policy-B** | 95% CI | *(superseded: vs policy-A)* |
|---|---|---|---|---|---|
| 1 | **0.00000** | [0, 0] exact | **0.00000** | [0, 0] exact | *+0.00246 / +0.00594* |
| 2 | +0.00528 | [+0.00324, +0.00725] | **−0.00262** | [−0.00481, −0.00051] | *+0.01052 / +0.01273* |
| 4 | +0.00891 | [+0.00632, +0.01147] | +0.00482 | [+0.00214, +0.00756] | *+0.01322 / +0.01494* |
| 8 | +0.01062 | [+0.00775, +0.01354] | +0.00835 | [+0.00543, +0.01130] | *+0.01574 / +0.01544* |
| **16** | **+0.01035** | [+0.00739, +0.01323] | **+0.00705** | [+0.00394, +0.01012] | *+0.01371 / +0.02092* |

**"All ten CIs exclude zero" is RETRACTED.** Against the clean comparator:

- **K=1 is exactly zero by construction** on both axes and both cohorts — that is the contract gate
  passing, not a null result, and it is not a claim about anything.
- **Seven of the eight K≥2 intervals exclude zero in the tree's favour.**
- **One excludes zero AGAINST the tree: tail at K=2, −0.00262 [−0.00481, −0.00051].** With two
  questions the tree is significantly *worse* on the tail metric than the rebuilt static order. On
  validation the same cell is −0.00175 [−0.00402, +0.00052] — same sign, not separating. This is a
  real, reproducible early-budget tail deficit, and it must be reported, not smoothed over.

### The correction is large, and it is not a subtraction

The rebuilt order shares **4 of 16 questions** with the one it replaces (items 19, 22, 83 and concept
173, all at different positions) and **0 of 16 positions**. The masking change moved the argmax at
every position; the old caveat was never a constant "+0.00246 offset". Measured effect of swapping
comparators, on test at K=16: full **+0.01371 → +0.01035** (−25%), tail **+0.02092 → +0.00705**
(−66%). **Two thirds of the reported tail advantage was comparator weakness.**

---

## 2. The headline finding

**Conditioning on cold, closed, charged questions beats a strong static order on held-out users —
from four questions onward, on both axes.**

The tree is grown on validation, frozen, and evaluated on a disjoint cohort. Every one of its
questions is charged and folded — there is no free side information anywhere in it. It is the first
arm in this chapter to clear that bar, and it still clears it against a comparator built by the same
greedy machinery, on the same objective, under the same masking policy, seeded from the tree's own
root heap.

The qualifier is now part of the claim: **at K=2 the tree is behind on tail**, and at K=1 it is
exactly the static order. Conditioning needs a few answers before it pays.

---

## 3. Changing the *bank* buys MORE than conditioning does

Static items-only → static combined is **+0.01544 with no conditioning at all** (0.24356 → 0.25900),
now **larger than the tree's entire conditioning gain (+0.01035)** and nearly double the genre arm's
(+0.00817). Against the superseded comparator these two were a dead heat (+0.01209 vs +0.01371); with
the clean comparator the bank wins outright.

As a share of what the interview buys over asking nothing (items static lifts +0.0850, rebuilt
combined +0.1005):

| change | gain | share of the static's own lift |
|---|---|---|
| items pool → combined pool | +0.01544 | 18.2% |
| genre conditioning (items pool) | +0.00817 | 9.6% |
| tree conditioning (combined pool) | +0.01035 | 10.3% |

**CAVEAT, stated rather than buried: the items-only order is still policy-A-selected**, so the
+0.01544 is an upper bound on the pool effect exactly the way the tree's +0.01371 was. A3 rebuilds it
and this row must be recomputed. The *direction* — a bank change bought at least as much as
conditioning — held under both comparators for the tree, so it is unlikely to reverse, but the
magnitude is not yet clean.

Small effects beside the interview itself either way. **If you can change one thing, change what you
ask about, not how you decide what to ask.** Consistent with the concept-bank result
(`ACT2_CONCEPT_BANK_PV.md`): the bank's value is real and it is static.

---

## 4. Both shrink val→test; tail still shrinks less, but the effect is much weaker

**RETRACTED IN PLACE: "tail does not shrink" was an artifact of the policy-A comparator.** It rested
on a 1.03× ratio measured against an order whose tail was weak (0.10982) because it had been selected
on the wrong ruler. Against the rebuilt comparator (tail 0.12370):

| | validation | test | ratio |
|---|---|---|---|
| tree, full@10, K=16, **vs policy-B** | +0.02794 | +0.01035 | **2.70×** |
| tree, tail@10, K=16, **vs policy-B** | +0.01055 | +0.00705 | **1.50×** |
| *(superseded: full vs policy-A)* | *+0.03091* | *+0.01371* | *2.26×* |
| *(superseded: tail vs policy-A)* | *+0.02161* | *+0.02092* | *1.03×* |
| genre, full@10, K=15 (policy-A comparator) | +0.02046 | +0.00817 | ~2.5× |

The asymmetry **survives in direction** — tail generalises better than full, 1.50× against 2.70× —
but "tail does not shrink" is gone, and with it the strongest version of the niche-items reading.
What can still be said: the full-metric edge is the more heavily validation-fitted half. It is a
weaker claim and should be written as one.

Supporting evidence that held up: the validation overfitting signature **vanished** on test.
Spearman(leaf n, gain) = −0.48 (p=0.071) on validation against the rebuilt comparator, **+0.10
(p=0.71)** on test. Small leaves no longer gain more, which is what a fitted-partition story would
require.

---

## 5. Two fifths of users are hurt

Against the rebuilt comparator, **6 of 15 leaves are negative on test** (was 4 of 15 against the
policy-A order; 0 of 15 on validation), covering **39.3% of the cohort** (3,933 of 10,000):

| leaf | n_test | gain vs **policy-B** | *(vs policy-A)* |
|---|---|---|---|
| `root.noanswer.noanswer` | 1,696 | −0.00086 | *−0.00000* |
| `root.dislike.dislike.dislike.noanswer.dislike` | 745 | −0.00649 | *−0.00921* |
| `root.dislike.dislike.noanswer` | 611 | −0.00010 | *−0.00381* |
| `root.dislike.dislike.dislike.dislike` | 350 | −0.00593 | *+0.00727* |
| `root.like.dislike` | 308 | −0.00868 | *+0.00609* |
| `root.dislike.dislike.dislike.noanswer.noanswer` | 223 | −0.00527 | *−0.01510* |

Two leaves crossed from positive to negative once the comparator was fixed — exactly what a stronger
comparator should do, and a reminder that the old per-leaf table was flattered too, not just the
headline.

This is a result, not a harness fault: a leaf carries its ancestors' questions, which were chosen for
its *parent's* cohort, so the static order is genuinely not inside a leaf's search space and a leaf can
lose.

The claim below is kept but was measured against the superseded comparator and is now weaker: three
of the four then-negative leaves were among the four lowest-absolute-NDCG leaves; Spearman(absolute,
gain) = +0.47 (p=0.079). **The tree helps least, and sometimes hurts, where users are hardest to
serve.** Held at *suggestive* — the fourth-lowest-absolute leaf has the largest positive gain (+0.0584), so
there is no clean rule.

### The reach problem, which conditioning does not solve

The three children of `root.noanswer` have baselines **0.374 / 0.320 / 0.087** against a parent at
0.119. Answering the question *at all* — like or dislike, barely matters which — is worth ~+0.25. The
largest question-*choice* effect anywhere in the tree is +0.048.

1,785 users answered neither of the first two questions, and 1,438 of them cannot answer the third
either. **~14% of the cohort is effectively untouched by the interview**, sitting at 0.087 while
everyone else moves.

---

## 6. What this does to the chapter's hypotheses

**DEAD: "cold closed answers do not differentiate users enough to adapt to."** The tree conditions on
nothing else and wins on held-out users **from K=4 onward on both axes**, against a comparator built
by the same machinery on the same objective under the same masking policy. There is enough to adapt
to. (At K=2 it is behind on tail; at K=1 it is the static order exactly.)

**DEAD, and this is the sharper version: the emulated opener lost to the closed interview.** The genre
split gave the selector one free fact — no turn charged, not folded into the recommender — and bought
+0.00817. Charged closed questions bought +0.01035 against the clean comparator. The margin is now
thin, and **both figures still carry a policy-A items comparator on the genre side** (A3), so this
comparison is not yet settled — but the opener did not beat the thing it was introduced to rescue.

**ALIVE: reach.** Closed questions cannot get anything out of ~14% of users. That is measured, and it
is a different claim from differentiation. **Act 3's motivation should move from "closed questions
can't differentiate" (false) to "closed questions can't reach everyone" (true).**

**The chapter's spine flips from a null to a mechanism.** Twenty of twenty-one published selection
rules optimise a surrogate for the metric they report (`BASELINE_OBJECTIVE_AUDIT.md`), and they lose to
a strong static order. Build an adaptive rule that optimises the reported metric directly, and it wins
on held-out users. **The field's null is about objectives, not about adaptivity.**

---

## 7. Comparator hygiene — DISCHARGED FOR THE TREE, still open for the genre arm

**The tree's caveat is closed (A2, 2026-08-12).** `greedy_static_policyB_combined` rebuilt the
combined static order under policy B — same 10,000 validation users, same 1,000-candidate pool, same
objective, same certified stack, 16 picks, 2,167 cumulative evaluations, resumed from the tree root's
own CELF heap — and both cohorts were re-measured against it. Three gates carried it:

1. **The contract gate.** The rebuilt order's first pick is **concept 114**, the tree's root splitter,
   as it must be (same argmax, same pool, same cohort, same objective). At K=1 the per-user difference
   is **exactly 0.0 for all 10,000 validation and all 10,000 test users, on both axes** — asserted
   with `== 0.0`, not a tolerance. The old +0.00246/+0.00093 K=1 leads are gone, because they were the
   selection-protocol artifact they were suspected of being.
2. **The reuse gate.** The tree's per-user vectors and the policy-A comparator's were recomputed from
   the environment and asserted **bit-identical** to the banked test look, as were the routed row ids
   per leaf. Nothing was carried over on trust.
3. **The banked-headline assert.** Every published policy-A test number (+0.01371 full / +0.02092 tail
   at K=16 and the four other budgets) is reproduced by this run to **0.0e+00** — a free contract check
   on the whole chain, and the same class of check that caught the contamination originally.

**What the caveat was worth, measured rather than derived:** the old wording ("a +0.00246 offset")
was wrong in kind. The rebuilt order shares 4 of 16 questions and 0 of 16 positions with the one it
replaces; the masking change moved the argmax everywhere, so every per-user difference the bootstrap
resamples is different. On test at K=16 the tree's lead falls from +0.01371 to **+0.01035** on full
and from +0.02092 to **+0.00705** on tail — a quarter of the full advantage and **two thirds of the
tail advantage** were comparator weakness.

**Still open:**
- **The genre arm's items-only comparator has NOT been rebuilt** (A3). Every genre number in §1 and §3,
  and the items→combined bank comparison, is still against a policy-A-selected order and is an upper
  bound.
- Budgets differ across arms (15 vs 16). The 16th question is worth +0.0010 to +0.0037 with one leaf
  negative, so that difference is cosmetic (A5).
- The one test shot so far included a mid-run 16th-step extension and now a comparator swap, so it is
  **exploratory** in the ledger's sense; a clean confirmatory evaluation is A8.

---

## 8. The tree, for reference

Depth **5**, 15 leaves, 22 nodes, grown on 10,000 validation users, combined pool (500 popularity-ranked
items + 500 answerable concepts in one argmax), min-leaf 200 as the only stopping rule, 16 charged
questions total — splitters plus a greedy tail in each leaf.

Root splitter `concept 114`, answer rate 0.742, splitting 2,606 / 4,814 / 2,580. For contrast, the best
*item* split (item 83) gives 2,099 / 856 / **7,045** — item splits dump 70% of the cohort into one
branch, concept splits do not.

**The partition generalises even where the orders do not.** All 15 leaf sizes are within ±10% val→test
and every leaf clears the 200 floor on test (smallest 223). Whatever fails here, it is not the routing.

**No channel dominance anywhere**: 5 concepts / 4 items at depth 2; 8 concepts / 7 items among the
16th questions. Two channel patterns were proposed off partial data during the run and both were
retracted — see `PAPER_B_TODO.md` 4.5.

**Depth is capped by arithmetic, not by conditioning running out.** A ternary split above a 200-user
floor needs n ≥ 600 to be possible at all. This must not be read as "conditioning stops paying" —
see `PAPER_B_TODO.md` 1.18.

---

## 9. The rebuilt comparator, for reference

`greedy_static_policyB_combined` — the static order every tree number above is measured against.
Built by the same `greedy_certified` machinery on the same objective (cohort-mean held-out full
NDCG@10 over all 10,000 validation users), the same 1,000-candidate pool (500 popularity-ranked items
+ 500 answerable concepts, items first), under policy B, resumed from the tree root's completed CELF
heap. 16 picks, 2,167 cumulative evaluations (670 new beyond the root's seed pass), 104.6 min.

| # | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| question | c114 | c173 | i19 | c748 | c785 | i83 | i583 | c243 | c231 | c809 | c455 | i22 | i77 | i88 | c18 | c838 |
| gain | .0205 | .0163 | .0115 | .0099 | .0070 | .0052 | .0054 | .0037 | .0034 | .0029 | .0027 | .0024 | .0024 | .0021 | .0020 | .0015 |

Validation curve ends at 0.25561 full / 0.12366 tail; on test the same order gives **0.25900 /
0.12370**. Channel mix **6 items / 10 concepts** (the policy-A order it replaces was 7 items / 9
concepts). Overlap with the superseded order: **4 of 16 questions** (i19, i22, i83, c173), **0 of 16
positions**.

The front of the order is concept-heavy (c114, c173, then one item) and the item block arrives late —
the same answerability-orders-the-channels pattern the policy-A order showed, so that reading is not
an artifact of the retired mask.
