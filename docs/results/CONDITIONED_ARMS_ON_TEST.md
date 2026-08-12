# Conditioned selection on held-out users: the tree, the genre split, and the static orders

Written 2026-08-12. **REVISED 2026-08-12 17:45 (A2, the tree) and again 2026-08-12 21:50 (A3/A4, the
single-channel orders). §7's contamination caveat is now DISCHARGED FOR EVERY ARM IN THIS FILE** —
the items-only and concepts-only orders have been rebuilt under policy B, the genre arm has been
re-measured against the rebuilt items order, and the bank decomposition is now at equal budget with
both sides policy-B-selected. Every number below is on the **10,000 canonical TEST users** unless marked
otherwise, scored with the certified pair (`t2i26c_ep31.pt` + `concept_cv3_ep08.pt`), NO-FLOOR,
policy-B candidate pool, learned decoder bias, behavioural answers. Paired percentile bootstrap over
per-user differences, 10,000 resamples, seed 98765 — never the unpaired form.

Sources: `experiments/battery/ndcgtree16test_policyB.json` (the current tree numbers),
`greedy_static_policyB_combined.json` / `_items.json` / `_concepts.json` (the three rebuilt static
orders), `channel_orders_policyB_test.json` (the three orders on test + the bank decomposition),
`groupgreedy_genre15_TEST_policyB.json` (the genre arm against the rebuilt comparator),
`groupgreedy_genre15_VAL_policyB.json` (its validation side, re-differenced onto the same ruler),
`ndcgtree16.json` (validation), `ndcgtree16test.json` and `groupgreedy_genre15_TEST.json` (the
superseded policy-A measurements, kept for the before/after), `ndcgtree.json`,
`docs/results/MASK_POLICY_B_MIGRATION.md`.

---

## 1. The table

| arm | pool | Q | full@10 | tail@10 | gain vs its own static |
|---|---|---|---|---|---|
| **ask nothing (m=0)** | — | 0 | 0.15855 | 0.02819 | — |
| static items-only (policy-A selected, superseded) | items | 15 | 0.24356 | 0.09689 | — |
| **static items-only, REBUILT under policy B (A3)** | items | 15 | **0.24295** | **0.10041** | — |
| **static items-only, REBUILT under policy B (A3)** | items | 16 | **0.24470** | **0.10171** | — |
| **static concepts-only, REBUILT under policy B (A4)** | concepts | 16 | **0.24467** | **0.11607** | — |
| static combined (policy-A selected, superseded) | combined | 16 | 0.25565 | 0.10982 | — |
| **static combined, REBUILT under policy B (A1)** | combined | 16 | **0.25900** | **0.12370** | — |
| genre split, fallback-200 | items | 15 | 0.25157\* | 0.10632\* | **+0.00862 / +0.00591** |
| genre split, all-groups | items | 15 | 0.25310\* | 0.10760\* | **+0.01015 / +0.00719** |
| **NDCG tree** | combined | 16 | **0.26936** | **0.13074** | **+0.01035 / +0.00705** |

\* derived as comparator + difference; the genre artifact stores differences and the comparator's absolute.
**All three static orders are now policy-B-selected**, so every gain column compares an arm against a
comparator chosen under the arm's own masking protocol.

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

## 3. Changing the *bank* buys MORE than conditioning does — now measured cleanly

**CLEAN NUMBER (A3/A4, `channel_orders_policyB_test.json`).** Both sides policy-B-selected, both at
**16 questions**, paired over the same 10,000 test users:

> **static items-only → static combined = +0.01430 full [+0.01124, +0.01739] and +0.02199 tail
> [+0.01872, +0.02516], with no conditioning at all** (0.24470 → 0.25900 full, 0.10171 → 0.12370 tail).

The published +0.01544 was two defects in one figure: an items order selected under policy A, and a
15-vs-16 budget mismatch. Fixing both moves it to +0.01430. For continuity, the unequal-budget form
on clean orders (items 15 vs combined 16) is +0.01605 — i.e. roughly +0.0018 of the old figure was
the free 16th question, and the rest was comparator selection.

**The conclusion survives the fix.** +0.01430 is still larger than the tree's entire conditioning
gain (+0.01035) and larger than the genre arm's (+0.00862). Against the superseded comparator the
tree and the bank were a dead heat (+0.01209 vs +0.01371); on clean orders the bank wins outright,
and on tail it is not close (+0.02199 vs +0.00705).

As a share of what the interview buys over asking nothing (rebuilt items static lifts +0.08615 full
at 16, rebuilt combined +0.10045):

| change | gain (full) | share of the static's own lift |
|---|---|---|
| items pool → combined pool, equal budget | **+0.01430** | 16.6% |
| genre conditioning (items pool, K=15) | +0.00862 | 10.2% |
| tree conditioning (combined pool, K=16) | +0.01035 | 10.3% |

Small effects beside the interview itself either way. **If you can change one thing, change what you
ask about, not how you decide what to ask.** Consistent with the concept-bank result
(`ACT2_CONCEPT_BANK_PV.md`): the bank's value is real and it is static.

### 3a. Which single channel is the better bank — Act 4's comparison on rebuilt orders

On test, at 16 questions, all three orders policy-B-selected:

| contrast @K=16 | full | 95% CI | tail | 95% CI |
|---|---|---|---|---|
| concepts − items | **−0.00003** | [−0.00315, +0.00310] | **+0.01436** | [+0.01093, +0.01776] |
| combined − items | +0.01430 | [+0.01124, +0.01739] | +0.02199 | [+0.01872, +0.02516] |
| combined − concepts | +0.01434 | [+0.01161, +0.01699] | +0.00763 | [+0.00466, +0.01063] |

**On full, items and concepts are a dead heat at 16** — the interval is symmetric about zero and
0.0003 wide of it. The chapter's current sentence (concepts .2473 / .1154 vs items .2449 / .0967,
concepts the strongest single channel) was measured on validation with policy-A-selected orders; on
test with rebuilt orders the honest statement is **concepts win the tail decisively and tie on full**,
and the combined bank beats both on both axes.

The channel ordering *does* depend on budget, and that is the answerability story, not a wobble:
concepts lead on full at every early budget (+0.00481 at K=1, +0.00781 at K=2, +0.01035 at K=4,
+0.01116 at K=8, all excluding zero) and the items order only catches up by 16. Cold users can answer
concepts; the item bank needs a longer interview to find questions they have seen.

At K=1 and K=2 the combined order IS the concepts order (its first two picks are c114, c173), so
those two contrast cells are exactly 0.0 by construction — a nesting check, not a result.

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
| genre, full@10, K=15, **both sides vs the rebuilt policy-B comparator** | **+0.01831** | **+0.01015** | **1.80×** |
| genre, tail@10, K=15, **both sides vs the rebuilt policy-B comparator** | **+0.00730** | **+0.00719** | **1.02×** |
| *(superseded: genre full, mixed rulers)* | *+0.02046* | *+0.00817* | *~2.5×* |

The genre row needed its own correction beyond the comparator swap: the stage-2 re-run's `shrinkage`
column subtracts a validation gain measured against the **policy-A** order from a test gain measured
against the **rebuilt** one. `groupgreedy_genre15_VAL_policyB.json` re-differences the validation side
onto the rebuilt comparator (stage 1's own banked per-user vectors, nothing re-searched; the policy-A
order re-scored as a free contract check and bit-identical in all 90 group × budget cells). On one
ruler the genre arm shrinks **1.80×** on full, not 2.5× — and its **tail barely shrinks at all
(1.02×)**, which is the same asymmetry the tree shows, in a stronger form, on a different partition.

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
**+0.00862** against its own rebuilt comparator. Charged closed questions bought +0.01035 against
theirs. **Both sides are now on policy-B-selected comparators (A3), so the comparison is settled as
far as comparator hygiene goes**: the free opener did not beat the thing it was introduced to rescue,
though the margin is thin (+0.00173) and the two arms sit on different pools and budgets, so this is
a direction, not a ranking.

**ALIVE: reach.** Closed questions cannot get anything out of ~14% of users. That is measured, and it
is a different claim from differentiation. **Act 3's motivation should move from "closed questions
can't differentiate" (false) to "closed questions can't reach everyone" (true).**

**The chapter's spine flips from a null to a mechanism.** Twenty of twenty-one published selection
rules optimise a surrogate for the metric they report (`BASELINE_OBJECTIVE_AUDIT.md`), and they lose to
a strong static order. Build an adaptive rule that optimises the reported metric directly, and it wins
on held-out users. **The field's null is about objectives, not about adaptivity.**

---

## 7. Comparator hygiene — DISCHARGED FOR EVERY ARM IN THIS FILE

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

**The genre arm's caveat is closed too (A3, 2026-08-12 19:25).** `greedy_static_policyB_items` rebuilt
the items-only order under policy B (16 picks, 839 evaluations, 52.7 min) and the genre arm was
re-measured against it as a **stage-2 re-run only** — the 17 group orders, the grouping rule, the
budgets, the primary K, the two pre-specified arms, the 200 threshold and the bootstrap seed are the
stage-1 ones, unchanged and not re-searched. What the fix was worth, and it is **not a subtraction and
not one-signed**:

| genre arm, K=15 | full, vs policy-A | full, vs rebuilt | tail, vs policy-A | tail, vs rebuilt |
|---|---|---|---|---|
| FALLBACK-200 | +0.00817 | **+0.00862** [+0.00595,+0.01131] | +0.00891 | **+0.00591** [+0.00326,+0.00862] |
| ALL-GROUPS | +0.00955 | **+0.01015** [+0.00733,+0.01296] | +0.01071 | **+0.00719** [+0.00443,+0.01002] |

Both primary intervals still exclude zero. The reason full goes *up* while tail goes *down* is that
the rebuilt items order is not uniformly stronger on the reporting cohort: it beat the policy-A order
by +0.00328 full on the **selection** cohort, but on **test** at K=15 it scores 0.24295 / 0.10041
against 0.24356 / 0.09689 — marginally weaker on full, clearly stronger on tail. A rebuilt comparator
is a *correctly selected* comparator, not necessarily a better-scoring one.

Two further changes to what the genre arm licenses, both of which must travel with it:
- **K=2 and K=4 now separate** on both axes and both arms; they were nulls. "Null at K=1, 2, 4, only
  separating at K=8" is retracted.
- **K=8 tail stops separating** (+0.00934 → +0.00203 [−0.00073, +0.00482]).
- K=1 is unchanged to the bit (+0.00144 all-groups): both comparators open with item 83, so the
  per-user vectors are identical (max abs diff 0.0). Unlike the tree, the genre orders are **not**
  nested in the comparator at the first question, so K=1 is a small real difference, not a gate.
- Drama — 55% of the cohort — is still a null, but its point estimate flips sign, −0.00111 →
  **+0.00150** [−0.00187, +0.00486].

**Still open:**
- Budgets differ across arms (15 vs 16). The 16th question is worth +0.0010 to +0.0037 with one leaf
  negative, so that difference is cosmetic (A5). The rebuilt items order now has a 16th pick banked
  (i135, +0.00294 on validation), so A5 is cheap.
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

---

## 10. The two single-channel orders, for reference (A3 / A4)

Same machinery, same objective, same 10,000 validation users, same certified stack, policy B, 16
picks. Both reuse the tree root's completed 1,000-candidate CELF seed pass — a seed gain does not
depend on the rest of the pool — transplanted into a sidecar signed by the current v2 signature and
verified three ways before use (banked signature fields; `base` recomputed and equal exactly; a
sample of candidates re-evaluated from scratch and exact; plus pick #1's gain re-derived afterwards).

**`greedy_static_policyB_items`** — 839 cumulative evaluations (339 new beyond the seed), **52.7 min**.

| # | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| item | 83 | 7 | 19 | 631 | 2195 | 140 | 22 | 396 | 494 | 343 | 873 | 506 | 212 | 549 | 228 | 135 |
| gain | .0174 | .0140 | .0079 | .0059 | .0054 | .0046 | .0039 | .0039 | .0038 | .0036 | .0027 | .0023 | .0022 | .0027 | .0028 | .0029 |

Validation at K=16: **0.24281 / 0.10112** (the policy-A order rescored under policy B: 0.23953 /
0.09410). Shares **7 of 16 questions** and **2 of 16 positions** with the order it replaces.

**`greedy_static_policyB_concepts`** — 1,546 cumulative evaluations (1,046 new), **134.3 min**.

| # | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| concept | 114 | 173 | 785 | 312 | 215 | 377 | 124 | 1010 | 233 | 12 | 912 | 294 | 488 | 467 | 753 | 851 |
| gain | .0205 | .0163 | .0108 | .0081 | .0069 | .0054 | .0039 | .0034 | .0031 | .0021 | .0015 | .0010 | .0009 | .0006 | .0006 | .0005 |

Validation at K=16: **0.24250 / 0.11809** (policy-A order rescored: 0.24283 / 0.11576). Shares **3 of
16 questions** and **1 of 16 positions**. Note the sign: the rebuilt concept order is ahead at K=1, 2,
4 and 8 on full and at every budget on tail, but **0.00032 behind at exactly K=16 on full**. Greedy is
prefix-optimal, not optimal at a chosen length, so "rebuilt is better" is not a law — the point of the
rebuild is that the order was *selected under the protocol it is scored under*, not that it wins.

**Cost is a property of the bank, not the budget.** Identical pool size (500) and identical budget:
the item search paid 339 lazy re-evaluations, the concept search 1,046. Concept marginal gains sit
close together, so CELF's upper bounds go stale far more often. Both came in well under the
`PAPER_B_TODO` 1.17 estimates (~2 h items, ~3.5 h concepts) — 52.7 min and 134.3 min.
