# Conditioned selection on held-out users: the tree, the genre split, and the static orders

Written 2026-08-12. **Provisional — see §7.** Every number below is on the **10,000 canonical TEST
users** unless marked otherwise, scored with the certified pair (`t2i26c_ep31.pt` + `concept_cv3_ep08.pt`),
NO-FLOOR, policy-B candidate pool, learned decoder bias, behavioural answers. Paired percentile
bootstrap over per-user differences, 10,000 resamples, seed 98765 — never the unpaired form.

Sources: `experiments/battery/ndcgtree16test.json`, `groupgreedy_genre15_TEST.json`,
`ndcgtree.json`, `docs/results/MASK_POLICY_B_MIGRATION.md`.

---

## 1. The table

| arm | pool | Q | full@10 | tail@10 | gain vs its own static |
|---|---|---|---|---|---|
| **ask nothing (m=0)** | — | 0 | 0.15855 | 0.02819 | — |
| static items-only | items | 15 | 0.24356 | 0.09689 | — |
| static **combined** | combined | 16 | 0.25565 | 0.10982 | — |
| genre split, fallback-200 | items | 15 | 0.25173\* | 0.10580\* | +0.00817 / +0.00891 |
| genre split, all-groups | items | 15 | 0.25311\* | 0.10760\* | +0.00955 / +0.01071 |
| **NDCG tree** | combined | 16 | **0.26936** | **0.13074** | **+0.01371 / +0.02092** |

\* derived as comparator + difference; the genre artifact stores differences and the comparator's absolute.

The m=0 intercept reproduces `MASK_POLICY_B_MIGRATION.md`'s banked policy-B test figure
(0.15855 / 0.02819) **exactly** — an independent check that the ruler is right, against a number
recorded long before this run.

### Gains by budget — the tree

| K | full | 95% CI | tail | 95% CI |
|---|---|---|---|---|
| 1 | +0.00246 | [+0.00064, +0.00425] | +0.00594 | [+0.00407, +0.00778] |
| 2 | +0.01052 | [+0.00816, +0.01290] | +0.01273 | [+0.01016, +0.01533] |
| 4 | +0.01322 | [+0.01020, +0.01624] | +0.01494 | [+0.01197, +0.01797] |
| 8 | +0.01574 | [+0.01276, +0.01872] | +0.01544 | [+0.01231, +0.01867] |
| **16** | **+0.01371** | [+0.01061, +0.01680] | **+0.02092** | [+0.01761, +0.02427] |

**All ten intervals exclude zero.**

---

## 2. The headline finding

**Conditioning on cold, closed, charged questions beats a strong static order on held-out users.**

The tree is grown on validation, frozen, and evaluated on a disjoint cohort. Every one of its
questions is charged and folded — there is no free side information anywhere in it. It is the first
arm in this chapter to clear that bar.

---

## 3. Changing the *bank* buys as much as conditioning does

Static items-only → static combined is **+0.01209 with no conditioning at all** — larger than the
genre arm's entire conditioning gain (+0.00817) and essentially equal to the tree's (+0.01371).

As a share of what the interview buys over asking nothing (items static lifts +0.0850, combined
+0.0971):

| change | gain | share of the static's own lift |
|---|---|---|
| items pool → combined pool | +0.01209 | 14.2% |
| genre conditioning (items pool) | +0.00817 | 9.6% |
| tree conditioning (combined pool) | +0.01371 | 14.1% |

Three comparable-sized effects, all small beside the interview itself. **If you can change one thing,
change what you ask about, not how you decide what to ask.** This is consistent with the concept-bank
result (`ACT2_CONCEPT_BANK_PV.md`): the bank's value is real and it is static.

Note also that roughly half the tree's advantage over the genre arm is the pool, not the method.

---

## 4. Full shrinks, tail does not

| | validation | test | ratio |
|---|---|---|---|
| tree, full@10, K=16 | +0.03091 | +0.01371 | **2.26×** |
| tree, tail@10, K=16 | +0.02161 | +0.02092 | **1.03×** |
| genre, full@10, K=15 | +0.02046 | +0.00817 | ~2.5× |

**This is the most interesting number in the run.** Tail is the sparser, harder metric — if the gains
were fitted noise, tail should collapse first. It does not move. The reading that fits: conditioning
genuinely helps surface niche items, while much of the full-metric advantage was fitted to validation
users' head-item idiosyncrasies.

Supporting evidence: the validation overfitting signature **vanished** on test. Spearman(leaf n, gain)
= −0.60 (p=0.018) on validation, **−0.05 (p=0.86)** on test.

---

## 5. A third of users are hurt

**4 of 15 leaves are negative on test** (0 of 15 on validation), covering **32.8% of the cohort**:
`root.noanswer.noanswer` (1,696 users, −0.00000), `…noanswer.dislike` (745, −0.00921),
`root.dislike.dislike.noanswer` (611, −0.00381), `…noanswer.noanswer` (223, −0.01510).

This is a result, not a harness fault: a leaf carries its ancestors' questions, which were chosen for
its *parent's* cohort, so the static order is genuinely not inside a leaf's search space and a leaf can
lose.

Three of the four are among the four lowest-absolute-NDCG leaves; Spearman(absolute, gain) = +0.47
(p=0.079). **The tree helps least, and sometimes hurts, exactly where users are hardest to serve.**
Held at *suggestive* — the fourth-lowest-absolute leaf has the largest positive gain (+0.0584), so
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
nothing else and wins on held-out users at every budget on both axes. There is enough to adapt to.

**DEAD, and this is the sharper version: the emulated opener lost to the closed interview.** The genre
split gave the selector one free fact — no turn charged, not folded into the recommender — and bought
+0.00817. Charged closed questions bought +0.01371. The opener underperformed the thing it was
introduced to rescue.

**ALIVE: reach.** Closed questions cannot get anything out of ~14% of users. That is measured, and it
is a different claim from differentiation. **Act 3's motivation should move from "closed questions
can't differentiate" (false) to "closed questions can't reach everyone" (true).**

**The chapter's spine flips from a null to a mechanism.** Twenty of twenty-one published selection
rules optimise a surrogate for the metric they report (`BASELINE_OBJECTIVE_AUDIT.md`), and they lose to
a strong static order. Build an adaptive rule that optimises the reported metric directly, and it wins
on held-out users. **The field's null is about objectives, not about adaptivity.**

---

## 7. Why none of this is publishable yet

**Both conditioning arms are measured against policy-A-selected comparators.** Their orders were
*chosen* by a greedy search running under the old candidate-masking rule and *scored* under the
current one. Every arm is scored on one ruler, so Act 1 is unaffected — but a comparator optimised for
the wrong ruler is weaker than it should be, and both conditioning arms are flattered.

The size of the contamination is measured, not assumed: **at K=1 no conditioning is possible** — every
user gets the same single question from each arm — **yet the tree leads by +0.00246.** That is pure
selection-protocol difference and it sits inside every K≥2 row. So the tree's conditioning contribution
on full is nearer **+0.011** than +0.0137.

Simple subtraction is *not* a valid correction: a better first question changes which users land where,
so its effect compounds through the tree rather than sitting as a constant offset.

**The fix is running.** `greedy_static_policyB_combined` rebuilds the comparator under policy B; then
both validation and test are re-measured against it. When that lands the K=1 difference should be
**exactly zero**, because the tree's root splitter and a policy-B static greedy's first pick are the
same question by construction.

Also outstanding: the genre arm's items-only comparator needs the same rebuild, and budgets differ
across arms (15 vs 16). The 16th question is worth +0.0010 to +0.0037 with one leaf negative, so that
difference is cosmetic.

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
