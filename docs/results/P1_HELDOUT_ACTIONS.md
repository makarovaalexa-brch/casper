# P1 — GENERALISATION ACROSS QUESTIONS: can the policy emit a question it was never taught?

**Status: COMPLETE, 2026-08-15. Validation only — the 10,000 canonical TEST users were never built,
never loaded, never named. Every artifact stamps `test_touched=false`.**

Design sheet: `docs/design/PAPER_B_POLICY_DESIGN.md` **Part I.1 is governing**. Code:
`src/instrument/policy_p1.py` and the `--heldout_json` path in `src/instrument/policy_net.py`, both
committed before the run (HARD RULE 10). The held-out question set and the k=5 neighbour lists were
computed and **committed as a separate commit before a single gradient step** (`1dc7c95`).

> **AUTHOR SAID (2026-08-15):** *"please run."* And, on why: *"i want to show generalisability working
> here and my recommender support ood questions etc. i need a policy capable of understanding that and
> emitting a questions."*

---

## ★ THE VERDICT, FIRST — THE PRE-REGISTERED FALSIFIER FIRED

The falsifier was: *held-out top-5 emission ≤ what a label-shuffled policy achieves ⇒ the policy
reaches questions by memory, not similarity; the OOD-emission claim fails and is reported failed.*

**It fired, and not marginally.** At the 38,336 validation states whose true label was a held-out
question, the retrained policy emits that question **0.0000 of the time at top-1 and 0.0000 of the
time in its top 5** — as does the label-shuffled floor. The neighbour-credit rate is **0.0000** too:
when it does not emit the question, it does not emit one of the question's five nearest neighbours in
the frozen descriptor space either. And the **dumb comparator beats it**: a fixed popularity ranking
puts the held-out question in its top 5 at **1.45%** of those states, which is 1.45 percentage points
more than the trained policy manages.

> **The claim "the policy emits questions it was never taught" is not available on this evidence, and
> is reported failed.**

Two further readings that are not softeners:

1. **The label-shuffled control does not merely tie — it ranks the held-out question BETTER.** Median
   rank of the held-out question, out of 1,000 candidates: **465 for the trained arm, 338 for the
   shuffled one.** The shuffled arm also scores higher end-to-end (0.23887 vs 0.23185, paired
   difference **−0.00702 [−0.00984, −0.00414]** against the trained arm). A control that beats its
   treatment is the sharpest form this falsifier could have taken.
2. **The removed vocabulary was load-bearing.** Hiding 36 of the 172 label questions moves the policy
   from **0.28355 to 0.23185** — it stops beating the metric-optimised static order and starts losing
   to it by **−0.02376 [−0.02661, −0.02087]**. Similarity did not substitute for supervision at all;
   removing the labels removed the behaviour.

**What this experiment therefore is:** a measurement of how much of the policy's competence lives in
the identity of the questions it was supervised on. The answer is: essentially all of it. **What it
is not:** a clean test of the architecture's ability to rank an unseen question, because the policy
it measures has been damaged in order to create the test. That confound is real, it was the reason
the author asked for a non-crippling version, and P1b (`docs/results/P1B_NEVER_TAUGHT.md`) is that
version. **P1 does not carry the generalisation headline.**

---

## The split, and its composition — printed before the time was spent

**36 of the 172 label questions held out BY IDENTITY** (20.9%), stratified by **channel × label-mass
quartile within channel**, seed 98765. Label mass = the number of TRAIN (user, position) examples
carrying that question as its label; stratifying on it was not cosmetic, because the mass
distribution runs from 2,948 to 140,768 and an unstratified draw would have held out only rare
labels.

| | held out | kept |
|---|---|---|
| questions | **36** (16 items / 20 concepts) | 136 (60 items / 76 concepts) |
| train label mass, quantiles [0/25/50/75/100] | 2,948 / 5,053 / 9,405 / 14,287 / **140,768** | 2,948 / 5,053 / 8,769 / 15,509 / 75,891 |
| train examples affected | **538,620 of 2,252,288 (23.91%) DROPPED** | 1,713,668 kept |
| validation states measured | **38,336 of 160,000** | — |

⚠ **The draw took the tree's ROOT.** Concept 114 carries 140,768 train examples — every user at
position 0 — and the stratified draw put it in the held-out set. So **position 0 carries no
supervision for any user**, and every later state is reached through an opening the policy was never
taught. The root alone contributes 10,000 of the 38,336 measured states, which is why the
per-question table below matters more than the pooled rate, and why a forced-opening diagnostic was
added (§4). This is what the pre-registered seed drew; it was not reselected.

Held-out validation states by position: `[10000, 2580, 0, 1189, 795, 2694, 2311, 2510, 0, 2314, 911,
2441, 3869, 1283, 2482, 2957]`.

**Neighbour lists**, k=5, cosine in the frozen 205-dim descriptor space, computed and committed
before training. **100% of held-out questions have an all-same-channel neighbourhood** — items sit
next to items and concepts next to concepts, which is the descriptor's own structure and is worth
knowing before reading a neighbour-credit number of zero.

**Gates passed before any number was trusted:** canonical m=0 snap `0.15686538118354684` exact;
`greedy_static_policyB_combined` rescored to its banked `0.25561016113602253` exactly; pool identity
against the trajectory bank; `branch_of_per_user == branch_of`; leak check 0/10,000. **And the
instrument check that matters here:** the unmodified E1 checkpoint, run through this same audit,
emits the held-out question at **top-1 1.0000** with median rank 0 — the measurement can see the
signal when the signal is there.

---

## RESULT 1 — emission at the held-out states

38,336 (user, position) states, four arms and the dumb comparator through one accumulator.

| arm | val full @ K=16 | **top-1 emission** | **top-5 emission** | neighbour credit | median rank of the held-out question (of 1,000) | mean rank |
|---|---|---|---|---|---|---|
| **E1, unmodified** *(reference: it was taught them)* | 0.28355 | **1.0000** | 1.0000 | 0.0000 | **0** | 0.0 |
| **P1 main** (dropped from supervision, still emittable) | 0.23185 | **0.0000** | **0.0000** | 0.0000 | **465** | 387.5 |
| **P1 control** (also removed from the training candidate table) | 0.22955 | 0.0000 | 0.0000 | 0.0000 | 553 | 463.9 |
| **label-shuffled floor** (the falsifier's comparator) | 0.23887 | 0.0000 | 0.0000 | 0.0000 | **338** | 391.9 |
| popularity-ranked emitter (the dumb comparator) | — | 0.0000 | **0.0145** | 0.0000 | — | — |

Per-question, the story does not change with question size — the ten largest held-out questions all
sit at 0.0000 top-1 and 0.0000 top-5:

| channel | id | states | top-1 | top-5 | median rank |
|---|---|---|---|---|---|
| concept | 114 (the root) | 10,000 | 0.0000 | 0.0000 | 478 |
| item | 514 | 2,580 | 0.0000 | 0.0000 | 136 |
| item | 8 | 2,130 | 0.0000 | 0.0000 | 118 |
| item | 123 | 1,785 | 0.0000 | 0.0000 | **9** |
| item | 442 | 1,785 | 0.0000 | 0.0000 | 460 |
| concept | 247 | 1,785 | 0.0000 | 0.0000 | 318 |
| concept | 251 | 1,382 | 0.0000 | 0.0000 | 90 |
| concept | 280 | 1,215 | 0.0000 | 0.0000 | 317 |
| item | 463 | 1,085 | 0.0000 | 0.0000 | 44 |
| concept | 801 | 1,010 | 0.0000 | 0.0000 | 65 |

**Exactly one cell in the whole audit is non-zero**: the control arm puts concept 251 in its top 5 at
0.07% of that question's 1,382 states. There is no "small positive" here to write up as a trend.

**What the policy does instead.** All three retrained arms collapse to a single opening question and
never emit any held-out question at all: P1 main opens with concept 15 for all 10,000 users
(**0.0000** of its top-1 picks anywhere are a held-out question), the control with item 19 (0.64% of
its picks are a held-out question), the shuffled arm with concept 343. Removing a question from
supervision removes it from the policy's behaviour outright.

---

## RESULT 2 — the score consequence

| arm | val full @ K=16 | val tail | vs the static order (paired) | vs E1 / the tree (paired) |
|---|---|---|---|---|
| E1, unmodified | 0.28355 | 0.13421 | +0.02794 [+0.02493, +0.03099] | — |
| static order | 0.25561 | 0.12366 | — | — |
| **P1 main** | **0.23185** | 0.10328 | **−0.02376 [−0.02661, −0.02087]** | **−0.05170 [−0.05502, −0.04847]** |
| P1 control | 0.22955 | 0.10289 | −0.02606 [−0.02900, −0.02315] | −0.05400 [−0.05739, −0.05073] |
| label-shuffled floor | 0.23887 | 0.10609 | −0.01675 [−0.01958, −0.01395] | — |

Teacher-forced top-1 agreement with the tree falls from **1.0000 to 0.7604** — and 0.7604 is
*exactly* the share of validation states whose label was not held out (121,664 / 160,000). The
retrained policy is right on essentially 100% of the states it was supervised at and essentially 0%
of the states it was not. There is no partial credit anywhere in the middle.

**P1 main vs P1 control is a null** (+0.00230 [−0.00090, +0.00552]). Whether the held-out questions
were merely unsupervised or actively removed from the training candidate table makes no measurable
difference — which is the control's job and it answers cleanly: **the policy did not "learn to reach
them and then fail to use them"; it never had any pull toward them at all.**

⚠ **The label-shuffled arm scoring ABOVE the trained arm needs saying plainly rather than buried.**
A shuffled-label policy learns the position-wise label marginal and nothing else (E1's own shuffle
control did exactly this, landing at 0.24072). With the root removed, imitating the tree at the
remaining positions no longer composes into a coherent interview — the policy is conditioning on a
first answer it was never taught to elicit — and a marginal-frequency order does better. Both are far
below the static order. Under HARD RULE 6 this is a **weak-method** result about what is left of an
imitation policy when its opening is deleted, not evidence about architectures.

---

## RESULT 3 — the forced-opening diagnostic (post-hoc, declared as such)

Because the draw took the root, one obvious question is how much of the −0.0517 is that one question.
Imposing the tree's root at position 0 and letting the policy run freely from position 1:

| arm | free | with the root imposed | share of the gap to E1 recovered | vs static, with the root imposed | agreement with the tree after the forced opening |
|---|---|---|---|---|---|
| P1 main | 0.23185 | **0.26616** | **66.4%** | **+0.01055 [+0.00760, +0.01354]** | 0.3727 |
| P1 control | 0.22955 | 0.25987 | 56.2% | +0.00426 [+0.00126, +0.00725] | 0.3764 |

**Two thirds of the damage is the root alone**, and with the root handed back the policy beats the
static order again. But it still sits **−0.01739 [−0.01958, −0.01524]** below E1 and agrees with the
tree at only **37%** of the remaining positions, so the other 35 missing questions cost real
behaviour too. This is a mechanism note, not a rescue: it explains the size of the drop, and it does
not move the emission result, which is 0.0000 either way.

---

## WHAT THIS SETTLES

1. **The falsifier fired.** Held-out top-5 emission (0.0000) does not exceed the label-shuffled floor
   (0.0000), the shuffled arm ranks the held-out question *better* (median 338 vs 465), and a fixed
   popularity ranking beats both (top-5 1.45%). **The OOD-emission claim is reported failed.**
2. **Neighbour credit is zero.** The policy does not reach for the removed question's descriptor
   neighbours either, so the failure is not "it found a near-synonym instead" — it is an absence.
3. **The supervised label vocabulary is load-bearing.** 36 of 172 questions are worth −0.0517 of full
   NDCG@10 and the whole of the conditioning advantage over a static order.
4. **Unsupervised and unavailable are the same thing here** (main vs control, a null), so nothing was
   learned-then-unused.
5. **This does not measure the architecture's similarity claim cleanly**, because the policy under
   test was damaged to create the test. That is what P1b was run to fix.

**Nothing about test.** No test user was built, loaded or touched.

---

## Cost — a property of this box

| step | measured |
|---|---|
| split + neighbour lists + gates | 1.1 min |
| retrain, main arm (8 epochs, 1.71M examples) | **62.5 min** |
| retrain, control arm | 61.9 min |
| retrain, label-shuffled floor | 41.6 min |
| emission audit, 4 arms × 38,336 states + popularity | 0.4 min |
| forced-opening diagnostic, 2 arms × full 16-turn rollout | 1.8 min |
| **total** | **≈ 2.8 h** |

The Part I.1 estimate was 2–3 h. It came in at 2.8 h.

## Artifacts

| what | where |
|---|---|
| the pre-registered split, neighbour lists, composition | `experiments/battery/p1_heldout/p1_split.json` (committed before training, `1dc7c95`) |
| emission audit + forced-opening diagnostic | `experiments/battery/p1_heldout/p1_audit.json` |
| per-arm gates + per-user vectors | `experiments/battery/policy_{main,shuffle}_equivariant_both_p1{,ctrl,shuf}_gates.json` (+ `_per_user.npz`) |
| logs | `experiments/battery/p1_heldout/train_p1*.log`, `audit.log` |
| code | `src/instrument/policy_p1.py`, `policy_net.py` (`--heldout_json`, `--heldout_mask_train`) |

**Reproduce:**
`python src/instrument/policy_p1.py --mode split --threads 6 --i_have_author_goahead`, then
`python src/instrument/policy_net.py --arm main --tag p1 --heldout_json experiments/battery/p1_heldout/p1_split.json --threads 6 --i_have_author_goahead`
(and `--tag p1ctrl --heldout_mask_train`, `--arm shuffle --tag p1shuf`), then
`python src/instrument/policy_p1.py --mode audit --ckpts ... --forced_arms p1_main,p1_control --threads 6 --i_have_author_goahead`.
