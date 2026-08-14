# PART F — THE TWO HEADS: outcome `P(o | state, q)` and lift `V(o | state, q)`

**Status: COMPLETE (2026-08-13/14). One head passes, one fails. Validation and TRAIN only — no test
look, no NDCG on a train user, `test_touched=false` on every artifact.**

**One-line verdict: the outcome head works and the entire margin is the interview — AUC 0.98243
against the per-question popularity prior's 0.96944, rising 0.9687 → 0.9856 across the 16
positions while the prior stays flat, with the shuffle control landing exactly on the prior, and
it buys a further +0.00392 [+0.00105, +0.00676] inside a policy. The learned lift head fails its
ρ ≥ +0.40 gate at +0.062 — and is beaten by a parameter-free mean of six gain vectors at +0.4163,
which passes. With that working value half the combined P×V policy improves to −0.02911 and
STILL loses to bank-alone (+0.01280) — and degrading the value estimator from ρ=1.00 down to
ρ=0.50 changes the policy by less than the seed noise, so value-estimator accuracy is NOT the
binding constraint and no learning curve is worth running.**

Design sheet: `docs/design/PAPER_B_POLICY_DESIGN.md` — **Part F is governing**.

> **AUTHOR SAID (2026-08-13):** *"nah, plz train on all and see if it even works. after than we can
> build curves and stuff. how long to train one version?"*
> **AUTHOR SAID (2026-08-13):** *"not following. how you not running 2nd head?"*
> **AUTHOR SAID (2026-08-13):** *"maybe having 2 heads, answerability and ndcg lift would be good?"*

---

## What this run is, and what it deliberately is not

Two heads, both fitted on data that already exists, neither costing a single new NDCG evaluation:

| | what it predicts | fitted on | cost of the labels |
|---|---|---|---|
| **outcome head** | `P(like / dislike / no-answer \| state, question)` | 140,768 TRAIN users × 16 interview positions × **all 19,390 catalogue questions** | **zero** — the outcome is the structural answer rule, evaluated |
| **lift head** | `V(like / dislike / no-answer \| state, question)` | the exact NDCG gains already banked in the tree's CELF heap sidecars | **zero** — already on disk |

Combined, they score a question the way Part F.1 specifies: `Σ_o P(o | q, s) · V(o | q, s)` — the
decomposition of the target quantity itself, with the information-theoretic value of EDDI's identical
decomposition replaced by **measured NDCG**.

**Deferred by the author's instruction, and not built:** the held-out-QUESTION generalisation test and
the 10 / 25 / 50 / 100% sample-efficiency curve of F.5. This run trains on all questions and all train
users. *"train on all and see if it even works"* is the question being answered.

**Three outcomes, never two.** The root's three children sit at 0.374 / 0.320 / 0.087 and ARM N
measured a "no" worth +0.023 at k=2. A binary answered/not encoding throws away the like/dislike
distinction exactly where it pays, and would not match the tree's own ternary branching.

## The firewall

- The 10,000 canonical TEST users are **never built, never loaded, never touched**. Every artifact
  stamps `test_touched=false`; neither file's parser accepts `test`.
- **No NDCG is measured anywhere in the outcome head.** That is what makes fitting on the 140,768
  TRAIN users legitimate: the tower memorised their ratings, so NDCG there would be meaningless — but
  an *observed answer outcome* is not a model output. Train users supply interview STATES and
  OBSERVED OUTCOMES, both functions of the user's own fold-in half.
- The certified pair (`t2i26c_ep31.pt`, `concept_cv3_ep08.pt`) is frozen; the tree is never regrown.
- Every NDCG number in the lift-head half is on the **10,000 validation users**, through the existing
  scoring funnel, after the canonical m=0 snap `0.15686538` reproduced exactly.

## ★ The no-profile guard, and why it is structural rather than declarative

Part F.5 is explicit: **never any profile feature**, because answerability is a structural rule partly
computable from the profile, so a head that sees profile-like input learns an arithmetic identity and
the generalisation win is hollow. The head sees the ternary answer history so far, the tower latent
`z_t`, depth/remaining budget, and the scored question's frozen descriptor. It never sees the user's
rating of the question it is scoring.

Reading the feature list is not proof. The guard asserted in code is:

> **At position 0 the interview has revealed nothing, so the state must be BIT-IDENTICAL across every
> user.** Anything that distinguishes two users at an empty interview is a profile feature.

`guard_no_profile` asserts max |z₀(u) − z₀(u′)| **== 0.0** over 20,000 user-states before a single
gradient step. Any leak — the user's ratings, their popularity profile, their own latent — breaks that
equality exactly.

## Coverage: no sampling anywhere

Every state scores **all 19,390 candidates** (18,359 items + 1,031 concepts) and the loss is taken
over all of them — one matmul, so full coverage is affordable. Per-user outcome labels are stored
**sparsely** (answered items as a CSR, concepts dense) rather than as a 140,768 × 19,390 dense block,
which is the difference between 0.25 GB and 2.7 GB. Nothing is subsampled: not users, not positions,
not candidates. Already-asked candidates are excluded from both the loss and every metric — at most 15
of 19,390 cells — because their outcome is literally in the input and scoring them would be measuring
a copy.

## Where the labels come from, and the gates they passed

`policy_traj.py --mode gen` applies the **frozen** tree to all 140,768 train users and emits, per
chunk: the 16 interview states, and the three-way outcome of every catalogue question.

| gate | what it asserts | result |
|---|---|---|
| **canonical snap** | validation m=0 intercept == `0.15686538` exactly | ✅ `0.15686538118354684` |
| **G-T1** | the re-applied tree reproduces every internal node's banked ternary `child_sizes` and `answer_rate` exactly | ✅ 7/7 |
| **G-O1** | the harvested outcome table reproduces `branch_of` — the tree's own answer rule — exactly, on questions from both channels | ✅ 24 questions × every user, every chunk |
| pool identity | the train env's 1,000-candidate pool is byte-identical to the validation env's | ✅ |
| leak | 0 users have a held-out target inside the routing lookup | ✅ |

---

## RESULT 1 — ★ THE OUTCOME HEAD BEATS THE POPULARITY PRIOR, AND THE MARGIN IS THE INTERVIEW

Held-out slice of TRAIN: **10,000 users never fitted on**, all 16 positions, all 19,390 candidates —
**3,101,200,000 scored (state, question, outcome) cells**, nothing subsampled. All three predictors are
scored cell-for-cell through the same accumulator, so this is a like-for-like comparison, not two
tables placed side by side.

| | accuracy | cross-entropy | **AUC macro** | AUC like | AUC dislike | AUC no-answer | **ECE macro** |
|---|---|---|---|---|---|---|---|
| **outcome head** | 0.98199 | **0.05476** | **0.98243** | 0.97288 | 0.98720 | 0.98720 | 0.00147 |
| per-question popularity prior | 0.97756 | 0.06974 | 0.96944 | 0.95562 | 0.97655 | 0.97616 | 0.00029 |
| global base rate (floor) | 0.96607 | 0.16672 | 0.50000 | 0.50000 | 0.50000 | 0.50000 | 0.00010 |

**The head works.** It beats the state-free per-question prior on every discrimination measure:
AUC macro **+0.0130**, cross-entropy **−0.0150** (a 21% reduction). The global base rate sits at AUC
0.5 by construction and is included only as the floor.

**Read the accuracy column and then ignore it.** The global outcome base rate over the whole
(user × catalogue) grid is **0.00817 like / 0.02587 dislike / 0.96596 no-answer** — over the full
catalogue almost nothing is answerable by any given user, so "predict no-answer always" already scores
96.6%. Accuracy cannot separate these arms and is reported only so nobody quotes it. The comparison
lives on AUC, CE and calibration.

**Calibration (a hard reported requirement, F.1).** Expected calibration error is **0.00147** macro
for the head (like 0.00144 / dislike 0.00086 / no-answer 0.00210). It is *worse* than the popularity
prior's 0.00029 — which is expected, because an empirical frequency table is calibrated by
construction — but both are small in absolute terms. F.1's warning stands and is now quantified: the
head's probabilities are good enough to multiply against a value estimate, and the ~0.0015 miscalibration
is the term that will compound with the lift head's error, not a free pass.

### ★★★ The per-position curve — the direct evidence that the state carries signal

This is the result, more than the headline row. The popularity prior cannot use the interview, so it is
**flat by construction**. The head is given one more answer at every position.

| position (answers heard) | 0 | 1 | 2 | 3 | 4 | 6 | 8 | 11 | 15 |
|---|---|---|---|---|---|---|---|---|---|
| **head, AUC macro** | 0.9687 | 0.9755 | 0.9780 | 0.9800 | 0.9816 | 0.9831 | 0.9844 | 0.9852 | **0.9856** |
| popularity prior, AUC macro | 0.9696 | 0.9695 | 0.9695 | 0.9695 | 0.9695 | 0.9694 | 0.9694 | 0.9694 | 0.9694 |
| **head, cross-entropy** | 0.07071 | 0.06464 | 0.06141 | 0.05878 | 0.05643 | 0.05407 | 0.05176 | 0.05032 | **0.04930** |
| popularity prior, cross-entropy | 0.07004 | 0.06999 | 0.06995 | 0.06991 | 0.06986 | 0.06978 | 0.06970 | 0.06960 | 0.06948 |

**At position 0 the head is very slightly WORSE than the popularity prior** — AUC 0.9687 vs 0.9696,
CE 0.07071 vs 0.07004 — and that is exactly right. At an empty interview there is no state to read, so
the head is reduced to expressing a per-question prior *through a 205-dimensional frozen descriptor*,
while the baseline has a free parameter per question. It loses that contest by a hair. **Then it hears
one answer and overtakes, and it keeps gaining for all sixteen turns.**

The whole margin is therefore attributable to the interview, not to the head having a better prior.
That is the cleanest form the "does the state carry signal" question could have been answered in, and
it comes with its own control built in: the position-0 cell.

**A conservatism worth stating rather than smoothing over:** the baseline gets one free parameter per
question and the head must route the same information through the frozen descriptor. The comparison is
biased *against* the head, and it wins anyway.

### Cost, measured — and no figure here is a property of the method

| stage | measured |
|---|---|
| harvest regeneration (140,768 users × 16 positions + the full outcome table) | **17.8 min** |
| environment, descriptors, split, both baselines | 2.3 min |
| **one training epoch** (2,252,288 states × 19,390 candidates × 3 outcomes = **43.7 billion label terms**) | **118.0 min** |
| held-out evaluation, head only, per epoch | 11.3 min |
| full metric suite (head + both baselines, 3.1 billion cells) | **32.1 min** |
| **one version, end to end** | **≈ 3.0 h** |

⚠ **The cost history is itself a finding, and it is not flattering.** The first working version projected
**42 min/epoch** from a synthetic bench and delivered **104 and rising — 148 min/epoch by step 1700**,
degrading superlinearly at ~49,000 page faults/second. Three causes, all measured, none of them the model:

1. **Batch assembly.** Drawing (user, position) pairs uniformly meant 1,024 *random* row reads per step
   from the 1.8 GB state memmap. Once that working set stopped being resident the run thrashed.
   Batching by **user** — `Z[u]` as one contiguous read, the 19,390-candidate outcome block built once
   per 16 examples instead of 16 times — took batch assembly from **1.9 s to 0.01 s per step**, and the
   degradation stopped: the rerun held 117–118 min/epoch flat from step 100 to step 2044.
2. **★ `torch.argmax(s, 0)` on the (3, 1024, 19390) logits takes 5.5 seconds** — against **0.8 s** for
   `logsumexp` and **0.08 s** for `gather` on the same tensor. It was the largest single cost in the
   step and it computes the **accuracy printout**, not the loss. Three explicit comparisons do it in
   ~0.1 s.
3. **Oversubscription and probable thermal throttling.** The identical `--bench` reported 1.15 s/step at
   18:11 and 2.6–3.2 s/step at 21:06 on the same box. This is an i7-7820HQ — **4 physical cores** — and
   the run requested 8 threads while other work (including the agent session driving it) took ~1.7 cores.

**The durable lesson:** the bench measured the parts that were *written* — the model, the loss — and not
the parts that were *assembled* around them: the batch construction and the diagnostic on the next line.
For the next version: batch by user (done), keep `argmax` off the outcome axis (done), and try
`--threads 4` or `6` rather than 8 on this chip.

### One epoch, and why

Trained for **one epoch**, stopped deliberately. One epoch is 43.7 billion label terms, the training
loss had flattened (0.0913 → 0.0580, with the last 500 of 2,044 steps moving it only 0.0006), and the
held-out numbers above are unambiguous — the head clears the baseline on every discrimination measure
and the per-position curve is monotone. A second epoch was not needed to answer *"does it even work"*.
**That one epoch sufficed is itself informative**: the signal is not scarce. What a second epoch is
still owed for is the F.5 held-out-**question** test, where generalisation, not fit, is the question.

### The shuffle control (HARD RULE 5) — and it lands exactly where theory says it must

Same states, same architecture, same optimiser, **same budget (one epoch, 2,044 identical steps)**;
only the target moves. The outcome labels of user *u* are taken from a **different user**, so the
state→label association is destroyed while **every per-question marginal is preserved**. That makes the
popularity prior the control's exact theoretical ceiling — and it is a *harder, fairer* control than a
global shuffle, which would also wreck the marginal and hand the control an implausible label
distribution to beat.

| | accuracy | CE | **AUC macro** | ECE macro |
|---|---|---|---|---|
| **main head** | 0.98199 | **0.05476** | **0.98243** | 0.00147 |
| **shuffle control** | 0.97743 | 0.07021 | **0.96884** | 0.00192 |
| per-question popularity prior | 0.97756 | 0.06974 | 0.96944 | 0.00029 |
| global base rate | 0.96607 | 0.16672 | 0.50000 | 0.00010 |

**✅ The control passes, and it passes in the most informative way available.** It does not merely fail
to beat the head — **it converges to the popularity prior almost exactly** (AUC 0.96884 vs 0.96944,
CE 0.07021 vs 0.06974, both a hair *below*). A head that can no longer associate a state with an
outcome falls back precisely onto the per-question marginal it can still see, which is what "learned
nothing about the state" looks like when the marginal is preserved.

**And its per-position curve is FLAT**: 0.9687 → 0.9688 across all sixteen positions, against the main
arm's 0.9687 → 0.9856. The main arm's rise is therefore not an artefact of position, of interview
length, or of anything that survives label permutation. **It is the state.**

| position | 0 | 4 | 8 | 15 |
|---|---|---|---|---|
| main head, AUC macro | 0.9687 | 0.9816 | 0.9844 | **0.9856** |
| **shuffle control** | 0.9687 | 0.9688 | 0.9689 | **0.9688** |
| popularity prior | 0.9696 | 0.9695 | 0.9694 | 0.9694 |

⚠ **A wall-clock asymmetry to declare, since it looks like one and is not.** The main arm took 129.3
minutes and the shuffle 48.0 for the *identical* work — one epoch, 2,044 steps, same batch, same
model. The difference is the box, not the arms: the main arm ran while the machine was oversubscribed
and the shuffle ran on a quiet one. **The arms are budget-matched in gradient steps and examples, which
is the matching that matters; they are not matched in wall time and nothing is inferred from it.**
This is also the cleanest available measurement of the throughput term in the cost section above:
**118 vs 43 minutes per epoch for the same computation, a 2.7× swing from load alone.**

---

## RESULT 2 — THE LIFT HEAD: what the banked gains actually contain, and one number that decides it

### 2.1 The composition audit — the finding that reframes the label question

All 22 CELF heap sidecars, audited at each node's own state before anything was fitted:

| | states | level-0 gains | pool coverage | bias |
|---|---|---|---|---|
| **complete vectors** (the 7 internal split nodes) | **7** | **6,988** | 996–1000 of ~997 available | **none** — measured before any CELF pop |
| survivor sets (the 15 leaves) | 15 | 8,737 | 16.0%–84% (mean 58.4%) | **biased LOW** — CELF pops the highest gains first |

**The clean training set is SEVEN STATES.** ⚠ *When this was written I took that to be the binding
constraint on a state-conditioned value model. It is not — §2.5 shows six states suffice for a
parameter-free average to clear the gate, and §4 shows estimator accuracy does not drive the policy
at all. The state count is a fact about the label bank, not the reason anything failed.*

★ **A correction carried in full, because it was asserted before it was checked.** These two families
were briefed — and initially reported by me — as holding *different scalars*: the internal nodes' gains
as the "ternary-conditioned" criterion and the leaves' as "unconditioned greedy". **That is false.**
The tree's own `criterion_note` states the identity and the artifact confirms it: *NDCG@10 is per-user,
so the node-size-weighted mean over the ternary partition is algebraically the node mean.* Re-derived
from `ndcgtree.json` rather than quoted: over the 7 internal nodes,
|size-weighted ternary mean − banked node mean| ≤ **5.6e-17**. Both families hold the **same** scalar —
the plain cohort-mean full NDCG@10 gain — and differ only in **completeness**. The tree's split search
is therefore myopic in Golbandi's sense, which the chapter's F.3 language already covers; it must be
read as covering the teacher too.

### 2.2 Every banked reuse gate passed

| gate | result |
|---|---|
| **G-L0** pool identity | derived contract `nnz=1356475 per_user=135.6475 fp=c282962b…` |
| **G-L1** base equality | **exact `==`, all 22 states** (root `0.15686538118354684`, deepest `0.10435931424782907`) |
| **G-L2** per-candidate byte-identity | **66 candidates re-evaluated from scratch, every one exact (0.0e+00)** |
| **G-L3** partition recompute | `child_sizes` + `answer_rate` exact at every internal node |
| **G-L4** argmax consistency | complete vector's argmax **is** the tree's pick, **7/7**. *Not asserted on survivor sets* — a survivor set cannot support that check, so claiming it would be a fake gate |
| **G-L5** no test | validation only |

### 2.3 The gate result — FAIL, on a stabilised estimate

Pre-registered gate (F.2), fixed before looking: held-out-state Spearman **≥ +0.40**.

**Read the second table, not the first.** The first LOSO run took the last epoch of a constant-LR fit
whose training loss oscillated by nearly an order of magnitude across its final 60 epochs, and whose
**in-sample** Spearman at the root was **−0.015** — a model that cannot fit points it was trained on has
measured nothing. That run is retained only to show what it cost to believe it.

The rerun fixes the estimator rather than the answer: an inner-validation split of the **training**
states' candidates (never the held-out state), cosine LR decay, best-inner-epoch selection instead of
last, LR 3e-4 → 1e-4, **3 seeds per fold reported as a range**, and V predicted as `base(s) + δ` (§2.4).

| held-out state | first run (unstable) | **stable, 3 seeds** | seed range | mean argmax rank |
|---|---|---|---|---|
| root | +0.033 | **+0.112** | [−0.106, +0.230] | 583/1000 |
| root.like | +0.405 | **+0.035** | [−0.045, +0.077] | 907/999 |
| root.dislike | −0.104 | **+0.028** | [−0.019, +0.118] | 650/999 |
| root.noanswer | +0.124 | **+0.039** | [−0.017, +0.069] | 593/999 |
| root.dislike.dislike | +0.168 | **+0.072** | [+0.062, +0.087] | 436/998 |
| root.dislike.dislike.dislike | +0.312 | **+0.101** | [+0.070, +0.122] | 56/997 |
| root.dislike.dislike.dislike.noanswer | +0.183 | **+0.050** | [−0.033, +0.110] | 481/996 |
| **mean** | +0.160 | **+0.0623** | | |

**✗ FAIL: +0.062 against the +0.40 bar** — beside the measured criterion's **+0.784** and the
training-loss surrogate's **−0.149** on the same axis. **Five of the seven seed-ranges straddle zero.**

**The first run's headline fold was noise.** `root.like` read +0.405 unstable and **+0.035** stable;
`root.dislike` read −0.104 and **+0.028**. A single point estimate was never distinguishable from any
other, which is precisely why the range is now reported.

**The sizing datum, on common held-out states** (both arms hold out the same seven complete states, so
the contrast isolates the effect of *adding* training states): 7 → 22 states moves the mean
**+0.0623 → +0.0819**. Tripling the state count — with biased states, but the same scalar — buys about
**+0.020 of Spearman**. To reach the +0.40 gate on that slope you would need far more than the 32 h
sweep buys. *(The first, unstable run reported this delta going the other way, −0.086. That reversal is
itself a demonstration that the unstable estimate could not support any conclusion.)*

### 2.4 ★ WHY it fails — the no-answer branch, diagnosed rather than guessed

The environment supplies a near-analytic constraint the head must satisfy for its output to mean
anything: **an unanswered question folds an UNSEEN/refusal token, so the state barely moves** and
`V(no-answer | q, s)` should sit close to `base(s)` and be **near-flat across q**. That is a property of
the environment, not of the fit — a free correctness test. It matters arithmetically because
**P(no-answer) is 0.966 over the catalogue** (0.45–0.66 within the 1,000-pool), so in `Σ_o P(o)·V(o)`
that branch dominates the product by construction.

| state | base | V(no-answer) mean | vs base | **sd across q** | V(like) | V(dislike) | mean P(no-answer) |
|---|---|---|---|---|---|---|---|
| root | 0.15687 | 0.12865 | −0.02821 | **0.05814** | 0.16871 | 0.16759 | 0.561 |
| root.like | 0.21971 | 0.19210 | −0.02760 | **0.05992** | 0.23475 | 0.21782 | 0.525 |
| root.dislike | 0.18577 | 0.15742 | −0.02835 | **0.05866** | 0.20204 | 0.19170 | 0.513 |
| root.noanswer | 0.11888 | 0.10614 | −0.01274 | **0.05320** | 0.14053 | 0.13251 | 0.658 |
| root.dislike.dislike | 0.21142 | 0.18535 | −0.02607 | **0.05829** | 0.22112 | 0.20942 | 0.494 |
| root.dislike.dislike.dislike | 0.24924 | 0.22424 | −0.02500 | **0.06009** | 0.25049 | 0.23937 | 0.452 |
| root.ddd.noanswer | 0.18309 | 0.16022 | −0.02286 | **0.05985** | 0.17866 | 0.17390 | 0.491 |

**The ordering is right and the magnitudes are wrong.** like > dislike > no-answer at every state, as
the tree's own children require. But:

1. **V(no-answer) is biased low by ~0.023–0.028** where it should be ≈ 0 away from base; and
2. **it varies across questions with sd ≈ 0.058 — roughly three times the entire range of the quantity
   being predicted** (the true gains span about ±0.02). A branch that should be flat is the noisiest
   thing in the model.

And it is what the score is actually reading. Decomposing the composed score at the root against each
factor: `corr(composed, V_no-answer) = **+0.379**`, against **+0.018** for `V_like` and **+0.069** for
`V_dislike`, with `corr(composed, P_no-answer) = −0.161`. **The argmax is driven by the value assigned
to silence.**

**The structural fix this implied was made and it worked, partially.** Predicting `V = base(s) + δ`
instead of free-form removes a known per-state constant (the heap's own `base`, gated exact by G-L1)
from the learning problem, so the seven states teach the *gain* rather than the *level*. Its effect on
the policy is large and in exactly the predicted direction (§3). It did **not** rescue the gate.

**In-sample, the fit is still weak** — ρ = +0.167 / +0.065 / +0.152 / +0.086 / +0.114 / +0.149 / +0.197
on the seven states it was trained on. A model that fits its own training data at ρ ≈ 0.13 has not
learned the function. HARD RULE 6: **this is a weak method, reported as one.**

> ### ★ RETRACTED — and the retraction is the finding
>
> An earlier version of this section concluded that *"the limiting factor is the learning problem — the
> question descriptor plus the interview state, as currently represented, does not resolve measured
> NDCG gain"*, and that seven states were too few. **Both statements are withdrawn. They were never
> tested, because HARD RULE 2 was not applied: the lift head was never compared against the simplest
> possible estimator.** See §2.5 — a parameter-free average clears the gate the learned head fails.

### 2.5 ★★★ THE BASELINE THAT SHOULD HAVE RUN FIRST: a parameter-free mean beats the learned head

Predict a held-out state's entire gain vector with **the plain mean of the other six states' gain
vectors.** No features, no parameters, no learning, and — the part that matters — **no state
information at all**: it does not know which state it is predicting for.

| held-out state | learned lift head (stable, 3 seeds) | **parameter-free mean of the other six** | its argmax rank |
|---|---|---|---|
| root | +0.112 | **+0.4642** | 27/999 |
| root.like | +0.035 | **+0.3939** | 692/999 |
| root.dislike | +0.028 | **+0.6521** | 67/999 |
| root.noanswer | +0.039 | **+0.2380** | 128/999 |
| root.dislike.dislike | +0.072 | **+0.3445** | 152/998 |
| root.dislike.dislike.dislike | +0.101 | **+0.2880** | 48/997 |
| root.dislike.dislike.dislike.noanswer | +0.050 | **+0.5332** | 14/996 |
| **mean** | **+0.0623 — FAIL** | **+0.4163 — ✅ PASSES ρ ≥ +0.40** | |

**The learned, state-conditioned, 527k-parameter head is beaten by an average of six vectors, by a
factor of nearly seven.** What this overturns:

- *"the descriptor and the interview state cannot resolve measured NDCG gain"* — **false.** A trivial
  average resolves it to +0.416, comfortably inside the gate and on the same axis as the measured
  criterion's +0.784.
- *"seven states are not enough"* — **false. Six** suffice for a parameter-free mean to clear the bar.
- The correct statement is the narrow one: **our learned estimator was weak.** The label material was
  always sufficient; the model built on it was not.

★ **And the estimator that wins is STATE-INDEPENDENT.** It ignores the interview entirely and still
beats the state-conditioned model. That is this chapter's own headline reappearing one level down —
**the bank beats conditioning (+0.01430 > +0.01035)** — and now *the mean value of a question beats a
model of how that value depends on the state.* What you ask about matters more than how you decide
what to ask, and it holds for the value estimator as well as for the policy.

**Extraction check, because the trap is real and silent.** Filtering heap entries on `upd == 0` alone
does **not** give the complete level-0 vector: CELF *pops* the candidate it selects, so the pick must be
added back from `seq[0]`/`gains[0]`. Miss it and every training target loses the single highest-gain
candidate at every state, and the root argmax comes out as `[1,994]`. **Our loader adds it back, and
G-L4 is the gate that catches the mistake if it is not** — verified as a number: root vector
**1000/1000**, argmax **[1,114]** at gain **0.02049056**, byte-matching `ndcgtree.json`'s recorded root
splitter and gain.

## RESULT 3 — the combined P×V policy: it still loses to the bank, with the value half FIXED

Comparators in Part F.4's order. **The bar is bank-alone, not the tree:** *a policy that beats the tree
but not the static bank has not earned adaptive language.*

| arm (validation, K=16, full NDCG@10) | full | gain vs combined static, paired CI |
|---|---|---|
| `greedy_static_policyB_combined` (banked, reproduced **exactly**) | 0.25561016113602253 | — |
| `greedy_static_policyB_items` | 0.24280812656671960 | — |
| **★ bank-alone bar** (combined − items) | | **+0.01280 [+0.00972, +0.01580]** |
| the tree (banked validation gain) | | +0.02794 |
| P×V, learned lift head, free-form V | 0.18640 | −0.06921 [−0.07286, −0.06560] |
| P×V, learned lift head, `V = base + δ` | 0.19721 | −0.05840 [−0.06180, −0.05495] |
| **P×V, parameter-free mean value** | **0.22650** | **−0.02911 [−0.03231, −0.02596]** |
| value table alone, **no outcome head** (a static order) | 0.22258 | −0.03303 [−0.03602, −0.03006] |

**Fixing the value half halves the loss — and the policy still loses to the bank.** −0.05840 →
**−0.02911**, against a bank-alone bar of **+0.01280**. So the earlier reading of −0.058 as evidence
about the *estimator* was right, and it was not the whole story: **even with a value estimator that
clears the gate, the two-head policy does not beat a static order chosen on the bank.** That conclusion
is now attributable to the design rather than to a broken fit, which is the first time in this report it
can be.

### ★ What the outcome head contributes inside a policy — measured, paired, and positive

Both arms carry the **identical** value table. The only difference is whether the per-user answer
probability reweights it. So the paired difference isolates the outcome head's contribution:

| | value table + outcome head | value table alone |
|---|---|---|
| full NDCG@10 | **0.22650** | 0.22258 |
| **paired difference** | **+0.00392 [+0.00105, +0.00676], excludes zero** | |
| interview answer rate | 0.5650 | 0.5623 |
| concept share of picks | 0.3931 | 0.5625 |
| **distinct realised interviews** | **9,421** | **1** |

**This is the cleanest positive result for the two-head design in the report.** The value table alone
is a *static order* — literally one interview for all 10,000 users, the `conlinucb_bs_static` failure
mode by construction. Adding the outcome head makes it genuinely adaptive (9,421 distinct interviews)
and buys **+0.00392 full NDCG@10 with a paired CI excluding zero.** Small, real, and the first
demonstration that `P(outcome | state, question)` pays *inside a policy* rather than only on AUC.

For scale, and stated so the number is not oversold: **+0.00392 is under a third of the bank-alone bar
(+0.01280) and about a seventh of the tree's +0.02794.** It is a real contribution to a policy that
still loses.

### Open vocabulary — structurally unavailable to the estimator that works

The mean-vector value is a **lookup table over the 1,000-candidate pool**. It has no value at all for a
question outside it, so the open-vocabulary arm **cannot be run** with it. That is not a null result, it
is a scope statement — and it is the argument *for* a parametric value head reappearing as a limitation
of the estimator that beats it. The open-vocab numbers in this report therefore stand only for the
learned head, where they were **−0.04016** with 100% out-of-pool picks and a 0.6% answer rate.

---

## RESULT 4 — ★★★ THE CEILING EXPERIMENT: value-estimator accuracy is NOT the binding constraint

The ρ ≥ +0.40 gate was set by asking whether an estimator sits nearer the measured criterion (+0.784)
than the training-loss surrogate (−0.149). The parameter-free mean **passes** it at +0.4163 and the
policy built on it **still loses** at −0.02911. **So the gate was too low, and we could state a slope
but not a finish line.** This experiment states the finish line, and it costs nothing new.

**Method.** Degrade the truth and watch the policy. The value vector is perturbed in normal-score space
and mapped back onto `sorted(t)`, so the simulated estimator has **exactly the same multiset of values**
and is wrong only about the ORDER, by a controlled amount — it cannot cheat by shrinking toward a
constant, which would flatten the policy for reasons unrelated to accuracy. The latent correlation is
solved by bisection against the realised Spearman (targets hit within 0.004). **Three noise draws per
level**, the combined P×V policy run on all 10,000 validation users at each.

| value-estimator accuracy (ρ vs the value table) | policy gain vs combined static | seed range | answer rate |
|---|---|---|---|
| ρ = 0.50 | **−0.02512** | [−0.02740, −0.02172] | 0.6188 |
| ρ = 0.60 | **−0.02495** | [−0.02785, −0.02186] | 0.6141 |
| ρ = 0.70 | **−0.02698** | [−0.02924, −0.02516] | 0.6067 |
| ρ = 0.80 | **−0.02398** | [−0.02627, −0.02235] | 0.5956 |
| ρ = 0.90 | **−0.02587** | [−0.02785, −0.02263] | 0.5842 |
| ρ = 0.95 | **−0.02608** | [−0.02864, −0.02221] | 0.5779 |
| **ρ = 1.00** (the undegraded table) | **−0.02911** | — | 0.5650 |
| **★ bank-alone bar** | **+0.01280** | | |
| the tree | +0.02794 | | |

### The curve is FLAT, and the answer is outcome (a)

**Across 19 runs spanning ρ = 0.50 to 1.00 the policy gain moves between −0.02924 and −0.02172 — a
total spread of 0.0075, smaller than the seed-to-seed spread within a single accuracy level.** The
correlation between accuracy and policy score is **−0.18**: if anything, *slightly negative*. The best
point of the entire sweep is still **0.0345 below the bank-alone bar**.

**Doubling the value estimator's rank correlation from 0.5 to 1.0 buys nothing.** So:

- the pre-registered reading *"even ρ=0.95 loses ⇒ the design is dead on its own terms"* — **fires**,
  and more strongly than anticipated, because even **ρ = 1.00** loses by the same margin;
- **no learning curve is worth running.** There is no accuracy target to aim at, because accuracy is
  not what is missing. That decision is now made on evidence rather than on a slope;
- a mean-vector-plus-learned-residual would be **building a better estimator of a quantity that does
  not drive the outcome.** Do not build it.

### ★ Why it is flat — and this is the real finding

The mean gain vector is a **marginal** value: what each question is worth *on its own* at a state.
The policy takes the top-16 by that score. **A one-shot marginal value cannot express redundancy
between questions**, so it will happily ask sixteen near-duplicates of the same highly-valued question.
The greedy static comparator does the one thing this design structurally cannot: it picks a question,
**folds it, re-scores everything, and picks again**, so the second pick is valued *given* the first.

That is why accuracy does not help. Making the marginal values more accurate makes the policy more
confident about a ranking whose top-16 is redundant. **It is visible in the answer-rate column above:
answer rate falls monotonically from 0.6188 at ρ=0.50 to 0.5650 at ρ=1.00** — the more accurate the
estimator, the more the policy concentrates on the same narrow set of questions, and the noise levels
are *helping* only by scattering the picks.

**The consequence for the chapter, stated plainly:** `Σ_o P(o|q,s)·V(o|q,s)` with a **state-marginal**
`V` is not a competitive selector at any estimator quality. To compete it would need `V` conditioned on
*what has already been asked* — which is a state-conditioned value model, i.e. exactly the object the
learned lift head was supposed to be and which lost to a static average. **The two failures compose:
the state-conditioned estimator we could learn was bad, and the static estimator we can build well is
structurally incapable.** That closes the line cleanly, cheaply, and on its own terms.

### The two limitations that outlive this result

1. **★ The estimator that wins cannot do the thing the author wants.** The mean vector is a **lookup
   table over the 1,000-candidate pool**; it has no value at all for a question outside it, so it is
   *structurally incapable of open vocabulary*. The parametric head existed precisely to fix that, and
   it is the one that lost.
2. **★ Every measured gain we hold is for a POOL question**, and that pool is top-500-popular ∪
   top-500-answerable. **No model in this report has ever seen what a *typical* catalogue question is
   worth.** For open vocabulary that sampling bias is plausibly a larger problem than the state count —
   and it is consistent with what the learned head did when the vocabulary was opened (100% out-of-pool
   picks, 0.6% answer rate). Any future open-vocab value model needs gains measured on a *representative*
   sample of the catalogue, not on the answerable head of it.

---


## What this settles, and what it licenses

**Settled, on held-out data, with controls:**

1. **The outcome head works and the interview is why.** AUC macro 0.98243 vs the per-question
   popularity prior's 0.96944 over 3.1 billion scored cells; the per-position curve rises
   0.9687 → 0.9856 while the prior is flat; the head is *behind* at position 0 where there is no state
   to read. **F.5's pre-registered falsifier did not fire.**
2. **The shuffle control converges exactly onto the popularity prior** (0.96884 vs 0.96944) with a flat
   per-position curve, so (1) is not an instrument artefact.
3. **The no-profile guard holds structurally** (position-0 states bit-identical, max |diff| = 0.0).
4. **The outcome head also pays inside a policy: +0.00392 [+0.00105, +0.00676]** over the identical
   value table without it, turning a single static order into 9,421 distinct interviews.
5. **★ The learned lift head is a weak estimator, and we know that because a parameter-free mean of six
   gain vectors beats it +0.4163 vs +0.0623 and clears the gate it fails.** The label material was
   sufficient all along.
6. **★ The winning value estimator is state-independent** — the chapter's "bank beats conditioning"
   finding reappearing inside the value estimator.
7. **Even with that working value half, the P×V policy loses to bank-alone** (−0.02911 vs +0.01280).
8. **★ Value-estimator accuracy is not the binding constraint.** Over 19 runs spanning ρ = 0.50 to
   1.00 the policy gain moves within 0.0075 — less than the seed spread — with correlation **−0.18**
   between accuracy and score, and the best point still 0.0345 below bank-alone. A **state-marginal**
   value cannot express redundancy between questions, which is the one thing the greedy comparator
   does by re-folding after every pick. **The line closes on its own terms.**

**What it does NOT settle:**

- **Not** that a value head cannot be learned. It says *ours* was worse than an average — a statement
  about our model, not about the problem. **But Result 4 removes the reason to build a better one:**
  a mean-vector-plus-learned-residual would be a better estimator of a quantity that does not drive
  the outcome. Do not build it.
- **Nothing about test.** No test user was built, loaded or touched at any point.

**Two things to carry forward, both cheap and both caught something here:**

> **(i) HARD RULE 2 applies to estimators, not just to arms.** The lift head ran for hours against no
> baseline simpler than itself, and the conclusions drawn from it — "the state cannot resolve the gain",
> "seven states are not enough" — were both false and both retracted. A parameter-free average takes
> milliseconds and would have preceded every fit.
>
> **(ii) The no-answer flatness check.** `V(no-answer | q, s)` must sit at `base(s)` and stay flat across
> `q`, because an unanswered question folds an UNSEEN/refusal token. It is free, it is a property of the
> environment rather than of the fit, and it localised this failure to the exact branch that
> `P(no-answer) ≈ 0.966` makes dominant.

---


## Artifacts

| what | where |
|---|---|
| trajectory + outcome-label bank (2.25M states, full catalogue outcomes, contract-signed manifest) | `experiments/battery/policy_traj/` (`meta_*.npz`, `ans_*.npz`, `z_train.npy`, `manifest.json`) |
| outcome head — main arm, gates and full metric suite | `experiments/battery/outcome_head_main_both.json` (+ `.pt`, `_history.json`, `_pop.npz`) |
| outcome head — shuffle control | `experiments/battery/outcome_head_shuffle_both.json` (+ `.pt`) |
| lift head + combined policy — **stable, citable** | `experiments/battery/lift_head_stable.json` |
| lift head — first run, unstable, retained for the comparison only | `experiments/battery/lift_head.json` |
| **parameter-free mean value: LOSO gate, the P×V rerun, and the ceiling sweep** | `experiments/battery/meanvec_value.json` (+ `_per_user.npz`), `meanvec.log` |
| logs | `experiments/battery/outcome_main.log`, `outcome_main_evalonly.log`, `outcome_shuffle.log`, `lift_head_stable.log`, `policy_traj/gen.out` |
| code | `src/instrument/outcome_head.py`, `src/instrument/lift_head.py`; harvest extension in `src/instrument/policy_traj.py` (`outcome_labels`, `gate_outcome_labels`) |
| run chain | `experiments/battery/run_partF.sh`, `experiments/battery/run_partF2.ps1` |

**Reproduce:**

```
python src/instrument/policy_traj.py --mode gen --full_threads
python src/instrument/outcome_head.py --arm main    --epochs 1 --batch 1024 --d_k 64 --threads 8 --full_threads --i_have_author_goahead
python src/instrument/outcome_head.py --arm shuffle --epochs 1 --batch 1024 --d_k 64 --threads 8 --full_threads --i_have_author_goahead
python src/instrument/lift_head.py --label _stable --lift_seeds 3 --lift_epochs 60 --lift_lr 1e-4 --threads 8 --full_threads --i_have_author_goahead
python src/instrument/lift_head.py --meanvec --threads 8 --full_threads --i_have_author_goahead
```

`--selftest` on either head runs pure-tensor checks with no data and no fit: the sparse-to-dense label
reconstruction, the exact per-question priors, the hand-written masked cross-entropy against the library
form, the binned AUC against the exact rank AUC, and the user-block batching (coverage of every
(user, position) pair exactly once, the de-duplicated asked-cell index list, and `block_ce == masked_ce`).

## What is still owed

1. **F.5's held-out-QUESTION test and the 10/25/50/100% sample-efficiency curve** — deferred by the
   author's instruction this round. The outcome head is the arm to run them on; `--prior_feat` must stay
   **off** there, since a per-question prior does not exist for a never-seen question.
2. **A second outcome-head epoch**, if the held-out-question test needs it. One epoch sufficed here and
   that is itself informative.
3. **The next value model, whatever it is, must pass the no-answer flatness test in §2.4 before its
   Spearman is believed.** It is free and it caught this one.
4. **Any open-vocabulary value model needs gains measured on a REPRESENTATIVE sample of the
   catalogue.** Every gain we hold is for a pool question, and the pool is top-500-popular ∪
   top-500-answerable — no model here has ever seen what a typical catalogue question is worth.
5. **A selector that scores questions independently is owed a redundancy mechanism** (re-folding,
   or a value conditioned on what has already been asked). Result 4 says that, not estimator
   quality, is what separates this design from the greedy comparator.

---

## Appendix — a blocker worth recording

The harvest died twice with **exit 1 and no traceback**. Cause: `train_tower_t2.load_recvae_teacher`
hard-codes `.cache/baselines/recvae_ml25m_liang.pt`, and yesterday's disk-recovery commit (`49b3586`)
deleted it as an MD5-verified duplicate of `models/baselines_ml25m_liang/recvae_ml25m_liang.pt`. It is
a live dependency of the certified tower's build path, not a dead copy. `raise SystemExit` prints only
its message, which is why it read as a benign warning twice before being diagnosed. Restored as a
**hard link** (zero additional disk), MD5 re-verified equal on both paths;
`models/DELETED_CACHE_CHECKPOINTS.md` corrected.

**The lesson, and it generalises:** byte-identity to a surviving copy proves a file is redundant as
*data*, not that its *path* is unreferenced. Grep for the path before deleting a "duplicate".
