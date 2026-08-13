# PART F — THE TWO HEADS: outcome `P(o | state, q)` and lift `V(o | state, q)`

**Status: RUNNING (2026-08-13). Numbers below are filled in as each stage lands; every table states
its own cohort. Validation and TRAIN only — no test look, no NDCG on a train user.**

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
