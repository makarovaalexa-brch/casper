# SNAP-4: Greedy SLIM — native snap, and the asked-item-exclusion ablation

Target: **Claudius Proissl, Amel Vatić, Helmut Waldschmidt, "Greedy SLIM: A SLIM-Based Approach
For Preference Elicitation", arXiv:2406.06061 [cs.IR], 10 Jun 2024.**
**Provenance, stated up front: CoRR only — dblp lists no venue.** Its user study is n=103, with the
authors' own words *"many of them were somehow linked to the authors"*. Cite it as a preprint.

Run 2026-08-09. Script `scripts/baselines/greedyslim_native_snap.py`, tables
`scripts/baselines/greedyslim_ablation_tables.py`.
Artifacts `experiments/baselines/greedyslim/native_ablation_full.{json,log}` +
`native_ablation_full_per_user/` (per-user NDCG vectors for paired differencing).

---

## Why this baseline mattered more than the rest of the queue

It is **the only published protocol sibling**: same dataset (ML-25M), same primary metric
(NDCG@10), and a long-tail definition that is *verbatim* our head mask — *"Following Cremonesi et
al. (2010), we refer to the set of most popular items that represent 33% of the ratings as
short-head items."*

It is **not** on our ruler and its numbers are **not orderable against ours**: 90/10 random user
split (not Liang 80/20 fold-in), explicit 1–5 ratings (we binarise at >3.5), graded gain
`2^r − 1`, budgets starting at k=5, and a candidate rule of its own. The snap therefore happens
**entirely inside their setup**; nothing here may be tabled beside 0.3496/0.2491.

## Step 0 — the published target, with its source  [V]

Table 3, ML-25M, NDCG@10:

| k | ALL QGSLIM | ALL QBandit | TAIL QGSLIM | TAIL QBandit |
|---|---|---|---|---|
| 5 | .3390 | .3382 | .1163 | **.1183** |
| 10 | **.3594** | .3456 | **.1543** | .1465 |
| 15 | **.3709** | .3221 | **.1674** | .1488 |
| 20 | **.3752** | .2916 | **.1779** | .1466 |

(Netflix reverses at k≤10: QBandit .3030/.3251 vs QGSLIM .2859/.3042.)

Protocol quotes used to build the harness, all [V] from the arXiv full text:

* §3.3.2 — *"Let I_Q be the set of items asked by the questionnaire and let I_R be the set of
  recommended items. **We always require I_Q ∩ I_R = ∅ to avoid trivial recommendations.**"*
* §3.3.1 — gain *"**2^{x_ui} − 1**"*; *"The NDCG is the DCG normalized by the **maximum DCG among
  all subsets I ⊆ I_R**"* — **IDCG is taken over the candidate set**, which is what makes the arms
  below commensurable.
* Candidates are **all items with x_ui = 0**; *"we do not apply any kind of item sampling"*.
* ML-25M **unfiltered** — 162,541 users / 59,047 items / 25,000,095 ratings.
* λ₁ = 2¹², λ_F = 2¹⁶. QBandit's latent factor model is **PureSVD**.
* §6.5 — *"The NDCG of Q_Bandit converges or even decreases after ten questions. **We think that
  the main reason for the decreasing NDCG is that the questionnaires are not allowed to recommend
  items that they contain as questions.**"*

Our rebuild reproduces the dataset exactly: 162,541 × 59,047, 25,000,095 ratings, 90/10 split →
146,287 train / **16,254** test users, Cremonesi 33% head = **296 items**.

---

## 1. THE STRUCTURAL POINT THAT DECIDES THE ABLATION

Their candidate rule is already *"all items with x_ui = 0"* — every item the questionnaire
**answered** is out of the pool anyway. So the extra constraint `I_Q ∩ I_R = ∅` bites on exactly
one set: **asked-but-UNANSWERED items**. Under their offline protocol an asked-but-unanswered
item is one the test user has no rating for, hence a **guaranteed non-target** (gain 0). Removing
a guaranteed non-target from I_R cannot lower DCG and cannot change IDCG.

**⇒ The rule §6.5 blames is, by construction, a monotone BENEFIT, not a tax.** This was
pre-registered in the script docstring before the run.

Three evaluation arms, everything else identical (same users, same interviews, same score
vectors — only the candidate mask I_R changes):

| arm | I_R | ground truth |
|---|---|---|
| **C** *theirs* | `{x_ui = 0}` \ I_Q | user's ratings ∩ I_R |
| **B** *exclusion off* | `{x_ui = 0}` | user's ratings ∩ I_R |
| **A** *no self-eating* | I \ (ask-side rated) | a fixed 20% held-out side, never answerable |

Because C and B are produced in one run from one set of interviews and one set of scores, the
comparison is **paired by construction** and differenced per user.

---

## 2. SNAP VERDICT: **PARTIAL — shape and decline magnitude reproduce, level does not**

QBandit, ARM C (= their protocol), 16,254 test users:

| k | ours full | published | Δ | ours tail | published | Δ | answers/user |
|---|---|---|---|---|---|---|---|
| 5 | .3142 | .3382 | −.0240 | .0823 | .1183 | −.0360 | 2.04 |
| 10 | .3035 | .3456 | −.0421 | .0956 | .1465 | −.0509 | 4.09 |
| 15 | .2841 | .3221 | −.0380 | .1055 | .1488 | −.0433 | 6.02 |
| 20 | .2696 | .2916 | −.0220 | .1135 | .1466 | −.0331 | 7.83 |

**What reproduces.** The *shape* on the full axis — rise-then-decline — and, more importantly, the
**magnitude of the decline**: theirs falls .3456 → .2916 = **−.0540** from its peak, ours falls
.3142 → .2696 = **−.0446**. The tail axis rises in both. Our peak sits one budget earlier (k=5
rather than k=10).

**What does not.** Levels are ~.03 low on full and ~.04 low on tail, roughly uniformly. This is a
**recommender gap, not a protocol gap**, and it is diagnosed rather than tuned away:

* **Canary (2,000-user diagnostic):** Most-Popular with **zero questions** already scores
  **.2960 full / .0570 tail** under their metric. Their k=5 QBandit number is only +.042 above a
  no-interview popularity ranking, so the metric scale is right; what we are short of is the
  quality of their latent factor model, not the protocol.
* Our consumer is a rank-50 PureSVD with an empirical Gaussian prior over user factors. The paper
  states only *"PureSVD"* — no rank, no fold-in rule, no prior. Its code is an **OSF view-only
  link**, unversioned, so the consumer cannot be matched exactly.
* Rank was **not** tuned upward to close the gap: measured, higher rank is much *worse*
  (d=200 → .0992 at k=5), because a fold-in from ~2 revealed ratings is under-determined.

**Per the standing rule, we report the miss rather than tune to pass.** Calibration for this
project: Mult-DAE was dropped from Paper A for a −.0221 miss; DRE passed at −.0095; EDDI failed
outright. Greedy SLIM's QBandit arm sits between DRE and Mult-DAE on level, but unlike Mult-DAE it
**reproduces the phenomenon the paper is cited for**. So it may be cited for the *mechanism* and
the *curve shape*, and its absolute numbers may not be re-used as ours.

**QGSLIM (their own static method) was not reached in this session** — see §6. The greedy fit is
implemented and **verified exactly against a brute-force reference** (identical 5-round selection
sequence, weights agreeing to 1.2e-08, reported gain equal to the true SLIM loss drop), but the
item-item gram it needs had not been built when the session closed.

---

## 3. ★ THE ABLATION — outcome (b), and the sign is the OPPOSITE of their diagnosis

ARM C minus ARM B, paired per user, 16,254 users:

**Full axis** — positive means *their exclusion rule HELPS that arm*:

| k | QBandit C | QBandit B | **C − B** | QPop C | QPop B | **C − B** |
|---|---|---|---|---|---|---|
| 5 | .3142 | .2802 | **+.0340** | .2835 | .2096 | **+.0739** |
| 10 | .3035 | .2360 | **+.0675** | .2893 | .2023 | **+.0870** |
| 15 | .2841 | .1888 | **+.0953** | .2914 | .1811 | **+.1103** |
| 20 | .2696 | .1509 | **+.1187** | .2796 | .1417 | **+.1379** |

Paired 95% bootstrap CIs over users — every difference is computed **per user**, from the same
interviews and the same score vectors, with only the mask changing:

| k | QBandit Δfull [95% CI] | QPop Δfull [95% CI] |
|---|---|---|
| 5 | +.0339 [+.0332, +.0346] | +.0739 [+.0727, +.0750] |
| 10 | +.0675 [+.0664, +.0686] | +.0870 [+.0857, +.0883] |
| 15 | +.0953 [+.0939, +.0966] | +.1103 [+.1088, +.1118] |
| 20 | +.1187 [+.1172, +.1202] | +.1379 [+.1362, +.1396] |

**Tail axis — the in-run control, and it earned its keep:**

| k | QBandit tail C / B | QPop tail C / B | users with Δtail ≠ 0 (QBandit / QPop) |
|---|---|---|---|
| 5 | .0823 / .0823 | .0754 / .0754 | 2 / **0** |
| 10 | .0956 / .0956 | .0887 / .0887 | 2 / **0** |
| 15 | .1055 / .1055 | .0938 / .0938 | 1 / **0** |
| 20 | .1135 / .1135 | .0972 / .0972 | 1 / **0** |

The tail metric restricts I_R to long-tail items, so removing an asked item from I_R can only move
the tail score of a user who was asked a **long-tail** item.

* **QPop** asks the 20 most-rated items and all 20 lie inside the 296-item head, so its per-user
  tail difference is *required* to be exactly zero. It is — `max|d| = 0.0e+00` at every budget
  across all 16,036 users with a non-empty tail ground truth. **That is the harness proving it
  differences the right object**, per user rather than on a mean.
* **QBandit** is adaptive, and Thompson sampling occasionally draws a tail item (measured: 0.127%
  of asks at k=5 falling to 0.057% at k=20; 101/129/161/182 users are asked ≥1 tail item at
  k=5/10/15/20). Its non-zero cells are therefore *expected* and must be a subset of those users.
  Observed: 2/2/1/1 — subset holds; the rest were asked a tail item that was neither a target nor
  inside their top 10. The check binds in both directions: a non-zero QPop cell would mean a leak,
  and an all-zero QBandit column would mean the rule was never applied.

**The first version of this check asserted "0.0 everywhere" and reported CONTROL FAILED on
QBandit.** The mean was 0.0000 to four decimals in every one of those cells; only the per-user
check saw the handful of moved users. That is exactly the failure mode the standing rule warns
about — *a matching mean is never a licence to difference* — caught by the control rather than by
a reviewer.

### Interpretation

1. **§6.5's diagnosis is wrong, and wrong in sign.** The rule they blame for QBandit's decline is
   worth **+.034 to +.119 NDCG in its favour**, growing monotonically with budget. Far from
   causing the decline, the exclusion **masks** one: strip it and QBandit falls .2802 → .1509
   (−.129) instead of .3142 → .2696 (−.045). *"The questionnaires are not allowed to recommend
   items that they contain as questions"* is a benefit these questionnaires are collecting, not a
   tax they are paying.

2. **The mechanism is now explicit.** In an offline replay an asked-but-unanswered item is one the
   user never rated, i.e. a guaranteed non-target — and because questionnaires ask *popular* items,
   it is a non-target the recommender ranks very highly. Excluding it frees a top-10 slot for a
   real target. The benefit scales with the number of unanswered asks, which is why it grows with k
   (answer rates are 41% at k=5 falling to 39% at k=20).

3. **On the tail axis the rule does essentially nothing** — exactly 0.0e+00 per user for the
   static popular questionnaire, and it moves 1–2 users in 16,036 for the adaptive one. Yet the
   *published* tail curve for QBandit also stops climbing (.1465 → .1488 → .1466). Whatever
   flattens that curve, it is not `I_Q ∩ I_R = ∅`, and for QPop that is a proof, not an estimate.

4. **Adaptivity, not popularity, carries the decline.** The QPop control asks the 20 most-rated
   items and achieves an answer rate indistinguishable from QBandit's (7.68 vs 7.83 revealed items
   per user at k=20), yet under ARM C it declines only **−.0118** from its peak while adaptive
   QBandit declines **−.0446** — nearly 4×. So the shape is not simply "asks popular items"; the
   adaptive arm is doing something extra that costs it as the budget grows.

---

## 4. ★ ARM A — the actual cause, and the decline disappears

ARM A holds out a fixed 20% of each test user's ratings **before** the interview. That side is
never answerable and is the only ground truth, so no questionnaire can consume its own targets.
Everything else is unchanged.

| k | QBandit full | QBandit tail | QPop full | QPop tail |
|---|---|---|---|---|
| 5 | .1637 | .0315 | .1571 | .0289 |
| 10 | .1761 | .0375 | .1715 | .0348 |
| 15 | .1857 | .0415 | .1811 | .0361 |
| 20 | **.1936** | **.0456** | **.1892** | **.0383** |

**Both questionnaires rise monotonically on both axes, at every budget. The decline is gone —
completely, not partially.**

Side by side, QBandit full NDCG from k=5 to k=20:

| arm | k=5 → k=20 | change from peak |
|---|---|---|
| **C** (their protocol) | .3142 → .2696 | **−.0446 (−14.2%)** |
| **B** (their rule removed) | .2802 → .1509 | **−.1293 (−46.1%)** |
| **A** (fixed held-out ground truth) | .1637 → .1936 | **+.0299 (monotone rise)** |

Absolute levels are not comparable across arms — ARM A masks the whole 80% ask side out of the
pool (~123 items/user) and scores against a 20% target set, so it is a harder and differently
normalised task. **Only the shape in k is comparable, and that is the estimand.**

### The mechanism, now identified

Greedy SLIM gives a test user **no held-out split**: the row is zeroed, the questionnaire reveals
some of that user's real ratings, and the recommender is then scored against *the rest of the same
rating set*. Every question the user can answer therefore **deletes one of that user's own targets
from their own ground truth** — and not a random one: it deletes a popular, highly-rated, easily
predicted item, precisely the kind the recommender would have placed in the top 10. IDCG is
recomputed over what is left, so the task gets strictly harder with every answered question. The
cost compounds with k and falls hardest on the arm best at guessing what this user has seen — the
adaptive one, which is why QBandit declines ~4× faster than QPop at a matched answer rate.

Fix that one thing — a fixed, arm-independent held-out target set — and the curve turns around and
climbs monotonically for both questionnaires.

**So the decline IS a protocol artefact, but of a different rule than the one the authors name.**
It is not `I_Q ∩ I_R = ∅` (§6.5's answer, which we measure to be a large *benefit*); it is the
absence of a held-out split, i.e. the *answered*-item consumption that their evaluation design
forces. This is the pre-registered outcome (b) plus a positive identification of the cause.

---

## 5. Consequence for how we cite this paper

The correction logged in `findings/candidate_masking_protocol.md` — *"Greedy SLIM can no longer be
cited as clean evidence that static beats adaptive"* — **stands, but its stated reason must be
replaced.** That file says the QBandit decline is *"a protocol artefact by its own authors'
diagnosis"*. We have now measured the authors' diagnosis and it is **false, and false in sign**:
removing the rule they blame makes the decline three times worse.

Three sentences that are now defensible, in preference to what we had:

1. *"Greedy SLIM's adaptive-degradation curve replicates in an independent reimplementation on
   their own dataset, split and metric; its published cause does not."*
2. *"The asked-item exclusion the authors blame is worth +.034 to +.119 NDCG **in the
   questionnaire's favour**, growing with budget, and has no effect at all on the long-tail metric
   for a head-only questionnaire."*
3. *"The degradation is produced by evaluating a user against a ground-truth set the questionnaire
   is simultaneously consuming; under a fixed held-out target set both a static and an adaptive
   questionnaire improve monotonically to k=20."*

**Never write "the decline is a protocol artefact of the asked-item exclusion".** Write "the
decline reproduces; the authors' stated cause is measurably not the cause; the operative cause is
the absence of a held-out split."

★ **And this is direct external support for our own standing protocol ruling.** Policy B (mask
every rated fold-in-side item, fixed per user before the interview, arm-independent) is exactly
ARM A. This run is a published-paper case study of what goes wrong without it — a strategy-dependent
ground truth manufacturing a monotone adaptive-degradation curve that vanishes when the ground
truth is fixed. That belongs in the Paper B protocol paragraph.

---

## 6. What is NOT done

* **QGSLIM — their own static method — was not run.** The greedy fit is implemented and **verified
  exactly against a brute-force reference** (identical 5-round selection sequence, weights agreeing
  to 1.2e-08, reported gain equal to the true SLIM loss drop), and the item-item gram it needs
  costs ~16 min. It simply did not fit in the session after the recommender diagnosis and the
  three-arm run. **Launch command (one heavy job at a time; resumes per cell):**

  ```
  python -u scripts/baselines/greedyslim_native_snap.py --run --label native_qgslim \
      --questionnaires qgslim --budgets 5,10,15,20
  ```
  Expect ~16 min gram + 20 greedy rounds + 12 cells at ~4 min. Targets: .3390/.3594/.3709/.3752
  full, .1163/.1543/.1674/.1779 tail. **Note the prediction this run tests:** QGSLIM's recommender
  is `x'W` with only the k selected rows of W non-zero, so a user who answers nothing scores an
  all-zero vector and ranks at chance. With ~19% of users answering nothing at k=5, either their
  answer rate is much higher than ours or they have an unstated fallback.
* **Step P (port to our ruler as a `Selector`) was not reached.** It is gated on the QGSLIM fit
  anyway, since the ported object is the greedy question order.
* The snap verdict is therefore **partial in coverage as well as in level**: the adaptive arm is
  replicated and diagnosed, the static arm is built and unit-verified but unrun.

---

## 7. Reproduction notes, and two environment findings worth keeping

* **The gram makes the greedy fit ~25× cheaper than the paper's own cost.** The authors report
  *"19 minutes per row"* on ML-25M (~6.5 h for 20 rows) because they recompute the residual every
  round. But `c = XᵀX̂ = G − G[:,S]·W[S,:]` with `G = XᵀX` constant, so G is built **once**
  (~16 min) and each round is a rank-|S| correction. Thresholding G at λ₁/2 is **exact, not a
  truncation**: `G ≥ 0`, `W ≥ 0` ⇒ `c ≤ G` elementwise, so `c_ij > λ₁/2 ⇒ G_ij > λ₁/2`.
* **★ numpy environment defect, worth a memory note.** On this box (numpy 1.26.4), **batched
  `np.linalg.inv` on (256,50,50) costs 4.38 s and batched `np.linalg.solve` 3.24 s, while batched
  `np.linalg.cholesky` costs 0.0067 s** — a ~500× pathology in the batched `getrf/getri` path.
  The first version of the bandit interview used `inv` and projected **4.5 hours**; rewritten on
  Cholesky plus triangular solves it takes **3.6 minutes**. Anything in this repo doing batched
  small-matrix inversion is a latent 100× slowdown.
* A second, independent trap in the same code: the empirical prior covariance spans ~10 orders of
  magnitude, so its inverse is full of **denormals**. The fix (whitening to a prior of N(0, I)) is
  an exact change of variables and is required for correctness of scale as well as speed — with a
  mis-scaled `λI` prior the model scored *below* the zero-question popularity floor.
* Persistence per HARD RULE 11: every (arm, questionnaire, k) cell is written the moment it
  completes, per-user vectors to a sibling directory, contract-signed sidecars for the gram and
  the SVD, a unique `--label`, a lock file refusing a second writer, and cell-level resume.
