# E1 — THE PARAMETRIC POLICY: imitating the frozen NDCG tree

**Status: E1 COMPLETE (validation only; no test look taken). 2026-08-13.**

**One-line verdict: every E1 gate passes, and the policy is an *exact* functional replica of the tree
on the validation cohort — top-1 1.0000 at every position, the same 15 realised interviews, the same
gain to 1e-9, zero deviations. E1 buys no gain (it was never meant to); what it buys is the licence for
E2, one measured design consequence for E2's DAgger step, and a demonstrated — and honestly
priced — open-vocabulary head.**

Design sheet: `docs/design/PAPER_B_POLICY_DESIGN.md` — Part D is Fable's architecture review and
**D.6b is the author's binding cohort ruling**; where Part D contradicts Parts A–C, D wins.

> **AUTHOR SAID (2026-08-13):** *"plz start on policy and tell me eta"*
> **AUTHOR SAID (2026-08-13):** *"i want it to scale not only by interview length, but also by vocab,
> so i want to be able to predict out of vocab questions … continuous embedding emitted and snapped to
> closest vocab question is my preferred design"*
> **AUTHOR SAID (2026-08-13):** *"so we train policy on train dataset, but we do not regrow the tree,
> and we select model variant or chkp on val. is that right?"* — confirmed.

**The firewall, restated because it is the thing that makes this legitimate.** The 10,000 canonical
TEST users are never built, never loaded, never touched anywhere in E1 — `policy_traj.py --split`
accepts `train` only and `policy_net.py` has no test code path. **No NDCG is ever measured on a train
user**: the tower was trained on them, so any NDCG there would be memorisation. Train users contribute
interview STATES and the frozen tree's LABELS, both functions of the user's own answers only. Every
reported number comes from the 10,000 VALIDATION users. The tree is never regrown; it is loaded frozen
and only applied.

---

## Step 0 — feasibility, and the answer-distribution problem

### 0.1 The problem, and why it was not optional

Validation users answer only out of their **80% fold-in half** — the answer lookup is built from
`ctx.allb`, every catalogue rating of the user *minus* the 20% held-out target half. Train users have
no such split on disk. Answering them from 100% of their ratings would have given train interviews a
systematically higher answer rate than the deployment distribution, and answerability is the single
most load-bearing property in this chapter (8 of the tree's 15 stops were forced by a did-not-answer
branch; the root's three children sit at 0.374 / 0.320 / 0.087). The policy would have been trained on
states that cannot occur at test time.

**The fix, as briefed:** apply the same per-user 80/20 recipe, same seed 98765, same code path, and
answer only from the 80% half. The 20% is discarded — nothing needs a target on a train user.

### 0.2 G-TS1 — the reproducer is byte-faithful, not merely similar

`liang_split.split_train_test_proportion` draws sequentially from one `RandomState(98765)` over
sorted-userId pandas groups, so a user's draw depends on every earlier user's size. That stream cannot
be vectorised. `train_split.py` reproduces it with a numpy loop over group boundaries and then
**proves** it: rebuilding the VALIDATION split from the raw ratings reproduces the shipped
`validation_tr.csv` (619,066 rows) and `validation_te.csv` (149,764 rows) **exactly, as sets**.

> A faster reimplementation that reproduces the shipped artifact byte-for-byte is the same code path.
> One that merely looks similar is not.

Applied to the train cohort: 10,851,052 liked train interactions → **2,114,192 target-half (19.48%) /
8,736,860 fold-in half**, cached under a contract signature that refuses to resume across a changed
recipe, seed, vocabulary or `train.csv`.

### 0.3 The answer rates match

10,000 train users (uids 0–9,999) against the 10,000 validation users, same pool, same env code path:

| quantity | validation | train | diff | rel |
|---|---|---|---|---|
| interview answer rate (all 16 positions) | 0.57889 | 0.57913 | +0.00024 | **0.04%** |
| mean answered of 16 | 9.2623 | 9.2661 | +0.0038 | 0.04% |
| users answering nothing in 16 turns | 0.0000 | 0.0000 | 0 | — |
| answer rate over the 500 pool ITEMS | 0.11446 | 0.11691 | +0.00245 | 2.14% |
| answer rate over the 500 pool CONCEPTS | 0.79806 | 0.79646 | −0.00161 | 0.20% |
| outcome mix like / dislike / no-answer | .1936 / .3853 / .4211 | .1903 / .3889 / .4209 | — | — |

Refusal threshold was 5% relative on the three answer-rate rows; the largest gap is **2.14%** on the
item channel. **The train cohort carries the deployment answer distribution.**

Two things worth keeping: **nobody answers nothing across 16 questions** (the "~14% answer nothing"
figure is a property of the *root question alone*, not of the interview), and the tree's label channel
mix reproduces from the data at **54.57% concept** on validation — the design sheet's 54.6%.

### 0.4 Contract checks that passed before anything was generated

| gate | result |
|---|---|
| **G-E1e canonical snap** | validation m=0 intercept `full=0.15686538118354684`, tail `0.02747733123666661` — the canonical policy-B value, exact. Also the regression gate on the `genv_latent` extraction (see 1.2). |
| **G-T1 banked partition** | all **7/7** internal tree nodes reproduce their banked ternary `child_sizes` and `answer_rate` **exactly** when the tree is re-applied. This is the frozen tree, applied — not a lookalike. |
| pool identity | the train env's 1,000-candidate pool is **byte-identical** to the validation env's. It is derived from `train.csv` counts and the frozen concept stats, so it cannot move with the cohort. |
| leak | 0/10,000 train users have a held-out target inside the routing lookup; likewise validation. |
| answered-item-is-never-a-target | asserted inside every fold. |

### 0.5 Measured cost, and the one real constraint

| | measured |
|---|---|
| tree traversal + ternary outcomes | **0.02 s / 1,000 users** (i.e. free) |
| the 16 state latents `z_t` per user | **3.53 s / 1,000 users** |
| total | **3.56 s / 1,000 → 0.14 h of folding for all 140,768 train users** |
| wall clock including the 15 per-chunk environment builds | ~45 min |
| `z` cache | 140,768 × 16 × 200 float32 = **1.80 GB** |

⚠ **Disk is the binding constraint on this box, not compute.** `C:` had **3.07 GB free** before the
run and 1.13 GB after the `z` caches (1.80 GB train + 0.13 GB validation) were preallocated. The
generator refuses to start if the cache would not fit with 5% headroom, and writes into ONE
preallocated memmap slice by slice so a kill costs at most the chunk in flight. **If E2 needs a second
state cache of this size, space will have to be freed first.**

---

## Step 1 — trajectory generation

The frozen tree applied to all **140,768** train users: ≤5 branch questions on the ternary rule, then
the reached leaf's banked greedy tail, one training example at **every** position 0…15 including the
empty state. **2,252,288 examples**, uniform over interview lengths, 15 chunks, **31.3 minutes** wall.

| | |
|---|---|
| examples | 2,252,288 = 140,768 × 16 |
| distinct label questions behind them | **172** — the same 172 the tree uses on validation, independently reproduced |
| label channel mix, user-weighted | **54.84% concept** (validation: 54.57%) |
| interview answer rate, user-weighted | **0.58115** (validation 0.57889, +0.39% relative) |
| mean answered of 16 | 9.298 (validation 9.262) |
| per-chunk consistency | every 10,000-user chunk asserted its own pool identity, ternary partition and 16-position completeness before being banked |

### 1.1 Leaf assignment — train vs validation

A different population, so these should be *similar in shape, not identical*. They are.

| leaf | train % | validation % | ratio |
|---|---|---|---|
| root.noanswer.noanswer | 17.67 | 17.85 | 0.99 |
| root.like.noanswer | 12.20 | 12.56 | 0.97 |
| root.like.like | 9.88 | 10.10 | 0.98 |
| **root.dislike.dislike.dislike.like** | **7.87** | **7.02** | **1.12** |
| root.dislike.like | 7.68 | 7.23 | 1.06 |
| root.dislike.dislike.dislike.noanswer.dislike | 7.29 | 7.39 | 0.99 |
| root.dislike.dislike.like | 6.23 | 6.60 | 0.94 |
| root.dislike.dislike.noanswer | 5.71 | 5.56 | 1.03 |
| root.dislike.noanswer | 5.24 | 5.29 | 0.99 |
| root.noanswer.dislike | 4.41 | 4.38 | 1.01 |
| root.noanswer.like | 3.59 | 3.57 | 1.01 |
| root.dislike.dislike.dislike.dislike | 3.54 | 3.55 | 1.00 |
| root.dislike.dislike.dislike.noanswer.like | 3.36 | 3.45 | 0.97 |
| root.like.dislike | 3.24 | 3.40 | 0.95 |
| root.dislike.dislike.dislike.noanswer.noanswer | 2.09 | 2.05 | 1.02 |

Eleven of fifteen leaves land within 5% relative of their validation share. The largest divergence is
**+12% on `root.dislike.dislike.dislike.like`** (7.87% vs 7.02%), with `root.dislike.like` at +6%. Both
sit on the dislike spine, and the drift is *toward* the deeper dislike branches — i.e. train users are
marginally more likely to dislike concept 114 and then like item 83's branch. It is a population
difference of the size one would expect between two random 10k/140k draws of the same distribution, it
is reported rather than hidden, and it does not move the label set (172 questions either way).

### 1.2 One code change worth naming

`genv_latent` — the per-user tower+concept latent — was **extracted out of `genv_eval`**, which now
calls it, rather than reimplemented next to it. The policy's state input and the scored interview are
therefore the same object and cannot drift. The extraction is regression-gated by the canonical m=0
snap (exact) and by the banked-headline assert on `greedy_static_policyB_combined`, which rescores on
validation to `full=0.25561016113602253 / tail=0.12366492106552922` — the banked values, exactly.

---

## Steps 2–3 — training and the E1 gates

### 2.1 What was fitted, and on what

| | |
|---|---|
| head | the **query-vector (dot-product) head** — the state encoder emits a continuous vector, every candidate has a precomputed lifted descriptor, the score is an inner product, and `argmax` over the vocabulary **is** the exact nearest-neighbour snap (design sheet D.1, the author's design) |
| trainer | **softmax cross-entropy over the 1,000-candidate pool**, never regress-to-an-embedding-then-snap (D.2: k=1 snap aliasing and the DKQS unroll collapse are *measured* failures of that trainer) |
| state | tower latent `z_t` + **explicit ternary answer history** (three-way one-hot per asked question, a distinct "asked, no answer" outcome, per-outcome counts, `n_asked`) + position / remaining budget |
| descriptors | frozen: decoder row + log-popularity for an item, cv3 embedding + answer rate + polarity for a concept, plus a 2-dim channel one-hot; **per-channel lift** learned jointly (D.3(a)) |
| parameters | 526,542 |
| fit cohort | **140,768 TRAIN users**, 2,252,288 examples. No NDCG measured there, ever. |
| selection cohort | **10,000 VALIDATION users**, by running the policy's OWN interviews and scoring real full/tail NDCG@10 — learned decoder bias, NO-FLOOR, full 18,359-item catalogue, policy-B mask, the existing scoring funnel, no second scorer |
| epochs | 8; best epoch **2**; 92.8 min wall |

The training curve, for the record — convergence is at epoch 2 and the one wobble is an optimiser
spike that val selection correctly rejected:

| epoch | train loss | train top-1 | val top-1 (teacher-forced) | val full | distinct interviews |
|---|---|---|---|---|---|
| 1 | 0.1785 | 0.9675 | 0.9998 | 0.28350 | 36 |
| **2** | **0.0002** | **1.0000** | **1.0000** | **0.28355** | **15** |
| 3 | 0.0000 | 1.0000 | 1.0000 | 0.28355 | 15 |
| 4 | 0.1300 | 0.9815 | 0.9993 | 0.28346 | 71 |
| 5 | 0.0007 | 0.9999 | 1.0000 | 0.28355 | 16 |
| 6–8 | 0.0000 | 1.0000 | 1.0000 | 0.28355 | 15 |

### 2.2 The gates

| gate | criterion | result | verdict |
|---|---|---|---|
| **G-E1e** canonical snap | m=0 validation intercept = 0.15686538 exactly | `0.15686538118354684` | ✅ **PASS**, run first, before anything was fitted |
| **G-E1a** fidelity | top-1 agreement with the tree's label on held-out validation states ≥ 50% | **1.0000**, and **1.0000 at every one of the 16 positions** | ✅ **PASS** |
| **G-E1b** gain vs static | K=16 validation gain over `greedy_static_policyB_combined`, paired percentile bootstrap (10,000 resamples, seed 98765) over per-user differences | **full +0.02794 [+0.02493, +0.03099]**, **tail +0.01055 [+0.00749, +0.01373]**, both exclude zero | ✅ **PASS** |
| **G-E1c** anti-collapse | > 3 distinct realised interviews AND Pep divergence ≥ the tree's own on the same cohort | 15 distinct interviews, largest share 17.85%; divergence **0.66370** vs the tree's **0.66370** over 68,571,148 pairs | ✅ **PASS** — but see the caveat below |
| **G-E1d** shuffle control | a label-shuffled policy must NOT clear G-E1b | *(running)* | *(pending)* |
| channel mix | policy vs the tree's 45.4/54.6 user-weighted | **54.5706% concept**, tree **54.5706%** | no channel collapse |

### 2.3 The headline, stated plainly

**The parametric policy is an exact functional replica of the tree on the validation cohort.**

- Top-1 agreement is **1.0000 at every position** — not "≥ 50%", not 49% as in June 2026, but perfect.
- Its own rollout — where it picks its own questions, with no teacher forcing — reproduces **exactly
  the tree's 15 distinct interviews**, and its per-user Pep divergence rate is the tree's to every
  digit because the realised interviews are the same object.
- Its validation gain over the metric-optimised static order is **+0.027939557**; the tree's banked
  validation gain is **+0.027939556**. They agree to 1e-9 — the residual is float32 banked vectors
  against float64 fresh ones.
- **It never deviates from the tree. Not once, on any of the 10,000 validation users, at any of the 16
  positions** (`frac_never_deviates = 1.0`).

So: **the observable state carries the tree's policy completely.** The design sheet's stated worry
(A.5: 2.25M examples behind only 172 distinct label questions, so "fitting that perfectly means
learning to ignore the answers") resolves the other way — the policy *must* read the ternary answers to
identify which leaf a user is in, and it does so without error.

**Three things this result is NOT, and they matter more than the passes:**

1. **It is not a gain.** E1 was never a contribution on its own (design sheet A.0). An exact
   amortisation of the tree scores exactly what the tree scores. The +0.02794 in the table is the
   *tree's* number, re-earned by a different mechanism; it is not evidence that parameters beat a
   partition.
2. **G-E1c passed vacuously.** The anti-collapse gate is a null-detector, and there was nothing to
   detect: the policy's realised interviews *are* the tree's, so its divergence rate is the tree's by
   identity. It rules out the June BC failure (one fixed 17-item list) and the `conlinucb_bs_static`
   failure (1 distinct interview), and it licenses nothing beyond that.
3. **The covariate-shift question is now measured, and the answer is that E1 cannot answer it.** The
   deviation histogram is **10,000 users at "never deviates"**. A perfect imitator never leaves the
   tree's own trajectory manifold, so E1 generates **zero** off-tree states and the per-position
   agreement curve has no decay to show. **This settles a live design question in the E2 direction:
   E2's DAgger step is not optional and it is not a safety net — it is the only mechanism that can
   produce an off-tree state at all.** Running the trained policy to collect the states *it* visits
   would, at this fidelity, collect the tree's states back.

### 2.4 Why the task turned out to be perfectly learnable

Worth naming, because a 1.0000 invites suspicion and the mechanism is not subtle. The label at
position *t* is a deterministic function of the leaf; the leaf is a deterministic function of the first
≤5 ternary answers; and those answers are in the state **explicitly**, as the design sheet insisted
(A.8, "did not answer is a THIRD outcome and must never be encoded as two"). The policy therefore has
to learn a lookup from an explicitly-provided path to one of 15 tails — and 2.25M examples over 172
label questions is far more than enough. The genuine open question — whether the tower latent `z_t`
*alone* carries the branch path, i.e. whether it is a sufficient statistic for question selection — is
the `--state z` ablation, reported below.

### 2.5 G-E1d — the shuffle control (HARD RULE 5)

Same states, same architecture, same optimiser, same budget; **only the target moves**: the label at
each (user, position) is another user's label *at the same position*, so the state→label association is
destroyed while the position's label marginal is preserved. (A global shuffle would also destroy the
marginal, which would hand the control an implausible label distribution and make it easy to beat —
this is the harder, fairer control, and it still fails.)

| | main | shuffle |
|---|---|---|
| train loss | 0.0002 | 2.2268 |
| val top-1 (teacher-forced) | **1.0000** | **0.2356** |
| val full @ K=16 | 0.28355 | 0.24072 |
| **gain vs static, paired CI** | **+0.02794 [+0.02493, +0.03099]** | **−0.01489 [−0.01803, −0.01172]** |
| Pep divergence | 0.66370 (= the tree's) | 0.22992 (tree: 0.66370) |
| channel mix, concept share | 0.5457 (= the tree's) | 0.3901 |

**✅ G-E1d PASSES: the shuffled arm does not clear G-E1b — it lands significantly BELOW the static
order.** Two details worth keeping. Its position-0 agreement is 1.0000 by construction (every user's
first question is concept 114, so permuting within position 0 changes nothing), and its position-1
agreement is 0.4814 — exactly the share of users in `root.dislike`, i.e. it learned the label marginal
and nothing else. That is what "learned nothing about the state" looks like when the marginal is
preserved.

### 2.6 The state-encoding ablation — is the tower latent a sufficient statistic?

Design sheet A.8 owes this: `z` only vs history only vs both. The interesting cell is **`z` only** —
does the recommender's own belief carry what question to ask next, or does the explicit ternary history
carry something `z` structurally discards?

| | `both` (z + explicit ternary history) | **`z` only** |
|---|---|---|
| val top-1 (teacher-forced) | 1.0000 | **0.9951** |
| val full @ K=16 | 0.28355 | 0.28284 |
| gain vs static | +0.02794 [+0.02493, +0.03099] | **+0.02723 [+0.02424, +0.03031]** |
| tail gain | +0.01055 | +0.01019 |
| distinct realised interviews | 15 | 228 |
| users who ever deviate from the tree | **0.00%** | **6.19%** |
| channel mix, concept share | 0.54571 | 0.54567 |

**Reading: the tower latent is very nearly, but not exactly, a sufficient statistic for question
selection.** Dropping the explicit answer history costs **0.49 percentage points of top-1** and about
**3.6% of the conditioning gain** (+0.02723 against +0.02794) — a real but small residual, and both
arms' CIs overlap heavily, so the ablation does **not** separate on NDCG. The honest sentence is *"`z`
alone recovers 99.5% of the tree's decisions and 97% of its gain"*, not *"the history is unnecessary"*
and not *"the history is load-bearing"*.

### 2.7 The like-for-like variant — no position / budget feature (A.4)

Design sheet A.4 flags an asymmetry to declare rather than smooth over: the tree is budget-unaware, and
giving the policy `t` and `16 − t` is a real extra input. Both variants were therefore run.

| | with position/budget | **without (like-for-like)** |
|---|---|---|
| val top-1 | 1.0000 | 0.9998 |
| gain vs static, full | +0.02794 [+0.02493, +0.03099] | **+0.02796 [+0.02495, +0.03101]** |
| gain vs static, tail | +0.01055 | +0.01056 |
| distinct interviews / divergence | 15 / 0.66370 | 38 / 0.66400 |

**The extra input buys nothing measurable** (+0.02796 vs +0.02794, CIs essentially coincident), which
is unsurprising once stated: the explicit history already contains `n_asked`, so the position is
recoverable. **The like-for-like number is the one to quote, and it is the same number.** The
asymmetry named in A.4 is therefore closed, not merely declared.

There is one genuinely useful by-product: **the `z`-only arm is the only arm that leaves the tree's
trajectory manifold at all** (6.19% of users, first deviating anywhere from position 2 onward). If E2
wants off-tree states cheaply before its DAgger loop exists, a deliberately imperfect imitator is a
source of them.

---

## Step 4 — the scalability demonstration, and what it actually showed

Trained on 1,000 candidates; at inference the **full 18,359-item + 1,031-concept catalogue (19,390
candidates)** is scored in one forward pass, **with no retraining**, using the *training* pool's
descriptor standardisation. This is the author's stated motive made measurable, and per D.4/A.11 the
mechanism claim stands independently of whether any gain separates. A 1,000-way head could not run
this experiment at all.

### 4.1 The mechanism works

| | |
|---|---|
| candidates scored at inference | **19,390** (trained on 1,000) |
| questions picked from OUTSIDE the training pool | **29.99%** of all 160,000 asked questions |
| distinct never-trained questions actually asked | **189** |
| users who receive at least one out-of-pool question | **82.31%** |
| out-of-pool rate by position | 0.00 at k=1–3, then 0.08 / 0.26 / 0.27 / 0.20 / 0.37 / 0.22 / 0.49 / 0.47 / 0.46 / 0.43 / 0.44 / 0.55 / **0.57 at k=16** |

The policy really does score and select candidates it never saw a gradient for, and it does so more as
the interview deepens (the early questions, where the tree's signal is strongest, stay in-pool).

### 4.2 And it costs NDCG — here is exactly why

| | 1,000-candidate pool | full catalogue |
|---|---|---|
| val full NDCG@10 @ K=16 | 0.28355 | **0.26321** |
| val tail NDCG@10 @ K=16 | 0.13421 | 0.12181 |
| paired difference (catalogue − pool) | — | **−0.02034 [−0.02236, −0.01837]**, excludes zero |
| interview answer rate | 0.5789 | **0.4139** |
| answer rate of the IN-pool picks | — | 0.5523 |
| **answer rate of the OUT-of-pool picks** | — | **0.0909** |
| concept share of picks | 0.5457 | 0.3793 |

**The mechanism is not mysterious and it is this chapter's own thesis pointed at itself.** The
1,000-candidate pool is the top-500 items *by popularity* and the top-500 concepts *by answer rate*, so
everything outside it is, by construction, less answerable. The out-of-pool picks are answered
**9.1%** of the time against **55.2%** for in-pool picks — the qualitative look confirms it: the most
frequently chosen never-trained questions are items at popularity ranks 752 … 12,852 (*It's Such a
Beautiful Day* (2011), rank 6,971, 2,320 picks; *Everything Will Be OK* (2006), rank 8,332; *Masquerade*
(2012), rank 12,852) and concepts with answer rates 0.27–0.52. A burned turn costs the interview a
question, so an interview that spends 30% of its budget on questions nobody can answer scores less.

**The honest statement, and it is two sentences, not one:**

> The dot-product head generalises **mechanically** to an unseen vocabulary exactly as designed — 19,390
> candidates in one forward pass, 189 never-trained questions actually selected, no retraining. But the
> value it assigns to a never-trained candidate is **not** calibrated: E1's supervision is a one-hot
> over a pool of pre-filtered *answerable* candidates, so nothing in the training signal ever taught the
> policy that an unseen question is usually unanswerable, and opening the vocabulary costs
> **−0.0203 [−0.0224, −0.0184]** of full NDCG@10.

That is a **weak-method** result in the HARD RULE 6 sense, and the fix is legible rather than
speculative: the descriptor already carries popularity and answer rate, so what is missing is a
training signal that *prices* them — E2's per-state measured labels over a wider candidate set, or
simply extending the label pool. It is not evidence that open-vocabulary selection cannot work, and it
must not be written as if it were.

---

## What E1 settles, and what it licenses

**Settled, on 10,000 held-out validation users, every number paired-bootstrapped, every control run:**

1. **The observable interview state carries the tree's policy completely.** Top-1 agreement 1.0000 at
   every position; the policy's own rollout reproduces the tree's 15 interviews and its validation gain
   to 1e-9. The June 2026 failure mode (clone at 14%, collapse to one fixed 17-item list) does not
   recur, and the June calibration bar (49% was the teacher that worked) is cleared by a factor of two.
2. **A single set of 526,542 parameters replaces a 22-node recursive partition with no loss.** That is
   the amortisation claim, and it is now measured rather than argued.
3. **The tower latent alone recovers 99.5% of the decisions and 97% of the gain.** Not a sufficient
   statistic, but close, and the two arms do not separate on NDCG.
4. **The head is vocabulary-agnostic in the way the author asked for** — 19,390 candidates in one
   forward pass, 189 never-trained questions selected — **and the value it puts on an unseen candidate
   is not yet calibrated**, costing −0.0203 full NDCG when the vocabulary is opened.
5. **The shuffle control fails every gate**, so none of the above is an instrument artefact.

**What it does NOT show, stated so nobody later reads it that way:** E1 produces **no gain over the
tree**, and was never designed to (design sheet A.0: "E1 is not a contribution on its own"). It is the
instrument check that licenses E2. The +0.02794 in these tables is the *tree's* number re-earned by a
different mechanism.

**The one design consequence, and it is the most useful thing E1 produced:**

> **A perfect imitator never leaves the teacher's trajectory manifold.** `frac_never_deviates = 1.0000`
> on all 10,000 validation users. So E1 cannot exhibit the Ross & Bagnell covariate shift that D.6b
> named as the known gap — and, more importantly, **running the trained policy to harvest the states it
> visits would harvest the tree's own states back**. E2's DAgger step is therefore not a robustness
> patch bolted onto the label generator: it is the only mechanism in the design that can produce an
> off-tree state at all, and the labeller must be driven to states chosen deliberately (below the
> floor stops, per `TREE_DEPTH_AND_MINLEAF.md`) rather than states the policy happens to wander into.
> A deliberately imperfect imitator — the `z`-only arm, 6.19% deviation — is the cheapest available
> source of genuinely off-tree states if one is wanted before E2's labeller exists.

**Not taken, and not to be taken without a separate decision: the test look.** E1 reports validation
only. Nothing here has touched the 10,000 test users.

## Artifacts

| what | where |
|---|---|
| trajectory bank (2.25M examples, contract-signed manifest) | `experiments/battery/policy_traj/` |
| Step-0 probe | `experiments/battery/policy_traj/step0_probe.json` |
| main arm gates + per-user vectors | `experiments/battery/policy_main_equivariant_both_gates.json` (+ `_per_user.npz`) |
| shuffle control | `experiments/battery/policy_shuffle_equivariant_both_gates.json` |
| `z`-only ablation | `experiments/battery/policy_main_equivariant_z_gates.json` |
| no-position ablation | `experiments/battery/policy_main_equivariant_both_nopos_gates.json` |
| scalability | `experiments/battery/policy_scalability.json` |
| code | `src/instrument/train_split.py`, `policy_traj.py`, `policy_net.py`; `genv_latent` in `strategy_channel_suite.py`; `branch_of_per_user` in `ndcg_tree.py` |
