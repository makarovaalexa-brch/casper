# E1 — THE PARAMETRIC POLICY: imitating the frozen NDCG tree

**Status: IN PROGRESS.** Steps 0 and 1 are done and reported below with their gates. Steps 2–4
(training, the promotion gates, the scalability demonstration) follow in this same document as they
land. Design sheet: `docs/design/PAPER_B_POLICY_DESIGN.md` — Part D is Fable's architecture review and
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
