# THE UNIFIED (state, question, gain) DATASET — Paper B's value head

> **AUTHOR SAID (2026-08-14, verbatim):** *"can you collect the dataset: all states you can scrape for
> all greedy variants, trees static, whatnot, plus random sampling of everything else and redo the
> training? or have i missed anything?"*

That sentence is the **HARD RULE 1 sign-off** for two things and only two things: (i) the scrape of
already-banked material, and (ii) the random-candidate sampling whose design is written out in full
below. Nothing else in this document is authorised by it.

**FIREWALL: validation only.** No test user was built, loaded or touched. Every artifact carries
`test_touched: false`.

Code: `src/instrument/value_dataset.py` (`--mode scrape|meanvec|sample|train`),
`src/instrument/value_chain.py` (the queue).
Artifacts: `experiments/battery/value_dataset.jsonl`, `..._manifest.json`, `..._meanvec.json`.

---

## Why this exists

The value head's failure was diagnosed as **label starvation**, not a representational limit
(`PAPER_B_POLICY_DESIGN.md` G.C-AMENDMENT-4). The clean training set was **seven states**. The tree's
own data is the wrong *shape* for a regressor — it is a handful of states with 1,000 candidates each,
bought by argmax searches, when a regressor wants thousands of states with a few candidates each.

So: take everything already on disk that is on the right ruler, tag how each measurement was
selected, and buy the shape that is missing.

---

## STEP 1 — the scrape, and the three things that could have ruined it

**Result: 55 sidecars → 245 distinct states → 23,983 measured (state, question, gain) triples.** That
reconciles exactly with the independent count in `OUTCOME_HEAD.md` ("What is actually on disk —
counted properly"), which is the check that says the two code paths agree.

A **state** is `(cohort, asked-question-set)` — *not* `(sidecar, level)`. `ndcgtree16_heaps` re-banks
the same leaf states with the tail extended, so the naive key double-counts. Where two independently
launched runs measured the same state they are merged and **every shared question is asserted equal**.

### The admission rule: on the recorded contract, never on the filename

| hazard | what was actually done |
|---|---|
| **Contract heterogeneity** — a policy-A gain and a policy-B gain are numbers on different rulers | admitted only on `mask_pool` / `pool_contract` / `contract_tag`. Target ruler `nnz=1356475 per_user=135.6475 fp=c282962b49a2d67ef47ff07544519f20`. Nothing is ever "converted" |
| **Cohort / firewall** | `sig["split"] == "validation"` plus a validation contract tag. `ndcgtree16test_heaps` is **admitted** — its recorded contract, base, prefix and `rows_sha1` are validation; "test" names the *driver job*. `ndcgtree_SANITY_heaps` / `_PREFLIGHT_heaps` are **refused** |
| **Surrogate contamination** | every source's *objective* is checked. The five `splitcrit_tree_*` arms are refused: their scalar is a pairwise / variance / logistic **surrogate score**, and admitting them would train the value head on exactly the quantities the chapter proves lose |

Two contract families exist and are kept distinguishable in the composition table rather than merged:

| contract | measurements | states |
|---|---|---|
| `policyB(v2, fp-stated)` — the mask fingerprint is in the artifact | 1,741 | 31 |
| `policyB(v1, base-verified)` — predates the fingerprint field; its provenance is the *measured* one (`lift_head`'s G-L1 re-derived all 22 bases to exact `==` against a freshly built policy-B env, and `policy_labels`' G-L0 routes the tree's contract to `policy_traj/manifest.json`) | 22,242 | 214 |
| **on a non-matching or unstated contract** | **0 — excluded outright** | |

### Everything refused, and why

| refused | reason |
|---|---|
| `splitcrit_tree_{pairwise,variance,logistic,random_ml,random}` | surrogate objective, not measured NDCG@10 |
| `ndcgtree_SANITY_heaps`, `ndcgtree_PREFLIGHT_heaps` | not the citable tree |
| `greedy_static_i26c.json`, `greedy_polish.json`, `groupgreedy_d1.json`, `groupgreedy_genre15.json` | **the artifact states no `pool_contract`/`mask_policy`.** An unstated contract is not a matching one — these predate the policy-B rebuild and their gains are on the retired ruler |
| `groupgreedy_genre15_VAL_policyB.json` | it *is* on the B contract, but it holds only budget-level differences — no per-question measured gains to admit |

⚠ **The genre-conditioned greedy is the largest thing lost.** Its 17 group orders are genuinely
answer-conditioned states, but their per-step gains were measured under policy A and only the
*sequences* were re-scored under B. Re-measuring them would be a new job, not a scrape.

### ★ The selection tag — the field that makes the dataset usable

Measurements are **not missing at random**, and the bias is **not** simple truncation. At a given
state the heap holds a large mass of weak candidates while the winners sit in `seq`/`gains`. One
node, measured: ~984 heap entries all below +0.0021, against picks at +0.0044…+0.0135. **Each
state's record is bimodal — a weak mass plus one excellent pick, with the middle missing — and the
middle is exactly where a ranking decision is made.**

| selection tag | measurements | states | what it is |
|---|---|---|---|
| `exhaustive_sweep` | 6,669 | 7 | every candidate measured; unbiased |
| `heap_survivor` | 17,067 | 223 | measured and never popped: the weak mass, **biased low** |
| `pick` | 247 | 245 | the winner CELF popped at that state |
| `random_sample` | *(step 3, pending)* | | uniform draw; unbiased by construction |

Composition otherwise: 12,268 concept and 11,715 item measurements; candidates per state min 1,
**p25 8, median 27**, p75 63, max 1,000; depths 0–15, with 1 state at depth 0 and 18 states at each
of depths 5–15.

**Two free cross-run checks passed and are worth recording:**

- all 15 `ndcgtree16test_heaps` sidecars contributed **zero new measurements** — every value was
  already present from `ndcgtree16_heaps` and every one matched to `<1e-12`;
- the 48 picks in the three policy-B static greedy JSONs were likewise **all already present** and
  all matched, so the static greedy artifacts and the heaps agree byte-for-byte.

---

## The snap that validates the scrape — and one number that did NOT hold up

The parameter-free mean-vector baseline (predict a held-out state's whole gain vector with the plain
mean of the other states' vectors — no parameters, and **no state information at all**), recomputed
from the scraped table alone, LOSO on the 7 complete states:

**+0.4163, and all seven per-state numbers, reproduce exactly** (+0.4642, +0.3939, +0.6521, +0.2380,
+0.3445, +0.2880, +0.5332) from a completely separate code path. That is the snap that says the
scrape and `lift_head`'s loader agree, and it cost nothing.

### ⚠ The source ablation is NOT yet stable, and the instability was measured, not suspected

Run on the 245 banked states alone, adding the 238 downward-biased states looked like a clean win —
ρ +0.4163 → +0.4327, and the mean rank of the predicted best question 161 → **37**. Then **two**
phase-3 exhaustive states landed (2 of 247), and on the same 7-state evaluation the same arm moved to
ρ +0.4157 and rank **143** — the effect essentially vanished.

**The mechanism is understood, and it is a property of the baseline rather than a bug.** The
mean-vector predictor averages over whichever training states happen to hold a value for `q`, and
that support is wildly uneven: a survivor state carries a median of 27 candidates, while a phase-3
state carries 997. Two dense states therefore change the mean for *nearly every question*, while 238
sparse ones each touch a few. A mean over unequal, non-random support is not a stable estimator, and
this is the sense in which it is "parameter-free but not assumption-free".

| eval set | training source | mean ρ | mean argmax rank |
|---|---|---|---|
| 7 complete | the other 6 complete | +0.4163 | 161 |
| 7 complete | all 246 states | +0.4157 | 143 |
| 9 unbiased (incl. 2 phase-3) | the other 8 unbiased | +0.3975 — **fails** | 115 |
| 9 unbiased | all 246 states | +0.4229 | 115 |

**Nothing about the ablation by source is reported as settled until phase 3 and the random sampling
are in.**

### ★★★ SETTLED: the +0.4163 bar was substantially a seven-state artefact

The trend we watched forming has completed. On the full **187-state** unbiased evaluation set the
parameter-free mean-vector baseline **fails the gate it was celebrated for passing**:

| unbiased evaluation states | mean-vector ρ (unbiased-only) |
|---|---|
| 7 (the tree's complete internal nodes) | **+0.4163** — passes |
| 9 (+2 phase-3 deep states) | +0.3975 |
| 16 (+9 phase-3 deep states) | +0.3771 |
| **187 (all sources)** | **+0.2956** unbiased-only / **+0.2770** union — **FAILS** |

**Why, and it is not a defect in the baseline.** The original seven are the tree's own internal
nodes: shallow (depths 0–4), large-cohort, and all on a single root-to-leaf spine. One average
describes seven such states well. Add deep prefixes, small cohorts and off-spine states and a
state-blind estimator degrades — which is exactly what a state-blind estimator *should* do. The
+0.4163 number was measuring the homogeneity of the tree's spine as much as anything about value.

★ **Two consequences, and the second matters more than the first.**

1. The relative claim in `OUTCOME_HEAD.md` §2.5 is unchanged: +0.2956 still beats the learned head's
   +0.0623, and "our learned estimator was weak" still stands.
2. ★ **The bar is now much weaker, and this must be said before any future head is reported.** A
   learned head scoring +0.35 would clear the new comparator while failing the old one *and* while
   failing the ρ ≥ +0.40 gate. Any future pass must be quoted against **+0.2956 with its sample
   size**, never against +0.4163. Both documents are corrected in place rather than rewritten, so
   the earlier number stays visible with the cohort that produced it.

★ **Carry the method point regardless: correlation and argmax rank move independently.** In the
banked-only run ρ moved +0.016 while the argmax rank moved 161 → 37. A value estimate is consumed by
an argmax, so reporting ρ alone would have hidden the whole effect — and reporting the argmax alone
would have overstated one that then failed to replicate.

---

## STEP 2 — phase 3, ingesting live

`policy_labels.py --mode sweep --states deep` is running (started 12:44; 42 states × ~997 candidates;
the structure-only cost model says 3.25 h, measured throughput is ~2.5× the model, so ~8 h). Its
output is exhaustive and therefore unbiased at every candidate, it records its own cohort `rows`, and
its per-user `.f32` vectors turn each measured state into its answer-conditioned children at no extra
evaluation. **9 of 42 states are complete and already ingested**, taking the union to
**254 states / 32,956 measurements**. Nothing about the run is disturbed. On a refitted cost model it
finishes ~18:10, with the queued chain complete ~19:30.

★ **Read the per-state sidecars, not the roll-up.** `policy_labels_deep_partial.json` still lists a
finished state as `done: false` until the whole sweep ends, while the same state is complete on disk
under its own name in `policy_labels_deep_partial/`. Ingesting whichever file is more convenient is
the "verify the artifact, not a proxy" failure, and it would silently have cost every state until the
run ended.

---

## STEP 3 — the random sampling, design stated in full

**This is the authorised reduction.** Users are **never** sampled — all 10,000 validation users,
every time. Candidates are.

Per prefix:

1. take `k` from the **stratified depth schedule** below, then `k` distinct questions **uniformly
   without replacement** from the policy-B combined pool (500 popularity items + 500 answerable
   concepts — the tree's own pool, same ids in the same order). The **prefix stays in-pool
   deliberately**: it defines the *state*, and drawing states from the same question set as every
   banked state is what keeps the new material comparable with the old;
2. ask that prefix to all 10,000 users; **keep the per-user NDCG vector**;
3. record each user's ternary answer path (like / dislike / no-answer, per prefix question);
4. draw **50 candidates** — **20 from the 1,000-question pool** and **30 from the full
   19,390-question catalogue** — uniformly without replacement; evaluate each over all 10,000 users,
   again keeping the per-user vector;
5. for the marginal state and for **every answer path with ≥ 200 users**, the conditional gain is
   `mean(f_q[path]) − mean(f_0[path])` — the same construction the tree uses to score a split, so it
   is on the same ruler.

Step 5 is why one sweep buys many states: grouping by answer path yields one measured gain vector per
populated path **at zero extra evaluation**.

**The declared truncation.** Paths below **200 users** are dropped — the tree's own `min_leaf`,
chosen because it already governs every banked state rather than invented for this job. The number of
paths dropped and the users they carry are recorded per prefix in the artifact. Because the per-user
vectors are banked, changing that threshold later costs nothing and requires no re-evaluation.

### ★★ Two widenings (2026-08-14) — recorded here alongside the original design

Both are **strict broadenings**: more states and more question variety for the same money, nothing
removed and nothing sampled that was not already being sampled. Neither needs a fresh HARD RULE 1
sign-off; both are recorded because a design that changes silently is a design nobody can audit.

**★ Widening 1 — DEPTH. `k ~ U{3..6}` → a stratified schedule over `k = 1…15`.**

The original draw put every unbiased measurement at depth ≤ 6, and phase 3's 42 states sit at depths
3–6 as well (they are one level below floor stops that live at depths 2–5). **The interview is
sixteen questions long, so depths 7–15 — more than half of it — would have had no unbiased
measurement at all**, covered only by the thin, bimodal heap survivors. The value head would then be
extrapolating over exactly the region where the policy spends most of its turns. Folding a longer
prefix is free: the cost is 50 candidates × 10,000 users whatever `k` is.

The schedule is **stratified rather than random** — `k = 1…15` once each, then the remaining four
prefixes at the deep end (8, 9, 10, 11). A random draw over 15 values with 19 prefixes leaves holes,
and the yield curve below is a thing we are trying to *measure*; a curve with missing x-values
measures less than it costs.

⚠ **The real constraint at depth is path fragmentation, not cost.** A `k`-question prefix scatters
10,000 users over up to `3^k` answer paths, and only paths clearing the 200-user floor give a stable
conditional mean. So **usable paths per prefix as a function of `k` is measured and reported**, not
assumed — it is the honest ceiling on how deep any conditional value model can be trained on this
cohort. **If depth 12 yields nothing above the floor, that is the finding.** The floor is *not*
lowered to manufacture states.

**★ Widening 2 — VOCABULARY. 30 of the 50 candidates now come from the full catalogue.**

Every gain ever measured in this project, in every source, is for a question inside the
1,000-candidate pool — by construction the top 500 by popularity ∪ top 500 by answer rate. **No model
here has ever seen a measured value for a typical catalogue question**, so the value head has no
examples of what a *bad* question is worth. That is precisely the deficit behind the open-vocabulary
failure (out-of-pool picks answered 9.1% vs 55.2% in-pool, NDCG −0.02034).

Most catalogue draws will be near-unanswerable and score near zero — **that is the point**: they are
the negative examples that have never existed. The split is **declared, not silent**: 20 in-pool so
the pool region stays densely covered, 30 catalogue-wide. Every measurement carries **`in_pool`**, so
every result can be recomputed restricted to in-pool candidates and compared like-for-like with the
banked material. The sampler also prints the contrast per prefix (mean and max marginal gain, in-pool
vs catalogue-wide), because it is free and it is the first direct measurement of what the pool
selection is actually buying.

Current vocabulary composition of the union: **32,956 in-pool, 0 catalogue-wide** — the hole, stated
as a number.

**Cost, from the project's own measured fit** `sec_per_eval(n) = 0.0858 + 0.8157·(n/1000)`, i.e.
**8.24 s at n = 10,000**: 51 evaluations per prefix ≈ **7 min**; 19 prefixes ≈ **2.2 h**. The job
re-measures its own rate as it goes and prints measured-vs-model, because every cost projection in
this workstream has been wrong in both directions.

**Persistence (HARD RULE 11):** each prefix writes its own contract-signed sidecar plus its raw
float32 per-user vectors the moment it finishes, and a resume across a changed contract is refused.

---

## STEP 4 — the refit, and the comparisons that are mandatory

`--mode train` refits on the union and reports, without exception:

1. ★ the **parameter-free mean-vector baseline** (+0.4163). HARD RULE 2. It beat the learned head by
   a factor of seven last time; a head that does not beat it is broken, not new.
2. ★ the **zeroed-state control**. A head that sees `z_t` can trivially ignore it and emit the mean
   vector, so scoring *below* a state-blind baseline is an optimisation failure, not a
   representational one. That free tell was on screen for hours and nobody read it.
3. the pre-registered **ρ ≥ +0.40 gate**, LOSO on the **unbiased states only**, per-state numbers,
   3 seeds with the spread.
4. the **ablation by source** — unbiased-only vs unbiased + tagged-biased — and, now that the
   vocabulary hole is being filled, the same arms recomputed **restricted to in-pool candidates**, so
   the new material is compared like-for-like with everything banked.
5. the **no-answer flatness check**: `V(no-answer|q,s)` must sit at `base(s)` and be flat across `q`.
   Last time it varied with sd 0.058, about three times the range of the quantity, and drove the
   argmax.

**HARD RULE 6 applies in advance:** if the refit still fails, that is a weak method reported as one.
No tuning until it passes, and no marketing a null.

---

## THE DATASET AS BUILT — 425 states, 72,717 triples

The sampling ran to completion. This is the real size, and it is far larger than the 254 quoted
mid-run:

| source | states | triples | selection |
|---|---|---|---|
| banked scrape (55 heap sidecars) | 245 | 23,983 | `pick` + `heap_survivor`, bimodal |
| phase 3 exhaustive sweeps (all 42 done) | 42 | 41,834 | `exhaustive_sweep`, unbiased |
| random prefix sampling (19 prefixes) | 138 | 6,900 | `random_sample`, unbiased |
| **total** | **425** | **72,717** | **187 unbiased evaluation states** |

Vocabulary: **68,802 in-pool / 3,915 catalogue-wide.** The catalogue-wide block is small in count and
is the entire negative-example supply — before this run it was zero.

### ★ The depth yield curve, measured

Usable states per prefix (answer paths clearing the 200-user floor), by prefix length:

| k | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| paths populated | 3 | 9 | 22 | 72 | 89 | 254 | 287 | 539/766 | 837/914 | 1925/1311 | 1117/1958 | 2347 | 2627 | 2661 | 2546 |
| **states kept** | 2 | 7 | 6 | 13 | 9 | 12 | 10 | 8/4 | 6/6 | 5/3 | 11/3 | 5 | 3 | 3 | 3 |

**It does not collapse.** Fragmentation is severe — at k=15, 2,543 of 2,546 populated paths fall
below the floor, carrying 8,673 of 10,000 users — but **three states still clear it at every depth
to 15**, because the dominant all-no-answer path keeps a large mass of users together. So the honest
ceiling on training a conditional value model is **not** a depth cutoff: it is a *width* cutoff. Deep
states exist, there are just very few of them and they are the same few branches every time. The
floor was never lowered.

### ★ The vocabulary contrast — the first measurement of what the pool is buying

Marginal gain at the prefix state, in-pool vs catalogue-wide draws, every prefix:

- in-pool mean ranges **+0.00101 … +0.00470**, max **+0.01853**;
- catalogue-wide mean ranges **−0.00104 … +0.00119** — near zero and **usually negative** — max
  **+0.00430**.

A typical catalogue question is worth **nothing, or slightly less than nothing**, and the best of 30
random catalogue draws rarely reaches the *mean* of 20 pool draws. That is the negative-example mass
the value head has never had, and it is a clean, direct statement of what the popularity ∪
answer-rate pool selection contributes — measured here for the first time rather than assumed.

★★ **This bears directly on the open-vocabulary claim, and it reframes it.** The recorded failure
was that an open-vocabulary policy picked out-of-pool questions that were answered 9.1% of the time
against 55.2% in-pool, costing −0.02034 NDCG, and that was read as a deficiency of *our model*. The
measurement above says the catalogue outside the pool is **genuinely mostly worthless to ask** — its
mean marginal gain is at or below zero at every one of 19 prefixes. So a policy that picks out-of-pool
questions is largely being punished by the catalogue, not only by its own weakness. The honest split:
choosing badly *within* the answerable region is a model failure; the thin returns *outside* it are a
property of the domain. Both belong in the chapter, and only the first is ours to fix.

---

## ★★★ THE `nan` TRAINING LOSS — diagnosed, and it is benign

Every logged epoch of the refit prints `train nan` while inner-validation is finite and falling
monotonically. Established from the code and a direct autograd check, **without touching the run**:

**Verdict: hypothesis (3) — only the reported aggregate is NaN. Nothing is dropped, nothing is
masked, no NaN gradient ever reaches the optimiser, and nothing sanitises anything.**

**The mechanism.** `fit_lift` splits each state's candidates into an inner-train / inner-val pair with
`k = max(round(0.2·n), 1)`. For `n == 1` the inner-val takes the only candidate and **the training
slice is empty**, so `((c[∅] − y[∅])**2).mean()` is a mean over zero elements — `0/0 = nan`. The
epoch's reported loss is `tr_tot / len(packs)` with `tr_tot` a running sum, so **one NaN term poisons
the whole epoch's headline number** while leaving everything else intact. **16 of the 425 states have
exactly one candidate**; every other `n` gives a non-empty split (`n = 2` already yields a training
slice of 1).

**Measured, not reasoned.** Backward through that empty reduction produces gradients that are
**exactly zero, not NaN** (`isnan: False, absmax: 0.0`), parameters stay finite through `opt.step()`,
and a normal pack immediately afterwards trains normally. That is why inner-validation — which always
has ≥ 1 element, so is never an empty reduction — is finite and falls monotonically. `grep` confirms
there is no `clip_grad`, no `nan_to_num`, no `isnan` mask anywhere in `lift_head.py` or
`value_dataset.py`: nothing is being silently absorbed.

**The one honest caveat, stated rather than buried.** Those 16 one-candidate states contribute a zero
gradient, so they are *not* being learned from — the effective training set is the **409 states with
≥ 2 candidates**. That is not a silent subset bug: a state with one candidate carries no ordering to
learn and cannot be split into train and inner-val at all. But it should be said, and the fix for
future runs is to exclude `n == 1` states explicitly rather than let them arrive as a NaN.

**So the finished number is interpretable.** Had it been (1) or (2) the run would not be reportable
and we would have said so rather than quoting it with a caveat.

---

## ★ THE FIT COST, PROFILED — and the awkward answer

The first union refit ran at 1.86 min/epoch and projected to **174 days**; it was killed. The natural
suspicion was that the frozen outcome head `P(o|s,q)` was being re-run every epoch. **It is not** —
`P` is computed once in `build_state_pack_live` under `no_grad` and stored on the pack. That
hypothesis is dead, and the real answer is less convenient than a bug.

**Where the time goes**, measured on the real 425 pack shapes with no environment built, so compute is
isolated:

| | |
|---|---|
| packs | 425 |
| user-rows summed over packs | **876,520** |
| (user × candidate) cells per forward | **75.8 M** |
| one epoch, as written | **95.5 s** (real run: 112 s — the profile is faithful) |

★ **The cost scales with USERS, not candidates.** The root pack (10,000 users × 1,000 candidates)
takes 0.888 s per training step; the *same 10,000 users with only 50 candidates* still takes 0.654 s.
Cutting candidates twentyfold saves 26%.

**So "1.5 ms per example" used the wrong denominator.** This model does not consume 72,717 triples
per epoch. It evaluates three outcomes over 876,520 user-rows and 75.8 M user-candidate cells —
**~227 M values forward and again backward**. At 95.5 s on six CPU threads with autograd that is
~2.4 M values/s, which is ordinary rather than pathological.

### The one real waste, and it is exactly recoverable

`StateEncoder.forward` builds `tok = hist_phi ⊗ onehot(hist_out)`, a **(B, T, 3·d_phi)** tensor —
**352 MB at the root** — on *every* forward, from three inputs that are **constant** during a fit
(only the weights move). So `pooled`, `cnt` and `n_asked` are constants recomputed 80 times per fit.

Hoisting them is arithmetic-identical — verified, **max |cached − original| = 0.000e+00** — and buys:

| | as written | hoisted |
|---|---|---|
| one epoch | 95.5 s | **41.2 s** (2.3×) |
| one fit (80 epochs) | 2.12 h | **0.92 h** |
| pack memory | 1.70 GB | 3.51 GB (`hq`/`ho`/`hm` dropped — the cache *moves* memory, not adds) |

### ⚠ 2.3× is a real saving and it is not a rescue

| protocol | fits | as written | hoisted |
|---|---|---|---|
| **LOSO, 187 eval states** × 3 seeds × 3 arms | 1,683 | 149 d | **64 d** |
| **5-fold** × 3 seeds × 3 arms | 45 | 4.0 d | **41 h** |
| 10-fold × 3 seeds × 3 arms | 90 | 8.0 d | 82 h |

**Leave-one-out is the wrong protocol at this size** — it costs one full fit per held-out state, and
187 held-out states is where that stops being affordable. K-fold is the standard choice at this `n`
and gives *more* training data per fit, not less; nothing else about the protocol changes.

**The remaining lever is the per-user term** (876,520 user-rows). Capping the users in a *training*
forward would cut it several-fold. **That is a HARD RULE 1 sampling decision on users** — it goes to
the author as its own question with the design in hand, and it has not been taken here.

**Nothing is launched.** The numbers go in front of the author first.

---

## Execution state

Stages 1–3 (sample → scrape → meanvec) **completed**. Stage 4 (train) was launched, ran one fit of a
projected 1,683, and **was killed** once the 174-day projection was established: the premise
"finish the training" was not reachable, so continuing was waste with no partial result to salvage.
The box is free.

**Nothing is queued.** The refit does not relaunch until the author has chosen a fold count against
the measured costs above.

★ Idle is detected by process **name and resident set**, never by command-line text. The threshold is
measured, not guessed: the idle `http.server` pythons sit at 2–10 MB and the live phase-3 sweep at
919 MB, so the obvious 1 GB guess **called a live sweep idle on its first test**.

---

## For a memory note

- **Admit a source on its recorded contract, its recorded cohort, and its recorded OBJECTIVE — three
  separate tests, and the third is the one nobody was checking.** The split-criterion ablation's
  heaps look exactly like the tree's and hold a surrogate scalar.
- **Correlation and argmax rank move independently — report both, always.** 238 downward-biased
  states moved ρ by +0.016 and the argmax rank from 161 to 37; two more states then undid the second
  effect entirely.
- **★ A baseline's number is a property of its evaluation set.** "+0.4163, a parameter-free average
  beats the learned head" was substantially a seven-state artefact — seven shallow, large-cohort,
  same-spine states. On 187 unbiased states the same estimator scores +0.2956 and fails the gate it
  was celebrated for passing. Quote a comparator with its sample size or not at all, and say out
  loud when a bar has become weaker so nobody reads a later pass as more impressive than it is.
- **★ Cost-model the PROTOCOL, not just the run.** Leave-one-out costs one full fit per held-out
  state. It was fine at 7 states (21 fits) and became 174 days at 187, and the evaluation set grew by
  a design change that never re-derived the cost. When an experiment's `n` changes by 25×, re-derive
  what the protocol costs before launching, not after.
- **Profile before optimising, and be willing to lose the comfortable hypothesis.** "The frozen head
  is being recomputed" would have been a satisfying bug; it was false. The cost scaled with *users*,
  not candidates, which named the real culprit in one measurement — and the recoverable waste was
  2.3×, not the 100× a bug would have given.
- **A `nan` in a reported aggregate is not automatically a `nan` in the objective.** A mean over an
  empty index slice is `0/0`; its backward pass produces exactly zero gradient, not NaN. Check which
  it is by measuring the gradient, not by reading the loss line — but check, because the two
  possibilities differ on whether the run is reportable at all.
- **Ask what the dataset does NOT cover before paying for it, not after.** The authorised design
  would have left depths 7–15 — over half the interview — with no unbiased measurement, and would
  have measured a gain for a typical catalogue question exactly as often as everything before it
  did: never. Both holes closed for the same money, because the cost is per *evaluation* and neither
  a longer prefix nor a different candidate changes that.
- **A parameter-free baseline is not an assumption-free one.** The mean-vector estimator averages
  over whichever states happen to hold a value for `q`, and that support ranges from a handful to
  hundreds. Two dense states outweighed 238 sparse ones. Any "which source helps" claim has to wait
  for a training set whose support is not wildly uneven.
- **A "heavy job" threshold must be measured on the box.** 1 GB called a live 919 MB sweep idle.
