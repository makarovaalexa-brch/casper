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
are in.** The one thing the table does say now is that the state-blind bar is **not** a fixed
+0.4163: on the two deep states phase 3 has finished so far it drops to +0.3975 and fails its own
gate. Deep, small-cohort states are harder for a state-blind average than the tree's shallow ones —
which is the direction that would make a *conditional* value easier to justify, and it is exactly
why it must not be claimed on two states.

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
evaluation. **2 of 42 states are complete and already ingested** — 1,994 measurements — taking the
union to **247 states / 25,977 measurements**. Nothing about the run is disturbed.

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

1. draw `k ~ Uniform{3,4,5,6}`, then `k` distinct questions **uniformly without replacement** from
   the policy-B combined pool (500 popularity items + 500 answerable concepts — the tree's own pool,
   same ids in the same order);
2. ask that prefix to all 10,000 users; **keep the per-user NDCG vector**;
3. record each user's ternary answer path (like / dislike / no-answer, per prefix question);
4. draw **50 candidates** uniformly without replacement from the pool minus the prefix; evaluate each
   over all 10,000 users, again keeping the per-user vector;
5. for the marginal state and for **every answer path with ≥ 200 users**, the conditional gain is
   `mean(f_q[path]) − mean(f_0[path])` — the same construction the tree uses to score a split, so it
   is on the same ruler.

Step 5 is why one sweep buys many states: grouping by answer path yields one measured gain vector per
populated path **at zero extra evaluation**.

**The declared truncation.** Paths below **200 users** are dropped — the tree's own `min_leaf`,
chosen because it already governs every banked state rather than invented for this job. The number of
paths dropped and the users they carry are recorded per prefix in the artifact. Because the per-user
vectors are banked, changing that threshold later costs nothing and requires no re-evaluation.

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
4. the **ablation by source** — unbiased-only vs unbiased + tagged-biased.
5. the **no-answer flatness check**: `V(no-answer|q,s)` must sit at `base(s)` and be flat across `q`.
   Last time it varied with sd 0.058, about three times the range of the quantity, and drove the
   argmax.

**HARD RULE 6 applies in advance:** if the refit still fails, that is a weak method reported as one.
No tuning until it passes, and no marketing a null.

---

## Execution state

`value_chain.py` is armed (detached, pid 14652). It waits for a **sustained idle box** — five
consecutive quiet polls — then runs sample → scrape → meanvec → train, each stage skipped if its
artifact exists, aborting rather than building on a failed stage.

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
- **A parameter-free baseline is not an assumption-free one.** The mean-vector estimator averages
  over whichever states happen to hold a value for `q`, and that support ranges from a handful to
  hundreds. Two dense states outweighed 238 sparse ones. Any "which source helps" claim has to wait
  for a training set whose support is not wildly uneven.
- **A "heavy job" threshold must be measured on the box.** 1 GB called a live 919 MB sweep idle.
