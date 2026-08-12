# Act 2 / C1 / TODO 1.11 — does a published selector's own acquisition score track the metric?

> **STATUS 2026-08-12 23:xx: DESIGN AND METHOD SECTIONS WRITTEN; NO RESULT IN THIS FILE YET.**
> Phase 1 (the realised-gain measurement, ~12 h) is **held** — the box belongs to the Hu & Yu
> replication tonight (author priority). Everything below §5 is pre-written from the design sheet so
> that the moment phase 1 lands the numbers drop into place and nothing about the *design* can be
> retro-fitted to the answer. **Every table below with `···` in it is a placeholder and is not a
> result.** Sections 1–4 are pre-registered claims about the method, not about the outcome.

Design sheet: `docs/design/DS_ACT2_SURROGATE_RANK_CORR.md` (pre-registered before any scoring run).
Code: `scripts/act2_surrogate_rank_corr.py` (phases 1/2) + `scripts/act2_surrogate_stats.py`
(phase 3 statistics). Ledger status: **EXPLORATORY**.

AUTHOR SAID (2026-08-12): *"all our baselines are optimised for approximated objective and that seems
to be the problem, not that cold interview answers do not differentiate."*

---

## 1. The question, and why it is not already answered

Act 2 currently asserts objective mismatch from a **code audit** — `BASELINE_OBJECTIVE_AUDIT.md`
establishes that 1 of 21 selection rules maximises the quantity anyone reports, and that one is a
static order. An audit says the surrogates are **different objects**. It does not say they are
**uninformative about the reported one**. A surrogate can be a different object and still rank
questions almost the way the metric does, and if it does, objective mismatch is not the mechanism and
Act 2 is wrong.

So: at matched user states, rank every candidate two ways — by each method's own acquisition score,
and by the **actual realised NDCG@10 gain of asking it** — and report the agreement.

The decision rule was fixed in advance:

| outcome | consequence |
|---|---|
| published surrogates near the shuffle FLOOR, our leg-1 criterion `U` near the reliability CEILING | Act 2's mechanism paragraph becomes a measurement; **1.15 (criterion-argmax) is green-lit** |
| `U` also near the floor | **1.15 is KILLED, not built.** Act 2 narrows to "no available surrogate tracks the metric — only direct measurement does", and workstream E collapses to E1/E2 with E4 dropped |
| some published surrogate has HIGH agreement yet loses in Act 1 | **the objective-mismatch story is falsified** and Act 2 is rewritten around the answer channel or the action space |
| all rows equal, floor and ceiling equal | instrument broken — report nothing |

## 2. The states, and what they cost in interpretability

**Cohort: the 10,000 canonical TEST users.** States are the reference static interview
`fixedorder:entropy0`, read out of the harness's own banked checkpoints through the same
`read_interview` / `states_of` pair `act2_pv_quartiles.py` uses — imported, never re-derived, so this
exhibit and the PV strata sit on one ruler. That cell asks one fixed item order of everybody and
spends **zero** concept turns, so a state is exactly *(asked items, the user's own half-star levels,
with an UNSEEN sentinel where they could not answer)*.

Every method is scored on the **same** inputs, so a difference between rows is the criterion and
nothing else. The cost of that choice, stated as a limitation rather than discovered later: a
surrogate might track realised gain better at the states **it** actually visits. The arm-own-states
variant is a re-run of phase 1 restricted to each arm's states; it is **not** run here and no claim
is made about it.

**All distinct states at every budget. No threshold, no sampling, no minimum state size.** Every
population statistic is weighted by user count, so a 1-user state contributes its one user and
nothing more — the weighting, not a cut, is what stops the long tail of tiny states dominating.

| k | states | singleton states | largest state | user mass in states with ≥2 users | ≥10 users |
|---|---|---|---|---|---|
| 0 | 1 | 0 | 10,000 | 1.000 | 1.000 |
| 1 | 11 | 0 | 5,633 | 1.000 | 1.000 |
| 2 | 112 | 18 | 3,933 | 0.998 | 0.984 |
| 4 | 1,463 | 755 | 2,531 | 0.924 | 0.721 |
| 8 | 5,767 | 5,066 | 1,463 | **0.493** | 0.301 |

### ★ The k=8 caveat, pre-registered

**At k=8 only 49.3% of users sit in a state with ≥2 users.** Three consequences, all of which travel
with every k=8 number in this document:

1. the split-half reliability **ceiling is only defined for 49.3% of the cohort** — a state with one
   user cannot be split, so there is nothing to correlate;
2. a per-state ρ at a singleton state is a one-user quantity, and a rank correlation computed on one
   user's realised gains is not a population statistic;
3. therefore **every ρ table prints a companion column restricted to states with ≥10 users** (30.1%
   of the cohort at k=8, 72.1% at k=4, 98.4% at k=2, 100% at k=1) **with its coverage beside it.**

That restriction is a **reporting stratification of a fully computed quantity** — every state and
every user is scored, nothing is dropped from the computation — and it is explicitly *not* a data
reduction. **No cell is dropped from the regret statistic**, which is well defined at n=1 and is the
reason regret, not ρ, is the headline number at deep budgets.

**k=1 is under-powered for ρ** (11 states, ≈0.15 half-width) and is carried for the regret statistic,
which pools 10,000 users. k≥2 resolves ρ differences of 0.05, far below the floor-to-ceiling span the
design needs.

## 3. The candidate pool, and what is and is not restricted by it

The tree's **combined** action space: top-500 items by train popularity + top-500 concepts by the
`answerable` bank, built by `ndcg_tree.build_pool` — the action space our own method searches, so
agreement is measured on the space the chapter's claim is about.

> **★ DECLARED RESTRICTION, AUTHOR SIGN-OFF OBTAINED 2026-08-12 (HARD RULE 1).** The published
> item-side selectors argmax over all 18,359 items and the conversational arms over all 1,031
> concepts, so the 1,000-candidate pool is a restriction of their native action space. The non-lossy
> alternative — all **19,390** candidates, costed at **≈230 wall-clock hours** — was put to the
> author and **declined**. The approval rests on the phase-1b mitigation below. **No other reduction
> is approved**: not users, not states, not budgets, not resamples, not rows.

**Phase 1b makes the money number pool-free.** The *correlation* is reported inside the common pool,
but the **regret** statistic additionally evaluates each arm's **true unrestricted argmax** at every
state — the question the arm would actually have asked, even when it falls outside the pool. That is
≤13 extra candidates per state against 1,000, so the headline "what does this selector's own rule
cost, in NDCG@10" number is **not** pool-restricted. Every ρ in this document carries the pool
restriction in its caption; every regret number carries whether it is the pooled or the unrestricted
form.

## 4. The rows — and three structural facts that are results, not inconveniences

Authoritative list: `BASELINE_OBJECTIVE_AUDIT.md`. Each row emits, at a state, **the very vector its
own `select()` argmaxes** — asserted by gate G-F, not paraphrased.

### 4a. ★ Nine of the twenty rows have ONE score vector for the entire population, per budget

At `entropy0`'s states every user has the **same asked set** at a given budget; only the answers
differ. Nine of the twenty rows depend on the asked set alone:

`bald`, `aopt`, `dre`, `popularity`, `helf`, `entropy0`, `pure_entropy`, `conlinucb_bs_static`, and
`greedy_static` — plus the key-term half of `conlinucb_mcr`, `conlinucb_bs` and `conucb`, and
`igcn`'s global phase.

For those rows the top-1 pick is **the same question for all 10,000 users at all 5,767 states**, so
they cannot track a state-varying optimum *even in principle*. Their correlation is reported and is,
by construction, constant across states. **That constancy is the reportable fact.** It is the
sharpest available form of the audit's "12 of 23 rows cannot personalise on answer content", and it
is exactly what the **state-swap control** is built to expose: a value-free row must survive the swap
with its ρ unchanged, which also makes it the positive control for the control itself.

Note what this does and does not say. A constant vector is not a *bad* ranking — a value-free row can
still have a high ρ against the *average* realised gain. What it cannot do is differ between two
users who answered differently, which is the whole content of "adaptive".

### 4b. ★ Golbandi's tree has no per-candidate score at a user state at all

The field's canonical adaptive interview criterion **is not defined at a test user's state**, and
this is reported in the table rather than the row being quietly omitted.

Its criterion `_e2` (`golbandi_tree_selector.py:195`) is a within-node squared error over **TRAIN**
users, evaluated during tree growth. `release()` (`:512`, called at `:678`) frees the matrices it
needs, and the object at interview time does pure ternary routing (`:714`). Measuring a per-candidate
vector at a test user's state would mean keeping the tree unreleased, persisting per-node train row
sets, and re-running `_best_split` at each of the ≤345 nodes the population routes to. That is
feasible — the Eq. 3 subtraction trick is six sparse products — but it is a separate build.

**The row appears in every table with `no per-candidate score at a user state` in its cells.** That
absence is an Act-2 sentence in its own right: a criterion that cannot be evaluated at the state
where the question is asked is not a ranking of that state's candidates.

### 4c. Other rows carried with their deviations named

- `fixedorder:greedy_{items,concepts,combined}` at k>0 — a true per-candidate gain vector at a
  *conditioned* state is exactly what phase 1 computes; CELF only ever holds true gains for the whole
  pool at step 0. So the `greedy_static` row uses the **k=0** vector and is labelled a STATIC score,
  which is what it is.
- `dpe` / `dpe_oracle` — the realised argmax is gated to a randomly drawn topic, so a full-pool
  vector is a different object from the policy. Reported only with the gate reproduced (`-inf`
  outside the drawn topic), or not at all.
- `greedy_slim` — folded in as a value-free row if its blob carries the per-slot vector, otherwise
  reported as an ORDER only (rank = position).
- `thompson` — randomised, so its vector is a **draw**, not a fixed function of the state; its ρ
  carries the draw-to-draw component and is labelled.

## 5. Gates

| gate | check | tolerance | status |
|---|---|---|---|
| G-A contract | our exact-NDCG path reproduces the banked NDCG-tree TEST per-user vectors for ≥3 leaves at K ∈ {1,2,4,8,16} | **0.0 exactly** | pending phase 1 |
| G-B state reconstruction | baseline NDCG at the reconstructed entropy0 states == the harness's own scored vector, per user | ≤ 1e-6 | pending phase 1 |
| G-C multistate identity | `genv_eval_multi` == `genv_eval` on a homogeneous user set | bit-identical | pending phase 1 |
| G-D partition | cohort mean == state-size-weighted mean of state means | < 1e-9 | pending phase 1 |
| G-E leak | `leak_assert` — no held-out target inside the routing lookup | 0 users | pending phase 1 |
| G-F score-is-the-argmax | for every row and state, `argmax` of the emitted masked score vector == the action `select()` returns for a user in that state | exact, every row | pending phase 2 |
| G-G shuffle floor | the candidate-shuffle floor's bootstrap CI covers 0 for every row | pre-registered | pending phase 3 |
| G-H self-difference | a row against itself: ρ = 1, regret = 0 | exact | **statistics side PASSES** (`act2_surrogate_stats.py --selftest`) |

A failed control means a broken instrument, not a result (HARD RULE 5).

---

## 6. RESULTS — *not yet measured*

### 6a. The primary table — Spearman ρ against realised NDCG@10 gain

*Placeholder. Judged against the MEASURED ceiling of §6d, never against 1.0.*

| row | channel | value-free? | k=1 | k=2 | k=4 | k=8 | k=8, states ≥10 users |
|---|---|---|---|---|---|---|---|
| `golbandi_tree` | items | — | *no per-candidate score at a user state* | | | | |
| *(19 rows)* | | | ··· | ··· | ··· | ··· | ··· |
| **FLOOR** (candidate shuffle) | | | ··· | ··· | ··· | ··· | ··· |
| **CEILING** (split-half, S-B corrected) | | | ··· | ··· | ··· | ··· | ··· |
| *ceiling coverage (user mass)* | | | 1.000 | 0.998 | 0.924 | **0.493** | 1.000 |

### 6b. Top-1 and top-10 agreement

*Placeholder.*

### 6c. REGRET, in NDCG@10 units — the money number

*Placeholder. Reported twice per row: pool-restricted, and phase-1b **unrestricted** (each arm's true
argmax over its native action space).*

### 6d. The three controls and the identity

*Placeholder. FLOOR, STATE SWAP (with the value-free rows' survival as the positive control), CEILING
with coverage, self-difference identity.*

### 6e. Verdict against the §1 decision rule

*Not yet measured. Nothing is written here until every gate in §5 has passed.*

---

## 7. Not claimed

- Nothing here is a claim about held-out generalisation — it is a diagnostic on frozen criteria.
- Nothing here says a selector *would* have won with a better criterion; it says its own score does or
  does not rank questions the way the reported metric does.
- The pool restriction travels with every ρ; only the regret statistic is pool-free.
- `golbandi_tree` is absent from the ρ table **with the reason printed in the table**, not omitted.
- A null on any row is not citable as a result (author ruling); it is reported as a null with its
  floor and ceiling beside it.
