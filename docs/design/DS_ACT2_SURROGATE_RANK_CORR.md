# DS — ACT 2 / C1 / TODO 1.11: does a published selector's own acquisition score track the metric?

Design sheet, 2026-08-12. **Pre-registered before any scoring run.** Workstream C1 of
`PAPER_B_PLAN_2026-08-12.md`; gates TODO 1.15.

AUTHOR SAID (2026-08-12): *"all our baselines are optimised for approximated objective and that seems
to be the problem, not that cold interview answers do not differentiate."*

---

## 1. The question, and what changes on the answer

Act 2 currently asserts objective mismatch from a **code audit** (`BASELINE_OBJECTIVE_AUDIT.md`:
1 of 21 rules optimises the reported quantity). An audit says the surrogates are *different objects*.
It does not say they are *uninformative about the reported one* — a surrogate can be a different
object and still rank questions almost the way the metric does, in which case objective mismatch is
not the mechanism and Act 2 is wrong.

**The measurement.** At a set of matched user states, rank all candidates two ways — (a) by each
method's own acquisition score, (b) by the ACTUAL realised NDCG@10 gain of asking the question —
and report the rank agreement.

**Decision rule (this is why it runs).**

| outcome | consequence |
|---|---|
| published surrogates near the shuffle FLOOR, our leg-1 criterion `U` near the reliability CEILING | Act 2's mechanism paragraph becomes a measurement; **1.15 (criterion-argmax) is green-lit** — the criterion demonstrably tracks the metric |
| `U` also near the floor | **1.15 is KILLED, not built.** Act 2 narrows to "no available surrogate tracks the metric — only direct measurement does", and workstream E collapses to E1/E2 (imitate the tree) with E4 dropped |
| some published surrogate has HIGH agreement yet loses in Act 1 | **the objective-mismatch story is falsified.** Act 2 is rewritten: the loss lives in the answer channel or the action space, not the criterion. (TODO 1.2 already killed the action-space branch, so this would be a genuine surprise and a genuine redirect) |
| all rows equal, floor and ceiling equal | instrument broken — the reliability control failed; report nothing |

No cell is reported unless the gates in §7 pass.

## 2. The states (exact Ns)

**Cohort: the 10,000 canonical TEST users.** Forced: the banked Act-2 states live there, and reusing
them is what makes this cheap. Nothing is selected, tuned, or fitted on the outcome — this is a
descriptive diagnostic of criteria that were all frozen before it — so no firewall is spent. Recorded
in the exploratory/confirmatory ledger as EXPLORATORY.

**States = the reference static interview `fixedorder:entropy0`**, read out of the harness's own
checkpoints `experiments/baselines/interview/adaptive_table_act2pv_interviews/fixedorder__entropy0_k{k}.json`
(the same source `act2_pv_quartiles.py` uses, via the same `read_interview`/`states_of` pair, imported
not re-implemented). Verified properties of that cell: one fixed item order `[8,121,59,123,5,4,631,0]`
asked of everybody, **zero concept turns**, so a state is exactly `(asked items, own half-star level
or the UNSEEN sentinel 10)`.

| budget k | distinct states | users |
|---|---|---|
| 0 (cold) | 1 | 10,000 |
| 1 | 11 | 10,000 |
| 2 | 112 | 10,000 |
| 4 | 1,463 | 10,000 |
| 8 | 5,767 | 10,000 |

**State sizes are extremely skewed, and this drives how the statistics are read** (measured by
`--selftest`, reproduced from the banked interviews):

| k | states | singleton states | largest state | user mass in states with ≥2 users | ≥10 users |
|---|---|---|---|---|---|
| 1 | 11 | 0 | 5,633 | 1.000 | 1.000 |
| 2 | 112 | 18 | 3,933 | 0.998 | 0.984 |
| 4 | 1,463 | 755 | 2,531 | 0.924 | 0.721 |
| 8 | 5,767 | 5,066 | 1,463 | 0.493 | 0.301 |

Consequences, pre-registered rather than discovered: at k=8 the split-half reliability ceiling is
only defined for **49.3%** of the cohort, and a per-state ρ at a singleton is a one-user quantity. So
the primary table reports the user-weighted ρ over ALL states with a companion column restricted to
states with ≥10 users and the coverage printed beside it. That restriction is a **reporting
stratification of a fully computed quantity** — every state and every user is scored — not a data
reduction, and no cell is dropped from the regret statistic, which is well defined at n=1.

**ALL distinct states at every budget. No threshold, no sampling, no minimum state size.** Every
population statistic is weighted by user count, so a 1-user state contributes its one user and
nothing more — the weighting, not a cut, is what stops the long tail of tiny states dominating.

*Why entropy0's states and not each arm's own states.* Matched states are the entire point: every
method is scored on the SAME inputs, so a difference between rows is the criterion and nothing else.
On each arm's own states a method could look good merely by visiting easy states. **Limitation,
stated in the write-up:** a surrogate might track realised gain better at the states it actually
visits. The arm-own-states variant is a re-run of phase 1 restricted to that arm's states (same cost
per arm-budget cell, §6); it is NOT run now and no claim will be made about it.

**k=0 is carried** even though it has one state: it is free, it is the state the static greedy order
was built at, and its realised-gain vector IS `fixedorder:greedy_*`'s own acquisition score (§4).

## 3. The candidate pool

**The tree's COMBINED pool: top-500 items by train popularity + top-500 concepts by the `answerable`
bank**, built by `ndcg_tree.build_pool` — the same two `_ranked_pool` calls, same order (items
first). Chosen because it is the action space our own method searches, so surrogate-vs-metric
agreement is measured on the space the chapter's claim is about.

Candidates already in a state's prefix are excluded per budget (marked NaN, never scored, never
counted); at k=8 that is at most 8 of the 1,000.

**DECLARED SHORTCUT (HARD RULE 1 — needs author sign-off before launch).** The published item-side
selectors argmax over all 18,359 items and the conversational arms over all 1,031 concepts. The pool
is therefore a restriction of their native action space.
- **Non-lossy alternative:** 18,359 items × 1,031 concepts = 19,390 candidates → **19.4× the cost,
  ≈ 230 wall-clock hours.** Not run.
- **Mitigation that recovers the part that matters, at 1.3% extra cost — phase 1b.** The
  *correlation* is reported inside the common pool, but the *regret* statistic uses each arm's TRUE
  UNRESTRICTED argmax: for every (state, arm) we additionally evaluate the realised gain of the
  question the arm would actually have asked, even when it is outside the pool. That is ≤ 13 extra
  candidates per state (≤ 130,000 extra user-evaluations per budget against 10,000,000), so the
  headline "what does this selector's own rule cost, in NDCG@10" number is **not** pool-restricted.
  Phase 1b runs after phase 2 (it needs the arms' argmaxes) and costs ~10 min per budget.

## 4. The rows (the full action space of methods)

Authoritative list: `docs/results/BASELINE_OBJECTIVE_AUDIT.md`. Each row must emit, at a state, a
score over the pool that is **the very vector its `select()` argmaxes** — asserted, not paraphrased
(§7 gate G-F).

| # | row | score source | value-free at these states? | channel |
|---|---|---|---|---|
| 1 | `glm_ucb` | `Y@mu + alpha*width` (`bandit_selectors.py:334`) | no | items |
| 2 | `bald` | `_bald_vec` :406 | **YES** | items |
| 3 | `aopt` | `_gain_vec` :498 | **YES** | items |
| 4 | `thompson` | `S = U @ Y.T` :572 — **per user**, not per state | no (randomised) | items |
| 5 | `eddi_ig` | `_acquisition` `eddi_selector.py:102` | no | items |
| 6 | `eddi_ig3` | `_acquisition` :274 | no | items |
| 7 | `igcn` | `ig_global` :429 (cold phase) / `personalised_ig` :306 scattered dense | global phase YES | items |
| 8 | `dre` | static ask order, budget-specific | **YES** | items |
| 9 | `popularity` | `interview_strategies.build_orders` :77 | **YES** | items |
| 10 | `helf` | :81 | **YES** | items |
| 11 | `entropy0` | :78 | **YES** | items |
| 12 | `pure_entropy` | :78 | **YES** | items |
| 13 | `conlinucb_mcr` | `_item_scores` :617 + `_keyterm_scores` :699 | key-term half YES | both |
| 14 | `conlinucb_bs` | :739 | key-term half YES | both |
| 15 | `conlinucb_bs_static` | :739, `b_schedule=always` | **YES and state-independent** | concepts |
| 16 | `clime` | :792 | score YES, trigger no | both |
| 17 | `conucb` | Eq. 8 :889 | key-term half YES | both |
| 18 | `dpe_concept` | `theta` `dpe_selector.py:461` | no | concepts |
| 19 | **`leg1_U` (OURS, the 1.15 candidate)** | banked `U_item`/`U_conc` in `.cache/act2_pv/` | no | both |
| 20 | **`greedy_static` (OURS)** | the realised-gain vector measured at **k=0** — by definition the static greedy order's own criterion | **YES** (static by construction) | both |

**The VALUE-FREE block is a result, not an inconvenience.** At entropy0's states every user has the
same ASKED SET at a given budget, and 9 of the 20 rows depend on the asked set alone. Those rows
therefore have **ONE score vector for the entire population at each budget**: their top-1 pick is the
same question for all 10,000 users at all 5,767 states, so they cannot track a state-varying optimum
even in principle. Their correlation is reported and is *by construction* constant across states —
that constancy is itself the reportable fact, and it is the sharpest form of the audit's "12 of 23
rows cannot personalise on answer content". It is exactly what the STATE-SWAP control (§5) is built
to expose.

**Rows deferred, with reasons stated rather than hidden.**
- `golbandi_tree` — **has no per-candidate score at a user state at all.** Its criterion `_e2`
  (`golbandi_tree_selector.py:195`) is a within-node squared error over TRAIN users evaluated during
  growth; `release()` (:512, called at :678) frees the matrices it needs, and the object at interview
  time does pure ternary routing (:714). Measuring it means keeping the tree unreleased, persisting
  per-node train row sets, and re-running `_best_split` at each of the ≤345 nodes the population
  routes to. That is feasible (the Eq. 3 subtraction trick is six sparse products) but it is a
  separate build; **the fact that the field's canonical adaptive interview criterion is not even
  defined at a test user's state is itself an Act-2 sentence** and is written that way unless the
  author asks for the build.
- `fixedorder:greedy_items/concepts/combined` at k>0 — a true per-candidate gain vector at a
  conditioned state is exactly what phase 1 computes; CELF only ever holds true gains for the whole
  pool at step 0. So row 20 uses the k=0 vector and is labelled a STATIC score, which is what it is.
- `dpe`/`dpe_oracle` — the realised argmax is gated to a randomly drawn topic, so a full-pool vector
  is a different object from the policy. Reported only with the gate reproduced (`-inf` outside the
  drawn topic), or not at all.
- `greedy_slim` — offline static order; folded in as a value-free row if its `.pt` blob carries the
  per-slot vector, otherwise reported as an ORDER only (rank = position).

## 5. Statistics, controls, MDE

Per (row, budget, state) with pool candidates `q`:
- realised gain `g_s(q) = mean_{u in S_s} [ NDCG@10(u | state_s + q) − NDCG@10(u | state_s) ]`,
  measured exactly the way the NDCG tree measures it (§6);
- method score `a_s(q)` over the candidates that row can score.

Statistics, all weighted by `|S_s|`, all also reported per channel (items-only, concepts-only,
combined) because a combined-pool number conflates *which channel* with *which question inside it*:
1. **Spearman ρ(a_s, g_s)** — primary.
2. **Top-1 agreement** — does the row's argmax equal the realised argmax.
3. **Top-10 overlap** — |top10(a) ∩ top10(g)| out of 10. This is the selector-relevant one.
4. **REGRET, in NDCG@10 units** — `max_q g_s(q) − g_s(argmax_q a_s(q))`. The money number: it puts
   rank disagreement on the reported metric's own scale, and it is well defined even for a 1-user
   state. Phase 1b makes it pool-free (§3).

**Controls (mandatory, HARD RULE 5).**
- **FLOOR — candidate shuffle.** Within each state, permute `g_s` across candidates (RNG seed 98765,
  per-state stream). Every statistic recomputed. Expected ρ ≈ 0 ± 1/√999, top-1 ≈ 1/1000, top-10
  overlap ≈ 0.1. A row that does not separate from this floor tracks nothing.
- **STATE SWAP — the control that decides the chapter's sentence.** Pair `a_s` with `g_{σ(s)}` for a
  fixed derangement σ of the same budget's states. A row whose ρ SURVIVES the swap is not reacting to
  the state at all — it is tracking global candidate quality ("ask popular things"). Every value-free
  row must survive it by construction; that is the positive control for the control itself.
- **CEILING — split-half reliability.** Split each state's users deterministically (by user id
  parity within the state), compute `g_s` on each half, correlate half A against half B,
  Spearman–Brown corrected. Defined for states with ≥2 users, which is 100% / 99.8% / 92.4% / 49.3%
  of the cohort at k = 1 / 2 / 4 / 8 — printed with every ceiling. **Every row is
  judged against this ceiling, not against 1.0** — a ρ of 0.3 against a ceiling of 0.35 is a
  different chapter from a ρ of 0.3 against a ceiling of 0.9.
- **Self-difference / identity control:** a row scored against itself must give ρ = 1.0 and regret
  = 0.0 exactly.

**Uncertainty.** Paired bootstrap over STATES (10,000 resamples, seed 98765, the canonical setting),
states drawn with replacement and weighted by user count; for regret also a paired bootstrap over
USERS. Rows are compared pairwise on the same resamples.

**MDE.** Between-state SD of ρ is conservatively 0.25, so the 95% half-width on a weighted mean ρ is

| k | states | ≈ 95% half-width on mean ρ |
|---|---|---|
| 1 | 11 | 0.15 |
| 2 | 112 | 0.046 |
| 4 | 1,463 | 0.013 |
| 8 | 5,767 | 0.006 |

**k=1 is under-powered for ρ and is carried for the regret statistic** (which pools 10,000 users,
MDE ≈ 0.003 NDCG@10); this is stated in the write-up rather than discovered afterwards. k≥2 resolves
ρ differences of 0.05, which is far below the floor-to-ceiling span the design needs to distinguish.

## 6. How the realised gain is computed, and what it costs

**Exactly the tree's path**: certified pair `t2i26c_ep31.pt` + `concept_cv3_ep08.pt`, policy-B
candidate mask, learned decoder bias, no floor, behavioural answers, full 18,359-item catalogue,
uncapped signed concept values with `cap_negatives` over the revealed set. Concretely
`build_greedy_env(..., split="test", ...)` and the same `ndcg10_from_scores` funnel `genv_eval` uses.

**Phase 1 needs NO new scoring code, and that is the point.** Because `fixedorder:entropy0` asks the
SAME questions of everybody and the states differ only in the ANSWERS — which `genv_eval` already
looks up per user — the realised gain of candidate `q` at ALL states at once is exactly one call

```
genv_eval(E, prefix + [q], all 10,000 rows)      # the tree's own per-candidate path, unmodified
```

and the states are recovered afterwards by grouping users on the state id. So phase 1 runs on
literally the same function the NDCG tree's split search calls, with no wrapper, which is the
strongest available form of "same ruler as the tree".

**One new function is added to the existing module — for PHASE 1b, not for phase 1.**
`strategy_channel_suite.genv_eval_multi(E, rows, item_seqs, conc_seqs, ...)`: per-user (full, tail)
NDCG@10 where **each user carries their own ask sequence**. Phase 1b needs it because each arm's
unrestricted argmax differs BY STATE, so there the users genuinely carry different candidates and
`genv_eval`'s one-sequence-per-cohort shape does not fit. It batches ACROSS states at zero
approximation cost (`fold_items_frozen` and `ConceptFoldNet` already take per-user lists), with three
exact optimisations: state-major batching; within-batch deduplication of bit-identical recommender
inputs before the fold and the decoder matmul (exact, because the state key carries the exact
half-star levels); and reuse of the item fold across a whole concept block. G-C (§7) asserts
`genv_eval_multi == genv_eval` bit-for-bit on a homogeneous user set.

**Cost, derived from a MEASURED per-evaluation time.** The 22 nodes of the banked NDCG-tree build
give `sec_per_eval` for this exact code path at known cohort sizes: 0.397 s at n=205 … **8.484 s at
n=10,000** (linear, ≈ 0.14 s fixed + 0.83 ms/user).

| item | value |
|---|---|
| full-cohort candidate pass (measured) | **8.484 s** |
| per budget | 1,001 passes = **2.36 h** |
| 5 budgets (k = 0,1,2,4,8) | **≈ 11.8 h** |
| phase 1b (unrestricted argmaxes) | ≈ +50 min total |
| phase 2 (surrogate vectors, no NDCG) | minutes |

Dedup (2) makes k∈{0,1,2} cheaper than the linear model; the state-major batching adds a little
overhead. The `--smoke` step re-measures s/eval on one state and prints a refreshed ETA **before**
the long run commits — the number above is the pre-launch estimate, the smoke's is the one to act on.

**Reductions available if the box is needed elsewhere — NONE taken without sign-off.** Budgets are
scored in the priority order k=1, 2, 0, 4, 8 and each budget is a complete, self-contained,
publishable cell, so a stop after any budget yields a valid partial table rather than nothing.
Dropping k=8 saves 2.4 h and costs the deepest state set. Nothing else is reducible without losing
either candidates or users.

**Persistence (HARD RULE 11).** Per budget: `gain_full_k{k}.npy`, `gain_tail_k{k}.npy` memmaps of
shape (1000 candidates × 10,000 users) float32, `base_full/tail_k{k}.npy`, `done_k{k}.npy`
(per-candidate bool), flushed **after every candidate** — the blast radius of a kill is one candidate
(~8 s). Contract-signed sidecar (`contract.json`); a resume against a different contract is refused
loudly, never merged. Total on disk ≈ 400 MB. Memory is bounded by the 500-user batch (37 MB of
scores) — no per-state cache, float32 for everything cached.

**The per-user × per-candidate gain matrix is itself the deliverable of phase 1** and is reusable by
1.14 (kernel-weighted local greedy), 1.15 and E1 (policy imitation labels) without re-scoring.

## 7. Gates — all must pass before any number is reported

| gate | check | tolerance |
|---|---|---|
| **G-A contract** | our exact-NDCG path reproduces the banked NDCG-tree TEST per-user vectors (`ndcgtree16test_per_user.npz`, `test\|<nid>\|K\|own_full`) for ≥3 leaves at K ∈ {1,2,4,8,16} | max abs diff **0.0 exactly** |
| **G-B state reconstruction** | the baseline NDCG at the reconstructed entropy0 states equals the harness's own scored vector `ours\|fixedorder:entropy0\|k{k}\|full` in `adaptive_table_act2pv_per_user.npz`, per user | ≤ 1e-6; a non-zero difference must be explained before the run continues |
| **G-C multistate identity** | `genv_eval_multi` == `genv_eval` on a homogeneous user set | bit-identical |
| **G-D partition** | cohort mean = state-size-weighted mean of state means (algebraic identity; catches a state-map bug) | < 1e-9 |
| **G-E leak** | `leak_assert` — no held-out target inside the routing lookup | 0 users |
| **G-F score-is-the-argmax** (phase 2) | for every row and state, `argmax` of the emitted masked score vector == the action `select()` returns for a user in that state | exact, every row |
| **G-G shuffle** | the candidate-shuffle floor is within 3 SE of 0 for every row | pre-registered |
| **G-H self-difference** | a row against itself: ρ = 1, regret = 0 | exact |

A failed control means a broken instrument, not a result (HARD RULE 5).

## 8. Execution

Three phases, deliberately split so the only heavy thing is phase 1 and it needs **no selector code
at all**:

1. **`--score` (heavy, ~12 h, resumable)** — realised per-user NDCG for every (budget, candidate).
2. **`--surrogates` (cheap)** — per-state score vectors for every row in §4, via a `score_vector`
   hook added to `Selector` returning `None` by default (so no existing arm changes behaviour) and
   implemented as a 1–3 line delegation on the arms that already compute the vector internally. For
   `glm_ucb` this needs `bandit_selectors.py:334` extracted into `_ucb_vec(key)` — a pure extraction,
   the same pattern as `_a_inv_Yt`. The probe walks the BANKED interview sidecars; **no interview is
   re-run.** Gate G-F is what makes the extraction self-verifying.
3. **`--analyze` (cheap)** — statistics, the three controls, `docs/results/ACT2_SURROGATE_VS_NDCG.md`.

Phases 2–3 are built while phase 1 runs; they contend for nothing.

**Launch sequence.** Nothing starts until the box is free — workstream A1 (`ndcg_tree_extend16`,
PID 24120) and the re-measurement queued behind it own the machine, and HARD RULE 11(d) forbids a
second heavy job.

```
python scripts/act2_surrogate_rank_corr.py --selftest      # seconds, no env  -- ALREADY PASSES
python scripts/act2_surrogate_rank_corr.py --smoke         # ~10 min: every gate + measured s/candidate
python scripts/act2_surrogate_rank_corr.py --score --full_threads
```

**The smoke has NOT been run yet, deliberately.** It builds the full scoring environment (the 25M-row
ratings frame plus the certified pair), which is a multi-GB peak; with 5.6 GB free next to a live
12-hour job, running it would be exactly the "two heavy jobs on the box" that HARD RULE 11(d) was
written after. The `--selftest` path — which verifies the state reconstruction against all four
published state counts, the memmap resume and the contract guard, with no model loaded — has been run
and passes. The cost figures in §6 come from 22 MEASURED `sec_per_eval` points of the identical
`genv_eval` path banked by the NDCG-tree build; the smoke re-measures on this cohort before the long
run commits. **Run the smoke first; if its ETA disagrees materially with 2.4 h/budget, stop and
re-plan rather than launch.**

No gate is skippable by skipping the smoke: G-A and G-C run at the start of `--score`, and G-B and
G-D run per budget inside it.

## 9. Not claimed

- Nothing here is a claim about held-out generalisation — it is a diagnostic on frozen criteria.
- Nothing here says a selector *would* have won with a better criterion; it says its own score does
  or does not rank questions the way the reported metric does.
- The pool restriction (§3) travels with every ρ; only the regret statistic is pool-free.
- `golbandi_tree` is ABSENT from the ρ table, with the reason printed in the table, not omitted.
- A null on any row is not citable as a result (author ruling); it is reported as a null with its
  floor and ceiling beside it.
