# DS_C5 — the PROFILE-PARAMETERISED emitter: predict the gain profile, construct the emission

**Status: PRE-REGISTERED, NOT RUN. 2026-09-01.** Code `src/instrument/c5_profile_policy.py`
(unit-tested, committed before any run). Predecessor and the failure this exists to address:
`docs/results/C1_FINAL.md` + `docs/design/DS_C1_FINAL.md`. Artifact locations:
`docs/design/C1_FINAL_SCOPING.md`. **Every population caveat and every claim restriction in
DS_C1_FINAL carries over verbatim** — C5 changes the parameterisation, not the data.

> ⚠ **NAMING COLLISION, flagged not silently resolved.** The label "C5" is already taken by
> `src/instrument/c5_diagnostics.py` (commit `cf49372`, "decide whether C1-final's generalisation
> failure is the MODEL or the INPUTS"), and "C6" by `3787ad6` (the closed-form eigen-emitter). This
> file is named `c5_profile_policy.py` because the brief named it so; **the stage should probably be
> renumbered C7 before anything is written up.**

## 0. ★ WHAT THE C5 DIAGNOSTICS ALREADY MEASURED ABOUT THIS DESIGN'S PREMISE — READ BEFORE LAUNCHING

`experiments/paperC/c5_diagnostics.json` (finished 2026-09-01 21:26, 4.4 min) ran **D5**, which is a
closed-form ridge version of this sheet's reading 6: predict the measured gain profile at held-out
states from the state, then score Spearman and NDCG@5 against the same two baselines this sheet
registers. **On the primary split it already says the premise does not hold.**

| primary split (`root.like.*` held out) | Spearman | NDCG@5 |
|---|---|---|
| ridge on the cohort-mean latent (200-d) | −0.0413 | 0.0749 |
| ridge on the warm-start emission (129-d) | +0.1460 | 0.2120 |
| ridge on structure only | +0.2872 | 0.2432 |
| ridge on size + depth only | +0.3239 | 0.3273 |
| **baseline: global mean TRAIN profile (state-blind)** | **+0.3103** | **+0.3405** |
| baseline: popularity prior | +0.2058 | 0.0446 |

| secondary split (depth ≥ 5) | Spearman | NDCG@5 |
|---|---|---|
| ridge on the cohort-mean latent | **+0.3747** | +0.3128 |
| ridge, latent + warm + structural | +0.3783 | **+0.3539** |
| baseline: global mean TRAIN profile | +0.2598 | +0.2512 |

**Plain reading: on the primary split no state-conditioned probe beats the state-blind global mean —
the one that knows nothing about which state it is scoring. Only the secondary (depth ≥ 5) split
shows a modest edge**, and depth is confounded with cohort size, so a target-noise explanation is
live there. D2 separately shows profiles *are* predictable from a NEIGHBOURING trained state
(kNN +0.4666 over mean-other), i.e. **locality, not extrapolation** — exactly the C1-final shape one
level down.

**What this does and does not settle for C5.** A ridge on 72 state-mean rows is not the policy: the
policy carries the head's inductive bias, sees ~510,000 user-rows rather than 72 means, and is fit
end to end. So D5 is a strong prior, not a ceiling. But it means **C5 is now a test of a hypothesis
the cheap probe already leans against**, and it must be launched with that expectation on the record:
the honest pre-registration is *"D5 says the state-blind mean is hard to beat on the primary split;
C5 asks whether a trained profile head, on user-rows rather than state means, can beat it."* If
reading 6 lands on top of the global-mean baseline, that is a **replication of D5 at the policy
level**, not a new disappointment — and it is a clean, publishable statement of where the ceiling is.

## 1. Why this run exists

C1-final regressed the policy onto a 128-d blend **direction**: one supervised vector per state, 72
states. It fits (cos **0.970** at trained states) and does not extrapolate (**−0.119** at held-out
states), and the one statistic that looked like extrapolation was matched by the **untrained warm
start**. That is *action-shaped* supervision. Paper B already measured which shapes generalise on
this stack:

| shape | artifact | result |
|---|---|---|
| per-question **REACTION** head | `docs/results/HELDOUT_QUESTIONS.md` | generalises to unseen questions: held-out-question AUC macro **0.98167** against the fair backoff **0.96285** (that doc's binding rule: the aggregate is never quoted without its baseline; the defensible part is the interview-accrued **+0.0172**, which no state-free predictor can have) |
| **ACTION** policy | `P1_HELDOUT_ACTIONS.md`, `P1B_NEVER_TAUGHT.md` | does not |

C5 makes the emission reaction-shaped. The supervised object becomes the per-candidate gain
**profile** `s(q | state)` — which the incumbent head already computes, one scalar per candidate —
and the emission is *derived* from it rather than regressed:

```
p_hat = softmax(s / tau_m)             over the state's own unasked table (~1,000 candidates)
z_hat = sum_{j<=m} p_j e_{q_j}         over the predicted top-m, p renormalised on that top-m
```

## 2. The four properties, and where each is enforced

| property | how it is enforced | code |
|---|---|---|
| (a) supervision is ~1,000 numbers per state, not 128 | `p*` is dense over the whole measured table; the table is asserted to be exactly the unasked pool | `gain_rows`, `build_profile_targets` |
| (b) the emission cannot drift off-manifold and its norm cannot blow up | it is a **convex combination of real bank entries** (softmax weights ≥ 0, sum 1) ⇒ `‖z_hat‖ ≤ max_j ‖e_{q_j}‖` | `assert_convex`, called on every emission |
| (c) the discrete policy is nested **exactly** | `tau → 0` ⇒ one-hot on the argmax ⇒ `z_hat = e_{argmax_q s(q)}` | `selftest_nesting` (exact to 0.0), plus the run's report of whether that argmax is the tree's splitter |
| (d) differentiable | topk + softmax + weighted sum of constant bank rows | `emit_convex`; the run's gradient path is `enc → head.q → logits → weights → z_hat` |

**★ Property (b) deletes C1-final's scale term.** The pilot's cosine-only loss let the emission drift
38× out of scale; C1-final bolted on `λ(‖z_mr‖/‖t_mr‖ − 1)²`, λ = 0.1, to contain it, and that term
was learned per trained state and **did not transfer** (held-out norm ratio 3.480, unchanged from the
warm start, i.e. still unrenderable). **C5 carries no scale term at all** and cannot need one.

**★ And the objective nests C1-final's.** With `tau_t → 0`, `p*` collapses onto the argmax-gain
candidate — which is the tree's splitter, verified at **72/72** — and `KL(p* ‖ p_hat)` becomes exactly
the one-hot cross-entropy the incumbent distillation uses. Asserted to a relative 7.7e-08.

## 3. Declared constants (chosen once, before the run, never swept)

| constant | value | how it was fixed |
|---|---|---|
| `tau_m` (the loss/profile temperature) | **1.0** | the head's own logit scale. Fixing it at 1.0 is what puts the blend KL and the replay CE on one temperature and what makes the `tau_t → 0` nesting an identity rather than an approximation |
| `m` (blend width) | **5** | C1-final's registered width, so the two runs' targets are the same object |
| `tau_t` (target temperature) | **0.00883176** | **solved once from the sweeps alone** by the rule below — no model, no held-out feedback |
| `alpha` (blend-loss weight) | **1.0** | as C1-final |
| epochs / batch / lr | **3 / 512 / 3e-4** | C1-final's, unswept, so the comparison is like-for-like |
| `kl_topk` | **0 = the whole table** | any non-zero value truncates the supervision (HARD RULE 1) and needs sign-off |

### The `tau_t` rule, stated as a rule

> Choose `tau_t` so the target's entropy matches the measured profile's own spread. Operationally:
> `p*` **restricted to the top-m and renormalised** must reproduce, in the median over the 72
> states, the top-1 weight of the **gain-proportional** blend C1-final used as its target.

Restricting a softmax to a subset and renormalising *is* the softmax on that subset, so this is a
scalar equation in `tau` on the top-5 gains only; it is monotone (τ→0 gives w₁→1, τ→∞ gives w₁→1/m),
so a bisection solves it exactly. **Measured on the real sweeps:**

* `w1*` (median gain-proportional top-1 weight over the 72 states) = **0.2254** — identical to the
  0.2254 banked in `C1_FINAL.md` §3, so the calibration target is the pilot's flat-profile number,
  reproduced independently here. ✔ consistency check passed.
* `tau_t` = **0.00883176**, at which the median top-5-restricted top-1 weight is **0.2254**. At
  `tau_t` the top-5 of `p*` *is* C1-final's blend, in the median.

**Resulting effective support, reported as required — and it is the run's main open risk.**

| quantity | value over the 72 states |
|---|---|
| supervision width (candidates per state) | min 989, median **996**, max 1000 |
| positive-gain candidates | median **484** |
| target effective support `exp(H(p*))` | q1 943.9, **median 970.6**, q3 985.0 |
| full-table top-1 weight of `p*` | median **0.00301** (uniform would be 0.00100) |

**Read that honestly: the registered target is nearly uniform over the full table** (971 of 996
effective), because the criterion is set by the *top-5's* flatness while the tail is ~1,000 long, and
because `max(gain, 0)` ties roughly half the pool at exactly zero. Three consequences, all declared
now rather than discovered later:

1. It is **not incoherent** — it is the point. The measured profile really is that flat (`C1_FINAL`
   §3, median top-1 blend weight 0.2254, 0/72 above 0.5), and a faithful profile predictor should
   reproduce that flatness. If `p_hat ≈ p*` then the emission's top-5 weights are *already* ≈ 0.2254
   at `tau_m = 1`, and the read-out calibration below is near-null.
2. The **KL magnitude is ~1000× the replay CE** (≈ 6.9 nats at a peaked warm start vs 0.006 per
   replay example). The blend term will therefore flatten the head's logits at node states. That is
   intended; the risk is **collateral damage at ordinary states through the shared encoder**, and
   that risk is exactly what the blocking no-collapse gate exists to catch.
3. The information the loss carries is mostly in the **ranking**, not the probability mass. Reading 6
   scores the ranking directly, which is why it is the reading the re-parameterisation lives or dies
   on.

### The one constant fitted at run time, disclosed

`tau_emit`. The head's logit **scale** is not pinned by anything in the objective, so `softmax(s/1)`
over the top-5 can be arbitrarily peaked or flat, and with it the emission slides between "exactly
the argmax bank entry" (on the grid, no off-catalogue claim) and "the uniform centroid of five".
`tau_emit` is therefore declared by the **same rule as `tau_t`**: match the median top-1 emission
weight to the measured `w1* = 0.2254`, solved on **trained states only**. It is a **single global
scalar shared by every state including the held-out ones**, so it cannot carry per-state information
and cannot manufacture generalisation. The **warm start gets its own `tau_emit` under the same
rule**, otherwise reading 1b would be confounded by logit scale alone. The `tau_m = 1.0` read-out is
reported alongside every cell so the calibration is auditable.

## 4. Loss

```
L = alpha * KL(p* || softmax(s/tau_m))   at blend states (the 72 searched node states)
  +          CE(s, teacher one-hot)      everywhere else  [C1-final's replay, unchanged]
```

The replay CE is kept **unchanged and for its original reason**: it is the single difference between
C1-final's policy and the pilot's collapse (163 → 4–6 distinct questions at 0.0000 agreement), and
the no-collapse gate is blocking here too.

**Rulings inherited verbatim from C1-final** (both re-stated in code):
1. `head.k` / `head.b` are **FROZEN**. The bank `e_q = [k(φ_q), b(φ_q)]` is the space the registered
   cosines live in and the space the warm-start null is measured against; letting the loss move it
   would compare numbers from two different banks. C5 adds a second reason: the emission is *built
   out of* that bank, so a moving bank makes "convex combination of real bank entries" a moving
   claim. The run asserts post-hoc that both are bit-identical to the warm start's.
2. **A held-out state receives no loss at all** — not the profile KL and not the incumbent CE.
   Supervising it with the teacher's one-hot would pull the emission back onto the grid at exactly
   the states reading 1 scores.

**Listwise alternative, implemented and UNRUN:** `--loss approxndcg` (ApproxNDCG, Qin et al. 2010 —
differentiable rank via `1 + Σ σ((s_j − s_i)/T)`). It is restricted to the top-50 by relevance
because the pairwise term is O(P²) at P = 1000; **that restriction is a truncation of the supervision
and would need explicit sign-off before the arm is ever run.** It is tested for shape and gradient
only and is not registered.

## 5. Data, splits, warm start — identical to C1-final

* Teacher `coldtree_v2_train10000.json`; the **72 full exact whole-pool gain tables**
  (`argmax(gain) == splitter` at 72/72, `|max(gain) − splitter_gain| = 0`, re-asserted at load).
* Trajectory bank: the existing **85,289-user** cold arm of `m6/traj_dj` — reused, never rebuilt
  (a rebuild is ~1.42 GB against ~2.4 GB free and is not authorised).
* Warm start `m7dj_policy.pt`; **phi bank asserted to be 1,004 entries**; TRAIN canonical snap
  **0.15830706760250668** asserted exact (**not** the validation 0.15686538…).
* Splits: **primary** = the whole `root.like.*` subtree (15 held out / 57 trained, prefixes
  disjoint); **secondary** = depth ≥ 5 (30 / 42). **Random state splits stay FORBIDDEN** — node
  cohorts nest and a random split leaks.
* HARD RULE 10 dirty-`src/instrument` guard; contract-signed JSON; atomic writes; per-epoch flush
  (HARD RULE 11); threads ≤ 4; free-RAM refusal before the bank is materialised.
* **Firewall: train-side only. The 10,000 canonical TEST users are not built, not loaded, not
  touched.**

## 6. Registered readings

**Readings 1–5 are C1-final's, reused verbatim** (`reading1` and the gate machinery are *imported*
from `c1_final.py`, not reimplemented):

1. **GENERALISATION (headline).** At held-out states, `cos(z_hat, target_s)` against (a) a seeded
   within-split random other held-out state's target and (b) the warm start's own cos at the same
   states. **★ The WARM-START NULL is registered, not optional** — the same own-minus-other statistic
   with the warm start in both slots, under the same seeded pairing. It is what turned C1-final's
   headline (the untrained warm start's margin was *larger*), and reading 1a may not be quoted
   without it.
2. **FIT** at trained blend states (reported, not a gate).
3. **★ NO-COLLAPSE GATE (hard, blocking).** 1,000 ordinary non-blend states: distinct emitted
   questions ≥ 0.75× the warm start's, AND snap agreement ≥ 0.90.
4. **★ RENDERABILITY GATE (hard, blocking).** Median `‖z_hat_mr‖/‖target_mr‖ ∈ [0.5, 2.0]`. Expected
   to pass ~1.0 *by construction* under property (b); it stays a gate so that expectation is tested,
   not assumed, and so the held-out/trained split of it is on the record as in C1-final.
5. **Off-grid geometry**: `cos₁`, margin, and whether the emission's nearest bank question is the
   target's.

**NEW — reading 6, the profile level.** At held-out states, **Spearman** (ties averaged) and
**NDCG@5** of the predicted per-state profile against the **measured gains** over that state's own
table, against three comparators:

* the **untrained warm start** (the same null discipline as reading 1);
* a **popularity prior** — `policy_p1.py`'s declared dumb comparator, one scale across channels: the
  number of train users who can answer the question (`ctx.cnt[i]` / `answer_rate[c]·140,768`);
* a **global-mean-profile** baseline — the mean of `max(gain,0)` over the **trained** states,
  remasked to the scored state. This is the "no state information at all" predictor and is the one
  the reaction-shaped claim must beat: **if a single fixed profile does as well, the head is not
  conditioning on the state and C5 has bought nothing.**

Paired bootstrap (10k, seed 98765); unit = states. Reported at trained states too, for reference.

**NEW — the nesting report.** How often the modal per-user `argmax_q s(q)` equals the tree's own
splitter, at trained and at held-out states, with the warm start's value beside it. Property (c) says
the discrete policy is nested *exactly* as `tau → 0`; this measures *which* discrete policy that is.

## 7. `--space` — and why it is nearly free

The **profile is space-free**: it is a distribution over candidates and the head's scores do not
depend on any read-out embedding. `--space` therefore changes only *which embedding the convex
combination is taken in*, and hence which space readings 1/4/5 are measured in.

* **`kphi` (default)** — `e_q = [k(φ_q), b(φ_q)]`, C1-final's space, so the headline comparison with
  C1-final is like-for-like.
* **`tower`** — item = the centred, popularity-stripped decoder row `Ew[i]`; concept =
  `build_concept_dirs`' direction `d_c[c]`. This is the space `c3b_adapter.TowerBridge` and the
  renderer's target actually live in, so it is the space in which a "semantic blend" is a semantic
  claim rather than a classification read-out. Built with the bridge's own control: averaging `Ew`
  over each concept's members must reproduce the env's `d_c` to < 1e-3.

Because the profile is space-free, **both spaces are computed in every run** from one cache of top-m
logits, and `--space` only names which is the headline. ⚠ **Declared asymmetry in the tower space:**
`build_concept_dirs` returns unit-normalised concept directions while item rows are not normalised,
so a cross-channel convex combination there is scale-inhomogeneous. Reported per channel, not
corrected — correcting it would be a different space from the renderer's.

## 8. What may NOT be claimed, whatever the run returns

* **Nothing about the five ingredients.** C5 widens the *supervision* from 5 numbers to ~1,000; it
  does **not** re-measure the gains. Node cohorts are min 200 / median 356 with 43 of 72 below 500,
  and no per-user gain sidecar exists for this tree, so split-half direction stability is **not
  computable** and the pilot's 0.656 (mean-removed) is carried as a **labelled prior**. The
  ingredient-instability prescription remains **UNMET**. Binding wording: *we train toward a measured
  profile; we do not claim the profile's top five are the right five.*
* **Nothing unconditioned on the population.** Every number is conditioned on the cold bank's
  **selected 60.6% of train** (users endorsing no anchor concept), a by-design consequence of Paper
  B's forest split, and must say so.
* **No value claim.** C5 measures profile fidelity and emission geometry. It does **not** measure
  whether emitting the blend helps a user, and no NDCG is reported on this cohort (the tower was
  trained on these users).
* **No test-set claim.** Paper B's test look was spent 2026-08-29; C5 is a Paper C object and may not
  reuse it.
* **Reading 1a may never be quoted without its warm-start null.** Third time a control decided a
  headline in this workstream (C3B's directionless C2, C4B's reversed twin, C1-final's warm start).
* If gate 3 or gate 4 fails, **the run fails regardless of readings 1/2/6.**

## 9. Where the design as briefed did not survive contact with the code

1. **Convexity bounds the norm ABOVE only.** A convex combination of five vectors can be *shorter*
   than every one of them. So the registered assert is `‖z_hat‖ ≤ max_j ‖e_{q_j}‖`; where it sits
   inside `[min_j, max_j]` is reported, never asserted.
2. **The C5 emission is not of the form `[u(h), 1]`.** C1-final's emission lived on the affine slice
   `b = 1` of the homogeneous space and therefore had an *analytic ceiling* on `cos(z_hat, target)`
   (`c1_pilot.ceiling`). C5's emission is a genuine point in the span of bank rows: its last
   coordinate is `Σ p_j b(φ_{q_j})`, not 1. The ceiling therefore does not apply and is not reported;
   the comparison to C1-final's *fit* numbers is between an unconstrained and a constrained
   parameterisation, and should be read that way.
3. **`tau_emit` had to be introduced.** The brief specified the emission but not its temperature, and
   the emission is *entirely* determined by it (from on-grid argmax to uniform centroid). It is
   declared by the same measured rule as `tau_t`, fitted on trained states only, one global scalar,
   and disclosed as the only read-out constant fitted in the run.
4. **`--space` turned out to be an evaluation-time flag, not a training-time one**, because the
   profile loss is space-free. Both spaces are therefore computed in one run instead of two.
5. **The registered target is near-uniform over the full table** (§3). This is a real property of the
   declared rule meeting a 1,000-long table with ~half the pool tied at `max(gain,0) = 0`, not a bug,
   but it moves the loss's information into the ranking. Two declared, unrun levers exist if the
   author wants them: `--alpha` (down-weight the blend term) and `--kl_topk` (which **truncates
   supervision and needs sign-off**).
6. **`p*`'s zero-clamp creates a large tie block.** `max(gain, 0)` maps ~516 of ~996 candidates to
   exactly the same target probability. Reading 6's Spearman therefore averages ties explicitly (a
   naive argsort would invent an order inside that block and hand every baseline a spurious
   correlation). A rank-based target would avoid the block entirely and is *not* registered here.
</content>
