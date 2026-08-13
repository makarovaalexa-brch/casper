# What does each selector in the battery actually optimise?

Audit 2026-08-10, read off the code with line citations (agent pass, spot-checked). Prompted by an
author objection that reframes Acts 1–2: *a fixed order is an adaptive policy that ignores its inputs,
so the optimal adaptive policy cannot be worse than the optimal fixed one.* Any measured loss is
therefore a statement about the selectors we ran, never about adaptivity. This table is what that
statement turns out to be.

## Headline

**1 of 21 distinct selection rules optimises the quantity the field reports.** That one is ours
(`greedy_certified`, run on three action pools = 3 of the 23 table rows). Every published or named
prior optimises a surrogate: squared error (3), posterior variance / design criteria (4), information
gain about a latent or a cluster label (3), a reward/UCB index over predicted *rating* (6), or a static
heuristic score (4).

**12 of 23 rows are VALUE-FREE** — the rule adapts to *which* questions were answerable and never to
*what* was answered. Over half the battery cannot personalise on answer content even in principle.

## The table

| # | selector | what it optimises | uses a ranking metric / held-out targets? | value-free? |
|---|---|---|---|---|
| 1 | `golbandi_tree` | within-node squared error of rating prediction (Eq. 1) | no | no (route branches L/H/U) |
| 2 | `glm_ucb` | UCB index over predicted rating | no | no |
| 3 | `thompson` | sampled posterior utility | no | no |
| 4 | `bald` | posterior variance of latent utility | no | **YES** |
| 5 | `aopt` | trace reduction of posterior covariance | no | **YES** |
| 6 | `conlinucb_mcr` | key-term posterior width / item UCB | no | key-term half only |
| 7 | `conlinucb_bs` | uniform spanner draw / item UCB | no | key-term half only |
| 8 | `conlinucb_bs_static` | uniform spanner draw, every turn | no | **YES** (and user-independent) |
| 9 | `clime` | key-term ∥ smallest eigendirection of `M` | no | no (width trigger is value-dep.) |
| 10 | `conucb` | key-term-attributable estimation-error reduction (Eq. 8) | no | key-term half only |
| 11 | `eddi` / `eddi_ig` | answerability-weighted KL movement of the latent posterior | no (target term deliberately dropped) | no |
| 12 | `dre` | squared reconstruction error of the full profile from k seeds | no | **YES** (static) |
| 13 | `dpe_concept` | Thompson index over concept answer value | no | no |
| 14 | `dpe_oracle` | the user's own rating of the candidate (Eq. 3) | reads fold-in ratings; no metric | no |
| 15 | `igcn` | information gain about the user's cluster label | no | no |
| 16 | `greedy_slim` / `qgslim` | SLIM elastic-net squared-loss decrease | no | **YES** (static) |
| 17 | `fixedorder:entropy0` | entropy with "not rated" as an 11th band | no | **YES** |
| 18 | `popularity` | train interaction count | no | **YES** |
| 19 | `helf` | harmonic mean of norm. entropy & norm. log-freq | no | **YES** |
| 20 | `pure_entropy` | rating-band entropy over raters | no | **YES** |
| 21 | `fixedorder:greedy_items` | **cohort-mean held-out NDCG@10** | **YES** | **YES** (static) |
| 22 | `fixedorder:greedy_concepts` | **cohort-mean held-out NDCG@10** | **YES** | **YES** (static) |
| 23 | `fixedorder:greedy_combined` | **cohort-mean held-out NDCG@10** | **YES** | **YES** (static) |

## Notes that matter for the write-up

**`conlinucb_bs_static` is a third member of the BALD/A-optimal class.** With `b_schedule="always"` and
the spanner draw seeded on `[seed, turn]` only, it is *one static random order over the spanner*, the
same for all 10,000 users — which is why the concept-bank run measured it conducting **1 distinct
interview**. It should be named alongside `bald` and `aopt` as structurally non-adaptive, not reported
as an adaptive arm.

**Row 5 (`aopt`) ports the recommender-side A-optimal rule, i.e. Anava et al. WWW'15 — and identically
Sepliarskaia et al. RecSys'18, who derive the same rule from a flat pairwise loss.** What
`AOptimalSelector._gain_vec` computes is the greedy Sherman–Morrison trace reduction of the **user-side
posterior covariance** over the frozen iALS item-factor basis, `argmin_i tr(A_{t+1}(i)^-1)`, with no
target-pool weighting. The class docstring also names Yu, Bi & Tresp (ICML'06) as the criterion's
origin; that is a lineage credit, not what is implemented — TED weights predictive variance over a
target pool and this trace is untargeted, so the docstring's "(transductive)" is loose. **When the arm
is labelled by venue anywhere, the venue is WWW'15.**

**The conversational trio is value-free on its key-term turns only.** Under `b(t)=5⌊log t⌋` that is a
deterministic subset of turns (3 and 8 within our budgets), so those arms alternate between a
value-free key-term rule and a value-dependent item rule on a fixed schedule.

**EDDI's target-aware term was dropped deliberately**, and the header records why: using held-out
targets in selection "would leak the evaluation targets into the question selection".

> ### ⚠ CORRECTION, 2026-08-13 — the sentence that used to follow is RETRACTED.
>
> This paragraph previously concluded: *"So its row is not an oversight — it is the honest version of
> that method under our protocol."* **That is false, and it was already false when it was written.**
> The SNAP-1 commit `b4a4783` (2026-08-05) had established that **Eq. 9's second term samples its
> hypothetical answers from the model and leaks nothing** — so the term was dropped for a reason that
> does not hold, and the leakage justification in `eddi_selector.py`'s header is wrong too.
>
> Three things follow, none of them cosmetic, all recorded in
> `docs/results/EDDI_IMPLEMENTATION_STATUS.md`:
> 1. **The reduction is not defensible as "honest under our protocol".** It is a documented
>    deviation whose stated reason has been withdrawn. What plays the target `x_phi` in cold-start
>    ranking is a real modelling question (TODO 1.3(a)) and it needs a design sheet, not a footnote.
> 2. **The authors argue against the first term alone** (App. C.3): it is the BALD-style gain. We
>    ship the reduction they warn about, under their name — which is why the row is relabelled
>    *latent-only info-gain* in `tab:battery`.
> 3. **The native snap fails.** `eddi_latent` (ours) pooled average rank **3.402** vs **random
>    3.236** on Ma et al.'s own Fig-3 datasets — the reduced acquisition ranks below random ordering.
>
> ★ **A third, independent defect that this row's "answerability-weighted" description conceals:**
> the retained weight `p_i` is a Mult-VAE^PR **softmax across items**, i.e. a relative next-draw
> probability, not the per-variable marginal `p(x_i | x_O)` Ma et al.'s factorised likelihood
> (their Eq. 2) requires. The docstring records `corr(log p, log R) = 0.965` at the empty state —
> the arm is close to a most-probable-item selector — **but that diagnostic has no reproducible
> artifact and must be re-derived before it is quoted as a measured number.**
>
> **What has since been discharged, and what has not.** TODO 1.3(b) — the three-outcome response
> model, which is the defect the *two-outcome* collapse of "not seen" and "disliked" causes — was
> run at contract scale on 2026-08-13 → `docs/results/EDDI_THREE_OUTCOME_RESULT.md`. Row 11's
> "target term deliberately dropped" column entry is **still true and still open**: fixing the
> response model does not implement Eq. 9, and does not touch the weight.

**The heuristic orders were never validated on a ranking metric by their own authors.**
`interview_strategies.py:24-26`: Rashid measured rating-prediction MAE and Golbandi RMSE; we measure
NDCG@10, so "the published ranks transfer as PRIORS, not as certified numbers."

## Consequence for the two proposed follow-ups

**(a) "Refit Golbandi on our objective" is a STRUCTURAL REWRITE, not a local change.** The scorer is one
function (`_e2`, `golbandi_tree_selector.py:195-200`) with one call site, so the *interface* is local —
but three things break together:
1. **No held-out targets exist in the growth contract.** The tree is grown on `g_train` only and never
   sees a test user; cohort-mean NDCG of a group needs a fold-in/target split per training user plus a
   recommender to score them.
2. **The subtraction trick dies.** Eq. 3/4's affordability comes from `sU = s_tot − sL − sH`: six sparse
   products give all three children in one pass. NDCG@10 of a group is not a sum of per-item sufficient
   statistics and does not decompose that way.
3. **Cost.** At the greedy builder's measured 13.54 s/eval, one root split over 18,359 candidates is
   **~69 CPU-hours** before any deeper level.

The routing, checkpointing, leaf profiles and `release()` all survive; the stopping test (`err < e2_node`,
a minimisation) would have to invert.

**(b) Group-split greedy: cost is bounded, not known.** The recorded build is **723.6 min (12.06 h)** for
all three arms, **3,173 unique evals at 13.543 s each** on 10,000 build users; items-only alone is
**852 evals ≈ 3.2 h**. `n_evals` is set by pool size and `L`, **not** by cohort size, so G groups cost
≈ G × 852 evals for an items-only order. Whether per-eval cost falls to 13.54/G when each group holds
10,000/G users is **UNVERIFIED**:
* **optimistic** (per-eval linear in cohort size): ≈ 3.2 h total for items-only across all G groups;
* **pessimistic** (fixed overhead dominates): ≈ G × 3.2 h, i.e. ~16 h at G = 5.

**One 2,000-user smoke of `greedy_certified` on the items-only pool settles the scaling in ~40 min**, and
should be run before committing to the full design.

## Why this is a contribution and not a complaint

It reframes Act 1 from "adaptive loses to static" — which cannot be true as a general claim, since
static is nested inside adaptive — to something defensible and, we think, more interesting: *the
cold-start elicitation literature optimises surrogates for the thing it reports, and half of it cannot
condition on answer content at all.* That is a statement about the field's evaluation practice, it is
verifiable line by line, and it explains the battery result without appealing to anything about
adaptivity.
