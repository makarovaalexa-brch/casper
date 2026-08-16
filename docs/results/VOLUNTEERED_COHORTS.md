# EXPERIMENT V — VOLUNTEERED COHORTS: does evidence the user *offers* route as well as an answer to a question we *ask*?

**Status: RUNNING (started 2026-08-16). Validation only — the 10,000 canonical TEST users are not
built, not loaded, not named. Numbers below marked `PENDING` are not yet measured.**

> **AUTHOR SAID (2026-08-16, verbatim): "ok, go ahead."**

Code: `src/instrument/volunteered_cohorts.py` (+ `volunteered_cohorts_report.py`), committed before
the run (HARD RULE 10). Artifacts: `experiments/battery/volcohorts/`.

---

## THE QUESTION

The **genre arm** (`group_greedy_genre.py`) splits users by their answer to **one asked genre
question**, gives each group its own greedy-optimised static order, and that conditioning is worth
about as much as the whole depth-5 tree. Experiment **R** then showed the distilled policy is
**inert** when a user volunteers evidence instead, and the z-only ablation showed that when forced to
respond it responds **badly**. Neither measured whether there is anything **to** respond to.

**V measures the ceiling, not a policy.** It hands each volunteered-evidence cohort its own optimised
order and asks what that conditioning is worth, on the same ruler as the genre arm.

---

## ★ THE LIMITATION, STATED UP FRONT

**V measures a CEILING.** Every cohort is handed an order fitted on the very users it is scored on.
It does **not** show that a learned policy can capture any of it. The gap between this ceiling and
what a policy achieves is the chapter's open question and must not be blurred with the ceiling
itself.

Two consequences that shape how the table must be read:

1. **On the selection cohort a group's own order is ≥ the global order by construction**, because the
   global order lies inside every group's search space. So an in-sample gain is ≥ 0 automatically and
   only its **magnitude** carries information — against *what* is the question, which is why the
   **placebo arm exists**.
2. **The volunteered evidence is treated as FREE.** No turn is charged for it; the comparison is 15
   asked items vs 15 asked items. The genre arm made the identical generous assumption about its
   genre answer. Every number here is an upper bound.

---

## DESIGN — reuse the genre machinery, change only what defines a cohort

| | genre arm (banked) | **V (new)** |
|---|---|---|
| cohort | answer to one **asked** genre question: most frequent MovieLens genre over the user's **liked fold-in items** | the same genre signature computed over the **k=4 items the user VOLUNTEERED** |
| evidence folded before the interview | none (empty state) | **the 4 volunteered items** (R's state structure) |
| candidate pool | items-only, bank `pop`, top-500 | identical |
| interview length | L = 15 items | identical |
| budgets scored | K ∈ {1, 2, 4, 8, 15} | identical |
| small-group threshold | 200 (flagged, never merged, never dropped) | identical |
| arms | `all_groups`, `fallback_200` | identical |
| comparator | `greedy_static_policyB_items` (16-item global order), first K | identical, **seeded the same way** |
| inference | paired percentile bootstrap, 10,000 resamples, seed 98765, over per-user differences | identical |

**The draw (HARD RULE 1, declared).** k = 4 items per user, **without replacement,
popularity-weighted** (p ∝ train rating count), from that user's rated **fold-in** items —
`policy_resume.draw_volunteered`, **imported, not re-implemented**, one `default_rng(98765)` stream,
users in canonical row order. It is asserted field-by-field against experiment R's banked k=4 draw
block and is **bit-identical** to it (mean seed size 4.0, 62.8875% of volunteered items are likes).

**The one structural difference that is forced.** "Volunteered" means the items are folded as
ordinary evidence *before* the interview. So **every arm — group orders and comparator alike — is
scored from the seeded state**: user *u*'s folded item sequence is `dedup(vol_u + asked[:K])`. A
re-asked volunteered item folds **once** and the turn is burnt (R's semantics,
`policy_resume.build_seqs`). Because each user carries a different sequence the scorer is
`genv_eval_multi` rather than `genv_eval`; gate 3 below asserts these are the same ruler.

**The placebo arm is not optional (HARD RULE 5).** `--cohorts placebo` re-runs the identical
machinery on **random cohorts with V's exact size profile** (a permutation of the real label vector,
`default_rng(98765)`). That is the null level for "how much does splitting 10,000 users into groups
of these sizes and re-optimising buy when the split carries no information". **V minus placebo is the
reading; V alone is not.**

---

## GATES — all four passed before any search started

| gate | result |
|---|---|
| canonical m = 0 snap, policy B, 10,000 validation users | `0.15686538118354684` **exact** |
| the banked global items-only comparator reproduces its banked unseeded curve | **5/5 bit-identical** at K ∈ {1,2,4,8,16} |
| `genv_eval_multi` == `genv_eval` on an identical-sequence cohort | **bit-for-bit**, full and tail — the seeded scorer is the same ruler, not a second one |
| the volunteered draw == experiment R's banked k=4 draw | **field-by-field identical** |
| leak check: held-out target inside the grouping lookup | **0 / 10,000** |

---

## ★ RESULT 0 — THE COHORT COMPOSITION, DECLARED BEFORE THE ORDERS WERE BUILT

**20 groups over 10,000 validation users. 13 groups (576 users, 5.76%) fall below the 200-user
threshold; the `all_groups` arm runs every one of them.**

| group | n (V, volunteered) | % | n (genre arm, asked) | % | below 200? |
|---|---|---|---|---|---|
| Action | 2758 | 27.58% | 1582 | 15.82% | |
| Drama | 2228 | 22.28% | **5481** | **54.81%** | |
| Comedy | 1323 | 13.23% | 1475 | 14.75% | |
| Adventure | 1180 | 11.80% | 534 | 5.34% | |
| Crime | 1027 | 10.27% | 79 | 0.79% | |
| **none** | 517 | 5.17% | 0 | 0.00% | |
| Thriller | 391 | 3.91% | 451 | 4.51% | |
| Sci-Fi | 146 | 1.46% | 161 | 1.61% | YES |
| Mystery | 99 | 0.99% | 3 | 0.03% | YES |
| Animation | 90 | 0.90% | 44 | 0.44% | YES |
| Romance | 76 | 0.76% | 88 | 0.88% | YES |
| Children | 62 | 0.62% | 29 | 0.29% | YES |
| Horror | 41 | 0.41% | 51 | 0.51% | YES |
| Fantasy | 28 | 0.28% | 11 | 0.11% | YES |
| IMAX | 12 | 0.12% | 0 | 0.00% | YES |
| Film-Noir | 7 | 0.07% | 1 | 0.01% | YES |
| Documentary | 6 | 0.06% | 3 | 0.03% | YES |
| Musical | 3 | 0.03% | 3 | 0.03% | YES |
| War | 3 | 0.03% | 2 | 0.02% | YES |
| Western | 3 | 0.03% | 2 | 0.02% | YES |

**Three readings, before any score exists.**

1. **V's partition is much flatter than the genre arm's.** The genre arm's biggest group holds 55% of
   users; V's holds 27.6%. The mechanism the genre arm's test result exposed — *the gain lives in
   small distinctive groups and the big group is a null* — has more room to operate here, because
   V has no 55% group to dilute it.
2. **Only 517 users (5.17%) volunteer nothing likeable enough to route on** (no liked volunteered
   item), and 0 users' liked volunteered items lack a genre. Coarse volunteered evidence does produce
   a label for 94.8% of people.
3. **★ Part of the reshuffle is the tie-break, not taste.** With ~2.5 liked items out of 4, genre
   counts tie constantly and the rule breaks ties **alphabetically** — which is exactly the direction
   Action (+11.8 points), Adventure (+6.5) and Crime (+9.5) moved, and the direction Drama (−32.5)
   moved. The tie-break census quantifies this and is reported below; the honest statement is that
   V's partition is **partly a signature of the volunteered set and partly an artifact of the rule
   used to summarise it**. It is the same rule the genre arm used, which is the point — but the genre
   arm applied it to ~60 items per user, where ties are rare.

`PENDING` tie-break census.

---

## RESULT 1 — THE HEADLINE

`PENDING`

## RESULT 2 — PER-GROUP

`PENDING`

## RESULT 3 — REDUNDANCY

`PENDING`

## WHAT THIS SETTLES

`PENDING`
