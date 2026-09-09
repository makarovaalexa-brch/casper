# DS-A12 — is the concept channel's gain answerability, or breadth per answer?

**Status: PRE-REGISTERED, arms fixed 2026-09-09 after a code trace. Nothing has run.**
Author instruction 2026-09-09: "ok, do that", referring to the control both delta reviewers
independently asked for (`new_chapters/paperB_ecir/reviews_v4/review_A2_delta.md`, `review_D2_final.md`).

## The question, stated so a result can contradict it

Paper B reports that enlarging the question bank from films to films-plus-concepts is worth
**+0.01430 full / +0.02199 tail** at sixteen questions with no conditioning of any kind
(`docs/results/CONDITIONED_ARMS_ON_TEST.md` §3), larger than the tree's entire conditioning gain
(+0.01035 / +0.00705). The paper's practical advice rests on it: change what the system asks about
rather than how it decides what to ask. The paper attributes the gain to **answerability**: sixteen
concept questions collect 12.36 answers where sixteen film questions collect 3.11.

Both round-2 reviewers rejected the attribution as unearned, for the same reason: a concept answer
also *carries* more than an item answer, since it summarises many films. Two explanations, one number:

- **H-answerability.** Concepts win because a cold user *can answer* them at all.
- **H-breadth.** Concepts win because each answer *carries more*.

Under H-answerability the advice is non-obvious and transfers. Under H-breadth it collapses into
something ordinary. The chair would ask for the control before camera-ready; the strict reviewer
names it as one of two conditions for moving past borderline. Both say either outcome is publishable.

## What the code trace settled, and what it kills

**The control as the reviewers described it is NOT CONSTRUCTIBLE, and proposing it would be an error.**
The chair asked us to "re-run the bank comparison with item questions scored under the same
implicit-watch answerability predicate the concept channel uses, instead of *rated*". That knob does
not exist, for a reason that is a property of the data and not of our implementation:

- The item predicate is `i in lvl[u]` — the user rated that film at any band —
  (`src/baselines/interview_strategies.py:115`, `src/baselines/adaptive_harness.py:477`,
  `src/instrument/strategy_channel_suite.py:522`), with the answer value the half-star level
  (`interview_strategies.py:45`).
- The concept predicate is `(n_c >= 2) AND (e_c >= 1.5)` (`src/instrument/signed_answers.py:130,138`,
  `TAU_REF = 1.5` at `:29`), consumed as `foldable = Fb & ansm` (`strategy_channel_suite.py:461`).
- **`n_c` is itself counted from the rated matrix** (`signed_answers.py:43`, `nc = Xbin @ Mm`), and
  the module says so: *"'watched' = rated at any band"* (`signed_answers.py:6`). The only ratings
  source is `data/movielens/ratings.csv` (`src/baselines/graded_data.py:30`). **ML-25M ships no watch
  log, so "watched" and "rated" are the same event for a film.**

Consequently the item predicate is *already the widest one available*, both channels answer from one
and the same evidence set, and there is no watch-versus-rate asymmetry to remove. **The real
asymmetry is threshold and aggregation:** an item needs one rating of that exact film; a concept
needs two rated member films out of the hundreds it aggregates (bank rule: genome relevance ≥ 0.5 and
≥ 30 member items, `src/baselines/conversational_bandit_selectors.py:415-419`). The concept's
answerability advantage *arises from* its breadth. The two hypotheses are not independent mechanisms.

That is exactly why they can still be separated **as channels of value**, and it is what this run does.

## The arms

Value decomposes without assumption into a rate and a per-answer worth:

    lift(arm, K)  =  answers(arm, K)  ×  worth-per-answer(arm, K)

Both factors are read off the same banked run that produced the published number, so neither can be
chosen after seeing the other.

**Arm 1, the decomposition.** For each of the three banked orders (items-only, concepts-only,
combined) at each budget, report lift over the answer-free reference, mean answers collected, and
lift per answer, on both axes.

**Arm 2, the load-bearing comparison: answer-matched.** Hold the number of answers roughly fixed and
ask which channel bought more with them. Items reach 3.108 answers only at K=16; concepts pass that
between K=2 (1.751) and K=4 (3.568), so **both neighbours are reported** and the reading does not
depend on an interpolation. Paired percentile bootstrap over per-user differences, 10,000 resamples
clustered over users, seed 98765.

**Arm 3, per-user matched band.** Cohort means can hide a mismatch, so the same contrast is repeated
on the subset of users whose concept answers at K=4 fall within 0.5 of their own item answers at
K=16. Reported only if that band holds at least 200 users.

**Nothing is refitted.** The orders are the banked policy-B validation-selected sequences; the cohort
is the same 10,000 test users; per-user scores come from the existing npz. The only new computation
is per-user answer counts, which the source run stored as cohort means only.

## Registered outcomes, written before the run

- **Concepts win on rate and LOSE on worth per answer** → H-answerability is supported and H-breadth
  is *contradicted*, not merely unsupported. Section 6 keeps its mechanism sentence, now earned, and
  the Limitations concession is narrowed to what the data actually shows.
- **Concepts win on both** → H-breadth is real, Section 6's mechanism sentence is wrong and is
  replaced by the split, and the advice is restated as "ask broader questions".
- **They cancel** → the dead heat between the single channels at K=16 is explained, the +0.01430 is a
  complementarity result rather than a channel-superiority result, and the paper must say so.

All three change the paper, so the run is admissible under the design-sheet rule.

## Gates, both of which must pass before any new number is read

1. **Published cells.** The banked per-user vectors must reproduce, to five decimals: items
   0.24470/0.10171, concepts 0.24467/0.11607, combined 0.25900/0.12370 at K=16, and the contrast
   combined − items = +0.01430 full / +0.02199 tail.
2. **Answer counts.** The per-user answer counts recomputed here must reproduce every cohort mean in
   `channel_orders_policyB_test.json` to 1e-6, on both channels at every budget. This is the gate
   that matters, because the whole decomposition divides by these numbers.

Also asserted: identical row order to the banked npz, or per-user pairing would be silently wrong;
`contract_selftest` and `leak_assert` on the test cohort as in the source run.

## Fixed protocol

Cohorts: fitted on the 10,000 validation users, scored on the disjoint 10,000 test users, seed 98765.
Metric: full AND tail NDCG@10, learned decoder bias, full catalogue, no sampling, no popularity floor.
Budgets 1, 2, 4, 8, 16, because the budget-dependence is part of the evidence.

## Hazards recorded before launch

- **The resume guard does not cover the answer table.** `mask_contract.pool_contract` guards the NDCG
  candidate pool, not the answer oracle, so a run with a modified answer rule would happily resume a
  sidecar written under the old rule. This run does not modify the answer rule, but a follow-up that
  narrows the concept value **must not reuse a label** and must not resume an existing sidecar.
- **`e_c` mixes two populations.** `e_c = n_u^{all-bands} × pexp^{likes-only}`
  (`strategy_channel_suite.py:455`, `pexp = (ctx.cnt @ Mm)/cnt.sum()` with `ctx.cnt` from the binary
  train matrix). Whether that was intentional is not determinable from the code or its comments. It
  does not affect this run, which changes no predicate, but it is a real mixed-population
  construction and should be resolved before any paper sentence leans on the exact value of `e_c`.
- **Two incompatible concept-answerability arrays exist.** `tradeoff_ledger.py:90-95` builds a second
  one (`counts >= 2` over the binary fold-in, no `e_c` gate) used by older ledger rungs. The
  certified path recomputes from `ctx.allb` and does not read it. Confirmed for this run; flagged in
  case a currently-cited number flows through a rung that does.
- HARD RULE 1: no truncation or sampling of users, items, questions or candidates.
- HARD RULE 10: code committed before launch.
- HARD RULE 11: one heavy job at a time, free RAM checked immediately before launch. The source run
  cost 213 s, so this is a light job, but the box was at 1.74 GB free at 21:15.

## Reporting

Numbers to `docs/results/A12_ANSWERABILITY_VS_BREADTH.md`, the finding to a memory note, status to
`docs/STATE.md`. The paper edit, if any, is decided after the numbers, not before.
