# A12 — the concept channel's gain is the answer RATE, not breadth per answer (2026-09-09)

**Status: COMPLETE.** Design `docs/design/DS_A12_ANSWERABILITY_VS_BREADTH.md`. Code
`src/instrument/a12_answerability_vs_breadth.py` and `src/instrument/a12b_burn_value.py`, both
committed before launch (`7295f9e`, `ec4d3d1`). Artifacts
`experiments/battery/a12_answerability_vs_breadth.json` + `_per_user.npz`, `a12b_burn_value.json`.
A pure re-analysis of banked test interviews: no refit, no new selection, no new scoring of any arm.

**Why it exists.** Both round-2 reviewers refused the paper's attribution of the +0.01430 bank gain
to *answerability*, on the ground that a concept answer also *carries* more than an item answer.
Two explanations, one number, and the battery as published could not separate them.

## The reviewers' proposed control does not exist, and it is a property of the data

The chair asked for item questions to be re-scored under "the same implicit-watch answerability
predicate the concept channel uses, instead of *rated*". **That knob cannot be built.** The item
predicate is `i in lvl[u]`, the user rated that film at any band
(`interview_strategies.py:115`, `adaptive_harness.py:477`, `strategy_channel_suite.py:522`). The
concept predicate is `(n_c >= 2) AND (e_c >= 1.5)` (`signed_answers.py:130,138`). But **`n_c` is
itself counted from the rated matrix** (`signed_answers.py:43`), and the module says so at
`signed_answers.py:6`: *"'watched' = rated at any band"*. The only ratings source is
`data/movielens/ratings.csv`; ML-25M ships no watch log.

So **"watched" and "rated" are the same event for a film**, both channels answer from one and the
same evidence, and the item predicate is already the widest available. The real asymmetry is
threshold and aggregation: an item needs one rating of that exact film, a concept needs two rated
members out of the hundreds it aggregates. **The concept channel's answerability advantage arises
*from* its breadth**, so the two hypotheses are not independent mechanisms — which is precisely why
they must be separated as channels of *value* instead.

## Both gates PASSED

1. Banked per-user vectors reproduce every published cell to five decimals, including
   combined − items at K=16 = +0.01430 full / +0.02199 tail.
2. Per-user answer counts recomputed here reproduce every banked cohort mean, on both channels at
   every budget. *(The banked field is written `round(x, 3)` at `channel_orders_on_test.py:126`, so
   the gate is against the full precision the artifact retained, not a loosened tolerance.)*

## The decomposition: lift = answers × lift-per-answer, full NDCG@10, 10,000 test users

Lift is over the answer-free reference 0.15855.

| arm | K=16 lift | answers | lift per answer |
|---|---|---|---|
| items-only | +0.08615 | 3.108 | **0.02771** |
| concepts-only | +0.08611 | 12.360 | **0.00697** |
| combined | +0.10045 | 10.757 | 0.00934 |

**An item answer is worth 3.98× a concept answer, and the concept channel collects 3.98× as many.**
The two factors cancel to three decimals, which is exactly why the single channels are a dead heat at
sixteen questions (−0.00003 [−0.00315, +0.00310]).

## The load-bearing comparison: answer-matched

Hold the answers roughly fixed, ask which channel bought more. Paired percentile bootstrap, 10,000
resamples clustered over users, seed 98765; `*` = interval excludes zero.

| pair | answers | full | tail |
|---|---|---|---|
| items@K16 − concepts@K4 | 3.108 vs 3.568 | **+0.03059\*** [+0.02736, +0.03377] | +0.01594\* [+0.01274, +0.01922] |
| items@K8 − concepts@K2 | 1.790 vs 1.751 | **+0.02734\*** [+0.02430, +0.03037] | +0.01586\* [+0.01289, +0.01882] |
| items@K4 − concepts@K1 | 1.112 vs 0.752 | **+0.02297\*** [+0.01989, +0.02614] | +0.01394\* [+0.01089, +0.01705] |
| per-user matched band (n=1,430) | matched per user | **+0.04956\*** [+0.04041, +0.05875] | — |

The `items@K8 − concepts@K2` row is the cleanest: the answer counts differ by 0.04 and the item arm
still wins by +0.02734. **At matched answers the item channel wins every comparison, on both axes.**

## The burn is not the explanation (A12b)

The item arm reaches 3.108 answers by asking sixteen questions and burning ~12.9, and a burned film
question folds an informative exposure token. Users who answered *nothing* receive burns only, so
their lift **is** the burn value, measured rather than estimated:

| arm | K | zero-answer users | burn-only lift | share of the arm's lift |
|---|---|---|---|---|
| items | 16 | 1,754 (17.5%) | +0.00116 | **1.3%** |
| items | 8 | 2,627 (26.3%) | +0.00063 | 1.0% |
| items | 4 | 3,616 (36.2%) | +0.00094 | 2.1% |
| items | 2 | 5,779 (57.8%) | −0.00073 | −2.4% |
| items | 1 | 7,017 (70.2%) | +0.00298 | 17.1% |
| concepts | 1 | 2,475 (24.7%) | +0.00499 | 22.4% |

**A burned film question buys almost nothing beyond one question in.** Net of burn, the item arm's
lift per answer at K=16 moves only 0.02771 → 0.02734, and the answer-matched contrast at the one
budget where both burns are measurable is +0.02702\* [+0.02394, +0.03019]. The reading survives.

Two incidental findings worth keeping: an item *exposure* token is worth less than a concept
*refusal* token at the same budget (+0.00298 against +0.00499 at K=1), and **at two concept questions
only 11 of 10,000 users answer nothing**, against 5,779 for two film questions. The second is the
answerability effect in its rawest form.

## Verdict against the pre-registered outcomes

The registered reading was: *concepts win on rate and lose on worth → H-answerability supported and
H-breadth contradicted*. **That is what happened, on every comparison.**

- The concept channel's entire advantage is the rate at which it collects answers.
- Each concept answer is individually **less** informative than a film answer, by about 4×. The
  breadth hypothesis is not merely unsupported, it runs the wrong way: aggregating hundreds of films
  into one question *dilutes* what the answer says, and the channel wins anyway because it is
  answerable four times as often.
- The +0.01430 is therefore a **complementarity** result, not a channel-superiority result: the
  combined order mixes 1.5 film answers with 9.2 concept answers and beats both single channels,
  which tie with each other.

## What goes in the paper

Section 6's "Answerability carries most of the mechanism" can be stated outright and earned, and the
Limitations concession must be **narrowed**, because the caveat as written concedes something the
data contradicts. Replace "that channel's advantage is answerability together with breadth, not
answerability alone" with the measurement: at a matched number of answers the film channel buys more
(+0.02734 at ~1.8 answers), so the concept channel's advantage is its answer rate and not the reach
of any one answer. One row of the decomposition table carries it.

⚠ **Do not state the reviewers' proposed control as "not run".** It is not constructible on this
dataset, and the paper should say why in one clause if it says anything at all.
