# M2 — the screen: does a stated opener change which question should be asked next?

**Design sheet** `PAPER_B_POLICY_DESIGN.md` Part M (M2) · **code** `src/instrument/m2_screen.py` ·
**artifact** `experiments/battery/m2/m2_screen.json`. Ran 2026-08-17.
**VALIDATION ONLY — the 10,000 canonical TEST users were not built, loaded or named.**

## The question

A user states one of the six anchors mined in M1. Their evidence is folded, so they sit at a state
the cold tree never reaches. At that state, is any question better than the one the cold tree would
ask — its root, concept 114 — by enough to justify a teacher?

This was the front-loaded bet of the programme: if no opener changes the best next question, there is
nothing for a forest to teach.

## THE VERDICT: the falsifier did not fire

| | | |
|---|---|---|
| **pooled best − root** | **+0.00384** | **[+0.00061, +0.00712]** — separates from zero |
| pooled shuffle − root | −0.01699 | [−0.02061, −0.01343] |

n = 2,704 held-out users pooled across six anchors, paired percentile bootstrap, 10,000 resamples,
seed 98765.

**A stated opener carries information the tree's root does not already extract** — worth roughly
**37% of the tree's entire conditioning gain** (+0.01035 on test), concentrated in one anchor.

⚠ **Two qualifications that must travel with this number.** The pooled point estimate **+0.00384 is
marginally BELOW the pre-registered +0.004 bar**, so the interval excludes zero while the effect size
just misses the threshold set in advance. And `marvel`'s own shuffle came in at **+0.0021** rather
than negative — its interval includes zero, but it is the weakest control in the set.

## Per anchor

Selection on a fit half, evaluation on the held-out half — so the reported difference is not the
maximum of a search scored on its own data.

| anchor | concept | n | fit gain | **holdout** | 95% CI | shuffle |
|---|---|---|---|---|---|---|
| `marvel` | 581 | 734 | +0.02835 | **+0.01540** | **[+0.00320, +0.02787]** | +0.00205 |
| `teenagers` | 922 | 786 | +0.02104 | +0.00364 | [−0.00699, +0.01415] | −0.03415 |
| `dialogue driven` | 275 | 730 | +0.00492 | +0.00318 | [−0.00370, +0.01007] | −0.01212 |
| `inspiring` | 503 | 1,822 | +0.00964 | +0.00254 | [−0.00204, +0.00721] | −0.01054 |
| `gore` | 417 | 747 | +0.00422 | +0.00146 | [−0.00584, +0.00855] | −0.04766 |
| `art house` | 68 | 587 | +0.00307 | −0.00242 | [−0.01262, +0.00771] | −0.00479 |

**`marvel` separates on its own**, at a magnitude 1.5× the tree's whole conditioning gain: for a user
who says "marvel", a specific film is a decisively better opening question than the tree's root.

## ★ The methodological result: selection shrinks ~4.5×

Cohort-weighted **fit +0.00913 → holdout +0.00203**. Per anchor the shrinkage runs 1.5× (`dialogue
driven`) to **5.8×** (`teenagers`, +0.0210 → +0.0036), with one sign flip (`art house`).

Reporting `teenagers`' fit number would have claimed twice the tree's entire conditioning gain from a
single question. This is the `Spearman(n_users, gain) = −0.724` lesson in a new costume: **a maximum
selected on its own data overstates**, and the fit/holdout split is what makes the number honest.

## Design decisions, declared

- **The cohort is who STATES the anchor**, not who is secretly in its cluster — routing at deployment
  is by what a user says. Clusters were the device for *finding* anchors, not the signature.
- **The anchor is volunteered and costs no turn** in both arms; exactly one question is asked, so the
  budgets match. The anchor is masked from the candidate pool.
- **Power was declared before the run.** A cohort is ~700 users, so a paired interval is ~±0.013 and a
  +0.004 effect is undetectable per anchor. Hence the **pooled** test as the headline and per-anchor
  cells as description. No per-anchor null should be read as a teacher failing.
- The baseline is the **policy-B** cold argmax: concept 114, the tree's root and also the first pick
  of `greedy_static_policyB_combined.json`. (The policy-A static order opens with concept 630 — a
  different ruler, not used here.)

## Independent cross-check

The M3 grower, searching the full cohort by CELF with a different baseline, selected **concept 833
for `gore`** — the same question the screen picked on a fit half. Two procedures converging on one
question is reassuring about both.

## ⚠ OUTSTANDING CONTROL — do not put this result in the chapter until it runs

The screen shows the selected questions beat the root **for users who state an anchor**. It never
tested whether they also beat the root for **everyone**. If they do, the gain is not conditioning —
it is the root being beatable in general.

Expected clean, because concept 114 is the policy-B cold argmax (the tree's splitter criterion
reduces algebraically to the greedy argmax over mean NDCG). But that is an argument, not a
measurement. The test is cheap: evaluate each selected question against concept 114 at the **cold**
state on all 10,000 validation users, and confirm it wins only for anchor-stating users.

## The limitation the screen cannot address

The measurement is at a **one-question horizon**. It shows the opener changes which question is best
*next*; it does not show the advantage survives a sixteen-question interview — the cold tree may
reach the same information a few questions later, which is exactly what Phase 3 found for the deeper
labels (real signal against a shuffle, no value against the tree once the full interview ran).
**M3/M5 settle this** by measuring the grown teachers at K ∈ {1, 2, 4, 8, 16}.
