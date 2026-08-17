# L2a — THE TWO-TEACHER FOREST

> **STATUS: NOT RUN YET (2026-08-17). Gated on L1's cell 1. Every section below is PENDING.**
> Design sheet: `docs/design/PAPER_B_POLICY_DESIGN.md` Part L.2; Part K.6 is why per-user labels are
> gone for good. Harness `src/instrument/forest_policy.py`. **Validation only; `test_touched=false`.**

## ★ THE FALSIFIER'S VERDICT — INTERFERENCE

**PENDING.** If fidelity to T1 (stated genre) costs fidelity to T0 (the cold tree), the single-policy
claim fails at two teachers and scaling the forest is dead. Both fidelities are reported together at
every checkpoint, always.

## The design law being obeyed

A conditioning grouping is admissible only if (a) its signature is **observable at ask-time** — else
the label is clairvoyant and unlearnable (K.6) — and (b) its groups are **large enough to measure
honestly** (≥ ~500 users; below 200 is the −0.724 noise regime).

| teacher | signature | label | status |
|---|---|---|---|
| T0 the cold tree | nothing volunteered | the tree's next question at that state | banked (E1's trajectory bank, 140,768 train users) |
| T1 stated genre | "my favourite genre is G", encoded as an answered question per L.1 | the G-group's own greedy item order at that depth | banked (`groupgreedy_genre15.json`) |
| ✗ inferred-genre-of-items | — | — | DEAD (V: null in every big group) |
| ✗ per-movie | — | — | DEAD (n = 1, the clairvoyant trap) |

## Declared before the run

T1's orders are **item-only** — the genre question was free in that experiment, so each group
searched 15 items. The tree spends most of its budget on concepts. T1 is therefore a teacher over a
poorer action space, and at genre-seeded states the tree may well **score better than T1 does**. That
does not make the forest wrong (its claim is *exact where a teacher is defined*), but the score table
must show policy, T1's own order and the tree side by side at those states — and it does.

T1's own measured strength, per group, at K=15 items against the policy-B items order on validation
(`groupgreedy_genre15_VAL_policyB.json`) — only the groups at or above the 500-user cohort floor can
be teachers:

| group | n (validation) | full vs static items | tail vs static items | teacher? |
|---|---|---|---|---|
| Drama | 5,481 | +0.00538 | **−0.00280** | yes |
| Action | 1,582 | +0.02216 | +0.01515 | yes |
| Comedy | 1,475 | +0.02292 | +0.01399 | yes |
| Adventure | 534 | +0.03934 | +0.02484 | yes |
| Thriller | 451 | +0.03635 | +0.01530 | **no — below the floor** |
| Sci-Fi and 12 smaller | ≤161 each | large but unmeasurable | — | no — the noise regime |

Drama is 55% of the cohort and is the weakest teacher of the four, negative on the tail. That is a
known property of this teacher (memory: *genre-conditioned greedy shrinks to a null on the big
group*), not something L2a discovers.

The genre orders were **searched on all 10,000 validation users**, so HOLDOUT is held out from the
policy, not from the teacher. Every comparison here is policy-vs-that-same-teacher, so both sides
carry the teacher's selection equally; no comparison treats a group order as held-out-clean.

## Composition (printed before any time was spent on it)

The genre partition **reproduces all 18 banked group sizes exactly** — a free chain check on the
teacher cohort. Four groups clear the 500-user floor and become teachers, covering **90.72%** of the
validation cohort. Each stated genre has a literal genome-tag concept, so it can be asked as a
question and encoded as an answered one.

| teacher | concept | n (all) | FIT users | HOLDOUT users | FIT states |
|---|---|---|---|---|---|
| Drama | 292 | 5,481 | 4,417 | 1,064 | 66,255 |
| Action | 15 | 1,582 | 1,255 | 327 | 18,825 |
| Comedy | 203 | 1,475 | 1,171 | 304 | 17,565 |
| Adventure | 25 | 534 | 421 | 113 | 6,315 |
| **total** | | **9,072** | **7,264** | **1,808** | **108,960** |

Not taught: Thriller (451 users) and 12 smaller groups — below the floor, so they keep the tree's
behaviour by default. Firewall: 8,000 FIT / 2,000 HOLDOUT by user, seed 98765 (fit sha1 `a0fd82f2a8b5`,
holdout sha1 `3300ad7182e7`), recomputed here rather than read from a file.

Incidental, and worth a line in the chapter: **the tree's root question is the concept
"better than expected"** — not a genre. The cold interview opens on a reaction, and the stated-genre
teacher opens on a category; the two teachers really do partition the state space differently.

## Gates (pre-registered, L.2)

- **spanning** — cold-state behaviour ≥ E1, exact: **PENDING**
- **per-teacher fidelity** — top-1 against each teacher's own order at its own signature: **PENDING**
- **behavioural** — on fresh seeded HOLDOUT states, policy vs the matching teacher's own order, null
  or better: **PENDING**
- **shuffle control** — teacher assignments permuted across signatures: **PENDING**
- ★ **interference** — T0 and T1 fidelity together: **PENDING**
