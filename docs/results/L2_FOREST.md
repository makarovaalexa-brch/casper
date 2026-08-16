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

The genre orders were **searched on all 10,000 validation users**, so HOLDOUT is held out from the
policy, not from the teacher. Every comparison here is policy-vs-that-same-teacher, so both sides
carry the teacher's selection equally; no comparison treats a group order as held-out-clean.

## Composition

**PENDING** — teachers, cohort sizes, FIT/HOLDOUT split, state counts.

## Gates (pre-registered, L.2)

- **spanning** — cold-state behaviour ≥ E1, exact: **PENDING**
- **per-teacher fidelity** — top-1 against each teacher's own order at its own signature: **PENDING**
- **behavioural** — on fresh seeded HOLDOUT states, policy vs the matching teacher's own order, null
  or better: **PENDING**
- **shuffle control** — teacher assignments permuted across signatures: **PENDING**
- ★ **interference** — T0 and T1 fidelity together: **PENDING**
