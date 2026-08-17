# L1 — THE AS-IF-ASKED ENCODING TEST

> **STATUS: RUNNING (2026-08-17). Every number below is marked PENDING until its cell lands.**
> Design sheet: `docs/design/PAPER_B_POLICY_DESIGN.md` Part L.1. Harness `src/instrument/policy_asif.py`
> (commit `c79b6b8`). Artifacts `experiments/battery/asif/`. **Validation only; `test_touched=false`.**

## THE FALSIFIER'S VERDICT — **CELL 1 PASSES, EXACTLY**

With the volunteered fact encoded as an answered question, the policy branches as the tree's
post-root arm **for 100.00% of the 10,000 users** — all 15 remaining questions, every user, no
exceptions. Under R's encoding (fold into `z`, history empty) the same policy follows that
continuation for **0.00%** of users: it re-asks the root it was just told the answer to and then
re-runs the tree from the beginning.

The as-if-asked encoding is therefore **correct and load-bearing**, and R's null was an encoding
artefact at exactly the point Part L.1 predicted.

## What is being tested

Experiment R fed volunteered evidence into the tower latent `z` only and left the explicit ternary
asked-history empty. The policy therefore saw a state that says *nothing has been asked*, and ran its
unseeded interview for 95–99% of users. L.1's claim is that a volunteered stated fact is **an answered
question nobody asked**, and must enter through the same channel an asked question does:

1. as a history token (`phi_q` tensored with the ternary outcome),
2. folded into `z`,
3. **masked out of the candidate table** — the answer is already held, so re-asking burns a turn.

R did (2) alone. L1 does all three and measures what changes.

## The budget, declared

Every arm in a cell holds **16 evidence questions in total**, of which `s` are volunteered; the policy
asks `16 − s`. That is the only budget under which the arms are comparable, and the only one that
keeps the position feature inside the range E1 was trained on. `tree_full16` is reported alongside as
the **handicapped** comparator: the tree gets all 16 of its own questions *on top of* the volunteered
evidence (R's protocol). It is a handicap in the tree's favour and is labelled as such everywhere.

| arm | what it does |
|---|---|
| `none` | the volunteered evidence alone; asks nothing |
| `tree_budget` | volunteered evidence + the tree's first `16 − s` questions from its root |
| `tree_full16` | volunteered evidence + all 16 of the tree's questions (**handicap**) |
| `zonly_enc` | R's encoding at the matched budget: folded into `z`, history empty, nothing masked |
| `asif` | the L.1 encoding: history token + fold + mask |

## Cell 1 — root-matching (the volunteered fact IS the tree's root concept)

The volunteered fact is concept 114, the tree's root splitter, carried by every one of the 10,000
users with that user's own ternary answer (26.1% like, 48.1% dislike, 25.8% did-not-answer). The
policy then asks 15 questions, so it holds 16 evidence questions like everything else.

| arm | full@10 | tail@10 | asks | evidence questions | re-asks of the volunteered fact | distinct interviews |
|---|---|---|---|---|---|---|
| `none` (the fact alone) | 0.17736 | 0.05235 | 0 | 1 | 0 | 1 |
| `tree_budget` | 0.28188 | 0.13308 | 15 | **15** | **1.000** | 15 |
| `zonly_enc` (R's encoding) | 0.28188 | 0.13308 | 15 | **15** | **1.000** | 15 |
| **`asif`** | **0.28355** | **0.13421** | 15 | **16** | **0** | 15 |
| `tree_full16` (handicap) | 0.28355 | 0.13421 | 16 | 16 | 1.000 | 15 |

**Behaviour.** `asif` reproduces the tree's post-root continuation for **100.00%** of users — the
falsifiable prediction, fired exactly. `zonly_enc` reproduces it for **0.00%**: told the answer to the
root, it asks the root anyway and then walks the tree from the top, which is precisely R's null.

**Score.** `asif − tree_budget` = **+0.00167 full [+0.00096, +0.00238]** (paired, 10,000 resamples,
seed 98765). That interval is the value of the one turn the encoding saves, and nothing else: the two
arms fold the same evidence and differ only in whether the volunteered fact had to be re-asked.
Against the handicapped `tree_full16` the difference is exactly 0.0 per user — `asif` with 15 asks
lands on the identical 16-question interview the tree needs 16 asks to reach.

**Redundancy is 0 by construction and asserted**, which is what validates the mask wiring.

## Cell 2 — pool-matching, non-tree (the genuinely new cell)

Two pool concepts per user, drawn uniformly from the concepts that user can actually answer and that
are **not** tree splitters (496 distinct concepts used across the cohort; 21.3% like, 78.7% dislike,
no did-not-answer by construction). The policy then asks 14 questions.

| arm | full@10 | tail@10 | re-asks | distinct interviews | follows the tree |
|---|---|---|---|---|---|
| `none` (the 2 facts alone) | 0.17152 | 0.04713 | 0 | 1 | — |
| `tree_budget` | 0.27569 | 0.13173 | 0.026 | 15 | — |
| `zonly_enc` (R's encoding) | 0.27569 | 0.13173 | 0.026 | 15 | **100.00%** |
| **`asif`** | **0.22906** | **0.10021** | 0 | **9,085** | **0.01%** |
| `tree_full16` (handicap) | 0.27948 | 0.13494 | 0.030 | 15 | — |

**`asif − tree_budget` = −0.04663 full [−0.04998, −0.04333]**; tail −0.0315 in the same direction.
The policy leaves the tree at the very first question for 90.4% of users and reaches 9,085 distinct
interviews — it reacts strongly to the volunteered facts, and the reaction is **costly**. R's
encoding, which cannot see the facts at all, is inert here and inertness is worth +0.047.

**Is this a harness bug?** No, and cell 1 is the proof: the identical code path — same history
tokens, same mask, same position feature, same fold — reproduces the tree's interview *to the last
question and to the last decimal* when the volunteered fact is one a teacher has an opinion about.
The only thing that changes in cell 2 is **which** questions occupy the history slots. So the
plumbing is verified by cell 1, and cell 2's loss is behaviour, not wiring.

**What it means.** The as-if-asked encoding does not, by itself, make the policy generalise. It puts
the volunteered fact where the policy can read it — and the policy then extrapolates off the
teacher's manifold, badly. This is the case *for* the forest: a state whose signature no teacher
covers is a state where the policy should be defaulting to a teacher's behaviour, not improvising.

## Cell 3 — arbitrary items (R's own draw, k ∈ {2, 4, 8})

**PENDING**

## Chain checks (free, and they either hold exactly or the run stops)

- **PENDING** cell 1's `tree_full16` must reproduce banked E1 *per user*, exactly.
- **PENDING** cell 3's `tree_full16` and `none` must reproduce R's banked cells at each k, exactly.
- **PENDING** `asif` redundancy must be exactly 0 in every cell — it validates the mask wiring.
