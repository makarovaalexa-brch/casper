# L1 — THE AS-IF-ASKED ENCODING TEST

> **STATUS: RUNNING (2026-08-17). Every number below is marked PENDING until its cell lands.**
> Design sheet: `docs/design/PAPER_B_POLICY_DESIGN.md` Part L.1. Harness `src/instrument/policy_asif.py`
> (commit `c79b6b8`). Artifacts `experiments/battery/asif/`. **Validation only; `test_touched=false`.**

## THE FALSIFIER'S VERDICT

**PENDING** — cell 1 (root-matching) decides it. The prediction is exact: with the volunteered fact
encoded as an answered question, the policy must branch exactly as the tree's post-root arm. If it
does not, the encoding is wrong and the theory is untouched.

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

**PENDING**

## Cell 2 — pool-matching, non-tree (the genuinely new cell)

**PENDING**

## Cell 3 — arbitrary items (R's own draw, k ∈ {2, 4, 8})

**PENDING**

## Chain checks (free, and they either hold exactly or the run stops)

- **PENDING** cell 1's `tree_full16` must reproduce banked E1 *per user*, exactly.
- **PENDING** cell 3's `tree_full16` and `none` must reproduce R's banked cells at each k, exactly.
- **PENDING** `asif` redundancy must be exactly 0 in every cell — it validates the mask wiring.
