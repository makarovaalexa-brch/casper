# L2a — THE TWO-TEACHER FOREST

> **STATUS: COMPLETE (2026-08-17) — main arm and shuffle control. Follow-up T1' is running; its section says so.**
> Design sheet: `docs/design/PAPER_B_POLICY_DESIGN.md` Part L.2; Part K.6 is why per-user labels are
> gone for good. Harness `src/instrument/forest_policy.py`. **Validation only; `test_touched=false`.**

## ★ THE FALSIFIER'S VERDICT — **INTERFERENCE DOES NOT FIRE**

On the 2,000 HOLDOUT users, after learning T1 from nothing:

| | before (E1) | after (forest) |
|---|---|---|
| **T0 fidelity** — top-1 vs the tree's own next question, at the tree's own states | 1.000000 | **0.999969** |
| **T1 fidelity** — top-1 vs the matching group order, at genre states | 0.000369 | **0.999963** |

**One tree state out of 32,000 changed.** The policy went from knowing nothing about the stated-genre
teacher (0.04%) to reproducing it essentially perfectly, and paid a single state for it. Per-teacher
on HOLDOUT: Drama 1.0000, Action 0.9998, Comedy 1.0000, Adventure 1.0000 (E1: 0.0000 / 0.0010 /
0.0000 / 0.0029).

**The single-policy claim survives two teachers.** One parametric policy holds a 22-node decision tree
and four measured group orders simultaneously, exactly, with no visible trade-off between them.

## ★ AND THE RESULT THAT MATTERS MORE — the second teacher is not worth following

The forest reproduces T1 perfectly. T1 is **worse than the tree at T1's own states**:

| group (HOLDOUT) | n | forest = T1 order | the tree, same budget | E1 improvising | forest − tree | forest − E1 |
|---|---|---|---|---|---|---|
| Drama | 1,064 | 0.26082 | **0.28730** | 0.24854 | **−0.02648** [−0.03561, −0.01750] | +0.01228 [+0.00277, +0.02171] |
| Action | 327 | 0.30968 | **0.34258** | 0.27768 | **−0.03290** [−0.05316, −0.01319] | +0.03200 [+0.01314, +0.05078] |
| Comedy | 304 | 0.20211 | **0.23286** | 0.19796 | **−0.03075** [−0.04968, −0.01112] | +0.00415 [−0.01516, +0.02346] |
| Adventure | 113 | 0.29117 | 0.29705 | 0.24172 | −0.00587 [−0.03450, +0.02377] | +0.04945 [+0.01290, +0.08800] |

Every arm holds 16 evidence questions: the stated genre plus 15 more. `forest − T1 order` is **exactly
0.00000** with a zero-width interval in all four groups — the policy *is* its teacher there.

So the ordering at a genre-seeded state is **tree > forest (= stated-genre teacher) > E1 improvising**.
Three separated conclusions:

1. **Adding a teacher beats improvising.** Where a teacher exists, following it is better than letting
   the policy extrapolate — +0.012 to +0.049, three of four intervals excluding zero. That is L1's
   cell-2 lesson repaid.
2. **But this teacher is worse than the one we already had.** The stated-genre orders were searched
   over **items only** (the genre answer was free in that experiment, so each group searched 15 items),
   while the tree spends most of its budget on concepts. Conditioning on a stated genre does not buy
   back what an item-only action space gives away.
3. **Therefore admissibility is not sufficiency.** The design law's two conditions — observable
   signature, honest cohort size — are what make a teacher *learnable*. They say nothing about whether
   it is *good*. A third condition is now measured, not assumed: **a teacher earns its place only if it
   beats the teacher already covering those states, on the same action space and the same budget.**

## Spanning gate — EXACT

Cold-state K=16 rollout on the 2,000 HOLDOUT users: forest **0.29280463882263225**, E1
**0.29280463882263225**. Paired difference **0.0 with a zero-width interval**, and **100.00% of users
are per-user identical to E1**. Learning four new teachers moved nothing at all about the cold
interview. 16 distinct interviews (E1's 15 leaves plus the one changed state).

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

| gate | bar | result |
|---|---|---|
| spanning | cold-state behaviour ≥ E1, exact | **PASS** — per-user identical, 2,000/2,000 |
| per-teacher fidelity | top-1 vs each teacher's own order at its signature | **PASS** — 0.9998–1.0000 |
| behavioural | policy vs the matching teacher's own order, null or better | **PASS** — exactly 0.0, all four groups |
| ★ interference | T1 fidelity must not cost T0 fidelity | **PASS** — 1 state of 32,000 |
| shuffle control | teacher assignments permuted across signatures | **PENDING** — running |

## Training record

Warm-started from E1 (16/16 tensors byte-identical after the load), batch 256 = 128 stated-genre
states + 128 tree replay (`t1_frac` 0.5, declared), lr 1e−4, 6 epochs of 768 steps, **6.7 minutes**.
Selection on T0 fidelity + T1 fidelity over a FIT-dev slice of 800 users; best epoch 3. Both
fidelities, every epoch, on FIT-dev:

| epoch | T0 fidelity | T1 fidelity |
|---|---|---|
| 0 (E1) | 1.0000 | 0.0002 |
| 1 | 0.9998 | 0.9910 |
| 2 | 1.0000 | 0.9997 |
| 3 | 1.0000 | 1.0000 |
| 4–6 | 1.0000 | 1.0000 |

T1 is learned to saturation inside one epoch and T0 never moves. The 108,960 training states carry
only 60 distinct (signature, depth) label pairs, so this is a small thing to learn — which is the
point: **cohort teachers are cheap to add, and the cost of adding one is the question.**

## Shuffle control — PASSES

Teacher assignments permuted across signatures (Drama←Comedy, Action←Adventure, Comedy←Drama,
Adventure←Action; seed 98771), everything else identical. Same 6 epochs, same selection rule.

- **T1 fidelity against the true orders: 0.000000** (main arm: 0.999963). The shuffled arm learns the
  orders it was given and none of the ones it was not — the conditioning is what the main arm learned,
  not the item sequences in general.
- **T0 fidelity: 0.999969**, the same single state. Learning *some* second teacher costs the tree
  nothing whichever teacher it is; the cost measured for the main arm is not a cost of T1 in
  particular.
- Spanning: +0.000005 full [+0.000000, +0.000015] — cold behaviour intact under the control too.
- Behavioural at genre states, full@10: Drama **0.23702** (main 0.26082), Action **0.29542** (main
  0.30968), Comedy **0.19583** (main 0.20211), Adventure **0.24211** (main 0.29117). The control is
  worse in every group, and reproduces the correct order for **0.00%** of users everywhere.

## What L2a settles

1. **One policy, many cohort teachers, no interference** — the architectural claim holds at two
   teachers, exactly, and the shuffle control shows the fit is conditioning rather than memorised
   sequences.
2. **Adding a teacher beats improvising** at states the teacher covers.
3. **A teacher must beat the incumbent, not merely be admissible** — the stated-genre orders lose to
   the cold tree by 0.026–0.033 on three of four groups, and the most likely reason is that they were
   searched over items only.

## Next — T1' (running)

`src/instrument/group_greedy_signature.py`: the stated-genre teacher **rebuilt over the tree's own
1,000-candidate pool** (500 items + 500 concepts), same objective, same budget, FIT users only, so the
comparison against the tree is like-for-like. **This is the decision experiment for the forest**: if a
genre-conditioned order over the full action space beats the tree inside its group, conclusion 3 was
about the action space and T2/T3 are worth their search time; if it does not, conditioning on a
volunteered category genuinely does not pay against a cold interview that may ask about concepts, and
the forest stops at one teacher. **PENDING.**
