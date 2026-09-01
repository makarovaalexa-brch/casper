# How the six groups were chosen — and what was tried and rejected

**Status: design of record, 2026-08-29.** Everything fitted on TRAIN, reported on VALIDATION. The
test look is unspent. Companion artefacts: figure `docs/figures/m7c_group_atlas.{png,pdf}`,
exemplars `experiments/battery/m6/m7c_group_atlas.json`, literature
`external_literature/findings/user_segmentation_for_elicitation_2026-08.md`.

---

## 1. What ships

Six groups, each user in exactly one, built from **behavioural concept endorsement** and then made a
**partition** by two argmaxes (`src/instrument/m7b_disjoint.py`):

- **users** — each user goes to the group they endorse most strongly (highest signed value `V`)
- **material** — each film and each concept goes to the group whose centroid lift for it is highest

3,980 of 10,000 validation users (39.8%) land in a group; the rest are served the cold interview.
**No k-means is used anywhere in the shipped system.**

## 2. The selection principle — routability, not tidiness

A grouping is only useful if the system can tell which group a user is in **from what they
volunteered, before asking anything**. That is not a nice-to-have: a grouping the router cannot
recover produces teachers that are never reached. It is also the criterion that makes the choice
non-arbitrary, so it is stated once and applied to every candidate:

> **A candidate grouping is admissible only if a linear probe recovers it from the folded opener
> state.** Coherence and size are secondary; unrecoverable groups are worthless regardless of how
> clean they look.

This is measured by `m6x_state_separability.py` and is the axis on which the alternatives below were
rejected or kept.

## 3. What was tried

| # | grouping | outcome | why rejected |
|---|---|---|---|
| 1 | **17 genres** | ⛔ dead | genre-conditioned greedy: Drama (55% of users) is a **null**, all 17 groups shrank; the genre teacher **loses at its own states** (−0.026 / −0.033 / −0.031). Genres are a taxonomy of films, not of people. |
| 2 | **k-means on raw latents** (`m6y`, k=20) | ⛔ dead | release-year buckets. Under the corrected era test (**sd alone**), **14 of 20** clusters have distinctive films spanning under 2.5 years (c12 all 1994, c7 1995–96, c14 1993–95). In MovieLens the dominant variance in `z` is *when the user was active*, not taste. Projecting out one era axis does not fix it — era lives in many directions. |
| 3 | **k-means on concept-decoded latents** (`m7a`, k=12) | ⛔ dead | halves the era problem (**6 of 12** era buckets vs 14 of 20) but the clusters are **not nameable**: c0 = *memory loss, product placement, unrealistic, plot, creativity*; c1 = *san francisco, dog, 18th century, divorce, courtroom*. A group you cannot name cannot motivate a question. |
| 4 | **endorsement cohorts, overlapping** | ⚠ superseded | good teachers, invisible grouping. Probe **0.371** on the worst group, policy **0.364** — an *information* ceiling, not an optimisation one. Cause: marvel/inspiring shared **10 of 12** characteristic words and **31%** of opener films sat in more than one pool. |
| 5 | **endorsement cohorts, DISJOINT** | ✔ **shipped** | probe **0.959**, achieved routing **0.858** (marvel 0.364 → **0.958**), teacher value preserved (`opener−cold +0.00688*`), and the distilled policy beats cold on the deployment mix (**+0.00333\***). |

⚠ **An honest limit.** Teachers were **never fitted** on the k-means groupings (2 and 3) — those were
rejected on era contamination and non-nameability, which are diagnostic, not NDCG. **We therefore
cannot and must not claim our grouping beats k-means grouping on accuracy.** What we can claim is
that it is the only candidate that is simultaneously nameable, era-free by construction, and
recoverable from the opener.

## 4. Why the shipped groups are coherent

Each group's **exclusive** vocabulary — concepts it wins outright — reads as a taste, not a
statistical residue:

| group | exclusive vocabulary | users |
|---|---|---|
| arthouse | melancholic, bleak, quirky, disturbing, dark, documentary | 473 |
| marvel | thriller, adventure, action, fun, fantasy, funny | 593 |
| teenagers | comedy, romance, musical | 540 |
| inspiring | sad, emotional, feel-good, heartwarming | 1,557 |
| gore | tense, scary | 419 |
| dialogue | gritty, drama | 398 |

These cover the everyday tastes a reader will test the scheme against: `teenagers` is the
comedy/romance group, `inspiring` the feel-good/family one. Per-group exemplar films are generated
into `m7c_group_atlas.json` and the chapter table is built from it — **never retyped**.

⚠ `dialogue` and `inspiring` keep only 113 and 152 exclusive films and lean on generic vocabulary;
they are the two groups with no significant gain. Declared before the fit, not discovered after.

## 5. Literature

Full notes: `external_literature/findings/user_segmentation_for_elicitation_2026-08.md` (searched
2026-08-29, abstracts only, `[A/S]`).

- ⛔ **"Cold-start = identify the user's cluster" is the standard motivation of the line we build on**
  — Golbandi/Koren/Lempel WSDM'11 (our tree's source) and Cluster-Based Bandits (SIGIR'21) both frame
  it exactly this way, both using "distinguisher items". **Routing to a group teacher is not a
  contribution and must not be claimed as one.**
- ⚠ arXiv **2510.27342** (Oct 2025) elicits genre attributes *"to better cluster users"* — but the
  full text (read 08-29) shows **Yahoo! Music only, 1,701 users, RMSE only, ONE global tree**, and
  *"the system always chooses what to ask; users do not volunteer preferences."* It shares the
  attribute channel and nothing else: different dataset, different metric, no per-group interviews,
  no volunteered input. Cite as current state of the art and separate on those four axes.
- Cluster-based CF (co-clustering/CCCF, k-means CF, per-cluster local models) is a large, old
  literature — context, not competition, since none of it concerns elicitation.

★ **What is left to us is the SOURCE of the routing signal.** Golbandi and Shams choose *distinguisher
items to ask*; we read *what the user said first, unprompted*. No prior work was found doing that.

## 6. Still owed

1. ~~Separability across all candidate groupings~~ — **DONE 2026-08-29.** See §8.
2. ~~Scrambled-opener control~~ — **DONE 2026-08-29, PASSED.** See §7.
3. Read arXiv 2510.27342 properly before writing related work.

---

## 7. THE SCRAMBLED-OPENER CONTROL (A1) — the claim's existential test

Each warm user was handed **another user's real opener**, permuted across the whole warm set (never
within group, which would leave the group signal intact). 3,980 users, **0 fixed points**. Film
popularity, length, coherence and the group-marginal mix are preserved exactly; only the link between
this user and this opener is destroyed — *scramble the correspondence, never the distribution.*

| group | routing, real | routing, scrambled | value vs cold, real | value vs cold, scrambled |
|---|---|---|---|---|
| arthouse | 0.888 | **0.148** | +0.03077\* | +0.00292 |
| dialogue | 0.834 | **0.040** | +0.00155 | −0.00199 |
| marvel | 0.958 | **0.094** | +0.00643 | **−0.01190−** |
| gore | 0.857 | **0.038** | +0.01980\* | −0.00693 |
| teenagers | 0.757 | **0.111** | +0.01313\* | +0.00036 |
| inspiring | 0.852 | **0.121** | +0.00074 | **−0.00572−** |

Chance over six groups is ≈0.167.

**Two things this establishes, and they are the strongest evidence in the section.**

1. **The router reads the opener.** Routing collapses to chance on every group. It cannot be running
   on popularity, opener length, or a generic profile leak — those all survive the permutation
   untouched.
2. **The teachers are genuinely group-specific.** Value does not merely fall back to cold; on
   `marvel` and `inspiring` it goes **significantly negative**. Being routed to the *wrong* group's
   interview is worse than not being routed at all, which is only possible if each tree encodes
   something specific to its own group rather than being a generically better interview.

★ Point 2 also closes the reviewer objection that the group trees are just better-fitted trees that
would help anyone.

---

## 8. SEPARABILITY ACROSS CANDIDATE GROUPINGS (A2) — and what it does NOT show

| grouping | k | probe accuracy | chance | x chance |
|---|---|---|---|---|
| raw-latent k-means (`m6y`) | 20 | 0.535 | 0.050 | **10.7x** |
| concept k-means (`m7a`) | 12 | 0.603 | 0.083 | 7.3x |
| endorsement, **DISJOINT** (shipped) | 6 | **0.956** | 0.167 | 5.7x |
| endorsement, overlapping | 6 | 0.615 | 0.167 | 3.7x |

⛔ **DO NOT claim "we chose the most separable grouping."** It is false under one normalisation and
true under the other: the shipped grouping wins on raw accuracy, but **both k-means groupings beat it
on lift over chance**, because a larger `k` lowers chance and inflates the ratio while changing the
task. Neither column is a fair comparison across different `k`.

★ **The one like-for-like comparison is within k=6, and it is clean:** the same six groups, same
chance, same probe — **0.615 → 0.956 from disjointing alone**. That is the number the chapter should
carry, and it is a statement about *the fix*, not about a ranking of groupings.

⚠ The concept k-means grouping is separable **by construction** — it is a deterministic function of
`z`, so a probe reading `z` must recover it. Its separability is therefore not evidence of merit, and
it was rejected on nameability and era contamination (§3), which are the axes that actually
discriminate. The defensible selection argument is the conjunction, not any single number:

| grouping | separable | nameable | era-free |
|---|---|---|---|
| raw-latent k-means | yes | no | **no** (14/20 under 2.5 yrs) |
| concept k-means | yes (by construction) | **no** (*memory loss, product placement, unrealistic*) | **no** (6/12) |
| endorsement, overlapping | **no** (0.371 worst group) | yes | yes |
| endorsement, **DISJOINT** | **yes** | **yes** | **yes** |

Genre is absent from this table: it was rejected on **value** (Drama a null, all 17 groups shrank,
the genre teacher loses at its own states), never reaching a separability test.
