# Paper B — DESIGN SHEET FOR THE CONFIRMATORY TEST LOOK

**Status: AWAITING AUTHOR SIGN-OFF. Nothing here has touched test.**
Internal design sheet per `CLAUDE.md` ("design sheet before any run"). **Not a public preregistration** —
that remains ruled out (`paperB-kickoff-artifacts-2026-08-03`).

Written 2026-08-29, **before** any test data is built or loaded, so that the contrast is fixed by the
claim rather than by the outcome.

---

## 0. WHAT IS ALREADY SPENT

One test look was spent **2026-08-12** on Act 1 (`tab:act1`) and Act 3 (`+0.01035/+0.00705` @K=16).
Acts 1–4 are frozen. **This sheet governs the ONE remaining look, for Act 5 only.**

## 1. THE CLAIM BEING TESTED

> A user's **unprompted opening request** carries enough signal to route them to a group-specific
> interview, and a **single distilled policy** can carry all six of those interviews at once —
> delivering more of the recommender's value at the same question budget than one global cold-start
> interview.

⚠ Deliberately NOT claimed (see `PAPER_B_MASTER_TODO.md` §B): that routing users to a group-specific
interview is itself novel — Golbandi WSDM'11 and Cluster-Based Bandits SIGIR'21 both motivate
cold-start exactly that way. The novel element is **the source of the routing signal**.

## 2. PRIMARY CONTRAST — one, fixed in advance

**`policy − cold` on the full deployment mix, K=15, all 10,000 test users, full NDCG@10,
paired percentile bootstrap (10,000 resamples, cluster over users, seed 98765).**

★ **This is the ONLY confirmatory contrast.** Everything in §3 is secondary and descriptive. Naming a
single primary is our multiplicity control, which `HARSH_REVIEW_POINTS` flags as missing elsewhere —
we are not correcting across a battery because we are not running a battery.

**Declared readings, before the data exists:**

| outcome | reading |
|---|---|
| CI excludes zero, positive | claim **supported** on test |
| CI contains zero | claim **not supported**; report as such, keep the validation result labelled validation-only |
| CI excludes zero, negative | claim **refuted**; report it |

⚠ On validation this contrast is **+0.00333 [+0.00158, +0.00499]**. The lower bound is about half the
point estimate, so a null on test is a live possibility and is **not** grounds for a re-look.

## 3. SECONDARY, REPORTED REGARDLESS OF OUTCOME

Descriptive. No claim rests on them and none is corrected for multiplicity.

1. `policy − static` (validation: +0.01151 [+0.00849, +0.01453])
2. The three-way order `static < cold < opener`
3. **Do-no-harm** on users who volunteer nothing (validation: −0.00037, n.s. but **nominally
   negative** — reported as measured, not rounded to zero)
4. **All three budgets K = 1, 4, 15.** ⚠ Declared now: on validation **K=4 straddles zero**
   (+0.00119 [−0.00029, +0.00269]) while K=1 and K=15 clear it. This non-monotonicity is reported
   whatever test shows; we are not selecting a budget after the fact.
5. Per-group breakdown, including the two groups that are flat (`dialogue`, `inspiring`)

## 4. ARMS — all scored in ONE pass, none added afterwards

| arm | artefact | fitted on |
|---|---|---|
| static (combined) | `greedy_static_policyB_combined_train10000.json` | train, 10,000 |
| **GreedyExtend analogue** (items-only static) | `greedy_static_policyB_items_train10000.json` | train, 10,000 |
| cold tree | `coldtree_v2_train10000.json` | train, 10,000 |
| six group teachers | `m6k_dj<group>_train2000.json` | train, 2,000/group |
| **policy** (the arm under test) | `m4/m7dj_policy.pt`, epoch 5 | train |
| **scrambled-opener policy** (control) | same ckpt, `--scramble_openers` | — |

Every arm is train-fitted, so no arm is at home while another is away (home advantage measured at
+0.02881; this is why the asymmetry matters). Openers are **half B** — films excluded from every
training opener.

⚠ **GreedyExtend's own objective is RMSE, which is NOT COMPUTABLE here**: `liang_split.py:169`
binarizes at `rating > 3.5` and discards rating values. We run its *algorithm* — greedy static
selection over items only — under our objective, and say so explicitly in the paper. This is the
"explicit incompatibility statement" `HARSH_REVIEW_POINTS` permits in place of running a named prior.

## 5. RULES BINDING THIS RUN

1. **ONE pass.** No second look, whatever the result. A null is a result.
2. **No arm added after seeing test numbers.** Anything missing from §4 is validation-only, forever.
3. **No budget, group, or subset selected after the fact.** §2 and §3 fix them now.
4. The test set is built once, by the canonical loader, with the standing leak assert
   (`0/10000 users have any held-out target inside the routing lookup`).
5. Report the primary with its CI whatever it says — including if it kills the claim.

## 6. WHAT THIS RUN CANNOT SETTLE

- Nothing about **real users**: openers are corpus-matched synthetic requests, not collected from
  people. The human round-trip study remains owed and still blocks the strongest claims across A–E.
- Nothing about **robustness to training seed** — the policy is trained once. Either three seeds are
  run first, or robustness language is dropped (author decision, `MASTER_TODO` A9; current plan is to
  drop the language, which costs nothing because no claim needs it).


## 7. WHAT DIFFERS FROM THE VALIDATION RUN (and what must NOT)

The protocol is otherwise identical: same certified pair, same trees, same policy checkpoint
(`m7dj_policy.pt` ep5), same K, same metric, same paired bootstrap, same half-B openers. Three
things necessarily differ, and the third is a contamination risk.

| | | rule |
|---|---|---|
| 1 | **the split** | `m6d_eval.py:112` currently HARDCODES `build_greedy_env(..., "validation", ...)`. The split must be plumbed through. Code change, not a flag. |
| 2 | **per-user group assignment** | `m7b_disjoint.npz` banks only `val_rows/val_group/val_has`. Test users get the SAME rule -- argmax over their own signed endorsements. This reads only the user's own ratings and fits nothing, so it is legitimate. |
| 3 | **opener material** | ⛔ **MUST BE REUSED, NEVER REFITTED.** |

### The contamination risk, stated plainly

`m6d_eval.py:156-157` recomputes the material on whatever split is loaded:

```python
coh     = rows_all[endorse[rows_all, ANCHORS[g]]]
tabs[g] = centroid_lift(E, coh, ZF, ZR)
```

On a test run this would derive **each group's centroid and diagnostic film pool from test users** --
fitting the opener generator on the confirmatory set. The output would look entirely normal. This is
the same failure family as the 2026-08-29 `m6v` bug: a silent cross-population substitution that
prints plausible numbers.

**Binding rule for the test pass:** `own_item`, `own_concept`, the six group centroids and the
half-B pools are taken from the BANKED validation-derived artefacts and reused byte-for-byte. Only
the users, their own-data group assignment, and their own liked films (filtered through the banked
pools) come from test.

**Guard, mandatory before the run:** assert that the material actually used on test is identical to
the banked validation material -- compare the pool ids and `own_item`/`own_concept` arrays and refuse
to proceed on any difference. Guard the MATCH, not non-emptiness.

## 8. FRAMING (author ruling, 2026-08-29)

> *"I want to show that the policy is a neater, cleaner, more extendable approach and it doesn't have
> to beat every tree on every setting -- it's more about capability to generalise and have all built
> in one by design."*

The contribution is the **architecture and the capability**, not a scoreboard win. Consequences:

- A per-group loss is not a failure; the six trees are six artefacts and the policy is one.
- ★ The trees are handed their group label for free (`m6d_eval.py:257` runs `trees[g]`, the correct
  tree, chosen by the banked assignment). The policy is given no label at any point and must infer it
  from the folded state. **`policy - tree` therefore understates the policy**, and this asymmetry is
  stated in the text rather than hidden.
- Forward-looking, NOT claimed as result: more teachers, or a different training regime (e.g. RL given
  enough observable episodes), would extend coverage within the same single-artefact design.
- ⛔ A deployable forest + router comparator was considered and **ruled out by the author** -- the
  claim is not superiority over a two-stage system.

---

## SIGN-OFF

- [ ] Author approves the claim in §1
- [ ] Author approves the single primary contrast in §2
- [ ] Author approves the arm list in §4 as final
- [ ] Author approves the reuse-not-refit rule in §7
