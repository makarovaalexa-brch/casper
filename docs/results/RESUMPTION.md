# EXPERIMENT R — RESUMPTION: can the policy condition on evidence it did not choose?

**Status: COMPLETE, 2026-08-15. Validation only — the 10,000 canonical TEST users were never built,
never loaded, never named. Every artifact stamps `test_touched=false`.**

Design sheet: `docs/design/PAPER_B_POLICY_DESIGN.md` **Part H.2 is governing**. Code:
`src/instrument/policy_resume.py` (committed before the run, HARD RULE 10).

> **AUTHOR SAID (2026-08-15):** *"plz run the experiments."*
> **AUTHOR SAID (2026-08-13), on why:** *"policy is supposed to eb scalable, generalisable (it knows
> similarity between concepts and items, tree doesn't), could support new vocab, can support new
> states, most improtantly i guess."*

---

## THE VERDICT, FIRST

**The pre-registered falsifier did NOT fire — and the claim it was guarding is still not
demonstrated.** The falsifier was *"if the policy adapts and still loses to the tree on seeded
states, the 'supports new states' claim is dead."* The policy did not lose. It also did not adapt.

| k volunteered items | users whose interview changes at all | mean questions changed (of 16) | policy − tree, full NDCG@10 |
|---|---|---|---|
| 2 | **1.01%** | 0.13 | **+0.00002 [−0.00019, +0.00023]** |
| 4 | **2.48%** | 0.30 | **−0.00031 [−0.00067, +0.00004]** |
| 8 | **5.11%** | 0.60 | **−0.00026 [−0.00073, +0.00020]** |

For 95–99% of users the distilled policy conducts **exactly the interview it would have conducted if
the person had volunteered nothing** — which is exactly the interview the tree conducts. The
architecture *can* read the seeded state (the latent is right there in its input, and 1–5% of users
prove the wiring is live); the distillation simply put no behaviour there. So the honest sentence is:

> **The capability is structural, not behavioural. One-hot distillation of a tree teaches the policy
> the tree's function and nothing outside it — including how to use evidence the tree cannot see.**

And the second half of it, which is the part that would have been easy to miss:

> **Where the policy does change its mind, it is not better for it.** Among the users who deviate,
> the paired policy − tree difference is **+0.00241 (n=101) at k=2, −0.01232 (n=248) at k=4 and
> −0.00500 (n=511) at k=8**. Two of the three are losses, and the largest effect is a loss.

---

## The design, as run

Each of the 10,000 validation users **volunteers k items before the interview starts**. Those items
are folded as ordinary evidence. Then each arm conducts a **16-question** interview from that seeded
state, and all four arms are scored through the identical funnel (full and tail NDCG@10 at K=16,
learned decoder bias, policy-B mask, `genv_eval_multi`).

| arm | what it can do with the volunteered evidence |
|---|---|
| **ask-nothing-more** | nothing further is asked. The k items alone. Isolates what the interview adds. |
| **static combined order** | nothing. The same 16 questions for every user, always. |
| **tree** | nothing. Its routing consumes the ternary answers to *its own* questions only, so it starts at its root and asks its fixed branch sequence regardless. Verified rather than assumed: the tree's own ternary answers are unchanged under seeding (the answer rule reads the user's ratings, not what they volunteered), so its realised interview is bit-identical to the unseeded one. |
| **distilled policy** (E1, frozen, `policy_main_equivariant_both.pt`, val 0.28355) | reads the seeded latent `z₀`. May adapt. **Whether it does is the result.** |

**The draw, declared (HARD RULE 1).** k items per user drawn **without replacement, popularity-
weighted** (p ∝ the train rating count) from the items that user rated on their **fold-in half** —
`E["lvl"][r]`, which is built with the held-out targets removed, so nothing here can touch a target.
One `numpy.default_rng(98765)` stream, users consumed in canonical row order. Every user had at
least 8 rated fold-in items, so no user was short of k at any level. **63% of volunteered items are
likes** (level ≥ 4.0 stars), 37% are dislikes — a realistic mixed profile, not a wish-list.

Popularity-weighting is not cosmetic: a uniform draw over a user's ratings over-represents the
catalogue tail and would make the seeded state unrealistically informative.

**Gates passed before any number was trusted:** canonical m=0 snap `0.15686538118354684` exact;
`greedy_static_policyB_combined` rescored to its banked `0.25561016113602253` exactly; pool identity
against the trajectory bank; leak check (0/10,000 users have a held-out target in the routing
lookup); `branch_of_per_user == branch_of`; E1's banked per-user vector re-asserted equal to the
tree's realised interview.

---

## RESULT 1 — the scores

Reference points on this cohort: empty interview **0.15687**, static order at K=16 **0.25561**,
E1 / the tree unseeded **0.28355 full / 0.13421 tail**.

| k | arm | full NDCG@10 | tail NDCG@10 |
|---|---|---|---|
| 2 | ask-nothing-more | 0.21274 | 0.08775 |
| 2 | static + 16 | 0.26951 | 0.13695 |
| 2 | **tree + 16** | **0.28615** | **0.14415** |
| 2 | **policy + 16** | **0.28617** | **0.14421** |
| 4 | ask-nothing-more | 0.23936 | 0.10966 |
| 4 | static + 16 | 0.27930 | 0.14782 |
| 4 | **tree + 16** | **0.29230** | **0.15323** |
| 4 | **policy + 16** | **0.29199** | **0.15305** |
| 8 | ask-nothing-more | 0.27339 | 0.14296 |
| 8 | static + 16 | 0.29225 | 0.16706 |
| 8 | **tree + 16** | **0.30410** | **0.16951** |
| 8 | **policy + 16** | **0.30384** | **0.16953** |

Paired percentile bootstrap, 10,000 resamples, seed 98765, over per-user differences (never the
unpaired `1.96·√2·SE` form):

| comparison | k=2 full | k=4 full | k=8 full |
|---|---|---|---|
| **policy − tree** | +0.00002 [−0.00019, +0.00023] | −0.00031 [−0.00067, +0.00004] | −0.00026 [−0.00073, +0.00020] |
| policy − static | +0.01666 [+0.01383, +0.01945] | +0.01269 [+0.01003, +0.01536] | +0.01159 [+0.00921, +0.01406] |
| tree − ask-nothing-more | +0.07341 [+0.07005, +0.07680] | +0.05293 [+0.04980, +0.05608] | +0.03070 [+0.02777, +0.03366] |
| seeded tree − unseeded E1 | +0.00260 [+0.00035, +0.00482] | +0.00875 [+0.00612, +0.01144] | +0.02055 [+0.01746, +0.02363] |

**Three things worth reading off this table.**

1. **Volunteered evidence pays, and the recommender is what banks it.** The seeded tree beats the
   unseeded tree by +0.0026 / +0.0088 / +0.0206 full and +0.0099 / +0.0190 / +0.0353 tail. None of
   that comes from better questioning — the questions are identical. It is the fold. So *"the system
   uses evidence it did not ask for"* is true of the **recommender**, and this experiment was asking
   whether it is also true of the **policy**. It is not.
2. **The interview keeps paying on top of the volunteered evidence, with diminishing returns.**
   Sixteen questions add +0.0734 over the k=2 seed, +0.0529 over k=4, +0.0307 over k=8. Eight
   volunteered items are worth roughly what a fifth of the interview is worth, and the interview
   still more than doubles the seed's own gain.
3. **The conditioning advantage over a static order survives seeding but shrinks with it**
   (+0.0167 → +0.0127 → +0.0116). More free evidence, less room for question choice to matter.

---

## RESULT 2 — ★ THE REDUNDANCY COUNT, which is the mechanism made visible

How often does an arm ask an **item question the volunteered evidence has already answered** — i.e.
the person is asked about a film they just named? (A re-asked item is folded once; the turn is burnt
and no new evidence arrives, which is the honest semantics and is how it was scored.)

| k | static | tree | **policy** |
|---|---|---|---|
| 2 | 0.081 re-asks/user (7.9% of users) | 0.103 (10.0%) | **0.102 (9.9%)** |
| 4 | 0.168 (15.8%) | 0.211 (19.0%) | **0.209 (18.9%)** |
| 8 | 0.312 (27.1%) | 0.397 (32.0%) | **0.388 (31.6%)** |

Per item question asked (the tree asks 7.27 item questions of 16, the static order 6.0, the policy
7.24), the re-ask rate is **5.5% for the tree, 5.2% for the static order and 5.4% for the policy** —
indistinguishable. **The policy, which can see the volunteered items, avoids them at the rate of an
arm that cannot see them at all.** At k=8, roughly one user in three is asked about a film they had
already volunteered, and the arm with the information does nothing about it.

This is the cleanest single fact in the experiment. The claim under test was *"the policy can
condition its questioning on evidence it did not choose."* The redundancy count says it does not,
and it says so without reference to any NDCG.

---

## RESULT 3 — how much the policy moves, and whether the movement helps

E1's unseeded rollout **is** the tree's realised interview (top-1 agreement 1.0000, gain equal to
1e-9), so "deviates from its unseeded self" and "leaves the teacher's manifold" are the same event.

| k | frac never deviating | mean positions changed | distinct realised interviews (tree: 15) | Pep divergence (tree: 0.66370) |
|---|---|---|---|---|
| 2 | 0.9899 | 0.131 | 40 | 0.66721 |
| 4 | 0.9752 | 0.300 | 75 | 0.67167 |
| 8 | 0.9489 | 0.601 | 177 | 0.68018 |

Deviation begins at **position 3** at every k (positions 1–2 are identical for every user at every
k) and then holds roughly flat across the remaining positions — i.e. when a seeded state does move
the policy, it moves it early and the change persists, rather than the interview drifting further
apart as it goes.

**And the movement does not pay.** Restricting to the users who deviate:

| k | n deviators | mean (policy − tree) among them |
|---|---|---|
| 2 | 101 | **+0.00241** |
| 4 | 248 | **−0.01232** |
| 8 | 511 | **−0.00500** |

Two of three are losses. This matters for how the result is written: it is not *"the policy adapts a
little and it is a wash"*; it is *"the policy adapts on a small minority of users and is, on
balance, slightly worse when it does."*

The channel mix is untouched by seeding — concept share 0.5457 for the tree, 0.5458 / 0.5458 /
0.5475 for the policy at k=2/4/8 — so nothing about the seeded state pushes the policy toward one
channel; it simply keeps doing what it was distilled to do.

---

## WHAT THIS SETTLES, AND WHAT IT LICENSES

**Settled, on 10,000 held-out validation users, every number paired-bootstrapped:**

1. **The distilled policy does not condition on volunteered evidence.** 95–99% of users receive the
   identical unseeded interview; the redundancy rate matches an arm that cannot see the evidence;
   the deviations it does make are, on balance, small losses.
2. **It does not lose to the tree either** — all three policy − tree intervals include zero. The
   architecture is not damaged by an off-manifold state; it is simply indifferent to it.
3. **The falsifier did not fire, and no capability claim is earned.** "Supports new states" may be
   written as an **architectural property** (the state is an input; the wiring is live on 1–5% of
   users) and **never as a demonstrated behaviour**. The sentence *"the policy adapts to evidence it
   did not choose"* is not available on this evidence.
4. **What the system does bank from volunteered evidence is the fold, not the questioning.** Up to
   +0.0206 full / +0.0353 tail at k=8, identical questions.
5. **The tree's structural inability to route on volunteered evidence costs it nothing here**,
   because nothing else exploited that evidence either. The setting was built to disadvantage the
   tree and did not.

**What it does NOT show:** that a parametric policy *cannot* use volunteered evidence. Under HARD
RULE 6 this is a **weak-method result about one-hot distillation from a tree teacher**, not a
statement about the architecture. The fix named in Part H.2 is legible and its data already exists:
a **fine-tune on evidence-seeded states**, for which the 138 random-prefix, evidence-seeded states
sampled on 2026-08-14 carry measured labels at zero new cost. **It was not run** — Part H.2 requires
reporting first, and this document is that report.

**Nothing about test.** No test user was built, loaded or touched.

---

## ★★★ ADDENDUM, 2026-08-15 — THE BELIEF-ONLY (z-ONLY) ARM: it adapts, and the falsifier FIRES

> **AUTHOR SAID: "ok, go ahead."**

### The hypothesis, stated before the run

R showed the both-inputs policy runs its *unseeded* interview for 95–99% of users. The proposed
mechanism: the state has two parts, and **volunteered evidence moves the tower latent `z_t` while
leaving the explicit ternary asked-history EMPTY — a pairing that never occurs anywhere in E1's
training**, because the only way `z_t` ever moved there was that a question was asked and answered.
On an unfamiliar pairing the model falls back to its empty-interview behaviour, which is the tree's
opening.

**The `z`-only policy cannot make that distinction by construction:** volunteered items and answered
items fold into `z_t` identically, so it has no way to tell *"three films you offered"* from *"three
films I asked about"*. If the mechanism is right, it should adapt. It is a fair arm, not a straw man
— top-1 **0.9951**, gain **+0.02723** (97% of E1's +0.02794), and it already deviates from the tree
on **6.19%** of users with no seeding at all, where the both-inputs model deviates on 0%.

Same protocol, same k ∈ {2,4,8}, same popularity-weighted draw, seed 98765, same gates, same paired
bootstrap. **Every cell R published was asserted against R's banked value** — resumed or recomputed —
and all 12 reproduce exactly. One new comparator was needed: **`policyz_frozen`**, which replays the
z-only policy's **own unseeded question order** at the seeded state, so it differs from `policyz` in
the questions asked and in nothing else. (For the both-inputs policy that arm *is* the tree arm,
which is why R did not need it.)

### 1. It adapts — the prediction is confirmed, and not marginally

| k | both-inputs: users changed / positions | **z-only: users changed / positions** |
|---|---|---|
| 2 | 1.01% / 0.13 of 16 | **87.66% / 9.30 of 16** |
| 4 | 2.48% / 0.30 | **94.59% / 11.38** |
| 8 | 5.11% / 0.60 | **98.60% / 12.91** |

Distinct realised interviews go from 228 (its own unseeded rollout) to **4,899 / 6,900 / 8,288**, and
the amount of change **scales with the amount of volunteered evidence**. The belief-only policy is
unambiguously responding to evidence it did not choose. **The state-features half of the mechanism is
real: the asked-history feature is what pins the both-inputs policy to its opening.**

### 2. The deviations do not pay — they cost, at every k

`policyz` − `policyz_frozen` isolates the deviation exactly: same user, same fold, same volunteered
evidence, **only the questions differ**.

| k | z-only − its own frozen order | among the deviators only | z-only − tree |
|---|---|---|---|
| 2 | **−0.01246 [−0.01471, −0.01015]** | −0.01422 [−0.01675, −0.01165] (n=8,766) | −0.01255 [−0.01481, −0.01025] |
| 4 | **−0.01234 [−0.01467, −0.00996]** | −0.01304 [−0.01559, −0.01047] (n=9,459) | −0.01254 [−0.01489, −0.01016] |
| 8 | **−0.00860 [−0.01094, −0.00633]** | −0.00872 [−0.01108, −0.00637] (n=9,860) | −0.00873 [−0.01109, −0.00644] |

Every interval excludes zero. Scores: at k=8 the z-only policy reaches **0.29536** against the tree's
**0.30410** and its own frozen order's **0.30397**.

> ### ★ THE FALSIFIER FIRES — for the belief-only policy
>
> R's pre-registered falsifier: *"if the policy adapts and still loses to the tree on seeded states —
> the setting built to disadvantage the tree — the 'supports new states' claim is dead and is
> reported dead."* In R it did not fire because the policy never adapted. **Here it fires.** The
> z-only policy adapts on 88–99% of users and loses to the tree at every k with intervals excluding
> zero. **For this arm the claim is dead and is reported dead.**

### 3. ★ The redundancy count says the change is not *about* the evidence

This was pre-registered as the sharpest single diagnostic: *a policy that reads the volunteered set
has no reason to ask about a film whose answer it already holds, so its re-ask rate should fall below
the tree's.* Re-asks of a volunteered item, **per item question asked**:

| k | static | tree | both-inputs policy | z-only frozen order | **z-only** |
|---|---|---|---|---|---|
| 2 | 1.348% | 1.417% | 1.404% | 1.425% | **1.442%** |
| 4 | 2.792% | 2.900% | 2.875% | 2.881% | **2.857%** |
| 8 | 5.195% | 5.459% | 5.355% | 5.441% | **5.429%** |

**It changes 9 to 13 of its 16 questions and re-asks volunteered films at exactly the tree's rate** —
higher than the tree's at k=2. Not one of the five arms separates from any other on this diagnostic.

> **So the honest sentence is: the belief-only policy is SENSITIVE to volunteered evidence but not
> INFORMED by it.** Its interview moves, and the amount it moves scales with how much was
> volunteered, but the movement does not carry the one property that would show the content was
> used — and it costs 0.9 to 1.2 points of NDCG.

### What this settles, and what it corrects

1. **The mechanism's first half is confirmed and is worth keeping.** The explicit asked-history is
   what freezes the both-inputs policy on seeded states: remove it and the policy responds. That is a
   real, cheap, measured finding about *why* R came out null.
2. **The mechanism's second half is refuted.** *"The asked-history feature was the thing breaking
   it"* implies that removing it fixes the behaviour. It does not. It converts a policy that ignores
   the evidence into one that churns on it — **undirected churn, not conditioning**, on the one
   diagnostic that can tell them apart.
3. **Neither policy has a behavioural "supports new states" capability.** The both-inputs model does
   not respond; the belief-only model responds badly. **The capability remains architectural in both
   arms, and the falsifier has now fired on one of them.**
4. **This is the exact case the 138-state fine-tune exists for** — the capability is real and the
   behaviour is untrained — and the fine-tune was **not run**, per instruction.

**One thing that is NOT a defect of the z-only arm:** it is still above the static order at k=2
(+0.00409 [+0.00130, +0.00692]) and k=8 (+0.00311 [+0.00053, +0.00571]), and a null at k=4. It has
not collapsed; it has traded a good fixed order for a worse adaptive one. Its answer rate is
unmoved (0.576 / 0.580 / 0.591 against the tree's 0.579) and its concept share drifts up
(0.5608 / 0.5718 / 0.5877 against 0.5457), so the churn is not a channel collapse either.

**Cost:** 6.1 minutes for all three k (four of six cells per k resume from R's banked artifacts).

## Cost — a property of this box, not of the method

| | measured |
|---|---|
| environment build (25M-row ratings, tower, concept net, pool) | ~1.2 min |
| one static/tree/ask-nothing arm at one k (10,000 users, 16 questions, scored) | **~0.2 min** |
| one seeded **policy rollout** at one k (16 turns, refold each turn) | **~1.8 min** |
| all 12 cells + all comparisons | **~12 min total** |

The Part H.2 estimate was 3–5 h. It came in at **12 minutes**, because the scoring funnel evaluates
a whole 10,000-user cohort in ~15 s and this experiment needs one evaluation per cell rather than a
candidate sweep. Costs are properties of this box; the estimate was wrong and it is recorded as
wrong.

## Artifacts

| what | where |
|---|---|
| per-cell results + **per-user vectors** (full, tail, asked, outcomes, redundancy, seed sizes) | `experiments/battery/resume/resume_k{2,4,8}_{none,static,tree,policy}.json` (+ `_per_user.npz`) |
| the combined artifact (all k, all comparisons, all CIs) | `experiments/battery/resume/policy_resume.json` |
| logs | `experiments/battery/resume_k2.log`, `resume_k48.log`, `resume_all.log` |
| code | `src/instrument/policy_resume.py` |

**Reproduce:** `python src/instrument/policy_resume.py --ks 2,4,8 --threads 6 --i_have_author_goahead`
(every cell resumes from disk under a contract check, so a rerun costs the environment build only).
