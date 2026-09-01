# THE PARAMETRIC POLICY — design sheet and full option map (workstream E / TODO 1.16)

**Status: DESIGN ONLY. Nothing here has been run. No model has been trained, not one epoch.**
Written 2026-08-13. Owner doc for workstream E of `PAPER_B_PLAN_2026-08-12.md` §6 and TODO 1.16.

> **AUTHOR SAID (2026-08-12):** *"i really dislike tree for lack of scalability … do a parameterised
> policy, that would avoid the weakness of subbing objective."*
> **AUTHOR SAID (2026-08-12):** *"policy that beats the tree ideally, as it can go to deeper states and
> scales better."*
> **AUTHOR SAID (2026-08-13):** *"i wanna focus on policy"* … *"do not train yet, just document all ideas."*

Everything below the line marked **PART A** is the design sheet proper (the project rule: question,
Ns, action space, baseline symmetry, metric + MDE, every shortcut with its non-lossy alternative and
cost, gates, and the abandon condition). **PART B** is the full option map — including the options we
are *not* taking, each with its reason, its cost, and the result that would revive it. **PART C** is
what was built (code, uncommitted-to-run) and the exact commands.

---

# PART A — THE DESIGN SHEET

## A.0 The question, in one sentence

*Can a single set of parameters, reading only observable interview state, reproduce the NDCG tree's
held-out conditioning gain — and then exceed it at the states the tree's min-leaf floor forbids it
from reaching?*

Two sub-questions, gated in order:

- **E1 (fidelity):** does a parametric policy trained on the tree's own exact-NDCG argmaxes reproduce
  the tree's test gain over the static order, within the tree's own CI?
- **E2 (payoff):** trained additionally on labels at states below the tree's floor stops, does it
  **beat** the tree on the disjoint test cohort, paired CI?

E1 is not a contribution on its own. It is the instrument check that licenses E2. If E1 passes and E2
does not, the honest result is *"the observable state carries the tree's edge but no more"* — which is
a measurement, not a paper section.

## A.1 Why this can work now when it failed in June 2026

Three root causes were **measured** in June, not guessed. All three have a fix that did not exist then.

| root cause, as measured (June 2026) | evidence | the fix now |
|---|---|---|
| **The teacher was unlearnable or wrong.** Cloning the clairvoyant oracle gave **top-1 agreement 14%** and lost to greedy — the oracle's edge is privileged (it sees the target). The realizable teachers were surrogate-objective (greedy info-gain) and near-static: BC collapsed to **a fixed 17-item list asked to all 291 users in identical order**. | `night-adaptive-edge-investigation`, `clairvoyant-imitation-breakthrough` | **The NDCG tree is the first teacher that is simultaneously realizable, metric-aligned, and proven to win held-out.** Realizable: every branch is decided from fold-in evidence only (`leak_assert`, 0/10,000). Metric-aligned: the split scalar *is* cohort-mean measured NDCG@10. Proven: **+0.01035 full / +0.00705 tail at K=16 on 10,000 disjoint test users, K=1 exactly 0.0 by construction** (`CONDITIONED_ARMS_ON_TEST.md`). And the split-criterion ablation proves it is the *measured* criterion that does the work: swap only the scalar and the sign of the conditioning gain flips (pairwise −0.03041, variance −0.03061, logistic −0.08627). |
| **Reward SNR.** The routing signal was **2.1% of cross-user reward variance**, so softmax policy gradient provably collapses to the best fixed order. | `night-adaptive-edge-investigation` (M1) | **Imitation first.** Supervised cross-entropy on a fixed label has no reward-variance problem at all. RL only as an RLOO finetune *on top of a working imitator*, where the leave-one-out baseline removes the per-user variance that is the 98%. |
| **Wrong reward, wrong actor.** Coverage/entropy rewards *degraded* the belief state (cos(u,u\*) 0.731→0.710); a flat MLP has no per-candidate inductive bias (M2/M3). | `policy-ladder-ndcg-reward` (P1), `night-adaptive-edge-investigation` | **Exact held-out NDCG@10** as the only reward (P1 already showed this reverses the decoherence: cos 0.717→0.782, adaptivity tripled). **Equivariant per-candidate scorer** — the same architecture that, distilled from a *realizable* greedy teacher, reached top-1 49% and the best realizable score in June. |

Old **magnitudes** from June are demoted (retired stack, retired answer environment). The
**mechanisms** are stack-independent and are what is being relied on here.

## A.2 The firewall — non-negotiable, and stated in code

- Labels, training, early stopping and model selection use **validation states only**.
- **The 10,000 canonical TEST users are never built, never loaded, never touched** during any part of
  E1 or E2 label extraction or training. The extraction script hard-refuses `--split test`.
- Exactly **one** test look per gate, reported with the paired percentile bootstrap (10,000 resamples,
  seed 98765) over per-user differences, and the **val→test gap reported explicitly**.
- Prior burn respected: held-out evaluation is what exposed RL gaming in June, with realisable gain
  ≈ **one fifth** of the oracle's. **An unfalsified training-set win means nothing here.**

### A.2.1 The one leak this design cannot fully close, stated rather than buried

The tree was **grown on the validation cohort**. Its labels are therefore already fitted to those
10,000 users, and a policy trained on those labels is fitted to validation *twice*. Worse, splitting
validation into policy-train / policy-dev **by user** does not fix it: a node's label was computed
from **all** the node's users, dev users included, so the dev fold's labels are contaminated by the
dev fold's own targets.

Three honest options, with cost. **An author ruling is wanted before E2, not before E1.**

| option | what it buys | cost |
|---|---|---|
| **(i) Accept it for E1.** Use a user-level 8,000/2,000 dev split (seed 98765) as an *in-sample* early-stopping signal only, label the contamination in the artifact, and treat the single test look as the only out-of-sample number. | E1 costs nothing extra and E1 is only an instrument check. | **0 h** |
| **(ii) Regrow the tree on an 8,000-user sub-fold** so the 2,000-user dev fold is genuinely clean, and use it for model selection before any E2 test look. | Makes E2's model selection defensible. This is the option I recommend *before* any E2 claim. | **~8–10 h** (the full tree cost 10.3 h on 10,000; ~0.8× on 8,000) |
| **(iii) Generate labels on the 140,768 TRAIN users.** 14× more users, the natural fix. | **BLOCKED, and it is worth knowing why:** `data/ml-25m/proc/` has no `train_tr`/`train_te`, so train users have **no fold-in/target split**, and the tower was **trained on them** — any NDCG measured there is memorisation, not held-out value. | building a train-side fold-in/target split + re-certifying the ruler; **not costed, and not recommended without a separate design sheet** |

## A.3 The cohort, the action space, the budget — identical to the tree, deliberately

| | |
|---|---|
| **Selection cohort** | the 10,000 canonical VALIDATION users (`liang_split.py --data ml-25m`, seed 98765) |
| **Reporting cohort** | the 10,000 canonical TEST users, disjoint, one look per gate |
| **Action space** | the **combined 1,000-candidate pool**: 500 popularity-ranked items (`_ranked_pool("pop")`) + 500 answerable concepts (`_ranked_pool("answerable")`), items first — byte-for-byte the tree's pool, built by the same `build_pool` call. No truncation anywhere (HARD RULE 1). |
| **Budget** | **16 charged questions**, every question folded into the recommender. An asked-but-unrated item folds as the i26 native UNSEEN exposure token; an unanswerable concept as a cv3 refusal token. **No free side information.** |
| **Recommender** | the certified pair `t2i26c_ep31.pt` + `concept_cv3_ep08.pt`, frozen |
| **Mask** | **policy B**, derived `pool_contract` fingerprint asserted on every artifact; `mask_contract.assert_fixed_mask` live in the scoring funnel |
| **Metric** | full **and** tail NDCG@10, learned decoder bias, NO-FLOOR, full 18,359-item catalogue. Full is primary; **tail is a secondary outcome and no search optimises it** |

## A.4 Baseline symmetry — what the policy is compared against, and why it is fair

Two comparators, both already banked, both on the same ruler:

1. **`greedy_static_policyB_combined`** (16 picks, policy-B-selected, `CONDITIONED_ARMS_ON_TEST.md` §9)
   — the same greedy machinery, same objective, same pool, same cohort, same stack, no conditioning.
   This is the comparator the tree's own headline is measured against, so a policy gain quoted
   against it is directly comparable to **+0.01035 / +0.00705**.
2. **The NDCG tree itself** (`ndcgtree16test_policyB.json`) — the arm E2 must beat.

The comparison is fair because *conditioning is the only thing that differs*: identical action space,
identical budget, identical charged-question accounting, identical recommender, identical mask,
identical cohorts, identical answer rule. The policy's single extra degree of freedom over the tree is
that **it may ask two users in the same tree node different questions**. That degree of freedom is the
treatment.

**Two asymmetries that must be declared, not smoothed over:**

- The tree is **budget-unaware** (its leaf tails are greedy prefixes, prefix-optimal, not optimised
  for a chosen length). The policy will be given the remaining budget as an input, which is a real
  advantage. Either give the tree the same information (it cannot use it — its structure is fixed) or
  report the policy's gain with and without the budget feature. **Recommend: report both; the
  no-budget-feature variant is the like-for-like number.**
- The policy sees the tower latent `z_t`; the tree sees only the branch path. This is *also* a real
  advantage and it is the point (a policy that only saw the path could not generalise off-tree). It
  must be named in the chapter, because a reviewer will otherwise name it.

## A.5 Exact Ns

Measured from `ndcgtree.json` / `ndcgtree16.json` / the banked CELF heap sidecars.

| quantity | N |
|---|---|
| tree nodes | 22 (7 internal + 15 leaves), depth 5 |
| **distinct states the tree visits at 16 questions** | **202** (7 split states + Σ over leaves of `16 − depth`) |
| distinct states by position k=0…15 | 1, 3, 9, 11, 13, then **15 at every k ≥ 5** |
| **E1 training examples (user, position) pairs** | **160,000** = 10,000 users × 16 positions |
| **distinct label values behind those 160,000 examples** | **202** — one per state |
| distinct questions used as a label | **172** (76 items, 96 concepts) of the 1,000-candidate pool |
| label channel, user-weighted | **45.4% item / 54.6% concept** |
| (state, candidate) exact-gain pairs needed for full gain vectors at every visited state | **200,244** |
| of those, **already banked exactly** in the CELF heap sidecars | **21,943 (11.0%)** |

**The single most important N on this page: 160,000 examples carry only 202 distinct targets, and
only 15 of them at any position ≥ 5.** That is the E1 label set's real information content, and it is
why E1 is an instrument check rather than a result. Two users in the same leaf at position 9 have the
same asked list but **different answers** — so the policy's *input* varies while its *label* does not.
Fitting that perfectly means learning to ignore the answers. **This is the exact mechanism by which
E1 could pass its numerical gate and still be a collapsed static order**, and it is why the
anti-collapse gate (A.9, G-E1c) is mandatory and not decorative.

E2 exists precisely to break this: kernel-weighted labels are computed *per state*, so distinct
states get distinct labels and the label count stops being 202.

## A.6 Cost — from a MEASURED per-state cost, not a projection

Fitted over all 22 real tree nodes (`ndcgtree.json`, `sec_per_eval` vs `n_users`):

> **s/eval = 0.0858 + 0.8157 × (n_users / 1000) seconds**, R² = 0.985, max residual 0.79 s.

Anchors, all measured, none extrapolated: the **root** paid **8.48 s/eval** at n=10,000, so its
1,000-candidate sweep took **141.8 min**. The whole 15-question tree cost **612 node-minutes / 10.3 h
wall**. The policy-B combined static rebuild cost **104.6 min** for 2,167 evals; items-only **52.7
min**; concepts-only **134.3 min** (search cost is a property of the *bank*, not the budget).

| job | evals | cost |
|---|---|---|
| **E1 labels (argmax only) — the briefed label set** | **0** | **0 h. They are already banked**, in `ndcgtree16.json`'s node splitters and leaf `own_sequence`s. |
| Gate the banked reuse (base equality + per-state re-evaluation sample + partition recompute) | ~200–400 | **0.5–1.0 h** |
| Full 1,000-candidate exact gain **vectors** at every visited state, cold | 200,244 | **40.8 h** |
| …the same, reusing the 11.0% banked | 178,301 | **31.8 h** |
| One state's full vector at the **root** (buys the NDCG-vs-DCG argmax disagreement number, A.8) | 1,000 | **2.4 h** |
| One deeper state's full vector (n ≈ 700) | ~985 | **0.18 h** |
| E2 labels at the 15 floor-stopped nodes, one level deeper, exact | ~14,800 | **3.3 h** (measured by `policy_labels.py --cost_only` over the 42 below-floor states; supersedes the earlier ~5–7 h estimate) |
| Regrow the tree on an 8,000-user sub-fold (option (ii), A.2.1) | ~30,000 | **~8–10 h** |
| E1 training itself (160k examples, ~2M-parameter scorer, CPU) | — | **~1–3 h**, and it is the *cheap* part |

**The cost structure, in one sentence:** at any fixed position the states partition the whole 10,000-user
cohort, so a full sweep at one depth always costs about one root sweep (~2.4 h) — which is why label
cost scales with **depth × pool**, is ~40 h for the tree's own states, and is the binding constraint on
this entire workstream. Model capacity is not the constraint. **It never was.**

## A.7 The architecture, and why equivariance matters *here*

**Scorer:** `s(q | state) = g_θ(φ_q, h_ψ(state))`, one shared parameter set applied to every candidate
independently, softmax over the 1,000 (minus already-asked). Policy = `argmax_q s(q | state)`.

`φ_q` is a **frozen** candidate descriptor built from the certified stack, so no new representation is
learned and nothing can drift from the ruler:

- **item** *i*: the decoder row `W_d[i]` (d-dim), `log(1 + train count)`, item answer rate;
- **concept** *c*: the cv3 concept embedding `emb.weight[c]`, the bank's answer rate, the signed-value
  std (`polar`);
- plus a 2-dim channel one-hot, so items and concepts are scored by one head without a channel prior
  being hard-coded.

**Why equivariance and not a 1,000-way softmax head.** Three reasons, in order of force:

1. **It is the scalability claim.** A 1,000-way output head is a lookup table over *this* pool: it
   cannot score a candidate it never saw, so the policy is exactly as un-scalable as the tree, and the
   author's stated motive evaporates. An equivariant scorer runs on the **full 18,359 items + 1,031
   concepts** as 19,390 dot products in one forward pass. *That* is "scales better", and it is
   measurable rather than asserted.
2. **June measured the alternative failing.** H3 was proven: a flat MLP actor lacks per-candidate
   inductive bias; the equivariant per-item actor is what made distillation work (top-1 14% → 49%).
3. **Statistical.** 172 distinct label questions over 202 states. A 1,000-way head has 828 outputs that
   never appear as a positive. A shared scorer has none.

**Formal property to unit-test (it is in the committed code):** permuting the candidate list permutes
the scores exactly, and dropping a candidate does not change any other candidate's score.

**One correction to the borrow list.** `POSITION_VS_MODERN_CRS.md` B.1 §1 recommends MCMIPL's dueling
decomposition `Q = f_V(s) + f_A(q, s)` as the fix for the 2.1% SNR pathology. It is the right idea in
the wrong stage: **in a softmax policy a state-only term cancels exactly** and buys nothing in E1. Its
correct home is **E3**, where `f_V(s)` is the value baseline that removes cross-user reward variance —
i.e. it is a *critic*, and RLOO's leave-one-out baseline is a cheaper, unbiased instance of the same
idea. **Recommendation: skip the dueling head; use RLOO's paired baseline, which is the same variance
fix with no extra parameters.**

## A.8 State encoding — and the one thing that must not be collapsed

`h_ψ(state)` is fed **three** things, and the recommendation is *all three*, with a two-cell ablation:

1. **The tower latent `z_t`** — the actual fold of the answers so far, i.e. the recommender's own
   belief. Free (already computed inside the scoring env).
2. **The answer history, explicitly** — for each asked question, `φ_q ⊗ (three-way outcome one-hot)`,
   pooled. `z_t` is a *lossy* summary trained for reconstruction, not for question selection; the
   history carries what `z_t` discarded, in particular **which questions were asked and not answered**,
   which by construction leaves `z_t` almost unchanged (an unanswered item folds as the UNSEEN token, an
   unanswerable concept as a refusal token).
3. **Position / remaining budget** `t`, `16 − t` — with the caveat in A.4 that the like-for-like
   comparison against the tree is the variant without it.

> ### ★ "Did not answer" is a THIRD outcome and must never be encoded as two
>
> A two-outcome encoding (`liked` / `not-liked`) would erase the property this entire chapter turns on.
> The measured stakes:
> - **8 of the tree's 15 stopped nodes** were stopped by the **did-not-answer branch** of a concept
>   **92.7–100%** of the node could answer (`TREE_DEPTH_AND_MINLEAF.md` §2);
> - the root's three children are the three outcomes, and their baselines are **0.374 / 0.320 / 0.087**
>   — answering *at all* is worth **~+0.25**, against **+0.048** for the largest question-choice effect
>   anywhere in the tree;
> - **~14% of users answer nothing** and sit at 0.087 while everyone else moves;
> - the tree's own branch rule is explicitly ternary, and every gate in this project is stated on
>   ternary child sizes.
>
> So: **three-way one-hot per asked question, and a distinct learned "asked, no answer" token.** In
> addition the state carries a scalar count of unanswered turns, because a user with six unanswered
> turns is a different object from a user with none, and `z_t` cannot say so.

**The cheap ablation E1 owes:** `z_t` only vs history only vs both. Three training runs at ~1–3 h each.
If `z_t` alone matches "both", the chapter can say the tower's latent is a sufficient statistic for
question selection, which is a clean, quotable result. If it does not, that is a **measured defect of
the tower's latent** and is worth more.

## A.8b Both normalisations — the author's standing decision, implemented

**Decision, already taken by the author and implemented in the extractor:** every candidate sweep
computes **both** cohort-mean per-user-normalised NDCG@10 (the current criterion) **and** cohort-mean
**unnormalised** DCG@10, off the same scored per-user vectors, and reports how often their argmaxes
disagree.

**It is genuinely free, and here is why:** for binary relevance, `DCG_u@10 = NDCG_u@10 × IDCG_u@10`,
and `IDCG_u@10 = Σ_{i≤min(n_u,10)} 1/log2(i+1)` depends only on the user's target count — **no model
call, no extra scoring pass**. The extractor computes the weight vector once and reduces twice. The
expensive part is scoring the candidates, and both readings come off the same vectors.

**The cheap number I can report today without a run** — the per-user weights, measured on the 10,000
validation users:

| IDCG@10 over the 10,000 validation users | value |
|---|---|
| targets per user | mean 14.98, median 8, p05 2, p95 52, max 468 |
| IDCG@10 | mean 3.464, median 3.954, **min 1.000, max 4.544**, sd 1.161 |
| users with ≤2 targets (the low-IDCG users Bilgic & Bennett warn about) | **12.7%** |
| users with ≤5 targets | 38.9% |

**Reading:** per-user normalisation upweights the sparsest 12.7% of users by up to **4.5×** relative to
the densest. That is a large, structured reweighting, not a rounding difference — so the two criteria
*can* pick different questions, and Bilgic & Bennett's mechanism (selection dragged toward low-IDCG
queries) has a live target in our cohort. It also touches an open puzzle: **6 of 15 leaves are negative
on test and ~14% of users answer nothing** — if the criterion is being dragged by low-IDCG users, that
is a candidate mechanism for both.

**The disagreement rate itself cannot be got for free.** The banked CELF heaps store only cohort-mean
**NDCG** scalars — no per-user vectors per candidate — so the DCG reading cannot be reconstructed from
them. **The cheapest instance that would produce a real number is one full root sweep: 1,000
candidates × 10,000 users = 2.4 h**, and it would give the disagreement rate at the single most
consequential state in the chapter (the root, whose argmax is the first question of every arm). At a
deeper state (n ≈ 700) it is **11 minutes**. Recommend running the root sweep as the first paid job of
workstream E, whatever else is decided — it is 2.4 h for a number that bears directly on TODO 1.22/C3
*and* on E3's reward normalisation.

**What we will NOT do:** change the reported metric. HARD RULE 8 — the ruler is fixed, full and tail
NDCG@10. This is a question about the *selection criterion and the training reward*, never about the
report.

## A.9 The gates — stated before anything runs

**Label-extraction gates (every banked reuse, enforced in `policy_labels.py`):**

| gate | what it asserts |
|---|---|
| **G-L0** pool identity | the env's derived `pool_contract` (`nnz`/`per_user`/`fp`) and the pool's sha1 equal the tree run's recorded ones. A missing field is a refusal, never a pass (`resume_ok`). |
| **G-L1** base equality | `genv_mean(E, prefix, rows)` recomputed in-process **`==`** the banked heap `base`, exact `==`, no tolerance. This is the check that separates policy A (0.12987336) from policy B (0.15686538) instantly. |
| **G-L2** per-candidate byte-identity | `--verify_n` candidates per state re-evaluated from scratch; the reconstructed value must equal `base + banked gain` **exactly (0.0e+00)**. |
| **G-L3** partition recompute | the ternary child sizes and answer rate at every reused node reproduce the banked `child_sizes` / `answer_rate` **exactly** (the G1/G2 gates `tree_minleaf_audit.py` already passed 1000/1000 and 22/22). |
| **G-L4** argmax consistency | where a banked level-0 vector is complete, its argmax **is** the banked pick. |
| **G-L5** no test | the env is built with `split="validation"`; `--split test` is refused by the parser. |

**Promotion gates:**

| gate | criterion | why this number |
|---|---|---|
| **G-E1a** fidelity | top-1 agreement with the tree's label on held-out policy-dev states **≥ 50%** | June: the teacher that *worked* cloned at 49%; the one that failed cloned at 14%. This is a floor read off our own measured history, not a convention. |
| **G-E1b** ★ the briefed gate | the policy, restricted to tree-visited states, scores a K=16 test gain over `greedy_static_policyB_combined` **inside the tree's own CI: [+0.00739, +0.01323] full** | it is the tree's published interval; reproducing the teacher within its own uncertainty is what "imitates it" means numerically |
| **G-E1c** ★ anti-collapse | **> 3 distinct realised interviews**, and the Pep divergence rate (two users answer the same question differently ⇒ the follow-up differs) **≥ the tree's own value on the same cohort** | the exact June failure (BC → one fixed 17-item list) and the exact live failure we have already caught once: `conlinucb_bs_static` had **1 distinct interview** — a static order wearing an adaptive label |
| **G-E1d** shuffle control (HARD RULE 5) | a policy trained on **label-shuffled** states must NOT clear G-E1b | a gate a broken instrument can pass is not a gate |
| **G-E1e** canonical snap (HARD RULE 2/5) | the policy's m=0 intercept reproduces **0.15686538** on validation and **0.15855 / 0.02819** on test, exactly | every new regime in this project is snapped on Most-Popular/m=0 first (`surprise-means-debug-the-harness`) |
| **G-E2** ★ payoff | **beat the tree on test at K=16 on FULL, paired percentile bootstrap CI excluding zero** | the author's stated motive |
| **G-E2-pre** ★ pre-registered no-go | if E2's **validation** gain over the tree is **< +0.011 full**, do NOT take the test look | see the MDE below: this chapter's measured val→test shrinkage on full is **2.70×**, and the test MDE is +0.0042, so 2.70 × 0.0042 ≈ **+0.0114** is the smallest validation gain that could plausibly survive. Spending a test look below it is spending the one thing we cannot get back. |

### MDE

The tree's own K=16 test interval is `+0.01035 [+0.00739, +0.01323]` on full ⇒ half-width 0.00292 ⇒
**SE ≈ 0.00149** on 10,000 paired test users; tail SE ≈ 0.00158. For a **policy − tree** difference the
per-user differences will be *smaller* in spread than policy − static (the two adaptive arms share
their prefix on most users), so this is a conservative SE. At α = 0.05 two-sided and 80% power the
**MDE ≈ 2.8 × SE ≈ +0.0042 full**.

**State this plainly, because it decides whether the workstream is worth running:** the tree's entire
conditioning gain is **+0.01035**. For E2 to separate, the policy must add **at least ~40% on top of
everything the tree achieves**. That is a demanding target and the author should see it before, not
after.

## A.10 What would make us ABANDON this line rather than iterate

The June failure was *iterating on a signal that was not there*. Pre-registered abandon conditions,
so that does not recur:

1. **E1 imitates well and still loses.** G-E1a passes (top-1 ≥ 50%) but G-E1b fails low. Conclusion:
   the tree's advantage is **not a function of the observable state**, so no parametric policy on this
   state can carry it. Record as a measurement (it is a genuine, publishable decomposition of the same
   shape as June's privileged-oracle result) and **stop**.
2. **E1 passes only by collapsing.** G-E1b passes but G-E1c fails. A static order that clears the gate
   already exists — it is `greedy_static_policyB_combined` — so this is a null wearing a policy's
   clothes. **Stop**; do not "fix" it with entropy bonuses (June: entropy/coverage rewards *degraded*
   the belief state).
3. **E2 cannot clear its own pre-registered validation bar.** E2's validation gain over the tree lands
   below **+0.011**. Do not take a test look, do not tune, **stop**.
4. **Encoding-and-architecture exhaustion.** Three state encodings (`z` / history / both) and two
   architectures (equivariant scorer / pointer-attention) all land inside the same null band on
   policy-dev. The finding is "the observable state does not carry the tree's edge"; **stop**.
5. **A shuffle control passes any promotion gate.** Broken instrument. Stop and debug; never report.

**Not abandon conditions** (HARD RULE 6 — a failed experiment defaults to a weak method): a single
architecture failing; a single learning rate failing; E1 missing the CI on the *tail* only (tail is a
secondary outcome nothing optimises).

## A.11 What is claimed, and what is never claimed

- **Never** "adaptivity is large". The size qualifiers of the plan's §0 travel with every statement:
  the tree's conditioning gain is ~10% of the interview's own lift, and it is **smaller than the gain
  from changing the question BANK with no conditioning at all** (+0.01430 at equal budget).
- **Never** a routing claim. D2 settled it negative: the tree **partitions and diverges but does not
  demonstrably route** (answer-congruence null, p=0.82 / p=0.61). A policy that diverges is not a
  policy that routes, and G-E1c's divergence diagnostic detects *collapse* — it licenses nothing.
- **Never** "first ranking objective in an interview tree", "nobody compared adaptivity against a
  metric-optimised static order", "nobody used a truncated discounted ranking metric as a criterion".
  All three are false (§C.4 of the Hu & Yu write-up). Claim the **conjunction**.
- The parametric claim, if it lands, is: *the same exact objective, amortised into parameters, with no
  partition, reaching states a recursive partition cannot* — and the scalability half of it is
  **demonstrable** (one forward pass over the full 19,390-candidate catalogue) independently of whether
  E2's gain separates.

---

# PART B — THE FULL OPTION MAP

Everything considered, including what we are not doing. Costs are wall-clock on this box, derived from
the measured fit in A.6 where they involve scoring. **"Prereq"** names the result that must land first;
where an option is only worth doing if an earlier result goes a certain way, the way is stated.

## B.1 Ranked shortlist — what to do, in order

| # | option | cost | prereq | verdict |
|---|---|---|---|---|
| 1 | **E1a — imitate the tree on the banked argmax labels** (§B.2) | labels **0 h**, gating 0.5–1 h, training 1–3 h | none | **DO FIRST.** The labels are already on disk. Nothing else in this workstream has a better cost/information ratio. |
| 2 | **Anti-collapse diagnostic** (Pep-style divergence + distinct-interview count) (§B.9) | 2–4 h, reuses D2's code | E1a trained | **DO WITH E1a.** Without it, E1a's headline number is uninterpretable. |
| 3 | **Root sweep under both normalisations** (§B.6) | **2.4 h** | none | **DO EARLY, regardless.** Buys the NDCG-vs-DCG argmax disagreement number at the chapter's most consequential state, and de-risks E3's reward. |
| 4 | **State-encoding ablation** (`z` / history / both) (§A.8) | 3 × 1–3 h | E1a trained | **DO.** Cheap, and one of the two outcomes is a quotable result about the tower's latent. |
| 5 | **E2 — kernel-weighted labels below the floor stops** (§B.3) | labels ~5–7 h (one level) or ~31.8 h (all states, full vectors), training 1–3 h | E1 gates pass | **THE PAYOFF. Do only after E1 passes G-E1a/b/c.** |
| 6 | **Regrow the tree on an 8,000-user fold** for a clean dev split (§A.2.1(ii)) | 8–10 h | before any **E2** claim | **DO before E2's test look**, not before E1. |
| 7 | **E3 — RLOO finetune on exact NDCG reward** (§B.4) | ~19 h + build | only if E2's supervised ceiling binds | **DEFER.** Design recorded; do not build yet. |

## B.2 The staged imitation line (the briefed design)

### E1 — imitate the tree

- **Labels:** the tree's exact measured-NDCG argmax at each visited state — the same quantity the split
  search computes — plus the leaf greedy tails. **16 labels per validation user, 160,000 (user,
  position) examples, 202 distinct states, 172 distinct label questions.** No surrogate anywhere in the
  label: it *is* the metric argmax.
- **Provenance:** entirely banked. Splitters from `ndcgtree.json` nodes, tails from `ndcgtree16.json`
  `leaf_table[*].own_sequence`, all gated per A.9.
- **Loss:** cross-entropy over the 1,000-candidate pool (masking already-asked). Optional listwise
  upgrade in §B.7.
- **Gates:** G-E1a…e.

### E2 — beat the tree where it starves

The target is measured precisely, which is what makes E2 a design rather than a hope
(`TREE_DEPTH_AND_MINLEAF.md`):

- **all 15 stopped nodes were FLOOR stops with a strictly positive gain still in hand** (+0.00672 …
  +0.04251). Not one stopped because no question helped;
- **8,398 users (84.0%) sit below a node that had more to ask** but could not split under the
  ternary-plus-200 rule;
- 8 of 15 stops were forced by the **did-not-answer branch of a concept 92.7–100% of the node could
  answer** — the property that makes a concept worth asking is the property that ends the branch;
- the admissible-candidate pool collapses from 876/1,000 at the root to **0 at n = 660**.

**A parametric policy has no floor**, because it does not partition: every user's every turn trains the
same weights. That is the mechanism, and it is the author's stated motive stated as a testable claim.

**Label generation at those states** = TODO 1.14's kernel-weighted cohort estimate: at a state, weight
**all** 10,000 validation users by similarity to that state and maximise the weighted objective. Soft
partitioning, so the effective sample never shrinks and depth *sharpens localisation* instead of
starving the estimate. Nothing is fitted, so nothing overfits.

⚠ **The kernel bandwidth is a shortcut, and here is its non-lossy alternative and cost.** Truncating a
Gaussian kernel at some radius to save evaluations is a HARD RULE 1 truncation and needs sign-off. The
non-lossy alternative is to **declare a compactly-supported kernel as the estimator itself** (Epanechnikov
/ tricube), in which case the zero weights are a property of the estimator, not a sample reduction, and
nothing is being silently dropped. **Recommend the declared compact kernel.** Cost: identical.

⚠ **Second shortcut, and this one is the expensive one.** Full exact labels at deeper states cost a
1,000-candidate sweep each. One level below all 15 floor stops ≈ **5–7 h**; every visited state with
full vectors ≈ **31.8 h**. The non-lossy alternative to shortlisting is *paying it*, and at one level
deep that is affordable — **so do not shortlist for E2's first level.** If E2 needs several levels, see
§B.5.

### E3 — RLOO finetune on exact NDCG reward

Designed, **not built** (per the brief and the author's "do not train yet").

- **Reward:** the exact held-out full NDCG@10 of the realised 16-question interview. Nothing else. Not
  EIG, not coverage, not entropy — June measured those *degrading* the belief state.
- **Variance control:** RLOO — `k` rollouts per user, each one's baseline being the mean of the other
  `k−1`. This is the direct fix for the measured 2.1% SNR, it is unbiased, and it needs no critic
  (which is why the MCMIPL dueling head is redundant here, A.7).
- **Cost:** the reward for one rollout set over 10,000 users is one root-scale evaluation (~8.5 s);
  RLOO with k=4 ⇒ ~34 s/gradient step; 2,000 steps ⇒ **~19 h**, plus build time.
- **Firewall:** rollouts and rewards on validation users only.
- **Trigger:** only if E2's supervised ceiling looks binding — i.e. E2 passes G-E2 but the policy's
  gain plateaus while the label set is still growing, or E2's labels are exhausted and the gap to a
  measured per-user oracle stays large.

## B.3 The non-parametric alternatives already in the TODO

### TODO 1.14 — kernel-weighted local greedy: **a tool, and (barely) not a method**

This is the sharpest cost finding in the sheet, so it is stated with its arithmetic.

- **As E2's label generator (a TOOL): affordable.** Labels are needed at a *chosen, small* set of
  states, offline. One state ≈ 2.4 h at full cohort weight; one level below all 15 floor stops ≈ 5–7 h.
- **As a deployable adaptive METHOD (an ARM in the battery): infeasible at our scale, measurably.** A
  deployed kernel-greedy must score 1,000 candidates *at every state the population actually occupies*.
  Act 2 measured that population: **11 / 112 / 1,463 / 5,767 distinct states at k = 1 / 2 / 4 / 8**. Each
  (state, candidate) evaluation folds all 10,000 weighted users, i.e. the root price of 8.48 s. At k=8
  alone that is 5,767 × 1,000 × 8.48 s ≈ **13.6 million CPU-seconds ≈ 3,772 h**. `genv_eval_multi`
  batches heterogeneous per-user sequences in one pass, but it cannot reduce the *fold count*: the
  kernel estimate at state *s* requires every weighted user folded with *s*'s question list, so the
  10,000 × n_states cross product is irreducible.

**Verdict: build 1.14 as E2's label generator; do NOT propose it as a battery arm.** If it is ever
wanted as an arm it needs §B.5's shortlist, and then it is no longer selecting on the exact metric over
the full pool — which is the one thing this chapter cannot give up.

### TODO 1.15 — criterion-argmax: **cheap, orthogonal, and gated on a result we have not got**

Ask the question maximising expected NDCG gain **under the tower's own predictive** at that user's
state. Zero parameters ⇒ no fragmentation, no val→test shrinkage, and accumulated state strictly
improves the estimate. Genuinely attractive.

**But it is gated on 1.11 (C1), and the gate is real.** 1.11 measures whether that criterion tracks
*realised* NDCG gain. Without it, a failure is uninterpretable — "just another weak method" (HARD RULE
6). Two partial readings exist and they point in **opposite** directions:

- **Encouraging:** the completed root-state probe (`pwheldout_root.json`, all 1,000 candidates) gives
  the held-out pairwise criterion **Spearman +0.784, regret +0.00459 vs +0.00161 random (29%), rank 2
  of 1,000** — a *measured-value* criterion tracks the metric well.
- **Discouraging, and this is the one that matters for 1.15:** at the same node the **training-loss**
  pairwise criterion gives **Spearman −0.149, rank 75, regret 92% of random**. Separation (2) —
  training loss versus measured value — does almost all the work. A model-predictive criterion is on
  the *training-loss* side of that line unless proven otherwise.
- And the argmax **disagrees** even for the good criterion (concept 704 vs our concept 114).

**Verdict: keep 1.15 gated on C1 (~12 h). Build it only if C1's rank correlation holds at depth, not
just at the root.** Ranked below E1a because E1a needs nothing and 1.15 needs 12 h of prerequisite.

## B.4 TODO 1.13 — the differentiable shortlist (the scalability lever)

Linearise the objective in the tower's latent `z`: compute `∂L/∂z` once per state, estimate `Δz` per
candidate answer from the encoder, and the per-candidate cost becomes a **dot product** instead of a
full 18,359-item decode. Then **re-rank the top ~m with the exact metric**, so selection stays exact.

**What recall the filter needs for the claim to survive.** The chapter's claim is that the arm selects
on **exact measured NDCG@10 over the full declared pool**. A shortlist preserves that claim **only if
the shortlist contains the true argmax**. So:

- **The bar: the shortlist must contain the exact full-pool argmax at ≥ 95% of the states it is used
  at, verified against full exact sweeps at a sampled set of states — and the sample must include deep,
  small-*n* states, not just the root.**
- At any state where it does not, the label is *not* the exact argmax, and the honest description is
  "**the exact argmax over an m-candidate shortlist**". Written that way it is defensible; written as
  "the exact argmax" it is a false claim about the objective, in a chapter whose whole thesis is
  measured-versus-surrogate.
- **What we know about achievable recall:** the best surrogate we have measured at the root ranks the
  true argmax **2 of 1,000** (top-10 recall 0.40, Spearman +0.784). So **m = 20 would have retained the
  root argmax**. One state is not evidence; it is encouraging.
- **Hard rule:** shortlisting the pool is a truncation (HARD RULE 1) and needs the author's sign-off
  **with the C1 correlations in hand**, not a hand-wave.

**Where it is allowed and where it is not:**

| use | verdict |
|---|---|
| E1's labels (tree-visited states) | **NOT ALLOWED.** Exact labels are already banked at zero cost. There is nothing to buy. |
| E2's first level below the floor stops | **NOT NEEDED.** 5–7 h exact. Pay it. |
| E2 at several levels, or a full-catalogue (19,390-candidate) pool | **The only route.** This is what 1.13 is for. |
| a deployable 1.14 kernel-greedy arm | the only thing that makes it feasible — and it costs the "exact over the full pool" claim |

**Cost:** ~6–10 h to wire, **after** C1 (~12 h). Partly built already: `split_criteria.py` implements
Hu & Yu's gradients and diagonal Hessians, verified against central finite differences.

## B.5 Soft / approximate ranking losses (ApproxNDCG, NeuralNDCG, SoftRank)

A three-way distinction, and getting it wrong would undercut the chapter's central claim.

| where | verdict | reason |
|---|---|---|
| **As the selection criterion / the label / the reward** | **NEVER.** Non-negotiable. | The chapter's separating axis is **measured versus surrogate**. B1 measured it: hold the tree code, split rule, min-leaf, answers, recommender, mask, pool, cohorts and budget fixed, change **only the scalar a question is scored by**, and the **sign** of the conditioning gain flips (ours +0.01035; pairwise −0.03041; variance −0.03061; logistic −0.08627). And upgrading a rating surrogate to a *ranking* surrogate buys nothing measurable: (b) ≈ (a), **+0.00020 [−0.00331, +0.00371]**, the only null in the matrix. Putting a smooth ranking surrogate in the objective is re-running the experiment we already showed loses, on ourselves. |
| **As the shortlist filter's scoring function (1.13)** | **ADMISSIBLE**, under §B.4's recall bar | the surrogate only *proposes*; the exact metric *selects*. This is exactly the boundary Hu & Yu themselves drew: *"since there are approximations in the derivations, we only use it to find the interview question."* |
| **As the policy's own training loss shape, over exact-gain targets** | **ADMISSIBLE, and worth doing** | This is not a surrogate for the metric — the *targets are the exact measured gains*; ApproxNDCG/NeuralNDCG is just a differentiable way to fit a ranking of candidates to those targets instead of a one-hot. It converts a 202-distinct-target problem into a 202×1,000-value problem, which is the single biggest statistical upgrade available to E1. **Prereq: the paid full-vector extraction (31.8 h).** |

**Ranking:** the listwise upgrade is worth 31.8 h **only if E1a's one-hot version passes G-E1a but
misses G-E1b** — i.e. the policy imitates the argmax but the realised interview underperforms. That is
precisely the symptom of a target set too coarse to convey *how much* each candidate was worth.

## B.6 Both normalisations — see A.8b

Implemented in the extractor at zero marginal cost. Cheap evidence today: **IDCG@10 spans 1.000–4.544
(sd 1.161), 12.7% of users have ≤2 targets**, so per-user normalisation upweights the sparsest users by
up to 4.5×. The disagreement rate needs one paid sweep (**2.4 h at the root**). Ranked #3 above.

## B.7 Architecture options for the actor

| option | inductive bias | cost | verdict |
|---|---|---|---|
| **Equivariant per-candidate scorer** (shared `g_θ(φ_q, h(s))`) | one parameter set scores any candidate ⇒ generalises to unseen candidates and to the full 19,390 catalogue; 172 distinct label questions is enough to fit it | 1–3 h/run | **PRIMARY.** June: this is the actor that made distillation work (top-1 49%). |
| **Pointer / attention over the pool** (query = state, keys = candidates) | adds candidate-candidate interaction, so the policy can express "don't ask two near-duplicates" — which greedy expresses only through refolding | 1–3 h/run | **THE ABLATION.** Run it against the primary once E1a has a number; it is a strict superset of the primary and therefore also strictly more able to overfit 202 targets. |
| **Flat MLP with a 1,000-way head** | none. A lookup table over this pool. | 1 h | **CONTROL ARM ONLY.** Its failure is measured (June H3) and reproducing it on the new stack is the HARD RULE 2 "start simple" anchor. **Never the primary** — and note it cannot make the scalability claim at all. |
| **Wolpertinger** (continuous proto-action + kNN in action space) | designed for 10⁵–10⁶ actions | 8–12 h | **REJECT at our scale, keep the reasoning.** Our pool is 1,000 and the full catalogue is 19,390 — the equivariant scorer scores all of them exhaustively in one forward pass, so the approximation buys nothing and costs a non-differentiable nearest-neighbour step plus the proto-action head, which is the flat-actor pathology reintroduced. Revive only if the action space becomes genuinely continuous — which is **Paper C's** continuous-*answer* line, not this discrete selector. |
| **Dueling head** `f_V(s) + f_A(q,s)` (MCMIPL borrow) | absorbs cross-user variance | 4–8 h | **SKIP in E1** — a state-only term cancels in a softmax policy. **Its correct home is E3**, and there RLOO's leave-one-out baseline is the same variance fix with zero extra parameters. See A.7. |

## B.8 The borrow list from `POSITION_VS_MODERN_CRS.md` — what survives our contract

Each with its mandatory *must NOT come with it*.

| borrow | source | take? | must NOT come with it |
|---|---|---|---|
| **Amortised experimental design as the framing** ("we amortise the design"; DAD/iDAD, Step-DAD, decision-aware BED) | Foster/Ivanova/Rainforth line | **YES — 0 h, writing only.** It gives 1.16 a name, a citation trail, and a pre-emption of "why not just run the greedy at test time?" (the amortisation literature already answered it: per-step optimisation does not scale with depth and cannot reach states the offline search never visited — which *is* E2's argument) | **the EIG reward.** Our reward is the measured metric and the chapter's whole point is that information-theoretic surrogates lose to it. Borrow the amortisation, not the objective. |
| **Expensive-exact-scoring → filtered labels → supervised distillation → one-forward-pass inference** | CLARINET (2024) | **YES — 0 h.** This is E1's exact recipe, published, in another domain, beating EIG and KL acquisition by 17% relative. It converts E1 from "an idea we had" into "the move that worked when someone else made it" | their **user simulator** (a GPT-3 model with access to the true item), their **target-conditioned rank objective**, and their **per-user label filter** (ours are cohort-level argmaxes) |
| **Adaptivity diagnostic** (two users answer differently ⇒ does the follow-up differ?) | Pep (arXiv:2602.15012) | **YES — 2–4 h, reusing D2's code. Make it a GATE (G-E1c).** | Pep's **numbers** (abstract-only — never quote as a bar), its domain, and any inference that divergence implies *useful* divergence. D2 already measured that our tree **diverges without routing**. |
| **State/graph encoder** from the deep-RL CRS family (CRM/EAR/SCPR/UNICORN/ConTS) | §2.3 of the position doc | **CONCEPTUALLY ONLY.** The survey's own verdict: breaking their circularity *deletes the method* and leaves **workstream E with their state encoder**. What survives is the shape — pool the asked-attribute embeddings into the state — which A.8 already specifies more carefully (three-way outcomes, which none of them have) | their **reward (target-hitting)**, their **simulator** (answers from the target item's ground-truth labels ⇒ every question answerable by construction ⇒ a burned turn costs zero — the disqualifier), their **pruned action space** (UNICORN top-K), their **ask-vs-recommend action** (no analogue at fixed k) |
| **Candidate-scoring amortisation** (second-order expansion, diagonal Hessian ⇒ every candidate is a pair of masked sums) | Hu & Yu, RecSys 2013 | **AS A PROPOSAL DISTRIBUTION ONLY**, and only after C1 — see §B.4 | their **objective** (a training loss), their **sampled negatives**, and any use of the Taylor gain **as the label** |
| **Unnormalised DCG@10 for selection** | Bilgic & Bennett, SIGIR 2012 | **YES — already the author's decision, implemented** (A.8b) | their **regret framing**, their query/document label-acquisition setting, and **any change to the reported metric** |
| **Hindsight reward shaping** `w = log(ρ(s,v)+1)` | CRSIRL, NeurIPS 2023 | **NEVER.** The shaping potential is an explicit function of the **target item**. Under our contract there is no target. | cite it as the cleanest published exhibit of the line optimising against the answer key; never implement |
| **Inverse RL to learn the reward** | CRIF, SIGIR 2022 | **NEVER** — the learned reward is *defined by* target-hitting; also a dead stack (torch 1.4.0) | — |
| **Learned world model / environment simulator** | FeedRec S-Network, Pseudo Dyna-Q | **NEVER** — squarely under the simulator disqualification | (and do not cite a DOI for Pseudo Dyna-Q; it 403'd and could not be verified) |
| **UpsRec's greedy-NDCG selector as a deployable policy** | WWW 2023 | **NEVER as a policy** — it reads the user's held-out validation labels at ask time. Run it as a **declared oracle ceiling** or not at all | — |
| **Non-myopic MCTS lookahead** | SAPIENT | **NO, unless challenged.** We measured that endpoint polish is +0.002 at q16 and −0.029 at q2 — greedy is prefix-optimal in the regime that matters. 25–35 h | their simulator (accepts only lists containing the target) and their reward constants |
| **Any LLM in the selection loop** | GATE, RSO, PPDPP | **NEVER** — project rule (no LLM API calls), and unnecessary: the good LLM elicitation systems put acquisition *outside* the LLM anyway | — |

## B.9 Diagnostics and controls to run alongside (cheap, and each has caught something before)

| diagnostic | cost | what it catches |
|---|---|---|
| distinct-realised-interview count | ~0 | the `conlinucb_bs_static` failure (1 distinct interview = a static order wearing an adaptive label) |
| Pep divergence rate vs the tree's own | 2–4 h | policy collapse to a fixed order — the June BC failure |
| shuffled-label control | 1–3 h | a gate a broken instrument can pass |
| m=0 canonical snap (0.15686538 val / 0.15855 test) | minutes | a wrong ruler, before it contaminates anything (`surprise-means-debug-the-harness`) |
| banked-headline assert (re-derive a published number to 0.0e+00) | minutes | the whole class of contract drift that caught the policy-A contamination |
| per-position top-1 agreement curve | ~0 | *where* the imitation fails — early (answerability) vs late (fragmentation) |
| item/concept channel mix of the policy's picks vs the tree's (45.4/54.6 user-weighted) | ~0 | silent channel collapse, which would quietly undo Act 4 |

## B.10 Things that must NEVER be tried, collected in one place

1. Any reward, shaping potential, label filter or criterion that is **a function of the target item**.
2. Any **model-generated or LLM-generated answers** (the LLM answer apparatus is retired for new work).
3. Any use of the **10,000 test users** for labels, training, early stopping, model selection, or
   hyper-parameter choice.
4. A **two-outcome** answer encoding (see A.8).
5. A **surrogate objective** as the label, criterion or reward (see B.5).
6. **Truncating** the candidate pool, the cohort, or the kernel without sign-off (HARD RULE 1); and
   never describing a shortlisted argmax as "the exact argmax over the pool".
7. **Two heavy jobs on the box at once** (HARD RULE 11d). An EDDI run is live now (PID 26676, 5.7 GB
   free of 15.8 GB) — nothing in workstream E launches until it is done.
8. Any run before its code is **committed** (HARD RULE 10).

---

# PART C — WHAT WAS BUILT, AND THE EXACT COMMANDS

Both files are committed **before** anything runs (HARD RULE 10) and neither has been run in anger.

### `src/instrument/policy_labels.py` — the label-extraction pipeline

- `--mode banked` — assembles E1's label set from the banked artifacts. **0 scoring cost.** Runs
  gates G-L0…G-L5 (A.9): derived `pool_contract` identity, exact `base` equality, per-state
  re-evaluation of `--verify_n` candidates to **0.0e+00**, ternary-partition recompute against the
  banked `child_sizes`/`answer_rate`, and argmax consistency where a banked vector is complete.
- `--mode sweep` — pays for full 1,000-candidate exact gain **vectors** at a declared state set
  (`--states root|tree|deep|<file>`), computing **cohort-mean NDCG@10 and cohort-mean unnormalised
  DCG@10 from the same per-user vectors** and reporting the argmax disagreement rate.
- **HARD RULE 11:** a contract-signed `*_partial.json` sidecar is rewritten after **every candidate**;
  resume refuses across a changed contract via `mask_contract.resume_ok`; float32 for cached vectors;
  no unbounded per-state cache.
- **HARD RULE 1:** no sampling anywhere. `--verify_n` samples only which candidates get *re-verified*,
  never which get used.
- **Firewall:** `--split` accepts `validation` only; `test` is refused by the parser.
- `--cost_only` prints the measured-fit cost estimate for the requested state set and exits without
  building an environment.

### `src/instrument/policy_net.py` — the E1 training code

Equivariant per-candidate scorer, three-outcome state encoder, cross-entropy over the pool, the
user-level 8,000/2,000 dev split (seed 98765) with A.2.1's contamination stated in the artifact, the
shuffled-label control arm, and the anti-collapse diagnostics. Frozen candidate descriptors from the
certified stack. `--selftest` runs pure-tensor invariance checks (permutation equivariance, masking,
three-outcome distinctness) on random weights — **no data, no optimiser, no fit.**

### Commands (do not run until the EDDI job is done and the author says go)

```bash
# 0. cost model only, no environment built, seconds
python src/instrument/policy_labels.py --mode sweep --states tree --cost_only

# 1. E1's label set, from banked artifacts, with every reuse gate. ~0.5-1 h (gating only).
python src/instrument/policy_labels.py --mode banked --label e1 --verify_n 8 --full_threads

# 2. THE FIRST PAID JOB, and the one to run whatever else is decided: the root sweep under
#    BOTH normalisations -> the NDCG-vs-DCG argmax disagreement rate. ~2.4 h.
python src/instrument/policy_labels.py --mode sweep --states root --label rootdual --full_threads

# 3. E1 training (author's go-ahead required; currently blocked by "do not train yet").
python src/instrument/policy_net.py --labels experiments/battery/policy_labels_e1.npz --arm main
python src/instrument/policy_net.py --labels experiments/battery/policy_labels_e1.npz --arm shuffle
```

**The one command to start E1** — the label set, gated, at zero scoring cost:

```
python src/instrument/policy_labels.py --mode banked --label e1 --verify_n 8 --full_threads
```

---

# PART D — THE CONTINUOUS-OUTPUT HEAD (Fable review, 2026-08-13)

> **AUTHOR SAID (2026-08-13):** *"i want it to scale not only by interview length, but also by vocab,
> so i want to be able to predict out of vocab questions. i want a continuous output model, akin to
> what we had in old paper b / c, just needs to work! so continuous embedding emitted and snapped to
> closest vocab question is my preferred design, but let me know if concerns."*

This part reviews Parts A–C against that preference. Verdict up front: **the preference is adopted,
and it costs nothing — the continuous-emit-and-snap model and Part A's equivariant scorer are the
same model once the scoring head is an inner product.** PART D supersedes the head specification of
B.7 row 1; every other element of Parts A–C (labels, gates, firewall, costs, cohorts) is unchanged.

## D.1 The reconciliation — snap-and-score are one model

Give every vocab question a **precomputed, lifted descriptor** `e_q = ψ(φ_q) ∈ R^m` (φ_q frozen, ψ
small and learned). Let the state encoder **emit a continuous query vector** `ẑ = f(h_ψ(state)) ∈ R^m`.
Score every candidate as `s(q|state) = ⟨ẑ, e_q⟩`. Then:

- **inference = the author's snap, exactly**: `argmax_q ⟨ẑ, e_q⟩` is nearest-neighbour retrieval of
  the vocab question closest to the emitted point (under the inner product). One matmul over the whole
  vocab — 19,390 candidates in a single forward pass.
- **it is also Part A's equivariant scorer**, with the compatibility function specialised from a deep
  pairwise MLP `g_θ(φ_q, h)` to a dot product. All of A.7's arguments (scales to unseen candidates,
  June's H3 evidence, 172-label statistics) carry over unchanged.

So there is no decision to make between "Opus's design" and "the author's design" — only a decision
about the **head**, and the dot-product head is the one that makes the emitted point exist. Call this
the **query-vector head**; it is E1's primary.

**One honest expressiveness caveat.** A single emitted point induces a candidate ranking that is
linear in `e_q` — it cannot express candidate–candidate interactions ("don't ask a near-duplicate of
what was just asked"). Already-asked masking removes the worst case; beyond that, the deep pairwise
head and the pointer-attention head (B.7 rows 1–2 as previously specified) remain **the ablations**.
If the query-vector head misses G-E1a while the deep pairwise head passes, that gap is the measured
price of snap-ability — report it, don't hide it.

## D.2 How to train it — softmax over the vocab, never regress-and-snap

The old Paper B/C continuous line implemented "emit a point, snap it" literally: regress the actor's
output toward a target embedding, then kNN-snap at inference. Two of its failures are **measured** and
both are properties of that trainer, not of continuous emission
(`external_literature/findings/continuous_action_policy.md`):

- **k=1 snap aliasing** — the CASPER V1–V4 FAISS snap collapsed distinct proto-actions onto the same
  executed question; Wolpertinger's own authors report k=1 failing on a 13k-action recommender.
- **the DKQS T=10 unroll collapse** — the differentiable selector collapsed to an input-independent
  static sequence; diagnosed as a trainer failure (conditioning-collapse, gradient starvation), not an
  adaptivity verdict.

The fix is to make the snap part of the loss: **softmax cross-entropy over the whole candidate pool**
on the logits `⟨ẑ, e_q⟩` (already Part A's loss — nothing changes). This is a contrastive loss with
every other vocab question as a negative, so the decision boundary is shaped exactly against the
near-neighbours that aliased the old regressor, and the "snap error" is zero **by construction** —
the executed action is the argmax of the trained objective, not a post-hoc projection of an
unconstrained point. L2-regression to the label's `e_q` is admissible only as an auxiliary term.
DGRL [was mis-cited as "DBU"; DBU is only its update rule]-style blend targets (arXiv:2602.08616) become relevant only if a pure regression head is ever
run; it is not planned.

**Why Wolpertinger's REJECT in B.7 stands anyway:** Wolpertinger = proto-action + **approximate**
kNN + critic re-rank, machinery for 10⁵–10⁶ actions that cannot be scored exhaustively. At 19,390 we
score everything, so the snap is exact and the machinery buys nothing. B.7 rejects the machinery, not
the emission. Cite `dulacarnold2015deep` as the mechanism ancestor; state this distinction in the
chapter or a reviewer will state it for us.

## D.3 The one real design problem — two channels, two spaces

`φ_item = W_d[i]` (decoder row) and `φ_concept = cv3 emb` live in **different spaces with different
dimensions**; nearest-neighbour across the union is undefined without a learned joint lift. Options:

| option | verdict |
|---|---|
| **(a) per-channel lifts** `ψ_item`, `ψ_concept` — two small MLPs into a shared `R^m`, trained jointly with the policy; inputs frozen so nothing drifts from the ruler; the whole vocab's `e_q` table precomputed once per checkpoint | **RECOMMENDED.** Snap runs over the union in one matmul; channel mix stays an *outcome* (Act 4: 45.4/54.6 user-weighted), monitored by the B.9 channel-mix diagnostic. |
| (b) two query heads + a learned channel gate | REJECT — hard-codes a channel prior into the architecture; Act 4's finding is precisely that the channel balance is an empirical outcome. |
| (c) zero-pad / concat into one raw space | REJECT — the geometry is meaningless; distances across channels would be artefacts of scale. |

## D.4 What "out of vocab" genuinely buys here — and its measurable form

The vocab becomes a **test-time input, not a parameter**: any question with a descriptor is scorable
by a policy that never saw it in training.

- **The cheap, honest demonstration:** train on the 1,000-candidate pool, then score the **full
  18,359-item + 1,031-concept catalogue** at inference with no retraining, and report what the policy
  picks outside its training pool. One forward pass; this is the "scales by vocab" claim made
  measurable, and it is available even if E2's gain does not separate (A.11 already says so).
- **Concepts are the real OOV story, because they are compositional.** A *new* concept is defined by
  its member set; its cv3 embedding, answer rate and polar statistic are all computable from train
  data without touching the policy. A never-before-asked concept becomes askable at test time. This
  is the concrete, defensible instance of "predict out-of-vocab questions".
- **Items are bounded by the recommender, say so.** An item the tower doesn't know has no decoder row
  and cannot be folded, so item-side OOV extends exactly as far as the ruler's vocab (18,359) — a
  scope statement, not a defect.
- **The boundary with Paper C, stated so we don't cannibalise it:** the emitted `ẑ` *is* Paper C's
  continuous action, which gives the thesis a clean spine (Paper B trains the emission against a
  closed vocab; Paper C opens the vocab and decodes to NL). Paper C's novelty is the **conjunction**
  (continuous actor + open-vocab unified space + decode-to-NL / novel-direction); Paper B claims only
  the closed-vocab scalability half and must not claim the conjunction.

## D.5 Input sufficiency — the author's "model needs to know enough"

A.8's three-part state stands and is the answer: the tower latent `z_t` (the recommender's own
belief), the **explicit ternary answer history** (which carries what `z_t` structurally discards —
above all *asked-and-unanswered*, which folds to almost nothing in `z_t`), and position/budget. The
`z`/history/both ablation is the empirical test of sufficiency, and either outcome is quotable.

One thing to see plainly before training (A.5): **E1's labels cannot reward using the answers.** Two
users in the same leaf at the same position have different answer histories but the identical label —
202 distinct labels behind 160,000 examples — so E1 measures whether the input *can carry* the tree's
policy, not whether richer input helps. Input richness only becomes *testable* under E2's per-state
kernel labels, or under the listwise exact-gain targets of B.5. If E1 shows history features getting
zero gradient use, that is expected, not a defect.

## D.6 Bibliography actions (from `continuous_action_policy.md`, binding fixes carried over)

- **Add to `references.bib`:** HyAR (Li, ICLR 2022 — and keep the factual fixes: LSC = output box
  constraint, RSC = replay relabeling, NOT loss regularizers); Huang et al. NeurIPS 2024
  (TNDP/DUG, amortised decision-utility BED); Blau et al. ICML 2022; DAD/iDAD already on the B.8 list.
- **Pre-empt in the chapter:** a reviewer WILL say "this is DUG applied to recommendation" — claim
  the application + action space + boundary (catalogue, collaborative prior, answerability, refusals,
  strong recommender), never the estimator.
- **Cite, never implement:** Wolpertinger as ancestor (D.2); GATE/PEBOL for why open-vocab NL is not
  claimed here (that is Paper C's fight).
- **Never write "lossless" transfer** — Chandak's Theorem 1 was refuted in verification; the phrase
  is "knowledge-preserving warm-start".

## D.6b ★ AUTHOR RULING (2026-08-13) — the training cohort

> **AUTHOR SAID:** *"so we train policy on train dataset, but we do not regrow the tree, and we
> select model variant or chkp on val. is that right?"* — confirmed.

- **The tree is never regrown and never learned on train.** Growing needs the measured criterion,
  which is only honest on val (train users: no fold-in/target split, and the tower memorised them).
- **The policy trains on the 140,768 TRAIN users, generated by APPLYING the frozen tree to them.**
  Traversal needs only the user's ratings (answers) and the frozen tower (state) — **no NDCG is ever
  measured on train users**, so the memorisation objection does not attach and no `train_tr/train_te`
  split is required. This supersedes A.5's "160,000 examples" as the E1 training set: same 202
  states and 172 labels, but ~14× the input variety per label — which is what the state encoder
  needs, given that label diversity, not user count, is E1's binding constraint.
- **Selection** (checkpoint, head variant, state-encoding ablation) is on the 10,000 val users by
  running the policy's own interviews and scoring real NDCG. This replaces A.2.1's 8,000/2,000
  in-sample dev split for E1 — training data now never overlaps the selection cohort.
- **Nuance carried, not acted on:** val does double duty (tree grown there, selection there). For E1
  this is harmless; for E2 it is *conservative* — the tree is flattered on its own fitting cohort, so
  the +0.011 validation no-go bar (G-E2-pre) is cleared on the tree's home turf. Test remains one
  look per gate.
- **E2 extension under the same ruling:** train users may supply *states* (where to localise); the
  kernel label at any state is always *measured on the val cohort*. Train-user-**measured** labels
  remain a fallback only, behind a ~5 h two-cohort argmax-agreement calibration check, if E2
  provably plateaus on label count.

**Example generation (confirmed with the author 2026-08-13):** applying the tree to a train user
yields one deterministic 16-question trajectory (≤5 branch questions, then the leaf's greedy tail),
and **every prefix is a training example** — 16 (state → next-question) pairs per user, one at every
interview length 0…15, the empty state included. Over 140,768 train users: **~2.25M examples**
(vs 160k from val), uniform over lengths, answers from the user's own ratings via the standard
answerability rule.

**The known gap, named:** every one of those states lies ON the tree's own trajectories (one
deterministic path per user). The first time the trained policy deviates mid-interview it lands in a
state that was never labelled, where errors compound — Ross & Bagnell covariate shift; June's BC
collapse is the local precedent. Containment: (a) E1 tolerates it by design (a high-fidelity
imitator rarely deviates); (b) **E2's kernel labeller IS the DAgger step** — run the trained policy
on train users, collect the states *it* visits (off-tree included), label them with the
kernel-weighted estimate measured on val. The covariate-shift fix and the licence to exceed the
teacher are the same loop.

## D.7 Decisions requested from the author

1. **Head:** adopt the query-vector (dot-product) head as E1's primary, deep-pairwise + pointer as
   ablations, flat 1,000-way as the control arm. (Recommended; this is your design.)
2. **Channel lift:** option D.3(a). (Recommended.)
3. **A.2.1 contamination:** option (i) for E1 now; ruling on (ii) — the 8–10 h tree regrow — needed
   only before E2's test look.
4. **Go/no-go on the first jobs** once the EDDI run frees the box: the banked E1 label extraction
   (0.5–1 h, no training) and the 2.4 h dual-normalisation root sweep. Training itself stays blocked
   until you say go.

---

# PART E — FABLE'S VERDICT ON WHAT COMES AFTER E1 (2026-08-13) — AWAITING AUTHOR RULING

> **AUTHOR ASKED (verbatim):** *"i don't see why we would continue with supervised. can you ask fable
> if it is not bttr to start rl now? or maybe nor rl but some explore exploit?"*

**VERDICT: D, with A1 as the bounded closer. NOT RL. NOT a bandit.** Ordered: (i) the cheap
hard-partition labels **plus the deeper-tree control** as one batch (~4–7 h) to close the
beat-the-tree line honestly under its pre-registered no-go; (ii) all remaining compute to **pricing
answerability into the open-vocabulary head**.

Deciding sentence: *the exploitable routing signal is 2.1% of cross-user reward variance and the bank
already beats conditioning without any policy at all (+0.01430 vs +0.01035), so every hour spent
trying to add 40% on top of the tree's gain is spent against the chapter's own evidence, while the
author's stated motive — vocab scaling and out-of-vocab prediction — has a demonstrated mechanism
with one measured, fixable defect.*

## E.1 Why not RL

**States are not the bottleneck; labels AT states are.** RL does not remove that cost, it hides it
inside a reward whose usable signal was measured at 2.1% of cross-user variance. That is a property of
the DATA, so RLOO fixes the *variance of the estimator*, not the *size of the signal*, and a good
initialisation does not fix it either. Supporting measured history: BC collapsed to a fixed 17-item
list; coverage/entropy rewards degraded the belief state; realizable gain ≈ 1/5 of oracle. ~19 h per
attempt against a +0.011 validation bar the tree itself never had to clear. **Rejected.**

## E.2 The explore/exploit question, answered on its merits

**The author's instinct points at something real that the four published bandit arms did NOT test:**
they do online *arm selection for reward*; she plausibly means *exploration as a STATE GENERATOR for
training*. That distinction is correct and worth telling her. It does not rescue the line, for RL's
reason: generated states still need measured labels (32 h–3,772 h ceiling). **Keep in pocket as a
DIAGNOSTIC only:** ε-perturbed rollouts of the E1 policy on validation to see *where* it goes
off-manifold — informs the answerability head, is not a training method, costs an afternoon.
**Do not build a fifth bandit.**

## E.3 ★ A2 (kernel-weighted labels) is KILLED — this corrects Part B and my own earlier framing

"A2 genuinely is not a deeper tree" was **over-claimed**. Kernel-weighted labels are still one-step
greedy split-search argmaxes with soft assignment; **a smoothed deeper tree is still a deeper tree**.
The only objects categorically not a deeper tree are one trained on a different objective (RL —
unsupportable) or one emitting out-of-vocab actions (D). **Drop A2: 20–100 h for a distinction that
will not survive review.**

## E.4 The job that replaces supervised E2

An **answerability head**: train `P(answered | z, q)` on **real answer events harvested from tree
application to the 140,768 TRAIN users** (same pipeline as E1's harvest; answer events only, no NDCG
on train — answerability is the structural rule, so this is measured, not surrogate, and is not a
function of the target item). Selection score = `policy logit + λ · log P(answered)`; sweep λ on the
10,000 VAL users by real interview NDCG. Actor = the existing E1 head, frozen or lightly fine-tuned.
No variance control needed (supervised). **Cost: hours, not tens.**

Success is **not** "beat the tree". It is: the open-vocab policy stops paying **−0.02034** for leaving
the pool — out-of-pool answer rate rises from **9.1%** toward the in-pool **55.2%**, and the
full-NDCG deficit against the policy's own in-pool run closes on validation.

## E.5 Pre-registered abandon conditions

- **A1:** existing no-go stands — validation gain over the tree < **+0.011** ⇒ no test look, line
  closed. **Control (mandatory, nearly free):** grow the tree one level with the floor lowered on the
  same labels. If deeper-tree ≈ policy on validation, write *"the policy is a re-parameterisation of a
  deeper tree"* and claim nothing. Fable's prior: **the no-go fires** — that is a clean, reportable
  closure, not a waste.
- **D:** if λ-swept pricing recovers **less than half** of the −0.02034 deficit, **or** if recovering
  it drives out-of-pool picks to ~0% (the head merely learns to stay in-pool), abandon the pricing
  claim and report the mechanism finding as-is. **The deeper-tree control is irrelevant to D — no tree
  can emit an out-of-vocab question, which is the cleanest "not just a deeper tree" argument the
  chapter has.**

## E.6 Framing corrections that bind

- **The 82.3% / 189-questions / 30%-out-of-pool figures are NOT a scaling win as they stand.** Until
  answerability is priced they are a measured **defect** (9.1% answered, −0.02034). Present as
  mechanism + measured failure + fix. Never as a result on its own.
- **Verified by Opus after the verdict:** every E1 artifact carries `test_touched=false` and the
  scalability numbers are `val_*` — **the −0.02034 is validation; no test look has been spent.**
- Also measured, and useful to the fix: out-of-pool picks are **0% at the first three positions** and
  rise to **57% by position 16** — the policy stays in-pool early and wanders late.

---

# PART F — THE TWO-HEAD DESIGN (author's proposal, Fable-reviewed) — SUPERSEDES PART E

> **AUTHOR SAID (2026-08-13):** *"maybe having 2 heads, answerability and ndcg lift would be good?"*
> and, on why the policy is wanted at all: *"policy is nice becuase it is generalisable and scalable.
> tree treats questions as discrete entities. policy has their embedding and know their similarity. so
> it is supposed to be able to learn from fewer examples. polciy is supposed to be able to learn
> strategic questioning, not greedy. and most importantly, keep adapting fter 5 levels"*

## F.1 The design — ADOPTED, three-way, supersedes Part E's λ·log P

Score a question by the **decomposition of the target quantity itself**:
`Σ_o P(o | q, state) · V(o | q, state)` over the three outcomes **like / dislike / no-answer**.
This replaces E.4's `policy logit + λ · log P(answered)`, which was a ranking heuristic with a
validation-fitted knob. Three-way, not binary: the root's three children sit at **0.374 / 0.320 /
0.087**, and ARM N measured a "no" worth **+0.023 at k=2** — a binary answered/not encoding throws
away the like/dislike distinction exactly where it pays. Three-way also matches the tree's branching,
so outcome labels exist at every state we hold gains for.

**Name the resemblance in the chapter:** this is EDDI's own decomposition with the information-theoretic
value replaced by **measured NDCG** — the thesis expressed in the architecture rather than asserted.

**★ The load-bearing property is the DATA ASYMMETRY.** Outcome labels are observable for **all 19,390
catalogue questions** on **140,768 TRAIN users** at **zero NDCG cost** (they fall out of the
tree-application harvest; answerability is the structural rule, hence measured, not surrogate, and not
a function of the target item). Lift labels exist only at the 202 tree states for the 1,000-pool, with
**21,943 exact (state,candidate) gains banked (~11%)** and 32 h for the rest. Supplying the *cheap*
signal is the direct fix for the diagnosed open-vocab failure (**9.1% vs 55.2% answered**).

⚠ **DEFLATION, binding:** "no free parameter" is over-claimed (my error). The *arithmetic* is
parameter-free; the *estimators* are not. A miscalibrated `P` multiplied by an extrapolated `V` is two
learned errors multiplied. **Calibration of the outcome head is a hard requirement and must be
reported (ECE).** Never promise "no knobs".

## F.2 The thesis question — no violation, but one sentence and one table are owed

The surrogates that lost (training loss ρ=−0.149, variance, pairwise logistic) were *different
quantities used in place of the target*. A regressor **fit on** the measured target is an *estimator of
the measured quantity*, not a surrogate objective. The firewall holds; what changes is certainty.

**The sentence the chapter must write:** *"The policy selects on a prediction of measured NDCG gain,
fit to measured gains; it inherits the tree's criterion but not its certainty, and every claim about
the policy is a claim about how well that estimator transfers, which we test on held-out states."*

**The table:** the lift head's Spearman against measured gain at held-out states, printed beside the
measured criterion's **ρ=+0.784** and the training-loss surrogate's **−0.149**.
**Pre-registered gate, fixed before looking: ρ ≥ +0.40 on held-out states.** Below it, HARD RULE 6
applies — report a weak method, do not tune until it passes.

## F.3 ★ MY ERROR, corrected: properties (2) and (3) do NOT share a root cause

I told the author that "strategic questioning" and "keeps adapting past depth 5" were blocked by the
same thing. **False, and it inflated the RL case by bundling a buyable property with an unbuyable one:**

- **(3) past-depth-5 continuation is BUYABLE WITH SUPERVISION** — past-depth-5 greedy labels are
  cheaply harvestable; that is literally the deeper-tree control.
- **(2) strategy is the only structurally RL-shaped one.** Every label we hold or can cheaply mint is
  one-step greedy, so a policy fit to them is greedy by construction.

**And structural possibility is not headroom.** Nothing measured says myopia costs us anything: the one
endpoint measurement (polish **+0.002@q16 / −0.029@q2**) points the other way, and the floor-stop
pattern (all 15 stops FLOOR stops with positive gain in hand, 84% of users below one) is evidence for
**more greedy DEPTH**, not for strategy.

**Chapter consequence, binding: the policy may NEVER be described as strategic or non-myopic. It is a
greedy policy with a learned reach. Write that.**

**RL re-open condition, pre-registered:** RL becomes arguable only if the deeper-greedy control
**saturates while floor-stop gain demonstrably remains**. Then one bounded RLOO attempt (19 h),
exact-NDCG reward only. Not this cycle.

## F.4 ★ COMPARATOR DISCIPLINE — the bar is the BANK, not the tree

Everything new must clear **bank-alone: +0.01430 [+0.01124,+0.01739] full / +0.02199 tail**, not merely
the tree's +0.01035/+0.00705. **A policy that beats the tree but not the static bank has not earned
adaptive language.** My briefs never stated this; that omission is how "beats the tree" quietly becomes
a claim the evidence does not license.

## F.5 The first experiment — held-out QUESTIONS, specified so it cannot be trivially won

- **Inputs:** interview-visible state only (answers so far, depth) + question embedding. **Never
  profile features** — answerability is a structural rule partly computable from the profile, so a head
  that sees profile-like inputs learns an arithmetic identity and the "generalisation" win is hollow.
- **Data:** tree-application harvest on the 140,768 TRAIN users; three-way outcome per (user-state,
  question). **No NDCG anywhere. TEST untouched.**
- **Split:** hold out **20% of the 19,390 catalogue questions by question IDENTITY**, stratified by
  channel and question popularity; plus a held-out user slice of TRAIN.
- **Measure:** three-way one-vs-rest AUC and **calibration (ECE)** on held-out questions × held-out users.
- **Baselines it must beat:** (a) global outcome base rates; (b) ★ **a per-question popularity
  heuristic that ignores state entirely** — the killer. If popularity alone matches the head on
  held-out questions, the embedding contributes nothing.
- **★ The sample-efficiency curve that actually tests the author's claim:** train on 10 / 25 / 50 /
  100% of questions and plot held-out-question AUC. "Learns from fewer examples via similarity"
  predicts **early saturation**; a curve still climbing steeply at 100% is memorisation-shaped.
- **Falsifier, pre-registered:** held-out-question AUC ≤ popularity baseline (b) ⇒ the "generalises via
  embedding similarity" claim fails on the cheapest, most learnable property it will ever be offered,
  and fails *a fortiori* for lift. **Report it; do not retry with a bigger head.**

## F.6 Revised ordering (supersedes E's)

1. **Outcome head + the held-out-question test** (F.5). Hours–1 day, zero NDCG, no test look.
   **Required under every path, including RL.**
2. **A1 + deeper-tree control STILL RUNS (~4–7 h)** and now has three jobs: the beat-the-tree closer
   under its no-go; the source of past-depth-5 greedy labels that buys property (3) **without RL**; and
   the control that **bounds how much room strategy could possibly have**. May run in parallel with 1.
3. **Lift head from the banked 11% FIRST** (21,943 exact gains, three-way conditional value). Evaluate
   held-out-state Spearman against F.2's ρ≥0.40 gate **before** buying the 32 h harvest.
4. **Combined P×V policy vs the tree AND vs bank-alone on validation.** One test look at the final gate.
5. **RL dead this cycle**, re-open condition in F.3.

## F.7 Two more corrections to my own framing

- **E1's top-1 = 1.0000 is not evidence OF memorisation** — perfect reproduction *was* the fit target.
  It is the **absence of evidence for generalisation**, not proof against it. Do not overstate it either way.
- Part E's kill of A2 (kernel labels) **stands**.

---

# PART G — THE STAGED TRAINING PLAN (Fable, 2026-08-13) — the curriculum, settled

> **AUTHOR SAID (verbatim):** *"i am very confused about rl phase of policy. plz review our
> conversation with opus about it and propose a clean training plan so: 1. we do not lose gains from
> distillation 2. we are able to extend adaptivity deeper than 5 questions 3. we might be able to
> discover non-myopic strategy 4. the model is learning to generalise by embedding -- my hunch is we
> must generate heaps of training examples, and i am a bit confused about training curriculum"*

**The resolution of the confusion:** RL is not an alternative to distillation; it is the last stage on
top of it. **The curriculum is over DATA SOURCES, from dense-and-cheap to sparse-and-expensive** —
imitation labels (unlimited, teach copying) → deeper greedy labels (cheap, teach reach) → measured-gain
labels at diverse states (expensive, teach the value model) → reward rollouts (most expensive, weakest
signal per example, the only source that can teach strategy). Not over interview length.

**★ THE ONE GATE THAT SPANS ALL STAGES (HARD RULE 8):** after every stage, re-run the fixed check —
the policy's own interviews on the 10,000 validation users, full+tail NDCG@10, canonical m=0 snap
first — and **no stage may score below the previous stage's number**. This is requirement 1 made
mechanical.

## G.A Stage A — ANCHOR (done / running)

E1 = the tree's exact replica (+0.02794 val, 1e-9). The two heads (Part F) training now: outcomes from
free labels, values from the 21,943 banked gains (survivor-bias handled per the scope message). Stage
A's val number is the floor for everything after.

## G.B Stage B — REACH: deeper than 5, supervised, NO RL (requirement 2)

Depth is NOT the property that needs RL (F.3). Harvest greedy labels one level below all 15 floor
stops (**3.3 h measured**); fine-tune the policy on them **MIXED with the original tree labels** —
replay is the anti-forgetting mechanism; grow the deeper tree on the same labels as the control. If
the policy ≈ the deeper tree, write "re-parameterisation" and claim nothing. **4–7 h total.**

## G.C Stage C — COVERAGE: the author's "heaps of examples" hunch, aimed correctly (requirement 4)

The hunch is right with one refinement: the outcome head already has 2.25M free examples; **the scarce
commodity is measured gains at DIVERSE STATES.** The tree's data (1,000 candidates × 202 on-path
states) is the wrong SHAPE for a value regressor — it wants **few candidates × thousands of states**.
Generate off-path states by ε-perturbing the policy's own interviews (the author's explore/exploit
instinct, placed where it belongs: generating training states, never selecting questions); measure a
small pre-registered candidate set at each; retrain the value head. This is what makes
embedding-generalisation real; the held-out-question test and 10/25/50/100% curves (F.5) slot in here.
**15–50 h. ⚠ The per-state candidate subset is a SAMPLING decision — HARD RULE 1 sign-off required
from the author before any run.** (Selection at inference still scores all 19,390 — only the value
head's TRAINING set is sparse by design.)

### ★★★ G.C-AMENDMENT-4 (2026-08-14, Fable ruling) — **AMENDMENT-3 OVER-RECORDS. THE LINE IS NOT CLOSED.**

**The author objected — *"why has it no way of knowing... it has user embedding, right?"* — and she is right.**
The ceiling experiment perturbs **one fixed vector reused at every state**, so every row of that table,
including ρ=1.00, is a **MARGINAL** value.

| licensed by the experiment | NOT licensed |
|---|---|
| **"a state-MARGINAL value fails at any accuracy"** — strong, clean, and mechanistically interesting | "the two-head design fails" |
| "accuracy does not help, and ρ=1.00 is numerically worst" | ★ "more accurate is worse" — cross-level spread is BELOW within-level seed spread; do not let this sentence into the chapter |

**Two independent reasons the stronger claim fails.** (1) The state-conditioned value was never shown
impossible — the learned head at ρ=+0.06 is a **weak method** (HARD RULE 6), not a proof. (2) ★ **The
state-conditioned ceiling is already known to be POSITIVE: the tree IS a state-conditioned value
policy** (it conditions on answers; its leaf tails are measured-gain greedy sequences), **and E1
reproduced it on this exact architecture at +0.02794 to 1e-9.** `⟨ẑ(z_t), e_q⟩` demonstrably carries
conditioning. The syllogism "marginal V fails; conditional V lost to a static average; therefore
closed" uses a **failed training run** as its second premise.

**Corrected statement: marginal value fails at any accuracy; the learned conditional value was never
obtained (7-state label supply); the conditional ceiling is bounded below by the tree.**

### ★ THE DIAGNOSIS — label starvation, with a tell nobody read

**+0.06 < +0.42 is impossible for a healthy learner.** A network that sees `z_t` can trivially
**ignore** `z_t` and emit the mean vector — so landing far *below* the state-blind baseline means it
failed to learn even the **state-common** component. That is an optimisation/target failure (tiny
NDCG-gain magnitudes, noise-dominated regression, six training examples under LOSO), **not** a
representational one. **This should have stopped the run at hour one, alongside the HARD RULE 2 miss.**
Bank it as a standing check: *any learner that scores below the trivial baseline available to it by
construction is broken, and no further interpretation of its output is admissible.*

### THE FIX — labels, not architecture

Generate measured gain vectors at **many** states (sampled interview prefixes) — which is exactly what
phase 3's deeper greedy labelling produces — then **skip the P×V factorisation and distil the
measured-gain greedy directly into `ẑ`, as E1 distilled the tree.** E1 is the existence proof that
distillation onto this architecture works; "imitate the tree" → "imitate greedy at sampled states" is
the natural method claim. Training is cheap; the cost is label generation, already running.

**On the state-gradient add-on (my Part G proposal) — I presented it with its redundancy story leading
and its surrogate nature trailing; the honest order is reversed.** It is a **first-order SURROGATE**,
and this project's most repeated finding is that surrogates for measured gain lose to measured gain.
**Do not ship it as the score.** Its correct role is a candidate *feature*, gated on experiment (b)
below, not presumed.

### THE THREE CHEAP SETTLING EXPERIMENTS (validation only, hours, no test look)

- **(a) Settles the ruling by measurement.** Rerun the composed policy with `V(state)` = the banked
  gain vector of the **nearest banked state** (a crude 7-way conditional table) instead of the flat
  mean. If it beats 0.22650, conditioning in V moves the policy and the line is open by measurement
  rather than argument. Same harness as the ceiling run, 19 rollouts × 3 seeds.
- **(b) Gates the gradient add-on.** At each of the 7 banked states compute `⟨∇, enc_q⟩` and Spearman
  it against the banked measured gains; check it down-ranks already-asked questions. Pure computation,
  no training. **If per-state ρ does not beat +0.42, drop the add-on.**
- **(c) Separates optimiser failure from label failure**, before phase-3 labels are spent: train the
  identical head with the **state input zeroed**. If it reaches ~+0.42 the architecture and optimiser
  are fine and the failure is purely label supply; if it also sits near zero, the loss/target scaling
  is broken and must be fixed before any more labels are bought.

### WHAT THE CHAPTER CLAIMS (coherent without any new positive result)

1. **E1** — a single emitted vector exactly carries a sequential, redundancy-aware policy (existence proof).
2. **The outcome head** — answerability is learnable, state-conditioned, calibrated, worth **+0.00392
   [+0.00105,+0.00676]** inside a policy, and turns 1 realised interview into 9,421.
3. ★ **The ceiling experiment, correctly stated: *the worth of a question is not a property of the
   question*.** Even a perfect per-question value table loses — value is intrinsically conditional.

**Effort:** let phase 3 finish, run (a)–(c), then one distillation-of-greedy attempt. **STOP** tuning
the two-head value regression on the 7-vector supply — every further hour there is compute theatre.

⚠ **And do not let the fallback framing oversell conditioning:** bank-alone (+0.01280) is nearly half
the tree's gain (+0.02794).

---

### G.C-AMENDMENT-3 (2026-08-14) — SUPERSEDED IN ITS CONCLUSION by AMENDMENT-4 above. Its ceiling numbers stand; its "line closes" verdict does not.

**The ceiling experiment** (degrade the true value table to a target ρ, rank-preservingly — perturbed in
normal-score space and mapped back onto `sorted(t)`, so the value multiset is exactly preserved and a
simulated estimator cannot cheat by shrinking toward a constant — then run the combined P×V policy;
19 rollouts, 3 seeds per level, 10,000 validation users):

| ρ vs true value table | 0.50 | 0.60 | 0.70 | 0.80 | 0.90 | 0.95 | **1.00** |
|---|---|---|---|---|---|---|---|
| policy gain vs combined static | −0.02512 | −0.02495 | −0.02698 | −0.02398 | −0.02587 | −0.02608 | **−0.02911** |

**Total spread 0.0075 — smaller than the seed spread within a single level. corr(accuracy, score) =
−0.18. EVEN ρ = 1.00 LOSES, by the largest margin of the sweep**, and the best point is still 0.0345
below bank-alone. The pre-registered "even ρ=0.95 loses ⇒ dead on its own terms" condition fires.

**★ WHY, and this is the part that transfers to any future design.** The mean gain vector is a
**MARGINAL** value — what a question is worth *on its own*. **Taking the top-k by a marginal value
cannot express REDUNDANCY between questions.** Greedy does the one thing this design structurally
cannot: pick, **re-fold, re-score**, pick again, valuing the second question *given* the first. It is
visible in the data rather than inferred: **answer rate falls monotonically 0.6188 → 0.5650 as
accuracy rises** — a more accurate marginal value concentrates the policy on a narrower, more
redundant set, and the noise levels were helping only by scattering the picks.

So `Σ_o P(o)·V(o)` with a **state-marginal** `V` is not competitive at any estimator quality; to
compete, `V` must be conditioned on what has already been asked — which is exactly the object that
lost to a static average. **The two failures compose and the line closes.**

**Consequences, binding:**
- **Stage C (buying states) is DEAD** — there is no accuracy target to buy toward.
- **No learning curve is worth running**; the mean-vector-plus-residual model is **WITHDRAWN**.
- The 2.3 h noise-ceiling test is **moot**.
- **Stage B (phase 3) is UNAFFECTED** — it uses labels, not predicted values.
- **Stage D (RL) is arguably STRENGTHENED**: a whole-interview reward handles sequence redundancy
  natively, which is precisely what this design could not. The diagnosis now points at *sequential
  conditioning* as the missing ingredient — which the tree gets by re-folding and RL would get from
  the reward.
- ★ The **arithmetic add-on** (appending the recommender's state-gradient to the emitted vector, and
  the candidates' encoder rows to the question table) is **NOT killed by this result** and is now the
  most interesting survivor: because the gradient is recomputed at each state, a question that
  duplicates one already asked scores lower *by construction*. It is a state-conditioned quantity that
  is **computed rather than learned**, so it addresses the actual failure rather than the one we
  mis-diagnosed. Untested.

**Two limitations carried forward as first-class:** the winning estimator is a **lookup table over the
1,000-pool** and is structurally incapable of open vocabulary; and **every gain we hold is for a pool
question** (top-500-popular ∪ top-500-answerable), so no model here has ever seen what a *typical*
catalogue question is worth — plausibly a bigger problem for open vocabulary than the state count.

---

### G.C-AMENDMENT-2 (2026-08-14, 03:0x) — a parameter-free baseline passes the gate (STILL VALID; its "fix the estimator" recommendation is superseded by AMENDMENT-3)

**HARD RULE 2 was not applied to the lift head: it was never compared against the simplest baseline.
It should have been, and the baseline wins.**

Predict a held-out state's gain vector with **the plain mean of the other six states' gain vectors** —
no features, no parameters, no learning, no state information at all:

| estimator | LOSO Spearman over 7 complete states |
|---|---|
| **parameter-free mean of the other six** | **+0.4136** ✅ clears the ρ ≥ +0.40 gate |
| the learned lift head (stable, 3 seeds) | +0.0623 ❌ |
| *(reference)* measured criterion | +0.784 |
| *(reference)* training-loss surrogate | −0.149 |

Per state: root +0.461, root.like +0.389, root.dislike +0.649, root.noanswer +0.237, root.dd +0.341,
root.ddd +0.286, root.dddn +0.533.

**Extraction verified before use** (`verify-the-artifact-not-a-proxy`): the reconstructed root vector's
argmax is `[1,114]` with gain `0.02049056`, byte-matching `ndcgtree.json`'s recorded root splitter and
gain. NB the heap's `upd==0` entries alone are NOT complete — CELF pops the selected candidate, so the
level-0 pick must be added back from `seq[0]`/`gains[0]`. A first pass that skipped this put the root
argmax at `[1,994]` and was wrong.

**What this overturns:**
- ✗ "The target is noise / measurement-limited." **False.** A trivial average predicts held-out states
  at +0.41; there is ample learnable structure.
- ✗ "Seven states are not enough." **Not supported.** Six states suffice for a parameter-free average
  to clear the bar.
- ✗ "The learning problem is unlearnable from question descriptor + interview state." **Not supported.**
  What is established is only that **our learned head is far worse than doing nothing**, which under
  HARD RULE 6 is a weak method and nothing more.
- The **2.3 h split-half noise-ceiling test is DEMOTED** from precondition to optional — the target
  demonstrably carries structure, so the ceiling is not near ρ≈0.13.

**★ And note what the winning baseline IS: a STATE-INDEPENDENT ranking of questions.** It ignores the
interview entirely and still beats the state-conditioned learned model. That is the same shape as the
chapter's own headline — *what you ask about matters more than how you decide what to ask* (bank
+0.01430 > conditioning +0.01035) — arriving independently in the value estimator.

**Actions, in order:**
1. **Re-run the combined P×V policy with the mean-vector value in place of the learned head.** Free
   (no new measurement), and it is the honest test of whether the two-head *design* works when its
   value half is not broken. Phase 2's −0.05840 is a verdict on our estimator, not on the design.
2. Only then decide whether a learned lift head is worth fixing at all.
3. Stage C's sign-off request must be re-derived from these numbers, not from the withdrawn ones below.

---

### G.C-AMENDMENT-1 (superseded in its conclusions by AMENDMENT-2 above; kept for the record)

Phase 2 finished and **two of its results attack Stage C's premise directly**:

1. **The sizing slope is far too shallow.** On common held-out states, going from 7 → 22 states moved
   the value head's Spearman **+0.0623 → +0.0819 — about +0.020 per TRIPLING**. On that slope the
   32 h sweep does not reach the ρ ≥ +0.40 gate, and nothing affordable does.
2. **★ The binding constraint is NOT the state count.** The stabilised head — with the no-answer
   level removed from its target — still fits **its own seven TRAINING states at only ρ ≈ 0.13**. A
   model that cannot fit what it was trained on is not short of states.

**So "buy more states" no longer follows from the evidence, and Stage C must not be purchased on the
old rationale.** Before any 15–50 h spend, run the cheap test that distinguishes the two live
explanations:

| explanation | consequence |
|---|---|
| **(i) the features are wrong** — question descriptor + interview state genuinely do not resolve measured NDCG gain | a different representation might work; Stage C still pointless, since more states of an unlearnable function teach nothing |
| **(ii) ★ THE TARGET IS NOISE** — measured gain at a state has a reliability ceiling near ρ≈0.13, so the head is already fitting it as well as it can be fitted | **the entire measure-gains-and-learn-from-them line is measurement-limited.** Stage C is dead, Stage D's dense reward inherits the same noise, and the chapter says so |

**THE TEST — split-half reliability of the target, ~2.3 h, one state.** At the root, split the 10,000
users into two halves, measure all ~997 candidates' gains on each half independently, and correlate
the two gain vectors. That correlation is the **ceiling any model could reach**. If it is ~0.15, then
ρ≈0.13 is near-perfect fitting of a noisy target and no amount of data or capacity helps. If it is
~0.9, the target is clean and the failure is ours.

**This test is a PRECONDITION on Stage C's sign-off request.** Asking the author to authorise 15–50 h
of sampling without it would be asking her to buy a cure for an undiagnosed illness.

**Also binding, and free (phase 2 earned it):** ★ **any future value model must pass the no-answer
flatness check before its Spearman is believed.** `V(no-answer|q,s)` should equal `base(s)` and be
flat across q; measured it was biased low by 0.023–0.028 and varied with **sd ≈ 0.058, ~3× the entire
range of the quantity**, and `corr(composed score, V_no-answer) = +0.379` against **+0.018** for
`V_like` — the argmax was being driven by the value assigned to silence. Predicting `V = base(s) + δ`
fixed the mechanism (answer rate 0.1241 → 0.5050, concept share 0.0150 → 0.7026, distinct interviews
2,548 → 7,428) without rescuing the gate.

## G.D Stage D — STRATEGY: one bounded RL attempt (requirement 3), doubly guarded

**Trigger (pre-registered, unchanged from F.3):** opens ONLY if Stage B's deeper-greedy control
saturates while floor-stop gain demonstrably remains. Everything measured says the remaining prize
looks like depth, not strategy (polish +0.002@q16/−0.029@q2; all 15 stops floor stops) — Stage D
exists to answer the question, not because I expect it to fire.

1. **★ PREFIX LOCK:** turns 1–5 frozen to the distilled policy; RL adjusts turns 6–16 only.
   Requirement 1 becomes STRUCTURAL (the anchor's behaviour cannot be lost where it earns its gain)
   and the credit horizon shrinks 16 → 11.
2. **★ DENSE MEASURED REWARD:** per-turn reward = the CHANGE in measured full NDCG@10 caused by that
   turn. The per-turn rewards telescope to the final NDCG — same objective, no surrogate, credit
   spread so the gradient can see which question helped. ~16× reward cost per rollout, paid for by
   needing far fewer steps.
3. Variance: RLOO leave-one-out baseline. **One attempt, ~19–25 h.** Failure is reported under HARD
   RULE 6, never silently retried.

## G.E Requirements → owners

| requirement | owner |
|---|---|
| 1. never lose distillation gains | the spanning regression gate + replay (B) + prefix lock (D) |
| 2. deeper than 5 | Stage B, supervised, with its own control |
| 3. non-myopic strategy | Stage D only, gated on B's control saturating with gain left |
| 4. generalise by embedding | shared frozen descriptors (in place) + Stage C's state diversity — the missing ingredient is DIVERSITY, not capacity |

**Chapter caveat, binding:** strategy has NO measured evidence of headroom here; the policy is never
described as strategic unless Stage D separates (F.3 stands).

**Immediate sequence:** two-head run finishes → Stage B batch → author's Stage C sampling sign-off.

---

# PART H — THE CAPABILITIES PROGRAMME (Fable, 2026-08-15) — the redirect after the value line closed

> **AUTHOR SAID (verbatim):** *"policy is supposed to eb scalable, generalisable (it knows similarity
> between concepts and items, tree doesn't), could support new vocab, can support new states, most
> improtantly i guess. so i am ok not to beat tree on these 5 levels or whatnot, but i wanna show how
> this architecture in general is better."* And, after the component split was explained: *"can we get
> rid of value at all?"* → yes; and *"write up the experiment design plz."*

## H.0 The strategic reframe, in two sentences

We had been trying to prove the architecture better by **beating the tree at the tree's own game**
(empty start, the tree's own pool, the cohort it was grown for) — a game in which the tree is
near-optimal by construction and the best any learner can do is copy it, which E1 did to 1e-9. The
author's list — similarity, new vocab, new states — is a list of **games the tree cannot enter**, and
that is where superiority is demonstrable without beating 0.28355.

## H.1 The system under test — and the VALUE MODEL IS DELETED

**System = the DISTILLED POLICY (E1, frozen, 0.28355) + the REACTION HEAD (outcome head, frozen,
AUC 0.98243).** No value model anywhere.

The value line is CLOSED as a measured negative, five arms deep: flat table −0.02911 · learned
two-head −0.05840 · **perfect measured values** (ceiling, ρ=1.00) −0.02911 · nearest-state lookup
−0.01978 · warm-started direct regression −0.03559, all against the combined static, all losing to
E1's +0.02794. Mechanisms recorded: argmax-over-noisy-estimates selects the over-estimates
(answer rate 0.58→0.37); the value surface is rough across states (inter-state rank corr ~+0.29,
range −0.17…+0.83) so it cannot be interpolated from any affordable label set; and action learning
needs one bit per state where value learning needs the whole surface. **Chapter sentence:** *at
feasible label budgets, learning the action beats learning the value — and even perfect per-question
values lose.* The `base + δ correction` sixth attempt is **CANCELLED** (revivable only by explicit
author request); the author's instinct that answerability multiplies in nothing is confirmed
(+0.1726 vs +0.1727) and `P(o)` survives only as an input/guard, never a factor.

Standing constraints on every experiment below: 10,000 validation users for all development, **no
test look without an explicit author decision**, certified pair frozen, policy-B mask, paired
percentile bootstrap (10k resamples, seed 98765), canonical m=0 snap before anything is trusted,
one heavy job on the box, code committed before running, `test_touched=false` stamped everywhere.

## H.2 Experiment R — RESUMPTION: "supports new states" (★ the headline; run FIRST)

**Claim under test:** the policy can condition its questioning on evidence it did not choose; the
tree structurally cannot (its routing consumes answers to ITS OWN questions only — volunteered
evidence enters the recommender's fold but can never move which question the tree asks next).

**Design.** For each validation user, sample **k volunteered items** from their fold-in half
(k ∈ {2, 4, 8}; popularity-weighted draw so the volunteered set is realistic, seed 98765; the draw is
declared, not silent). Fold them as ordinary evidence. Then each arm conducts a **16-question**
interview from that seeded state:

| arm | how it uses the evidence |
|---|---|
| tree | cannot — starts at its root, asks its fixed branch sequence regardless (it may literally re-ask what was volunteered) |
| static combined order | cannot — same 16 questions always |
| **distilled policy** | reads the seeded `z_t` — MAY adapt off-manifold; whether it does is itself a result |
| ask-nothing-more | the k volunteered items alone (isolates what the interview adds on top) |

All arms score identically: full+tail NDCG@10 at K=16, same recommender, same mask. **Report also the
redundancy count** — how often each arm asks a question whose answer the volunteered evidence already
determines — because that is the mechanism made visible, not just the score.

**The honest fork, declared up front:** E1 never deviates on-manifold, so it may not adapt to seeded
states either. **Both outcomes are results.** If it adapts and wins: distillation generalises off the
teacher's manifold — the architecture claim, demonstrated. If it does not adapt: that measures what
one-hot distillation fails to buy, and the fix is a **fine-tune on evidence-seeded states, for which
the training data already exists** — the 138 random-prefix states sampled on 2026-08-14 are exactly
evidence-seeded states, with measured labels, at zero new cost. Gate for the fine-tune: the spanning
regression gate (never below E1 at k=0 seeded evidence) plus the shuffle control.

**Cost:** rollout + scoring per arm per k, ~3 arms × 3 k ≈ 9 cells ≈ 3–5 h total on this box.
**Falsifier:** if the policy adapts and still cannot beat the TREE on seeded states — the setting
built to disadvantage the tree — the "supports new states" claim is dead and is reported dead.

## H.3 Experiment Q — HELD-OUT QUESTIONS: "it knows similarity" (the deferred F.5, now due)

**Claim under test:** the reaction head predicts a person's reaction to a question it has NEVER seen
in training, from the question's embedding alone. A tree cannot represent such a question at all.

Design exactly as F.5 specified (it was deferred by the author on 2026-08-13, "train on all and see
if it even works" — it works, so the deferral expires): hold out **20% of the 19,390 questions by
identity**, stratified by channel and popularity; retrain the reaction head on the remaining 80%;
evaluate three-way AUC/ECE on held-out questions × held-out users. **Baselines:** global base rates,
and the per-question popularity prior — which for a held-out question has NO per-question row, so the
prior must back off to channel/popularity-bucket means; that backoff IS the fair comparator, state it.
**The curve that answers the author's real question:** train on 10/25/50/100% of questions, plot
held-out-question AUC. Early saturation = "learns from fewer examples via similarity" demonstrated;
still-climbing = memorisation-shaped, reported as such. **Falsifier (pre-registered, unchanged from
F.5):** held-out-question AUC ≤ the popularity backoff ⇒ the similarity claim fails on the easiest
property it will ever be offered, and is reported failed. **Cost:** ~2 retrains + suite ≈ 4–6 h.

## H.4 Experiment G — THE GUARD: "new vocab means knowing what NOT to ask"

**Reframe, forced by measurement:** catalogue-wide questions are worth ≈ nothing (−0.00104…+0.00119
per draw vs +0.00101…+0.00470 in-pool; best-of-30 catalogue rarely reaches mean-of-20 pool). So the
honest open-vocab claim is not "it finds treasure outside the pool" — the treasure mostly is not
there — but **"it can score any question and correctly declines the junk."**

**Design.** Open the vocabulary to all 19,390. Score = the distilled policy's logit, with the
reaction head as a **hard guard**: candidates whose predicted P(no-answer) exceeds a threshold are
masked. Sweep the threshold on validation (declared sweep, not a tuned knob smuggled in — report the
whole curve, pick nothing post-hoc). Compare against: E1 open-vocab unguarded (−0.02034, 9.1%
answered out-of-pool), E1 pool-restricted (the ceiling), and guard-by-popularity (the dumb guard — if
popularity alone recovers the same deficit, the learned head adds nothing; say so). **Success:** the
guarded open-vocab policy recovers most of the −0.02034 while retaining a nonzero out-of-pool rate
whose picks get answered at a rate near in-pool. **Falsifier:** if recovery requires driving
out-of-pool picks to ~0%, the claim collapses to "the pool was already right" — report that; it is
the F.5-era pre-registered failure mode and it is informative about the catalogue, not just the model.
**Cost:** rollouts only, ~2–3 h.

## H.5 What the chapter then says

Tree = existence proof that the measured criterion pays. Distilled policy = the same criterion in a
form that generalises. Capabilities section = one experiment per claimed property (R: new states ·
Q: similarity · G: vocabulary), each with its falsifier fired-or-not. Value model = the measured
negative, five arms, with mechanisms — *learning the action beats learning the value, and even
perfect values lose* — which rhymes with the chapter's thesis (measured beats modelled) rather than
apologising for it. **Never claim routing; never claim strategy; never claim "adaptivity is large";
the bank result (+0.01430 > +0.01035) travels with every adaptive sentence.**

**Order: R → Q → G.** R first because it is the headline and its fine-tune data already exists.

---

# PART I — THE GENERALISATION PAIR: P1 (questions) and P2 (states) — DESIGNED, NOT RUN

> **AUTHOR SAID (2026-08-15, verbatim):** *"i want to show generalisability working here and my
> recommender support ood questions etc. i need a policy capable of understanding that and emitting a
> questions. how can we achieve this?"* — and, on this part: **"go, but do not run, just design."**
> **NOTHING IN PART I MAY RUN WITHOUT AN EXPLICIT AUTHOR GO.**

## I.0 A reversal, recorded

Part H's verdict §2 ("the evidence-seeded fine-tune — DO NOT RUN") is **REVERSED**, and the reasoning
error was mine. "138 states cannot teach what 72,717 triples could not" conflated **value** learning
(the dense surface that resisted everything) with **action** learning (E1 absorbed the whole tree from
**202 labels**; G.D's own text says action learning needs one bit per state). And the phase-3
precedent cut differently than I claimed: that fine-tune *disturbed states the policy already handled*
(deviating at 14.2/16 positions on-manifold); P2 trains states where the policy has **no behaviour to
disturb** — R showed it is inert there (1–5% deviation). Opus registered the objection; it was right.

**Expectation-setting, binding on any write-up:** even full success moves the SCORE very little — the
mechanical room is small (redundancy upside ~5% of asks; conditioning's edge shrinks as free evidence
grows, +0.0167→+0.0116 across three points, no interval — a trend, not a finding). **The prize is the
capability demonstration, behavioural, on the author's three axes** — not a number that beats anything.

## I.1 — P1: GENERALISATION ACROSS QUESTIONS ("emits questions it was never taught")

**Claim under test.** The distilled policy reaches a question through embedding similarity, not label
memory. The tree cannot do this even in principle — a question it was not built with does not exist
for it. This is Q's logic applied to the ACTION policy, never yet run.

**Design.** The tree's labels use **172 distinct questions** (76 items, 96 concepts; 45.4/54.6
user-weighted). Hold out **~20% of them (~34 questions) by identity**, stratified by channel ×
label-mass (how many (user,position) examples carry that question as label — stratify on it so the
held-out set is not accidentally all rare labels). **Retrain E1 from scratch** on the remaining ~138
label questions, with the held-out questions **still present in the candidate table** (they must be
*emittable*, never *supervised*). Everything else identical to E1: same trajectories, same replay,
same cohorts, same gates, canonical snap first.

**Measurements, in order of evidential weight:**
1. **Emission rate at the right states:** at (user, position) pairs whose held-out label was question
   X — does the retrained policy emit X anyway? Report top-1 and top-5 (X in the top 5 of 1,000).
2. **Neighbour credit, declared not smuggled:** if not X, is the emitted question among X's k=5
   nearest neighbours in the frozen descriptor space (neighbour list computed and committed BEFORE
   training)? Report separately from top-1; never sum them into one headline.
3. **The policy score consequence:** K=16 validation rollout of the retrained policy vs E1. Expected
   near-parity; a large drop means the 34 held-out questions were load-bearing and similarity did not
   substitute — itself a finding.
4. **Control:** the same retrain with the held-out questions ALSO REMOVED from the candidate table,
   then restored at inference — distinguishes "learned to reach them" from "never needed them".

**Pre-registered falsifier:** held-out emission top-5 rate ≤ the base rate a label-shuffled policy
achieves ⇒ the policy reaches questions by memory, not similarity — the OOD-emission claim fails and
is reported failed. **Baselines:** shuffled-label policy (floor), and a popularity-ranked emitter
(the dumb comparator — if popularity alone "finds" X as often, similarity added nothing).

**Cost (this box, from measured anchors):** one E1 retrain ≈ 25–93 min + rollout ~15 s/cohort + the
emission audit (banked trajectories, no new NDCG). **≈ 2–3 h.** No new labels. No test look.

## I.2 — P2: GENERALISATION ACROSS STATES (the evidence-seeded fine-tune, properly designed)

**Claim under test.** Trained on states where evidence is already on the table, the policy stops being
inert there (R: 95–99% unchanged) without becoming arbitrary (R2: deviations cost −0.012).

**Training set.** A MIXTURE, and the mixture is the design: the original E1 tree-state labels as
**replay** (the proven anti-forgetting mechanism) + the evidence-seeded states, each with its
measured-argmax label, upweighted so seeded states are ~1/8 of each batch (declared constant, not
tuned post-hoc).

> ### ★ CORRECTION, 2026-08-15 (author's catch) — "138" WAS AN UNDERCOUNT; THE NUMBER IS **209**
>
> This section originally said *"the 138 evidence-seeded states from the 2026-08-14 random-prefix
> sampling"*. The 425-state / 72,717-triple bank was built for the **value** model and has only ever
> been used for value regression — but **every one of its states also carries a measured ACTION label,
> and none of them had ever trained anything.** Counted by source rather than assumed:
>
> | source | states | on E1's manifold? | action-label teacher |
> |---|---|---|---|
> | `ndcgtree16_heaps` + `ndcgtree_heaps` | 216 | **yes** (the tree's own greedy tails) | — this is the replay, excluded |
> | `policy_labels_deep_partial` | 42 | no (below the floor stops) | **exact** (~996 of 1,000 measured) |
> | random-prefix sweeps | 138 | no | **weak** (argmax of ~22 in-pool of 50 sampled) |
> | `greedy_static_policyB_{items,concepts}_heaps` | 31 | no (static-order states) | **exact** (the banked `pick` is greedy's own choice) |
> | | **209 off-manifold** | | |
>
> **210 states were used** (138 + 42 + 30): one of the 31 static-order states is the EMPTY state,
> which E1 already handles with the tree's root, and relabelling it would disturb behaviour the policy
> already has — the phase-3 failure mode I.0 warns about — so it was dropped and the drop is counted.
> Together they carry **125 distinct label questions, 77 of which the tree never asks**, so the
> fine-tune genuinely widened the supervised action vocabulary rather than re-running the tree's own
> labels at new states. Result: `docs/results/P2_EVIDENCE_SEEDED.md`.

⚠ **Teacher honesty, carried on every number:** the seeded-state labels are **argmax over the 50
sampled candidates** (20 pool + 30 catalogue), NOT over the full 1,000-pool. An imperfect teacher is
acceptable for a capability test and must be named. **Optional upgrade, priced:** full-pool sweeps at
the 138 states ≈ 12–23 h on the phase-3 cost model (s/eval = 0.0857 + 1.57·n/1000, states are
200–800 users). Run the cheap version first; upgrade only if direction is right but magnitude is shy.

**Evaluation — all on FRESHLY DRAWN seeded states (new seed for the volunteered draw, disjoint from
both the 138 training states and R/R2's evaluation draw), through R2's existing harness:**

| gate | bar |
|---|---|
| **spanning gate (G, HARD RULE 8)** | K=16 unseeded validation rollout must not fall below **E1's 0.28355** — the replay's job; if it fails, the arm is rejected regardless of anything else |
| **★ the deviation-cost flip** | on fresh seeded states, fine-tuned policy − its own frozen order: R2 measured **−0.01246/−0.01234/−0.00860**; the bar is **≥ 0** (CI excluding a loss), with positive as the win |
| vs the tree on seeded states | report, paired CI — the capability sentence needs "not worse", not "beats" |
| redundancy | report with an interval this time; expectation is "no detectable avoidance" persists (base rate ~1.4–5.5% is too small to carry a claim either way — H's correction) |
| controls | shuffle the seeded-state labels (HARD RULE 5); anti-collapse diagnostics as in E1 |

**Pre-registered falsifier:** if, trained ON evidence-seeded states, deviations on fresh seeded states
still cost (CI excludes zero from below), then **this action space does not reward state-conditioning
beyond the tree's** even when taught — the strongest honest closure available, and it is reported as
the finding, not buried.

**Cost:** fine-tune ≈ 1–2 h (E1-scale, replay-dominated batches) + R2-harness evaluation ≈ 15 min +
controls ≈ 1 h. **≈ 3–6 h total.** No test look.

## I.3 The sentence the pair is for

If P1 and P2 both pass: *"the policy conducts sensible interviews from states it was never trained
on, and emits questions it was never taught"* — with Q (the reaction head's unseen-question result)
as the third leg. Generalisability demonstrated **behaviourally** on all three of the author's axes
(new states, new questions, new users), with the tree structurally unable to enter any of the three
games. If either fails, its falsifier sentence goes in instead — both are honest chapter text.

**Order if run: P1 first** (cheaper, no new machinery, and its negative would lower the prior on P2).

---

# ★ PART J — EXPERIMENT V IS DEAD, AND THE DAgger QUESTION ANSWERED (2026-08-16)

> **AUTHOR SAID:** *"plz kill running jobs, discard the latest experiments chkp, make a note against
> the design that it is dead"* … *"we already have most of it via a tree. what exactly do we build
> this on / for?"*

## J.1 V is DEAD — killed mid-placebo, checkpoints discarded

Killed at ~17:00 during the placebo arm. Heavy intermediates (heaps, partials, per-user vectors)
**discarded**; the small result JSONs and logs are kept because the per-group table is a real finding.
**No number from V's headline (+0.00549 all_groups) may be quoted — its placebo never completed, and
an in-sample cohort gain is ≥ 0 by construction.**

**Design faults, all identified by the author, all conceded:**
- **500 candidates, not 1,000** — inherited from the genre arm for comparability, not because 500 is
  the right action space. So V never searched the true best question.
- **Cohorts by INFERRED genre** of 4 volunteered films is a crude proxy, and with ~2.5 liked films per
  user genre counts tie constantly, **broken alphabetically** — the tie-break census was queued and
  never ran.
- **k = 4 volunteered films is arbitrary** (R's middle value).
- **It measured a CEILING with per-group static lists — a diagnostic, never a step toward a method.**

**What survives, and it is worth keeping:** the per-group shape at K=15 on 10,000 validation users.
**Every group with n ≥ 200 is an interval-backed null except `none`.** Action (2,758) +0.0002,
Drama (2,228) −0.0005, Comedy (1,323) +0.0028, Adventure (1,180) −0.0006, Crime (1,027) +0.0069,
Thriller (391) +0.0092 — none separating. All apparent gain sits at n ≤ 146 (Sci-Fi +0.038, Horror
n=41 +0.113, **Film-Noir n=7 +0.258**) — the textbook overfitting signature. Contrast the genre arm on
the same cohort: Action (n=1,582) **+0.0222** and Comedy (n=1,475) **+0.0229**, both separating.
⚠ And the killer scale fact: **greedy's own step-wise suboptimality (~5e-4, visible as Drama and
Adventure going slightly negative at K=15 only) is the SAME SIZE as the conditioning signal in V's
big groups.** The effect is at the noise floor of the instrument measuring it.

**Licensed conclusion:** *what a user volunteers carries far less routing information than their
answer to a question we chose.* That is a mechanical explanation for R and R2 and it closes the
resumption line. It is NOT licensed to quote V's headline gain.

## J.2 ★★ DAgger: the right technique, and it CANNOT help here — the reason matters

The author independently reconstructed **DAgger** (Ross & Bagnell 2011): train, run the learner,
collect **the states the learner visits**, label them with the expert, retrain **with the old data
retained** (her "sampled step 1" = the replay). Her worry about "unlikely states with rubbish
questions" is exactly why DAgger samples the *learner's* distribution rather than at random. Correct
on every point.

**But DAgger closes the gap between LEARNER and EXPERT, and here that gap is already ZERO.** E1
reproduces the tree to **1e-9** with `frac_never_deviates = 1.0000`. There is no distribution shift to
repair on-manifold, and no new states to collect, because a perfect imitator never leaves the
teacher's trajectories (the same fact that killed the E2 DAgger plan in Part A).

**★ THE BINDING CONSTRAINT IS THE EXPERT, NOT THE IMITATION.** To beat the tree you need a better
teacher, not a better student. And every attempt to improve the teacher has been measured:

| better teacher | result |
|---|---|
| deeper greedy (floor removed) | phase 3: **null** vs the tree, and the fine-tune on those labels **failed the spanning gate** |
| per-state / kernel greedy | infeasible at scale (~3,772 h at k=8) |
| a value model, so any state can be scored | **five arms, all lose**, even with perfect values |
| conditioning on volunteered evidence | **V: every big group a null** |

**And a second, independent reason DAgger cannot deliver what the author wants:** our expert is a
**MEASUREMENT, not a function** — one label costs a cohort fold plus ~500–1,000 catalogue-wide scorings,
i.e. 15 min to 2.4 h. Standard DAgger assumes a cheaply queryable expert. Every shortcut we took to
dodge that cost (best-of-50 candidates, sub-200-user cohorts, borrowed states) produced labels too
weak to teach from — which is most of this week in one sentence.

## J.3 What is actually left

**Every greedy-derived expert is MYOPIC.** Its labels can only ever teach "the best single next
question", so no amount of DAgger, data or replay produces non-myopic behaviour. The only two ways to
get a non-myopic policy are (a) an explicit lookahead expert — combinatorial and unpriced — or
(b) **reward-driven learning (RL)**, which needs no expert at all.

That is why RL keeps reappearing: **it is the only remaining lever, because it is the only one that
does not require an expert we cannot afford.** Its own prior remains poor (Part G.D: 2.1% routing
signal; endpoint polish +0.002@q16 / −0.029@q2 says the non-myopic prize is small), so the honest
position is *"the only untried lever is also the one our own evidence says is least likely to pay"* —
and that sentence, with the table above, is a defensible chapter closing.

**No new experiment is proposed here.** Part I's P1/P2 are spent, V is dead, and the next move is the
author's.

---

# PART K — PER-USER GAIN DISTILLATION (Fable, 2026-08-16) — the design the author asked for

> **AUTHOR SAID (verbatim):** *"what I want is basically the same model as we have now (distilled
> policy that emits actions embedding), but I want it to generalise to more states, ideally lots lots
> more sttaes, so a user can come with an unseen combination of things we know aout them and emitted
> question makes sense. so i guess model needs to sample a lot of combos and know which question would
> give th very best ndcg uplift for that user in that circumstance."* Budget: *"a day or 2 of training
> is still fine."* Then: **"ok, nice, plz generate the design doc and the code to implement this."**

## K.0 The unlock — label USERS, not COHORTS

Every prior attempt inherited the tree's frame: a label = the cohort argmax at one shared state, price
~2.4 h per state (1,000 candidates × 10,000 users). **`genv_eval_multi` batches heterogeneous per-user
sequences in one pass** — so give each user their OWN state and sweep the 1,000 candidates once across
all of them: **the same compute as ONE cohort sweep (~3.5 h at n=8,000) yields ~8,000 distinct states**,
each with a full per-user measured gain vector. That is the author's sentence — *the best NDCG uplift
for that user in that circumstance* — as a dataset, and it is what makes DAgger-style iteration
affordable (Part J.2's "the expert is a measurement" objection priced away by batching).

## K.1 The central bet, and how it is handled

**A single user's gain vector is maximally noisy** (the −0.724 small-cohort lesson at n=1). Handling:
- ★ **NEVER a per-user argmax as a hard label** — argmax-of-noise is the value-head death. Train with
  **soft LISTWISE targets over all 1,000 candidates** (per-user measured gains, temperature-scaled
  softmax target; the exact form is the pilot's one free hyperparameter, declared not tuned-post-hoc).
  Labels are noisy but **unbiased**; SGD over thousands of users in similar states is the smoother.
- This also ends the 202-distinct-label poverty: every candidate at every state carries a graded target.
- Useful structure in the noise: unanswerable candidates have measured gain ≈ 0 **by construction**
  (a refusal/UNSEEN fold moves nothing), so **answerability is taught automatically**.
- ★ **The PILOT decides whether the bet is live before real money is spent** (K.4).

## K.2 Firewall — one forced change

Per-user labels are computed from **validation users' target halves**, so training on them burns those
users for selection. **Split the 10,000 validation users 8,000 (FIT) / 2,000 (HOLDOUT) by user, seed
98765, fixed once.** All labels, training and model selection on the 8,000; **every behavioural claim
proven on the 2,000**, whose states and labels are never touched. Train users remain unusable (tower
memorisation, settled in D.6b). The one unspent TEST look stays reserved for Q. `--split test` refused
by the parser; `test_touched=false` everywhere.

## K.3 Design

| element | spec |
|---|---|
| **states** | sampled per FIT user from the deployment mixture: tree trajectories (replay share), volunteered-evidence seeds k∈{1..8}, partial random interviews, and (round 2+) the policy's own rollout states — the affordable DAgger step. Mixture weights declared in the artifact per round. |
| **labels** | full 1,000-candidate per-user measured gain vectors, one pass per round (~3.5 h at n=8,000), heterogeneous batching, per-user vectors persisted float32 append-only (the finest-grained rule) |
| **model** | UNCHANGED: the E1 emitted-vector architecture, warm-started from `policy_main_equivariant_both.pt` |
| **loss** | listwise CE against the soft target; tree-label replay mixed in as the anti-forgetting mechanism |
| **rounds** | pilot → 2–4 labelling rounds; round 2+ states include the current policy's own visited states |
| **budget** | pilot ~0.5 h · rounds ~3.5 h each · training 1–3 h each ⇒ comfortably inside 1–2 days |

## K.4 ★ THE PILOT — run first, decides everything (~30–40 min)

1,000 FIT users, each at one sampled state, full 1,000-candidate per-user labels (~27 min), one
training run (~minutes at this scale). **Gates:**
- **sanity:** canonical m=0 snap exact; banked-headline assert on the static order; leak 0/users;
  composition printed BEFORE training (state-type mix, prefix-length histogram, per-user nnz of the
  gain vector).
- **signal:** the trained pilot must beat the **shuffled-target control** on held-out FIT states, and
  must not fall below the warm-start's own numbers on-manifold (a pilot that only preserves its
  initialisation has learned nothing — report which).
- **If the pilot shows no trainable signal, the design is DEAD at pilot price** and is recorded dead.

## K.5 Gates for the full run (pre-registered)

| gate | bar |
|---|---|
| spanning (HARD RULE 8) | unseeded K=16 rollout on the 2,000 HOLDOUT users must not fall below E1 on the same users |
| ★ behavioural win | on fresh seeded states from HOLDOUT users: policy − its own frozen order **≥ 0, CI excluding a loss** (R2 measured −0.01246/−0.01234/−0.00860) |
| vs tree | report, paired CI; "not worse" is the sentence, "beats" is not expected (bank-beats-conditioning stands) |
| controls | shuffled-target arm; anti-collapse diagnostics; redundancy reported with an interval |
| honesty | myopia declared (one-step labels ⇒ greedy with reach, never strategic); Q's test look untouched |

**Expectation, binding on the write-up:** score lands ≈ E1; the prize is behavioural — *the emitted
question makes sense at combinations never seen* — proven on held-out users. Failure of the
behavioural gate after a passing pilot is reported as the finding, not retried silently.

---

# ★★★ PART K.7 — NUMBER AUDIT (2026-08-18): K.6's ARGUMENT STANDS, ITS NUMBERS DO NOT

A provenance audit against the artifacts (chapter-draft agent, commit `f78ee2a`) found **all three of
K.6's headline numbers wrong or unbanked**, plus two mis-stated mechanisms in the value-line summary
and **one arm that does not exist**. The conclusions survive; the figures must be replaced everywhere
they appear, including in anything already relayed to the author.

| K.6 claimed | truth |
|---|---|
| oracle **+0.229** | that is the mean over only the 845 users with a positive gain. **All-user value is +0.19366.** |
| best cohort-wide question **+0.0115** | matches no computable variant; **the banked field is +0.009664** |
| trained cold arm **+0.0153** | ★ **DOES NOT EXIST ANYWHERE IN `experiments/`.** Use the real per-k cells: k=2 **+0.00422** (null), k=4 **+0.00074** (null), k=8 **+0.00771** (separates) — and the same arm **fails the spanning gate at −0.06798** |

**The ~95% conclusion survives on either pairing (5.0%), so K.6's argument is unaffected** — the
per-user prize is overwhelmingly carried by information the interview has not revealed. Only the
figures change.

## ★ Two mechanisms I asserted that the artifacts do NOT support — retracted

1. **"Argmax over noisy estimates selects the over-estimates (answer rate 0.58→0.37)."** **False.**
   0.58→0.37 is the *warm regressor walking off its E1 start* (14.20 of 16 positions changed), not a
   winner's curse. And the ceiling sweep's answer rate **falls as accuracy RISES** (0.6188→0.5650),
   which is the opposite of the maximisation-bias story, and its rank-preserving construction rules
   that story out. **I relayed this mechanism to the author at length; it is withdrawn.**
2. **"The value surface is rough across states, inter-state correlation ~+0.29."** Mis-sourced: that
   figure is a **state-blind predictor's LOSO Spearman**, not a correlation between states, and it is
   **log-only** (`value_train.out:35`, the run was killed before its JSON). The JSON-backed value is
   **+0.2564 over 49 states**. Re-derive or drop the roughness claim; do not quote +0.29.

## ★★ THE CEILING ARM DOES NOT EXIST — Fable's Friday suspicion confirmed

The "flat table −0.02911" and "**perfect measured values** −0.02911" identity was flagged on
2026-08-15 as *a surprising number is a harness bug until proven otherwise*. It is a bug:
**`ceiling[18]` is field-for-field identical to `arms.meanvec_PxV` — the same run recorded twice.**
ρ=1.00 therefore means perfect fidelity **to the flat mean table**, not to true per-user gains.
⇒ **"even perfect values lose" is NOT established and must be struck from the value-line summary.**
The value line remains a four-arm negative, not five.

## Other audit findings, all binding on the write-up

- **`docs/results/PHASE3_DEEPER.md` contains no results** — its RESULTS section is still a PENDING
  checklist. Every phase-3 number quoted this week came from the dated artifacts and **does** verify;
  the doc needs filling from them.
- **Q has NO confidence intervals anywhere**, over 620M highly non-independent cells, and
  57.0% / 78.1% / 69.6% / 91.6% are unstored derived ratios. **Write as a live limitation.**
- The bank tail contrast is **n = 9,281**, not 10,000 (validation tail is 9,217).
- **CI audit CLEAN:** no `1.96*sqrt(2)*SE` signature anywhere; `paired_bootstrap` reproduces
  bit-identically at seed 98765 / 10,000 resamples.
- **In the current tex and unsourceable: `0.374 / 0.320 / 0.087`** in the Reach paragraph — it carries
  the "+0.25 for answering at all vs +0.048 for question choice" headline and **no artifact holds
  those baselines. Highest-priority fix before submission.**
- Q's evaluation users are held out from the head but **not from the tower** that produces their state
  latent — defensible only because no NDCG is measured there, and it must be stated, not left implicit.

---

# PART K.6 — THE CLOSING VERDICT (Fable, 2026-08-17): why distillation works and per-user labels cannot
### ⚠ NUMBERS SUPERSEDED BY K.7 ABOVE — the argument below stands, its figures do not

> **AUTHOR ASKED:** *"why policy can distill a tree but cannot learn on this dataset? surely if we show
> it enough combinations of state and optimal question, it could?"*

**The tree's label is computed FROM the state; the per-user label is computed FROM THE ANSWER KEY.**
The tree decides its question from the asked-and-answered history alone — its label is a true function
of the policy's input, and a function that exists can be learned (E1: exactly). The per-user optimal
question is found by scoring against the user's **held-out targets** — information not in the state and
mostly not derivable from it. Two users indistinguishable to any policy have different best questions
because their hidden tastes differ: **the mapping is not a function of the input, and no quantity of
data supplies a function that is not there.**

**What "enough combinations" converges to:** the best state-conditioned answer — the question best *on
average* for users at that state. **That object is the greedy criterion; at coarse states it IS the
tree.** The per-user route, taken to its limit, converges back to the thing distillation already gives
exactly. Measured (⚠ **CORRECTED — the K.6 figures +0.229 / +0.0115 / +0.0153 were audited WRONG and must
never be quoted; see the correction table above**): per-user oracle **+0.19366** (all users; +0.229
was the mean over only the 845 users with a positive gain), best cohort-wide question **+0.009664**,
and the "trained cold arm +0.0153" **does not exist anywhere in `experiments/`** — the real per-k
cells are k=2 **+0.00422** (null), k=4 **+0.00074** (null), k=8 **+0.00771** (separates), with the
same arm failing the spanning gate at **−0.06798**. **~95% of the per-user prize is carried by information the
interview has not yet revealed — which is not a modelling failure but the reason interviews exist.**

**This is the June clairvoyant lesson in a new costume** ([[clairvoyant-imitation-breakthrough]],
[[night-adaptive-edge-investigation]]: cloning the oracle gave top-1 14% and lost to greedy, because
the oracle's edge is privileged). The per-user argmax label is a privileged teacher. I designed Part K
without recognising it; the lesson was in our own notes.

**What remains learnable** — conditioning finer than the tree's 22 nodes — has now been measured
independently four ways (nearest-state lookup +0.009 in-sample; V's cohort nulls; K's realised-gain
null; R2's costly deviations) and is real, small, and mostly captured already.

**The chapter's closing sentence:** *the policy learns everything the state can teach — and the state
can teach almost nothing about which question is best, because that knowledge is precisely what the
interview exists to acquire. Distillation is not the fallback; it is the optimum given the information
available.*

**PART K IS CLOSED. The policy workstream's experimental record is closed with it.** The remaining
work is writing (Fable verdict 2026-08-15, §5), plus the reserved Q test look at draft time.

---

# PART L — THE FOREST: as-if-asked encoding (L1) and multi-teacher distillation (L2) — SPEC ONLY, NOT RUN

> **AUTHOR SAID (2026-08-17, verbatim):** *"can you spec that test and put it in todo? and have another
> think how to cover question choice at novel states? ... maybe we can devise multiple types of
> groupig? cold intervirew tree that we have + volunteered genre tree + maybe something else ... we
> need multiple examples for a state and the next question to denoise, it that it?"*

**Her closing sentence is the design principle: DENOISING IS COHORT MEASUREMENT.** A conditioning
grouping is admissible iff (a) its signature is **observable at ask-time** (else the label is
clairvoyant — K.6), and (b) its groups are **large enough to measure honestly** (≥~500 users; the
−0.724 exposure says <200 is noise). Every failure this week violated (a) (per-user labels) or (b)
(below-floor states, V's small groups). Every success satisfied both.

## L.1 — R3: the as-if-asked encoding test (no training, ~minutes)

**Claim:** volunteered STATED facts should enter the policy through the ASKED-HISTORY, not only the
belief. A stated fact is an answered question nobody asked. R fed evidence into `z` only, history
empty ⇒ the policy saw "nothing asked" and was correctly inert. Encoding it as history moves the novel
state ONTO the trained manifold instead of asking the model to generalise off it.

**Design.** Re-run R's harness with a third encoding arm. For each validation user, the volunteered
evidence is written as history tokens `φ_q ⊗ outcome` (like/dislike from their rating) AND folded into
`z` AND masked from re-asking. Three cells:
1. **root-matching:** volunteered fact = the tree's root concept (the stated-genre case). **Prediction,
   falsifiable: the policy branches exactly as the tree's post-root arm — E1 fidelity guarantees it if
   the encoding is right.** If it does not, the encoding is wrong, not the theory.
2. **pool-matching, non-tree:** volunteered facts are pool questions that are not tree splitters.
   Unknown behaviour — this is the genuinely new cell.
3. **arbitrary items** (R's original draw) as history tokens. Measures nearest-manifold behaviour.

**Report per cell:** K=16 full/tail vs the tree-from-root on the same seeded users (paired CI);
redundancy (must be **exactly 0 by construction** — assert it, it validates the mask wiring);
distinct interviews; where first divergence occurs. **Falsifier:** if cell 1 does not reproduce the
tree's branch behaviour, the as-if-asked claim dies and is reported dead.

## L.2 — FOREST DISTILLATION: multiple cohort-level teachers, one policy

**Teachers, ranked by evidence:**
| teacher | status | prior |
|---|---|---|
| T0: the cold tree | banked (E1's) | +0.028 val / +0.010 test |
| **T1: stated-genre orders** | **17 per-group orders ALREADY BANKED** (`groupgreedy_genre15`, sequences re-scored under policy B, test-grade +0.010) | +0.018 val |
| T2: evidence-count bands (none / 1–2 / 3–8) | needs ~3 greedy orders, cohorts ≥1,000 by construction | R's shifting interview value |
| T3: taste-popularity band (mainstream/mixed/obscure by fold-in tail mass) | needs ~3 orders | the thesis's own head/tail axis |
| ✗ inferred-genre-of-items | — | V: null in every big group. DEAD |
| ✗ per-movie | — | n=1, the clairvoyant trap. DEAD |

**Training set:** for each teacher, seed states matching its signature (genre stated ⇒ genre-concept
answered in history per L.1; count-band ⇒ k volunteered items as history; cold ⇒ E1's trajectories),
label = **that teacher's next question at that state**. All labels are cohort-measured functions of
observable state — learnable AND denoised, by construction. Mix across teachers + tree replay;
`pu_policy`'s trainer shape reusable (hard labels this time — each teacher's pick, not per-user soft
targets).

**Gates:** spanning (cold-state behaviour ≥ E1, exact); **per-teacher fidelity** (at states matching
teacher T's signature, top-1 vs T's own order — the multi-teacher analogue of G-E1a); behavioural
(on fresh seeded states, policy vs the matching teacher's own order: null or better); shuffle control
(teacher assignments permuted across signatures). **Falsifier:** if fidelity to T1 costs fidelity to
T0 (interference — the P2 lesson), the single-policy claim fails at 2 teachers and scaling the forest
is dead.

**Cost:** L2a (T0+T1 only, all material banked): state-gen + training ≈ **3–5 h**. T2/T3 orders:
~1–2 h per group × ~6 groups, only if L2a passes. **Order: L1 → L2a → (decision) → T2/T3.**

**What the forest honestly buys:** coverage of the *evidence-type* axis (stated facts, evidence
amount, taste band) at cohort reliability — not coverage of all conversational states. The residual
(fine-grained novel states) stays bounded small by V/K/nearest-state. The chapter claim if it works:
*one policy, many measured teachers, exact where any teacher is defined, graceful default elsewhere.*

---

## L.3 — RESULT (2026-08-17, run under "plz run l1 and l2")

**L1 — `docs/results/L1_ASIF_ASKED.md`.** Cell 1's prediction fired exactly: with the volunteered fact
encoded as an answered question, the policy branches as the tree's post-root arm for **100.00%** of
10,000 users (R's encoding: 0.00%), scores identically to banked E1 **per user**, and re-asks the
volunteered question **zero** times. The turn this saves is worth **+0.00167 full [+0.00096,
+0.00238]**. ★ But cells 2 and 3 measured the other half: where no teacher has an opinion about the
volunteered fact, reading it makes the policy improvise and LOSE — **−0.04663** (random answerable
concepts), **−0.02231 / −0.01643 / −0.01070** (R's item draw at k = 2 / 4 / 8). **The encoding is
necessary and not sufficient; the fix is more teachers, which is what L.2 is.**

**L2a — `docs/results/L2_FOREST.md`.** The interference falsifier **did not fire**: T0 fidelity
1.000000 → 0.999969 (**one tree state of 32,000**) while T1 went 0.000369 → 0.999963; spanning is
per-user IDENTICAL to E1 on 2,000/2,000 HOLDOUT users; the shuffle control scores 0.000000 against the
true orders. All five pre-registered gates pass. ★ **And the second teacher is not worth following:**
at genre-seeded states the ordering is **tree > forest (= T1 exactly) > E1 improvising**, with
forest − tree = −0.02648 / −0.03290 / −0.03075 (Drama / Action / Comedy, all excluding zero) and a
null on Adventure.

### ★ A CAUTION ON T3, BEFORE ANYONE PAYS FOR ITS ORDERS (2026-08-17)

L.2 lists **T3 = taste-popularity band (mainstream / mixed / obscure by fold-in tail mass)**. Check it
against condition (a) before spending: **a cold user's tail mass is NOT observable at ask time.** It is
computed from the fold-in profile the system does not yet have — the same shape of error as the
per-user label of Part K.6, in a new costume. T3 is admissible **only** as a band estimated from
evidence the user has actually volunteered, i.e. inside T2's regime, and its cohorts must then be
counted per evidence level, not over the whole cohort. **T2 (evidence-count band: none / 1–2 / 3–8) is
the sound one** — how much someone has told you is observable by construction.

A second practical note for T2: its states carry PER-USER prefixes (each user volunteered different
items), which `greedy_certified` cannot search — its `prefix` is one shared head. A per-user-prefix
CELF loop over `genv_eval_multi` is needed, and its gate is that it reproduces `greedy_certified`
bit-for-bit when every user carries the same prefix.

**★ THE AMENDMENT THIS FORCES ON L.2's DESIGN LAW.** Observability and cohort size make a teacher
**learnable**. They say nothing about whether it is **good**. A third condition, now measured:

> **A teacher earns its place in the forest only if it beats the teacher already covering those
> states — on the same action space and the same budget.**

T1 was searched over 500 ITEMS ONLY; the tree spends most of its budget on CONCEPTS, so the comparison
was never like-for-like. **T1' RAN AND ANSWERED IT: the forest stops at one teacher.** The stated-genre teacher rebuilt over
the tree's own combined pool (FIT users only, ~4.7 h) does spend turns on concepts — 5 to 9 of 15,
where the item-only teacher had none by construction — and on the HOLDOUT users of each group it
**still loses to the tree in 0 of 4 groups won, 3 lost, 1 null**: −0.01650 [−0.02476, −0.00836]
(Drama), −0.04789 [−0.06821, −0.02820] (Action), −0.03734 [−0.05634, −0.01803] (Comedy), −0.02806
[−0.05522, +0.00068] (Adventure), against item-only −0.02648 / −0.03290 / −0.03075 / −0.00587. The
comparison is generous to the item-only orders (they were searched on all 10,000 validation users, so
these HOLDOUT users were inside their cohort) and clean for the rebuilt ones — and the clean one still
loses. **The action space was not the explanation.** Conditioning on a volunteered category does not
pay against a cold interview that may ask about concepts. `docs/results/L2_FOREST.md` carries the
table; T2 is NOT started on the back of this.

# PART L.4 — WHY THE GENRE TEACHER LOST: a FIXED LIST vs an ADAPTIVE TREE (2026-08-18)

> **AUTHOR ASKED:** *"how can it be if we literally select best questions for them, and they turn out
> to be not the best?"*

**The comparison was between two different KINDS of object, and that resolves the paradox.**

- **T1/T1' is a FIXED ORDER** — the same 15 questions, in the same sequence, for every member of the
  group, whatever they answer along the way. **Zero branch points.**
- **The tree is ADAPTIVE** — it branches on every answer. **Fifteen decision points**, each
  conditioned on everything heard so far.

So L2's cells did not measure *category-conditioning vs none*. They measured **one-off conditioning at
the start followed by a script, versus no category conditioning but fifteen rounds of adaptation** —
and within-interview adaptation wins by −0.017 to −0.048.

**★ The confirming evidence is in the genre arm's own construction:** its published **+0.018 val /
+0.010 test was measured against the GLOBAL STATIC ORDER** (`global_comparator` / `global_sequence` in
`groupgreedy_genre15_TEST_policyB.json`), **not against the tree**. Against a fixed list, per-category
lists genuinely help. Against an adaptive tree, they lose. **Both results are correct and answer
different questions** — the chapter must name the comparator on every such figure, or the two read as
contradictory.

**Licensed:** *knowing someone's stated category up front is worth less than being able to react to
their next fifteen answers; a per-cohort fixed order buys the first by giving up the second.*
**NOT licensed:** "the tree is optimal", or "category conditioning does not pay" without the qualifier
**"as a fixed list"**.

## The fair fight, and its price

**Grow a TREE per stated category, not a list** — then the arm has both the conditioning and the
within-interview adaptation, and the comparison is like-for-like. **Nothing measured so far rules this
out; L2 tested lists only.**

- **Cost ~10 h per tree** (the cold tree's measured build) ⇒ ~40 h for four categories. Drama, Action
  and Comedy hold 1,000+ FIT users each; **Adventure's 421 is already below the design law's own 500
  floor** and would not qualify.
- **Prior, honestly poor:** the cold tree's root split already separates people usefully, so a category
  tree may rediscover much of the same structure; and all 17 genre groups shrank validation→test, so
  part of T1's loss is transfer rather than structure.
- **Verdict: designed, NOT recommended at 40 h against that prior — but it is the only remaining way to
  test the author's intuition fairly, and until it runs, "category conditioning does not pay" stands
  for FIXED LISTS ONLY.**


## Cross-references

`PAPER_B_PLAN_2026-08-12.md` §6 (workstream E) · `PAPER_B_TODO.md` 1.13/1.14/1.15/1.16/1.22 ·
`docs/results/CONDITIONED_ARMS_ON_TEST.md` · `docs/results/TREE_DEPTH_AND_MINLEAF.md` ·
`docs/results/HU_YU_2013_REPLICATION_AND_ABLATION.md` §B ·
`docs/results/TREE_ROUTING_SPECIALISATION.md` · `docs/results/POSITION_VS_MODERN_CRS.md` Part II ·
memory `clairvoyant-imitation-breakthrough`, `night-adaptive-edge-investigation`,
`policy-ladder-ndcg-reward`, `banked-global-greedy-is-a-policy-A-artifact`,
`surprise-means-debug-the-harness`.

---

## Part M — THE PLAN FOR THE REST OF THE CHAPTER: the COVERAGE programme (2026-08-17, author + Fable review + author's corrections)

**This Part supersedes Fable's 2026-08-17 vision where they conflict.** The author's corrections that
bind it:
- The chapter's spine is the **elicitation audit** (field's methods on a real NDCG ruler, all losing;
  measured-vs-surrogate 56/56 sign flip; the tree beating everything) — NOT "the value of volunteered
  information". Volunteered info is the closing act, not the spine.
- **"The interview is evidence-invariant" is BANNED as a headline.** It is a negative inferred from
  arms never trained to adapt (R's broken encoding, R2 an ablation, P2 on 210 states with mostly
  approximate labels) — HARD RULE 6 forbids marketing it as a finding. Claim 2 is UNTESTED, not
  anti-supported: nothing was ever trained to change its order with the opener. The forest is that
  training.
- E1 holding the tree is an instrument check, never a contribution.
- Hu & Yu stays IN the chapter; it is not split out as a standalone paper.
- **The bar is COVERAGE, not dominance** (author): show one tree does not cover all bases and the
  policy can cover several — ideally recognising overlap. ONE cluster where a dedicated teacher beats
  the incumbent at its own states is a positive result. Fable's uniform-dominance abandon rule is
  replaced accordingly.

**What survives from Fable's review, verbatim binding:**
- Bar for any "beats the tree" cell: **≥ +0.004 full NDCG@10, paired 10k bootstrap, CI excluding
  zero, validation** (~40% of the tree's +0.01035).
- **~8 clusters, never 20** — 500/cluster sits at the floor of the −0.724 noise regime.
- The skip-redundancy channel is arithmetically dead (5.4% redundancy ⇒ ~0.8 saved turns ×
  +0.00167 ≈ +0.0013 < MDE/3) — print the multiplication in the chapter.
- K.6 enters the chapter as a **scoping theorem** (the per-user label is not a function of the state;
  the oracle gap is why interviews exist), 2–3 sentences in any paper's method section. Audited
  figures ONLY: oracle +0.19366, cohort-best +0.009664. The K.6 originals are wrong.
- The single unspent TEST look is spent at the END, on ONE pre-registered comparison.

### The stages

**M0 — bank claim 3, the cheap half (NO linking).** Fold k volunteered items drawn from ML fold-in,
ask nothing, vs the m=0 snap 0.15687. Falsifier: ≤ snap at any k ∈ {2,4,8}. Hours. The real-text
(SBERT, Reddit-linked) version is M5b and is a SEPARATE claim — the cheap one does not wait for it.

**M1 — cluster ML users by TASTE, then NAME each cluster.** Cluster folded-evidence `z` (TRAIN users,
k≈8 volunteered draw), ~8 clusters. Then, per cluster, test which observable descriptor best
separates it: genre, anchor film, mood/tag (genome), era, or a mixture. The best-separating
descriptor becomes the cluster's OPENER — an open question, not a ternary split — and the
admissibility signature ("drama" may be useless where "dark" is useful; that is now an empirical
output, not an assumption). Gates: descriptor observable at ask time; cohort ≥ ~1,200 val users;
a cluster with no separating descriptor is reported and dropped (it cannot be routed to).
Cost: ≤ a day, mostly CPU-light.

**M2 — THE SCREEN.** Per named cluster, measured-greedy sweep over the full question vocabulary at
that cluster's evidence states vs the tree's root at the same states, paired, identical fold both
arms. Controls: shuffled cluster assignment (must show nothing); zero-evidence snap (sweep must
recover the tree's root). **Read-out under the coverage bar: each cluster clearing +0.004 is a base
the tree does not cover; ZERO clusters clearing ends the beat-the-tree line for good** (action space,
encoding, and conditioning signal all separately falsified) — the chapter then claims audit + tree +
mechanism record, and NOTHING is marketed in its place. ~1 day; bench one cluster first.

**M3 — adaptive teachers for the clusters that cleared.** Grow a ternary TREE (not a static order)
per clearing cluster on TRAIN via tree application, select on val. **Each teacher is individually
gated by the L2 law: it must beat the incumbent at its own states (+0.004, CI excluding zero) or it
is replaced by the cold tree — fallback pre-registered here, not post-hoc.** Cost ~2–3 days if
several clusters clear; bench first.

**M4 — the forest distilled into ONE policy.** L2a machinery, warm-start E1. Gates: every surviving
teacher's fidelity ≥ 0.999 at its own states; cold-tree fidelity unchanged (zero interference, the
L2a result at N=2 re-proven at N=final); shuffle control. **The policy's contribution is stated as:
N teachers, one model, zero interference — where N separate trees are N separate models that cannot
share or recognise overlap.** Overlap probe: users matching two signatures — report which teacher the
policy follows and at what cost vs both.

**M5 — Reddit, two independent halves.**
(a) **Opener census** (no linking): type the demo-half openers (exemplar/genre/mood/era/constraint +
frequencies); report whether the M1 cluster descriptors correspond to opener types people actually
lead with — that is the "supported by data, not a lucky guess" sentence. If exemplar films dominate,
that is direct support for the fold-is-the-mechanism story.
(b) **Real-text consumption demo** (linking + SBERT phrase→genome-concept→fold): out-of-vocab request
text into recommendations beating popularity. Feasibility gate per `DS_REDDIT_LINKING.md` first.
Either half failing is omitted, never written as a limitation (author ruling).

**M6 — claim 2 counted + the TEST look.** Trajectory divergence by opener over the final policy:
fraction of interviews that differ by cluster, depth of first divergence, semantic read as garnish.
Then the single TEST look on ONE comparison, chosen by what survived: forest-policy vs cold tree at
evidence states if M3/M4 landed; else the strongest banked validation claim is stated as
validation-only.

### Decision points
- **After M2 (~2 days in): the fork.** ≥1 cluster clears → M3–M6 (the jazzy chapter). Zero clear →
  skip M3/M4, keep M5+M6 census/demo, chapter = audit + tree + mechanisms.
- **~1 Sep: venue call.** M3/M4 landed with the TEST look → **ECIR 2027 (2 Oct)**; RecSys 2027
  fallback. Not landed → TORS with the audit+mechanism paper; ECIR is not attempted with a spine of
  negatives.

### Standing constraints (unchanged)
Validation only until M6; certified pair; policy-B mask; canonical snap 0.15686538; paired percentile
bootstrap 10k seed 98765; design sheet before every run; commit before launch; persist per cell;
bench one cell before any multi-day estimate; no LLM calls without author approval; never the Reddit
API; never votes as labels; nulls not citable; no truncation without sign-off.

### M.1 — REPLAN (author corrections, 2026-08-17 late)

**Supersedes the stage list above where they conflict.**

**M0 is DELETED.** Fold-pays-without-questions is Chapter A's territory and already proven; re-proving
it here was padding. What the author actually wants demonstrated is that **volunteered info from
REDDIT is used** — real request text, including OOD attributes projected into rec space — and that
belongs INSIDE the Reddit section (new M5a) as its intro, not as a standalone stage.

**Data sufficiency, MEASURED (question 5): we hold enough and process nothing further.**
- 4,431 threads fully adjudicated (batches 0000–0099 of 681), each ≥5 in-catalogue GT films WITH
  request text; ruler built (`data/reddit_he/ruler/ruler_v1.csv.gz`; GT median 10, mean 14.9).
- **Typed slots ALREADY EXTRACTED for all 4,431** (`data/reddit_he/slots/full_out_*.jsonl`):
  anchors_pos/neg, genres_pos/neg, era, tone, constraints, context.
- The thread shuffle (seed 98765, threads never split) makes any batch prefix an unbiased random
  subsample ⇒ the remaining 581 batches stay unprocessed. **No new adjudication, no new LLM calls.**

**TWO OPENER FAMILIES (author's design, question 3):**
- **DREAM openers (M1)** — cluster ML users by taste, name each cluster by its best-separating
  observable descriptor. The upper bound: what openers *should* be if users could be asked anything.
- **REALISTIC openers (M1b, NEW)** — mine the opener distribution from the 4,431 Reddit slot records
  (which genres/tones/eras/anchor-types people actually lead with, at what frequencies), map each
  opener type onto the ML attributes we already hold, and form cohorts of ML users matching each.
  **Headline comparison: how much taste-separation (and downstream screen gain) do realistic openers
  lose vs dream ones?** Cost: the census is a groupby over existing slot files (~hours); cohort
  construction ~a day.
- **M2 screens BOTH families** under the same +0.004 bar. A realistic opener that clears is worth
  more to the story than a dream one (deployable as asked); a dream-only clearance quantifies what
  is lost by asking natural questions.

**M4 clarified (question 2): yes** — the forest is the COLD tree (no opener, the incumbent, kept
exactly) **plus** the opener-routed teachers from M3, all distilled into ONE policy. A user who gives
no opener gets the cold tree verbatim; an opener routes to its teacher. The L2a zero-interference
result is re-proven at final N.

**REDDIT'S ROLE, now in one place (question 4) — section M5, three parts, linking scoped exactly:**
- **M5a (intro demo): a real request becomes recommendations.** Request text → slots (held) →
  anchors linked to catalogue via the **rules tier** (no LLM; 0.971 precision, 48.6% coverage —
  losses printed) → OOD phrases (tone "dark", constraints) via SBERT phrase→genome-concept →
  **fold** → score on the thread's own reply-set GT on the EXISTING 4,431-thread ruler, vs
  MostPop gate + shuffle control. **Linking needed: request-side anchors only, rules tier only.**
- **M5b (realistic-opener mining): NO LINKING AT ALL** — slot distributions feed M1b directly.
- **M5c (optional, only if M3 lands): openers on Reddit frequencies.** Weight the M6 trajectory
  count by the realistic opener distribution, so "the interview visibly adapts" is stated under the
  empirical opener mix, not uniform. No new data.
- The transplant variant (`DS_REDDIT_LINKING.md` 2026-08-17) is PARKED unless M5a's demo needs
  reinforcing; its feasibility gate is not spent now.

**Stage order becomes: M1 ∥ M1b → M2 (both families) → fork → M3 → M4 → M5a → M6.**
M5b precedes M1b by definition (it is its input); M5a can run any time after its linking pass.

### M.2 — THE FINAL TABLE: the Reddit-weighted mixed-opener benchmark (2026-08-17, author's question "can we win on the mix?")

**The trap, named:** per-scenario evaluation shows only distillation fidelity — the policy ties its
teacher in every column. A multi-teacher policy can only WIN on a **mixed population**, where every
single system must commit to one behaviour and the mix demands several. The win on the mix is
guaranteed BY CONSTRUCTION once each teacher clears the M2 screen — say so in the chapter; the
science is the screen, the mix table is the demonstration that it assembles into ONE deployable
model (N teachers otherwise = N models, no overlap handling).

**The benchmark:** ML-25M canonical test users, each assigned a scenario drawn from the REALISTIC
opener distribution (the 4,431-thread slot census — x% cold, y% genre-led, z% anchor+tone, …),
evidence constructed from the user's own fold-in to match the scenario, ground truth their own
held-out ratings. **Reddit supplies the MIX; ML supplies the TRUTH.** No linking beyond M5a's demo.

**Arms, all on the same population:** best static + fold · **cold tree + fold (THE hard baseline —
the evidence-informed incumbent, not the bare tree)** · each specialist teacher applied globally
(expected to win its segment, lose elsewhere — L2a's −0.03 at foreign states) · the forest policy.
Read-out: per-scenario columns + the MIX column; policy must match best-per-segment and win the mix.
The branching figure (claim 2) is trajectory divergence by opener, counted on this population.

**The one cell that is more than routing:** OVERLAP states — users matching two signatures, no
teacher covering the pair. Pre-registered single cell: policy vs both parent teachers at those
states. Could fail; reported either way it lands inside a winning table, per the author's omission
rule applied to CELLS, not to the table.

**Sequencing note:** the benchmark is built in M1b (scenario assignment) and scored in M6; it
replaces M6's bare trajectory count as the chapter's closing exhibit. The single TEST look is spent
HERE: forest policy vs cold-tree+fold on the MIX, pre-registered.

### M.3 — rulings from the 2026-08-17 late discussion (author Qs 1–5)

- **M5a is OPTIONAL, not load-bearing.** The spine consumes no request text (Reddit contributes the
  opener DISTRIBUTION only). M5a is the deployability exhibit — real words → recommendations — built
  only after the fork lands, cuttable without damage. Cutting it ⇒ zero linking in the chapter.
- **K.6 GOES IN, as the granularity justification, not a confession:** global tree = too coarse
  (author's objection) · per-user = unlearnable (labels are functions of held-out targets; oracle
  +0.19366 vs cohort-best +0.009664) · the forest = the finest scale measurement supports. An
  identification argument — "nulls not citable" does not bite.
- **THE CONTRIBUTION STATEMENT (thesis-key):** *we establish the conditions under which an interview
  policy should condition on volunteered information — (i) signature observable at ask time (K.6),
  (ii) cohort ≥ measurement floor (−0.724 regime), (iii) specialist beats the incumbent at its own
  states (L2a) — show the field's alternatives each violate one, and deliver a single policy covering
  the resulting mix.* The design law is the science; distillation is the delivery mechanism. The
  dream-vs-realistic opener comparison adds the price of natural openers vs optimal ones.
- **Reviewer-risk register for the mix table** (each pre-empted in the design, to be pre-registered):
  1. mix weights fixed from the census BEFORE any arm runs + uniform-mix sensitivity row;
  2. hard baseline = tree+FOLD (the evidence-informed incumbent);
  3. robustness arm: cold tree allowed to ASK the opener as a charged root question (genre study
     says a charged turn eats short-budget gains — likely strengthens us);
  4. simulator circularity is the field's lead objection — the behavioural-answer firewall is
     foregrounded in the section, not footnoted;
  5. single dataset (ML-25M) stated plainly.
- **Risk split, stated for the author:** conditional on the M2 screen, the mix win is near-arithmetic
  and robust; the open bet is the screen itself (Fable ~25–30%; somewhat higher under the coverage
  bar with two opener families). The bet is front-loaded into ~1 day of compute — that is the design
  working as intended.

### M.4 — final rulings and the writing programme (author, 2026-08-17 night). NOTHING RUNS YET.

**1. Risk 1 is SOFTENED by a banked test number — but not closed.** The genre-conditioned greedy
(`groupgreedy_genre15_TEST`) proved conditioning on an opener moves the optimal ITEM ORDER on test:
FALLBACK-200 **+0.00817 [+0.00539, +0.01097]** vs the global static. So the conditioning signal
exists — the author's inference is correct as far as STATIC goes. What stays open is only whether the
signal survives against the TREE, which already conditions on five answers of its own and may absorb
most of what an opener reveals (L2a says exactly this happened for genre: the genre orders beat
static but LOSE to the tree at their own states). **The screen's question is therefore precisely:
"does the opener carry information the tree's own five questions do not already extract?"** — a
sharper and more interesting question than "does conditioning work" (it does).

**2. Simulator circularity: our firewall is real AND largely unwritten.** Our answers are behavioural
— real held ratings for items, SEL behavioural signal for concepts — never generated by the model
being evaluated (the Jul-25 ruling; the LLM apparatus was retired Jul-22 for popularity dominance).
The field's recent neural CRS evaluations are largely circular/contaminated (positioning survey;
blind-reviewer panel says this is THE lead objection AND that our firewall is under-sold). **Audit
finding 2026-08-17: `chapterB_v2.tex` contains ONE circularity mention in 1,647 lines (line 1542,
deep-RL label circularity) — the simulator-circularity section does not exist.** → TODO W1.

**3. Qualitative trajectory panel is ADDED to the chapter's exhibits (author request):** rendered
example interviews — what the static order sounds like, what the tree asks after a like vs a
dislike, what each cluster teacher asks after its opener — for gut-check semantic coherence; distinct
trajectory counts; coarse-to-fine quantified ONLY where the notion is coherent (an adaptive tree can
be coarse-to-fine; a static order cannot — author's earlier correction stands). → TODO W2. This is
also the natural home of the branching figure (M.2).

**4. "Win by construction", resolved:** the objection is only that the mix-table victory follows
logically once each teacher passes its per-segment gate — i.e. the table's win adds no NEW
information beyond the screen. The author's counter is accepted and becomes the framing: nobody has
built this; it beats every baseline; the chapter's job is to show WHY it works here (the design law),
not to claim a new learning mechanism. The fix is presentational: the screen is the discovery, the
table is the assembly, and the MIX WEIGHTS ARE DATA-DRIVEN (the Reddit census distribution, fixed
before any arm runs — never a uniform or hand-picked mix, which would look artificial because it
would be). With data-driven weights the benchmark itself is a claimable artifact.

**5. The Reddit data pipeline is a CLAIMED CONTRIBUTION and gets its own methodology section:**
corpus (He raw, no Reddit API), username hashing + sensitive-content redaction at ingest, the
500-mention gold set, rules tier (0.971 precision / 48.6% coverage) + Opus adjudication of DEFER
(0.949; pipeline 0.960 vs He's raw 0.572), thread-level shuffle making any batch prefix an unbiased
subsample, 4,431-thread ruler (≥5 in-catalogue GT + request text), typed slot extraction ×4,431, the
~7% label-error floor stated. → TODO W3.

**6. Writing rules for the whole chapter (standing):** well sold, zero overclaims, ZERO
self-deprecating language — candour lives in tables, not adverbs; every self-critical sentence must
pass the "does the chapter read better with it in or out" test.

**WRITING TODOS (schedulable now, no compute):**
- **W1** — simulator-circularity + contamination section for chapterB: the field's evaluations are
  circular (survey evidence), ours is behavioural with the firewall; sell it, cite the Jul-25 ruling
  chain. Sonnet-suitable draft, author-reviewed.
- **W2** — qualitative trajectory panel spec + rendering harness (uses banked interviews only; no new
  runs until M-stages execute).
- **W3** — Reddit data + linker methodology section, claiming the contribution.
- **W4** — K.6 scoping-theorem paragraph (audited numbers only: +0.19366 / +0.009664).
- **W5** — the skip-redundancy multiplication printed (5.4% × 16 × +0.00167 ≈ +0.0013 < MDE/3).

---

## Writing / experiment TODOs added 2026-08-20 (Fable verdict + author direction)

**W6 — ELEVATE THE ABLATION FROM ANALYSIS TO PRESCRIPTION.** `sec:criterion` already contains the
finding ("Rank-awareness is not the fix. Measuring is.") but states it as an observation inside Act 3.
The author's own summary is the prescription: **"NDCG and NDCG only should be optimised if you want
the best ranking."** Fable's verdict makes the ablation the load-bearing result and the paper's hero,
so this belongs in the **abstract and the introduction**, phrased as guidance to anyone building an
elicitation criterion, not merely as a fact about our tree. Supporting numbers already measured:
+0.01035 vs −0.03041 / −0.03061 / −0.08627, 56/56 intervals excluding zero, and the mechanism (the
measured value of *asking* rather than a training loss over the fit) which is already written.

**W7 — RUN GreedyExtend (`golbandi2010bootstrapping`), the missing EXTERNAL STATIC anchor.**
⚠ Note what is and is not already done: Golbandi's *interview tree* IS implemented
(`src/baselines/golbandi_tree_selector.py`) and loses at every budget 1–16. **GreedyExtend — their
published static seed set, greedily minimising RMSE — is only CITED, never run.** It is the one
baseline that would answer "your strong static comparator is self-built", which is currently
answered only by self-citation. Hours of work, highest effort-to-value ratio on the list.
⚠ Do not assume the outcome: the chapter's own positioning paragraph concedes that Golbandi's tree
beat GreedyExtend *decisively* in their paper. The claim to target is **"our metric-optimised static
is stronger than the published static of that era"** — i.e. the comparator is not a straw man.
HARD RULE 2 applies: run the named prior, do not reason about it.

**W8 — FRAMING CHANGE (Fable, adopted).** Tell the paper as *"the objective is the bug"*, not
*"adaptive elicitation fails, here is a fix"*. The fix is a +0.01 effect that loses to a bank swap, so
making it the hero invites judgement by the weakest link. Spine: frontier fails on an honest ruler →
the ablation proves the objective is why → the corrected objective reveals cold conditioning is small
and bank choice dominates → the information that pays arrives **volunteered**, and answerability
orders the channels → fitting the interviewer at realistic volunteered states is where measured-objective
conditioning finally earns its keep. This turns "conditioning is small" from a buried confession into
the hinge of the argument.

**W9 — DEMOTE DISTILLATION to infrastructure.** E1's top-1 1.0000 is a replication that buys zero
gain by our own record, and M4b's seven-tree result is engineering fidelity. Present both as the
machinery that makes the generalisation claim possible, never as a contribution in their own right.

**W10 — TEST-LOOK BUDGET.** One look, one run, covering the forest, the group mix, the unseen-film
generalisation cell and the do-no-harm cell. **Retreat lines decided before the run:** if `vs cold`
shrinks to zero the claim retreats to `vs anchor`; if routing is accurate but the NDCG delta is ~0
that is M4b redux (behaviour without value) and is reported as such. Pre-register the **value**
criterion, not the routing criterion.

**W11 — STATE THE ABLATION'S HELD-FIXED DESIGN MORE PLAINLY. (Revised; the earlier version of this
item was wrong.)**

Fable proposed transplanting our measured-NDCG scalar into Golbandi's tree as the highest-value
missing experiment. I first recorded that as "already done, because our tree IS their machinery, so we
have rescued a published method". ⚠ **The author rejected that and is right.** A greedy ternary tree is
a generic object; the min-leaf, stopping rule, pool, candidate mask and answer contract are ours, not
Golbandi's. **"We repair a published method by changing one scalar" is an OVERCLAIM and must not be
written.** Attribution of the structure is a positioning move, not evidence.

★ **What actually answers the circularity objection is the held-fixed design, and it needs no
attribution.** The tree is constant across all four criteria — same code, branching, min-leaf,
stopping rule, pool, mask, answers, recommender, cohorts, budget. A constant cannot explain a
+0.01035 / −0.03061 gap, because it is equally the structure of the arms that lose. That is a
complete answer to "you designed the tree around your criterion".

**The actionable residue is small and is about emphasis, not attribution:** Fable held every number
and still reached for circularity, which suggests the one-variable-changed design is not landing on a
first pass. Say it earlier and flatter — in the abstract and the table caption — as *"only the scalar
changes"*, not as a claim about whose tree it is.
