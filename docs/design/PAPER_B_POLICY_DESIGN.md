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
DBU-style blend targets (arXiv:2602.08616) become relevant only if a pure regression head is ever
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

## Cross-references

`PAPER_B_PLAN_2026-08-12.md` §6 (workstream E) · `PAPER_B_TODO.md` 1.13/1.14/1.15/1.16/1.22 ·
`docs/results/CONDITIONED_ARMS_ON_TEST.md` · `docs/results/TREE_DEPTH_AND_MINLEAF.md` ·
`docs/results/HU_YU_2013_REPLICATION_AND_ABLATION.md` §B ·
`docs/results/TREE_ROUTING_SPECIALISATION.md` · `docs/results/POSITION_VS_MODERN_CRS.md` Part II ·
memory `clairvoyant-imitation-breakthrough`, `night-adaptive-edge-investigation`,
`policy-ladder-ndcg-reward`, `banked-global-greedy-is-a-policy-A-artifact`,
`surprise-means-debug-the-harness`.
