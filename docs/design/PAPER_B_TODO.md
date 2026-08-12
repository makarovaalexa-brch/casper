# Paper B — the single TODO (state of the whole paper, one page)

Maintained 2026-08-10 (TODO 1.2 closed). This is the one place to see the paper's state; per-stream task lists and the
per-act design sheets hang off it. Draft: `new_chapters/chapterB_v2/chapterB_v2.tex` (14 pp, builds
clean, 8 red REDO + 7 blue PLAN markers so nothing unfinished is invisible).

---

## THE CLAIM, as the evidence now supports it

Adaptive question selection does not pay in cold-start interviews **and the reason is not the one we
first proposed.** It is not that there is nothing to adapt to. Where the criterion says personalising
is worth the most, the adaptive methods are furthest behind — measured on one population, 38 of 48
trend cells separated, all negative, shuffle control clean. So the finding is *methods cannot find
the structure that exists*, which is stronger than a flat null and different from what Acts 1–2 said
a week ago.

**And it is not the action space either (settled 2026-08-10, TODO 1.2).** Handing the selector the
concept bank — the channel the criterion says is optimal for ~100% of users — does not fix it:
77 of 96 trend cells separate, 75 negative, shuffle 1/96, and the concept askers fail *harder* than
the item askers. Two guard arms ask concepts on 100% of turns at every budget, so no null rides on an
unfired schedule. The claim's final form: **adaptive selection fails to convert personalisation value
even when the optimal action type is available to it.** The best use of the concept bank is a FIXED
ORDER — `fixedorder:greedy_concepts` is the strongest arm we have at every budget on both axes
(0.2362/0.1022 at k=8). → `docs/results/ACT2_CONCEPT_BANK_PV.md`.

---

## ACT 1 — the battery. **EXPERIMENTALLY COMPLETE.**

21 selectors across six families, one contract (certified pair, policy-B pool, NO-FLOOR, behavioural
answers, 10,000 test users), paired intervals throughout. Six native snaps: EDDI **FAILED** (reported
as failed), DRE / Golbandi / IGCN **PASSED**, DPE setup passed, Greedy SLIM **PARTIAL + the ablation
finding**. Floored arm done. Oracle ceiling measured.

| # | item | state |
|---|---|---|
| 1.1 | Revise the Act-1 conditional — **the tree's crossover is a PV=0 effect** (see Act 2) | **OPEN, blocking** |
| 1.2 | Concept-bank test: give an adaptive selector the concept bank, see if it can then convert PV | **DONE 2026-08-10 — NO. 77/96 trends separate, 75 negative, shuffle 1/96; concept askers fail HARDER. Mechanism hypothesis RETRACTED in `ACT2_PV_QUARTILES.md`. → `docs/results/ACT2_CONCEPT_BANK_PV.md`** |
| 1.2b | `_pick_mixed` extension of `glm_ucb` to the unified action space — now the worth-building follow-up, since the question has narrowed to "does ANY acquisition rule convert PV?" | open, next |
| 1.3 | EDDI full Eq. 9 (both terms); current row is relabelled, not fixed | open |
| 1.4 | CLiME width-rule run (measured negative: zero concept questions under a pure uncertainty trigger) | open |
| 1.5 | Re-run DRE + DPE with in-run `fixedorder:` comparators (currently cross-run, admitted on a 0.0e+00 control) | open, cosmetic |
| 1.6 | ConUCB Eq. 8 caching fix (value-free but uncached; 451× redundant) | open, low |
| **1.7** | **★ GROUP-CONDITIONED GREEDY (author proposal 2026-08-10).** Partition users, run the SAME greedy search inside each group, route at test time. Like-for-like: same objective, same optimiser, same val/test discipline, and static is NESTED (groups may all pick the same order), so the question reduces to whether the conditioning GENERALISES from validation to test. **A0 ceiling** = group by true taste (privileged; answers "is there anything to condition on at all"). **A1 deployable** = group by the answer to question 1. | **OPEN — the decisive experiment** |
| **1.8** | **Cost smoke for 1.7**: one 2,000-user `greedy_certified` run on the items-only pool, ~40 min, to settle whether per-eval cost scales down with cohort size. Recorded build = 723.6 min / 3,173 evals / 13.543 s per eval at 10,000 users; items-only alone 852 evals ≈ 3.2 h. Bounds for G groups: 3.2 h (optimistic) to G×3.2 h (pessimistic). **Run this BEFORE committing to 1.7's full design.** | open, cheap, blocking 1.7 |
| **1.9** | Golbandi with OUR objective = the recursive version of 1.7 (split at every turn, not once). **Structural rewrite, not a local change** — see `docs/results/BASELINE_OBJECTIVE_AUDIT.md`: no held-out targets in the growth contract, the Eq. 3/4 subtraction trick dies, and the stopping test inverts. Do NOT start before 1.7. | open, expensive |
| — | **★ AUDIT LANDED**: `docs/results/BASELINE_OBJECTIVE_AUDIT.md`. **1 of 21 distinct selection rules optimises held-out ranking quality — ours.** 12 of 23 rows are VALUE-FREE (adapt to which questions were answerable, never to what was answered). `conlinucb_bs_static` joins `bald`/`aopt` as structurally non-adaptive. This reframes Act 1 and belongs in the paper as a table. | recorded |
| — | fMF | **RULED OUT by the author** |
| **1.10** | **★ RBMF (maxvol selector) — UN-DROPPED 2026-08-10.** The reported-metric audit puts it in the *fair-fight* half: Liu et al., RecSys 2011 reported **P@10, MAP, AUC as its primary result**, claimed an improvement over MostPopular (.501 vs .432), and evaluated at **k = 5, 10, 20** — straddling our budgets exactly. Dropping a baseline whose authors claimed the very thing we measure, at our budgets, is the kind of gap a reviewer finds. Re-run the snap (was task "SNAP 2"), then port. | **OPEN — reinstated** |
| **1.11** | **★★ SURROGATE-vs-NDCG RANK CORRELATION — the cheap direct test of H2.** At each of a set of matched user states, rank all 500 candidate questions two ways: (a) by each method's OWN acquisition score (entropy₀'s entropy, BALD's posterior variance, Golbandi's variance reduction, GLM-UCB's index, ConUCB Eq. 8, …), and (b) by the ACTUAL realised NDCG gain of asking it. Report the rank correlation per method across states. **This measures the surrogate→ranking transfer directly, with no new arm, no adaptivity and no confound.** Machinery already exists (leg-1 criterion + the banked Act-2 states). Turns "1 of 21 optimises the metric" from a description of the literature into a measurement of why it matters. | **OPEN — cheap, high value** |
| **1.12** | **Hu & Yu (RecSys 2013) faithful replication + improvement.** Nearest prior work; PDF now in `external_literature/papers/`, analysis in `findings/hu_yu_2013_interview_process_learning.md`. Their setup: interview tree, split by a general loss incl. **pairwise rank loss**, **Taylor expansion (gradient + Hessian)** to avoid re-solving, trained on the **full train cohort**, evaluated with **MAP on 10 sampled negatives per positive**, static comparator = **top-k popular** (`FixQ`). Do the snap on their protocol first, then port to our ruler. Their approximation is the published route to the scale our exact objective cannot reach. | open, next after the tree |
| **1.13** | **Differentiable-surrogate search (scalability).** Linearise the objective in the tower's latent `z` — compute ∂L/∂z once per node, estimate Δz per candidate answer from the encoder, per-candidate cost becomes a dot product instead of a full 18,359-item decode. Candidate objectives: the tower's own multinomial log-likelihood (its training loss, fully differentiable) or a soft-NDCG (ApproxNDCG / NeuralNDCG / SoftRank). **Use it as a SHORTLIST filter only — re-rank the top ~20 with the exact metric — so the arm still selects on exact NDCG and the H2 claim survives.** Report the filter's recall against one full exact node. | open, unblocks scale |
| | **★ THREE NON-TREE ADAPTIVE ARMS (author, 2026-08-12).** Trees break down with depth because they use HARD partitions with PER-NODE fitting — every split divides the data that estimates the next question. All three below avoid that, either by fitting nothing or by sharing parameters across states, so they get *stronger* as user state accumulates. | |
| **1.14** | **Kernel-weighted local greedy.** At a state, weight ALL validation users by similarity to that state and maximise the weighted objective — soft partitioning, so every user contributes to every state and the effective sample never shrinks. Depth sharpens localisation instead of starving the estimate. Tractable because the population occupies few distinct states (11 / 112 / 1,463 / 5,767 at k=1/2/4/8, Act 2), so compute per STATE not per user. Reuses `genv_mean`; kernel weights enter the mean. Nothing is fitted, so nothing overfits. | open |
| **1.15** | **Criterion-argmax** — at each turn ask the question maximising expected NDCG gain under the tower's own predictive at that user's state. Zero parameters, so no fragmentation and no val→test shrinkage; accumulated state strictly improves the estimate. **Gate on 1.11**: the rank-correlation test measures precisely whether this criterion tracks realised NDCG gain, so run it first and only build this if the correlation holds. A failure here is otherwise uninterpretable ("just another weak method", HARD RULE 6). | open, gated on 1.11 |
| **1.16** | **★★ PARAMETRIC POLICY — author explicitly wants this (2026-08-12).** Learn a function from accumulated state (the tower's own latent `z` plus answer history) to the next question, sharing parameters across all states AND all depths. This is the textbook answer to "grows stronger with depth": every user's every turn trains the same weights, so depth adds signal instead of dividing data. **Scope note:** this is a DISCRETE question selector and is distinct from the continuous-ANSWER policy the author held out of Paper B for Paper C — confirm framing before writing. **Staged design, safe first:** (a) **imitation** — supervised, train the policy to reproduce the exact-NDCG-best question computed offline on validation states; stable, no reward hacking, and this project has prior art (`clairvoyant-imitation-breakthrough`). (b) **policy gradient** with NDCG reward only after (a) works. **Firewall, non-negotiable:** train on train/validation states only, TEST never seen; report the held-out gap explicitly. Prior burn to respect — the night adaptive-edge investigation found held-out evaluation exposes RL gaming with realisable gain ~1/5 of oracle, so an unfalsified training-set win means nothing here. | **OPEN — author priority** |
| **1.17** | **★★ REBUILD THE STATIC GREEDY ORDERS UNDER POLICY B — scheduled for after the tree (author, 2026-08-12).** **Verified, not inferred:** `greedy_static_i26c.json` records `intercept_eval` = 0.13139 / 0.02343, which `MASK_POLICY_B_MIGRATION.md` lists verbatim as the **policy-A** intercept (policy B is 0.1586 / 0.0282). So the greedy *search* ran under policy A. Its sequences are re-scored under policy B everywhere downstream, so all reported numbers are on one ruler and **Act 1 is unaffected — an order optimised for policy A and scored under B is handicapped, not flattered.** The problem is confined to comparisons where a conditioned arm was *selected* under B against a comparator *selected* under A: the **genre result** (items pool) and the **NDCG tree** (combined pool). Author has removed the in-chapter disclosure on the understanding that the rebuild closes it. **Rebuild all three arms, not just items** — items ≈852 evals ≈2 h (fixes genre), combined ≈1,971 evals ≈4.7 h (the tree's comparator), concepts ≈1,486 evals ≈3.5 h (quoted in-chapter as the strongest arm, and any future conditioned concept arm needs it). ~10 h total at the measured 8.53 s/eval; one env build serves all three. **Order: combined → items → concepts.** Then re-run the genre comparison and the tree measurement against the rebuilt orders. **Act 1's greedy numbers get SWAPPED for the rebuilt ones** — the sequence changes, so the table changes; the *outcome* does not, because an order optimised under A and scored under B is handicapped, i.e. the currently-reported version is conservative. **Agreed sequence (author, 2026-08-12): (1) tree finishes → (2) rebuild the whole static family → (3) re-measure the tree and re-run the genre experiment against the rebuilt orders → (4) swap Act 1's greedy row.** ★ **THE CONFOUND IS NOW MEASURED, 2026-08-12 02:45.** The tree's root argmax over the full 1,000-candidate combined pool under policy B is **concept 114 at +0.02049**; the banked policy-A comparator opens with **concept 630, measured at +0.019563 under policy B**. So the policy-A order is suboptimal on its own first pick by **≈ +0.0009 full**, and concept 114 appears nowhere in the comparator's 15 questions. That offset is *selection-policy*, not conditioning, and it sits inside every K≥2 number against the `combined` arm — small against the genre result's +0.008, but real and compounding. ★ **AND THE COMBINED REBUILD IS NOW NEARLY FREE:** the root's completed CELF heap (1,000 candidates × 10,000 users) **is** the first pick of a policy-B static global greedy, so extending it to 15 picks costs only the lazy re-evaluations — **~45–85 min instead of ~4.7 h**. Must copy `node_root.json` to its own output path first (HARD RULE 11(e)). Revised cost for the family: combined ~1 h + items ~2 h + concepts ~3.5 h ≈ **6.5 h, not 10**. | **OPEN — scheduled after the tree** |
| — | RMVA, PERE, GreedyExtend, MeLU, PEBOL, a-stratification, SEL-BALD | dropped, reasons recorded |

## ACT 2 — the mechanism. **MEASURED, AND IT CHANGED THE STORY.**

Dispersion collapse measured two independent ways: effective distinct optimal questions
(1.00 → 2.88 → 5.04 on ML; 95.13 on real requests) **and** distinct states (11 / 112 / 1,463 / 5,767
at k=1/2/4/8). Quartile stratification done on 10,000 users.

| # | item | state |
|---|---|---|
| 2.1 | Fold the quartile result into the chapter — it changes Act 2's claim **and** Act 1's | **OPEN, blocking** |
| 2.2 | State the confound in-chapter: PV ↔ answer rate ρ=+0.76 at k=2; clean at k=4/8 | open |
| 2.3 | Answerability-prediction selector → decompose the oracle gap into predictable vs unknowable | open, high value |
| 2.4 | Test "popularity is the answerability prior": correlate each selector's answer rate against the popularity of what it asks | open, cheap |

## ACT 3 — the intervention (open the opener). **NOT STARTED.**

Emulate an open opener on ML-25M (favourite film / favourite genre / distinctiveness-framed), with
the question distribution mined from the 4,431 real requests; users, answers and scoring stay
MovieLens. Difference-in-differences against the closed opener. 9-point threat model already written
(`PAPER_B_DESIGN.md`), including the dispersion-reproduction validation gate that must pass first.

## ACT 4 — coarse-to-fine. **PARTLY MEASURED, NEEDS REDIRECTING.**

Greedy emits `c c c i c c i i i i c c i i i i` with no curriculum specified; the channel flips with
placement. ~~**New lead from Act 2:** ... a concrete hypothesis for why they cannot convert PV~~ —
**TESTED AND FALSE (1.2, 2026-08-10).** Concept-asking selectors do not convert PV either, and fail
harder. What survives is the *bank*, not the *selection*: the fixed concept order is the strongest arm
we have and is answered ~3× as often as an item order. See 4.4.

| # | item | state |
|---|---|---|
| 4.1 | `--objective tail` greedy sequence (required before any tail-optimised claim) | open |
| 4.2 | Face-validity exhibits: hand-picked trajectories + genre-congruence vs matched-divisiveness control | open |
| 4.3 | `--conc_bank polar` in the greedy suite (needs the `H(bands\|answered)` estimator ported) | open |
| 4.4 | **NEW from 1.2**: the concept channel's value is real but STATIC — `greedy_concepts` beats `greedy_items` and `entropy0` at every budget on both axes, and is answered on 79–90% of turns vs 30–44%. Act 4's coarse-to-fine story should be built on the fixed concept order, not on adaptive concept selection | open, high value |

## ACT 5 — Reddit external validity. **ARTIFACTS BUILT, CELLS NOT RUN.**

4,431 linked threads, slot parse done (94% genuine requests, median 5 slots), linking precision
0.960 vs a 500-mention gold standard. Withheld-slot replay (both variants) not run.

---

## CROSS-CUTTING

* **Writing**: related-work section (survey exists in `external_literature/`, bib entries not
  written); the concept-fold result must be stated narrowly (it compares *initialisation recipes*,
  not our trained embedding — author correction 2026-08-10).
* **Protocol paragraph**: Greedy SLIM is now a published case study — a real adaptive-degradation
  curve manufactured by target-set shrinkage, which our policy B removes. Written in.
* **Bib hazard**: adding a key that collides with `references.bib` once aborted Paper A's whole
  bibliography (366 citations → `[?]`). Collision-check before adding.
* **Reproducibility debt**: `glm_ucb`/`thompson` ride an iALS basis that is not bit-reproducible
  across BLAS thread counts; the basis is now banked, but pre-fix cells must not be compared to
  post-fix cells at 4 dp.

* **Harness debt — `greedy_certified` should persist its CELF heap alongside the picks.** Right now
  only the picked sequence and gains are written, so *extending a stored greedy order by one step*
  costs a full cold seeding pass (~485 evals for a 500-item pool) instead of the ~55 it costs
  in-process, where the lazy heap and its staleness counters are still warm. On 2026-08-11 that
  turned "add a 16th item to four banked 15-item orders" into a ~3 h job and forced the genre-split
  endpoint down to K=15. Cheap to fix — bank `[(neg_gain, q, upd)]` plus `cur` into the sidecar next
  to `gains` — and it makes **every banked order extendable**, which every group-conditioned or
  per-cohort design will want.

## ORDER I WOULD WORK IN

1. **1.1 + 2.1** — the chapter currently states a conditional the data no longer supports. Writing, no compute.
2. ~~**1.2** — the concept-bank test.~~ **DONE 2026-08-10; the hypothesis is dead and the retraction is made.** Successor is **1.2b** (`_pick_mixed`), and it is now optional rather than blocking: four published concept-askers already failed, so a fifth negative adds little, while 4.4 (build on the fixed concept order) is a constructive lead the same run produced.
3. **2.3 / 2.4** — the answerability decomposition; cheap and it closes the chapter's own loop.
4. **4.1, 4.2** — tail-objective greedy and face validity.
5. **Act 3** — the intervention, once the mechanism is settled.
