# PAPER B — MASTER PLAN (2026-08-12, post-redirect)

Single source of truth for the redirected chapter. Supersedes `PAPER_B_REDIRECT_2026-08-12.md`
(now a pointer stub). Experimental items carry `PAPER_B_TODO.md` numbers where they exist.

AUTHOR SAID (2026-08-12): "adaptivity based on a simple tree wins hands down, and also outperforms
all baselines … all our baselines are optimised for approximated objective and that seems to be the
problem, not that cold interview answers do not differentiate."
AUTHOR SAID: "i really dislike tree for lack of scalability … do a parameterised policy, that would
avoid the weakness of subbing objective."
AUTHOR SAID: "static coarse-to-fine makes no sense whatsoever … in static nothing informs anything."
AUTHOR SAID: "policy that beats the tree ideally, as it can go to deeper states and scales better."

---

## 0. The claim

Published adaptive selectors lose to a strong fixed order **not because cold closed answers fail to
differentiate users, but because none of them optimises the quantity anyone reports** (audit: 1 of
21 rules does, and it is a static order — `BASELINE_OBJECTIVE_AUDIT.md`). Swap only the split
criterion in the field's own tree machinery to measured NDCG@10 and adaptivity wins on held-out
users **from K=4 onward on both axes**. The tree is the **existence proof**; the **parametric policy
is the method** — same exact objective, no partition, scales with depth.

**UPDATED 2026-08-12 17:45 — A2 landed; the numbers below are the clean-comparator ones**
(`docs/results/CONDITIONED_ARMS_ON_TEST.md`, `experiments/battery/ndcgtree16test_policyB.json`).
"All ten CIs exclude zero" is **RETRACTED**: K=1 is now exactly 0.0 by construction (the contract
gate), seven of the eight K≥2 intervals exclude zero in the tree's favour, and **tail at K=2
excludes zero AGAINST the tree** (−0.00262 [−0.00481,−0.00051]).

**UPDATED 2026-08-12 21:50 — A3 and A4 landed; the whole static family is now policy-B-selected.**

Size qualifiers that travel with every statement of the claim:
- Conditioning gain on test at K=16 is **+0.01035 full / +0.00705 tail** against the rebuilt policy-B
  comparator (was +0.01371/+0.02092 against the policy-A-selected one — a quarter of the full and
  **two thirds of the tail** advantage was comparator weakness). That is **~10% of the interview's own
  lift**, and it is **smaller** than the gain from changing the question BANK with no conditioning,
  which is now clean: **+0.01430 full [+0.01124,+0.01739] / +0.02199 tail at EQUAL budget** (16 vs 16,
  both orders policy-B-selected). The old +0.01544 carried two defects — a policy-A items order and a
  15-vs-16 budget mismatch — and the conclusion survives fixing both. "Adaptivity pays", never
  "adaptivity is large".
- "Tail does not shrink" is **RETRACTED** too: val→test is 2.70× on full and 1.50× on tail. The
  asymmetry survives in direction only. The genre arm, re-differenced onto one ruler, shrinks 1.80×
  on full and **1.02× on tail** — the same asymmetry, stronger, on a different partition.
- The genre arm against its own rebuilt comparator is **+0.00862 full [+0.00595,+0.01131] /
  +0.00591 tail** (fallback-200) and **+0.01015 / +0.00719** (all-groups) at K=15. Full went up, tail
  went down: the rebuilt items order is stronger on the selection cohort but marginally weaker on
  test's full metric and clearly stronger on its tail. K=2 and K=4 now separate (they were nulls);
  **K=8 tail stops separating**. Drama (55% of users) is still a null with its sign flipped to +.
- Channel comparison at 16 on test, rebuilt orders: **concepts − items is a dead heat on full**
  (−0.00003 [−0.00315,+0.00310]) and **+0.01436 on tail** for concepts. Concepts lead full at every
  budget below 16 and items only catch up at 16 — answerability ordering the channels, measured on
  held-out users. The chapter's ".2473/.1154 vs .2449/.0967, concepts strongest" sentence is on
  validation with policy-A orders and must be re-cut.
- 4/15 leaves negative on test (32.8% of cohort); ~14% of users answer nothing and sit at 0.087.

Historical frame (verified, `rating_elicitation_family_deep.md` §2.4–2.5): Golbandi WSDM'11 beat his
own metric-aligned greedy static (GreedyExtend, RMSE) because his variance split essentially WAS the
era's reported metric. The field moved the metric to ranking and kept the rating-era criteria;
nobody rebuilt the greedy comparator. We change the criterion back to the metric and adaptivity wins
again.

## 1. Chapter structure

- **Act 1 — the puzzle.** Published adaptive selectors vs a static order built by maximising the
  reported metric: all lose, every budget, held-out. Genre split closes the act as the first crack.
- **Act 2 — the diagnosis: objective mismatch.** (a) the objective audit; (b) PV-quartile trend —
  deficit widens where personalising pays most; (c) the direct test, TODO 1.11 rank correlation.
  Placement/dispersion material MOVES OUT (to Act 5, as a reach diagnostic).
- **Act 3 — the demonstration.** The NDCG tree: design, val→test shrinkage (**2.70× full / 1.50×
  tail** against the clean comparator — the "tail does not shrink" 1.03× figure was a comparator
  artifact and is RETRACTED; the asymmetry survives in direction only), partition generalises,
  negative leaves reported in full (**6/15, 39.3% of the cohort**, on the clean comparator),
  bank-vs-conditioning decomposition (**the bank exceeds conditioning: +0.01430 [+0.01124,+0.01739]
  at equal budget vs +0.01035** — A3/A4 done, both sides policy-B-selected). Hu & Yu objective
  ablation lives here (nearest rival).
- **Act 4 — REFRAMED (author correction 2026-08-12): "answerability orders the channels", NOT
  "coarse-to-fine curriculum".** A static order is not a strategy — nothing informs anything. The
  measured fact: concepts front-load because cold users can answer them (three independent
  measurements: greedy order composition c c c i…, per-state criterion ×12 concept advantage cold
  flipping to items when placed, tree root = concept while the best item split dumps 70% into
  no-answer). Any ROUTING claim ("broad answers determine which finer questions follow") is a claim
  about the TREE and is **now SETTLED NEGATIVE by 1.20** (`TREE_ROUTING_SPECIALISATION.md`): the tree
  **partitions and diverges but does not demonstrably route.** Licensed: 80/105 leaf pairs share no
  free pick, median 14/15 positions differ, overlap 3.5× below matched chance. NOT licensed:
  answer-congruence is a null — matched-availability control p=0.82, branch-label shuffle p=0.61.
  **No routing claim goes in the chapter.**
- **Act 5 — reach.** ~14% answer nothing; answering at all is worth ~+0.25 vs +0.048 for the best
  question choice. Open opener as the only instrument that can touch these users; Reddit corpus,
  DiD with init-aware static comparator, withheld-slot replay. CUTTABLE if scope demands — the core
  chapter (1–4) stands without it.
- **End-state method (threads Acts 3→5):** the parametric policy (1.16), pitched to beat the tree
  by reaching deeper states.

## 2. Workstream A — comparator hygiene: every greedy/tree number onto one clean ruler

The contamination: the banked greedy orders were SELECTED under candidate-mask policy A and are
scored under policy B (`banked-global-greedy-is-a-policy-A-artifact`). Measured size: the tree's
K=1 lead of +0.00246, where conditioning is impossible.

- **A1 (1.17, DONE 17:25).** Static COMBINED order rebuilt under policy B on validation
  (`greedy_static_policyB_combined.json`, 16 picks, 2,167 evals, resumed from the tree root's CELF
  heap). Shares 4/16 questions and **0/16 positions** with the order it replaces — the contamination
  was never an offset, it moved the argmax at every position.
- **A2 (DONE 17:43).** Tree re-measured on val AND test against A1 (`ndcgtree16test_policyB.json`).
  **Contract gate PASSED: K=1 difference exactly 0.0 on all 20,000 users, both axes.** Reuse gate
  (bit-identical per-user vectors + routed row ids) and banked-headline assert (every policy-A number
  reproduced to 0.0e+00) both passed. New K=16 test: +0.01035 full / +0.00705 tail.
- **A3 (DONE 19:25).** `greedy_static_policyB_items.json`, 16 picks, 839 evals, **52.7 min** (est. 2 h).
  Shares 7/16 questions and 2/16 positions with the policy-A order. Genre arm re-measured against it
  (`groupgreedy_genre15_TEST_policyB.json`, stage-2 only) and its validation side re-differenced onto
  the same comparator (`groupgreedy_genre15_VAL_policyB.json`, banked assert bit-identical in all 90
  cells). Both primary intervals still exclude zero; see §0 for what moved.
- **A4 (DONE 21:44).** `greedy_static_policyB_concepts.json`, 16 picks, 1,546 evals, **134.3 min**
  (est. 3.5 h). Shares 3/16 questions and 1/16 positions. Act 4's channel table re-scored on test
  (`channel_orders_policyB_test.json`), which also carries the clean bank decomposition and a **banked
  assert that PASSED**: the combined comparator re-scored from scratch reproduces the tree run's
  per-user test vectors bit-identically at all five budgets on both axes.
  *Cost lesson worth keeping: identical pool size and budget, 339 lazy re-evaluations for items vs
  1,046 for concepts. Concept marginal gains sit close together, so CELF's bounds go stale far more
  often — search cost is a property of the BANK, not of the budget.*
- **A5.** Budget consistency: 16 everywhere (genre arm gets its 16th pick from the persisted heaps,
  now cheap after the `heap_path` fix; the rebuilt items order already carries its 16th pick, i135).
- **A6 (REDO in tex).** Tail-objective greedy sequence — required before any tail-optimised claim.
- **A7 (1.19).** CELF signature debt.
- **A8.** Clean confirmatory tree evaluation logged in the exploratory/confirmatory ledger — the
  one test shot so far had a mid-run 16th-step extension and a comparator swap (exploratory).
  The TREE ITSELF is not regrown (author ruling: "we are NOT changing this tree").
- **A9.** Swap Act 1's greedy row for rebuilt numbers; assert banked numbers wherever a new arm
  nests in an old one (free contract check).
- **A10 (1.18/1.18b).** The min-leaf/answerability interaction: the 200-user floor penalises
  answerable questions (a question everyone answers splits three ways and hits the floor sooner).
  Quantify before any depth conclusion is written.

## 3. Workstream B — Act 1 baseline additions

- **B1 (1.12, promoted MANDATORY).** Hu & Yu 2013: (i) snap on their protocol (MAP, 10 sampled
  negatives, FixQ) per HARD RULE 2; (ii) **objective ablation in OUR tree harness** — same code,
  same answers, three split criteria: variance (Golbandi) / pairwise rank loss (Hu & Yu) / measured
  NDCG@10 (ours). (c)>(b)>(a) makes the mechanism airtight; (b)≈(c) narrows novelty to "the exact
  reported metric", field claim survives. Bib key `hu2013interview`, check collision first.
- **B2 (1.10).** RBMF snap + port (static; threatens the comparator, not the adaptive claim).
- **B3 (task 9).** EDDI full Eq. 9 re-run.
- **B4 (task 25).** ConUCB k=8: resolve or report UNINFORMATIVE with the b(t) schedule stated.
- **B5 (task 10).** CLiME width-rule run — make the degeneration citable.
- **B6 (1.21, NEW).** Deep-RL CRS family (CRM/EAR/SCPR/UNICORN) as AUDIT ROWS + positioning text:
  adaptive, metric-as-RL-reward, but simulator-circular (our firewall) and attribute-loop protocol.
  Cite, don't run. Our 1.16 = the non-circular version.

## 4. Workstream C — Act 2 exhibits

- **C1 (1.11, cheap, run FIRST).** Surrogate-vs-NDCG rank correlation per method per state. Turns
  "1 of 21" from description into measurement; gates 1.15.
- **C2 (writing).** Rewrite Act 2 around objective mismatch; move placement/dispersion to Act 5.

## 5. Workstream D — Act 4 reframe

- **D1 (writing).** Kill "curriculum"/"coarse-to-fine" as strategy language for static orders;
  reframe as answerability ordering the channels.
- **D2 (1.20, NEW, cheap — banked data only).** Routing-specialisation measurement on the tree:
  (i) pairwise overlap of leaf greedy tails across leaves (do different branches ask different
  questions?); (ii) congruence of each branch's questions with that branch's answers (e.g. does the
  like-branch of the root concept get items/concepts related to it?) vs a matched-divisiveness
  random control. This replaces the vague "face validity" PLAN with a quantitative test, and it is
  the ONLY thing that can support a routing claim.

## 6. Workstream E — the policy that beats the tree (1.16, author priority)

Why the June attempts sat below entropy (all three root causes MEASURED then, all fixed now):

| root cause (proven Jun 2026) | fix now |
|---|---|
| Teacher unlearnable (clairvoyant = privileged, top-1 14%) or wrong-objective (greedy infogain, near-static); only realizable-teacher distillation worked | **Teacher = the NDCG tree**: realizable (observable answers only), metric-aligned, and PROVEN to win held-out — this teacher did not exist in June |
| Reward SNR: routing signal = 2.1% of cross-user reward variance ⇒ PG collapses to best fixed order | Imitation FIRST (supervised, no variance); RL only as RLOO finetune with paired baselines on top of a working imitator |
| Wrong reward (coverage/entropy degraded cos(u,u*)); flat-MLP actor lacks per-candidate bias | Exact held-out NDCG@10 as reward (old P1 already showed this fixes decoherence); equivariant per-item actor |

Old magnitudes are demoted (retired stack); the mechanisms are stack-independent.

Staged design, gates before each promotion:
- **E1 — imitation of the tree.** Input: observable state only (tower latent z + asked/answered
  history). Labels: the tree's cohort-level exact-NDCG argmax at each node, PLUS the leaf greedy
  tails (16 labels per validation user). Architecture: equivariant per-candidate scorer over the
  1,000-candidate pool. GATE: reproduce the tree's own test gain within CI (policy ≈ tree when
  restricted to tree-visited states). No surrogate anywhere: the label IS the metric argmax.
- **E2 — exceed the tree where it starves.** The payoff case: states the tree cannot condition on
  (post-depth-5, fragmented cohorts). Generate labels at deeper states with the kernel-weighted
  cohort estimate (1.14 machinery — soft partition, sample never shrinks) and train the SAME
  policy on them. GATE: beat the tree on test, paired CI.
- **E3 — RLOO finetune, exact NDCG reward,** only if E2's supervised ceiling looks binding.
  Firewall non-negotiable: train/val states only, TEST never seen, held-out gap reported.
  Prior burn: realizable RL ≈ 1/5 of oracle once held out — an unfalsified training win is nothing.
- **E4 — non-parametric middle rungs** (cheap sanity anchors): 1.14 kernel-weighted local greedy;
  1.15 criterion-argmax, gated on C1.

## 7. Workstream F — Reddit / reach (Act 5)

- **F1 (writing).** Rescope the DiD hypothesis from "starved openers" (dead — the free genre fact
  lost to the charged closed interview) to REACH: the open opener is the only instrument that can
  touch the ~14% who answer nothing.
- **F2.** The init-aware-but-fixed static comparator requirement STANDS (or we commit the
  comparator error this chapter criticises).
- **F3.** Withheld-slot replay (both variants) — external-validity endpoint.
- **F4.** "What does an adaptive interview add once a real request is given?" — newly runnable:
  place users with the request, then run the tree/policy after the opener. This is the July
  survivor question and now has a working adaptive method to answer it with.
- Whole workstream is the designated CUT if scope demands.

## 8. Execution order

1. **DONE:** A1 → A2 (K=1 gate) → A3 → genre re-measure → A4 → channel table + bank decomposition.
   **Next:** A9 (swap Act 1's greedy row for the rebuilt numbers). In parallel: C1 (cheap),
   D2 (done), B1(i) snap.
2. **Then:** A5, B1(ii) ablation, B2, A6, A10.
3. **Then:** E1 → E2 (→ E3 if needed), with F and writing (C2, D1, F1) interleaved.
4. Confirmatory pass A8 + ledger before any abstract number is final.

## 9. Not claimed / guarded

- Never "adaptivity is large" — size qualifiers of §0 travel everywhere.
- No routing/curriculum claim unless D2 separates from its control.
- No tail-optimised claim before A6.
- Genre numbers remain UPPER BOUNDS — the free genre fact costs no turn — and **all 17 groups still
  shrink val→test on ONE ruler** (1.80× cohort-weighted on full), with Drama still a null.
- Nulls are not citable (author ruling); Hu & Yu ablation outcome (b)≈(c) is reported as a
  narrowing, not marketed.
