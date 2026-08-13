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

**CONFIRMED 2026-08-12 by the modern-CRS positioning survey (B6/1.21) — and this is now the chapter's
LEAD concession, not a background note.** GreedyExtend is the *only* metric-optimised static comparator
found anywhere in the literature, so **"nobody ever compared adaptivity against one" is FALSE and must
never be written**; the claim is scoped to the period after the reported metric moved to truncated top-k
ranking. The deep-RL CRS family was checked baseline-table by baseline-table and has none; the LLM
elicitation line has none; the conversational-bandit line has none. ⚠ *GreedyExtend's own CIKM'10 full
text is [A/S] — closed on every route tried — so this concession rests on a reconstruction from
WSDM'11 §3.1.1 [V], patent US8510248B2 [V] and RMVA's description [V], plus WSDM'11 Fig. 3 [V]. Get the
PDF through institutional access before it goes to print.*

## 1. Chapter structure

- **Act 1 — the puzzle.** Published adaptive selectors vs a static order built by maximising the
  reported metric: all lose, every budget, held-out. Genre split closes the act as the first crack.
- **Act 2 — the diagnosis: objective mismatch.** (a) the objective audit; (b) the criterion-regret
  exhibit (22 nodes × ~1,000 candidates of exact measured NDCG gains) plus the split-criterion
  ablation. **(c) PV-QUARTILE TREND: ✘ CUT FROM PAPER B (author, 2026-08-13)** — it carried a
  comparator debt (cells differenced against policy-A-selected orders) and Act 2's argument does not
  need it. **`docs/results/ACT2_PV_QUARTILES.md` and `ACT2_CONCEPT_BANK_PV.md` STAY ON FILE** — the
  measurements are not retracted, they are simply not in this paper; the Act-1 stratified paragraph
  that cross-referenced them is removed too. (d) TODO 1.11 rank correlation — to be run for Paper B,
  but NOT NOW (see §4). Placement/dispersion material MOVES OUT (to Act 5, as a reach diagnostic).
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
- **A6. ✂ CUT FROM PAPER B (author, 2026-08-13).** Tail-objective greedy sequence. The chapter now
  states plainly, where the greedy orders are introduced, that every order maximises **full** NDCG@10
  and that tail is a **secondary outcome** — which is what we did. No claim in the chapter is
  conditioned on a tail-objective search, so the run is not owed. Do not re-add the `\REDO`.
- **A7 (1.19).** CELF signature debt.
- **A8. ✂ CUT FROM PAPER B — author ruling 2026-08-13: no confirmatory re-run is owed, and the
  reason it is sound is recorded here rather than in the chapter.** The tree was grown on the
  **validation** cohort and was never modified after any test look ("we are NOT changing this tree").
  Two things changed between the test looks and **neither was a selection decision taken on the test
  cohort**: (i) the sixteenth question, added for **budget consistency** with every other arm in the
  chapter (16 everywhere, A5), not because a K=15 test result disappointed; (ii) the comparator
  rebuild (A2/A3/A4), which fixed a **provenance bug** — the static orders had been *selected* under
  the retired policy-A mask — found by the `ndcgtree` sanity gate and the policy-A provenance audit,
  i.e. discovered from an artifact field, independently of any test outcome, and applied uniformly to
  every arm in the chapter including ones it hurt (it cut the tree's own tail gain from +0.02092 to
  +0.00705 and retracted a separated interval in our favour). A fix that costs the proponent arm two
  thirds of its headline is not test-set selection. Nothing about the tree's structure, criterion,
  pool, depth or stopping rule was chosen with knowledge of a test number.
- **A9 (MEASURED 2026-08-12 22:37; tex merge outstanding).** `docs/results/BATTERY_TABLE_POLICYB_ROWS.md`
  carries drop-in replacements for **both** `tab:battery` and `tab:greedy` (the latter's `emergent
  order` string changes to `c c i c c i i c c c c i i i c c`, 10 of 16 concepts), the per-pick gains,
  and every stale prose line. Banked-number assert passed to 5 dp on all three arms; two new admission
  gates (cross-family per-user identity at 0.0e+00, cross-run entropy0 at 0.0e+00) license pairing the
  instrument-side static rows against the arm-N selector rows for the first time.
  **The "thirty comparisons separate" sentence HOLDS against all three rebuilt comparators (30/30
  each).** ★ **But Golbandi's tail-at-eight win over the strongest static (+.0068 [+.0039,+.0098])
  is RETRACTED — it was comparator weakness; against the rebuilt items order it is −0.00159
  [−0.00489,+0.00167], ns. No published adaptive selector now beats the strong static order on either
  axis at any budget.** Also: items answered@16 4.20 → 3.11 (channel ratio ~4×, not 3×); the
  `tab:battery` caption's "lower bounds" flag, the `\REDO` on the greedy re-run and the `\PLAN`
  clause are all discharged; TODO 4.6's "concept 114 is absent from the combined order" is now false
  (it is the first pick, by the A2 gate).
- **A10 (1.18/1.18b, DONE 2026-08-12).** The min-leaf/answerability interaction, quantified:
  `docs/results/TREE_DEPTH_AND_MINLEAF.md` — all 15 stops are floor stops with a positive gain in
  hand (0 for want of a split), 8 forced by node size and 7 by shape, the floor is a window
  `[400/n, 1−200/n]` that closes at n=600, and 8 of the 15 stops are a concept 92.7–100% of the node
  could answer; **depth measures the rule, not the data — and no claim that deeper would pay is
  licensed either.**

## 3. Workstream B — Act 1 baseline additions

- **B1 (1.12, promoted MANDATORY). ★ FINALISED 2026-08-13 — VERDICT: the pre-registered mechanism
  branch fired on the (c)-vs-(b) leg and NOT on the (b)-vs-(a) leg. Write the outcome as
  `(c) ≫ (b) ≈ (a)`, NEVER as `(c) > (b) > (a)`.** Full write-up, six arms, per-budget CIs, structure,
  overlap, Part A/B reconciliation, novelty-as-conjunction and chapter-ready prose:
  `docs/results/HU_YU_2013_REPLICATION_AND_ABLATION.md` §B.0–§B.7 and §C.3–§C.5. **Nothing is left for
  the tex agent to compute — lift §C.5.**
  On 10,000 TEST users at K=16, gain vs the policy-B static order:
  **ours +0.01035 [+0.00749,+0.01328] full / +0.00705 [+0.00394,+0.01022] tail; Hu & Yu pairwise
  −0.03041; Golbandi variance −0.03061; Hu & Yu logistic (the loss they ACTUALLY RAN) −0.08627;
  random_ml (fair shuffle) −0.04817; random (degenerate → a random STATIC order) −0.03934.**
  Of the 60 test cells, 4 are nested at K=1 and **all 56 remaining intervals exclude zero**; ours is
  the only arm above the static order; the adverse case did not occur.
  **Contract check that licenses putting these arms in one table with the published tree number:**
  `banked_headline_assert` reproduces `ndcgtree16test_policyB.json` at **exactly 0.0 in nine of ten
  cells**, K16_full at 2.81e-07. Re-assembly from `splitcrit_per_user.npz` reproduces all 120 cohort
  cells at 1.17e-10 and the static comparator's per-user vector is **bit-identical across all six
  arms**.
  **Unpredicted and chapter-worthy: (b) ≈ (a) is now INTERVAL-BACKED** — paired, K=16, test:
  **+0.00020 [−0.00331,+0.00371] full and −0.00092 [−0.00448,+0.00277] tail**, the only null in the
  matrix. Upgrading a rating surrogate to a ranking surrogate buys nothing measurable; the separating
  axis is measured-vs-surrogate, not ranking-vs-rating. Act 3 should state this: hold the tree
  code, the split rule, min-leaf, the answers, the recommender, the mask, the pool, both cohorts
  and the budget fixed, change only the scalar a question is scored by, and the SIGN of the
  conditioning gain flips.
  **Three things the previously-unreported shuffle arms add, all of which must be written:**
  (1) `random_ml` is deeper (7 vs 5), wider (27 leaves vs 15) and more varied (286 distinct questions
  vs 159) than our tree and still loses −0.04817 ⇒ **no "deeper tree = better" argument survives**;
  (2) `random` degenerates to depth 0 (root item answered by 4.5%, 9,552/10,000 into one branch) and
  therefore doubles as a **random static comparator** — against which **neither published criterion
  separates on the TAIL at K=16** (pairwise −0.00212 [−0.00540,+0.00127] ns; variance −0.00120
  [−0.00486,+0.00245] ns), though both do on full (+0.00893 / +0.00873, separating). Report both
  halves; never write "worse than random";
  (3) **the logistic arm loses to BOTH shuffle controls decisively** (−0.03809 vs random_ml,
  −0.04693 vs random, on full).
  **Every criterion that is allowed to choose splits the root on a CONCEPT; only the two random arms
  pick an item, and both pick an unanswerable one** — a criterion-free corroboration of Act 4.
  **Separation (1) probe COMPLETE 05:46** (`pwheldout_root.json`, 1,000/1,000 candidates, 500 items /
  500 concepts, banked-NDCG gate 0.0 on all 1,000). The §B.6 preliminary (140 candidates, Spearman
  +0.637) is SUPERSEDED: **Spearman +0.784, regret +0.00459 vs +0.01608 random (29%), rank 2 of
  1,000, top-10 recall 0.40, argmax DISAGREES (concept 704 vs our concept 114)**. Against the
  training-loss pairwise criterion at the same node (Spearman −0.149, rank 75, regret +0.01473 = 92%
  of random), this says **separation (2) — training loss vs measured value — does almost all the
  work, and truncation/discounting is real but second-order.** Root state only; the `pairwise_heldout`
  TREE arm is still unrun.
  **Artefact defect to know about:** the tail metric excludes users with no tail targets, so tail
  means/CIs are over **9,281 test / 9,217 val** users, but the artefact's `n_users` field reads
  `10000` on tail cells too. Labelling defect, not numerical (recomputed tail means match at 4e-11).
  Part A's red flag (EIP below most-popular) was at **d=3 and is RESOLVED** — it was the latent
  dimension (d=10 → 20) plus a non-monotone fit; EIP now clears most-popular (.5277 vs .5129) and
  FixQ does not (.4910). The absolute level gap remains **unexplained and is reported as such**.
  **Part A's shuffle control CLOSED 2026-08-13 on both
  readings of the answerability protocol, and it corrected an earlier entry:** EIP − RandQ is
  +0.01365 → +0.00288 across depths 3→5 under the protocol as literally stated, but +0.00661 →
  **+0.00866** once training and test users are made equally answerable — so **the depth-5 collapse
  is an artefact of the asymmetry and the strong "beats no random tree past three questions on either
  reading" sentence is withdrawn.** Cutting the other way: EIP − FixQ shrinks to +0.0048 under
  symmetric answers (a quarter of their reported gap) and RandQ crosses below FixQ, reversing whether
  the advantage is attributable to the tree or to the criterion. Single seed, no CIs, nothing past
  depth 5. Everything lives in
  `docs/results/HU_YU_2013_REPLICATION_AND_ABLATION.md` (design sheet §0, pre-registered
  interpretation §0.8); the findings file is rewritten; bib key `hu2013interview` added to
  `new_chapters/references.bib` after checking all four `.bib` files for a collision (none).
  (i) snap on their protocol (MAP, 10 sampled negatives, FixQ) per HARD RULE 2 — ML-1M only, we do
  not hold Netflix or Yahoo; their results are FIGURE-ONLY, so "matching" can mean the arm ordering,
  the direction with budget and the gap sizes to pixel precision, and cannot mean a published digit.
  (ii) **objective ablation in OUR tree harness** — same code, same answers, same min-leaf, same
  certified pair, same policy-B mask, same 1,000-candidate pool, same cohorts, same 16-budget:
  variance (Golbandi) / pairwise rank loss (Hu & Yu) / **logistic loss (what Hu & Yu ACTUALLY RAN)**
  / measured NDCG@10 (ours), plus two shuffle controls. Three corrections the full read forced are
  recorded in TODO 1.12 and in the findings file; the biggest is that their ranking experiments ran a
  logistic loss and pairwise rank loss appears exactly once, in a list, never instantiated.
  The `pairwise_heldout` arm — their LOSS under OUR protocol, the clean single-factor test of the
  training-loss-vs-measured-value separation — is specified and QUEUED, not run: it needs a tower
  fold per candidate, so it costs what the NDCG tree cost (10.3 h). The cheap exact substitute that
  IS run: criterion-vs-MEASURED-NDCG rank correlation and regret over all 1,000 candidates at every
  node of the real tree, read out of the banked CELF heap sidecars for free — plus the completed
  root-state probe above.
  **NOVELTY, as a CONJUNCTION and never as a technique (§C.4).** Three prior-art facts bound it and
  all three must be stated by US: Hu & Yu OFFERED a rank loss in an interview-tree split (and The FacT
  SIGIR'19 instantiated one), so "first ranking objective in an interview tree" is unavailable;
  Golbandi WSDM'11 DID compare against a metric-optimised static (GreedyExtend, RMSE era), so "nobody
  compared adaptivity against one" is FALSE; Bilgic & Bennett SIGIR'12 DID put DCG@10/NDCG@10 inside
  an acquisition function, so "nobody used a truncated discounted ranking metric as a criterion" is
  FALSE. What survives is the six-way conjunction of §C.4 (measured improvement · exact reported
  NDCG@10 · full catalogue, nothing sampled · a question asked of a cold user · grown on one cohort
  and reported on a disjoint one · tested against a static order maximising the same quantity), plus
  the scoped field-level claim: *since the reported metric moved to truncated top-k ranking, no
  adaptive selector has been tested against a static order built by maximising it.*
  **UNRUN, listed as unrun and never omitted:** Part A depths 8/10 (EIP and RandQ), the Part A
  pairwise arm, the fMF RMSE panel, Netflix and Yahoo! Music, the `pairwise_heldout` tree arm, the
  `nonempty` min-leaf robustness mode, and seed variation on the two random arms.
  **Outstanding for the tex agent (owner of `chapterB_v2.tex`):** lift §C.5.1–§C.5.3; fix the
  Positioning paragraph that currently says nothing in the literature tests adaptivity against a
  metric-optimised static order (§C.2's fourth separation has been corrected in place).
**★ AUTHOR RULING 2026-08-13 — B2/B3/B4/B5 are NEEDED, BUT NOT NOW.** All four are owed for Paper B
and none blocks the current draft; they are Act-1 completeness, not claim support. Do not drop them
and do not schedule them ahead of workstream E.

- **B2 (1.10). NEEDED, NOT NOW.** RBMF snap + port (static; threatens the comparator, not the
  adaptive claim).
- **B3 (task 9). NEEDED, NOT NOW.** EDDI full Eq. 9 re-run. ★ Separately and *cheaply*: the
  three-outcome selector `eddi_ig3` is implemented but has only ever run on a 200-user smoke
  (`SMOKE_NOT_CITABLE`), so the chapter now carries a `\REDO` saying exactly that; the contract-scale
  run is 2–6 CPU-hours with no new code (TODO 1.3(b)) and is the cheapest debt on this list.
- **B4 (task 25). NEEDED, NOT NOW.** ConUCB k=8: resolve or report UNINFORMATIVE with the b(t)
  schedule stated.
- **B5 (task 10). NEEDED, NOT NOW.** CLiME width-rule run — make the degeneration citable.
- **B6 (1.21, SURVEY DONE 2026-08-12).** Deep-RL CRS family (CRM/EAR/SCPR/UNICORN/ConTS + the 2022–26
  successor layer) and the LLM elicitation line as AUDIT ROWS + positioning text. Evidence:
  `external_literature/findings/modern_crs_positioning.md`; chapter-ready prose and the borrow list for
  E/1.16: `docs/results/POSITION_VS_MODERN_CRS.md`. **Cite, don't run — confirmed, with the rebuild cost
  itemised: their RL reward IS target-hitting, so breaking the circularity deletes the method and leaves
  workstream E with their state encoder.** Four things this survey changes for the chapter:
  1. **★ The comparator claim must be SCOPED.** *"Nobody compared adaptivity against a metric-optimised
     static order"* is **FALSE**: Golbandi WSDM'11 beat **GreedyExtend** (CIKM'10), a static seed set
     built by greedily minimising **RMSE with the recommender refit on seed-restricted data** — the
     reported metric of its era — needing 6 questions where GreedyExtend needed >20. **This is evidence
     FOR §0's objective-mismatch frame, and it must be presented by us, not found by a reviewer.** The
     surviving claim: *since the reported metric moved to truncated top-k ranking, no adaptive selector
     has been tested against a static order built by maximising it.*
  2. **★ The objective claim must be SCOPED too.** A truncated, position-discounted ranking metric HAS
     been an acquisition function — **Bilgic & Bennett, SIGIR 2012** (`M = DCG@10 / NDCG@10` inside an
     ELO regret; arms `dcgELOq`/`ndcgELOq`), and **ELO-DCG, SIGIR 2010** has the discount without the
     truncation. Both are **regret**, both select **training labels**, neither asks a user anything.
     Our four surviving deltas: improvement-not-regret · held-out cohort metric · a question to a user ·
     full catalogue. **Also: greedy forward selection with an NDCG wrapper is ordinary in LTR feature
     selection — claim the conjunction, never the technique.**
  3. **★ Two more concessions.** **The FacT (SIGIR 2019)** is a confirmed ranking-split *interview* tree
     (BPR term in the split, "interview questions" verbatim) — so "a ranking loss in an interview tree's
     split" is 2019 prior art, not only Hu & Yu's 2013 framework. And **Hu & Yu has SEVEN citers in
     thirteen years**, none an interview-tree method paper, **nobody ever instantiating the pairwise
     rank loss** — the third measured instance of this subfield's dormancy pattern, and an argument for
     running the B1 ablation rather than citing a number.
  4. **Corrections to our own records:** SCPR is **KDD** 2020; **hDCG is UNICORN-onward** and
     CRM/EAR/SCPR/ConTS report no rank-sensitive metric at all; the circularity is **label-based, not
     representational** (never write "the simulator is built from the recommender's embeddings");
     PEBOL's stored DOI was wrong. 16 bib keys added, collision-checked across all four `.bib` files;
     **three carry PLACEHOLDER TITLES flagged in-file and must be fixed before any compile.**
  Outstanding under B6: add the audit rows to `BASELINE_OBJECTIVE_AUDIT.md`, and fold the position doc
  into the chapter's Positioning section.

## 4. Workstream C — Act 2 exhibits

- **C1 (1.11). ★ TO BE RUN FOR PAPER B, BUT NOT NOW — author, 2026-08-13.** Surrogate-vs-NDCG rank
  correlation per method per state; turns "1 of 21" from description into measurement and gates 1.15.
  Code is BUILT and GATED (`scripts/act2_surrogate_rank_corr.py`, phases 2/3 complete, self-tests
  pass); cost ~12 h; the author's sign-off on the 1,000-candidate pool is recorded in TODO 1.11.
  **Interim cover:** the Hu & Yu criterion-regret exhibit (§B1) measures the same claim at the 22
  nodes the tree actually visits, so the chapter is not blocked on C1.
- **C3 (NEW, author 2026-08-13). Unnormalised cohort DCG@10 as the selection criterion.** Bilgic &
  Bennett (SIGIR'12) find DCG-loss beats NDCG-loss **for selection**, because per-query normalisation
  drags selection toward low-IDCG queries. Our criterion is per-USER-normalised, so the same pathology
  may apply to us. Single-factor ablation: same greedy/tree machinery, swap the scored scalar from
  cohort-mean NDCG@10 to cohort-mean **unnormalised** DCG@10, report the resulting order and its
  NDCG@10. **~4–6 h.** Most relevant when the parametric policy's reward is designed (workstream E) —
  a reward that is the wrong normalisation is a defect baked into training, not a reporting choice.
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
2. **Then:** A5, B1(ii) ablation, B2, A10. (**A6 and A8 are CUT — author, 2026-08-13.**)
3. **Then:** E1 → E2 (→ E3 if needed), with F and writing (C2, D1, F1) interleaved.
4. ~~Confirmatory pass A8 + ledger before any abstract number is final.~~ **CUT 2026-08-13** — see A8
   above for the ruling and why it is sound. The ledger stays; no re-run is owed.

## 9. Not claimed / guarded

- Never "adaptivity is large" — size qualifiers of §0 travel everywhere.
- No routing/curriculum claim unless D2 separates from its control.
- Tail is a **secondary outcome** everywhere: every order in the chapter maximises full NDCG@10
  and no search optimises tail. State it once; do not claim a tail-*optimised* result.
- Genre numbers remain UPPER BOUNDS — the free genre fact costs no turn — and **all 17 groups still
  shrink val→test on ONE ruler** (1.80× cohort-weighted on full), with Drama still a null.
- Nulls are not citable (author ruling). **The pre-registered (b)≈(c) narrowing branch did NOT fire**
  — (b) loses to (c) by +0.04076 [+0.03741,+0.04412] full, paired — so no narrowing is written. The
  branch that DID fire unexpectedly is (b)≈(a): write **`(c) ≫ (b) ≈ (a)`**, never `(c) > (b) > (a)`.
- Never write "the published criteria are worse than random": on full they beat the degenerate random
  static order (+0.0089 / +0.0087, separating); only on the TAIL at K=16 are they nulls against it.
- Never write "first ranking objective in an interview tree's split", "nobody compared adaptivity
  against a metric-optimised static order", or "nobody used a truncated discounted ranking metric as
  a selection criterion" — all three are FALSE (§C.4). Claim the CONJUNCTION, never the technique.
