# DESIGN SHEET — ARENA V2 (the corrected arena)
Status: DRAFT FOR AUTHOR SIGN-OFF. Nothing runs until signed. Supersedes the V300 run (300-user
selection) entirely — that run is void (noise-fit, mismatched train/val, wrong baselines).

## 0. The question, and the ONE number that decides it
Does any deployable adaptive policy beat the strongest *honest* static, on a cohort large enough
that the static isn't noise-fit — and **how much of the static's own gain survives out-of-sample?**
THE HEADLINE DIAGNOSTIC (printed first): b2's per-turn NDCG@10 on TRAIN vs held-out TEST, matched
cohort + metric, side by side. If TEST collapses after turn 1-2, the real elicitation ceiling is
~1-2 questions and everything downstream is measuring noise. This number gates the rest.

## 1. Why V2 exists (the V300 failures being corrected)
- SELECTION ON 300 USERS: b2 greedy-selected on 300; policies trained on 400-1000. We built a
  162k-user world precisely to escape 300-user starvation, then didn't use it. Cause of both the
  noise-fit picks ("voyeurism" at turn 4) AND the flat policy learning (700 users, 2428-way choice).
- NO TRAIN/VAL GAP MEASURED: train ran @10 on 300, val ran @50 on 40 — mismatched metric AND cohort,
  so the generalization gap was never computable. #0 fixes this.
- BROKEN/ABSENT BASELINES: b1 was ANSWERABILITY-entropy (asks most 50/50-answerable questions, zero
  taste content) and its curve DECREASED. No random baseline. No popularity baseline. Cannot judge
  b2's strength without honest floors.
- @50 metric drift (reverted to @10, program-canonical).

## 2. Cohorts (exact Ns) — all disjoint, all synthetic population users, 173 UNTOUCHED
- TRAIN (selection + policy training): N_tr = 3,000  [see shortcut S1]
- VAL (model/checkpoint selection): N_va = 1,000
- TEST (all headline numbers, held out): N_te = 1,000
- All drawn disjoint from the trU synthetic population (v2.1 answerer, fold-v3 belief), 300 study
  IDs excluded. The 173 real-judged users are NOT touched by this run at all.

## 3. Action space & horizon
- Full universe: 2,428 questions (1,128 concept / 500 entity / 800 item). No pools, no sampling of
  the action set.
- Horizon T=10 curve; HEADLINE = NDCG@10 at TURN 8 (program-canonical budget). Per-turn curve 1->10
  printed for every arm. (Budget-finding from the b2 curve confirmed the value is front-loaded;
  no need for T=25.)
- Metric: NDCG@10 primary; tail-NDCG@10 reported alongside (Paper B headline). MDE printed per CI.

## 4. Baselines (the honest floor — all on the SAME cohorts/metric)
- b0  COLD (ask nothing) — reference line.
- bR  RANDOM question each turn (unasked, uniform) — the true floor: does *ordering* beat *asking
  anything*? Seed-averaged (>=5 seeds) with CI.
- bP  POPULARITY — ask most-popular / most-answerable first (Golbandi's classic strong baseline;
  we had it in Paper A, dropped from the arena). Deterministic.
- bE  TASTE INFO-GAIN entropy — expected reduction in taste-posterior uncertainty / expected NDCG
  gain (Paper B's realizable info-gain policy — PORT it, do not reinvent). Replaces the broken
  answerability-entropy b1. This is the canonical EIG baseline reviewers demand.
- b2  LEARNED STATIC — greedy forward selection maximizing NDCG@10 at turn 8, on N_tr. The
  strongest fair fixed order. Reported TRAIN and TEST (the #0 gap).
- b4  MODEL-BASED MYOPIC GREEDY — per-turn argmax expected @10 gain, same beliefs/fold; deployable,
  no learning. If b4 > b2, that's the first honest adaptive-by-computation win.

## 5. Adaptive arms (the author's objective test)
Same class-A scorer architecture, trained on N_tr, two objectives:
- ARM-END: maximize endpoint NDCG@10 at turn 8 (discrete, high-variance target).
- ARM-BEL: Paper B's belief objective — pick questions that move the belief embedding toward the
  full-profile target z* (dense, low-variance). z* = fold of all the user's known ratings,
  privileged TRAINING teacher only; eval policy uses beliefs alone (deployable).
DECISIVE READ: does ARM-BEL's per-turn @10 curve RISE where ARM-END stays flat? Rises => the signal
was too noisy, belief-matching is the fix. Both flat => model/representation, not objective.

## 6. Verdicts (pre-registered, printed before numbers)
- PRIMARY: best adaptive arm vs b2 AND vs b4, endpoint@10 turn 8, TEST, paired CI + MDE.
- The #0 train-vs-test b2 gap, printed first.
- Every arm's per-turn curve 1->10 with error bars. No post-hoc metric/turn selection.
- Mechanism: channel mix + refusal rate over turns for the winning adaptive arm.

## 7. Shortcuts flagged (with alternatives)
- S1 (COMPUTE): full greedy on the whole population is CPU-prohibitive (300-user prescreen took
  ~27 min; the population would take days). N_tr=3,000 is a feasibility cap — ~10x the noise-fit
  scale, tractable in a few hours. ALTERNATIVE if 3k still noise-fits at the tail: stochastic
  greedy (re-sample the scoring cohort each turn) or accept a 1-2 question honest ceiling. Flag,
  don't hide.
- S2: policies use the class-A scorer only (B/C/D deferred) until ARM-END/BEL show a learner exists.
- S3: bE ported from Paper B; if the port doesn't fit the arena interface, implement EIG directly
  and say so.

## 8. Scope / cost
DEV-synthetic only. $0. No LLM calls. 173 untouched. Non-idling execution: the driver script runs
+ self-polls to completion; no agent waits on a background signal. ~few hours CPU.

## SIGN-OFF
[ ] Author: cohorts (3k/1k/1k), baselines (random/popularity/taste-EIG), dual objectives,
    #0 train-vs-test gap first, S1 compute cap accepted.
