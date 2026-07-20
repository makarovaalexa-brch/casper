# DESIGN SHEET — EVOI policy on the additive-surrogate arena (AWAITING AUTHOR SIGNATURE)
Date: 2026-07-14. No run starts until this sheet is signed. HARD RULES #1-#3 (casper/CLAUDE.md) bind
this sheet and every subagent brief derived from it: NO data reduction of any kind without author
sign-off; the adaptivity question is settled (+47% tail, Jul-14 probe); the eight dead policies are
not cited, analysed, or reasoned from.

## QUESTION
Does a deployable adaptive policy (greedy EVOI over the full 800-item bank, enumerated via an
additive fold-surrogate, deployed on the frozen pb2 recommender) beat the hardest static and
cluster-adaptive baselines on TAIL NDCG@10 under an unpunished masking protocol?

## ARENA (changes from the current one)
- MASKING, PRIMARY (P-NEUTRAL — AUTHOR DECISION 2026-07-14, matches Paper B `mf_foldin.py:51,57`):
  per-user credit-neutral. rel_u = held-out ≥4, not head (190), MINUS items asked-and-answered.
  An asked item leaves the ranking AND the IDCG denominator, per user. No arm is punished for asking
  a good question; no privileged info (asking is not chosen from labels); users who were never asked
  a given item keep it in their denominator. This is the rule CASPER has used since June.
  * The Fable "wart" (a shrinking denominator mildly flatters asking about a target) is SYMMETRIC
    across all arms and REPORTED, not hidden.
  * THIS IS THE GREEDY-SLIM-FAITHFUL RULE. arXiv 2406.06061's I_Q ∩ I_R = ∅ excludes the ITEMS
    ACTUALLY ASKED (the ≤20-item questionnaire), per user — NOT a candidate bank. Fable's earlier
    "P-GLOBAL exclude all 800" was a MISREAD: our 800 is the pool we SELECT questions from, not their
    I_Q. Excluding only the ≤q asked items (P-NEUTRAL) is exactly their rule and comparable to them.
  * Global 800-exclusion REJECTED outright (author, 2026-07-14): a recommender that refuses to surface
    the 800 most popular films is not deployable and would corrupt the denominator for users never
    asked those items. It is not what SLIM did and it is not run, not even as a comparability row.
- CANNIBALISATION STUDY (DIRECT measurement, not a masking variant): for each question the policy
  asks, record whether the asked item WOULD HAVE BEEN in that state's top-10 recommendation, and its
  popularity rank. Tests the prediction cost-of-asking scales with asked-item popularity. Independent
  of the masking rule → no denominator-shrink confound.
- The current-code rule (mask from numerator, KEEP in IDCG) is ABOLISHED — it punishes the best
  question. The 2026-07-13 2.4M target file is contaminated by it and is not trained on. The author's
  exclude-targets-from-askable fix is set aside as privileged (uses the eval split); P-NEUTRAL gets
  the same "no penalty" property without touching the askable set.

## EXACT Ns — NO REDUCTIONS
- Users: 150,239 (all). Quarantined 173/300 study users excluded as always. Train/eval = the
  standing disjoint halves; selection and evaluation never share users.
- Action space: ALL 800 bank items per state (KCAND=16 is RETIRED). All answer levels (10 item
  star-levels + refusal/knows states) per candidate at EVOI time.
- States: all interview depths 0-8, all users. Interview length budget q identical across arms;
  burned (refused) turns count against every arm.
- f_r training: ALL (user, bank-item) answerer labels (~120M), streamed/micro-batched, no sampling.

## MACHINERY (the compute fix — algebraic, non-lossy)
Surrogate scorer from the EXISTING pb3 belief-pool checkpoint (no retraining):
score'(k,l,i) = Σ_r a_r(t)[U_r(i) + λ_k T_r(k,l,i)] + c_i, t = Λ_state + λ_k, where N, Λ are cached
exactly per state; x_{k,l} is the exact context-free FiLM token; λ_k uses frozen state context.
Precompute: R tables T_r (8,000 × 18,430, fp16) + R decodes U_r per state. Deployment/eval folding
and final recommendations are ALWAYS true pb2.

## SHORTCUTS — every one flagged, with its non-lossy alternative
- S1 rank-R Cauchy separation of 1/(p0_j+t): numerical approximation; R chosen for max rel. error
  < 1e-3 on the realized t-range, verified on the full t distribution. Fallback (exact): per-state
  per-level exact decodes (~10x cost, still feasible).
- S2 frozen-context λ for the appended token (pb3's λ is set-contextual): one-token, one-step
  approximation. Validated against TRUE pb3 refolds on the surrogate's argmax candidate for ALL
  eval states (cheap), plus the regenerated uniform probe.
- S3 scorer/deployer mismatch (pb3-surrogate ranks, pb2 folds): GATED, see G1-G2. Fallbacks in
  order: distil pb2's own fold into additive per-(item,level) deltas; adopt the additive pool
  end-to-end (−0.007 full NDCG, accepted only with author sign-off).
- S4 f_r is trained on the distilled answerer (no LLM calls — standing rule respected).
- S5 the 16/state probe regeneration (~3h) is a VALIDATION measurement of the surrogate, uniform
  over candidates; it is not training data and not a reduction of the experiment.

## GATES (in order; a failed gate stops the line, per standing rules)
- G1 surrogate fidelity: within-state Spearman(surrogate score, pb2-realized outcome) on the
  regenerated probe; report the full distribution.
- G2 argmax transfer: true-pb2-realized tail-NDCG of the surrogate's per-state top-1 ≥ 0.7 × the
  clairvoyant best-of-16 realized lift, on ALL states (150k true folds, ~15 min at measured rate).
- G3 potential validity: state-level Spearman(dense potential, tail-NDCG@10) ≥ 0.8 before m trains;
  else switch potential to held-out multinomial log-likelihood.

## POLICY + ESTIMATOR
Greedy myopic EVOI(k) = Σ_l f_r(l|z,k) · m(z'_{k,l}) over all 800 × all levels at each turn.
f_r: multinomial CE (knows + graded level). m: per-STATE value on the dense potential
(truncation-free tail-DCG over all held-out tail targets), two-part head (P(hit) × magnitude).
EVAL METRIC IS ONLY TAIL NDCG@10 (P-GLOBAL). Winner's-curse controls: full-800 coverage; ensemble
LCB argmax (β on val); checkpoint selection on realized argmax tail-NDCG on disjoint val users,
never on regression loss; predicted-vs-realized calibration at the argmax reported.
Published ablation: amortized Q-head, within-state listwise softmax over full-800 surrogate targets.

## ARMS (all under P-GLOBAL, all deployed on pb2, identical q and checkpoint-selection budget)
A0 no-question base. A1 random-q. A2 static top main-effect questionnaire. A3 HARD STATIC: fixed
sequence greedily forward-selected on downstream tail-NDCG on the train half using the SAME delta
table and SAME selection budget as the policy. A4 genre-cluster adaptive (Jul-14 incumbent — the
bar). A5 EVOI policy. A6 clairvoyant per-state best-of-800 (privileged ceiling, labelled as such).

## METRIC + MDE
Paired per-user tail NDCG@10, disjoint eval half (n ≈ 75k), 95% bootstrap CI. Jul-14 run achieved
CI halfwidth ≈ 0.0009 at this n → MDE ≈ 0.002 vs prize scale 0.01-0.06. Seeds: standing set
{123,1,2,3,7,11} for any stochastic training. All checkpoints persisted to .cache/ with durable
peak files (standing rule).

SIGNATURE (author): ______________________  DATE: __________
