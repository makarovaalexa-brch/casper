# Design Sheet — Amortized-E3 (learned continuous q2 policy) + blend-answer honesty pre-check

Signed off by author 2026-07-18 ("i am ok with the design, just record it and go ahead").

## Question
Can a LEARNED, per-user, answer-value-adaptive selector match or beat E3's answer-branching prize with a
CONTINUOUS action (a query direction), trained end-to-end through the exact Kalman fold (DKQS, no RL)?
Motivation: the discrete tree beats a fixed questionnaire (+0.0198 tail) but LOSES to per-user variance-greedy
(-0.0078); the optimum must be BOTH per-user AND answer-adaptive in one function = a learned policy.

## Data (HARD RULE #1 — stated in full)
- Recommendation catalog / NDCG target: ALL 18,430 items. Tail-NDCG@10 over held tail-likes (non-head).
- Concept candidates: ALL 1,317 usable concepts (usable = >=20 rated members, a calibration necessity; refusal
  encoding => universal coverage => the old COV>=0.30 filter is vacuous, all 1,317 in play).
- Users: the sanctioned 2,500-user probe (val split), disjoint train/eval halves (E3's exact split, seed 7).
- Items as questions: NOT included (concept-only shared bank; item-as-shared-question needs universal answers).
- A HEADLINE number requires the all-users run (150k) + a fresh sign-off. This probe is diagnostic only.

## Pre-check FIRST (gate): blend-answer honesty  [`scripts/bpool_honesty.py`]
DKQS trains on a BLEND answer: for d = normalize(sum_i w_i d_{c_i}), observation y = sum_i w_i y_{c_i}
(w-weighted blend of component calibrated answers), sigma^2 = sum_i w_i^2 sigma^2_{c_i}. This is honest ONLY if
the blend answer tracks the TRUE projection <u*, d>. Test: random 2- and 3-sparse blends over the bank; per user
compute true = <us, d> (us = cached belief), blend = sum w_i y_{c_i}; report corr(blend, true) and R^2 vs the
single-best-component baseline. GO if median corr(blend,true) >= ~0.8 and comparable to single-concept fidelity;
else the training signal leaks -> fall back to Fable Arch 3 (learned gain-critic, supervised).

## Method (if gate passes)
- State: after static q1 (concept 128), Sig1 is user-independent => state = mu1 (512-d) only. No history encoder.
- Actor f: mu1 -> sparsemax weights over the q2 concept bank -> d = normalize(sum w_c d_c). On-manifold; contains
  the discrete tree (one-hot = a tree move); sparse blend = LLM-verbalisable ("between X and Y", Paper E OMP-k=3).
- Fold: exact Kalman with the blend (y, sigma^2). Objective: soft tail-NDCG (NeuralNDCG / softmax-DCG over the
  18,430-item score vector) on the eval-tail held likes. Differentiable end-to-end in the actor params.
- FROZEN: the observation model (Bc/EDG/Yc/S2) and sigma^2 — the policy MUST NEVER output/scale sigma^2 (else it
  fakes confidence and games the non-degradation invariant). Only the SELECTION (weights w) is learned.

## Comparators (the known ladder, same eval half)
q1-only 0.0710 | static-q2 +0.0169 | E3 cluster-adaptive +0.0254 | per-user peeking oracle (ceiling) | AND
variance-greedy (the champion). Report snap agreement (top-weight == E3 cluster's q2) and blend sparsity.

## Controls (day one)
- Frozen-mu twin: actor fed mu1=prior for SELECTION, real answers in the UPDATE -> learned-minus-twin = the
  provable answer-contingent prize.
- Answer-permutation null: select on user i's answers, score user j -> must vanish.

## Metric + MDE
Exact tail-NDCG@10 on the eval half. MDE anchored at the E3 prize (+0.0085). Decision: learned-q2 >= +0.0254
=> matched adaptivity with a continuous action & no clustering => green-light the full T=8 differentiable unroll +
the soft-concept M(d) answer test. Between static and E3 => run blend-answer honesty deep-dive. Below static-q2
=> gradient path broken => fall back to Arch 3 critic.

## Flagged shortcuts
blend-answer linearity (gated by the pre-check); soft-NDCG surrogate (vs exact @eval); frozen sigma^2; static q1
prefix (collapses state to mu1 — a deliberate simplification of the first experiment, not the final method).

## No LLM calls in this phase (geometry + answer model are LLM-free).
