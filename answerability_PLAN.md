# Answerability Study — EXECUTION PLAN (2026-07-06, post-Fable-review)

Incorporates answerability_design_review_2026-07-06.md. This doc + answerability_study_design.md =
the PRE-REGISTRATION once cutoffs (§B) are frozen. Arena v1 = ML-25M. NOT RUN until §A done + owner go.

## VERDICT ON THE REVIEW
Accept all 6 holes + smaller gaps. Two are load-bearing: Hole 1 (shared-prior circularity → mask-and-
measure), Hole 2 (stereotyping vs knowledge → held-out-half validity gap, zero extra cost). Hole 3
(exploitability G2) saves ~2 weeks by upper-bounding the agent prize before training. This converts a
1-stage "does the LLM emit heterogeneity" gate into a 3-part "is the heterogeneity ABOUT THE USER and
EXPLOITABLE" gate — which is what the thesis actually needs.

## A. PRE-REGISTRATION FREEZE (do first, no LLM calls)
A1. Fill the Q-E cutoffs into §B below; date + commit design + review + this plan as the frozen prereg.
A2. Design decisions locked from the review:
    - Value: DATA-SIDE PRIMARY everywhere (real ratings for items/pairs; mean-of-member-real-ratings for
      concepts when ≥k rated members). LLM-predicted value = SENSITIVITY-ONLY, no headline depends on it.
    - Two structural artifacts, distinct roles: (i) A-PRIORI rule = the robustness witness (τ,k set BEFORE
      seeing LLM output; anchor τ=top-5% popularity, k=2; match global scale only, never per-user structure);
      (ii) FITTED P(answerable|features) from Q-B = scale surrogate/plumbing, NEVER cited as independent.
    - "maybe" → refuse (primary), →yes (sensitivity). Pre-registered, not a free parameter.
    - Prompt "meaningful answer" defined PER CHANNEL: items="has seen/has a real opinion"; concepts=
      "familiar enough to state a preference". Sample-not-complete framing kept.
    - Held-out TARGETS excluded from the interview question bank (Hole 6). Masked-item validation is a
      SEPARATE calibration pass, never inside evaluated interviews.
    - Model snapshot PINNED (exact gpt-5.4-mini snapshot id) for the cached grid; temperature pinned.

## B. FROZEN CUTOFFS (the prereg numbers — fill/confirm with owner before run)
- Heterogeneity: mixed-effects logistic can_answer ~ logpop + breadth + (1|user)+(1|question); LRT that
  user-variance>0 AND report ICC. Threshold: ICC ≥ 0.05 (user structure non-trivial). [CONFIRM]
- Taste-tracking: add taste-match covariate (cosine(user known-profile genre/tag dist, question genre/tag
  vector)); OR ≥ 1.5 per sd, or ΔAUC ≥ 0.05 vs popularity-only. Report raw effect regardless. [FROZEN: OR>=1.5/sd OR dAUC>=0.05]
- Validity gap (Hole 2): judged answer-rate on held-out-RATED minus matched never-rated, within pop tiers;
  require >= 15pt gap in low/mid tiers. [FROZEN 2026-07-06]
- Exploitability G2 (Hole 3): ΔNDCG(answerability-aware greedy - blind greedy) CI excludes 0, AND dNDCG>=0.005 (material). [FROZEN 2026-07-06]
- ADAPTIVE-CLAIM GATE = heterogeneity AND taste-tracking AND validity-gap AND G2>0. All four, else static map.

## C. SCRIPT UPGRADES (from pilot → gate script; cheap, no full run)
C1. Log response.usage every call; print running $ (measured, not assumed). Pin model snapshot + temperature.
C2. Fix nits: label ratings correctly (not "liked"), say "a sample of size n of N (they have seen more)",
    drop json_object-vs-list mismatch, add masked-item + held-out-half batteries to the question set.
C3. Stratified sampling: 300 users by profile-size × dominant-taste-cluster, ≥20 users/taste cell
    (NOT 1-user-per-genre — that confounds user with taste, kills taste-tracking power).
C4. Question bank: concepts = genome tags binned by breadth (~40, tiered); items = designed bank ~500-2000
    (popularity strata × genre coverage + per-user taste-adjacent) — judge-all on the bank, batched; PLUS
    held-out-rated + matched never-rated items (validity), PLUS masked-rated items (Hole-1 calibration).
C5. Small spot-checks to schedule (cheap): batch-vs-single agreement (~subset), pair-answerability battery
    (~200 calls), cross-family (Haiku/Flash on ~30 users).

## D. RUN ORDER (ML-25M)
1. FREEZE prereg (A) + upgrade script (C). [no calls]
2. MINI-PILOT on ML-25M (~15 users): sanity + the held-out-half positive control works + cost measured. [cents]
3. GATE RUN (~300 stratified users, bank per C4, cached, usage-logged, pinned): compute heterogeneity +
   taste-tracking + validity-gap + G2. [2-3 days; measure cost — expected low single-$; CONFIRM by usage]
4. SPLIT-SENSITIVITY (Hole 5): 50 users × 2-3 alt splits; report judgment agreement/answer-rate sd.
5. DECISION (pre-registered): all four gates pass → MAIN STUDY (cached grid; masked-item MAE validation
   → primary value predictor + fidelity σ; both structural artifacts; both value sources for headline
   robustness) + build the adaptive agent (measured against the G2 oracle-headroom). Any fail → STATIC
   channel map, adaptive clause cut. EITHER WAY the gate result is a citable finding.

## E. CLAIM BOUNDARY (Q-F, print in every paper)
"Results hold under two external answer models (an LLM judge and an a-priori structural rule); the LLM
judge is validated against human answers in §X [pending human study]." NEVER "realistic users" / "human-
level answerability." LLM study = load-bearer of recomputed results; human study = validator of the judge.

## F. FROZEN 2026-07-06 (owner: reasonable cutoffs set, ML-25M only, proceed, not worried re single-$)
- Cutoffs above FROZEN. Arena = ML-25M only (v1). Budget: proceed through gate (single-$ ok).
- Execution: agent does script-upgrade + ML-25M mini-pilot, STOPS for review before the 300-user gate
  (the pre-registered decision spend). STOP-ON-BLOCKER stands.

## F-OLD. (superseded)
- Confirm the four cutoffs in §B (or adjust).
- Confirm arena = ML-25M only for v1.
- Confirm budget to proceed through the gate (expected low single-$, measured at step 2).
- Note: STOP-ON-BLOCKER rule stands — any gate failing halts to owner, does not auto-fallback.
