# The Answerability Study — Results & Paper Placement (2026-07-06)

Status: GATE + CROSS-CHECKS final; MAIN STUDY in progress (fitted P(answerable) pending). This is the
publication-facing write-up; raw log = ANSWERABILITY_RESULTS_LOG.md, per-experiment json in experiments/.

## 1. The problem this solves
Every prior elicitation evaluation answers a silent, un-deployable assumption: either "a user can answer
about any item" (false at scale) or "answerable = rated" (a lower bound that grossly underestimates
knowledge), or answers come from the recommender's own latent geometry (circular). We replace the assumption
with a MEASUREMENT: an external LLM judge reads a user's known profile and judges what they could answer —
independent of any rule we authored, so it cannot be exploited circularly. It removes SELF-AUTHORED
circularity; it does not remove model-prior dependence (the LLM is a learned proxy) — the human study
validates the judge (pending). Claim boundary, stated everywhere: "results hold under two external answer
models (an LLM judge + an a-priori structural rule); LLM judge validated against humans in §X [pending]."

## 2. Design (pre-registered; frozen before any run)
ML-25M, gpt-5.4-mini-2026-03-17 (pinned), temp 0, fixed answerer split seed 123, "maybe"->refuse.
Answers are non-circular: real ratings for rated items/pairs; data-side member-rating aggregate for concepts;
LLM value only where no ground truth exists, sensitivity-only. Answerability judged externally (LLM), scored
everywhere by a fitted P(answerable|features) (LLM-derived plumbing, never the independent witness). Four
pre-registered gate tests, cutoffs frozen in answerability_PLAN.md.

## 3. GATE RESULTS — PASS 4/4 (300 users, $1.29)
| Test | What it rules out | Cutoff | Measured | |
|---|---|---|---|---|
| Heterogeneity | "everyone answers the same" | ICC>=0.05 | ICC=0.174 (perm p=.003) | PASS |
| Taste-tracking | "answerability = popularity only" | OR>=1.5/sd or dAUC>=.05 | OR=2.71/sd, dAUC=.106 (AUC .946 vs .840) | PASS |
| **Validity gap** (the key one) | "LLM just stereotypes the prompt" | >=15pt low/mid | **moderate +44.1pt** (n=768); famous +22; obscure +10 | PASS |
| G2 exploitability (privileged UPPER BOUND) | "heterogeneity is useless for NDCG" | CI>0 & dNDCG>=.005 | dNDCG=0.287, CI[.268,.307] | PASS |
The validity gap is the load-bearing result: for mid-popularity films the user did NOT show the judge, the
judge rates them MORE answerable when the user actually rated them than when matched on popularity+genre —
i.e. it infers REAL user-specific knowledge from taste (the scandi-noir-fan hypothesis, confirmed), not
stereotype. G2=0.287 says the fuel is large; it is a privileged oracle bound, NOT the realizable agent prize.

## 4. CROSS-CHECKS — both PASS (robustness Fable required)
- **Independent-CF value (shared-prior circularity):** bank-restricted EASE (known-portion only, 4147-item
  universe) predicts the masked items at MAE 0.74 / corr 0.43 vs the LLM's MAE 0.68 / corr 0.50. Two
  independently-built predictors agree within 0.06 stars on real held-out ratings; LLM marginally better.
  -> LLM value is not a shared-prior artifact; both reportable as sensitivity sources.
- **Cross-family judge (one-model artifact):** GPT-5.4-mini vs Claude-Haiku-4.5 answerability agreement
  83.2%, Cohen's kappa 0.67 over 2250 questions; validity-gap sign replicates under Haiku. -> the fuel is a
  property of capable LLMs reading profiles, not a GPT quirk.

## 5. MAIN STUDY — IN PROGRESS
Judging full concept set + large stratified item sample; then fit P(answerable|log-pop, count, decade,
genre-match, franchise), validated on HELD-OUT USERS (AUC + calibration). [numbers TBD; append here.]

## 6. STRUCTURE DECISION (owner, 2026-07-06): SEPARATE THESIS CHAPTER, FOLDED FOR PAPERS
- **THESIS: its own chapter** — working title *"Measuring Answerability: An External-Judge Testbed for
  Deployable Elicitation."* Sits between the instrument chapter and the flagship; makes everything after it
  non-circular. Contents: the problem (silent un-deployable answer assumptions in the field, with citations
  = the deployability audit), the external-LLM-judge method, the 4-test pre-registered gate, the 2
  cross-checks (independent-CF + cross-family), the fitted P(answerable) model, the validity-gap FINDING
  (external answerability is heterogeneous + taste-tracking + NOT stereotype + exploitable), claim boundary
  (removes self-authored circularity, not model-prior; human study validates). MUST carry the AUDIT + the
  FINDING, not just "we built a judge" (else it reads as plumbing). Strongest viva material.
- **PAPERS: FOLD IN, do NOT spin a 4th paper for Oct 2.** Primary home = Paper 1 (which BECOMES the
  answerability/measurement paper; old instrument content = its foundation). Post-ECIR, IF main study +
  human validation are strong, a short standalone "released calibrated answerer + audit" resource paper is
  an option — decide after the fitted-model numbers, not now.

## 7. WHICH PAPER — placement (paper-side detail)
This is a CROSS-CUTTING METHODS CONTRIBUTION; it does not belong to one legacy chapter. Placement:
- **PRIMARY = Paper 1 (the instrument/testbed paper).** The answerability judge + validity-gap methodology +
  the deployability-assumptions audit are apparatus, exactly Paper 1's remit ("how do we measure elicitation
  honestly"). The four-test gate and the two cross-checks are its rigor showcase. Headline addition: "prior
  simulators answer with the evaluator's own assumptions; we measure answerability externally and validate it."
- **ENABLES = Paper 2 (the flagship / channel map).** The fitted P(answerable) is the environment the
  adaptive coarse->granular agent runs in; the G2 upper bound is the oracle-headroom the agent is measured
  against. Paper 2 CITES Paper 1's judge; the adaptive result lives in Paper 2 (pending agent build).
- **SUPPORTS = Paper 3 (open recall).** Answerability judge gives D's framing-lever / recall claims an
  external (non-privileged) answer model, replacing the rigged/oracle answerers.
Net: the study is Paper 1's new backbone and Paper 2's enabling engine. It is NOT a standalone paper —
it is the thing that makes the other two non-circular. (If it grows, a short "resource" paper on the
released calibrated answerer + audit is possible, but default is Paper 1.)
