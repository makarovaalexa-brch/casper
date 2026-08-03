# Design sheet — Dispersion census on existing question banks (task #52)

**Status: DESIGN ONLY, not run. CPU cost ~minutes on the existing harness; no training.**
**Do not run while the concept thread is unsettled if it needs the concept bank — the item-bank
census is unblocked now.**

## Question (fixed before running)
Under the standard question banks (items, existing static orders, greedy bank), how much do the
first answers actually DIFFERENTIATE users' posteriors — measured as across-user dispersion of the
loss-optimal next question — and is measured dispersion ≈ 0, which would explain every adaptivity
null in the corpus (ours and the field's) as "the openers never create divergence"?

## Why it matters / what changes on the outcome
- Dispersion ≈ 0 on standard banks → the adaptivity null is CONDITIONAL (the field never created
  divergence); the splitter probe becomes the decisive experiment; §Adaptivity is written as the
  conditional structure (design doc Challenge 2 ruling).
- Dispersion substantially > 0 on standard banks WITH our measured gap < MDE → the null HARDENS
  ("divergence exists and cannot be cashed") — the off-diagonal cell; print it.
- If no decision changes on the outcome, don't run — decision above shows both branches change the
  writing plan, so it runs.

## Measurement (the M11 correction is binding)
Dispersion is computed on the **loss-optimal** argmax, not the information argmax: for user u with
posterior state after answers a_1..k, the next question q*(u) = argmax_q E[ΔNDCG@10 | u, q] under
the behavioral answer model. Dispersion at step k = 1 − (max_q share of users whose q*(u) = q),
plus entropy of the q* distribution, reported both. (Σ-only criteria are answer-independent by the
theorem — their dispersion is 0 by construction and is the null row.)

## Exact Ns and data
- Users: the 10k canonical VAL users (never test; test stays untouched for the splitter probe).
- Instrument: frozen i26c ep31 (`models/i26c_ep31/t2i26c_ep31.pt`), sha-pinned. No floor
  (primary contract), floored as robustness.
- Banks: (a) item bank as used in the interview tables; (b) the greedy best-static order;
  (c) entropy0 / helf / popularity heuristic orders. Concept bank DEFERRED until #74's
  niche/uniform check settles the tail question.
- Steps: k ∈ {0, 1, 2, 4} (dispersion after 0/1/2/4 answers).
- Answers: behavioral (real ratings / SEL), the standard firewall. No LLM anything.

## Controls (Hard Rule 5)
- Shuffled-answer control: permute answers across users; dispersion must collapse toward the
  k=0 value (if it doesn't, the dispersion measures bank structure, not user information).
- Existential control: a hand-built 2-cluster synthetic user pair must produce dispersion ≈ 1
  (instrument can express divergence at all).
- Canonical snap: k=0 recommendation must reproduce the known prior (Spearman ≈ 0.96 vs pop).

## Metric + MDE
Descriptive census — no MDE gate on dispersion itself. The MDE (0.004 NDCG@10, paired bootstrap)
applies only when a dispersion cell is later paired with a gap measurement (splitter probe).

## Shortcuts and their non-lossy alternatives
- Scoring E[ΔNDCG] for every (user, q) pair at k=4 is the expensive cell. Non-lossy: batch the
  candidate-question fold (the set encoder is order-invariant; fold once per (user, q)); stream
  users. NO user subsampling without sign-off (Hard Rule 1). If it is still too slow, STOP and ask
  — do not silently cap the candidate set.

## Output
`docs/results/DISPERSION_CENSUS.md`: dispersion per (bank × step), both metrics, controls, and one
sentence per bank: "under this bank, adaptivity had X% of users sharing the same best next
question at step 1." Update `PAPER_B_EVIDENCE_POLICY.md` SEEN list same session.
