# ANSWERABILITY STUDY — RESULTS LOG (persisted; append as results land)

## GATE (2026-07-06) — PASSED all four pre-registered tests. ML-25M, gpt-5.4-mini-2026-03-17, $1.29, 300 users.
| Test | Cutoff | Measured | Verdict |
|---|---|---|---|
| Heterogeneity | ICC>=0.05 | ICC=0.174, perm p=0.003 | PASS |
| Taste-tracking | OR>=1.5/sd or dAUC>=0.05 | OR/sd=2.71, dAUC=0.106 (AUC .946 vs .840) | PASS |
| Validity gap (Hole 2, the key one) | >=15pt low/mid | moderate +44.1pt (n=768), obscure +10 (n=40), famous +22 | PASS |
| G2 exploitability (privileged UPPER BOUND) | CI>0 & dNDCG>=0.005 | dNDCG=0.287 CI[0.268,0.307], A=0.568 B=0.281 | PASS |
Masked-item LLM value MAE = 0.68 stars (corr 0.50, n=2400) = counterfactual-channel fidelity sigma.
Verdict: ambitious/adaptive thesis has FUEL. Files: experiments/ANSWERABILITY_GATE_RESULT.md,
answerability_gate_ml25m_results.json, cached grid .cache/instrument2/answerability_grid_ml25m.json (committed).
CAVEATS: G2 is a privileged upper bound (selector sees answerability table + true taste) not the realizable
agent prize; two cross-checks SKIPPED (independent-CF EASE inverse memory-heavy; cross-family LLM no 2nd key).

## NEXT (owner-flagged 2026-07-06): MAIN STUDY = bigger LLM run + FIT A MODEL — BEFORE the agent.
Why: the gate judged a ~600-call BANK to test fuel. The AGENT needs answerability for ANY item/concept, not
just the bank. So the main study must:
1. Judge the FULL concept set (~all breadth-tiered genome tags) x the study users, cached.
2. Judge a LARGE stratified ITEM sample (thousands, popularity x genre x taste-adjacent), cached.
3. FIT P(answerable | features) — features: log-popularity, ratings-count, decade, genre-match-to-profile,
   franchise/sequel flag; logistic; VALIDATE on held-out USERS (fit on train users only) with AUC + calibration.
   This fitted model = the scale surrogate that scores answerability everywhere the agent asks (PLAN Q-B/Q-D:
   it is LLM-DERIVED plumbing, NEVER cited as the independent structural witness).
4. THEN: masked-item validation -> primary value predictor + fidelity sigma; independent-CF cross-check
   (do the EASE-on-known-portion now, memory permitting); cross-family LLM spot-check (needs a 2nd provider key).
5. Cost: bigger but still low tens of $ (batched, cached). Model + split PINNED, reuse the gate cache.
Only AFTER this: build the adaptive coarse->granular agent, measured against the G2 headroom.

## CROSS-CHECK A (2026-07-06) — independent-CF value predictor (Hole 1, shared-prior). PASS.
Bank-restricted EASE: 4147-item universe (top-4000 popular UNION the 1300 masked targets => trivial
inverse), trained on KNOWN-portion ratings of all 162k users, gate-user held-out interactions dropped.
Predicted the SAME 2400 masked known-half items the LLM predicted (context = known-half minus masked,
mirroring the LLM profile). **EASE MAE=0.740 stars, pred/true corr=0.428, n=2400** (beats item-mean
0.777 / global-mean 0.847). LLM reference MAE=0.679/corr=0.500. VERDICT: sane independent CF predictor
=> we HAVE a 2nd value predictor for sensitivity. Does NOT gate the adaptive claim (value is
sensitivity-only per Hole-1); closes shared-prior circularity (CF shares no text-source prior w/ LLM).
Files: scripts/answerability_crosscheck_ease.py, experiments/answerability_crosscheck_ease.json.

## CROSS-CHECK B (2026-07-06) — cross-family LLM judge agreement on ANSWERABILITY. PASS.
Re-judged 30 gate users with **claude-haiku-4-5** (snapshot **claude-haiku-4-5-20251001**, temp 0) on
the BYTE-IDENTICAL replayed prompt/bank (2250 interview questions). **Answerability agreement GPT-5.4-mini
vs Haiku = 83.2%, Cohen kappa 0.668 (substantial).** Validity-gap SIGN replicates on diagnostic low/mid
tiers (moderate +3.9pt rated>never n=77; famous +20pt n=159); obscure ties at 0.0 for BOTH models (n=4,
below gate's rated_n>=5 => non-diagnostic). Haiku is systematically more conservative (yes-rate .477 vs
GPT .619) but PRESERVES the user-specific direction. VERDICT: fuel is not a one-model-family artifact.
Cost $0.44. Files: scripts/answerability_crosscheck_haiku.py, experiments/answerability_crosscheck_haiku.json,
.cache/instrument2/answerability_haiku_grid.json.

GATE ON CROSS-CHECKS: BOTH PASS -> proceeded to MAIN STUDY.

## MAIN STUDY (2026-07-06/07) — bigger LLM pass + FITTED P(answerable|features). ML-25M, gpt-5.4-mini-2026-03-17.
Same 300 study users + PINNED answerer split seed 123. Per user: 1 answerability call (ALL **200**
breadth-tiered genome tags + 50 rotating stratified pool items + 20 taste-adjacent = 270 Q) + 1 mask
call (12 known-half items). 600 calls, **$3.20** (usage sidecar exact). Cached/resumable
(.cache/instrument2/answerability_mainstudy_grid.json).
- COVERAGE: **1,716 distinct items** judged (popularity strata x genre x per-user taste-adjacent, union
  of the rotated pool) + **200 concepts**; **108,224** labeled (user,question,answerable) fit
  observations over 300 users (main-study + the gate's already-paid interview judgments, same judge/split).
- FIT P(answerable|features), logistic, features [pop_pct, log_ratings_count, decade, genre_match,
  franchise_flag, is_concept], VALIDATED on HELD-OUT USERS (fit on train users only):
  **AUC held-out-users = 0.921, grouped-5fold-CV (by user) = 0.916, ECE = 0.008** (well-calibrated across
  all 10 bins, empirical ~ predicted), base answer-rate 0.732. Standardized coef: log_ratings_count +1.55
  (dominant), is_concept +1.27, genre_match +0.63 (the user-specific fuel), decade +0.37, pop_pct -0.27
  (collinear w/ rcount), franchise +0.02. This fitted model = the SCALE SURROGATE (LLM-derived plumbing
  per Q-D; NEVER cited as the independent structural witness). Artifact: .cache/instrument2/
  answerability_pmodel.json (+ .npz).
- MASKED-VALUE (primary predictor + fidelity sigma), n=3547 across the larger sample: **LLM MAE=0.697
  stars (corr 0.508)** vs independent-CF **EASE MAE=0.745 (corr 0.453)** => fidelity sigma ~0.70 stars,
  consistent with the gate (0.68) and cross-check A (0.74). LLM = primary; EASE = sensitivity control.
Files: scripts/answerability_main_study.py, experiments/answerability_mainstudy_results.json,
.cache/instrument2/answerability_mainstudy_grid.json + _usage.json + answerability_pmodel.{json,npz}.
VERDICT: the agent has a validated, well-calibrated P(answerable) surrogate to score answerability for
ANY item/concept (AUC .92 on held-out users), plus a two-model / two-value-predictor robustness envelope.

## TODO REMINDERS (owner)
- [x] MAIN STUDY (full 200 concepts + 1.7k-item sample, FIT P(answerable|features) AUC .92 held-out users) DONE.
- [x] Close 2 skipped cross-checks: independent-CF EASE (MAE 0.74) + cross-family Haiku (agree 83%, sign holds). DONE, both PASS.
- [ ] NEXT (owner decision): build the adaptive coarse->granular agent, measured vs the G2 upper bound (dNDCG 0.287). NOT started (owner's call).
