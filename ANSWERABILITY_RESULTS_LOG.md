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

## TODO REMINDERS (owner)
- [ ] MAIN STUDY (judge full concepts + large item sample, FIT P(answerable|features), validate on held-out users) BEFORE agent.
- [ ] Close 2 skipped cross-checks: independent-CF (EASE on known portion) + cross-family LLM (2nd provider key).
- [ ] Then adaptive agent vs best static, measured vs G2 upper bound.
