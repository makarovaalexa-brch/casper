# ANSWERABILITY GATE - RESULT (ML-25M, 300-user run)

Date 2026-07-06. Pre-registration: `answerability_PLAN.md` sec.B (FROZEN cutoffs) + `answerability_design_review_2026-07-06.md` (6 holes). Script `scripts/llm_answerability_gate.py --gate`. This is the owner-approved decision spend.

## Config (resolved snapshot + pins)

- Model **gpt-5.4-mini** -> resolved snapshot **gpt-5.4-mini-2026-03-17**, temperature **0.0** (pinned). `maybe`->refuse (primary).

- Dataset ML-25M; meta `data/movielens/.cache/ml25m/meta.npz`; instrument `.cache/instrument2/ml25m_recvae_d512_best.pt`; belief eta=16.0, T=8.

- Answerer split seed **123** (fixed, cached); user sample seed 0; **300 users** stratified size-tercile x 5 taste-clusters.

- Question bank/user: 39 breadth-tiered concepts + 30 designed items + 6 taste-adjacent + validity (held-out-rated / matched never-rated) + masked-rated.


## Measured cost
600 LLM calls (2/user over 300 users), **$1.291** total (documented-rate estimate; per-call response.usage exact), $0.00215/call. Analyze pass (tests+G2) makes 0 LLM calls (all-cached grid), wall 12.06 min. usd = sum of 6 collection-chunk response.usage subtotals (each exact, logged to 3dp); per-chunk token totals not persisted (cumulative accumulator added after collection ran). Approx tokens ~1.6M prompt / ~0.46M completion by the mini-pilot per-call rate. usd is a documented-rate estimate (in $0.25/M, out $2.00/M); tokens exact per call.


## The four pre-registered gate tests

| Test | Cutoff | Measured | Verdict |
|---|---|---|---|
| Heterogeneity | ICC>=0.05 (+perm p<.05) | ICC=0.174, perm p=0.003 (var_user=0.693, var_q=0.000) | **PASS** |
| Taste-tracking | OR>=1.5/sd OR dAUC>=0.05 | OR/sd=2.71, dAUC=0.106 (full 0.946 vs pop 0.840) | **PASS** |
| Validity gap (Hole 2) | >=15pt low/mid tier | famous: 22.0pt (n=1589); moderate: 44.1pt (n=768); obscure: 10.0pt (n=40) | **PASS** |
| G2 exploitability (Hole 3) | CI excl 0 AND dNDCG>=0.005 | dNDCG=0.2868, CI95=[0.2678, 0.3066] (A=0.5681 B=0.2812, n=298) | **PASS** |

## VERDICT

**ADAPTIVE-CLAIM GATE = heterogeneity AND taste-tracking AND validity-gap AND G2 = PASS.**


## Method notes

- Heterogeneity: EB two-step crossed random-intercept logistic on (logpop+breadth)-adjusted answers; latent-scale variance components (statsmodels GLMM unavailable in env); ICC=var_user/(var_user+var_question+pi^2/3); permutation LRT-surrogate on user labels.

- Taste-tracking: unpenalized logistic; OR per sd from full-data fit; dAUC = grouped-5fold-CV (split by user) AUC(logpop+breadth+tastematch) - AUC(logpop-only).

- G2: answerability-AWARE per-user greedy oracle (A) vs single BLIND greedy schedule (B) on the certified ML-25M RecVAE-d512 fold-in; graded answer a=cos(z*,q), z'=z+16*a*q, T=8, NDCG@10 full; paired user bootstrap.

- Masked-rated-item MAE (Hole 1/4, fidelity sigma): LLM MAE=0.6789583333333333 stars, pred/true corr=0.5004255779445698, n=2400. Independent-CF cross-check SKIPPED (EASE on 18,430 items = a dense 18k^2 Gram inverse, memory-heavy; LLM predictor is the pre-registered primary and value is sensitivity-only per Hole-1 rule).

- Cross-family spot-check: SKIPPED (no 2nd model-family provider configured in env; only OPENAI_API_KEY present). Judge-disagreement rate remains an unmeasured uncertainty; the claim boundary already restricts to 'one LLM judge + a-priori structural rule, human study pending' (PLAN sec.E).


## Honest caveats (the de-risk-not-de-circularize boundary)

- The LLM judge removes SELF-AUTHORED circularity (no designer-written rule to exploit) but NOT model-prior dependence: the judge shares text-sources with the world RecVAE models (Hole 1 shared-prior). Claim boundary for every paper: 'results hold under two external answer models (LLM judge + a-priori structural rule); the LLM judge is validated against human answers in a pending human study.' Never 'realistic users'/'human-level'.

- G2 dNDCG is an UPPER BOUND on the adaptive prize (Selector A gets the answerability table for free; a real agent must spend turns learning it). It bounds, not estimates, the agent gain.

- Validity gap is a ONE-SIDED signal (never-rated != can't-answer): a positive low/mid-tier gap proves user-specific signal; a zero gap is ambiguous at high popularity. Only low/mid tiers are diagnostic.

- Single fixed answerer split (seed 123): split-level variance is unmeasured here (Hole 5 split-sensitivity is a separate later pass, per PLAN sec.D step 4).

