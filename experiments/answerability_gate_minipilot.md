# Answerability GATE — ML-25M MINI-PILOT (plumbing sanity, NOT a result)

Status: DONE (2026-07-06). 15 ML-25M users. This is a PLUMBING check of `scripts/llm_answerability_gate.py`
before the owner-gated 300-user run. Numbers below are eyeball sanity on n=15 — **not** the fuel-gate
verdict. STOP here for review per `answerability_PLAN.md` §F.

## Config (pinned)
- Model: `gpt-5.4-mini` → **resolved snapshot `gpt-5.4-mini-2026-03-17`** (recorded from `resp.model`).
- Temperature **0.0** (pinned). `response_format=json_object` with an `"answers"` list key (fixes the
  pilot's json_object-vs-list mismatch).
- Dataset: ML-25M. Arena/meta `data/movielens/.cache/ml25m/meta.npz` (ni=18430, nu=162541).
- **Answerer split (§5): ONE FIXED split, `ANSWERER_SPLIT_SEED=123`** = the I2 `ml25m_arena` convention
  (per-user shuffle ALL rated items, first half = KNOWN/profile the answerer sees, second half = held-out
  targets; ≥6 rated). Cached to `.cache/instrument2/answerability_answerer_split.json`.
- User sampling `USER_SAMPLE_SEED=0`, 15 users from the te cohort, stratified by profile-size tercile ×
  dominant genre (NOT 1-per-genre). Cells hit: sizes 0/1/2 across Action, Children, Crime, Comedy,
  Mystery, IMAX, Adventure, Romance, Sci-Fi, Drama. Profile sizes N spanned 20 → 806.
- Question bank per user: 39 genome-tag concepts (breadth-tiered: broad/mid/niche) + 30 designed items
  (popularity strata × genre) + 6 taste-adjacent items = 75 **interview** questions; PLUS validity battery
  (8 held-out-rated + 8 matched-never-rated) and 8 masked-rated items. Interview bank EXCLUDES held-out
  targets (Hole 6); validity + masked are SEPARATE passes.
- 2 LLM calls/user (1 answerability, 1 masked-rating). 30 calls total.

## Cost (measured from response.usage — the hard number)
- **$0.00219 per call**; 30 calls = **$0.066** total; 80,901 prompt + 22,749 completion tokens; 2.2 min wall.
- $ uses documented rate guesses (in $0.25/M, out $2.00/M) — **tokens are exact**, $ scales linearly if the
  billed rate differs. Extrapolation: 300-user gate ≈ 600 calls ≈ **~$1.3** (matches the prereg "low single-$").

## Plumbing sanity (n=15 — do not cite)
- **JSON parses cleanly**: 91/91 (or 89/89) answers returned every user; 8/8 mask preds every user. No parse fails.
- **Heterogeneity** (interview answer-rate, maybe→refuse): mean 0.644, sd 0.077, range 0.53–0.76 (spread 0.23).
  Real per-user variation present. Full mixed-effects model is a stub w/ TODO.
- **Taste-tracking**: point-biserial r(taste-match, can_answer)=**0.14** over 585 concept judgments; mean
  taste-match 0.73 when "yes" vs 0.58 when "no" — **correct sign**.
- **VALIDITY GAP (the key new machinery — Hole 2)** held-out-RATED minus pop+genre-MATCHED never-rated,
  within popularity tier:
  | tier | rated-rate | never-rate | gap | n/side |
  |---|---|---|---|---|
  | famous | 0.848 | 0.772 | **+7.6pt** | 79 |
  | moderate | 0.529 | 0.176 | **+35.3pt** | 34 |
  | obscure | 0.00 | 0.00 | 0.0pt | 4 |
  **Sign is POSITIVE and largest in the diagnostic mid tier** — exactly the review's predicted shape
  (famous never-rated are probably known too → small gap; obscure n=4 too thin to read). Machinery computes
  cleanly and produces sane, interpretable numbers.
- **Masked-item MAE (Hole 1)**: LLM rating predictor MAE **0.625 stars**, pred/true corr **0.62** over 120
  masked items. Sane predictor; supplies the counterfactual-value fidelity σ (Hole 4). CF/EASE cross-check
  is a full-gate TODO.
- **G2 exploitability (Hole 3)**: STUB — requires the instrument fold-in loop on the cached grid; computed
  in the 300-user gate, not the mini-pilot.

## Verdict
Machinery works end-to-end: fixed cached answerer split, stratified sampling, 3 passes (interview /
validity / masked), clean JSON, usage-logged cost, and all four metric families (heterogeneity,
taste-tracking, validity-gap, masked-MAE) compute cleanly with the right signs at n=15. Per-call cost
measured ($0.0022). **No blocker.** Ready for owner review before the 300-user gate spend.

Artifacts: `scripts/llm_answerability_gate.py`, `experiments/answerability_gate_minipilot_results.json`,
`.cache/instrument2/answerability_answerer_split.json`.
