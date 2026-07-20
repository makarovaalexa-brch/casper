# DESIGN — Unified item+concept belief recommender (PrecAcc, increment objective)

Status 2026-07-20: **coded + adversarially reviewed (`scripts/train_precacc.py`), NOT yet run.** Supersedes the
learned-head VarHead (`DESIGN_SHEET_VARHEAD.md`), which failed (amortized covariance was content-blind).
Always report FULL **and** TAIL NDCG@10, scored `z @ Wd.T + decoder.BIAS` (learned bias, NEVER popb) — CLAUDE.md HARD RULE #4.

## Goal
One recommender that is (1) SOTA on full-profile items (pbC 0.4946 full / 0.3372 tail), (2) interview-native (folds
a variable partial evidence set), (3) multi-channel (items + concepts + future open, unified tokens), (4)
uncertainty-native (a posterior that SHRINKS with evidence → non-degradation + question selection), (5)
concept-preserving (adding concepts must not cost item strength). Prior attempts failed: a bolted-on Kalman on a
popb floor CRATERED under the real decoder; conjugate pooling INTO the mean cost -0.0074 (convex-avg can't do
concept intersections). Key insight: **decouple mean and covariance** — keep the nonlinear SOTA encoder as the
MEAN, put the uncertainty as a SEPARATE analytic covariance in the SAME z-space the real decoder consumes.

## Architecture
- **Mean μ = the FROZEN pbC encoder fold** of the evidence set S. Untouched → item strength preserved BY DEFINITION
  (gate G0: eval on μ is bit-identical 0.4946/0.3372).
- **Covariance = analytic PRECISION ACCUMULATOR** in z-space: `Λ(S) = Λ0 + Σ_{a∈S} α_{ch(a),lev(a)} d_a d_aᵀ`,
  `Σ = Λ⁻¹`. `Λ0 = 1/(empirical z-variance)` (the salvaged bpool2 anisotropic prior). Woodbury/Sherman-Morrison
  (low-rank in |S|), never a dense 512×512 inverse.
- **Directions `d_a`** (the leverage fix): item = `normalize(Wd[i])` (exact decoder row); concept =
  **encoder MEAN-SHIFT** `normalize(μ({c}) − μ(∅))` (`conc_dirs_meanshift.npy`, `--conc_dirs`).
  NOT the whitened member centroid — whitening strips the popularity PC that the raw item-score dirs `w_i=Wd[i]`
  live in, making concept dirs ~orthogonal to the scored space (leverage 0.021 vs 0.35 mean-shift vs 0.66 item →
  the α-fit had no gradient and zeroed concepts). Mean-shift = the direction the mean actually moves → real leverage.
- **α = ≤13 scalars** (channel {item,concept} × level {hated,meh,liked,loved,refuse}) + Λ0 diag (s0/vfloor) + τ².
  NOT per-concept (would overfit ΔNDCG/Δe noise).

## The α objective — INCREMENT calibration (fixes the "decoupling trap")
Calibrating ABSOLUTE predicted variance to ABSOLUTE residual zeros concepts (a good frozen mean leaves no absolute
leftover). Fix: calibrate REDUCTION-to-REDUCTION (no sign problem). Per train user u, held-probe item i, atom a∈S:
- ACTUAL (frozen, NO α, precomputed once & cached):
  `Δe_{u,i,a} = (s*_i − s_i(μ(S\a)))² − (s*_i − s_i(μ(S)))²`, signed; `s*_i = s_i(μ_FULL-KNOWN-HALF fold)` = the
  belief's destination (no held-liked other-half leakage). Δe>0 ⇔ atom a is informative.
- PREDICTED (Sherman-Morrison, where α lives):
  `Δv_{u,i,a}(α) = α·(w_iᵀΣ(S\a)d_a)² / (1 + α·d_aᵀΣ(S\a)d_a)`.
- LOSS: `Σ Huber(Δv − Δe)`.
The frozen mean already using a concept is now the SOURCE of credit (large Δe), not the destroyer. It's Δv↔Δe
calibration (NOT logdet-EIG); the `w_i`-weighting = MORE right than EIG (recommendation-relevant reduction).

## Pre-fit SIGN PROOF (mandatory gate, seconds, asserted before training)
At α_c=0, `dL/dα_c < 0` (loss falls as α rises off 0) iff
`G = Σ_{concept triples} clamp(Δe,−β,β)·(w_iᵀΣ(S\c)d_c)² > 0`. Compute over precomputed quantities; **assert G>0**.
Positive ⇒ α_conc provably leaves zero and cannot collapse the way the absolute-NLL did. If G≤0 → stop (bug/data).

## Selection & non-degradation (A-vs-B: build B's spine with A's Σ as the gate)
- **SELECTION = direct informativeness (B, the SPINE):** expected-ΔNDCG (the Golbandi tree already made this work)
  or `‖μ(S∪c)−μ(S)‖` weighted by answerability. Direct, non-circular, track record here, ZERO covariance-calibration
  risk. EIG ≈ E[ΔNDCG] anyway.
- **NON-DEGRADATION gate = the increment-calibrated Σ (A):** the fold gate `μ' = μ + Σ(Σ+S)⁻¹(μ_new−μ)` needs a
  DIRECTIONAL Σ; non-degradation is a WEAKER bar than selection-optimality (Σ only has to shrink sensibly per
  direction + be roughly calibrated). This is exactly what the analytic accumulator is correct-by-construction for.
- **G2 runs BOTH selection arms + random**, reports FULL+TAIL per step; success keyed on the SPINE (spine>random@q4
  tail; gated-monotone; full≥cold), Σ-greedy reported but NOT required to win. Residual risk: additive Σ
  double-counts correlated concepts (over-shrink) — tolerable for the GATE, watch only if Σ-greedy becomes policy.

## Gates
- **G0** eval_student(μ) bit-identical 0.4946/0.3372 + zero trunk drift (asserted pre/post-fit).
- **G1a** tr(Σ) strictly ↓ over k∈{0,1,2,4,8,16,32,full} and **G1b** directional (fold c drops var along d_c ≥2×
  vs other concepts) — hold BY CONSTRUCTION (additive precision + mean-shift dirs); violation-is-a-bug asserts.
- **G1c** calibration Spearman ρ(√(wᵀΣw), held-item NLL) ≥ 0.25 (must beat the failed head's 0.219).
- **G2** cold q=0..8, real decoder, FULL+TAIL, dual selection arms as above.

## COMPUTE REALITY + the sanctioned reduction (the fix this session)
Exact Δe = `Σ m(m−1)` LOO folds; avg m≈147 atoms/user over 146k users → **~3.1B encoder token-forwards** — the
exact run reached only 6k/150k users in 4h AND slowed ~10× (growing `out` list + high-m tail = memory pressure).
INFEASIBLE (~days). The α-fit is a ≤13-scalar regression → does NOT need 9.5M atoms.
- **`--max_users N`**: subsample train users for the fit (deterministic, SEEDS[0]); bounds compute + memory. Gates
  and G0 still use FULL cohorts.
- **`--max_atoms N`**: cap LOO-evaluated atoms per user (the mu(S\a) folds still use the FULL S; we only compute Δe
  for a deterministic subset of atoms a — each atom is an independent (channel,level) triple → unbiased bucket
  coverage, tames the O(m²) tail).
- Both flagged HARD RULE #1 SANCTIONED (author sign-off, given this session); cache filename `inc_<base>_u<U>_a<A>.pt`
  so a subsampled cache never collides with the exact one. 0/0 = exact.
- Planned first run (NOT yet executed, awaiting author go): `--max_users 30000 --max_atoms 40` → ~175M tokens,
  bounded memory, ~2-4h; produces the sign-proof G, the α's (now crediting concepts), G1c, and the G2 curves.

## Run command (when green-lit)
`python scripts/train_precacc.py --conc_dirs .cache/set_mn/conc_dirs_meanshift.npy --tag precacc_incr30k
  --max_users 30000 --max_atoms 40`
Then, if the α's/gates look right, the exact all-atoms cache can be produced later (efficiency fix for the memory
growth first) for the headline — but the ≤13 α's should be statistically identical.
