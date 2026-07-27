# K-map -- offline validation of a LEARNED answerability/knowledge model

Date 2026-07-08. Scripts `scripts/kmap_build.py` + `scripts/kmap_validate.py`. NO LLM calls; deterministic (seed 0); local compute. Artifacts in `.cache/instrument2/`.

## Recipe
- **Item knowledge embeddings** (population-scale, learned): logistic MF of the binary KNOWN matrix, P(u knows j)=sigmoid(b_j + k_u . e_j), d=16, item intercepts b_j + embeddings e_j. Trained on 40000 ML-25M training users (downsampled to 40000 of 161541 trU users (seed 0)), 6116834 positives, 11502 items (>= 20 raters). ALL 300 study-user rows excluded (leak=0).
- **Coverage**: probe bank 160/160 (1.000); judged bank 3007/3411 (0.882).
- **Online inference** (no hand rules): Bayesian logistic MAP (Newton, 12 it) for k_u AND a per-user intercept alpha_u on FIXED e_j/b_j; priors k_u~N(0,1.0^2 I), alpha_u~N(0,2.0^2). Surprise-weighting falls out of the likelihood only.
- **Study cohort**: 298 users; judged item-cells/user mean 114.8 (min 52, max 116); judged concept-cells/user mean 200.0. Measured labels use ONLY LLM-judged cells (pmodel-synthesized labels NEVER used as ground truth -- circularity firewall).

## Arena: STRUCTURAL (knows = rated in known half; universe = 160-item probe bank)

### Held-out AUC vs t (mean over users)

| method | t=0 | t=1 | t=2 | t=4 | t=8 | t=16 |
|---|---|---|---|---|---|---|
| full (b_j + alpha_u + k_u.e_j) | 0.626 | 0.628 | 0.652 | 0.654 | 0.669 | 0.683 |
| b_j only (= popularity) | 0.626 | 0.626 | 0.625 | 0.623 | 0.623 | 0.621 |
| b_j + alpha_u (intercept) | 0.626 | 0.626 | 0.625 | 0.623 | 0.623 | 0.621 |
| genre-ghat rule (a4, to beat) | 0.596 | 0.588 | 0.596 | 0.606 | 0.607 | 0.613 |
| pmodel + TRUE genre_match (PRIV ceiling) | 0.699 | 0.699 | 0.699 | 0.697 | 0.696 | 0.695 |

### Gates
- **G1 (monotone in t, within noise):** PASS (net AUC t16 vs t2 = +0.033[+0.020,+0.047]; no significant consecutive decrease).
- **G2 (full beats popularity by >=0.03 at t=8):** PASS (delta +0.046[+0.031,+0.060]).
- **G3 (full beats genre-ghat at t=8):** PASS (delta +0.062[+0.042,+0.081]).
- **G4 (HEADLINE: user term k_u.e_j adds over b_j alone):** PASS -- t=4: +0.031[+0.019,+0.043]; t=8: +0.046[+0.031,+0.060]; t=16: +0.062[+0.046,+0.077].

### G4 three-way (AUC on held-out judged/bank cells)

| term | t=4 | t=8 | t=16 |
|---|---|---|---|
| b_j only | 0.623 | 0.623 | 0.621 |
| b_j + alpha_u | 0.623 | 0.623 | 0.621 |
| full (b+alpha+k.e) | 0.654 | 0.669 | 0.683 |

## Arena: MEASURED (knows = LLM judged YES; universe = full judged battery, judged cells only)

### Held-out AUC vs t (mean over users)

| method | t=0 | t=1 | t=2 | t=4 | t=8 | t=16 |
|---|---|---|---|---|---|---|
| full (b_j + alpha_u + k_u.e_j) | 0.915 | 0.914 | 0.914 | 0.916 | 0.917 | 0.915 |
| b_j only (= popularity) | 0.915 | 0.913 | 0.913 | 0.914 | 0.910 | 0.907 |
| b_j + alpha_u (intercept) | 0.915 | 0.913 | 0.913 | 0.914 | 0.910 | 0.907 |
| genre-ghat rule (a4, to beat) | 0.912 | 0.910 | 0.911 | 0.914 | 0.914 | 0.913 |
| pmodel + TRUE genre_match (PRIV ceiling) | 0.922 | 0.921 | 0.920 | 0.921 | 0.918 | 0.915 |

### Gates
- **G1 (monotone in t, within noise):** PASS (net AUC t16 vs t2 = +0.001[-0.003,+0.004]; no significant consecutive decrease).
- **G2 (full beats popularity by >=0.03 at t=8):** FAIL (delta +0.007[+0.004,+0.010]).
- **G3 (full beats genre-ghat at t=8):** FAIL (delta +0.003[-0.000,+0.007]).
- **G4 (HEADLINE: user term k_u.e_j adds over b_j alone):** PASS -- t=4: +0.002[+0.000,+0.004]; t=8: +0.007[+0.004,+0.010]; t=16: +0.008[+0.004,+0.011].

### G4 three-way (AUC on held-out judged/bank cells)

| term | t=4 | t=8 | t=16 |
|---|---|---|---|
| b_j only | 0.914 | 0.910 | 0.907 |
| b_j + alpha_u | 0.914 | 0.910 | 0.907 |
| full (b+alpha+k.e) | 0.916 | 0.917 | 0.915 |

## Concept unification (items are points, concepts are regions -- ONE learned space)

Concept embedding e_c = relevance-weighted centroid of member items' e_j (genome membership); b_c = population concept answer-rate logit. Validated on 200 judged concepts x 298 users (LLM concept labels).

| method | t=0 | t=1 | t=2 | t=4 | t=8 | t=16 |
|---|---|---|---|---|---|---|
| full (b_c + alpha_u + k_u.e_c) | 0.983 | 0.983 | 0.983 | 0.982 | 0.982 | 0.981 |
| b_c only (concept popularity) | 0.983 | 0.983 | 0.983 | 0.982 | 0.982 | 0.981 |

**G4-concept (user term adds over b_c):** t=4: +0.000[-0.000,+0.000] FAIL; t=8: +0.000[-0.000,+0.000] FAIL; t=16: +0.000[-0.000,+0.000] FAIL.

**Caveat (honest):** concept answerability is a near-ceiling POPULATION property here -- LLM concept yes-rate = 0.932, so concept 'popularity' b_c already scores AUC 0.981 and the per-user held-out set is class-imbalanced (few 'no' concepts; 282/298 users have both classes at t=8), making AUC a LOW-POWER instrument for a user-specific concept tilt. The unification holds representationally (e_c lives in the same learned space, norms ~1.5), but concepts do NOT carry a measurable user-specific answerability signal over their population rate -- concept answerability is population-driven, unlike STRUCTURAL item knowledge which is strongly user-specific.

## Intercept vs embedding ablation

alpha_u is a per-user CONSTANT, so within a user it is rank-invariant: AUC(b_j only) == AUC(b_j + alpha_u) by construction (structural t=8 0.623 vs 0.623; measured 0.910 vs 0.910). **The entire held-out per-user AUC lift comes from the embedding direction k_u** (structural full 0.669 vs pop 0.623; measured full 0.917 vs pop 0.910); alpha_u carries the per-user base-rate LEVEL (matters for calibration, not for within-user ranking).

## Example trajectories (measured-arena held-out AUC by t)

| user type | u | n_known | n_judged | bank pos-rate | t=0 | t=1 | t=2 | t=4 | t=8 | t=16 |
|---|---|---|---|---|---|---|---|---|---|---|
| mainstream | 72975 | 702 | 115 | 0.481 | 0.886 | 0.899 | 0.895 | 0.912 | 0.927 | 0.939 |
| niche_heavy | 8777 | 10 | 115 | 0.0 | 0.943 | 0.938 | 0.935 | 0.938 | 0.933 | 0.929 |
| sparse | 39510 | 9 | 115 | 0.025 | 0.877 | 0.872 | 0.872 | 0.864 | 0.894 | 0.912 |

## Calibration note
Platt/isotonic mapping is AUC-invariant (monotone), so all gates above are unaffected by calibration; per-arena base rates (structural << measured 0.73) motivate recalibrating the PROBABILITY level before any thresholded deployment. Calibration to arena base rate is a level-only transform applied at deploy time, fit on a train split of grid users.

## ASSUMPTIONS / judgment calls
1. KNOWN(u,j)=1 iff u rated j (implicit); training users = 40000 downsampled trU (study users disjoint from trU, leak=0). Items >= 20 raters; uncovered items get a popularity-logit fallback intercept (e_j=0).
2. STRUCTURAL universe = 160-item top-coverage probe bank; reveal order = the committed s3 schedule (dense ids), then remaining bank items in coverage order. Label = rated-in-known-half.
3. MEASURED universe = the FULL judged battery per user (union of gate+main item cells); reveal order = popularity-descending; label = LLM 'yes'. Only judged cells used as ground truth -- pmodel-synthesized labels NEVER used (circularity firewall).
4. Online inference: MAP Newton, priors tau(k)=1.0, tau_a(alpha)=2.0; no hand rules; t=0 => full==popularity by construction.
5. genre-ghat baseline = a4's online g-hat (answered += Gmat, refused -= 0.5*phat*Gmat) with the fitted pmodel; pmodel-TRUE = pmodel with true known-half genre_match (PRIVILEGED, labelled).
6. Concept e_c = genome-relevance-weighted centroid of member items' e_j (kept items only); b_c = population concept answer-rate logit (concept 'popularity'); reveal order = concept answer-rate desc.
7. Per-user AUC requires both classes in the held-out set; users lacking both are dropped at that t (n reported). Paired per-user bootstrap BOOT=5000 seed=0.
8. Attributes (decade/genre member-sets) not yet validated -- same construction as concepts; PENDING.

