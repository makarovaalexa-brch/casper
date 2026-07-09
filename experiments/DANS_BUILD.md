# D-ANS BUILD LOG

Contract: DESIGN_SHEET_DISTILLED_ANSWERER.md (signed 2026-07-09). NO LLM calls. Seeds deterministic (123). Script `scripts/dans_build.py`.

## G-pre -- reputation-confidence prevalence (informational; NO correction, author decision)

Operationalization: an item cell is reputation-confidence if it is `know_well` on an UNRATED item with NO nearby engagement -- co-knowledge proximity (max cosine to the rated set in the population kmap knowledge space) < 0.30 AND genre affinity below the user's median.

Overall: 124387 item cells; 86286 know_well-on-unrated; 284 reputation-confidence (**0.003** of know_well-unrated; 0.002 of all item cells).

| within-bank pop tercile | item cells | know_well(unrated) | reputation-conf | rep/know_well | rep/all |
|---|--:|--:|--:|--:|--:|
| low | 43615 | 20833 | 139 | 0.007 | 0.003 |
| mid | 42229 | 29999 | 109 | 0.004 | 0.003 |
| high | 38543 | 35454 | 36 | 0.001 | 0.001 |

## FIT -- per-channel ordered logistic (LOUO over 173; final refit on all 173)

Knowledge = proportional-odds ordinal {no_clue<rough_idea<know_well}; value = ordinal {hated<meh<liked<loved} on cells with knowledge!=no_clue (rated items pass through the real rating -- no model). Saturation exponent gamma (concept rated-member count, entity rated-filmography count) chosen by full-data log-likelihood over a grid. Coefficients standardized.

| model | feature | coefficient |
|---|---|--:|
| know:concept | sat_ratedmembers | +0.609 |
| know:concept | tag_logmemb | -0.043 |
| know:concept | tag_logpw | +0.929 |
| know:concept | taste_align | +0.195 |
| know:concept | saturation_gamma | +0.250 |
| know:entity | sat_ratedfilmo | +0.361 |
| know:entity | frac_filmo | +0.204 |
| know:entity | entity_logpop | +0.098 |
| know:entity | taste_align | +0.206 |
| know:entity | saturation_gamma | +0.500 |
| know:item | coknow_mean | +0.391 |
| know:item | coknow_max | +0.044 |
| know:item | genre_align | +0.097 |
| know:item | fame_pr | +0.223 |
| know:item | fame_logcnt | +1.010 |
| know:item | decade_align | -0.098 |
| value:concept | member_mean_ctr | +0.738 |
| value:concept | has_rated_member | -0.465 |
| value:concept | taste_align | +0.265 |
| value:concept | tag_logpw | +0.801 |
| value:concept | user_cmean | +0.351 |
| value:entity | member_mean_ctr | +1.370 |
| value:entity | has_rated_member | -0.915 |
| value:entity | taste_align | -0.092 |
| value:entity | entity_logpop | +0.041 |
| value:entity | user_cmean | +0.516 |
| value:item | genre_align | +0.173 |
| value:item | fame_pr | -0.567 |
| value:item | fame_logcnt | +1.796 |
| value:item | user_cmean | +0.534 |

Sign sanity (intuition): concept_engagement_pos=True, entity_engagement_pos=True, item_coknow_pos=True, item_fame_pos=True, concept_saturation_lt1=True, entity_saturation_lt1=True.

## G1 -- AGREEMENT (leave-one-user-out)

**Pre-registered**: knowledge accuracy must beat the population-rate (per-stratum majority) baseline overall and in the niche stratum; kappa>0 (niche decisive). Value MAE vs LLM stars per channel (LLM masked MAE ~0.70 ref).

### Knowledge (accuracy / Cohen kappa vs LLM; baseline = per-stratum majority class)

| channel | stratum | n | acc(model) | acc(base) | kappa | verdict |
|---|---|--:|--:|--:|--:|:--:|
| concept | all | 195144 | 0.578 | 0.386 | 0.354 | beat |
| concept | low (NICHE) | 65048 | 0.555 | 0.517 | 0.176 | beat |
| concept | mid | 65048 | 0.494 | 0.436 | 0.151 | beat |
| concept | high | 65048 | 0.684 | 0.674 | 0.143 | beat |
| entity | all | 86500 | 0.557 | 0.525 | 0.127 | beat |
| entity | low (NICHE) | 28891 | 0.484 | 0.458 | 0.069 | beat |
| entity | mid | 28891 | 0.627 | 0.636 | 0.094 | tie/lose |
| entity | high | 28718 | 0.560 | 0.480 | 0.183 | beat |
| item | all | 124387 | 0.726 | 0.694 | 0.292 | beat |
| item | low (NICHE) | 41521 | 0.567 | 0.504 | 0.148 | beat |
| item | mid | 41530 | 0.695 | 0.690 | 0.072 | beat |
| item | high | 41336 | 0.917 | 0.917 | -0.000 | tie/lose |

### Value (expected-value MAE in stars vs LLM labels; LLM masked-MAE ~0.70 ref)

| channel | n | MAE(exp) | MAE(base=user-mean) |
|---|--:|--:|--:|
| concept | 144547 | 0.536 | 0.631 |
| entity | 71309 | 0.395 | 0.439 |
| item | 123226 | 0.411 | 0.512 |

**G1 knowledge verdict**: beats population-rate baseline overall in all channels and is non-worse in the niche stratum = True.

## G2 -- FUEL REPRODUCTION (decisive)

**Pre-registered**: each Stage-A fuel statistic's synthetic value must lie within the real 95% CI (or CIs overlap). Sampling only.

| statistic | real [95% CI] | synth [95% CI] | match |
|---|---|---|:--:|
| A1 concept ICC(k>=1) | +0.034 [+0.026,+0.041] | +0.012 [+0.009,+0.014] | MISS |
| A1 concept ICC(k>=2) | +0.052 [+0.042,+0.062] | +0.022 [+0.017,+0.026] | MISS |
| A1 concept base(k>=1) | +0.741 [+0.741,+0.741] | +0.743 [+0.743,+0.743] | MISS |
| A1 attribute ICC(k>=1) | +0.092 [+0.075,+0.108] | +0.006 [+0.004,+0.008] | MISS |
| A1 attribute ICC(k>=2) | +0.080 [+0.066,+0.095] | +0.020 [+0.013,+0.026] | MISS |
| A1 attribute base(k>=1) | +0.824 [+0.824,+0.824] | +0.822 [+0.822,+0.822] | MISS |
| A1 item ICC(k>=1) | +0.016 [+0.011,+0.021] | +0.001 [+0.000,+0.001] | MISS |
| A1 item ICC(k>=2) | +0.137 [+0.114,+0.159] | +0.004 [+0.003,+0.005] | MISS |
| A1 item base(k>=1) | +0.991 [+0.991,+0.991] | +0.991 [+0.991,+0.991] | MISS |
| A2 item-low near-far(k2) | +0.009 [-0.004,+0.023] | +0.051 [+0.041,+0.061] | MISS |
| A2 item-mid near-far(k2) | +0.028 [+0.017,+0.039] | +0.032 [+0.023,+0.041] | OK |
| A2 item-high near-far(k2) | -0.007 [-0.014,+0.001] | +0.012 [+0.007,+0.018] | MISS |
| A2 concept near-far(k1) | +0.105 [+0.095,+0.115] | +0.126 [+0.120,+0.133] | MISS |
| A3 concept-niche k>=1 | +0.528 [+0.528,+0.528] | +0.544 [+0.544,+0.544] | MISS |
| A3 item-niche k>=2 | +0.475 [+0.475,+0.475] | +0.469 [+0.469,+0.469] | MISS |
| A3 concept-broad k>=1 | +0.928 [+0.928,+0.928] | +0.915 [+0.915,+0.915] | MISS |
| A3 item-broad k>=2 | +0.917 [+0.917,+0.917] | +0.915 [+0.915,+0.915] | MISS |
| A3 overall rough share | +0.374 [+0.374,+0.374] | +0.374 [+0.374,+0.374] | MISS |

**G2 verdict**: decisive (ICC + validity-gap) 1/10 within CI -> FAIL.

## G2 FAIL (v1.0 model) -- diagnosis + ITERATION-1 (the one documented iteration)

**v1.0 G2 result (see table above): FAIL, 1/10 decisive.** What matched: base rates (0.741/0.743,
0.824/0.822, 0.991/0.991), strata rates (within 0.016), rough share (0.374/0.374 exact), item-mid
validity gap. What was LOST: the per-user ICC in EVERY channel (item k>=2: real 0.137 vs synth 0.004;
attribute k>=1: 0.092 vs 0.006; concept k>=2: 0.052 vs 0.022) -- independent per-cell sampling with
no user-level terms destroys the user random intercept. This is exactly the author's warned failure
("sampling without signal = noise") and the gate caught it.

**Diagnosis (measured):** regressing per-user real answer rates on observable known-half user
statistics gives R2 = 0.44-0.60 (concept), 0.12-0.20 (attribute), 0.09-0.10 (item). So observable
features can restore part of the concept ICC but the item/attribute user heterogeneity is largely
IDIOSYNCRATIC (consistent with the 0.725 rated-ness predictability ceiling) -- features alone
mathematically cannot reach ICC 0.137.

**ITERATION-1 (documented, the one allowed):**
1. FEATURES (within sheet section 3 consumption family): user-level known-half consumption statistics
   added to all three knowledge channels -- log profile size, genre entropy, number of distinct
   decades, rating dispersion (u_logprofile, u_genre_entropy, u_n_decades, u_rating_sd).
2. MODEL-FORM AMENDMENT (**flagged for author review**): a per-channel FITTED user random intercept
   -- empirical-Bayes per-user shifts on the fixed-effects ordinal fit, between-user variance
   sampling-corrected -> sigma_u per knowledge channel (3 extra parameters). Generation DRAWS
   b_u ~ N(0, sigma_u) once per synthetic user per channel (seeded), shared across that user's cells.
   Reading of the signed sheet: this is the fitted distribution carrying uncertainty at the USER level
   (a variance component of the ordered logistic), NOT injected ad-hoc noise; but it exceeds a pure
   "feature iteration", so it is called out here for explicit author sign-off. E5-clean: real users'
   own estimated intercepts are never used; only random draws from the fitted population distribution.

Also corrected (reporting only, verdict unchanged): the decisive-set filter now matches the printed
pre-registration exactly (A1 ICCs; A2 concept near-far k1 + item-MID near-far k2; A3 rates), and
zero-width point statistics use a documented |diff|<=0.03 tolerance (v1.0 marked exact matches like
0.374 vs 0.374 as MISS). Under the corrected prereg the v1.0 run still FAILS (all 6 ICCs miss).

## FIT -- per-channel ordered logistic (LOUO over 173; final refit on all 173)

Knowledge = proportional-odds ordinal {no_clue<rough_idea<know_well}; value = ordinal {hated<meh<liked<loved} on cells with knowledge!=no_clue (rated items pass through the real rating -- no model). Saturation exponent gamma (concept rated-member count, entity rated-filmography count) chosen by full-data log-likelihood over a grid. Coefficients standardized.

| model | feature | coefficient |
|---|---|--:|
| know:concept | sat_ratedmembers | +0.531 |
| know:concept | tag_logmemb | -0.040 |
| know:concept | tag_logpw | +0.983 |
| know:concept | taste_align | +0.199 |
| know:concept | u_logprofile | +0.069 |
| know:concept | u_genre_entropy | -0.038 |
| know:concept | u_n_decades | +0.075 |
| know:concept | u_rating_sd | +0.021 |
| know:concept | saturation_gamma | +0.250 |
| know:entity | sat_ratedfilmo | +0.345 |
| know:entity | frac_filmo | +0.207 |
| know:entity | entity_logpop | +0.102 |
| know:entity | taste_align | +0.215 |
| know:entity | u_logprofile | +0.037 |
| know:entity | u_genre_entropy | -0.119 |
| know:entity | u_n_decades | +0.079 |
| know:entity | u_rating_sd | +0.074 |
| know:entity | saturation_gamma | +0.350 |
| know:item | coknow_mean | +0.412 |
| know:item | coknow_max | -0.032 |
| know:item | genre_align | +0.104 |
| know:item | fame_pr | +0.229 |
| know:item | fame_logcnt | +1.027 |
| know:item | decade_align | -0.060 |
| know:item | u_logprofile | -0.081 |
| know:item | u_genre_entropy | -0.130 |
| know:item | u_n_decades | +0.246 |
| know:item | u_rating_sd | +0.093 |
| value:concept | member_mean_ctr | +0.738 |
| value:concept | has_rated_member | -0.465 |
| value:concept | taste_align | +0.265 |
| value:concept | tag_logpw | +0.801 |
| value:concept | user_cmean | +0.351 |
| value:entity | member_mean_ctr | +1.370 |
| value:entity | has_rated_member | -0.915 |
| value:entity | taste_align | -0.092 |
| value:entity | entity_logpop | +0.041 |
| value:entity | user_cmean | +0.516 |
| value:item | genre_align | +0.173 |
| value:item | fame_pr | -0.567 |
| value:item | fame_logcnt | +1.796 |
| value:item | user_cmean | +0.534 |

Sign sanity (intuition): concept_engagement_pos=True, entity_engagement_pos=True, item_coknow_pos=True, item_fame_pos=True, concept_saturation_lt1=True, entity_saturation_lt1=True.

## G1 -- AGREEMENT (leave-one-user-out)

**Pre-registered**: knowledge accuracy must beat the population-rate (per-stratum majority) baseline overall and in the niche stratum; kappa>0 (niche decisive). Value MAE vs LLM stars per channel (LLM masked MAE ~0.70 ref).

### Knowledge (accuracy / Cohen kappa vs LLM; baseline = per-stratum majority class)

| channel | stratum | n | acc(model) | acc(base) | kappa | verdict |
|---|---|--:|--:|--:|--:|:--:|
| concept | all | 195144 | 0.577 | 0.386 | 0.353 | beat |
| concept | low (NICHE) | 65048 | 0.555 | 0.517 | 0.180 | beat |
| concept | mid | 65048 | 0.491 | 0.436 | 0.144 | beat |
| concept | high | 65048 | 0.684 | 0.674 | 0.144 | beat |
| entity | all | 86500 | 0.556 | 0.525 | 0.131 | beat |
| entity | low (NICHE) | 28891 | 0.483 | 0.458 | 0.069 | beat |
| entity | mid | 28891 | 0.626 | 0.636 | 0.111 | tie/lose |
| entity | high | 28718 | 0.560 | 0.480 | 0.184 | beat |
| item | all | 124387 | 0.727 | 0.694 | 0.297 | beat |
| item | low (NICHE) | 41521 | 0.566 | 0.504 | 0.146 | beat |
| item | mid | 41530 | 0.698 | 0.690 | 0.096 | beat |
| item | high | 41336 | 0.917 | 0.917 | 0.000 | tie/lose |

### Value (expected-value MAE in stars vs LLM labels; LLM masked-MAE ~0.70 ref)

| channel | n | MAE(exp) | MAE(base=user-mean) |
|---|--:|--:|--:|
| concept | 144547 | 0.536 | 0.631 |
| entity | 71309 | 0.395 | 0.439 |
| item | 123226 | 0.411 | 0.512 |

**G1 knowledge verdict**: beats population-rate baseline overall in all channels and is non-worse in the niche stratum = True.

## G2 -- FUEL REPRODUCTION (decisive)

**Pre-registered**: each Stage-A fuel statistic's synthetic value must lie within the real 95% CI (or CIs overlap). Sampling only.

| statistic | real [95% CI] | synth [95% CI] | match |
|---|---|---|:--:|
| A1 concept ICC(k>=1) | +0.034 [+0.026,+0.041] | +0.029 [+0.023,+0.034] | OK |
| A1 concept ICC(k>=2) | +0.052 [+0.042,+0.062] | +0.042 [+0.033,+0.050] | OK |
| A1 concept base(k>=1) | +0.741 [+0.741,+0.741] | +0.739 [+0.739,+0.739] | OK |
| A1 attribute ICC(k>=1) | +0.092 [+0.075,+0.108] | +0.055 [+0.042,+0.067] | MISS |
| A1 attribute ICC(k>=2) | +0.080 [+0.066,+0.095] | +0.070 [+0.056,+0.086] | OK |
| A1 attribute base(k>=1) | +0.824 [+0.824,+0.824] | +0.807 [+0.807,+0.807] | OK |
| A1 item ICC(k>=1) | +0.016 [+0.011,+0.021] | +0.012 [+0.007,+0.020] | OK |
| A1 item ICC(k>=2) | +0.137 [+0.114,+0.159] | +0.115 [+0.093,+0.140] | OK |
| A1 item base(k>=1) | +0.991 [+0.991,+0.991] | +0.989 [+0.989,+0.989] | OK |
| A2 item-low near-far(k2) | +0.009 [-0.004,+0.023] | +0.039 [+0.030,+0.049] | MISS |
| A2 item-mid near-far(k2) | +0.028 [+0.017,+0.039] | +0.025 [+0.016,+0.033] | OK |
| A2 item-high near-far(k2) | -0.007 [-0.014,+0.001] | +0.011 [+0.006,+0.016] | MISS |
| A2 concept near-far(k1) | +0.105 [+0.095,+0.115] | +0.125 [+0.118,+0.131] | MISS |
| A3 concept-niche k>=1 | +0.528 [+0.528,+0.528] | +0.543 [+0.543,+0.543] | OK |
| A3 item-niche k>=2 | +0.475 [+0.475,+0.475] | +0.502 [+0.502,+0.502] | OK |
| A3 concept-broad k>=1 | +0.928 [+0.928,+0.928] | +0.911 [+0.911,+0.911] | OK |
| A3 item-broad k>=2 | +0.917 [+0.917,+0.917] | +0.905 [+0.905,+0.905] | OK |
| A3 overall rough share | +0.374 [+0.374,+0.374] | +0.363 [+0.363,+0.363] | OK |

**G2 verdict**: decisive (ICC + validity-gap) 11/13 within CI -> FAIL.

## G3 -- ERROR PROFILE (rated cells, value passthrough disabled)

Sampled value vs real rating on 6962 rated cells: MAE=0.754 stars, |err| dispersion=0.674, pred-true corr=0.177. LLM reference MAE~0.70, corr~0.5. Too-clean flag=False. **G3 verdict**: PASS (not measurably too clean; no calibration applied).

## STOP STATE (2026-07-09) -- build halted at G2 per contract

**Verdict chain**: G-pre measured (0.3% impurity). FIT clean (LOUO, ~57 params, all sign checks pass).
G1 PASS (all channels beat the population-rate baseline overall AND in the niche strata). G2 v1.0
FAIL 1/10 decisive (user-ICC collapse, diagnosed). ITERATION-1 (documented above) recovered 11/13
decisive -- item ICC k>=2 0.115 vs real 0.137 [0.114,0.159] (overlap OK), concept ICCs OK -- but two
decisive stats still miss:
1. **A1 attribute ICC(k>=1)**: synth 0.055 [0.042,0.067] vs real 0.092 [0.075,0.108]. A single
   per-channel random intercept cannot carry the real data's LARGER user variance at the
   knows-of-it margin than at the know-well margin (k>=2 matches: 0.070 vs 0.080 OK). Named fix:
   per-cut user random effects for the entity channel (+2 params) -- REQUIRES AUTHOR SIGN-OFF.
2. **A2 concept near-far(k1)**: synth +0.125 [+0.118,+0.131] vs real +0.105 [+0.095,+0.115]. The
   single fitted taste_align fixed effect slightly OVERSTATES the within-user taste gradient (part
   of the real align-answerability correlation is between-user). Named fix: fit the within-user
   gap directly / shrink taste_align; also the non-decisive item-low/high near-far misses point to
   a genre_align x fame-band interaction. REQUIRES AUTHOR SIGN-OFF.

G3 (informational, run for this review packet; NOT a gate-pass claim): sampled unrated-value on the
6,962 rated cells (passthrough disabled) MAE 0.754 vs LLM 0.70 -- magnitude matches; corr 0.18 vs
LLM 0.5 -- the synthetic unrated-value channel is NOISIER than the LLM, not cleaner (too-clean flag
False). In real generation rated cells pass through the true rating, so this bounds only unrated
value fidelity.

**Not run (contract: only if G1-G3 pass)**: 20k generation, G4, 162k extension.

**Options for the author** (per sheet section 1 "gates fail in specific strata -> decide then"):
- (a) Sign the two named fixes above as ITERATION-2 (both are small, parameter-countable, and
  target exactly the two missing stats); or
- (b) Accept the arena restricted to the passing strata (item + concept ICCs, all base/strata rates,
  item-mid validity gap all reproduce; the attribute channel and the concept taste-gradient
  magnitude are the two flagged caveats); or
- (c) Option D (wide-sparse LLM judging, ~$20-40) if neither is acceptable.

Artifacts: .cache/dans/{g_pre,g1,g2,g3,sign_checks}.json, models.json (fitted coefficients +
sigma_u), louo.npz (LOUO predictions), logs .cache/dans/*_iter1.log. Scripts
scripts/dans_build.py + scripts/dans_stages.py.
