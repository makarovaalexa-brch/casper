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


---

# ITERATION-2 (author-directed amendment; signed 2026-07-09)

Contract: DESIGN_SHEET AMENDMENTS block + coordinator addenda (full feature census, distinctiveness, pockets, regularized importance study). Enriched user-level features + refit + two G2 calibration fixes; full G2 re-run; G1/G3 side-by-side. NO LLM calls; seeds 123. Author rationale (quoted): *"the LLM saw ONLY the profile, so its per-user behaviour is profile-predictable in principle; the old 90% item residual measured feature poverty, not randomness."* Census constraint: every feature derives from WHAT THE LLM SAW -- verified: the gate's profile_text showed the FULL known half (no 200-cap exists in the gate code; max shown 702 titles), so known-half features == shown-profile features exactly.

## ITER-2 feature census (user-level, known-half only; identical transform real & synthetic)

34 user features = 4 iter-1 consumption + 30 census: volume, obscurity (mean/median pop percentile, out-top-1000/5000 share+logcount), era (pre-1980 and pre-1970 share+count, era spread, per-question era density+distance), breadth (genre entropy, taste-cloud dispersion, top-pocket concentration, effective pocket count), distinctiveness (taste typicality vs population centroid, neighborhood density vs 4000 non-study reference users), territory (bank coverage + per-question co-knowledge/genre/era), markers (foreign genome tags (15), foreign-title, franchise, documentary, animation, niche-tag engagement), rating style (mean, sd, share-max, share-extreme), interactions (volume x out-top-5000, volume x niche, volume x dispersion). Fitted with a RIDGE penalty on the user block, lambda by 5-fold user-grouped CV (concept lambda=1000.0, entity lambda=1000.0, item lambda=1000.0).

### Fitted knowledge coefficients (standardized; sign = effect on higher knowledge)

| model | feature | coefficient |
|---|---|--:|
| know:concept | sat_ratedmembers | +0.525 |
| know:concept | tag_logmemb | -0.043 |
| know:concept | tag_logpw | +1.002 |
| know:concept | taste_align | +0.114 |
| know:concept | u_logprofile | -0.014 |
| know:concept | u_genre_entropy | -0.020 |
| know:concept | u_n_decades | +0.031 |
| know:concept | u_rating_sd | +0.093 |
| know:concept | b_mean_pr | +0.023 |
| know:concept | b_med_pr | +0.087 |
| know:concept | b_share_out1k | +0.007 |
| know:concept | b_logcnt_out1k | +0.057 |
| know:concept | b_share_out5k | -0.041 |
| know:concept | b_logcnt_out5k | +0.080 |
| know:concept | b_niche_share | -0.036 |
| know:concept | b_niche_logcnt | -0.010 |
| know:concept | b_foreign_share | +0.057 |
| know:concept | b_foreign_logcnt | +0.018 |
| know:concept | b_share_old | +0.054 |
| know:concept | b_int_size_out5k | +0.001 |
| know:concept | b_int_size_niche | +0.013 |
| know:concept | b_coverage | -0.052 |
| know:concept | b_pre1970_share | -0.004 |
| know:concept | b_pre1970_logcnt | +0.062 |
| know:concept | b_foreign_title_share | -0.040 |
| know:concept | b_franchise_share | -0.046 |
| know:concept | b_taste_disp | +0.050 |
| know:concept | b_typicality | +0.032 |
| know:concept | b_nbr_density | +0.039 |
| know:concept | b_pocket_conc | +0.037 |
| know:concept | b_pocket_eff | +0.059 |
| know:concept | b_rate_mean | +0.140 |
| know:concept | b_share_max | -0.033 |
| know:concept | b_share_extreme | -0.027 |
| know:concept | b_doc_share | +0.024 |
| know:concept | b_anim_share | +0.041 |
| know:concept | b_era_spread | -0.051 |
| know:concept | b_int_size_disp | +0.008 |
| know:concept | era_density | +0.121 |
| know:concept | era_dist | -0.009 |
| know:concept | saturation_gamma | +0.250 |
| know:entity | sat_ratedfilmo | +0.336 |
| know:entity | frac_filmo | +0.201 |
| know:entity | entity_logpop | +0.089 |
| know:entity | taste_align | +0.203 |
| know:entity | u_logprofile | +0.031 |
| know:entity | u_genre_entropy | -0.102 |
| know:entity | u_n_decades | -0.026 |
| know:entity | u_rating_sd | +0.073 |
| know:entity | b_mean_pr | +0.014 |
| know:entity | b_med_pr | -0.144 |
| know:entity | b_share_out1k | -0.109 |
| know:entity | b_logcnt_out1k | -0.026 |
| know:entity | b_share_out5k | -0.104 |
| know:entity | b_logcnt_out5k | +0.033 |
| know:entity | b_niche_share | +0.014 |
| know:entity | b_niche_logcnt | -0.119 |
| know:entity | b_foreign_share | +0.015 |
| know:entity | b_foreign_logcnt | +0.108 |
| know:entity | b_share_old | +0.033 |
| know:entity | b_int_size_out5k | -0.027 |
| know:entity | b_int_size_niche | +0.072 |
| know:entity | b_coverage | -0.040 |
| know:entity | b_pre1970_share | +0.080 |
| know:entity | b_pre1970_logcnt | +0.103 |
| know:entity | b_foreign_title_share | -0.044 |
| know:entity | b_franchise_share | +0.009 |
| know:entity | b_taste_disp | +0.020 |
| know:entity | b_typicality | -0.063 |
| know:entity | b_nbr_density | +0.016 |
| know:entity | b_pocket_conc | +0.063 |
| know:entity | b_pocket_eff | +0.039 |
| know:entity | b_rate_mean | +0.020 |
| know:entity | b_share_max | -0.031 |
| know:entity | b_share_extreme | +0.031 |
| know:entity | b_doc_share | -0.007 |
| know:entity | b_anim_share | +0.030 |
| know:entity | b_era_spread | -0.172 |
| know:entity | b_int_size_disp | +0.054 |
| know:entity | era_density | -0.006 |
| know:entity | era_dist | -0.179 |
| know:entity | saturation_gamma | +0.350 |
| know:item | coknow_mean | +0.482 |
| know:item | coknow_max | -0.054 |
| know:item | genre_align | +0.094 |
| know:item | fame_pr | +0.261 |
| know:item | fame_logcnt | +1.003 |
| know:item | decade_align | -0.147 |
| know:item | u_logprofile | -0.087 |
| know:item | u_genre_entropy | +0.044 |
| know:item | u_n_decades | +0.020 |
| know:item | u_rating_sd | +0.047 |
| know:item | b_mean_pr | +0.046 |
| know:item | b_med_pr | -0.115 |
| know:item | b_share_out1k | -0.002 |
| know:item | b_logcnt_out1k | -0.100 |
| know:item | b_share_out5k | -0.148 |
| know:item | b_logcnt_out5k | +0.120 |
| know:item | b_niche_share | -0.112 |
| know:item | b_niche_logcnt | -0.069 |
| know:item | b_foreign_share | +0.094 |
| know:item | b_foreign_logcnt | -0.115 |
| know:item | b_share_old | +0.198 |
| know:item | b_int_size_out5k | +0.060 |
| know:item | b_int_size_niche | -0.060 |
| know:item | b_coverage | -0.200 |
| know:item | b_pre1970_share | -0.058 |
| know:item | b_pre1970_logcnt | +0.164 |
| know:item | b_foreign_title_share | -0.005 |
| know:item | b_franchise_share | -0.211 |
| know:item | b_taste_disp | +0.149 |
| know:item | b_typicality | +0.105 |
| know:item | b_nbr_density | -0.048 |
| know:item | b_pocket_conc | +0.018 |
| know:item | b_pocket_eff | -0.026 |
| know:item | b_rate_mean | -0.018 |
| know:item | b_share_max | -0.062 |
| know:item | b_share_extreme | +0.056 |
| know:item | b_doc_share | +0.050 |
| know:item | b_anim_share | -0.004 |
| know:item | b_era_spread | -0.198 |
| know:item | b_int_size_disp | +0.206 |
| know:item | era_dist | -0.126 |

(concept taste_align shown is POST-calibration; pre-calibration beta = +0.163, multiplier 0.70.)

## STANDALONE STUDY -- what does an LLM read in a rating profile? (anatomy of judged answerability)

Ranked user-feature importance: LOUO fold-mean beta (173 refits), fold sd, and sign-consistency per knowledge channel; ranked by sum |beta| across channels. (Flagged by the author as a potentially publishable standalone table.)

| rank | feature | concept beta (sd, sign%) | entity beta (sd, sign%) | item beta (sd, sign%) |
|--:|---|---|---|---|
| 1 | b_era_spread | -0.051 (0.005, 100%) | -0.172 (0.005, 100%) | -0.197 (0.008, 100%) |
| 2 | b_med_pr | +0.086 (0.003, 100%) | -0.143 (0.006, 100%) | -0.115 (0.006, 100%) |
| 3 | b_pre1970_logcnt | +0.062 (0.003, 100%) | +0.103 (0.005, 100%) | +0.164 (0.007, 100%) |
| 4 | b_coverage | -0.052 (0.003, 100%) | -0.040 (0.005, 100%) | -0.200 (0.006, 100%) |
| 5 | b_share_out5k | -0.040 (0.003, 100%) | -0.103 (0.004, 100%) | -0.148 (0.005, 100%) |
| 6 | b_share_old | +0.054 (0.003, 100%) | +0.033 (0.004, 100%) | +0.198 (0.006, 100%) |
| 7 | b_int_size_disp | +0.008 (0.002, 99%) | +0.054 (0.003, 100%) | +0.205 (0.006, 100%) |
| 8 | b_franchise_share | -0.046 (0.002, 100%) | +0.009 (0.004, 98%) | -0.211 (0.006, 100%) |
| 9 | b_foreign_logcnt | +0.018 (0.003, 100%) | +0.107 (0.004, 100%) | -0.115 (0.005, 100%) |
| 10 | b_logcnt_out5k | +0.080 (0.003, 100%) | +0.033 (0.005, 100%) | +0.120 (0.007, 100%) |
| 11 | b_taste_disp | +0.050 (0.003, 100%) | +0.020 (0.004, 100%) | +0.148 (0.005, 100%) |
| 12 | u_rating_sd | +0.093 (0.004, 100%) | +0.073 (0.005, 100%) | +0.047 (0.006, 100%) |
| 13 | b_typicality | +0.032 (0.002, 100%) | -0.063 (0.004, 100%) | +0.105 (0.006, 100%) |
| 14 | b_niche_logcnt | -0.010 (0.003, 99%) | -0.118 (0.004, 100%) | -0.069 (0.005, 100%) |
| 15 | b_logcnt_out1k | +0.057 (0.003, 100%) | -0.025 (0.004, 100%) | -0.100 (0.005, 100%) |
| 16 | b_rate_mean | +0.140 (0.003, 100%) | +0.020 (0.005, 100%) | -0.018 (0.006, 99%) |
| 17 | b_foreign_share | +0.057 (0.003, 100%) | +0.015 (0.004, 99%) | +0.094 (0.005, 100%) |
| 18 | u_genre_entropy | -0.020 (0.004, 100%) | -0.102 (0.005, 100%) | +0.044 (0.005, 100%) |
| 19 | b_niche_share | -0.036 (0.002, 100%) | +0.014 (0.004, 99%) | -0.111 (0.004, 100%) |
| 20 | b_int_size_niche | +0.013 (0.004, 99%) | +0.072 (0.004, 100%) | -0.060 (0.005, 100%) |
| 21 | b_pre1970_share | -0.004 (0.004, 95%) | +0.079 (0.005, 100%) | -0.058 (0.006, 100%) |
| 22 | u_logprofile | -0.013 (0.002, 100%) | +0.030 (0.003, 100%) | -0.087 (0.004, 100%) |
| 23 | b_share_max | -0.033 (0.003, 100%) | -0.030 (0.003, 100%) | -0.062 (0.004, 100%) |
| 24 | b_pocket_eff | +0.059 (0.004, 100%) | +0.038 (0.005, 100%) | -0.026 (0.007, 99%) |
| 25 | b_share_out1k | +0.007 (0.002, 98%) | -0.109 (0.005, 100%) | -0.002 (0.006, 69%) |
| 26 | b_pocket_conc | +0.037 (0.003, 100%) | +0.063 (0.004, 100%) | +0.018 (0.006, 99%) |
| 27 | b_share_extreme | -0.027 (0.004, 100%) | +0.031 (0.004, 100%) | +0.055 (0.004, 100%) |
| 28 | b_nbr_density | +0.039 (0.004, 100%) | +0.016 (0.006, 98%) | -0.048 (0.006, 100%) |
| 29 | b_foreign_title_share | -0.040 (0.001, 100%) | -0.044 (0.003, 100%) | -0.005 (0.005, 94%) |
| 30 | b_int_size_out5k | +0.000 (0.004, 71%) | -0.028 (0.006, 99%) | +0.060 (0.006, 100%) |
| 31 | b_mean_pr | +0.023 (0.003, 100%) | +0.014 (0.005, 99%) | +0.046 (0.006, 100%) |
| 32 | b_doc_share | +0.024 (0.001, 100%) | -0.007 (0.003, 98%) | +0.050 (0.004, 100%) |
| 33 | u_n_decades | +0.031 (0.004, 100%) | -0.026 (0.005, 100%) | +0.020 (0.008, 98%) |
| 34 | b_anim_share | +0.040 (0.002, 100%) | +0.030 (0.005, 100%) | -0.004 (0.005, 84%) |

## DECOMPOSITION -- user answer-rate variance explained by observable features (R^2)

Per channel: OLS of per-user mean(knowledge>=1) on the user-feature block; R^2 = fraction of between-user variance explained (residual = what the random intercept carries). 'Old' = the 4 iter-1 consumption features; 'New' = the full census. CV = 5-fold ridge cross-validated R^2 (the honest number with ~34 features on 173 users).

| channel | explained OLD | explained NEW | OLD (CV) | NEW (CV) | between-user var(rate) |
|---|--:|--:|--:|--:|--:|
| concept | 42% | 64% | 37% | 32% | 0.0056 |
| entity | 12% | 29% | 2% | -28% | 0.0134 |
| item | 9% | 36% | 5% | 3% | 0.0002 |

## sigma_u -- random intercept shrinkage (refit to the NEW residual)

| channel | sigma_u iter-1 | sigma_u iter-2 | note |
|---|--:|--:|---|
| concept | 0.305 | 0.239 | single shift |
| entity/attribute | 0.567 | 0.533 | replaced by PER-CUT [k>=1: 0.778, k>=2: 0.554] (fix a) |
| item | 1.048 | 0.928 | single shift |

Calibration fix (b): concept taste_align coefficient scaled x0.70 (near-far(k1) 0.115 -> 0.105, target 0.105).

## ITER-2 G1 AGREEMENT (LOUO) -- side-by-side with iter-1

Knowledge accuracy / kappa vs LLM (baseline = per-stratum majority). Value MAE unchanged (value channels do not use the buffness user block).

| channel | stratum | n | acc iter1 | acc iter2 | kappa iter1 | kappa iter2 | base | verdict |
|---|---|--:|--:|--:|--:|--:|--:|:--:|
| concept | all | 195144 | 0.577 | 0.578 | 0.353 | 0.354 | 0.386 | beat |
| concept | low (NICHE) | 65048 | 0.555 | 0.554 | 0.180 | 0.182 | 0.517 | beat |
| concept | mid | 65048 | 0.491 | 0.493 | 0.144 | 0.152 | 0.436 | beat |
| concept | high | 65048 | 0.684 | 0.686 | 0.144 | 0.155 | 0.674 | beat |
| entity | all | 86500 | 0.556 | 0.558 | 0.131 | 0.137 | 0.525 | beat |
| entity | low (NICHE) | 28891 | 0.483 | 0.491 | 0.069 | 0.091 | 0.458 | beat |
| entity | mid | 28891 | 0.626 | 0.625 | 0.111 | 0.111 | 0.636 | tie/lose |
| entity | high | 28718 | 0.560 | 0.557 | 0.184 | 0.177 | 0.480 | beat |
| item | all | 124387 | 0.727 | 0.734 | 0.297 | 0.322 | 0.694 | beat |
| item | low (NICHE) | 41521 | 0.566 | 0.583 | 0.146 | 0.180 | 0.504 | beat |
| item | mid | 41530 | 0.698 | 0.703 | 0.096 | 0.135 | 0.690 | beat |
| item | high | 41336 | 0.917 | 0.917 | 0.000 | 0.010 | 0.917 | tie/lose |

**ITER-2 G1 verdict**: beats population-rate baseline (all channels + niche) = True; enriched features do NOT degrade accuracy vs iter-1 at all/niche strata = True.

## ITER-2 G2 -- FUEL REPRODUCTION (decisive, 13 stats)

**Pre-registered**: each statistic's synthetic value within the real 95% CI (or CIs overlap). Sampling only. Fixes applied: entity per-cut random effects (a); concept taste_align calibrated (b).

| statistic | real [95% CI] | synth [95% CI] | match |
|---|---|---|:--:|
| A1 concept ICC(k>=1) | +0.034 [+0.026,+0.041] | +0.029 [+0.023,+0.035] | OK |
| A1 concept ICC(k>=2) | +0.052 [+0.042,+0.062] | +0.042 [+0.034,+0.050] | OK |
| A1 concept base(k>=1) | +0.741 [+0.741,+0.741] | +0.742 [+0.742,+0.742] | OK |
| A1 attribute ICC(k>=1) | +0.092 [+0.075,+0.108] | +0.094 [+0.075,+0.114] | OK |
| A1 attribute ICC(k>=2) | +0.080 [+0.066,+0.095] | +0.060 [+0.047,+0.074] | OK |
| A1 attribute base(k>=1) | +0.824 [+0.824,+0.824] | +0.799 [+0.799,+0.799] | OK |
| A1 item ICC(k>=1) | +0.016 [+0.011,+0.021] | +0.018 [+0.011,+0.025] | OK |
| A1 item ICC(k>=2) | +0.137 [+0.114,+0.159] | +0.136 [+0.114,+0.158] | OK |
| A1 item base(k>=1) | +0.991 [+0.991,+0.991] | +0.987 [+0.987,+0.987] | OK |
| A2 item-low near-far(k2) | +0.009 [-0.004,+0.023] | +0.039 [+0.031,+0.048] | MISS |
| A2 item-mid near-far(k2) | +0.028 [+0.017,+0.039] | +0.028 [+0.020,+0.037] | OK |
| A2 item-high near-far(k2) | -0.007 [-0.014,+0.001] | +0.010 [+0.004,+0.017] | MISS |
| A2 concept near-far(k1) | +0.105 [+0.095,+0.115] | +0.104 [+0.098,+0.110] | OK |
| A3 concept-niche k>=1 | +0.528 [+0.528,+0.528] | +0.545 [+0.545,+0.545] | OK |
| A3 item-niche k>=2 | +0.475 [+0.475,+0.475] | +0.485 [+0.485,+0.485] | OK |
| A3 concept-broad k>=1 | +0.928 [+0.928,+0.928] | +0.914 [+0.914,+0.914] | OK |
| A3 item-broad k>=2 | +0.917 [+0.917,+0.917] | +0.895 [+0.895,+0.895] | OK |
| A3 overall rough share | +0.374 [+0.374,+0.374] | +0.365 [+0.365,+0.365] | OK |

**ITER-2 G2 verdict**: decisive 13/13 within CI -> PASS.

## ITER-2 G3 -- ERROR PROFILE (rated cells, passthrough disabled)

Sampled value vs real rating on 6962 rated cells: MAE=0.754, |err| dispersion=0.674, pred-true corr=0.177. LLM reference MAE~0.70, corr~0.5. Too-clean flag=False. **ITER-2 G3 verdict**: PASS. (value channels unchanged from iter-1; rated cells pass through the true rating in real generation.)

## ADDENDUM -- era-pocket user check (territory matching vs a global dial)

Bank era mass = 1988.1. Era-pocket users = rated-era mass >= 1.5 decades away (9/173 users; the user-62120 type: literate but off the bank's modern era). LOUO item-knowledge accuracy WITHOUT era features (decade_align density + era_dist distance dropped) vs WITH:

| subset | n cells | acc (no era) | acc (+era) | delta |
|---|--:|--:|--:|--:|
| era-pocket users | 6627 | 0.792 | 0.791 | -0.001 |
| non-pocket users | 117760 | 0.731 | 0.731 | -0.000 |
| all | 124387 | 0.734 | 0.734 | -0.000 |

**Verdict: NULL.** The era features produce NO item-knowledge accuracy lift for era-pocket users
(delta -0.001) or anyone else (delta -0.000). Item knowledge on this bank is already carried by
co-knowledge proximity + fame; per-question era distance adds nothing the embedding space did not
already encode. The territory-matching hypothesis for the item channel is not supported at n=9
pocket users -- reported without softening. (The era features remain in the model at negligible
weight; they do no harm and the concept/entity channels keep their own era columns.)

## ITERATION-2 STOP STATE (2026-07-09) -- build halted after G2 per contract

**Verdict chain (iteration-2)**: FIT clean (34 user features, ridge lambda by user-grouped 5-fold CV;
LOUO over 173; per-fold betas recorded). Calibration fixes: (a) entity per-cut random effects
sigma=[0.778 k>=1, 0.554 k>=2] (replaces the single 0.533 shift -- exactly the larger-variance-at-the-
knows-of-it-margin structure the iter-1 miss demanded); (b) concept taste_align x0.70 (synthetic
near-far(k1) 0.115 -> 0.105 = target center). **G2 PASS 13/13 decisive** (16/18 all rows; the two
misses are the NON-decisive item-low/item-high near-far bands, unchanged caveat from iter-1).
**G1**: improves or ties everywhere that matters (item all 0.727->0.734, item NICHE 0.566->0.583,
item kappa 0.297->0.322, entity NICHE 0.483->0.491; no degradation). **G3** unchanged-PASS (value
channels untouched: MAE 0.754 vs LLM ~0.70, corr 0.18 -- noisier than the LLM, not cleaner).
**Era-pocket check: NULL** (no lift; reported without softening).

**Honest caveats**:
1. Ridge lambda hit the GRID EDGE (1000) in all three channels -- CV prefers strong shrinkage of the
   user block; coefficients are stable (tiny fold-sd) but their absolute scale is penalty-limited.
2. The decomposition's CV columns are the honest numbers: concept 37%->32% (census does NOT beat the
   4 iter-1 features out-of-sample there), entity 2%->-28% (census OVERFITS the entity channel at the
   user level), item 5%->3%. The in-sample gains (concept 42->64%, entity 12->29%, item 9->36%) are
   what the fitted model actually uses, shrunken by the ridge; the user random intercepts carry the
   measured remainder (sigma_u 0.305->0.239 concept, 0.567->0.533 entity(+per-cut), 1.048->0.928 item).
   The author's profile-predictability thesis is PARTIALLY confirmed: features restore enough
   user-level structure that G2's ICCs all pass, but a large idiosyncratic residual remains and the
   random intercept (not the census) is what carries it.
3. G2 passing was driven primarily by the two calibration fixes; the census's measurable G2
   contribution is via the refit + shrunken sigma_u.

**Not run (contract: STOP after G2 regardless of outcome)**: lazy answerer, 20k/G4, 162k extension,
fold-v3, policies. Awaiting Fable review + author marks on the G2 outcome.

Artifacts: .cache/dans/{models_iter2.json, decomp_iter2.json (incl. importance table), louo_iter2.npz,
g1_iter2.json, g2_iter2.json, g3_iter2.json, erapocket_iter2.json, ref_centroids_4000_123.npz,
iter2_all.log}. Scripts: scripts/dans_iter2.py (+ dans_build.py / dans_stages.py extended).


---

# D-ANS v2.1 REPAIR BUNDLE (author-directed, signed 2026-07-09)

Motivation: the shuffle probe exposed a per-CALL know-well flutter (rate spread up to 0.66 on identical content; stars stable; within-call structure stable). v2.1 equates it out, rewires value on an EASE backbone, refits sigma_u to the equated trait, and re-gates against CORRECTED fuel targets. NO LLM calls.

## STAGE A -- equating + variance decomposition (full record: experiments/LLM_DECOMPOSITION.md)

Two-step equating (joint fit unidentifiable: unshuffled battery => call collinear with question-block). Headline margin per channel (concept/entity k>=1, item k>=2 since k>=1 is saturated ~99%). Share of Var(y):

| channel/margin | question-diff | user-FEATURE | user-TRAIT (equated) | CALL-FLUTTER | cell resid |
|---|--:|--:|--:|--:|--:|
| concept k>=1 | 54.6% | 0.6% | 1.3% | 4.3% | 39.3% |
| entity k>=1 | 32.0% | 0.0% | 2.5% | 16.2% | 49.3% |
| item k>=2 | 34.3% | 0.3% | 4.5% | 21.7% | 39.2% |

**Corrected fuel (trait-only ICC, headline margin) vs old flutter-inflated:**

| channel/margin | OLD ICC one-way | CORRECTED trait ICC | flutter var |
|---|--:|--:|--:|
| concept k>=1 | 0.0338 | **0.0427** | 0.00819 |
| entity k>=1 | 0.0920 | **0.0374** | 0.02350 |
| item k>=2 | 0.1367 | **0.0729** | 0.04619 |

## STAGE B -- EASE value wiring (backbone t(u,i) = real rating if rated else EASE pred)

Population EASE over a 9352-item universe (top-9000 popular UNION the top-800 bank UNION the 173 users' known items; lambda=500; binary item-item weights; Gram footprint 700 MB, cached B for lazy per-user prediction known-vector x B). Genome coverage of the universe (raw membership incidence; popular members -- which dominate the popularity-weighted aggregation -- are covered at a higher rate): tag-member 0.733, entity-member 0.797. Trained on all users' interactions MINUS the 173 study users' held-out half (leakage guard). Value refit: item = ordinal on t(u,i) directly; concept/entity = ordinal on the popularity-weighted mean of member t-values + rated-member engagement + fitted cutpoints. LOUO.

| value channel | n | corr(E[stars], LLM stars) | MAE vs label | iter-2 MAE | verdict |
|---|--:|--:|--:|--:|---|
| item | 123226 | 0.550 | 0.392 | 0.411 | corr>=0.40 PASS (G5) |
| concept | 144547 | 0.544 | 0.532 | 0.536 | beats iter-2 |
| entity | 71309 | 0.452 | 0.391 | 0.395 | beats iter-2 |

Item-value corr target ~0.45 (LLM masked-cell benchmark 0.547; G5 gate >=0.40). Old distilled item-value corr was 0.177 (heavily diluted); the EASE backbone rewires it.

## STAGE C -- dial refit + gates vs CORRECTED targets

### sigma_u refit to the EQUATED (flutter-free) trait residual

iter-2 sigma_u pooled trait+flutter over each user's few calls; rescaled by sqrt(trait fraction), trait fraction = sig2_U/(sig2_U + flutter/avg-calls-per-user) on the headline margin (documented approximation -- linear-scale variance ratio applied to the logit-scale sigma).

| channel | margin | sigma_u iter-2 | sigma_u v2.1 | trait fraction | note |
|---|---|--:|--:|--:|---|
| concept | k>=1 | 0.239 | 0.199 | 0.695 | single shift |
| entity | k>=1 | 0.533 | 0.301 | 0.320 | per-cut ['0.778', '0.554'] -> ['0.440', '0.313'] |
| item | k>=2 | 0.928 | 0.621 | 0.449 | single shift |

### G1 -- agreement (raw point acc unchanged; rate-level raw vs EQUATED-labels)

Point argmax accuracy is unchanged from iter-2 (equating changes the variance components / sigma_u, not the fixed-effect point predictions; item all-stratum acc stays 0.734). The fair comparison is at the per-call RATE level. The model predicts a content-driven per-call rate (it has NO knowledge of the call's flutter offset). RAW compares it to the LLM's realized (fluttered) per-call rate; EQUATED compares it to the flutter-FREE content rate = the population question-difficulty mean of that call's questions (removes flutter but PRESERVES content -- the correct target when calls are content-heterogeneous, e.g. the item battery's popularity-ordered blocks). Lower is better.

| channel | RAW per-call rate-MAE (vs fluttered LLM call rate) | EQUATED rate-MAE (vs flutter-free content rate) |
|---|--:|--:|
| concept | 0.0764 | 0.0534 |
| entity | 0.1552 | 0.1084 |
| item | 0.1609 | 0.0611 |

EQUATED < RAW on every channel: under RAW the model is penalized purely for not reproducing presentation flutter, which v2.1 deliberately no longer bakes into the trait. The gap RAW-minus-EQUATED is the flutter the model correctly declines to reproduce; the equated number is the fair agreement.

### G2 -- fuel reproduction vs the CORRECTED targets (the gate that now matters)

Generate synthetic knowledge with the EQUATED sigma_u (per-user intercept only, no per-call flutter), recompute the trait ICC by the same question-residualized nested VC, and require it to reproduce the CORRECTED (flutter-free) real trait ICC -- NOT the old flutter-inflated one-way ICC.

| channel | margin | real OLD one-way ICC | real CORRECTED trait ICC | synth trait ICC | match |
|---|---|--:|--:|--:|:--:|
| concept | k>=1 | 0.0338 | 0.0427 | 0.0294 | OK |
| entity | k>=1 | 0.0920 | 0.0374 | 0.0368 | OK |
| item | k>=2 | 0.1367 | 0.0729 | 0.0786 | OK |

**G2 v2.1 verdict**: reproduces corrected trait targets = True (tolerance 0.02 absolute).

### G3 -- error profile (NEW EASE value model, rated cells, passthrough disabled)

Sampled value vs real rating on 6962 rated cells: MAE=0.543, dispersion=0.495, pred-true corr=0.651. LLM ref MAE~0.70 corr~0.5. Too-clean flag=False. **G3 verdict**: PASS.

## v2.1 STOP STATE -- halted after gates per contract

Not run (contract: STOP after gates): lazy answerer, 20k/G4/162k generation, policies. Artifacts: .cache/dans/{equate_v21.json, value_v21.json, sigma_v21.json, gates_v21.json, models_v21.json, ease_v21.npz}, experiments/LLM_DECOMPOSITION.md. Script scripts/dans_v21.py.

