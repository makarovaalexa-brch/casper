# Vivid-Swap Mechanism Tests -- Answerer v1 (DIRECTIONAL 173/300)

> **DIRECTIONAL ONLY** -- 173/300 users, answerer-v1 working grid NOT frozen. Every number is provisional; re-run on the frozen 300-user grid before any citation.

Date 2026-07-08. Script `scripts/vivid_swap_tests.py`. NO LLM API calls; local compute on the cached judged grid through the I2.5 learned fold (`.cache/i25_fold_best.pt`, val 0.4629). 173 users pass the fold filter; 180 shared candidates; cold NDCG@10 0.1502. Paired per-user bootstrap BOOT=5000 seed=0. Vividness = know_well (k>=2); rough_idea (k=1) folds at w_rough=0.5.

**Value model (used identically across arms, E2-clean):** per-candidate mean single-answer NDCG@10 lift from cold, over its answering users (>= 10); value TIERS = 8 octiles of that value. Matching on the tier controls question value; the residual k2-vs-k1 gap is the mechanism. klev/answerability mismatches vs build_cands: 0 (0 = consistent).

## T1 -- VIVIDNESS VALUE CHECK (the premise)

**PRE-REGISTERED GATE:** k2-set beats k1-set at matched value tiers, paired CI excl 0. If FAIL: the premise is dead -> STOP.

Design: per user, for each value tier holding BOTH a k=1 and a k=2 grid cell, slot in one k=1 rep and one k=2 rep (rep = cval closest to tier median). The k1-set and k2-set are matched tier-for-tier and differ ONLY in knowledge composition. Both folded through the real I2.5 fold (know_well weight 1.0, rough_idea 0.5; item rated cells = real ratings, native RecVAE token where stars>=4). NDCG@10 on the untouched held-out halves.

| contrast | k2 NDCG | k1 NDCG | delta [95% CI] | n users | verdict |
|---|--:|--:|---|--:|---|
| **k2-set vs k1-set (matched, ANY k2)** | 0.1945 | 0.2127 | **-0.0182**[-0.0398,+0.0024] | 173 | FAIL |
| k2-DATA (real rating) vs k1 | 0.2779 | 0.2611 | +0.0169[-0.0217,+0.0535] | 97 | decomposition |
| k2-LLM/concept/attr vs k1 | 0.1913 | 0.2127 | -0.0214[-0.0417,-0.0026] | 173 | decomposition |

Mean matched value tiers per user = 7.1. **T1 GATE = FAIL -> STOP (premise dead).**

### Diagnosis (why the premise fails, and what actually survives)

The aggregate k2-vs-k1 delta is negative (-0.018, CI touches 0) because the two knowledge channels
pull in OPPOSITE directions and the noisy one dominates the matched sets:

- **Real-rating vividness HELPS (directionally):** k2-DATA (source="data" real ratings, zero noise,
  folded at weight 1.0) beats its matched k1 by **+0.0169** -- the sign the mechanism predicted. But
  CI includes 0 (n=97) so it is not established, and it only exists where the user has a genuine rating
  in that value tier (i.e. their known half -- already answerable by construction, not a discovery win).
- **LLM-guessed vividness HURTS (significantly):** k2-LLM/concept/attr (know_well but value is
  LLM-inferred, sigma ~0.70 stars) folded at weight 1.0 is **-0.0214 [CI excl 0]** vs rough_idea (k1)
  folded at weight 0.5. Full-weighting a noisy inferred value is WORSE than half-weighting a hedge.

So the mechanism's stated premise -- "k=2 beats k=1 because weight 1.0 + real ratings + no dilution" --
is **refuted in aggregate**: the fold's `know_well = 1.0` weight is MIScalibrated for LLM-guessed
answers (it amplifies fidelity noise), and it is exactly this LLM-know_well cell type that fills most
matched tiers. The vividness premium is real ONLY on genuinely-rated cells, which are the user's own
known-half items -- not something a peer-swap policy can manufacture by picking a "more vivid" question.

**Consequence for the vivid-swap policy:** the swap would trade an equal-value peer for one the user
"knows well", but in this arena most "know_well" answers are LLM-valued, so the swap would on average
INJECT fidelity noise at full weight, not remove dilution. T2/T3 are NOT run (T1 gates them). This is
the abundance-regime twin of the Stage-C conclusion (STATE_2026-07-08): the game here is
value/FIDELITY, not knowledge level -- and the actionable fix is a fidelity-aware fold weight
(`w_know_well_llm < 1.0`, i.e. down-weight LLM-inferred know_well toward the rated-only channel), a
mechanical change testable before any policy is built. A cheap, diagnostic death.


---

# FIDELITY-AWARE FOLD-WEIGHT CALIBRATION + GATED RERUN (held-out half B)

> **DIRECTIONAL ONLY** -- 173/300 users, grid UNFROZEN. FIT on half A (~86 users), ALL verdicts from half B (~87 users). Re-run on the frozen grid before any citation.

Date 2026-07-08. Script `scripts/fold_weight_calibration.py` (reuses `vivid_swap_tests.py`). NO LLM calls. Motivated by the T1 FAIL diagnosis above: the fold weight was keyed on KNOWLEDGE LEVEL (know_well=1.0, rough_idea=0.5), which full-weights a ~0.70-star LLM guess. We re-key it on **FIDELITY CLASS**: `w_data`=1.0 (real rating, FIXED reference), `w_k2_llm` (know_well + LLM value), `w_k1_llm` (rough_idea + LLM value). Deterministic split (seed=0 permutation); users split once, weights fit on A, verdicts on B.

## STEP 1 -- Calibration (fit on half A)

**OBJECTIVE (pre-registered):** maximize mean NDCG@10 of FIXED reference answer-sets = a MIXTURE per user of 4 random 8-answer tier-spanning draws + the coverage-popularity static's answered subset (so weights are not tuned to one selector). Coarse grid {0,.1,..,1.0}^2 over (w_k2_llm, w_k1_llm); w_data fixed 1.0. Default fold (1.0, 0.5) lies inside the grid.

| quantity | value |
|---|---|
| **FITTED weights (half A)** | w_data=1.0, **w_k2_llm=0.6**, **w_k1_llm=0.0** |
| objA at fitted / at default(1.0,0.5) | 0.1770 / 0.1749 |
| flatness (obj span over grid) | 0.0079 |
| grid pts within 0.001 of best | 6 (w_k2_llm in [0.6,0.9], w_k1_llm in [0.0,0.1]) |
| half-A calibrated-vs-default objective | +0.0022[-0.0028,+0.0076] (n=86) |
| **half-B calibrated-vs-default objective (held out)** | **+0.0005[-0.0041,+0.0057]** (n=87) |

objA surface (rows w_k2_llm 0..1, cols w_k1_llm 0..1), leading 0 stripped:

```
w2\w1  0.0  0.1  0.2  0.3  0.4  0.5  0.6  0.7  0.8  0.9  1.0
 0.0  .172 .171 .171 .170 .170 .169 .170 .170 .170 .169 .170
 0.1  .172 .171 .170 .170 .169 .169 .170 .169 .170 .169 .170
 0.2  .172 .171 .170 .169 .170 .170 .170 .169 .169 .169 .170
 0.3  .173 .173 .172 .171 .171 .171 .171 .171 .171 .171 .171
 0.4  .175 .175 .173 .173 .172 .173 .173 .172 .171 .172 .172
 0.5  .176 .175 .174 .174 .173 .173 .173 .172 .172 .172 .172
 0.6  .177 .176 .174 .174 .174 .174 .174 .173 .173 .173 .173
 0.7  .177 .176 .175 .175 .175 .174 .174 .173 .173 .174 .174
 0.8  .177 .176 .175 .175 .175 .175 .175 .174 .174 .175 .175
 0.9  .177 .176 .175 .175 .175 .175 .174 .174 .174 .175 .174
 1.0  .176 .175 .174 .174 .174 .175 .174 .174 .174 .174 .174
```

**Diagnosis prediction check:** w_k2_llm << 1.0 and ~ w_k1_llm -> this fit **does NOT cleanly confirm** it (w_k2_llm=0.6, w_k1_llm=0.0). 

## STEP 2 -- T1 / T1b on held-out half B (calibrated weights everywhere)

Calibrated fold: w_data=1.0, w_k2_llm=0.6, w_k1_llm=0.0. Value tiers recomputed under the calibrated fold on half B (87 users). Matched-tier design identical to T1 above; the ONLY change is the fidelity-aware weights.

**PRE-REGISTERED GATES:** T1 = k2-set vs k1-set (CI excl 0); T1b = RATED(source=data) vs matched-tier k1-LLM (CI excl 0). NOTE: if w_k2_llm~w_k1_llm the LLM-valued cells are neutralized BY DESIGN, so T1 may go null while T1b isolates the real source=data premium.

| contrast | tgt NDCG | k1 NDCG | delta [95% CI] | n | verdict |
|---|--:|--:|---|--:|---|
| **T1 k2-set vs k1-set (ANY k2)** | 0.2085 | 0.2217 | **-0.0132**[-0.0449,+0.0158] | 87 | FAIL |
| **T1b RATED(data) vs k1-LLM** | 0.3570 | 0.3022 | **+0.0547**[+0.0022,+0.1124] | 47 | PASS |
| **(ctx) k2-LLM vs k1** | 0.2003 | 0.2217 | **-0.0215**[-0.0531,+0.0066] | 87 | context |

**STEP 2 verdict (half B):** T1 FAIL, T1b PASS.

## STEP 3 -- gated T2/T3 on half B (calibrated fold, TARGET = RATED (source=data real ratings; item channel only))

STEP 2 opened the gate via **T1b (rated premium)** so the swap target is the class that carries the premium: **RATED (source=data real ratings; item channel only)**. The reused `vivid_swap_tests.py` harness prints its generic labels ('k>=2', 'population-k2-rate', 'know_well'); under target='data' READ every such label as the TARGET class above (the target switch `V.TGT` re-points the label, the population rate, and the LOUO belief at source=data cells). Calibrated weights (w_data=1.0, w_k2_llm=0.6, w_k1_llm=0.0) are applied EVERYWHERE incl. the reproduced statics (fair). T2 = can a blind belief pick THIS user's rated peer at matched value? T3 = does the rated-swap policy beat the fair static?

## T2 -- PEER-RANKING ACCURACY (the belief where it matters)

**PRE-REGISTERED GATE:** belief (iii) LOUO-MF+TASTE beats belief (i) population-k2-rate on precision@1, paired CI excl 0, at t=8 evidence. If FAIL: personalization has no purchase -> run T3 with the population-vivid variant only.

Peer sets = (channel, value-tier) groups of shared candidates (size 3-12, capped by coverage); 14 sets. Evaluated on sets where the user has >=1 k=2 AND >=1 answerable-non-k2 peer (a non-trivial pick). Beliefs (E5 firewall): (i) population k>=2 rate; (ii/iii base) 5-fold LEAVE-ONE-USER-OUT logistic MF on k>=2 (item + concept channels), online user vector from t evidence events; (iii) 5-fold logistic P(k=2 | pop, louo_score, taste_dot, log-cov) where taste_dot = cosine(fold of first-t answer VALUE tokens, candidate direction). Evidence excludes the scored peer set (no leakage). precision@1 = the belief's top-1 peer is actually answered k=2 by the user.

| evidence | belief(i) pop prec@1 | belief(iii) MF+taste prec@1 | delta [95% CI] | n sets | realized fold-wt iii vs sched |
|---|--:|--:|---|--:|---|
| t=4 | 0.265 | 0.088 | -0.176[-0.309,-0.044] | 68 | 0.794 vs 0.816 |
| t=8 | 0.265 | 0.088 | -0.176[-0.309,-0.044] | 68 | 0.794 vs 0.816 |

**T2 GATE = FAIL -> T3 runs the population-vivid variant only (r-vivid reduces to s-vivid).**

## T3 -- THE POLICY (fair harness)

T=12, NDCG@10 anytime, all 87 users (E1). Statics reproduced via the Stage-C canary + greedy build on the shared pool. NO oracle / target-peek / answerability-table arms (Stage-C retraction). r-vivid belief = **pop** (population k2-rate only (T2 failed)). eps/delta grid-searched on a 40% val split of users (chosen eps=0.0087, delta=0.100; val r-vivid-s-vivid +0.0207).

**PRE-REGISTERED CONTRASTS:** (1) r-vivid vs s-best = headline (must not lose beyond noise, E2); (2) r-vivid vs s-vivid = personalization increment (THE thesis quantity); (3) s-vivid vs s-best = vividness-prior gain (credited to the SCALE design, not adaptivity).

| arm | anytime NDCG@10 | class |
|---|--:|---|
| s-best (s-item) | 0.2522 | strongest static (reproduced) |
| s-vivid | 0.2317 | population-vivid static (deployable, no user evidence) |
| r-vivid | 0.2522 | full vivid-swap router (pop) |

### Three pre-registered contrasts (paired per-user bootstrap)

- **(1) r-vivid vs s-best (headline):** +0.0000[+0.0000,+0.0000] -> ties / within E2 floor.
- **(2) r-vivid vs s-vivid (personalization increment -- THE thesis quantity):** +0.0205[+0.0020,+0.0431] -> personalization ADDS (CI excl 0).
- **(3) s-vivid vs s-best (vividness-prior gain, SCALE design):** -0.0205[-0.0431,-0.0020] -> no prior gain (ties).

E2 floor (r-vivid does not lose to s-best beyond noise): **True**.

### DIRECTIONAL caveats

- 173/300 users, grid UNFROZEN; re-run on the frozen grid before citation.
- Value model = per-candidate mean single-answer NDCG lift (population; identical across arms).
- s-vivid/r-vivid swap only WITHIN channel and within +/- eps value of the scheduled question; r-vivid at t=0 (no evidence) reduces to the population-vivid pick (~s-vivid).

### STEP 3 conclusion (target=data)

**The rated-composition premium (T1b, +0.0547) is NOT routable.** T2 personalization FAILS (belief(iii) does not beat the population rate at picking the user's rated peer -- consistent with the rated-ness predictability ceiling in STATE_2026-07-08: which popular items a user happens to have rated is largely idiosyncratic + popularity-driven). The population rated-swap (s-vivid) even HURTS the static (-0.0205) -- swapping a scheduled item for a higher-rated-rate (more popular) peer trades value for rated-probability and loses; the router only avoids the loss by declining to swap (reducing to s-best). **Net: source=data vividness is the user's own known-half property, not something a peer-swap policy can manufacture by picking a 'more vivid' question -- exactly the abundance-regime twin of the rated-ness ceiling. Answer-composition routing has no realizable premium here.**

**ROUTABLE = NO.**

