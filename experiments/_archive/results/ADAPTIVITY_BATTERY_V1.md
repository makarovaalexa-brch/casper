> **STAGE-C RETRACTION (Fable, 2026-07-08, same day):** the "+0.040 prize" cited from u-clair is a
> TARGET-PEEK (clairvoyant NDCG-greedy) upper bound — the exact class of number the program's own
> rules (E0f: peek-UBs are labelled leaky, never cited as results) forbid citing as a prize. It is
> hereby retracted as a prize measurement. The u-table "answerability oracle loses" arm used a
> different value ordering than the static it was compared to — the E2 value-model confound applied
> to a ceiling arm. Stage C therefore establishes NOTHING about the adaptivity prize; a legitimate
> ceiling (information-bounded, not target-peeking; one value model across all arms) has not been
> run. Stages A and B stand. r-blind's tie stands as "no loss under E2 floor" only.

# Directional Adaptivity Battery -- Answerer v1 (PARTIAL 173-user)

> **DIRECTIONAL ONLY** -- 173/300 users (163 shards + 10 grid10 merged in-memory); the answerer-v1 grid is NOT frozen. Every number below is provisional and must be re-run on the frozen 300-user grid before any citation.

Date 2026-07-08. Script `scripts/adaptivity_battery_v1.py` (+ `battery_stage_b.py`, `battery_stage_c.py`). NO LLM API calls; all local compute on cached grids. Environment = the judged answerer-v1 grid merged in-memory (163 shard users + 10 grid10 = 173 raw; 173 pass the fold filter). Working merge `.cache/instrument2/answerer_v1_grid173_WORKING.json` (NOT FROZEN). Paired per-user bootstrap BOOT=5000 seed=0. Answerability = knowledge != no_clue (k>=1); strong = know_well (k>=2); w_rough=0.5.

## DIRECTIONAL VERDICTS (synthesis, 173/300 -- provisional)

- **Stage A (fuel) -- PASS.** User-level heterogeneity in answerability exists beyond popularity, concentrated in the *strong-knowledge* (know_well, k>=2) signal: ICC(k>=2) item 0.137, attribute 0.080, concept 0.052; attribute also passes k>=1 (0.092). k>=1 is near-ceiling for items (0.99) and broad concepts -- this is an ABUNDANCE regime where the fuel lives in "how well", not "at all". Validity gap (user-specific taste signal) is strongly positive on concepts (+0.105 k>=1 / +0.148 k>=2, CI excl 0) and on mid-popularity items (k>=2 +0.027, CI excl 0). Scarcity habitat (answer-rate 0.2-0.7): concept-niche, item-bank-low/mid (by k>=2), attribute-director.
- **Stage B (discovery) -- FEASIBLE, and MORE feasible than the old arena.** The firewall-clean LEAVE-ONE-USER-OUT judged-grid MF is the strongest belief; on the informative know_well target its split-half asymptote is **0.938 (items) / 0.956 (concepts)** -- far ABOVE the old rated-ness ceiling **0.725** -- and beats popularity by **+0.134 (item k>=2) / +0.054 (concept k>=2)**, CI excl 0. Caveat: within-interview climb by t=16 is modest/flat (asymptote needs many more than 16 events); the latent predictability is high but evidence-starved in a 12-16 turn budget.
- **Stage C (prize) -- PRIZE EXISTS; blind router TIES the static (does not lose).** All three channels pass the E4 fold canary. Best static = s-item (any 0.2226). Privileged NDCG-greedy ceiling u-clair = 0.2627 -> **prize = +0.040 [+0.022, +0.058]** (EXISTS). Tie-by-construction blind router r-blind = 0.2166 -> **-0.006 [-0.0155, +0.0020]** vs s-item: does NOT beat it, does NOT lose beyond noise (E2 floor verified: turn-1 == s-mixed[0] for all users). Consistent with Branch B: the prize is real, discovery is the bottleneck, blind answerability adaptivity ties. NOTE: the value-greedy u-table LOSES (-0.043) -- knowing *which* questions are answerable is worthless without knowing which are *high-NDCG*; the ceiling is an NDCG-value oracle, not an answerability oracle. In this abundant-answerability arena the game is value/fidelity, not answerability discovery -- the abundance-regime twin of the prior structural (scarcity) finding.

## STAGE A -- FUEL GATES

**Pre-registered thresholds** (from the original study): A1 ICC>=0.05 beyond log-popularity, per channel; A2 user-specific-signal validity gap >0 with CI excl 0 in the moderate band; A3 locate the scarcity habitat (answer-rate in 0.2-0.7).

### A1 Heterogeneity (one-way random-effects ICC of answerability across users, log-popularity partialled out)

| channel | base k>=1 | base k>=2 | ICC(k>=1 \| logpop) [95% CI] | ICC(k>=2 \| logpop) [95% CI] | raw ICC k>=1 |
|---|--:|--:|---|---|--:|
| concept | 0.741 | 0.386 | 0.034[0.026,0.041] | 0.052[0.042,0.062] | 0.028 |
| attribute | 0.824 | 0.299 | 0.092[0.075,0.108] | 0.080[0.066,0.094] | 0.091 |
| item | 0.991 | 0.694 | 0.016[0.011,0.021] | 0.137[0.114,0.159] | 0.016 |

A1 verdict (ICC>=0.05 threshold): {'concept': True, 'attribute': True, 'item': True}. Item channel base rate on LLM-judged (genuinely uncertain) cells only; rated known-half items are data-filled (all know_well) and reported separately in A3.

### A2 Validity gap (user-specific signal)

The top-800 item bank is entirely global-famous (pr>=0.95), so the design's MODERATE band is realized as WITHIN-BANK popularity terciles (low/mid/high cnt); mid = moderate. Non-circular test: within a band, answerability of taste-NEAR items (genre matches the user's known-liked taste, above per-user median) minus taste-FAR. Item k>=1 is near-ceiling so the k>=2 (know_well) gap is the informative one for items; concepts span the range.

| stratum | near-far k>=1 [95% CI] | near-far k>=2 [95% CI] | n users |
|---|---|---|--:|
| item-low | -0.000[-0.004,+0.003] | +0.009[-0.005,+0.023] | 173 |
| item-mid | +0.000[-0.001,+0.002] | +0.027[+0.016,+0.039] | 173 |
| item-high | -0.001[-0.001,+0.000] | -0.007[-0.014,+0.001] | 173 |
| concept (all tags) | +0.105[+0.095,+0.115] | +0.148[+0.136,+0.161] | 173 |

A2 verdict (user-specific gap>0, CI excl 0): item-mid k>=2 True; concept k>=1 True; overall A2 pass = True.

### A3 Knowledge structure / scarcity habitat

Overall rough_idea share = 0.374. Answer-rate in the 0.2-0.7 band = the adaptivity habitat (scarcity a router can exploit).

| channel | stratum | n | rate k>=1 | rate k>=2 | rough share | in 0.2-0.7 habitat |
|---|---|--:|--:|--:|--:|:--:|
| item | famous | 124387 | 0.991 | 0.694 | 0.297 | - |
| item | tercile_bank_low | 41521 | 0.980 | 0.475 | 0.504 | YES |
| item | tercile_bank_mid | 41530 | 0.993 | 0.690 | 0.303 | YES |
| item | tercile_bank_high | 41336 | 0.999 | 0.917 | 0.082 | - |
| concept | all | 195144 | 0.741 | 0.386 | 0.355 | - |
| concept | tercile_niche | 65394 | 0.528 | 0.169 | 0.359 | YES |
| concept | tercile_mid | 64702 | 0.768 | 0.357 | 0.411 | - |
| concept | tercile_broad | 65048 | 0.928 | 0.632 | 0.296 | - |
| attribute | director | 25950 | 0.664 | 0.161 | 0.503 | YES |
| attribute | actor | 34600 | 0.920 | 0.362 | 0.558 | - |
| attribute | composer | 8650 | 0.772 | 0.112 | 0.661 | - |
| attribute | writer | 4325 | 0.814 | 0.181 | 0.633 | - |
| attribute | franchise | 12975 | 0.927 | 0.573 | 0.354 | - |

Scarcity-habitat strata (adaptivity has purchase here): ['item/tercile_bank_low', 'item/tercile_bank_mid', 'concept/tercile_niche', 'attribute/director'].

## STAGE B -- DISCOVERY FEASIBILITY

**Pre-registered directional reads:** (i) AUC climbs with observed events t; (ii) split-half asymptote beats popularity-only by >=0.05; (iii) compare the true-answerability asymptote to the OLD arena's rated-ness ceiling 0.725 (higher = discovery more feasible than the old arena allowed -- the headline either way).

Beliefs (E5 firewall -- eval user's cells NEVER train the belief): (i) population answer-rate logit; (ii) kmap POPULATION rated-matrix embeddings (items) / member-item centroid (concepts); (iii) NEW 5-fold LEAVE-ONE-USER-OUT judged-grid logistic MF (d=16). AUC held out over unrevealed cells; needs both classes (item k>=1 is near-ceiling so many users are dropped -- n reported).

### item_k>=1

| belief | t=0 | t=2 | t=4 | t=8 | t=16 | split-half asymptote |
|---|--:|--:|--:|--:|--:|--:|
| popularity (i) | 0.877 | 0.876 | 0.876 | 0.875 | 0.874 | 0.880 |
| kmap population (ii) | 0.877 | 0.861 | 0.856 | 0.852 | 0.842 | 0.870 |
| LOUO judged-MF (iii) | 0.891 | 0.892 | 0.893 | 0.895 | 0.898 | 0.884 |

- n users with both classes held out by t: {0: 147, 2: 147, 4: 147, 8: 147, 16: 147}.
- (i) climb (LOUO-MF t16 vs t2): +0.005[+0.005,+0.006] (n=147).
- (ii) best belief = **louo_mf**; asymptote vs popularity +0.004[-0.029,+0.036] (ties/below).
- (iii) best asymptote 0.884 vs rated-ness ceiling 0.725: **ABOVE**.

### item_k>=2

| belief | t=0 | t=2 | t=4 | t=8 | t=16 | split-half asymptote |
|---|--:|--:|--:|--:|--:|--:|
| popularity (i) | 0.805 | 0.804 | 0.803 | 0.802 | 0.799 | 0.804 |
| kmap population (ii) | 0.805 | 0.784 | 0.780 | 0.783 | 0.775 | 0.840 |
| LOUO judged-MF (iii) | 0.885 | 0.884 | 0.884 | 0.883 | 0.881 | 0.938 |

- n users with both classes held out by t: {0: 173, 2: 173, 4: 173, 8: 173, 16: 173}.
- (i) climb (LOUO-MF t16 vs t2): -0.004[-0.005,-0.003] (n=173).
- (ii) best belief = **louo_mf**; asymptote vs popularity +0.134[+0.124,+0.146] (BEATS +0.05).
- (iii) best asymptote 0.938 vs rated-ness ceiling 0.725: **ABOVE**.

### concept_k>=1

| belief | t=0 | t=2 | t=4 | t=8 | t=16 | split-half asymptote |
|---|--:|--:|--:|--:|--:|--:|
| popularity (i) | 0.952 | 0.952 | 0.952 | 0.952 | 0.952 | 0.952 |
| kmap population (ii) | 0.952 | 0.952 | 0.952 | 0.952 | 0.952 | 0.961 |
| LOUO judged-MF (iii) | 0.950 | 0.950 | 0.950 | 0.950 | 0.950 | 0.957 |

- n users with both classes held out by t: {0: 173, 2: 173, 4: 173, 8: 173, 16: 173}.
- (i) climb (LOUO-MF t16 vs t2): +0.000[-0.001,+0.001] (n=173).
- (ii) best belief = **kmap**; asymptote vs popularity +0.008[+0.007,+0.010] (beats pop).
- (iii) best asymptote 0.961 vs rated-ness ceiling 0.725: **ABOVE**.

### concept_k>=2

| belief | t=0 | t=2 | t=4 | t=8 | t=16 | split-half asymptote |
|---|--:|--:|--:|--:|--:|--:|
| popularity (i) | 0.903 | 0.903 | 0.903 | 0.902 | 0.902 | 0.903 |
| kmap population (ii) | 0.903 | 0.904 | 0.905 | 0.905 | 0.898 | 0.924 |
| LOUO judged-MF (iii) | 0.937 | 0.939 | 0.938 | 0.936 | 0.936 | 0.956 |

- n users with both classes held out by t: {0: 173, 2: 173, 4: 173, 8: 173, 16: 173}.
- (i) climb (LOUO-MF t16 vs t2): -0.002[-0.007,+0.000] (n=173).
- (ii) best belief = **louo_mf**; asymptote vs popularity +0.054[+0.047,+0.059] (beats pop).
- (iii) best asymptote 0.956 vs rated-ness ceiling 0.725: **ABOVE**.

## STAGE C -- PRIZE DECOMPOSITION (mini-arena)

T=12 turns; question universe = the judged grid (channels passing the E4 canary); NDCG@10 on the untouched held-out halves; ALL 173 users every arm (E1: refusal=no-op turn, no-answer users at cold z=0 = 0.1502); I2.5 fold (val 0.4629). Paired per-user bootstrap.

### E4 fold canary (gate)

| channel | mean single-answer lift from cold [95% CI] | n users | verdict |
|---|---|--:|---|
| concept | +0.0576[+0.0320,+0.0834] | 173 | PASS |
| item | +0.0463[+0.0252,+0.0680] | 173 | PASS |
| attr | +0.0588[+0.0351,+0.0834] | 173 | PASS |

Channels entering the arena: ['concept', 'item', 'attr'].

### Arena arms (NDCG@10)

| arm | anytime | endpoint | mean answered turns | class |
|---|--:|--:|--:|---|
| s-concept | 0.2166 | 0.2150 | 12.0 | static |
| s-item | 0.2226 | 0.2213 | 11.8 | static |
| s-mixed | 0.2166 | 0.2150 | 12.0 | static (baseline) |
| u-table (PRIV) | 0.1800 | 0.1793 | 12.0 | PRIVILEGED value-greedy ceiling |
| u-clair (PRIV) | 0.2627 | 0.2632 | 12.0 | PRIVILEGED NDCG-greedy ceiling |
| r-blind | 0.2166 | 0.2150 | 12.0 | blind router (tie-by-construction) |

Best static opponent = **s-item**.

### C3 pre-registered directional reads

- **(i) prize** (best privileged ceiling - s-item, anytime): u-clair (NDCG-greedy) = **+0.0400**[+0.0219,+0.0582]; u-table (value-greedy) = -0.0427[-0.0665,-0.0181] -> prize EXISTS (CI excl 0 AND >=0.01): **True**.
- **(ii) r-blind** vs s-item (anytime) = **-0.0060**[-0.0155,+0.0020] -> blind win (CI excl 0, +): **False**; loses beyond noise: **False**. Tie-by-construction floor (turn-1 == s-mixed[0], E2): **True**.

Interpretation: prize EXISTS; blind router TIES the static (discovery still binding).

### DIRECTIONAL caveats

- 173/300 users, grid UNFROZEN; re-run on the frozen grid before citation.
- Attribute embeddings built from member-item bags (director/actor/etc); concept embeddings from genome membership bags -- both via the frozen RecVAE encoder, not the curated 200-concept fold interface (documented construction).
- r-blind LR tilt uses the kmap POPULATION item posterior (validated, ready infer); concept/attr candidates carry LR=1 (no tilt). Stage B reports whether the LOUO-MF belief is stronger -- if so, a follow-up router should use it.

## STAGE C -- PRIZE DECOMPOSITION (mini-arena)

T=12 turns; question universe = the judged grid (channels passing the E4 canary); NDCG@10 on the untouched held-out halves; ALL 173 users every arm (E1: refusal=no-op turn, no-answer users at cold z=0 = 0.1502); I2.5 fold (val 0.4629). Paired per-user bootstrap.

### E4 fold canary (gate)

| channel | mean single-answer lift from cold [95% CI] | n users | verdict |
|---|---|--:|---|
| concept | +0.0576[+0.0320,+0.0834] | 173 | PASS |
| item | +0.0463[+0.0252,+0.0680] | 173 | PASS |
| attr | +0.0588[+0.0351,+0.0834] | 173 | PASS |

Channels entering the arena: ['concept', 'item', 'attr'].

### Arena arms (NDCG@10)

| arm | anytime | endpoint | mean answered turns | class |
|---|--:|--:|--:|---|
| s-concept | 0.2166 | 0.2150 | 12.0 | static |
| s-item | 0.2226 | 0.2213 | 11.8 | static |
| s-mixed | 0.2166 | 0.2150 | 12.0 | static (baseline) |
| u-table (PRIV) | 0.1934 | 0.1843 | 12.0 | PRIVILEGED ceiling |
| r-blind | 0.2166 | 0.2150 | 12.0 | blind router (tie-by-construction) |

Best static opponent = **s-item**.

### C3 pre-registered directional reads

- **(i) prize** = u-table - s-item (anytime) = **-0.0292**[-0.0481,-0.0110] -> prize EXISTS (CI excl 0 AND >=0.01): **False**.
- **(ii) r-blind** vs s-item (anytime) = **-0.0060**[-0.0155,+0.0020] -> blind win (CI excl 0, +): **False**; loses beyond noise: **False**. Tie-by-construction floor (turn-1 == s-mixed[0], E2): **True**.

Interpretation: prize absent; blind router TIES the static (discovery still binding).

### DIRECTIONAL caveats

- 173/300 users, grid UNFROZEN; re-run on the frozen grid before citation.
- Attribute embeddings built from member-item bags (director/actor/etc); concept embeddings from genome membership bags -- both via the frozen RecVAE encoder, not the curated 200-concept fold interface (documented construction).
- r-blind LR tilt uses the kmap POPULATION item posterior (validated, ready infer); concept/attr candidates carry LR=1 (no tilt). Stage B reports whether the LOUO-MF belief is stronger -- if so, a follow-up router should use it.

