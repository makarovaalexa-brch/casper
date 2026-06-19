# CASPER-U — LIVE results ledger

This is the **single source of truth** for go-forward results. Abandoned avenues live in `archive/`
(the frozen-instrument RL saga E1–E13 + early elicitation experiments; superseded — see `archive/README.md`).
Plan + fixed ruler: `CASPER_U_PLAN.md`. One entry per result; never silently overwrite — append a dated row.

**FIXED RULER (sacred):** ML-1M, like=rating≥4, users≥5 pos, fixed 80/10/10 split, FULL-catalogue ranking,
STANDARD NDCG@10 (=DCG/IDCG) + Recall@10 (Recall = lead elicitation metric), hyperparams tuned on VAL → reported on TEST.
External-validity anchor: MOSTPOP NDCG@10 ≈ 0.39–0.41 (DRE/Kweon WWW2020). If it drifts, pipeline is broken.

Status key: ✅ done · 🔄 in progress · ⬜ todo · ❌ negative/failed-to-reproduce (kept, not abandoned)

---

## PART A — Replications (faithful on the PAPER's own ruler, THEN port to ours)

### R1 — Representative-MF cold-start rating elicitation
- Papers: Liu RecSys 2011 (RBMF) · Fonarev ICDM 2016 (rect. maxvol) · Shi/Zhao/Shen TOIS 2017 · fold-in Zhou SIGIR 2011
- Script: `scripts/paper1/mf_foldin.py`
- **Paper-ruler faithful replication:** ✅ done (2026-06-17) — ML-1M is the papers' own dataset; full-cat NDCG@10 protocol.
  MOSTPOP 0.4134 ≈ DRE/Kweon reported 0.3921; RMVA-maxvol+pop 0.5011 vs DRE-reported RMVA 0.5387 (scale+ordering
  match faithfully; residual gap = our fold-in decoder vs their reconstruction net). RMVA = rectangular max-volume
  (Fonarev ICDM2016) via pivoted-QR on WRMF item factors.
- **Ported to OUR ruler:** ✅ done (2026-06-17)

  | arm (ML-1M, full-cat, std NDCG, val-tuned) | NDCG@10 | Recall@10 | published ref |
  |---|---|---|---|
  | MOSTPOP (popularity) | 0.4134 | 0.0645 | DRE 0.3921 |
  | RANDOM seeds + pop (w*=12) | 0.4140 | 0.0659 | — |
  | POPULAR seeds + pop (w*=4) | 0.3547 | 0.0707 | — |
  | REPRESENTATIVE seeds + pop (w*=8) | 0.4816 | 0.0818 | — |
  | **RMVA max-volume seeds + pop (w*=8)** | **0.5011** | **0.0902** | DRE-RMVA 0.5387 |
  - Verdict: ✅ elicitation beats popularity (+21% NDCG, +40% Recall); RMVA faithfully matches published scale/ordering.
  - Mechanism note: requires popularity-floor + personalization residual (pure personalization LOSES NDCG).

### R2 — EAR unified item+attribute FM scorer
- Paper: Lei et al. WSDM 2020 (score = u·v + Σ_{p∈P_u} v·p)
- Script: `scripts/paper1/ear_lastfm.py`
- Scope (user decision): MECHANISM + qualitative claim (the scorer we port), not full SR@15 RL pipeline.
- **Replicated (LastFM hetrec, items=artists, attrs=tags, cold-start, std NDCG@10/Recall@10):** ✅ done (2026-06-17)

  | arm | NDCG@10 | Recall@10 |
  |---|---|---|
  | POPULARITY | 0.1758 | 0.0754 |
  | ATTR-only (Σ v·p) | 0.1359 | 0.0630 |
  | item-only (b_v) | 0.1268 | 0.0576 |
  | **ATTR + pop (EAR mechanism)** | **0.2583** | **0.1070** |
  - Verdict: ✅ confirmed-attribute conditioning beats popularity (+47% NDCG, +42% Recall) — EAR's core claim.
  - CROSS-SETTING CONFIRMATION: same popularity-floor+residual law as R1 (attributes alone lose; attributes ON TOP
    of popularity win). Holds across ML-1M items (R1), LastFM attributes (R2), EDDI tabular (R3).
- **Ported to OUR ruler:** the scorer is folded into the unified instrument at S2/S3.

### R3 — EDDI / Partial-VAE set-encoder
- Paper: Ma et al. ICML 2019
- Script: `scripts/paper1/eddi_replicate.py`
- **Paper-ruler faithful replication:** ✅ done (2026-06-17) — Partial-VAE (permutation-invariant encoder over observed
  (feature,value) tokens, random-mask trained) + information-reward active acquisition on California-housing (UCI-style
  regression). Reproduces EDDI's HEADLINE claim: active acquisition dominates random at every step.

  | #features | ACTIVE (EDDI info-gain) RMSE | RANDOM RMSE |
  |---|---|---|
  | 1 | 0.812 | 0.946 |
  | 4 | 0.787 | 0.895 |
  | 7 | 0.787 | 0.830 |
  | mean-over-curve | **0.816** | 0.894 |
  - Verdict: ✅ active dominates random (+0.078 mean RMSE). Validates the permutation-invariant set-encoder + info-gain
    acquisition (the exact instrument mechanism).
- **Ported to OUR ruler:** ⬜ pending — instantiate this set-encoder as the recsys instrument (= ladder S2).

### R-extra — Golbandi adaptive decision-tree interview (RMSE)
- Paper: Golbandi, Koren, Lempel WSDM 2011
- Script: `scripts/paper1/golbandi_tree.py`
- **Replicated (ML-100k, RMSE — paper's own metric):** ✅ done (2026-06-17)
  - Adaptive tree vs static-popular interview, equal #questions: **0.9804 < 0.9885 RMSE @ depth 6** (adaptive wins every depth ≥1). Faithful magnitude. Canonical *adaptive*-elicitation win.

### R-extra — DRE faithful re-implementation
- Paper: Kweon et al. WWW 2020 (arXiv:2402.16327 is a reupload)
- Script: `scripts/paper1/dre_faithful.py`
- **Status:** ❌ partial — baseline scale reproduced (MOSTPOP 0.41 ≈ paper 0.39) but DRE's *neural gain* did NOT reproduce (autoencoder stayed below MOSTPOP; Dacrema-2019 reproducibility caveat). Recorded as honest negative; the win came from R1 (representative-MF), not DRE-neural.

---

## PART B — CASPER-U ladder (each stage: Definition-of-Done + regression-check prior stage)

| Stage | What | Script | Status | Headline result | DoD met? |
|---|---|---|---|---|---|
| S0 | Ruler + acceptance suite locked (gates: beats-pop, law, monotone, ≥WRMF) | CASPER_U_PLAN.md / instrument_u.py | ✅ | MOSTPOP 0.4134 ≈ 0.39; gates defined+coded | ✅ |
| S1 | WRMF+foldin+RMVA-maxvol seeds+pop > popularity (item-only) | mf_foldin.py | ✅ | 0.5011 vs 0.4134 (+21% NDCG, +40% Recall) | ✅ |
| S2 | Learned set-encoder instrument (item-only) = WRMF-init Q + differentiable ridge fold-in + conf-scaled pop-blend | instrument_u.py | ✅ | FULL 0.4863 vs MOSTPOP 0.4134 (+17.6%); ALL 4 GATES PASS (matches WRMF 0.5011) | ✅ |
| S3b | Elicitation EFFICIENCY: concepts/attributes vs movies (oracle, per-user) | concept_efficiency.py, concept_efficiency_v2.py | ✅ KEY | GENRES-only (18): beat movies for 27/21/19% users @1/3/5q. FINE GENOME CONCEPTS (745): far stronger — NDCG@4q items 0.611, genres 0.446, genome 0.581, MIXED 0.623 (mixed BEATS items on avg!). Genome concepts>movies for 38% of users (vs genres 16%); mixed>items 49%. Winning concepts are moods/themes: "utopia" 1.00 vs 0.43, "nostalgic" 0.92 vs 0.54, "affectionate" 0.64 vs 0.00, "oscar(visual fx)", "sad but good". => strong motivation for UNIFIED action space; non-genre concepts worth a lot for ~38% users. Caveat: ORACLE (potential, realizable=S5). Added to paper+thesis chapter (tab:eff). |
| S6-LLM | CHEAP LLM asker baseline (gpt-4o-mini) on calibrated instrument | llm_asker.py | ✅ | 60u, T=6, menu=50 (popular+RMVA), held-out, leakage-free answers, CACHED (llm_asker_cache.jsonl, 0 parse failures). random-from-menu 0.304→0.310 (+0.005); **LLM 0.304→0.274 (−0.031, WORSE than random)**; oracle-over-menu 0.304→0.484 (+0.180). VERDICT: naked LLM asker FAILS — its world knowledge biases to famous/popular titles (redundant direction), can't pick items that discriminate users in the collaborative latent space (no access to it). Strong negative for paper A: even an LLM can't capture the headroom → need a policy that knows collaborative structure. Cheap (69 API calls, rest cached; early-turn prompts identical across users). VERIFIED not-a-bug: prompt DOES include answers (dumped: "Star Wars IV: not seen"), cache keyed on full prompt+style. FAILURE MODE (minimal prompt): LLM marches through famous franchise (Star Wars I-VI) ignoring "not seen" feedback. FAIR RETRY (adaptive prompt: use answers, pivot on not-seen, prefer divisive over famous; STYLE=adaptive cache-versioned): behavior changed (now opens Fight Club, pivots) but STILL fails — llm 0.304→0.283 (−0.022), below random (+0.005), far below oracle (+0.180). CONCLUSION (fair, non-strawman): world knowledge ≠ collaborative informativeness; cheap LLM can't access the latent geometry that decides which questions discriminate. Expensive model/dedicated LLM-RS deferred. |
| ATTR-ORACLE | Attribute-only (genre+concept) oracle: is there ANY headroom? | attr_oracle.py | ✅ KEY | Calibrated instrument, held-out, 120u, T=8. realizable attributes (affinity order) 0.301→0.304 (+0.003 dead); **ATTRIBUTE ORACLE (18 genres + 100 genome concepts, privileged greedy) 0.301→0.332 (+0.031 = LOW CEILING)**; ITEM ORACLE 0.301→0.485 (+0.184) for reference. BREAKOUT (cheap realizable askers all near-dead): random -0.002, popular -0.001, affinity-all +0.003 (genre-affinity +0.007, concept-affinity -0.004). ORACLE ceiling: genre +0.025, concept +0.012, combined +0.031 (concepts even weaker than genres). VERDICT: attribute elicitation has a LOW ceiling (~6× below items) even with a perfect selector → NO attribute-elicitation baseline (EAR/SCPR/UNICORN/PEBOL) can beat +0.031 → SKIP building them; the oracle already bounds them. Cheap attribute askers ADDED for completeness (random/popular/affinity × genre/concept) — all land ≤+0.007. CAUSE (proven at oracle level, not answer-model/selector): a genre/concept centroid is a COARSE direction (avg of 100s of items) — localizes a broad region, can't pinpoint specific held-out items; items are precise points (~6× better). Paper A: elicitation story = ITEMS; attributes = clean rigorous NEGATIVE (oracle is the proof). |
| Q1 | RECONCILE R1(+0.09) vs calibrated(+0.025): controlled 2x2xK on P1 | reconcile_r1.py | ✅ KEY | MOSTPOP 0.4134. Best@K50 NDCG: **WRMF+z 0.491(+0.077, =R1✓)**, WRMF+noz 0.414(+0.000 DEAD), SVD+z 0.476(+0.062), **SVD+noz 0.467(+0.053, our no-harm instrument)**. Gap was NOT instrument weakness, it was: (1) BUDGET — all climb steeply with K; our +0.025 was @q15, @K50 SVD+noz=+0.053; (2) z-scoring ESSENTIAL for WRMF (noz→dead, factors entangle popularity) but ~irrelevant for SVD (+0.009); (3) protocol level. CONCLUSION: biased-SVD+noz DOMINATES — only config that is strong (+0.053) AND no-harm AND attribute-aware (WRMF can't be no-harm; strip z → dies). No-harm costs only ~0.009 for SVD. Instrument SOLIDIFIED = biased-SVD + no-z. Gains grow past K=50 (not plateaued); learned-policy job = capture oracle-level (+0.19@q15) at LOW budget. |
| S4-CAL2 | CLEANEST RESULT: calibrated instrument, held-out, q0-IDENTICAL, 3 metrics monotone | full_breakdown.py | ✅ KEY | ML-1M, 604 test users, held-out=half of rated, ask from answerable PROFILE, shared FIXED split → **q0=0.305 NDCG/0.084 Rec/0.960 RMSE identical all selectors**. q0→q15 ALL MONOTONE: NDCG popularity 0.332(+0.027) ≈ HELF 0.329 ≈ RMVA 0.328 > random/entropy 0.322(+0.017); RMSE all −0.035; Rec@10 +0.005. GENRES weak (NDCG +0.004, RMSE −0.004). TAKEAWAYS: (1) revealing helps monotonically on ALL 3 metrics (earlier 'NDCG flat' was P1/all-likes protocol artifact, q0=0.41 too strong); (2) realizable methods (HELF/RMVA) ≈ random (~0.33); (3) **ORACLE headroom HUGE: privileged greedy reaches NDCG 0.493 (0.301→0.388@q1→0.493@q15) vs realizable 0.327 vs random 0.328 → realizable captures ~0% of +0.165 headroom**. So selection matters ENORMOUSLY; established heuristics are weak (not the task ill-posed); the 0.33→0.49 gap = the LEARNED-POLICY PRIZE (motivates paper B). (4) genre elicitation flat (+0.004) but genres fine for RETRIEVAL (10/18 pure-taste) — never meant to win elicitation, not a regression. LOCKED checkpoint (code+paper+git+memory). full_breakdown.py has the oracle. |
| S4-CAL | RECOMMENDER RECALIBRATION (remove z-scoring → biased-MF; established) | instrument_svd.py, instrument_cal.py | ✅ KEY | DIAGNOSIS: score=W*z(pop)+conf*z(Q.u) Z-SCORED the residual to unit variance → undid ridge shrinkage → revealing could HURT. FIX (Koren'09 biased-SVD + HKV'08 WRMF fold-in, NO z-score, score=β*popb+Q.u): no-harm now holds for ALL REASONABLE selectors (RMVA/HELF/random/entropy flat-or-up under BOTH P1 and held-out fixed-target). HELF climbs (calibrated +0.016 P1 / +0.006 held-out; WRMF variant +0.048). TRADEOFF: calibrated gains << z-scored (the big earlier gains were partly artifact). POPULARITY irreducibly HURTS (no hyperparam fixes it w/o killing gains): asking only top-popular = degenerate narrow subspace → bad extrapolation; would need full Bayesian predictive-variance (out of scope); literature says don't use pop selection anyway → treat as known-degenerate baseline. ATTRIBUTES: biased-SVD taste factors understand genres (pure-taste prec@10: 10/18 ≥0.5; Animation .90 Children's 1.0 Horror 1.0 Musical .90 Drama .90 Comedy .80); pop-floor must be dropped for attribute queries. Saved Q_svd.npy/bi_svd.npy. NEXT: adopt calibrated instrument; rerun S4 panel on it. |
| S4-A6 | P1 panel SAVED as checkpoint but NOT USABLE (recommender mis-calibrated: z-scored residual) | s4_panel.py /tmp/s4i.log | ⚠️ | T=15/200u, P1 q0=0.411: EIG 0.452(+0.041), HELF 0.449(+0.038), RMVA 0.427(+0.015) climb MONOTONE; ENTROPY flat; RANDOM 0.396(-0.015); POPULARITY 0.321(-0.090). Attrs P2 q0=0.309: GENRES 0.364(+0.055). PROBLEM: popularity/random DECLINE = revealing true answers HURTS. ROOT CAUSE confirmed across P1+held-out+R1: score=W*z(pop)+conf*z(Q.u) Z-SCORES the personalization residual to unit variance, UNDOING the ridge's natural shrinkage -> redundant/weak reveals amplified to full strength -> corrupt the prior. Held-out does NOT fix it (popularity still -0.076). A calibrated recommender MUST guarantee revealing can't hurt. NEXT: S4-CAL = biased-MF/WRMF fold-in WITHOUT z-scoring (Hu-Koren-Volinsky'08, Koren'09). DO NOT ship this panel. |
| S4-A5 | CHEAP panel — fold-skip fixed [SUPERSEDED by S4-A6 save + S4-CAL recalibration] | s4_panel.py | ✅ | BUG: panel folded ONLY rated items, skipped "no" answers (dropped the discriminative signal). FIX: fold every asked seed y=1 liked/0 not (= instrument_u L97 / mf_foldin L59). RESULT T=30/200u: good selection now climbs MONOTONICALLY = revealing helps. HELF 0.411→0.454(+0.043,Rec0.080), RMVA 0.411→0.433(+0.022), EIG 0.438@q8(+0.027); ENTROPY flat(no-op); POPULARITY 0.411→0.294, RANDOM →0.382 (decline = redundant answers + P1 excludes asked-popular from re-rec → exhausts easy targets; matches R1 POPULAR 0.355<MOSTPOP). Magnitude: generic ridge+saved-Q ~0.43-0.45; full instrument K=50+learned enc =0.486 (validated num is @50 reveals). q8 NDCG: EIG 0.438, HELF 0.431, RMVA 0.426. ATTRS P2 q0=0.309: GENRES 0.350(+0.040). |
| S4-A4 | CHEAP panel — ON-RULER (canonical P1 + VAL-tuned blend) [SUPERSEDED by S4-A5: fold-skip bug undercounted reveals] | s4_panel.py | ✅ | ML-1M cold-test 200u. **ITEMS: canonical P1 (rel=all likes−seeds, exclude seeds), q0=MOSTPOP=0.411 EXACTLY (matches DRE/mf_foldin anchor).** Blend (W,conf-cap) tuned on VAL. NDCG@10 q8 / Rec@10: **EIG 0.438/0.075 (+0.027 BEST)**, HELF 0.422/0.070 (+0.011), RMVA 0.418/0.071 (+0.007), ENTROPY 0.410/0.065 (~flat no-op), RANDOM 0.401/0.065 (no-op), **POPULARITY 0.361/0.060 (−0.050 BELOW baseline)**. **ATTRS: leakage-free P2, q0=0.309: GENRES 0.350/0.092 (+0.040 BEATS)**, GENOME 0.276 (−0.033), OOS 0.283 (−0.027). ORDERING = EIG>HELF>RMVA>entropy≈random>popularity — LITERATURE-CONSISTENT (Rashid: HELF>pop, entropy poor; RBMF: representative>pop; popular=redundant low-info). Popularity-DECLINE corroborated by R1 (POPULAR seeds 0.355 < MOSTPOP 0.413). Two protocols: items P1 (no leakage, 0.41 anchor), attrs P2 (held-out required to avoid answer-scores-own-target leakage, 0.32 anchor). |
| S4-A3 | CHEAP panel — fixed shared split [SUPERSEDED by S4-A4: off-ruler protocol, q0=0.32 not 0.41; untuned blend] | s4_panel.py | ✅ | ML-1M cold-test 120u, q0=0.326 identical all policies (fixed split fixed the per-policy-split confound). But protocol was P2-style (rel=half likes, exclude whole profile) → MOSTPOP 0.32 drifted off the 0.41 ruler; blend untuned. Headline then: GENRES 0.383, POPULARITY 0.345. Replaced by on-ruler S4-A4. |
| S4-A2 | CHEAP panel on corrected model [SUPERSEDED by S4-A3: per-policy split confound] | s4_panel.py | ✅ | ML-1M cold-test 120u, NDCG@10 q5 / Rec@10 q5: RANDOM-item 0.320/0.078, POPULARITY 0.318/0.073, ENTROPY 0.314/0.077, HELF 0.309/0.078, EIG-item 0.291/0.070, **GENRES(lift+additive) 0.362/0.091 = BEST**, GENOME-CONCEPTS 0.278/0.072, OOS-CONCEPTS 0.301/0.069. Recipe: graded base-rate answer + EAR-additive + bounded weight + profile-only. (Numbers confounded by per-policy random split — see S4-A3.) |
| S4-A | CHEAP baseline panel (no-train, item+attribute) on instrument [SUPERSEDED by S4-A2: used wrong attr fold-in] | cheap_baselines.py | ✅ | ML-1M cold-test, held-out-half protocol, NDCG@10 q6 / Recall@10 q6: RANDOM-item 0.322/0.078, POPULARITY-item 0.323/0.073, ENTROPY 0.315/0.077, HELF(Rashid) 0.326/0.079, RANDOM-genre 0.313/0.076, POPULAR-genre 0.331/**0.088**, MAXENT-genre 0.325/0.084, UNCERTAINTY-item 0.328/0.084, **EIG-MIXED 0.337/0.084 (best NDCG, principled anchor)**. Realizable gains modest (q0~0.31→0.33) = realizable<<oracle(0.6) gap. Genres competitive/lead Recall; HELF>pop>entropy (replicates Rashid). NEXT (Panel B trained): Golbandi tree, RMVA, DRE, EAR/UNICORN-DQN, PEBOL. |
| S3c | Plot themes (Tag Genome) + established text→CF projection | plot_themes.py | ✅ | Enriching item text with top-18 Tag-Genome themes improves concept→item retrieval: median expected-film rank 104(title+genres)→38(title+themes); "time travel"→#1, "talking animals"→25, "dinosaurs"→52. Replicated established CONTRASTIVE projection (RLMRec/CLCRec InfoNCE) — WORSE (206) than linear ridge for ARBITRARY-concept projection (overfits item manifold, poor off-manifold extrapolation; cf DaRec). KEEP ridge, cite contrastive. Residual ceiling = theme≠taste in collaborative Q. Genome local: genome-scores.csv (95% coverage). |
| S3 | Unified items+attributes(+concepts) in one space | instrument_u_attr.py, project_concepts.py, project_concepts_nogenre.py | ✅ items+attrs · ⚠ concepts PARTIAL | genres: genre-like lifts items +0.36, mixed>item-only (DONE). OPEN CONCEPTS (honest, no-genre-word test + title-only ablation): MIXED — works for title-recoverable + taste-clustered concepts ("outer space aliens"→Day the Earth Stood Still/Mars Attacks; "AI robots"→Akira/Ghost in the Shell; "boxing"→Rocky; "spy cold war"→From Russia with Love) but FAILS for plot themes absent from item text ("dinosaurs"→Snow White; "talking animals"→Field of Dreams; "prison escape"→Se7en). CAUSE: item text=title+genres only (no plot synopsis). FIX: enrich item text. Earlier "all concepts work" was CONTAMINATED (every phrase had a genre word). | ⚠ enrich text to complete |
| S4 | Run all elicitation baselines on the instrument | ⬜ | ⬜ | — | — |
| S5 | Continuous-action (Wolpertinger) policy — beat baselines | ⬜ | ⬜ | — | — |
| S6 | Naked LLM askers as baselines | ⬜ | ⬜ | — | — |

## PART C — Paper B (learned policy, S5)
| Stage | What | Script | Status | Headline | DoD? |
|---|---|---|---|---|---|
| B0-BC | Oracle-imitation (behaviour cloning) of the greedy oracle | scripts/paper2/policy_bc.py | ❌ (expected negative) | Frozen calibrated instrument; state=u_t; Wolpertinger query head snapped to pool. BC reaches only ~0.11 oracle-match-acc on TRAIN (≈16× chance over pool=150) and transfers to BELOW random on TEST (policy 0.304→0.243, −0.060; HELF +0.013; random −0.004). DIAGNOSIS = IMITATION GAP: the oracle is PRIVILEGED (peeks at held-out to pick the NDCG-maximizing item), so its choices are not reproducible from observable state u_t → BC can't fit them and confident-wrong picks mislead the fold-in. Oracle = good ceiling, bad teacher. NEXT: (1) RL finetune (optimize real NDCG/Recall reward from observable state = thesis Wolpertinger method); (2) realizable teacher (amortized-EIG over belief, or Golbandi tree) then imitate. | partial |
| B1-RL | BC(HELF)+REINFORCE learned policy [PLAYLIST COLLAPSE] | scripts/paper2/policy_rl.py | ❌ KEY | 400 train, BC30+RL25 ep, pool=120, T=8, 250 test. RL train reward FLAT (~0±0.008, never climbed over 25 ep). TEST q0→q8: random −0.008, HELF +0.000, RL/BC −0.001, oracle +0.187. BC mechanically imitated HELF (learnable fixed order, unlike oracle) but HELF≈random so inherited ~0 value; RL added nothing. CONVERGENT EVIDENCE (4 lines) that realizable selection ceiling on dense ML-1M is LOW: (1) BC-on-oracle 11% match<random (imitation gap); (2) all heuristics≈random; (3) RMVA=D-optimal≈random; (4) RL flat. Oracle's +0.187 headroom is LARGELY PRIVILEGED (knows answerability + held-out). Implication: learned-policy contribution NOT demonstrable on dense ML-1M; needs selection-friendly testbed (sparse/cold/cross-domain) OR reframe as rigorous low-realizable-ceiling finding. Consistent with prior memory (realizable captures ~1/5 oracle; collaborative generalisation substitutes for adaptivity on dense catalogs). |

**B1-RL trace (what it asks):** the REINFORCE policy collapsed to a STATIC PLAYLIST — identical question sequence for every user (Star Wars VI, Jurassic Park, Babe, Star Wars I, American Pie, Men in Black, Independence Day, Titanic), IGNORING answers (u310 likes Jurassic Park, u908 dislikes it → both get same next Q). State-independent (query MLP ≈ constant); questions are popular blockbusters mostly UNSEEN (no signal). = the playlist-collapse failure mode (cf. bot-play saga). Clean fixed-split eval (q0=0.277 identical all methods): RL +0.002 ≈ HELF +0.002 ≈ random −0.003 vs oracle +0.187. CAUSE: flat/high-variance REINFORCE reward + BC-fixed-order warm start → no pressure to use state. FIX = Wolpertinger DDPG ACTOR-CRITIC (per-state low-variance signal, the standard remedy). NEXT: replicate Wolpertinger (Dulac-Arnold 2015) on its original benchmark, validate, then apply → disambiguates method-weakness vs realizable-ceiling-wall.
| B2-WolpRep | Wolpertinger replication (Dulac-Arnold 2015), mechanism-level | scripts/paper2/wolpertinger_replicate.py | ✅ | Self-contained contextual bandit, N=2000 discrete actions in R^8, optimal known (full-enum ceiling). DDPG actor-critic + kNN proto-snap. REPRODUCES paper: (1) critic learns — norm-score 0.919 at k=N (critic over all actions); (2) k-tradeoff (k=1 0.007→k=5 0.177→k=10 0.306→k=50 0.522→k=200 0.653→k=N 0.919): more neighbours→closer to ceiling = Wolpertinger's headline. Machinery VALIDATED → a CASPER failure is now attributable to the task, not the impl. Provides the per-state low-variance critic signal REINFORCE lacked (the playlist-collapse fix). NEXT: port DDPG actor-critic to CASPER (frozen instrument env, reward=NDCG). |
| B3-Wolp | Wolpertinger DDPG applied to CASPER (the thesis method) | scripts/paper2/wolpertinger_casper.py | ❌ KEY | Validated DDPG actor-critic + kNN on frozen instrument, dense ΔNDCG reward, shared split. Clean (q0=0.277 identical, 250 test): WOLPERTINGER +0.000 ≈ HELF +0.002 ≈ random +0.002, oracle +0.186. The replication-VALIDATED method is FLAT on CASPER → failure attributable to the TASK not the impl (the disambiguation). **5 CONVERGENT LINES now confirm the realizable selection ceiling on dense ML-1M is ~nil:** (1) BC-oracle imitation gap; (2) heuristics≈random; (3) RMVA/D-optimal≈random; (4) REINFORCE playlist-collapse; (5) validated Wolpertinger≈flat. Oracle's +0.19 headroom is LARGELY PRIVILEGED (answerability + held-out). CONCLUSION: strategic what-to-ask has no realizable value on dense ML-1M cold-start (collaborative prior already explains the variance; matches prior memory). DECISION: Paper B "method beats baselines" NOT demonstrable on ML-1M → pivot to selection-friendly testbed (sparse/cold/cross-domain, large oracle-random gap) OR reframe as rigorous low-realizable-ceiling negative. |
| B4-RealCeil | Realizable-ceiling test in PROFILE mode (decides if learned policy worthwhile) | scripts/paper2/realizable_ceiling.py | ✅ KEY DECISIVE | 250 users, NDCG@10 on true held-out: HELF +0.017, **REALIZABLE-ORACLE (greedy on profile-internal validation, NO test peek) +0.013 ≈ HELF**, TRUE-ORACLE (test-peek) +0.131. => NO realizable headroom even in profile mode: the best any realizable policy can do ≈ the simple baseline; the +0.13 oracle gap is ENTIRELY PRIVILEGED (needs the actual held-out). Cause: taste vector well-pinned from any few (redundant) items → which to ask barely changes the realizable estimate. **6th convergent line.** CONCLUSION (airtight): strategic what-to-ask has ~nil realizable value on dense ML-1M (global OR profile). Building a learned policy here is pointless — saved by this cheap test. Decision: pivot to selection-friendly dataset (sparse/cold/cross-domain, weak collaborative prior) OR ship the rigorous negative. Real elicitation (answerable Q's) DOES work (+0.015 NDCG/−0.035 RMSE/+Recall, Paper A); only strategic SELECTION adds nothing realizable. |

## PART D — Multi-dataset replication of the realizable-ceiling probe (established datasets)
scripts/paper2/dataset_probe.py (explicit) + npz_probe.py (binary-implicit profiles). PROFILE-mode realizable-ceiling
(HELF vs REALIZABLE-ORACLE[profile-internal val, no test peek] vs TRUE-ORACLE[test peek]); NDCG@10 q0→q8 delta.
| dataset | q0 | HELF Δ | REALIZABLE-ORACLE Δ | TRUE-ORACLE Δ | headroom |
|---|---|---|---|---|---|
| ML-1M (dense) | 0.277 | +0.017 | +0.013 | +0.131 | NO (privileged gap only) |
| ML-100k (dense) | 0.327 | +0.007 | −0.001 | +0.117 | NO (same) |
| LastFM (curated 360-item) | 0.541 | +0.011 | +0.016 | +0.019 | NO (saturated prior) |
| Yelp multi-city (648-item) | 0.152 | +0.042 | +0.041 | +0.041 | NO (oracle=HELF; elicitation helps, selection doesn't) |
| Amazon cross-domain (648) | 0.335 | +0.000 | +0.000 | +0.000 | NO (flat) |
| Amazon balanced (648) | 0.270 | +0.001 | +0.001 | +0.001 | NO (flat) |
| Amazon raw (Digital_Music/All_Beauty/Appliances) | — | — | — | — | DEGENERATE (no usable 5-core) |
**VERDICT (general, 6 configs): NO established dataset shows realizable selection headroom (realizable-oracle ≈ HELF).**
Asking answerable Q's helps (e.g. Yelp +0.04); learning WHAT to ask adds ~nil realizable value. Dense=privileged gap;
sparse/curated=no gap or saturated; genuinely-sparse Amazon=degenerate. The Paper-B negative GENERALIZES across
ML-1M/ML-100k/LastFM/Yelp/Amazon. (Bug fixed: SGD LR too high → NaN factors → false q1=0 collapse; LR≤0.01, 0.003 for very sparse.)

## PART E — THE METRIC WAS THE PROBLEM: long-tail / debiased regime (BREAKTHROUGH)
The dense full-catalogue NDCG@10 "no realizable value / popularity-is-ceiling" conclusion was a METRIC ARTIFACT.
Evidence: folding a user's WHOLE profile (realizable) barely moves full-cat NDCG@10 (0.291→0.326) but clearly improves
RMSE (0.976→0.908) — personalization IS real, just invisible in a blockbuster-saturated top-10. The TRUE-oracle's
advantage is real on ALL metrics (NDCG +0.16, Recall@50 +0.04, RMSE −0.08), NOT a top-10 gaming artifact.
**TAIL-RESTRICTED (exclude top-300 popular), NDCG@10 (tail_probe.py, 296 users):**
| selector | q0 | q1 | q4 | q8 | qALL |
|---|---|---|---|---|---|
| random | 0.039 | 0.049 | 0.057 | 0.067 | 0.093 |
| HELF | 0.039 | 0.053 | 0.065 | 0.070 | 0.093 |
| realizable-oracle (pv-greedy) | 0.039 | 0.049 | 0.054 | 0.065 | 0.072 |
| **TRUE-ORACLE** | 0.039 | **0.185** | 0.279 | **0.302** | 0.297 |
KEYS: (1) on the tail, realizable elicitation DOUBLES NDCG (0.039→0.093) — real value, visible. (2) TRUE-oracle is
HUGE (+0.26, ~8×; +0.15 from ONE question) — enormous selection headroom, opposite of dense. (3) the pv-greedy
"realizable-oracle" is UNRELIABLE (0.072 < HELF 0.093 — overfits the noisy profile-internal proxy), so it is NOT a
valid realizable-ceiling estimator — earlier "realizable≈HELF→no headroom" claims that leaned on it are NOT
trustworthy. => REALIZABLE CEILING IS OPEN on the tail, and the prize is large (+0.26). This is the regime to build
Paper B (learned policy): dense top-10 hid everything; the long-tail/debiased lens reveals real, large elicitation +
selection value. User's skepticism vindicated: wrong metric/regime, not a broken field.

### PART E confirmation under ESTABLISHED protocols (tail_principled.py)
Cremonesi/Koren/Turrin RecSys2010 long-tail (head = items covering X% of interaction mass; swept) — ROBUST:
| head mass (% items) | random Δ | HELF Δ | ORACLE Δ | oracle−HELF |
|---|---|---|---|---|
| 20% (2.1%) | +0.043 | +0.039 | +0.270 | +0.231 |
| 33% (4.3%) | +0.033 | +0.042 | +0.280 | +0.238 |
| 50% (8.8%) | +0.025 | +0.029 | +0.275 | +0.246 |
IPS-debiased Recall@10 (Schnabel ICML2016/Yang RecSys2018, full catalogue): random/HELF +0.000, oracle +0.018 (flat).
CONCLUSION: the tail finding REPLICATES under Cremonesi's principled head/tail split, robust across cutoffs (NOT the
arbitrary top-300). HONEST CAVEAT: IPS-over-full-catalogue is flat — reweighting scores doesn't change which items
reach top-10 (still blockbusters); elicitation value appears ONLY when ranking IN tail space (head excluded from
CANDIDATES = the long-tail recommendation task). SCOPED CLAIM: on the long-tail recommendation task (Cremonesi 2010),
elicitation has large realizable value (+0.04, robust) + large oracle headroom (+0.27); on full-cat top-N it's hidden.
This is the citable home for Paper B (learned policy) — robust prize, open realizable ceiling.

## PART F — Reconstruction encoder (Paper B Phase A) — GATES PASS
scripts/paper2/encoder_recon.py. Learned attention fold-in trained on MASKED/SHUFFLED reveals to reconstruct the
user's UNREVEALED likes, IPS/tail-weighted (no popb crutch, balanced pos/neg). Eval = long-tail task (Cremonesi
head-33%), HELF reveal order, vs ridge fold-in & popularity (frozen Q_svd decoder for ridge/pop).
| method | q0 | q1 | q2 | q4 | q8 |
|---|---|---|---|---|---|
| popularity | 0.064 | 0.064 | 0.064 | 0.064 | 0.064 |
| ridge fold-in | 0.064 | 0.076 | 0.073 | 0.091 | 0.098 |
| encoder (FROZEN Q) | 0.064 | 0.065 | 0.074 | 0.077 | 0.093 | (ties ridge — ridge is optimal linear fold-in on fixed factors)
| **encoder (JOINT Q)** | 0.064 | **0.077** | **0.085** | **0.098** | **0.119** | (BEATS ridge +21% @q8, at every q)
GATES: A1 (enc>ridge & >pop @q4,q8) PASS; A2 (enc>ridge @q1,q2) PASS; A3 (monotone) PASS. Bugs fixed en route: popb
offset swamped encoder + 37:1 neg imbalance -> encoder learned nothing (worse than pop); fix = train on pure Q.u
(no popb) + balance pos/neg weight. KEY: frozen-Q caps encoder at ridge (optimal linear fold-in); the win needs
JOINT factor learning (encoder's degree of freedom ridge lacks). => reconstruction/encoder-decoder premise VALIDATED
on the long-tail task. PROCEED to Phase B (elicitation policy on reconstruction info-gain).
