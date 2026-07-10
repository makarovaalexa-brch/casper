# THE POLICY ARENA -- BUILD + DEV RESULTS (v2, GATED WORLD)

Per DESIGN_SHEET_POLICY_ARENA.md (signed 2026-07-09), rebuilt after the pre-verdict audits (experiments/ARENA_CODE_AUDIT.md; experiments/BLIND_VALIDATION.md F14). DEV evaluation on SYNTHETIC users ONLY; the 173/300 real-judged users are NEVER touched. NO LLM calls. Deterministic (seed 123).

## FOLD PICK (STEP 0)

**Deep-Sets fold-v3** (`.cache/i25_fold_v3_best.pt`, clean_frac=0.30, val NDCG@10 0.4356) is the arena belief. The Set-Transformer contingency FINISHED: best val 0.4123 (ep6), decisively below Deep-Sets' 0.4311, and re-gated WORSE -- FAIL on G2/G3/G4/G5/G6 (only G1/G2b/G7 pass), including the decisive G2 GoT gate that DS PASSES, and even G6 (its own remedy target) worse than DS (-0.0119 vs -0.0035). The contract required strict dominance (all gates + val + G4) to displace; it fails on every count. DS fold-v3 posture: decisive two-channel gates (G2 GoT +0.184, G2b prolific +0.316, G7 implicit-ablation +0.020) PASS; G5 (-0.028 vs v2 on v2's item-heavy regime) and G6 (-0.0035 one-step, marginal) are diagnosed failures, not the two-channel purpose. `ALL GATES PASS: False` is the honest record; we do NOT fall back to fold-v2.

## THE WORLD (fix #7 -- certification chain repaired)

The arena answerer is THE FITTED, GATED v2.1 MODEL, loaded and run (NOT the hand-parameterized sampler the blind validation flagged as credential transfer):

- knowledge: per-channel ordinal logistics from `.cache/dans/models_v21.json` (sha f53e23a7692d) -- full feature set (co-knowledge, fame, census/buffness, era), EQUATED flutter-free trait sigma (concept 0.199 / entity 0.301 / item 0.621), entity per-cut random effects; generation recipe = dans_stages.know_probs verbatim (per-user seed 123*7777+uid).
- values: v2.1 EASE-backbone value models (t(u,i) = real rating if rated else EASE prediction over the 9352-item universe); rated bank items pass through (know_well + real centered rating, fid=data).
- universe (fix #5): ALL judged questions = 1128 concepts + 500 attributes (IMDb entities) + 800 bank items = 2428 -- the signed sheet's 2,428. No pools, no coverage sampling, no alphabetical ties.
- belief: the picked Deep-Sets fold-v3, FIXED. Documented deviation: the fold was trained on the earlier sampler world and is deployed unchanged on the gated world (same token vocabulary; mild distribution shift; retraining the fold on the gated world is future work).
- LENIENT regime: know>=1 answered; no_clue = refusal = consumed turn, belief unchanged (refusal tokens NEVER fold -- the fold-v3 no-clue-not-negative caveat), user kept (E1).

### WORLD FUEL CHECK (the world's own gate stamp, not inherited credentials)

Trait ICC per channel on the ARENA WORLD AS BUILT (600 synthetic users, question-residualized one-way user share; flutter=0 by construction) vs the CORRECTED targets (equate_v21.json icc_trait_corrected), tolerance 0.02 (the v2.1 G2 rule):

| channel | margin | arena trait ICC | corrected target | verdict | base rate |
|---|---|--:|--:|:--:|--:|
| concept | k>=1 | 0.0284 | 0.0427 | MATCH | 0.750 |
| entity | k>=1 | 0.0431 | 0.0374 | MATCH | 0.821 |
| item | k>=2 | 0.0710 | 0.0729 | MATCH | 0.721 |

**ALL MATCH: True**

## AUDIT FIXES APPLIED (methods; credit: the blind pre-verdict code audit + program validation)

1. b4 target-peek FIXED: objective is now the blind smooth ranking utility J(z)=tau*logsumexp(decode(z)/tau); expected gain = p_ans_blind * (J(z')-J(z)); never touches held-out targets.
2. b2 partial-cache FIXED: resume-from-partial + assert len(seq)==Tmax (the 10/24 truncated cache was discarded with the old world's caches).
3. b2 construction cohort raised to 300 TRAIN users (was 50); candidate PRE-SCREEN = top-300 by 1-question cohort gain over ALL 2,428 (documented compute deviation; seed-repeat robustness in the addenda).
4. answer caches keyed by cohort-config sha + per-user known-set hashes verified on load; ALL old-world caches invalidated (archived to .cache/arena_old_prefix4/).
5. universe rebuilt to the signed sheet (2,428 judged questions; was 1,530 sampler vocab).
6. b3 gets the pre-registered value-ranked tail (prescreen-gain order) when its list exhausts under refunds.
7. THE WORLD runs the fitted gated v2.1 models (see above) + carries its own fuel stamp.
8. one shared cand_M=100 for b4/A/B/C; all blind arms share ONE population prior (per-question TRAIN answer rate); policies receive only the observed dialogue, never the answerer's internals.

## DEPLOYABILITY (blind agents)

b4/A/B/C choose questions from the current belief z + the shared TRAIN population prior only; value forecasts decode z over region members; the true answer is observed only AFTER asking. D branches on OBSERVED answer polarity of already-asked questions. Only the labelled context arms (true-table, clairvoyant) peek at realized answers -- never cited.

## E7 SYMMETRY TABLE

All arms share: gated v2.1 world, fold-v3 belief, full 2,428 universe, lenient regime, SAME DEV-TEST users (E1), SAME shared population prior + cand_M for blind arms. Differs ONLY by selection policy + construction cohort:

| arm | family | learns from | construction cohort | reachable set/turn | privileged? |
|---|---|---|---|---|---|
| b0 cold | orientation | nothing | - | - | no |
| b1 concept-entropy | static | TRAIN answer-rates | 1000 TRAIN | 1128 concepts | no |
| b2 learned static | static | TRAIN NDCG greedy | 300 TRAIN | prescreen top-300 | no |
| b3 = b2+skip+tail | static+skip | = b2 | = b2 | = b2 + gain-ranked tail | no |
| b4 myopic (J) | model-based blind | NONE | - | cand_M=100 | no |
| A scorer-v2 | adaptive learned | TRAIN 1+2-step gains | 1000 TRAIN | cand_M=100 | no |
| B ask-gradient | adaptive | calibration only | - | cand_M=100 | no |
| C CAT-router | adaptive | NONE | - | cand_M=100 | no |
| D Golbandi tree | adaptive learned | TRAIN split NDCG | 300 TRAIN | tree + b2 + tail | no |
| ttab router | CONTEXT labelled | peeks realized answers | - | cand_M=100 | YES -- never cited |
| clairvoyant | CONTEXT labelled approx | peeks realized answers | - | M=200 answered | YES -- never cited |

## PRE-REGISTERED DEV VERDICTS (printed BEFORE results; E5)

Primary: endpoint NDCG@50 at T=24; also endpoint@10, anytime@10, budgets 8/16/24; paired bootstrap CI + MDE on every delta (E4).

- **b4 vs b2** (computation-suffices): b4 WIN (CI excl 0) = adaptivity pays via computation; learned policies must then beat b4. TIE = computation alone does not pay here.
- **each class (A,B,C,D) vs b2 AND vs b4**: WIN = CI excl 0 above b2 AND >= b4. If only b4 wins: 'adaptivity pays via computation; learning adds nothing yet' (stated exactly).
- TAU on DEV-VAL (TAU=24 == b2 exactly); pure TAU=0 reported alongside. No post-hoc metric selection. DEV ONLY -- no headline claims; the 173/300 are out of bounds.

## DEV-TEST RESULTS (endpoint = T=24)

| arm | NDCG@50 T24 | NDCG@10 T24 | anytime@10 | @50 T8 | @50 T16 |
|---|--:|--:|--:|--:|--:|
| b0 cold | 0.2369 | 0.2424 | 0.2424 | 0.2369 | 0.2369 |
| b1 concept-entropy | 0.2113 | 0.2217 | 0.2383 | 0.2344 | 0.2193 |
| b2 learned static | 0.2884 | 0.2709 | 0.2936 | 0.2945 | 0.2911 |
| b3 b2+skip+tail | 0.2684 | 0.2563 | 0.2903 | 0.2926 | 0.2901 |
| b4 myopic (blind J) | 0.2296 | 0.2625 | 0.2700 | 0.2517 | 0.2443 |
| A scorer (tau24) | 0.2884 | 0.2709 | 0.2936 | 0.2945 | 0.2911 |
| A scorer pure | 0.2464 | 0.2601 | 0.2673 | 0.2417 | 0.2470 |
| B gradient (tau24) | 0.2884 | 0.2709 | 0.2936 | 0.2945 | 0.2911 |
| B gradient pure | 0.2328 | 0.2637 | 0.2697 | 0.2475 | 0.2517 |
| C CAT (tau24) | 0.2884 | 0.2709 | 0.2936 | 0.2945 | 0.2911 |
| C CAT pure | 0.2546 | 0.2742 | 0.2643 | 0.2576 | 0.2571 |
| D Golbandi (tau0) | 0.2743 | 0.2603 | 0.2792 | 0.2791 | 0.2900 |
| D Golbandi pure | 0.2743 | 0.2603 | 0.2792 | 0.2791 | 0.2900 |
| [ctx] true-table (n=50) | 0.4672 | 0.5330 | 0.5119 | 0.4479 | 0.4642 |
| [ctx] clairvoyant approx (n=50) | 0.4935 | 0.5657 | 0.5512 | 0.4755 | 0.4940 |

(context rows are computed on the first 50 DEV-TEST users -- do NOT difference them against full-cohort rows.)

### Paired deltas vs b2 and vs b4 (DEV-TEST endpoint@50 T=24; E4)

| arm | vs b2 | vs b4 |
|---|---|---|
| b1 concept-entropy | -0.0772 [-0.1001,-0.0541] MDE 0.0328 (n=160) | -0.0184 [-0.0414,+0.0046] MDE 0.0327 (n=160) |
| b3 b2+skip+tail | -0.0201 [-0.0308,-0.0095] MDE 0.0151 (n=160) | +0.0387 [+0.0133,+0.0636] MDE 0.0364 (n=160) |
| b4 myopic (blind J) | -0.0588 [-0.0835,-0.0344] MDE 0.0356 (n=160) | - |
| A scorer (tau24) | +0.0000 [+0.0000,+0.0000] MDE 0.0000 (n=160) | +0.0588 [+0.0344,+0.0835] MDE 0.0356 (n=160) |
| A scorer pure | -0.0421 [-0.0647,-0.0192] MDE 0.0326 (n=160) | +0.0167 [-0.0033,+0.0384] MDE 0.0296 (n=160) |
| B gradient (tau24) | +0.0000 [+0.0000,+0.0000] MDE 0.0000 (n=160) | +0.0588 [+0.0344,+0.0835] MDE 0.0356 (n=160) |
| B gradient pure | -0.0556 [-0.0805,-0.0309] MDE 0.0354 (n=160) | +0.0032 [-0.0017,+0.0101] MDE 0.0086 (n=160) |
| C CAT (tau24) | +0.0000 [+0.0000,+0.0000] MDE 0.0000 (n=160) | +0.0588 [+0.0344,+0.0835] MDE 0.0356 (n=160) |
| C CAT pure | -0.0338 [-0.0587,-0.0083] MDE 0.0362 (n=160) | +0.0250 [+0.0062,+0.0450] MDE 0.0276 (n=160) |
| D Golbandi (tau0) | -0.0141 [-0.0257,-0.0028] MDE 0.0165 (n=160) | +0.0447 [+0.0196,+0.0702] MDE 0.0362 (n=160) |
| D Golbandi pure | -0.0141 [-0.0257,-0.0028] MDE 0.0165 (n=160) | +0.0447 [+0.0196,+0.0702] MDE 0.0362 (n=160) |

**b4 vs b2 (computation-suffices check): -0.0588 [-0.0835,-0.0344] MDE 0.0356 (n=160)**

b2 per-step greedy gains (first 8): [0.0253, 0.014, 0.0118, 0.0099, 0.0063, 0.0105, 0.0034, 0.0035]

## MECHANISM READOUTS (does coarse-to-fine EMERGE?)

| arm | refusal% (T1-4/T21-24) | channel mix T1-4 -> T21-24 (conc/ent/item %) | mean prior-p_ans asked (early/late) | divergence-from-b2 turn |
|---|---|---|---|---|
| b1 | 46%/49% | 100/0/0 -> 100/0/0 | 0.50/0.50 | 1.0 |
| b2 | 17%/17% | 75/25/0 -> 75/0/25 | 0.81/0.82 | 24.0 |
| b3 | 0%/0% | 66/33/0 -> 49/40/10 | 0.83/0.86 | 7.2 |
| b4 | 0%/0% | 0/0/100 -> 0/0/100 | 1.00/1.00 | 1.0 |
| A | 17%/17% | 75/25/0 -> 75/0/25 | 0.81/0.82 | 24.0 |
| B | 17%/17% | 75/25/0 -> 75/0/25 | 0.81/0.82 | 24.0 |
| C | 17%/17% | 75/25/0 -> 75/0/25 | 0.81/0.82 | 24.0 |
| D | 1%/18% | 28/0/71 -> 74/18/7 | 0.99/0.80 | 1.0 |

**Channel-mix one-liners:**
- b1: STATIC mix (100/0/0)
- b2: mix moved 75/25/0 -> 75/0/25
- b3: mix moved 66/33/0 -> 49/40/10
- b4: STATIC mix (0/0/100)
- A: mix moved 75/25/0 -> 75/0/25
- B: mix moved 75/25/0 -> 75/0/25
- C: mix moved 75/25/0 -> 75/0/25
- D: mix moved 28/0/71 -> 74/18/7

## TRANSCRIPTS (3 DEV-TEST users, b4 arm; question names + answers)

**user 88848** (b4):

- T1: item[Alien (1979)] -> know_well val=liked
- T2: item[Spirited Away (Sen to Chihiro no kamikakushi) (2001)] -> know_well val=loved
- T3: item[Aliens (1986)] -> know_well val=liked
- T4: item[Shining, The (1980)] -> know_well val=liked
- T5: item[City of God (Cidade de Deus) (2002)] -> know_well val=liked
- T6: item[Clockwork Orange, A (1971)] -> know_well val=liked
- T7: item[Being John Malkovich (1999)] -> know_well val=liked
- T8: item[Apocalypse Now (1979)] -> know_well val=liked
- T9: item[Dr. Strangelove or: How I Learned to Stop Worrying and Love the Bomb (1964)] -> know_well val=liked
- T10: item[2001: A Space Odyssey (1968)] -> rough val=meh
- T11: item[Full Metal Jacket (1987)] -> know_well val=loved
- T12: item[Blade Runner (1982)] -> know_well val=liked

**user 121233** (b4):

- T1: item[Alien (1979)] -> know_well val=loved
- T2: item[Aliens (1986)] -> know_well val=meh
- T3: item[Spirited Away (Sen to Chihiro no kamikakushi) (2001)] -> know_well val=liked
- T4: item[Dr. Strangelove or: How I Learned to Stop Worrying and Love the Bomb (1964)] -> know_well val=liked
- T5: item[Chinatown (1974)] -> know_well val=liked
- T6: item[Citizen Kane (1941)] -> know_well val=liked
- T7: item[Apocalypse Now (1979)] -> know_well val=loved
- T8: item[Ocean's Eleven (2001)] -> know_well val=liked
- T9: item[2001: A Space Odyssey (1968)] -> know_well val=liked
- T10: item[This Is Spinal Tap (1984)] -> know_well val=liked
- T11: item[Lord of the Rings: The Two Towers, The (2002)] -> know_well val=liked
- T12: item[Graduate, The (1967)] -> rough val=liked

**user 86711** (b4):

- T1: item[Alien (1979)] -> know_well val=liked
- T2: item[Shining, The (1980)] -> know_well val=liked
- T3: item[Dr. Strangelove or: How I Learned to Stop Worrying and Love the Bomb (1964)] -> know_well val=loved
- T4: item[Chinatown (1974)] -> know_well val=liked
- T5: item[Citizen Kane (1941)] -> know_well val=liked
- T6: item[Apocalypse Now (1979)] -> know_well val=loved
- T7: item[2001: A Space Odyssey (1968)] -> know_well val=liked
- T8: item[Vertigo (1958)] -> know_well val=meh
- T9: item[Sting, The (1973)] -> know_well val=liked
- T10: item[Exorcist, The (1973)] -> know_well val=liked
- T11: item[North by Northwest (1959)] -> know_well val=liked
- T12: item[Clockwork Orange, A (1971)] -> know_well val=meh

## VERDICTS (DEV; pre-registered contrasts)

- **b4 vs b2: LOSS** (-0.0588 [-0.0835,-0.0344] MDE 0.0356 (n=160)) -> blind model-based myopia hurts here.
- **A vs b2: TIE** (+0.0000 [+0.0000,+0.0000] MDE 0.0000 (n=160)); **vs b4: WIN** (+0.0588 [+0.0344,+0.0835] MDE 0.0356 (n=160))
- **B vs b2: TIE** (+0.0000 [+0.0000,+0.0000] MDE 0.0000 (n=160)); **vs b4: WIN** (+0.0588 [+0.0344,+0.0835] MDE 0.0356 (n=160))
- **C vs b2: TIE** (+0.0000 [+0.0000,+0.0000] MDE 0.0000 (n=160)); **vs b4: WIN** (+0.0588 [+0.0344,+0.0835] MDE 0.0356 (n=160))
- **D vs b2: LOSS** (-0.0141 [-0.0257,-0.0028] MDE 0.0165 (n=160)); **vs b4: WIN** (+0.0447 [+0.0196,+0.0702] MDE 0.0362 (n=160))

NO headline claim -- DEV only; the 173/300 real users are out of bounds (execution scope guard).


---

## RELIABILITY UPGRADES (author-directed addendum)

### (1) Training-seed repeats (TRSEED lesson: a single-seed win is not a result)

b2 (greedy static) and the DEV-VAL leader (**A**) rebuilt on 3 DISJOINT TRAIN user subsets (b2 on 120 users/seed; leader retrained per seed). DEV-TEST endpoint@50 (mean +/- across-seed sd). Calibration-only classes (B, C) have no training seed.

| arm | seed endpoints@50 | mean +/- sd |
|---|---|---|
