# Attribute/Concept ANSWER-MODEL investigation log

Question: how to determine a cold-start user's ANSWER to "do you like genre/concept X?" so that POPULAR-attribute
elicitation grows MONOTONICALLY (sanity) and attributes/concepts add value. The count-threshold (>=2 liked items)
gave a near-constant "+1" for popular categories -> uninformative -> NDCG DECLINED with more questions.

Lit (deep-research): CRS sims use TARGET-ITEM oracle (wrong for cold-start profile). Profile-based answering is open
but building blocks: genre lift/PMI (Church&Hanks'90; Steck'18) + diff-in-means debiased (Koren'09); concept =
relevance-weighted avg rating DEBIASED by user mean (Sen et al. WWW'09 'movie-ratings', the best inferrer;
Nguyen&Riedl UMAP'13). MUST debias by user mean (relevance is applicability not valence). Answer model is a research
variable (Zhang&Balog'20; Bernard&Balog'25). Protocol: ML-1M cold-test, held-out half, profile excluded, W=2, T=10.

## Answer models tested (change one thing at a time)
GENRE:
- M1 COUNT (old): y=+1 if >=2 liked items in g else -1.
- M2 DIFF-IN-MEANS (graded, Koren bias): y = mean(rating_u over g items) - mean(rating_u all), scaled.
- M3 LIFT (binary, base-rate norm): lift=P(g|user rated)/P(g|catalog); y=+1 if lift>1.1, -1 if <0.9, skip else.
- M4 LIFT (graded): y = tanh(log lift).
CONCEPT (Tag Genome):
- C1 COUNT (old): y=+1 if >=2 liked items with rel>0.5 else -1.
- C2 RELWEIGHTED-DEBIASED (Sen'09, graded): y = sum_m rel(m,c)*(r_um - mean_u) / sum_m rel(m,c).
- C3 RELWEIGHTED sign (binary).

## Results (scripts/paper1/attr_answer.py; ML-1M cold-test, 120 users, W=2, T=10, NDCG@10 q0..q10)
GENRE:
- M1 count (pop):  0.317->0.256  DECLINES (broken: popular genres get constant +1 -> drift)
- M2 diff-in-means (pop): 0.320->0.196 DECLINES WORSE
- M3 lift-binary (pop): 0.320->0.365(q2)->0.346  grows early then wobbles
- M4 lift-GRADED (pop): 0.318->0.374(q3)->0.336  grows then declines
- **M4 lift-GRADED (AFFINITY sel): 0.328->0.376(q4)->0.365  GROWS +0.048 and HOLDS (~monotone) -- FIX**
CONCEPT (Tag Genome):
- C1 count: declines (~0.32->0.19). C2 relweighted-debiased (Sen, pop/aff): still DECLINES (~0.31->0.21).

## CONCLUSIONS
1. ANSWER MODEL matters hugely: graded base-rate-normalized LIFT/PMI (M4) >> count (M1) >> diff-in-means (M2).
   The count-threshold was the bug (constant +1 for popular categories). NO training / complex mapping needed.
2. SELECTION matters: AFFINITY order (ask most-opinionated genres first) + graded-lift => grows 0.328->0.376 and
   PLATEAUS (essentially monotone). Beats the item baselines (popularity-item ~0.362). GENRES FIXED by a simple calc.
3. CONCEPTS NOT fixed by answer model alone: relweighted-debiased (lit-best) still declines. Likely causes:
   concept CENTROID factor too coarse/overlapping; over-weighting many concept reveals drifts. NEXT to try (one at
   a time): (a) fewer concept reveals / stronger floor; (b) better concept factor (relevance-weighted centroid, or
   SBERT-adapter factor, or maxvol-representative concept); (c) lower concept fold-in weight vs items; (d) graded y
   scale. Recommendation to adopt going forward: GENRE answer = graded lift/PMI; CONCEPT answer = relweighted-debiased
   (Sen'09) but treat concept fold-in carefully. This is a research variable (Zhang&Balog'20; Bernard&Balog'25).

## UPDATE 2 (leakage-free + 3-valued answers; scripts/paper1/concept_answer.py)
User fixes: (1) LEAKAGE -> compute answers PROFILE-ONLY (rated minus held-out test). (2) THREE-VALUED answer ->
abstain "don't know" (no fold-in) vs "dislike" (-1). Faithfully reproduced: C2 Sen'09 relweighted-debiased,
C5 Bayesian shrinkage (movie-bayes), C6 per-user ridge regression beta_u over tag relevance (Nguyen&Riedl UMAP'13),
+ C4 confidence-gated abstain. RESULTS (leakage-free, affinity-sel, W=2 T=10): genre 0.317->0.351(q4)->0.309;
C2->0.185; C4->0.212; C5->0.213; C6(best concept)->0.246. ALL DECLINE after q1-4.
CORRECTION: earlier genre "0.376 holds" was partly LEAKAGE; leakage-free genre also declines after q4.
DIAGNOSIS (key): decline is RECOMMENDER-SIDE not answer-model. Folding many COARSE attribute/concept CENTROIDS
(averages not points) over-fits coarse directions -> drifts from specific held-out items; items (precise points)
don't drift. Attribute/concept value is FRONT-LOADED (coarse localization ~first 3q); they cannot refine.
Lit answer models tried FAITHFULLY (Sen, Nguyen&Riedl, Bayes) -> fix early growth only, do NOT stop drift -> report
this honestly in paper. PRINCIPLED FIXES (not hacks): (a) PRECISION-WEIGHTED fold-in (attribute=higher-noise row,
down-weight vs item; heteroscedastic Bayesian fold-in); (b) COARSE-TO-FINE mixed policy (attributes localize, items
refine)=the unified-action-space+learned-policy contribution.

## UPDATE 3 (precision-weighted fold-in; concept_answer.py attr-weight sweep)
Tried down-weighting coarse attribute/concept rows in ridge fold-in u=(XtCX+lI)^-1 XtCy, w in {1,0.5,0.2,0.1}.
Marginal: genre ends 0.329(w=0.2) vs 0.309(w=1); concept C6 ends 0.257 vs 0.246. Reduces drift slightly but does
NOT stop decline. Precision-weighting DISCONFIRMED as a full fix.
FIRM CONCLUSION (4 faithful attempts: answer-model lift, leakage/3-valued abstain, per-user regression Nguyen&Riedl,
precision-weighting): attribute/concept value is FRONT-LOADED (peaks ~q3-5, genre 0.317->0.351, then plateaus/declines)
because coarse CENTROIDS localize but cannot refine; only precise ITEM answers refine. No single-knob fix makes a PURE
attribute/concept interview monotone. REAL FIX = MIXED COARSE-TO-FINE policy (few concept/attr to localize, then items
to refine) over the unified action space = the S5 learned-policy contribution. Paper: report all four attempts faithfully
as tried-and-insufficient; constructive answer = mixed coarse-to-fine.

## UPDATE 4 — RESOLVED: the decline was the GROWING WEIGHT on coarse evidence (not answer model, not foldin)
Controlled test (concept_answer.py: foldin vs additive vs additive-with-BOUNDED-weight): additive (no Gram inverse)
ALSO declined -> inverse-amplification hypothesis WRONG. Re-diagnosed: score=W*z(pop)+c(m)*z(coarse-dir), c(m)=m/(m+5)
GROWS -> coarse genre/concept direction increasingly overrides popularity -> demotes blockbuster held-out likes ->
decline (the popularity-floor vs coarse-residual law again). FIX = BOUND the coarse-signal weight (constant, ~0.4),
NOT growing. 120-user result: G additive w=0.4: 0.320->0.393 (monotone after q1 dip, beats item baselines ~0.36);
G w=0.2: 0.320->0.357; C6 w=0.2: 0.328->0.331 (HOLDS, no decline). High weight (0.8) declines. PRINCIPLE: coarse
attribute = low-precision evidence -> bounded total weight; more answers refine DIRECTION not TRUST magnitude.
FINAL RECIPE: graded base-rate-normalized answer (lift/PMI debiased; concept=Sen relweighted-debiased) + EAR-style
ADDITIVE channel (separate from item fold-in) + BOUNDED coarse-signal weight. Genres grow monotonically & beat items;
concepts hold (NDCG gain small=blockbuster-saturation, value on Recall/per-user). Adopt this in instrument + baselines.
(Minor residual: q1 dip from first single coarse answer; fixable by ramping weight. Not critical.)

## UPDATE 5 — OUT-OF-SAMPLE (non-genome, SBERT) concept ELICITATION (oos_concept.py)
Tested arbitrary multi-word concept phrases (NOT genome tags: "mind-bending plot twist","slow-burn character study"
...), grounded by SBERT: concept factor = SBERT-sim-weighted Q-centroid; answer = SBERT-sim-weighted debiased rating
(profile-only, gated); corrected recipe (additive + bounded weight). Controls: affinity vs random selection; genres.
RESULT (120 users, NDCG@10 q0->q10): GENRES aff 0.317->0.388 (win), genres rand 0.314->0.333; OOS-CONCEPTS aff
0.322->0.305 (FLAT/slightly down), OOS rand 0.326->0.305. Rec@10: genres 0.089, OOS 0.078.
## UPDATE 6 — HARNESS BUG: per-policy random split made the panel incoherent (NOT a model issue)
User flagged: "popularity now doesn't monotonically increase, ranking makes no sense." ROOT CAUSE = each policy run
RE-DREW its own random held-out split (global rng advanced between runs) -> every method scored on DIFFERENT data.
PROOF: q0 (pure popularity, zero questions = should be byte-identical) ranged 0.301-0.328 across policies; that
~0.03 split-variance > the method gaps -> ranking was noise; popularity looked non-monotone purely from an unlucky
split. FIX: precompute ONE fixed per-user split (SPL, dedicated rng seed=123) shared across ALL policies. After fix
q0=0.326 identical everywhere; panel coherent (s4_panel.py, s4d.log): GENRES 0.326->0.383 Rec0.094 BEST; POPULARITY
0.326->0.345 MONOTONE; RMVA 0.329; RANDOM 0.325 (no-op floor: random items not in profile -> no fold-in -> sits at
pop prior); EIG 0.323 (bad proxy); ENTROPY 0.319; HELF 0.318; OOS 0.311; GENOME 0.288 (below baseline). Lesson:
always fix the eval split across compared arms; per-arm resampling confounds. Paper tab:s4panel updated (both tex).

## UPDATE 7 — METRIC/PROTOCOL REALIGNMENT (q0 must = MOSTPOP 0.41) + FINAL ON-RULER PANEL
User: "confirm the metric aligns with lit." Formula = standard (identical to mf_foldin ndcg_recall). But the s4 panel
PROTOCOL had drifted: it targeted a random HALF of likes and excluded the WHOLE profile -> MOSTPOP 0.32, OFF the
project's 0.41 anchor. metric_check.py proves it's 100% protocol (same data/pop vector): CANONICAL (rel=all likes-seeds,
exclude only seeds) = 0.4134 == anchor; S4-protocol = 0.3187. TWO valid protocols: P1 cold-elicitation (items, no
leakage, anchor 0.41) vs P2 held-out (attributes MUST use it -- a profile-derived attribute answer would re-score its
own targets = leakage; anchor 0.32). FIX: items -> P1 (q0=MOSTPOP=0.411 exactly); attrs stay P2 (q0=0.309); blend
(W,conf-cap) tuned on VAL per the ruler. s4_panel.py rewritten.
FINAL ON-RULER PANEL (s4f.log, 200 users): ITEMS P1 q8 NDCG/Rec: EIG 0.438/0.075(+0.027 BEST), HELF 0.422/0.070(+0.011),
RMVA 0.418/0.071(+0.007), ENTROPY 0.410(~flat no-op), RANDOM 0.401(no-op), POPULARITY 0.361/0.060(-0.050 BELOW).
ATTRS P2 q8: GENRES 0.350/0.092(+0.040 BEATS), GENOME 0.276(-0.033), OOS 0.283(-0.027).
KEY: ordering EIG>HELF>RMVA>entropy~random>popularity is LITERATURE-CONSISTENT. Popularity-item elicitation DECLINES
below baseline -- NOT a bug: a popular-item rating is REDUNDANT with the popularity prior (everyone likes blockbusters)
-> injects noise not discriminative signal. Corroborated by R1 (POPULAR seeds 0.355 < MOSTPOP 0.413). Discriminative
signal lives in REPRESENTATIVE (RMVA) + INFO-GAIN (EIG) items = founding insight of cold-start interviewing (Rashid,
Golbandi, RBMF: don't ask popular, ask representative). q0 = the popularity baseline itself; "below q0" = worse than
popularity = the strong-floor problem. Paper tab:s4panel + RESULTS S4-A4 updated. LESSON: always re-check q0==anchor.

## UPDATE 8 — THE REAL BUG: s4 panel skipped "no" answers (fold-skip). Revealing now HELPS monotonically.
User pushed: "revealing ratings MUST help or stay flat; your strong baselines do nothing; we lost something vs the
classic MF result." CORRECT. ROOT CAUSE: run_item folded in ONLY items the user RATED (`if e in rd: rows.append`)
and SILENTLY SKIPPED the rest -- instead of folding the ANSWER as y=0 ("don't like / haven't seen"). The validated
instrument (instrument_u.py L97: yv=[rd.get(it,0.0) for it in seeds] -- ALL seeds, 0 for not-liked) and R1
(mf_foldin L59: prefs=B[ui][seeds]) BOTH fold every asked seed with 0/1. The dropped "no" answers ARE the
discriminative signal of representative-item elicitation -> without them RMVA collapsed to +1.7%. FIX: fold EVERY
asked item, y=1 if liked else 0; conf=n_liked/(n_liked+5) (instrument recipe; dropped the ad-hoc cap).
RESULT (s4h.log, T=30, 200u): HELF 0.411->0.454 (+0.043, MONOTONE), RMVA 0.411->0.433 (+0.022, MONOTONE), EIG
0.438@q8; ENTROPY flat (no-op, high-entropy items unrated->y=0 no signal); POPULARITY 0.411->0.294, RANDOM ->0.382
(DECLINE). Good selection now improves monotonically = user's intuition confirmed. Popularity/random decline for two
reasons: (1) REDUNDANCY (popular answers near-identical across users -> no discrimination); (2) EXCLUSION (P1 excludes
asked items from re-recommendation -> asking top-popular exhausts the easy targets, leaves the hard tail). Matches R1
(POPULAR seeds 0.355<MOSTPOP 0.413) + literature. Magnitude vs R1 0.50: generic-ridge+saved-Q here reaches 0.45-0.43;
full instrument (K=50+learned encoder) =0.486. Same shape. Confirms instrument_u K=50 (validated num is at 50 reveals,
not 8 -> earlier "small gain" was budget). LESSON: in elicitation, ALWAYS fold the ANSWER (incl negatives), never skip.

KEY DIAGNOSTIC: for genres affinity>>random (answer carries signal); for OOS concepts affinity~=random (answer
carries ~NO usable signal). => realizable OOS-concept elicitation does NOT pay off on NDCG. Cause: arbitrary phrases
match movies diffusely via SBERT -> mushy factor + noisy answer; and niche gain misaligned with blockbuster-saturated
NDCG. Honest NEGATIVE for paper. VALUE GRADIENT (realizable NDCG): genres(win) > genome-concepts(flat,no decline) >
OOS-concepts(flat/none). Concept value is oracle-only (38% users); realizing it needs better grounding (plot synopses
fixed RETRIEVAL not elicitation), informative (not affinity) selection, Recall lens, i.e. the learned policy (S5).
