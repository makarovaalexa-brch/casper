# DESIGN SHEET — D-ANS: the Distilled Answerer (v1)
Status: DRAFT FOR AUTHOR SIGN-OFF. Nothing executes until marked. (Rule: design-sheets-before-execution.)

## 1. The question this answers, and what changes per outcome
Build a calibrated generative answerer that converts ANY ML-25M user's known-half ratings into
interview answers (knowledge level + value, per the locked schema) statistically faithful to the
LLM answerer — so policies can finally train on thousands of users instead of 173.
- Gates pass → the population-scale training world exists; fold-v3 (Item 2) proceeds on top.
- Gates fail in specific strata → arena restricted to passing strata, or ONE documented feature
  iteration; decide then.
- Gates fail globally → stop; the wide-sparse LLM judging option (Option D, ~$20-40) becomes the
  necessary path and gets its own sheet.

## 2. Data and splits (exact)
- FIT TARGETS: the LLM-judged cells of the 173 usable study users (~420k cells: 1,128 concepts +
  500 attributes + 800 items each, knowledge+value) + the masked-value star predictions.
- STRUCTURE SOURCES (population, no LLM): ML-25M ratings (162k users); tag genome (1,128 tags ×
  13.8k movies, graded relevance); IMDb join (directors/actors/composers/writers/franchises);
  co-knowledge embeddings from the population rated matrix; item/entity/tag popularity.
- SPLITS: calibration by leave-one-user-out over the 173 (never a user's own cells in their fit;
  final model refit on all 173). ALL features computed from known halves only — for real users and
  synthetic users alike. Held-out halves: never touched by anything, ever (the eval firewall).

## 3. Model form (hand-designed rule families carrying world-knowledge proxies; ~30–60 fitted
##    parameters total; every rule inspectable)
- INTERNAL CONSUMPTION VARIABLE per (user, item): rated ⇒ consumed. Unrated: logistic on
  [co-knowledge proximity of the item to the user's rated set, genre/tag affinity alignment,
  item fame, decade alignment]. (The GoT insight lives here: consumption is first-class.)
- ITEM knowledge (ordered logistic → {no_clue, rough_idea, know_well}): driven by consumption +
  fame; a small calibrated FAME-ONLY term allows reputation-confidence on mega-famous items (the
  "Titanic impurity") — its size is measured, not assumed (see G-pre).
- CONCEPT knowledge: relevance-weighted rated-member count with a LEARNED SATURATION exponent +
  tag fame + taste-alignment.
- ENTITY knowledge (director/actor/composer/writer/franchise): ENGAGEMENT (count and fraction of
  filmography rated; learned saturation) + entity fame + alignment.
- VALUES: rated items → the real rating mapped to the scale (no model, no noise). Everything else
  → ordered logistic over {hated, meh, liked, loved} on [predicted affinity from population
  embeddings, data-side member aggregates where present, engagement bonus for entities (the
  watched-everything-fan rule: engagement lifts value independently of valence), fame prior].
- DISLIKED MEMBERS COUNT toward knowledge/engagement everywhere (locked).
- GENERATION = SAMPLING ONLY: knowledge sampled from its fitted categorical; value sampled from
  its fitted conditional. No point estimates, no thresholds, NO injected noise — the fitted
  distributions carry the uncertainty (author decision, locked).

## 4. Gates (pre-registered; printed before results)
- G-pre (free, before fitting): measure the reputation-confidence prevalence on the EXISTING grid
  (know_well on unrated famous items with no nearby engagement) — bounds the Titanic impurity and
  calibrates the fame-only term.
- G1 AGREEMENT (leave-user-out): per channel × popularity stratum — knowledge-level accuracy and
  kappa vs the LLM's actual labels, compared against a population-rate-only baseline; the NICHE
  strata rows are decisive (that is where the fuel lives and where data is thinnest). Values:
  MAE vs the LLM's values; on rated cells, MAE vs REAL ratings benchmarked against the LLM's own
  0.70-star mark.
- G2 FUEL REPRODUCTION (the decisive gate): generate synthetic answers for the same 173 users from
  their known halves; recompute the Stage-A fuel statistics — ICC beyond popularity per channel,
  within-tier validity gap, know-well strata rates, habitat sizes — synthetic must match real
  within CIs. If the synthetic world loses the correlations (author's warning: sampling without
  signal = noise), this gate catches it and the build FAILS here.
- G3 ERROR PROFILE: sampled answers' error on masked rated cells matches the LLM's error in
  magnitude AND dispersion (the too-clean/too-smooth check; calibrate only if measurably too clean).
- G4 SCALE SANITY: generate a 10k-user synthetic population; base rates monotone in popularity,
  no degenerate columns, heterogeneity present at the measured levels.

## 5. Shortcuts, flagged, with alternatives (DECIDE each)
- S1 conf field: omit from the distilled answerer (nothing downstream uses it) — OR fit it (cheap).
  DEFAULT: omit, documented.
- S2 reputation-vs-seen at the top level: handled via the calibrated fame term + G-pre measurement —
  OR re-judge a small sample with an explicit seen/heard question (~$1-2). DEFAULT: measure free
  first; escalate only if G-pre shows large impurity.
- S3 feature set closed as listed (no LLM-text embeddings — keeps the judge's text prior out of the
  distillation; firewall). DEFAULT: keep closed.
- S4 synthetic population size: 10-20k users first (compute), extensible to all 162k later.
  DEFAULT: 20k.
- S5 all calibration rests on 173 users — inherent to the budget; mitigated by the low parameter
  count (~30-60) + LOUO validation; the human study later re-validates externally. Acknowledged.

## 6. What this build explicitly does NOT do
No policies. No fold training (Item 2 gets its own sheet). No LLM calls. No touching held-out
halves. No fitting on evaluation users' own cells. The 173 remain the untouchable eval world.

## 7. Cost and execution
$0 LLM. Local CPU: fitting = minutes; 20k-user generation = hours. Implementation by Opus agents
ONLY after this sheet is signed; Fable reviews implementation and all gate outputs before anything
(fold-v3, policies) consumes the result.

## SIGN-OFF — SIGNED BY AUTHOR 2026-07-09
[x] Model form (§3) — signed, with author amendment: know_well is NOT relabeled as "watched"; the
    consumption variable stays INTERNAL to generation; know_well labels match the LLM's behavior
    as-is and are used as the strong signal downstream ("leave it as llm decided").
[x] Gates (§4) — signed; G2 decisive; G-pre impurity measurement = informational only, no
    correction applied in v1.
[x] S1 conf: OMIT (author: "didn't know about conf, ok, drop").
[x] S2 impurity: free measurement only; no re-judging spend.
[x] S4 synthetic N: 20k for gates, THEN — AUTHOR CONDITION — extend to ALL 162k users BEFORE any
    next step (no fold-v3, no policies, until everyone has an answerer).
[x] Split simplification ratified: train = synthetic users (173 real IDs excluded); test = the 173
    real users (touched sparingly, headline-only); known/held-out halves = the task definition,
    not a split; all tuning on synthetic validation users.
