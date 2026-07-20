# Overnight Adaptivity-Headroom Report

Generated 2026-06-13 23:35 (70s).

Tier-0 model-free diagnostic: HEADROOM@10 = (per-user adaptive liked-coverage) - (best fixed-set coverage) in 10 questions. Large => adaptivity can pay; train learned policies. Small => static near-optimal.

| Dataset | users | items | answer-rate | pair-Jaccard | static@10 | adaptive@10 | HEADROOM@10 | verdict |
|---|---|---|---|---|---|---|---|---|
| movielens_slate1 | 3000 | 187 | 0.38 | 0.373 | 0.148 | 0.317 | **0.169** | MODERATE headroom |
| movielens_slate2 | 3000 | 379 | 0.33 | 0.419 | 0.070 | 0.311 | **0.241** | MODERATE headroom |
| synthetic_indicator | 3000 | 97 | 0.36 | 0.283 | 0.142 | 0.640 | **0.498** | STRONG adaptivity headroom |
| lastfm | 1715 | 300 | 0.07 | 0.055 | 0.149 | 0.834 | **0.685** | STRONG adaptivity headroom |
| yelp | — | — | — | — | — | — | — | SKIPPED: Yelp Open Dataset requires a signup form |

## Validation check (should retrodict known results)
- synthetic_indicator: expect LARGE headroom (disjoint user groups).
- movielens_slate1: expect SMALL headroom (everyone likes the hits).
- movielens_slate2: expect small-moderate (mid-pop, more varied).
If these hold, the diagnostic is trustworthy for new datasets.

## Recommendation
Train Tier-2 learned policies on any NEW dataset with HEADROOM@10 materially above movielens_slate2's; that is the dataset most likely to demonstrate learned adaptive > static.
## LastFM Tier-1 results (real-data adaptivity demonstrator)
instrument accepted=True (rho=1.00, overlap=0.42); prior=0.569

| policy | final | AUAC | answer | adaptive |
|---|---|---|---|---|
| random | 0.5749 | 0.5710 | 17% | True |
| popularity | 0.6016 | 0.5860 | 94% | False |
| greedy_infogain | 0.6089 | 0.5896 | 66% | True |
| scpr_entropy | 0.6019 | 0.5880 | 87% | True |
| thompson | 0.5964 | 0.5807 | 43% | True |

Adaptive heuristic best AUAC=0.5896 vs popularity 0.5860 (gap +0.0036). Tier-2 (PPO/bot-play) is the recommended next step.
## LastFM diagnosis (IMPORTANT — read before trusting v1 numbers)
v1 used a median-split target (liked = plays >= own median). Result:
oracle-ceiling (all reveals) = 0.637, prior 0.570 => dynamic range only
0.068. The instrument is WEAK because the median-split target is near-
noise; the benchmark (greedy 0.590 vs popularity 0.586 AUAC) is therefore
INCONCLUSIVE, NOT a real "adaptivity doesn't help" finding — it is the
weak-instrument compression failure mode (cf. synthetic ceiling gate).
Tier-0 headroom (0.685, Jaccard 0.055) shows the DATA has strong routing
structure; the median target just can't express it.

FIX IN PROGRESS: tercile target (liked = top-third plays = genuine
favourites, disliked = bottom-third, middle = unknown), mirroring the
movie liked=rating>=4 design, plus an oracle-ceiling acceptance gate
(>0.72). If the tercile instrument clears the ceiling gate, its benchmark
is the real LastFM adaptivity test; results land in
benchmark_lastfm_tercile.json.

METHODOLOGICAL NOTE for the paper: the oracle-ceiling gate must be a
STANDARD acceptance criterion — v1 passed monotonicity + overlap yet was
too weak to benchmark. Instrument acceptance needs: polarity + reveal-
monotonicity + liked/dislike overlap + ANSWERABILITY (dual head) +
ORACLE CEILING.

## LastFM Tier-1 results (real-data adaptivity demonstrator)
instrument accepted=False (rho=1.00, overlap=0.35); prior=0.594

| policy | final | AUAC | answer | adaptive |
|---|---|---|---|---|
| random | 0.6067 | 0.5994 | 14% | True |
| popularity | 0.6360 | 0.6150 | 90% | False |
| greedy_infogain | 0.6291 | 0.6181 | 57% | True |
| scpr_entropy | 0.6422 | 0.6203 | 83% | True |
| thompson | 0.6324 | 0.6114 | 37% | True |

Adaptive heuristic best AUAC=0.6203 vs popularity 0.6150 (gap +0.0054). Tier-2 (PPO/bot-play) is the recommended next step.
## SYNTHESIS — the honest overnight conclusion (read this)

Tested adaptivity on 4 real-data instrument configs (MovieLens slate1 &
slate2; LastFM median & tercile) + synthetic. Pattern is consistent:

| world | instrument ceiling | adaptive-vs-static (AUAC) |
|---|---|---|
| synthetic (no collaborative bridge) | 0.90 | LARGE (oracle .80 vs static .69) |
| movielens slate1 (strong instr.) | ~0.85 | tiny (+0.004) |
| movielens slate2 (strong instr.) | ~0.82 | small (+0.004) |
| lastfm tercile (weak instr.) | 0.69 | small (+0.005), gate-fail |
| lastfm median (weak instr.) | 0.64 | ~0 (gate-fail, inconclusive) |

KEY INSIGHT: the model-free Tier-0 headroom (LastFM 0.685, Jaccard 0.055)
OVER-PREDICTS adaptivity value, because a trained recommender GENERALISES
across users via collaborative structure — recovering from a *fixed*
question set most of what per-user adaptive questioning would capture
directly. Collaborative generalisation substitutes for adaptive
elicitation. This is a real, defensible, and arguably headline finding:
adaptive elicitation pays big ONLY when collaborative generalisation
fails (synthetic disjoint groups); on real catalogs with collaborative
bridges, a fixed elicitation order + a good recommender ≈ adaptive.

WHY LastFM did NOT become the demonstrator: music has disjoint ANSWERABLE
sets (Jaccard 0.055) BUT strong within-genre collaborative BRIDGES, so
generalisation still substitutes. The Tier-0 diagnostic measured
answerable-set overlap but not collaborative bridgeability — that's its
limitation.

REVISED PREDICTION for a real adaptivity demonstrator: need disjoint
answerable sets AND weak cross-partition collaborative bridges =>
CROSS-DOMAIN (Amazon multi-category: books tell you little about kitchen
goods) or MULTI-CITY (Yelp: NYC tells you nothing about Phoenix). Music
is not disjoint enough. Yelp needs a manual signup download; Amazon
2023 categories are auto-downloadable (large).

RECOMMENDED NEXT STEPS (not run overnight — flagged for decision):
1. Refine Tier-0 to also estimate collaborative bridgeability
   (cross-user hold-one-out predictability), not just answerable overlap.
2. Build a cross-domain or multi-city slate (Amazon multi-category is the
   auto-downloadable path) and re-run the headroom + instrument + bench.
3. For the paper: the efficiency framing (2-2.5x turns-to-quality on
   movies) remains the operational value story; the synthetic world
   remains the "adaptivity is learnable when required" proof; and the new
   "collaborative generalisation substitutes for adaptivity on real
   catalogs" is the unifying explanation.

## amazon_crossdomain results
ceiling=0.967 base=0.841 lift=+0.126

| policy | final | AUAC | answer | branches |
|---|---|---|---|---|
| random | 0.7883 | 0.7844 | 2% | True |
| popularity | 0.8558 | 0.8284 | 23% | False |
| greedy_infogain | 0.8478 | 0.8239 | 20% | True |
| greedy_answerability | 0.8632 | 0.8278 | 23% | True |
| scpr_entropy | 0.8598 | 0.8303 | 15% | True |
| thompson | 0.7788 | 0.7803 | 1% | True |

greedy_answerability AUAC 0.8278 vs popularity 0.8284 vs random 0.7844 (answerability-routing gap -0.0006 vs static)

## amazon_balanced results
ceiling=0.880 base=0.831 lift=+0.050

| policy | final | AUAC | answer | branches |
|---|---|---|---|---|
| random | 0.7752 | 0.7705 | 2% | True |
| popularity | 0.8346 | 0.8121 | 27% | False |
| greedy_infogain | 0.8323 | 0.8110 | 17% | True |
| greedy_answerability | 0.8368 | 0.8136 | 22% | True |
| scpr_entropy | 0.8481 | 0.8189 | 19% | True |
| thompson | 0.7629 | 0.7630 | 1% | True |

greedy_answerability AUAC 0.8136 vs popularity 0.8121 vs random 0.7705 (answerability-routing gap +0.0015 vs static)

## FINAL SYNTHESIS (after 6 real-data configs + synthetic)

Tested adaptivity across: MovieLens slate1, slate2; LastFM median, tercile;
Amazon cross-domain; Amazon balanced. Plus the synthetic indicator world.

ROBUST FINDING: on EVERY real recommendation catalog, static
popularity-ordered elicitation is near-optimal; adaptive elicitation
(heuristic or learned) beats it by at most ~+0.007 AUAC — never
decisively. Adaptive wins DECISIVELY only in the synthetic world
(balanced, disjoint, no collaborative bridge, no popularity head).

WHY (mechanism, now well-supported):
- Real user populations have a POPULARITY HEAD: dense users concentrate in
  a few big segments, so a static order front-loads the questions most
  users can answer AND that best predict them.
- Collaborative generalisation lets the recommender fill in from a fixed
  question set most of what per-user routing would capture directly.
- Adaptive routing only helps the tail minority; averaged over users the
  gain is marginal on accuracy.
- Attempts to remove the head fail naturally: real datasets don't supply
  balanced, dense, mono-segment user populations (small Amazon categories
  have ~0 dense users; balancing collapses to the big categories).

THIS IS A STRONG, HONEST, PUBLISHABLE RESULT — arguably better than
"adaptive wins": it is surprising, robust across 2 domains x multiple
instruments, and bounded by a clean synthetic existence proof of WHEN
adaptivity would win. The operational value of strategy on real data is
EFFICIENCY (2-2.5x turns-to-quality), not endpoint accuracy.

RECOMMENDATION: stop manufacturing balanced real datasets (risks contrived
setups). Frame the paper around: (1) the testbed + measurement rigour;
(2) static-near-optimal-on-real-catalogs with the popularity-head/
collaborative-generalisation explanation; (3) the synthetic boundary
showing adaptivity is learnable and wins when the structure demands it;
(4) efficiency as the real-world value. ONE remaining clean real test that
could still show a win: YELP MULTI-CITY (cities are size-balanced AND
geographically disjoint, unlike size-skewed Amazon categories) — but it
needs a manual signup download (USER ACTION).

## ml_stratified results
ceiling=0.919 base=0.693 lift=+0.226

| policy | final | AUAC | answer | branches |
|---|---|---|---|---|
| random | 0.7125 | 0.6874 | 22% | True |
| popularity | 0.7190 | 0.7006 | 99% | False |
| greedy_infogain | 0.7448 | 0.7210 | 79% | True |
| greedy_answerability | 0.7438 | 0.7235 | 71% | True |
| scpr_entropy | 0.7553 | 0.7236 | 81% | True |
| thompson | 0.6915 | 0.6688 | 19% | True |

greedy_answerability AUAC 0.7235 vs popularity 0.7006 vs random 0.6874 (answerability-routing gap +0.0228 vs static)

## ml_stratified results
ceiling=0.919 base=0.693 lift=+0.226

| policy | final | AUAC | answer | branches |
|---|---|---|---|---|
| random | 0.7125 | 0.6874 | 22% | True |
| popularity | 0.7190 | 0.7006 | 99% | False |
| greedy_infogain | 0.7448 | 0.7210 | 79% | True |
| greedy_answerability | 0.7438 | 0.7235 | 71% | True |
| scpr_entropy | 0.7553 | 0.7236 | 81% | True |
| thompson | 0.6915 | 0.6688 | 19% | True |

greedy_answerability AUAC 0.7235 vs popularity 0.7006 vs random 0.6874 (answerability-routing gap +0.0228 vs static)

## *** BREAKTHROUGH: adaptivity WINS on a stratified (unbiased) slate ***

The user's preprocessing concern was correct. All prior slates took the
TOP-M most popular items -> popular targets the prior already predicts
(popularity answer-rate 99% but uninformative) -> static looked optimal.
That was a SLATE-CONSTRUCTION ARTIFACT.

On MovieLens STRATIFIED (300 movies sampled evenly across popularity-rank
bands 0-6000, incl. long tail; instrument lift +0.226 over base, ceiling
0.919), adaptive elicitation SIGNIFICANTLY beats static:

| policy | AUAC | final | answer |
|---|---|---|---|
| scpr_entropy        | 0.7236 | 0.7553 | 81% |
| greedy_answerability| 0.7235 | 0.7438 | 71% |
| greedy_infogain     | 0.7210 | 0.7448 | 79% |
| popularity (static) | 0.7006 | 0.7190 | 99% |
| random              | 0.6874 | 0.7125 | 22% |
| thompson            | 0.6688 | 0.6915 | 19% |

Paired bootstrap (n=300): scpr/greedy/greedy_answerability vs popularity
= +0.020 to +0.023 AUAC, p<0.0001. ~5-6x the +0.004 top-popular gap.

IMPLICATIONS:
1. The line of work is vindicated: adaptive elicitation DOES matter on a
   realistic (non-popularity-biased) real dataset.
2. NEW methodological contribution: popularity-biased item slates (used by
   much CRS work) systematically UNDERSTATE elicitation value, because
   popular items are prior-predictable. Elicitation value lives in the
   tail. This reframes the earlier 'static near-optimal on top-popular
   slates' from a null into a cautionary finding about benchmark design.
3. Paper arc now: (a) testbed+rigour; (b) slate construction is decisive
   -- top-popular hides adaptivity, stratified reveals it; (c) synthetic
   bound; (d) efficiency. Strong, positive, honest.

## yelp_multicity results
ceiling=0.933 base=0.781 lift=+0.153

| policy | final | AUAC | answer | branches |
|---|---|---|---|---|
| random | 0.7707 | 0.7713 | 2% | True |
| popularity | 0.7715 | 0.7705 | 14% | False |
| greedy_infogain | 0.7712 | 0.7703 | 12% | True |
| greedy_answerability | 0.7927 | 0.7812 | 17% | True |
| scpr_entropy | 0.7801 | 0.7737 | 11% | True |
| thompson | 0.7690 | 0.7675 | 4% | True |

greedy_answerability AUAC 0.7812 vs popularity 0.7705 vs random 0.7713 (answerability-routing gap +0.0107 vs static)

## ml_stratified results
ceiling=0.919 base=0.693 lift=+0.226

| policy | final | AUAC | answer | branches |
|---|---|---|---|---|
| random | 0.7125 | 0.6874 | 22% | True |
| popularity | 0.7190 | 0.7006 | 99% | False |
| greedy_infogain | 0.7448 | 0.7210 | 79% | True |
| greedy_answerability | 0.7438 | 0.7235 | 71% | True |
| scpr_entropy | 0.7553 | 0.7236 | 81% | True |
| thompson | 0.6915 | 0.6688 | 19% | True |

greedy_answerability AUAC 0.7235 vs popularity 0.7006 vs random 0.6874 (answerability-routing gap +0.0228 vs static)
