# THE CLEAN REFERENCE LADDER (fold-v2; 173 users; NO LLM)

> **DIRECTIONAL 173/300** -- answerer-v1 WORKING grid (`.cache/instrument2/answerer_v1_grid173_WORKING.json`), grid NOT frozen. Re-run on the frozen 300-user grid before citation. Instrument = fold-v2 `.cache/i25_fold_v2_best.pt` (val 0.4256). Script `scripts/reference_ladder.py`; NO LLM API calls.

173 users; 180 candidates. Greedy statics built to maximize cohort-mean NDCG@10 (canonical objective), reported at BOTH @10 and @50. Paired per-user bootstrap BOOT=5000, seed 0. Everything below is on the SAME instrument and the SAME 173 users.

## The ladder (NDCG@10 / NDCG@50)

| # | rung | @10 | @50 | notes |
|---|---|--:|--:|---|
| 1 | COLD (z=0, no answers) | 0.1502 | 0.1575 | floor |
| 2 | learned ITEM static -- anytime@12 (in-sample all-173) | 0.2251 | 0.2082 | anchor@10 0.2251 vs 0.2251 (MATCH); carries test-fit bonus (see 2b) |
| 2 | learned ITEM static -- endpoint t=24 (in-sample) | 0.2256 | 0.2167 | endpoint after 24 turns |
| 2b | learned ITEM static -- anytime@12 (FAIR cross-fit, out-of-fold) | 0.2046 | 0.2033 | de-contaminated; the honest static number |
| 2c | concept static -- anytime@12 (in-sample, contrast) | 0.2240 | 0.2148 | for contrast only |
| 3 | FULL-PROFILE via fold-v2 (all rated known items) | 0.3589 | 0.3780 | noise-robust fold, full clean profile (out-of-regime) |
| 4 | FULL-PROFILE via NATIVE RecVAE (liked items) | 0.4938 | 0.4883 | **instrument ceiling** (its own interface) |
| 5 | static t=24 as % of (3) / (4) | 62.9% / 45.7% | 57.3% / 44.4% | orientation |

## Learned ITEM static -- full per-turn ENDPOINT curve (in-sample all-173)

| metric | t1 | t2 | t3 | t4 | t5 | t6 | t7 | t8 | t9 | t10 | t11 | t12 | t13 | t14 | t15 | t16 | t17 | t18 | t19 | t20 | t21 | t22 | t23 | t24 |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| @10 | 0.2158 | 0.2153 | 0.2194 | 0.2216 | 0.2254 | 0.2274 | 0.2281 | 0.2286 | 0.2284 | 0.2295 | 0.2302 | 0.2309 | 0.2305 | 0.2305 | 0.2305 | 0.2315 | 0.2297 | 0.2280 | 0.2279 | 0.2274 | 0.2270 | 0.2265 | 0.2273 | 0.2256 |
| @50 | 0.2032 | 0.2041 | 0.2034 | 0.2079 | 0.2082 | 0.2089 | 0.2091 | 0.2096 | 0.2097 | 0.2107 | 0.2088 | 0.2143 | 0.2155 | 0.2154 | 0.2170 | 0.2159 | 0.2172 | 0.2195 | 0.2174 | 0.2162 | 0.2183 | 0.2190 | 0.2191 | 0.2167 |

**Endpoint** @10: t8=0.2286, t12=0.2309, t24=0.2256  |  @50: t8=0.2096, t12=0.2143, t24=0.2167

**Anytime** @10: T8=0.2227, T12=0.2251, T24=0.2268  |  @50: T8=0.2068, T12=0.2082, T24=0.2127

## Concept static -- full per-turn ENDPOINT curve (in-sample, contrast)

| metric | t1 | t2 | t3 | t4 | t5 | t6 | t7 | t8 | t9 | t10 | t11 | t12 | t13 | t14 | t15 | t16 | t17 | t18 | t19 | t20 | t21 | t22 | t23 | t24 |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| @10 | 0.2176 | 0.2250 | 0.2243 | 0.2249 | 0.2253 | 0.2247 | 0.2243 | 0.2239 | 0.2243 | 0.2235 | 0.2247 | 0.2251 | 0.2253 | 0.2252 | 0.2234 | 0.2233 | 0.2233 | 0.2231 | 0.2222 | 0.2224 | 0.2229 | 0.2226 | 0.2219 | 0.2222 |
| @50 | 0.2106 | 0.2134 | 0.2149 | 0.2160 | 0.2157 | 0.2154 | 0.2140 | 0.2142 | 0.2152 | 0.2158 | 0.2156 | 0.2171 | 0.2164 | 0.2156 | 0.2156 | 0.2146 | 0.2149 | 0.2144 | 0.2152 | 0.2151 | 0.2153 | 0.2147 | 0.2141 | 0.2148 |

## Monotonicity verdict (cohort-mean endpoint curve, max per-step decline)

| schedule | metric | max per-step decline | at turn | verdict |
|---|---|--:|--:|---|
| item | @10 | -0.0018 | t16->t17 | near-monotone (decline < 0.002) |
| item | @50 | -0.0024 | t23->t24 | NON-monotone (decline >= 0.002) |
| concept | @10 | -0.0018 | t14->t15 | near-monotone (decline < 0.002) |
| concept | @50 | -0.0015 | t6->t7 | near-monotone (decline < 0.002) |

## Static test-fit contamination (in-sample minus fair cross-fit, anytime NDCG@10)

| budget | in-sample - fair [95% CI] | n |
|---|---|--:|
| T=8 | +0.0179[+0.0068,+0.0306] | 173 |
| T=12 | +0.0205[+0.0094,+0.0329] | 173 |
| T=24 | +0.0260[+0.0144,+0.0383] | 173 |

> Positive = the all-173 greedy static gains from being fit to its own evaluation cohort. The FAIR (out-of-fold) numbers in rung 2b are the de-contaminated reference; the in-sample rung-2 curve is kept because it is the single canonical decodable schedule (and reproduces the 0.2251 anchor). Consistent with experiments/STATIC_CONTAMINATION.md.

## Full-profile: fold-v2 vs native RecVAE

| metric | v2 fold | native RecVAE | v2 - native [95% CI] |
|---|--:|--:|---|
| @10 | 0.3589 | 0.4938 | -0.1349[-0.1681,-0.1021] |
| @50 | 0.3780 | 0.4883 | -0.1102[-0.1322,-0.0891] |

> The v2 fold is trained for PARTIAL, NOISY interviews (k<=16, mixed fidelity, sigma=0.70); a full CLEAN profile of all rated items is out-of-regime, so the noise-robust fold is deliberately conservative and gives up NDCG to the native encoder on clean full profiles. Anyone needing a full-profile score uses the native RecVAE encoder (rung 4 = the instrument ceiling).

## Decoded schedule -- learned ITEM static (first 24, greedy order)

Item names via `D['title'][key]`.

| turn | tagId/itemId | name | pop-cov |
|---|--:|---|--:|
| 1 | 14234 | Hobbit: The Desolation of Smaug, The (2013) | 0.983 |
| 2 | 1262 | American Werewolf in London, An (1981) | 0.988 |
| 3 | 680 | Truth About Cats & Dogs, The (1996) | 0.983 |
| 4 | 10955 | Watchmen (2009) | 0.983 |
| 5 | 1019 | Sleepers (1996) | 0.983 |
| 6 | 7531 | Fahrenheit 9/11 (2004) | 0.994 |
| 7 | 2934 | Fatal Attraction (1987) | 0.988 |
| 8 | 256 | Little Women (1994) | 0.983 |
| 9 | 7833 | Finding Neverland (2004) | 0.988 |
| 10 | 1217 | Highlander (1986) | 0.988 |
| 11 | 988 | Dumbo (1941) | 0.983 |
| 12 | 16117 | Spotlight (2015) | 0.988 |
| 13 | 2783 | Dirty Dozen, The (1967) | 0.983 |
| 14 | 9620 | Thank You for Smoking (2006) | 0.988 |
| 15 | 4399 | Ghost World (2001) | 0.983 |
| 16 | 1046 | Basic Instinct (1992) | 0.983 |
| 17 | 159 | Desperado (1995) | 0.983 |
| 18 | 4209 | Legally Blonde (2001) | 0.988 |
| 19 | 1072 | People vs. Larry Flynt, The (1996) | 1.0 |
| 20 | 3171 | Bull Durham (1988) | 0.994 |
| 21 | 15912 | The Revenant (2015) | 0.994 |
| 22 | 1661 | Wedding Singer, The (1998) | 0.983 |
| 23 | 9813 | Stranger than Fiction (2006) | 0.988 |
| 24 | 6406 | Monty Python's The Meaning of Life (1983) | 0.988 |

## Decoded schedule -- learned CONCEPT static (first 24, greedy order)

Concept names via the arena's OWN map (tagId -> `tag` field on the grid Q cells). Map size 1128; unmapped in schedule 0.

| turn | tagId | tag name | pop-cov |
|---|--:|---|--:|
| 1 | 231 | comic | 1.0 |
| 2 | 154 | books | 1.0 |
| 3 | 122 | beautifully filmed | 1.0 |
| 4 | 269 | crime gone awry | 1.0 |
| 5 | 274 | cult | 1.0 |
| 6 | 291 | deadpan | 1.0 |
| 7 | 163 | brilliant | 1.0 |
| 8 | 235 | coming of age | 1.0 |
| 9 | 107 | based on a book | 1.0 |
| 10 | 129 | better than expected | 1.0 |
| 11 | 62 | animals | 1.0 |
| 12 | 285 | dark | 1.0 |
| 13 | 54 | amazing cinematography | 1.0 |
| 14 | 67 | anti-hero | 1.0 |
| 15 | 260 | courage | 1.0 |
| 16 | 86 | atmospheric | 1.0 |
| 17 | 220 | clever | 1.0 |
| 18 | 195 | chase | 1.0 |
| 19 | 83 | assassins | 1.0 |
| 20 | 282 | cynical | 1.0 |
| 21 | 239 | competition | 1.0 |
| 22 | 267 | creepy | 1.0 |
| 23 | 121 | beautiful scenery | 1.0 |
| 24 | 307 | disappointing | 1.0 |

## Anomaly flag: the concept POOL is an alphabetical-prefix artifact (mapping is correct)

The decode above is correct by construction (tagId -> `tag` taken from the arena's own grid cells;
0 unmapped). But the candidate pool has a tie-break artifact: 308 concepts are answered by ALL 173
users (coverage tie at 1.0), and `repair_probes.setup()` selects the top-60 concepts by coverage via
`Counter.most_common()`, whose ties break by insertion order = ascending tagId = ALPHABETICAL genome
tag order. The concept pool is therefore the alphabetically-first ~60 universally-answered tags
(0-9/a-d: "007" ... "disappointing"), NOT a value-chosen subset of the 308. This is why a previous
quick decode "read alphabetical" -- the pool itself is; the suspected tag_questions.json mapping error
was not the (only) issue. The greedy ORDER within the pool is genuine (comic, books, beautifully
filmed, crime gone awry, cult, deadpan, ...). Any conclusion about WHICH concepts are best is
pool-limited; the concept-vs-item CONTRAST (rung 2c, near-tie) is likely a LOWER bound for concepts.

## Monotonicity note

Endpoint curves are near-monotone but not strictly monotone: worst step @10 is -0.0018 (item,
t16->t17); item @50 dips -0.0024 at t23->t24 (the schedule was built to maximize @10, so mild @50
non-monotonicity is expected). The item @10 curve effectively plateaus at ~t8-t12 (0.2286-0.2309)
and drifts slightly down to 0.2256 by t24 -- late questions add no value on this instrument.

