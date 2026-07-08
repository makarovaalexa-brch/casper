# Answerer v1 -- CORRECTED design SMALL BATCH (10 users)

Judge `gpt-5.4-mini-2026-03-17` temp 0.0, split seed 123, user seed 1. NO MASKING; full known-half profile (cap 200). 10 users x (1128 tags + 500 attrs + top800-minus-rated items) + 1 quarantined validation call each. Grid `.cache/instrument2/answerer_v1_grid10.json`; validation `.cache/instrument2/answerer_v1_valid.json` (QUARANTINED). NOT the 300-user run.

Users: [757, 2389, 2988, 22718, 27396, 30250, 31858, 65649, 74895, 98268]

## (a) Cost + token compression

- measured: 107 calls, 426578 prompt + 191017 completion tokens, **$0.4887** (rates $0.25/M in, $2.0/M out)
- per-user **$0.0489** | **PROJECTED 300-USER = $14.66**
- production completion tok/Q = **8.05** vs pilot 19.14 => **compression ratio 0.421** (PASS <0.50)
- production prompt tok/Q = 17.98
- tripwire (2 users): 10-user proj $0.48, 300-user proj $14.33

## (b) Parse / core-contract rate

| channel | cells | returned | core-valid | core% | conf% | hard-invalid |
|---|--:|--:|--:|--:|--:|--:|
| concept | 11280 | 11280 | 11280 | 100.00% | 100.00% | 0 |
| attribute | 5000 | 5000 | 5000 | 100.00% | 100.00% | 0 |
| item | 7307 | 7307 | 7307 | 100.00% | 100.00% | 0 |
| **data-filled** | 337 | 337 | 337 | 100.00% | 100.00% | 0 |

Overall LLM core-contract = **100.00%** (PASS >=99%).

Of the core-valid cells, 0 were COERCED to no_clue (judge emitted k>=1 with an out-of-range/zero half-star = failed to commit a rating; treated as a refusal per the safety net).

## (c) Validation call (held-out-half items -- QUARANTINED)

- n held-out items judged = 143; non-no_clue with stars = 140; no_clue = 3
- **corr(predicted stars, true held-out rating) = 0.547** (micro-check baseline 0.359; expect >= since full profile)
- star MAE = 0.732 stars
- binned exact = 0.386 | adjacent = 0.921
- **liked-share = 0.536** (n=140)
- marginals -- PRED: hated 0.114, meh 0.279, liked 0.536, loved 0.071 | TRUE: hated 0.064, meh 0.257, liked 0.400, loved 0.279

## (d) Knowledge base rates per channel

| channel | n | no_clue | rough_idea | know_well |
|---|--:|--:|--:|--:|
| concept | 11280 | 0.318 | 0.339 | 0.342 |
| attribute | 5000 | 0.268 | 0.541 | 0.191 |
| item | 7307 | 0.007 | 0.396 | 0.596 |

Overall rough_idea share = 0.400 (USED).

Attribute answerability by type (composers/writers expected weakest):

| type | n | no_clue | rough_idea | know_well |
|---|--:|--:|--:|--:|
| director | 1500 | 0.511 | 0.405 | 0.084 |
| actor | 2000 | 0.131 | 0.669 | 0.200 |
| composer | 500 | 0.330 | 0.600 | 0.070 |
| writer | 250 | 0.192 | 0.680 | 0.128 |
| franchise | 750 | 0.132 | 0.385 | 0.483 |

## (e) Degenerate tag columns (>=90% of these 10 users share one label; n>=8)

401 degenerate tag columns (of 1128). By label: {'know_well': 183, 'no_clue': 175, 'rough_idea': 43}. (10-user batch => small-n; illustrative, not the prune decision.)

all-no_clue degenerate example tags: ['70mm', 'almodovar', 'amy smart', 'argentina', 'bdsm', 'beauty pageant', 'berlin', 'bollywood', 'bowling', 'brainwashing', 'brazil', 'broadway', 'bullshit history', 'c.s. lewis', 'canada', 'cheerleading', 'china', 'dolphins', 'dr. seuss', 'east germany']

## (f) Assumptions

- **Profile cap**: full known-half shown uncapped for 9/10 users; 1 exceeded 200 titles => deterministic star-stratified subsample to 200, flagged in-prompt (size string). Capped users: [31858].
- **Battery**: rebuilt attr_battery_500.json from attr_membership.json, top-pop per type dir150/act200/comp50/writ25/franch75; verified an EXACT id/phrasing subset of the prior 700 battery's per-type top-N (no phrasing drift).
- **Item grid**: top-800 popular catalog. Known-half-rated items => source="data" (real rating, know_well, conf 1.0). Held-out-half items => EXCLUDED from grid entirely (leakage guard; runtime assertion enforced). Remaining (unrated) => LLM graded stars.
- **Validation quarantine**: 15 held-out items/user (deterministic rng 5000+u), predicted k+stars, written ONLY to .cache/instrument2/answerer_v1_valid.json; never enters the grid.
- **Selection**: 10 users, seed 1, profile-size-tercile x dominant-genre round-robin, disjoint from the pilot's 20.
- **Compression**: output is a COMPACT integer array row [i,k,h,c] per question -- i=index (kept as the alignment anchor so the grid never misaligns), k=knowledge 0/1/2, h=half-stars 1..10 (0 iff k==0), c=confidence in tenths 0..10. Decoded back to knowledge/stars/value/conf on parse. The literal {"i","k","s","c"} DICT form the author sketched was measured first and gave ~20 tok/Q (ratio ~1.0 vs pilot) -- JSON dict punctuation/keys dominate and the pilot's word-labels already tokenize cheaply, so it did NOT meet the <50% target; the array form does (ratio 0.421). Question lines rendered "idx: text" (no [type] wrapper).
- **Constant completion length / truncation check**: full 260-question batches return a near-constant ~2091 completion tokens. This is NOT a max_completion_tokens truncation (none is set): every 260-row batch parsed 260/260 rows as complete well-formed JSON (finish_reason=stop; a mid-list cut would null the whole batch on json.loads), zero None and zero _raw cells. The constant is the compact fixed-width row format's signature at temperature 0 (each [i,k,h,c] row is structurally identical length regardless of the answer).
- **h=0 contradiction fix**: a first pass (compact prompt without the h-range constraint) had 2.2% invalid cells = rows [i,k>=1,h=0] (knows-it-but-zero-stars), 416/514 from one user. Fixed by a prompt constraint ('when k>=1, h MUST be 1..10; use k=0 if you cannot rate') + a parse safety net (k>=1 with h<1 -> no_clue). Re-collected fresh: 0 invalid, 0 coercions.
