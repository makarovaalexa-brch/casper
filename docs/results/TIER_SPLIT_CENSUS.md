# Volume-weighted rules-tier split over the full He corpus (2026-08-03)

**Question:** what fraction of the corpus's mentions does the deterministic rules tier
(`scripts/reddit/linker_rules_tier.py`) auto-accept, and how many DEFER to the Opus tier?
The audition (`REDDIT_LINKER_AUDITION.md`) measured 48.6% ACCEPT coverage on the 500-mention
gold set, but that set is stratified evenly over (year x turn role) and so massively
over-samples request/seeker_followup turns. This census is volume-weighted: every mention in
the corpus, no sampling, no cap.

**Method.** One streaming pass over `large_v1/{train,valid,test}.csv` (1.83 GB + 346 MB +
187 MB), threads deduped by submission id and turns by reddit `turn_id` exactly as
`census_he.py`; each mention's surface span recovered by `align_mentions` (word-diff of
`processed` against `raw`); a second streaming pass over `title.basics.tsv` keyed by all
82,893 distinct normalized spans builds the span-only candidate
index; `DECIDE` is then applied per distinct (span, span-year, He-tconst, role) key and
weighted by its count. Thresholds untouched.

## Corpus streamed

| quantity | count |
|---|---|
| CSV rows (path-expanded) | 1,669,720 |
| unique turns | 971,185 |
| threads | 58,623 |
| threads in eval scope (>=5 in-catalogue reply items) | 35,982 |
| tconst mentions in `processed` (census-parity count) | 1,355,236 |
| tconst mentions in the parsed utterance text | 1,734,534 |
| of those, span-alignable (adjudicable) | 949,067 |

### Controls (all three passed before any number below was read)

1. **Corpus snap.** Rows / unique turns / threads / eval threads reproduce `REDDIT_HE_CENSUS.md`
   exactly: 1,669,720 / 971,185 / 58,623 / 35,982.
2. **Alignment snap.** 949,067 alignable mentions reproduces the gold
   sampler's own population exactly (`census/gold_sample_summary.json`,
   `population_mentions_alignable`).
3. **Rules-tier snap.** The `DECIDE` copied here, re-run on `gold_combined_500.csv`, returns
   243/500 ACCEPT = 0.486 coverage at 0.971 precision -- the audition's published row to three
   decimals.

### A regex quirk worth recording

The two mention counts differ by 379,298
(22%).
`REDDIT_HE_CENSUS.md` applies `\btt\d{7,8}\b` to the raw CSV cell, where the list-style
replies that dominate this corpus write one title per line -- in the cell that newline is the
two literal characters `\` `n`, so `\ntt0177789` glues a word character onto `tt` and the `\b`
fails. Applied to the parsed utterance the same regex finds those. The census's mention totals
(and every percentage derived from them) are therefore conservative by roughly that margin;
eval-thread membership here is deliberately kept on the census-parity count so the thread set is
bit-identical to the published 35,982, but on the parsed text it would be
39,613 threads.

Only span-alignable mentions can be adjudicated at all -- a mention whose surface string cannot
be recovered from the `processed`/`raw` diff has nothing to show a model. All tier numbers below
are over the 949,067 alignable mentions.

## Tier split -- OVERALL (whole corpus)

| scope | mentions | ACCEPT | DEFER | distinct spans | spans w/ any ACCEPT | spans w/ any DEFER |
|---|---|---|---|---|---|---|
| ALL mentions | 949,067 | 517,925 (54.57%) | 431,142 (45.43%) | 82,893 | 15,943 | 74,785 |
| - reply | 824,517 | 457,312 (55.46%) | 367,205 (44.54%) | 74,098 | 15,563 | 65,870 |
| - request | 83,011 | 40,987 (49.38%) | 42,024 (50.62%) | 18,884 | 5,941 | 15,064 |
| - seeker_followup | 41,539 | 19,626 (47.25%) | 21,913 (52.75%) | 12,353 | 3,992 | 8,935 |

## Tier split -- EVAL SCOPE (35,982 threads with >=5 in-catalogue reply items)

| scope | mentions | ACCEPT | DEFER | distinct spans | spans w/ any ACCEPT | spans w/ any DEFER |
|---|---|---|---|---|---|---|
| ALL mentions | 823,128 | 465,916 (56.6%) | 357,212 (43.4%) | 72,559 | 15,152 | 64,754 |
| - reply | 737,338 | 422,207 (57.26%) | 315,131 (42.74%) | 66,758 | 14,936 | 58,829 |
| - request | 51,852 | 26,985 (52.04%) | 24,867 (47.96%) | 12,972 | 4,526 | 9,994 |
| - seeker_followup | 33,938 | 16,724 (49.28%) | 17,214 (50.72%) | 10,534 | 3,622 | 7,408 |

## Why each DEFER happened

Classification only (never feeds back into the decision). Precedence: empty candidate list
first, then stop-listed/short span, then ambiguous.

| scope | DEFER | empty candidates | ambiguous candidates | stoplist / span <4 chars |
|---|---|---|---|---|
| ALL mentions | 431,142 | 93,459 (21.7%) | 323,570 (75.0%) | 14,113 (3.3%) |
| - reply | 367,205 | 78,598 (21.4%) | 276,997 (75.4%) | 11,610 (3.2%) |
| - request | 42,024 | 9,592 (22.8%) | 31,054 (73.9%) | 1,378 (3.3%) |
| - seeker_followup | 21,913 | 5,269 (24.0%) | 15,519 (70.8%) | 1,125 (5.1%) |
| EVAL ALL mentions | 357,212 | 78,494 (22.0%) | 267,322 (74.8%) | 11,396 (3.2%) |
| EVAL - reply | 315,131 | 68,467 (21.7%) | 236,925 (75.2%) | 9,739 (3.1%) |
| EVAL - request | 24,867 | 5,779 (23.2%) | 18,293 (73.6%) | 795 (3.2%) |
| EVAL - seeker_followup | 17,214 | 4,248 (24.7%) | 12,104 (70.3%) | 862 (5.0%) |

Of the eval-scope ambiguous DEFERs, 98,766
have a candidate that clears the confidence bar (>=2000 votes, >=10x runner-up) but is NOT the
tconst He's release assigned -- i.e. the rules tier and He actively disagree there, which is
precisely the population the audition showed He getting wrong (57.2% precision as shipped).

## Cost arithmetic for the Opus adjudication run

The audition observed ~400-550 tokens per mention on the LLM tier under production batching
(`REDDIT_LINKER_AUDITION.md`, reading #4); **500 tokens/mention** is the midpoint assumption used
here. It covers the prompt share (policy + sentence + candidate list) plus the answer, amortised
over batches of ~500 mentions per agent.

| scope | DEFER mentions | x 500 tok/mention | = estimated Opus tokens |
|---|---|---|---|
| eval scope (the run of record) | 357,212 | 500 | **178,606,000** (178.6M) |
| whole corpus | 431,142 | 500 | 215,571,000 (215.6M) |

At the audition's observed band the eval-scope figure ranges
142.9M-196.5M tokens
(400-550 tok/mention).

## Caveats

- The tier split is computed from the span, its parenthetical year, and He's tconst only -- no
  context, exactly as the deployed rules tier works. It is a *coverage* number, not an accuracy
  number; accuracy on the ACCEPT tier was 0.971 on gold.
- ACCEPT requires agreement with He's link, so ACCEPT volume is bounded by He's own precision.
  The tier cannot accept a mention He linked wrongly -- those land in DEFER by construction.
- **785,467 mentions
  (45%
  of the parsed-text total) are not span-alignable** and are excluded from both tiers. This is
  the same phenomenon as the regex quirk above: `align_mentions` attributes a surface span only
  when a diff block contains exactly one tconst, and the list-style reply ("title\ntconst\ntitle
  \ntconst...") collapses into one block holding many. Recovering them needs a line-aware
  alignment, not a bigger LLM budget -- and it would enlarge the DEFER tier, so the token
  estimate below is a floor for full coverage, not a ceiling.

Raw counts: `data/reddit_he/census/tier_split.json`. Script: `scripts/reddit/tier_split_census.py`.
