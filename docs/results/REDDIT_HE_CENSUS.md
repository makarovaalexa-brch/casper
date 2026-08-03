# He-corpus census (`ZhankuiHe/reddit_movie_raw` + `…_large_v1`) — CENSUS phase, no LLM

Executes steps 1–3 of `docs/design/DS_REDDIT_LINKING.md`. Run 2026-08-03.
Reproduce (from `C:/dev/phd/casper`):

```
poetry run python scripts/reddit/download_he.py --which raw
poetry run python scripts/reddit/download_he.py --which v1_large
poetry run python scripts/reddit/census_he.py raw
poetry run python scripts/reddit/census_he.py v1
poetry run python scripts/reddit/census_he.py gold       # also dumps v1 thread ids
poetry run python scripts/reddit/census_he.py coverage    # needs the gold pass first
```

Machine-readable outputs: `data/reddit_he/census/{raw_census,v1_census,v1_coverage,
gold_sample_summary}.json` (not committed — `data/reddit_he/` was added to `.gitignore`).

**No LLM call was made. No Reddit API call was made** — everything below comes from the
public Hugging Face release. Nothing was sampled, capped or truncated in any count: all
18 raw files and all three V1 CSVs were streamed end to end. The only sample drawn is the
300-mention gold set requested for human labelling.

---

## 0. What actually ships (this reframes questions 3 and 4)

| repo | what it is | size |
|---|---|---|
| `ZhankuiHe/reddit_movie_raw` | 5 raw subreddit `*.jsonl` (one JSON object per **thread**), `sentences.jsonl`, `IMDB-database/` | 3.01 GB |
| `ZhankuiHe/reddit_movie_large_v1` | `{train,valid,test}.csv` + `id2name.json` — **this is the only place their entity links live** | 2.36 GB |
| `ZhankuiHe/reddit_movie_small_v1` | same, 2022 only | 0.51 GB |
| `ZhankuiHe/reddit_cikm` | the CIKM'23 *benchmark* subset (`*_data_processed.jsonl`) | 0.18 GB |

**The raw release carries NO entity annotations at all.** `sentences.jsonl` is sentence-split
raw text keyed by `("<subreddit>", "<t3_/t1_ id>", "<field>")` — no ids, no spans.
`IMDB-database/` is a *gazetteer* (`title.basics` + `title.ratings` filtered to
`titleType=='movie'`, runtime 60–360 min), i.e. the input to a matcher, not a matching.
So "their machine-matched movie mentions" = the **V1 processed release**, where the
`processed` column is the utterance with every linked mention substituted **in place** by an
IMDb tconst and `raw` is the same utterance with the original surface strings. Recovering
(tconst, surface span) pairs is a word-level diff of `processed` against `raw` — done here in
`align_mentions()`.

Raw thread record: `{submission: {t3_id: {...}}, valid_comments: {t1_id: {...}}, head_comments,
ghost_comments}`; comments carry `parent_id`, `child_ids`, `author`, `score`, `is_submitter`.

### Protocol (fixed for everything below)
- **Thread = one `t3_…` submission id.** Raw: the single key of the `submission` dict.
  V1: `conv_id` minus the trailing `_<path>/<npaths>` (`t3_rt7enj_0/14` → `t3_rt7enj`).
- **Turn dedup:** V1 rows are *path expansions* of a thread, so the same reddit turn recurs
  many times; turns are deduplicated by `turn_id` **within** a thread. Confirmed safe: all
  rows of a thread are contiguous (`threads_reopened_out_of_order = 0`).
  Scale of the trap here: **1,669,720 CSV rows → 971,185 unique turns → 58,623 threads.**
- **Linking source:** He et al.'s own V1 links. No new linker was run.
- **Catalogue map:** `data/movielens/links.csv` (`movieId`↔`imdbId`, 62,423 rows) restricted to
  the 18,359 ids in `data/ml-25m/proc/unique_sid.txt`. All 18,359 resolve.
- **Year** = UTC year of `created_utc` / `utc_time`.
- **Privacy:** every author/user string is sha256-hashed (first 12 hex) before it is written
  anywhere. No request text is reproduced in this document.

---

## 1 ★ THE DATE QUESTION — settled, and the answer is split

### Raw release: pre-2019 data DOES exist, but not in r/MovieSuggestions

Threads (deduplicated by submission id) per year per file. Every file's per-year counts sum
exactly to its total and no duplicate submission ids were found (sanity control passed for all
five files; `year_sum_check: true`, `duplicate_submission_ids_dropped: 0`).

| year | bestofnetflix | movies | moviesuggestions | netflixbestof | truefilm | **total** |
|---|---|---|---|---|---|---|
| 2012 | 227 | 0 | 0 | 7 | 0 | 234 |
| 2013 | 959 | 0 | 0 | 24 | 0 | 983 |
| 2014 | 783 | 853 | 0 | 48 | 0 | 1,684 |
| 2015 | 675 | 2,490 | 0 | 771 | 0 | 3,936 |
| 2016 | 649 | 2,168 | 0 | 805 | 21 | 3,643 |
| 2017 | 307 | 1,201 | 0 | 845 | 1 | 2,354 |
| 2018 | 441 | 1,206 | 0 | 763 | 10 | 2,420 |
| 2019 | 542 | 1,617 | 4,931 | 1,351 | 27 | 8,468 |
| 2020 | 363 | 3,199 | 17,032 | 2,192 | 87 | 22,873 |
| 2021 | 640 | 3,101 | 15,054 | 2,032 | 54 | 20,881 |
| 2022 | 519 | 3,855 | 17,529 | 1,298 | 108 | 23,309 |
| **total** | **6,105** | **19,690** | **54,546** | **10,136** | **308** | **90,785** |

- **Reproduction control PASSED exactly:** `moviesuggestions.jsonl` = **54,546 threads,
  2019–2022 only** — the Jul-31 measurement reproduces to the unit, and the 0/1,480 Eberhard
  overlap is fully explained.
- The HF card's "Jan 2012 – Dec 2022" is true **of the bundle**, not of any single file.
  Earliest data: r/bestofnetflix and r/netflixbestof from 2012, r/movies from 2014,
  r/truefilm from 2016, r/MovieSuggestions from 2019.
- **≤2018: 15,254 threads. ≤2019 (inclusive): 23,722 threads.** None of them is a
  r/MovieSuggestions request.

### V1 (the linked data): pre-2019 does NOT exist

Measured directly on the `utc_time` column: the large-V1 range is
**2018-06-29 03:36:03 UTC → 2022-12-31 23:59:46 UTC.** Per-year V1 threads:
2018 → 958, 2019 → 7,172, 2020 → 16,830, 2021 → 15,271, 2022 → 18,392 (sum = 58,623 = total,
sanity control passed). Its own dataset card also claims 2012–2022; **that claim is false for
the file that actually carries the links.**

**Verdict (a).** Pre-2019 threads exist in the raw dump — 15,254 of them — but (i) none is
from the request subreddit and (ii) none of them was processed into the entity-linked V1
release. A ≤2019 evaluation slice therefore exists **only if we link it ourselves**, and it
would be built from r/movies / r/bestofnetflix / r/netflixbestof, which are discussion and
"here's a good film" posts, not requests. The pre-2019 *request* corpus remains Eberhard's,
and Eberhard still ships no request text.

---

## 2. Threads and replies (raw release, all five files)

| file | threads | comments | mean | p10 | median | p90 | max |
|---|---|---|---|---|---|---|---|
| bestofnetflix | 6,105 | 82,667 | 13.54 | 1 | 8 | 31 | 760 |
| movies | 19,690 | 806,478 | 40.96 | 1 | 12 | 71 | 4,519 |
| **moviesuggestions** | **54,546** | **1,346,056** | **24.68** | **2** | **12** | **43** | **3,140** |
| netflixbestof | 10,136 | 409,453 | 40.40 | 4 | 16 | 89 | 1,089 |
| truefilm | 308 | 6,348 | 20.61 | 1 | 6 | 58 | 252 |
| **all** | **90,785** | **2,651,002** | — | — | — | — | — |

`ghost_comments` is empty in every thread of every file (0 across the corpus).
V1 turns-per-thread (post-dedup): mean 16.57, p10 3, median 10, p90 30, max 2,037; of these,
non-seeker turns mean 13.96, median 8.

---

## 3. Catalogue fit — and the 61.3% figure reconciled

All numbers from He et al.'s own links, turn-deduplicated, over **all** of large-V1
(1,355,236 mentions, 50,075 distinct tconsts).

| slice | mentions | % mentions in catalogue | distinct items | % distinct in catalogue |
|---|---|---|---|---|
| **large-V1, all years** | 1,355,236 | **74.59%** | 50,075 | **28.63%** |
| 2018 (n=958 threads) | 24,250 | 66.70% | 6,833 | 54.35% |
| 2019 | 211,023 | 74.35% | 21,739 | 45.33% |
| 2020 | 498,652 | 75.51% | 31,773 | 37.80% |
| 2021 | 266,386 | 74.20% | 24,118 | 44.50% |
| 2022 | 354,925 | 74.27% | 28,628 | 40.58% |

**The 75.1% mention figure reproduces (74.59%). The 61.3% distinct-item figure does not —
and the disagreement is a denominator, not an error.** Both numbers are reported; neither is
averaged away. Measured variants:

| item universe | distinct in 18,359 catalogue | distinct in full `links.csv` (62,423) |
|---|---|---|
| large-V1 mention vocabulary (50,075) | 28.63% | 49.34% |
| `id2name.json` (51,203) | 28.22% | 50.35% |
| small-V1 (2022 only) vocabulary (28,138) | 41.20% | 59.42% |
| **`reddit_cikm` test split (7,413)** | **63.87%** | 74.07% |
| `reddit_cikm` valid split (10,691) | **60.26%** | 72.26% |
| `reddit_cikm` train split (21,032) | 48.51% | 65.46% |

The Jul-31 pair "61.3% distinct / 75.1% mentions" sits exactly on the `reddit_cikm`
**benchmark subset** (valid 60.26%/74.97%, test 63.87%/74.00%) — the CIKM'23 release, whose
item vocabulary is far smaller than V1's (7,413 in test, 21,032 in train, vs 50,075) and
therefore much better covered. Over the full V1 corpus the same catalogue covers only 28.6%
of the *types* while still covering 74.6% of the *tokens*: the 35,741 uncovered tconsts are
long-tail and post-2019 titles (all of them are IMDb `titleType=='movie'` — see §4 — so the
gap is genuinely catalogue coverage, not junk ids). **Practical read: the mention-level number (~75%) is the one that governs
ground-truth retention and it is stable across every slice (73.6–75.5%); the distinct-item
number is only interpretable against a stated vocabulary.**

The by-year split does **not** show the hoped-for pre-2019 lift, because V1 has no real
pre-2019 mass: 2018's 66.7% is *worse* than 2020's 75.5% (n=958 threads, and those threads
are r/movies-style discussion of then-current releases). The recency-coverage argument cannot
be tested on this corpus.

---

## 4. How good is their entity linking?

**Coverage.** Joining V1 thread ids back to the raw dump (sanity control: the per-subreddit
kept-counts sum to 58,623, exactly the V1 thread total):

| file | raw threads | kept in V1 | % kept | kept ≤2017 |
|---|---|---|---|---|
| bestofnetflix | 6,105 | 1,492 | 24.4 | **0** |
| movies | 19,690 | 8,077 | 41.0 | **0** |
| moviesuggestions | 54,546 | 42,901 | 78.7 | (no pre-2019 data) |
| netflixbestof | 10,136 | 5,980 | 59.0 | **0** |
| truefilm | 308 | 173 | 56.2 | **0** |
| **all** | **90,785** | **58,623** | **64.6** | **0** |

r/MovieSuggestions supplies **73.2% of all V1 threads**. 2018 is partial in every file
(157/441, 460/1,206, 340/763, 1/10) consistent with the 2018-06-29 start; every year before
that is kept at exactly zero.

- V1 keeps **58,623 of the 90,785 raw threads (64.6%)**, and none from before mid-2018.
- **99.87% of V1 threads (58,544)** carry ≥1 link; **73.67% of V1 turns (715,482 of 971,185)**
  carry ≥1 link, 255,703 turns carry none.
- Mentions: 1,355,236 total — 180,479 in seeker turns, 1,174,757 in replier turns
  (mean 23.1 linked mentions per thread).
- Independent free signal in the raw dump (no model needed): **19,070 threads (21.0%) contain
  an explicit `imdb.com/title/tt…` URL, 157,879 such URLs in total** — mostly bot posts
  (`movie_info_bot` and kin), which V1 drops. This is a usable, zero-cost precision anchor
  for validating any future linker.

**Quality.** Their pipeline is gazetteer string matching (`IMDB-database/clean.py`:
`titleType=='movie'`, runtime 60–360 min → 355,749 candidate titles), and its failure mode is
over-linking. Two deterministic checks:

*Type check passes trivially.* All 51,203 ids in `id2name.json` are in the movie gazetteer —
100% are IMDb `titleType=='movie'`. So type filtering catches nothing; the errors are all
*wrong-entity*, not *wrong-type*.

*Surface-span audit* (all 949,067 alignable mentions in large-V1, spans recovered by diffing
`processed` against `raw`; 100,365 distinct spans). The top-60 spans by frequency contain
unmistakable false-positive classes:
- **platform / generic nouns**: `netflix` (2,173 mentions), `dark` (2,517 — the adjective in
  "dark thriller"), `mother`, `drive`, `arrival`, `prisoners`, `alien`, `the thing`,
  `about time`, `good time`, `coherence`;
- **TV series matched to same-titled films**: `breaking bad`, `black mirror`, `ozark`,
  `mindhunter`, `the office`, `stranger things`, `the good place`, `the oa` — each linked to
  an IMDb *movie* with that title, so the type check cannot see them.

Two worked examples from the shipped data: a `reddit_cikm` test record links the ordinary
word *"Movie"* to `tt1040023` ("Movie (2007)"), twice, creating a spurious recommendation
target; a 2018 large-V1 comment links the genre phrase *"True Crime"* to `tt0139668`
(Eastwood's *True Crime*). Note also He et al.'s own disclaimer: *"processing is not
perfect… we do observe some failure cases"*, and the explicit invitation to build a V2.

No precision/recall number is claimed here — that is exactly what the gold set in §7 is for.

**Verdict (c).** Their linking is good enough for a *coverage* claim and useless as a
*precision* claim. A re-linking pass is still needed — but the census changes its shape and
its size: it is **not** a corpus-wide extraction job, because V1 already supplies candidate
spans. The job is (i) precision filtering of the ~1.36 M existing links (the single largest
win, and a large fraction is reachable with a stop-title blocklist plus a year/context check —
no LLM), and (ii) recall recovery on the 255,703 unlinked turns and the 32,162 raw threads V1
dropped, which *is* LLM-shaped. Scoped to the threads that would actually become evaluation
items (see §6), the LLM job is ~36 k threads, not 90 k.

---

## 5. Organic follow-up subset (OP answered in the reply tree)

Measured on the **raw** files, since V1 discards the `is_submitter` flag. Two definitions,
both reported:
- *OP follow-up*: OP posted ≥1 comment in their own thread.
- *OP answered someone*: OP posted a comment whose parent is a comment by a **different**
  author — the strict "answered a clarifying question" event.

`is_submitter` is missing on older records (present in 2,693/6,105 bestofnetflix threads but
all 54,546 moviesuggestions threads), so the flag is OR-ed with an author-identity check
against the hashed submission author (`[deleted]` and `AutoModerator` excluded).

| file | threads | OP follow-up | % | OP answered someone | % |
|---|---|---|---|---|---|
| bestofnetflix | 6,105 | 2,796 | 45.80 | 2,238 | 36.66 |
| movies | 19,690 | 10,550 | 53.58 | 9,665 | 49.09 |
| **moviesuggestions** | **54,546** | **24,216** | **44.40** | **23,434** | **42.96** |
| netflixbestof | 10,136 | 5,607 | 55.32 | 5,462 | 53.89 |
| truefilm | 308 | 155 | 50.32 | 149 | 48.38 |
| **all** | **90,785** | **43,324** | **47.72** | **40,948** | **45.10** |

This is much larger than expected and is the one place on this corpus where adaptivity can be
measured without simulating an answerer: **23,434 r/MovieSuggestions threads (43.0%)** contain
a genuine OP reply to someone else's comment.

---

## 6. Ground-truth viability

Ground truth = distinct in-catalogue items mentioned in **replier** turns, minus every
in-catalogue item the **seeker** named anywhere in the thread (V1's `is_seeker` makes this
distinction cleanly, so the Eberhard OP-exclusion protocol is directly implementable).

- Set size per thread: mean **11.26**, p10 1, median 7, p90 25, max 579; 660,345 items total.
- **35,982 of 58,623 threads (61.38%) have ≥5 in-catalogue reply items** — these survive the
  design sheet's filter; the filter drops 22,641 threads (38.6%).
- By year: 2018 → 420, 2019 → 4,642, 2020 → 11,434, 2021 → 8,254, 2022 → 11,232.
- **≤2019 slice: 5,062 threads** pass the ≥5 filter. Non-trivial in isolation, but it is
  2018–19 only, mostly non-request subreddits, and it is *not* the pre-2019 slice the design
  sheet hoped for.
- Ceiling reminder (unchanged): with median 7 / mean 11.3 ground-truth items, Recall@10 is
  capped well below 1 and must be reported alongside every number.

---

## 7. Gold sample for hand labelling

`data/reddit_he/gold_sample_unlabeled.csv` — **300 mentions**, stratified over
year × mention-context, context ∈ {`request` (seeker, turn 0), `seeker_followup` (seeker,
later turn), `reply` (non-seeker)}. Reservoir sampling, seed 98765, over **every** alignable
mention in large-V1 whose surface span could be recovered unambiguously (see the note on the
sampling frame below). Columns: `thread_id`, `turn_id`,
`user_hash` (sha256[:12]), `year`, `context`, `he_tconst` (their link, to be judged),
`mention_span`, `sentence`, and an **empty** `label_tconst_or_NONE`.

**It is unlabelled by construction — no LLM and no guessing was applied.** Label it before
looking at any linker output, per step 3 of the design sheet.

Population and allocation (15 non-empty strata = 5 years × 3 contexts, 20 mentions each):

| year | request | seeker_followup | reply |
|---|---|---|---|
| 2018 | 1,493 | 904 | 16,438 |
| 2019 | 9,439 | 6,122 | 135,583 |
| 2020 | 23,222 | 15,635 | 297,045 |
| 2021 | 21,797 | 8,291 | 149,249 |
| 2022 | 27,060 | 10,587 | 226,202 |

Population total **949,067 alignable mentions** (of 1,355,236 linked mentions; the 30% gap is
mentions whose surface span the word-level diff could not isolate unambiguously — e.g. two
tconsts inside one replaced run. This affects the *sampling frame* for hand labelling only,
not any count in §1–§6, and it is deliberately conservative: ambiguous alignments are
excluded rather than guessed). Equal allocation over-samples `request` and
`seeker_followup` relative to their natural frequency — that is intended: those are the
contexts the ruler conditions on, and the 2018 cells are otherwise too thin to estimate.
Reweight to the population column when reporting a corpus-level precision.

---

## Verdicts

**(a) The date question.** Settled, both ways. The HF card is right about the *bundle* and
wrong about r/MovieSuggestions: that file is 2019–2022, exactly 54,546 threads, reproducing
the Jul-31 measurement to the unit. Pre-2019 raw threads do exist — 15,254 of them, from
r/movies, r/bestofnetflix, r/netflixbestof and r/truefilm — but **He et al.'s linked V1
release starts 2018-06-29 and contains none of them.**

**(b) Does the ≤2019 slice become the primary evaluation slice?** **No.** It fails on three
counts: the pre-2019 threads are not requests (no r/MovieSuggestions before 2019); they carry
no entity links at all, so building the slice means linking 15,254 threads from scratch with
nothing to validate against; and the coverage premise behind the idea is not supported —
V1's 2018 mention-level catalogue fit is 66.7%, *below* 2020's 75.5%. The primary slice stays
the 2019–2022 r/MovieSuggestions request corpus, and the recency-bias scope condition stays a
first-class limitation rather than shrinking to a caveat. The ≤2019 idea is not dead, but it
is a separate, linking-first work item, not a shortcut.

**(c) Is LLM re-linking still needed, and how big?** Yes, but smaller and differently shaped
than the "3–5 weeks of extraction" estimate. V1 already gives 1.36 M candidate (span → tconst)
pairs at 99.9% thread coverage / 73.7% turn coverage, so the work is *adjudication*, not
extraction. Sized against the evaluation set that actually matters — the 35,982 threads with
≥5 in-catalogue reply items, ~23 linked mentions per thread — the batch is on the order of
1 M mention judgements, and a large share of the precision problem (generic-word titles) is reachable by
a deterministic blocklist first. Recommended order: gold-label the 300 → measure V1's raw
precision/recall → apply the no-LLM blocklist and re-measure → only then take a scoped LLM
pass to the author-approval gate (step 4), sized on what the first two steps leave.

## Known limitations of this census
- Catalogue fit inherits He et al.'s linking errors: a mention wrongly linked to an
  in-catalogue title counts as in-catalogue here. §7's gold set is what corrects this.
- `by_year` mention counts are keyed on the **turn's** timestamp, thread counts on the
  submission's; they differ for threads whose replies cross a year boundary.
- The OP-follow-up count uses hashed-author identity where `is_submitter` is absent; a
  deleted/renamed account can therefore be missed (counted conservatively, never inflated).
- V1's `is_seeker` is He et al.'s own label; it was not independently verified.
