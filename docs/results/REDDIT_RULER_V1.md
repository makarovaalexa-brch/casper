# Reddit He-corpus evaluation ruler v1 (2026-08-03)

Assembled by `scripts/reddit/assemble_ruler.py` from the completed adjudication
(batches 0000-0099) plus the rules-tier ACCEPTs. No LLM call, no Reddit API, no sampling:
every mention of every completed batch is carried through, and each reduction below is one of the
four pre-registered ones (NONE, unresolvable TITLE, out-of-catalogue, <5 GT items).

Ruler definition (`docs/design/DS_REDDIT_LINKING.md` step 6): **GT = in-catalogue items mentioned
in REPLY turns, minus anything the OP named** (request / seeker_followup turns = OP_NAMED, the
Eberhard protocol); a thread enters the ruler when it has >= 5 GT items.

## Funnel

| stage | count | share |
|---|---|---|
| adjudicated mentions in (manifest 53,051) | 53,051 | 100% |
| - model answered NONE (dropped) | 9,349 | 17.6% |
| - model answered a tconst | 39,995 | 75.4% |
| - model answered TITLE:... | 3,707 | 7.0% |
| -- TITLE resolved to a tconst | 3,330 | 89.8% of TITLE |
| -- TITLE unresolved (dropped) | 377 | 10.2% of TITLE |
| - malformed answers (dropped) | 0 | 0.0% |
| **model links** | **43,325** | 81.7% of mentions in |
| rules-tier ACCEPT links in these threads | 70,028 | (of 465,916 over all 35,982 eval threads) |
| **links total** | **113,353** | |
| - in the 18,359-item catalogue (model) | 30,121 | 69.5% |
| - in the 18,359-item catalogue (rules) | 64,896 | 92.7% |
| threads in scope | 5,441 | |
| **threads kept (>= 5 GT)** | **4,431** | 81.4% |
| threads dropped (< 5 GT) | 1,010 | 18.6% |
| kept threads with request text | 4,431 | 100.0% |

## Ruler shape (kept threads)

| quantity | mean | median | p10 | p90 | max | total |
|---|---|---|---|---|---|---|
| GT items / thread | 14.883 | 10 | 5 | 28 | 299 | 65,948 |
| OP_NAMED items / thread | 1.709 | 1 | 0 | 4 | 56 | 7,572 |

OP_NAMED coverage: 2,639 of
4,431 kept threads name at least one in-catalogue item in the request or a
seeker follow-up (59.6%).

## TITLE resolution outcomes

| outcome | count |
|---|---|
| no_title_match | 277 |
| resolved_no_year | 3 |
| resolved_with_year | 3,327 |
| year_mismatch | 100 |

Resolver: normalized match (case / punctuation / hyphen / diacritics insensitive, `&`==`and`,
apostrophes dropped) against BOTH `primaryTitle` and `originalTitle` of `title.basics.tsv`,
restricted to the same title types the candidate index uses; article variants ("the X" / "X")
tried; year within +-1 when the answer gave one; most-voted title wins ties. `year_mismatch` =
a title matched but no candidate sat within +-1 of the stated year -- dropped, not guessed.

## Spot check (10 random kept threads, seed 98765; reddit handles redacted)

| thread | GT | OP | first GT links (span -> tconst -> movieId) | request excerpt |
|---|---|---|---|---|
| `t3_g776t2` | 7 | 0 | `The Usual Suspect -> tt0114814 -> mid 50`<br>`Bicycle Thieves -> tt0040522 -> mid 3089`<br>`Go for Birdman (2014) -> tt2562232 -> mid 112183` | Suggest me a movie. Can you suggest me a movie that looks plain and not that deep at first but after you finish the movie it suddenly hits you and you realize what it all meant. |
| `t3_f4qq0g` | 5 | 0 | `Clueless -> tt0112697 -> mid 39`<br>`Reality Bites (1994) -> tt0110950 -> mid 372`<br>`The Breakfast Club -> tt0088847 -> mid 1968` | Movies like the TV show FRIENDS. Doesn't have to be lighthearted or just comedic.. Relatable interesting characters belonging in the same age group dealing with everyday life with interesting outcomes. |
| `t3_v6rs7e` | 8 | 0 | `Predator -> tt0093773 -> mid 3527`<br>`The others -> tt0230600 -> mid 4720`<br>`Oldboy -> tt0364569 -> mid 27773` | Unpredictable movies with plot twist. Been loving these type of movies since then. It just give satisfaction, and make my time watching it 10x worth it. Something that will leave me in awe. Would love to discover more. U |
| `t3_xc11v0` | 19 | 2 | `The Umbrellas of Cherbourg (1964) -> tt0058450 -> mid 199`<br>`Ghost (1990) -> tt0099653 -> mid 587`<br>`Sophie's Choice (1982) -> tt0084707 -> mid 1096` | Tragedy Heart-wrenching Romance movies. Tragedy and Heart-wrenching Romance movies that leave your heart in pain after watching the movie. Movies like 'Never let me go' and 'Bridge to Terabithia'. |
| `t3_l95v30` | 12 | 10 | `'Twelve Monkeys' -> tt0114746 -> mid 32`<br>`Seven -> tt0114369 -> mid 47`<br>`'2001: A Space Odyssey' -> tt0062622 -> mid 924` | Searching for the best action/thriller/sci-fi films. Hi guys! As background, every weekend I go over to my parents' house, where we usually watch a movie (*don't worry re: COVID: we are all teleworking and going literall |
| `t3_vksskw` | 27 | 0 | `To Die For -> tt0114681 -> mid 45`<br>`Swimming with Sharks -> tt0114594 -> mid 322`<br>`Singin' In The Rain (1952) -> tt0045152 -> mid 899` | Movies about trying to make it in show business.. It's all in the title. Maybe an unknown trying to make it or celebrity's kid living in their parents shadow or something similar. Cant think of many examples. |
| `t3_cjz32l` | 13 | 1 | `The Abyss -> tt0096754 -> mid 1127`<br>`The Fog -> tt0080749 -> mid 1128`<br>`for a Jaws -> tt0073195 -> mid 1387` | Nautical Horror films?. Just saw the trailer for The Lighthouse and now I’m in the mood for some nautical horror. Any suggestions? |
| `t3_ui838c` | 12 | 1 | `There's something about Mary, -> tt0129387 -> mid 1923`<br>`What Women Want -> tt0207201 -> mid 4018`<br>`Alfie -> tt0060086 -> mid 4191` | What are some good movies where the protagonist is irresistible and attractive to other people around them?. Looking for recommended films where the protagonist (either male or female) are sexually attractive to people a |
| `t3_dh2jlt` | 7 | 0 | `Withnail and I -> tt0094336 -> mid 1202`<br>`The Boondocks Saints. -> tt0144117 -> mid 3275`<br>`Snatch -> tt0208092 -> mid 4011` | Movies with British/Irish/Scottish accents with lots of cursing and swearing. I loved the first two seasons of Shameless (UK) and loved the unabashed, brash portrayal of the dysfunctional family. I especially loved the c |
| `t3_o5httg` | 11 | 8 | `Inferno -> tt0080923 -> mid 3587`<br>`"StageFright (1987)" -> tt0092576 -> mid 5163`<br>`"Irreversible" -> tt0290673 -> mid 6214` | Horror films similar to the original Suspiria?. I’m looking for something visceral and surreal. What I love most about Suspiria is Dario Argentos use of colour and sound. The score and imagery is phenomenal. The only fil |

## Anomalies / notes

- Threads dropped for < 5 GT items: 1,010. These threads
  entered the adjudication scope on He's OWN links (>= 5 in-catalogue reply tconsts in the
  `processed` cell); after adjudication corrected those links some fall below the bar. That is the
  ruler doing its job, not a leak -- but it means the ruler's thread count is not a fixed fraction
  of the 35,982.
- Out-of-catalogue loss is the disclosed recency truncation (corpus 2019-22, ML-25M ends late
  2019); reported above per link source rather than hidden. It is strongly asymmetric between the
  two link sources -- 92.7% of
  rules-tier links are in catalogue vs 69.5% of model
  links -- which is expected rather than alarming: the rules tier auto-accepts unambiguous
  high-vote classics, so everything hard, new, foreign or TV-shaped was DEFERred to the model.
- 9,349 DEFER mentions (17.6%)
  were adjudicated NONE -- i.e. the model rejected He's link outright rather than repairing it.
  That is the intended behaviour on the population where He's raw precision was 0.572.
- Ground truth is entity-anchored: it can only contain titles someone actually named in a reply.

Raw counts: `data/reddit_he/ruler/RULER_STATS.json`. Ruler: `data/reddit_he/ruler/ruler_v1.csv.gz`.
Script: `scripts/reddit/assemble_ruler.py`.
