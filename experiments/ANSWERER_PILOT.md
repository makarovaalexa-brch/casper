# Answerer v1 -- PILOT report

Judge `gpt-5.4-mini-2026-03-17` temp 0.0, answerer split seed 123, user seed 0. Pilot = 20 users x (1128 tags + 700 attrs + 100 items). Schema: `casper/answerer_schema.json`. NOT the full run.

Users: [700, 2110, 2227, 2276, 3239, 3761, 5609, 8040, 8777, 15131, 15241, 17247, 19019, 22719, 27826, 28973, 64620, 91639, 93314, 102367]

## 1. Parse / schema-validity rate per channel (target >=99%)

Two grades: STRICT = full schema (knowledge+value contract+conf float); CORE = knowledge label valid + value present-iff-knowledge!=no_clue (conf may be omitted).

| channel | cells | returned | strict-valid | strict% | core-valid | core% | missing | hard-invalid |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| concept | 22560 | 22560 | 19499 | 86.43% | 22553 | 99.97% | 0 | 7 |
| attribute | 14000 | 14000 | 13208 | 94.34% | 14000 | 100.00% | 0 | 0 |
| item | 2000 | 2000 | 1999 | 99.95% | 2000 | 100.00% | 0 | 0 |

Failure-reason census (all returned cells): {'bad_conf': 3847, 'bad_knowledge': 7}

The dominant deviation is SYSTEMATIC, not random: the judge omits `conf` on a subset of `no_clue` cells (`{"q":i,"knowledge":"no_clue"}` with no value, no conf). The knowledge/value contract is intact on those cells; only the optional-for-analysis conf float is missing. Fix for the full run: declare conf optional-on-no_clue in the schema, or one prompt line ('always include conf').

HARD failures (7 listed = all, unless noted):

- [concept] u2276 `hunting`: bad_knowledge:meh
- [concept] u3239 `happy ending`: bad_knowledge:liked
- [concept] u5609 `happy ending`: bad_knowledge:meh
- [concept] u8777 `happy ending`: bad_knowledge:liked
- [concept] u15241 `feel good movie`: bad_knowledge:liked
- [concept] u15241 `feel-good`: bad_knowledge:liked
- [concept] u15241 `happy ending`: bad_knowledge:liked

## 2. Degenerate columns (>=95% of users share one knowledge label) -- FLAGGED, NOT pruned

### concept: 68 degenerate columns (of 1128 with >=10 users)

By label: {'know_well': 11, 'no_clue': 29, 'rough_idea': 28}

| entity | label | share | n | example |
|---|---|--:|--:|---|
| adventure | know_well | 100% | 20 |  |
| animated | know_well | 100% | 20 |  |
| comedy | know_well | 100% | 20 |  |
| amy smart | no_clue | 100% | 20 |  |
| author:alan moore | no_clue | 100% | 20 |  |
| author:neil gaiman | no_clue | 100% | 20 |  |
| c.s. lewis | no_clue | 100% | 20 |  |
| carrie-anne moss | no_clue | 100% | 20 |  |
| chris tucker | no_clue | 100% | 20 |  |
| easily confused with other movie(s) (title) | no_clue | 100% | 20 | awkward |
| free to download | no_clue | 100% | 20 |  |
| gypsy accent | no_clue | 100% | 20 |  |
| hitchcock | no_clue | 100% | 20 |  |
| incest | no_clue | 100% | 20 |  |
| indians | no_clue | 100% | 20 |  |
| liv tyler | no_clue | 100% | 20 |  |
| long | no_clue | 100% | 20 |  |
| lynch | no_clue | 100% | 20 |  |
| neil gaiman | no_clue | 100% | 20 |  |
| heroine in tight suit | rough_idea | 100% | 20 |  |
| swashbuckler | rough_idea | 100% | 20 |  |
| television | rough_idea | 100% | 20 |  |
| texas | rough_idea | 100% | 20 |  |
| too long | rough_idea | 100% | 20 |  |
| too short | rough_idea | 100% | 20 |  |
| underwater | rough_idea | 100% | 20 |  |
| utopia | rough_idea | 100% | 20 |  |
| vampire human love | rough_idea | 100% | 20 |  |
| very good | rough_idea | 100% | 20 |  |
| action | know_well | 95% | 20 |  |
| action packed | know_well | 95% | 20 |  |
| alien | know_well | 95% | 20 |  |
| assassin | know_well | 95% | 20 |  |
| based on a book | know_well | 95% | 20 |  |
| based on book | know_well | 95% | 20 |  |
| books | know_well | 95% | 20 |  |
| car chase | know_well | 95% | 20 |  |
| best of 2005 | no_clue | 95% | 20 | awkward |
| cool | no_clue | 95% | 20 |  |
| finnish | no_clue | 95% | 20 |  |
| ...(+28 more) | | | | |

### attribute: 201 degenerate columns (of 700 with >=10 users)

By label: {'know_well': 171, 'no_clue': 10, 'rough_idea': 20}

| entity | label | share | n | example |
|---|---|--:|--:|---|
| Steven Spielberg | know_well | 100% | 20 | director |
| Robert Zemeckis | know_well | 100% | 20 | director |
| James Cameron | know_well | 100% | 20 | director |
| John Carpenter | know_well | 100% | 20 | director |
| Harrison Ford | know_well | 100% | 20 | actor |
| Tom Hanks | know_well | 100% | 20 | actor |
| Brad Pitt | know_well | 100% | 20 | actor |
| Bruce Willis | know_well | 100% | 20 | actor |
| Robert De Niro | know_well | 100% | 20 | actor |
| Kevin Spacey | know_well | 100% | 20 | actor |
| Matt Damon | know_well | 100% | 20 | actor |
| Morgan Freeman | know_well | 100% | 20 | actor |
| Tom Cruise | know_well | 100% | 20 | actor |
| Leonardo DiCaprio | know_well | 100% | 20 | actor |
| Samuel L. Jackson | know_well | 100% | 20 | actor |
| Ian McKellen | know_well | 100% | 20 | actor |
| Orlando Bloom | know_well | 100% | 20 | actor |
| Al Pacino | know_well | 100% | 20 | actor |
| Jack Nicholson | know_well | 100% | 20 | actor |
| Johnny Depp | know_well | 100% | 20 | actor |
| Robin Williams | know_well | 100% | 20 | actor |
| Keanu Reeves | know_well | 100% | 20 | actor |
| Mel Gibson | know_well | 100% | 20 | actor |
| Sean Connery | know_well | 100% | 20 | actor |
| Christian Bale | know_well | 100% | 20 | actor |
| Arnold Schwarzenegger | know_well | 100% | 20 | actor |
| Sigourney Weaver | know_well | 100% | 20 | actor |
| Uma Thurman | know_well | 100% | 20 | actor |
| Elijah Wood | know_well | 100% | 20 | actor |
| Gene Hackman | know_well | 100% | 20 | actor |
| John Travolta | know_well | 100% | 20 | actor |
| Edward Norton | know_well | 100% | 20 | actor |
| Jodie Foster | know_well | 100% | 20 | actor |
| Anthony Hopkins | know_well | 100% | 20 | actor |
| Julianne Moore | know_well | 100% | 20 | actor |
| Diane Keaton | know_well | 100% | 20 | actor |
| Bill Murray | know_well | 100% | 20 | actor |
| Jim Carrey | know_well | 100% | 20 | actor |
| Michael Caine | know_well | 100% | 20 | actor |
| Nicolas Cage | know_well | 100% | 20 | actor |
| ...(+161 more) | | | | |

## 3. Value agreement where checkable

### 3a. ITEMS -- LLM value vs masked real rating (mapped to 4-level)

n=632 masked-and-answered cells | exact=0.407 | adjacent=0.918 | corr=0.165

Marginals -- LLM: hated 0.000, meh 0.059, liked 0.937, loved 0.005 | TRUTH: hated 0.085, meh 0.190, liked 0.400, loved 0.324

FLAG: the LLM value marginal COLLAPSES to 'liked' (~0.94) while real ratings spread over all four levels -- the judge hedges to the modal answer. The adjacent-match rate is inflated by this collapse ('liked' is adjacent to both 'meh' and 'loved'); the correlation is the honest number. Author options for the full run: (a) anchor the 4 levels with explicit frequency guidance / examples in the prompt; (b) have the judge predict a 0.5-5.0 star value (the old masked-pass MAE 0.70 protocol showed real variance) and bin it to the scale mechanically; (c) accept and document the compression. This is the main schema risk the pilot surfaced.

### 3b. TAGS / ATTRIBUTES -- LLM value vs data-side member aggregate (user rated >=3 members)

| channel | checkable cells | answered pairs | exact | adjacent | corr |
|---|--:|--:|--:|--:|--:|
| concept(tag) | 8832 | 8256 | 0.535 | 0.947 | 0.140 |
| attribute | 648 | 642 | 0.478 | 0.978 | 0.212 |

(aggregate->scale: >=4.25 loved, >=3.25 liked, >=2.25 meh, else hated.)

## 4. Consistency vs OLD cached can_answer (knowledge {know_well,rough_idea}->yes, no_clue->no)

Two mappings shown: SPEC = {know_well,rough_idea}->yes (pre-registered); STRICT = know_well->yes only (rough_idea->no). Old maybe->no (primary refuse convention).

| channel | mapping | overlap cells | %agree | kappa |
|---|---|--:|--:|--:|
| item | SPEC | 1288 | 46.6% | 0.000 |
| item | STRICT | 1288 | 58.3% | 0.197 |
| concept | SPEC | 3994 | 90.7% | 0.300 |
| concept | STRICT | 3994 | 48.4% | 0.071 |

Item-overlap diagnosis: the overlap cells are top-1000 (famous) items. On RATED overlap cells (ground truth = answerable) new judge yes 110/110, old judge yes 110/110. On UNRATED overlap cells the new judge marks essentially everything >=rough_idea (recognition of famous films), while the old can_answer asked 'has SEEN it' -- a stricter question. The SPEC-mapping kappa on items is therefore 0 by construction (constant new marginal = all yes), NOT judge noise: 'knows OF a famous film' (rough_idea share 0.22) and 'has seen it' are different constructs. The STRICT mapping (know_well only) partially recovers the old construct.

## 5. Knowledge-label distributions per channel + stratum; rough_idea usage

| channel | n | no_clue | rough_idea | know_well |
|---|--:|--:|--:|--:|
| concept | 22560 | 0.154 | 0.520 | 0.325 |
| attribute | 14000 | 0.068 | 0.404 | 0.527 |
| item | 2000 | 0.001 | 0.149 | 0.851 |

rough_idea marginal usage (is the middle level actually used?):
- overall rough_idea share = 0.459 (17695/38560)

### 5a. Attribute answerability by TYPE (composers/writers expected LOW)

| type | n | no_clue | rough_idea | know_well | can_answer(!=no_clue) |
|---|--:|--:|--:|--:|--:|
| director | 4000 | 0.116 | 0.405 | 0.479 | 0.884 |
| actor | 6000 | 0.018 | 0.315 | 0.667 | 0.982 |
| composer | 1000 | 0.056 | 0.735 | 0.209 | 0.944 |
| writer | 1000 | 0.088 | 0.753 | 0.159 | 0.912 |
| franchise | 2000 | 0.120 | 0.330 | 0.549 | 0.879 |

Lit-pass expectation check: composers/writers ARE the weakest channel, but the signal shows up in `know_well` (composer 0.21, writer 0.16 vs actor 0.67), NOT in `no_clue` -- the judge prefers `rough_idea` ('knows the films, not the name') over a refusal for top-50 famous composers/writers. Ordering know_well: actor > franchise ~ director >> composer > writer, as the lit predicts. Note these are the TOP 50 most-popular composers/writers; the full-run battery is the same 50, so this is the real base rate for the channel.

### 5b. Concept answerability by popularity tercile (answer_rate_prior)

| tercile | n | no_clue | rough_idea | know_well | can_answer |
|---|--:|--:|--:|--:|--:|
| low | 7520 | 0.285 | 0.514 | 0.201 | 0.715 |
| mid | 7520 | 0.117 | 0.557 | 0.326 | 0.883 |
| high | 7520 | 0.061 | 0.490 | 0.448 | 0.938 |

### 5c. Item answerability by popularity tercile (cnt within top-1000)

| tercile | n | no_clue | rough_idea | know_well | can_answer |
|---|--:|--:|--:|--:|--:|
| low | 667 | 0.001 | 0.366 | 0.633 | 0.999 |
| mid | 667 | 0.000 | 0.081 | 0.919 | 1.000 |
| high | 666 | 0.000 | 0.000 | 1.000 | 1.000 |

Base-rate sanity: can_answer should be monotone increasing in popularity within each channel.

## 6. Cost

- measured: 160 calls, 627713 prompt + 738011 completion tokens, **$1.6330** (rates in $0.25/M, out $2.0/M)
- per-call $0.0102 | per-user $0.0816 (1928 Q/user)
- **projected FULL run** (2828 Q/user x 300 users, richer output) = per_user x (2828/1928) x 300 = **$35.93**
- tripwire (3 users): $0.2390; projected pilot $1.59; projected full $35.06

## 7. The awkward-flagged tags (do they behave worse on parse/degeneracy/knowledge?)

(valid% here = STRICT validity; the gap is conf-omission on no_clue cells, which awkward tags hit more often because they are no_clue more often.)

| group | tags | valid% | no_clue | rough_idea | know_well | degenerate cols |
|---|--:|--:|--:|--:|--:|--:|
| awkward | 54 | 76.02% | 0.273 | 0.601 | 0.126 | 8 |
| normal | 1074 | 86.96% | 0.148 | 0.516 | 0.335 | 60 |

Awkward degenerate rate 8/54 = 14.8% vs normal 60/1074 = 5.6%.

## AUTHOR DECISION

- **Parse/validity (CORE contract)**: 99.98% overall (concept 99.97%, attribute 100.00%, item 100.00%). PASS (>=99%). STRICT validity is lower only because the judge omits `conf` on some no_clue cells (systematic; fixable by one prompt line or schema note).
- **Degenerate columns** (evidence for a PRUNE decision, author's call): concept 68, attribute 201. Proposed empirical prune candidates = the all-`no_clue` degenerate columns (near-zero information):
  - concept: 29 all-no_clue columns -> e.g. ['amy smart', 'author:alan moore', 'author:neil gaiman', 'c.s. lewis', 'carrie-anne moss', 'chris tucker', 'easily confused with other movie(s) (title)', 'free to download', 'gypsy accent', 'hitchcock', 'incest', 'indians', 'liv tyler', 'long', 'lynch']
  - attribute: 10 all-no_clue columns -> e.g. ['Norman Ferguson', 'District', 'Wilfred Jackson', 'Hamilton Luske', 'Clyde Geronimi', 'Ben Sharpsteen', 'David Hand', 'Wolfgang Reitherman', 'Ash Brannon', 'Much Ado About Nothing']
- **rough_idea usage**: overall 0.459; the middle level is USED.
- **Consistency vs old judge**: item kappa 0.000, concept kappa 0.300.
- **Value agreement**: items exact 0.407/adj 0.918; tags exact 0.535/adj 0.947; attrs exact 0.478/adj 0.978. **MAIN RISK**: the LLM value marginal collapses to 'liked' (~0.94 on items) -- see 3a; decide (a) prompt anchoring, (b) predict stars then bin, or (c) accept-and-document, BEFORE the full run.
- **Consistency construct note**: item SPEC-kappa 0 is a CONSTRUCT difference (knows-of vs has-seen on famous items), not judge noise -- STRICT mapping gives kappa 0.197, and on rated (ground-truth-answerable) overlap cells both judges are 110/110 correct. If the fold needs the old 'has seen' construct for items, use know_well (not !=no_clue) as the item can_answer rule.
- **Full-run cost estimate** for the signed config (2828 Q x 300 users, richer output) = **$35.93**.
- **Schema issues found**: see the failure list in section 1 (empty = none).
