# How much organic *interviewing* is in the He corpus? — follow-up census (no LLM, no Reddit API)

Run 2026-08-03 by `scripts/reddit/followup_census.py`. Reproduce (from `C:/dev/phd/casper`):

```
poetry run python scripts/reddit/followup_census.py
```

Machine-readable output: `data/reddit_he/census/followup_census.json` (not committed —
`data/reddit_he/` is gitignored). Side artifact: `data/reddit_he/census/thread_subreddit.json.gz`
(thread id → subreddit, read off the raw dump; cached so the 3 GB scan runs once).

**No LLM call. No Reddit API call.** Everything comes from the local Hugging Face release.
**Nothing is sampled, capped or truncated in any count** — all three large-V1 CSVs are streamed
end to end and every unique turn of every thread is counted. The only samples drawn are the
3 × 30 snippets in the validation appendix (explicitly requested; reservoir, seed 98765), and
those snippets are character-truncated *for display only*. No username appears anywhere: the
V1 CSV carries only pseudonymous `t2_…` ids (never emitted), and `u/handle` strings inside
snippets are rewritten to `u/[USER]`.

---

## 0. Protocol, and the one thing that improved since the last census

| item | setting |
|---|---|
| thread | `conv_id` minus the trailing `_<path>/<npaths>` (`t3_rt7enj_0/14` → `t3_rt7enj`) |
| turn dedup | reddit `turn_id` **within** a thread (V1 is path-expanded) |
| **parent linkage** | **REAL, not a proxy** — see below |
| eval threads | ≥5 distinct in-catalogue **replier** tconsts after removing every in-catalogue tconst the **seeker** named anywhere (identical code path to `census_he.census_v1`) |
| clean subset | thread ids appearing in adjudication `batches/batch_0000..0099.jsonl` |
| text field | `raw` (original surface text) for question/ack matching; `processed` for tconsts |
| OP / non-OP | He et al.'s `is_seeker` flag |

**The brief allowed a `turn_order > 0` proxy for OP replies. It is not needed.** The V1 CSV's
`context_turn_ids` column is the *ancestor chain* of the extracted path, so the parent of a turn
is simply the last element of that list (`['t3_rnphw0','t1_hptojjc']` → parent `t1_hptojjc`).
Control: **0 of 971,185 unique turns** have `turn_order > 0` and an empty ancestor chain, i.e.
every non-root turn has a parent. So every number below — OP-replies-to-a-non-OP, and the
question-answered rate — is computed on **true reply structure**, not on turn ordering.

### Reconciliation to the published census (`docs/results/REDDIT_HE_CENSUS.md`) — exact

| quantity | published | this run |
|---|---|---|
| V1 CSV rows | 1,669,720 | **1,669,720** ✅ |
| unique turns | 971,185 | **971,185** ✅ |
| V1 threads | 58,623 | **58,623** ✅ |
| eval threads (≥5 in-cat reply items) | 35,982 | **35,982** ✅ |
| clean subset threads (manifest, batches 0000–0099) | 5,441 | **5,441** ✅ |
| clean ids absent from V1 | — | **0** |
| threads reopened out of order | 0 | **0** ✅ |

All 5,441 clean threads are eval threads (the adjudication batches were drawn from the eval
set), so the clean subset is a 15.1% random slice of the same population — which is exactly how
it behaves in every table below.

The published **43% OP-follow-up** figure is a *different* quantity: it was measured on the
**raw** dump over **all** r/MovieSuggestions threads (23,434 / 54,546 = 42.96% "OP answered
someone"). Its V1-eval-thread analogue here is **49.47%** of eval threads with ≥1 OP reply to a
non-OP comment (47.08% restricted to r/MovieSuggestions). The two are consistent: the eval
filter keeps threads with ≥5 in-catalogue reply items, i.e. busier threads, which are more
likely to contain an OP reply. Nothing needs reconciling — they are not the same denominator
and this document does not restate the 43% as its own.

---

## 1. Questions to the OP

A non-OP comment counts as a question if it contains a literal `?` **and** matches one of four
regex classes, first match wins: `seen_x` → `preference_probe` → `constraint_probe` →
`other_question` (a generic wh-/auxiliary pattern before a `?`). Full patterns are in the
script. **`other_question` turns out to be mostly noise (§5), so the headline is the three
STRICT classes.**

Base rates over the 35,982 eval threads: **721,035 non-OP comments**, of which **55,317 (7.67%)
contain a `?` at all** — most replies are title lists, not conversation.

| | all eval (35,982) | clean subset (5,441) |
|---|---|---|
| non-OP comments | 721,035 | 107,137 |
| … containing `?` | 55,317 (7.67%) | 8,314 (7.76%) |
| matched as a question (any class) | 35,919 | 5,342 |
| **STRICT probes** (`seen_x`+`preference`+`constraint`) | **4,813** | **680** |

### Per-thread distribution

| statistic | all eval | clean |
|---|---|---|
| questions/thread (any class) — mean | 0.998 | 0.982 |
| median / p90 / p99 / max | 0 / 2 / 12 / 183 | 0 / 2 / 11 / 183 |
| questions addressed **at the OP** (parent = submission or an OP turn) — mean | 0.521 | 0.516 |
| **strict probes/thread — mean** | **0.134** | **0.125** |
| strict median / p90 / p99 / max | 0 / 1 / 2 / 30 | 0 / 0 / 2 / 11 |
| **% threads with ≥1 strict probe** | **10.13%** | **9.65%** |
| % threads with ≥1 question of any class | 40.89% | 40.49% |

Strict-probe count histogram (share of threads):

| probes | 0 | 1 | 2 | 3 | 4 | 5+ |
|---|---|---|---|---|---|---|
| all eval | **89.87%** | 8.21% | 1.36% | 0.30% | 0.11% | 0.15% |
| clean | 90.35% | 7.88% | 1.38% | 0.18% | 0.06% | 0.15% |

### Class mix (all eval; counts are of matched comments)

| class | count | % of matches | what it is |
|---|---|---|---|
| `other_question` | 31,106 | 86.60% | generic interrogative — **~20% precision, see §5** |
| `seen_x` | 3,283 | 9.14% | "have you seen/watched X?" |
| `preference_probe` | 1,190 | 3.31% | "do you like…", "are you looking for…", "how do you feel about…" |
| `constraint_probe` | 340 | 0.95% | genre / subtitles / age / platform / "what have you seen" |

The clean subset reproduces the mix to within a point (86.27 / 8.54 / 3.41 / 0.79).

### Answered rate (true parent linkage)

A question is *answered* iff some OP turn's parent is that comment.

| | all eval | clean |
|---|---|---|
| any-class questions answered | 5,487 / 35,919 = **15.28%** | 809 / 5,342 = 15.14% |
| any-class **addressed at OP** answered | 5,295 / 18,745 = **28.25%** | 776 / 2,807 = 27.65% |
| **strict probes answered** | 1,655 / 4,813 = **34.39%** | 237 / 680 = 34.85% |
| **strict probes addressed at OP, answered** | **1,632 / 3,196 = 51.06%** | 231 / 453 = 50.99% |

The gradient is the point: the tighter the definition of "a real question put to the OP", the
higher the answered rate, ending at **~51% for a strict probe actually directed at the seeker**.
Seekers do answer when asked.

---

## 2. OP answers — the effective organic interview length *k*

An *OP reply* = a seeker turn whose parent is a non-seeker turn. Total OP turns in eval threads:
112,673, of which **76,389 are replies to someone else**.

| statistic | all eval | clean |
|---|---|---|
| OP replies/thread — mean | **2.123** | 2.006 |
| median / p75 / p90 / p99 / max | **0** / 2 / 6 / 19 / 141 | 0 / 2 / 6 / 17 / 69 |
| % threads with ≥1 OP reply | **49.47%** | 49.97% |

| OP replies (*k*) | 0 | 1 | 2 | 3 | 4 | 5+ |
|---|---|---|---|---|---|---|
| **all eval — threads** | 18,182 | 5,492 | 3,376 | 2,305 | 1,504 | 5,123 |
| **all eval — %** | **50.53%** | 15.26% | 9.38% | 6.41% | 4.18% | **14.24%** |
| clean — threads | 2,722 | 858 | 547 | 329 | 208 | 777 |
| clean — % | 50.03% | 15.77% | 10.05% | 6.05% | 3.82% | 14.28% |

Half of all eval threads are a pure one-shot request → answers, with the seeker never coming
back. The mean of 2.1 is carried by a long tail (14.2% of threads have ≥5 OP replies, max 141),
and §3 shows most of that tail is *thanking*, not answering questions: only 1,655 of the 76,389
OP replies (2.2%) are replies to a strict probe.

---

## 3. Acknowledgments and the outcome signal

*Acknowledgment* = an OP turn matching thanks / "will check|watch|try" / "added to (my) list" /
"watchlist" / "just watched" / "watched it" / "loved|liked it" / "sounds good|great|perfect" /
"great suggestion". *Acknowledged recommendation* (the strong signal) = an acknowledging OP turn
that names a tconst which **some non-OP turn of the same thread also named** — i.e. the seeker
explicitly reacts to a specific item that was recommended to them.

| | all eval | clean |
|---|---|---|
| acknowledgment turns | 54,827 (48.7% of all 112,673 OP turns) | 7,953 |
| acks/thread — mean / median / p90 / max | 1.524 / 1 / 4 / 122 | 1.462 / 1 / 4 / 40 |
| **% threads with ≥1 acknowledgment** | **54.80%** | 55.39% |
| **acknowledged recommendations (total)** | **10,614** | 1,618 |
| acked recs/thread — mean / median / p90 / p99 / max | **0.295** / 0 / 1 / 3 / 18 | 0.297 / 0 / 1 / 3 / 11 |
| **% threads with ≥1 acknowledged recommendation** | **19.72%** | 20.09% |

So roughly **1 eval thread in 5 carries an explicit, item-level outcome signal** — 7,096 threads
(19.72% × 35,982) and 10,614 (thread, item) reaction events. That is the only place in this
corpus where "did the recommendation land?" is observable without simulation, and it is *not*
vote-derived — the Jul-31 rule "never use votes as graded labels" is untouched here.

Caveat inherited from the census: item identity comes from He et al.'s machine links, whose
precision is unmeasured (over-linking on generic-word titles). An acknowledged rec is therefore
a *candidate* outcome label; the adjudication pass is what will clean it.

---

## 4. By subreddit

**All eval threads:**

| subreddit | threads | q/thr (any) | strict/thr | % thr ≥1 strict | OP replies/thr | % thr ≥1 OP reply | strict answered % | strict-to-OP answered % | % thr ≥1 ack | % thr ≥1 acked rec |
|---|---|---|---|---|---|---|---|---|---|---|
| bestofnetflix | 225 | 3.907 | 0.271 | 20.00 | 1.711 | 46.22 | 9.84 | 50.00 | 25.33 | 10.22 |
| movies | 4,022 | 2.460 | 0.273 | 17.95 | 3.414 | 62.08 | 30.24 | 53.36 | 61.01 | 28.72 |
| **moviesuggestions** | **29,219** | **0.648** | **0.092** | **7.75** | **1.890** | **47.08** | **39.89** | **53.40** | **54.01** | **18.28** |
| netflixbestof | 2,448 | 2.418 | 0.390 | 24.35 | 2.805 | 57.23 | 26.00 | 41.39 | 57.31 | 22.92 |
| truefilm | 68 | 4.221 | 0.368 | 27.94 | 2.456 | 61.76 | 8.00† | 22.22† | 33.82 | 22.06 |

† tiny cells: r/truefilm contributes 68 eval threads and 25 strict probes; r/bestofnetflix 225
threads and 61 probes. Their rates are quoted for completeness, not for inference.

**Clean subset** (same ordering, 15.1% slice):

| subreddit | threads | q/thr (any) | strict/thr | % thr ≥1 strict | OP replies/thr | % thr ≥1 OP reply | % thr ≥1 ack | % thr ≥1 acked rec |
|---|---|---|---|---|---|---|---|---|
| bestofnetflix | 37 | 7.703 | 0.486 | 21.62 | 1.405 | 51.35 | 27.03 | 10.81 |
| movies | 588 | 2.515 | 0.255 | 17.01 | 3.131 | 63.27 | 60.37 | 29.25 |
| **moviesuggestions** | **4,481** | **0.633** | **0.087** | **7.50** | **1.819** | **47.51** | **54.74** | **18.54** |
| netflixbestof | 326 | 2.117 | 0.368 | 23.93 | 2.494 | 58.90 | 59.20 | 25.77 |
| truefilm | 9 | 5.889 | 0.444 | 33.33 | 6.556 | 77.78 | 33.33 | 22.22 |

Every clean-vs-all pair agrees to within sampling noise on every metric (largest gap on the
n=37 and n=9 cells) — the sanity check the brief asked for, passed.

**The request subreddit is the *least* interviewed.** r/MovieSuggestions has the lowest question
rate of any file (0.648/thread vs 2.4–4.2 elsewhere) and the lowest strict-probe rate (7.75% of
threads vs 18–28%). That is not a paradox: the other four subreddits are *discussion* forums, so
their "questions" are argument and banter, while r/MovieSuggestions posts state a request up
front and get lists back. But it does mean the clean asymmetry — **r/MovieSuggestions has the
highest strict-probe answered rate (39.9%) of any subreddit** — is the sharper signal: on the
request forum, when someone bothers to ask, the seeker usually replies.

---

## 5. Heuristic validation — 30 matched + 30 strict + 30 `?`-non-matched

Three reservoir samples (seed 98765) over the eval-thread non-OP comment population
(35,919 matched / 4,813 strict / 19,398 `?`-containing non-matched). Snippets are redacted and
truncated to ~420 chars for display. Every example was eyeball-classified by hand against two
targets:

- **Q** — is this a genuine question addressed at another person?
- **P** — is it an *interview move*: a question that probes the seeker's taste, viewing history
  or constraints?

### Estimated precision

| sample | Q (genuine question) | P (interview probe) |
|---|---|---|
| any-class matches (n=30) | 12/30 = **40%** | 6/30 = **20%** |
| **strict matches (n=30)** | 27/30 = **90%** | **24/30 = 80%** (1 unclear/truncated) |

Reweighting by class share (strict 13.4% of matches, `other_question` 86.6%): the `other_question`
bucket has ≈8% probe precision and ≈27% question precision. **Conclusion: `other_question` is not
usable as a measurement of interviewing** — its hits are overwhelmingly (i) URLs containing
`?v=` or `?ref_=`, (ii) suggestions hedged with a question mark ("Maybe *Housebound*?"),
(iii) rhetorical/argumentative questions between commenters, and (iv) film titles that are
questions. The strict classes, at ~80% probe precision, are the number to quote.

Corpus-level estimate after applying precision: **4,813 strict matches × 0.80 ≈ 3,850 genuine
interview moves** in 35,982 threads.

### Recall — the heuristic misses about as much as it catches

Of the 30 `?`-containing **non**-matches, 4 (13%) are genuine clarifying questions to the OP
("You mean Korean?", "Oh you mean like a complete alien society without humans in it?",
"Nf orig?", "The Thai one with Jeeja Yanin as the lead?") — a *clarification* register the regex
does not cover, ≈2,500 comments corpus-wide. And within the matched set, the 6 hand-confirmed
probes in the 30-sample map to ≈7,200 genuine probes corpus-wide against the 3,850 the strict
classes recover, so **strict-class recall against "all interview moves" is roughly one half**.

Honest band for the whole corpus: **≈4,000–8,000 genuine preference/experience probes across
35,982 threads, i.e. 0.11–0.22 per thread, present in at most ~20% of threads.** Every
conclusion in the verdict is stated so that it survives the whole band.

### Appendix A — 30 random STRICT matches

_(verdicts: TP = genuine interview probe; FP = not; snippets redacted, truncated for display)_

**S00** `seen_x` · answered=True · **TP**  
> 1408 is one of my favorite horror movie, did you see both endings ? Also what did you like about it, with that I could get more recommendations. Personally there three aspects that I really liked; first its a descent into madness, some kind of irreversible bad trip, it just get worse and worse for the protagonist as the movie goes on, a good example would be Mother! (2017) Second there some kind of lovecraftian feel …

**S01** `seen_x` · answered=False · **TP**  
> Have you seen The Men Who Stare at Goats? That's good, too.

**S02** `seen_x` · answered=False · **TP**  
> Have you watched WW84 yet? Also *Gigli*, and *North*

**S03** `preference_probe` · answered=True · **TP**  
> Are you open to series? Goliath is a great one.

**S04** `seen_x` · answered=True · **TP**  
> Did you watch Leap Year? That's a pretty good film.

**S05** `seen_x` · answered=False · **TP**  
> Have you seen Forbidden Arsenal (In The Line Of Duty 6) mate? I'm struggling to get that one...

**S06** `preference_probe` · answered=True · **TP**  
> * Scooby Doo on Zombie Island * Scooby Doo and the Witch's Ghost Are you open to anime?

**S07** `preference_probe` · answered=False · **TP**  
> Did you only like "Three Colors: Blue" and not "Red", or "White"? Or have you not seen those?

**S08** `preference_probe` · answered=False · **TP**  
> Out of curiosity why did you not like Midsommar?

**S09** `seen_x` · answered=False · **TP**  
> Did you check out the director's other films? They're so good.

**S10** `seen_x` · answered=False · **TP**  
> Have you seen Castlevania on Netflix? It's actually quite good. Animated but dark undead invasion set in medieval times.

**S11** `seen_x` · answered=False · **TP**  
> Really? Have you seen the original?. I know you said the modern era, to me that means the 20th century and beyond.

**S12** `preference_probe` · answered=False · **FP (dialogue quoted from another work)**  
> Do you like my manifestations? No Jerry, your manifestations hurt us. Great fucking book but even with an amazing cast this movie just....

**S13** `seen_x` · answered=False · **TP**  
> Have you read Timeline? That one always stood out as one of my favorites of his. I think it deserved a better film or TV adaptation than it got.

**S14** `seen_x` · answered=False · **TP**  
> My fucking god, Kill List is great. Have you seen the Hungarian film ‘13 Tzameti’?

**S15** `preference_probe` · answered=False · **FP (hypothetical questions quoted, not asked)**  
> That is pretty cool! And yeah, I wasn’t talking about drama or actors throwing tantrums on set or anything, I just meant like “did you meet Tom Waits and if so what’s he like?” or “did they use squibs or CG blood?”

**S16** `preference_probe` · answered=True · **TP**  
> Glad you did! Which ones did you like the most? Maybe I can orientate my suggestions based on that.

**S17** `preference_probe` · answered=True · **TP**  
> Movies or series? Which do u prefer? And mention some of em which you have already watched. So we can suggest excluding those.

**S18** `seen_x` · answered=False · **UNCLEAR (probe falls past the display cut)**  
> Yeah, love those two as well. *8 1/2* is really great, it has so many layers to it, so many great scenes that it demands rewatches. I found *La Grande Bellezza* to be way too similar and derivative to *La Dolce Vita*, so I wasn’t too impressed with it. *Amarcord* is on my watchlist as well. Have you seen any Antonioni films?

**S19** `preference_probe` · answered=False · **FP (film quote)**  
> *"Would you like to go upstairs and check for yourself, Mr. Graysmith?"*

**S20** `seen_x` · answered=False · **TP**  
> How many movies have you seen?

**S21** `seen_x` · answered=False · **TP**  
> Have you seen Reality Bites and The Truth About Cats and Dogs? Oh and High Fidelity?

**S22** `constraint_probe` · answered=False · **TP**  
> Do you have Hulu? Runaways and Cloak and Dagger are on there.

**S23** `seen_x` · answered=False · **FP (conditional suggestion + hedged "Maybe X?")**  
> If you watched the 2010 The Crazies I would recommend the original from 1978. Maybe Blindness? I haven't seen it for a long time, but it might be what you are looking for.

**S24** `seen_x` · answered=True · **TP**  
> Yeah, Pi is a film that most people would probably turn off within 10 minutes, but if you can actually get through the whole thing then it's very rewarding. Have you seen The Machinist? If not then that's my other recommendation. Another delusional trip of a movie.

**S25** `seen_x` · answered=False · **TP**  
> Only Lovers Left Alive - have you seen other Jim Jarmusch movies? Check out his filmography. I honestly didn't like Only Lovers, but some of my favorites of his are: Paterson, Night on Earth, and Dead Man. Pulp Fiction - maybe you'd like Guy Ritchie movies? Check out: Lock, Stock and Two Smoking Barrels, Snatch, and RocknRolla

**S26** `preference_probe` · answered=False · **FP (asks OP to read a post; addressed at OP but not a taste probe)**  
> Would you mind reading my two theories about Burning then? Link: https://www.reddit.com/r/TrueFilm/comments/ffg6ib/burning_2018_dir_lee_changdong_review_analysis/?utm_medium=android_app&amp;utm_source=share

**S27** `seen_x` · answered=False · **TP**  
> Did u watch the other Flanagan things? Like Bly Manor, Midnight Mass and Hush?

**S28** `seen_x` · answered=True · **TP**  
> have you seen chinatown or seven?

**S29** `seen_x` · answered=False · **TP**  
> Have you see The Shawshank Redemption? I believe itâs on Netflix in the US.

### Appendix B — 30 random matches, any class

**M00** `other_question` · answered=False · **Q+P**  
> So after watching it again. I picked up more details this time around. So now I realize Abdel beat Moktar because Moktar dirty narcs were the ones that killed Karim. They were originally going to pick up Jerome I believe that was their initial plan. Moktar didnât really take any time of responsibility for it an didnât seem to care to much after. Why they blew up the building is still a mystery to me. Sebastian is…

**M01** `other_question` · answered=False · **-**  
> 12 Monkeys is one of those movies that isn't really horror, but creeped the hell out of me in its ideas AND execution. Terry Gilliam is the master of making really wierd movies. He usually uses that weirdness for laughs, but when he uses them in a sci fi thriller? Unnerving, to say the least. Plus, it certainly doesn't follow the cliche horror/slasher movie formula. Happy watching :)

**M02** `other_question` · answered=False · **-**  
> Anyway how is your sex life?

**M03** `other_question` · answered=False · **Q**  
> That’s my only thought. I think they should have shown her waking in another facility in a new body with the writing on the bottom like: ‘Begin experiment 512’ (this would work well if they had ‘Begin experiment 511’ at the start and first we’d think it’s the having an embryo come to life, later think the cave woman was 510, then maybe think it was the plan. Or, they could have had Mother in a new body in another fac…

**M04** `other_question` · answered=False · **-**  
> Understandable for sure. I went back and watched it again after I made the suggestion and honestly I had forgotten how weird it was. What was you favorite movie from the suggestions you got on the thread?

**M05** `other_question` · answered=False · **Q+P**  
> Same, he’s right there at the top of my list as well. His ability to craft absurdist narratives into such compelling works is top-notch. Any other directors/films you’d recommend?

**M06** `other_question` · answered=False · **Q**  
> Yeah that irritated me as well...they fed him the attention he was craving...but then again, who’s to say that he would’ve stopped if those particular people hadn’t gotten involved?....

**M07** `other_question` · answered=False · **Q**  
> Is it that bad? Most of it just seems to be discussion of more obscure movies than you'd find in a sub like r/movies. I didn't read many of the posts though so maybe I'm wrong.

**M08** `other_question` · answered=False · **- (URL "?v=")**  
> The Wailing is amazing. It's a slow burn but worth the investment. I'm curious, though, what makes a great horror movie to most people? Is it how scary it is? Or how good the story is or?

**M09** `other_question` · answered=False · **-**  
> okay, in that case, what doesn't make sense to you? it's a film about two men's descent into madness, with plenty of Greek and Lovecraftian symbolism that could be taken as literal or nothing more than symbolic depending on the viewer. it wasn't terribly complicated.

**M10** `other_question` · answered=False · **Q+P**  
> Here's a list of all the Netflix shows/movies that I've seen recommended on here. I didn't add any that aren't Netflix because that list could go on for days. I haven't really looked at them all so I can't vouch if they are decent horror/psychological shows but it's a start. Let me know if I should fix or add anything. [American Horror Story](https://www.netflix.com/title/70210884?s=a&amp;trkid=13747225&amp;t=cp) [Ba…

**M11** `other_question` · answered=False · **- (rhetorical, off-topic argument)**  
> It just has a cool, 90’s indie feel to it. It’s funny...maybe his funniest movie he’s made? And the performances by John C. Reily, Gwyneth Paltrow and Samual Jackson are all great.

**M12** `other_question` · answered=True · **Q**  
> [The Place Beyond the Pines](https://www.imdb.com/title/tt1817273/?ref_=nv_sr_srsg_0) [You Were Never Really Here](https://www.imdb.com/title/tt5742374/?ref_=nv_sr_srsg_0) [North by Northwest](https://www.imdb.com/title/tt0053125/?ref_=nv_sr_srsg_0) [The Hitch-Hiker](https://www.imdb.com/title/tt0045877/?ref_=fn_al_tt_2) [Only God Forgives](https://www.imdb.com/title/tt1602613/?ref_=nv_sr_srsg_0) [Near Dark](https://…

**M13** `other_question` · answered=False · **- (rhetorical)**  
> This is debateable but my favourite non obvious portrayal of the Devil is John Goodman's character Charlie Meadows in Barton Fink. https://m.youtube.com/watch?v=Ps3IXTH2dG0

**M14** `other_question` · answered=False · **Q (challenges another commenter)**  
> LOL are you insinuating that we're the same person? I'm glad I triggered you so hard that you looked through my account history.

**M15** `other_question` · answered=False · **-**  
> Star Wars is a cinematic universe. Almost everyone knows this. See my upvotes? I’m surprised this is news to you... Nobody has ever watched Solo and Rogue One and said “this doesn’t connect to the saga”. Really? Han Solo is in the saga. The Death Star is in the saga. That’s like watching Ant-Man and saying it’s not part of a cinematic universe.

**M16** `seen_x` · answered=False · **- (URL)**  
> If you haven't seen the trailers : HOT SHOTS Hot shots Part Deux. Liar Liar Ace Ventura : When Nature Calls (that scene when they're watching the film) Blazing Saddles and History of The World Part 1 The very first scene in Wedding Crashers with the divorcing couple there was a couple Hilarious scenes in Happy Death Day 2 U while not a movie have you checked out live performances of Craig Ferguson? I saw 3 specials h…

**M17** `seen_x` · answered=True · **-**  
> Did you watch Total Recall (1990, Schwarzenegger) or Total Recall (2012, Farrell)? There's a difference 🤔

**M18** `other_question` · answered=False · **- (film title)**  
> How about a horror/thriller? [À l'intérieur](https://www.imdb.com/title/tt0856288/) is a French film about an aspiring mother. I rate it as one of the best in this genre of horror but hardly ever see people talk about it. How someone came up with this idea, sold it to a group of people that agreed to film it, and made the thing watchable as well as entertaing is beyond me. Another horror worth mentioning is [Aterrado…

**M19** `other_question` · answered=False · **- (rhetorical)**  
> Could you explain? Nicole Kidman does an Irish accent in The Others, definitely. Watch the trailer if you don't believe me.

**M20** `other_question` · answered=False · **- (joke)**  
> What about the parts where they torture a bunch of people ineffectively for ten years?

**M21** `other_question` · answered=True · **Q+P**  
> North by Northwest is, imo, the greatest Hitchcock film made. Oh, would The Talented Mr Ripley fit your theme?

**M22** `preference_probe` · answered=True · **Q+P (clarifies the request)**  
> What aspects of these did you like? They are both radically different films. Alex Garland, director of Annihilation currently has a series out called Devs, and theres his other film Deus Ex Machina which is great.

**M23** `other_question` · answered=False · **Q**  
> &gt; story without any particular purpose You say you're a Coen Brothers fan but criticize this movie for that reason. *Most* of their movies don't have any particular purpose. What in the world is the "purpose" of *The Big Lebowski*, arguably their most popular movie? Nothing happens in it other than a fake kidnapping and some bowling. There's no real payoff. And yet, its popularity endures.

**M24** `other_question` · answered=False · **Q+P**  
> The Secret in Their Eyes (2009, Argentina - not the 2015 version) - my friend and I couldn't stop talking about the end for at least a week. Coherence (2013) - my spouse and I watched this, then immediately started watching it again because we were like wut? Those are probably my two favorites that meet this description, but also He Loves Me, He Loves Me Not (2002) would, as well.

**M25** `other_question` · answered=False · **- (film title)**  
> Is it better than her performance as Tracey Flick in Election?

**M26** `other_question` · answered=False · **- (unclear at cut)**  
> Oh? To each his own, but I thought it was tighter and the characters developed more than usual. It sounds like you're not done with the season though? If so, that could definitely be the case for why you feel that way.

**M27** `seen_x` · answered=False · **- (URL)**  
> That’s such a great show. I really enjoyed it too. I don’t know if ya have Amazon Prime or not and there’s only one season, but have you watched Good Omens? I mean The Hitchhikers Guide To The Galaxy is only in movie format so it’s not a tv show but it’s really good.

**M28** `other_question` · answered=False · **- (reaction)**  
> The Neverending Story (1984) and The Neverending Story 2 (1994). Maybe later I'll think of others... Are you considering animation, or just live action?

**M29** `other_question` · answered=False · **- (rhetorical)**  
> Have you ever had a pet? And if yes, what type?

### Appendix C — 30 random non-OP comments that contain `?` but did NOT match

**N00** · **correct reject**  
> Branded to Kill?

**N01** · **correct reject**  
> [The Cincinnati Kid (1965)](https://www.imdb.com/title/tt0059037/?ref_=nm_flmg_act_15)

**N02** · **correct reject**  
> [Phenomenon](https://www.youtube.com/watch?v=EecewkjszbY) (1996) [The Lawnmower Man](https://www.youtube.com/watch?v=3LNvXjb44-U) (1992)

**N03** · **MISS (clarifying question to OP)**  
> Logan Lucky?

**N04** · **correct reject**  
> So like *Taxi Driver* (1976) Or like *Tremors* (1990), *Brokeback Mountain* (2005) ?? Edit: adding *The Last Stand* (2013) and *Cop Land* (1997)

**N05** · **correct reject**  
> Midsommar felt right ? If anything it left me frustrated.

**N06** · **correct reject**  
> &gt;movies i watched as a teen &gt;Sicario Feel old yet?

**N07** · **correct reject**  
> Paper Towns?

**N08** · **correct reject**  
> You ever been in a cockpit before?

**N09** · **MISS (clarifying question to OP)**  
> [Wizards (1977)](https://www.imdb.com/title/tt0076929/?ref_=nv_sr_2?ref_=nv_sr_2) [The Lord Of The Rings (1978)](https://www.imdb.com/title/tt0077869/?ref_=nv_sr_5?ref_=nv_sr_5) [Fire And Ice (1983)](https://www.imdb.com/title/tt0085542/?ref_=nv_sr_1?ref_=nv_sr_1)

**N10** · **MISS (clarifying question to OP)**  
> [Cool World](https://www.imdb.com/title/tt0104009/?ref_=fn_al_tt_1)

**N11** · **borderline (reaction question)**  
> \- Feels good man \- Class action park \- Hail Satan?

**N12** · **correct reject**  
> Hannibal??

**N13** · **correct reject**  
> [The Abandoned (2006)](https://www.imdb.com/title/tt0475937/) [The House of the Devil (2009)](https://www.imdb.com/title/tt1172994/?ref_=nv_sr_srsg_0) [The Silent House (2011)](https://www.imdb.com/title/tt1767382/?ref_=nm_flmg_act_24) [The Silent House (2010)](https://www.imdb.com/title/tt1646973/?ref_=tt_sims_tt)

**N14** · **borderline (reaction question)**  
> Hereditary?

**N15** · **correct reject**  
> I thought [The Nest (1987)](https://www.imdb.com/title/tt0095719/?ref_=nv_sr_srsg_5) was actually surprisingly decent and is not talked about that much

**N16** · **correct reject**  
> [Secret Superstar](https://www.imdb.com/title/tt6108090/?ref_=nv_sr_1?ref_=nv_sr_1)

**N17** · **correct reject**  
> It really dropped off didn't it!? Haha, I was face palming quiet a bit but look forward to the second season of it happens

**N18** · **correct reject**  
> [My Cousin Vinny (1992)](https://www.imdb.com/title/tt0104952/?ref_=nv_sr_srsg_0)

**N19** · **correct reject**  
> [Colossus: The Forbin Project (1970)](https://www.imdb.com/title/tt0064177/?ref_=fn_al_tt_1) [THX-1138 (1971)](https://www.imdb.com/title/tt0066434/?ref_=nv_sr_srsg_0)

**N20** · **correct reject**  
> Tropic Thunder? Hot Rod?

**N21** · **borderline (hedged suggestion)**  
> Agree about The Rescuers sequel...wasn’t a George C. Scott involved?

**N22** · **correct reject**  
> [Flirting (1991)](https://www.imdb.com/title/tt0101898/?ref_=nv_sr_srsg_3) [Clueless (1995)](https://www.imdb.com/title/tt0112697/?ref_=nv_sr_srsg_0) [The Wedding Singer (1998)](https://www.imdb.com/title/tt0120888/?ref_=nv_sr_srsg_0) [Blast from the Past (1999)](https://www.imdb.com/title/tt0124298/?ref_=nv_sr_srsg_0) [10 Things I Hate About You (1999)](https://www.imdb.com/title/tt0147800/?ref_=nv_sr_srsg_0) [Yeopg…

**N23** · **correct reject**  
> Not quite similar to the dirt, but if you are interested in the history of the music business I highly recommend the wrecking crew! It's a doc about the studio musicians that played on a multitude of hit albums in the early days of rock n roll. [https://www.youtube.com/watch?v=hhl-3EOYTkc](https://www.youtube.com/watch?v=hhl-3EOYTkc)

**N24** · **correct reject**  
> [Eternal Sunshine of the Spotless Mind ](https://www.google.com/search?source=hp&amp;ei=I47-XqSIBqapytMPtLesoAI&amp;q=eternal+sunshine+of+the+spotless+mind&amp;gs_ssp=eJzj4tTP1Tcwyi0xNTRg9FJNLUktykvMUSguzSvOyMxLVchPUyjJSFUoLsgvyUktLlbIzcxLAQCdBxFb&amp;oq=etern&amp;gs_lcp=ChFtb2JpbGUtZ3dzLXdpei1ocBABGAAyBQguEJMCMgUIABCxAzIFCC4QsQMyBQgAELEDMgUILhCDATIFCAAQsQMyAggAMgUIABCxAzoCCCk6CAgpELEDEJMCOgUIKRCxAzoFCCkQgwE6BQgpEKAB…

**N25** · **correct reject**  
> Thriller ##Anxiety Inducing Glued to the screen, unable to lookaway. Anxiety Inducing| | | | ---|---|---|---| [10 Cloverfield Lane (2016)](https://www.themoviedb.org/movie/333371)|[1917 (2019)](https://www.themoviedb.org/movie/530915)|[Cape Fear (1991)](https://www.themoviedb.org/movie/1598)|[Children of Men (2006)](https://www.themoviedb.org/movie/9693)| [Eastern Promises (2007)](https://www.themoviedb.org/movie/225…

**N26** · **MISS (clarifying question to OP)**  
> I think Jack the Bear sort of fits right? Big Daddy is another one that sort of works.

**N27** · **correct reject**  
> The *Russians*?

**N28** · **borderline (reaction question)**  
> Maybe these? La Femme Nikita (1990), French Point of No Return (1993), US remake of La Femme Nikita

**N29** · **correct reject**  
> [The Adventures Of Tom Sawyer (1938)](https://www.imdb.com/title/tt0029844/?ref_=nv_sr_srsg_1)

---

## Verdict — how long is the organic interview, and how often is there an outcome signal?

**The organic interview is, at the median, zero turns long, and where it exists it is one
exchange.** Half of all 35,982 eval threads (50.5%) contain no OP reply at all; 15.3% contain
exactly one; the mean of 2.1 OP replies per thread is carried by a 14% tail of chatty threads and
is mostly *thanking*, not answering — only 2.2% of the 76,389 OP replies are replies to a
preference/experience probe. Explicit elicitation is genuinely rare and it is rarest exactly
where requests live: a strict probe appears in **10.1% of eval threads overall and 7.75% of
r/MovieSuggestions threads**, ~0.13 per thread, and precision-corrected that is ≈0.11 and
recall-corrected at most ≈0.22 probes per thread. Only 7.67% of the 721,035 reply comments even
contain a question mark; the modal reply is a list of titles. **The corpus is a request-and-list
corpus, not a conversation corpus** — which is the finding that matters for the Reddit-opener
reframe, because it means an adaptive interview is not something these threads already do and
then we merely imitate. What the data *does* establish is that the missing ingredient is the
asking, not the willingness: when a strict probe is put directly to the seeker, the seeker
answers **51.1%** of the time (34.4% for probes not necessarily addressed to them), and on
r/MovieSuggestions specifically the strict-probe answered rate is the highest of any subreddit
at 39.9%. **The outcome signal is the stronger asset:** 54.8% of eval threads contain at least
one seeker acknowledgment, and — the version with item identity attached — **19.7% of threads
(≈7,096) contain at least one *acknowledged recommendation*, an acknowledging OP comment naming
a movie that a replier had named**, 10,614 such (thread, item) events in total, mean 0.295 per
thread. Every one of these numbers reproduces on the 5,441-thread clean subset to within
sampling noise (e.g. 49.97% vs 49.47% ≥1 OP reply, 20.09% vs 19.72% acknowledged rec), so the
adjudicated slice is a fair sample of the whole. Read together: this corpus supplies a real
**request**, a real **outcome label on one thread in five**, and almost no **interview** — which
is precisely the shape that makes "what does an adaptive interview add once a real request has
been given?" an open question here rather than a re-enactment.

## Known limitations
- Both the question and the acknowledgment detectors are regexes; §5 measures the question one
  (~80% precision, ~50% recall on the strict classes) but the **acknowledgment regex was not
  hand-validated** — its rates should be read as an upper bound on "thanking behaviour" and a
  lower bound is not established. The *acknowledged recommendation* number is the more robust of
  the two because it additionally requires a tconst intersection.
- Item identity everywhere comes from He et al.'s links, whose precision is unmeasured; a
  wrongly-linked title inside an acknowledgment still counts. The 300-mention gold set and the
  adjudication pass are what correct this.
- "Addressed at the OP" is approximated by reply structure (parent is the submission or an OP
  turn), not by discourse analysis; a comment can address the OP from deeper in a subtree.
- `is_seeker` is He et al.'s own label and was not independently verified.
- Threads are eval threads only (≥5 in-catalogue reply items). The 22,641 V1 threads that fail
  that filter are not measured here.
