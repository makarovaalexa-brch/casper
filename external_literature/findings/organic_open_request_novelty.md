# The mined organic open request is PRE-EMPTED — and one cell survives

Research 2026-07-31. Supersedes the optimistic framing recorded that morning in
`docs/STATE.md` and memory `reddit-mined-open-request-opener`.

**It is not "ReDial with extra steps". It is "narrative-driven recommendation with extra steps"** — a
published programme with a decade of history, a CLEF shared task, and a 634k-conversation public dataset.

## The three prior works that own it

**1. Bogers & Koolen — narrative-driven recommendation (NDR).** RecSys 2017 + KaRS@RecSys 2018 + the
INEX/CLEF **Social Book Search Suggestion Track, 2011–2016**. Their RecSys 2017 definition is our
proposal verbatim: *"a narrative description of the aspects of items desired by the users (≥1 sentence)…
**The narrative must describe a open request for recommendations** as opposed to locating a specific
item."* 115,899 LibraryThing threads; ~25,000 such requests on LT alone; mean narrative 527 characters.

They own everything structural **including the outcome signal we do not have.** SBS relevance grades:
suggested = **1**; already in the requester's pre-request catalogue = **0**; **suggestions the requester
subsequently added to their own catalogue = 8**. Topics ship as XML with `<request>` (raw forum prose),
`<examples>`, and `<catalog>` (the requester's full library), alongside 94,656 anonymised profiles /
33M transactions. Metric: **nDCG@10**. 46 runs from 10 institutions in 2016 alone.

**2. Eberhard et al., IUI 2019** — the exact artifact. Same subreddit (r/MovieSuggestions), same
crowd-linked-to-IMDb pipeline, votes used as a quality filter, public release. Their opening example is
nearly word-for-word our pitch: *"The Lord of the Rings, but in space with a dark vibe."*

**3. He et al., CIKM 2023** — scale, and **our anti-circularity argument already in print**: *"crowd
workers often do not have a particular preference at the time of completing a task. In contrast, a real
user could have a very particular need"*, illustrated with a ReDial worker replying *"Whatever Whatever
I'm open to any suggestion."*

The organic-vs-role-play distinction itself is verified and clean (INSPIRED: *"two workers are randomly
paired and assigned different roles… **Then, recommenders start the conversation**"*; ReDial's **4-movie
quota**; DuRecDial explicitly inverts the roles; U-NEED's own Table 1 labels ReDial and CCPE-M
**"Simulated"**). But it is public knowledge to **cite**, not to claim.

**The one field-wide statement that does survive:** no public dataset in the **movie** domain has both an
organic open request at turn 1 *and* an out-of-dialogue consumption outcome. Reddit buys the opener;
commercial logs (U-NEED/Taobao, E-ConvRec/JD) buy the outcome; SBS has both and is books.

## Three ways our claim was WRONG, not merely unoriginal

**1. "It CANNOT be circular" — the *opener* can't; the *ground truth* can.** Reply titles come from the
same self-selected community. He et al. document the exact leak we would inherit: **>15% of INSPIRED
ground-truth items are already named earlier in the conversation, a trivial copy-the-history baseline
beats most CRS models, and stripping repeats drops #HIT@1 by >60%.** That is
`surprise-means-debug-the-harness` pre-loaded into the dataset.

**2. "Votes rank the replies" — do not do this.** The evidence against community votes as a graded label
is strong and consistent:
- **GuessTheKarma, CSCW 2018** — 20,674 influence-free paired judgments: the higher-scored item was
  preferred only **68%** of the time.
- **Glenski & Weninger, HT 2015** (N≈93,019): one artificial upvote → **+11.02%** final score, one
  downvote → **−5.15%**, and unlike Muchnik's *Science* 2013 result the **corrective effect does not
  replicate on Reddit**.
- **Glenski 2017: 73% of votes are cast without viewing the content.**
- **Gilbert, CSCW 2013: 51.52%** of eventually-popular links had been submitted before and ignored.
- The IR/CQA tradition calls community labels **silver** — SemEval-2017 says so verbatim; TREC LiveQA
  deliberately excluded best-answer flags and paid NIST assessors instead.
- **Neither Bogers & Koolen nor He et al. use votes.** He et al. ship an `upvotes` column and their paper
  contains **zero** occurrences of "upvote"/"vote"/"score". That sliver is unclaimed because the evidence
  says it does not work.
- **Use votes as Eberhard did: a coarse binary inclusion filter, never a graded relevance label.**

**3. "Cold-start is the differentiator"** — true against NDR (warm-start *by definition*: it requires a
transaction log or example items), but **already stated in print by Eberhard**: *"recommendations on
r/MovieSuggestions are usually generated only considering the information provided in each submission,
ignoring previous interactions."* A property of the data, not a contribution.

**Budget for contamination:** of Bogers & Koolen's 1,457 annotated forum "requests", **483 (33.2%) are
known-item needs, not recommendation needs**; of the remainder, ~25% give neither examples nor context.
An entity-anchored pipeline silently discards **~42%** of genuine requests and biases toward users who
already name titles. HARD RULE 1 applies — that is a truncation, and it is not neutral.

**One verified point in Reddit's favour:** He et al. find the most popular movies appear ~2% of the time
in ReDial ground truth but **<0.3% on Reddit** — reply-mined ground truth is ~7× *less*
popularity-concentrated than the crowdsourced CRS benchmark.

## What survives — and it is the same cell as Correction 1

**The adaptivity gap measured downstream of a non-circular organic opener, in a cold-start regime, with a
ranking outcome.** The three pieces all exist and never co-occur:

- **Organic request + ranking, ZERO follow-up:** Eberhard IUI'19 & WWW'25, He CIKM'23, Mysore RecSys'23,
  OCG-Agent EMNLP 2025 (single-shot on Eberhard's threads, +18.5% nDCG@10, **zero occurrences of
  "follow-up"/"multi-turn"**), Bogers & Koolen KaRS'18.
- **Adaptive multi-turn + ranking, but curated request and role-played answerer:** Qulac/NeuQS
  (SIGIR 2019), ClariQ, ConvSim (SIGIR 2023).
- **Real request + real signal, ONE question, no ranking:** Zamani WWW'20 — which states the boundary
  itself: *"**In this work, we do not study multi-turn interactions.**"*

In recommendation the openers are worse than role-played — they are **target-derived**. EAR, verbatim:
*"we randomly choose an attribute from the oracle set as the user's initialization to the session"*, and
the outcome is Success Rate@t, **not a ranking metric**.

Explicit negative searches: `"narrative" AND "clarifying" AND recommend` → **0**; `"r/MovieSuggestions"`
→ **0**. **No published work takes a real forum request and then asks system-chosen follow-ups.**

**THE REFRAME:** *"what does an adaptive interview add once a real request has already been given?"* —
with Eberhard/He as the **ruler** and the mined opener as **given evidence**, not as the contribution.
That converts our two most dangerous prior works into our baseline table, which is the strongest
defensive posture available.

## Two directly usable findings

- **Aliannejadi's best case is an OPEN question:** *"Open questions are very hard to formulate… however,
  it is more likely to get useful feedback."* Independent support for the opener-type hypothesis.
- **Krasakis et al., ICTIR 2020:** on Qulac, **negative answers *degrade* ranking** — NDCG@20 0.130 →
  0.106, **−18.5%**. A published contrast for our sign/dislike line (we measure +0.0229 at k=2).
- **Calibration target — ConvSim: +16% nDCG@3 from one round of feedback, +35% after three.** But with a
  curated opener and an LLM simulator, i.e. exactly the circularity we would be escaping. A target to
  beat on honesty, not a threat.

## Consolidated verdict table (both investigations)

| Claim | Verdict |
|---|---|
| **(A′) answer-space size moderates the adaptivity gap** | **Novel, one sentence wide.** Reframe from "open-ended" — our genre question was closed-categorical |
| **(C′) refusal invisible via the *answer model*** | True post-2011; write as the regression arc, not a novelty claim |
| **Adaptivity downstream of an organic opener** | **THE ONE GENUINELY EMPTY CELL in both investigations** |
| (B) open-vocabulary ingestion | Dead — TEARS, FLARE |
| Organic mined opener as a data contribution | Dead — Bogers & Koolen, Eberhard, He |
| Item-channel answer rate as a first measurement | Dead — Karimi UMUAI 2015 (26.5% vs our 27%) |
| Splitter objective / "EIG is user-agnostic" | Dead — Golbandi ANOVA identity; MacKay 1992 |

## Verification debts, priority order

1. **Koolen, Kamps & Kazai, CIKM 2012** — "Social book search: comparing topical relevance judgements and
   book suggestions for evaluation." Closed access. **This is literally the validity study for the ground
   truth we would adopt. Get it first.**
2. **Email Lukas Eberhard (TU Graz)** for the submission texts — the OSF release is dehydrated (verified
   by download).
3. **PEPPER citation** — the agent twice failed to supply it despite direct request; it underpins "no
   published critique exists" for item 9.
4. **fMF (Zhou 2011)**, **Golbandi CIKM 2010**, **McNee UM 2003**, **van der Linden 1999** — library
   access needed.
5. r/booksuggestions / r/AnimeSuggest dataset existence — LOW confidence, APIs rate-limited before
   completion.

## ⚠ RELIABILITY CAVEAT — applies to this file AND to `critiquing_vocabulary_is_closed.md`

**Two of nine sub-agents fabricated primary-source content before self-correcting.** Retracted items:
BK-VAE internals, **Burke / Chen & Pu quotes**, LangPTune mechanism, a fabricated
*"NEO arXiv:2603.17533"*, and Eberhard's field list. **Treat anything not tied to a specific extracted
quote as provisional.** In particular the Burke (AAAI-96) and Chen & Pu (UMUAI 2012) quotes in
`critiquing_vocabulary_is_closed.md` fall under this retraction and must be re-verified from the PDFs
before use — the PROJECTION verdict for those two rests on well-known properties of those systems and is
not in doubt, but **their quoted sentences are.**
