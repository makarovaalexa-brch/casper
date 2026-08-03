# Gold-sample labels for the He-corpus linker (300 mentions, labeled by Fable 2026-08-03)

**File:** `data/reddit_he/gold_sample_labeled.csv` (adds `label_tconst_or_NONE` + `note` to the
census's stratified sample: 15 strata = 5 years × 3 contexts, 20 each, seed 98765).
Labeled single-annotator (Fable in-session, per author instruction 2026-08-03), adjudicated against
the full IMDb basics dump (all title types + vote counts) with sentence context; 5 hard cases
verified by targeted lookup. **17 rows carry an UNCERTAIN/AMBIGUOUS flag — spot-check those first
if auditing.**

## Labeling policy (fixed before labeling)
1. Label = tconst of the work the commenter actually refers to, **any IMDb type** (movie, TV,
   mini, short) — movie-only filtering is a downstream eval decision, not a gold decision.
2. **NONE** when the span is not a title reference: conversational fragments ("Tyvm.", "true"),
   platforms (Netflix, Mubi, Hulu), people (Lovecraft, Elvira, Paul Schrader, Cantinflas),
   characters (Mr. Miyagi, Maverick, Spider-Man-as-character, Wiccan), video games (Doom, Diablo),
   books/comics being discussed as books ("best movie adaptation for Crime and Punishment?"),
   genre terms ("bottle movie", "True Crime" the genre), anaphora ("That", "the British Mafia one").
3. Truncated spans get the intended full-title referent ("In → In & Out 1997; "Missouri" →
   Three Billboards; "Dark" [sentence: The Dark Knight] → tt0468569).
4. Bare ambiguous titles → the exact-title match with dominant fame, flagged AMBIGUOUS
   (Italian Job→2003, Evil Dead→2013, War of the Worlds→2005, The Game→1997).
5. Franchise mentions → the first/founding film bearing that title, flagged franchise
   (Star Wars→ANH, Harry Potter→Sorcerer's Stone, Carry On→1958, King Naresuan→pt1 2007).
6. Post year is a hard constraint (a 2018 comment cannot mean a 2020 title) and decides
   TV-vs-movie same-name cases together with context words ("show", "series", "episodes").

## THE HEADLINE: their linker's mention-level precision is 0.607
| verdict | n | share |
|---|---|---|
| he_tconst correct | 182 | 60.7% |
| wrong entity (a real title was meant, he picked another) | 79 | 26.3% |
| span is NONE (he hallucinated a title onto a non-mention) | 39 | 13.0% |

By context (n=100 each): **reply 67% / request 63% / seeker_followup 52%.** The stratification
oversamples followups relative to the corpus, so corpus-weighted precision is likely a bit above
0.61 but nowhere near the 99.87% *coverage* the release advertises. Coverage ≠ precision.

## Error taxonomy (what the adjudication pass must fix)
- **Same-name wrong pick** (biggest class): TV-vs-movie (New Girl→1985 film, Skins→2002 film,
  Mindhunter→Mindhunters, The Last Kingdom→The Last King) and same-title year confusions
  (Castaway'86 vs Cast Away'00, The Way Back'10 vs '20, I Want to Live 2015 vs 1958).
- **Near-string garbage**: Casablanca→Inglourious Basterds, United 93→Alien, Deadpool→Office Space,
  Coco→Rent, A Clockwork Orange→Tex, Eraserhead→24, matrix→Matria, Dexter→Lexter, NTTD→Dune,
  the crown→The Crow. These are not typos of the user — the matcher picked an unrelated entry.
- **Non-mentions linked** (13%): conversational one-liners, platforms, characters, games, genre
  terms. A stop-list + POS/NER-free heuristics catch some; most need context judgment.
- **Truncation artifacts**: multi-word titles split at commas/quotes ("In, "Me, "Missouri, "Dark).
- **Abbreviations/typos resolved fine by context but not by string match**: TNCFU, NTTD,
  "rimless"→Timeless, "Horribile bosses".

## Implications for the pipeline (updates DS_REDDIT_LINKING.md step 4–5)
1. **The adjudication LLM pass is not optional.** A blocklist fixes only the non-mention class
   (~13%); the 26% wrong-entity class needs context reading. Target: ≥0.95 precision on this gold
   set. The cheap-model pass (Sonnet/Haiku per author) gets: span + full sentence + candidate list
   from the gazetteer with types/years/votes + post year, and may output NONE.
2. **Ground-truth quality bound**: reply mentions are the evaluation ground truth; at 67% raw
   precision, roughly a third of unadjudicated ground-truth items are wrong — adjudication must
   run on reply mentions at minimum (~44k in the annotated subset; full corpus scoped in the sheet).
3. **Sensitivity check to report**: metric stability under gold-labeled vs raw links on the 300.

## Caveats
- Single annotator; 17 flagged uncertain (TV-vs-movie coin flips like The Outsider 2020-TV vs
  2018-film, US-vs-UK Office/Shameless, Legion film-vs-TV, Stargate film-vs-SG-1, Maniac 2018).
- The labeler (Fable) must NOT be the adjudication model — author's plan (cheaper model does the
  corpus, this gold set grades it) keeps grader and gradee separate.
- These 300 rows are for MEASURING linkers; never train on them.
