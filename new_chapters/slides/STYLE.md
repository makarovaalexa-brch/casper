# Supervisor deck — style sheet (binding for every slide)

Derived from the author's own decks (`old presentations/CoR Meeting 23_07_05.pptx`,
`Supervision Meeting 26_01_28.pptx`) on 2026-09-06. The deck must read as if the
author wrote it. Build with `deck_lib2.py`; check with `preview.py`.

## Voice
- Titles are **labels**, 2–5 words, in the author's `topic` or `topic: sub-topic` form.
  Good: "Research goals", "Recommendation engine: results", "Split criteria: ablation",
  "Interview tree: example paths". Bad: any sentence, any claim, anything with a verb phrase
  longer than the noun, "1 of 2", em-dashes.
- **No subtitles, no claim lines, no kickers, no hairlines, no footers, no source lines.**
- Bullets are **short phrases, not sentences**: no full stop, no "The X." bold lead-ins,
  no rhetorical questions, no taglines, no "and yet", no "this is what".
  Good: "Reward = NDCG delta", "Agent stuck in repetitive loops", "Same entity asked 5+ times".
  A number may sit inside a phrase: "conditioning beats static order, +0.0104 full (TEST)".
- 3–6 top-level bullets per slide, second level allowed, third level never.
- Plain academic register. No bold inside bullets. No colour in text. No italics except
  film titles.
- Chapters are "Chapter 1/2/3"; sections within a chapter are "section". Never "Paper",
  "Act", "arc", "walkthrough", "story", "hook".
- CASPER = Cold-start Adaptive Selection for Preference Elicitation in Recommendation.
  Expanded once on the title slide, plain CASPER after. Never CASPER-R / CASPER-U.

## Density
- Body text 18 pt (16 pt beside a figure or table, never below 14 for prose).
- Tables carry the substance and may use 9–12 pt. Keep every key table. Trim rows that
  do not change the reading; keep the interval columns.
- One idea per slide. A table slide has the table and at most 3 bullets. A figure slide
  has the figure, a caption, and at most 3 bullets. Split rather than shrink.

## Figures and flow charts
- Captions below the artwork in the author's form: "Figure N. …" / "Table N. …", 12 pt,
  grey, numbered per chapter (1.1, 1.2 … / 2.1 …).
- Flow charts are native shapes in the author's palette: orange rounded boxes
  (`obox`), cream container for a subsystem (`container`), white boxes with thin black
  outline for utterances and data (`wbox`), speech bubbles (`bubble`), thin black arrows
  with short lower-case labels (`arrow`), black outline icons (`icon("robot")`,
  `icon("chat")`, `person()`, `database()`). One dark-orange box (`dark=True`) at most
  per chart, for the one thing to remember.
- Extracted chapter figures from `img/` are placed with `figure()`, as large as the
  body area allows.

## Geometry (inches, 10 × 5.625 canvas)
- Body area: x 0.34 → 9.66, y 1.26 → 5.00. Logo occupies x 7.83–9.71, y 0.22–0.85 — the
  title must be short enough not to run under it (≈ 42 characters at 28 pt).
- Nothing below y = 5.25. Two-column split at x = 5.0 with a 0.3 gutter.

## Speaker notes
- Every content slide carries 3–6 lines of talking points in `notes()`, plain phrases,
  about 60–80 seconds of speech. Total main deck ≈ 50 minutes at few interruptions.
- Anything the author must decide is a separate line starting `NOTE TO AUTHOR:`.

## Numbers
- Every number comes from the chapter `.tex` files or `build_deck.py` (which was built
  from them). Never invent, round beyond the source, or drop an interval that the
  source states next to a headline number.
- Chapter 2's finding is "published adaptive selectors optimise the wrong objective";
  conditioning beats the static order on TEST (+0.01035 [+0.00739, +0.01323] full).
  Never "adaptivity does not pay".
- No absolute NDCG claim for Chapter 3 (declared-circular answer model): relative only.
