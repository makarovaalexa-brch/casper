# Reddit slot parse v2 — every ruler request decomposed into typed preference slots

Run 2026-08-03/04. Input: all **4,431** ruler threads (`ruler/ruler_v1.csv.gz`, request text capped
at 8,000 chars — cap raised from the pilot's 4,000 after one mega-list truncated). Parser: Opus,
via the Workflow fleet (40 slices × ~111 requests; slice 15 lost to an API connection drop and
rerun clean; coverage asserted **4,431/4,431** by `scratchpad/consolidate_slots.py`).
Output: `data/reddit_he/slots/slots_v2.jsonl` + `slots_v2_stats.json` (both committed).
Approved LLM use (author, 2026-08-03). No Reddit API. No usernames in outputs.

## Schema v2 (per thread)

```json
{"idx", "thread_id", "is_request",
 "slots": {"anchors_pos", "anchors_neg", "genres_pos", "genres_neg",
           "era", "tone", "constraints", "context", "other"},
 "n_slots"}
```

Conventions the parser was held to:
- **Verbatim-substring rule** — every slot value is a literal substring of the request text
  (modulo whitespace), so any downstream resolution failure is auditable against the source.
- **Seen-list rule** — films the OP lists as *already seen / please exclude* go to `anchors_pos`
  or `anchors_neg` by expressed sentiment, never silently dropped: the seen-list is the closest
  thing the request has to a rating profile.
- **Character anchors kept verbatim** ("movies like John Wick the character") — resolution to a
  film is a *later*, separately-audited step, not the parser's job.
- **`is_request`** — false for meta posts, discussion threads, "what was that movie" identification
  posts; those stay in the file (with slots still parsed where present) but are excluded from
  interview experiments.

## Aggregate stats

| stat | value |
|---|---|
| threads parsed | 4,431 |
| genuine requests (`is_request`) | 4,166 (**94.0%**) |
| slots per thread | mean **5.91**, median **5**, p90 **10**, max 241, zero-slot 56 |

Slot volume and coverage (threads with ≥1 value in the field):

| field | total values | threads with field | share of threads |
|---|---|---|---|
| anchors_pos | 9,655 | 2,793 | **63.0%** |
| other | 6,388 | 3,518 | 79.4% |
| anchors_neg | 3,194 | 730 | 16.5% |
| genres_pos | 2,378 | 1,763 | **39.8%** |
| constraints | 1,930 | 1,360 | 30.7% |
| tone | 1,365 | 946 | 21.4% |
| context | 612 | 612 | 13.8% |
| era | 433 | 433 | 9.8% |
| genres_neg | 242 | 184 | 4.2% |

## What this feeds

- **Withheld-slot replay** (`DS_REDDIT_LINKING.md` §ADAPTIVITY-ON-REDDIT): the per-thread slot
  set IS the askable pool — fold a partial request, hold out volunteered slots, and the
  refusal-free variant chooses among slots the OP actually volunteered. Median 5 slots/thread
  means the pool is small but real; 2,793 threads have positive anchors to withhold.
- **Init-shaping bridge** (Act 3): `anchors_pos`/`anchors_neg`/`genres_pos` are the model-free
  matching criteria for pairing ML-25M users to request shapes.
- The 3,194 negative anchor values are the request-side face of the sign question (R5 lineage):
  16.5% of threads volunteer explicit dislikes unprompted.

## Known limits

- `other` is a catch-all (79% of threads) — plot points, actor names, platform mentions; it is
  deliberately not typed further at parse time.
- The max-241-slot outlier is a mega-list thread; the 8,000-char cap already truncates a handful
  of such lists (cap disclosed, not silent — the ruler text itself is uncapped).
- `is_request=false` (265 threads, 6%) still carry ground-truth mentions in the ruler; whether
  they enter any experiment is a per-experiment decision, default exclude.
