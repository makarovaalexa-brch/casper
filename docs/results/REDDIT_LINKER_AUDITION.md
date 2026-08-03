# Linker audition on the 500-mention gold set (2026-08-03)

**Question:** who should adjudicate the He-corpus entity links — a deterministic script, Haiku,
Sonnet, or Opus — measured against 500 hand-labeled mentions (batch 1: 300, seed 98765; batch 2:
200, seed 13579, disjoint; labeler = Fable, policy in `REDDIT_HE_GOLD_LABELS.md`; batch-2 file
`gold_sample2_labeled.csv`, combined `gold_combined_500.csv`). Batch 2 measured HARDER
(He precision 52.0% vs batch 1's 60.7%; combined 57.2%) — same stratification, different seed, so
treat the strata as oversampling hard contexts relative to corpus volume.

## Setup
- **Rules tier** (`scripts/reddit/linker_rules_tier.py`): span-only, no context. ACCEPT iff the
  span's top IMDb candidate has ≥2000 votes AND ≥10× the runner-up AND matches He's tconst AND
  span ≥4 chars, not stop-listed; or an explicit "(year)" uniquely pins a candidate He agrees with.
  Thresholds fixed before scoring, not tuned.
- **Models**: 15 subagents (3 models × 5 batches of 100), same prompt = the gold labeling policy +
  sentence + post year + span-only candidate list; may answer tconst / TITLE:name (year) / NONE.
  Blind to He's link and to the gold. TITLE: answers resolved to tconsts by exact-title(+year)
  match, most-voted on ties. Outputs archived per batch in `data/reddit_he/audition/`.

## RESULTS (exact-match accuracy vs gold, N=500; 31 rows flagged uncertain/ambiguous)

| stage | accuracy | excl. 31 uncertain |
|---|---|---|
| He release (as shipped) | **0.572** | — |
| Rules tier alone | 0.971 on its 48.6% coverage | — |
| Haiku solo | 0.772 | 0.778 |
| Sonnet solo | 0.942 | 0.953 |
| **Opus solo** | **0.972** | **0.981** |
| Pipeline rules→Haiku | 0.844 | — |
| Pipeline rules→Sonnet | 0.930 | — |
| Pipeline rules→Opus | 0.960 | — |
| On the 257 rules-DEFER rows | Haiku 0.724 · Sonnet 0.891 · Opus 0.949 | |

## Reading
1. **Opus passes the ≥0.95 gate; Sonnet is borderline; Haiku fails.** Sonnet's 0.953
   excluding gold-uncertain rows is defensible but its 0.891 on the hard (DEFER) tier — exactly
   the tier a pipeline would give it — is not.
2. **Opus's 14 "errors" mostly aren't:** 6 are TITLE: answers where Opus named the correct work
   but the exact-string resolver failed ("V Wars"→IMDb "V-Wars", "The Legend of King Naresuan"→
   "Kingdom of War: Part 1", "Kuroneko"→primary title "Black Cat", "Death(s) of Ian Stone");
   4 are coin flips on rows the gold itself flags ambiguous (Evil Dead 1981/2013, Dawn of the
   Dead 1978/2004, Lost in Space, "Mafia"). Genuine misses ≈ 4/500. **With a fuzzy/aka-aware
   resolver, effective Opus accuracy ≈ 0.98+.**
3. **The rules tier is worth keeping for cost, not accuracy**: it halves LLM volume at 0.971,
   costing the rules+Opus pipeline ~1 point vs Opus-solo (0.960 vs 0.972). Both pass the gate.
4. Observed audition cost: ~60–70k tokens per 100 mentions per agent (incl. agent overhead);
   production batching should land ~400–550 tokens/mention on the LLM tier.

## Plan of record for the corpus run (pending author go + volume census)
- Rules tier auto-accepts; **Opus adjudicates the DEFER tier**; TITLE: answers resolved with a
  punctuation/alias-tolerant resolver (fixes the 6 near-misses; add IMDb akas if needed).
- **Before pricing: run the tier-split census on the full corpus** — gold strata over-represent
  hard cases, so volume-weighted DEFER share is likely well under 51%; that number × ~500
  tokens/mention is the real Opus budget. Scope = mentions in the 35,982 eval threads first.
- Batches of ~500 mentions per agent, one output file per batch (granular, resumable, nothing
  lost on interruption), spot-check drift against a held-out slice of the gold mid-run.

## Caveats
- Gold is single-annotator (Fable); Opus/Sonnet/Haiku were graded by labels written by a
  same-family model — an independent human spot-check of the 31 uncertain rows would firm the
  gate. The 6 resolver near-misses are counted AGAINST Opus in the table (conservative).
- The audition models saw the same candidate lists the pipeline will supply; accuracy on
  mentions with empty candidate lists rides on TITLE: resolution quality.
