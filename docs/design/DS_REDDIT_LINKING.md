# Design sheet — He-corpus request linking (organic-opener data, plan of record)

**Status: DESIGN ONLY. The LLM batch run requires explicit author approval — this sheet is what
gets approved.** Origin: author ruling 2026-08-03 (no emails; He corpus is the primary route).
Feasibility facts → `external_literature/findings/reddit_corpus_feasibility.md`.

## Goal
Turn `ZhankuiHe/reddit_movie_raw` (public, ships request TEXT) into an evaluation ruler for
"what does an adaptive interview add once a real request has been given": request text as given
evidence, reply-mentioned movies (mapped to our 18,359-item catalogue) as ground truth,
OP-named items excluded at scoring.

## Steps, in order (1–3 need NO LLM and can start any time)
1. **Pull the corpus** (git clone / direct download — never the Reddit API). Hash usernames at
   ingest; flag-and-redact Art. 9 sensitive content at ingest (illness, sexuality, religion…);
   never quote a sensitive request verbatim anywhere.
2. **Census**: thread counts by year, mentions per thread, existing entity annotations in the
   release (their processed files carry links — measure how far they get us before any new
   linking), reproduction of the 61.3%/75.1% catalogue-fit number, thread-level dedup
   (the V1-CSV trap: 17,176 conv_ids ≈ 1,620 real threads — evaluate at THREAD level only).
   **★ DECISIVE CHECK (added 2026-08-03): the HF dataset card claims the LARGE version covers
   Jan 2012 – Dec 2022 (5 subreddits, 3.01 GB), but our Jul-31 measurement of the
   moviesuggestions file found ONLY 2019–2022 (54,546 submissions; consistent with 0/1,480
   Eberhard overlap). Verify per-file, per-year counts on the LARGE download. If pre-2019
   threads exist, the ≤2019 slice has near-Eberhard catalogue coverage (~99%) WITH request
   text — it becomes the primary evaluation slice and the recency-bias scope condition shrinks
   to a secondary-slice caveat.**
   Also census: count threads where the OP answered clarifying questions in comments
   (`is_submitter` in the reply tree) — the organic-follow-up subset that permits any
   adaptivity measurement on this corpus (see "What this ruler is NOT").
3. **Validation sample**: hand-label ~300 mentions (text span → IMDb tconst or NONE), drawn
   stratified over years and thread lengths. This is the gold set every linker is measured on.
   Label before seeing any linker output.
4. **LLM-assisted linking** (⚠ AUTHOR APPROVAL GATE — model, cost estimate, and prompt go to the
   author first): extract movie mentions from request + replies, resolve to IMDb tconst.
   Constraints: temperature 0, output must be a tconst present in an IMDb gazetteer (no free-form
   IDs — hallucinated tconsts are the known failure), NONE allowed and expected. The 173/300
   quarantined study users are irrelevant here (different population) but NO Reddit text ever
   enters any training set — this corpus is EVALUATION ONLY.
5. **Measure the linker** on the gold set: precision/recall by year and by mention type (exact
   title / paraphrase / nickname). GATE: precision ≥0.95 on linked mentions, else iterate or
   fall back to gazetteer-only exact matching with the loss reported.
6. **Build the ruler**: IMDb→MovieLens via `links.csv` (lookup, lossless); keep threads with
   ≥5 in-catalogue ground-truth items; exclude OP-named items from scoring (Eberhard protocol).
7. **Gates before any model number is interpreted** (Hard Rule 5 + the Jul-29 rule):
   Most-Popular NDCG@10 must stay ≲0.15 and ~3× below a text baseline (prior measurement:
   0.031–0.082 across corpora — healthy); shuffle control (permute requests across threads →
   any request-aware method must collapse); report Recall@10 ceilings given reply-set sizes.

## Known truncations to PRINT, not hide (Hard Rule 1 — these are disclosed, not silent)
- Catalogue coverage: corpus is 2019–22, ML-25M ends late 2019 → ~25% of mentions unmappable,
  concentrated in the newest films. Report retention per thread-year; state the recency bias as
  a scope condition.
- The ≥5-items filter: report how many threads it drops.
- Entity-anchored ground truth biases toward users who name titles (the Bogers 42% lesson) —
  one sentence in limitations.

## What this ruler is NOT
- Not a training set (nothing trains on Reddit text — legal + quarantine).
- Not graded by votes (votes at most a coarse binary inclusion filter).
- Not a data contribution claim (pre-empted: Bogers & Koolen, Eberhard, He). The contribution is
  the MEASUREMENT downstream of a real request.
- **Not an adaptivity simulator.** No OP is available to answer follow-up questions; simulating
  answers from the reply set is circular (firewall) and from the request text alone makes
  follow-ups uninformative by construction. Adaptivity on this corpus is measured ONLY on the
  organic-follow-up subset (OP actually answered clarifying comments); everything else here
  evaluates the request-conditioned cold ranking. Controlled adaptivity magnitudes stay on
  ML-25M. Ground truth = union of reply-suggested movies, OP-named excluded (crowd relevance —
  say so); secondary = OP acknowledgement via `is_submitter` (no LLM needed).

## Effort estimate
Steps 1–3: ~2–3 days. Step 4–5: days once approved (batch size ~54k submissions + reply trees —
cost estimate delivered with the approval request). Total 1–2 weeks, no external dependency.
