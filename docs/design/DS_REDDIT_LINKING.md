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

## ★ ADAPTIVITY-ON-REDDIT PROTOCOL (author + Fable, 2026-08-04 early hours)
The Reddit cell's thesis: real requests give users an INITIAL PLACEMENT; show adaptivity pays
downstream of it. Two legs + a bridge:
- **Leg 1 (dispersion, no answers needed):** fold each request into the certified tower; compute
  the loss-optimal next question per thread; measure across-thread dispersion of that argmax.
  Contrast with the (expected ≈0) dispersion under standard banks on ML-25M. Fully offline.
- **Leg 2 (payoff, thread-revealed-preference oracle):** LLM pass extracts each OP's statements
  into typed slots (genre/era/tone/seen-X/constraints) = the ANSWER POOL. Answers come ONLY from
  the OP's own words (behavioral firewall; reply-set GT stays quarantined for scoring).
  **UPDATED after the follow-up census (2026-08-04): the REQUEST TEXT is the primary pool
  source** — organic follow-ups are near-absent (median 0 OP replies; ~0.1–0.2 genuine probes per
  thread; see `REDDIT_FOLLOWUP_CENSUS.md`), but requests are multi-sentence and carry several
  distinct revealed preferences. **WITHHELD-SLOT REPLAY:** fold a PARTIAL request as the initial
  placement (e.g. anchor films only); the remaining slots the user volunteered are the answerable
  pool; adaptive picks which withheld slot to ask for from the placement, static uses a fixed
  slot order. Claim shape: "could an interviewer have elicited what this user was going to say,
  in fewer questions, by choosing per-user?" Nothing simulated — every answer is the user's own
  written words. CONCEDE PLAINLY: no live user answers our question anywhere on this corpus;
  realism tops out at replaying verifiably-possessed information (the 300-user study is the only
  path above that ceiling). Census bonus figures: crowd asks probes in only ~10% of threads yet
  users answer 51% when asked (the unfilled-niche motivation panel); 19.7% of threads carry an
  acknowledged-recommendation outcome label (secondary scoring target, ~10.6k events).
  TWO VARIANTS, run BOTH (author ruling):
  1. **PRIMARY — pool-restricted, refusal-free:** both arms choose only from that thread's
     available pool; budget counts answered questions only. Static = global priority order over
     question types fitted on train threads, take highest-priority available; adaptive = choose
     from pool by posterior placement, re-decide per answer. Isolates SELECTION given
     answerability — the ML-25M SEL-firewall analogue. HEAVY CAVEATS to print: pool composition
     is itself informative and shaped by what commenters happened to ask (selection bias); gains
     conflate answer content with pool information; comparison is clean, absolutes are not.
  2. **SECONDARY — unrestricted with refusals:** both arms ask anything; unanswered = refusal,
     counts toward budget. Measures the answerability-targeting channel that variant 1
     deliberately neutralizes. Sparse and noisy — report as the channel decomposition, never as
     the headline.
- **Bridge:** measure the dispersion real requests induce (leg 1), reproduce that dispersion on
  ML-25M with constructed initial-evidence sets, run full adaptive-vs-static there with complete
  behavioral answers. Claim shape: "real requests place users at dispersion X (ecological); at
  dispersion X adaptivity buys Y (controlled)".
- Sequencing: smoke trio (MostPop gate, copy-OP, tower request-fold) → leg 1 figure → ML-25M
  matched-dispersion probe → oracle extraction pass → leg 2 both variants.

## Effort estimate
Steps 1–3: ~2–3 days. Step 4–5: days once approved (batch size ~54k submissions + reply trees —
cost estimate delivered with the approval request). Total 1–2 weeks, no external dependency.

## ★ PIPELINE OF RECORD (author decision 2026-08-03, after the 500-gold audition)
- **Rules tier auto-accepts** (48.6% coverage @ 0.971 on gold; `linker_rules_tier.py`, thresholds
  frozen) → **OPUS adjudicates the DEFER tier** (audition: 0.949 on DEFER; pipeline 0.960 overall
  vs He's raw 0.572). FINAL author decision 2026-08-03 (after initially weighing Sonnet 0.930):
  label errors are permanent bias in the ground truth while sample size is recoverable variance,
  and the thread-shuffle makes stopping early the clean budget lever — so maximize quality per
  label. Run in waves of 10 batches with a progress/percent report between waves; stoppable at any
  wave boundary.
- **Thread-level shuffle, fixed seed**: scope = DEFER mentions in the 35,982 eval threads;
  threads shuffled once (seed 98765), batched ~500 mentions keeping threads intact. Any prefix of
  completed batches = unbiased random subsample of eval threads with FULLY adjudicated ground
  truth (label errors are bias, sample size is variance — downsampling by stopping early is the
  clean budget lever).
- Batches run as subagent waves (~10 at a time), one checkpoint file per batch under
  `data/reddit_he/adjudication/`; resumable across plan resets. TITLE: answers resolved with a
  punctuation/alias-tolerant resolver (fixes the exact-string near-misses found in the audition).
- Print in the paper: linker precision measured on the 500-mention gold set, rules/Sonnet split,
  and the residual ~7% label-error floor as a stated limitation.
