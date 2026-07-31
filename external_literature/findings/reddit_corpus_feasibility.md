# Organic-request corpora: what exists, what fits OUR catalogue, and what it costs

Research 2026-07-31 (feasibility stream; companion to `organic_open_request_novelty.md`, which covers
novelty). **Verdict: six weeks, not six months — but only via a third-party-released corpus, and the
best-fitting one needs an email.** Numbers below marked "measured" are the agent's own measurements
against our `data/ml-25m/proc/unique_sid.txt`, reproducible from its scratchpad scripts.

## The linking question is already solved on our side

`data/movielens/links.csv` (62,423 rows, ML-25M) maps **movieId → imdbId → tmdbId**. Checked locally
2026-07-31: **all 18,359 catalogue items resolve, 18,359/18,359 have an imdbId** (18,341 have tmdbId).
So any corpus carrying **IMDb tconsts joins to our catalogue by lookup** — no entity linking, no model,
no LLM, no author approval needed. The entity-linking cost only appears for corpora that ship raw text
without resolved ids.

## The three candidate corpora

### 1. Eberhard et al. — BEST FIT, but ships without the request text
OSF `10.17605/osf.io/ma2bj`, public, ungated.
- **1,480 crowd-annotated r/MovieSuggestions requests (2011–2017)**, 21,032 annotated comments,
  **44,448 recommendation mentions**, **IMDb tconsts throughout**.
- **Positive and negative movies annotated separately** — directly relevant to the sign/Arm-N line.
- **MEASURED catalogue fit: 86.2% of distinct items and 98.9% of mentions land in our 18,359-item
  catalogue; all 1,480 threads keep ≥5 items.** Human annotation plus a pre-2019 window makes it a near-
  perfect fit for ML-25M — dramatically better than Reddit-Movie's 61.3%/75.1%.
- **THE BLOCKER: it ships without the submission text.** And the obvious workaround is **proved dead** —
  **0/1,480 id overlap with He et al.'s corpus**, because their r/MovieSuggestions file contains **only
  2019–2022** (54,546 submissions) despite a README claiming 2012–2022.
- **Honest read on the negatives: thin.** Only **75 of 1,480 submissions (5.1%)** carry any — **177
  negative mentions against 5,344 positive**. A real seam for Arm-N, not a windfall.

### 2. He et al. — `ZhankuiHe/reddit_movie_raw` — the only public corpus that ships the free text
Ungated, full thread tree, actual request prose.
- **MEASURED catalogue fit: 61.3% of distinct items / 75.1% of mentions.**
- Needs re-linking: budget **3–5 weeks** with ReFinED + an IMDb gazetteer.

### 3. `yaochenzhu/Rank-GRPO` Reddit-v2 — the scale complement
383k / 9.4k / 11k. Google Drive link is in **`README.MD`** (uppercase — `README.md` is a 12-byte stub).
**Positives-only and fuzzy-matched, not human-annotated.**

### Books: DROP
No public dataset exists for r/booksuggestions or r/suggestmeabook — that would be a from-scratch mine
into the legally cornered route. (Bogers & Koolen's SBS remains the books benchmark, but it is
LibraryThing, not Reddit, and warm-start by construction.)

## The Most-Popular gate PASSES on both corpora — and the number is unclaimed

Per HARD RULE 5 / `surprise-means-debug-the-harness`, the new regime was gated on Most-Popular first.

- **Eberhard ruler** (OP-named items excluded, 740 eval threads): Prec@10 **0.0772**, Recall@10
  **0.0273**, **NDCG@10 0.0821**; random@20 Recall **0.0036**. Implied **F1@10 ≈ 0.040** against the
  published **doc2vec 0.1258 [0.1125, 0.1388]** and **GPT-4o >0.21**. Most-Popular sits ~3× below the
  best classical baseline — clear headroom, no degeneracy.
- **Reddit-Movie:** NDCG@20 **0.031**; the top-20 items are only **2.8% of mentions**.
- **Nobody has published a Most-Popular number on either corpus. That is ours to report.**

**Broken-ruler tripwires, pre-registered:** Most-Popular NDCG@10 above ~0.15; Most-Popular beating a text
model; or a method ranking that changes when OP-named items are excluded.

**The copy-the-history shortcut does NOT bite in this framing.** At thread level the copy-the-OP baseline
recovers only **4.4%** of the reply set — so the leak that broke He et al.'s *turn-level* protocol
(>15% of INSPIRED targets pre-named, copy-history beating most CRS) is a turn-level artifact, not a
thread-level one. This substantially softens correction (1) in `organic_open_request_novelty.md`, though
the crowd-relevance caveat below still stands.

## Two traps that would invalidate the statistics

1. **The V1 CSV has 17,176 `conv_id`s but only 1,620 real threads.** Evaluate at **thread level** or n
   inflates ~10× and every significance test is wrong.
2. **Report the metric ceilings.** With mean reply sets of ~30, **Recall@10 ≤ 0.43 and F1@10 ≤ 0.57.**

## Choosing the outcome signal

- **(i) Union of reply titles — PRIMARY.** It is the published WWW-2025 protocol and it survives the
  gate. But it is a **crowd relevance** set, not the OP's revealed preference, and we must say so.
- **(ii) Upvotes — NO.** Median 2, **44% at ≤1**, plus vote fuzzing. Consistent with the vote-reliability
  evidence in `organic_open_request_novelty.md`.
- **(iii) OP acknowledgement — good SECONDARY.** Extractable **without an LLM** via `is_submitter` and the
  `parent_id`/`child_ids` tree. Low volume, high precision.
- **(iv) Input-only against a held-out ML-25M user — the circularity-safe option**, and it fits our
  existing behavioural firewall.
- **(v) Human judgement — Eberhard already paid for it.** Reuse rather than repeat.

## Legal and ethical constraints (these shape the build, not just the write-up)

- **Reddit Developer Terms §4.2 bar using the data "to train large language, artificial intelligence, or
  other algorithmic models"; §5.2 says no right to use User Content "for training a machine learning or
  artificial intelligence model."** We train models, so this goes to our core use. **The distinction that
  saves us: those terms bind us only if we accept them by using Reddit's API. Consuming a third-party-
  released dataset does not.** → **Do not touch the Reddit API.**
- **ID-rehydration is dead as a reproducibility mechanism** — not throughput, but because Reddit requires
  deletion within ~48h and treats retention of deleted content as a violation "even if disassociated,
  de-identified or anonymized". Every rehydration yields a different, monotonically smaller corpus with
  non-random attrition: **our numbers would be unreproducible by construction.**
- **All pre-2024-04 Academic Torrents archives 404** (verified with real hashes), including the
  "2005-06–2023-12 top 40k subreddits" bundle. 2024-04 → 2026-06 return 200. Arctic Shift is alive
  (commit 2026-07-25) but is an "unauthorized third-party tool" by Reddit's definition.
- **The Zurich precedent — and our safe harbour.** Reddit's CLO called it *"deeply wrong on both a moral
  and legal level"*, banned the accounts and issued formal legal demands; UZH formally warned the PI and
  the researchers **decided not to publish**. The line the CMV mods drew: *"they were actively
  manipulating members of the subreddit rather than simply observing data."* **Mining already-written
  requests is observation.** Write two red lines into the protocol: **no posting to Reddit**, and **no
  LLM-inferred demographic attributes of posters** — the latter is exactly what drew the harshest
  reaction.
- **GDPR: "public" is not an exemption.** EDPS Opinion 6/2020: *"Personal data which are 'publicly
  available' — such as those collected from social media sites — are still personal data."* Usernames
  single out individuals → **hash at ingest**. Art. 89 is not a legal basis; still need Art. 6 (likely
  6(1)(e) for a university), and Art. 89(2) derogations depend on the **national** implementing act.
  **A DPIA is very likely required.**
- **Art. 9 special-category data is embedded in the very requests we want to model** — *"I'm a gay man
  looking for…"*, *"recovering from chemo"*, *"I have PTSD, no war films."* The "manifestly made public"
  exemption does **not** automatically apply (EDPB ChatGPT Taskforce; CJEU C-252/21). **Filter or redact
  at ingest; never quote a sensitive request verbatim in the paper.**

## Recommended action, in order

1. **Email Lukas Eberhard (TU Graz) for the submission texts.** Highest value per unit effort in the
   whole report: it converts the best-fitting corpus from unusable to usable. Institution-to-institution
   sharing is normal and the WWW-2025 follow-up is active.
2. **Meanwhile work with `ZhankuiHe/reddit_movie_raw`** — the only public corpus shipping free-text
   requests. Budget 3–5 weeks for re-linking.
3. **`yaochenzhu/Rank-GRPO` Reddit-v2** as the scale complement (note the uppercase `README.MD`).
4. **Drop books.**
5. **Frame it as narrative-driven recommendation** (Bogers & Koolen's established term). We join an
   existing leaderboard rather than inventing a ruler — which also answers "isn't this ReDial with extra
   steps?" more cleanly than anything else: it is a different, named task with its own benchmark, one
   that ReDial's UI-generated gold markup structurally cannot pose.

**⚠ Reliability:** this agent states it **fabricated a substantial block of its first report** and
corrected it only after checking. Everything retained above is either its own reproducible measurement or
a verified fetch; where research streams and direct checks disagreed it went with the artifact — notably
the dehydrated-dataset finding, where **the artifact contradicts the paper**.
