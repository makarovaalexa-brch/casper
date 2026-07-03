# Goodreads Cross-Domain Replication — Phase 1 Result

Scope: port the Paper-B (ML-25M) instrument pipeline to the **Goodreads Mystery/Thriller/Crime**
byGenre slice (McAuley Lab UCSD) as an independent cross-domain replication. Build data artifacts,
biased-SVD linear reference + V1 neural fold-in encoder, run the health gate, measure concept
effective-rank and answerability. **Phase 2 (concept/item B-battery + policy) is a separate run,
gated on the instrument-health verdict below.**

Raw data (academic-use-only, no redistribution) lives under `casper/.cache/goodreads/` (gitignored).
Derived artifacts (`base.npz`, `Q_svd.npy`, …) live in the same dir. Build scripts under
`casper/scripts/paper2/gr_*.py`.

---

## 1. Data prep (EXPLICIT track) — counts

Stream-parsed `goodreads_interactions_mystery_thriller_crime.json.gz` once (never fully decompressed).
Keep interactions with **rating>0** (explicit ratings only). like = rating≥4, dislike ≤2 (mirrors ML
semantics; Goodreads ratings are integer 1–5). Item filter: catalog = books with **≥20 ratings**
(stated preprocessing filter, NOT sampling). **FULL user set** (no activity sampling).

| quantity | value |
|---|---|
| raw interaction rows scanned | 24,799,896 |
| rated (rating>0) interactions | 11,715,518 |
| **items after ≥20-rating filter** | **52,613** |
| **users (full set)** | **568,218** |
| **ratings kept** | **10,772,997** |
| ratings/user | **median 6**, mean 19.0, p25 2, p75 17, p90 43, max 3707 |

User split (mirrors ML-25M): rng(0) shuffle, last 1000 held out → 500 val / 500 test; rest (567,218) train.

**This is a genuinely cold-start-realistic regime** (median 6 ratings/user), unlike the ML-25M
phase-1 heavy-rater subsample (median 726) that saturated on popularity. It is the regime where
elicitation should earn its keep.

### Popularity concentration (vs ML-25M reference)
| dataset | items | top-1% items' rating share | top-0.1% share | Gini |
|---|---|---|---|---|
| **Goodreads MTC** | 52,613 | **35.5%** | 14.2% | 0.759 |
| ML-25M subsample (ref) | 18,430 | 25.1% | — | 0.837 |

Goodreads is **more concentrated in its head** (top-1% hold 35.5% vs 25.1%) but has a **lower overall
Gini** (0.759 vs 0.837) — the ML-25M subsample's Gini is inflated by the heavy-rater regime's very
long near-zero tail. (ML-25M reference is the local 10k-heavy-rater subsample, the only ML-25M
collaborative signal on disk; treat as indicative.)

---

## 2. THE RULER (pre-stated 2026-07-03, BEFORE any recommender/eval numbers below)

- **Primary K = catalogue-fraction-matched:** K = round(0.0027 × N), N = 52,613 items ⇒ **K = 142.**
  (Same fraction that gives ML-1M K=10 at ~3,700 items and ML-25M K=45–50 at ~16.7k items.)
- **Reported alongside:** NDCG@10 (fixed shallow depth, cross-dataset comparability).
- **Tail:** Cremonesi head-33% — the cumulative-popularity head is masked from both candidates and
  relevant set; NDCG re-scored on the tail remainder.
- **Recommender decoder convention (V1, matches `encoder_recon`):** score = popb + Q·u,
  popb = log(cnt+1) (β=1); ridge fold-in λ=5. MOSTPOP q0 = popb (u=0).
- **Held-out disjoint-targets protocol** (the no-harm / headroom ruler): per test user, likes shuffled
  with rng(123), second half = held-out target; first half + asked items excluded from candidates.
- **Health gate (primary K=142):** encoder full-profile fold NDCG@K − MOSTPOP q0 **substantially
  positive** AND monotone no-harm elicitation (revealing never hurts) for random/entropy selectors at
  reveals {0,8,20}. Encoder headroom substantially positive = PASS.
- **Item factors:** biased SVD (Koren09) mini-batch SGD, D=64, LAMF=0.05, LR=0.01, EP=15 — identical
  recipe to ML-1M/ML-25M. **V1 neural encoder:** attention fold-in set-encoder, d=64 (mirrors
  `ml25m_train_enc.py` / `encoder_recon`), best-val on 500-user val cohort (NDCG@K full catalogue).

*(Results below were produced after this ruler was fixed.)*

---

## 3. Item factors (biased-SVD linear reference)
Biased SVD (Koren09) mini-batch SGD on 10.75M train ratings, D=64, LAMF=0.05, LR=0.01, EP=15
(identical recipe to ML-1M/ML-25M). Train RMSE 0.823 → **0.698** (converged, healthy; cf. ML-25M 0.728).
Artifacts `Q_svd.npy` (52613×64), `bi_svd.npy`.

## 4. SBERT content embeddings
`all-MiniLM-L6-v2` over title + first 200 chars of description (dense item order) → `item_sbert.npy`
(52613×384, L2-normalized). Parity with the CASPER content pipeline; not on the gate's critical path
(the linear reference + V1 encoder use the collaborative Q). Run note: encode is CPU-bound ~22 min on
this box; set `HF_HUB_OFFLINE=1` (the hub check hangs otherwise) — baked into `gr_sbert.py`.

## 5. Concept vocabulary (popular_shelves)
Streamed the books file; parsed `popular_shelves`. Membership: a shelf tags a book if its shelf-count ≥5.
Junk-filtered (to-read, currently-reading, owned, favorites, kindle, library, default, wish-list, to-buy,
series, dnf, abandoned, audiobook, own-*, tbr, arc, giveaway, netgalley, star/rating shelves, bare years,
≤2-char, + substring rules). Frequency floor ≥30 catalogue items; kept **top 1500 content shelves by document
frequency** (DF range 42,368 → 50). All 52,613 catalogue items matched their book record; **95.3% carry ≥1
concept shelf**; mean 621 items/concept. Concept centroid directions `Ac` = mean Q over member items (same
construction as ML-25M's `Ac`). Examples: `mystery` (42,368 items), `thriller` (27,000), `crime` (25,564),
`suspense`, `historical-fiction`, `courtroom`, `monsters`, `will-trent-series` (50). The vocabulary mixes
broad genres, affect/structure shelves, and author/series shelves — richer and less curated than the
ML-25M genome.

## 6. V1 neural encoder training
V1 attention fold-in set-encoder (mirrors `ml25m_train_enc.py` / `encoder_recon`): d=64 attention pooling
over revealed (Q[item], residual) tokens → u; frozen decoder popb + Q·u; IPS-weighted BCE reconstruction
of the unrevealed like-set; random-count reveals k=1..12. Trained on 175,524 train users with ≥13 ratings,
~340 s/epoch, chunked+resumable (`gr_train_enc.py`). **Val-selected** on the 500-user val cohort
(full-profile fold, NDCG@142, n=265 usable):

| ep | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|
| val NDCG@142 | .1894 | .1986 | .2017 | .2014 | .2002 | .2011 | **.2024** | .2011 | .2001 |

Plateaued from ep3; stopped at 9/12 epochs. **Best-val checkpoint = ep7 (0.2024)** → `enc_v1_gr.pt`
(+ `enc_v1_gr_state.pt` resume state, `enc_v1_gr_peak.txt` durable peak log). Used by the gate.

## 7. Health gate — VERDICT: **mechanically healthy; headroom WEAK/FAIL-band on the pre-stated bar; item-asking channel structurally starved (the strong cross-domain answerability datapoint)**

`gr_health.py`, held-out disjoint targets, n=272 test users (≥4 likes), n_tail=229. All numbers NDCG.

**(a) Full-profile fold (the headroom measurement):**
| model | @10 full | @10 tail | @142 full | @142 tail |
|---|---|---|---|---|
| MOSTPOP (q0) | 0.1239 | 0.0126 | 0.2047 | 0.0329 |
| ridge (λ=5) | 0.1245 | 0.0298 | 0.2072 | 0.0476 |
| **ENCODER** | **0.1357** | 0.0152 | **0.2151** | 0.0343 |

- **Encoder headroom over MOSTPOP: @142 full +0.0104 (+5.1% rel), @10 full +0.0118 (+9.5% rel), @142 tail +0.0014.**
- Ridge headroom @142 full is only +0.0025 → the neural encoder ~4× the linear fold's headroom; ordering
  encoder > ridge > MOSTPOP holds on full (mechanics healthy). On **tail** ridge (+0.0147) beats the
  encoder (+0.0014) — under the popb decoder the encoder chases the head in this regime.
- Headroom by profile size (@142 full): 2–5 items **+0.0304** (n=76), 6–15 +0.0062 (n=102),
  16–40 +0.0114 (n=55), 41+ +0.0039 (n=39). Headroom does **not** grow with profile size → the weakness
  is not profile starvation; per-item personalization signal within the genre slice is genuinely small.

**(b) Monotone no-harm elicitation (encoder fold; reveals {0,8,20}, @142 full):**
| selector | q0 | q8 | q20 | no-harm | note |
|---|---|---|---|---|---|
| random | 0.2047 | 0.2047 | 0.2046 | OK | ~zero seeds hit (see below) |
| entropy | 0.2047 | 0.2047 | 0.2047 | OK | ~zero seeds hit |
| pop | 0.2047 | 0.2748 | 0.3333 | OK | **exclusion-confounded** (see below) |

- **No-harm holds everywhere** (revealing never hurts) — the instrument is sound.
- **Item-asking answerability starvation (the striking cross-domain datapoint):** test users rated a
  median of **6 of 52,613** items, so random/entropy asks essentially never hit a rated item → **zero
  realizable elicitation signal from arbitrary item-asking**. Even popularity-ordered asking hits only
  ~1.1 rated items in 8 asks (50.8% of users get ≥1 hit; ~1.7 hits in 20 asks). This is the Paper-B
  thesis made extreme: on a realistic sparse book cohort, item-level elicitation is structurally starved
  and an answerable (concept) channel is *necessary*, not merely better.
- The pop-selector "gain" (+0.1286 @142 by q20) is **not clean elicitation evidence**: the held-out
  protocol excludes asked items from the candidate list, so asking the top-20 popular items also removes
  the strongest popular distractors (ML-25M's health run showed the same "pop rises" behaviour). Treat
  as an upper bound with a known artifact; the honest realizable item-elicitation gain here is ≈ 0.

**(c) GATE (pre-stated: encoder full-profile − MOSTPOP substantially positive AND monotone no-harm):**
- Monotone no-harm: **PASS**. Mechanics (SVD converged, encoder > ridge > MOSTPOP full, val-selected
  encoder): **PASS**.
- Headroom: **+0.0104 @142 / +0.0118 @10 — positive but NOT substantial** (ML-1M reference ≈ +0.10).
  **GATE VERDICT: WEAK / FAIL-band.**
- Diagnosis (a *different* failure mode from ML-25M phase 1): not popularity-typical heavy raters, but
  **single-genre homogeneity + sparse explicit ratings** — within Mystery/Thriller/Crime, popularity
  explains most of the ranking and per-user explicit signal is thin (median 6 ratings), so collaborative
  personalization headroom at the full-catalogue task is intrinsically small (~0.01).
- Recommendation for the coordinator: the slice is uniquely strong as an **answerability/concept-necessity
  showcase** (§7b, §9) with the concept channel already built (§5); it is weak as a **policy-efficiency
  ruler** (a phase-2 battery would be measuring a ~0.01 prize). Either run Phase 2 scoped to the
  concept-vs-item story, or re-scope the ruler first (e.g. multi-genre Goodreads composite to restore
  cross-genre taste differentiation, or within-shortlist ranking) before the full battery.

## 8. Effective rank (concepts vs items) — the cross-domain rank datapoint
Participation ratio (effective dimensionality) of unit direction vectors, D=64 (same method as ML-25M):

| space | Goodreads effrank (uncentered) | centered | ML-25M ref | ML-1M ref |
|---|---|---|---|---|
| **Shelf concepts** (1500 shelves, ≥30 items) | **34.65 / 64** | 35.68 | 4.11 | 2.25 |
| **Pool items** (top-600 popular) | **30.48 / 64** | 30.48 | 9.09 | 27.1 |

**Cross-domain divergence (honest, important):** on ML-25M the *genome* concept directions collapsed to a
very low-rank subspace (4.11 ≪ items 9.09) — the qualitative "concepts saturate in few directions" claim.
On Goodreads the **shelf** concepts do NOT collapse: they span **34.65 / 64**, *slightly more* than the
popular-item pool (30.48). Two drivers: (i) the shelf vocabulary is far more heterogeneous than curated
genome tags (1500 folksonomy shelves incl. micro-genres and author/series tags spread across many
directions); (ii) the realistic (non-saturated) collaborative Q here spans a much higher item rank than the
ML-25M heavy-rater subsample's compressed Q (30.48 vs 9.09). Net: the ML-25M "low-rank concept subspace"
finding is **construction-specific (genome), not universal** — folksonomy-derived concepts occupy a
high-rank space comparable to items. This is a genuine cross-domain result, not a bug.

## 9. Answerability (concepts vs items)
Realistic FULL cohort (who-rated-what; concept answered if user has ≥2 rated items sharing that shelf):

| channel | answer-rate (mean) | median | notes |
|---|---|---|---|
| **Shelf concepts** (≥2 tagged rated items) | **0.076** | 0.024 | max 0.901; 46/1500 ≥0.5; 12/1500 ≥0.8 |
| **Items** (all, frac users who rated) | 0.0005 | 0.0001 | most items rated by a tiny fraction |
| **Items** (top-600 popular pool) | 0.012 | — | min 0.0035 |

**Concepts are far more answerable than items** — ~150× the all-item mean and **~6× the popular-item pool**
(0.076 vs 0.012). The Paper-B concept-answerability advantage **replicates cross-domain**. Absolute levels
are much lower than ML-25M (concepts 0.076 vs 0.945) because this is a **realistic sparse cohort** (median 6
ratings/user) rather than heavy raters (726) — users have simply read few books, so few shelves reach the
≥2-item bar. The most-answerable shelves are the broad genre labels (`mystery` 0.90, `thriller` 0.88,
`crime` 0.87, `suspense`, `contemporary`). Relative advantage is regime-independent; absolute magnitude
tracks cohort density.

## 10. Disk status + durable artifacts
- Free space: **37 GB before → 36 GB after** (well under the 6 GB peak-footprint cap; total added ~1.8 GB).
- Raw data (gitignored, academic-use-only, no redistribution) + derived artifacts, all under
  `casper/.cache/goodreads/`:
  - `goodreads_books_mystery_thriller_crime.json.gz` (231 MB, gzip-verified)
  - `goodreads_interactions_mystery_thriller_crime.json.gz` (1.33 GB, gzip-verified)
  - `base.npz` (134 MB: uu/ii/rr, cnt, mu, split, keepB book-id map, H5)
  - `Q_svd.npy` (52613×64), `bi_svd.npy`
  - `books_meta.npz` (titles+descriptions), `item_sbert.npy` (52613×384)
  - `concepts.npz` (1500 shelf concepts: names, membership, Ac centroids)
  - `enc_v1_gr.pt` (best-val encoder ep7), `enc_v1_gr_state.pt` (resume), `enc_v1_gr_peak.txt`
- `.gitignore` (casper repo) now contains `.cache/goodreads/` (verified with `git check-ignore`).
- Scripts (committed): `casper/scripts/paper2/gr_prep.py`, `gr_build_svd.py`, `gr_books.py`,
  `gr_sbert.py`, `gr_train_enc.py`, `gr_health.py`, `gr_effrank_answer.py`.

---

## Headline numbers
- Counts: **52,613 items / 568,218 users / 10.77 M explicit ratings; median 6 ratings/user** —
  genuinely cold-start-realistic. Top-1% items hold 35.5% of ratings (Gini 0.759).
- Ruler (pre-stated): **K = 142** (= round(0.0027×52,613)); NDCG@10 alongside; Cremonesi head-33% tail.
- Gate: monotone no-harm PASS; mechanics PASS (encoder > ridge > MOSTPOP); headroom
  **+0.0104 @142 / +0.0118 @10 = WEAK/FAIL-band** (single-genre homogeneity + sparse explicit signal,
  not fixable by more profile: headroom flat in profile size). **Phase-2 battery on this ruler would
  chase a ~0.01 prize — re-scope or run Phase 2 as the concept-necessity story.**
- Item-asking is **structurally starved** here (random/entropy asks hit 0 rated items; pop asks ~1/8) —
  the strongest cross-domain evidence yet that an answerable concept channel is necessary.
- Effrank: shelf-concepts **34.65/64** vs items **30.48/64** — the ML-25M genome low-rank concept
  finding does **not** transfer to folksonomy shelves (construction-specific, not universal).
- Answerability: concepts **0.076** vs popular-item pool **0.012** vs all items **0.0005** —
  the concept-answerability advantage replicates (~6× the popular pool, ~150× all items).
