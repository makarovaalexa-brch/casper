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

---

## PHASE 1D — COLD-COHORT re-gate (dated 2026-07-03; eval-only, `gr_cold_regate.py`)

**Motivation (stated before any policy result).** Phase-1's +0.010 average headroom hides a strong
gradient by profile size: **+0.0304 on 2–5-rating users** (n=76) shrinking toward zero for heavier
raters. Cold users are the thesis's target population and — per Paper A's visibility principle — the
policy battery needs an arena where the instrument can actually *see* preference differences. This
cohort choice is made **BEFORE** any phase-2 policy result, on gate-visibility grounds only.

**(1) Cold cohort.** Test/val users with **≤10 kept ratings** (pre-stated threshold). Cohort is
**large**, as expected from median 6: **65.1% of ALL 568,218 users** (369,807), **62.6% of test**
(313/500), 65.2% of val. Since ≥300 test users qualify at ≤10, the threshold is **NOT** relaxed
(the ≤15 fallback would have given 72.2% test / 361 users). Gate is evaluated on the **cold test**
users. Full-profile fold usable n = **88** (of 313; the rest lack ≥4 likes → no held disjoint target,
or <2 profile items after the held-half split); avg profile = **4.0 items**; n_tail = 59.

**(2) Health gate on the cold cohort (K=142 primary, +K=10; held-out disjoint targets):**

*(a) Full-profile fold:*
| model | @10 full | @10 tail | @142 full | @142 tail |
|---|---|---|---|---|
| MOSTPOP (q0) | 0.1522 | 0.0000 | 0.2409 | 0.0176 |
| ridge (λ=5)  | 0.1604 | 0.0049 | 0.2470 | 0.0215 |
| **ENCODER**  | **0.1749** | 0.0000 | **0.2576** | 0.0161 |

- **Encoder headroom over MOSTPOP: @142 full +0.0167, @10 full +0.0227, @142 tail −0.0015.**
- Ridge headroom @142 full +0.0062. Ordering encoder > ridge > MOSTPOP holds on full (mechanics
  healthy); on tail ridge again edges the encoder (head-chasing under the popb decoder), as in phase 1.
- The cold cohort **does** lift the headroom (+0.0167 vs the full-cohort +0.0104), consistent with the
  motivating profile-size gradient — but it **dilutes** the very-cold +0.0304 (2–5 raters) with the
  6–10 bucket and lands **below the pre-stated substantiality band (≥ +0.025)**.

*(b) Monotone no-harm concept-channel elicitation (wasted-turn; graded geometric answers; q0/8/20):*
| concept selector | q0 @142 | q8 @142 | q20 @142 | ans_tok q8 / q20 | no-harm |
|---|---|---|---|---|---|
| random  | 0.2409 | 0.2407 | 0.2405 | 0.19 / 0.57 | OK |
| entropy | 0.2409 | 0.2450 | 0.2443 | 0.14 / 0.48 | OK |
| pop     | 0.2409 | 0.2357 | 0.2367 | 6.89 / 13.88 | HURTS (exclusion + head-chasing, as in phase 1) |

- **Monotone no-harm holds for random + entropy** (revealing never hurts) — instrument sound. The pop
  selector "hurts" purely from the held-out exclusion artifact (asking popular items removes popular
  distractors), same known confound as §7b; not clean evidence either way.

**(3) Concept-channel elicited gain on the cold cohort.** Entropy concept-asking gains
**+0.0041 @q8 / +0.0035 @q20** (@142 full) = **~21% of the encoder full-profile headroom (+0.0167)**;
tail is flat-to-slightly-negative (−0.0013). **Crucially the answerable-token counts are tiny:** entropy
folds **0.14 answered concepts by q8, 0.48 by q20** (random 0.19 / 0.57). Cold users have ≤10 ratings, so
almost no shelf reaches the ≥2-tagged-item answerability bar → **even the concept channel is
answerability-starved on this cohort**, and asks mostly waste the turn. The absolute realizable prize
is **+0.003–0.004 NDCG@142 with <1 answered concept per 8 turns.**

**(4) VERDICT: FAIL (headroom below the pre-stated band).** Encoder full-profile − MOSTPOP =
**+0.0167 @142** (< +0.025 target); monotone no-harm PASS. The cold cohort **raises** headroom vs the
full cohort but **does not clear the substantiality bar**, and the answerable concept channel it was
meant to feed is itself starved (0.14 answered tokens @q8) — so the arena still does not give the
instrument enough to see policy differences. **Per the pre-stated rule, this is a FAIL → the phase-2
battery is NOT a go on this cohort.**

**Recommendation — the multi-genre composite path.** The failure mode is now doubly confirmed:
single-genre homogeneity gives an intrinsically small (~0.017) personalization prize, and the ≤10-rating
sparsity thins *both* channels (item channel dead in phase 1; concept channel to 0.14 answered tokens
here). Taking the cohort colder does not fix it — it trades a little more headroom for even sparser
answerability. **Recommend building a multi-genre Goodreads composite** (e.g. union of several byGenre
slices — mystery/thriller/crime + romance + fantasy + …) to restore **cross-genre taste
differentiation** (the axis popularity cannot explain within a single genre), while **keeping the
concept/shelf answerability channel** that already replicates. That arena should widen the encoder −
MOSTPOP headroom back toward a substantial band and give more shelves that clear the ≥2-item bar, so
the policy battery measures a real prize with a live answerable channel. Alternatively, if staying
single-genre, re-scope phase 2 explicitly to the **concept-necessity / answerability** story (not a
policy-efficiency ruler), as §7 already recommended.

*(Script: `casper/scripts/paper2/gr_cold_regate.py`; eval-only, no training, no commits. Cohort chosen
before any policy run.)*

---

## PHASE 1E — MULTI-GENRE COMPOSITE (dated 2026-07-04)

**Motivation.** Phases 1 and 1D failed the headroom gate twice on the single Mystery/Thriller/Crime
slice (+0.0104 full-cohort, +0.0167 cold — both below the ≥+0.025 substantiality band). Diagnosis:
*within one genre, popularity explains most of the ranking and cross-user taste barely varies*. The
composite is the fix attempt: **union of 3 byGenre slices — mystery+thriller+crime ∪
fantasy/paranormal ∪ history/biography** — to inject **cross-genre taste differentiation** (an axis
popularity cannot explain within a genre) while keeping the shelf-answerability channel. Explicit
ratings only (like ≥4 / dislike ≤2), item filter ≥20 ratings. Prep (Q_svd_comp/bi_svd_comp/base_comp/
books_meta_comp/concepts_comp) was built by a prior run; this phase does the split, encoder, gate,
effrank/answerability. Scripts: `gr_comp_train_enc.py`, `gr_comp_health.py`, `gr_comp_effrank_answer.py`.

### 1E.1 Composite counts + the cross-genre differentiation signal
| quantity | value |
|---|---|
| **items** (≥20-rating filter, union) | **188,867** (mystery 52,613 · fantasy/paranormal 77,889 · history/biography 58,365) |
| **users** (full set) | **768,746** |
| **ratings kept** (explicit) | **46,143,044** (per slice: 10.77M · 24.71M · 10.66M) |
| ratings/user | **median 24**, mean 60.0, p25 9, p75 64, p90 146, max 8165 |
| mean rating μ | 3.9717 |
| **cross-genre share: users with ratings in ≥2 of 3 slices** | **83.4%** (641,092/768,746) |
| users spanning all 3 slices | **59.4%** (456,473); exactly-1 slice 16.6% |

The composite delivers exactly the intended differentiation signal: **83.4% of users span ≥2 genres,
59.4% all three**, and the cohort is **~4× denser** than the single slice (median 24 vs 6 ratings/user).
If cross-genre taste were going to rescue collaborative headroom over popularity, this is the arena.

### 1E.2 Split (unchanged convention)
rng(0) shuffle, last 1000 → **500 val / 500 test**, rest (**767,746**) train. Verified byte-identical to
the rng(0) convention used in phases 1/1D. (Splits already materialised in `base_comp.npz`.)

### 1E.3 THE RULER (pre-stated per the phase-1E task spec, BEFORE any eval below)
- **Primary K = round(0.0027 × 188,867) = 510.** NDCG@10 reported alongside. Tail = Cremonesi head-33%.
- V1 decoder sc = popb + Q·u (popb = log(cnt+1)); ridge λ=5; MOSTPOP q0 = popb. Held-out disjoint targets.
- **Gate = encoder full-profile NDCG@510 − MOSTPOP ≥ +0.025 AND monotone no-harm** (random+entropy
  concept selectors, reveals {0,8,20}).

### 1E.4 V1 encoder training
Same recipe as `gr_train_enc.py` (d=64 attention fold-in over (Q[item], residual) tokens; frozen popb+Q·u
decoder; IPS-weighted BCE over unrevealed like-set; k=1..12 reveals; Adam 1e-3). Batch **512**.
**Compute-budget note:** the full 514,344 train users (≥13 ratings) give a ~3,160 s/epoch pass — over the
10-min foreground cap — so each epoch trains on a **rotating deterministic (rng0) 90k-user window**
(covers all users across epochs), ~555 s/epoch, chunked+resumable. Val-selected on the 500-user val
cohort, full-profile fold, NDCG@510 (n=418 usable):

| ep | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| val NDCG@510 | .2889 | .2893 | .2899 | **.2904** | .2897 |

Plateaued by ep4; ep5 declined. **Best-val = ep4 (0.2904)** → `enc_v1_grcomp.pt`
(+ `_state.pt` resume, `_peak.txt`). Used by the gate.

### 1E.5 Health gate — VERDICT: **FAIL (headroom +0.0106 full-test / −0.0003 cold, both < +0.025)**
`gr_comp_health.py`, held-out disjoint targets, K=510 primary.

**(a) FULL-TEST cohort, full-profile fold (n=423, n_tail=392, avg_profile=44.1):**
| model | @10 full | @10 tail | @510 full | @510 tail |
|---|---|---|---|---|
| MOSTPOP (q0) | 0.1973 | 0.0038 | 0.2650 | 0.0373 |
| ridge (λ=5)  | 0.1923 | 0.0222 | 0.2639 | 0.0498 |
| **ENCODER**  | **0.2058** | 0.0104 | **0.2756** | 0.0457 |

- **Encoder headroom over MOSTPOP: @510 full +0.0106, @10 full +0.0085, @510 tail +0.0084.**
- Ridge headroom @510 full is **−0.0011** (encoder > MOSTPOP > ridge on full; on tail ridge leads, as in
  phases 1/1D — head-chasing under the popb decoder). Encoder mechanics healthy; **headroom is the same
  ~0.010–0.011 band as the single-genre slice** despite 4× density and cross-genre membership.

**(b) CONCEPT-channel elicitation (FULL-TEST; wasted-turn, graded geometric answers, q0/8/20 @510 full):**
| selector | q0 | q8 | q20 | ans_tok q8 / q20 | no-harm | elicit gain @q8/@q20 |
|---|---|---|---|---|---|---|
| random  | 0.2650 | 0.2646 | 0.2657 | **1.01** / 2.62 | OK | −0.0004 / +0.0007 |
| entropy | 0.2650 | 0.2675 | 0.2682 | **0.96** / 2.27 | OK | +0.0025 / +0.0032 |
| pop     | 0.2650 | 0.2664 | 0.2665 | 6.76 / 15.88 | OK | +0.0014 / +0.0015 |

- **Monotone no-harm PASS (random+entropy).** Concept channel is now genuinely answerable
  (random/entropy fold ~1 token by q8, pop ~6.8) — a real improvement over the single slice's ~0 tokens.
  But the realizable elicited prize is **tiny (+0.0025 @q8 entropy)** and the informative selectors sit
  **right at ~1 answered token by q8 (0.96–1.01), NOT clearly above 1** — the bar for a meaningful
  phase-2 policy battery is not cleanly met.

**(c) COLD cohort (≤10 ratings; n=57 usable, n_tail=42, avg_profile=3.9):** 132/500 test users qualify.
| model | @10 full | @510 full | @510 tail |
|---|---|---|---|
| MOSTPOP (q0) | 0.0844 | 0.1924 | 0.0159 |
| ridge | 0.0792 | 0.1894 | 0.0214 |
| **ENCODER** | 0.0899 | **0.1921** | 0.0223 |
- **Encoder headroom @510 = −0.0003** (encoder ties MOSTPOP on full; +0.0064 tail). Concept entropy
  elicits +0.0022 @q8 but folds only **0.11 answered tokens** — the concept channel is answerability-
  starved again on very-cold users, and pop-asking **HURTS** (−0.0082, the held-out exclusion artifact).

**(d) GATE VERDICT: FAIL.** FULL-TEST headroom **+0.0106** and COLD **−0.0003**, both far below +0.025;
monotone no-harm PASS. **Third consecutive Goodreads headroom failure** (single-genre +0.010/+0.017,
composite +0.011/−0.000). The multi-genre composite added real cross-genre coverage and revived the
concept channel to ~1 answered token, **but it did NOT widen the collaborative personalization prize** —
across the union catalogue, shared popularity structure still explains the @510 ranking, and per-user
taste over-and-above popularity remains ~0.01 NDCG. Denser profiles (avg 44 items) did not help, so the
weakness is intrinsic to the domain's popularity-dominated signal, not to profile starvation.

### 1E.6 Effective rank + answerability (composite)
`gr_comp_effrank_answer.py`, participation ratio (D=64):
| space | composite effrank (uncentered / centered) | single-genre MTC | ML-25M genome |
|---|---|---|---|
| Shelf concepts (1500 shelves) | **7.41 / 8.00** | 34.65 / 35.68 | 4.11 |
| Pool items (top-600 popular) | **12.30 / 12.50** | 30.48 | 9.09 |

- **The single-genre inversion (concepts 34.65 > items 30.48) does NOT survive the composite:** here
  **concepts 7.41 < items 12.30**, back to the MovieLens ordering (concepts below items). Top-1 concept
  direction alone holds **33.5%** of variance — the genre axis dominates the folksonomy shelf geometry
  once three genres are pooled. So the phase-1 "folksonomy concepts are high-rank" finding was
  **construction-specific to the single-genre Q**, not a stable cross-domain law.
- **Answerability (full realistic cohort):** concepts **mean 0.226** (median 0.126, max 0.997; 217/1500
  ≥0.5; 69 ≥0.8) vs **items-pool 0.0296** vs **all-items 0.0003**. Concepts are **~7.6× the popular-item
  pool and ~750× all items** — the concept-answerability advantage replicates strongly, and absolute
  levels are far higher than the single slice (0.226 vs 0.076) thanks to the denser cohort. Most
  answerable: `fiction` 0.997, `novels` 0.987, `contemporary` 0.986, `mystery` 0.982, `fantasy` 0.966,
  `romance` 0.962 — broad genre/era labels, as before.

### 1E.7 Pre-registered predictions (PREREG_GOODREADS_PREDICTIONS.md) — retrospective
The prereg was written on the single-genre effrank basis (concepts 34.65 ≫ items 30.48); the composite
**changes that basis** (concepts 7.41 < items 12.30), so predictions (1)–(3), which were conditioned on a
rank-35 concept menu, no longer apply to this arena — recorded, not reinterpreted. Prediction (4)
**holds**: concept answerability advantage persists (0.226 vs 0.0003) and the selection hierarchy
entropy ≥ random ≥ pop holds on the realizable gain (entropy +0.0025 @q8, the only clean positive).

### 1E.8 Disk + durable artifacts
- Free space **5.5 GB** (above the 5 GB floor). Composite artifacts under `casper/.cache/goodreads/`:
  `base_comp.npz`, `Q_svd_comp.npy`, `bi_svd_comp.npy`, `books_meta_comp.npz`, `concepts_comp.npz`,
  `enc_v1_grcomp.pt` (best-val ep4), `enc_v1_grcomp_state.pt`, `enc_v1_grcomp_peak.txt`.
- Scripts: `gr_comp_train_enc.py`, `gr_comp_health.py`, `gr_comp_effrank_answer.py`.

---

## FINAL VERDICT (2026-07-04): Goodreads FAILS as a policy-efficiency ruler — conclude as the concept-necessity story; point the cross-domain bet at Steam

Three health-gate attempts (single-genre, cold re-gate, multi-genre composite) all land in the
**+0.010–0.017 / −0.000 headroom band — none clears the pre-stated ≥+0.025 substantiality bar.** The
composite specifically **falsifies the hypothesis** that single-genre homogeneity was the cause:
adding cross-genre differentiation (83% of users span ≥2 genres) and 4× density **did not widen the
prize** (+0.0106, statistically indistinguishable from the single-genre +0.0104). The domain is simply
**popularity-dominated**: on this book catalogue, per-user collaborative taste over-and-above popularity
is ~0.01 NDCG@K regardless of genre breadth or profile depth, so a phase-2 policy battery would be
optimising within a ~0.01 envelope — not a publishable efficiency ruler.

**What Goodreads DID establish (keep this):** the **concept-necessity / answerability** story is strong
and replicates cleanly. Item-level elicitation is structurally dead (0.0003 answer-rate; random/entropy
item-asks hit ~0 rated items) while the shelf-concept channel is **~750× more answerable** (0.226) and
monotone-no-harm — the clearest cross-domain evidence that an **answerable concept channel is necessary,
not merely nicer**, on realistic sparse catalogues. That is the Goodreads contribution.

**Recommendation — GO to Steam for the cross-domain policy-efficiency demonstrator.** Reasoning:
1. **A live, high-variance taste axis popularity cannot explain.** Steam playtime/genre-tag profiles are
   far less popularity-collinear than book ratings (a handful of blockbusters do not dominate every
   user's library the way head books dominate reading), so the encoder − MOSTPOP headroom should clear
   the substantiality band that Goodreads cannot.
2. **A rich, genuinely answerable concept vocabulary already exists** (Steam user-generated tags —
   thousands of them, densely applied), giving the concept channel the "clearly >1 answered token by q8"
   that Goodreads only reached marginally (0.96–1.01).
3. **Implicit but abundant per-user signal** (ownership + playtime) sidesteps the explicit-rating
   sparsity (median 6–24) that thinned both channels here.
Retain Goodreads solely as the **concept-necessity/answerability** datapoint (§7b, §9, §1E.6); do **not**
run a phase-2 policy battery on it. **PHASE 2 IS NOT A GO on Goodreads.**

*(Scripts committed under `casper/scripts/paper2/gr_comp_*.py`; eval + one encoder train, no git
commits, no background monitors, foreground chunked.)*

### 1E.9 Independent replication of the gate (2026-07-04, second run)
A parallel recovery run independently re-ran the full gate (`gr_health_comp.py`, same ruler K=510,
same protocol) against a later val-selected checkpoint from the same training ladder (ep6-best
0.2900 vs the ep4-best 0.2904 above; ladder continued to ep9, val plateau 0.2889–0.2904 from ep3 —
peak log `enc_v1_grcomp_peak.txt`; gate checkpoint frozen as `enc_v1_grcomp_GATE.pt`):
- FULL-TEST (n=423): MOSTPOP 0.2650 / ridge 0.2639 / **encoder 0.2757** @510 full →
  **headroom +0.0107** (@10 +0.0094; tail +0.0094); ridge −0.0011. Matches §1E.5 to ±0.0001.
- **Headroom by profile size (@510 full): 2–5 +0.0070 (n=53), 6–15 +0.0102 (n=116), 16–40 +0.0103
  (n=126), 41+ +0.0130 (n=128)** — flat ~0.01 across a 20× profile-depth range, confirming §1E.5(d):
  the small prize is intrinsic, not profile starvation.
- Concept channel (entropy): +0.0022 @q8 / +0.0034 @q20, ans_tok 0.96 / 2.27, monotone no-harm OK —
  matches. COLD (≤10; n=57): headroom **−0.0030** (vs −0.0003 above; the delta is checkpoint noise —
  both are ≈0 and far below the band). Effrank/answerability re-run identical (7.41/12.30; 0.226).
- **Verdict unchanged: WEAK/FAIL — phase 2 is NOT a go on Goodreads.** Checkpoint-to-checkpoint
  variance (~0.003 on a 57-user cold cohort, ~0.0001 on the full cohort) does not move the verdict.
- Artifact addendum: composite SBERT content embeddings `item_sbert_comp.npy` (188,867×384;
  52,613 mystery rows reused, 136,254 new items embedded) built by resumable `gr_sbert_comp.py`
  (partial-progress file `item_sbert_comp_part.npz`; rerun the script to resume/finish if absent).
