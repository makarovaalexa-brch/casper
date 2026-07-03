# ML-25M Replication Campaign — Phase 1 Result

Date: 2026-07-03. Scope: port the CASPER pipeline to MovieLens-25M; build data artifacts, train
the fold-in instrument, run the mini acceptance suite, measure concept effective-rank and
answerability. **Phase 2 (policy/ladder) is gated on the instrument-health verdict below.**

All ML-25M artifacts live under `casper/data/movielens/.cache/ml25m/` (tag `ml25m_*`), durable.
Build scripts: `casper/scripts/paper2/ml25m_build_svd.py`, `ml25m_concepts.py`.

---

## 1. Disk status
- **9.3 GB free** at start (NOT the ~27 GB assumed historically — `C:` is 476 GB, 99% used).
- Phase-1 artifacts added ~112 MB (`meta.npz` 107 MB + `Q_svd.npy` 4.3 MB + `bi_svd.npy`/`Ac_concept.npy` small).
- **~9.2 GB still free.** Adequate for Phase-1; the full 25M `ratings.csv` (~650 MB) is **absent** —
  only the genome files + a pre-made ratings *subset* are on disk (see below). Keep an eye on headroom
  before any large cache in Phase 2.

## 2. What ML-25M data is present (inventory)
Under `casper/data/movielens/`:
- `genome-scores.csv` (415 MB, full Tag Genome, ~14 M rows), `genome-tags.csv` (1128 tags),
  `movies.csv` (62 423 movies), `links.csv`, `README.txt` (confirms ml-25m: 25 M ratings, 162 541 users).
- **No `ratings.csv`.** The collaborative signal is instead a pre-made subsample:
  `casper/data/processed/ratings_subset.pkl` (364 MB) = **9 102 115 ratings, 10 000 users, 57 606 items**,
  0.5–5.0 scale. These 10 000 users are **heavy raters** (min 490, median 726, max 32 202 ratings each).
- Cached SBERT content embeddings for all 62 423 movies exist
  (`.cache/movie_embeddings_62423_4b8e49f7.pkl`, 96 MB) — not needed here (concepts come from the genome).

## 3. Subsample protocol (decided)
- **Users:** all 10 000 in the subsample; split 80/10/10 by seed-0 shuffle → **train 7995 / val 999 / test 1000**
  (mirrors `instrument_svd.py`). Cold-start is simulated by the *reveal protocol* (0→T answers), as in ML-1M.
- **Items:** catalog = items with **≥20 total ratings** (collaborative-signal filter; drops ~41 k ultra-rare
  items) → **16 709 items**, remapped dense. 8.92 M of the 9.10 M ratings retained.
- **Like:** rating ≥ 4.0. Median 313 likes/user.
- **Item factors:** biased SVD (Koren 2009) via mini-batch SGD, **D=64, LAMF=0.05, LR=0.01, EP=15** —
  identical recipe/hyperparams to ML-1M `instrument_svd.py`. Train RMSE 0.827→**0.728** (converged, healthy).

## 4. Instrument health — VERDICT: **mechanically healthy, but the subsample regime is popularity-saturated → Phase 2 GATED**

Ridge fold-in acceptance suite (`ml25m_build_svd.py`, tuned β=4 λ=5 on val), NDCG@10:

**(a) Monotonicity 0→20 reveals (held-out disjoint-target protocol — the correct no-harm test):**
| selector | q0 | q4 | q8 | q12 | q16 | q20 | monotone |
|---|---|---|---|---|---|---|---|
| rmva | 0.709 | 0.717 | 0.721 | 0.721 | 0.722 | 0.722 | OK |
| helf | 0.709 | 0.716 | 0.718 | 0.720 | 0.721 | 0.724 | OK |
| pop  | 0.709 | 0.757 | 0.800 | 0.834 | 0.862 | 0.882 | OK (rises) |
| random | 0.709 | 0.710 | 0.711 | 0.710 | 0.710 | 0.709 | OK |

Revealing **never hurts**; fold-in raises NDCG monotonically → instrument is **sound** (no-harm holds, RMSE
converges, attribute/fold mechanics work).

**(b) Full-profile fold reference:** NDCG@10 = **0.729** (fold ALL profile items → held-out likes).

**(c) The saturation problem (why Phase 2 is gated):** the standard `rel = all likes − asked` metric gives
**q0 (MOSTPOP) = 0.74** — saturated, because every user has ~300 likes among 16 709 items, so popularity
alone nearly solves the task. Under a **proper cold-start metric** (small held-out target of 10 likes vs the
full catalog, all other rated items excluded):

| target size | MOSTPOP q0 | RMVA @8 reveals | elicitation gain |
|---|---|---|---|
| 10 held-out likes | 0.147 | 0.153 | **+0.006** |
| 5 held-out likes | 0.115 | 0.117 | +0.002 |

q0 = 0.147 is a healthy magnitude (cf. ML-1M 0.310), confirming the 0.74 was purely a target-set-size
artifact. **BUT** the gap between MOSTPOP and full-profile personalization is only ~**+0.02** (ML-1M: +0.10),
and 8 RMVA reveals gain only **+0.006**. These 10 k heaviest raters are **popularity-typical** — their taste ≈
the popular prior — so there is **little elicitation headroom**. This is a regime problem, not an instrument bug.

**Gate decision:** the instrument passes health checks (monotone, no-harm, converged), but the *available
subsample* is unsuitable for the elicitation research question (no cold-start realism, tiny headroom).
**Before Phase 2**, build a cold-start-realistic split. Options, in order:
1. Obtain the full ML-25M `ratings.csv` and sample *typical / sparse* users (20–100 ratings) and/or
   tail-preferring users whose taste diverges from popularity — this is where elicitation earns its keep.
2. If ratings.csv cannot be added under the 9 GB disk limit: construct pseudo-cold users from the current
   pkl by (i) selecting users with above-median tail-like mass and (ii) capping profiles, so MOSTPOP is not
   already optimal. (Weaker — the user pool is still only heavy raters.)

*(The neural V1 attention set-encoder was not trained in Phase 1: the ridge fold-in already establishes
instrument health as the linear fold reference/control, and training the neural encoder on a
popularity-saturated split would not change the gate verdict. Deferred to the start of Phase 2, on the
cold-start-realistic split, where the fold-vs-ridge control becomes informative.)*

## 5. Effective rank of the ML-25M concept space (feeds the cross-dataset rank prediction)

Participation ratio (effective dimensionality) of unit direction vectors, D=64
(`ml25m_concepts.py`, same method as `effrank_concept_vs_item.py`):

| space | ML-25M effrank (uncentered) | ML-25M (centered) | ML-1M ref |
|---|---|---|---|
| **Genome concepts** (1031 tags, ≥30 items) | **4.11 / 64** | 4.03 | 2.25 |
| **Pool items** (top-600 popular) | **9.09 / 64** | 9.12 | 27.1 |

- **Concepts occupy a markedly lower-rank subspace than items (4.11 < 9.09)** — the thesis's qualitative
  prediction (concept channel saturates in few directions; item channel spans more) **holds cross-dataset**.
  Concept top-5 dirs explain 85% of variance vs items' 66%.
- Absolute values differ from ML-1M (concepts higher, items *much* lower). Two caveats: (i) ML-25M concepts
  here use the **centroid** construction `Ac` (mean Q over tagged items) = the *init* of ML-1M's learned `Ec`
  (2.25); a learned `Ec` on 25M would likely be lower. (ii) The low item effrank (9.09 vs 27.1) reflects the
  **concentrated power-user Q** — the saturated regime again compresses the factor space. Re-measure after the
  cold-start resample for the clean cross-dataset number.

## 6. Answerability stats (first cross-dataset datapoint for the B story)

| channel | answer-rate (mean) | median | notes |
|---|---|---|---|
| **Concepts** (≥2 tagged rated items) | **0.945** | 0.998 | 1008/1031 concepts ≥0.5; 931/1031 ≥0.8 |
| **Items** (all, frac users who rated) | 0.054 | 0.012 | most items rarely rated |
| **Items** (top-600 popular pool) | 0.466 | — | min 0.198 |

- **Concepts are ~17× more answerable than items overall (0.945 vs 0.054)**, and 2× more than even the popular
  item pool (0.945 vs 0.466). The concept-answerability advantage from Paper B **replicates on ML-25M, more
  strongly** (power users hold an opinion on nearly every tag). Most-answerable concepts are broad affective/
  structural tags ("twist ending", "period piece", "heartbreaking", "passionate", "prejudice").
- Regime-independent (based on who-rated-what, not the recommender), so this datapoint stands regardless of the
  cold-start resample.

---

## Durable artifacts (tag `ml25m_*`)
- `casper/data/movielens/.cache/ml25m/Q_svd.npy` (16709×64 item factors), `bi_svd.npy` (item biases)
- `casper/data/movielens/.cache/ml25m/meta.npz` (uu/ii/rr, cnt, mu, train/val/test split, catalog ids)
- `casper/data/movielens/.cache/ml25m/Ac_concept.npy` (1031×64 concept centroids), `ctags_concept.npy`
- Scripts: `casper/scripts/paper2/ml25m_build_svd.py`, `ml25m_concepts.py`

## Headline numbers
- Disk: 9.2 GB free (tight; no ratings.csv on disk).
- Instrument: monotone + no-harm + RMSE 0.728 (healthy) BUT regime popularity-saturated (full-profile beats
  MOSTPOP by only +0.02; 8 reveals +0.006) → **Phase 2 gated on a cold-start-realistic resample.**
- Effrank: **concepts 4.11/64, items 9.09/64** (concept < item confirmed; absolute values pending clean resample).
- Answerability: **concepts 0.945 vs items 0.054** (advantage replicates, stronger than ML-1M).
