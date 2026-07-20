# ML-25M Replication — Phase 1b Result (FULL-DATASET protocol)

Date: 2026-07-03. **Corrects Phase 1**, which accidentally trained on a legacy 10k-heaviest-raters
subsample (`data/processed/ratings_subset.pkl`, min 490 / median 726 ratings per user). Phase 1b uses the
**official full MovieLens-25M `ratings.csv`** with a standard by-user split — no activity-based sampling.
STOP point: instrument-health verdict (V1 encoder + Phase-2 battery remain gated on it).

Build/eval scripts (new, full-data; phase-1 subset scripts kept intact):
`scripts/paper2/ml25m_prep_full.py`, `ml25m_train_svd_full.py`, `ml25m_eval_full.py`, `ml25m_concepts_full.py`.
Durable artifacts overwrite the phase-1 (saturated) cache under `data/movielens/.cache/ml25m/`.

---

## 1. Disk status
- Downloaded official `ml-25m.zip` (249.8 MB) → extracted `ratings.csv` only (646.8 MB) → deleted zip.
- Intermediate `prep.npz` (568 MB) + `svd_state.npz` (45 MB) deleted after training/eval (regen in ~12 s).
- **37.4 GB free at finish** (started ~21 GB; freed headroom during run). Never near the 4 GB floor.
- `ratings.csv` (647 MB) kept under `data/ml25m/` as the durable full-data input.

## 2. Full-dataset protocol & counts
- **Source:** official `ratings.csv` — **25,000,095 ratings**.
- **Item filter (stated preprocessing, not sampling):** keep items with **≥20 total ratings** → **18,430 items**;
  **24,810,483 ratings** retained (99.2%). Dense-remapped.
- **Users:** ALL **162,541** users. Split by user via fixed `rng(0)` shuffle: hold out the **last 1,000**
  → **500 val / 500 test**; remaining **161,541 train** (mirrors ML-1M `te[:300]/te[300:]`). Cold-start = the
  reveal protocol (0→T answers).
- **Like** = rating ≥ 4.0. **Factors:** biased SVD (Koren09) mini-batch SGD, **D=64 LAMF=0.05 LR=0.01 EP=15**
  — identical recipe to ML-1M `instrument_svd.py`. Train RMSE 0.863 → **0.7534** (monotone, converged).
- **Rating-count distribution (per user, after item filter):**
  **median 71**, mean 152.6, min 6, max 14,040 (p10 24 · p25 36 · p75 162 · p90 352).
  **This is a realistic cohort (median 71, NOT 726).** The heavy-tail mean/p90 reflect a few power users,
  but the median user is light — exactly the cold-start-relevant regime Phase 1 lacked.

## 3. Instrument health — VERDICT: **mechanically healthy, but headroom gate FAILS on the full cohort too → Phase 2 GATED**

Ridge fold-in acceptance (`ml25m_eval_full.py`; val-tuned β=4 λ=5), full-catalogue NDCG@10, n=500 test.

**(a) Reveal monotonicity, standard `rel = all likes − asked` (q0 = MOSTPOP = 0.352):**
| selector | q0 | q4 | q8 | q12 | q16 | q20 | RecT | mono |
|---|---|---|---|---|---|---|---|---|
| rmva   | 0.351 | 0.352 | 0.352 | 0.355 | 0.355 | 0.356 | 0.072 | OK |
| helf   | 0.351 | 0.352 | 0.353 | 0.354 | 0.356 | 0.355 | 0.072 | OK |
| entropy| 0.351 | 0.351 | 0.351 | 0.351 | 0.351 | 0.351 | 0.069 | OK |
| random | 0.351 | 0.351 | 0.351 | 0.351 | 0.351 | 0.352 | 0.069 | OK |
| pop    | 0.351 | 0.299 | 0.284 | 0.267 | 0.255 | 0.228 | 0.047 | (rel-shrink artifact) |

`pop`'s decline is the standard "remove-asked-from-rel" artifact (it spends its own popular targets), **not**
instrument harm — confirmed by the held-out no-harm test below where `pop` rises.

**(b) Held-out no-harm (fixed disjoint targets — the correct no-harm / headroom ruler), q0 = 0.2685:**
| selector | q0 | q4 | q8 | q12 | q16 | q20 | no-harm |
|---|---|---|---|---|---|---|---|
| rmva   | 0.268 | 0.270 | 0.270 | 0.271 | 0.271 | 0.272 | OK |
| helf   | 0.268 | 0.269 | 0.270 | 0.272 | 0.274 | 0.282 | OK |
| random | 0.268 | 0.268 | 0.268 | 0.268 | 0.268 | 0.269 | OK |
| pop    | 0.268 | 0.336 | 0.390 | 0.434 | 0.466 | 0.494 | OK (rises) |

Revealing **never hurts** (all monotone); RMSE converges. **Instrument is mechanically sound.**

**(c) Full-profile fold reference (same held-out disjoint targets):** NDCG@10 = **0.2773**, Rec@10 0.094.

**(d) The headroom problem (why Phase 2 stays gated):**
- **Headroom = full-profile (0.2773) − MOSTPOP q0 (0.2685) = +0.009** (ML-1M ref: **+0.10**).
- **Smart-selector elicitation gain is negligible:** RMVA @8 = **+0.0016** (all-likes); RMVA @20 = +0.004,
  HELF @20 = +0.014 (held-out). Only `pop` moves a lot (+0.226 @20) — but that is popularity **reveal**
  exploiting popularity-skewed held-out likes, i.e. the **saturation signature**, not genuine elicitation.
- **Tail NDCG@10 = 0.000 everywhere** (including full-profile): with the β=4 popularity prior on an
  18k-item catalog, the linear fold never lifts a non-popular (rank>500) like into the top-10.

**KEY CORRECTIVE:** Phase 1 attributed the tiny headroom to the heavy-rater subsample and predicted a
realistic resample would restore it. **Phase 1b refutes that.** On the full, realistic cohort (median 71
ratings/user) the headroom is **still only +0.009** and smart-selector gain **still ~+0.002**. The
popularity-saturation of ML-25M full-catalogue NDCG@10 under this **linear biased-SVD ridge-fold + popularity
prior** instrument is **regime-intrinsic, not a subset artifact.** Popularity nearly solves the task.

**Gate decision:** instrument passes health (monotone · no-harm · RMSE-converged) but **FAILS the headroom
gate** (full-profile − MOSTPOP = +0.009 ≪ ML-1M +0.10; RMVA @8 +0.002; tail = 0). **Phase 2 (V1 encoder +
policy/ladder battery) remains GATED.**

Caveats / phase-2 levers (do NOT run without addressing at least one):
1. **Popularity-prior strength is not tuned below β=4** (the val sweep floor); β<4 would weaken the prior and
   could surface tail/personalisation signal — the most likely lever to open headroom on this dataset.
2. **Metric:** full-catalogue NDCG@10 with a popularity prior is popularity-dominated at this catalog size; a
   tail-restricted or popularity-debiased ruler is where elicitation could earn its keep.
3. **Instrument capacity:** the ridge fold is the *linear reference*. The V1 neural attention set-encoder
   (deferred) may extract nonlinear personalisation, but on the linear reference the headroom is absent.

The V1 neural encoder was **not** trained (as in Phase 1): the linear ridge fold establishes the health/headroom
reference, and the gate verdict — small linear headroom — does not depend on the neural encoder.

## 4. Effective rank — concepts vs items (clean full-data cross-dataset datapoint)

Participation ratio of unit direction vectors, D=64 (`ml25m_concepts_full.py`):
| space | ML-25M full (uncentered) | centered | Phase-1 subset (caveated) | ML-1M ref |
|---|---|---|---|---|
| **Genome concepts** (1031 tags, ≥30 items, `Ac` centroid) | **3.99 / 64** | 3.92 | 4.11 | 2.25 (learned `Ec`) |
| **Pool items** (top-600 popular `Q`) | **8.35 / 64** | 8.16 | 9.09 | 27.1 |

- **Concepts occupy a markedly lower-rank subspace than items (3.99 < 8.35)** — the thesis prediction
  (concept channel saturates in few directions) **holds cross-dataset and is now clean** (full-data Q).
  Concept top-5 dirs explain 88% of variance vs items' 69%.
- Values are close to the phase-1 caveated numbers (concepts ~4, items slightly lower ~8 on full data), so the
  qualitative concept < item gap is **robust to the cohort correction**. (Concept `Ac` is the centroid *init*
  of ML-1M's learned `Ec`=2.25; a learned encoder on 25M would likely be lower still. Item effrank 8.35 vs
  ML-1M 27.1 reflects the different SVD/catalog regime, not the subset.)

## 5. Answerability — concept vs item channel (B's cross-dataset row, realistic cohort)

| channel | answer-rate mean | median | notes |
|---|---|---|---|
| **Concepts** (frac users with ≥2 rated tagged items, 3000-user sample) | **0.574** | 0.596 | 600/1031 ≥0.5; 303/1031 ≥0.8; max 1.000 |
| **Items** — all | **0.008** | 0.0008 | most items rarely rated |
| **Items** — top-600 popular pool | 0.120 | (min 0.040) | the entity set the policy asks from |

- **Concepts are ~72× more answerable than items overall (0.574 vs 0.008)** and **~4.8× more than even the
  popular item pool (0.574 vs 0.120)**. The Paper-B concept-answerability advantage **replicates on ML-25M**.
- **As predicted, the absolute concept rate drops for realistic light users** (0.574 here vs **0.945** on the
  phase-1 power-user subset): a median-71-ratings user holds an opinion on ~57% of broad tags, not ~95%.
  Most-answerable concepts are broad affective/structural tags ("great ending", "story", "good soundtrack",
  "dialogue", "storytelling", "great acting"). The **advantage direction and magnitude-ratio are regime-robust**;
  the absolute level is cohort-dependent (this is the honest realistic-cohort number for the B story).

---

## Durable artifacts (`data/movielens/.cache/ml25m/`, tag `ml25m_*`, overwritten with full-data)
- `Q_svd.npy` (18430×64 item factors), `bi_svd.npy` (item biases) — train RMSE 0.7534.
- `meta.npz` (uu/ii/rr, cnt, mu, ni/nu, train/val/test split, keepI catalog ids) — full-data.
- `Ac_concept.npy` (1031×64 concept centroids), `ctags_concept.npy`.
- Input: `data/ml25m/ratings.csv` (full 25M). Scripts under `scripts/paper2/ml25m_*_full.py`.
- (`prep.npz`/`svd_state.npz` deleted post-run to reclaim disk; regen: `ml25m_prep_full.py` ~12 s +
  `ml25m_train_svd_full.py` ~36 min in 3-epoch chunks.)

## Headline numbers
- **Counts:** 24.81 M ratings kept, **18,430 items** (≥20), **162,541 users** (train 161,541 / val 500 / test 500);
  **median 71 ratings/user** (realistic — vs 726 in the phase-1 subset).
- **Health gate: FAIL.** Instrument mechanically healthy (monotone · no-harm · RMSE 0.7534) BUT
  full-profile − MOSTPOP = **+0.009**, RMVA @8 = **+0.002**, tail = **0.000** → popularity-saturated;
  **Phase 2 GATED.** Corrects Phase 1: the small headroom is **regime-intrinsic, not a subset artifact.**
- **Effrank:** concepts **3.99/64**, items **8.35/64** (concept < item confirmed, clean full-data).
- **Answerability:** concepts **0.574** vs items **0.008** (all) / **0.120** (pool) — advantage replicates
  (~72×); absolute concept rate drops from 0.945 (power users) as predicted for the realistic cohort.
