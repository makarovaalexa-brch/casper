# Baseline-Replication Harness — Design Sheet (Paper A, obligation 1)

Created 2026-07-20. Snaps classic/SOTA recommender baselines to their **published ML-20M (Liang
strong-generalization split)** numbers, then runs the identical code on CASPER's **ML-25M** harness.
Code: `scripts/baselines/`. Results: `experiments/baselines/{snap_ml20m,ml25m}/`.

> **STATUS: code written + smoke-tested end-to-end. NOT launched.** A review gate sits before the full
> runs (HARD RULE 2). This sheet is the pre-registration + the honest wall-time/RAM extrapolation.

## Runtime environment (FLAG for review)
- The real interpreter is **system Python 3.12.7** (`C:\Program Files\Python312\python`), which has
  numpy 1.26 / scipy 1.17 / torch 2.3.1+cpu / pandas / sklearn. The **Poetry venv is empty** (no numpy)
  — the project's scripts are run with system python, not `poetry run`. `implicit==0.7.3` was therefore
  `pip install`ed into system python (verified `import implicit` OK) AND recorded in `pyproject.toml`.
  **If the reviewer wants a clean `poetry run` path, the poetry env must first be rebuilt** (`poetry install`).
- CPU-only, `torch.set_num_threads(os.cpu_count())`; `OPENBLAS_NUM_THREADS=1` set for `implicit`.

## Data / split fidelity
- `download_ml20m.py` → `data/ml-20m/ratings.csv` (idempotent, ~190 MB zip from grouplens.org).
- `liang_split.py` → `data/ml-20m/proc/` — **bit-faithful** to `dawenl/vae_cf` VAE_ML20M_WWW2018.ipynb
  (verified 2026-07-20 by fetching the raw notebook). Every parameter pinned with inline provenance:
  rating `> 3.5` (≡ `>=4` for half-star data), `min_uc=5, min_sc=0`, 10k val + 10k test held-out users,
  item vocab from **train users only**, per-user 80/20 fold-in/target (`test_prop=0.2`, users `>=5` items),
  **seed 98765** for both the user permutation and the tr/te split. Expected split (published): ~9.99M
  events, ~136,677 users, **20,108 items** (the script prints a warning if the item count differs).
- `metrics.py` — NDCG@100 / Recall@20 / Recall@50 are a **verbatim port of the vae_cf metric code**
  (argpartition top-k → argsort; gains `1/log2(2..k+1)`; IDCG = first `min(n_pos,k)` gains; fold-in items
  masked to −inf before ranking). NDCG@10 added with the same formula for the ML-25M bridge.

## ML-25M harness (obligation: reuse the canonical split, do not reinvent) — IDENTIFIED UNAMBIGUOUSLY
Reused verbatim from `scripts/signed_latent.py` (the eval that produced the canonical pbC set-encoder
**0.4946/0.3372** and reproduces RecVAE-d512 0.4998/0.3443):
`load_arena_base()` + `build_splits(base,seed)` (per-user half-split) + `cohort(base,SPL,"test")`
(= te(500) **minus the 300 quarantined study users**) + `ndcg10(...,tail=...)` (tail excludes top-33%
popular items/targets) + seed-avg over `SEEDS=[1,2,3,7,11]`. Train matrix = train users × 18,430 items,
binary r≥4 likes (`base tr_u/tr_i`) — the same signal `signed_latent`'s in-house EASE/RecVAE rulers use.
**Smoke check:** Most-Popular through this path returns full@10≈0.286 / tail@10≈0.057 (1 seed) — same
ballpark as the harness's canonical MOSTPOP 0.2522/0.0502. (Small gap is definitional: the canonical
ruler's `popb` uses **global** item counts incl. eval users + 5-seed avg, whereas our Most-Popular uses
**train-only** popularity — the methodologically correct strong-gen choice, kept identical across both
harnesses. Not a bug; documented.)

## Per-baseline fidelity, published target, wall-time, RAM

| # | Baseline | Source ported | Hyperparameters (provenance) | Published NDCG@100 (ML-20M) |
|---|---|---|---|---|
| 1 | **pop** | trivial | none | — (floor; must be beaten) |
| 2 | **itemknn** | standard cosine item-kNN | topk=200, cosine, no shrink (**standard default, not a paper constant — sweepable**) | ~0.36 class (approx) |
| 3 | **ease** | Steck WWW'19 closed form (Eq. 8) | **λ=500 verbatim** (paper ML-20M); float64 solve | **0.420** (R@20 .391, R@50 .521) |
| 4 | **ials** | `implicit` 0.7.3 ALS (Hu'08) | factors=200; **reg/alpha selected on val** (no paper constant — **tuned baseline, flagged**) | ~0.386 (approx) |
| 5 | **dae** | Mult-DAE, dawenl/vae_cf | [200,600,I] tanh, dropout .5, wd .01, lr 1e-3, ~200 ep, early-stop val | **0.419** |
| 6 | **multvae** | Mult-VAE^PR, dawenl/vae_cf | [200,600,I], anneal β 0→0.2 over 200k steps, lr 1e-3, ~200 ep | **0.426** (R@20 .395, R@50 .537) |
| 7 | **recvae** | ilya-shenbin/RecVAE | d200/h600, γ=0.005, 3enc:1dec, composite prior [3/20,3/4,1/10], dropout .5, lr 5e-4, ~50 ep | **0.442** (R@20 .414, R@50 .553) |
| 8 | **edlae** | Steck NeurIPS'20 full-rank, Eq. 8/9 — **Status: REIMPLEMENTED** (no official repo found; equations extracted from the NeurIPS PDF) | full-rank, full-emphasis (b=0): Λ=(p/q)·diag(G), C=(G+Λ)⁻¹, B=I−C·dMat(1/diag C); **p selected on val** (paper reports no single ML-20M p — **flagged deviation**) | ~0.427 (approx, ties RecVAE class) |

**Snap gate:** `run_snap_ml20m.py` prints PASS/FAIL vs the target at **±0.005 NDCG@100** for the exact
anchors (ease, dae, multvae, recvae); anchors marked *approx* (itemknn, ials, edlae, and pop=floor)
print the delta and defer the verdict to the reviewer.

### Measured timings → honest full-run extrapolation (calibrated on the real ML-25M train matrix, CPU)
Calibration points (this machine): dense `np.linalg.inv` scales cubically — n=3k 1.7s, 6k 12s, 9k 40s;
Gram `XᵀX` full = **12s / 2.72 GB**; iALS 3 iters = 10.6s; MultVAE 1 epoch on 20k users = 19.6s;
RecVAE 1 epoch(3enc+1dec) on 20k users = 66s.

| Baseline | ML-20M (20,108 items) est. | ML-25M (18,430 items) est. | Peak RAM |
|---|---|---|---|
| pop | <1 min | <1 min | <0.5 GB |
| itemknn | ~10–30 min (cosine + 20k-row prune loop; **measured-uncertain, flagged**) | ~10–30 min | ~4–6 GB (near-dense S) |
| ease (λ=500, 1 solve) | Gram 12s + inv **~7.5 min** ≈ **8 min** | inv **~5.8 min** ≈ 6 min | **~10 GB** (20k²) / ~8 GB (18k²) |
| ials (9-point val sweep + final) | ~10 min | ~10 min | ~1 GB |
| edlae (**5× inv** for p-sweep) | 12s + 5×7.5 min ≈ **~40 min** | ≈ ~32 min | ~10 / ~8 GB |
| **dae** (200 ep worst-case) | ~158 s/ep → **~8.8 h** (early-stop likely ~4–6 h) | ~same | <1 GB |
| **multvae** (200 ep) | ~158 s/ep → **~8.8 h** (early-stop ~5–7 h) | ~same | <1 GB |
| **recvae** (50 ep) | ~533 s/ep → **~7.4 h** (early-stop ~4–7 h) | ~same | <1 GB |

- **RAM flag:** EASE/EDLAE need a machine with **≥12–16 GB free RAM** (20,108² float64 Gram + LAPACK
  inverse workspace ≈ 10 GB peak on ML-20M). If unavailable, run EASE/EDLAE alone (not concurrently
  with neural jobs).

## Run plan (what runs tonight, in what order) — and the feasibility flag
Order (both runners): **pop, itemknn, ease, ials, dae, multvae, recvae, edlae** (closed-form first so a
result bank exists before the long jobs). Both runners are **resumable**: they skip any baseline whose
result JSON exists; neural baselines additionally checkpoint every epoch to `.cache/baselines/*.pt` and
resume mid-training via `--max_minutes`.

- **Closed-form bank (pop, itemknn, ease, ials, edlae), both harnesses:** ≈ **2–3 h total.** Fits one night.
- **Neural (dae, multvae, recvae) × 2 harnesses = 6 runs:** at paper epochs ≈ **6 × ~6 h ≈ 30–40 h.**
  **This CANNOT fit in one night sequentially — FLAGGED FOR REVIEW.** Do **not** silently cut epochs
  (HARD RULE 1 / task rule). Options, in preference order:
  1. **Paper epochs + checkpoint-resume across ~4 nights** (primary recommendation). Val-NDCG early
     stopping (patience: multvae/dae 15, recvae 10) already bounds this below the worst case; the runner
     resumes each night until each model early-stops or hits `--epochs`.
  2. **Parallelize the 3 neural models on separate processes** (CPU contention will roughly halve
     per-model throughput; net wall-clock ~2 nights). Needs ≥3 free cores and RAM headroom.
  3. **Paper epochs with a documented early-plateau cut** — only if the reviewer approves, backed by the
     val-NDCG curve showing a plateau; the curve is logged per epoch, so the evidence will exist.
- **Total estimated compute:** closed-form ~2–3 h + neural ~30–40 h ≈ **~35–43 h** across ~4 nights.

## Deviations from the papers (bold = needs reviewer sign-off)
- **itemknn** topk=200/cosine is a *standard* default, not a single published ML-20M constant.
- **ials/WMF** reg & alpha are **selected on validation** (the Mult-VAE/RecVAE tables give a tuned WMF
  ~0.386 but no verbatim hyperparameters); factors fixed at 200.
- **edlae** is **reimplemented** from the paper's closed form (Eq. 8/9); its dropout **p is selected on
  validation** because the paper publishes no single ML-20M p (and the 5-point p-sweep costs ~5 inverses).
- **ML-25M Most-Popular** uses train-only popularity (vs the canonical ruler's global `popb`) — kept
  identical across both harnesses for code parity; explains the ~0.286 vs 0.252 difference.
- **ML-25M neural early-stop** uses the arena **VAL NDCG@10** (not NDCG@100) as the stop signal, to match
  the ML-25M primary metric; the reported test metrics still include NDCG@100 (secondary).

## Unsure / open for the review gate
- Exact **EDLAE ML-20M target** (0.427 used as approx) — confirm against Steck 2020 Table before gating.
- **itemknn / ials approx targets** (0.36 / 0.386) come from the landscape findings, not a re-derived
  primary table — treat their PASS/FAIL as advisory.
- Whether to **rebuild the poetry env** or accept the system-python runtime (see Runtime flag above).
- Whether ML-25M neural baselines should be trained at all, or only the closed-form + the existing
  in-house RecVAE-d512 ruler used (the neural ML-25M runs are the most expensive and least novel part).
