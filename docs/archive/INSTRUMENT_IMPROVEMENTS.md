# Instrument Improvements — Fair List

Scope: improving the ML-100k cold-start elicitation instrument (set-encoder,
ranking+MSE) **without** producing artifacts or unfair comparisons. Context:
the instrument currently beats biased-MF on full-catalogue NDCG/Recall and lags
it on RMSE; both monotone in true reveals. Established protocol: ML-100k, warm/cold
USER split (200 cold), MF fold-in baseline, RMSE (Golbandi/Zhou) + full-catalogue
NDCG@10/Recall@10 (NCF lineage).

## 0. MANDATORY fairness protocol (applies to every item below)
- **Tune hyperparameters ONLY on the val (warm held-out) split.** Lock the config,
  then evaluate **once** on cold/test users. Never pick the config that maxes the
  cold/test number — that's test-set leakage and won't replicate.
- **Report 3 seeds + 95% CIs.** Full-catalogue NDCG (few relevant items, ~1600
  negatives) is high-variance; single-seed gaps may be noise.
- **Keep the warm/cold split FIXED (200 cold)** so runs stay comparable.
- **Report BOTH** RMSE and ranking (NDCG@10/Recall@10), full-catalogue, and watch
  all of them together (guards against gaming one).

## 1. FAIR BASELINES — add before any "instrument > MF" claim
- biased MF (RMSE baseline) — DONE (RMSE 0.94 warm / ~0.97 cold).
- **WRMF / BPR-MF (ranking-trained MF)** — REQUIRED. The current NDCG win compares
  a ranking-trained instrument to an MSE-trained MF; the fair ranking baseline is
  MF trained for ranking. Until the instrument beats THIS, the ranking win is not
  clean. (WRMF already implemented in scripts/paper1/mf_compare.py.)
- popularity-rank, random, HELF.

## 2. FAIR IMPROVEMENTS — try as single-change A/Bs, tuned on VAL
**Tier 1 (highest leverage, low risk)**
1. Early-stop on **smoothed val NDCG** (not val MSE) — align stopping with the
   target metric. Smooth because full-cat NDCG is noisy.
2. **RANK_W Pareto sweep** (0.2→0.6) on val — pick the NDCG/RMSE frontier point.
3. **Attention pooling** (set-transformer) instead of mean-pool — more expressive;
   watch overfit via val (663 warm users is small).

**Tier 2**
4. **More augmentation** S (reveal-subsets/user), FIXED split — modest, diminishing
   (same users → limited real diversity).
5. **Rating centering**: encode `r − item_mean` using **WARM-ONLY** item means
   (warm-only or it leaks).
6. **Ensemble** 3–5 instruments (avg scores) — reliable small boost; keep compute
   fair vs MF (or ensemble MF too).

**Tier 3**
7. Capacity/reg sweeps (`d`, dropout, weight-decay) on val.
8. **LambdaRank / ApproxNDCG** loss — only with multi-metric monitoring (can game
   NDCG@10 and resurrect the rated-ness artifact).

## 3. USE ONLY WITH HONEST LABELLING (not "independent instrument > MF")
- **MF-anchored residual** (`r̂ = MF_pred + neural_residual`) — folds MF in; a good
  MODEL but circular for "instrument beats MF". Label as an MF+neural hybrid.
- **MF-init item embeddings** — reduces independence from MF; note it explicitly.

## 4. AVOID (fishy)
- **Popularity debias at inference** (subtract log-pop) — post-hoc manipulation of a
  popularity-sensitive metric; can inflate NDCG without real improvement.
- **Tuning any hyperparameter on the cold/test set.**
- **Shrinking the cold set** to get more warm data — breaks comparability AND makes
  the test noisier.
- **Optimizing NDCG@10 alone** without watching RMSE and the disliked control.

## 5. Order of operations
1. Add WRMF/BPR-MF fair ranking baseline; re-check instrument still leads. (gate)
2. If it leads: Tier-1 A/Bs on val → lock → cold eval + 3 seeds/CIs.
3. Then Tier 2/3 as warranted.
If the instrument does NOT beat WRMF on ranking, STOP and reframe — don't polish.
