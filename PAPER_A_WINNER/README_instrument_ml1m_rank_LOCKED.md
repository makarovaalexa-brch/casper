# LOCKED instrument — instrument_ml1m_rank (backup 2026-06-15)

**File:** `instrument_ml1m_rank_LOCKED_2026-06-15.pt`
**Live copy:** `data/movielens/.cache/checkpoints/instrument_ml1m_rank.pt`

## What it is
The validated ML-1M elicitation instrument (the fixed recommender the testbed
measures policies against). Trained by `scripts/paper1/train_instrument_rank.py`.

- Architecture: `DualHeadSetEncoder` (set encoder over revealed (item,polarity)
  tokens; liked head + answerability/rated head), arch tag `dual_set_encoder`,
  d_model=128, 4 heads, 2 layers.
- **Loss: listwise RANKING (softmax) on the liked head** (the key fix; per-item
  BCE made it rank ~random on large catalogs). Rated head = BCE.
- Data: ML-1M full catalog `ml1m_profiles.npz` — 3706 movies + 18 genres +
  10 decades + 60 genome tags = 3784 items, 6040 users (WITH decades; the
  no-decades variant did not clear the intrinsic era effect and had lower lift).

## Validated metrics (honest protocols)
- Attribute-reveal NDCG@10: turn0 ~0.395 -> full ~0.51 (lift +0.12); random ~0.03.
- **Hard popularity-matched LOO (the headline metric): Hit@10 0.433 (no reveal)
  -> 0.647 (20 reveals), spread +0.214**; clean monotonic per-turn curve.
- Permutation risers sensible (Toy Story->animation, Silence->thrillers,
  L.A. Confidential->arthouse) with an intrinsic MovieLens era component.

## Why locked
Two-tower dot-product (rank2) was tried and was WORSE (spread +0.16, noisier
risers). No-decades was tried and didn't clear the era effect (it's intrinsic
to co-rating) and lowered lift. So this flat-head ranking-loss instrument is
the chosen measuring stick. See SESSION_LOG.md for the full iteration history.

## Reproduce
`DATASET_NAME=ml1m poetry run python scripts/paper1/train_instrument_rank.py`
