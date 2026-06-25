# 🏆 PAPER A WINNER — the RS model / instrument  (DO NOT DELETE / DO NOT OVERWRITE)

**THE model:** `instrument_ml1m_rank` — the validated ML-1M **DualHeadSetEncoder** recommender
that the whole testbed measures elicitation policies against.

**Files (this archive):**
- `instrument_ml1m_rank_LOCKED_2026-06-15.pt` — the locked weights
- `README_instrument_ml1m_rank_LOCKED.md` — full provenance (architecture, data, metrics, repro)

**SHA256:** `10a7999faa91542b309f60b25a3b7c244afe89b5f1da20157e463a75062f3ede`
**Live load path:** `data/movielens/.cache/checkpoints/instrument_ml1m_rank.pt` (byte-identical — sha verified)
**Also locked at:** `experiments/instruments/instrument_ml1m_rank_LOCKED_2026-06-15.pt` (committed `3a3950d`)

## What it is (one line)
DualHeadSetEncoder (d=128, 4 heads, 2 layers) over revealed (item,polarity) tokens; **listwise
ranking loss** on the liked head (the key fix vs per-item BCE), BCE on the rated/answerability head.
ML-1M full catalog (3706 movies + 18 genres + 10 decades + 60 genome tags = 3784 items, 6040 users).

## Headline metrics (honest protocols)
- Hard popularity-matched LOO **Hit@10: 0.433 (no reveal) → 0.647 (20 reveals)**, spread **+0.214**, monotone.
- Attribute-reveal NDCG@10: ~0.395 → ~0.51 (lift +0.12); random ~0.03.

## Guard status (2026-06-25)
Triple-guarded + offsite: locked backup + live copy + this archive, all **on the clean remote**
`recommender-improvement @ e3707e0` (GitHub). Trained by `scripts/paper1/train_instrument_rank.py`.
Parallel to `PAPER_B_WINNER/` (the CASPER-R policy). _Paper A is frozen on this instrument._
