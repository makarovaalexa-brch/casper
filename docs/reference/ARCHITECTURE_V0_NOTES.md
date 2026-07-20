# V0 architecture (Jan–Feb 2026) — what it was, why it was abandoned

> A one-page record of the original `src/casper/` package (38 files, ~8k lines), removed 2026-07-20
> (recoverable from git history). Its surviving ideas all live on in the current papers/vision; this note
> exists so the *first* design and its failure modes aren't forgotten.

## What it was
The original **CASPER = Continuous-Action Preference Elicitation via RL**:
- **Policy:** actor-critic (`rl_actor_critic`, `casper_agent`) emitting a **continuous action in SBERT
  semantic-embedding space**, snapped to the nearest movie/entity. Trained with an RL harness
  (`episode_runner`, `parallel_rl_trainer`, `rl_actor_pretraining`).
- **Recommenders:** `two_tower_recommender` and `lstm_attention_recommender` (explicit-rating encoding).
- **LLM pieces:** `question_generator` (GPT asker), `preference_extractor` (LLM answer→preference).
- **Users/data:** MovieLens `user_simulator`; a **Reddit** scrape→pretrain pipeline for supervised
  bootstrapping; `embedding_space` (SBERT movie entities).

## Why it was abandoned
- **Recommenders collapsed on polarity:** two-tower 0.389 (68% like/dislike overlap); LSTM+attn 0.428
  (100% collapse). See `EXPERIMENTS.md` (recommender-core).
- **RL was information-theoretically broken:** ~1–2.5k transitions for a 384-dim continuous action
  (needs 10⁵–10⁶); per-turn reward SNR ≈ 0.025; all 3 DDPG runs converged to a **user-agnostic** policy.
  See `HARSH_REVIEW_POINTS.md` (expert review, Flaws A/B).
- **Reddit pretraining direction: dropped** (never paid off; data removed).

## Where the surviving ideas went
- Continuous action over embeddings → **Paper C / VISION #3** (now a policy over a *belief*, LLM-verbalised).
- LLM asker / interpreter → **Paper E / VISION #6** (renderer-only; never scores).
- The recommender → **Paper A** (RecVAE → set-encoder; strong instrument first).
- The user simulator → the **distilled answerer v2.1** (apparatus specs in `reference/`).
- The learned RL policy → superseded; the learning moved into the **recommender + answer model**
  (elicitation ≈ selection), with the belief-pool / uncertainty-shrinkage line as Paper B's lead.
