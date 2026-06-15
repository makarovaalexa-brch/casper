# CASPER — Consolidation (2026-06-15)
Single source of truth after a long, sprawling session. Addresses: unified naming,
checkpoint manifest, best results, gap analysis, plan. (Interim false paths to be
pruned LATER, not now — see SESSION_LOG.md for the full history.)

## 0. Method naming — UNIFIED (use these names everywhere)
| canonical name | what it is | scripts | NOTE on inconsistency |
|---|---|---|---|
| `random` | random question | policies.py RandomPolicy | |
| `popularity` | static popularity order | policies.py PopularityPolicy | |
| `greedy_infogain` | myopic EIG (instrument) | policies.py | |
| `scpr` | weighted-entropy (SCPR-style) | policies.py SCPREntropyPolicy | |
| `greedy_answerability` | dual-head answerability routing | policies.py | |
| `thompson` | PEBOL-style Bayesian | policies.py ThompsonPolicy | |
| `HELF` | pop×entropy (Rashid IUI'02) | bench_appropriate.py | only ml1m so far |
| `PPO` | discrete dual-belief PPO | train_discrete_ppo.py | synthetic uses in-script PPO (synthetic_rl_improved.py) — SAME algo, diff impl |
| `DQN` | dueling double DQN | train_dqn_policy.py | synthetic in-script DQN |
| `bot-play` / `REINFORCE` | IJCNN REINFORCE (return-to-go, entropy, dual belief) | train_botplay_dual.py | **synthetic = reinforce_v2/v3 (synthetic_rl_improved.py) — DIFFERENT impl. UNIFY LATER.** |
| `CASPER` (the method) | equivariant actor, greedy-distill + RLOO finetune | oracle_distill.py + train_finetune.py (stratified); train_method_ml1m.py (ml1m) | only stratified + ml1m |

## 1. Checkpoint manifest (what to keep)
**Instruments** (`data/movielens/.cache/checkpoints/`):
- `instrument_ml1m_rank.pt` — **LOCKED ml1m ranking instrument** (backed up in experiments/instruments/).
- `instrument_ml_stratified.pt` (binary AUAC), `instrument_v5_set.pt` (slate1), `instrument_slate2_*`, `instrument_yelp_multicity.pt`, `instrument_amazon_*`, `instrument_lastfm_dual*`, `instrument_s1_dual.pt`.
- superseded/diagnostic: `instrument_ml1m.pt` (BCE, ranks~random), `instrument_ml1m_rank2.pt` (two-tower, worse), `instrument_ml1m_nd_rank.pt` (no-decade, no gain), `instrument_v2/3/4_onehot` (old).
- DELETE: `instrument_ml1m_big.resume.pt` (107MB OOM junk).

**Learned policies**:
- CASPER: `experiments/paper2/distill_ml_stratified_greedy.pt`, `finetune_ml_stratified.pt` (=0.726), `method_ml1m.pt` (=0.507).
- PPO: `discrete_ppo_{ml_stratified,slate2,yelp_multicity}_dual.pt`, `discrete_ppo.pt` (slate1).
- DQN: `dqn_policy_{ml_stratified,slate2,yelp_multicity}.pt`, `dqn_policy.pt`.
- bot-play: `botplay_{ml_stratified,slate2,yelp_multicity}_dual.pt`, `botplay_dual.pt`.
- Legacy V1–V3 SBERT-era: `experiments/checkpoints/rl_episode_*` etc. — KEEP as history, NOT used.

## 2. BEST RESULTS so far (honest, per testbed — METRIC DIFFERS, flagged)
### Synthetic indicator world — AUAC (the CLEAN adaptive win)
| policy | AUAC | branches |
|---|---|---|
| PPO | **0.628** | Yes |
| adaptive oracle | 0.623 | Yes |
| DQN | 0.623 | Yes |
| scpr | 0.609 | Yes |
| static-oracle (best fixed) | 0.591 | No |
| **bot-play (reinforce)** | 0.591 / 0.574 | **No (FAILS to route)** |
→ **PPO/DQN learn adaptive routing, beat best-static by +0.04. bot-play does NOT.**

### Stratified MovieLens — AUAC (held-out)
| policy | AUAC | branches |
|---|---|---|
| **CASPER (finetune)** | **0.726** | Yes |
| scpr | 0.721 | Yes |
| PPO | 0.720 | No |
| bot-play | 0.714 | No |
| popularity (static) | 0.701 | No |
→ CASPER best; adaptive ≈ best-static (edge over naive popularity +0.02).

### ML-1M full catalog — Hit@10 hard-LOO (ranking; the rigorous, "real-ish" testbed)
| policy | AUC | t15 |
|---|---|---|
| HELF | 0.509 | 0.525 |
| **CASPER** | 0.507 | 0.530 |
| popularity | 0.498 | 0.495 |
| thompson | 0.485 | 0.520 |
| random | 0.477 | 0.485 |
→ CASPER ties best heuristic (HELF); none beats it. Ceiling (20 reveals) 0.647.

### Yelp multi-city — AUAC (older metric): greedy_answerability 0.781 > popularity 0.771 (+0.011).

## 3. GAP ANALYSIS (what's missing / weak)
1. **No real, full dataset with a clean adaptive win.** Clean win is synthetic (toy); ml1m washes out; stratified is bespoke; yelp marginal. **This is the central gap** (user point #2).
2. **bot-play (REINFORCE) fails to learn routing** (only PPO/DQN do) — the user's own method needs a variance-reduced optimizer (RLOO + equivariant actor) to cross the adaptive boundary.
3. **Metric inconsistency**: synthetic/stratified use AUAC; ml1m uses Hit@10/NDCG. Need ONE metric (recommend ranking: NDCG@10/Hit@10 hard-LOO) everywhere.
4. **CASPER only on stratified + ml1m** — not run on synthetic/slate2/yelp. Not unified.
5. **Missing faithful CRS-RL baselines**: UNICORN (graph dueling-DDQN), EAR, CRM, SAC-discrete. We have generic PPO/DQN/REINFORCE, not the published CRS-RL agents.
6. **Naming/impl drift**: bot-play = trainer vs synthetic reinforce_v2/v3 (different code). PPO/DQN similarly. Unify to one trainer set.

## 4. PLAN
**NOW (consolidate):** this doc + safe checkpoint backup (git-tracked) + draft the paper around the *honest best* results (synthetic adaptive-RL win + the rigorous testbed/metric findings + CASPER matching SOTA heuristic on ml1m).
**SIDE (ongoing):** bigger/real dataset hunt for a clean adaptive win (user point #2).
**LATER (cleanup):** unify metric + trainers across testbeds; add UNICORN/EAR/CRM baselines; fix bot-play optimizer; prune all interim false paths from code + write-up.
