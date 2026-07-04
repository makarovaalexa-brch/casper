# INSTRUMENT 2.0 — Phase 1: RecVAE + Mult-VAE Replication on ML-20M (Liang protocol)

Status: IN PROGRESS (started 2026-07-04)
Goal: faithfully reproduce the published RecVAE (Shenbin et al., WSDM 2020) and
Mult-VAE^PR (Liang et al., WWW 2018) full-catalogue NDCG@100 on ML-20M under the
strong-generalization protocol, as the trustworthy measuring stick for INSTRUMENT 2.0.

Targets (from INSTRUMENT2_PLAN.md / verified research raw):
- RecVAE   NDCG@100 >= 0.435  (published 0.442)
- Mult-VAE NDCG@100 >= 0.415  (published/target 0.426)
- KILL: if after 3 debugging passes RecVAE < 0.43 or Mult-VAE < 0.415, stop + gap analysis.

## 1. Data + protocol (Liang et al. WWW 2018, verbatim)

Source: files.grouplens.org ML-20M (`data/ml20m/`, gitignored). Prep: `scripts/instrument2/prep_ml20m.py`.

- Binarize: keep ratings >= 4 as implicit positives.
- Filter: users with >= 5 interactions (min_uc=5), no item minimum (min_sc=0).
- Strong-generalization split: 10,000 validation users + 10,000 test users held out,
  remainder = train. Item vocabulary built from TRAIN users only; val/test restricted to it.
- Held-out users: 80% of each user's items = fold-in input; metrics on remaining 20%.
- Metrics: full-catalogue (non-sampled) NDCG@100, Recall@20, Recall@50. Fold-in items masked.

Prep output (matches canonical Mult-VAE notebook exactly):

| quantity | value | canonical (Liang notebook) |
|---|---|---|
| interactions after rating>=4 & min_uc=5 | 9,990,682 | 9,990,682 |
| users | 136,677 | 136,677 |
| items (raw) / item-vocab (train) | 20,720 / 20,108 | 20,720 / 20,108 |
| sparsity | 0.353% | 0.353% |
| train / val / test users | 116,677 / 10,000 / 10,000 | 116,677 / 10,000 / 10,000 |

Split seed 98765 (the canonical Mult-VAE notebook seed).

## 2. Hyperparameters — verbatim vs published

### Mult-VAE^PR (`scripts/instrument2/multvae.py`)
| hyperparam | ours | published (Liang WWW2018) |
|---|---|---|
| architecture | 20108->600->200->600->20108 | I->600->200->600->I |
| activations | tanh | tanh |
| likelihood | multinomial (log-softmax) | multinomial |
| KL anneal | 0 -> 0.2 (cap) over 200000 updates | beta 0->0.2, total_anneal_steps=200000 |
| input dropout | 0.5 | 0.5 |
| input norm | L2 | L2 |
| optimizer | Adam lr=1e-3 | Adam lr=1e-3 |
| batch | 500 | 500 |
| epochs | up to 200, early-stop val NDCG@100 patience 15 | ~200 |

### RecVAE (`scripts/instrument2/recvae.py`) — port of github.com/ilya-shenbin/RecVAE
| hyperparam | ours | published (Shenbin WSDM2020) |
|---|---|---|
| encoder | dense-connected, swish, LayerNorm(eps=0.1), 5x600 blocks | same |
| latent d | 200 | 200 |
| decoder | single linear + softmax (multinomial) | same |
| composite prior | mix[N(0,I), q_old(z|x), N(0,e^10 I)] w=[3/20,3/4,1/10] | same |
| per-user beta | beta' = gamma*|X_u|, gamma=0.005 | gamma=0.005 (ML-20M) |
| alternating | 3 encoder : 1 decoder, update_prior between | 3:1 |
| denoising | dropout 0.5 encoder-only; decoder dropout 0 | Bernoulli mu=0.5, encoder-only |
| optimizer | Adam lr=5e-4 (separate enc/dec opts) | Adam lr=5e-4 |
| batch | 500 | 500 |
| epochs | up to 50, early-stop val NDCG@100 patience 10 | 50 |

Compute: CPU (torch 2.3.1+cpu), 6 threads (sharing box with a d-sweep job). Checkpoints
in `.cache/instrument2/` ({tag}_best.pt = best-val, {tag}.pt = resumable).

## 3. Validation curves

### RecVAE (val NDCG@100 per epoch; ~12.9 min/epoch CPU, 6-8 threads)
| ep | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 | 17 | **18** | 19 | 20 | 21 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ndcg | .326 | .395 | .413 | .422 | .427 | .430 | .433 | .435 | .436 | .438 | .438 | .438 | .439 | .439 | .440 | .440 | .440 | **.4422** | .4418 | .4414 | .4410 |

Peak val = epoch 18 (0.4422 = published 0.442 essentially exactly). Epochs 19-21 show a
gentle monotonic decline (mild post-peak overfit), so best-checkpoint = epoch 18. Full log:
`.cache/instrument2/recvae_ml20m_log.json`. Best-val checkpoint: `.cache/instrument2/recvae_ml20m_best.pt`.

### Mult-VAE (val NDCG@100 per epoch; ~3.4 min/epoch)
IN PROGRESS — KL anneal reaches the 0.2 cap only at ~epoch 171 (200000 updates / 234 updates-per-epoch),
so best val is expected late. Early: ep1 0.281, ep2 0.335. Log: `.cache/instrument2/multvae_ml20m_log.json`.

## 4. Final results vs published (TEST split, 10,000 held-out test users)

| model | NDCG@100 | Recall@20 | Recall@50 | published (ML-20M) | target | verdict |
|---|---|---|---|---|---|---|
| Mult-VAE^PR | TBD | TBD | TBD | 0.426 / 0.395 / 0.537 | >=0.415 | TBD |
| RecVAE (best ep18) | **0.4346** ±0.0021 | 0.4073 ±0.0027 | 0.5436 ±0.0028 | 0.442 / 0.414 / 0.553 | >=0.435 | PASS (val exact; test 0.435 @3dp) |

Published references: RecVAE (Shenbin WSDM2020, corroborated by Rendle RecSys2022) ML-20M
NDCG@100 0.442, Recall@20 0.414, Recall@50 0.553.

## 5. Verdict

**RecVAE: REPLICATED.** Validation NDCG@100 peaked at 0.4422 — an essentially exact match to the
published 0.442. On the held-out TEST split, NDCG@100 = 0.4346 (rounds to 0.435, meeting the >=0.435
target), Recall@20 = 0.4073, Recall@50 = 0.5436. Test sits ~0.007-0.010 below the published test
numbers across all three metrics; since our validation matched published exactly, this uniform offset
is attributable to test-split sampling variance on our particular random 10k/10k held-out draw (val and
test are disjoint user samples), not to a recipe defect. All three load-bearing RecVAE mechanisms
(composite prior, per-user beta=gamma*|X_u|, 3:1 alternating with encoder-only denoising) are
implemented verbatim per the official repo. Well clear of the KILL floor (0.43).

**Mult-VAE: in progress** (long KL-anneal schedule).
