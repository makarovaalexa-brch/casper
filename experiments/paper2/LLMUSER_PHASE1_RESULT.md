# LLM-Simulated-User Study — Phase 1 (Paper C/D simulator reality check)

Tests whether the CASPER answer simulator's assumptions hold up against an LLM standing in for a real
user: (a) does 'hidden gem' framing pull named items toward the tail, (b) does an LLM user agree with the
geometric answer sign(u*.(e_i-e_j)), (c) does the >=2-tagged-items answerability proxy predict LLM refusal.

## Config
- Data: ML-1M, canonical harness split (continuous_actor.py); test pool = te[300:] = 304 users; catalog ni=3706; concepts NC=761
- Answering model (default): **gpt-4o-mini** (temperature 0.7)
- Cross-model subsample: **gpt-5-nano** (reasoning_effort='minimal', max_completion_tokens) on 30 users
- Persona: 3-sentence taste summary generated ONCE/user by gpt-4o-mini (cached)
- FULL context = user's rated titles+ratings capped to <=80 items (high+low extremes)
- Recommender fold u* = frozen enc_concept.pt attention encoder; item factors = Q_svd (D=64)
- Pair questions: pair-native actor policy_pairtrain_d1warm_best.pt, 8 realized item-pairs/user (PAIRSNAP top-K realization)
- All API responses cached to llmuser_cache.jsonl (reruns free)

## (a) Framing lever — popularity percentile of the named movie (0=obscure tail, 100=most-rated head)

### gpt-4o-mini
| condition | question | n | match% | mean pct | median pct | p25 | p75 |
|---|---|---:|---:|---:|---:|---:|---:|
| FULL | love | 600 | 96% | 94.7 | 98.7 | 95.4 | 99.4 |
| FULL | gem | 600 | 72% | 61.9 | 71.5 | 47.7 | 82.3 |
| FULL | dislike | 600 | 99% | 78.6 | 86.7 | 67.4 | 95.6 |
| PERSONA | love | 600 | 83% | 93.9 | 97.5 | 90.0 | 99.4 |
| PERSONA | gem | 600 | 62% | 76.6 | 83.3 | 62.4 | 89.2 |
| PERSONA | dislike | 600 | 38% | 88.8 | 94.3 | 86.2 | 96.8 |

- **FULL: gem − love** paired shift = -33.4 pct-points (n=181 users), Wilcoxon p=0 (gem significantly LOWER — toward tail, as the simulator assumes)
- **PERSONA: gem − love** paired shift = -18.0 pct-points (n=153 users), Wilcoxon p=0 (gem significantly LOWER — toward tail, as the simulator assumes)

### gpt-5-nano
| condition | question | n | match% | mean pct | median pct | p25 | p75 |
|---|---|---:|---:|---:|---:|---:|---:|
| FULL | love | 90 | 93% | 96.6 | 99.3 | 98.1 | 99.4 |
| FULL | gem | 90 | 86% | 80.9 | 87.5 | 75.1 | 92.7 |
| FULL | dislike | 90 | 98% | 83.3 | 89.1 | 77.6 | 96.4 |
| PERSONA | love | 90 | 74% | 91.0 | 98.7 | 91.4 | 99.5 |
| PERSONA | gem | 90 | 53% | 73.6 | 78.0 | 63.7 | 88.6 |
| PERSONA | dislike | 90 | 29% | 84.5 | 92.1 | 76.0 | 96.6 |

- **FULL: gem − love** paired shift = -15.4 pct-points (n=29 users), Wilcoxon p=1.3e-05 (gem significantly LOWER — toward tail, as the simulator assumes)
- **PERSONA: gem − love** paired shift = -18.9 pct-points (n=22 users), Wilcoxon p=0.0027 (gem significantly LOWER — toward tail, as the simulator assumes)

## (b) Pair agreement — LLM choice vs geometric sign(u*·(e_i−e_j))

- Overall agreement: **71.5%** over 694 decided pairs (6 undecided/unparsed of 700 total).

Calibration by |u*·Δ| (geometric signal strength; stronger should agree more):
| |u*·Δ| quartile | n | agreement |
|---|---:|---:|
| Q1 weakest | 174 | 52.9% |
| Q2 | 173 | 65.9% |
| Q3 | 173 | 82.1% |
| Q4 strongest | 174 | 85.1% |

## (c) Answerability realism — LLM refusal vs the >=2-tagged-items proxy

| proxy | LLM answered (yes/no) | LLM refused | refusal rate |
|---|---:|---:|---:|
| answerable (>=2 tagged) | 355 | 49 | 12.1% |
| NOT answerable (0 tagged) | 80 | 301 | 79.0% |

- Proxy↔LLM agreement (answerable→answer, non→refuse): **83.6%** over 785 concepts (197 unparsed 'other').

## Caveats (honest)
- **Clairvoyant reader, not rememberer.** In FULL context the LLM can *read off* the answer from the rating
  list rather than recalling like a real user with imperfect memory. This INFLATES pair agreement and
  answerability realism upward; treat them as optimistic upper bounds. PERSONA (lossy summary) is the more
  realistic-recall condition and is why phase (a) runs both.
- **Fuzzy title match.** LLM free-text titles matched to ML-1M by normalized-string + difflib (cutoff 0.86).
  Unmatched names are dropped from percentile stats (match rates reported per cell). ML-1M's ', The'
  comma-title format and year suffixes are normalized; rare/foreign titles may still miss.
- **Popularity percentile = rating COUNT** over full ML-1M (not the harness's train-like count), ranked across the catalog.
- u* = full-profile fold (single per-user taste vector); pair questions fold the full profile too, so actor,
  geometric answer, and agreement ground-truth all share one u*. PAIRSNAP eval instead folds a half-profile.
- gpt-5-nano is a reasoning model; 'minimal' effort gives clean 0-reasoning-token answers but is a different
  model family than gpt-4o-mini — cross-model rows test robustness of the framing effect, not identical wording.

## Spend
-  gpt-4o-mini: 186 calls, 181792 in / 598 out tok -> $0.0276
- **TOTAL ~$0.0276** (well under the $5 cap)