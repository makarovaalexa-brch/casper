# Reddit smoke trio -- first evaluation on the He-corpus ruler v1

Run 2026-08-03 23:35:37 by `scripts/reddit/smoke_trio.py` (10.1 min, CPU, 4 threads).
Ruler `data\reddit_he\ruler\ruler_v1.csv.gz` -- **all 4431 threads, no sampling** (HARD RULE 1). No LLM call, no
Reddit API. Tower `models\i26c_ep31\t2i26c_ep31.pt` (frozen, read-only import). Concept channel `.cache\instrument\cv3_ep07.pt`.

## Protocol

* Candidate set = the full 18,359-item catalogue in every arm, with **the thread's OP_NAMED items
  masked to -inf in every arm** (given evidence, never targets; GT already excludes them). Identical
  candidate sets across arms is what makes the paired deltas a model comparison.
* Item evidence = the thread's `op_named_movieIds` folded as LIKED tokens at level
  8 (4.5 stars). Threads with zero anchors are bit-identical to the prior
  arm (asserted in code).
* SBERT recipe (arm 4): `all-MiniLM-L6-v2`, request text split into sentences and each
  sentence embedded (no truncation of the request); each genome concept scored by its MAX cosine over
  the request's sentences; top-4 concepts with cosine >=
  0.3 folded as positive answers (value 1.0,
  refusal flag 0) through the C-LITE `ConceptFoldNet` on the frozen tower's latent. Vocabulary =
  1031 genome concepts; mean 3.997 concepts
  folded per thread. Arm 4c pools the whole request into one embedding instead (MiniLM truncates at
  256 word-pieces, which is why sentence pooling is the primary).
* Metrics: binary relevance against `gt_movieIds`, IDCG over min(|GT|,k). Paired bootstrap,
  10,000 resamples over threads.

## Gate: Most-Popular

MostPop NDCG@10 = **0.0188** -- <=0.15:
True; below the items arm: True.

## Arms -- overall

(4431 threads)

| arm | NDCG@10 | NDCG@20 | Recall@10 |
|---|---|---|---|
| 1. Most-Popular | 0.0188 | 0.0187 | 0.0100 |
| 2. Tower prior (no evidence) | 0.0169 | 0.0192 | 0.0087 |
| 3. Tower + items (OP_NAMED) | 0.0325 | 0.0352 | 0.0182 |
| 4. Tower + items + SBERT concepts | 0.0393 | 0.0420 | 0.0219 |
| 4b. Tower + SBERT concepts only | 0.0222 | 0.0247 | 0.0122 |
| 4c. arm 4, doc-pooled SBERT | 0.0401 | 0.0428 | 0.0225 |

## Split by anchor coverage

**>=1 OP_NAMED item** (2639 threads)

| arm | NDCG@10 | NDCG@20 | Recall@10 |
|---|---|---|---|
| 1. Most-Popular | 0.0164 | 0.0169 | 0.0092 |
| 2. Tower prior (no evidence) | 0.0149 | 0.0173 | 0.0083 |
| 3. Tower + items (OP_NAMED) | 0.0411 | 0.0442 | 0.0241 |
| 4. Tower + items + SBERT concepts | 0.0491 | 0.0523 | 0.0280 |
| 4b. Tower + SBERT concepts only | 0.0205 | 0.0232 | 0.0116 |
| 4c. arm 4, doc-pooled SBERT | 0.0499 | 0.0533 | 0.0289 |

**zero anchors** (1792 threads)

| arm | NDCG@10 | NDCG@20 | Recall@10 |
|---|---|---|---|
| 1. Most-Popular | 0.0224 | 0.0213 | 0.0111 |
| 2. Tower prior (no evidence) | 0.0199 | 0.0219 | 0.0094 |
| 3. Tower + items (OP_NAMED) | 0.0199 | 0.0219 | 0.0094 |
| 4. Tower + items + SBERT concepts | 0.0248 | 0.0269 | 0.0130 |
| 4b. Tower + SBERT concepts only | 0.0248 | 0.0269 | 0.0130 |
| 4c. arm 4, doc-pooled SBERT | 0.0257 | 0.0275 | 0.0131 |

## Paired bootstrap (NDCG@10, 10k resamples over threads)

| delta | mean | 95% CI | verdict |
|---|---|---|---|
| items_vs_mostpop | +0.0137 | [+0.0118, +0.0157] | SEPARATED |
| items_vs_prior | +0.0156 | [+0.0136, +0.0176] | SEPARATED |
| sbert_vs_items | +0.0067 | [+0.0050, +0.0084] | SEPARATED |
| prior_vs_mostpop | -0.0019 | [-0.0024, -0.0014] | SEPARATED |
| sbertonly_vs_prior | +0.0053 | [+0.0042, +0.0065] | SEPARATED |

Anchored-subset-only deltas, NDCG@10:

| delta | mean | 95% CI | verdict |
|---|---|---|---|
| items_vs_mostpop | +0.0248 | [+0.0217, +0.0279] | SEPARATED |
| items_vs_prior | +0.0262 | [+0.0230, +0.0295] | SEPARATED |
| sbert_vs_items | +0.0080 | [+0.0054, +0.0106] | SEPARATED |
| prior_vs_mostpop | -0.0015 | [-0.0021, -0.0008] | SEPARATED |
| sbertonly_vs_prior | +0.0056 | [+0.0040, +0.0072] | SEPARATED |

## Controls

**Shuffle** -- the (OP_NAMED, request) evidence is permuted across threads while each thread keeps
its own targets and candidate mask, so any collapse is the evidence losing its owner, not a change
of candidate set:

| arm | NDCG@10 | NDCG@20 | Recall@10 |
|---|---|---|---|
| items, shuffled | 0.0184 | 0.0207 | 0.0097 |
| items+SBERT, shuffled | 0.0167 | 0.0184 | 0.0088 |
| (prior, for reference) | 0.0169 | 0.0192 | 0.0087 |

**Ceilings** -- NDCG@10 and NDCG@20 have ceiling 1.0 for every thread (|GT| >= 5). Recall@10 is
GT-size-capped: mean ceiling 0.7976 overall,
0.7938 on anchored threads,
0.8034 on zero-anchor threads.

**OP-overlap leak stat** -- {"threads": 4431, "op_named_items": 7572, "raw_reply_items": 68385, "op_also_in_replies": 2437, "share_of_op_repeated_by_crowd": 0.3218, "share_of_raw_reply_removed_as_op": 0.0356}

## Sensitivity

Item fold level (arm 3): {"level7": {"ndcg@10": 0.032, "ndcg@20": 0.0351, "recall@10": 0.0173}, "level9": {"ndcg@10": 0.0316, "ndcg@20": 0.0345, "recall@10": 0.0172}}

SBERT concept budget (arm 4): {"top1": {"ndcg@10": 0.0395, "ndcg@20": 0.042, "recall@10": 0.0222}, "top2": {"ndcg@10": 0.0393, "ndcg@20": 0.0417, "recall@10": 0.022}, "top8": {"ndcg@10": 0.0409, "ndcg@20": 0.0432, "recall@10": 0.023}}

## Harness validation (surprise-means-debug-the-harness)

MostPop at 0.0188 is ~4x below the reference numbers the design sheet quotes for the Eberhard
variant (MostPop 0.0821 / doc2vec 0.126 / GPT-4o >0.21), so the metric code was snapped before any
number here was interpreted:

* `smoke_trio.ndcg_recall` vs the canonical `metrics.NDCG_binary_at_k_batch` on 500 ML-25M test
  users, identical scores and masks: **0.13809 vs 0.13809, max per-user difference 0.0** -- the
  metric is bit-identical to the one every published CASPER number uses.
* Most-Popular scored through this same function on the canonical ML-25M test cohort (10k users,
  fold-in masked) = **0.1345**, i.e. the code reproduces a familiar popularity level on a familiar
  task.

So the low absolute level is a property of THIS ruler, not of the harness: the ground truth is what
a crowd *suggested in reply to a specific niche request*, drawn from an 18,359-item catalogue, which
is exactly the population a popularity ranking cannot serve. The Eberhard-variant references were
measured under a different corpus, candidate contract and GT construction -- per the standing rule,
numbers from different information contracts are **not ordered**; they are not a target to hit here.

## Verdict

**The gate PASSES and the ruler behaves.** Most-Popular scores 0.0188 NDCG@10, far under the 0.15
tripwire and below every informative tower arm; the tower's zero-evidence prior sits just *below*
Most-Popular (-0.0019, separated), which is the expected signature of i26c's learned popularity
prior rather than a bug. Folding the OP's own named films through the standard item-evidence
contract is worth **+0.0156** overall and **+0.0262** on the 2,639 anchored threads (both separated,
paired bootstrap over threads) -- so a real request's anchor films do place the user in the latent,
and the shuffle control kills it (items 0.0325 -> 0.0184, essentially back to the 0.0169 prior;
items+SBERT 0.0393 -> 0.0167). The 1,792 zero-anchor threads are, by construction, arm 2 = arm 3;
they dilute the overall table and are the reason the anchored split is the one to read.

**The SBERT rung earns iteration -- but it earns rung 2, not more of rung 1.** Zero-shot
request-text -> genome concepts -> C-LITE fold adds **+0.0067** over items-only (+20% relative,
separated) and is the *only* thing that beats the prior on zero-anchor threads (0.0248 vs 0.0199).
That satisfies the design sheet's decision rule in the direction of "text carries signal we are not
yet using". But the rung is clearly reading something coarse: the concept budget barely matters
(top-1 0.0395 -> top-8 0.0409), sentence pooling and whole-document pooling are indistinguishable
(0.0393 vs 0.0401 -- so the pooling choice is not load-bearing), and the cosine floor of 0.30 never
binds (mean 4.00 of 4 concepts folded). A one-line zero-shot map into a 1,031-concept vocabulary
recovers genre-level intent and stops there. The measured gap it leaves is large in both absolute
and relative terms, so **rung 2 (LLM slot parsing, gated on author approval) is justified**; further
tuning of rung 1 (thresholds, budgets, pooling) is not -- it has been swept here and does nothing.

**For the record on leakage:** 32.2% of OP-named items are also named by repliers (the crowd
re-recommends what the OP already listed); the Eberhard exclusion therefore removes 3.6% of raw
reply items from the ground truth. Masking OP_NAMED out of the candidate list in *every* arm keeps
that from becoming a free win for the arms that get to see them.

**Scope conditions to carry:** the corpus is 2019-22 against a catalogue that ends in late 2019
(disclosed recency truncation); ground truth is entity-anchored, so it only contains films someone
actually named; and this cell measures the *request-conditioned cold ranking* only -- no adaptivity
claim is made or measurable here (`DS_REDDIT_LINKING.md`, "What this ruler is NOT").
