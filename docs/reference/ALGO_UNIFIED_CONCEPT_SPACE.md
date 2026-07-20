# ALGO — Unified Concept+Item Space: pre-VAE origin + current model (2026-07-12)
For Fable adversarial review: find issues + how to improve further. The thesis is now empirically alive —
concepts and items in ONE space, concepts encoded like items, beat the popularity intercept.

## DATA — PRECOMPUTED ANSWERER TABLES ALREADY EXIST (do NOT re-run the answerer)
The realistic answerer answers are already on disk (built by rich_signal, leak-safe):
- `.cache/rich_signal/mm_train_{know,val,crval,uids}.npy` = **(150000, 2428) int8** answerer answers over the
  2,428 universe: concepts `[0:1128]` + entities `[1128:1628]` + items `[1628:2428]`, **aligned 1:1 to
  ConceptBank** (007/zombies/Spielberg verified). `mm_val_*` = **(3000, 2428)**.
- **Split** = leak-safe uid-seeded half-split via `rich_signal.build_cohorts` (per-user rng seed =
  `SEED*1_000_003 + uid`; known = half, held-liked = other half with r≥4). Phase 2/3 must reconstruct held-liked
  targets with THIS split so known/held match the tables.
- **Batched answerer** (`gen_user_fast` + `ease_pred_batch`, rich_signal.py:119/132) exists — ~40× faster than
  per-user; use it to regenerate fast when needed (e.g. EASE-retrain-excluding-eval-held for a leak-clean eval).
- Phase 2/3 read `know/val[:, :1628]` for the concept channel; confidence = `know` (0/1/2), value from `val`.
- NOTE: `derive_concepts` (oracle member-mean) is NOT the answerer — do not use it for training/eval.

## 0. THESIS (one line)
Embed concepts and items in the SAME space and let a strong model encode a concept exactly like an item.
The pre-VAE proved concepts-in-one-space help (ML-1M); the current model scales it with a stronger model +
richer dataset and reproduces the concept lift on ML-25M.

## 1. THE CORE RESULT SO FAR (why this doc exists)
- **Additive concept operators FAIL** (documented): adding a concept vector to the belief drags cold-start
  NDCG BELOW the popularity intercept (0.258→0.226). Diagnosis (8 research streams): summing CORRELATED
  concept vectors amplifies the shared genre/popularity axis and cancels the discriminative residue → you
  recover popularity. Collinearity, not lack of signal.
- **Encoding concepts like items WORKS** (current, `u1`, epoch 1): concept-only NDCG@10 BEATS the intercept
  (0.2551) from k=4 upward and CLIMBS monotonically: k1 0.235 / k2 0.246 / k4 0.261 / k8 0.273 / k16 0.288 /
  k32 0.304; full-profile strength held 0.494. First concept model in the project to clear popularity.

## 2. PART A — THE PRE-VAE ALGORITHM (ML-1M, the origin, PROVEN)
Model `DualHeadSetEncoder`: `scripts/paper1/synthetic_sanity.py:84-112`.
Trainer: `scripts/paper1/train_instrument_rank.py`. Locked ckpt + card:
`PAPER_A_WINNER/instrument_ml1m_rank_LOCKED_2026-06-15.pt`, `PAPER_A_WINNER/README_instrument_ml1m_rank_LOCKED.md`.

- **ONE index space** (train_instrument_rank.py:42, 87): items = indices `0..nt` (3706 movies = the scored
  targets); attributes = indices `nt..ni` (18 genres + 10 decades + 60 genome tags). Same embedding table
  `item_emb = Embedding(n_items, d)` (synthetic_sanity.py:91) — a concept is literally another entity id.
- **Polarity token** (synthetic_sanity.py:92,108): `tok = item_emb(id) + pol_emb(polarity)`, polarity∈{0,1}
  (like/dislike). Dislike native as a token feature.
- **Set encoder** (synthetic_sanity.py:94-111): TransformerEncoder over the revealed (entity,polarity)
  tokens + a CLS token; pooled CLS → two heads.
- **Dual head** (synthetic_sanity.py:98-103): `head_liked` (n_items) + `head_rated` (n_items).
- **Losses** (train_instrument_rank.py:58-65,123): liked head = LISTWISE ranking softmax over targets, revealed
  masked out (`liked_rank_loss`); rated head = BCE (answerability).
- **Concept-forcing curriculum** (train_instrument_rank.py:28,42-44): `ATTR_REVEAL_P = 0.5` — HALF of training
  reveals are ATTRIBUTE-ONLY (`cand = attrs`), forcing concepts to carry the belief. Log-uniform reveal length.
- **Eval** (train_instrument_rank.py:77-104): turn-0 (empty) vs FULL-ATTRIBUTE-reveal (concepts only) NDCG@10.
- **Results** (README): attribute-reveal NDCG@10 turn0 ~0.395 → full ~0.51 (+0.12); Hit@10 0.433→0.647
  (+0.214) over 20 reveals; sensible risers. Concepts helped, as part of a strong-ranker.
- **Known weakness** (INSTRUMENT_REVIEW.md; scale research): the small set-encoder + listwise softmax COLLAPSED
  on large catalogs (0.002 on 18k items). Binary polarity only. ML-1M scale.

## 3. PART B — THE CURRENT MODEL (`u1`, ML-25M, WORKING)
Model `UnifiedEncoder` + `UnifiedAE`: `scripts/reconciled.py` (classes `UnifiedEncoder`, `UnifiedAE`).
Trainer: `cmd_utrain` + `make_unified_example` in `scripts/reconciled.py`. Warm-start: `UnifiedAE.warm_start_a03b`
from `.cache/signed_latent/a0c_best.pt` (0.4961). Log: `.cache/reconciled/u1b.log`.

- **ONE input space**: encoder input = `concat[ item_val(ni), item_mask(ni), concept_val(C), concept_mask(C) ]`
  → `2*(ni+C)` dims (`UnifiedEncoder.forward`). Items (18,430) AND concepts (C=1,628 = 1,128 genome tags +
  500 entities incl. directors/actors/franchises) are input dims to the SAME encoder. A concept is encoded
  identically to an item.
- **Graded value** (not binary): concept/item value = signed graded scalar (loved..hated → centered valence).
  Dislike = negative input value.
- **Strong encoder + decoder**: SignedAE-class deep residual MLP (swish, LayerNorm, DenseNet residuals) →
  z(512); decoder Linear z→18,430 items. MULTINOMIAL log-likelihood over items (Mult-VAE/RecVAE-class — scales
  to catalogs where the pre-VAE listwise collapsed).
- **Warm-start**: item columns of fc1 = a0c; concept columns init 0 → item-only path == a0c EXACTLY (0.4961)
  at init (`warm_start_a03b`); concepts learned from zero, co-trained.
- **Concept-forcing curriculum** (`make_unified_example`): mixed regimes — item-only (strength, w=0.4),
  CONCEPT-ONLY (forcing, w=0.3), mixed. Target = liked NOT revealed (leak-safe). (Pre-VAE's ATTR_REVEAL_P=0.5
  analog.)
- **Eval**: intercept (empty) vs concept-only k-curve (`eval_concept_only`, `eval_intercept`) + full-profile
  strength (`eval_strength`). Full catalog, no caps. 300 study users quarantined.
- **Result** (§1): concept-only beats intercept from k4, monotone; strength held.

## 4. CORRESPONDENCE (pre-VAE ↔ current) — the author's "better model + better dataset"
| | pre-VAE (ML-1M, proven) | current u1 (ML-25M, working) |
|---|---|---|
| items+concepts same space | one Embedding table, entity ids | one encoder input, item+concept dims |
| force concepts | ATTR_REVEAL_P=0.5 (attr-only reveals) | concept-only regime (w=0.3) |
| value | BINARY polarity token | GRADED signed value (loved..hated) |
| backbone | small Transformer set-encoder | strong deep dense-AE (warm 0.496) |
| objective | listwise softmax (COLLAPSES @18k) | MULTINOMIAL (scales to 25M) |
| catalog / concepts | 3.7k items / 60 tags+genres+decades | 18.4k items / 1,628 concepts (+entities) |
| answerability | rated head (BCE) | (not yet — candidate re-add) |

## 5. OPEN QUESTIONS / IMPROVEMENT DIRECTIONS (for Fable)
1. **BELIEF CONFIDENCE + SHRINKING (author-flagged, the headline improvement).** The current model produces a
   POINT belief z — no uncertainty. ConTS/ConUCB research: the lever that BEATS popularity is a POSTERIOR
   narrowing — each answer is a MEASUREMENT that shrinks the belief's covariance along its direction
   (`B += φφᵀ`), and DISLIKE both shrinks uncertainty AND repels the mean (an asymmetry additive can't do).
   Could a last-layer Bayesian head on the (frozen or trained) encoder features add confidence + narrowing +
   adaptive question selection (value-of-info) on top of the working unified encoder? Does it beat the point
   model, and where does it break (feature alignment, tail separation, graded-scale calibration)?
1b. **CONFIDENCE = MEASUREMENT PRECISION (author-flagged; SAME mechanism as Q1).** Users answer with a
   KNOWLEDGE level (no_clue / rough_idea "probably know" / know_well). Currently a FIXED multiplicative
   down-weight (`g = valence × CONF`, CONF={0, 0.5, 1.0}) — heuristic, not learned; can't tell "rough-but-right"
   from "rough-and-guessing." The principled home is the belief head (Q1): confidence = the observation
   precision `1/σ²` of each answer's measurement — know_well shrinks the belief a lot, rough_idea a little
   (auto-discounted, uncertainty retained), no_clue not at all. Open: does the training signal support a
   LEARNED confidence gate/precision (do rough-idea values predict held-liked worse), or is the fixed scale
   already near-optimal? Test: ablate confidence (flatten CONF to 1) vs learned precision; measure held-liked.

2. **TAIL.** The research says concepts' real value is on the TAIL (broad genres ≈ popularity by information
   theory; fine tags/entities carry the distinctive signal). u1 so far measured FULL only — does the concept
   lift concentrate on the tail? Is the honest bar popularity-within-the-attribute-filter on tail NDCG?
3. **STRENGTH DRIFT.** full-profile slipped 0.4961→0.494 at ep1. Does co-training erode item strength over
   epochs? Guard (freeze/low-LR item columns, replay, joint peak)?
4. **THE COLLINEARITY/WHITENING lens.** Additive failed by concept collinearity; the encoder's nonlinearity
   presumably learns an implicit whitening. Is that what's happening, and is an explicit whitening (FM cross /
   posterior B⁻¹) a better or redundant mechanism vs the encoder?
5. **DISLIKE usefulness.** Is the NEGATIVE half of the concept signal net-useful (dislike-repels), or is the
   lift all positive-half? (knockout split.)
6. **Re-add the answerability head?** (pre-VAE had it; dropped as a recommender concern — but it's the policy
   channel.)
7. **Circularity guard.** Concept values derive from member taste (answerer duality). De-bias / Ce / 173
   transfer so the win isn't "EASE folded twice" (a measurable test exists: concept whose members are already
   in history → ~0 new info under a correct model).
8. Scale/collapse guardrails (DIF-SR gradient theorem: early ADDITIVE fusion shares gradients → collapse; we
   use INPUT dims + multinomial, believed safe — confirm).

## 6. QUESTIONS FOR FABLE
- The single most dangerous flaw in the current unified-encoder result (is the concept-only lift real/useful,
  or an artifact — OOD single-concept, EASE circularity, strength-borrow)?
- Is the belief-confidence/shrinking head (Q1) the right next improvement, and can it bolt onto the working
  unified encoder without losing the 0.496? Sketch the mechanism + the decisive gate.
- The minimal ordered plan to (a) harden the current result and (b) add the belief head, with a kill-switch
  at each step. The 173 quarantine touched once at the end.
