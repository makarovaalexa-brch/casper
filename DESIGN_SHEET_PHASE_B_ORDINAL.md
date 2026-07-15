# DESIGN SHEET — Phase B (concept embeddings) on the UNIFIED-ORDINAL recommender (AWAITING SIGN-OFF)
Date 2026-07-15. No run until the author signs. HARD RULES #1-#3 bind. Base recommender = `paord`
(ordinal, NLEV=5: 0 hated / 1 meh / 2 liked / 3 loved / 4 refuse; a shared FiLM over items+concepts+embeddings).

## GOAL
Learn concept-token embeddings so that folding a user's CONCEPT answers into the FROZEN recommender ranks
their liked items well in the COLD regime (few answers). Concepts live in the same embedding table and share
the same FiLM grading as items — so the ONLY new thing is the concept embedding geometry.

## THE RECIPE (author's intent, made precise)
- **FROZEN:** decoder; `item_emb[:ni]`; the FiLM `gamma`/`beta` (trained on items in paord); `know_emb`;
  the whole attention (`I`, `mab_in`, `sab`, `pma`); `head`; `z0`. The item recommender does not move.
- **TRAINABLE:** ONLY the concept rows `item_emb[ni:ni+NC]` (NC=1628). Polarity is the shared item FiLM.
- **Warm-start** concept rows = pop-weighted member-bag mean of the item embeddings (ConceptBank.Mw @ item_emb).
- **Curriculum:** per user, fold ALL NON-REFUSED concept answers (know>=1), NO items. Token =
  `gamma[ord]*concept_emb + beta[ord] + know_emb[knowledge]`. Target = all liked items (rating>=LO=4).
  Multinomial NLL, same objective as Phase A.
- **Refusal:** refused concepts are DROPPED from the fold (a refused question contributes nothing); they still
  count toward the k budget at eval.
- **Eval** (`concept_eval.py`, ordinal): COLD, reveal k concepts (refused ones drop but count toward k),
  intercept + k=1,2,4,8,16,32, report FULL and TAIL NDCG@10 on held-out users (2956), disjoint from train.

## THE THREE FAILURE MODES I WANT KILLED (open questions for Fable)
1. **TRAIN-ALL vs EVAL-FEW distribution shift.** Training folds ~840 concepts/user (all non-refused); eval folds
   1-32. The frozen attention pools a very different token count/geometry in the two regimes. Do embeddings
   learned under "all concepts pooled" transfer to "3 concepts pooled"? Or must training MATCH eval by sampling
   k (contradicting the author's fold-all intent)? This is the highest-risk item.
2. **REFUSAL representation.** Dropping refused concepts = "refusal is absence." Clean, but is it faithful? A
   real interview refusal BURNS a turn (asked, no info) — dropping models exactly that (no belief change, turn
   spent). Alternative: a learned refusal token. NOTE the additive-null trap: if a refusal folded a token at
   level 4 with gamma=1/beta~0, it would ADD the concept's identity — the opposite of "no info." So if we ever
   fold refusals, level-4 FiLM must be trained to ~zero-out. Dropping avoids this. Which is right?
3. **CIRCULARITY / LEAK.** The answerer's concept answers are DERIVED from the user's taste (dans_build.py
   conditions knowledge on taste). Folding all taste-concepts to predict liked items — is that the legitimate
   elicitation task, or does it let the model shortcut via a taste signal no real interview provides? (We think
   legitimate: concepts ARE the elicited taste. But stress it.)

## SECONDARY QUESTIONS
4. **Is embedding-only underpowered?** With attention + FiLM + decoder all frozen, a concept's only degree of
   freedom is one 512-vector folded through item-trained machinery. Enough capacity, or does a sliver of the
   attention need to adapt to concept tokens (middle ground: train concept rows + a LoRA/bias on mab_in)?
5. **Negation transfer.** "Hated horror" uses the item-trained gamma[0]. Does an item-learned "hated" transform
   meaningfully negate a CONCEPT embedding (which is a member-bag mean, not a single item)?
6. **Warm-start bias.** Member-bag init may pin concepts near the popularity centroid of their members; does
   training escape it, and should the target down-weight popular items so concepts learn TAIL discrimination?

## EXACT Ns / METRIC / MDE (no reductions)
Train: all ~150k answerer-train users (fold-all-concepts). Eval: 2956 answerer-val users (disjoint). All 1628
concepts, all non-refused per user. Metric: cold concept NDCG@10 (full + tail), intercept + k in {1,2,4,8,16,32}.
Baselines to beat: intercept (paord z0), and the item-tree (does a k-concept cold interview beat a k-item cold
interview on the SAME ruler — the Paper-B claim, measured honestly for once). Per-epoch checkpoints; select the
epoch on cold tail-NDCG (NOT full-profile f, which barely moves when only concepts train).

## SIGNED (author, 2026-07-15) — Fable pass applied, decisions locked
- **Curriculum: MIXED 50% fold-all / 50% k~log-uniform[1,64]** of the non-refused set. (Every answer used across
  epochs; per-example curriculum only — not a data reduction.)
- **Refusals DROPPED at train AND eval** (padded out, still count toward k). Level-4 FiLM stays frozen/unused.
- **Target = HELD-half liked items** (same uid-seeded split the answerer's KNOWN half came from) — train task ≡
  eval task, removes the answer-source leak.
- **Freeze via a training design that avoids the AdamW-decay trap:** optimizer over item_emb.weight ONLY with
  **weight_decay=0**, and zero `item_emb.weight.grad[:ni]` each step so item rows never move (verified
  bit-identical after step 1). Everything else `requires_grad_(False)`. Full-profile f is then a CONSTANT leak
  canary.
- **pool = attn FORCED** (paord is attn; key-presence auto-detect is broken because SetEncoder always allocates
  belief params now). concept_eval keeps attn.
- **Selection:** mean cold TAIL over k∈{2,4,8,16,32} (fixed rng), confirm chosen epoch once on test.
- **Headline:** k random CONCEPTS (1628, refusal burns turn) vs k random top-800 ITEMS (same users/split/k,
  symmetric refusals) on ONE ruler; item-TREE reported as the strong item arm separately. NO fabricated
  realistic-answerability item subset (no answer model for tail items).
- **Pop-weighted NLL = CONTINGENT run 2**, triggered only by "cold FULL up, cold TAIL flat" (head capture).
- **Pre-run controls (minutes):** (1) evidence∩target=∅ assert; (2) wrong-user permutation control → cold NDCG
  ≈ intercept; (3) single-concept level-3 vs level-0 polarity probe; (4) frozen tensors bit-identical after one
  step.
- **GO/NO-GO:** GO if cold TAIL@k32 > 0.077 (beats old half-star Phase B) AND cold FULL@k2 > intercept.
  NO-GO → escalate to concept-FiLM-copy rung (not more epochs). f-drift or failed control → abort.

SIGNATURE (author): SIGNED 2026-07-15 (mixed curriculum, mean-tail selection, contingent pop-weight)
