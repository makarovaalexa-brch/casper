# DESIGN SHEET — Unified context-dependent FUSION token (replaces FiLM) — FOR FABLE REVIEW
Date 2026-07-15. Author directive: replace FiLM. NO item/concept split. Value AND confidence fed WITH the
embedding and attended, so the model LEARNS per-case whether an answer is a negative sign, a drop, or a like.
Must support an arbitrary (policy-emitted) question embedding.

## WHY FiLM FAILS (established)
- FiLM token = `gamma[level] * emb + beta[level]`. `gamma/beta` see ONLY the level, never the embedding, so
  they apply the SAME transform to a horror concept and a specific film. They cannot be context-dependent.
- Measured: `gamma ~= 1` at every level; the model never learned to negate. It responds to item PRESENCE.
- DATA says negation is real and CONTEXT-DEPENDENT: item-hate is weak-positive (exposure: a hater of film X
  still watches X's genre), but concept-hate is genuinely below prior (Test 0b: concept-haters like the
  concept's films 0.64-0.94x the population, know-gated so not laundered by exposure). Same word "hate", opposite
  meaning -- decided by WHICH embedding. FiLM structurally cannot express this.

## THE PROPOSED TOKEN (uniform for item / concept / arbitrary embedding)
```
token = emb + Fuse( concat[ emb ; Eval[value] ; Econf[confidence] ] )
```
- `emb` (512): item row, concept row (SAME table), or an arbitrary emitted vector. NO special-casing.
- `Eval` (learned, 4 levels: hated/meh/liked/loved -> 512). `Econf` (learned, 3 levels: no-clue/rough/know-well
  -> 512). REFUSAL = lowest confidence (no separate channel).
- `Fuse` = MLP(1536 -> h -> 512), LAST layer init 0  => at init `token = emb` (preserves the warm-start geometry
  that made FiLM beat concat-MLP historically). Then `Fuse` learns the value/confidence modulation.
- Attention (ISAB: tokens -> 32 inducing pts -> self-attend -> PMA pool) UNCHANGED -> z(512) -> multinomial
  decoder over 18,430 items.
- **The lever:** because `Fuse` SEES `emb`, it can output `~= -2*concept_emb` for (concept, hated) -> token
  points AWAY -> the concept's films sink below prior; and `~= +small` for (film, hated) -> stays weak-positive;
  and `~= -emb` (token ~ 0, dropped) for low confidence / refusal. All LEARNED, context-dependent, per-case.
- **Arbitrary-embedding ready:** a policy emits `emb`; attach value+confidence; same Fuse; fold. The continuous
  part is the EMBEDDING; value/confidence stay discrete (the user answers a graded scale).

## TRAINING (no split -- the crux)
UNIFIED curriculum, ONE Fuse, items AND concepts in the same training so the shared Fuse learns BOTH polarities
contextually from the embedding:
- item tokens: value = rating->ordinal (hated..loved), confidence = know-well (rated => known).
- concept tokens: value+confidence from the distilled answerer (mm tables); refusals = no-clue.
- target = held-half liked items (leak-free split). Multinomial NLL (the existing objective already carries the
  concept-negative signal: a horror-hater's target EXCLUDES horror -> loss WANTS fold(hate-horror) to down-rank
  horror; Fuse gives it the lever). No new competing negative-loss term.
- curriculum mixes item-only / concept-only / mixed folds, and SHORT (k=1..4) folds (interview regime) so
  cold-start is in-distribution (a fix vs the current curriculum that never folds singletons).
- warm-start: item_emb from a0c (or pb2), concept rows from member-bags, Fuse init-null.

## ACCEPTANCE GATES
1. full-profile NDCG@10 >= ~0.485 (the fusion must not lose to FiLM on the recommender's own job).
2. NEGATION learned & context-dependent: (concept, hated) fold sinks the concept's member films BELOW the
   refusal-fold baseline AND reproduces Test-0b ordering LOVE > refusal > HATE; (item, hated) stays weak-positive
   (>= its genre-exposed baseline). The model must learn the split ITSELF (no hard-coded item/concept branch).
3. cold concept tail-NDCG beats intercept and ideally beats the old FiLM Phase-B (0.06-0.077 half-star).
4. refusal/low-confidence folds move z ~ 0 (dropped), verified.

## OPEN QUESTIONS FOR FABLE
A. **Full-profile risk.** concat-MLP fusion LOST to FiLM historically (scrambled a0c geometry). Does the
   RESIDUAL form `emb + Fuse(...)` with null-init actually preserve enough to match FiLM's 0.485, while gaining
   negation? Or is there a better parameterization (e.g. Fuse outputs a gate + shift; or low-rank)?
B. **Joint vs staged training.** Author wants NO split, so train items+concepts jointly with one Fuse. Risk:
   concepts (rough member-bag init) interfere with item full-profile. Is joint safe, or do we warm-start a good
   item recommender first then joint-finetune? Does jointly-trained Fuse actually learn DIFFERENT polarity for
   item-region vs concept-region embeddings (the whole point), or collapse to one behaviour?
C. **Does one negation token survive attention pooling?** For a cold interview (few tokens), a single
   `-concept_emb` token has weight; but can ISAB->PMA actually produce a z that ranks the concept's films below
   prior, or does the pool wash it out? Is an explicit confidence GATE on attention weights safer than
   learned-drop for refusals?
D. **Value/confidence as learned codes vs scalars.** Learned 4/3-level embeddings vs continuous scalars. Which
   generalizes better, and does the policy ever need CONTINUOUS value (vs continuous embedding + discrete value)?
E. **Init & optimization.** Fuse-from-null must learn ALL value signal from zero (FiLM started gamma=1). Is that
   a harder optimization? Should Eval/Econf be non-null while Fuse is null? Any barrier like the gamma=1 trap?
F. **The negation identifiability trap Fable named before:** item-hate must NOT be forced below prior (that would
   fabricate an effect stronger than the data). Does the unified Fuse naturally keep item-hate weak-positive
   while making concept-hate negative, given the loss + exposure structure? How do we PROVE it learned the
   distinction rather than averaging?

## WHAT I NEED BACK
Rank the risks; the single most likely way this wastes a retrain; the corrected token+training design (residual
form, init, joint-vs-staged, gate-vs-learned-drop); a cheap pre-retrain test that predicts whether the fusion
can beat FiLM on full-profile AND learn context-dependent negation; and the exact go/no-go gates. Be brutal: if
`emb + Fuse([emb,val,conf])` cannot both preserve full-profile AND learn negation, say what does.
