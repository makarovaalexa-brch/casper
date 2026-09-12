# A17 — the instrument ported to Goodreads: tower OK, concept channel FAILS (2026-09-12)

**Status: STOPPED FOR AUTHOR DECISION.** Design `docs/design/DS_A17_GOODREADS_ARENA.md`.
Author instruction: *"replicate the process exactly without corner cutting. if you start getting
netaive results stop and tell me."* This is that report.

## What was built, and it is sound

| stage | result |
|---|---|
| arena | 18,430 most-rated books, matched to ML-25M's arena SIZE; keeps 56.7% of rating events |
| split | **the project's own `liang_split.py --data goodreads`**: 18,430 items, 140,768 / 10,000 / 10,000 |
| concept bank | 2,408 concepts from `popular_shelves`, per-book density median **37** vs ML-25M's 37 |
| RecVAE teacher | test **0.3456 full / 0.2702 tail** vs ML-25M's 0.3540 / 0.2497 — 2.4% lower on full, **8.2% higher on tail** |
| i26 tower | 60 epochs, 15.2 h; pre-registered rule selects **ep53** (sel 0.1074, val_full 0.3374) |
| concept channel | 12 epochs, 68 min, mixture verified identical to cv3's |

Every leak check passed on Goodreads at every stage: vocabulary, cohort disjointness, train likes
matching the split file, vocabulary set matching the item list.

## ⛔ THE NEGATIVE RESULT: concept answers make recommendations WORSE

Validation NDCG@10 after k concept questions, same script, same mixture, same ladder:

| | m0 (none) | m1 | m4 | m8 |
|---|---|---|---|---|
| **ML-25M (cv3, ep4)** | 0.1811 | 0.1774 | 0.1856 | **0.1916** |
| **Goodreads (cvg, ep12 final)** | 0.1877 | **0.1893** | 0.1817 | **0.1746** |

Stable across all twelve epochs. Best SEL 0.1830 at ep10; the shape never changed.

**The refinement that matters for diagnosis: one concept answer HELPS (+0.0016 over asking nothing).
Four hurt (-0.0060). Eight hurt more (-0.0131).** So the vocabulary is not signal-free — if shelves
carried nothing, one question would do nothing. What fails is **accumulation**: each additional
concept answer moves the user's representation further off target.

## What this threatens

The concept channel is the mechanism behind the **larger** of Paper B's two results: +0.01430 from
enlarging the bank, against +0.01035 from conditioning. If concept answers do not help on Goodreads,
**that result does not port**, and the honest statement is that it is MovieLens-specific.

The item-channel half is untouched by this. The arena, split, teacher and tower are sound.

## Candidate explanations, untested, cheapest first

1. **Answer-value scale.** Most consistent with "one helps, many hurt". The signed value
   (NPMI + w_val·VAL, clipped to ±1, negatives volume-capped) is calibrated by statistics re-derived
   on Goodreads only minutes before this run. Note `m0` **tail** differs enormously between datasets
   (0.0104 ML-25M vs **0.0679** Goodreads), which suggests the two answer distributions are not on
   comparable scales. **Check: the w_val target and the negative cap on Goodreads vs ML-25M.**
2. **The fold operator saturating.** The gated fold adds a norm-capped shift per answer; if the shift
   is too large for this latent geometry, repeated application overshoots.
3. **Interaction with the tower's weak zero-answer prior** (recorded watch item: k0 is 8.6% below
   most-popular on Goodreads, vs 1.1% on ML-25M).
4. **Genuine.** Books may be less separable by broad shelf attributes than films are by genome tags.

⚠ (1) and (2) are *implementation* explanations and would be fixable; (4) is a *finding* and would
have to be reported as one. **They are distinguishable and none has been tested.** Do not write
either story up until one is.

## Four dataset-portability bugs found on the way, all now fixed

The pipeline was built for one dataset, so anything cached was keyed by filename alone. Each was
caught by a size mismatch, not by design — **on a dataset with a coincidentally matching item count
every one would have produced plausible, wrong numbers.**

1. `arm_n` / `graded_data`: hardcoded ml-25m proc, movielens ratings, and a **shared cache filename**.
2. Tower leak asserts: hardcoded 18,359 / 140,768 literals.
3. `signed_answers`: **cached item-mean and pre-registered thresholds** shared across datasets.
4. `select_i26c`: printed three hardcoded ML-25M numbers labelled *"step-0 of THIS run"*.

Plus one stale docstring: `concept_train_v2`'s `--mix` help documents five weights (the tower's
signature); the concept channel takes four in a different order. It raised a TypeError, but a
permutation with matching arity would have trained a scrambled mixture silently.

## What has NOT been run

The Phase-A gate battery (g3a, g3b, g6 with `--force`, g9, g5) and a fresh `concept_identity_lift`.
Held pending the author's decision, because R2 depends on the concept fold that just failed.
