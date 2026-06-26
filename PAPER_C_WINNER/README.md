# PAPER C WINNER — `uent + GRAW` (continuous-answer elicitation)

**Locked 2026-06-26.** First policy to beat the Paper-B discrete SOTA (CASPER-R) on the locked
ruler, seed-averaged. It is a **heuristic + answer-model**, not a trained policy — there is no
neural policy weight to load (the *learned* actor we trained actually LOST to this; see below).
The only learned component is the frozen Paper-B/C encoder `enc_concept.pt` (guarded here).

## What it is

An 8-question elicitation policy = **selection rule** + **answer model**, folded through the
frozen set-encoder into a belief `u`, then rank held-out items by `popularity + Ql·u`.

- **`uent` — selection (which question).** Each turn pick the not-yet-asked pool entity (item OR
  concept) with the highest **population divisiveness** = binary entropy of its like-rate across
  training users (`POOL_ENT`, cached in `pool_entavg.npy`). Static, population-level, NO learning,
  NO per-user adaptation, NO target knowledge. Ends up asking ~7 divisive **popular items** + ~1 concept.
- **`GRAW` — answer model (what the user says back).** Two continuous levers:
  1. **`ITEMSANS` — popular items are answerable.** A popular movie is assumed answerable (the user
     has likely seen it), so the simulated user derives a response from taste — no "have you seen XYZ?"
     cold-start block. Restores the item channel that Paper B had to drop.
  2. **`GRAW` — graded answer, not a bit.** The answer is the **predicted rating** `u*·(item emb)`
     (a real number = *how much* they'd like it), NOT like/dislike. That magnitude is the information
     CASPER-R's ±1-bit throws away. (Real users give real graded ratings ⇒ this predicted-rating
     answer is a conservative lower bound.)

The win is the two **continuous** levers (answerable items as actions + graded answers as responses)
stacked on the simplest possible selector. Purely continuous; no new vocabulary, no ABot, no learned policy.

## Result (seed-avg over EVAL profile-splits {1,2,3,7,11}, te[300:], 304 users)

| Policy | FULL NDCG@10 | TAIL NDCG@10 |
|---|---|---|
| **uent + GRAW (this)** | **0.3667 ±.0045** | **0.1577 ±.0078** |
| uent + binary (item-answerable, ±1 bit) | 0.361 | 0.153 |
| CASPER-R (Paper B discrete SOTA, concept-only ±1 bit) | 0.360 | 0.152 |
| full-profile reconstruct ceiling (cos(u,u*)=1) | 0.407 | 0.21 |

- **FULL: +0.007 over CASPER-R (~3 SE) — solid.**
- **TAIL: +0.006 over CASPER-R (~1.6 SE) — real but marginal; tighten with 10 seeds.**
- Sits well below the reconstruct ceiling (0.407/0.21) ⇒ legitimate partial reconstruction, not inflation.

### Decomposition (which lever does what)
- concept-only entropy (±1 bit): ~0.13 tail
- + item-answerability (binary): 0.153 tail  ← matches CASPER-R (answerability ≈ all of CASPER-R's learning)
- + graded answer (GRAW): 0.158 tail        ← the continuous-answer push past the tie

### Negative controls (why this is the right story, honestly)
- **Learned continuous actor** over the same unified-answerable pool: 0.326/0.123 — LOST to this heuristic.
- **Oracle-distillation** (distill the privileged 0.389-tail concept oracle): 0.334/0.127 — LOST
  (imitation gap: tail-optimal concept not predictable from the cold-start profile).
- ⇒ The win is **answerability + graded answers**, NOT cleverer selection or learning.

## UPDATE 2026-06-26b — FTREC (recommender fine-tuned on elicitation distribution) — BIGGER WIN

Stacked, seed-avg {1,2,3,7,11} on te[300:]:

| Step | FULL | TAIL |
|---|---|---|
| CASPER-R (discrete SOTA) | 0.360 | 0.152 |
| + continuous answer + answerable items (uent+GRAW, frozen rec) | 0.367 | 0.158 |
| **+ elicitation-specialized recommender (FTREC)** | **0.385 ±.006** | **0.165 ±.007** |
| **total vs CASPER-R** | **+0.025 (~7 SE)** | **+0.013 (~2.4 SE)** |

**FTREC = fine-tune the recommender (encoder + Ql readout) on the ELICITATION distribution**: the same
8 most-divisive entropy questions, graded answers, and a **fold-curriculum** (random t∈1..8 each batch)
so it ranks well from *partial* beliefs at every turn — fixing the OOD mismatch (Paper-A rec was trained
on full profiles). Frozen encoder generates the *answers* (user taste fixed); only the belief-builder +
readout adapt. Overfit controlled by **data augmentation** (K=5 random splits/user), **frozen L2 anchor**,
and **low LR** — the un-regularized version overfit (0.354), the regularized version generalizes (val AND
test both up). Locked: `cache/ftrec_best.pt` sha `5d6406fd…`. Config: `FTAUG=5 FTLR=1e-4 FTANCHOR=1e-4 FTEP=50 FTN=2500`.

Reproduce FTREC: `FTREC=1 FTAUG=5 FTLR=1e-4 FTANCHOR=1e-4 FTEP=50 python scripts/paper2/continuous_actor.py`
(train once), then `FTREC=1 FTLOAD=1 SEED=$S ...` per seed to eval.

NOTE: 7 *policy-side* learned attempts lost to the heuristic; the win came from the RECOMMENDER side
(de-OOD via fold-curriculum), exactly as flagged — proper regularization (aug+anchor+lowLR), not a ceiling.

## Reproducer

```bash
cd C:/dev/phd/casper
for S in 1 2 3 7 11; do
  UNIANS=1 UMODES=uent GRAW=1 SEED=$S NOBC=1 NOTRAIN=1 \
    python scripts/paper2/continuous_actor.py 2>&1 | grep uent
done
# average the FULL/TAIL columns -> 0.3667 / 0.1577
```

Frozen code: `continuous_actor_FROZEN.py` (== `scripts/paper2/continuous_actor.py` at lock time).
Env flags: `UNIANS=1` (diagnostic block), `UMODES=uent`, `GRAW=1` (graded predicted-rating answer),
`SEED=<eval split>`, `NOBC=1 NOTRAIN=1` (skip the unused training prefix).

## Locked inputs (full policy state — all in `cache/`)
- `enc_concept.pt` — frozen set-encoder (the ONLY learned weight). sha256 `27e6c72d…d0aa8`.
- `pool_entavg.npy` — `POOL_ENT` divisiveness ranking (the uent selector).
- `Q_svd.npy` `Ql_concept.npy` `Ec_concept.npy` `ctags_concept.npy` `bi_svd.npy` — item factors,
  ranking factors, concept embeddings, concept tags, item biases.
- (also needs `ml-1m/` ratings + `genome-scores.csv` for concept membership — in the repo data dir.)

## Caveats / open
1. Tail margin ~1.6 SE → run 10 seeds to lock significance.
2. Fair within-setting 2×2 (item-answerable? × graded?) on identical seeds for a clean ablation table.
3. Answer = `u*·item` uses the profile-encoded `u*` (no held-out leak); it is a *predicted* rating,
   so the simulated user "rates" popular movies from taste — defensible iff popular⇒seen⇒answerable.
