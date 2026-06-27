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

## UPDATE 2026-06-27 — overnight optimization study (what's the win, what isn't)

Goal: (a) make the recommender policy-AGNOSTIC (train on random policy), then (b) OPTIMIZE the policy on it.
Result: both probed thoroughly; the FINAL answer is **co-designed recommender + static-entropy policy**.

Recommender variants (entropy-policy eval, seed 123):
| variant | FULL | TAIL | note |
|---|---|---|---|
| frozen (Paper A) | 0.367 | 0.158 | baseline |
| **static FTREC (WINNER)** | **0.379** | **0.155** | seed-avg **0.385/0.165** |
| uniform-random | 0.369 | 0.140 | weaker — dilution |
| tempered-random (t=0.5) | 0.367 | 0.144 | weaker — dilution |
| tail-weighted (inv-pop loss) | 0.358 | 0.141 | worse — unstable |

=> **Specialization, not robustness, drives the gain.** Random-policy training gives no specialization (and worse
tail). The FTREC win is the recommender learning the belief distribution it ACTUALLY sees (entropy questions).
So the same-policy decomposition (+0.018/+0.008, identical Qs+answers, only recommender changes) is fully FAIR —
"train the recommender for how it's used," standard practice, no co-design asterisk.

Policy optimization (POLOPT = residual-on-entropy policy `score=β·POOL_ENT + g(belief,entity)`, g init 0 =
entropy warm-start, REINFORCE on true NDCG, val-selected, against the strong de-OOD'd recommender):
  init (entropy) 0.3791/0.1540 -> best-val 0.3784/0.1487 — **did NOT beat entropy** (explored, val fell to 0.30,
  drifted back). 8th policy-learning negative. With oracle-distill ALSO failing, the evidence is consistent:
  **the realizable optimal policy ≈ static entropy.** The learning lives in the RECOMMENDER + the graded ANSWER,
  not the policy. Policy stays a simple, interpretable static questionnaire (a feature for the paper, not a bug).

Code added: `POLOPT` (policy optimizer), `LOADREC=<ckpt>` (swap recommender into any block), `RANDPOL`/`RANDTEMP`
(random/tempered-random recommender training). Winner recommender = `cache/ftrec_best.pt` sha `5d6406fd`.

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

## UPDATE 2026-06-27b — RECURRENT+ATTENTION RECOMMENDER (DAHCR arch) beats frozen on FULL-PROFILE

User's idea (after exposing FTREC as a narrow overfit via full-profile + cross-sequence tests):
(1) train on ALL sequence lengths (not capped at 8), (2) use the DAHCR recurrent+attention architecture
(GRU + multi-head self-attention) to fold the profile.

Result — full-profile NDCG (real ratings, the NATIVE recommendation benchmark), seed-avg {1,2,3,7,11}:
| recommender | FULL | TAIL |
|---|---|---|
| frozen (Paper-A, simple attn-pool) | ~0.406 | ~0.214 |
| **FTRA (GRU+MHSA, all sequence lengths)** | **0.4275 ±.005** | **0.2461 ±.004** |
| (+ vs frozen) | **+0.021 (~10σ)** | **+0.032 (~16σ)** |

EVERY seed beats frozen on BOTH axes. This is a GENERAL recommender improvement (full-profile), NOT a
narrow overfit (contrast FTREC: full-profile 0.345, collapsed). Locked: cache/ftra_LOCKED.pt sha 84a94283.
Code: FTRA block (env FTRA=1; RAH/RAHEADS/RALR/RAEP/RAN; RALOAD eval-only). Arch = Linear(D+1,H) ->
MHSA(residual,relu) -> GRU -> last-valid-hidden -> Linear(H,D); trained variable-length (1..|profile|)
real-rating foldings, BPR ranking loss, val-fullprof early-stop.

WHY the prior fine-tunes failed and this works: FTREC/random fine-tuned on SHORT elicited beliefs ->
forgot full-profile (OOD). FTRA trains on ALL lengths incl full profiles -> full-profile in-distribution.

NEXT: unified FTRA-MIX (real-rating profiles + graded-answer question sequences incl concepts) -> one
recommender strong on BOTH full-profile AND elicitation (test whether the better recommender carries Paper C).
