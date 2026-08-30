# C4C — decoding, not retraining: the mode collapse is fixable, but the mechanism is not what we predicted

**Status: COMPLETE (validation/train-side; no test look). 2026-08-30.** Author-approved as Step 1 of
the Paper C polish. Code `src/instrument/c4c_decoding.py` (committed pre-run `38f012b`); artifacts
`experiments/paperC/c4c_decoding.json` + per-state npz (`e9f2571`). Conclusions Fable's own.
Runtime 10.7 min. Decoding spec pre-registered before the run, with the pick rule deliberately
blind to the evaluation metric (best-of-k selects on **LM likelihood**, never on cos to R_s).

**Reproduction gate**: the greedy arm was regenerated, not read, and reproduced the banked row
exactly — 42 renders bit-identical, cos to R_s to max|diff| 0, triangulation split (18 unique,
median rank 91 / 362.5, floor 340) and the C4B cost (+0.018025 / +0.006721) all exact.

**One-line verdict: the hypothesis is confirmed on its number and refuted on its mechanism.
Decoding alone fixes the collapse (23 → 41 distinct renders) and cuts the translation cost 2–3.5×
with non-overlapping CIs — but the sampled renders are LESS faithful and land FURTHER from the
point's own neighbours, so the cost fell because the words became less harmful, not more accurate.
Best-of-8-by-likelihood is the one arm that buys diversity and cost without paying faithfulness,
and it is hereby the decoding of record.**

## The four arms

| arm | distinct/42 | cos(u_text, R_s) | Δ vs greedy | median rank (of 1,031) | L−T full [CI] |
|---|---|---|---|---|---|
| greedy (banked) | 23 | **+0.2039** [+0.117, +0.291] | — | **313** | +0.01802 [+0.01668, +0.01940] |
| sample T=0.7 | 38 | +0.1034 | **−0.1005** [−0.197, −0.004] | 441 | **+0.00520** [+0.00397, +0.00638] |
| sample T=1.0 | **41** | +0.0942 | **−0.1097** [−0.220, −0.002] | 464 | +0.00746 |
| **best-of-8 (LM)** | 37 | **+0.1846** | −0.0193 [−0.097, +0.058] **n.s.** | 395 | **+0.00884** [+0.00747, +0.01025] |

The 4.2× concentration of cost on stock-word states is gone: those 20 states fall
+0.02871 → +0.0099 / +0.0123 / +0.0095, while the other 22 stay flat. The cost is now roughly
uniform — which is itself the tell.

## ★ The mechanism, and why it qualifies C4B's headline

The absolute levels (declared-circular, never headlined) give it away: base (no fold) **0.16571**,
L (the direction's own top concepts) 0.17443, T under greedy **0.15641**, T under sampling
**0.16924**. **Greedy's stock words fold to BELOW the no-fold baseline** — asking "horror" of a
Harry-Potter-ish cohort is worse than asking nothing. Sampling's vaguer words simply damage less.

So "translation cost" is not purely fidelity lost in translation; a large part of it is **harm from
confidently committing to the wrong concept**. A hedged word costs less than a wrong-but-decisive
one. That is the same shape as C4B's retired-twin echo and the thesis's standing bank-beats-
conditioning lesson, and it means C4B's reading — *"the noise is decoding diversity, not
verbalisation"* — needs its second clause amended: **the noise is removable by decoding, but what
sampling removes is harm, not confusion.** C4B's numbers stand; that sentence is now qualified here.

Corroborating: triangulation gets *worse* under every non-greedy arm (top-10 hits 4 → 3 → 1), and
the unique/repeated split **inverts** — under greedy, escaping the mode meant landing on the right
neighbour (unique median rank 91); under sampling, the diverse words are mostly not neighbours
(unique 459) and the few repeats are. Sampling buys coverage from the distribution's tail.

## Ruling

1. **Best-of-8 by LM likelihood is the decoding of record** for anything measured: it is the only
   Pareto point (37/42 distinct, faithfulness statistically unchanged, cost halved).
2. **The demo table shows BOTH greedy and best-of-k**, side by side, because the variance is real
   and hiding it would be a choice. Best-of-k repairs greedy's collapses — `religion`→**`scifi`**
   (rank 620→61), `religion`→**`animals`** (479→54), `hollywood`→**`ocean`** (963→133),
   `science`→**`science fiction`** (31→**7**) — while greedy keeps hits best-of-k loses
   (`art` rank 4 → `sela` 873; `indictment` rank 5 → `standard-of-life` 972).
   Unchanged across both and the best single row in the set: `root.noanswer.noanswer.dislike` →
   **`action`, rank 1 of 1,031**.
3. **No retraining is needed for the renderer.** Step 2 (richer adapter supervision) is DEMOTED to
   optional — decoding recovered the diversity it was meant to buy.

## Caveats

42 nested, structurally-selected states (C1's population caveats travel) · concept channel, blend
targets only · the cost arm runs under the declared-circular simulator and carries only the
relative reading · sampled arms are reproducible only at fixed batching (declared; `gen_batch` ≥ 42
asserted) · the run's dirty-src guard was import-scoped to this run's own modules because four
Paper-B files are dirty in the tree; the intersection with loaded modules was empty and the
excluded files are named in the artifact.
