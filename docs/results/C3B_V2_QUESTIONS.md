# C3B-v2 — the renderer emits QUESTIONS, and the vector still conditions them

**Status: COMPLETE (validation/train-side; no test look). 2026-08-30.** Design
`docs/design/DS_C3B_V2_QUESTIONS.md` (pre-registered, incl. both amendments). Author instruction:
*"continue with 1, focus on llm emission of reasonable text."* Code `src/instrument/c3b_adapter.py`
(`--targets questions`, committed pre-run `5ffb0a4`; resumability + KV cache `68b3d21`); artifacts
`experiments/paperC/c3b_v2/`. Conclusions Fable's own. Adapters: true best ep5 (val CE 2.2620),
scrambled twin best ep5 (2.4634); registered early read **separated: true** (+0.177 val CE at ep2).

**One-line verdict: every registered reading PASSES on the gate (item) channel, under BOTH
decodings, and on the concept channel too — the vector conditions the text just as well when the
text is an askable question. 94.7% of held-out generations are well-formed questions (target 0.90).
The cost of naturalness is paid in the instrument, not the mechanism: templating drops the ceiling
23%, and content preservation is near zero (token-F1 0.055, exact match 0.0003) — the machine-only
synonym behaviour, now in question form.**

## The gate (item channel, 3,670 held-out questions; tower space; 10k paired bootstraps)

| reading | best-of-8 (primary) | greedy | bar |
|---|---|---|---|
| **1** A − A′ | **+0.05169** [+0.04595, +0.05750] | **+0.05855** [+0.05321, +0.06383] | CI excl. 0 ✅ |
| **1** A level vs required | **0.04869** vs 0.02731 | **0.07177** vs 0.02731 | ≥40% of ceiling ✅ |
| **2** identification | **0.6099** | **0.6392** | ≥0.5790 ✅ |
| **3** question-form rate | **0.9471** [0.9398, 0.9542] | — | ≥0.90 ✅ |
| **verdict** | **SUCCESS** | **SUCCESS** | |

Concept channel (not the gate, reported): A−A′ **+0.28625** [+0.24014, +0.33307], A 0.2795 vs
required 0.1663; identification 0.7810 vs floor 0.7343; question-form 0.9429 — all pass.

## What this establishes, and what it costs

1. **The mechanism survives reformatting.** The bare-phrase adapter gave A−A′ = +0.111 at 107% of
   its ceiling; the question adapter gives +0.052 at **71% of the templated ceiling** (0.0487 of
   0.0683) on items and +0.286 at 67% on concepts. Lower in absolute terms, comfortably above bar,
   and measured against a ceiling **re-measured under the new format** — which is why the drop is
   interpretable rather than alarming.
2. **★ Naturalness costs verifiability, and we can price it.** Wrapping a phrase in question
   boilerplate drops what a *perfect* rendering can score by **23% on items** (0.0892 → 0.0683) and
   17% on concepts. The boilerplate dilutes the signal the bridge reads. Nobody in the
   verbalisation literature reports this because nobody measures against a same-instrument ceiling;
   it is a small, honest methodological contribution of this chapter.
3. **Machine-only synonyms, in question form.** Content token-F1 **0.055**, exact content match
   **0.0003** — the adapter essentially never reproduces the gold title, yet its text identifies
   the right vector 61% of the time (chance 50%, ceiling 63%). Examples (held-out, best-of-8):
   *Shawshank Redemption* → **"Have you seen Pulp Fiction (1995)?"**; *Ghost Story (1981)* →
   **"Have you seen Battlestar Galactica (1980)?"**; *Time Traveler's Wife (2009)* → **"Did you
   enjoy My Little Pony: Friendship is Magic, The (2009)?"**; *Desperate Hours (1955)* → **"What
   did you think of Bunted, The (1941)?"**. Neighbouring era, neighbouring register, wrong film —
   exactly Hewitt's phenomenon, and exactly why a human plausibility panel would grade this
   renderer as failing while the instrument says it carries the vector.
4. **Degenerate outputs exist and are not hidden**: 5.3% of generations are not well-formed
   questions (e.g. *"Did you enjoy Til, Til, Til, …"*), and the strict all-zero-prefix prior never
   produces a question mark (0.00) while the channel-matched prior always does (1.00).

## Reviewed deviations (Fable's ruling)

Decode budget 16 → 32 tokens — **necessary**, a 16-token cap would truncate the closing "?" and turn
reading 3 into a measurement of the cap. Inner-val CE on a fixed epoch-0 template draw — **correct**,
it keeps best-epoch selection comparable. Content-F1 reported both literally and stop-set-symmetric
— **accepted**, titles legitimately contain boilerplate words. Criterion 2 (A′ vs prior) reported as
secondary, not part of the verdict — **correct**, per C3B's directionless-equivalence lesson.

## Engineering note that belongs in the record

The first attempt ran 4 hours without completing one arm and banked nothing; it was killed. Cause:
generation re-ran the full forward pass per token (no KV cache) — the fixed version is **11.1×
faster with byte-identical greedy output** (0 disagreements) — and stage 2 wrote only at the end,
violating HARD RULE 11. Both fixed: per-(decoding, arm) contract-signed sidecars with resume, and a
progress line every ~2 minutes. The resume test then caught a latent bug — `TowerBridge` was built
downstream of RNG state consumed by generation, so a resumed run moved cosines in the 5th decimal;
re-seeding before the bridge made resumed output byte-identical to fresh.

## What may NOT be claimed

That the questions are good because they read well (5.3% are malformed; content match ≈ 0) · any
value or NDCG-improvement claim · anything at states outside C1-final's fitted set · that a human
would answer them (no human study) · single training seed per arm, so no "robust".
