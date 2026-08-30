# DS_C3B_V2 — emit an askable QUESTION, not a bare phrase

**Status: PRE-REGISTERED 2026-08-30. Author instruction:** *"continue with 1, focus on llm emission
of reasonable text."* ⇒ (1) the generalisation claim is SCOPED, not chased — C1-final's state-local
result stands as written (`docs/results/C1_FINAL.md`); (2) the remaining work is rendering quality.
Predecessors: `DS_C3B_ADAPTER.md` (+ its 2026-08-24 amendments, still binding where not superseded
here), `docs/results/C3B_ADAPTER.md`, `C4C_DECODING.md` (best-of-8 is the decoding of record).

## The problem in one line

The adapter currently emits a **bare phrase** — `science fiction`, `animals`, `Rocky (2005)`. An
interview asks **questions**. A demo of a synthesised interview needs text a person could answer.

## The fix, and why it keeps the claim

Retrain the same adapter on **question-shaped targets** built from the same bank entries, so the
vector still conditions the whole utterance. This is NOT a cosmetic phrasing layer bolted on after
the fact (that would be the disqualified variant (a) — the LM would never see the vector).

**Targets.** Per bank entry, a small template family, one sampled per example **per epoch** (so the
model learns *question form + this content*, not one memorised string, without multiplying the data):
- concepts (1,031): e.g. *"Do you enjoy {tag} films?"* · *"Are you in the mood for something {tag}?"*
  · *"How do you feel about {tag} movies?"* · *"Do you like it when a film is {tag}?"*
- items (18,359): *"Have you seen {title}?"* · *"Did you enjoy {title}?"* ·
  *"What did you think of {title}?"*
Templates are fixed in code, listed in the manifest, and identical across both arms. ⚠ Some genome
tags are adjectival and some nominal, so no single template is clean for all — that is why there is
a family, and the awkward cases are a declared property of the vocabulary, not a bug.

**Everything else is held fixed** from C3B: same 19,390 pairs, same identity-held-out 20% split
(seed 98765, channel × popularity decile), same adapter (2-layer MLP → 8 soft prefix tokens into
FROZEN distilgpt2), same warm-free init, same lr 1e-3 / batch 32, same scrambled twin A′ trained
identically on **channel-preserving** permuted pairs. Epochs 6 (C3B's best was epoch 4).

## Registered readings — the claim must SURVIVE the reformatting

Evaluated on the held-out 20% of bank questions, in the amended **tower** space, per channel, every
statistic with a paired bootstrap CI (10k, seed 98765), reusing `c3b_adapter.py`'s `stage2`
machinery unchanged:
1. **Conditioning survives (the gate).** A − A′ > 0 with CI excluding zero, and A reaching ≥40% of
   the true-phrase ceiling on the gate channel — i.e. C3B's criterion 1, re-run. ⚠ The ceiling must
   be **re-measured against the templated gold** (a question containing the phrase), not reused from
   C3B, because the instrument's input distribution has changed. Re-run `ceiling` first.
2. **Identification** ≥ chance + 60% of the new ceiling's headroom, CI excluding 0.50 (the
   ceiling-relative rule from the 2026-08-24 ruling).
3. **★ Question-form rate (new, the point of this run).** Fraction of held-out generations that are
   well-formed questions: ends in `?`, ≥4 tokens, contains a content word that is not template
   boilerplate. Reported per channel with a CI. Target stated in advance: **≥0.90**; below that the
   reformatting failed regardless of readings 1–2.
4. **Content preservation.** Token-F1 between the generated question's non-template content and the
   gold phrase, and the exact-content-match rate — the C3B analogue, now measured on content words
   only. ⚠ C3B found exact recovery 0.0000 with above-ceiling re-encode (machine-only synonyms);
   expect that to persist and report it as such, not as failure.
5. Decoding: **best-of-8 by LM likelihood** (the C4C decoding of record) is primary; greedy is
   reported beside it. The pick rule may never consult the tower cosine.

## Then, and only if reading 1 passes: the demo

Re-render the C1-final policy's own emissions at its 57 trained states (in-distribution now — the
scale term holds there, ratio 1.045) and the 42 pilot blend targets, and build the **synthesised-
interview transcript**: one cold-start interview, every question emitted and rendered, with the
belief updating. That artifact is the capstone figure. Held-out-state emissions stay excluded
(out of scale, per `C4_PILOT.md` ruling 2).

## ★ AMENDMENT 2026-08-30 (Fable) — the demo paragraph above contains a SPACE MISMATCH

Caught by the transcript build, and it is my error: the sheet says "re-render the C1-final policy's
own emissions", but **C1-final warm-started from Paper B's `m7dj_policy.pt`, so its emissions live
in m7dj's `k(φ)` read-out, while this adapter is trained on E1's** (`c3b_pairs.npz`). Same width,
same frozen `φ`, two different learned matrices — nothing guarantees a shared basis, and feeding one
to the other would silently render a *different direction*. (The C4 pilot escaped this because
C1-pilot fine-tuned E1 itself.)

**Resolution, in order:**
1. **Run the diagnostic first** — `c4d_transcript.py` banks a per-construction snap agreement (does
   the fed point's nearest bank entry equal the emission's nearest entry in its own space?). If it
   is high, `--space direct` is valid and the demo renders the emission as written.
2. **Otherwise use `--space target`**: render the blend **rebuilt in the adapter's own bank** — the
   same gain-weighted construction, expressed in the renderer's read-out. This is honest and loses
   little, because at trained states **cos(emission, target) = 0.970**: the policy's emission and
   the target are nearly the same object exactly where the figure draws from. The caption then reads
   *"the direction the measured gains point to at this state"*, with the policy's ability to emit it
   (0.970) stated as the separate, already-banked result.
3. The `lstsq` bridge is available but is a FITTED component and would need its own justification
   and R²; use it only if 1 fails and 2 is judged insufficient.

**Not a blocker for the run in flight** — it changes only which vector the figure feeds the
renderer, not whether the renderer works.

## What may NOT be claimed

That the questions are good because they read well (fluency is not evidence — C3B's machine-only
synonyms) · anything at states outside C1-final's trained set · any NDCG or value claim · that a
human would answer them (no human study).

## Mechanics

Extend `src/instrument/c3b_adapter.py` with a `--targets questions` mode (templates in code, listed
in the manifest) rather than forking it; new outputs under `experiments/paperC/c3b_v2/`. **Run
DETACHED** (`Start-Process`, as `run_c3b_arms.ps1` does) — session background bash has been reaped
twice. Threads ≤4; ~6 epochs × 2 arms; commit code before the run. Validation/train-side only.
