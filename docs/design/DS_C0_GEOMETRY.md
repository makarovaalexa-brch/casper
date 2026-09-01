# DS_C0 — the geometry check on E1's emitted vector (Paper C, stage C0)

**Status: PRE-REGISTERED 2026-08-21, NOT RUN.** Owner design: `PAPER_C_DESIGN.md` (Fable-ruled,
binding). Blocker analysis: `PAPER_C_EMISSION_DESIGN.md`. This sheet exists per the project rule
"design sheet before any run"; the run itself is minutes of CPU on the existing checkpoint and may
not start while a heavy job (Paper B work) holds the box without checking free RAM/CPU first.

## The question

E1 is trained by cross-entropy toward the tree's one-hot pick, so nothing in its loss rewards
landing *between* two questions (blocker 1). **Does the emitted vector `ẑ` nevertheless carry any
off-catalogue content, or does it sit essentially on its nearest bank question?**

**Prediction to falsify (registered):** `ẑ` sits on `e₍₁₎` — cosine to the nearest bank question
near 1, margin ratio near 1 — and carries no off-catalogue content.

## What the answer changes (why this runs at all)

- It does **not** gate C1 — C1 (the DGRL [was mis-cited as "DBU"; DBU is only its update rule] blend target) runs on either outcome. C0 fixes the
  **pre-C1 baseline geometry** that C1's "the emissions moved off the grid" claim is measured
  against, and it is the harness check for every later geometry number.
- If `ẑ` already sits between candidates with small margins, C1 has a head start and the chapter
  says so; if it sits on the grid, that is the clean statement of blocker 1 with a number attached.
- Per `PAPER_C_DESIGN.md`: C1 failing to pull `ẑ` off the grid later kills the paper at the price
  of this check. C0 is what makes that price payable.

## Inputs (exact, all frozen — nothing is trained)

| what | where |
|---|---|
| policy | the banked E1 checkpoint (the artifact scored in `docs/results/E1_POLICY_IMITATION.md`; locate via `src/instrument/policy_net.py` conventions and **record path + sha in the output artifact**) |
| states | E1's own validation rollout: **10,000 validation users × 16 positions = 160,000 states** (the same states G-E1b scored; regenerate through `policy_net.py`'s existing rollout path — no new state construction) |
| bank | the full 19,390-question embedding table `e_q` E1 snaps over (the open-vocab table already in `policy_net.py`) |
| emitted vector | `ẑ` = the continuous query vector the head produces BEFORE the snap `argmax_q ⟨ẑ, e_q⟩` (`PAPER_C_EMISSION_DESIGN.md`'s definition) |

No test users. No training. No answer model is consulted beyond replaying the banked rollout
(the circularity ruling's harmless use: the simulator generates states, nothing is scored through it).

## Metrics (per state; report distributions over all 160,000 and per position 0…15)

Let `s_k = ⟨ẑ, e₍k₎⟩` over the bank sorted descending, and `cos₁ = cos(ẑ, e₍₁₎)`.

1. **`cos₁`** — cosine to the nearest bank question.
2. **Margin ratio `m = (s₁ − s₂) / (s₁ − s₁₀)`** — the registered statistic from
   `PAPER_C_EMISSION_DESIGN.md`. `m → 1` = one dominant target; `m → 0` = sitting among several.
3. **Effective support**: entropy of `softmax(s)` over the top-50, and the count of candidates
   within 5% of `s₁` — how many questions the vector is "between".
4. **Split by snap identity**: in-pool (1,000) vs out-of-pool picks (E1 makes 29.99% OOV picks —
   report whether OOV states have different geometry), and concept vs item channel.

Primary readout = the **median and IQR of `cos₁` and `m`** over all states. MDE logic does not
apply (no arm comparison; this is a measurement, and both outcomes are reportable).

## Registered reading rule

- **"On the grid" (prediction confirmed):** median `cos₁ ≥ 0.98` AND median `m ≥ 0.90`.
- **"Between candidates" (prediction falsified):** median `m ≤ 0.60`, or ≥25% of states with ≥3
  candidates inside the 5% band.
- Anything between: report as intermediate; the full distributions are the result either way and
  are banked regardless of which sentence they license.

## Controls / sanity gates (must pass or the instrument is broken)

1. **Snap identity**: `argmax_q ⟨ẑ, e_q⟩` at every state must reproduce E1's banked pick
   (top-1 agreement 1.0000 is the E1-certified property; any mismatch = harness bug, stop).
2. **Canonical snap**: the m=0 intercept and the rollout's realised interviews must match the
   banked E1 artifacts (`policy_main_equivariant_both_gates.json`) before any geometry number is
   read.
3. **Scale control**: a random unit vector in the same space, scored with the same code — its
   `cos₁`/`m` distribution is the floor that shows the metrics are not trivially high for ANY
   vector in this bank (the bank is large and anisotropic; this is the cheap version of the
   spectrum-matched null that C2 will need in full).

## Shortcuts and their non-lossy alternatives

- None taken: all 160,000 states, the full 19,390-question bank, no sampling (HARD RULE 1).
  If memory forces chunking, chunk over users (lossless), never subsample.

## Cost & outputs

- Cost: one forward pass per state + one 19,390-dot-product per state ≈ minutes; float32; check
  free RAM and that no heavy job is running before launch (CLAUDE.md rule 11d/11e).
- Outputs: `experiments/paperC/c0_geometry.json` (summary stats, gates, checkpoint sha, git rev)
  + `c0_geometry_per_state.npz` (per-state `cos₁`, `m`, support count, position, snap id).
  Write-up → `docs/results/C0_GEOMETRY.md`. Code: new `src/instrument/c0_geometry.py`, which
  CALLS `policy_net.py`'s existing rollout/scoring path rather than reimplementing it
  (the E1 lesson: same code path or it proves nothing).
- Commit the code before the run (HARD RULE 10).
