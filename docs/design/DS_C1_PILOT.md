# DS_C1_PILOT — proof-of-concept: can a blend target pull ẑ off the grid?

**Status: PRE-REGISTERED 2026-08-21. Author approved a pilot/PoC this date** (*"what i want here is
a pilot, a poc, whether we can achieve it"*), noting the tree will probably be retrained anyway —
so this sheet deliberately claims NOTHING about generalisation or deployment; it tests capability
to fit. Owner docs: `PAPER_C_DESIGN.md` (C1), `C1_GAIN_TABLE_AUDIT.md` (data), `DS_C0_GEOMETRY.md`
+ `docs/results/C0_GEOMETRY.md` (the baseline every number here is read against).

## The question

C0 showed E1's emission is a scoring functional (median cos₁ = −0.182, common-mode offset), because
the one-hot CE loss never pins ẑ to any point. **If we replace the target with a gain-weighted blend
of measured candidates — off-catalogue by construction — can the SAME architecture be trained to
emit a vector that (a) lands on its target and (b) is genuinely between bank questions rather than
on one?** If it cannot fit even this, C1 (and the paper) dies here, cheaply — the registered
kill-gate of `PAPER_C_DESIGN.md`.

## Data

The **42 states with full exact 1,000-candidate gain tables** (`C1_GAIN_TABLE_AUDIT.md`: 35 from
`policy_labels_deep_partial` + 7 single-pick heaps; contract-verified against the tree bit-identically
15/15 where they overlap). No partial-heap states, no stale bounds, no new measurement (HARD RULE 1:
this is the entire exact-coverage population, not a sample of it).

## Stage 0 (free, BEFORE any training) — are the targets themselves off-grid?

For each state build `target = Σ_{j≤m} w_j e_{q_j}` with `w_j ∝ max(gain_j, 0)` over the top-m
measured candidates, normalised; **m=5 primary, m∈{3,10} secondary**. Blends live in the snap
embedding space e_q (both with-bias homogeneous and bias-free variants, as C0). Report each target's
cos to its nearest bank question and margin ratio (same code path as C0).
**If the gain profiles are so peaked that the blends collapse onto e₍₁₎ (median target-cos₁ > 0.99),
the pilot STOPS and reports that** — the premise, not the optimizer, would be at fault; a
temperature-softened weighting may be reported as an exploratory row but must be labelled as
abandoning gain-proportionality.

## Stage 1 — fine-tune and measure

- Start from the banked E1 checkpoint (the C0-measured object; sha in `c0_geometry.json`).
- Loss at the 42 states: `1 − cos(ẑ, target)`. Two arms: (A) head-only fine-tune, (B) full-policy
  fine-tune. 3 seeds each. No replay/regularisation arm is required for the PoC — collateral damage
  to behaviour elsewhere is REPORTED (snap agreement vs banked E1 at 1,000 random ordinary states)
  but not gated: this pilot's claim is capability, not a deployable policy.
- ⚠ 42 examples vs 526k parameters: trivially fittable in the loss sense. That is WHY the registered
  success criterion is geometric, not loss-based — the question is whether the head can EXPRESS an
  off-grid point (C0 showed its trained output carries a large common-mode component; nothing yet
  shows the architecture can place a vector at an arbitrary point of e_q-space).

## Registered reading

**SUCCESS (all three):** at the 42 states, seed-averaged
1. `cos(ẑ, target) ≥ 0.90` (it fits its target);
2. the emission's off-grid signature matches its target's: |cos₁(ẑ) − cos₁(target)| median ≤ 0.05
   AND ẑ's nearest bank question is the target's nearest for ≥ 80% of states (it is off-grid the
   way the target is, not snapped back onto e₍₁₎ and not floating unrelated to the bank);
3. across 3 seeds, per-state emitted vectors agree pairwise cos ≥ 0.95 (stability).

**FAILURE:** criterion 1 misses ⇒ the architecture cannot place a point (kill finding, report as
weak-method per HARD RULE 6 only if a plausible optimizer fix exists; otherwise this is the paper's
registered death). Criterion 2 misses with 1 passing ⇒ the optimizer found a cheating direction
(high target-cos via the common-mode component) — diagnose before any claim. Criterion 3 misses ⇒
report instability; no claim of a well-defined emitted point.

**What may be claimed on success:** *"the architecture can be trained to emit stable off-catalogue
points at states with measured blend targets"* — capability to fit, nothing about value,
generalisation, or interviews. What may NEVER be claimed from this pilot: any NDCG number, any
generalisation, any "novel question".

## Controls

1. C0's own machinery re-used verbatim for all geometry (its gates already certified it).
2. Shuffle control: targets permuted across the 42 states — trained policy must fit its OWN targets
   better than the shuffled assignment fits (else the fit is state-independent collapse).
3. The random-direction floor from C0 carried into every table.

## Cost & outputs

Minutes to low tens of minutes (42-example fine-tune × 6 runs + geometry); threads ≤ 4; check the
box before launching. Commit code before run (only own files staged — the tree is dirty with a
colleague's work). Outputs: `experiments/paperC/c1_pilot.json` (+ per-state npz), write-up
`docs/results/C1_PILOT.md`, code `src/instrument/c1_pilot.py` (calls `c0_geometry.py`/`policy_net.py`
paths, never reimplements).
