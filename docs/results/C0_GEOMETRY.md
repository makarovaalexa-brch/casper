# C0 — the geometry of E1's emitted vector, before the snap

**Status: C0 COMPLETE (validation only; no test look taken). 2026-08-21.**
Design sheet: `docs/design/DS_C0_GEOMETRY.md` (pre-registered 2026-08-21, binding). Code:
`src/instrument/c0_geometry.py`, committed **before** the run at
`e5fa5dc5f55e5464e31dfb6c3309862184f68e3b` (HARD RULE 10). Checkpoint:
`experiments/battery/policy_main_equivariant_both.pt`, epoch 2, sha256
`9c4c61e7387d29c61f4cee62600627eff6178d2dbe3fb4ea6b5a19f7e6003058`. Wall clock **17.6 minutes**.

**One-line verdict (REWRITTEN 2026-08-24 after the hostile-review + repair pass; the original
overclaimed — see the Repaired reading below): the registered on-grid prediction is falsified —
max cos₁ over every state is 0.1203, so no emission sits on its question — but the registered
reading rule has no power to characterise the emission beyond that: a norm-matched random direction
lands in the same registered cell, so the cell describes the rule, not the policy. The informative
statement needs the mean-removed diagnostic added in the repair pass: the emission is a large
common-mode offset PLUS a genuine bank-aligned direction (mean-removed cos₁ to the snapped entry:
median +0.488 vs +0.170 for a random direction against the same entry).**

> Superseded original ("FALSIFIED, and not narrowly… −0.182… 0.410… `between_candidates`"): the
> point estimates stand but the framing implied an effect size the rule cannot license, quoted
> "160,000 states" without the clustering statement, and the raw-space "random is closer to its
> snap than the policy to its selection" comparison was a common-mode artifact.

The one metric that does *not* say "between": the softmax over the top-50 is essentially degenerate
(median entropy 0.0055 nats), because the absolute gap to the runner-up is large — median s₁ − s₂ =
7.81 logits. Both are true at once and the write-up keeps them together: **one dominant target in the
softmax sense, but a vector that is nowhere near that target's direction, sitting inside a spread of
runners-up 2.4× wider than its own lead.**

---

## What was run

| | |
|---|---|
| policy | the banked E1 checkpoint, unmodified, frozen |
| states | E1's own validation rollout — **10,000 validation users × 16 positions = 160,000 states**, regenerated through `policy_net.rollout` unchanged |
| bank | the full **19,390**-question table (18,359 items + 1,031 concepts), descriptors standardised with the *training* pool's statistics from the checkpoint |
| emitted vector | captured by a forward hook on `pol.head.q` during the rollout — no rollout, scoring or selection code was reimplemented |
| sampling | **none**. All 160,000 states × all 19,390 bank entries; chunked over states at 1,000/chunk, losslessly |
| firewall | validation only; the 10,000 test users are not built, not loaded, not touched. Nothing was trained. |

### ★ The bias term, and why the emitted vector is 129-dimensional

The design sheet writes the snap as `argmax_q ⟨ẑ, e_q⟩`. The head's actual score is

```
s(q | state) = ⟨u(h_state), k(φ_q)⟩ + b(φ_q)
```

— there is a **per-candidate bias**, so a bare `u(h)` / `k(φ)` inner product is not the rule the policy
maximises. Discarding it would have described a scoring function the policy does not use, and gate 1
could not have passed exactly. The lossless fix is homogeneous coordinates — an identity, not an
approximation:

```
ẑ = [ u(h), 1 ]        e_q = [ k(φ_q), b(φ_q) ]        ⟨ẑ, e_q⟩ = s(q | state)   exactly
```

so `d_k = 128` and the emitted vector is 129-dimensional. The identity is asserted against the head's
own forward at run time: max |difference| **3.05e-4** on scores of magnitude ~80 (float32 rounding).
The bias-free variant is reported alongside as a secondary number; it changes nothing (median cosine
−0.179 against −0.182).

---

## Gates — all passed, and they were read before any geometry number was

| gate | criterion | result |
|---|---|---|
| **G-C0-2a canonical snap** | validation m=0 intercept exact | `full = 0.15686538118354684`, `tail = 0.02747733123666661` — the canonical value, exact ✅ |
| **G-C0-2b banked rollout** | the regenerated realised interviews must equal the banked E1 artifacts | **byte-identical** to `policy_main_equivariant_both_gates_per_user.npz` (all 10,000 × 16); val full `0.28354971825495506` and tail `0.13421311934575303` reproduce the banked values **exactly**; 15 distinct interviews; answer rate 0.57889 ✅ |
| **G-C0-1 snap identity** | `argmax_q ⟨ẑ, e_q⟩` from the reconstructed emitted vector and the reconstructed bank must reproduce E1's own pick at every state | **0 mismatches of 160,000; top-1 agreement 1.0000** — in *both* arms ✅ |
| **G-C0-3 scale control** | a random vector scored by the same code | run; numbers below ✅ |

A fourth check fell out for free. The secondary arm (below) independently reproduces E1's banked
scalability table: out-of-pool pick rate **0.29992** (banked 0.299925), val full **0.263209** (banked
0.26320921), answer rate **0.41394** (banked 0.41394375), concept share **0.37932** (banked
0.37931875).

---

## The distributions (primary arm, 160,000 states, full 19,390 bank)

| statistic | median | IQR | p5 / p95 | min / max |
|---|---|---|---|---|
| **cos₁** (cosine to the snapped question) | **−0.18201** | [−0.32830, −0.06928] | −0.4567 / −0.0021 | −0.5543 / **0.1203** |
| **margin ratio m = (s₁−s₂)/(s₁−s₁₀)** | **0.40985** | [0.21470, 0.57909] | 0.0424 / 0.6850 | 0.0 / 0.9178 |
| entropy of softmax over the top-50 | 0.00549 | [0.00004, 0.41493] | 0 / 1.156 | 0 / 2.427 |
| candidates within 5% of \|s₁\| | 1 | [1, 2] | 1 / 13 | 1 / 47 |
| candidates within 5% of the s₁…s₁₀ range | 1 | [1, 1] | 1 / 2 | 1 / 4 |
| cos₁, bias-free bilinear space (secondary) | −0.17936 | [−0.32558, −0.06699] | | −0.5525 / 0.1258 |

Tail fractions that matter for the reading rule: **cos₁ > 0.98 at 0 states** (0.0000), cos₁ > 0 at
4.76%, p99 = 0.030. **m ≥ 0.90 at 0.006% of states**; m ≤ 0.60 at **86.3%**; m ≤ 0.25 at 28.7%.
≥3 candidates inside the 5%-of-|s₁| band at **20.8%** of states (the range-relative band: 0.26%).

### Why cos₁ is negative — mechanism, not a harness fault

The whole score vector sits far below zero: median s₁ = **−82.7**, and **95.2%** of states have a
*negative* top score. Cross-entropy constrains only score *differences*, so the head is free to place
a large common-mode offset in `k(φ)`/`b(φ)`, and it did. The consequence is that the argmax is decided
by a comparatively small variation on top of a large shared component that points away from ẑ:
median s₁ − s₂ = 7.81, median s₁ − s₁₀ = 17.50, against |s₁| ≈ 83. This is exactly the anisotropy the
sheet's scale control was written to expose, and it is why the two registered statistics disagree in
tone: the softmax has one clear winner, the direction does not.

### Per position

| t | cos₁ median | m median | m IQR | entropy median | ≥3 in 5% band | mean band count |
|---|---|---|---|---|---|---|
| 0 | −0.0693 | 0.5861 | [0.586, 0.586] | 0.0000 | 0.0000 | 1.000 |
| 1 | −0.3149 | 0.5322 | [0.261, 0.593] | 0.0000 | 0.0000 | 1.000 |
| 2 | −0.0564 | 0.4879 | [0.292, 0.570] | 0.0001 | 0.0227 | 1.119 |
| 3 | −0.1848 | 0.5097 | [0.442, 0.633] | 0.0000 | 0.0512 | 1.583 |
| 4 | −0.1255 | 0.4238 | [0.207, 0.574] | 0.0004 | 0.2538 | 2.026 |
| 5 | −0.1430 | 0.3255 | [0.229, 0.486] | 0.0337 | 0.1498 | 3.055 |
| 6 | −0.2622 | 0.4145 | [0.285, 0.590] | 0.0007 | 0.1630 | 2.776 |
| 7 | −0.2402 | 0.4559 | [0.246, 0.599] | 0.0081 | 0.1684 | 2.312 |
| 8 | −0.1827 | 0.3756 | [0.232, 0.470] | 0.0338 | 0.2651 | 3.493 |
| 9 | −0.2471 | 0.2803 | [0.133, 0.448] | 0.3983 | 0.3839 | 4.313 |
| 10 | −0.2035 | 0.3578 | [0.174, 0.568] | 0.1861 | 0.2262 | 2.853 |
| 11 | −0.1489 | 0.3906 | [0.168, 0.531] | 0.0497 | 0.2019 | 1.966 |
| 12 | −0.1689 | 0.3053 | [0.106, 0.612] | 0.1648 | 0.2035 | 2.082 |
| 13 | −0.1417 | 0.2378 | [0.113, 0.524] | 0.2693 | 0.2235 | 3.274 |
| 14 | −0.2458 | 0.2686 | [0.144, 0.414] | 0.2795 | 0.4114 | 5.020 |
| 15 | −0.2653 | **0.2009** | [0.086, 0.327] | 0.5320 | **0.6100** | 5.942 |

**The interview gets less decisive as it deepens, monotonically in the aggregate.** m falls from 0.586
at the root to 0.201 at the last position; the mean number of candidates inside the 5% band rises from
1.0 to 5.9; at position 15, **61% of states have three or more candidates in the band**. Position 0 has
a single state for all 10,000 users (every user gets the same root question), hence the degenerate IQR.

### By where the open bank sends the vector, and by channel

At **32.9%** of the primary arm's states, the argmax over the full 19,390 bank is a question *outside*
E1's 1,000-candidate training pool (the arm's own picks were made inside the pool; this is where the
open bank *would* send the same vector — the two agree at 67.1% of states).

| split | n states | cos₁ median | m median | ≥3 in 5% band | entropy median |
|---|---|---|---|---|---|
| full-bank snap **in pool** | 107,288 | −0.1867 | **0.4997** | 0.0897 | 0.00014 |
| full-bank snap **out of pool** | 52,712 | −0.1657 | **0.1816** | **0.4501** | 0.5329 |
| snap is an **item** | 100,367 | −0.1779 | 0.3466 | 0.2629 | — |
| snap is a **concept** | 59,633 | −0.1840 | 0.5398 | 0.1167 | — |

**The out-of-pool region is where the geometry is genuinely crowded** — margin ratio 0.18 against 0.50,
and 45% of those states have three or more near-ties. Cosine barely moves between the splits (−0.187
vs −0.166), i.e. the *direction* is equally far from the grid either way; what changes is how tightly
one bank entry wins. Concepts, the smaller and more separated channel, are picked more decisively than
items.

### The scale control (random unit direction, same code)

For every state, a random Gaussian direction rescaled to that state's own ‖u(h)‖ and augmented with
the same bias coordinate, then pushed through the identical `geometry()` call:

| | emitted ẑ | random floor |
|---|---|---|
| cos₁ median | −0.18201 | **+0.06418** (IQR [0.008, 0.118], max 0.398) |
| m median | 0.40985 | **0.26678** (IQR [0.129, 0.433]) |
| entropy median | 0.00549 | 0.90049 |
| ≥3 in 5%-of-\|s₁\| band | 0.2084 | 0.3264 |
| distinct questions snapped to | — | 8,243 |

Two things the floor establishes. **(1) m = 0.41 is not trivially high — but it is not far above the
floor either**: a vector with no information about this bank already scores 0.267, so the policy's lead
over "any vector at all" on this statistic is 0.14. **(2) The negative cosine is a property of the
trained emission, not of the bank**: a random direction achieves a *higher* cosine (+0.064) to whatever
it happens to snap to than the policy achieves to the question it deliberately selects. Neither reaches
0.98 anywhere (0 states in either).

### The secondary arm — E1's own open-vocabulary rollout

The same measurement on the states E1 visits when it chooses from all 19,390 candidates (its
scalability arm; 1,168 distinct interviews, val full 0.2632, answer rate 0.414). It carries no gate —
these are not the states G-E1b scored — and it exists so the in/out-of-pool split can also be read on
realised open-vocabulary interviews. It says the same thing: median cos₁ **−0.18063**, median m
**0.40693**, ≥3 in band 26.4%, random floor cos₁ 0.064 / m 0.267 — cell **`between_candidates`**.
Out-of-pool states again carry the crowded geometry (m 0.244 vs 0.467 in-pool; ≥3 in band 52.3%).

---

## The registered cell

| registered cell | criterion | this result |
|---|---|---|
| "on the grid" (prediction confirmed) | median cos₁ ≥ 0.98 **and** median m ≥ 0.90 | **NO** (−0.182; 0.410) |
| **"between candidates" (prediction falsified)** | median m ≤ 0.60 **or** ≥25% of states with ≥3 in the 5% band | **YES** — median m = 0.410 ≤ 0.60 (the second trigger, 20.8%, does *not* fire on the primary arm; it does on the secondary at 26.4%) |
| intermediate | otherwise | n/a |

**Both arms land in `between_candidates`.** Per the sheet, the distributions are the result either way
and are banked regardless of which sentence they license.

## What this does and does not license

**Does.** The pre-C1 baseline geometry is now a measured quantity rather than an assumption, and C1's
"the emissions moved off the grid" claim has something to be measured against. The `m` and `cos₁`
distributions above, and the random floor, are that reference.

**Does not.** This is not evidence that ẑ carries useful off-catalogue *content*. A vector can be far
from every bank entry and between several of them for the mundane reason that the head never had a
reason to place it anywhere in particular — blocker 1 is a statement about the *training target*, and
C0 measured the *emission*, so the sheet's prediction failing does not repair the blocker. Nor does a
low margin ratio mean the interview is uncertain in any decision-relevant sense: the softmax is
degenerate at the median state, and E1's realised interviews are still exactly the tree's. The honest
sentence is **"the emitted vector does not sit on its snapped question, and the runners-up are spread
widely enough that the top gap is under half the top-10 range — most of all late in the interview and
in the out-of-pool region"**, not "E1 already emits novel questions".

## Artifacts

| what | where |
|---|---|
| summary, gates, per-position / per-split tables, checkpoint sha, git rev + status | `experiments/paperC/c0_geometry.json` |
| per-state `cos₁`, `m`, entropy, band counts, `s₁ s₂ s₃ s₁₀`, position, user row, snap id, snap channel, in/out-of-pool, plus the random-floor per-state vectors — both arms | `experiments/paperC/c0_geometry_per_state.npz` |
| run log | `experiments/paperC/c0.out` |
| code | `src/instrument/c0_geometry.py` (commit `e5fa5dc`) |

## REPAIRED READING (2026-08-24) — Fable, from `experiments/paperC/stats_repair.json` (commit 019c060)

The hostile-review pass (statistics + harness lenses) forced five corrections; the repair pass
banked the numbers. Everything below supersedes the corresponding original sentences.

1. **The sample unit.** The 160,000 rows are 10,000 users × 16 nested positions over **31,874
   distinct asked-history states** (per position: 1, 3, 9, … 6,153) and 6,836 distinct
   (asked, answers) interviews; **99.95% of the cos₁ variance is between-state** (≥99.9% at every
   position). The emitted vector is essentially a function of the discrete answer history, not of
   the user. All original tail fractions are user-row-weighted; the declared unit is now stated and
   the state-weighted column is the robustness row: median cos₁ **−0.260** (vs −0.182), median m
   **0.286** (vs 0.410 — state-weighted, the margin sits almost on the random floor of 0.267).
   Cluster-bootstrap 95% CIs (cluster = user): median cos₁ [−0.18396, −0.18063], median m
   [0.40782, 0.41168].
2. **The rule has no power beyond the falsification.** Applying the registered reading rule to the
   norm-matched random floor lands it in `between_candidates` too. The cell is a property of the
   rule; only the max-statistic claim survives unconditionally (max cos₁ = 0.1203 ⇒ nothing sits on
   the grid).
3. **★ The mean-removed diagnostic (never run in the original) changes the interpretation.**
   Mean-removed cos₁ to the snapped entry: **median +0.488** (IQR [0.369, 0.587], positive at 98.9%
   of rows) vs **+0.170** for a norm-matched random direction scored against the same entries. The
   raw-space "a random vector is closer to its snap than the policy to its selection" was a
   common-mode artifact. Corrected mechanism statement: **the trained emission = a large shared
   common-mode component (which raw cosine sees) + a genuine bank-aligned informative direction
   (which raw cosine hides)**. "A scoring functional, not a semantic point" therefore holds only in
   the raw space; mean-removed, the emission is directionally meaningful though far from ON its
   question (0.49 ≪ 1) and still among many near-ties.
4. **What this changes downstream:** C1's baseline and the chapter's C0 section must quote the
   mean-removed pair (+0.488 vs +0.170) alongside the raw pair, and the bolded "not a semantic
   point in question space" sentence must carry the raw-space qualifier. The renderer counterfactual
   ("would be verbalising a common-mode offset") weakens to "would be verbalising a vector whose
   raw-space orientation is dominated by an offset".
5. Provenance repairs (harness lens): tail gates now asserted, E1 per-user npz sha pinned, index
   map asserted in all consumers, thread count recorded. ⚠ The edited asserts inside
   `c0_geometry.run()` itself have not executed since the edit (the 17.6-min rerun was not
   repeated); they are exercised by `paperc_stats_repair.py`. First re-run of C0 must confirm them.
