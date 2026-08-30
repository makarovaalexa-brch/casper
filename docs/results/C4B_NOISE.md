# C4B — the translation cost, under a DECLARED-CIRCULAR answer model (relative claim only)

**Status: COMPLETE (validation only; no test look). 2026-08-25.** Design
`docs/design/DS_C4B_TRANSLATION_NOISE.md` under the author's same-day ruling (recorded verbatim in
`PAPER_C_DESIGN.md`: circularity admitted, declared, field-consistent — Info-Synth / PEBOL / our
June cans rule cited beside the declaration — and only the RELATIVE claim carried). Code
`src/instrument/c4b_noise.py` (committed pre-run `1a9a4f7`); artifacts `experiments/paperC/`
`c4b_noise.json` + per-state npz (`b6766de`). Conclusions Fable's own. Runtime 7.8 min.
Fold = the env's own certified operator in all four arms (incl. its `cap_negatives`); leak assert
clean; answer rate 48.2% positive.

**One-line verdict: translating the direction into English and reading it back costs
+0.01802 full / +0.00672 tail NDCG@10 [CIs excl. zero] under the shared simulator — and the cost
decomposes: states with a UNIQUE render pay +0.0063 while the mode-collapsed stock-word states pay
+0.0263 (4.2×). The translation noise is mostly the generation-diversity failure already diagnosed
in the triangulation, not an intrinsic price of verbalisation.**

## The readings

| reading | full | tail | verdict |
|---|---|---|---|
| 1 **translation cost L − T** (cohort-weighted, 36,197 user-rows / 42 nested states) | **+0.01802 [+0.01668, +0.01940]** | **+0.00672 [+0.00541, +0.00805]** | a real, non-uniform cost: 21/42 states CI>0, **5/42 CI<0 — the English beats the direction there** |
| 2 T − T′ (directional) | −0.01367 [−0.01521, −0.01217] | −0.00254 | **FAIL, reversed — see below** |
| 3 T − P (directional) | +0.02332 [+0.02180, +0.02478] | +0.01125 | PASS, weak (P is the degenerate single word "bribery") |
| 4 overlap \|L ∩ T\| | 0 at 38/42 states (mean 0.19) | | T is NOT L by coincidence — the cost is a real semantic-transport measure |

Absolute rows, *under a declared-circular answer model*, printed and never headlined: base 0.16571/0.03041 ·
L 0.17443/0.05019 · T 0.15641/0.04347 · T′ 0.17008/0.04601 · P 0.13309/0.03222. (T sits below the
no-fold base on full while helping tail — a property of folding four ±1 concepts under this
simulator, shared machinery across arms.)

## ★ The decomposition (Fable's join of `c4b_noise` × `c4_triangulate`, banked per-state rows)

| render class | n | mean L−T (full) | median |
|---|---|---|---|
| unique render (triangulation's rank-91 group) | 18 | **+0.00633** | +0.00777 |
| **repeated** render (any word used more than once) | 24 | **+0.02628** | +0.01902 |
| **stock-word** render (`horror`×9 / `religion`×7 / `movies`×4) | **20** | **+0.02871** | — |

**[Corrected 2026-08-30]** The original table paired the *stock-word* label with the
*repeated-render* count and mean — two different groups in the artifact
(`repeated_render` n=24 also contains `fantasy`×2 and `good`×2). Both rows are now shown with their
own numbers. No conclusion changes (both classes pay ~4× the unique group), but the label and the
field must read the same thing — the project's own `verify-the-artifact-not-a-proxy` rule.

The worst three states are all `root.like.*` branches rendered as `horror` (up to +0.118); the
best include `theatrical` −0.079 [−0.119, −0.040] and `love` −0.035 — English *beating* the
direction. **The lever for C-final is decoding diversity, not the verbalisation principle.**

## ★ T − T′ reversed, and the twin is hereby RETIRED as a per-point render control

A′ emitted 8 distinct texts over 42 states (`horror`×16, `fantasy`×14, …): its "renders" are
near-constant generic popular concepts, which fold well for many users under this simulator. The
arm measured "generic popular word vs state-specific word", not correspondence — and the generic
word won on average. Two consequences, stated separately:
1. **Control verdict:** this is the third experiment where A′'s outputs fail as a per-point
   control (C3B's C2 fired downward; C4-pilot reading 1 n.s.; here reversed). The twin remains
   valid where it worked — TRAINING-time separation and held-out bank-question re-encode — and is
   retired from render-level comparisons. C-final's render controls are random-point renders,
   non-degenerate priors, and (new, this result) **a generic-popular-concept arm as a named
   comparator, not a null**.
2. **Substantive echo, worth one chapter sentence:** under the simulator, a generic popular
   concept beats the state-specific render on average — the same shape as the thesis's standing
   "the bank/static beats conditioning" lesson, now appearing at the render level. Whether a
   diversity-fixed renderer flips it is an open, testable question and is registered as such,
   not assumed either way.

## Caveats that travel with every number

Declared-circular simulator (both arms share it; only L−T and the directional controls are
readings, absolute rows are labelled) · 42 nested, structurally-selected states (C1 caveats; the
summary bootstrap treats states as strata and understates its interval — per-state rows are the
primary object) · concept channel only · blend targets only (emissions await the declared rescale)
· the fold uses the env's `cap_negatives` operator identically in all arms.

## What this buys the chapter

The author's question — "how much noise does translating back add?" — has a number with a CI, a
decomposition that names the mechanism (decoding diversity), five states where English beats the
direction, and an honest reversed control that sharpens C-final's design. Together with C4-pilot's
answer-free faithfulness and the triangulation, the C3/C4 sections can now be written end to end.

## AMENDMENT 2026-08-30 — the mechanism clause is qualified by C4C

`docs/results/C4C_DECODING.md` tested this doc's own reading by changing decoding only. The number
holds and is removable: sampling/best-of-k cut the cost 2–3.5× with non-overlapping CIs. **But the
mechanism claimed above is wrong in its second clause.** Sampled renders are LESS faithful
(cos to R_s +0.204 → +0.103, CI excluding zero) and land FURTHER from the point's neighbours, yet
cost less — because greedy's stock words fold to **below the no-fold baseline** (T 0.15641 vs base
0.16571) while vaguer words merely damage less. **The translation cost is therefore substantially a
measure of HARM from confidently committing to the wrong concept, not purely of meaning lost in
translation.** "The noise is decoding diversity, not verbalisation" stands only as: *the loss is
removable by decoding* — what sampling removes is harm, not confusion.
