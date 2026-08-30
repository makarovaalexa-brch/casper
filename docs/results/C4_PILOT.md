# C4 PILOT — the loop closed: policy point → English → fold, faithfulness measured answer-free

**Status: COMPLETE (validation only; no test look; no simulated answer to anything). 2026-08-25.**
Design `docs/design/DS_C4_PILOT.md` (pre-registered, author-approved); code
`src/instrument/c4_pilot.py` (committed pre-run `05e3225`); artifacts `experiments/paperC/`
`c4_pilot.json` + per-state npz (`247b5b5`). Conclusions Fable's own. Runtime 9.9 min.
All provenance gates passed (targets bit-identical; ingredients match stage0 exactly; adapters'
split hashes match; canonical snap exact).

**One-line verdict: for the BLEND TARGETS on the CONCEPT channel, the loop works end to end and
the readings say so with CIs — the rendered off-catalogue word carries the state's own
measured-ingredient direction through the independent fold, better than renders of random points
(+0.286 [+0.178, +0.398]) and better than the nearest catalogue question's own text
(+0.362, CI excl. 0), with the prior floor at zero. The EMISSION cells are retracted as artefacts
(the adapter cannot render the pilot's raw emissions — a scale mismatch, not a mechanism failure),
and the ITEM channel is disqualified as an instrument (its prior floor is not a floor).**

## What the vectors translate to (the author's question, answered concretely)

Concept-hot renders of blend targets are single words, and mostly sensible reads of the branch:
`root` → **"religion"** · `root.dislike` → **"horror"** · two dislikes deep → **"science"**
(+0.751 to its own reference) · the Toy-Story-adjacent depth-6 node → **"animals"** (+0.621).
Item-hot renders are hallucinated titles with real-title syntax ("Star Trek: The Next Generation
(1982)" +0.631, "Toy Story: The Movie (2003)" +0.698, "The Last of Us (1988)") — machine-only
synonyms again. The full 42-state verbatim table is in the npz; ~30 distinct texts for A vs 7 for
the twin (which collapses to near-constant "horror"/"fantasy"). Qualitative, not evidence — but it
is the first time a synthesised interview direction has been said in English on this project.

## The registered readings (paired 10k bootstrap, per cell, never pooled)

| reading | cell that carries it | result |
|---|---|---|
| 2 point-specificity A(real)−A(random) | **target/concept_hot** | **+0.28593 [+0.17820, +0.39777] — PASS** |
| 3 catalogue comparator (report-only) | all four cells | nearest bank question's gold phrase vs R_s = **−0.158 [−0.224, −0.089]**; A − catalogue positive with CI excl. 0 everywhere (target/concept +0.362) |
| 4 prior floor | concept channel | −0.013 [−0.074, +0.053] ✓ at zero |
| 1 faithfulness A−A′ | target cells | item −0.024 n.s.; concept +0.050 n.s. — **not separated**; A′ is not a clean null here (its near-constant "horror"/"fantasy" sometimes aligns by luck); reading 2 is the discriminating control at this n |

**★ The catalogue finding is the chapter's.** The nearest bank question sits at cos 0.93 to the
blend in the emitter's head space, yet its own text is *anti-aligned* (−0.158) with the blend's
measured-ingredient direction through the independent bridge — while the rendered off-catalogue
word is positively aligned (+0.204). Caveats attached: the k(φ)-nearest entry is itself a
common-mode artefact of that anisotropic space (it is frequently the same entry, `divorce`), and
the bridge favours concept-flavoured text; both stated wherever this number appears. Even so, this
is the first measured instance of the paper's motivating sentence: *the direction the policy wants
to ask about is expressible, and no near-catalogue question's text expresses it.*

## The emission cells are RETRACTED as artefacts (and the diagnosis is a finding)

34/42 (item-hot) and 32/42 (concept-hot) emission renders are the **empty string**; the bridge
embeds and folds even that, producing a fixed direction that spuriously scores +0.276. Restricted
to non-empty renders, every emission effect vanishes or flips (n ≤ 9). Cause, measured: the C1
emission's mean-removed norm is **38.3×** the target's and cos(target, emission) mean-removed is
**−0.078** — C1's "emission ≈ target at cos 0.99" was a raw homogeneous-space number. **Third
occurrence of the raw-vs-mean-removed trap in one week** (C0's cos₁, the split-half stability, now
this). The adapter was trained on target-scale inputs; the out-of-scale emission saturates the LM
into immediate EOS. Not a mechanism failure — an interface mismatch between the C1 pilot's
fine-tune and the C3B adapter's input distribution.

## Rulings for any C4-final (recorded now, before that data exists)

1. **An empty generation is not a rendering** — declared non-encodable, never folded.
2. **Emissions are rendered only after a DECLARED rescale to target-norm** in the mean-removed
   space (or C1-final trains the policy to emit in-distribution — preferred).
3. The random control must be norm-matched per point-kind; reading 1 needs a better null than A′
   at small n (A′'s degenerate outputs are not a per-point control); the item channel is reported
   as an instrument limit until its prior floor is fixed (the generic-title prior folds to +0.245
   against R_s — a floor above the signal).
4. R_s's item/concept norm disparity (Ew carries magnitude, d_c is unit) gets normalised per
   ingredient in C4-final; here it stands as the sheet defined it, disclosed.

## What may be claimed / never claimed

Claim (with the two catalogue caveats and the n=42 nested-population caveat): the loop
policy-point → English → fold works for blend targets on the concept channel; the rendered word is
point-specific (reading 2's CI) and carries the direction where the nearest catalogue text does
not. Never: NDCG, "asking it would help", any emission-cell number, any item-channel comparison,
human plausibility of the renders.

## Triangulation (author question, 2026-08-25) — Fable, from `experiments/paperC/c4_triangulate.json` (302af23/70b1754)

**Do the concept renders correspond to the point's own nearest bank entries? YES where the adapter
emits a distinctive word; NO where it repeats its favourites — and the split is diagnostic.**

- 18/42 states get a UNIQUE render: **median rank 91 of 1,031** in the tower ranking against the
  state's own reference, with genuine bullseyes — `action` rank **1**, `art` **4**, `indictment`
  **5** (a courtroom blend: Primal Fear / A Time to Kill), `animals` **7** (the Pixar branch),
  `military` **21**, `adventure` **28** (the Harry Potter blend), `science` **31**, `noir` **54**.
- **20**/42 states get one of the adapter's three stock words (`horror`×9, `religion`×7,
  `movies`×4); **24**/42 have a *repeated* render (those 20 plus `fantasy`×2 and `good`×2), and it
  is the repeated group whose median rank is **362.5** — no better than the random-tag floor (340).
  **[Corrected 2026-08-30: the original line attached the repeated-render count and median to the
  stock-word label. Pooled over all 42 the median is 313.0.]**
- Strict top-10 neighbourhood membership is 4/42 vs 0/42 random (Wilson intervals overlap) — the
  membership test is a harsher bar than fold-alignment and does not separate at this n.

**Reading: a GENERATION-DIVERSITY limit, not a geometry failure** — the same family as the
retracted emission cells (the adapter under-covers its output space; A′ collapses far worse, 7
distinct texts). Where decoding escapes the mode, the word lands among exactly the neighbours it
should. C3-final levers: sampling in place of greedy, a diversity term, or one model class up —
none needed for the mechanism claim, all needed before any per-state render is quoted as "the"
question. Caveats: the tower top ranks are award/production-tag dominated in action regions
(a `d_c` property, not the render's); the k(φ) ranking is the anisotropic space and stays a
secondary witness.
