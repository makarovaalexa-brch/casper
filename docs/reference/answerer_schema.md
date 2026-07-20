# Answerer v1 — the locked answer schema (single source of truth)

Status: **DRAFT — not locked until author sign-off.** Machine-readable form: `casper/answerer_schema.json`.
Origin: `casper/ANSWERER_V1_DESIGN_REVIEW.md` Decisions **D** (answer scale) and **E** (dual-answerer).

This file is imported by three consumers that must never disagree:
1. the **judge prompt** (what the LLM is asked to emit per cell),
2. the **cache format** (what each judged cell stores),
3. the **fold interface** (what the recommender consumes).

## Knowledge (the vividness hedge, 3-level)
`knowledge ∈ {no_clue, rough_idea, know_well}`

| level | meaning | value present? | fold weight (default) |
|-------|---------|----------------|-----------------------|
| `no_clue` | never heard of it / cannot answer | **no** | 0.0 |
| `rough_idea` | knows of it / barely remembers ("I think I liked it?") | yes | `w_rough` = **0.5** |
| `know_well` | seen it / remembers well | yes | 1.0 |

`can_answer` (the design's yes/no) is **derived**: `can_answer == (knowledge != no_clue)`. Not stored twice.

The design review's Decision D3 named this axis *vividness {strong / vague}* plus a separate *can_answer {yes/no}*.
v1 collapses those two binary fields into one ordered 3-level `knowledge` field: `no_clue` = can't answer,
`rough_idea` = vague, `know_well` = strong. Same information, one field, ordered for the fold weight.

## Value (4-level sentiment)
`value ∈ {hated, meh, liked, loved}`, present **iff** `knowledge != no_clue`.

Centered fold values (what the fold arithmetic uses):

| value | centered |
|-------|----------|
| hated | −1 |
| meh   | −1/3 |
| liked | +1/3 |
| loved | +1 |

## Fold confidence weight = f(knowledge)
The centered value is scaled by a knowledge-dependent weight before folding:
`fold_value = centered(value) × weight(knowledge)`.
`weight = {know_well: 1.0, rough_idea: w_rough, no_clue: 0.0}`.
**`w_rough` is a KNOB (default 0.5), to be fit later** by masked-value validation — documented as a parameter, not a constant.

## Rating → scale mapping (rated cells only)
A real ML-25M star rating maps to the value scale (and implies `knowledge = know_well`, since a rating means the user saw it):

| stars | value |
|-------|-------|
| ≥ 4.5 | loved |
| 3.5–4.0 | liked |
| 2.5–3.0 | meh |
| ≤ 2.0 | hated |

Fixed and documented. (0.5-star bins between the ranges, e.g. exactly 2.5 vs 3.0, fall as written: 2.5–3.0 → meh, ≤2.0 → hated. There is no 2.25 in a half-star dataset.)

**Value elicitation (stars-then-bin, judge-side).** The judge does **not** emit the 4-level `value` directly. For every non-`no_clue` cell it predicts a **star rating** (0.5–5.0, half-star steps), and we bin it deterministically to the value scale with the exact `stars → value` table above (≥4.5 loved, 3.5–4.0 liked, 2.5–3.0 meh, ≤2.0 hated). The cache stores **both** the raw `stars` and the binned `value`. Rationale: the pilot showed a direct 4-level ask collapses to a modal "liked" (~0.94); a continuous star ask recovers the full-range variance and is binned mechanically. This is a change to the judge's internal elicitation format only — the fold interface (this schema) is unchanged.

## Channels
`channel ∈ {item, concept, attribute, pair, recall}`.

- **item** — entity = dense item id (movieId = `keepI[id]`). Rated → real rating; answerable-but-unrated → LLM graded answer (Decision E canonical).
- **concept** — entity = genome tagId. Value primary = data-side member aggregate (where ≥k rated members); LLM fallback where <k (flagged).
- **attribute** — entity = director/actor/composer/writer/franchise id. Value = data-side aggregate of the user's ratings over the entity's member films (LLM fallback).
- **pair** — entity = (item_a, item_b), both answerable. Value = which rated higher (real where both rated; LLM-compared otherwise).
- **recall** — open-recall framing menu {favourite, hidden_gem, hated}, each usable **once** per interview. v1 = Paper-D popularity-tilted sampler; LLM recall model = v1.1.

## Dual-arena rule (Decision E, pre-registered)
Every headline runs under two answerers; claims stated where both agree, disagreements reported as scope.

- **Arena-F (full)** — PRIMARY. Real rating where rated; **LLM graded answer where answerable-but-unrated** (the LLM is the simulated human there).
- **Arena-R (rated-only)** — hard ground-truth lower bound. Only real-rating cells answered; answerable-but-unrated = not answered.

Fidelity σ for the LLM-valued channel: masked-item validation MAE ≈ **0.70 stars** ≈ human test-retest inconsistency (Amatriain). Re-anchored per validation gate V3.

## What the fold consumes (contract)
`(channel, entity, value_4level, knowledge, fold_value_centered, fold_weight)`.
Raw LLM confidence floats are kept for analysis but **not** exposed to the agent/fold.

## Author sign-off checklist
- [ ] knowledge = 3-level `{no_clue, rough_idea, know_well}` (collapses D3's vividness×can_answer). OK?
- [ ] value = 4-level, centered (−1, −1/3, +1/3, +1). OK?
- [ ] `w_rough` default 0.5 as a fit knob. OK?
- [ ] rating→scale bins as above. OK?
- [ ] five channels + dual-arena rule as above. OK?
