# Paper C — design of record for the EMISSION arc (continuous questions)

**Status: DESIGN ONLY, nothing here has run.** Recorded 2026-08-17 from the author's ruling.
Owner doc for Paper B's policy machinery: `docs/design/PAPER_B_POLICY_DESIGN.md` (Parts D, L).
This sheet covers only what Paper C adds on top: emitting a question that **is not in the bank**.

## The claim Paper C wants to make

Paper B's policy emits a continuous vector `ẑ` and snaps it to `argmax_q ⟨ẑ, e_q⟩` over the
19,390-question bank. Paper C's claim is that the emitted vector can name a question **the bank
does not contain**, which an LLM then renders into natural language.

## ★ The blocker, stated exactly (author + Fable, 2026-08-17)

**Two independent reasons the current stack cannot make that claim, and they need different fixes.**

### Blocker 1 — the training target forbids novelty

E1 is trained by cross-entropy toward the tree's chosen question. That loss pulls `ẑ` *onto* the
target embedding. **There is no term anywhere in the objective that rewards landing between two
questions**, so the trained model has no reason to emit anything off-catalogue. The author's own
framing: *"with distillation there is no reason to expect in-between."* Correct.

### Blocker 2 — the snap makes novelty unobservable to any reward

RL does **not** fix blocker 1 on its own, for a structural reason. Whatever vector is emitted, the
snap replaces it with a real bank question before the environment sees anything. Two emissions that
snap to the same question produce **byte-identical** interviews and therefore identical reward.
**A reward acting through the snap cannot distinguish a novel direction from its nearest neighbour.**
So "add RL" is not a route to novelty; it is a route to better *selection* within the bank.

## The two fixes, in order

### Fix A (training) — blend targets, no new answerer needed

Replace the one-hot target with a **gain-weighted blend of the top-m measured candidates** at each
state: `target = Σ_j w_j · e_{q_j}`, with `w_j` from the measured gains of those candidates.
The target is genuinely off-catalogue by construction, and **every gradient still comes from
questions whose value we actually measured** — no simulation, no clairvoyance. This is the
"Breaking the Grid" / DGRL [was mis-cited as "DBU"; DBU is only its update rule] pattern already summarised in `external_literature/`.

Cost: cheap. The per-state measured candidate gains are exactly what the labelling harness already
produces (`policy_labels.py`); this is a change of target, not of data.

### Fix B (environment) — an answerer that responds to arbitrary directions

Novelty only *pays* if something downstream consumes the vector rather than the snapped question.
Our current answer model is the structural rule (rated ∨ pop>τ; ≥k rated members) — it is defined
over bank entries and **cannot answer a direction that is not one**. So Paper C requires a new
answerer. That is a Paper C component, not a Paper B one, and it is where the LLM enters
(⚠ **no LLM API calls without explicit author approval** — standing rule).

**Only once Fix B exists does RL become meaningful**, because only then can the reward see the
difference between an emitted direction and its nearest bank neighbour.

## ★ FIRST EXPERIMENT — the geometry check (free, decides whether the arc has a foundation)

Before any of the above, measure what E1's existing checkpoint actually emits.

- Emit `ẑ` for every validation state E1 was evaluated on.
- Report, per state: `⟨ẑ, e_(1)⟩` vs `⟨ẑ, e_(2)⟩`, `⟨ẑ, e_(3)⟩` (nearest, 2nd, 3rd), the cosine to
  the nearest question, and the **margin ratio** `(s1 − s2) / (s1 − s10)`.
- **Prediction to falsify:** if blocker 1 is the whole story, `ẑ` sits essentially on `e_(1)`
  (cosine near 1, large margin), and the emitted point carries no off-catalogue content.
- **If instead `ẑ` sits between several candidates with small margins**, the geometry is already
  non-trivial and Fix A has a head start.

Either outcome is informative and it costs minutes. **Run this before committing to Paper C.**

## What may NOT be claimed

- Never call the snapped question "novel" — it is a bank entry by construction.
- Never report an RL gain as evidence of novelty (blocker 2: the reward cannot see it).
- A blend target is not a measured question; the blend's *value* is unmeasured until an answerer
  can be asked about it.
