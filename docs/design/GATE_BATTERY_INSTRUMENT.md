# The Instrument Gate Battery (G0–G9) — certification spec for Paper A / pillar 1

> Synthesized 2026-07-22 from the adversarial gate-sufficiency review + the full validity-check
> archaeology (all notes incl. archives). Supersedes the bare G0/G1/G2 wherever "gates" are cited.
> Rationale: G0/G1/G2 are OUTCOME gates; nine concrete constructions pass all three with broken
> elicitation (sign-blind fold, circular answer model, decorative Σ, popularity parasite, taste-leaking
> selection, …). The battery below adds the SEMANTIC and ARTIFACT-CONTROL gates that close them.
> Every gate is one-line testable; each carries its historical receipt (the disaster it would have caught).

**Design pre-condition (resolve before Step 2 lands):** the chapter and DESIGN_PRECACC currently describe
TWO different instruments — (i) rank on the conjugate POSTERIOR mean vs (ii) rank on the frozen ENCODER
fold with Σ separate. The battery applies to whichever is chosen, but G0's form and G8's teeth differ:
under (i) G0 is a CI-tie, under (ii) G0 is bit-identity + Σ must be load-bearing via G8 or R4 is dropped.

## Per-build gates (every build must pass; all cheap)

- **G0 — strength & identity.** Full-profile full+tail NDCG@10 reproduces the frozen tower (bit-identical
  under design (ii); paired-bootstrap tie under (i)). Includes standing asserts: score with the decoder's
  learned bias never popb *(popb disaster)*; empty evidence set → exactly the prior *(intercept bug)*;
  trunk-drift canary = 0 *(AdamW un-freeze)*; canonical-snap of all anchor arms before any delta is read
  *(corrupted-cache disaster)*.
- **G1 — usable posterior.** tr Σ shrinks IN EXPECTATION over confident answers (refusals explicitly
  EXEMPT — a correct model does NOT shrink on a refusal); directional: folding c shrinks variance along
  d_c ≥2× other directions; calibrated: Spearman(belief-σ, held-out NLL) ≥ 0.25; pre-fit sign proof
  dL/dα_c < 0 at α=0 *(decoupling trap)*.
- **G2 — cold curve rises.** Per-question full+tail NDCG@10 vs random-question control: success = curve/
  endpoint at budget (AUC), NOT every-step significance; must show INCREASE (turn8−turn1 ≥ MDE, CI>0),
  not merely no-drop *(flat-model artifact)*; tail restricted to users with ≥k tail targets; plus the
  out-of-envelope canary: one free TRUE answer from cold must help on EVERY channel *(algebraic-operator
  ambush)*.
- **G3 — sign & intensity fidelity.** (a) FLIP test: negate every answer's sign → ΔNDCG < 0, CI excl. 0,
  per channel; (b) VALUE-NONINERT: neutralizing values to "meh" (membership frozen) drops NDCG ≥ MDE
  *(the 7-fold value-inertness wall)*; (c) monotone-in-intensity: Spearman(level, Δ) > 0 and
  binarize-ablation must LOSE *(graded-collapse)*.
- **G4 — confidence & refusal.** Fitted precision ordered: α(refuse) < α(vague) < α(know-well); collapsing
  confidence levels must LOSE NDCG; a refusal burns the turn but moves belief ≈ 0 *(τ mis-ordering; the
  additive-null trap)*.
- **G5 — concept/channel specificity.** Folding concept c ranks member items above matched non-members
  (member-lift AUC > 0.8) AND beats folding c's own popularity-projection (top-PC partialled out); honest
  bar on tail = popularity-WITHIN-the-filter; dedicated channel ≥ trained member-bag injection
  *(raw-centroid popularity collapse; member-bag catastrophe)*.
- **G6 — existential controls (HARD RULE 5, now discharged for the fold itself).** Wrong-user answers,
  shuffled answers, and placebo-constant answers through the identical pipeline must gain ≈ nothing
  (≪ true-answer arm); duplicate answers ×2/×3 change nothing (information, not cardinality)
  *(cardinality confound; popularity parasite)*.
- **G8 — Σ load-bearing + set coherence.** Replace Σ → isotropic cI: the acceptance-deciding number must
  DEGRADE (else the R4 claim is dropped); permuting answer order leaves belief/NDCG within ε; conflicting
  re-asks resolve order-invariantly *(decorative-Σ; order-dependence)*.
- **G9 — answer-ingestion isolation.** On a FIXED user-independent question set, NDCG still rises with t
  from answer content alone *(separates instrument from policy — the E0/G2 taste-peek lesson: beats-random
  certifies selection, the instrument paper must certify ingestion)*.

## One-time certifications (expensive; per instrument, not per build)

- **C1 — published-number bridge.** Implementations snap to published numbers on the canonical split
  (DONE: EASE +0.0003, RecVAE +0.0005 ML-20M; Mult-VAE/DAE pending).
- **C2 / G7 — answer-model transfer (non-circularity).** The G2 gain survives under a held-out BEHAVIORAL
  answer model (the in-repo SEL watch-signal channel) + injected label noise, at ≥ a stated fraction of the
  self-model gain, CI excl. 0. THE single most important addition: without it, a G0–G2 pass is consistent
  with an instrument inverting its own simulator. Includes the firewall assert: recommender-geometry
  answers (cos(z*,q)) appear NOWHERE in train or eval *(circular-measurement flaw)*.
  **C2 leak clause (2026-07-22):** the SEL answer for a user MUST be computed with the user's held-out
  target items EXCLUDED from the watch counts — an answer derived from the targets is a leak dressed as a
  certification. **C2 scope clause:** SEL certifies *not-geometrically-circular*, NOT *human-realistic* —
  behavioral and stated answers agree only ~21% (Jul-17 finding); SEL is likely optimistic vs a real
  interview. Human realism is C3's job and only C3's.
- **C3 — human round-trip (VISION constraint; still owed project-wide).** ≥1 study: rendered questions →
  graded human answers → downstream NDCG.

## Standing protocol rules (not gates; from CLAUDE.md + archaeology, enforced on every experiment)
Paired per-user bootstrap for every bolded delta · ≥3 REAL training seeds before any "learned X wins"
claim · comparator symmetry (static arm beyond reproach; same selection budget both arms) · privilege
ladder: every oracle ceiling paired with a realizable probe · credit-neutral masking of asked items ·
known∩held=∅ + held-out targets un-askable + leak-free token asserts · no data truncation · commit code
before run.

## Coverage map (attack → killer)
sign-blind→G3a · intensity-blind→G3c · confidence-flat→G4 · non-specific concept→G5 · popularity
parasite→G6 · circular answer model→C2 · decorative Σ→G8 · taste-leak selection→G9 · order-dependence→G8 ·
strict-shrinkage false-negative→G1 exemption · noisy-answer false-negative→G2 AUC form · thin-tail
false-negative→G2 tail-target floor.
