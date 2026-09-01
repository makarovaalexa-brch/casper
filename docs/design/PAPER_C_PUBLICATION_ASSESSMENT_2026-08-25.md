# Paper C — what is proved, what is outstanding, the cheap path to a publishable paper

## ★★★ AUTHOR STEER 2026-08-30 — BINDING, overrides §C item 1 ("sell the audit")

**AUTHOR SAID (verbatim):** *"i just wanted sexy llm piece to top off my phd thesis, this one is
no load bearing. and it is a bit more risque, wanna do interesting stuff and less so lengthy field
audits."*

Consequences, replacing the §C-1 reframe:
- **The spine is the DEMO, not the audit.** Lead with: a policy emits a direction the catalogue
  does not contain, an adapter says it in English, and the word lands in the direction's own
  semantic neighbourhood. The 12-system scrambled-control audit shrinks to ONE related-work
  paragraph + one control we actually ran; it is no longer the contribution.
- **Target length/venue shifts DOWN**: a 4–6 page late-breaking/short/workshop piece + the thesis
  capstone chapter, not a full audit paper. Publication remains a bonus (`PAPER_C_TODO.md`).
- **Priority order changes**: (1) decoding diversity — it is what makes the demo table good;
  (2) the deployment-trajectory transcript figure (a whole cold-start interview whose every
  question is synthesised) — the capstone artifact; (3) write short. The bank-question arm, the
  second seed, and the emission re-render become optional colour, not requirements.
- What does NOT change: no absolute-value claim (the circularity ruling), no human study, the
  three-gap novelty wording, and every number keeps the strength its evidence licenses.
**Fable, 2026-08-25, at the author's request. Companion to `PAPER_C_DESIGN.md` (which stays the
design of record); this is the publication-strategy read of the banked evidence.**

## A. What is PROVED, at what strength

1. **The mechanism (strongest, large-n): the vector conditions the text.** C3B, 3,670 held-out
   item questions, independently-fitted tower bridge, pre-registered: A−A′ = +0.111 [+0.105,
   +0.117]; A at 107% of the true-phrase ceiling; identification 0.678 [CI excl. chance] vs the
   ceiling-relative floor 0.597; training curves separate by epoch 2. One criterion (a
   directionless equivalence) failed benignly and the registered FAILURE label is kept — with the
   directional rewrite pending the author's ruling, this is a clean, controlled, CI'd result.
2. **The loop closes (pilot-n): point → English → fold.** C4 pilot: renders of real points align
   with their own measured-ingredient reference at +0.286 [+0.178, +0.398] over random-point
   renders, prior at zero (concept channel). Triangulation: where the render is unique (18/42),
   it lands at median rank 91/1,031 in the point's own neighbourhood — `action` rank 1,
   `indictment` 5, `animals` 7, `science` 31.
3. **★ The motivating sentence, measured:** the nearest catalogue question (cos 0.93 in the head
   space) carries text ANTI-aligned (−0.158) with the point's direction while the rendered
   off-catalogue word carries it (+0.204, CI excl. 0) — *the direction is expressible and no
   near-catalogue text expresses it*. Two caveats travel (common-mode nearest; concept-favouring
   bridge).
4. **The price of language, relative and declared:** C4B under the author-ruled declared-circular
   simulator: translation costs +0.018 full / +0.007 tail [CIs excl. 0], decomposing 4.2× onto the
   mode-collapsed renders (+0.006 unique vs +0.026 stock; 5 states where English beats the
   direction). **[CORRECTED 2026-09-01 — the original clause "⇒ the noise is decoding diversity,
   not verbalisation" was RETRACTED by C4C and this doc was not updated.]** C4C removed the collapse
   by decoding alone and cut the cost 2–3.5×, but the sampled renders are LESS faithful, so what
   sampling removes is **harm from confidently committing to the wrong concept** (greedy's stock
   words fold BELOW the no-fold baseline), not confusion. The loss is removable by decoding; the
   mechanism is not "diversity".
5. **The audit (the durable methodological half):** no embedding-verbalisation system reports a
   correspondence-scrambled control (PDF-verified narrow wording); ELM self-grades and its
   faithfulness metric rises off-manifold; we document THREE ways naive controls fail (variant-(a)
   structurally unconstructible; directionless equivalence fires on a benign direction; the twin's
   degenerate renders reverse a per-point comparison) and contribute the **ceiling-relative
   evaluation recipe** (calibrate every gate to what the TRUE phrase earns through the same
   instrument, before training). Machine-only synonyms observed in our space (exact recovery
   0.0000 with above-gold re-encode) — our own evidence for why plausibility panels mislead, which
   licenses skipping the field's rater-panel ritual on principle rather than thrift.
6. Foundations: C0's corrected geometry (emission = common-mode + genuine bank-aligned direction;
   nothing on the grid, max cos₁ 0.120); C1's capability-to-fit with real-init stability and the
   paired sign test; the C3 PoC's structural disqualification of retrieval-anchored rendering.

## B. What is OUTSTANDING (what a reviewer names)

1. **Leg (i) — "a policy CHOOSES the point at a real interview state" — is the thin leg.** The
   pilot policy collapses off its 42 states; its raw emissions were unrenderable (38× scale
   mismatch, fix registered); the 42 states are nested and structurally selected. The full repair
   is C1-final on the retrained tree with persisted gains — the one genuinely expensive item.
2. **Renderer diversity**: distilgpt2 + greedy collapses to stock words on 24/42 states; every
   weak number traces to this.
3. **The item channel** is an instrument limit (concept-routed bridge; prior floor above signal).
4. Single training seed per adapter arm (our own checklist forbids "robust" language until ≥2);
   chapter C3/C4 sections and Introduction unwritten; C2's directional rewrite needs the author's
   one-line ruling.

## C. The cheap path to compelling (ranked; costs are box-hours/token-days, not weeks)

1. **Reframe leg (i) down — the zero-cost decision that unblocks publication.** Sell the paper as
   the RENDERING + AUDIT paper: *synthesised interview directions can be said in English,
   faithfully, auditably — and here is what saying them costs*. The policy leg becomes measured
   motivation (C0's between-ness, C1's capability) rather than a claim. C1-final then upgrades a
   future version instead of blocking this one.
2. **Diversity-fixed rendering (≈1 day, no retraining):** sample k renders per point at
   temperature with a pre-registered pick rule; re-run triangulation + C4B's T arm. If the stock-
   word cost collapses toward +0.006, the "noise is decoding diversity" claim gets its
   confirmation and the flagship table gets better words. Highest value-per-hour on the board.
3. **The bank-question arm in the C4B harness (≈1 day):** add arm B = the tree's own chosen
   question at each state, same simulator. Completes the triangle direction / its English / the
   catalogue's best at the very states the search flagged as near-ties — whichever way it lands,
   it is the head-to-head the reviewer will ask for, relative and declared.
4. **Second adapter seed pair (overnight box time, zero tokens)** — satisfies our own N-seed rule
   before any "robust" sentence. + **the emission rescale re-render (half a day)** to un-retract
   the emission cells properly.
5. **Writing (token-days, no compute):** C3B/C4/C4B chapter sections from the banked docs; the
   Introduction around the mechanism + the catalogue-anti-alignment; the audit section; the
   flagship figure (branch → word table + the loop diagram). Most sentences already exist in the
   results docs at the licensed strength.

**Explicitly NOT worth the time for v1:** a bigger LM (GPU/cloud), any human study (our
machine-only-synonym evidence is the argued-for reason to skip the rater panel), the full
item-channel bridge fix, Info-Synth replication, the 300 users (stay saved).

## D. The claim, as the evidence now supports it

*A tiny adapter proves that a recommender's interview directions condition generated language
(controlled, ceiling-relative, CI'd); the rendered words triangulate to the direction's own
neighbourhood while the nearest catalogue text does not; translating a direction into English and
reading it back costs ~0.018 NDCG under a declared-circular simulator shared by both arms, and the
cost is decoding diversity, not language; and the field's existing validation practice would have
caught none of the three control failures we document.* Capability + audit; worth inherited;
absolute value never claimed. The three-gap novelty (collaborative latent · language · unsnapped)
stands against the PDF-verified literature.

**Total incremental cost to a submittable draft: roughly 3–4 focused days of token work + 2–3
box-nights, PLUS the author's pass over the claims.** Venue thought (author's call): ECIR full or
RecSys short/LBR fit the capability+audit register; the old roster had ECIR 2027 (deadline
2026-10-02) — comfortably reachable on this plan.
