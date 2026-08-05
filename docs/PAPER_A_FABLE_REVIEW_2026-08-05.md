# Paper A — Fable balanced review, 2026-08-05

Author's request: a *balanced* review — venue recommendation and real improvements, **not** a nit-hunt.
Reviewed from a self-contained brief (HARD RULE 7; Fable read no repo files). The brief caveated the
three pending arm-N cells.

---

## 1. Venue

**TORS regular. ~65% accept-after-major-revision.** The shape — 35 pages, certification chain, two
protocols, verification battery, seven stated limitations — is a TORS paper. Its reviewer pool is the one
most likely to *reward* the bridge and the paired-bootstrap discipline rather than penalise "no new model".
The 1.2% shortfall, honestly stated, will not sink it there; at SIGIR main it is the first line of Review 2.

- **Do NOT use the RecSys reproducibility track.** The bridge is a component, not the contribution.
  Submitting there "would ratify the weakest possible reading — they re-ran Liang's split carefully."
- **ECIR ~50%**, but the cut to ~14 pages removes exactly the verification material that differentiates us.
- **Next tier up is RecSys main, not SIGIR** — ~40–50% *if* we re-lead with the inversion and close the
  weak-consumer hole; ~25% with the current framing. SIGIR main is the wrong venue: the contribution is
  systemic to RecSys, not to ranking.

## 2. The three highest-value improvements

**2.1 Lead with the payoff table.** (writing only; buys a category change) The current spine reads "we
built a careful thing". The inversion is what makes the instrument *necessary*. Put a compressed version of
that table on p1–2; everything else becomes the answer to "why should I believe this table", which is the
strongest possible role for the verification material.

**2.2 Add a strong-but-crude consumer to the payoff table.** (low cost; kills the best remaining objection)
The inversion currently rests on ours + Golbandi + three weak models, so a reviewer can ask whether it is
just "weak recommenders scramble everything". A frontier recommender with a crude interface settles it
either way: if it also disagrees with our tower, the claim is genuinely about the *pair*; if it agrees, the
claim narrows to "recommender quality, not interface, drives the ordering" — which we need to know before a
reviewer does. **Not running it leaves the central claim resting partly on strawmen.**

> **We may already have most of this.** The 2026-07-30 interview table included `recvae_likes` — a frontier
> recommender restricted to a likes-only interface — at k=8: HELF .1413 / entropy0 .1777 / popularity .1817,
> i.e. **P > E0 > H, a third distinct ordering**. That run was on the old tower and the floored protocol, so
> it needs re-running under ep31 / no-floor, which is cheap because RecVAE does not depend on our checkpoint.

**2.3 Promote the anti-circularity firewall to a first-class contribution.** (writing + one figure) It is
the strongest defence in this literature and it is under-sold. Named subsection *in the evaluation setup*,
not in limitations; an information-flow diagram showing where the firewall sits; a short checklist applied
to PEBOL / ConTS / unified-CRS showing each violating it. "That turns *we avoid circularity* into *we define
the test and prior work fails it*."

## 3. Strongest expected objection

**"Your users are logs, not people"**, in two halves.
- The **circularity** half our material fully answers — say it early, loudly, once (see 2.3).
- The **ecological validity** half we cannot answer without a user study. The correct move is *affirmative
  scoping*: this chapter measures what a recommender can extract from truthful, behaviour-grounded answers;
  human answer noise belongs to the elicitation layer, i.e. the companion chapter. It currently sits in the
  limitations list where it reads as a confession — **move the scoping sentence to the introduction where it
  reads as design.**

Second-strongest: *"the requirements are self-serving — you defined R1–R4 then built the only thing that
satisfies them."* Defence is the payoff table (the requirements are **consequential**: violating them changes
scientific conclusions) — one more reason it must lead.

## 4. Overclaims (author would rather cut than defend)

| claim | verdict |
|---|---|
| "architecturally **inexpressible** in the item-indicator basis" | **Too strong for 2 of 3 legs.** A concept has an obvious reduction — a weighted indicator over member items, foldable into EASE/RecVAE. It may fail, but we have not *shown* it failing, so a reviewer constructs it in the margin. Either run the reduction and show it loses, or rewrite. The **name-only held-out fold genuinely is inexpressible** — say that precisely. |
| "a measuring instrument for the **whole** elicitation-policy literature" | Falsified by our own tree-designer disclaimer. **Drop "whole".** |
| tail → "**parity**" with RecVAE | A CI containing zero is *failure to separate*, not demonstrated equivalence — and this chapter has been burned on tie language before. Write "statistically indistinguishable (Δ=−0.0006, CI [−0.0020,+0.0008])" everywhere "parity" appears. |

Rest is clean. "Concepts beat items" is properly conditioned on the bank ranking; the bifurcation claim is
survey-bounded. **The R4 decomposition (intensity at fixed support vs sign varying what is folded) is called
"exemplary" and worth imitating elsewhere in the paper.**

## 5. Best contribution

**Agrees: the strategy inversion — but sell it as a PINCER with the certification bridge.** The inversion
alone is an existence proof. What makes it the headline is that *the same codebase that reproduces published
numbers to three decimals also reproduces Rashid's ordering under Rashid-era consumers and inverts it under a
frontier one* — which closes the usual escape route, "your reimplementation is broken."

The answerability/concept-decoupling result (4.8% vs 97.2%) is the most *memorable* mechanism and the best
figure, but it is bank-conditional and instrument-specific: second act, not headline. The two-protocol
framing is excellent hygiene reviewers will trust us for, but it is infrastructure, not a contribution.

## On the pending cells
**None can change a conclusion.** RBMF/EDDI sit in the NATIVE column of weak baselines and would need ~+0.09
absolute to threaten any ordering. Only nonzero risk: TaNP's graded meta-training landing dramatically higher
and softening "ratings-native models don't benefit" — if so, report it and adjust one sentence.

## Bottom line (verbatim)
> "This is an accept-shaped paper at TORS as it stands, and a plausible RecSys-main paper if you re-lead with
> the inversion and add one strong-crude consumer. The calibration warning cuts one way here: the previous
> 'weak reject' pattern was a reviewer over-weighting simulator circularity, and this paper's firewall is the
> strongest answer to that objection in the current literature — its main defect is that it's presented as an
> apology instead of a contribution."
