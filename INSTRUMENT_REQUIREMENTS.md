# INSTRUMENT — reasoning, requirements, and gates (2026-07-11)
The fold work stalled on a deeper problem: the SCORER may be unfit for elicitation. This resets to
"what instrument do we actually need," with the pre-VAE gates (numeric AND illustrative) restored.

## 1. THE REASONING TRACE — how we got boxed in (why we can't go back OR stay)
- **Pre-VAE instrument** (biased-SVD linear ridge fold-in `u=(XᵀX+λI)⁻¹Xᵀy`, y∈{−1,+1}): GRADED POLARITY
  is NATIVE — the answer SIGN carries like/dislike, "meaning" (embedding) and "polarity" (sign y) are
  SEPARATED so like/dislike can't be conflated (G7). Purpose-built for elicitation; genre-purity,
  polarity-flip, franchise-coherence sanity all pass. Paper A is written on it.
  **BUT it fails two ways:** (a) WEAK recommender — non-competitive with EASE (captured ~4% of the prize
  on Goodreads, EASE ~23×); (b) **on it we could NOT show ANY adaptivity** — in the arena every adaptive
  selection class anchored to the static heuristic; nothing beat "ask the popular questions." So going
  back gives a weak recommender that ALSO can't demonstrate the thesis.
- **Swap to RecVAE (Instrument 2.0):** STRONG recommender (ties EASE, ~78% of the Goodreads prize).
  **BUT RecVAE is a POSITIVE-ONLY multinomial** — it ingests "which items you consumed" and structurally
  cannot express DISLIKE ("nonneg multinomial cannot express dislike", commit b67912e). Polarity/sign is
  demoted to a weak learned z-channel, and this session PROVED it comes out INERT: value-zeroing changes
  NDCG by 0.0000, flipping loved→hated doesn't move the region, confidence (know-well/rough) is unused.
  So RecVAE is strong but **unfit for the interview** — it drops the sign and the confidence, which ARE
  the elicitation signal (in a real interview everyone's consumed the popular stuff; the DISCRIMINATOR is
  how they FEEL, not what they watched).
- **We paid (real $) for LLM interviews** → a GRADED implicit signal {no_clue/rough/know_well} and a
  GRADED explicit signal {hated/meh/liked/loved}. This is the whole point; it MUST be used, not collapsed.
- **Conclusion:** we cannot go back (weak + no adaptivity) and cannot stay (deaf to dislike/confidence).
  We need an instrument that is BOTH a strong recommender AND natively hears graded, signed preference.

## 2. REQUIREMENTS (the box any new instrument must fit)
- R1 STRONG: competitive with EASE/RecVAE (not the weak linear pre-VAE).
- R2 NATIVE GRADED POLARITY: like/dislike + magnitude carried BY CONSTRUCTION — dislike DOWN-ranks the
  region; the sign-flip (G7) passes natively, not via a fragile add-on that can go inert.
- R3 USES THE GRADED LLM SIGNAL: confidence {no_clue/rough/know_well} AND value {hated/meh/liked/loved} —
  both, distinguished (not binarized).
- R4 ELICITATION-FIT: sparse cold-start (8–10 answers, many refusals), partial-reveal robust, ACCUMULATES
  (monotone in reveals), fast per-turn fold-in (real-time question scoring).
- R5 ADAPTIVITY — MUST BE SHOWN CLEARLY (author, non-negotiable). Asking a user tailored questions
  given what you already know MUST beat a fixed list, and we must demonstrate it. For that the
  instrument must be (a) NONLINEAR — a linear scorer + Gaussian belief makes optimal question-design
  provably NON-adaptive (the June theory note; the "linear-Gaussian trap"), so EASE/biased-SVD
  FORECLOSE adaptivity by construction; and (b) carry a POSTERIOR / uncertainty over the user latent
  so info-gain question selection is principled and heterogeneity across users can pay. The prior
  "adaptive prize ≈ 0" was measured only on DEAF (RecVAE, no dislike) or LINEAR (biased-SVD/EASE)
  instruments — never on a FIT + NONLINEAR one. This is the untested cell.
- R6 ATTRIBUTES/CONCEPTS: ingest the answerable concept/attribute/entity channel (Paper B), in the SAME space.
- R7 LEAK-FREE (belief invariant to unrevealed items), NO-HARM, MONOTONE.
- R8 CONTINUOUS USER LATENT (author): a d-dim continuous user representation z. Paper C's entire program
  — continuous query action space, snap-to-nearest-askable, triangulation, ask-the-gradient — requires
  it. EASE-family (item-item linear, "user rep" = raw item-space answer vector, NO latent) FAILS this.
  This, with R5's nonlinearity, is why FEASE is REJECTED as primary despite native polarity + strength.

## 3. GATES — restored from the pre-VAE paper + this session (numeric AND illustrative)
NUMERIC (pre-VAE G1–G8, each guards a real past failure):
- G1 beats-popularity; G2 floor+residual law (beats popularity-only AND personalization-only); G3 monotone
  in reveals; G4 ≥ MF/EASE (within 0.02); G5 attribute directionality ("like g" raises g's items);
  G6 attribute value (mixed ≥ item-only); G7 POLARITY (like vs dislike move a region OPPOSITELY);
  G8 out-of-envelope canary (one free TRUE answer from cold must HELP).
ILLUSTRATIVE — "look INSIDE the data" (the checks I skipped; you SEE the recommendations behave):
- IG1 GENRE PURITY: "like Sci-Fi" → the top-10 are actually Sci-Fi (pre-VAE: percentile ~0.90). Eyeball the list.
- IG2 POLARITY FLIP: flip like→dislike on a genre → its items move to the OPPOSITE end (0.90→0.10). Look at both lists.
- IG3 FRANCHISE/ITEM COHERENCE: reveal ONE liked film → the nearest recommendations cohere (same franchise/cluster). Look at the neighbours.
- IG4 GRADED VALUE (this session): sweep one item hated→meh→liked→loved → recs shift MONOTONICALLY (value not inert). Look at the shift.
- IG5 GRADED CONFIDENCE (this session): rough vs know-well on the same answer → the pull scales with confidence. Look.
- IG6 CONSUMPTION/GoT: "know-well but hated X" still pulls toward X's region (consumption dominates). Look.
THIS-SESSION QUANTITATIVE (retained): accumulation (clean ≥ native−0.03), G-no-profile-leak, G-intercept,
G-monotone-INCREASE, G-falsify-count, in-context GoT. Every illustrative check gets a numeric CI form too.

## 3b. DIRECTION AFTER REVISION (R5 nonlinear + R8 latent added)
FEASE (linear item-item, no latent) is REJECTED as primary: it reinforces the non-adaptivity trap (R5)
and kills Paper C (R8). Keep it only as a LINEAR BASELINE to prove a nonlinear instrument buys adaptivity.
The target class is CONTINUOUS-LATENT + NONLINEAR + DISLIKE-EXPRESSING + STRONG, with a POSTERIOR over z:
- Gaussian / ORDINAL rating-VAE (z + nonlinear + graded/signed output + q(z|x) posterior). Risk: Gaussian
  likelihood historically weaker than multinomial for top-N (Liang'18) — need a STRONG graded-VAE variant.
- Amortized latent models for sparse context (Neural-Process-style) — produce z + posterior from few graded
  answers by design.
- Deep / variational probabilistic MF (latent + posterior + graded, nonlinear).
- Hybrid: RecVAE's strong multinomial CONSUMPTION backbone + a graded/signed VALUE head/channel and a
  posterior — keep strength, add dislike + latent already present.
The R5 mechanism is a posterior q(z|context) + info-gain selection (reduce z-uncertainty); the region-
elicitation model (arXiv:2406.00973) is the ready-made selection layer over whichever scorer wins.

## 4. NEXT
(a) DEEP RESEARCH (revised): EXISTING architectures fitting R1–R8 — continuous-latent + nonlinear +
    dislike-expressing + strong + posterior-for-adaptivity + attributes. Find STRONG graded/ordinal VAEs,
    amortized/NP preference models, variational MF, hybrid consumption+value models. (b) Update proposal
    into SEVERAL VARIANTS. (c) Hand variants to Fable → add any missing + expected problems. (d) To author.
