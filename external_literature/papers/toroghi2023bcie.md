# toroghi2023bcie — Toroghi & Sanner 2023, Bayesian Critique-Improve-Explain (BCIE)

- **Venue/year:** SIGIR 2023
- **Link / DOI:** (SIGIR 2023 short/full; verify DOI before citing)
- **Status:** abstract-only (from elicitation findings; not re-fetched this pass)

## Essence
Conjugate-Gaussian Bayesian fold of user critiques into a FROZEN factorization (matrix-factorization) space,
producing updated recommendations plus explanations. A precedent for exact-conjugacy belief updates over a fixed,
pre-trained item embedding space — closest published analog to our R4/R5 mechanism.

## Method in one paragraph
Keeps the item factors fixed; represents the user as a Gaussian belief; each critique is a linear observation giving
a closed-form conjugate-Gaussian posterior update over the user vector, from which re-ranking and explanations follow.
Critique-based (not full-profile SOTA), fixed schema of critiques.

## Relevance to CASPER
- **Papers:** A (R4/R5 mechanism precedent), C (critique/elicitation).
- **Taxonomy slot:** (iv) Bayesian / Gaussian belief over a frozen item space.
- **Baseline candidate?** optional (Tier 3) — critique-based metrics, not the full-profile bar.
- **Pre-empts / supports:** PRE-EMPTS "conjugate-Gaussian fold over a frozen factorization"; our wedge is doing it
  over a certified RecVAE-CLASS tower with channel-agnostic (not just critique) tokens + monotone-per-question demo.

## Verdict
Cite as the exact-conjugacy-over-frozen-space precedent; differentiate on the SOTA tower + channel-agnostic input.
