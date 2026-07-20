# wang2025bdecf — 2025, Epistemic Uncertainty-aware Recommendation via Bayesian Deep Ensemble Learning (BDECF)

- **Venue/year:** arXiv 2025 (preprint)
- **Link / DOI:** arXiv:2504.10753
- **Status:** abstract-only

## Essence
Bayesian deep-ensemble collaborative filtering: places uncertainty on network weights (Bayesian NNs + ensembling) to
produce epistemic uncertainty estimates on recommendations, targeting sparse / low-activity users. Representative of
the crowded 2024–25 "uncertainty-aware recommender" wave.

## Method in one paragraph
Trains an ensemble of Bayesian neural CF models; disagreement across the ensemble + weight-posterior variance yields
per-prediction epistemic uncertainty; used to stabilize predictions for cold/low-activity users. Uncertainty is over
predictions/weights, NOT an actionable belief-covariance folded through an elicitation loop, and not tied to a frozen
full-profile SOTA tower.

## Relevance to CASPER
- **Papers:** A — evidence that "uncertainty-native recommender" alone is NOT novel in 2024–25.
- **Taxonomy slot:** (iv)-adjacent (deep-ensemble uncertainty, not conjugate belief).
- **Baseline candidate?** no — different objective (epistemic UQ), not an elicitation-compatible instrument.
- **Pre-empts / supports:** PRE-EMPTS a bare "uncertainty-aware recommendation" novelty; our claim must be the
  CONJUNCTION (frozen SOTA tower + channel-agnostic fold + monotone-per-question), not uncertainty per se.

## Verdict
Cite (with one or two peers) to concede uncertainty-aware rec is a mature area; it does not occupy the instrument intersection.
