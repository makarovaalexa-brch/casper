# vendrov2020gradient — Vendrov et al. 2020, Gradient-based Optimization for Bayesian Preference Elicitation

- **Venue/year:** AAAI 2020, 34(06), 10292-10301; arXiv:1911.09153
- **Link / DOI:** https://arxiv.org/abs/1911.09153
- **Status:** abstract-only

## Essence (3-6 lines)
Tackles the computational cost of expected-value-of-information (EVOI) query selection in Bayesian preference elicitation, which is normally prohibitive for large item spaces because it requires enumerating candidate items/queries. Reformulates EVOI as a continuous, differentiable objective and optimizes it with a novel scalable Monte Carlo + gradient-based method, implementable in standard ML frameworks (TensorFlow/PyTorch).

## Method in one paragraph
Rather than exhaustively scoring candidate queries by their expected reduction in posterior uncertainty (classical EVOI), the authors parameterize the query itself as continuous and differentiable, then backpropagate a Monte-Carlo-estimated EVOI objective through it — turning a combinatorial search into gradient-based optimization. This scales EVOI-driven elicitation to item spaces too large for explicit per-item enumeration.

## Relevance to CASPER
- **Papers:** A — a differentiable EVOI framing is directly analogous to any gradient-trainable component of CASPER's belief-pool/posterior update; B — this is a strong prior-art anchor for "principled Bayesian query selection at scale," a benchmark CASPER's static-entropy /learned-policy approaches should be measured against.
- **Taxonomy slot:** Bayesian-belief CRS, scalable EVOI
- **Baseline candidate?** Possibly — if reimplementable, the differentiable-EVOI machinery could be a baseline elicitation-policy comparator for CASPER's fold-in/belief-pool question selection at scale. Would need full text for item-space sizes and reported metrics.
- **Pre-empts / supports which of our claims:** Relevant to CASPER's repeated finding that policy learning underperforms static entropy — this paper is evidence that a purpose-built continuous-EVOI optimizer (not generic RL) is the "right" tool class for scalable Bayesian query selection, worth citing when discussing why naive RL policies lose.

## Verdict
Cite as the load-bearing scalable-Bayesian-elicitation prior art; the nearest "EVOI done right" reference against which CASPER's belief-pool question-selection mechanism should be positioned.
