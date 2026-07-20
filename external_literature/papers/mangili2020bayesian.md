# mangili2020bayesian — Mangili et al. 2020, A Bayesian Approach to Conversational Recommendation Systems

- **Venue/year:** AAAI'20 Workshop on Interactive and Conversational Rec Systems (WICRS); arXiv:2002.05063
- **Link / DOI:** https://arxiv.org/abs/2002.05063
- **Status:** abstract-only

## Essence (3-6 lines)
Bayesian CRS that maintains a probability mass function (posterior) over the item catalog, updated after every user interaction. Information-theoretic criteria decide (a) which question to ask next and (b) when to stop the conversation and recommend the current MAP item. Structural-judgement priors for the interaction-likelihood parameters are derived and shown combinable with historical interaction data.

## Method in one paragraph
Each user interaction (e.g. accept/reject a suggested item or attribute) is treated as an observation updating a discrete posterior over items via Bayes' rule; the likelihood model's parameters get elicited priors from structural judgements (rather than fit from scratch), letting the system cold-start reasonably and refine with data. Next-question and stopping decisions are driven by information-theoretic (entropy/info-gain-style) criteria over this posterior.

## Relevance to CASPER
- **Papers:** A — closely parallels CASPER's Bayesian/Kalman belief-pool posterior update (gate G1: posterior shrinks per question); B — the stopping-criterion and info-theoretic question-choice logic is directly comparable to elicitation-tree/policy work.
- **Taxonomy slot:** Bayesian-belief CRS
- **Baseline candidate?** No — item-level discrete posterior, no reported top-N/NDCG numbers found in abstract; would need full text for a quantitative baseline.
- **Pre-empts / supports which of our claims:** Supports `belief-pool-satisfies-elicitation-invariant` — an independent instance of "explicit posterior over items + monotone info-driven update" in the CRS literature, reinforcing that this design pattern is principled prior art, not novel by itself.

## Verdict
Cite as the nearest classical-CRS prior-art for CASPER's Bayesian belief pool; differentiate on CASPER's continuous/set-encoder posterior vs. this paper's discrete item-level PMF and hand-derived priors.
