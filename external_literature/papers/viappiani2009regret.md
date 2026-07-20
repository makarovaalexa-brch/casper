# viappiani2009regret — Viappiani & Boutilier 2009, Regret-based Optimal Recommendation Sets in CRS

- **Venue/year:** RecSys 2009, pp. 101-108
- **Link / DOI:** https://dl.acm.org/doi/10.1145/1639714.1639732
- **Status:** abstract-only

## Essence (3-6 lines)
Addresses recommendation under an unknown/uncertain user utility function by adopting minimax regret as the decision criterion instead of requiring a fitted preference model. Introduces setwise minimax regret (SMR) for constructing recommendation sets, gives algorithms to compute it, and proves SMR-selected sets/queries are myopically optimal.

## Method in one paragraph
The system maintains explicit constraints on the user's utility function derived from revealed preferences (accept/reject/choice actions), rather than a point-estimate or full posterior. For any candidate recommendation set, minimax regret bounds the worst-case loss relative to the (unknown) true-optimal item under the maintained constraint set. Setwise minimax regret extends this to jointly optimize an entire displayed set, and the paper shows the resulting sets/queries are myopically optimal for reducing regret at each step.

## Relevance to CASPER
- **Papers:** A — an uncertainty-carrying recommendation criterion (gate d: carries uncertainty over belief) that doesn't require a fully specified probabilistic posterior; B — regret-optimal set/query construction is a classical alternative to CASPER's info-gain/entropy elicitation criteria.
- **Taxonomy slot:** Bayesian-belief / robust-decision-theoretic CRS
- **Baseline candidate?** No — no NDCG/top-N numbers; a set-recommendation-under-uncertainty formalism, not a fold-in encoder.
- **Pre-empts / supports which of our claims:** Relevant background for framing "carries uncertainty" as a gate — minimax regret is a distinct (non-Bayesian) way to formalize belief uncertainty that CASPER should cite-and-differentiate from its Kalman/Gaussian posterior approach.

## Verdict
Cite as the classical robust-decision-theoretic alternative to Bayesian belief tracking; CASPER's posterior-based approach should be explicitly contrasted with this regret-minimization criterion.
