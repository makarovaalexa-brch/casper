# zhao2013interactive — Zhao, Zhang & Wang 2013, Interactive Collaborative Filtering

- **Venue/year:** CIKM 2013
- **Link / DOI:** https://wnzhang.net/papers/icf-cikm.pdf
- **Status:** abstract-only

## Essence (3-6 lines)
Studies collaborative filtering in an explicitly interactive setting: the system continuously recommends items to a user and receives interactive feedback, with predictions continuously refined using up-to-date feedback (a bandit-style, online-learning reformulation of CF rather than a one-shot batch fit).

## Method in one paragraph
CF is cast as a sequential decision process (multi-armed-bandit flavored): at each round the system recommends, observes feedback, and updates its latent factor estimates online, balancing exploration (learning the user's taste) against exploitation (recommending well) — the interactive analogue of the standard offline matrix-factorization fit.

## Relevance to CASPER
- **Papers:** A — early prior-art for online/incremental fold-in of user evidence into a CF model, structurally the predecessor problem to CASPER's cold-start fold-in encoder; B — the explore/exploit framing is relevant background for question/item-selection during elicitation.
- **Taxonomy slot:** bandit-CRS / online-CF
- **Baseline candidate?** No — pre-deep-learning matrix-factorization bandit method; not directly comparable architecture, but a citable ancestor for "interactive CF."
- **Pre-empts / supports which of our claims:** Establishes that "recommender that updates online from interactive feedback" is a long-standing CF sub-area CASPER's fold-in encoder must be positioned against (differentiator: CASPER folds in evidence via a trained encoder in one shot per question rather than per-round bandit updates).

## Verdict
Cite as the classical online/interactive-CF ancestor of CASPER's fold-in cold-start encoder; differentiate on architecture (trained set-encoder vs. bandit-updated MF).
