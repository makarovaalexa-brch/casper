# shen2021gfcf — Shen et al. 2021, How Powerful is Graph Convolution for Recommendation? (GF-CF)

- **Venue/year:** CIKM 2021
- **Link / DOI:** arXiv:2108.07567
- **Status:** abstract-only

## Essence
Reframes GCN-based CF as graph signal processing and derives a CLOSED-FORM linear graph filter (GF-CF) that, with
no training, matches or beats trainable GNNs (LightGCN etc.) on Gowalla/Yelp/Amazon-book. The strong-baseline that
launched the training-free graph-filter line continued by BSPM and Turbo-CF.

## Method in one paragraph
Interprets neighborhood aggregation as low-pass filtering on the user-item bipartite graph; combines a linear
filter over the normalized adjacency with an ideal low-pass component. No learned parameters; a new user's history
folds in as a new graph signal. Item-only, point estimate.

## Relevance to CASPER
- **Papers:** A — root of the closed-form graph-filter SOTA family; strong-baseline warning (`dacrema2019progress` spirit).
- **Taxonomy slot:** (ii) fold-in / closed-form graph filter.
- **Baseline candidate?** conditional — no ML-20M strong-gen NDCG published; Tier 2 only if re-run on our split.
- **Pre-empts / supports:** supports "shallow/closed-form frontier"; item-only ⇒ fails R3/R4.

## Verdict
Cite as the closed-form graph-filter progenitor; another R1-corner, item-only point estimator.
