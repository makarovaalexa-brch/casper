# weimer2007cofirank — Weimer, Karatzoglou, Le, Smola 2007, "CoFiRank"

- **Venue/year:** NIPS 2007
- **Link / DOI:** proceedings.neurips.cc/paper/2007 (pp.1593-1600)
- **Status:** read (abstract + summaries)

## Essence
First MF method to optimize a **rank-oriented metric** directly: maximum-margin matrix factorization with a
**structured-output (NDCG) surrogate** loss, plug-and-play for different ranking measures. Shows optimizing for
ranking beats optimizing regression (RMSE) MF for top-N on EachMovie/ML. The ancestor of the whole graded-listwise
CF line (ListRank, xCLiMF, SQL-Rank descend from it).

## Method in one paragraph
Structured-SVM style: a convex upper bound on (1 − NDCG) over each user's item ordering is minimized w.r.t. low-rank
user/item factors; the graded rating defines the target permutation/gains.

## Relevance to CASPER
- **Papers:** A.
- **Taxonomy slot:** structured-ranking MF (graded).
- **Baseline candidate?** weak — old struct-SVM code, heavy; hard at ML-25M. Cite as the origin, run only if feasible.
- **Pre-empts / supports:** PRE-EMPTS "optimize ranking on graded ratings instead of RMSE." No matched-input toggle,
  no frozen-tower fold-in.

## Verdict
Historical anchor of graded-ranking CF; cite for lineage, likely too heavy to replicate at scale.
</content>
