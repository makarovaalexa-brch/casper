# shi2013xclimf — Shi et al. 2013, "xCLiMF"

- **Venue/year:** RecSys 2013
- **Link / DOI:** dblp conf/recsys/ShiKBLH13 ; PDF alexiskz.wordpress
- **Status:** read (abstract + method)

## Essence
Listwise learning-to-rank CF that optimizes **Expected Reciprocal Rank (ERR)**, a generalization of reciprocal
rank to **multiple relevance levels** (graded ratings). Generalizes CLiMF (which used binary implicit feedback,
optimizing MRR). Headline: **xCLiMF beats CLiMF specifically when >2 relevance levels exist in the data** — the
single cleanest published "grades beat binary within one model family" statement. Data: TED/ML, ERR/NDCG.

## Method in one paragraph
Maximize a smoothed lower bound on ERR over each user's list; the graded rating enters the ERR gain/stop
probabilities, so higher-rated items get more listwise weight than a binary relevant/irrelevant split allows.
Frobenius regularization on latent factors; scales with observed ratings.

## Relevance to CASPER
- **Papers:** A, C.
- **Taxonomy slot:** graded listwise LTR-CF.
- **Baseline candidate?** yes — public py/spark; the nearest "grade on/off within a family" baseline for §5B.
- **Pre-empts / supports:** the closest prior art to our matched claim; PRE-EMPTS "grade beats binary within a
  model family." Differs: it swaps the *loss* (ERR vs RR), not just the input, and reports ERR, not a frozen-tower
  fold-in NDCG@10 delta at matched information.

## Verdict
The paper to cite as "most similar prior art" and to run as-published in the baseline table; our wedge is
matched-input toggle on a frozen SOTA tower vs their objective swap.
</content>
