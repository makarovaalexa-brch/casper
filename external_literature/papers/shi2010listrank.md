# shi2010listrank — Shi, Larson, Hanjalic 2010, "ListRank-MF"

- **Venue/year:** RecSys 2010
- **Link / DOI:** dl.acm.org/10.1145/1864708.1864764
- **Status:** read (abstract + repo)

## Essence
Listwise learning-to-rank + MF: minimizes cross-entropy between the model's **top-1 probability** distribution and
the ground-truth (graded) one over each user's items. Low complexity — **linear in the number of observed ratings**.
Beats item-based CF and CoFiRank on NDCG (ML/Netflix/EachMovie). The classic scalable listwise graded ranker
between CoFiRank (2007) and xCLiMF/SQL-Rank.

## Method in one paragraph
For each user, softmax-normalize predicted scores and observed ratings into top-1 probabilities; minimize their
cross-entropy w.r.t. latent factors — a listwise objective where the graded rating shapes the target distribution.

## Relevance to CASPER
- **Papers:** A.
- **Taxonomy slot:** listwise (top-1 cross-entropy) MF.
- **Baseline candidate?** optional — public Java/py; scalable; a clean classic listwise anchor for §5B if a second
  graded baseline beyond SQL-Rank/xCLiMF is wanted.
- **Pre-empts / supports:** PRE-EMPTS "listwise graded objective for top-N." No matched-input frozen-tower toggle.

## Verdict
Cite for lineage; optional replication as a lightweight listwise baseline.
</content>
