# koren2011ordrec — Koren & Sill 2011, "OrdRec"

- **Venue/year:** RecSys 2011
- **Link / DOI:** dl.acm.org/10.1145/2043932.2043956
- **Status:** read (abstract + summaries)

## Essence
Treats user feedback as **ordinal** (an order among rating levels) rather than numeric. Predicts a full
**personalized rating distribution** via a set of ordered thresholds on top of a latent-factor score, so the model
represents uncertainty and asymmetry across rating levels. Improves ranking-oriented measures (FCP/NDCG) over
pointwise numeric MF on Netflix/Yahoo/ML. The canonical "ordinal structure > numeric value" result.

## Method in one paragraph
An MF score per (user,item) is mapped through user-specific ordered cut-points into probabilities over rating
categories (ordered-logit style); training maximizes the ordinal likelihood, and ranking uses the expected
utility / P(rating ≥ threshold).

## Relevance to CASPER
- **Papers:** A (grade-as-ordinal precedent), C.
- **Taxonomy slot:** ordinal MF / distribution prediction.
- **Baseline candidate?** optional — reimplementations exist; mainly cited for the ordinal-beats-numeric point.
- **Pre-empts / supports:** PRE-EMPTS "ordinal treatment of the grade helps ranking." Reports FCP/RMSE, not a
  matched-information cold-start fold-in NDCG@10 delta.

## Verdict
Cite for the ordinal-value principle; not our matched measurement, and evaluated for prediction/FCP not fold-in top-N.
</content>
