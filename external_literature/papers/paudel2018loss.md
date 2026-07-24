# paudel2018loss — Paudel, Luck, Bernstein 2018, "Loss Aversion in Recommender Systems"

- **Venue/year:** arXiv 2018 (1812.11422); ICDM workshop line
- **Link / DOI:** arXiv:1812.11422
- **Status:** skimmed (abstract)

## Essence
Adds an explicit **negative-preference** term to a recommender so disliked items are actively pushed down the list.
On two public datasets, improves accuracy AND reduces the number of negative items at the top — the "loss aversion"
framing (a negative at the top costs more than a positive gains). Part of the negative-implicit-feedback line
(with Frolov 2016 and the 2021–2025 sequential-negative work).

## Method in one paragraph
Augments an embedding/graph recommender with a signal derived from low ratings / disliked interactions, jointly
optimizing to rank positives up and negatives down.

## Relevance to CASPER
- **Papers:** A (sign/negative-value leg), C.
- **Taxonomy slot:** negative-preference top-N.
- **Baseline candidate?** optional — supporting cite for "sign carries signal," not a primary baseline.
- **Pre-empts / supports:** PRE-EMPTS "modeling disliked items improves top-N." Not a matched-information grade toggle.

## Verdict
Secondary cite reinforcing that using negatives/sign is established; Frolov is the stronger precedent.
</content>
