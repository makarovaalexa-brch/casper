# volkovs2015effective — Volkovs & Yu 2015, "Effective Latent Models for Binary Feedback"

- **Venue/year:** SIGIR 2015
- **Link / DOI:** cs.toronto.edu/~mvolkovs/sigir2015_svd.pdf ; dl.acm.org/10.1145/2766462.2767716
- **Status:** skimmed (abstract)

## Essence
Argues latent models designed for **explicit ratings** perform poorly on **binary implicit** data and must be
redesigned; uses neighborhood-similarity information to guide latent factorization for accurate implicit-data
representations. A pillar of the "explicit and implicit are different regimes" framing that motivated binarizing.

## Method in one paragraph
Incorporates item-item neighborhood similarity as a guide/regularizer into an SVD-style latent factorization tuned
for 0/1 implicit feedback rather than graded ratings.

## Relevance to CASPER
- **Papers:** A (history of the binarization consensus).
- **Taxonomy slot:** implicit-tuned latent CF.
- **Baseline candidate?** no.
- **Pre-empts / supports:** SUPPORTS the §1 history — evidence the field treated explicit and binary as separate
  regimes without a matched graded-vs-binary top-N test.

## Verdict
History cite for why binary became default; not a baseline, not a pre-emption of our measurement.
</content>
