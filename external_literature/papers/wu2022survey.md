# wu2022survey — Wu et al. 2022, A Survey on Accuracy-oriented Neural Recommendation: From Collaborative Filtering to Information-rich Recommendation

- **Venue/year:** IEEE TKDE, 2022
- **Link / DOI:** https://doi.org/10.1109/tkde.2022.3145690
- **Status:** abstract-only

## Essence (3-6 lines)
Systematic survey of neural recommender models organized around the accuracy objective, spanning from pure collaborative filtering through to models that incorporate rich side information (content, context, knowledge graphs, etc.). Provides a broad architecture taxonomy intended as a field-summarizing reference for researchers/practitioners.

## Method in one paragraph
Not an original method — a taxonomy/survey paper categorizing neural recommendation architectures (CF-based, content-augmented, context-aware, knowledge-graph-based, etc.) by how they incorporate information toward the accuracy goal, with representative models and comparative discussion per category.

## Relevance to CASPER
- **Papers:** A — background architecture taxonomy for candidate-generation recommenders; useful as a citation when situating CASPER's set-encoder/fold-in recommender within the broader neural-CF landscape and when justifying baseline choices (EASE, RecVAE, Mult-VAE already covered elsewhere in this bib).
- **Taxonomy slot:** survey / background
- **Baseline candidate?** No — survey paper, no single reproducible baseline.
- **Pre-empts / supports which of our claims:** None directly; use only as a broad-coverage citation for "neural recommender landscape," not for specific numeric claims.

## Verdict
Cite as general background/taxonomy support when introducing Paper A's recommender-architecture landscape; not load-bearing for any specific claim.
