# park2024turbocf — Park et al. 2024, Turbo-CF: Matrix Decomposition-Free Graph Filtering

- **Venue/year:** SIGIR 2024
- **Link / DOI:** arXiv:2404.14243 ; 10.1145/3626772.3657916
- **Status:** abstract-only

## Essence
Training-free CF using a polynomial graph filter that avoids expensive matrix decomposition (no SVD/eigen), so it
runs on GPU in ~seconds while matching BSPM-class accuracy on Gowalla/Yelp2018/Amazon-book. **ML-20M not reported.**

## Method in one paragraph
Builds a normalized item-item adjacency and applies a hand-designed polynomial (low-pass) graph filter directly to
the interaction signal — no decomposition, fully GPU-parallel. Item-only, point estimate, instant fold-in of a new
user's history. Positioned as the speed-optimal member of the closed-form graph-filter family (GF-CF → BSPM → Turbo-CF).

## Relevance to CASPER
- **Papers:** A — the 2024 speed leader among closed-form graph filters.
- **Taxonomy slot:** (ii) fold-in / closed-form graph filter.
- **Baseline candidate?** conditional — same as BSPM: no ML-20M/25M number, admit to Tier 2 only if re-run on our split.
- **Pre-empts / supports:** reinforces that the graph-filter frontier is item-only, uncertainty-free — fails R3/R4/R5.

## Verdict
Cite alongside GF-CF/BSPM as the training-free graph-filter frontier; not a channel-agnostic or uncertainty-native rival.
