# choi2023bspm — Choi, Hong, Park & Cho 2023, Blurring-Sharpening Process Models for CF (BSPM)

- **Venue/year:** SIGIR 2023
- **Link / DOI:** arXiv:2211.09324 ; 10.1145/3539618.3591645
- **Status:** abstract-only (+ repo/abstract)

## Essence
Training-free CF via a blurring–sharpening ODE/SDE process inspired by score-based generative models: perturb
(blur) the interaction signal then recover (sharpen) it, discovering new item affinities. SOTA Recall/NDCG on
Gowalla, Yelp2018, Amazon-book with runtime comparable to fast baselines. **ML-20M NDCG not reported.**

## Method in one paragraph
Treats the user-item interaction matrix as a graph signal; runs a continuous blurring process (low-pass diffusion)
then a sharpening process (recovery) integrated with an ODE solver, no learned parameters beyond filter design.
Item-only, closed-form-ish, point estimate; a new user's history is just a new signal to filter (interactive-friendly).

## Relevance to CASPER
- **Papers:** A — 2023 graph-filtering accuracy/speed leader on the LightGCN benchmark family.
- **Taxonomy slot:** (ii) fold-in / closed-form graph filter.
- **Baseline candidate?** conditional — SOTA on Gowalla/Yelp/Amazon; NO ML-20M/25M number published, so admit to
  Tier 2 only if re-run on our split. Item-only, no uncertainty, no concept channel.
- **Pre-empts / supports:** supports the "R1-corner ≠ R3/R4" bifurcation; supports flat-frontier caveat (its wins
  are on a different dataset family than the ML-20M VAE split).

## Verdict
Cite as the current fast graph-filter SOTA; not directly comparable to us until re-run on the ML-20M/25M split.
