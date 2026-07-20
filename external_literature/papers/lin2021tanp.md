# lin2021tanp — Lin et al. 2021, Task-adaptive Neural Process for User Cold-Start Recommendation (TaNP)

- **Venue/year:** WWW 2021
- **Link / DOI:** arXiv:2103.06137
- **Status:** skimmed

## Essence
Casts per-user cold-start recommendation as a neural process: each user is a task / stochastic process; a
permutation-invariant encoder maps the user's observed (item, rating) support set to a predictive distribution over
preferences, with a task-adaptive customization module. Amortized variational inference, no per-user gradient steps.

## Method in one paragraph
Encoder mean-pools embeddings of the observed (item, rating) pairs into a latent task representation r; a task
identity network + global pool adapt decoder parameters to the user; an adaptive decoder predicts ratings. Because
it is a neural process, it yields a stochastic-process (Gaussian-like) predictive uncertainty over the value-carrying
SET input — the R2+R4 shape, but item+side-feature only (not channel-agnostic), and not benchmarked at full-profile SOTA.

## Relevance to CASPER
- **Papers:** A — ancestor of the value-carrying set-input cold-start recommender WITH uncertainty; precedent for R2+R4.
- **Taxonomy slot:** (i)/(iii) amortized meta-learning / neural-process set encoder.
- **Baseline candidate?** optional — cold-start metrics on MovieLens/others (not the full-profile split); public code.
- **Pre-empts / supports:** pre-empts "first value-carrying set-input cold-start rec" and "first uncertainty over the
  set input" — must cite; wedge is full-profile SOTA + channel-agnostic tokens + monotone-per-question.

## Verdict
Cite as the neural-process precedent that already pairs a value-carrying set input with predictive uncertainty; it is
not full-profile SOTA and not channel-open, which is where our instrument differs.
