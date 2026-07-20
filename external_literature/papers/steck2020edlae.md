# steck2020edlae — Steck 2020, Autoencoders That Don't Overfit Towards the Identity (EDLAE)

- **Venue/year:** NeurIPS 2020
- **Link / DOI:** (NeurIPS 2020)
- **Status:** abstract-only

## Essence
Denoising / higher-order linear autoencoder for CF that corrects EASE's tendency to overfit toward the identity
mapping; adds a principled dropout/denoising correction and higher-order terms, tying RecVAE-class accuracy on
ML-20M (NDCG@100 in the 0.42–0.44 band as reported). Confirms the linear frontier is essentially flat.

## Method in one paragraph
Derives a closed-form correction to the EASE solution that accounts for the denoising objective (emulating dropout on
inputs), preventing the trivial self-reconstruction and yielding better generalization; optional higher-order (item
pair/triple) extensions. Item-only, closed-form, instant fold-in, point estimate.

## Relevance to CASPER
- **Papers:** A — evidence the shallow/linear full-profile frontier is flat; a strong closed-form fold-in baseline.
- **Taxonomy slot:** (ii) fold-in / closed-form linear autoencoder.
- **Baseline candidate?** yes (Tier 2) — ML-20M ≈ RecVAE class (reported tie); public code.
- **Pre-empts / supports:** supports the "flat frontier / shallow tower is defensible R1 substrate" design fact.

## Verdict
Cite alongside EASE/RecVAE as the linear frontier's upper edge; another item-only point estimator (fails R3/R4).
