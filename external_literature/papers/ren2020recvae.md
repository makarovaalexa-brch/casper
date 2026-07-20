# ren2020recvae — Shenbin et al. 2020, RecVAE

- **Venue/year:** WSDM 2020
- **Link / DOI:** arXiv:1912.11160
- **Status:** skimmed (results table numbers extracted)

## Essence
A refined VAE for top-N implicit CF: composite prior (mixture of standard + previous-posterior), a new encoder
architecture, and an alternating-update / rescaled-KL training scheme. Top of the ML-20M strong-generalization
split: NDCG@100 **0.442**, Recall@20 **0.414**, Recall@50 **0.553** — beats Mult-VAE (0.426) and EASE (0.420).

## Method in one paragraph
Multinomial likelihood decoder over the item catalog (as Mult-VAE), encoder maps a user's item-indicator vector to
a Gaussian latent; RecVAE adds a composite prior anchored at the previous posterior for stability, and alternates
encoder/decoder updates. Fold-in = encode the new user's interaction set at inference (any-length, no retrain).
Latent posterior exists but is item-only and not used as an actionable elicitation belief.

## Relevance to CASPER
- **Papers:** A — the strongest full-profile amortized-encoder baseline; our in-house RecVAE-d512 ruler
  (0.4998 NDCG@10 on ML-25M) is this class.
- **Taxonomy slot:** (iii) amortized inference encoder.
- **Baseline candidate?** yes — ML-20M NDCG@100 0.442 (reported); public code; Tier 2, the R1 bar to tie.
- **Pre-empts / supports:** supports R1 bar; its item-indicator input is R3-blind (dislike-blind, no concept token).

## Verdict
The full-profile number to tie for G0; the amortized-encoder family CASPER's set encoder belongs to.
