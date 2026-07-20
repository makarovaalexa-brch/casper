# steck2019ease — Steck 2019, Embarrassingly Shallow Autoencoders for Sparse Data (EASE)

- **Venue/year:** WWW 2019
- **Link / DOI:** arXiv:1905.03375 ; 10.1145/3308558.3313710
- **Status:** skimmed (numbers cross-checked to RecVAE/Mult-VAE tables)

## Essence
Closed-form item-item linear autoencoder: a single dense weight matrix B with zero diagonal, solved in closed
form from the Gram matrix (X^T X) via a Lagrange constraint. No SGD, no hidden layer. On ML-20M strong-gen split:
NDCG@100 **0.420**, Recall@20 **0.391**, Recall@50 **0.521** — beats/ties much deeper models. THE weak-but-
unbeatable shallow baseline.

## Method in one paragraph
Minimize ||X - XB||^2 + λ||B||^2 subject to diag(B)=0. Solution B = I - P·diagMat(1/diag(P)) where P = (X^T X + λI)^-1.
Scoring a user = row of X times B; a new user folds in instantly (their interaction vector × B) with no retraining —
so it is natively a fold-in / any-length-input recommender. Item-only, one-hot indicator basis, point estimate.

## Relevance to CASPER
- **Papers:** A — the full-profile R1/G0 bar; instant fold-in exemplar; the item-indicator basis that R3 (concept/
  continuous tokens) is structurally impossible in.
- **Taxonomy slot:** (ii) fold-in linear / closed-form projection.
- **Baseline candidate?** yes — ML-20M NDCG@100 0.420 (reported); public code; Tier 2. Must snap on our harness.
- **Pre-empts / supports:** supports c3 "R1-corner models fail R3/R4"; supports flat-frontier design fact.

## Verdict
Cite as the shallow full-profile bar every claim must clear; its item-only basis is the clean illustration of the R3 gap.
