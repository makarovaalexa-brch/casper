# biyik2023soft — Bıyık et al. 2023, Preference Elicitation with Soft Attributes in Interactive Recommendation

- **Venue/year:** arXiv 2023 (preprint)
- **Link / DOI:** arXiv:2311.02085
- **Status:** skimmed (full-text confirmation of the VoI integrand still pending — see elicitation findings open Qs)

## Essence
Interactive recommendation with a multivariate-Gaussian belief over a user embedding in a learned item space. "Soft
attributes" (e.g. "scarier") are Concept Activation Vectors — DIRECTIONS in the item space — so item queries and
attribute queries fold into the SAME Gaussian belief via Bayes. Query selection by Entropy / InfoGain / EVOI ("BPER").
THE central near-miss for our instrument claim (R2 + R3 + R4 together).

## Method in one paragraph
User taste = latent vector with a Gaussian prior; each answer (item comparison or soft-attribute judgment) is a
noisy linear observation along a direction (item vector or CAV), giving a conjugate-ish Gaussian posterior update
(their Eq. 7); next query chosen to maximize an information/expected-value criterion (Eq. 13). Crucially the item
embedding space is CO-TRAINED with the belief model, and the true posterior is "generally not Gaussian" (approximated).

## Relevance to CASPER
- **Papers:** A (central threat to the instrument-conjunction claim), C (VoI/query selection).
- **Taxonomy slot:** (iv) Bayesian / Gaussian belief over an item space.
- **Baseline candidate?** yes (Tier 3, closest match) — reports on its own elicitation task, NOT the full-profile
  EASE/RecVAE bar; partial code.
- **Pre-empts / supports:** PRE-EMPTS "attribute-answers-as-directions + Bayesian fold" and "unified item+attribute
  VoI in one Gaussian belief" (see `elicitation_and_belief_pool.md`). Our surviving wedge: EXACT linear-Gaussian
  conjugacy over a TRULY FROZEN certified RecVAE-class tower + arbitrary-embedding/open-concept/continuous tokens.

## Verdict
The paper to cite on the first page of related work and to beat on differentiation: it fails R1 (co-trained, not a
frozen SOTA tower; no full-profile benchmark) and its R3 is attribute-directions, not arbitrary/open/continuous tokens.
