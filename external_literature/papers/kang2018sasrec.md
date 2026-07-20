# kang2018sasrec — Kang & McAuley 2018, Self-Attentive Sequential Recommendation (SASRec)

- **Venue/year:** ICDM 2018
- **Link / DOI:** arXiv:1808.09781
- **Status:** abstract-only

## Essence
Causal (unidirectional) self-attention over a user's interaction sequence to predict the next item; captures long-
and short-range dependencies, strong on sequential next-item benchmarks. The canonical self-attentive sequence rec.

## Method in one paragraph
Embeds each item + position, applies stacked masked self-attention blocks, predicts the shifted sequence (next-item).
At inference a new user's session is re-encoded with no retraining — implicit fold-in — but the input is an ORDERED,
ITEM-ONLY token sequence, not a value-carrying set; no uncertainty; a later token can reorder all scores (non-monotone).

## Relevance to CASPER
- **Papers:** A — the "token-input" strawman: tokens ≠ value-carrying mixed-type set; sequence ≠ interview.
- **Taxonomy slot:** (v) sequence model (implicit fold-in).
- **Baseline candidate?** yes (as a fold-in baseline) — strong on sequential splits, NOT the ML-20M top-N split; public code.
- **Pre-empts / supports:** supports the standing warning "do not claim interview-native = token input (SASRec is token-based)".

## Verdict
Cite to pre-empt the trivial "we take tokens" novelty framing; a point-estimate, order-sensitive, item-only baseline.
