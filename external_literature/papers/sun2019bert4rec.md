# sun2019bert4rec — Sun et al. 2019, BERT4Rec

- **Venue/year:** CIKM 2019
- **Link / DOI:** arXiv:1904.06690
- **Status:** abstract-only (+ replicability caveat, petrov2022replicability)

## Essence
Bidirectional self-attention (masked-item / cloze training) for sequential recommendation; at inference appends a
[mask] and scores catalog items for that position. Reported to beat SASRec, but Petrov & Macdonald (RecSys 2022)
found the gains fragile / under-tuned on replication.

## Method in one paragraph
Transformer encoder trained by masking random items in the sequence and reconstructing them (BERT-style); inference
masks the final position. Fold-in = re-encode the sequence, no retrain. Item-only, order/context-sensitive, point
estimate, non-monotone. Same taxonomy corner as SASRec.

## Relevance to CASPER
- **Papers:** A — sequence-model fold-in baseline; the replicability caveat is a `dacrema2019progress`-style warning.
- **Taxonomy slot:** (v) sequence model.
- **Baseline candidate?** optional — include with SASRec but tune carefully; watch `petrov2022replicability`.
- **Pre-empts / supports:** supports the sequence-≠-interview and token-≠-value-carrying points.

## Verdict
Cite as the bidirectional sequence baseline; treat reported wins skeptically per the replicability study.
