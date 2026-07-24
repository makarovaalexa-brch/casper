# frolov2016fifty — Frolov & Oseledets 2016, "Fifty Shades of Ratings"

- **Venue/year:** RecSys 2016
- **Link / DOI:** arXiv:1607.04228 ; ACM RecSys'16 p91
- **Status:** deep-researched (abstract + ar5iv body; results are largely graphical)

## Essence
THE load-bearing precedent for "use the full rating spectrum, including NEGATIVE feedback, to improve top-N."
Argues conventional CF and its metrics are *insensitive to negative feedback* because they only reward putting
relevant items high, never penalize putting irrelevant items high. Proposes **CoFFee** — a third-order tensor
factorization (user × item × rating-category) treating feedback as a categorical variable — with a higher-order
folding-in for online/cold recommendation. Claims **state-of-the-art in the standard scenario AND large wins in
the negative-only / "no positive feedback" cold-start** case. Data: ML-1M/10M (online eval up to 22M). Metrics:
precision/recall/nDCG plus a novel **nDCL** (normalized Discounted Cumulative Loss) that penalizes irrelevant hits.
Reported: exact-rating hit 47%, correct feedback-positivity 95% on ML-1M; head numbers are shown in figures, not
tables — no clean paired graded-vs-binary NDCG delta is printed.

## Method in one paragraph
Fold the (user,item,rating) triples into a Tucker/CP tensor so each rating value is its own slice; a new user's
few graded answers are folded in via a higher-order projection, yielding scores that are sensitive to the sign and
magnitude of feedback (a single disliked movie steers recommendations toward "opposite" features). Contrast to
matrix SVD/PureSVD which collapse the value and are shown "worst in nDCL."

## Relevance to CASPER
- **Papers:** A (graded/signed value leg — the strongest pre-empting cite), C (negative/graded answers steer rec).
- **Taxonomy slot:** graded/negative-feedback top-N (tensor fold-in).
- **Baseline candidate?** yes (optional) — Polara code public; ML-1M/10M nDCG+nDCL; a natural negative-feedback arm.
- **Pre-empts / supports:** PRE-EMPTS "using the sign/negatives for top-N is new" (findings §2, §6). Does NOT
  isolate the matched-information graded-vs-binary premium on a frozen tower → our measurement survives.

## Verdict
Cite prominently as the counter-current to the binarization consensus; a claim that ignores it is desk-flag risk.
Differentiate: bespoke tensor model + graphical nDCL results, NOT a controlled grade-on/off delta on a frozen SOTA tower.
</content>
