# wu2018sqlrank — Wu, Hsieh, Sharpnack 2018, "SQL-Rank"

- **Venue/year:** ICML 2018
- **Link / DOI:** arXiv:1803.00114 ; PMLR v80/wu18c ; code github.com/wuliwei9278/SQL-Rank (Julia)
- **Status:** read (abstract + repo)

## Essence
Listwise collaborative ranking cast as **maximum likelihood under a permutation (Plackett-Luce-style) model** over
a low-rank score matrix; **handles ties and missing data**, runs in linear time. Beats implicit baselines
(Weighted-MF, BPR) and is competitive-to-better than explicit MF/collaborative-ranking. Reported on ML-1M/10M and
Netflix, NDCG@k. The most modern, scalable graded/listwise ranker with clean public code.

## Method in one paragraph
Place probability mass on permutations of each user's items via the latent score matrix; ties (equal ratings) and
missing entries are handled in the likelihood, so graded ratings define partial orders rather than a binary bag.

## Relevance to CASPER
- **Papers:** A, C.
- **Taxonomy slot:** listwise permutation-model collaborative ranking.
- **Baseline candidate?** yes — the primary published graded-ranking baseline for §5B; ML-10M protocol published,
  25M plausible. Snap to its reported ML-10M NDCG before extending.
- **Pre-empts / supports:** PRE-EMPTS "listwise graded objective beats binary implicit for top-N." Does not do a
  matched-information fold-in on a frozen tower.

## Verdict
Run as-published in the baseline table so reviewers can't say we only compared to our own binary toggle.
</content>
