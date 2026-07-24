# steck2010training — Steck 2010, "Training and Testing of Recommender Systems on Data Missing Not at Random"

- **Venue/year:** KDD 2010
- **Link / DOI:** dl.acm.org/10.1145/1835804.1835895
- **Status:** deep-researched (abstract + secondary summaries)

## Essence
Foundational MNAR paper. Ratings are **missing not at random** — users disproportionately rate items they like, so
the *absence* of a rating is itself informative. Introduces performance measures estimable without bias under MNAR
and the **AllRank** training surrogate: account for ALL items per user (observed + missing), impute a low value for
missing, weight observed=1 vs missing=w_m, regularize. AllRank improves the top-k hit rate of a simple MF over
training only on observed ratings. Also a source of the ">4 = relevant" binarization convention.

## Method in one paragraph
Least-squares MF modified so the loss ranges over the full user×item grid with imputed values and per-cell weights,
turning "which items were rated at all" into signal rather than treating unobserved as unknown.

## Relevance to CASPER
- **Papers:** A (the theoretical root of the reveal-set effect).
- **Taxonomy slot:** MNAR / all-items training objective.
- **Baseline candidate?** no (principle cite, not a fold-in ranker to run).
- **Pre-empts / supports:** PRE-EMPTS "revealing more items (incl. dislikes) as membership helps top-N" as a novel
  phenomenon (findings §4). Our 0.3435→0.4022 reveal-set effect is a direct instance of AllRank/MNAR.

## Verdict
Cite as the reason the reveal-set effect is *expected*, not new; frame our contribution as its clean quantification
and the effect-size ordering (reveal-set ≫ grade), not as discovering that membership matters.
</content>
