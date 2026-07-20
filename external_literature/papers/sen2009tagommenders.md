# sen2009tagommenders — Sen, Vig & Riedl 2009, Tagommenders: Connecting Users to Items through Tags

- **Venue/year:** WWW 2009
- **Link / DOI:** https://dl.acm.org/doi/10.1145/1526709.1526800
- **Status:** abstract-only

## Essence (3-6 lines)
Proposes "tagommenders" — recommender algorithms that predict a user's preference for an item by first inferring their preference for the item's tags, then aggregating tag-preference to an item-level score. Evaluated using tag-preference ratings collected from 995 MovieLens users on actual movie tags.

## Method in one paragraph
Users' interactions with tags and movies are used to infer per-user tag-preference scores (tag preference inference algorithms); an item's predicted score is then computed by combining the user's inferred preferences for that item's associated tags. This makes tags an intermediate, semantically interpretable representation layer sitting between raw user-item interactions and the final recommendation score.

## Relevance to CASPER
- **Papers:** A — an early, non-neural instance of a channel-agnostic intermediate representation (tags) bridging users and items, conceptually analogous to CASPER's concept channel that folds through the same encoder interface as items.
- **Taxonomy slot:** tag-based / concept-channel recommender
- **Baseline candidate?** No — 2009-era linear tag-preference model on a small (995-user) MovieLens tag-rating sample; not a competitive numeric baseline for ML-25M scale, but a citable conceptual ancestor.
- **Pre-empts / supports which of our claims:** Supports the "concepts as an answerability channel" thread — tags-as-intermediate-representation is the direct conceptual predecessor of CASPER's concept-channel fold-in, decades before the learned-embedding version.
- **Note:** ML tag data is the same MovieLens family CASPER uses (ML-25M); worth checking if tag-genome data overlaps.

## Verdict
Cite as the earliest tag/concept-as-intermediate-channel prior art; useful historical anchor when motivating CASPER's channel-agnostic (item/concept/text) encoder interface.
