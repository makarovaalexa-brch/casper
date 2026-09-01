# User segmentation / clustering for cold-start elicitation — searched 2026-08-29

**Why this exists.** The author asked on 2026-08-28 whether MovieLens segmentation already exists in
the literature. It was NOT searched at the time — the clustering (`m6y`, `m7a`) was built from
scratch instead, against hard rule 4. This is the retro-search. **Verification level: abstracts,
publisher pages and search summaries only — NO full text was read.** Marked `[A/S]` throughout; any
number quoted from here must be upgraded to `[V]` before it enters the chapter.

---

## ★ THE FINDING THAT MATTERS FOR POSITIONING

**"Cold-start = identify which cluster the user belongs to" is the STANDARD motivation of the exact
line we build on — not a gap we found.**

- **Golbandi, Koren & Lempel, WSDM'11** (`golbandi2011adaptive`, the source of our tree) is motivated
  verbatim this way: users "can often be grouped into clusters based on the similarity of their
  preferences", which "allows accelerated learning of new user preferences since the task becomes one
  of identifying which cluster a user belongs to — typically there are far fewer clusters than items
  to be rated." The tree's job is to find **"distinguisher items"** that reveal the cluster. `[A/S]`
- **Shams, Anderson & Leith, "Cluster-Based Bandits: Fast Cold-Start for Recommender System New
  Users", SIGIR 2021**, DOI 10.1145/3404835.3463033 — the *same* framing and the *same* term
  "distinguisher items", as an online bandit rather than a tree. `[A/S]`

⛔ **Consequence: "we route each user to the teacher for their group" is NOT a contribution.** It is
the field's standard motivation, and a reviewer who knows Golbandi will say the tree already does
this internally, one question at a time. Our forest must be positioned against that, not as if it
were new. **This is the single most important thing this search produced.**

## What IS still ours, and what is now contested

| our element | status after this search |
|---|---|
| routing to a group-specific interview | ⛔ **standard** (Golbandi'11, Shams'21) |
| grouping users, fitting a model per group | ⛔ **large established literature** — CCCF and k-means CF, below |
| asking **attribute/concept** questions to place a user | ⚠ **contested since Oct 2025** — see 2510.27342 |
| routing from what the user **VOLUNTEERED unprompted** rather than from system-chosen probe items | ✔ **no prior found** — this is the real differentiator |
| collapsing the whole forest into **one distilled policy** | ✔ no prior found |
| optimising **NDCG@10** rather than RMSE | ✔ the elicitation-tree line is RMSE-only (see `baseline_reported_metrics.md`) |

★ The defensible line is therefore **the source of the routing signal**, not the routing. Golbandi and
Shams both choose *distinguisher items to ask*; we read *what the user said first, unprompted*. That
is exactly what the Reddit opener work supports, and it should carry the framing.

## Closest recent competitor — CITE AND DISTINGUISH

**"Pairwise and Attribute-Aware Decision Tree-Based Preference Elicitation for Cold-Start
Recommendation"**, arXiv **2510.27342** (Oct 2025, KU Leuven / Lirias). `[V, full text read 08-29]`

⚠ **CORRECTION to this file's first version (08-29, abstract-grade).** It said "Yahoo! Webscope;
ranking-based metrics (not RMSE)". **Both halves were wrong** — that came from PDF metadata, not the
paper. The full text says:

- **Dataset: Yahoo! Music ONLY** — final cohort **1,701 users**, 4,793 artists, 218 genres. Not
  MovieLens, not movies, and two orders of magnitude smaller than our 10,000-user evaluation.
- **Metric: RMSE, and nothing else.** No NDCG, no precision@k, no recall.
- **ONE global tree**, retrained each elicitation round — *not* per-segment trees. "Clustering" is the
  ordinary three-way branch at a leaf, not a user partition with its own fitted interview.
- Baselines: popularity, variance, entropy, HELF, IGCN, plus its own tree variants. Golbandi is cited
  as lineage.
- Novelty, verbatim: *"(i) ... a decision tree-based rating elicitation algorithm that gathers both
  item and attribute ratings outperforms one that collects only item ratings ... (ii) ... a decision
  tree constructed using pairs of items outperforms one based on individual items."*
- ★ **"The system always chooses what to ask. Users do not volunteer preferences."**

**Consequence: far less threatening than the abstract suggested, and it CORROBORATES us twice.**
(a) It is the newest paper in the line and it still reports **RMSE only** — direct support for our
"optimise the reported metric" spine. (b) It confirms by contrast that the elicitation line never
lets the user speak first, which is precisely our differentiator. Cite it as the current state of the
art in attribute-aware elicitation, and separate on metric, dataset scale, and volunteered input —
**not** on "concepts help place users", which it does share.

## Cluster-based CF generally (context, not competition)

Large and old. Co-clustering (CCCF), k-means CF, per-cluster local models: "the task of learning a
predictive model from the complete rating data space is reduced to each cluster of users". Standard
critique in that literature: one-dimensional clustering discards structure in the other dimension.
None of it is about *elicitation*, so it bounds our novelty claim without competing directly. `[A/S]`

## Not found (weak evidence of absence — ONE search, not a systematic review)

- **No paper found showing MovieLens latent structure is dominated by release year / joining cohort.**
  Our `m6y` result — 14 of 20 raw-latent clusters have distinctive films spanning under 2.5 years —
  may be original. ⚠ One search is not proof of absence; do NOT claim novelty on this without a
  proper systematic pass.
- No prior found for routing a cold-start interview from an unprompted free-text request.

## Sources
- Golbandi, Koren & Lempel, WSDM'11 — https://dl.acm.org/doi/10.1145/1935826.1935910
- Shams, Anderson & Leith, SIGIR'21 — https://dl.acm.org/doi/10.1145/3404835.3463033
- arXiv 2510.27342 — https://arxiv.org/abs/2510.27342
- Awesome-Cold-Start-Recommendation (survey list) — https://github.com/YuanchenBei/Awesome-Cold-Start-Recommendation
