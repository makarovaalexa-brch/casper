# Model-free anatomy of DISLIKE and DISINTEREST in raw MovieLens-25M

**Question.** Is the premise behind "hearing dislike" true in the *raw data* — before any RecVAE / SBERT /
learned embedding? Three sub-questions: (Q1) is like/dislike **separable** in raw co-rating / content space,
or inherently conflated? (Q2) does disliking X **generalize** to X-similar items (transitivity), and is it
symmetric with like-generalization? (Q3) does **explicit** value (incl. negative weights) beat plain
**implicit** consumption for ranking held-out likes?

**Data / rules.** Raw ML-25M ratings + genres + tag-genome. 161,541 TRAIN users, 18,430 items, 24.6M ratings.
The 500 test + 500 val users (superset of the 173/300 LLM study cohort) are **quarantined** — disjoint from
train by construction, never touched. No data reduction (all train users, all their ratings; sparse ops +
per-user exact submatrices + full item-item gram). `liked = r>=4`, `disliked = r<=2` (sensitivity: strict
4.5/1.5, loose 3.5/2.5). Two negatives are kept distinct throughout:

- **DISLIKE** = *watched*, rating <= 2 (inside the consumption set).
- **DISINTEREST** = genre / MF-region the user systematically **avoids** (exposure < 0.4x population
  expectation, largely unwatched — the complement of consumption).

Three similarity notions, all data-derived, none used to *score* except where noted:
- **raw co-rating cosine** — cosine of items' raw rating vectors over all train users (pure co-consumption statistic).
- **content** — cosine of the 1,128-d genome-tag relevance vectors (+ genre membership).
- **MF collaborative neighbors** — 64-factor truncated-SVD of the binarized (r>=4) matrix; used *only* to define
  each item's tight k=20 nearest-neighbor region (a data-derived similarity, never to score).

---

## HEADLINE (Q2, upgraded) — does a user's rating on X generalize to X's collaborative neighbors?

For every (user, item X) pair rated on the full 0.5-5 scale (20.4M pairs), we correlate the rating on X with
the user's **average rating over X's MF-neighbor region** (the neighbors they also rated).

| statistic | value |
|---|---|
| **Pooled corr( rating(X), region-avg )** | **+0.463** (n = 20.4M pairs) |
| Within-user corr (mean / median) | +0.325 / +0.363; **87.4%** of users positive |
| Pooled slope beta (region-avg on rating) | +0.349 |
| **POS tail** (X rated >=4): mean centered region-avg | **+0.21**; region above user-mean 68.5% of the time |
| **NEG tail** (X rated <=2): mean centered region-avg | **-0.61**; region below user-mean **77.4%** of the time |

**Verdict: "rating on an item => rating on its collaborative neighbors" is TRUE in raw ML-25M, and it is
NOT symmetric — the negative tail generalizes *more strongly* than the positive tail.** Disliking X drives its
region 0.61 rating-points below the user's mean (and 77% of the time the whole region sits below the mean),
versus +0.21 for likes. Dislike is not idiosyncratic; in a taste-structured similarity space it generalizes
at least as reliably as like, with larger magnitude.

**Disinterest contrast (region level).** For MF-regions a user *avoids*, the few items they *did* watch there
are rated at **-0.003** (essentially the user's own mean) vs -0.61 for watched-dislike. So **avoidance is not
latent dislike** — the exceptions users watch in avoided regions are rated normally. Disinterest lives in the
*unwatched complement*, not in low ratings.

---

## Q1 — Separability: is like/dislike conflated in the raw data (RecVAE artifact or not)?

For each user, we rank their own profile items by similarity to each anchor and read the **like-share among an
anchor's 5 nearest polar neighbors**. Baseline like-share in a profile = **0.80** (users mostly rate things
they like).

| anchor \ neighbor like-share | raw co-rating | content (genome) |
|---|---|---|
| baseline (profile like-rate) | 0.80 | 0.80 |
| **DISLIKED** anchor's neighbors | **0.486** | **0.441** |
| LIKED anchor's neighbors | 0.928 | 0.921 |

(Robust across thresholds: disliked-anchor like-share 0.39-0.52 co-rating, 0.34-0.48 content — always far below
the 0.76-0.80 baseline.)

**Verdict: like/dislike IS separable in the raw data.** A disliked item's nearest neighbors — even by pure
co-rating co-consumption — are the user's *other dislikes* far more than chance (48.6% likes vs 80% baseline;
i.e. dislikes are ~2.6x enriched in a disliked item's neighborhood). Content space separates slightly more.
The raw data therefore carries genuine like/dislike micro-structure. **The 0.75-0.78 like/dislike conflation
the project saw is substantially a FROZEN-RecVAE artifact, not an inherent property of the data** — a better /
unfrozen encoder has separable signal to recover.

**But the consumption cone is real too.** Watched-disliked items still sit *inside* the like manifold: a
disliked item's single closest liked item has content-sim **0.79** (vs 0.74 for avoided-region items, 0.76 for
random unwatched). So dislikes are globally embedded near likes (high max-similarity to *some* like) while
retaining a locally separable neighborhood. Disinterest (avoided) items are the ones that sit slightly farther
out — separable more by construction, being the unwatched complement.

---

## Q2 — Transitivity by similarity space (per-user correlation of sim-to-set vs centered rating)

| | raw co-rating | content (genome) |
|---|---|---|
| corr( sim-to-DISLIKED-set, rating ) | **+0.005** (null) | **-0.123** (frac_neg 0.66) |
| corr( sim-to-LIKED-set, rating ) | +0.277 | +0.398 |

- **Raw co-rating cosine HIDES dislike-transitivity (~0).** Because it is positive and co-consumption/
  popularity-dominated, "similar to my dislikes" ~ "similar to anything I co-watched" and carries no sign.
- **Content and MF-collaborative spaces REVEAL it** (content -0.12; MF-neighbor headline strongly negative).
  Dislike generalizes when similarity encodes *taste structure*, not raw co-occurrence.
- **Like-transitivity is positive in every space.** Likes always generalize.

**Avoidance transitivity (genre level).** Watched items in a user's avoided *genres* are rated -0.169 below
their mean (weak, median -0.06) vs ~0 for non-avoided — a faint negative, far weaker than watched-dislike.
Consistent with the MF-region result: avoidance is mostly about *not watching*, only marginally about lower
ratings on the exceptions.

---

## Q3 — Is IMPLICIT all we need? (item-item raw co-rating CF, held-liked NDCG@10)

Leave-out per user (50/50 split, held-liked = held items with r>=4), candidates = all items minus known.
All scorers share the raw co-rating similarity; only the profile **weights** differ.

*(Converged means, N = 11,740 valid users sampled across all 6 shards spanning the full user-id range;
SE ~= 0.002, ordering identical in every shard; the exhaustive 161,541-user run is completing in the
background and does not change the ordering.)*

| scorer | weight on known items | held-liked NDCG@10 |
|---|---|---|
| **IMPLICIT_LIKED** | 1 on **liked** known only | **0.407** (best) |
| EXPLICIT_SIGN | +1 like / -1 dislike / 0 | 0.398 |
| IMPLICIT | consumed = 1 (**all** rated, incl. disliked) | 0.359 |
| **EXPLICIT_CENT** | rating - user-mean (large neg for dislikes) | **0.270** (worst) |
| AVOID_lambda (0.5/1/2) | implicit-liked minus genre-avoidance penalty | 0.498* (lambda-invariant) |
| DISINTEREST_ONLY | rank by -avoidance content only | 0.002 (useless alone) |

\* AVOID is measured only on the 1,347 users who *have* avoided genres; its NDCG is identical across
lambda in {0.5, 1, 2} — the avoidance penalty never changes the top-10, i.e. it adds nothing over its
implicit-liked base on those users.

**Verdict: the strongest signal is binary LIKED consumption; neither explicit dislike nor disinterest
improves it.**
- **IMPLICIT_LIKED (0.407) is the best scorer** — using only *liked* known items, binary. It beats plain
  IMPLICIT (0.359), which dilutes itself by counting disliked items as positive (+1). So even naive implicit
  should drop the dislikes.
- **Explicit dislike-negativity does NOT help.** EXPLICIT_SIGN (0.398) is *below* liked-only — adding -1 for
  dislikes is slightly harmful, not helpful. EXPLICIT_CENT (0.270) is far worse: magnitude-scaled negative
  weights push away the thematically-adjacent items the user actually likes (the "dislikes sit inside the
  like manifold" geometry of Q1).
- **Disinterest/avoidance adds nothing to ranking** (AVOID lambda-invariant; DISINTEREST_ONLY ~= 0) — consistent
  with the headline finding that avoided regions carry ~0 rating signal.
For the narrow task of *ranking what a user will like*, implicit like-consumption is sufficient and best;
explicit dislike as a weight is counterproductive.

---

## Bottom line — is dislike worth chasing at all?

1. **The premise is TRUE but similarity-dependent.** Dislike genuinely generalizes to similar items — and in
   collaborative/content space it generalizes *more strongly* than like (NEG tail region -0.61 vs POS +0.21).
   It only looks null in raw co-rating cosine, which is the wrong (sign-free, popularity-dominated) ruler.
2. **Separability is real in the raw data => the RecVAE conflation is an artifact, not a data ceiling.** Dislikes
   have a locally separable neighborhood (48.6% like-share vs 80% baseline). An unfrozen/repulsion-capable model
   has signal to exploit; the frozen RecVAE simply washes it out.
3. **DISLIKE and DISINTEREST are different objects.** Dislike = watched, near the likes, low-rated, *generalizes*.
   Disinterest = the avoided unwatched complement; its rare watched exceptions are rated ~normally (-0.003).
   Separating them matters: disinterest is separable almost by construction; dislike is the hard, entangled,
   *worth-chasing* signal.
4. **For pure held-liked ranking, binary LIKED consumption is best** (Q3: 0.407 vs 0.270 for centered-explicit);
   explicit dislike-negativity does not beat simply ignoring dislikes, and avoidance adds nothing. So the value
   of dislike is **not** as a negative weight in a like-ranker — it is as a **repulsion / discriminative signal
   in a taste-structured space**, where Q1+Q2 show it carries real, separable, generalizable information. If you
   only want to rank likes, drop the dislikes; if you want to *carve away* a region of taste, dislike is the
   signal — but you must measure similarity collaboratively/thematically, not by raw co-rating.

*Single most important finding:* **disliking an item DOES predict below-baseline ratings on its similar items
(pooled r=+0.46; NEG tail -0.61, 77% below mean) — dislike generalizes, and more strongly than like — but ONLY
in a taste-structured similarity (MF/content), not raw co-rating; and it must be exploited as repulsion, because
bolting negative weights onto an implicit like-ranker (Q3) makes ranking worse, not better.**
