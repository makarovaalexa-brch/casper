# M1 — the forest's starting points: taste clusters, era removal, and plain-word anchors

**Design sheet** `docs/design/DS_M1_STATE_MATCHING.md` + `PAPER_B_POLICY_DESIGN.md` Part M.
**Code** `src/instrument/m1_train_states.py`, `m1_deera.py`, `m1_sweep.py`, `m1_era_probe.py`,
`m1b_vocab_full.py`. **Artifacts** `experiments/battery/m1/…`.
Fitted on the **140,768 TRAIN users**; **NDCG is never computed on train** (D.6b) — everything here
is a function of a user's own ratings. No TEST data built, loaded or named.

## The headline

**After removing a release-era axis, taste clusters at k=20 are semantically coherent, and four of
the six clean ones are anchored by a word people actually use — with zero drift from the optimal
attribute.**

| cluster | n | best attribute (992) | best plain word (341) | distinctive films |
|---|---|---|---|---|
| c3 | 6,882 | `arms dealer` (m .156) | **`marvel`** (m .100) | Iron Man 3 · Avengers: Age of Ultron · Captain America ×2 · Iron Man 2 |
| c7 | 4,576 | `art house` (m .210) | **`art house`** — same | Holy Motors · The White Ribbon · A Field in England · Pervert's Guide to Cinema |
| c5 | 7,475 | `dialogue driven` (m .120) | **same** | Birdman · Whiplash · Grand Budapest Hotel · Nightcrawler · Dallas Buyers Club |
| c4 | 5,825 | `gore` (m .072) | **same** | Dredd · Watchmen · The Hateful Eight · True Grit · The World's End |
| c10 | 9,743 | `inspiring` (m .168) | **same** | Dark Knight Rises · Interstellar · Wolf of Wall Street · Django · The Imitation Game |
| c6 | 4,945 | `teenagers` (m .067) | `school` (m .054) | Aquamarine · Princess Diaries 2 · Monte Carlo · Burlesque · Frozen |

The anchor and the films **agree** in every row — that is the check, and it is what four earlier
configurations failed.

Four further clusters clear the gate but are **not coherent**: c8 `1930s`, c13 `70mm`, c14 `spoof`,
c15 `teens`. Their distinctive films are all 1997–2000, i.e. **residual era structure** the single
projection did not remove. They are reported, not used.

## The coverage/granularity trade-off (de-era'd states, scale-aware gate)

| k | anchorable | users covered | share | median cluster |
|---|---|---|---|---|
| 6 | 3 / 6 | 72,151 | 51.3% | 23,210 |
| 8 | 5 / 8 | 80,952 | 57.5% | 16,465 |
| **12** | **7 / 12** | **97,280** | **69.1%** | 10,620 |
| 16 | 9 / 16 | 94,350 | 67.0% | 7,462 |
| 20 | 10 / 20 | 73,272 | 52.1% | 5,781 |
| 30 | 12 / 30 | 71,389 | 50.7% | 4,213 |
| 40 | 15 / 40 | 66,746 | 47.4% | 2,965 |

**Coverage peaks at k=12; nameability keeps improving to k=20.** The count of anchorable clusters
rises monotonically while their total reach falls after k=12 — more teachers, each serving fewer.

**Partial coverage is the design, not a shortfall.** Users in unanchorable regions get the cold tree,
which the policy keeps verbatim; the opener census found 17.0% of real requests volunteer nothing
routable at all.

## ★ The release-era axis, and why it had to go

The first film check showed three of four clusters defined by a *window of years* (1995–96, 2009–14,
2013–15) rather than by taste. That is a property of MovieLens — users join at a time and rate what
is current — and it is **not volunteerable**: nobody can state when they started rating.

Measured (`m1_era_probe.json`): decoding each principal state direction through the recommender's own
decoder and correlating with item release year gives **PC1 ρ = +0.709** (14.1% of variance),
PC0 +0.362 (29.1%), PC2 −0.352 (6.2%) — nearly half the variance era-loaded — while PC3 (5.3%)
carries **+0.036**, i.e. genuine taste sitting under an era axis three times its size.

Removal was **targeted, not PC surgery**: every item's year is known, so `d_era = Wᵈᵀ y` is the state
direction whose decoded scores are maximally year-aligned. Its own correlation is **ρ = +0.875**,
confirming the construction. Projecting out that one direction:

| | before | after |
|---|---|---|
| PC0 ρ(year) | +0.362 | **+0.061** |
| PC1 ρ(year) | +0.709 | **+0.169** |
| PC2 ρ(year) | −0.352 | −0.113 |

State norm fell only 10.868 → 10.598 (2.5%) — a thin direction with outsized influence. Dropping
whole PCs would have discarded the real taste PC0 carries alongside its era loading.

## Method decisions that were corrected along the way (all author-caught)

1. **Per-user identifying facts are not an opener.** A2 gave every user their own best fact (1,069–
   1,230 distinct items) — coherent as a statement about *volunteering*, useless as a question.
   Anchors are chosen **per cluster** so members share one entry state; `fold(fact)` does not depend
   on who states it, so within-cluster state variance is zero by construction.
2. **The fact must be an attribute, and it must push users AWAY from other clusters.** Selection is
   by **margin** — own share minus the largest share in any other cluster, on a held-out half — not
   by coverage or a coverage/purity balance. `pg-13` covers 90% of a cluster and separates nobody.
3. **The Reddit vocabulary was truncated.** `m1b_map.py` mapped only each slot's census **top-50**
   values, and its 34 targets were then used as if they were the whole vocabulary. The full tail
   (`m1b_vocab_full.py`) yields **352 tags**, of which 341 survive as candidate concepts. Under the
   truncated vocabulary two of four clusters had *negative* plain-word margins; that result is
   **retracted** — with the full vocabulary no cluster does.
4. **An absolute margin threshold is not comparable across k.** Shares fall mechanically as k grows,
   so one fixed number penalises fine granularity and drives a merge toward blobs. The gate is
   expressed in **prior units** (purity ≥ 2× prior, margin ≥ 0.5× prior).
5. **The merge loop was a ratchet** — it only stopped when *every* cluster passed, so one stubborn
   cluster dragged k down and destroyed good clusters on the way. Replaced by the sweep above.
6. **Raw top-N films are the popularity axis.** The first film check returned Shawshank for four of
   six clusters. Films are now reported as **lift over the global centroid**; both lists are kept in
   the artifact.
7. **Clusters must be fitted on TRAIN.** On 10,000 validation users a 1,200 floor caps k at ~8 and
   forces blobs. Clustering is a fitting step (D.6b): fit on 140,768, measure on validation.


## ★★★ Step 4 — where real requests actually land (independent validation)

`src/instrument/m1_reddit_clusters.py` · `experiments/battery/m1/m1_reddit_clusters.json`.
The 3,078 folded Reddit arrival states are projected onto the **same** era axis as the train states
(so both sit in one geometry) and compared on **direction** — a request folds ~2.9 named films
(state norm 6.15) against a train profile's ~62 (norm 10.60).

**The six coherent clusters hold 28.0% of train users and absorb 75.7% of real requests.**

| cluster | anchor | train share | request share | ratio |
|---|---|---|---|---|
| c10 | `inspiring` | 6.92% | **20.18%** | 2.91× |
| c5 | `dialogue driven` | 5.31% | **16.83%** | 3.17× |
| c4 | `gore` | 4.14% | **15.95%** | 3.85× |
| c3 | `marvel` | 4.89% | **11.50%** | 2.35× |
| c6 | `teenagers` | 3.51% | 6.11% | 1.74× |
| c7 | `art house` | 3.25% | 5.10% | 1.57× |

**And the clusters flagged as era artifacts are exactly the ones nobody requests:** c8 `1930s`
0.68% · c15 `teens` 0.88% · c13 `70mm` 1.14% · c14 `spoof` 1.33%; the largest train cluster c0
(21,477 users, 15.26%) draws **1.49%** of requests (0.10×).

**★ This is independent confirmation.** Coherence was judged on the MovieLens side — anchor agreeing
with distinctive films — and nothing in the clustering saw Reddit. That the same six clusters carry
three quarters of real traffic, and the chronological artifacts carry almost none, is a check the
procedure could have failed and did not.

**Arrival structure agrees with taste structure well above chance, and imperfectly: ARI = +0.269.**
Five of eight arrival clusters map cleanly onto a taste cluster with matching semantics —
r1 → `gore` (64%: Pumpkinhead, The Beyond, Halloween 6, Leatherface) · r6 → `dialogue driven`
(67%: A Field in England, Climax, The House That Jack Built) · r2 → `inspiring` (56%: Midnight in
Paris, The Theory of Everything, La La Land) · r4 → `marvel` (54%: Captain America, Doctor Strange,
Iron Man 2) · r0 → c11 (43%). The three diffuse ones (r3 28%, r5 35%, r7 27%) all show 1995–96
films, so **residual era is visible on the request side too**.

**Reading:** an opener places a user roughly, not exactly. The gap between +0.269 and perfect
agreement is the space an adaptive interview exists to close — now a measured quantity rather than
an assumption. For deployment the practical figure is the one that matters: **six teachers reach
75.7% of real traffic**, against the 28.0% that the MovieLens-side coverage alone would suggest.

## Provenance and gates

Certified pair `t2i26c_ep31` + `concept_cv3_ep08`; canonical m=0 snap `0.15686538118354684` asserted
exact in every stage that builds an env; `ctx.show2id == unique_sid.txt` asserted; leak assert clean.
Endorsement uses the env's own SEL rule (`foldable & V>0`), applied identically to train slices.
Purity and margin are measured on a **held-out half** of train users throughout. k-means on
L2-normalised states, seed 98765 (direction, not magnitude — the populations carry very different
evidence quantities).

## What M2 receives

The six coherent k=20 clusters as candidate forest starting points, each with a shared anchor whose
fold is the cluster's entry state, plus the k=12 configuration as the higher-coverage alternative.
The screen then asks whether any question beats the tree's own first question by ≥ +0.004 **at that
entry state**.
