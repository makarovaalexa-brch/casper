# Findings: Graded / Explicit Inputs for RANKING (not RMSE) — the binarization consensus, and whether our "graded premium at matched information" is novel

Deep-research pass 2026-07-24. Feeds **Paper A** (the instrument: graded/signed values per entity as an
architecturally-inexpressible leg) and **Paper C** (continuous-answer / graded-value elicitation).
Reads on top of `paperA_recommender_landscape.md` (do not duplicate the SOTA/taxonomy there).

**The claim under test:** on ML-25M, folding a user's revealed set into our frozen strong tower with the
*graded* rating values attached beats folding the *same* revealed set as *binary* membership by **≈ +1.7
NDCG@10 points** (0.3435 binary-membership → ~0.361 graded, with a separate reveal-set effect 0.3435→0.4022
from revealing more items as positives), plus the values carry sign/intensity semantics. Question: has anyone
already measured a **graded-vs-binary premium at matched information**, and what is the published-faithful
baseline plan to defend it?

**One-line verdict (see §6):** the *general* idea "grades help top-N, prior architectures threw them away" is
**well-established and must be cited** (Frolov & Oseledets 2016 is the load-bearing precedent; OrdRec/CoFiRank/
xCLiMF/ListRank/SQL-Rank are the graded-ranking model line). But the *specific* thing we measure — a **clean
matched-information decomposition** (identical reveal set, membership-only vs membership+values) folded into a
**frozen certified strong tower**, reported as a controlled NDCG@10 delta — **we are aware of no prior art that
isolates it this way.** Every prior "grades help" result is method-vs-method (confounded objective) or reported
graphically. Claim it narrowly as a *measurement*, not as "first to use grades."

---

## 1. HISTORY — how "binarize for top-N" became the default (and what it actually concluded)

The consensus is **empirical and architecture-conditional**, never "ratings are information-free for ranking."

- **`cremonesi2010performance` (RecSys'10, Cremonesi/Koren/Turrini).** The origin. Showed models tuned for **RMSE
  do not win at top-N**, and that **PureSVD** — SVD on the 0/1 matrix with *all missing = 0* — beats the
  RMSE-optimal latent-factor models (and non-personalized) on top-N precision/recall (ML, Netflix). The lesson
  the field *took* was "treat the matrix as binary implicit and evaluate ranking"; the lesson the paper actually
  proved was narrower: **RMSE-optimal use of the rating value is the wrong objective for top-N.** It did *not*
  test a ranking-objective model that uses the graded value. That gap is where our claim lives.
- **`hu2008collaborative` (Hu/Koren/Volinsky, ICDM'08).** The implicit-feedback template: split the raw number
  into **preference (0/1) × confidence (magnitude)**. The magnitude re-enters only as a *loss weight*, never as a
  signed direction. Institutionalized "value = confidence, not preference," which is the assumption our signed
  values break.
- **`steck2010training` (KDD'10, "…Data Missing Not At Random").** The other pillar. Argued the **absence** of a
  rating is itself informative (MNAR) and introduced **AllRank** — impute a low value for all unobserved items,
  weight observed vs missing — improving top-k over MF. This is the theoretical root of our **reveal-set effect**
  (§4): *which* items a user bothered to rate is signal, independent of the value. Steck also popularized the
  ">4 = relevant" binarization used everywhere downstream.
- **`liang2018variational` (Mult-VAE, WWW'18).** Cemented the modern protocol: **binarize at rating > 3.5**,
  strong-generalization split, multinomial likelihood over the 0/1 bag. The multinomial explicitly does *not*
  treat unobserved as disliked — but it also **discards the graded value entirely**. EASE (`steck2019ease`),
  RecVAE (`ren2020recvae`), EDLAE all inherit this exact >3.5-binary input. **The entire ML-20M NDCG@100 frontier
  is measured on binarized input by construction** — nobody on that leaderboard ever fed the model the grade.
- **`volkovs2015effective` (SIGIR'15).** Explicitly argued latent models built for *explicit* ratings do poorly
  on binary implicit data and must be redesigned — reinforcing the "different regime" framing, again without a
  matched graded-vs-binary top-N test.

**Precise conclusion of the consensus:** *"For the objectives and architectures we used, binarizing did not hurt
(and RMSE-on-grades hurt) top-N, so we standardized on binary."* NOT *"the grade contains no ranking signal."*
The distinction is exactly the wedge our measurement exploits, and reviewers who know this literature will accept
the wedge **only if** we cite Cremonesi/Steck/Liang for the consensus and Frolov for the counter-current.

## 2. PUBLISHED GRADED-FOR-RANKING ATTEMPTS (the ones that did try to keep the grade for ranking)

| Method | Venue | Uses grade how | Claims top-N gain vs binarized? | Data / metric | Code | ML-25M-scale? |
|---|---|---|---|---|---|---|
| **CoFiRank** `weimer2007cofirank` | NIPS'07 | Max-margin MF optimizing a **structured NDCG** surrogate on ordinal ratings | Beats regression MF at ranking; **no controlled binarized-at-matched-info arm** | EachMovie/ML-1M, NDCG@10 | yes (old C++) | struct-SVM, heavy; not at 25M easily |
| **ListRank-MF** `shi2010listrank` | RecSys'10 | Listwise top-1 cross-entropy over graded scores | Beats CoFiRank/itemCF at NDCG; grade enters via the softmax targets | ML/Netflix/EachMovie, NDCG | yes (Java/py) | linear in observed ratings → scalable |
| **OrdRec** `koren2011ordrec` | RecSys'11 | **Ordinal** model: predicts a full rating *distribution* via ordered thresholds | Improves ranking (FCP/NDCG) over pointwise MF; ordinal ≫ numeric | Netflix/Yahoo/ML, RMSE+FCP | reimpls exist | scalable (MF core) |
| **GAPfm / TFMAP** `shi2013gapfm` | (CIKM/RecSys'13) | Optimize **Graded Average Precision** directly | Beats binary-AP models on graded domains | ML/Amazon, GAP | partial | mid |
| **xCLiMF** `shi2013xclimf` | RecSys'13 | Optimize **Expected Reciprocal Rank** (multi-grade generalization of CLiMF) | **Beats CLiMF *when >2 relevance levels exist*** — the closest "grades beat binary, same model family" statement | TED/ML, ERR/NDCG | yes (py/spark) | mid |
| **SQL-Rank** `wu2018sqlrank` | ICML'18 | Listwise permutation likelihood, **handles ties + graded** | Beats WMF/BPR (implicit) and explicit MF/collab-ranking | ML1M/10M/Netflix, NDCG@k | yes (Julia) | 10M shown; 25M plausible |
| **CoFFee / "Fifty Shades"** `frolov2016fifty` | RecSys'16 | **3rd-order tensor** (user×item×rating-category); folds the *whole rating spectrum incl. negatives* | **THE load-bearing precedent**: "how to benefit from negative feedback in top-N"; SOTA standard + wins negative-only cold-start | ML-1M/10M, nDCG + novel **nDCL** | yes (Polara) | tensor fold-in, mid-scale |
| **Loss-Aversion / neg-preference** `paudel2018loss` | arXiv'18 | Adds a **negative-preference** term (graph/embedding) to push disliked items down | Improves accuracy *and* cuts negatives-at-top | 2 public sets, top-N | partial | mid |

**Reading of the table for our purpose:**
- The **graded-ranking-loss line** (CoFiRank → ListRank → xCLiMF → GAPfm → SQL-Rank) proves *a graded ranking
  objective can beat a binary one*, but almost always **compares different models/objectives** — the gain is
  confounded with the loss function, not isolated as "same model, same reveal set, grade on vs off." **xCLiMF is
  the single cleanest statement** ("advantage over CLiMF *when* >2 relevance levels exist"), and it is the one to
  cite as the nearest match to "grade beats binary within one family."
- The **negative-feedback line** (Frolov 2016, Paudel 2018, and the modern sequential-negative work — Xie 2021,
  Google 2023 `2308.12256`, 2025 surveys) proves **sign** (disliking) carries top-N signal. Frolov is the
  strongest and most citable: it explicitly frames the field as *insensitive* to negative feedback and shows a
  model that uses the full spectrum. **This pre-empts any claim that "using the sign/negatives is new."**
- **None** of them report our exact experiment: fold an identical revealed set into a **frozen SOTA tower** as
  binary-membership vs graded, and read off the NDCG@10 gap. They build bespoke graded models end-to-end.

## 3. MATCHED-INFORMATION COMPARISON (binary-fold vs graded-fold of the SAME reveal set) — closest prior art

This is the exact claim boundary. Searched hard; the honest result:

- **No paper isolates membership-vs-values on a fixed reveal set** as a controlled ablation with a graded-premium
  delta. The graded-ranking papers swap the *objective*; the negative-feedback papers swap the *model*; none hold
  the revealed set and the tower fixed and toggle only "value attached vs not."
- **Nearest analogues:**
  - **xCLiMF vs CLiMF** (`shi2013xclimf`) — same author, same CLiMF backbone, graded (ERR) vs binary (RR). This is
    the closest "same family, grade on/off" result, but it changes the *loss*, not just the input, and reports ERR
    not a fold-in NDCG@10 delta. **Cite as the prior art we are most similar to, and differentiate on
    frozen-tower-fold-in + matched-input.**
  - **Frolov CoFFee vs SVD** (`frolov2016fifty`) — full-spectrum vs binary, but different architecture (tensor vs
    matrix) and results are largely **graphical / nDCL**, not a clean paired NDCG@10 premium.
  - **Ordinal-NMF / ordinal-MF ablations** (Gouvert 2020; `koren2011ordrec`) — show ordinal treatment beats
    numeric, but for **rating prediction / FCP**, not a matched top-N fold-in.
- **Verdict:** our matched-information decomposition (identical reveal set → membership vs membership+value, both
  folded into the *same frozen tower*, reported as paired NDCG@10) appears to be **an unoccupied measurement.** We
  should present it as *"to our knowledge the first controlled measurement of the graded premium at matched
  information for cold-start fold-in,"* never as "first to use grades."

## 4. THE REVEAL-SET EFFECT (all-rated-as-positive helps: 0.3435 → 0.4022)

Our second observation — revealing **more** items (including dislikes) folded as **binary positives** raises top-N
even without the grade — is **a known principle, not novel**, and has a name-adjacent lineage:

- **Steck MNAR / AllRank** (`steck2010training`): the very fact that a user *rated* an item (regardless of value)
  is informative; imputing/accounting for the observed-vs-missing pattern raises top-k. Our effect is a direct
  instance: adding revealed items (even disliked ones, as membership) enlarges the observed set → better fold-in.
- **Hu/Koren/Volinsky** (`hu2008collaborative`): "the item was interacted with at all" = the preference bit;
  magnitude is only confidence. A disliked-but-rated item still carries the interaction bit.
- **Cremonesi PureSVD** (`cremonesi2010performance`): treating the rating pattern as binary presence already wins
  top-N — consistent with membership-alone being strong.

So the reveal-set effect is **explained/pre-empted in principle** by the MNAR + confidence literature; it is *not*
a novel phenomenon. What is fresh is the **clean quantification on our tower** and the fact that it **dominates**
the graded premium (0.0587 reveal-set vs ~0.017 grade), which is itself a useful, citable calibration: *most of
the cold-start signal is "which items," and the grade is a real-but-second-order refinement.* Frame it that way —
it disarms the "why bother with grades" reviewer by admitting the ordering of effect sizes up front.

## 5. PUBLISHED-FAITHFUL REPLICATION / BASELINE PLAN

To make "measured graded premium at matched information" bullet-proof at a RecSys/UMAP-class venue:

**A. The matched-information ablation itself (the headline).** On the canonical Liang-recipe ML-25M split
(`docs/STATE`/CLAUDE metric block), for each held-out user fold the *same* revealed set into the *same frozen
tower* under two encodings: (i) **binary membership** (all revealed items = 1, the Liang/EASE/RecVAE-native input
— this is the standard "binary upgrade" and needs no new baseline), (ii) **graded** (value attached: signed
intensity). Report **paired NDCG@10 (full + tail)** with a bootstrap CI over users. Add a **third arm** —
membership of *positives only* (>3.5) vs *all-rated* — to separate the reveal-set effect from the grade (this is
exactly the 0.3435 vs 0.4022 vs graded decomposition). Controls (HARD RULE #5): shuffle the values across items
(destroys grade, keeps membership → must collapse to the binary number), and a leak check on the fold/target split.

**B. Graded-ranking baselines to replicate *as published* (so a reviewer can't say "you only compared to your own
binary fold"):** at minimum **SQL-Rank** (`wu2018sqlrank`, ICML'18, public Julia, handles ties+grades, ML-10M
protocol published) and **xCLiMF** (`shi2013xclimf`, public, the graded-vs-binary within-family result). Optional
stretch: **CoFFee/Polara** (`frolov2016fifty`) for the negative-feedback arm and **ListRank-MF** as the classic
listwise anchor. Replicate each on its *own* reported dataset/metric first (HARD RULE #2 — snap to published
numbers) before running on ML-25M; if they don't reach ML-25M scale, report on ML-10M/1M and state the scale
caveat rather than under-tuning them.

**C. Binary "upgrade" baselines that are simply standard (no novelty claimed):** all-rated-as-binary fold into
EASE/RecVAE/the frozen tower (§A-i) IS the accepted strong binary baseline — cite Cremonesi (PureSVD = all-missing-0)
and Liang (>3.5 binarize) as the protocol authority. Most-popular + item-kNN floors per `dacrema2019progress`.

**D. What reviewers of this claim will demand (pre-register these):**
1. **Matched information, provably** — identical reveal set both arms; show the shuffle-value control collapses.
2. **A published graded-ranking model in the table**, run as-published, not just your own toggle (SQL-Rank/xCLiMF).
3. **Effect-size honesty** — report that the reveal-set effect > grade effect; don't oversell +1.7pt.
4. **Cite the consensus AND the counter-current** — Cremonesi/Steck/Liang (binarize) *and* Frolov/xCLiMF/OrdRec
   (grades carry ranking signal). A claim that ignores Frolov 2016 will be desk-flagged.
5. **Sign/intensity semantics** demonstrated separately (a disliked item pushes related items *down*), since the
   number alone (NDCG@10) doesn't prove the sign is used — mirror Frolov's negative-only probe.

## 6. NOVEL vs PRE-EMPTED — verdict on "measured graded premium at matched information"

**PRE-EMPTED (must cite, must NOT claim as first):**
- "Grades/ordinal structure carry top-N ranking signal that RMSE-binary throws away" → OrdRec, CoFiRank, ListRank,
  **xCLiMF (grade beats binary within a family)**, GAPfm, SQL-Rank.
- "Negative feedback / the sign of preference improves top-N and cold-start" → **Frolov & Oseledets 2016 (THE
  precedent)**, Paudel 2018, and the 2021–2025 sequential-negative line.
- "Which items a user rated (membership), independent of value, is informative" (the reveal-set effect) → **Steck
  2010 MNAR/AllRank**, Hu-Koren-Volinsky confidence, Cremonesi PureSVD.

**NOVEL / defensible as a *measurement* (aware-of-none, not "there is none"):**
- A **controlled matched-information decomposition** — identical reveal set, **frozen certified strong tower**,
  toggling only membership vs graded value — reported as a **paired cold-start NDCG@10 premium** with the
  reveal-set effect separated out. No prior work isolates the grade this cleanly on a frozen SOTA fold-in; they
  swap objectives or architectures. This is a *calibration/measurement* contribution, and its honest framing
  (reveal-set ≫ grade, grade still real at +~1.7pt with sign semantics) is itself the citable delta.
- Positioning the grade as an **architecturally-inexpressible input for the R1-bar item-indicator models**
  (EASE/RecVAE/Mult-VAE literally cannot ingest a signed value — they binarize by construction) — a *structural*
  argument the graded-ranking papers never make because they build bespoke graded models.

**Bottom line for the paper:** lead with the *measurement* framing ("we quantify, at matched information, how much
the grade is worth on a frozen SOTA tower — and find the reveal-set dominates, the grade is a real second-order
gain with usable sign"), cite Frolov 2016 + xCLiMF + Cremonesi/Steck/Liang prominently, run SQL-Rank/xCLiMF
as-published in the baseline table, and never write "first to use ratings for ranking."

## Bib entries to add (hand to author; do not edit references.bib here)
- `steck2010training` — Harald Steck. "Training and Testing of Recommender Systems on Data Missing Not at Random." KDD 2010.
- `hu2008collaborative` — Hu, Koren, Volinsky. "Collaborative Filtering for Implicit Feedback Datasets." ICDM 2008.
- `frolov2016fifty` — Frolov & Oseledets. "Fifty Shades of Ratings: How to Benefit from a Negative Feedback in Top-N Recommendation Tasks." RecSys 2016. (arXiv:1607.04228)
- `koren2011ordrec` — Koren & Sill. "OrdRec: An Ordinal Model for Predicting Personalized Item Rating Distributions." RecSys 2011.
- `weimer2007cofirank` — Weimer, Karatzoglou, Le, Smola. "CoFiRank: Maximum Margin Matrix Factorization for Collaborative Ranking." NIPS 2007.
- `shi2010listrank` — Shi, Larson, Hanjalic. "List-wise Learning to Rank with Matrix Factorization for CF (ListRank-MF)." RecSys 2010.
- `shi2013xclimf` — Shi et al. "xCLiMF: Optimizing Expected Reciprocal Rank for Data with Multiple Levels of Relevance." RecSys 2013.
- `shi2013gapfm` — Shi et al. "GAPfm: Optimal Top-N Recommendations for Graded Relevance Domains." CIKM 2013.
- `wu2018sqlrank` — Wu, Hsieh, Sharpnack. "SQL-Rank: A Listwise Approach to Collaborative Ranking." ICML 2018. (arXiv:1803.00114)
- `paudel2018loss` — Paudel, Luck, Bernstein. "Loss Aversion in Recommender Systems: Utilizing Negative User Preference to Improve Recommendation Quality." arXiv:1812.11422, 2018.
- `volkovs2015effective` — Volkovs & Yu. "Effective Latent Models for Binary Feedback in Recommender Systems." SIGIR 2015.
</content>
</invoke>
