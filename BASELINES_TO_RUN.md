# BASELINES TO RUN (2026-07-14) — verified from primary sources
Our ruler: ML-25M, strong-generalization fold-in (known half in, held-out LIKED items as targets), full
18,430-item catalog, NDCG@10 (+ head-masked TAIL). Recommender: set-encoder, full-profile **0.4852**
(RecVAE on the same harness: **0.4998**).

---

## ⭐ THE ANSWER TO "MARTIN ET AL. ON WHAT RECOMMENDER, WHAT NDCG?"
**Martin, Boutilier, Meshi, Sandholm, "Model-Free Preference Elicitation", IJCAI-24, pp. 3493-3503.**
[VERIFIED from the PDF]
- **Recommender: probabilistic matrix factorization, d=50** (Salakhutdinov & Mnih 2007). NOT neural. A
  Bayesian recommender (multivariate Gaussian over u, MCMC/HMC), treated as a black box.
- **Catalog: 1,000 ITEMS.** Verbatim: *"We use the 100 most common attributes and 1000 most common items for
  each dataset."* ML-25M in name only.
- **Users: ENTIRELY SYNTHETIC.** Responses ~ `Bernoulli(sigma(u.q))`; utility = `u.x`. **No held-out real
  ratings are ever scored.** Training episodes come from a UNIFORMLY RANDOM query policy.
- **Metric: expected utility of the recommended item.** `grep -c NDCG` over the full text = **0**. No NDCG, no
  recall, no rank metric. **All results are FIGURES — there is NO number to quote.**
- **Their only baseline is a RANDOM query policy.** They do not claim to beat a static questionnaire.
- **No code** (the only GitHub link is DeepMind's `mctx` MCTS library).
- **Method (this part IS portable):** two set-input models on logged episodes — a response model
  `f_r(h,q) -> Delta(R)` and a **value-of-history model `f_v(h) -> R` trained by regression on realized
  utility**; then `EVOI(h,q) = sum_r P(r|h,q) f_v(h + (q,r)) - f_v(h)`. **DeepSets wins** among 5 architectures.
  Planning: depth-0 (myopic) or MCTS.

### => OUR COMBINATION IS NOVEL, AND THE GAP IS LARGER THAN FEARED
| | Martin et al. | CASPER |
|---|---|---|
| recommender | PMF d=50 | set-encoder, **0.4852** (RecVAE-class) |
| catalog | **1,000 items** | **18,430 items** |
| users | **synthetic** Bernoulli(sigma(u.q)) | 150k real profiles + a distilled LLM answerer |
| answers | binary yes/no | graded (hated/meh/liked/loved) + **REFUSALS** |
| metric | expected utility (figures only) | **NDCG@10 / TAIL** |
| baseline | random query policy | static questionnaire (the hard one) |
| EVOI | **ENUMERATED** (needs a response model f_r) | **AMORTIZED** (regress the expectation directly) |
**The last row is a genuine methodological distinction: they enumerate the answer distribution; we regress the
expectation. Ours needs no response model at selection time.**

---

## ⭐ THE CHEAPEST NOVELTY AVAILABLE TO US
**Greedy SLIM's authors TRIED to reproduce Golbandi and Sepliarskaia, FAILED, and EXCLUDED BOTH.**
=> **There is NO published head-to-head of Golbandi's tree vs Sepliarskaia's SPQ vs Greedy SLIM.**
=> **Run all three on one ruler and THAT IS ITSELF A CONTRIBUTION.**
**Greedy SLIM (arXiv 2406.06061; PREPRINT, not RecSys 2024) reports ML-25M NDCG@10 = .339 / .359 / .375 at
5 / 10 / 20 questions** — **the ONLY prior ML-25M-scale NDCG@10 elicitation result in existence.** Our closest
direct comparator.

---

## ⭐ GREEDY SLIM — OUR EXACT COMPETITOR ON OUR EXACT RULER (priority 1)
[VERIFIED, and the facts below are NOT in dispute] ML-25M **UNFILTERED** (162,541 users / 59,047 items),
**full-catalog NDCG@10, UNSAMPLED** (they cite Krichene & Rendle), **WITH a tail split**, questions EXCLUDED
from the recommendable set. Reported ML-25M NDCG@10: **.3390 / .3594 / .3709 / .3752 at 5 / 10 / 15 / 20 Q.**
**COST: ~19 MINUTES PER ROW** on ML-25M (they compute 20) => ~6h one-time preprocessing.
**HOW TO RUN:** extract the top-k item ORDERING, feed that fixed question sequence into OUR answerer + OUR
recommender + OUR NDCG@10. Apples-to-apples on the POLICY only. 2-4 days.

### ✅ CONFLICT RESOLVED FROM THE PAPER'S OWN TEXT (arxiv.org/html/2406.06061v1, read 2026-07-14)
The lit agent's DETAILED report was RIGHT; its FINAL SUMMARY was WRONG. **The curve below IS static-vs-adaptive.**
Verbatim from the paper:
- **QGSLIM is STATIC:** *"Our approach describes a **static questionnaire** but we believe that our ideas can be
  easily extended to the dynamic setting."* Items are chosen in a PREPROCESSING phase; the same sequence goes to
  every new user.
- **QBandit is DYNAMIC:** Christakopoulou et al. 2016, **Thompson sampling**, LFM trained with **PureSVD**;
  response-conditioned question selection. **The paper uses the words "static" and "dynamic" explicitly and
  places QGSLIM in the first and QBandit in the second.**

| ML-25M NDCG@10 (their Table 3) | 5Q | 10Q | 15Q | 20Q |
|---|---|---|---|---|
| **QGSLIM (STATIC)** | .3390 | .3594 | .3709 | **.3752** |
| **QBandit (DYNAMIC)** | .3382 | .3456 | .3221 | **.2916** |
**The dynamic method is level at 5Q and then COLLAPSES — it gets WORSE with more questions.**

**⚠ HOW TO READ THIS WITHOUT VIOLATING HARD RULE #2.** This is NOT evidence that adaptivity does not pay — our
own 150k-user probe (+47% TAIL, no policy, no RL) settles that. It is evidence that **A THOMPSON-SAMPLING BANDIT
OVER A PureSVD LFM IS A WEAK ADAPTIVE METHOD**, and the paper names the reason itself:
> *"asking questions about popular items may have a negative effect ... if the recommender system is unable to
> generalize."*
**That is OUR probe's mechanism, stated from the losing side.** Their bandit chases POPULAR items (exactly what
our static arm picks: The Usual Suspects, rank 8/800). Our cluster-adaptive arm asks **NICHE in-genre polarisers**
(median rank 471/800). **Same ruler, same dataset, opposite outcome — and the difference is WHICH questions
adaptivity is allowed to reach for.** This is a headline contrast for the paper, not a threat.

### ⚠ CANNIBALISATION — CONFIRMED IN THEIR TEXT, AND IT APPLIES TO US
**Their exclusion rule, verbatim: `I_Q ∩ I_R = ∅` — "to avoid trivial recommendations".** IDENTICAL to our
masking rule. So when their bandit asks about the popular items it would otherwise have RECOMMENDED,
**ADAPTIVITY CANNIBALISES ITS OWN SLATE** — and the more questions it asks, the more of its own top-10 it burns.
**That is the shape of the .3382 → .2916 collapse.**
**WE HAVE THE SAME EXCLUSION RULE, SO THE SAME EXPOSURE.** Every time our Q asks about a film the user would
have loved, it burns that film out of its own top-10.
**⇒ MEASURE IT THE MOMENT THE TARGETS LAND: what fraction of Q's picks would have been in the top-10 anyway?**
**NOTE — WE MAY ALREADY BE IMMUNE, AND IT IS TESTABLE.** Our probe's adaptive arm asks NICHE items (rank
471/800) under the SAME masking rule and still wins **+47% TAIL**. A niche item is rarely in the top-10 it would
be cannibalising. **PREDICTION: cannibalisation cost scales with the POPULARITY of the asked item — which makes
"ask niche" not just informative but CHEAP.** If that holds it is a genuine finding and it explains their curve.

**⚠ AND IT CUTS AGAINST THE ANSWERABILITY STORY:** GSLIM asks about items users KNOW only **34.4%** of the time
vs QBandit's **48.7%** — **the WINNER deliberately asks LESS ANSWERABLE questions.** Plus a 103-person user
study where the adaptive LFM method is statistically indistinguishable from a NON-PERSONALISED static
recommender (52.6% vs 52.0%) while static-SLIM reaches 77.2%. (Consistent with our own E0 null: answerability
routing bought −0.0003. **Answerability is a CONSTRAINT, not an objective.**)

## GOLBANDI: RE-DERIVING HIS SPLIT FOR A RANKING LOSS IS ITSELF A CONTRIBUTION
[VERIFIED] His efficiency rests on a SUFFICIENT-STATISTICS trick (compute for LOVERS+HATERS, derive UNKNOWNS by
subtraction — Unknowns being ~99% of users). **It is SPECIFIC TO SQUARED LOSS and does not survive a switch to
NDCG. KOREN SAYS SO HIMSELF (§8):**
> *"we would like to experiment with other cost functions, especially ones related to the quality of the top-K
> item ranking. **The main challenge would be keeping computation efficient.**"*
Verified details: ternary like(4-5*)/dislike(1-3*)/unknown; node predictor = the node's item MEAN; smoothing
lambda1=200; ensemble weights **w_L=5, w_H=1, w_U=0.02, c=2** (a "like" is worth 5x a "dislike", **250x an
"unknown"**); depth 6 => Netflix RMSE 0.97172. **No public code.**

## CHEAP FLOOR BASELINES (Dacrema makes them awkward to skip)
**MostPop** (<0.5d), **ItemKNN** (0.5d), **SLIM** (0.5d), **ADMM-SLIM** (1d), **Mult-DAE** (0.5d). ~1.5d total.
Dacrema, Cremonesi & Jannach, **RecSys 2019 BEST LONG PAPER** [VERIFIED]: of 18 neural recommenders from top
venues **only 7 reproduced**, and well-tuned classical baselines matched or beat most of them.
**MeLU / TaNP: OUT OF SCOPE — and SAY SO explicitly rather than omitting silently** (rating prediction/MAE,
~3,900-item content-rich catalogs, and MeLU runs a per-user MAML inner loop at TEST time).

## THE MINIMUM SET — the six a reviewer will demand
| # | baseline | why | effort | code |
|---|---|---|---|---|
| 1 | **Static seed battery**: popularity, entropy, **Entropy0, HELF**, random | non-negotiable. "Did you beat popularity?" | **1-2 d** | trivial |
| 2 | **Sepliarskaia SPQ** (RecSys'18) | the static questionnaire that DIRECTLY OPTIMISES THE LOSS. **Our whole thesis lives or dies against it.** | 3-5 d | [Seplanna/pairwiseLearning] (live) |
| 3 | **Golbandi tree** (WSDM'11) | the adaptive ancestor. **If we do not run it, a reviewer assumes we are hiding from it.** | 4-7 d | none |
| 4 | **Martin et al. ported** (DeepSets `f_v` + greedy depth-0 EVOI, NDCG@10 as the realized utility) | the closest realizable rival | 3-5 d | none |
| 5 | **UpsRec greedy-NDCG as the PRIVILEGED ORACLE CEILING** | converts a threat into a contribution | 2 d | none |
| 6 | **EASE** (+ **gSASRec**) on our ruler | the post-Dacrema "did you beat the dumb linear model" check; and the sequential-model question | **<1 d** (+2-3 d) | closed form; [asash/gsasrec] |
**Optional 7th: Greedy SLIM's static ordering** — cheap, and the only prior ML-25M NDCG@10 elicitation result.

**CITE BUT DO NOT RUN** (with a one-line fairness reason each): PERE (simulated users, embedding-derived
ground truth), PEBOL (100-item shortlist, no recommender), FacT-CRS (different task/metric), EAR/CRM/ConTS/
UNICORN (self-pruned 10-item pools, SR@T metric), MeLU/TaNP (rank only the ~10-item query set).

---

## ⚠⚠ CORRECTIONS TO OUR OWN CLAIMS (I was about to overclaim — both are load-bearing)

### PERE: my numbers were MISLABELLED and the strong claim is UNSAFE
[VERIFIED] The triple I called "Gowalla" is actually **Amazon-Books / LightGCN**. Correct table:
| | PEO (static) | c-DPP | PERE | margin | c-DPP recovers |
|---|---|---|---|---|---|
| Amazon-Books / biVAE | 0.2218 | 0.2901 | 0.2918 | 0.0700 | **97.6%** |
| Amazon-Books / LightGCN | 0.3108 | 0.3575 | 0.3616 | 0.0508 | **91.9%** |
| **Gowalla** / LightGCN | **0.1307** | **0.1764** | **0.1806** | 0.0499 | **91.6%** |
So the honest range is **92-98%**, not "~97%".
⚠ **AND THEIR TABLE 10 CUTS AGAINST THE STRONG CLAIM:** a FULLY static DPP-100 at the same budget recovers
only **38-55%**. If c-DPP were merely a response-oblivious staged DPP it would land there. It does not.
**So c-DPP is doing something USER-DEPENDENT that the paper never documents, and
"their non-adaptive control recovers the margin, therefore adaptivity buys nothing" is NOT SAFE.**
**THE DEFENSIBLE SENTENCE:**
> *PERE's own sequential DPP control — which carries NONE of the paper's embedding-region machinery —
> recovers **92-98%** of its margin over the static PEO questionnaire, and PERE's advantage over that control
> is **NOT statistically significant on Gowalla (p=0.198, NDCG@10)**.*
Also: PERE's main tables use a **simulated user** (ground-truth "likes" = the top-k items closest to u0 in the
embedding). Venue **CONFIRMED: UAI 2024** (PMLR v244).

### UpsRec: it peeks at VALIDATION, not TEST. The strong charge is refutable.
[VERIFIED] It uses *"the groundtruth items of the user (in the **validation set**)"* — the authors explicitly
avoid TEST leakage. **State it as:** *"scores each candidate question by the NDCG it would achieve against
held-out ground-truth items, assuming the user answers 'liked' — information no deployed system has at
selection time."* The stronger charge would cost us in review.
Also: **"UpsRec" is not a paper title.** Cite as **Lin, Zhu, Wang, Caverlee, "Enhancing User Personalization in
Conversational Recommenders", WWW 2023, arXiv 2302.06656.**

---

## TRAPS THAT WOULD INVALIDATE A COMPARISON (each has killed someone)
1. **SAMPLED METRICS.** Krichene & Rendle (KDD 2020): sampled metrics are INCONSISTENT with exact ones — they
   do not preserve "A beats B", **not even in expectation**. Compute over the FULL catalog. Always.
2. **CANDIDATE SHORTLISTS.** PEBOL ranks **100 items** — and admits why: *"we have to limit |I| to 100 for fair
   comparison to the MonoLLM baseline."* MRR@10 over 100 items != NDCG@10 over 18,430.
3. **SELF-PRUNED ACTION SPACES.** UNICORN prunes to top-10 per turn; EAR fixes |V|=10. A system can score well
   by SHRINKING ITS OWN POOL rather than ranking well.
4. **SR@T / Average-Turns** is a dialogue-efficiency metric, NOT a ranking metric. Not convertible.
5. **PROTOCOL MISMATCH.** Sequential lit = leave-one-out (ONE target); ours = fold-in (MANY targets).
   **SASRec's 0.13-0.18 must NEVER be tabled beside our 0.4852 without re-running.** Also SASRec is
   ORDER-dependent and our profile is a SET — decide and DISCLOSE (timestamp order vs no positional encoding).
6. **DIFFERENT CATALOGS.** Martin caps at 1,000 items; PEBOL at 100. **No absolute number from any of these
   transfers.**
7. **NO REFUSAL CHANNEL EXISTS ANYWHERE IN THIS LITERATURE.** EAR/UNICORN simulated users consult the true
   target item's attribute set. Our (knowledge, value, refusal) answerer has no counterpart — a contribution,
   but it means answer-model parity must be ARGUED, not assumed.

## CITATION FIXES
- **Sepliarskaia et al. has FOUR authors** (Sepliarskaia, Kiseleva, Radlinski, de Rijke), not three.
- **HELF / Entropy0 / IGCN are defined in Rashid, Karypis & Riedl, SIGKDD Explorations 10(2), 2008** — NOT in
  the IUI 2002 paper (which has only popularity / entropy / pop*ent / item-item).
- **Greedy SLIM is a PREPRINT** (arXiv 2406.06061), not a RecSys 2024 paper.
- **PEBOL = arXiv 2405.00981** (RecSys 2024). "PEBOL" is the system, not the title.
- **gSASRec** (Petrov & Macdonald, RecSys'23, arXiv 2308.07192) is NOT the same paper as Klenitskiy & Vasilev's
  "Turning Dross Into Gold Loss" (also RecSys'23). Do not merge them.
- **EASE**: `B = I - P diagMat(1 / diag(P))`, `P = (X^T X + lambda I)^-1` — confirmed.
