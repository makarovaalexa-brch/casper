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
