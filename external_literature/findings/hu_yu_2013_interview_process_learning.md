# Hu & Yu, "Interview Process Learning for Top-N Recommendation", RecSys 2013 — RESOLVED

Status: **`[V]` full text read directly, TWICE** (author supplied the ACM PDF 2026-08-11; re-read
line by line 2026-08-12 for plan item B1 / TODO 1.12, and **corrected**). PDF at
`external_literature/papers/hu_yu_recsys2013_interview_process_learning.pdf`, extracted text at
`external_literature/sources_raw/hu_yu_recsys2013.txt` (line numbers below refer to it). 4 pages
(short paper), pp. 331–334. Bib key `hu2013interview`, added to `new_chapters/references.bib`
2026-08-12 (collision-checked against all four `.bib` files first — none).

Replication + objective ablation: `docs/results/HU_YU_2013_REPLICATION_AND_ABLATION.md`.

This closes the standing `[HIGH]` novelty threat in `ndcg_optimising_elicitation.md`. **The threat is
real but partial: this is the closest prior work to our construction, and it does not occupy our cell.**

## What it is

An interview decision tree in the fMF/Golbandi lineage. The user latent factor is the output of a
function `f` in decision-tree space over the interview answers (l. 141–148); each node is one **item
to rate** plus a **rating threshold**, and it has three branches — "higher than threshold", "lower
than threshold", "Unknown" (l. 143–146). The node latent is additive along the path,
`f(x_i) = SUM_{s in path} Δu_s` (Eq. 3, l. 155–158), regularised by `Ω(f) = Σ_s |Δu_s|²` (Eq. 4), so
each node's factor is pulled toward its parent's — they call this **layer-wise functional matrix
factorization**. Item factors `V` are fitted by regularised coordinate descent, alternating with the
tree fit (Eq. 5/6, l. 178–187). Depth limit "usually set to a small number between 3 and 10" (l. 201);
**depth == number of interview questions**, and it is the x-axis of every results figure. The method
is called **EIP** ("efficient interview process"). λ_u = 10 by cross-validation (l. 333–334); the
latent dimension `d` is never stated.

## The split criterion — the load-bearing fact

> "at each node, we select a best interview question (i.e. a best item to rate) and a best rating
> threshold **by optimizing the loss function Equation (2)**; then the tree will split the user set
> into three parts." (l. 196–199)

and the loss is general:

> "For the loss function L, it can be square loss, logistic loss, **pairwise rank loss**, etc."
> (l. 107–108)

So **yes — this is an interview tree whose split objective may be a ranking loss.** That is Golbandi's
stated future work, done in 2013, and it means "we are the first to put a ranking objective in an
interview tree's split" is **NOT available to us**. It must be cited.

### The Taylor trick (worth having independently of the replication)

Solving Eq. (7) — the optimal increment `Δu` per child — exactly, for every candidate item × threshold,
is a full re-fit per candidate. They avoid it by expanding to second order about `Δu = 0`
(l. 219–265): with `g = ∇L + 2λΔu` and `H̃ = H + 2λ`,

    min_Δu L̃(Δu) = L̃(0) − ½ gᵀ H̃⁻¹ g        (Eq. 11)

and then, because inverting `H̃` per candidate per child is still too expensive, `H̃` is replaced by
its **diagonal** `D` (l. 269–277):

    min_Δu L̃(Δu) ≃ L̃(0) − ½ gᵀ D⁻¹ g        (Eq. 12)

Two consequences we use directly. `Σ_children L̃(0)` is the node's own loss and is the same for every
candidate, so the argmin of the total child loss is the **argmax of `Σ_C ½ gᶜᵀ(H_C + 2λ)⁻¹gᶜ`** —
structurally an XGBoost split gain. And `g`, `H` are sums over a child's users of per-user quantities
that do **not** depend on the candidate, so they are computed once per node and every candidate is
then a pair of masked sums. **That is the published route to a scale our exact objective cannot reach:
our NDCG tree pays a tower fold per candidate (10.3 h), theirs pays none.** They are explicit that the
approximation only chooses the question — "since there are approximations in the derivations, we only
use it to find the interview question. After the question is found, we use coordinate descent to
obtain optimal Δu" (l. 280–283).

## CORRECTED 2026-08-12 — what the full re-read changed

**1. The ranking experiments they actually RAN used the LOGISTIC loss, not the pairwise rank loss.**
§3.2.1, l. 361–362: *"Firstly, we evaluate the MAP for EIP, FixQ and Warm-MF on MovieLens **since fMF
cannot handle the logistic loss**."* Pairwise rank loss occurs **exactly once in the paper**, in the
"square loss, logistic loss, pairwise rank loss, etc." list attached to **Eq. (1)** — the *basic* MF
objective, one subsection before the tree exists. It is never instantiated, never tuned, never
ablated and never evaluated. The prior version of this file said "a smooth pairwise surrogate" as if
that were what they ran. It is not. The framework admits it; the experiments do not use it.
*Consequence for us: the honest positioning is that Hu & Yu's FRAMEWORK admits a ranking loss and
their EXPERIMENTS optimise a pointwise classification loss on sampled negatives. Our ablation
therefore runs BOTH — `pairwise` and `logistic` — so we cannot be accused of choosing the version
that flatters us.*

**2. Their metric is untruncated and undiscounted.** Eq. (13), l. 302–319: MAP over each test user's
whole observed set, `P@(rank_i(j))` summed over all `j ∈ O_i`. No `@k` cutoff, no positional
discount. Separation (1) is therefore stronger than previously written: it is not merely "smooth
surrogate vs truncated metric", it is that **nothing anywhere in their pipeline — not the training
loss, not the split criterion, not the reported metric — is truncated or position-discounted.**

**3. Negative sampling is in the TRAINING data too, not only the evaluation.** l. 325–328: "our
experimental setting here is to sample negative instances for training data and test data
separately. For test data ... we randomly sample negative items 10 times as positive items for each
user as test data." So their split criterion is fitted against sampled negatives as well.

**4. Warm-MF beats EIP on two of three datasets, and on the third the text contradicts the figure.**
ML-1M: "Warm-MF works better than EIP ... about 6%" (l. 369–371). Netflix: Warm-MF "outperform EIP by
5% approximately" (l. 381–382). Yahoo: the text claims "EIP outperforms Warm-MF by a bit less than
3%" (l. 392–394), but **Figure 2(c) shows the Warm-MF bar clearly ABOVE the EIP bar at every depth,
by ≈0.025 MAP** (digitised: EIP ≈ .8795/.8802/.8807/.8828, Warm-MF ≈ .9058 flat). The paper's
"almost reaches the rank performance of warm-start" sentence is contradicted by its own figure.
The stated percentages are, on ML-1M and Netflix, **absolute MAP-point** gaps rather than relative.
There is also a plain typo at l. 380: "the MAP of EIP is still higher than that of EIP by about 2%"
(should be FixQ).

**5. Depth is the budget, and their own tree stops paying at 6.** Yahoo: "when the depth reaches 6,
the MAP begins to decrease, which means our model becomes overfitting" (l. 396–398); the RMSE panel
degrades with depth too (l. 408–411). Their budgets are 3/5/8/10 (ML-1M, Netflix) and 3/4/5/6
(Yahoo). Ours is 16.

Digitised figure values (400 dpi render of p. 4; the paper reports no scalar table):

| depth | ML-1M FixQ | ML-1M EIP | ML-1M Warm | NF FixQ | NF EIP | NF Warm | YM FixQ | YM EIP | YM Warm |
|---|---|---|---|---|---|---|---|---|---|
| 3 | .7175 | .7383 | .8018 | .7418 | .7590 | .8133 | .8195 | .8795 | .9058 |
| 5 (YM: 4) | .7233 | .7408 | .8018 | .7447 | .7627 | .8133 | .8237 | .8802 | .9058 |
| 8 (YM: 5) | .7280 | .7443 | .8018 | .7470 | .7677 | .8133 | .8250 | .8807 | .9058 |
| 10 (YM: 6) | .7303 | .7450 | .8018 | .7488 | .7715 | .8133 | .8262 | .8828 | .9058 |

## Why it nevertheless does not occupy our cell — the four separations, re-verified

1. **Nothing they optimise or report is truncated or discounted.** The framework's ranking option is a
   pairwise rank loss (a smooth surrogate over all pairs); what they ran is a pointwise logistic loss;
   what they report is untruncated MAP. NDCG@10 has a top-10 truncation and a positional discount and
   reduces to none of these. This is the same distinction that makes Sepliarskaia's A-optimality
   derivation work — an untruncated, undiscounted loss collapses to a trace criterion; NDCG@10 does not.
2. **They optimise a second-order APPROXIMATION of a TRAINING loss over the tree fit, not the expected
   value of asking a question.** Their split score is Eq. (12): a Taylor expansion, with the Hessian
   replaced by its diagonal, of how much the training loss on the node's own users would fall if the
   child got its optimal factor increment. Ours folds each candidate's answers through the certified
   tower and ranks users' **held-out** targets, i.e. it measures the thing that will be reported.
3. **Sampled-negative evaluation, not full-catalogue.** Ten sampled negatives per positive (l. 325–328),
   against our full 18,359-item catalogue. Sampled-negative ranking metrics are known to distort method
   ordering (Krichene & Rendle, KDD 2020), so their numbers are not comparable to ours and their
   ordering is not evidence about ours.
4. **Metric is MAP, not NDCG** (MAP ×16 in the text; NDCG/nDCG/DCG **zero** occurrences). RMSE is
   reported alongside for the rate-prediction comparison against fMF.

## What it does NOT threaten, and in fact supports

**Their static comparator is popularity.** > "**FixQ** It is short for 'fixed questions' which
represents the method that select **top-k popular items** as static interview questions, and then use
feature based MF to solve the problem by leverage the answers to interview as implicit feedback."
(l. 335–339)

Other baselines: `Warm-MF` (train + answers combined — an upper bound, not a cold-start method) and
`fMF` (Zhou et al., square loss only, so it appears only in the RMSE panel). So the strongest static
arm they beat is **top-k popular** — exactly the weak-static pattern our Act 1 identifies. Nothing
here tests a static order built by maximising the reported metric, which is our comparator. **Our
Act-1 result is not pre-empted.**

## Consequence for the novelty claim

Revised wording. NOT available: "first ranking objective in an interview tree." Still available, and
what we should claim:

> a fixed question order, and an interview tree, selected by directly maximising **measured held-out
> cohort-mean NDCG@10** under a certified recommender, on a full catalogue with no negative sampling,
> selected on a disjoint validation cohort and reported on a disjoint test cohort — and the finding
> that published adaptive selectors lose to the static version of it.

Cite Hu & Yu as the nearest prior attempt and state the four separations up front rather than leaving
a reviewer to find them. Note it is a 4-page short paper with figure-only results (MAP bar charts, no
scalar tables), that its ranking experiments ran a logistic loss, and that its own figure contradicts
its Yahoo warm-start sentence.
