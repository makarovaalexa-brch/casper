# Hu & Yu, "Interview Process Learning for Top-N Recommendation", RecSys 2013 — RESOLVED

Status: **`[V]` full text read directly** (author supplied the ACM PDF 2026-08-11; five prior access
routes had failed). PDF at `external_literature/papers/hu_yu_recsys2013_interview_process_learning.pdf`,
extracted text at `external_literature/sources_raw/hu_yu_recsys2013.txt`. 4 pages (short paper).

This closes the standing `[HIGH]` novelty threat in `ndcg_optimising_elicitation.md`. **The threat is
real but partial: this is the closest prior work to our construction, and it does not occupy our cell.**

## What it is

An interview decision tree (the fMF/Golbandi lineage) whose user latent factor is the output of a tree
over interview answers, trained by coordinate descent. Depth limit "usually set to a small number
between 3 and 10". Three-way splits, as in Golbandi.

## The split criterion — the load-bearing fact

> "at each node, we select a best interview question (i.e. a best item to rate) and a best rating
> threshold **by optimizing the loss function Equation (2)**; then the tree will split the user set
> into three parts."

and the loss is general:

> "For the loss function L, it can be square loss, logistic loss, **pairwise rank loss**, etc."

So **yes — this is an interview tree whose split objective can be a ranking loss.** That is Golbandi's
stated future work, done in 2013, and it means "we are the first to put a ranking objective in an
interview tree's split" is **NOT available to us**. It must be cited.

## Why it nevertheless does not occupy our cell

Four separations, all verified from the text:

1. **Pairwise rank loss, not a truncated discounted top-k metric.** A pairwise loss is a smooth
   surrogate over all pairs. NDCG@10 has a top-10 truncation and a positional discount, and does not
   reduce to a pairwise loss. This is the same distinction that makes Sepliarskaia's A-optimality
   derivation work — an untruncated, undiscounted loss collapses to a trace criterion; NDCG@10 does not.
2. **They optimise it as a TRAINING loss over the tree fit, not as the expected value of asking a
   question.** Ours scores each candidate question by actually folding users and ranking their held-out
   targets under the certified tower.
3. **Sampled-negative evaluation, not full-catalogue.** > "to simulate users' rank list, we randomly
   sample negative items 10 times as positive items for each user as test data." Ten sampled negatives
   per positive, against our full 18,359-item catalogue. Sampled-negative ranking metrics are known to
   distort method ordering (Krichene & Rendle, KDD 2020), so their numbers are not comparable to ours
   and their ordering is not evidence about ours.
4. **Metric is MAP, not NDCG** (MAP x16 in the text; NDCG/nDCG/DCG **zero** occurrences). RMSE is
   reported alongside for the rate-prediction comparison against fMF.

## What it does NOT threaten, and in fact supports

**Their static comparator is popularity.** > "**FixQ** It is short for 'fixed questions' which
represents the method that select **top-k popular items** as static interview questions, and then use
feature based MF to solve the problem by leverage the answers to interview as implicit feedback."

Other baselines: `Warm-MF` (train + answers combined) and `fMF` (Zhou et al., square loss only). So the
strongest static arm they beat is **top-k popular** — exactly the weak-static pattern our Act 1
identifies. Nothing here tests a static order built by maximising the reported metric, which is our
comparator. **Our Act-1 result is not pre-empted.**

## Consequence for the novelty claim

Revised wording. NOT available: "first ranking objective in an interview tree." Still available, and
what we should claim:

> a fixed question order selected by directly maximising **cohort-mean held-out NDCG@10** under a
> certified recommender, on a full catalogue with no negative sampling, selected on a disjoint
> validation cohort — and the finding that published adaptive selectors lose to it.

Cite Hu & Yu as the nearest prior attempt and state the four separations rather than leaving a reviewer
to find them. Note it is a 4-page short paper with figure-only results (MAP curves, no scalar tables).

## Bib

Not yet added to `new_chapters/references.bib`. **Check for key collision before adding** — a collision
once destroyed a 366-citation bibliography. Suggested key `hu2013interview`.
