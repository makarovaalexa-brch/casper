# Reddit leg 1 -- does a real request create divergence in the OPTIMAL NEXT QUESTION?

Run 2026-08-04 01:37:53 by `scripts/reddit/dispersion_leg1.py` (91.0 min, CPU, 4 torch threads).
Tower `models\i26c_ep31\t2i26c_ep31.pt` (frozen, read-only import), concept channel `.cache\instrument\cv3_ep07.pt`.
Ruler `data\reddit_he\ruler\ruler_v1.csv.gz` -- **all 4,431 threads**; ML-25M rows use **all 10,000
canonical VAL users**. No sampling anywhere (HARD RULE 1). No LLM call, no Reddit API.

## The criterion (M11: dispersion must be read off a LOSS-COUPLED criterion)

p(.|z)=softmax(z Wd^T+bd); V(z)=sum_{r<=10} disc_r p_(r)(.|z); U(q|z)=sum_a P(a|z) V(z (+) (q,a)); crit=U-V(z)

`p(.|z)` is the tower's own multinomial predictive -- the distribution its training loss optimises --
so "relevance" here is the model's belief, never held-out ground truth: the criterion is a function of
the STATE alone, which is what makes measurement B a genuine floor rather than a leak. `V(z)` is the
expected NDCG@10 of the slate the recommender would actually show at `z` (IDCG = 1 under a single-draw
relevance model, so DCG = NDCG). `U(q|z)` marginalises over the answer, so the criterion is
answer-dependent -- it is not a Sigma-only criterion whose dispersion is 0 by the theorem.

**Answer model** (behavioural, state-coupled, ground truth never read), L = 63
= the median rated-item count of a canonical VAL user:

* item -- P(seen)=1-(1-p_i)^L; LIKED folds at level 8, UNSEEN folds as the i26 exposure sentinel (level 10) -- both move the state
* concept -- P(answerable)=1-(1-m)^L-L m (1-m)^(L-1) with m=sum_{i in c} p_i; P(like)=lift/(1+lift), lift=m(z)/m(prior); refusal folds a refusal token

**Why this criterion and not one already in the tree.** repo audit: interview evals use static lit orders (interview_strategies.py); greedy machinery optimises a POPULATION mean over a shared sequence; the only per-state rule (adaptive_concept_arms 'greedy realizable') is |<d_c,z>|*std|v| = expected response magnitude, an information criterion, which the M11 rule forbids.
LIMITATION, printed rather than hidden: the target distribution is the model's own posterior, so this
measures the recommender's *self-consistent* expected gain, not gain against held-out items. That is
the price of a criterion that is a pure function of the state; a ground-truth-coupled version would be
clairvoyant and could not be computed at interview time.

## Banks (identical in all three measurements)

* **Concepts** -- 1031 genome concepts (genome relevance>=0.5, >=30 members), the full bank.
* **Items** -- 297 items, union of top-200 of popularity / entropy0 / helf. Covers the first 16 of every
  standard order: `{'popularity': True, 'entropy0': True, 'helf': True}`. This is a bank choice, not a
  per-user candidate cap: the criterion costs one encoder fold plus one full 18,359-way decode per
  (state, candidate, answer), so the full catalogue would be ~1.6e15 flops.

4,313 distinct states were scored (states are de-duplicated exactly; the
de-duplication itself is part of the finding -- see arm C).

## The comparison

**Concept bank (1031 questions)**

| state population | n | distinct q* | top-q* share | entropy (bits) | perplexity (eff. #questions) | top-10 cum. share |
|---|---|---|---|---|---|---|
| ML-25M cold, k=0 (10,000 VAL users) | 10,000 | 1 | 1.000 | 0.000 | 1.00 | 1.000 |
| ML-25M after 1 entropy0 question | 10,000 | 5 | 0.574 | 1.524 | 2.88 | 1.000 |
| ML-25M after 2 entropy0 questions | 10,000 | 9 | 0.407 | 2.332 | 5.04 | 1.000 |
| Reddit, request-placed (4,431 threads) | 4,431 | 442 | 0.141 | 6.572 | 95.13 | 0.441 |
| Reddit, >=1 OP_NAMED anchor | 2,639 | 413 | 0.107 | 7.005 | 128.46 | 0.365 |
| Reddit, zero anchors (SBERT concepts only) | 1,792 | 171 | 0.190 | 5.257 | 38.23 | 0.597 |

**Item bank (297 questions)**

| state population | n | distinct q* | top-q* share | entropy (bits) | perplexity (eff. #questions) | top-10 cum. share |
|---|---|---|---|---|---|---|
| ML-25M cold, k=0 (10,000 VAL users) | 10,000 | 1 | 1.000 | 0.000 | 1.00 | 1.000 |
| ML-25M after 1 entropy0 question | 10,000 | 5 | 0.574 | 1.582 | 2.99 | 1.000 |
| ML-25M after 2 entropy0 questions | 10,000 | 15 | 0.408 | 2.455 | 5.48 | 0.990 |
| Reddit, request-placed (4,431 threads) | 4,431 | 175 | 0.102 | 5.213 | 37.08 | 0.633 |
| Reddit, >=1 OP_NAMED anchor | 2,639 | 173 | 0.085 | 5.748 | 53.74 | 0.509 |
| Reddit, zero anchors (SBERT concepts only) | 1,792 | 44 | 0.179 | 3.629 | 12.37 | 0.896 |

**Unified bank (items + concepts compete in the same expected-NDCG units)**

| state population | n | distinct q* | top-q* share | entropy (bits) | perplexity (eff. #questions) | top-10 cum. share |
|---|---|---|---|---|---|---|
| ML-25M cold, k=0 (10,000 VAL users) | 10,000 | 1 | 1.000 | 0.000 | 1.00 | 1.000 |
| ML-25M after 1 entropy0 question | 10,000 | 5 | 0.574 | 1.524 | 2.88 | 1.000 |
| ML-25M after 2 entropy0 questions | 10,000 | 9 | 0.407 | 2.332 | 5.04 | 1.000 |
| Reddit, request-placed (4,431 threads) | 4,431 | 350 | 0.077 | 6.227 | 74.90 | 0.483 |
| Reddit, >=1 OP_NAMED anchor | 2,639 | 290 | 0.074 | 6.339 | 80.95 | 0.441 |
| Reddit, zero anchors (SBERT concepts only) | 1,792 | 131 | 0.138 | 5.027 | 32.61 | 0.671 |

Mean expected gain of the chosen question (`U(q*) - V(z)`, NDCG@10 units):

| state population | best concept | best item |
|---|---|---|
| ML-25M cold, k=0 (10,000 VAL users) | +0.02380 | +0.00193 |
| ML-25M after 1 entropy0 question | +0.02571 | +0.00340 |
| ML-25M after 2 entropy0 questions | +0.02830 | +0.00368 |
| Reddit, request-placed (4,431 threads) | +0.00762 | +0.01384 |
| Reddit, >=1 OP_NAMED anchor | +0.00818 | +0.01787 |
| Reddit, zero anchors (SBERT concepts only) | +0.00680 | +0.00791 |

## Controls

* **Shuffle** -- each thread reassigned a random other thread's folded state. Every dispersion
  statistic identical: **True** (it must be: the argmax multiset is a
  property of the state set, so a permutation cannot move it; a change here would mean the pipeline
  leaks thread identity into the criterion).
* **Duplicate-state census** -- 350 of
  4,431 threads share an exact state hash with at least one other thread;
  350 share both an exact state hash and an argmax.
* **Existential** -- two hand-built opposite states (top-8 popular `horror` likes vs top-8 popular
  `romance` likes): best concept `frightening` vs `sword fighting`
  (differ: True); best item sid 1835 vs 29
  (differ: True).
* **Determinism** -- 100 random threads recomputed from scratch return identical argmaxes:
  **True**.

## Verdict

**Yes -- and by roughly an order of magnitude.** At the ML-25M cold start every one of the 10,000 VAL
users has the same state, so the loss-optimal next question is one question for everybody: perplexity
1.00, top-question share 1.000. That is the trivial floor, and it is verified rather than assumed. The
standard openers barely move it: answering the first *and* second entropy0 question behaviourally
leaves the population sitting in **112 distinct states** (11 after one question) -- the bank asks the
same two items of everybody, so the only thing that can differ is the half-star band the user reports
or whether they had seen the film at all -- and the loss-optimal next question collapses to an
effective **5.0 distinct questions** over 10,000 users, with 41% of users sharing a single argmax. A
real Reddit request places its author in an essentially unique state (4,188 distinct states over 4,431
threads) and the loss-optimal next question spreads to an effective **95 concept questions** (top share
0.141, top-10 covering only 44% of threads) and **37 item questions**. Concept-bank perplexity is
**19x** the two-question-opener figure, item-bank perplexity **6.8x**. The split is informative: threads
with at least one named film are the most divergent (concept perplexity 128, 41% higher than the pooled
figure), while threads whose only evidence is SBERT-mapped request text still reach perplexity 38 --
above the openers by 7.6x, so the divergence is not merely "the OP listed some films".

Read against the M11 framing, this is the missing conditional — with a citation discipline the
write-up must keep (author ruling 2026-08-04): there is NO published field-wide adaptivity null to
cite, and OUR sub-MDE measurements are in-paper results, not citable background. The published
prior runs the other way — Golbandi 2011 and the tree line report adaptive GAINS, measured against
unoptimized static orders; whether those survive a strongest-static comparator is a question this
paper MEASURES (his tree is one of our consumers), not asserts. What this figure explains is our
own measured sub-MDE gaps, and it delimits rather than contradicts the tree line: those designs —
and ours — were measured downstream of openers that leave essentially no divergence to exploit: an
adaptive policy and a static one are choosing between at most a handful of distinguishable situations,
so there is nothing for adaptivity to be right about. Real requests do create that divergence for free,
before any question is asked. What this measurement does **not** show is that the divergence can be
cashed -- it is a property of the criterion surface, not a payoff. Leg 2 and the ML-25M
matched-dispersion probe are what decide that, and until they run the honest claim is "real requests
place users at a dispersion the standard banks never reach", never "adaptivity pays on Reddit".

Two limitations to carry forward. (1) The criterion scores against the tower's own predictive
distribution, so it measures self-consistent expected gain; a divergence in what the model *thinks*
is worth asking is necessary but not sufficient for a divergence in what actually pays. (2) The mean
expected gain of the best question is larger at the ML-25M states (+0.0257 NDCG@10 for the best concept
at k=2) than at the Reddit states (+0.0076) -- unsurprising, since a Reddit state already carries
anchors plus four folded concepts and is therefore further along the curve, but it means dispersion and
headroom move in opposite directions here and the two must not be conflated.
