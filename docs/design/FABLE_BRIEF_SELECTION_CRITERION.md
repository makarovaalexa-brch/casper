# Brief for Fable: is there a confidence/distribution measure better fitted to rank-targeted adaptive elicitation than EVOI?

## THE SETUP (self-contained; do not read the repo)

A frozen RecVAE-class tower exposes a latent $\mathbb{R}^{200}$. A user is a point $\mu$; **every queryable
thing is a direction in that same space**:

- an **item** $i$ → its decoder row $Q_i$ (score $= Q_i^\top\mu + \beta_i$)
- a **concept** (out-of-catalogue, e.g. a genome tag) → whitened member centroid $\varphi_c$
- an **arbitrary free-text phrase** → $W\cdot\mathrm{SBERT}(\text{text})$ via one globally-fitted linear adapter

One fold operator handles all three: an answer is an observation pair $(\varphi, y)$.

Over that latent sits an analytic conjugate-Gaussian belief, precision form:
$$\Lambda_t=\Lambda_0+\sum_j \alpha_j\varphi_j\varphi_j^\top,\qquad \Sigma_t=\Lambda_t^{-1}$$
with 8 fitted scalars ($\alpha$ per channel × confidence, prior scale, variance floor). Uncertainty along
ANY direction $w$ is $w^\top\Sigma w$ — so we can already ask "how unsure am I about this item / this
concept / this arbitrary phrase / this region?" with one operation.

## WHAT WE MEASURED

**The belief's structure is good.** Total posterior variance falls monotonically over answers
(1881.9 → 1805.1 across $q=0\ldots16$). Shrinkage is **11.5× larger along the queried direction** than
elsewhere, against a 2× bar. Order-invariant to $5\times10^{-7}$.

**Its calibration gate was garbage and we killed it.** Spearman(belief-$\sigma$, held-out NLL) $=0.239$ —
but a control showed *mean log-popularity of the user's held-out targets* predicts the same NLL at
$|\rho|=0.70$, and profile size at $0.52$. The gate was measuring how hard a user's targets happen to be,
not the covariance's quality. Dropped, not improved.

**Selection with it barely works.** $\Sigma$-greedy (ask where posterior variance is largest) beats an
isotropic $c\mathbf{I}$ control by $+0.0089$ full NDCG@10 at $q{=}16$ (CI-clean), so the covariance's
*shape* carries information — but beats **random question order by only $+0.0036$**.

## THE DIAGNOSIS (this is the crux)

$\Lambda_t$ depends only on **which** questions were asked, never on **what the user answered**. So
$\Sigma_t$ evolves identically for every user, and a max-variance selector emits the same sequence for
everyone: **it is a static policy in adaptive clothing.** That is exactly the linear-Gaussian +
variance-only static-optimality result (Krause–Guestrin ICML 2007; Jedynak–Frazier–Sznitman 2012). We built
the one selector the theorem says cannot win, and got the predicted null.

**Two escapes, both available to us:**
1. **Answerability makes $\Sigma$ answer-dependent after all.** Refusals carry $\alpha_{\text{refuse}}$,
   real answers $\alpha_{\text{answer}}$, fitted ratio **0.0007**. So belief states genuinely diverge across
   users according to *whether they could answer* — outside the theorem's setup. Measured answer rates:
   items **2.18 of 8** (27%), concepts **6.02 of 8** (75%).
2. **Couple the objective to the ranking.** NDCG depends on $\mu$, and $\mu$ *does* depend on answers.

## THE GEOMETRY WE WANT TO EXPLOIT (the coarse-to-fine mechanism)

The update is rank-1, so by Sherman–Morrison the variance along any other direction $w$ falls by
$$\frac{\alpha\,(w^\top\Sigma\varphi)^2}{1+\alpha\,\varphi^\top\Sigma\varphi}$$
i.e. **proportional to the squared covariance between $w$ and the queried direction**. One question
tightens a whole *correlated region*, not just the queried axis. A concept direction sits near many item
directions by construction, so it collapses a large region per question; an idiosyncratic item collapses
almost nothing else. **That is a geometric account of coarse-to-fine**, and it is what we want a selector
to exploit.

**Measured, consistent with it:** a greedy best-static sequence over a mixed item+concept bank (built on
10k val users, evaluated on 10k disjoint test users, pure NDCG objective, nothing encoding "broad first")
emits `c c c c i i i i c i i i i i i i` — four concepts, then items. Concepts lead full NDCG through $q{=}4$
and dominate tail at every budget; items win full from $q{=}8$. Intercept 0.1279/0.0192; items-only reaches
0.2098/0.0679 @q16, concepts-only 0.1808/0.0758, combined 0.2049/0.0776.

## WHAT WE WANT TO BUILD

An **adaptive, coarse-to-fine question selector** that is (a) driven by the belief, (b) targeted at NDCG
rather than at generic uncertainty, (c) genuinely adaptive rather than static-equivalent, and (d)
channel-agnostic — items, concepts and arbitrary embeddings compete in one action space.

**Our candidate criterion**, exploiting the geometry above:
$$\text{score}(\varphi)=\sum_{i\in\text{top-}K(\mu)}\frac{\alpha\,(Q_i^\top\Sigma\varphi)^2}{1+\alpha\,\varphi^\top\Sigma\varphi}$$
"Ask the question that collapses variance in the directions that currently decide my ranking." It is
adaptive because $\text{top-}K(\mu)$ moves with the answers, even though the coupling term does not.

## WHAT WE KNOW ABOUT THE OBVIOUS ALTERNATIVE

Bıyık et al. 2023 (arXiv:2311.02085) select queries by **EVOI**. We verified from their full text that
their EVOI is **expected utility, not rank-discounted** — so it is not the same object as an NDCG-targeted
criterion. Their belief is a sampled/HMC posterior, not closed-form; CAV directions are per-concept
supervised probes over 164 tags; they report no full-catalogue accuracy (their NDCG is in-simulator top-|S|
agreement over 16 users on slates of 5).

## THE QUESTIONS FOR YOU

1. **Is there a confidence / distribution measure better fitted to this purpose than EVOI?** We want
   something that (i) targets a *ranking* objective, not expected utility or generic entropy;
   (ii) is computable in closed form over a Gaussian belief in a 200-d latent, for arbitrary query
   directions; (iii) does not collapse to static-optimal under our own linear-Gaussian machinery.
   Candidates we are aware of but have not evaluated: rank-aware / decision-focused information gain,
   top-$K$ identification and best-arm-identification criteria, Bayesian experimental design with a
   decision loss, knowledge-gradient, Thompson-sampling-based acquisition, D/A/E-optimal design variants.
2. **Does our candidate criterion already have a name?** If it is a known object (or a special case of
   one) we want to cite it, not christen it.
3. **Where, if anywhere, is the contribution here?** Is "a rank-targeted acquisition function over a
   closed-form Gaussian belief in a frozen CF latent, with items/concepts/text competing in one action
   space" a real contribution, a known thing, or an engineering detail? Be blunt.
4. **What is the failure mode?** In particular: does top-$K(\mu)$ weighting actually escape
   static-optimality in a meaningful way, or does it merely add noise? Is there a degenerate solution
   (e.g. always ask about the current #1) we should expect?
5. Should this live in the instrument paper (A) or the policy paper (B)? A owns the recommender that can
   be interviewed; B owns question-selection policies.

Return a typed verdict, concrete and specific. Name papers where they exist.
