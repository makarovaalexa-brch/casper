# Findings: Rank-Targeted Acquisition (top-K-weighted variance reduction)

*Is our "collapse the variance in the directions that decide my ranking" criterion novel? Best-fitted published alternative to EVOI. The static-optimality escape.*
Deep lit sweep 2026-07-28. Verification levels: **[V]** verified from primary text (abstract or body I read), **[S]** secondary/abstract mention or search-snippet quote, **[A]** my inference/derivation.

---

## What we concluded

1. **The FORM of our criterion is fully pre-empted — it has a name, and the name is not ours.** Write it out:

   ```
   score(phi) = sum_{i in topK(mu_t)}  alpha (Q_i^T Sigma_t phi)^2 / (1 + alpha phi^T Sigma_t phi)
              = sum_{i in topK}  [ Q_i^T Sigma_t Q_i  -  Q_i^T Sigma_{t+1} Q_i ]
              = tr(P Sigma_t P^T) - tr(P Sigma_{t+1} P^T),     P = rows {Q_i : i in topK(mu_t)}
   ```
   That is **exactly the one-step greedy step of Bayesian L-optimality (linear/goal-oriented A-optimality)**, i.e. minimise the trace of the posterior covariance *mapped through a goal operator P*, not of Sigma itself. K=1 is **Bayesian c-optimality** verbatim. Named in the inverse-problems literature as **A-GOODE** (Attia, Alexanderian & Saibaba 2018) and in ML as **transductive experimental design** (Yu, Bi & Tresp, ICML 2006), where the "goal set" is the test pool. **[A]** for the algebraic identity (Sherman–Morrison, one line); **[V]** for A-GOODE being "minimise posterior uncertainty in the end-goal QoI rather than the parameter"; **[V]** for the linearised goal criterion being *literally* "equivalent to Bayesian c-optimality" (stated in Neuberger–Alexanderian–van Bloemen Waanders 2024, arXiv:2411.07532, on G_l-optimality). **Do not present the formula as a new acquisition function. It is a textbook design criterion with a substituted goal operator.**

2. **The one genuinely uncontested wedge is that our goal operator is POSTERIOR-DEPENDENT: P = P(mu_t).** Every goal-oriented OED paper found fixes the QoI / goal operator in advance and expands at a fixed point m̄. The 2024 quadratic-approximation paper's related-work section is explicitly about *fixed* goal functionals with a *fixed* expansion point, and does not discuss posterior-mean-dependent goals or sequential goal-oriented design **[V]** (fetched, related-work summary). More decisively, the sequential-OED literature states the gap in as many words: *"Existing goal-oriented efforts all focused on batch (i.e., non-sequential) or greedy OED, and integrating goal-oriented objectives into sOED has yet to be developed"* — Shen, Dong & Huan, *Variational Sequential Optimal Experimental Design using RL*, arXiv:2306.10430 (CMAME 2025) **[S]** (quote returned verbatim by search against the paper; PDF text extraction failed, re-verify before quoting in the thesis). So: **goal-oriented + adaptive is an acknowledged open cell as of 2023–2025.** That is the claim to stake — not the formula.

3. **The static-optimality escape is real but is a hypothesis-failure argument, not a theorem.** The linear-Gaussian null is a statement about the *objective class*, not the belief class. Jedynak–Frazier–Sznitman's theorem is: minimise expected posterior *entropy*, greedy one-step is full-horizon optimal, and one of the two Bayes-optimal policies **"asks a deterministic set of questions"** — the question sequence can be fixed before any answer arrives **[V]** (abstract, verbatim). Krause–Guestrin's line is the same for GP variance-reduction objectives **[S]**. Both hypotheses require the objective to be a functional of Sigma_t alone. Our objective is `J_t = tr(P(mu_t) Sigma_t P(mu_t)^T)` — a functional of **(Sigma_t, mu_t)**. Sigma_t remains answer-independent, but J_t is not, so the theorems' antecedent simply does not hold and no static-equivalence follows. **[A]** — but see the warnings in §Open questions: nothing here *guarantees* a positive adaptivity gap, and our measured +0.0036 is exactly what a degenerate P(mu_t) (near-identical top-K across users) would produce.

4. **The structural precedent that this kind of mean-dependence really does buy adaptivity is Bayesian ranking & selection.** Knowledge Gradient (Frazier–Powell–Dayanik 2009) and OCBA are computed over a Gaussian belief whose covariance update is *also* answer-independent (Gaussian, known sampling variance), yet their acquisitions depend on the posterior *means* — and they are decisively better than variance-only / equal allocation, and are not static policies **[S]**. This is the closest thing to an existence proof that "decision-dependent weighting escapes the null." It is also the alternative criterion we should actually run (see §2 of the verdict).

5. **Nobody in recsys/CRS has targeted a rank-discounted metric through a Bayesian posterior.** The elicitation literature uses: popularity/entropy/HELF heuristics; squared-loss tree splits (Golbandi); information gain / entropy (EDDI, most CRS); expected utility / EVOI (Boutilier line, Biyik 2023); bandit upper-confidence bounds (interactive CF). The single paper that names NDCG as the selection target — Lin, Zhu, Wang & Caverlee, *Enhancing User Personalization in Conversational Recommenders*, WWW 2023, "Algorithm 2: Greedy NDCG Attribute Selector" — computes a **deterministic NDCG difference from the current point estimate, with no posterior, no variance, and no expectation over answers** **[S]** (algorithm description confirmed via two independent search summaries; direct fetch 403'd — verify before the thesis claims "no posterior"). So the empty cell is precise: *posterior-variance-weighted, rank-discounted acquisition over a frozen CF latent.* This is consistent with, and sharpens, the Claim-5 wording already in `elicitation_and_belief_pool.md`.

---

## Key external methods

### The criterion's true lineage (cite these, do not claim them)
- **Attia, Alexanderian & Saibaba (2018), "Goal-oriented optimal design of experiments for large-scale Bayesian linear inverse problems", *Inverse Problems* 34(9); arXiv:1802.06517.** A-GOODE / D-GOODE = classical Bayesian A-/D-optimality with the posterior covariance mapped through a goal operator. Abstract verbatim: *"we seek experimental designs that minimize the posterior uncertainty in the experiment end-goal, e.g., a quantity of interest (QoI), rather than the estimated parameter itself"* **[V]**. Also notes the criterion is *cheaper* than classical OED because the QoI is low-dimensional — which is our K=10 vs d=200 situation exactly. **This is the correct citation for our formula.**
- **Neuberger, Alexanderian & van Bloemen Waanders (2024), arXiv:2411.07532, *J. Sci. Comput.* (2025).** Extends GOODE to nonlinear goal functionals via a quadratic approximation (G_q-optimality); states the linearised criterion `Psi^l = <C_post g, g>` is **"equivalent to Bayesian c-optimality"** **[V]**. Useful if we ever make the goal *soft* top-K (a smooth NDCG surrogate) rather than a hard set — that is precisely their G_q setting.
- **Yu, Bi & Tresp (2006), "Active learning via transductive experimental design", ICML.** The ML-side name: select points minimising predictive variance summed over a *target/test* set rather than over the parameter **[S]** (abstract + secondary summaries; PDF fetch failed). Our top-K set is a transductive target set that moves. Cheap and very citable in an ML venue.
- **Hübotter et al. (2024), "Transductive Active Learning: Theory and Applications", NeurIPS.** Modern re-statement/theory of target-set-directed acquisition **[S]**. Worth a one-line cite for currency.
- **Mussmann (2026), arXiv:2607.06642, "The Approximation Ratio for the Risk of Myopic Bayesian Active Learning for Linear Regression".** First approximation-ratio bound for the *greedy* OED risk in linear regression; bound is tight up to a constant and scales with a "maximum initial leverage score" **[V]** (abstract). This is the right citation for "our greedy is provably not far from the optimal non-adaptive design" — and it is a *warning*: in the plain linear-regression risk setting the greedy design is near-optimal, which is another way of saying there is little room for adaptivity there.

### Decision-focused / task-driven BED (the modern framing we sit inside)
- **Huang, Bickford Smith & Rainforth (2026), "Loss-Driven Bayesian Active Learning", arXiv:2604.11995.** *"any loss function can derive a unique objective for optimal data acquisition"*; losses expressible as **weighted Bregman divergences** admit analytic components **[V]** (abstract). Our criterion is an instance: squared-error Bregman loss with weights = 1 on the top-K item scores, 0 elsewhere. **We must cite this or a reviewer will say we reinvented their framework's special case.**
- **Rossa, Phillips & Rainforth (2026), "Action-BED: Task-Driven Bayesian Experimental Design with Singly Intractable Objectives", arXiv:2606.23662.** Reformulates BED as **expected future loss** of the downstream action, jointly optimising design policy and action policy; avoids explicit posterior estimation **[V]** (abstract). Same family; more general, more expensive, no closed form.
- **Huang et al. (2024), "Amortized Bayesian Experimental Design for Decision-Making", NeurIPS** **[S]** — amortised decision-utility-aware BED; explicitly motivated by "reducing parameter uncertainty does not necessarily improve downstream decisions."
- **GoBOED (2026), arXiv:2605.26093** — goal-driven BOED with a differentiable convex decision layer **[S]**. Background only.
- Framing quote worth reusing: *"in decision-critical settings, reducing parameter uncertainty does not necessarily improve downstream decisions, as only specific parameter directions relevant to the objective truly matter"* **[S]** (arXiv:2605.26093). That is our whole thesis in one sentence, already published — so it is *motivation*, not contribution.

### Genuinely adaptive closed-form Gaussian acquisitions (the real alternatives to EVOI)
- **Frazier, Powell & Dayanik (2009), "The knowledge-gradient policy for correlated normal beliefs", *INFORMS J. on Computing* 21(4):599–613.** Closed-form one-step value of a measurement under a **multivariate normal belief with correlations** — exactly our Sigma. KG values the improvement in `max_i mu_i` under the posterior predictive; the covariance update is answer-independent but the KG *value* is a function of the posterior means, so the policy is genuinely answer-dependent **[S]** for the paper, **[A]** for the structural point. Cost: for n alternatives, the standard KG computation is an O(n log n) sort of the (a_i, b_i) affine lines per candidate measurement.
- **Frazier & Powell, hierarchical KG (JMLR 12:2931–2974, 2011)** and **KG for pairwise/top-κ subset selection** (Bayesian R&S with pairwise comparisons) **[S]** — the top-K generalisation exists.
- **OCBA / OCBA-m (Chen et al.)** — asymptotic budget allocation maximising probability of correct selection of the top-m; *"explicit allocation of computational budget based on variance and mean-gap structure"* **[S]**. Mean-gap dependence = adaptivity. The `1/(gap^2/variance)` structure is the natural sanity baseline for a top-K interview.
- **"Ask the Right Comparison: Bias-Aware Bayesian Active Top-k Ranking with LLM Judges", arXiv:2607.02104 (2026).** Concurrent and uncomfortably close in *spirit*. Acquisition: `s(i,j) = p̂_ij(1-p̂_ij) · Var(θ_i - θ_j) · (H(p_i)+H(p_j))` over a Laplace-Gaussian posterior, explicitly targeting **top-k membership** rather than global ranking precision **[V]** (fetched HTML). Not our criterion (Bradley–Terry pairwise comparisons; a product of three heuristic factors; no NDCG discount; no closed-form variance-reduction), but it is a 2026 "top-k-aware Bayesian design" paper and must be cited as concurrent work. Cites Houlsby et al. 2011 (BALD) as its foundation.
- **"Targeted Variance Reduction" (Miao & Mak, arXiv:2403.03816)** — a *named* acquisition function in robust BO, `TVR`. Name collision only; it targets the objective-improvement region, not a ranking. Note it so we do not accidentally reuse the name.

### Preference elicitation / decision-theoretic (what the field actually uses)
- **Chajewska, Koller & Parr (AAAI 2000), "Making rational decisions using adaptive utility elicitation."** Prior over the utility function; at each step ask the question with the **highest value of information**, then recompute the best strategy; explicitly *"interleaves the analysis of the decision problem and utility elicitation to allow these two tasks to inform each other"* **[S]**. **This is the conceptual ancestor of "let the current decision decide the next question."** Our top-K(mu) weighting is the linear-Gaussian, ranking-metric instance of exactly this idea. Cite it prominently and honestly.
- **Viappiani & Boutilier (NeurIPS 2010), "Optimal Bayesian recommendation sets and myopically optimal choice query sets"** + **(RecSys 2009) regret-based version** + **(AIJ 2020) "On the equivalence of optimal recommendation sets and myopically optimal query sets"** + **(AAAI 2020) "Recommendation Sets and Choice Queries: There Is No Exploration/Exploitation Tradeoff!"**. Key result: **the optimal recommendation set coincides with the myopically optimal query set**, under both Bayesian and regret-based elicitation **[S]**. This is a *threat and a gift*: it is the published, formal statement that "ask about what you would currently recommend" is optimal — i.e. it independently justifies conditioning the query on top-K(mu). Their objective is **expected max utility (EVOI)**, with **no rank discount and no ranking metric**. Our criterion is not their result, but a reviewer who knows this line will ask why we do not just use the recommendation set as the query set. **Have an answer.**
- **Boutilier's EVOI/PEU definition**: `PEU(q) = sum_r P(r|q) EU(post_r)`, `EVOI = PEU - EU*` **[S]**. Confirms (again) that EVOI is expected *utility*, matching our earlier ruling on Biyik 2023.
- **Vendrov, Lu, Huang & Boutilier (2020), "Gradient-based optimization for Bayesian preference elicitation", AAAI; arXiv:1911.09153** — already in our bib; EVOI-family, linear utility, no rank discount **[S]** (PDF fetch failed; do not over-claim details).
- **Lin, Zhu, Wang & Caverlee (WWW 2023), "Enhancing User Personalization in Conversational Recommenders"** — Algorithm 2, greedy NDCG attribute selector: picks the attribute with the highest expected NDCG increase, computed as an NDCG difference from the current user representation, **no posterior / no variance** **[S]**. The nearest recsys neighbour; the mandatory delta-citation.
- **Elahi, Ricci & Rubens (2016), "A survey of active learning in collaborative filtering recommender systems", *Computer Science Review* 20:29–50** **[S]** — the AL-in-recsys map; criteria taxonomy (personalised vs not, single vs multi-criterion). Confirms nothing in the survey targets a rank-discounted metric through a posterior.
- **Rubens & Sugiyama (RecSys 2007), "Influence-based collaborative active learning."** Selects items by the **influence of a rating on the predictions of other items** **[S]**. This is the closest *pre-existing recsys* relative of our criterion: `(Q_i^T Sigma phi)` *is* an influence term. Difference: theirs is un-Bayesian and un-targeted (influence over *all* items, not the current top-K). Must cite; it makes the "targeted at the top-K" the load-bearing delta.
- **Jin & Si (UAI 2004), "A Bayesian approach toward active learning for collaborative filtering", arXiv:1207.4146.** Expected loss over the **posterior distribution of the model**, not the point estimate; beats point-estimate methods when ratings are scarce **[V]** (abstract). Prediction (RMSE-type) loss, not ranking. Cite as the Bayesian-AL-for-CF anchor.
- **Sutherland, Póczos & Schneider (KDD 2013), "Active learning and search on low-rank matrices"** **[S]** — posterior-variance querying in an MF latent space; the "highest posterior variance element" baseline that we already know is static-equivalent.
- **Karimi, Freudenthaler, Nanopoulos & Schmidt-Thieme (RecSys 2012 / ICTAI 2011)** — MF-specific AL, incl. explicitly **non-myopic** variants and "optimal AL selects queries that directly optimize expected error for test data" **[S]**. Their "expected error on test data" is transductive design in a CF space — check before claiming that framing is new.

### The static-optimality / adaptivity-gap theory
- **Jedynak, Frazier & Sznitman (2012), *J. Applied Probability* 49(1):114–136.** Abstract verbatim: *"we seek to minimize the expected entropy of the posterior distribution... we show that any policy optimizing the one-step expected reduction in entropy is also optimal over the full horizon. Two such Bayes optimal policies are presented: one generalizes the probabilistic bisection policy due to Horstein and the other asks a deterministic set of questions."* **[V]**. Scope: **entropy loss**. Our objective is not entropy.
- **Krause & Guestrin (ICML 2007) / Krause, Singh & Guestrin (JMLR 2008)** — variance/mutual-information observation selection in GPs; near-optimal greedy, and near-optimality of *a priori* (non-adaptive) designs when the adaptivity gap is small **[S]**. Their nonmyopic-AL paper *"proved bounds on how much better a sequential algorithm can perform than an a priori design"* and localises the improvement potential in **parameter entropy** (i.e. unknown hyperparameters), motivating explore-then-exploit **[S]**. **Important corollary for us: in their framing, a fully-known linear-Gaussian model has essentially no adaptivity potential — all the potential lives in what the covariance itself does not know.** Our escape must therefore come from the objective, not from the belief.
- **Golovin & Krause (JAIR 42:427–486, 2011), "Adaptive submodularity."** Adaptive greedy is (1−1/e)-competitive **with the optimal adaptive policy** for monotone adaptive-submodular f **[S]**. **Trap:** this bounds greedy-vs-optimal-adaptive; it says *nothing* about adaptive-vs-static. Do not cite it as evidence that adaptivity pays.
- **"Adaptivity in Adaptive Submodularity" (Esfandiari, Karbasi, Mirrokni; arXiv:1911.03620, COLT 2021)** — semi-adaptive policies achieve `1−1/e−eps` with `O(log n log k)` adaptive rounds, and it is **impossible** with `o(log n)` rounds **[S]**. i.e. the *value* of adaptivity is provably concentrated in a logarithmic number of rounds. With an 8–16 question budget and `log(18430) ≈ 9.8`, this is a quantitatively relevant statement about how many *genuinely adaptive* decisions can matter. Nice framing for the interview-length curve.
- **Batch-vs-sequential OED framing** (sOED literature): batch design *"is equivalent to the sOED formulation but restricting all policy functions to be only constant functions"*, hence the optimal adaptive policy weakly dominates **[S]**. Useful, but note it is a *weak* dominance — it does not say the gap is positive.
- **Adaptivity-gap literature** (stochastic probing, submodular/XOS) **[S]** — constant-factor gaps for combinatorial stochastic problems. Not directly applicable: our uncertainty is in a continuous Gaussian, not in item-realisation randomness.

---

## What is NOVEL vs pre-empted

**PRE-EMPTED — do not claim, cite:**
- The **formula**. `sum_i (c_i^T Sigma phi)^2 / (1 + alpha phi^T Sigma phi)` = greedy Bayesian L-optimality / A-GOODE / Sherman–Morrison c-optimality. **[A]+[V]** Attia et al. 2018; Yu, Bi & Tresp 2006. Calling this a "new acquisition function" is the single most likely way to get desk-rejected by a design-of-experiments-literate reviewer.
- **"Design for the decision, not the parameter."** Owned four times over: Chajewska–Koller–Parr 2000; goal-oriented OED; Loss-Driven BAL 2026; Action-BED 2026. Motivation only.
- **"Condition the query on the current recommendation set."** Viappiani & Boutilier prove the *optimal recommendation set = the myopically optimal query set*. Our top-K(mu) conditioning is a version of a published optimality result, not an invention. **[S]**
- **Top-k-aware Bayesian query selection.** arXiv:2607.02104 (2026), Bayesian R&S top-m (OCBA-m, KG top-κ). **[V]/[S]**
- **Influence-of-a-rating-on-other-predictions as a selection signal.** Rubens & Sugiyama 2007. **[S]**
- **"NDCG as the selection target in a CRS."** Lin et al. WWW 2023. **[S]** Our delta is *the posterior*, not *the metric*.

**NOVEL / defensible (narrow, and in this order of strength):**
1. **Adaptive (posterior-mean-dependent) goal operator in goal-oriented design.** All goal-oriented OED fixes the QoI; sequential goal-oriented OED is stated as undeveloped as of 2025 **[S]** (Shen–Dong–Huan). Our `P = P(mu_t)` makes the goal itself a random variable measurable w.r.t. the answer history. Frame the contribution as: *"a decision-dependent goal operator turns a provably static design criterion into a genuinely adaptive one, at zero extra cost."* **[A]**
2. **Rank-discount inside the goal operator.** Nothing found weights the goal directions by a positional discount. If we weight `Q_i` by `1/log2(1+rank_i)` (or by the NDCG swap-sensitivity `|1/log2(1+i) - 1/log2(1+j)|` over the boundary pairs), the criterion becomes a **rank-discounted L-optimal design** — an empty cell across OED, BO, R&S, and recsys. This is a cheap and real increment over both A-GOODE and Lin et al. **[A]**
3. **The recsys instantiation:** closed-form, in a *frozen* CF latent, over a heterogeneous query set (items, out-of-catalogue concepts, free text) that are all directions in the same space. Commensurability across query types via one closed-form number.
4. (**Weak, contested**) The empirical adaptivity-gap measurement itself — see C2 in `elicitation_and_belief_pool.md`.

**Explicitly NOT novel and NOT true as stated:** "we escape the linear-Gaussian null." We *fail its hypothesis*. That is a different, weaker, and honest sentence. Write it the weak way.

---

## The static-optimality question, stated precisely

**The null.** Let `f(S)` be the objective. If `f` is a functional of `Sigma_t` alone (entropy, A-/D-optimality, mutual information, max-variance), then because in the linear-Gaussian model `Lambda_t = Lambda_0 + sum_j alpha_j phi_j phi_j^T` depends only on *which* directions were queried, the whole objective trajectory is deterministic given the query sequence. The optimal policy is therefore realisable as a fixed sequence — Jedynak–Frazier–Sznitman's "deterministic set of questions" **[V]**; Krause–Guestrin's a-priori design **[S]**. Our measured +0.0036 for Sigma-greedy over random order is the predicted null, and it is *not evidence that adaptivity does not pay* — it is evidence that we ran an objective inside the theorem's hypothesis.

**The escape.** `J_t = tr(P(mu_t) Sigma_t P(mu_t)^T)` is a functional of the *full* belief `(mu_t, Sigma_t)`. `mu_t` is a martingale driven by the answers. Therefore `argmax_phi score(phi)` is a non-constant function of the answer history and no static sequence realises the greedy policy. The theorems do not apply. **[A]**

**What the escape does NOT buy — read this before writing any claim:**
- **Non-constancy is not a positive adaptivity gap.** A policy can be answer-dependent and still be beaten (or merely tied) by the *best* static sequence in expectation. The load-bearing experiment is therefore **our adaptive policy vs. the best static sequence found by the same greedy criterion run on the prior mean** — not vs. random order. No published theorem does this work for us.
- **The degeneracy failure mode is measurable and is probably what bit us.** If `topK(mu_t)` is nearly the same set for every user at every t (popularity domination — a documented property of our arena), then `P(mu_t) ≈ P̄` is effectively constant, `J_t` collapses back into the Sigma-only class, and the criterion is static-equivalent *in practice*. **Diagnostic to run before anything else: the across-user Jaccard/rank-correlation of `topK(mu_t)` at each turn t, and the induced dispersion of the argmax question.** If the argmax question is the same for >90% of users at t=1, the criterion has not escaped anything. This mirrors the C3 finding already on record (info-ordering → Spearman 0.795 → one list → static wins; task-ordering → Spearman 0.062 → adaptive wins). **The predictor of the adaptivity gap is the dispersion of the goal operator, not the form of the criterion.** That is a publishable, falsifiable statement and is probably the best theoretical sentence in this whole file.
- **Krause–Guestrin's diagnosis cuts against us**: they localise adaptivity potential in *parameter/hyperparameter entropy*. In a fully-specified linear-Gaussian model there is little to adapt to. Our answer must be that the *objective's* dependence on `mu_t` supplies the missing stochasticity — and we should say so explicitly, because a theory-literate reviewer will raise exactly this.
- **Adaptive submodularity does not help here.** `J_t` with a hard top-K is not obviously monotone or adaptive-submodular (the top-K set changes discontinuously), and even if it were, Golovin–Krause bounds greedy against the optimal *adaptive* policy, not against static. A soft/rank-discounted `P` would at least restore continuity; whether it restores adaptive submodularity is open.
- **Budget context:** `O(log n log k)` adaptive rounds suffice for near-optimal adaptive submodular maximisation and `o(log n)` provably does not **[S]**; `log2(18430) ≈ 14.2`. Our 8–16 question budget sits exactly on that boundary — a defensible reason to report the *interview-length curve*, and a warning that gains may only appear late.

---

## Answers to the four questions (short form)

**Q1 — already published?** The criterion: **yes, as a formula** (greedy Bayesian L-/c-optimal = A-GOODE = transductive experimental design). **No, as a policy**, because the goal operator is posterior-dependent and sequential goal-oriented design is an acknowledged gap. Nearest exact matches: Attia–Alexanderian–Saibaba 2018 (identical algebra, fixed goal); Yu–Bi–Tresp 2006 (identical algebra, test-set goal); Rubens–Sugiyama 2007 (same influence term, untargeted, un-Bayesian); Lin et al. WWW 2023 (same ranking target, no posterior); arXiv:2607.02104 2026 (top-k-aware Bayesian, different model/criterion).

**Q2 — best-fitted alternative to EVOI.** **Knowledge Gradient for correlated normal beliefs** (Frazier, Powell & Dayanik, *INFORMS J. Comp.* 21(4):599–613, 2009), in its top-K / rank-discounted form. Criterion: `KG(phi) = E[ V(mu_{t+1}) | phi ] - V(mu_t)` where `V(mu) = sum_{i=1..K} disc(i) * mu_{(i)}` (DCG of the induced ranking; standard KG is `V = max_i mu_i`). Under the rank-1 Gaussian update, `mu_{t+1} = mu_t + sigmã(phi) Z` with `Z ~ N(0,1)` and `sigmã(phi) = Sigma_t phi / sqrt(1/alpha + phi^T Sigma_t phi)`, so item scores are **affine in a single scalar Z**: `Q_i^T mu_{t+1} = a_i + b_i Z`. The expectation is then the classical piecewise-linear/epigraph computation — exact, closed-form, `O(n log n)` per candidate direction after one `O(d^2)` matvec. It is (a) closed-form over a Gaussian belief, (b) works for arbitrary query directions, (c) **provably decision-dependent hence not static**, (d) directly extensible to rank-discounted `V`, and (e) it is the *expected change in the realised ranking value*, which is much closer to NDCG than EVOI's expected utility. **This is what we should run against our Sherman–Morrison criterion.** Secondary/cheap alternative: OCBA-m-style gap-over-variance allocation on the top-K boundary. Framework-level alternative if we want the general machine: Loss-Driven BAL (Huang, Bickford Smith & Rainforth 2026), of which our criterion is the weighted-squared-error instance.

**Q3 — static-optimality escape.** The null's hypothesis fails (objective depends on `mu_t`), so the theorems are silent — but no published result promises a positive gap, and the gap is governed by the **across-user dispersion of `topK(mu_t)`**, which is a quantity we can and must measure. See §above.

**Q4 — what CRS elicitation actually uses.** Popularity/entropy/HELF heuristics; squared-loss tree splits; information gain/entropy (EDDI and most CRS); expected utility/EVOI (Boutilier line, Biyik 2023); bandit UCB/Thompson (interactive CF). Rank-discounted targeting: **only Lin et al. WWW 2023, and without a posterior**. Confirmed empty cell.

---

## Open questions / to verify before citing

- **[HIGH] Re-verify the Shen–Dong–Huan quote** (*"integrating goal-oriented objectives into sOED has yet to be developed"*) from the actual PDF/published CMAME version. It is the load-bearing novelty quote and I only have it via search snippet **[S]**.
- **[HIGH] Get Lin et al. WWW 2023 Algorithm 2 verbatim** (ACM 403'd; try arXiv:2302.06656 source or the authors' page). If their NDCG delta turns out to be an *expectation over a posterior*, our Claim-5 delta shrinks to "closed-form Gaussian + rank discount".
- **[HIGH] Check Karimi et al. (ICTAI 2011 / RecSys 2012)** for an explicit "minimise expected error on the target/test items" MF criterion. If they already do transductive design in a CF latent, our item (3) shrinks to the rank discount + heterogeneous query set.
- **[MED] Read Viappiani & Boutilier AIJ 2020 properly.** If the "optimal recommendation set = optimal query set" equivalence extends to ranking losses, it may *pre-empt* our top-K conditioning as a design principle. Currently our strongest justification and our strongest threat, simultaneously.
- **[MED] Search specifically for "rank-discounted" / "DCG-weighted" design criteria** in the R&S and BO literature (I found none, but absence-of-evidence at this depth is [A] not [V]).
- **[MED] Sequential/adaptive TED**: is there a paper doing transductive experimental design where the target set is re-selected each round from the current model? That would be the closest possible pre-emption of item (1).
- **[LOW] Verify Krause & Guestrin's adaptivity-gap bound statement** from the primary text before quoting the "potential lives in parameter entropy" reading.

## Bibliography additions required
```
attia2018goode      Attia, Alexanderian, Saibaba. Goal-oriented optimal design of experiments for
                    large-scale Bayesian linear inverse problems. Inverse Problems 34(9), 2018.
yu2006ted           Yu, Bi, Tresp. Active learning via transductive experimental design. ICML 2006.
frazier2009correlatedkg  Frazier, Powell, Dayanik. The knowledge-gradient policy for correlated
                    normal beliefs. INFORMS J. on Computing 21(4):599-613, 2009.
chajewska2000adaptive  Chajewska, Koller, Parr. Making rational decisions using adaptive utility
                    elicitation. AAAI 2000.
viappiani2010optimal  Viappiani, Boutilier. Optimal Bayesian recommendation sets and myopically
                    optimal choice query sets. NeurIPS 2010.
viappiani2020equivalence  Viappiani, Boutilier. On the equivalence of optimal recommendation sets
                    and myopically optimal query sets. Artificial Intelligence, 2020.
lin2023enhancing    Lin, Zhu, Wang, Caverlee. Enhancing user personalization in conversational
                    recommenders. WWW 2023.  [greedy NDCG attribute selector]
rubens2007influence Rubens, Sugiyama. Influence-based collaborative active learning. RecSys 2007.
jin2004bayesianal   Jin, Si. A Bayesian approach toward active learning for collaborative
                    filtering. UAI 2004.
elahi2016survey     Elahi, Ricci, Rubens. A survey of active learning in collaborative filtering
                    recommender systems. Computer Science Review 20:29-50, 2016.
huang2026lossdriven Huang, Bickford Smith, Rainforth. Loss-driven Bayesian active learning.
                    arXiv:2604.11995, 2026.
rossa2026actionbed  Rossa, Phillips, Rainforth. Action-BED: task-driven Bayesian experimental
                    design with singly intractable objectives. arXiv:2606.23662, 2026.
shen2025vsoed       Shen, Dong, Huan. Variational sequential optimal experimental design using
                    reinforcement learning. CMAME 2025 / arXiv:2306.10430.
mussmann2026myopic  Mussmann. The approximation ratio for the risk of myopic Bayesian active
                    learning for linear regression. arXiv:2607.06642, 2026.
golovin2011adaptive Golovin, Krause. Adaptive submodularity. JAIR 42:427-486, 2011.
esfandiari2021adaptivity  Esfandiari, Karbasi, Mirrokni. Adaptivity in adaptive submodularity.
                    COLT 2021 / arXiv:1911.03620.
neuberger2024goode  Neuberger, Alexanderian, van Bloemen Waanders. Goal-oriented optimal design of
                    infinite-dimensional Bayesian inverse problems using quadratic approximations.
                    J. Sci. Comput., 2025 / arXiv:2411.07532.
anon2026topk        Ask the right comparison: bias-aware Bayesian active top-k ranking with LLM
                    judges. arXiv:2607.02104, 2026.  [concurrent — get author list]
```
