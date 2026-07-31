# Splitter questions and the refusal channel: novelty audit (2026-07-31)

Five parallel agents, primary-source verified where obtainable. Commissioned after the author asked
whether 'open/partitioning questions as the key to adaptive elicitation' is new, and whether the
REFUSAL channel (rather than sign) is the unclaimed one.

**Headline: the splitter objective is Golbandi's split criterion exactly (ANOVA identity, derived from
his own equations). The refusal claim is FALSE for the decision-tree line and TRUE for the modern CRS
line. The safest ground is the ABLATION, not the channel.**

---

Research complete — five agents, all nine items, primary-source verified. Findings below.

# QUESTION 1 — The splitter framing

## Item 1 — Disagreement/divergence-maximizing query selection

**The decisive finding: Golbandi's split criterion IS the splitter objective, exactly.** Agent 1 obtained the WSDM 2011 full PDF and derived the ANOVA identity against their own equations. Their `splitter(t) ≝ argmin_i Err_t(i)` where `Err_t(i) = e²(t_L) + e²(t_H) + e²(t_U)`. Since `e²(t)` is a node constant:

```
e²(t) = Err_t(i) + B_t(i),  B_t(i) = Σ_j Σ_g n(t_g)_j (μ(t_g)_j − μ(t)_j)²
⇒ argmin_i Err_t(i) ≡ argmax_i B_t(i)
```

`B_t(i)` is precisely the count-weighted across-user squared deviation of the induced group-mean update. Applied greedily at every node **including the root**. Closeness 5/5. Agent 3 flagged the same reduction independently, from a different direction (`Δμ_u = K(y_u − ŷ_u)` with user-independent gain K ⇒ `Var_u[Δμ_u] = K Var_u[y_u] Kᵀ`, i.e. across-user answer variance up to a fixed gain).

Golbandi also pre-empts the naive version in prose (§3 "Contention"): *"finding that a user liked an item that is also liked by everyone else, provides less insight than discovering that a user likes a more controversial item"* — then explains why raw contention fails (Napoleon Dynamite) and why the squared-error split is the principled replacement.

Other hits: **Rashid/Karypis/Riedl 2008 IGCN** is the entropy version of the same object — `IG(a_t) = H(C) − Σ_r (|C^r|/|C|)H(C^r)` over **user-cluster labels**, i.e. "which item best separates the user population" (4/5). **Karimi et al.** same criterion with validation-split error and 6-way splits (5/5). **QBC** (Seung/Opper/Sompolinsky 1992, verbatim *"the principle of maximal disagreement"*) is 3/5 — formally a population of hypotheses about one target, but under the standard de Finetti reading (model prior = population distribution) the distinction dissolves, and QBC's own authors justify it by **information gain**. Hanneke's disagreement coefficient is 1/5 — an analysis constant, not a criterion. **IRT item discrimination** (`a_i` in 2PL) is the ~85-year-old population-separation quantity, 4/5.

Polarising/controversial-item selection is not a separate literature — it *is* Golbandi's contention paragraph and Rashid's entropy/HELF, both of which conclude raw controversy is insufficient.

## Item 2 — When does adaptivity pay

**Krause & Guestrin ICML 2007 already asks your question in its abstract** and answers it: *"a fundamental question is when an active learning, or sequential design, strategy will perform significantly better than sensing at an a priori specified set of locations... Central to our analysis is a theoretical bound which quantifies the performance difference between active and a priori design strategies."* Their **Theorem 1** is a computable, online, per-instance bound on the adaptivity gap; **Corollary 2** turns it into a stopping rule. Their diagnostic currency is hyperparameter entropy `H(Θ)`; yours is across-user argmax dispersion. Same intellectual move, different quantity.

**Golovin & Krause JAIR 2011** define the adaptivity gap formally but give worst-case bounds only, no per-instance diagnostic — and Theorem 25 shows it can be Ω(n/log n). Do not cite adaptive submodularity as evidence adaptivity pays; it bounds greedy vs optimal *adaptive*.

**Jedynak/Frazier/Sznitman 2012**: confirmed, one of the two Bayes-optimal policies *"asks a deterministic set of questions"* and is explicitly labelled non-adaptive in §4.2; adaptivity gap under entropy loss is exactly zero there.

**Empirical adaptivity-gap measurement in recsys cold-start interviews: none found.** Zero arXiv hits for `"adaptivity gap" AND "preference elicitation"`. This is a negative search result — phrase as "we are not aware of," not "no one has."

Counter-example to keep in view: **CAT/IRT adaptive testing halves item counts** — because IRT is logistic, so Fisher information depends on θ̂.

## Item 3 — Is "covariance is answer-independent ⇒ EIG is non-adaptive" novel?

**No. Textbook, three times over, and one of them is 34 years old.**

**MacKay 1992** (*Neural Computation* 4(4)), §4.2, verbatim: *"the solutions to the first and second tasks depend only on the x locations where data were previously gathered, not on the actual data gathered... A complete data-gathering plan can be drawn up before we start. It is only for a nonlinear model that our decisions about what data to gather next are affected by our previous observations!"*

**Krause/Singh/Guestrin JMLR 2008** §3.1: *"a sequential, closed-loop design taking into account previous measurements bears no advantages over an open-loop design."*

**Krause & Guestrin ICML 2007** §4 proves `max_π H(X_π) = max_A H(X_A)` and generalises: *"any objective function depending only on the predictive variances, such as mutual information, cannot benefit from sequential strategies."*

And in recsys specifically: **Sepliarskaia et al. RecSys 2018** minimise `tr[(V_FᵀV_F + λE)⁻¹]` — A-optimality on latent representations — observe it is answer-independent, build a **Static Preference Questionnaire**, and *"are the first to rigorously prove"* it is near-optimal, cutting question count up to 3×. That is your observation, proved and published, in your domain, eight years ago.

## Items 4–5 — Conjoint, decision analysis, LLM-era

**Toubia et al. 2003/2004 is the best defensive citation you have.** FastPACE has a subsection literally titled "First question": *"the polyhedral method offers little guidance for the choice of the first question."* And the 2004 CBC paper: *"we selected the first question by randomly choosing from amongst the axes... future research might use aggregate customization to select the first question."* A 22-year-old stated open problem you are answering. Good for you — but it means "nobody noticed" is unavailable.

Polyhedral/ACA criteria are all strictly per-respondent (1/5), and ACA's *utility balance* is the **opposite** of a splitter — it seeks per-respondent indifference. Hauser & Toubia MktSci 2005 is a useful cautionary cite: utility balance induces **endogeneity bias**.

**Sándor & Wedel 2005 is a false alarm** — "heterogeneous designs" means the *design set* is heterogeneous, optimised for D-efficiency, not questions that expose heterogeneity. The paper you actually want for "design targets the heterogeneity distribution" is **Yu, Goos & Vandebroek, MktSci 2008** (3/5) — semi-Bayesian D-optimal mixed-logit designs to estimate *"the mean vector and the variances of the multivariate heterogeneity distribution."*

**Atkinson & Fedorov 1975 (T-optimality)** — designs discriminating between rival *models*; maps conceptually onto discriminating user types (2/5), the sharpest classical-DOE analogue.

**Chajewska/Koller/Parr AAAI 2000** does use a population mixture prior (*"distinct clusters in the population"*) but selects by myopic VOI for the decision — exactly the immediate-gain criterion you're rejecting. Cite as the contrast.

**The 2024–26 danger: VPL (Poddar et al., NeurIPS 2024).** Variational encoder over a **user-type latent z** for "diverse populations of users with divergent preferences," with §4.2 "Active Learning of Preferences to Minimize Latent Uncertainty" selecting queries by **maximising mutual information with the latent**. Maximising `I(y; z | q)` over a multimodal population latent *is* operationally a splitter — MI with user type is large exactly when the population disagrees. Closeness 4/5.

**GATE (Li et al. 2023) §5.2 already publishes your motivating measurement**: *"we compute the entropy in p(yes) for each question across participants... average entropy of 0.77 bits,"* motivated by *"Elicitation is only helpful if there is variation in people's preferences."* They measure across-user answer dispersion; they don't select on it.

Also live: **arXiv:2605.00696** (May 2026), persona-mixture Bayesian design for cold-start sequential item selection — architecturally the nearest current competitor (3/5). PEBOL and the 2025 clarifying-question work are single-user, 0/5.

Negative results: zero arXiv hits for `"user segmentation" AND "question selection"`, `"heterogeneous users" AND "query selection"`. No 2024–26 paper selects a first question to segment users.

# QUESTION 2 — The refusal channel

## Item 6 — Who models UNKNOWN as a distinct updating signal

**The author's claim is FALSE as stated.** It fails on the decision-tree line by 15 years and on the CRS line by two.

Verified from full PDFs: **fMF (Zhou/Yang/Zha SIGIR 2011)** — *"the finite set {0, 1, Unknown}"*, ternary tree, and the unknown child gets **its own fitted latent profile** with hierarchical shrinkage, not a default branch. **Sun et al. WSDM 2013** — unknown group has its own regressor `T_U`, and their 76-user study had a literal **"Unknown" button** in the interface. **Rashid 2008** — *"treating the missing evaluations of an item as a separate category"*, scored inside the split criterion with tuned weight `w₀ = 1/2`.

⚠ **Golbandi WSDM 2011 itself could not be obtained in full text** (ACM 403, no OA copy anywhere — unpaywall/OpenAlex/CORE/IA-Scholar all negative). The ternary split is attested by two independent primary texts (fMF's *"[8, 20]... ternary"*; Sun'13) plus your own replication in `C:\dev\phd\casper\scripts\_archive\paper1_v1\golbandi_tree.py`. **Do not quote Golbandi's internal mechanism directly** — cite fMF/Sun, or get the PDF through your library. (Note: a different agent's session *did* extract Golbandi full text and quotes `w_L=5, w_H=1, w_U=0.02`, *"an unknown vote was found to be relatively insignificant"* — consistent with your existing `answerability_in_elicitation.md`. Treat as verified via that route, but reconcile the two before citing a section number.)

**Where the claim holds — the modern lines.** Zero occurrences of unknown/don't-know/neutral in the full text of EAR, SCPR, UNICORN, MCMIPL. SCPR verbatim folds attribute-absence into a NO. Both major CRS surveys have zero occurrences.

**The cleanest quotable fact you have: PEBOL computes a three-class NLI distribution and throws the third class away.** Verified twice (repo + arXiv PDF): the mNLI model *"predicts logits for entailment, contradiction, and neutral"*, but *"we pass the temperature-scaled entailment and contradiction scores through a softmax."* Responses are `r ∈ {0,1}`; the simulated user is *"instructed to provide only 'yes' or 'no' responses."*

**Adjacent fields you must cite yourself:** Rose, von Davier & Xu (ETS RR-10-11, 2010) model IRT omissions with an extra latent dimension built from response indicators — "omission as evidence about the person," solved in 2010 on ~250k students. And **Krosnick et al. POQ 2002 cuts against you**: across nine experiments, DK options attract satisficers, and data quality *"was not compromised by the omission of no-opinion options."* Your escape is that Krosnick studies *attitudes* (DK = non-attitude) while yours is *factual non-experience*, closer to IRT omits — but if you don't cite it you look naive.

## Item 7 — Has the branch been priced?

**In the interview-tree line: no ablation found.** Confirmed via OpenAlex/S2/arXiv sweeps. Your "no published ablation" claim survives *for that line*.

**But an equivalent measurement exists and is the single most dangerous prior work.** Shen et al., **ACM TORS 2(4) art. 27, 2024** — "Multi-Interest Multi-Round CRS with Fuzzy Feedback Based User Simulator" — introduces "I don't know"/"All right" as a third response, builds machinery to consume it, and reports the delta from consuming vs ignoring it. Table 5, Amazon-Clothing SR@15: SCPR 0.251 → 0.322; UNICORN 0.284 → 0.322; MIPL 0.341 → 0.382. Plus a reward sweep `r_ask^bord ∈ [−0.05, −0.03, −0.01, 0.01]`, optimum −0.03. **This is in your target venue.** (Caveat: their Adapted-SCPR and Adapted-UNICORN rows are byte-identical — likely a duplicated-row typo. Don't lean on those two rows.)

**The two-senses distinction is already named — by fMF §4.5, verbatim:** *"we assume that the user will respond 'Unknown'... if she does not rate the corresponding item. This assumption, however, might be inaccurate in practice... the user may actually respond 'Like' or 'Dislike'."* Footnote 3: *"The bias in training data seems to have been largely ignored in previous work."* They name the gap between *unknown = absent from the matrix* and *unknown = live refusal*, and do not close it. Naming is 15 years old; closure is open.

Rates: Sun'13 — *"the unknown branch capturing more than 80% of the users at each split"*; Rashid 2008 — users rate *"at least one third"* of presented items.

## Items 8–9 — Answerability and exposure

**Answerability is a named taxonomy category with a named metric.** Elahi/Ricci/Rubens 2016 §5.2.1 "**Acquisition probability based**": *"maximizing the probability that the selected items are familiar to the user, hence, are rateable"*; metric = *"success ratio (#acquired_ratings/#requested_items)"*; failure mode named: *"strategies that only focus on the informativeness... may fail to actually acquire ratings, by selecting obscure items that users do not know and cannot rate."*

Golbandi: *"the seed set should include familiar items, since asking users to rate obscure items is mostly futile"* — and answerability **falls out of the objective**: *"a non-familiar item will place, by definition, almost all users within the Unknowns user group... Clearly, this is not the best way to decrease the squared error."*

**Bıyık CoRL 2019 is 4/5 on the axis but a different quantity.** `I(ω;q|Q) = H(q|Q) − E_ω H(q|ω,Q)` — the subtracted term is *"the human's uncertainty when answering."* But his "hard" question is one the user **knows and is torn on** (taste indifference); yours is one the user has **never encountered**. Same slot in the objective, different construct. Do not conflate.

**ExpoMF (Liang et al. WWW 2016)** is the two-channel model: `a_ui ~ Bernoulli(μ_ui)`, `y_ui|a_ui=1 ~ N(θᵀβ)`, `y_ui|a_ui=0 ~ δ_0`. Decisive asymmetry: *"When y_ui > 0, we know that a_ui = 1"* — exposure is only ever **inferred**, never observed, and only for zeros. Citation-graph check: 6 of 377 citers mention "conversational," **none** applies an exposure latent to dialogue; 0 of 169 Golbandi citers touch exposure/awareness. **Consideration-set theory (Roberts & Lattin JMR 1991)** is the real 35-year-old prior art for two-channel awareness-then-preference.

MNAR is correctly distinguished: it treats missingness as a property of a static log, never observed, and *corrects for* it. An interview "I haven't seen it" is solicited, timestamped, attributable, and available before the next decision. No one has carried MNAR into dialogue.

---

# NOVELTY VERDICTS, RANKED

**SAFEST — (c) the refusal/unknown channel as a measured, priced signal, scoped to live-dialogue NL elicitation.** Three verified conjuncts: PEBOL computes and discards its neutral class; EAR/SCPR/UNICORN/MCMIPL/GATE admit no third response and SCPR folds absence into a NO; and where ternary unknown *is* standard, it means matrix-absence, which fMF flags as an inaccurate proxy and does not resolve. Claim the **ablation**, not the channel. Most dangerous: **Shen et al. TORS 2024**.

**SECOND — (a) the dispersion diagnostic law**, but only in its full three-part form: dispersion of the induced **argmax question** (not H(Θ)) + the paired **negative** claim that criterion form doesn't predict the gap + an **empirical** adaptivity-gap measurement in recsys cold start. All three cells are empty. Part 1 alone reads as a re-parameterisation of Krause & Guestrin Theorem 1 to the reviewer most likely to be assigned your paper. Most dangerous: **Krause & Guestrin ICML 2007**. Critically, this is also the **only** claim that explains both the EIG-selector collapse *and* the RL/distilled-actor collapse — the Gaussian fact covers only the first, and is published.

**THIRD, RISKY — (d) answerability as an elicitation-design axis. Reframing, largely pre-empted.** Do not claim a new axis; Elahi 2016 has the category, the rationale and the metric. What survives is at the **intersection of (d) and (e)**: reading the unknown response as a reusable **exposure** quantity rather than a local taste partition. Claim the *transfer* of ExpoMF's structure to solicited dialogue answers, and make the inversion (observed vs inferred `a_ui`) the contribution. Most dangerous: **Golbandi** for (d), **ExpoMF** for (e). Identifiability risk: ExpoMF's default exposure prior *is* item popularity, so a fitted exposure channel can collapse into popularity+activity — the exact failure your own `answerability-audit-2026-07-22.md` already recorded. You must show signal beyond both or the claim is unfalsifiable.

**PRE-EMPTED — (b) the splitter objective. Do not christen it.** It is the CART regression split criterion applied to user profiles, published WSDM 2011, by authors who explain in prose why it beats the contention/entropy heuristics you'd otherwise reach for. Restating it in latent-belief space is a change of representation; fMF may own even that. Most dangerous: **Golbandi WSDM 2011**; runner-up **Rashid 2008 IGCN**; dark horse **VPL NeurIPS 2024**.

# WHERE THE CLAIM WOULD BE WRONG, NOT MERELY UNORIGINAL

1. **"Explicitly NOT information/entropy/EIG" is mathematically false under your own model.** Agent 1 derived and numerically confirmed (d=6, 2M users): with population = prior, `tr Var_u[δμ_u] = α‖Λ₀⁻¹φ‖²/(1+αs) = tr(Σ₀ − Σ₁)` exactly — MC 0.29228 vs closed form 0.29261. This is just the law of total variance with the first term deterministic. **The splitter objective IS A-optimal variance reduction**, differing from EIG only by trace-vs-log-det. A Bayesian-OED reviewer finds this in five minutes.
   *The escape, and it is a real one:* the actual content of your proposal is that you measure dispersion under the **empirical answer distribution of real users**, not the model's prior predictive. The gap between model-EIG and empirical `Var_u[δμ_u]` is then **exactly a measure of answer-model misspecification** — which, given this project's two prior circular-simulator incidents, is a far stronger paper than a new objective.

2. **"EIG cannot capture population separation" is false under a mixture/latent-type prior.** VPL and arXiv:2605.00696 are direct counterexamples. The defensible sentence is scoped: "under the linear-Gaussian belief our tower induces."

3. **The reviewer objection "your non-adaptivity is a Gaussian/linear artefact" is CORRECT and the literature says so explicitly.** Krause & Guestrin: *"for non-Gaussian models, sequential strategies can strictly outperform a priori designs, even with known parameters."* MacKay: *"It is **only** for a nonlinear model..."* BALD's objective is manifestly mean-dependent under probit. CAT works *because* IRT is logistic. If your answers are binary or ordinal, the correct likelihood is probit/ordinal, under which `Λ_t` **does** depend on past answers. Best response: run the probit/ordinal variant and show dispersion stays low anyway — that converts the objection into your headline result.

4. **`Λ_t = Λ₀ + Σαφφᵀ` is answer-independent only if `α_j` is fixed.** Heteroscedastic precision, a "don't know" option, or confidence-weighted answers all break it — note that your own refusal channel is one of the things that breaks it.

5. **The Λ fact does not explain the RL collapse.** The posterior *mean* is emphatically answer-dependent, and a reward-maximising policy depends on the mean. Conflating the two is a non-sequitur a theory-literate reviewer will catch.

6. **"The decision-tree line skips unknown or folds it into a negative" is factually false** — fMF fits a dedicated child with its own latent profile.

7. **On your own numbers:** +0.0081 at k=8 vs −0.0193 is a **sign flip**, which under your own `surprise-means-debug-the-harness.md` rule should be gated on Most-Popular before publication as a price. And if that was measured with "unknown = not in the matrix," you have measured the data artefact (~80% of splits, per Sun'13), not the refusal — collapsing into precisely the confusion fMF flagged in 2011.

# VERIFICATION GAPS — do not rely on these without a second pass

- **Golbandi WSDM 2011 full text**: closed everywhere; two agents disagree on obtainability. Get the library PDF before citing any section number.
- **fMF split criterion in latent space** (whether it owns the latent-space version of the splitter): the abstract confirms ternary + per-node latent profiles; the exact objective is unverified. **Highest-value remaining unknown for (b).**
- Freund et al. 1997 full text (paywalled); Rashid IUI 2002 full text (ACM 403 — its popularity-as-answerability rationale is verified only *through* Elahi and Golbandi); Chaloner & Verdinelli 1995 (abstract only — MacKay is the stronger cite anyway); Sándor & Wedel and Yu/Goos/Vandebroek (abstracts only); Kelley 1939 / origin of the discrimination index.
- **Existence or absence of an optimal-design-for-latent-class-choice literature**: searches came back empty but the search budget was exhausted. Do not assert that gap in print without one clean Scopus pass.
- Medical-diagnosis question selection: not covered.
- All five agents exhausted the shared WebSearch quota early and worked via OpenAlex/Crossref/S2/arXiv APIs plus local PDF extraction. Coverage of unindexed 2025–26 preprints is weaker than a normal round.