# Findings: Agent-Independent Attribute-Affinity Estimators for a Non-Circular Concept-Answer Simulator

*Deep research 2026-07-25 (web only, no LLM calls). Builds ON — does not re-cover —
`concept_folding_and_implicit_attribute_inference.md` (NPMI≈SEL, ExpoMF, Qin attribute-propensity, Steck
calibration, Tagommenders) and `simulator_circularity_conv_rec.md` (the circularity ruling: PEPPER target-bias,
Panickssery self-preference, RecSim agent–environment separation).*

**The design question.** Our honest concept-answer = **SEL**: signed NPMI/PMI watch-lift of concept X's member
items in the user's consumption + shrunk residual rating. SEL is **agent-independent** (uses no parameter of the
evaluated recommender ⇒ non-circular) but a **raw behavioral moment**: exposure/popularity-biased, blind on
attributes the user was never exposed to, noisy for sparse users. We want a MIDDLE GROUND imputer that is
(a) agent-independent, (b) richer/less-biased than raw NPMI, (c) behavior-calibrated. Legitimate ceiling =
"oracle-B" (same behavioral formula over the user's FULL history).

**One-line bottom line (see §Synthesis).** SEL/NPMI is a **strong, principled, cheap baseline but clearly beatable
on its two known weaknesses** (exposure confound, unobserved-attribute coverage). No single published method is
simultaneously agent-independent + exposure-corrected + content-regularized + signed; the middle ground is a small
**assembly** — NPMI signed affinity **shrunk (empirical-Bayes) toward a content/tag-genome prior**, optionally with
**content-conditioned exposure correction** — none of whose components touch the evaluated recommender. The
circularity guard is a **CKA/HSIC representational-alignment cap + a cross-model differential-benefit test**.

---

## Q1 — How is per-user ATTRIBUTE affinity represented/estimated, and is it independent of a given CF recommender?

The candidates split by **what geometry they live in**, which determines independence from the evaluated tower.

### A. Content / relevance-projection spaces — MODEL-FREE, cleanest independence

- **`vig2012taggenome` — The Tag Genome (Vig, Sen, Riedl, ACM TiiS 2012)** [search + prior knowledge, high on mechanism]
  Each item → a dense **tag-relevance vector** (∈[0,1] per tag, ~1100 tags for ML), predicted by a **regression
  model** from community features: tag-application counts, text-search relevance of the tag against reviews/titles,
  and rating correlations. Computed **once, offline, from community data — independent of any downstream CF
  recommender** (it is a content/relevance model, not a user-factor model). A **per-user tag affinity** falls out
  by aggregating the genome vectors of the user's consumed items (mean / rating-weighted / TF-IDF-weighted — the
  Tagommenders `sen2009tagommenders` construction). Continuous, dense, and — crucially — **defined for EVERY tag
  even ones the user never directly consumed**, because member-item relevance carries over. Signed only if you
  center (affinity = user's tag-weighted profile minus population mean).
  → **Verdict-for-us:** the single best *coverage* fix and the cleanest independent substrate. Genome relevance is
  model-free; user affinity = a content projection of behavior, no CF params. Pair its *content prior* with NPMI's
  *behavioral sign*. The expanded/cross-domain genome (`kotkov2021genome`, HCIR 2026) extends tag coverage.

- **Content / text-embedding attribute spaces — attribute2vec / item2vec+content / SBERT centroid** [search, medium]
  Item2Vec (`barkan2016item2vec`) and attribute-aware embedding nets put items **and** attributes in one vector
  space; a concept = centroid/probe direction of its member items' content vectors; user affinity = projection of
  the user's consumed-item content vectors onto that direction. Fully **content-derived ⇒ agent-independent** of the
  CF tower (it is our own whitened-concept-centroid construction, but in a content/text space rather than the
  evaluated tower's latent). Covers unseen attributes via text. Weakness: no exposure correction, sign is only
  "more/less than average content exposure," not calibrated aversion.
  → **Verdict:** the coverage/OOD substrate; but on its own it is *content-similarity*, not *behavior-calibrated
  preference* — must be fused with a behavioral term or it drifts toward "what you watched a lot of," re-importing
  popularity.

### B. Separate latent factor models over user×item×tag — INDEPENDENT of the evaluated tower, but SOFT (shared CF-geometry family)

- **`rendle2010pitf` — Pairwise Interaction Tensor Factorization (PITF, WSDM 2010)** [search, high]
  Factorizes the **user×item×tag** tensor into pairwise-interaction factors (linear-time special case of Tucker),
  learned with a **BPR ranking** criterion; won the ECML/PKDD'09 tag-recommendation challenge. Yields explicit
  **per-user tag factors ⇒ a user-tag affinity by dot product.** Requires *tagging* data (user actually applied
  tags) — very sparse on ML (most users never tag), so coverage is the opposite of the genome's.
  → **Verdict:** gives a genuine per-user attribute factor, and as a **separate model it is independent of the
  evaluated recommender's parameters** — but it is the SAME *family* of learned CF geometry, so it carries a soft
  self-preference risk (Q4) and it needs tag-application data we mostly lack. Cite as the canonical "attribute
  factor" prior art; not the primary recipe.

- **`tagmf2018 / co-factorization` — Tag-Enhanced MF (TagMF, Loepp et al. IJHCS 2018); matrix co-factorization with
  tags (Zhen-style, ESWA 2019)** [search, medium]
  Jointly factorize {user×item rating, user×tag preference, tag×item relevance}. TagMF explicitly **derives
  user-tag relevance even for users who never tagged**, via the shared factor space. Produces a per-user tag
  affinity vector.
  → **Verdict:** same status as PITF — a *separate* CF-family model gives an independent-of-the-target affinity, but
  softly shares learned-CF geometry; useful as a second-opinion estimator under the Q4 guard, not as the sole oracle.

- **`rendle2010fm` — Factorization Machines (already in INDEX)**: attributes get their own latent factors; the
  user×attribute interaction term IS a per-user attribute affinity. Same soft-independence story; general engine
  behind EAR's fold-in. Background.

### C. Distribution / propensity profiles (already covered — recap for completeness)
- `steck2018calibrated` genre distribution p(g|u): positive-only, unsigned (cannot express aversion).
- `qin2020attributepropensity` attribute-exposure propensity + IPS: the debiasing layer, agent-independent, high variance.

**Q1 verdict.** Independence has two grades: **(i) model-free** — tag-genome projection & content-embedding
projection (no CF params at all; cleanest, best coverage); **(ii) separate-CF-model** — PITF / TagMF / FM attribute
factors (independent of the *evaluated* tower's parameters, but same learned-CF *family* ⇒ needs the self-preference
guard). SEL/NPMI is a grade-(i) *moment* estimator with no content smoothing. The upgrade path is grade-(i)
content smoothing on top of NPMI, NOT a grade-(ii) model (which reintroduces circularity risk and needs tag data).

---

## Q2 — Is NPMI/SEL SOTA, or is there a better AGENT-INDEPENDENT recipe?

NPMI/PMI log-lift is a **moment estimator**: `NPMI(u,g)` of consuming g-members vs base rate — signed,
popularity-corrected (÷ attribute base rate), volume-corrected (per-user rate), bounded [−1,1]. Its two documented
failure modes and the independent fixes:

| Weakness of raw NPMI/SEL | Best AGENT-INDEPENDENT fix | Cost |
|---|---|---|
| **Exposure confound** (a non-watch of a niche-member = unaware OR dislike; NPMI reads it as weak signal) | **ExpoMF** `liang2016expomf` (content/popularity-conditioned exposure prior) or **attribute-propensity IPS** `qin2020attributepropensity`; **causal PDA/CausalEPP** `zhang2021pda` deconfounds item popularity from preference | ExpoMF: EM, slow; IPS/causal: high variance |
| **Unobserved-attribute coverage / sparse users** (NPMI undefined or zero-support for attributes the user never hit) | **Tag-genome / content projection** `vig2012taggenome`,`sen2009tagommenders` — gives a value for every attribute via member-item content; **empirical-Bayes shrinkage** of low-support NPMI toward that content prior | needs genome/content; shrinkage hyperparam |
| **Heavy-user / high-count saturation** (raw counts dominated by prolific users & popular members) | **BM25 / TF-IDF weighting** (IR family; benfred/`implicit`) — term-frequency saturation + IDF down-weights common attributes; a strictly better weighting than raw count, still model-free | tune k1,b |
| **Noisy sign on thin negative tail** | ExpoMF posterior-exposure confidence, or Beta/Wilson shrinkage on support | — |

- **BM25/TF-IDF weighting** (`implicit` library; classic IR-CF) is the cheapest concrete improvement over raw NPMI:
  replace raw member-watch counts with BM25-saturated, IDF-weighted counts before computing the lift. Model-free,
  agent-independent, handles heavy users and popular members better than PMI's linear base-rate division.
- **Causal debiasing** (`zhang2021pda` PDA "Causal Intervention for Leveraging Popularity Bias"; CausalEPP 2025):
  separates item popularity (a confounder via the do-operator) from genuine preference. Agent-independent
  (operates on the interaction graph, not the target recommender). Principled popularity fix; heavier machinery,
  and mostly framed for item-level not attribute-level, so it needs adaptation.

**Q2 verdict.** NPMI/SEL is **NOT SOTA — it is the cheap moment baseline.** It is *clearly beatable, agent-
independently*, on (i) exposure (ExpoMF / IPS / causal-PDA), (ii) coverage (genome/content smoothing), (iii)
weighting (BM25 > raw count). None of these three fixes touches the evaluated recommender, so all preserve
non-circularity. The **single most cost-effective upgrade** is BM25-weighted lift **+ empirical-Bayes shrinkage to a
tag-genome content prior**; the **most principled** is content-conditioned **ExpoMF** over member items.

---

## Q3 — Is there an off-the-shelf agent-independent, exposure-corrected, content-regularized signed estimator?

**No single published method holds all four properties (independent + exposure-corrected + content-regularized +
signed).** The closest single names:

- **Content-conditioned ExpoMF** (`liang2016expomf`): exposure prior μ_i = σ(content/popularity features) gives
  *exposure correction + content regularization*, and it is agent-independent — but it is **positive-preference
  centric** (preference re-enters as a confidence weight, not a signed direction) and models *item* exposure, so
  attribute affinity requires aggregating exposure-corrected preference over member items. Gets 3 of 4; **sign is
  weak.**
- **Tag-genome-projected user profile** (`vig2012taggenome` + `sen2009tagommenders` TF-IDF aggregation):
  *content-regularized + coverage + agent-independent*, but **NOT exposure-corrected and essentially unsigned**
  (a positive emphasis profile). Gets ~2.5 of 4.

So the middle ground is a **buildable assembly.** Ranked:

**Assembly A (recommended middle ground): BM25-lift + EB-shrinkage to a genome content prior.**
`affinity(u,g) = shrink( signed_BM25_lift(u,g) , prior=genome_projection(u,g) ; weight=f(support) )`.
Signed BM25/NPMI carries calibrated behavioral sign & magnitude; for low-support attributes the estimate is pulled
toward the content/genome-predicted affinity (coverage); shrinkage weight = support count (confidence).
- Non-circularity: **full (grade-i, model-free)** — genome + behavior, no target params. ✔
- Sophistication: moment estimator + IR weighting + content prior + EB shrinkage — respectable, all cited. ✔
- Coverage of unobserved attributes: **good** (content prior fills gaps). ✔
- Calibration to behavior: **strong** (BM25 lift is the behavioral core; genome only fills the tail). ✔
- Cost: light, closed-form, no EM. Novel component: the *specific EB-shrinkage of a signed behavioral lift toward a
  genome prior* is an assembly, not a named method — cite Tagommenders (TF-IDF tag prefs) + James-Stein/EB shrinkage
  + BM25-CF as parents; claim "assembled," not "novel method."

**Assembly B (most principled): content-conditioned ExpoMF over member items → signed affinity + exposure confidence.**
- Non-circularity: full (independent generative model). ✔ Sophistication: highest (generative, principled confound
  fix). ✔ Coverage: good (content-conditioned exposure prior). ✔ Calibration: strong but **sign is engineered**
  (needs a negative-preference extension, e.g. below-prior exposure-corrected preference, à la `paudel2018loss`).
  Cost: EM, slow, popularity-biased on rare items. Use when the unaware-vs-dislike confound is load-bearing.

**Assembly C (second-opinion only): a separate PITF/TagMF user-tag factor model.**
- Non-circularity: **soft** — independent of the evaluated tower's params but same learned-CF family (Q4 risk).
  Sophistication: high. Coverage: needs tag-application data (sparse on ML). Calibration: implicit via BPR. Use only
  as a cross-model second opinion under the Q4 guard, never as the sole oracle.

**Q3 verdict.** Adopt **Assembly A** as the middle-ground answer imputer; keep **Assembly B (ExpoMF)** as the
principled ceiling/robustness check; treat **oracle-B (full-history BMI-lift)** as the declared upper bound. All
three are non-circular; A is the sweet spot of sophistication vs cost.

---

## Q4 — Validating that an answer simulator is REPRESENTATIONALLY independent of the model under test

Beyond "no target-item leak" (`kim2024targetfree` PEPPER) and the self-preference *concept*
(`panickssery2024self`), the buildable, quantitative guards:

- **Representational-similarity cap (CKA / HSIC / distance correlation).** `kornblith2019cka` **Centered Kernel
  Alignment** normalizes the **Hilbert-Schmidt Independence Criterion (HSIC)** to compare two representations on the
  same inputs; invariant to orthogonal transforms & isotropic scaling; `szekely2007dcorr` distance correlation
  captures nonlinear dependence; the RSA tradition (`kriegeskorte2008rsa`) compares systems via representational
  *distance* matrices. **Buildable guard:** over a held-out user set, form the answerer's affinity-vector geometry
  (users × concepts) and the recommender's `u*` geometry; require
  **CKA(answerer, u*) ≤ CKA(behavioral-SEL, u*) + ε.** The behavioral simulator sets the admissible ceiling; any
  answerer whose geometry aligns with the recommender *more than raw behavior does* is flagged as leaking model
  geometry. (`resi2024benchmark` warns similarity depends on the probe set — fix the probe set and report the family
  of measures, not one.)
- **Cross-model differential-benefit test (the decisive behavioral guard).** Self-preference manifests as an
  answerer that helps the **evaluated** recommender more than an **independent held-out** recommender. Fit ≥2
  structurally different towers (e.g. RecVAE-class vs EASE vs item-kNN); require the answerer's NDCG lift to be
  **statistically indistinguishable across towers** (no preferential lift for the one it was derived from). This is
  the recsys analogue of jury-of-independent-judges in LLM-eval and directly operationalizes PEPPER's shortcut
  signature (recall-on-selected vs residual). SEL, being behavioral, passes by construction — it is the reference.
- **Shortcut-signature diagnostic** (`kim2024targetfree`): compare recall on the answerer-favored slice vs the
  residual; a large gap = shortcutting. Reuse as a pre-registered control.

**Q4 verdict.** Build a **two-part guard**: (i) **representational** — CKA/HSIC alignment of the answerer's affinity
geometry with the recommender's `u*` must not exceed the behavioral baseline's (SEL defines the cap); (ii)
**differential-benefit** — the answerer must not lift the evaluated tower more than an independent tower. This is
exactly the requested "answerer's alignment with the recommender's u* must not exceed behavioral's," made concrete
with a named metric (CKA/HSIC) and a cross-model agreement test.

---

## SYNTHESIS (feeds the design decision)

**(i) Is SEL good enough or beatable?** SEL/NPMI is a **principled, agent-independent, cheap moment baseline — not
SOTA and clearly beatable**, agent-independently, on exposure bias, unobserved-attribute coverage, and heavy-user
weighting.

**(ii) Best respectable non-circular recipes, ranked:**
1. **BM25/NPMI signed lift + empirical-Bayes shrinkage to a tag-genome content prior** (Assembly A) — model-free,
   signed, coverage-complete, behavior-calibrated, light. *Maps onto us:* SEL's exact signal, IR-weighted, with the
   genome filling the tail SEL is blind to.
2. **Content-conditioned ExpoMF over member items** (Assembly B) — the principled exposure-confound fix; heavier,
   sign needs an extension. *Maps onto us:* the ceiling/robustness estimator when unaware-vs-dislike matters.
3. **Separate PITF/TagMF user-tag factor model** (Assembly C) — genuine per-user attribute factor, but soft
   independence + needs tag data. *Maps onto us:* a cross-model second opinion, gated by the Q4 guard.
4. **Tag-genome-projected profile alone** (`sen2009tagommenders`) — great coverage, but unsigned/positive-only;
   *maps onto us:* the content-prior component of Assembly A, not a standalone answer.

**(iii) Proposed recipe — non-circular sophisticated concept-answer imputer.**
`SEL⁺(u,g) = shrink( sign · BM25_lift(u, members(g)) + λ·resid_rating(u,g),  prior = center(genome_proj(u,g)); w = support(u,g) )`
- **BM25_lift**: replace raw member-watch counts with BM25-saturated, IDF-weighted counts, then signed
  log-lift vs base rate (upgrades SEL's raw NPMI).
- **genome_proj(u,g)**: tag-genome relevance of g projected through the user's consumed items (model-free content
  prior), centered to be signed.
- **shrink**: empirical-Bayes / James-Stein pull of the behavioral term toward the content prior with weight rising
  in support(u,g) — behavioral dominates when the user has evidence, content prior fills the tail.
- **(optional robustness)** swap BM25_lift for content-conditioned ExpoMF exposure-corrected preference when the
  exposure confound is load-bearing (Assembly B).
- **Novelty honesty:** every component is cited prior art (NPMI, BM25-CF, Tagommenders/tag-genome, EB shrinkage,
  ExpoMF); the *assembly* — a signed behavioral lift EB-shrunk toward a genome content prior as a **non-circular
  answer simulator** — is what is new, and it is an assembly, not a new estimator. Claim "assembled from," never
  "novel method."
- **Validity guard (ship with the recipe):** (1) CKA/HSIC(SEL⁺ geometry, recommender u*) ≤ CKA(SEL, u*) + ε;
  (2) cross-model differential-benefit test (indistinguishable lift on RecVAE-class vs EASE vs item-kNN);
  (3) PEPPER shortcut-signature control; (4) oracle-B (full-history lift) as the declared ceiling.

**(iv) Confidence + biggest uncertainty.** **Confidence HIGH** that SEL is beatable non-circularly and that
Assembly A is the right middle ground (all components are established; the fixes are orthogonal to the target
model). **Confidence HIGH** that CKA/HSIC + cross-model differential-benefit is a defensible, buildable circularity
guard. **Biggest uncertainty:** whether the genome/content prior *materially* improves oracle-B-relative fidelity or
merely re-imports content-popularity through the back door — the content prior's *sign* is weak, so if the tail it
fills is mostly aversion, it may add little over SEL and could even bias toward "watched-a-lot" content similarity.
Second uncertainty: the grade-(ii) models (PITF/TagMF) are the most "sophisticated" but their soft CF-family
independence is exactly what the Q4 guard exists to police — they may fail the CKA cap. **Test empirically:** measure
each recipe's correlation with oracle-B AND its CKA to u* against SEL's, and pick the point that maximizes
oracle-B agreement subject to the CKA cap.

## Caveats / confidence flags
- Tag-genome computation mechanism from search summary + prior knowledge (PDF didn't parse this pass) — high
  confidence it is a model-free community-data regression, independent of any CF recommender.
- PITF/TagMF "independent of the evaluated recommender" is w.r.t. *parameters*; the shared learned-CF *family* is a
  soft-circularity risk, not a hard leak — hence the Q4 guard. Medium confidence a reviewer accepts grade-(ii) as
  "independent" without the guard.
- CKA/HSIC as a *simulator-circularity* guard is our adaptation of a representation-comparison tool; no source
  applies CKA to answer-simulator validation specifically — say "we adapt," not "standard practice."
- "No single method holds all four properties" is a search-pass negative (medium confidence on absence) — say
  "we are aware of none."
