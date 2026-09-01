# IM1 / IM2 as a plausibility score for Paper C's C2 — deep read

**Status: NEW FILE, NOT RECONCILED.** `INDEX.md` and the `.bib` are deliberately **not** edited here;
a later pass reconciles. Do not cite from this file without adding the bib entry first.

**Target:** Arnaud Van Looveren & Janis Klaise, *Interpretable Counterfactual Explanations Guided by
Prototypes*. arXiv:1907.02584 (v1 2019-07-03, **v2 2020-02-18**).
**Read:** full PDF (17 pages incl. supplementary), v2, 2026-08-24. Both authors: Seldon Technologies Ltd, London.
**Consumer:** `docs/design/PAPER_C_DESIGN.md` §C2 item 3 — *"On-manifold plausibility — IM1/IM2 …
a user-free, answer-free autoencoder plausibility score. **Import it; do not invent one.**"*

---

## 0. Verdict up front

**IM1 and IM2 cannot be imported into C2 as written.** Both are **ratios between two
class-conditional autoencoders**, evaluated on a **paired** object `x_cf = x0 + δ` in the
**classifier's input space**. C2 has no classifier, no class partition, no `x0`, and its object is a
**direction/point in a CF latent space**, not a perturbed input. Every one of the four
preconditions fails. What *can* transfer is the weaker, older idea they build on — their `L_AE`
term, i.e. reconstruction error under an autoencoder fitted on real data as a manifold proxy — and
that is not IM1/IM2 and must not be labelled as such.

There is also an independent reason to be wary: **IM1/IM2 are under a documented, specific
critique** (§5), including one pathology (IM2 is monotone in the magnitude of the instance) that
would bite C1/C2 *directly*, because our emitted `ẑ` has an unconstrained norm.

Full argument in §2; the honest options in §3.

---

## 1. Exact definitions [V]

Both metrics are introduced in §4.1 "Evaluation", as evaluation instruments for the paper's own
method. Verbatim setup [V]:

> "In order to evaluate interpretability, we introduce two interpretability metrics IM1 and IM2. Let
> AEi and AEt0 be autoencoders trained specifically on instances of classes i and t0, respectively.
> Then IM1 measures the ratio between the reconstruction errors of x_cf using AEi and AEt0"

### IM1 — Eq. (13) [V]

```
IM1(AE_i, AE_t0, x_cf) := ||x0 + δ − AE_i(x0 + δ)||²₂ / ( ||x0 + δ − AE_t0(x0 + δ)||²₂ + ε )
```

Where `t0` = the **original / factual** class of `x0` (`t0 = argmax f_pred(x0)`), `i` = the
**counterfactual / target** class, `x_cf = x0 + δ`.

> "A lower value for IM1 means that x_cf can be better reconstructed by the autoencoder which has
> only seen instances of the counterfactual class i than by the autoencoder trained on the original
> class t0. This implies that x_cf lies closer to the data manifold of counterfactual class i
> compared to t0, which is considered to be more interpretable." [V]

**Lower is better.** The scale is anchored: IM1 = 1 means both class AEs reconstruct it equally
well; IM1 < 1 means it looks more like the target class than the origin class.

### IM2 — Eq. (14) [V]

```
IM2(AE_i, AE, x_cf) := ||AE_i(x0 + δ) − AE(x0 + δ)||²₂ / ( ||x0 + δ||₁ + ε )
```

Where `AE` (no subscript) is **an autoencoder trained on all classes**. Note the denominator is the
**L1 norm of the counterfactual instance itself**, not a reconstruction error — a scale
normalisation, explicitly motivated:

> "We scale IM2 by the L1 norm of x_cf to make the metric comparable across classes" [V]

> "A low value of IM2 means that the reconstructed instances of x_cf are very similar when using
> either AE_i or AE. As a result, the data distribution of the counterfactual class i describes x_cf
> as good as the distribution over all classes. This implies that the counterfactual is
> interpretable." [V]

**Lower is better.** Unlike IM1 the scale is *not* anchored — 0 is the floor, and the paper reports
it as `IM2 (×10)` in both tables.

### Which autoencoders, trained on what

Three (per counterfactual-class pair) distinct AEs are needed, and **all are evaluation-only
artefacts, trained separately from the AE used inside the method's own loss**:

| model | trained on | used in |
|---|---|---|
| `AE_i` | instances of the **target** class only | IM1 numerator, IM2 |
| `AE_t0` | instances of the **origin** class only | IM1 denominator |
| `AE` | **all** classes | IM2 |

Classes are **model-predicted, not ground-truth**: the recipe everywhere in the paper is
`X_i ← {x ∈ X | argmax f_pred(x) = i}` (Algorithms 1 & 2, step 3).

Architectures (supplementary) [V]: for **Breast Cancer**, the class-specific AEs are "3 dense layers
in the encoder with respectively 20, 10 and 6 neurons"; decoder 10, 20, then linear back to input
dim; Adam, 500 epochs, batch 128, **MSE** loss. For **MNIST**, "3 convolutional layers with 3×3
filters and ReLU activations in the encoder, each followed by 2×2 max-pooling layers. The first one
contains 16 filters while the others have 8 filters"; Adam, 30 epochs, batch 128. These are
*different, smaller* nets than the AE the method itself uses for `L_AE`/prototypes (3 conv + 3
deconv, 4 epochs) — deliberately, so the evaluator is not the optimiser's own model.

### The loss terms they are evaluating (context, and the thing that *does* transfer)

- `L_AE = γ · ||x0 + δ − AE(x0 + δ)||²₂` — plain reconstruction error under the **all-class** AE.
  The paper's own criticism of it [V]: *"The autoencoder loss term L_AE penalizes out-of-distribution
  counterfactual instances, but does not take the data distribution for each prediction class i into
  account."*
- `L_proto = θ · ||ENC(x0 + δ) − proto_j||²₂` (Eq. 9), with `proto_i = (1/K) Σ_{k=1..K} ENC(x_i^k)`
  (Eq. 7) — the mean encoding of the K nearest same-class instances. [V] *"It is important to note
  that the prototype is defined in the latent space, not the original feature space."* A k-d tree
  variant (Eq. 10) works in the original feature space when no encoder is available.

IM1/IM2 exist to show that `L_proto` beats `L_AE` at producing class-plausible counterfactuals.
**The metric's whole design purpose is to be class-relative** — that is precisely what makes it
inapplicable to a class-free plausibility question.

---

## 2. Their object vs ours — what would have to change, and whether it survives

| | Van Looveren & Klaise | Paper C, C2 |
|---|---|---|
| object scored | `x_cf`, a concrete **instance** | `ẑ`, an emitted **point/direction** |
| space | classifier **input** space (pixels, tabular features) | the **CF latent** space (already an encoder output) |
| paired with | a specific factual `x0` and its class `t0` | nothing — `ẑ` is emitted at an interview state |
| requires | a classifier `f_pred` inducing a discrete class partition | no classifier, no classes |
| question asked | "is `x_cf` more like class *i* than class *t0*?" | "is `ẑ` a point the data manifold could produce?" |

**Four gaps, of which two are fatal and two are merely expensive:**

1. **No classes (fatal-as-written).** Both metrics are ratios *between class-conditional AEs*. With
   one class the ratio is identically 1 and both metrics are undefined/degenerate. To instantiate
   them at all we would have to invent a class partition of the user latent space. The available
   candidates are the banked k=20 taste clusters (`m6p_clusters_k20.npz`) or the six endorsement
   cohorts — but note the standing memory finding that **cluster ≠ endorsement cohort**, and that 13
   of 20 clusters span <3 years (era structure). Choosing the partition would then *determine* the
   metric, and no choice is principled for a plausibility question.
2. **No `x0` (fatal for the answer-free framing).** `x_cf = x0 + δ` appears in every term of both
   equations. To supply one we would have to instantiate `ẑ` against real fold-in profiles — which
   turns C2's "a fact about the direction" into a per-user quantity again. It stays *answer*-free,
   so the C2 design constraint technically survives, but the metric is then no longer a property of
   the emitted point alone, which was the stated appeal.
3. **Wrong space (expensive, not fatal).** Their AEs reconstruct in the input space and the AE is a
   *separate model from the classifier*. Our `ẑ` already lives in a learned latent. Fitting AEs on
   top of a VAE latent is doubly compressive and the reconstruction error there has no
   distributional interpretation transferred from theirs. Nothing forbids it; it just means their
   reported numbers (§4) carry **zero** calibration information for us.
4. **IM2's denominator is a live hazard for us specifically (near-fatal).** IM2 divides by
   `||x_cf||₁`. Hvilshøj et al. show experimentally that IM2 **decreases monotonically as you shrink
   the instance toward its minimal value** (§5) — a near-black image beats a legible digit. Our
   emitted `ẑ` has an **unconstrained norm**, and C1 deliberately changes the target from one-hot to
   a gain-weighted blend, which will plausibly **shrink the norm**. IM2 would therefore reward the
   C1 arm for a reason that has nothing to do with plausibility. If IM2 is ever computed here it
   **must** be on norm-matched candidates, and the norm must be reported alongside.

**Transfer verdict.** *Can C2 honestly use this?* **Not as "IM1/IM2".** The design doc's instruction
"import it; do not invent one" is based on a description of the metric, not the metric. Three honest
readings, in order of preference:

- **(A) Import the idea, not the metric — recommended.** Fit one AE on **real fold-in user latents**
  and report `r(ẑ) = ||ẑ − AE(ẑ)||²₂`, positioned against (i) the distribution of `r` over real user
  latents, (ii) the **norm- and spectrum-matched** null the design doc already mandates, and (iii)
  the **correspondence-scrambled** control. Cite Van Looveren & Klaise for the `L_AE` manifold idea
  and cite Hvilshøj for why we did *not* use IM1/IM2. **Call it a reconstruction-plausibility score,
  never IM1.** This is one AE and one afternoon; it costs less than the class-conditional version
  and claims less.
- **(B) A genuine IM1 analogue — only if C2's claim becomes class-relative.** Partition the latent
  space (cohorts/clusters), fit per-partition AEs, report
  `IM1(ẑ) = ||ẑ − AE_target(ẑ)||² / ||ẑ − AE_origin(ẑ)||²`. Legitimate, and *defensible* to a
  reviewer since it is the published metric applied to a latent. But note it then answers **"is `ẑ`
  characteristic of cohort i rather than cohort t0"** — which is C2's **property 2**
  (discriminativeness) with extra machinery, and which the design doc already warns is *"nearly free
  geometry"* in a space with dominant principal components. It duplicates a property C2 has, and
  does not deliver the one it wants.
- **(C) IM2 analogue** (cohort AE vs all-user AE). Closest in spirit to "plausibility" — and the one
  with **both** published failure modes (out-of-sample blindness, magnitude monotonicity). Do not
  lead with it.

**One further honesty constraint regardless of option.** Hvilshøj et al. [V]: *"we know of no
publicly available pre-trained auto-encoders for computing the score. When new auto-encoders need to
be trained for each publication, results may not be comparable across publications."* ⇒ any absolute
IM-like value we report is a statement about our AE choice, not about our point. **Report only
differences against our own pre-registered controls; never quote an absolute number as a
plausible/implausible bar.**

---

## 3. Reported numbers — what good and bad look like *in their setting*

**MNIST** (Table 1; 1,500 counterfactuals per loss function — 10 classes × 50 test samples × 3
seeds; 95% CI). Loss functions: `A = c·L_pred + β·L1 + L2`; `B = A + L_AE`; `C = A + L_proto`;
`D = A + L_AE + L_proto`; `E = β·L1 + L2 + L_proto`; `F = E + L_AE`.

| Method | Time (s) | Gradient steps | **IM1** | **IM2 (×10)** |
|---|---|---|---|---|
| A | 13.06 ± 0.23 | 5158 ± 82 | **1.56 ± 0.03** | 1.65 ± 0.04 |
| B | 8.40 ± 0.38 | 2380 ± 113 | 1.36 ± 0.02 | 1.60 ± 0.03 |
| C | 3.06 ± 0.11 | 835 ± 36 | 1.16 ± 0.02 | 1.09 ± 0.02 |
| D | 2.31 ± 0.04 | 497 ± 10 | 1.21 ± 0.02 | 1.26 ± 0.03 |
| E | 1.93 ± 0.10 | 777 ± 44 | **1.10 ± 0.02** | 1.10 ± 0.03 |
| F | 4.01 ± 0.05 | 1116 ± 14 | 1.19 ± 0.02 | 1.27 ± 0.03 |

**Breast Cancer Wisconsin (Diagnostic)** (Table 2; 19 test instances × 5 seeds; `A` as above,
`B = A + L_proto`, `C = β·L1 + L2 + L_proto`):

| Method | Time (s) | Gradient steps | **IM1** | **IM2 (×10)** |
|---|---|---|---|---|
| A | 2.68 ± 0.20 | 2752 ± 203 | **2.07 ± 0.16** | **7.65 ± 0.79** |
| B | 0.35 ± 0.03 | 253 ± 33 | 0.94 ± 0.10 | 1.47 ± 0.15 |
| C | 0.22 ± 0.02 | 182 ± 30 | **0.88 ± 0.10** | 1.41 ± 0.15 |

**Single-instance illustration** (Fig. 2, MNIST) [V]: an uninterpretable counterfactual has
"an IM1 value of **1.81** compared to **1.04**" for the interpretable one; "The IM2 value of x_cf,1
is higher as well—**0.15** compared to **0.12**".

**Reported relative improvements** [V]: MNIST — IM1 improves "13% and 26% as L_AE and L_proto are
added", removing `L_pred` (E) gives "an improvement over A of 29%"; IM2 — `L_proto` "brings IM2 down
by 34% while the combination of L_AE and L_proto only reduces the metric by 24%". Breast Cancer —
`L_proto` "reduces IM1 and IM2 by respectively 55% and 81%".

**Calibration takeaways for us:**
- The whole discriminating range of IM1 across both datasets is roughly **0.88 – 2.07**, and
  method-level effect sizes are **13 – 55%**.
- ★ **On MNIST, even the best mean IM1 is 1.10 — above 1.** I.e. the paper's own best counterfactuals
  are, on average, *still* reconstructed slightly better by the **original**-class AE. "Good" does
  not mean "< 1" empirically. Anyone quoting "IM1 < 1 = plausible" is over-reading.
- IM2 is reported ×10 throughout; true values are ~0.088 – 0.765. Because it lacks an anchor, only
  its *ratios* are meaningful, and those ratios are contaminated by the norm effect (§2.4, §5).
- CIs are tight enough that 13% separations are significant at n=1500 — but see Hvilshøj's finding
  that on higher-complexity data the scores "get close to statistically insignificant" across
  methods.

Supporting instrument: they also run "a multiple model comparison test based on the maximum mean
discrepancy (Lim et al., 2019)" on MNIST; methods without `L_proto` (A, B) "result in high rejection
rates for faithfully modelling the predicted class distribution".

---

## 4. Venue resolution

**Resolved: ECML PKDD 2021.**

- **DBLP:** `ECML/PKDD (2) 2021`, **pp. 650–665**, DOI **10.1007/978-3-030-86520-7_40**. Springer
  LNCS **vol. 12976** (Machine Learning and Knowledge Discovery in Databases, Research Track, Part II).
- **Why the arXiv abs page could not confirm it:** the abs page carries **no `journal-ref` and no
  publisher DOI** — only the datacite `10.48550/arXiv.1907.02584`. The last version is **v2,
  2020-02-18**, i.e. it *predates* the 2021 publication and was never updated. The Comments field
  reads only "17 pages, 13 figures. For an open source implementation of the algorithm, see
  [github.com/SeldonIO/alibi]".
- **The PDF's own header/footer does NOT confirm the venue.** The running head is just the paper
  title; there is no conference banner, no page-1 copyright block, no "Under review by …" line. The
  layout is the **ICML template** (two columns, `¹Seldon Technologies Ltd, London, United Kingdom.`
  affiliation footnote + `Correspondence to: …` block, `Abstract` in a boxed head). So the artefact
  we read is an **ICML-formatted preprint that was ultimately published at ECML PKDD** — a
  submission-history fingerprint, not a venue claim.
- **Citation hygiene:** the arXiv v2 we read numbers the metrics **Eq. (13)** and **Eq. (14)**; the
  Springer version may renumber. Either cite the **Springer** paper without equation numbers, or
  cite **arXiv:1907.02584v2** *with* them. Do not mix. Most citers in this literature (incl.
  Hvilshøj) cite it as `CoRR abs/1907.02584, 2019`.
- Reference implementation: **alibi** (`github.com/SeldonIO/alibi`) — the `CounterfactualProto`
  method. Seldon's library, same authors.

---

## 5. Successors and critiques — the ones that matter before we adopt anything

**★ The direct critique — read before writing any C2 sentence that uses these metrics.**
Frederik Hvilshøj, Alexandros Iosifidis, Ira Assent, **"On Quantitative Evaluations of
Counterfactuals"**, arXiv:2111.00177 (2021-10-30). Code: `github.com/fhvilshoj/EvaluatingCounterfactuals`.
It restates IM1/IM2 correctly (their Eqs. 2 and 3 match Eqs. 13/14 above) and then lists specific
failures [V]:

> "The IM1 score can, however, be deceiving. We find experimentally that methods which make tiny
> changes to the input can get an undesired good score, presumably because tiny changes yields almost
> no error even if the changes are not preserved by the auto-encoders. Some classes may also be
> easier to reconstruct than others, resulting in skewed scores for different target classes."

> "Finally, we know of no publicly available pre-trained auto-encoders for computing the score. When
> new auto-encoders need to be trained for each publication, results may not be comparable across
> publications."

> "The applicability of the [IM2] score is however debatable. Schut et al. [24] demonstrate that the
> IM2 score fails to identify out-of-sample images. Mahajan et al. [17] also argue that both IM1 and
> IM2 are better reported by displaying both the denominator and numerator of each score. For both
> IM1 and IM2, we further find experimentally that when the complexity of the dataset increases, the
> computed scores get close to statistically insignificant amongst three different counterfactual
> methods. In turn, the two metrics may be best suited for datasets of lower complexity."

> "for the IM2 score … Figure 2c shows how the score gives an almost completely black image a better
> score than an image of an eight digit. … We demonstrate in the appendix that it holds more
> generally that the IM2 score decreases, when we decrease pixel values toward their minimal value.
> In turn, the IM2 score might wrongly give good scores to methods that produce less interpretable
> counterfactuals by removing information from the inputs."

Their concrete Fig. 2 counter-examples: **IM1 0.3 vs 0.6** and **IM2 0.000 vs 0.003**, where in both
cases the *better*-scoring image is the worse counterfactual. They partially rehabilitate IM1 [V]:
*"the metric does, however, seem to work well on average when comparing methods on MNIST. As such,
the IM1 score is best used as a summarizing statistic to compare averages of many samples across
methods."* — i.e. **IM1 is a method-level aggregate, not a per-instance plausibility score.** C2
wants a per-point score. That is a direct mismatch with our intended use.

**Proposed replacements** (Hvilshøj et al.): **Label Variation Score** and **Oracle score**, both
"less vulnerable to such tiny changes". Their conclusion is that no single metric suffices —
*"capturing all the desired properties is best done by reporting metrics concerning both realistic
changes and validity together"*. ⚠ Both replacements are **image-classifier-shaped** (the Oracle
score needs an independently trained oracle classifier over the same label set) and inherit the
class-partition requirement — so they do **not** rescue our transfer either. Their value to us is as
**the citation that justifies not using IM1/IM2**, plus the design principle: *report a
plausibility-ish score together with a validity-ish score, never one alone.* That principle is
already what C4's plug-in gap-closure does for us.

**Other named successors / relevant citers:**
- **Schut et al., AISTATS 2021** (PMLR v130), *Generating Interpretable Counterfactual Explanations
  By Implicit Minimisation of Epistemic and Aleatoric Uncertainties* — the source of the "IM2 fails
  to identify out-of-sample images" result. Its own proposal (drive down model uncertainty rather
  than fit an AE) is a genuinely different plausibility route and is the closest thing to a
  **successor** for "is this point one the model can be confident about". Worth a look if option (A)
  underdelivers, though it needs a predictive model with calibrated uncertainty over the latent.
- **Mahajan, Tan & Sharma**, arXiv:1912.03277 — argue IM1/IM2 should be reported with **numerator and
  denominator shown separately**. Cheap and we should do it under option (B)/(C) regardless.
- **FID** (Heusel et al. 2017) as the population-level realism alternative — also critiqued by
  Hvilshøj as unable to identify individual extreme samples. Population-level is arguably the right
  granularity for a *direction*, but FID needs an Inception-style embedding and does not exist for
  our latent.
- Survey-level citers that will frame a reviewer's expectations: *Counterfactual Explanations and
  Algorithmic Recourses for ML: A Review* (ACM CSUR, 10.1145/3677119), *Unifying VXAI: A Systematic
  Review and Framework for the Evaluation of Explainable AI* (arXiv:2506.15408), *Benchmarking
  Counterfactual Image Generation* (arXiv:2403.20287). **None** of these was checked for a
  latent-space IM1/IM2 instantiation; a web search for one returned nothing credible (the one
  suggestive hit was model-generated summary, not a located paper). ⇒ **We have found no precedent
  for applying IM1/IM2 in a latent space.** If we do option (B), we are the first, and we own the
  justification burden.

---

## 6. Recommended edit to `PAPER_C_DESIGN.md` §C2 (not applied here)

Replace *"On-manifold plausibility — IM1/IM2 … Import it; do not invent one"* with something
matching the above: import the **`L_AE` reconstruction-plausibility idea** with a single AE over real
fold-in latents (option A), scored against the norm/spectrum-matched null and the
correspondence-scrambled control the section already mandates; cite Van Looveren & Klaise for the
idea and **Hvilshøj et al. (2111.00177) for why the published IM1/IM2 form is not used** — that
sentence is cheap, pre-empts the obvious reviewer question, and is on-brand for a paper whose
contribution is *the audit*. If a class-relative version is wanted later, it is option (B) and it
must be sold as **discriminativeness**, not plausibility.

⚠ Standing caution from the C2 section applies unchanged and *harder* here: in a CF space with
dominant principal components, a reconstruction-error score may also be nearly free geometry.
**Pre-register the null and the pass/fail threshold before running the AE**, or this becomes another
number that is valid and claims nothing.
