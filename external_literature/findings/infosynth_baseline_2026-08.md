# Info-Synth — deep read (Paper C nearest-neighbour baseline)

> **Housekeeping:** this file was written on 2026-08-24 as a standalone finding.
> **INDEX.md and `new_chapters/references.bib` were NOT edited** — add the row and the bib entry
> separately. Nothing committed.

**Paper:** *Active Query Synthesis for Preference Learning* — Namrata Nadagouda (Georgia Tech),
Nauman Ahad (Gauss Labs), Maegan Tucker (Georgia Tech), Mark A. Davenport (Georgia Tech).
arXiv:2605.26072v1 [cs.LG], dated **25 May 2026** (title page 26 May 2026). 27 pages, 9 body + 18
appendix, 35 references.
**PDF:** `scratchpad/infosynth.pdf`, text dump `scratchpad/infosynth.txt` (page markers preserved).

---

## 5. Venue / status (answering this first because it constrains how we cite)

The footer of page 1 reads, in full: **"Preprint."** [V] There is **no venue, no submission note, no
"under review at", no acknowledgements section, no funding statement, and no code or data release
link anywhere in the 27 pages** (grepped `github`, `code`, `available`, `reproduc` — zero hits
outside the reference list). The NeurIPS-style formatting and the word "Preprint." are the
`neurips_2025.sty` default when `\usepackage[preprint]{neurips}` is set, so the layout implies a
NeurIPS-family submission but **the paper itself claims nothing**.

⚠ **Cite as an arXiv preprint only.** Do not write "NeurIPS 2026" or "under review"; there is no
evidence for either in the artifact.

---

## 1. The query object, the synthesis procedure, and the response model

### 1.1 What a query *is*

A query is an **ordered pair of points `(p, q)` in the same `d`-dimensional embedding space as the
user**, presented as "which of these two do you prefer?", answered by a **binary** `Y ∈ {0,1}`
(`Y = 1` ⇔ `p` preferred to `q`). The user is a point `w ∈ R^d` under the **ideal point model**
(Coombs 1950): *"we model preference based on distance, assuming the user prefers items that are
closer to them"* [V, Sec. A]. Belief over `w` is Bayesian, posterior sampled by **MCMC**, and the
estimate is the posterior mean `ŵ = E[W | Y_n]` [V, Sec. A].

The synthesised object is therefore **a pair of raw latent vectors** — no text, no item id, no
attribute, no catalogue lookup. There is **no renderer anywhere in the paper**.

### 1.2 THE RESPONSE MODEL — this is the circularity we point at

They start from the standard Bradley–Terry/ideal-point baseline [V, Sec. 2]:

```
P(Y = 1 | W = w) = Φ( k ( ‖w − q‖²₂ − ‖w − p‖²₂ ) )
                 = Φ( k (aᵀw − τ) ) = Φ( k‖a‖ (âᵀw − τ̂) )
with a = 2(p − q),  τ = ‖p‖² − ‖q‖²,  ‖a‖ = 2‖p − q‖,  Φ(x) = 1/(1 + e^{−x})
```

and they survey the usual choices of the noise parameter `k`: constant `k = k₀`; normalised
`k = k₀‖a‖⁻¹`; decaying `k = k₀ exp(−‖a‖)` [Canal et al. 2019]; normalised-with-probit [Chumbalov
et al. 2020] — *"highlighting different assumptions about how user uncertainty depends on query
geometry"* [V].

**Their contribution — the "confidence aware response model" (Sec. 3).** They make the *noise
variance itself* a function of the distances by writing the perceived squared distances as noisy:

```
d₁ = ‖w − p‖²₂ + e₁ ,  d₂ = ‖w − q‖²₂ + e₂
with sd(e₁) = σ₀‖w − p‖²₂  and  sd(e₂) = σ₀‖w − q‖²₂          [V, Sec. 3]
```

giving **the formula our chapter should quote — their Eq. (1)**:

```
                       ‖w − q‖²₂ − ‖w − p‖²₂
P(Y | w) = Φ( f(w) ),  f(w) = ─────────────────────────────────
                              σ₀ · sqrt( ‖w − q‖⁴₂ + ‖w − p‖⁴₂ )
```

equivalently, in midpoint/normal form, **their Eq. (2)**:

```
                              aᵀ(w − b)
P(Y | w) = Φ( ───────────────────────────────────────────────────── )
              σ₀ · sqrt( 2( ‖w−b‖²₂ + ‖a‖²₂/16 )² + ½ (aᵀ(w − b))² )
```

with `a = 2(p − q)` the hyperplane normal, `b = (p + q)/2` the midpoint, and `Φ` **any** symmetric
CDF (`Φ(−x) = 1 − Φ(x)`, `Φ'(−x) = Φ'(x)`). One free scalar hyperparameter: `σ₀`.

**★ The load-bearing observation for Paper C.** The probability that a user *can answer* a query is
a **closed-form function of Euclidean geometry alone** — the distances from the latent user point to
the two latent query points. Nothing else enters: not what `p` and `q` denote, not whether they
denote anything, not whether either point corresponds to an object the respondent has encountered,
not the modality of presentation. Their own stated motivation is exactly this geometric reading:
*"comparisons between points that are nearly identical (Fig. 1b) or entirely dissimilar (Fig. 1c) are
inherently ambiguous and yield low-confidence responses"* [V, Sec. 1] and *"When the points are too
far from each other, they might lose a common ground of comparison and again result in an unreliable
response."* [V, Sec. 2].

So **answerability is present as a modelled quantity and is defined to be a property of the
geometry.** That is a stronger position than "they ignore answerability", and our wording must
respect it (see §4).

### 1.3 The synthesis procedure

Objective: mutual information between the user point and the response,
`I(W;Y|(p,q)) = H(Y|(p,q)) − H(Y|W,(p,q))` [V, Eq. (3)], reparametrised onto `(a, b)` [Eq. (4)] and
optimised **analytically, term by term**, under a Gaussian posterior `W ~ N(µ, Σ)`:

- **Term 1** (marginal response entropy) is maximised at equiprobability, which forces the **query
  midpoint to the posterior mean**: `b̃ = (p̃ + q̃)/2 = µ` [Eq. (5)]. Proof is three lines:
  antisymmetry of `f` about `µ`, symmetry of the Gaussian, and `Φ(−x) = 1 − Φ(x)` ⇒ `E[Φ(f)] = 0.5`
  exactly.
- **Term 2** (expected conditional entropy) is minimised by maximising `E|f(W)|`. Writing
  `p̃ = µ + x`, `q̃ = µ − x`, `x = r·u`, Proposition 1 (Sec. B.3) shows the maximiser has the form
  `x̃ = r̃ v₁` where **`v₁` is the principal eigenvector of the posterior covariance `Σ`**.
- **Magnitude** `r̃`: *"Optimizing (6) for r̃ results in it being independent of σ₀. However, in
  practice we find that the optimal magnitude does depend on σ₀."* [V] — so `r̃` is obtained by a
  **1-D numerical entropy minimisation** over `r`. Final answer: **`ã = 4 r̃ v₁`**.

**In one line: the synthesised query is the pair straddling the posterior mean along the top
principal axis of the posterior covariance, at a numerically-tuned separation `r̃`.** Cost
`O(S·d²)` for `S` Monte-Carlo posterior samples — this is just the covariance estimate.

The `r̃` step is where their whole "confidence" story cashes out: *"choosing a larger value of r can
make both the numerator and denominator larger. Thus, we want to select a moderate value of r."* [V]

### 1.4 Constrained setting (the snap) — relevant to our bank-snap arm

When you cannot pose the synthesised pair, you must approximate it in a dataset `D`, and they
explicitly reject plain nearest-neighbour: *"Most works [17, 8] on synthesizing active queries select
the nearest neighbors in the dataset as the closest approximations. This works well for a single
point but for a pair of points, the resulting geometry could bear no association with the synthesized
optimal pair."* [V, Sec. 5]. Two filter-and-refine replacements:

- **Pair M-dist** (low `d`): Taylor-expand `I` at the optimum, gradient vanishes, so the loss is a
  Mahalanobis norm `ΔI(z) = ½‖Δz‖²_M` with `M = −H(z*)` the negative Hessian of the MI. Take the
  nearest `α` fraction of pairs under that metric, then rank them by true MI. Cost
  `O((S+P)d² + α·P·S·d)`, `P` = number of candidate pairs.
- **Pair Opt-dist** (high `d`, Hessian too expensive): a heuristic penalising deviation from the two
  optimality conditions `b̃ = µ` and `ã = 4r̃v₁` —
  `η(p,q,λ) = Σᵢ (bᵢ − µᵢ)²/Σᵢᵢ + λ‖(p − q) − 2r̃v₁‖²₂` [Eq. (8)] — keep the best `γ` fraction, then
  MI-rank. `λ = ζ·d / Tr(Σ)` with `ζ = 0.1`, auto-calibrated from the MCMC posterior. Cost
  `O(S·d² + P·d + γ·P·S·d)`.

Note their own reason for `ζ = 0.1`: *"enforcing the exact difference magnitude (2r̃v₁) might become
too restrictive, **starving the algorithm of valid candidate pairs**"* [V, Sec. D]. "Valid" here means
*a pair exists in the dataset near the optimum* — availability, not intelligibility. Do not let a
hostile reviewer read this as an answerability caveat; read in context it is a filter-yield issue.

---

## 2. What a replication on our space would require — and is it cheap?

### 2.1 Verdict

**Genuinely cheap as an ALGORITHM, a campaign as an EXPERIMENT — and the expensive half is exactly
the half we cannot honestly run.** Three tiers:

**Tier 1 — the synthesiser itself: half a day, no GPU. Real.**
`ã = 4r̃v₁`, `b̃ = µ` is an eigendecomposition of a `d×d` covariance plus a 1-D scalar search. Given
posterior samples this is `numpy.linalg.eigh` and `scipy.optimize.minimize_scalar`. Nothing to train.

**Tier 2 — the belief machinery it presumes: days, and it requires a modelling commitment we have
not made.**
- **An ideal-point posterior over `w`.** Info-Synth is *derived* under (a) preference = negative
  Euclidean distance in the embedding space, and (b) `W ~ N(µ, Σ)` (the Gaussian is used to get both
  optimality conditions; MCMC is only used to *estimate* `µ`, `Σ`). **Our recommender scores by
  inner product against the decoder plus a learned decoder bias, not by distance** (cf. the
  `popb`-vs-decoder-bias metric bug). Porting Info-Synth means *imposing* an ideal-point likelihood
  on a space that was not fitted as one — a substantive assumption we would be making on their
  behalf, and a reviewer-visible one. State it as an assumption in the arm's description.
- **MCMC per user per turn.** They run Stan for the gain-tuning study and "MCMC sampling" generally.
  For a 15-turn interview over even a few hundred users this is the dominant wall-clock, not the
  synthesis. Their own reported per-query selection times on the *128-dim* Reddit data are
  **3.77–6.19 s** for the main user and up to **57.33 s** (Active Discrete) / **17.55–24.19 s**
  (Pair Opt-dist) for User 2 [V, Figs. 5b/5d/11b] — i.e. **seconds to a minute per single query**,
  before any of our users exist.
- **`σ₀` must be chosen and it is not innocuous.** It is the only free parameter of the response
  model, it sets the optimal query separation `r̃`, and their synthetic results *change qualitatively
  with it*: Gauss Search Synthesis is indistinguishable from Info-Synth at `σ₀ = 0.001` and clearly
  worse at `σ₀ = 0.1` [V, Sec. 6.1]. On our space `σ₀` has **no calibration source** — there is no
  measurement in CASPER that fixes it. It is a knob we would be tuning to make their method look
  however we want. ⚠ If we run this arm, sweep `σ₀` and report the sweep; a single value is
  indefensible.

**Tier 3 — the constrained arm on our catalogue: this is where the cost blows up.**
Pair Opt-dist / Pair M-dist are `O(P·d)` and `O(P·d²)` in the number of **candidate pairs**. Our
item vocabulary is **18,430 items ⇒ P ≈ 1.70 × 10⁸ unordered pairs**, per user, per turn. Their
largest reported `P` comes from **500 points** (synthetic) and a single user's pool on Reddit.
**We are 3–4 orders of magnitude beyond their demonstrated scale**, and the `α`/`γ` fractions that
make the method cheap are exactly what they never report numerically (see 2.3). Any honest run needs
a blocked/ANN candidate generator that they do not describe.
⚠ Per the standing rule on this box: **bench the assembled path, not the eigendecomposition.**

**Tier 4 — the answers. This is the wall.**
Info-Synth needs a *response* to a **synthesised pair of latent points**. We have no non-circular
source for that. Simulating the answer with the same geometry the synthesiser optimised over is the
textbook version of the circularity our project already ruled on (`answer-simulator-circularity-
ruling`) — and here it is worse than usual, because **their response model IS the acquisition
criterion's own objective**: the simulator would be `Φ(f(w))` and the selector maximises MI under
`Φ(f(w))`. That arm cannot fail. It is a self-consistency check dressed as an experiment.

### 2.2 Recommended shape of the arm (cheap, honest, and it still earns its sentence)

Run Info-Synth **answer-free**, as the "emit freely with no answerability constraint" foil:
synthesise `(p̃, q̃)` on our latent under a stated ideal-point posterior, then **measure the emitted
points' properties, not their payoff** — how far off-catalogue they land, what the nearest bank
question is, what the SBERT→CF adapter makes of them, and what the snap costs in MI. That is C1/C2
territory, needs no answerer, and sidesteps Tier 4 entirely. **Do not claim Info-Synth helps or
hurts NDCG on our data; we cannot measure that without an answerer we have already disqualified.**

### 2.3 Reproducibility gaps (list these if a reviewer asks why our numbers differ)

Never stated anywhere in the paper: the MC sample count **`S`**; the filter fractions **`α`**,
**`β`**; the `k` in k-NN Approx; MCMC chains/warmup/draws; the prior `p₀(W)`; the link `Φ` actually
used in experiments (defined only abstractly as "the CDF of a symmetric noise distribution", while
Sec. 2's baseline uses the logistic); the number of synthetic trials is **5** [V, Sec. E.1.1] and the
Reddit study uses **3 users** total. **No code release.** A replication is therefore a
re-implementation from the equations, not a re-run.

---

## 3. Evaluation — simulated only, or any real respondent?

**Simulated only. There is no human respondent anywhere in this paper.** No user study, no
participants, no annotators recruited, no IRB/ethics statement, no consent language, no compensation
note. Three settings:

1. **Synthetic** (Sec. 6.1 / E.1): items `~ U[−4,4]^D`, user point `~ U[−1,1]^D`, `D = 4` and `10`,
   `N = 100` or `500` points, **5 trials**, 100 queries. Responses generated from their own model.
   Metrics: MSE to the true point, Kendall-tau distance on user–item distances.
2. **Reddit TL;DR summaries** (Sec. 6.2 / E.2): *offline replay of an existing preference dataset* —
   Stiennon et al. 2020 labels, embedded with OPT-350M + a frozen PAL projection head, **512-dim
   reduced to 128 by PCA**. *"We selected a user at random from this dataset"* [V]; three users total,
   with **1874 / 1010 / 837** existing query responses. Metric: preference-prediction accuracy.
   **The humans here answered in 2020, to pairs that already existed; nobody was ever shown anything
   this paper synthesised.**
3. **Gain tuning** (Sec. 6.3 / E.3): unicycle robot in simulation. And it is stated outright —
   *"**Instead of a real human, we use an automated oracle** that evaluates the simulated trajectories
   based on the trajectory tracking error (computed using Hausdorff distance). To simulate real human
   inconsistency, the oracle uses a sigmoid function to generate a preference probability"* [V, Sec.
   E.3.3]. Fig. 2's caption likewise calls the responder *"The oracle"* [V].

★ **The sharpest fact in the whole read, and the one to put in the chapter:** the paper has exactly
one *semantic* embedding space — the OPT-350M summary space — and **in that space the synthesised
query is never posed.** The Reddit experiments run entirely through **Pair Opt-dist**, i.e. they snap
to real summaries from the dataset before anything is "asked". The **continuous** setting, where the
synthesised point `(p̃, q̃)` is used directly, is exercised only on (i) uniform random synthetic
vectors and (ii) controller gains — and a gain vector is directly instantiable: you *run the robot
with it*. **The paper never once presents a synthesised point in a space where presentation would
require rendering it into something a person could perceive.** That is verifiable by any reviewer
from Secs. 6.1–6.3, and it is the gap Paper C occupies.

---

## 4. Do they discuss answerability / validity of synthesised queries?

### 4.1 The honest answer

**No — but the naive absence claim will not survive a hostile reviewer, because the paper's headline
contribution is *about* response reliability.** Anyone who has read it can point at the "confidence
aware response model", at Fig. 1, and at the related-work paragraph on cognitive load. We must claim
the *right* absence.

### 4.2 Everything in the paper that comes close (so nothing ambushes us)

- *"comparisons between points that are nearly identical (Fig. 1b) or entirely dissimilar (Fig. 1c)
  are inherently ambiguous and yield low-confidence responses, in contrast to ideal queries
  (Fig. 1a)"* [V, Sec. 1]
- *"When the points are located too close to each other, it might be hard for the user to select one
  of them and the response can be ambiguous. When the points are too far from each other, they might
  lose a common ground of comparison and again result in an unreliable response."* [V, Sec. 2]
- *"most methods assume all query feedback is equally reliable, ignoring that pairwise queries between
  nearly identical or entirely dissimilar items yield ambiguous, low-confidence responses"* [V,
  Abstract]
- *"The method proposed in [20] employs a specific information gain criterion that balances the
  uncertainty of the query and **the human ability to provide a response** to encourage selection of
  high confidence queries. In this work, we instead propose a modified response probability model
  that **not only doesn't increase the cognitive load for responding** but is also amenable to any
  active learning algorithm."* [V, Sec. 2.1 — ref. [20] is Bıyık et al., *Asking Easy Questions*,
  arXiv:1910.04365]
- *"potentially improving performance by **generating data points that do not currently exist in the
  dataset**"* [V, Sec. 1] — the one sentence acknowledging that a synthesised query may correspond to
  no real object. **It is offered as an advantage and is never revisited.**
- *"If we have the flexibility to query any pair of points, we can query the synthesized pair of
  points directly. Else, we need to approximate them with the best points in the dataset."* [V,
  Sec. 5] — the *entire* treatment of when a synthesised query can actually be posed. The condition
  is **dataset membership**, and the fix is snapping. Whether the pair *means* anything, whether a
  respondent could perceive it, and how it would be shown are not raised.
- *"starving the algorithm of valid candidate pairs"* [V, Sec. D] — "valid" = passes the geometric
  filter and exists in `D`. Not a validity-of-question claim.

**Absent from the paper entirely** (grepped): *answerab-*, *interpretab-* (of a query), *plausib-*,
*realistic*, *meaningful*, *semantic*, *render*, *verbal*, *describe/description* (of a synthesised
point), *circular*, and any discussion of how a synthesised `(p̃, q̃)` in the OPT-350M summary space
would be shown to a person.

### 4.3 Verdict, and the exact wording our chapter may safely use

The absence is **narrow, real, and survives**: they model *response confidence* as a function of
**query geometry only**, and never ask whether the synthesised query is a thing that can be posed.
Safe formulations, in descending strength — all four are checkable line-by-line against the paper:

> **(A) — the one-liner.** "Info-Synth models the reliability of a comparison as a closed-form
> function of the query's geometry — the distances from the user point to the two synthesised points
> — and never as a function of whether the synthesised query denotes anything a respondent could be
> shown."

> **(B) — the evidence sentence.** "The point is not that Info-Synth ignores response quality: its
> headline contribution is a confidence-aware response model built precisely to avoid pairs that are
> too close or too far apart. The point is where that model locates the difficulty. Confidence is
> `Φ(f(w))` with `f` a ratio of Euclidean distances and a single noise scale `σ₀`; nothing in it
> refers to the content of `p` or `q`."

> **(C) — the experimental gap (strongest, hardest to rebut).** "In the only semantically-embedded
> space they study — text summaries encoded with OPT-350M — the synthesised query is never posed; it
> is snapped back onto real summaries by Pair Opt-dist before any response is obtained. The
> continuous setting, in which the synthesised pair is used directly, is exercised only on uniform
> random synthetic vectors and on controller gains, which are instantiable by simulation. The
> question of how a synthesised point in a semantic space would be presented to a respondent does
> not arise in their experiments, and is not discussed."

> **(D) — the respondent.** "No human answers any query in the paper: the synthetic and robotics
> studies use a simulated oracle — explicitly, 'Instead of a real human, we use an automated oracle'
> — and the text experiments replay preference labels collected by Stiennon et al. (2020) over pairs
> that already existed."

**Do NOT write** any of the following — each is false against the artifact and a reviewer will say so:
- ❌ "they ignore the reliability of responses" / "they assume all queries are equally answerable"
  (the confidence-aware model is the paper's contribution #1);
- ❌ "they do not consider that a synthesised point may not exist in the dataset" (Sec. 5 is exactly
  that, and Sec. 1 flags off-dataset generation as a feature);
- ❌ "no discussion of the human's ability to respond" (Sec. 2.1 cites Bıyık et al. on precisely this
  and positions against it on cognitive-load grounds);
- ❌ "NeurIPS 2026" or any venue — it says "Preprint." and nothing else.

---

## 5b. One-paragraph positioning draft (for PAPER_C_DESIGN Baselines)

Info-Synth (Nadagouda et al., arXiv:2605.26072) is the closest published relative of our emitter: it
abandons the pool, synthesises comparison queries directly in a continuous embedding space by
maximising mutual information, and — uniquely among synthesis methods — models how *confidently* the
query will be answered. But its confidence is geometric: `P(Y|w) = Φ(f(w))` with `f` the normalised
difference of squared distances from the user point to the two synthesised points, scaled by a single
free `σ₀`. A query is "hard to answer" when its two points are too close together or too far apart in
the latent, never when it fails to denote anything. Consistent with that, the synthesised query is
posed directly only where a latent point is itself instantiable — random synthetic vectors and robot
controller gains — while in their one semantic space, OPT-350M summary embeddings, it is snapped back
onto real dataset items before any response is collected, and every response in the paper comes from
an automated oracle or from labels collected years earlier. Running Info-Synth on our latent gives us
the *emit-freely, no-answerability-constraint* arm; the renderer and the answerability question are
what we add.
