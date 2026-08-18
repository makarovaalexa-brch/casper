# Continuous / embedding-space preference elicitation — deep dive, 2026-08-18

*Brief served: `docs/design/PAPER_C_TODO.md` STEP A + the parent agent's Q1–Q5.*
*Read WITH `continuous_action_policy.md` (the machinery), `llm_elicitation_and_metalearning.md` §6
(the LDPP / ELM / Info-Synth sweep — NOT repeated here), `simulator_circularity_conv_rec.md`
(the firewall), `active_learning_elicitation.md` (BED/CAT portability).*

**Provenance flags — binding.** `[V]` full text read · `[A]` abstract / landing page, or a
tool-summarised fetch of the paper's own text · `[S]` secondary (search summary, third-party
description). **No number appears here that was not seen at the flagged level.** Anything at `[S]`
is a lead, not a citable fact.

**This pass's job was the cells earlier passes did not cover:** the *answer model* for a synthesised
query, continuous design spaces *outside* recsys where a human must answer, and how anyone measures
that a synthesised latent point is *meaningful*. The recsys/LLM sweep in
`llm_elicitation_and_metalearning.md` §6 is not redone — its verdicts stand and are cross-referenced.

---

## ★ THE ONE-PARAGRAPH ANSWER

The cell we thought was empty is **empty only under a specific restriction, and the restriction is
the contribution**. Synthesising a query as a continuous point and putting it to a **real human** is
a mature, 30-year-old practice — in psychophysics, adaptive design optimisation, preferential
Bayesian optimisation and preference-based robot learning, where the synthesised design is
**physically rendered** (a stimulus is displayed; an exoskeleton walks the gait). Query synthesis for
a human oracle **in a semantic space** is equally old and is *the field's canonical failure*: Baum &
Lang (1992) named it, Zhu & Bento (2017) reproduced it with GANs, and the standard fixes are exactly
the two moves we are debating — **snap to a real neighbour** (local membership queries; NN query
synthesis; Info-Synth 2026) or **verbalise the synthesised point** (GenMQD 2024). So Paper C's live
question is not "does anyone synthesise queries" (yes, everywhere) but "**can a synthesised point in
a collaborative-filtering latent be made answerable by a person, and how would you know**" — and
*that* is genuinely unoccupied.

---

# (a) DIRECT ANSWERS

## Q1 — Learned policy emits a latent point, a user answers it, in a loop. Is the cell empty?

**VERDICT: EMPTY for *semantic/CF* latents rendered as language. NOT empty for latents that are
physically executable. Our phrasing must carry the qualifier.**

Evidence *against* an unqualified "nobody does this":

- **CoSpar / LineCoSpar (Tucker, Novoseller, … Ames; ICRA 2020 / IROS 2020)** `[S]`. A posterior over
  a utility function on a **continuous gait-parameter space**; the algorithm proposes a **point that
  comes from no pool**, the exoskeleton **walks it**, and a **real human subject** gives a pairwise
  preference. Reported: **6 gait parameters, 6 able-bodied subjects, 30 gait trials (≤29 pairwise
  preferences + 30 coactive-feedback items)** `[S — search summary of the IROS paper; verify against
  the PDF before quoting]`. A full closed loop over synthesised continuous points with a real user.
  *The communication channel is execution, not language.*
- **Adaptive Design Optimization (Cavagnaro, Myung, Pitt; ADOpy, Behav. Res. Methods 2021)** `[S]`.
  Mutual-information-optimal **stimulus** chosen per trial over a continuous design grid, presented
  to a **real participant**. Same structure: the design is directly perceivable, so answerability is
  free.
- **Preferential Bayesian Optimization (González, Dai, Damianou, Lawrence, ICML 2017)** `[S]` and
  **PABBO (Zhang, Huang, Kaski, Martinelli, ICLR 2025, arXiv:2503.00924)** `[A]`. PABBO **amortises**
  surrogate and acquisition and emits a **continuous design pair in one forward pass** (transformer
  neural process, RL-trained). This is DAD-for-preferences and the closest *architecture* to "the
  policy outputs the query directly". No human experiments evidenced in the abstract `[A]`.
- **Info-Synth (Nadagouda, Ahad, Tucker, Davenport, arXiv:2605.26072)** `[A]` — already on record;
  this pass adds the author list and the detail that the **continuous setting uses the synthesised
  query directly** ("used directly"), and only the *constrained* setting snaps to dataset items via
  **Pair M-dist / Pair Opt-dist** `[A]`. So Info-Synth is *both* halves: unsnapped where the domain
  permits, snapped where it does not.

Evidence the *semantic* version is unoccupied — unchanged and reconfirmed: ELM decodes synthesised
points with **no loop and no policy**; LDPP's continuous action is the convex hull of a **strategy
codebook** with no user belief; PEBOL / GATE / OPEN / CLARINET / BED-LLM / COPE all **select then
phrase**; **Pep** (arXiv:2602.15012, Feb 2026) maintains a **posterior over a latent user variable**
but its action is verbatim `a_t ∈ 𝒞(x)∖{a_1..a_{t-1}}` — **selection from a fixed criterion list**
`[A]`. Pep is the newest and most dangerous-looking near-miss (latent posterior + multi-turn
questioning + cold start) and it is **discrete**.

**⇒ The sentence we may write:** *no published system has a learned policy emit a point in a
recommender's own latent space, verbalise it, and fold a user's answer back into that same space.*
**The sentence we may NOT write:** "nobody asks a user about a synthesised continuous query."

## Q2 — Answer models in continuous-action elicitation, and is circularity named?

**VERDICT: The field has exactly ONE serious answer model for synthesised continuous queries, it is
GEOMETRIC, and it is circular by our own standard — and nobody names the circularity. But the field
DOES name the adjacent problem (unanswerable synthesised queries), loudly and early.**

1. **Info-Synth's confidence-aware response model** `[A, formula seen]` is the state of the art, and
   it is the design we would otherwise have invented ourselves:
   `P(Y|w) = Φ( (‖w−q‖² − ‖w−p‖²) / (σ₀ √(‖w−q‖⁴ + ‖w−p‖⁴)) )` — a heteroscedastic probit whose
   noise **grows with the query's geometry**, encoding "responses are unreliable when the two items
   are nearly identical or wildly dissimilar", so the acquisition prefers pairs at **moderate
   distance**. Justification offered: empirical observation about human responses, plus a
   heteroskedastic-regression argument `[A]`.
   **This is a geometric simulator with an extra noise term.** The answer is still a deterministic
   function of the same latent `w` the model is estimating. Under our Jul-25 ruling it is circular;
   the paper does not treat that as a validity issue. Its own human-ish evidence is thin: the Reddit
   TL;DR experiment uses **one randomly selected real user's actual labels** `[A]`; the robot
   experiment has **no human subjects** and simulates responses from tracking error `[A]`.
2. **Nobody in this line names simulator circularity.** The named critiques we hold — PEPPER
   target-biased shortcutting (arXiv:2411.16160) and Panickssery et al. self-preference (NeurIPS
   2024) — remain the only load-bearing citations, and both come from *outside* continuous
   elicitation. A targeted search for "the model asks and answers" framings returned only generic
   LLM generator-evaluator-agreement work `[S]`, nothing on synthesised-query self-grading.
   **⇒ Our circularity ruling is ahead of this literature, not behind it. Safe to state as an
   observation about the field, and it is itself a contribution.**
3. **The field's *real* named problem is answerability, and it is 34 years old** — see Q4. Honest
   framing: *preference elicitation worries about answer NOISE (Info-Synth); active learning worries
   about answer IMPOSSIBILITY (Baum & Lang). Our setting inherits the second, and the second has no
   accepted solution.*

## Q3 — Continuous-action RL where the action must be COMMUNICATED to a human

**VERDICT: Essentially nothing new beyond what we hold. The 2024–26 large-action-space RL literature
moved AWAY from continuous proto-actions toward generative/combinatorial policies, and none of it
communicates its action to a person.**

- Sweep of 2024–26 large-discrete-action RL surfaced: **discrete diffusion policies for combinatorial
  action spaces (arXiv:2509.22963)**, **SAINT: attention policies for discrete combinatorial actions
  (arXiv:2505.12109)**, **stochastic Q-learning for large discrete action spaces (arXiv:2405.10310)**,
  **unimodal-distribution discretisation of continuous actions (arXiv:2408.00309)**, **geometry of
  neural RL in continuous state/action spaces (arXiv:2507.20853)** — all `[S]`, all machine-executed
  actions. **None is a Paper C baseline; one background sentence at most.**
- The Wolpertinger → HyAR → P-DQN lineage is unchanged; `continuous_action_policy.md` stands.
- The only "action → human" instances remain **ACING** (soft prompt → instruction), **VDLM**
  (embedding planner + vec2text) and **LDPP** (codebook blend → P-Former → frozen LLM) — all already
  on record, none with a user answering.
- **New framing available:** in *robotics preference learning* the action IS communicated to the
  human — by execution (CoSpar; PBO controller tuning `[S]`). Two-sentence positioning:
  *continuous queries reach humans wherever the design can be rendered by the world; recommendation
  has no such renderer, which is why the field snapped to a catalogue. We propose language as the
  missing renderer.*

## Q4 — Continuous design spaces that must be made human-answerable, outside recsys

**VERDICT: OCCUPIED and important — and it contains the single best citation this pass found.**

1. **★ Baum & Lang 1992, "Query learning can work poorly when a human oracle is used", IJCNN**
   `[S — canonical; described consistently across the AL survey literature]`. Membership-query
   synthesis generated images that were **fusions of two digits (e.g. a 3/5 hybrid)** which a human
   **cannot label at all**. This is *our* failure mode, stated in 1992, and it is why pool-based AL
   displaced query synthesis. **Open Paper C's answerability section with it.** It also retro-justifies
   the snap on exactly the ground `continuous_action_policy.md` already chose (askability /
   answerability, not "no decoder exists").
2. **Zhu & Bento 2017, "Generative Adversarial Active Learning", arXiv:1702.07956** `[S]`. First GAN
   query synthesis, explicitly **intended for expert annotation**; the reported outcome is that
   generated samples were **low quality and of indistinguishable class** — Baum & Lang reproduced
   with deep generative models. **The empirical precedent that a plausible-looking synthesised query
   can still be unanswerable — i.e. the exact need for our scrambled control.**
3. **Awasthi, Feldman & Kanade 2012 — local membership queries (COLT)** `[S]`. The theory fix:
   restrict synthesised queries to a neighbourhood of real examples, recovering learnability while
   keeping queries recognisable. **Our snap / trust region, with a learning-theory pedigree.**
4. **"Active learning via query synthesis and nearest neighbour search", *Neurocomputing* (2014/15),
   S0925231214008145** `[S — author list unverified]`. Synthesise the optimal query, then label the
   **nearest real instance**. **Wolpertinger's snap, in active learning, ten years before Info-Synth
   did it again.** Together they make "everyone snaps" a three-field claim.
5. **★ GenMQD — "Semantic-aligned Query Synthesis for Active Learning" (OpenReview `CZvbXXgjrn`,
   Oct 2024)** `[S — ⚠ UNVERIFIED; OpenReview served a bot-check page]`. As reported second-hand:
   **generate a textual DESCRIPTION of the wanted instance rather than the instance itself**, using a
   CLIP-style multimodal alignment to turn a synthesised feature direction into natural language used
   to gather matching real data. **If that is accurate, it is the closest published instance of
   "verbalise the synthesised query so a human can act on it" — our mechanism, in vision-AL rather
   than elicitation.** Highest-value verification debt from this pass.
6. **Zarecki & Markovitch, "Textual Membership Queries", IJCAI 2020, arXiv:1805.04609** `[A]`.
   Synthesises text queries by **modification operators over a small labelled core set** plus search
   — answerability is bought by *editing real text*, never by decoding a free vector. Evaluated on
   several text-classification tasks; a human-annotator study is **not evidenced** in the fetched page
   `[A]`. **The conservative alternative to our decoder: name it, and say why we do not take it (it
   cannot express a direction no phrase names).**
7. **Psychometrics — automatic item generation (AIG) / on-the-fly CAT** `[S]`. The field's answer to
   "you cannot calibrate a question you just invented" is **item-MODEL calibration**: generate items
   from templates ("item models"), calibrate the *family* with a multilevel model, treat instances as
   nested draws; plus **human expert review of every AI-generated item before use** `[S]`.
   **Transferable:** if we emit a novel direction we can claim properties of the *family/region*, not
   of the instance — and human review is the field-normal validation.
8. **Conjoint analysis — polyhedral question design (Toubia, Simester, Hauser, Dahan, *Marketing
   Science* 2003; Toubia, Hauser, Simester, *JMR* 2004)** `[S]`. Each question is **constructed** to
   shrink the feasible parameter polyhedron fastest; estimation by the **analytic centre** of that
   polyhedron. Answerability is bought structurally: a question is a **profile built from real
   attribute levels**, so any synthesised design is automatically describable. **Marketing's version
   of "the snap can't fail" — evidence our snap justification is the field-normal one.**
9. **Vendrov, Lu, Huang & Boutilier, "Gradient-Based Optimization for Bayesian Preference
   Elicitation", AAAI 2020 (arXiv:1911.09153)** `[S — two PDF extractions failed to yield reliable
   method detail]`. Continuous, differentiable relaxation of EVOI optimised by gradients for large
   item spaces, including **queries over subsets/slates**. Secondary reading: the relaxation is over
   **item-selection weights**, **projected back to real items** — which would make it *soft selection*,
   not synthesis. ⚠ **Do not state the mechanism until the PDF is read properly.** It is the nearest
   recsys-side precedent for "optimise the query continuously", and Boutilier is a likely reviewer.
10. **Latent-space Bayesian optimisation (LSBO)** — **Tripp, Daxberger & Hernández-Lobato, weighted
    retraining (NeurIPS 2020)**; **LOL-BO (Maus et al., NeurIPS 2022, arXiv:2201.11872)** `[S]`. The
    design lives in a VAE latent and must **decode to a valid object**; accepted fixes are **trust
    regions in latent space + retraining the generative model toward the promising region**. `[S]`
    reports LOL-BO "as much as 20×" over prior LSBO on six benchmarks — **do not quote unread.**
    Closest methodological family to "emit a latent point that must decode into something real", and
    a better technical neighbour for that argument than the RL action-space line.

## Q5 — Does anyone measure whether a synthesised latent point is MEANINGFUL, not merely decodable?

**VERDICT: YES, in three separate literatures, none of them recsys. We need not invent the metric —
we should import it and name its parent.**

1. **ELM's consistency metrics** — already on record, still the only *recommendation-domain* instance
   `[V, prior pass]`.
2. **Generative-model interpolation quality.** **Perceptual Path Length (Karras, Laine & Aila,
   StyleGAN, CVPR 2019)** `[S]`: subdivide the interpolation path and sum perceptual distance between
   adjacent points; a smooth latent gives low PPL, an entangled one shows **features appearing
   mid-path that exist at neither endpoint**. That phenomenon is exactly what our novel-direction
   claim must distinguish from noise. Related: **Arvanitidis, Hansen & Hauberg, "Latent Space Oddity"
   (ICLR 2018)** `[S]` — a Riemannian metric pulled back through the decoder, so *geodesic* rather
   than linear interpolation stays on-manifold. **⇒ If we interpolate linearly in a CF latent, a
   reviewer from this literature will ask why not geodesically. Have an answer.**
3. **★ Counterfactual-explanation plausibility — the best-fitting import.** **Van Looveren & Klaise,
   "Interpretable Counterfactual Explanations Guided by Prototypes" (ECML PKDD 2021,
   arXiv:1907.02584)** `[S]` define **IM1** = reconstruction error of the point under an autoencoder
   trained on the target class ÷ error under one trained on the original class, and **IM2** =
   discrepancy between the target-class AE reconstruction and an all-data AE reconstruction; **lower
   = more on-manifold = more plausible** `[S]`. This is a **user-free, answer-free plausibility score
   for a synthesised point** — exactly the shape the author's circularity ruling demands (validation
   must be a property of the POINT). **Recommendation: adopt an IM1/IM2 analogue over the CF latent
   as validation check 1b and cite the parent.** The whole counterfactual-XAI field is organised
   around "a generated point must be plausible, not merely classified correctly" (survey: *ACM CSUR*
   2024, DOI 10.1145/3677119 `[S]`) — a ready-made vocabulary for "meaningful vs merely decodable".
4. **Query-synthesis AL's own version** is the *negative* result (Baum & Lang; Zhu & Bento): the
   measurement is a **human labelling attempt that fails**. The strongest possible control, and the
   most expensive.

**Not found anywhere:** a metric for whether a synthesised point is meaningful *as a question* — i.e.
whether a population of users would answer it *differently from one another*. The author's proposed
"discriminative over real users" check (project real fold-in profiles onto the direction; look at the
spread) **appears unclaimed and is the most defensible novel piece of methodology in Paper C.** It is
also the only check that uses real behaviour and no answerer.

---

# (b) PER-PAPER ENTRIES

Only papers **new to the project or materially updated by this pass**. Everything in
`continuous_action_policy.md` and `llm_elicitation_and_metalearning.md` §6 is unchanged.

### Baum & Lang 1992 — "Query learning can work poorly when a human oracle is used"
IJCNN 1992 (Baltimore). `[S]`
- **Mechanism.** Neural-net membership-query learning synthesises input points at the decision
  boundary and asks a human to label them; the synthesised digits are fusions (a 3/5 hybrid) that
  correspond to no class.
- **Evaluation.** Handwritten-digit recognition; outcome reported qualitatively — human oracles could
  not label a large fraction of synthesised queries. **No numbers seen; quote none.**
- **Use: CITATION-ONLY, and load-bearing.** The canonical statement of our answerability risk and the
  historical reason the field snaps.

### Zhu & Bento 2017 — "Generative Adversarial Active Learning"
arXiv:1702.07956 (v5, Nov 2017). `[S]`
- **Mechanism.** A GAN generator synthesises instances near the classifier's uncertainty region;
  those synthetic instances are what gets labelled.
- **Evaluation.** Image classification, compared against pool-based AL; described in the survey
  literature as producing **low-quality, class-ambiguous** samples. No verified numbers.
- **Use: CITATION-ONLY.** Modern proof that a *generative* renderer does not solve answerability —
  the reason our brief demands a scrambled control.

### Awasthi, Feldman & Kanade 2012 — learning with local membership queries
COLT 2012. `[S]`
- **Mechanism.** Restrict synthesised queries to a small ball around real examples, recovering
  learnability while keeping queries recognisable.
- **Use: CITATION-ONLY.** The principled ancestor of "stay near the manifold" / trust-region snapping.

### "Active learning via query synthesis and nearest neighbour search"
*Neurocomputing* (2014/2015), ScienceDirect S0925231214008145. `[S — authors unverified]`
- **Mechanism.** Synthesise an optimal query point, then label the **nearest real instance**.
- **Use: CITATION-ONLY, high value.** Establishes the snap independently of Wolpertinger, inside
  active learning.

### GenMQD / "Semantic-aligned Query Synthesis for Active Learning"
OpenReview `CZvbXXgjrn`, Oct 2024. `[S — ⚠ UNVERIFIED]`
- **Mechanism (second-hand).** Synthesise a **textual descriptor** of the wanted instance rather than
  the instance; a CLIP-style aligned model turns the synthesised feature direction into natural
  language used to gather matching real data.
- **Use: potentially the NEAREST NEIGHBOUR for our renderer argument outside elicitation.**
  **Verification debt #1** — obtain the PDF, confirm venue and whether humans act on the descriptions.

### Zarecki & Markovitch 2020 — "Textual Membership Queries"
IJCAI 2020; arXiv:1805.04609. `[A]`
- **Mechanism.** Instance space treated as a search space; **modification operators** applied to a
  small labelled core set synthesise new *text* examples relevant to the current learner.
- **Evaluation.** Several text-classification tasks; classifier improves as more synthesised queries
  are labelled `[A]`. Real annotators **not evidenced** in the fetched page.
- **Use: CITATION-ONLY.** The text-domain answer to Baum & Lang: edit real language, never decode a
  free vector.

### Tucker, Novoseller, Kann, Sui, Yue, Burdick & Ames 2020 — CoSpar
ICRA 2020; arXiv:1909.12316. `[S]`
### Tucker, Cheng, Novoseller, Cheng, Yue, Burdick & Ames 2020 — LineCoSpar
IROS 2020; arXiv:2003.06495. `[S]`
- **Mechanism.** Bayesian posterior over a utility function on a continuous gait-parameter space;
  LineCoSpar samples the posterior along iteratively chosen **1-D subspaces** to scale past the
  dimensionality where a grid is hopeless. Queries are points, not catalogue entries.
- **Evaluation.** LineCoSpar: **6 gait parameters, 6 able-bodied human subjects, 30 gait trials per
  subject (≤29 pairwise preferences + 30 coactive-feedback items)** `[S — verify in the PDF]`. Users
  are **real**, so the simulator question does not arise, and there is no ground-truth utility.
- **Use: CITATION-ONLY, structurally essential.** Existence proof that a synthesised continuous query
  can be answered by a person **when the world can render it** — the sentence that makes our
  restriction precise instead of overclaiming.

### Cavagnaro, Myung, Pitt et al. — Adaptive Design Optimization; **ADOpy**
Yang, Pitt, Ahn & Myung, *Behavior Research Methods* 2021. `[S]`
- **Mechanism.** Mutual-information-optimal design (model discrimination or parameter estimation)
  chosen per trial over a design grid, with a grid/particle posterior over cognitive-model parameters.
- **Evaluation.** Psychometric-function estimation, delay discounting, risky choice; real
  participants. `[S]` claims delay-discounting *k* estimated in "fewer than 20 trials", "3–5× more
  precise than the staircase method" — **do not quote without reading the source.**
- **Use: CITATION-ONLY.** The human-subject BED tradition; the cleanest precedent for "continuous
  design space, directly perceivable design".

### González, Dai, Damianou & Lawrence 2017 — Preferential Bayesian Optimization (ICML) `[S]`
### Zhang, Huang, Kaski & Martinelli 2025 — PABBO, ICLR 2025, arXiv:2503.00924 `[A]`
- **Mechanism.** PBO optimises a latent utility observable only through **duels**, with
  Bradley–Terry–Luce / Thurstone–Mosteller likelihoods `[S]`. PABBO **fully amortises** surrogate and
  acquisition into a transformer neural process trained with RL plus auxiliary losses, emitting the
  next **query pair in one forward pass**, "several orders of magnitude faster" than GP-based PBO `[A]`.
- **Evaluation.** PABBO: "a benchmark composed of synthetic and real-world datasets"; **no human
  experiments evidenced**; response model not stated in the abstract `[A]`.
- **Use: CITATION-ONLY (PBO) / ARCHITECTURAL reference (PABBO).** PABBO pre-empts any claim that
  *amortisation* of preference acquisition is our novelty.

### Nadagouda, Ahad, Tucker & Davenport — Info-Synth (updated entry)
"Active Query Synthesis for Preference Learning", arXiv:2605.26072, 27 pp., 12 figs. `[A, incl. quoted formula]`
- **Mechanism.** Maximise a mutual-information objective **over a continuous query space** to
  synthesise a pairwise query; a **confidence-aware heteroscedastic probit response model** makes
  answer noise depend on the query's own geometry (identical or wildly dissimilar pairs are
  unreliable). Two pool-restricted variants — **Pair M-dist** (Mahalanobis, low-d) and **Pair
  Opt-dist** (optimality-criterion heuristic, high-d) — snap the synthesised pair onto dataset items.
- **Evaluation.** (i) synthetic Gaussian clouds in **4-D and 10-D**, responses simulated from the
  model with a **known ground-truth user point w**; (ii) **Reddit TL;DR summaries**, **128-D** space
  from **OPT-350M + PCA**, responses = **one randomly selected real user's actual labels**;
  (iii) simulated unicycle-robot gain tuning, responses simulated from tracking error, **no human
  subjects**. **No explicit treatment of answerability or the Baum–Lang problem** `[A]`.
- **Use: NEAREST-NEIGHBOUR PAPER (see summary). Citation + explicit differentiation; not runnable as
  a baseline (no catalogue, no ranking metric, pairwise-only).**

### Bose, Li, Brahman, Koh, Du, Tsvetkov, Fazel, Xiao & Celikyilmaz 2026 — Pep
"Cold-Start Personalization via Training-Free Priors from Structured World Models", arXiv:2602.15012
(16 Feb 2026). `[A]`
- **Mechanism.** A structured world model supplies a **training-free prior**; a posterior
  `P(z|H_t,x)` over a latent user variable (Bayesian linear regression, or a GMM over user types) is
  updated from answers; the policy **selects the next criterion to query** from a fixed per-task list.
- **Evaluation.** PrefDisco benchmarks **MedQA / AIME / CommonsenseQA / SocialIQA**, **100 problems
  per domain × 50 users per problem**, 80/20 split at problem level, **20–30 criteria per task, users
  care about 2–4**. Answers are **LLM-simulated** under a "passive user" protocol. Metrics: % of
  oracle (PrefAlign rubric), queries-to-threshold, and an explicit **adaptivity** statistic
  `P(a_{t+1}^{(1)} ≠ a_{t+1}^{(2)} | H_{t−1}, a_t, v_t^{(1)} ≠ v_t^{(2)})`. Reported **77–87 % of
  oracle vs 55–76 % for a GRPO RL baseline, ~10K parameters vs 8B, 3–5× fewer interactions** `[A]`.
- **Use: CITATION-ONLY for C. ★ NOTE FOR PAPER B:** their adaptivity statistic is *our* Part-H/R2
  question ("did the behaviour actually condition on the answer?") written as a one-line metric.
  Adopt the formulation and cite it rather than coining our own.

### Van Looveren & Klaise — counterfactuals guided by prototypes; IM1 / IM2
ECML PKDD 2021; arXiv:1907.02584. `[S]`
- **Mechanism.** Autoencoder-based plausibility scores for a *generated* point: IM1 = target-class AE
  reconstruction error ÷ original-class AE reconstruction error; IM2 = distance between target-class
  AE and all-data AE reconstructions. Lower = more on-manifold.
- **Use: METHOD TO IMPORT.** The user-free "is this synthesised point meaningful" measure, with a
  survey-level pedigree (*ACM CSUR* 2024, 10.1145/3677119 `[S]`).

### Karras, Laine & Aila 2019 — Perceptual Path Length (StyleGAN, CVPR); Arvanitidis et al. 2018 — Latent Space Oddity (ICLR)
`[S]`
- **Use: CITATION-ONLY.** PPL = the standard "is this latent smooth enough that interpolants mean
  something" measure; Latent Space Oddity = the geodesic-not-linear objection we must pre-empt.

### Tripp, Daxberger & Hernández-Lobato 2020 (weighted retraining, NeurIPS); Maus et al. 2022 (LOL-BO, NeurIPS, arXiv:2201.11872)
`[S]`
- **Mechanism.** BO inside a VAE latent where candidates must decode to valid structured objects;
  trust regions in latent space plus retraining the generative model toward high-scoring regions.
- **Use: CITATION-ONLY, but the right technical family to name** when we discuss controlling where
  the emitted point lands.

### Wang, Feng, He, Wang & Chua 2023 — GeneRec
"Generative Recommendation: Towards Next-generation Recommender Paradigm", arXiv:2304.03516. `[S]`
- **Mechanism.** An "AI editor" repurposes existing items and an "AI creator" **generates items not
  in the corpus**, steered by user instructions; the paper stresses **fidelity checks** (authenticity,
  legality) on generated items.
- **Use: CITATION-ONLY — but it constrains our language.** "Off-catalogue is a prize, not a bug" is
  claimed territory in recommendation as of 2023, for generated CONTENT. Ours must narrow to
  *off-catalogue in the QUESTION channel*.

### 2024–26 large-action-space RL, triaged and NOT read
Discrete diffusion policies for combinatorial actions (arXiv:2509.22963); SAINT (arXiv:2505.12109);
stochastic Q-learning for large discrete actions (arXiv:2405.10310); unimodal discretisation
(arXiv:2408.00309); geometry of neural RL in continuous spaces (arXiv:2507.20853). All `[S]`.
**None communicates its action to a human. None is a baseline.**

---

# (c) CELLS THAT ARE EMPTY

Each stated narrowly enough to survive a hostile reviewer.

1. **A learned policy emitting a point in a *recommender's own collaborative latent*, verbalised as a
   question, whose answer is folded back into that same latent.** Nothing found. The three halves
   (ELM renders; Wolpertinger/Info-Synth emit-and-snap; PEBOL/GATE select-and-phrase) exist only
   separately.
2. **Any measurement of whether a synthesised query would be answered *differently by different
   users*.** Info-Synth measures answer *noise* as a function of query geometry; nobody measures
   discriminativeness of a synthesised query over a real user population. **This is the author's
   check #2 and it is unclaimed.**
3. **A scrambled / shuffled-latent CONTROL for a renderer.** Not found in ELM, LDPP, ACING, VDLM, or
   the AL query-synthesis line. The nearest thing is Zhu & Bento's *incidental* observation that
   generated queries were class-ambiguous — an accident, not a designed control. **Our scrambled
   control appears genuinely novel as an experimental design.**
4. **Circularity of a simulated answer to a *synthesised* query, named as a validity problem.**
   Nobody names it; Info-Synth's response model is exactly the construction, presented without caveat.
5. **Answerability-aware acquisition in a *continuous* design space.** SEL-BALD (NeurIPS 2024) does it
   for discrete instances; Info-Synth's heteroscedastic model handles answer *reliability*, not answer
   *possibility*. Nobody penalises a continuous design for being unaskable.
6. **Item-model-style calibration of a synthesised question.** Psychometrics calibrates families of
   generated items; no elicitation paper does anything analogous for a synthesised query.

---

# (d) WHAT WE SHOULD STOP CLAIMING

1. **STOP: "off-catalog emission as a PRIZE (not a bug) appears UNCLAIMED"**
   (`continuous_action_policy.md`, "What is NOVEL vs pre-empted"). **GeneRec (arXiv:2304.03516)
   claims exactly the off-corpus prize in recommendation**, in 2023, with an explicit "AI creator"
   for items that do not exist. **Narrow to:** *off-catalogue emission in the QUESTION channel, where
   the object must be answered rather than consumed.*
2. **STOP: any unqualified "nobody asks a user about a synthesised continuous point."**
   CoSpar/LineCoSpar (real subjects), ADO (real participants) and PBO do exactly that. **The
   qualifier is the renderer.** Write: "no work renders a synthesised point **as language** and folds
   the answer back into the space it came from."
3. **STOP framing "the snap" as a mere engineering convenience we could have avoided.** It is the
   field-standard, independently reinvented answer to a 1992 failure (Baum & Lang → local membership
   queries → NN query synthesis 2015 → Info-Synth 2026 → polyhedral conjoint's real-attribute
   profiles). **Removing the snap is the risk we are taking, not a shortcut we are declining** — and
   the paper must argue why it is safe *here*, since four fields concluded it is not safe in general.
4. **SOFTEN: "the snap CANNOT fail, so there is no validity problem."** True for the *bank* snap. The
   moment we emit an unsnapped point we inherit **the entire query-synthesis failure literature**, and
   the burden of proof is ours. The current framing under-states the exposure.
5. **STOP calling ELM's consistency metrics the only precedent for validating a synthesised point.**
   IM1/IM2 (counterfactual XAI) and PPL (generative modelling) are older, more general and user-free.
   Present our checks as *imports with named parents*, not inventions — safer under review, and the
   *application* to elicitation is still ours.
6. **DO NOT write "amortising the acquisition into one forward pass" as any part of our novelty.**
   Already true for DAD; as of ICLR 2025 it is also true for *preference* acquisition specifically
   (PABBO).
7. **VERIFICATION DEBTS created by this pass — cite none of these until cleared:** (a) GenMQD's
   mechanism and venue; (b) Vendrov et al. 2020's actual query object (two PDF extractions failed);
   (c) LineCoSpar's 6/6/30 numbers; (d) ADO's "fewer than 20 trials / 3–5× more precise"; (e) LOL-BO's
   "20×"; (f) Baum & Lang's exact bibliographic details.

---

## Where this leaves Paper C's claim

> *Query synthesis for a human oracle fails in general (Baum & Lang 1992; Zhu & Bento 2017), and every
> field that needs it either executes the design physically (CoSpar, ADO, PBO) or snaps it back to a
> real instance (local membership queries; NN query synthesis; Info-Synth; polyhedral conjoint). We
> ask whether **language** can be the missing renderer for a collaborative-filtering latent, and we
> answer it with point-level tests — off-catalogue geometry, discriminativeness over real users,
> render-consistency, and a scrambled control — rather than with a simulated answer.*

Every clause is now backed by a citation, and the last clause is the part nobody else has done.
