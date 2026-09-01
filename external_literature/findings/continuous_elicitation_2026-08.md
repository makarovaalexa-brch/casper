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

**⚠ AMENDED by the 2026-08-21 verification pass below — the last clause must now be narrowed
(Tang et al. 2023 hold the scrambled-correspondence control for a latent→language decoder), and the
"snap" list gains Vendrov et al. 2020, which is *inside recsys*.**

---

# VERIFICATION PASS — 2026-08-21

*Clears the eight verification debts opened by the 2026-08-18 pass (§(d)7 above, plus the `Tang et
al.` debt recorded in `docs/design/PAPER_C_DESIGN.md`). Method: WebSearch + WebFetch against arXiv,
publisher pages, Semantic Scholar and the papers' own text. Provenance flags carry the same meaning
as at the top of this file; every `[S]` below is still a lead, not a citable fact.*

**Headline: 5 VERIFIED, 2 VERIFIED-WITH-CORRECTION, 1 PARTIAL. Nothing was refuted outright, but
three of the eight were wrong in a way that would have surfaced in review — one mis-attributed
number, one wrong conference city, and one mechanism read backwards.**

| # | Debt | Verdict |
|---|---|---|
| 1 | Baum & Lang 1992 bibliography + finding | **VERIFIED with correction** (city wrong; there are no page numbers) |
| 2 | Zhu & Bento 2017 reproduces the failure | **VERIFIED — upgraded `[S]`→`[V]`; stronger than we recorded** |
| 3 | LineCoSpar 6 params / 6 subjects / 30 trials | **VERIFIED exactly, from the paper's own text** |
| 4 | ADO precision claims | **VERIFIED but MIS-ATTRIBUTED — wrong paper, and "fewer than 20 trials" is wrong** |
| 5 | LOL-BO's "20×" | **VERIFIED — optimisation performance vs prior LSBO, NOT decoding validity** |
| 6 | GenMQD — what it actually is | **VERIFIED in mechanism, UNRESOLVED in bibliography** |
| 7 | Vendrov et al.'s query object | **VERIFIED — and our fallback reading was WRONG. Genuine synthesis-then-snap** |
| 8 | Tang et al., Nature Neuroscience 2023 | **VERIFIED — and it is the scrambled-correspondence control, in a top journal** |

---

## 1. Baum & Lang 1992 — **VERIFIED, with two corrections** `[S, corroborated across four independent bibliographies]`

- **Author order is Baum then Lang.** (One citing paper writes "Lang & Baum"; the majority form, and
  the one printed in the local-membership-query line's own bibliography, is Baum, E. B. and Lang, K.)
- **Canonical entry:** *Baum, E. B. and Lang, K. (1992). Query learning can work poorly when a human
  oracle is used. In International Joint Conference on Neural Networks, volume 8.*
- **⚠ There is NO page range.** The `pages={8}` circulating in BibTeX copies is a mis-parse of
  `volume 8`. **Do not emit a page field.** Compile with the entry as banked in `references.bib`
  (`baum1992query`).
- **⚠ The city in the 2026-08-18 entry was WRONG.** We wrote "IJCNN 1992 (Baltimore)". Sources that
  state a location say **Beijing, China** (IJCNN ran twice in 1992: Baltimore in June, Beijing in
  November). If a venue string is needed, write only "IJCNN 1992".
- **Finding CONFIRMED, qualitatively and consistently:** membership-query synthesis on handwritten
  digits produced query images that "often had no structure and looked meaningless to the human eye";
  the canonical restatement is that an informative query is a **fusion of two similar digits (a 3/5
  hybrid) that a user cannot label because it is not a recognisable digit**. One citing source adds
  that the consequence is **noisy labels**, not merely refusals — keep that, because our answer model
  has the same exposure.
- **The 1992 proceedings text was NOT obtained** (no digital copy located). **Cite the CLAIM; never
  quote a number from this paper.** Rating stays `[S]` — but `[S]` from four bibliographies that agree.

## 2. Zhu & Bento 2017, GAAL — **VERIFIED, and our record was too weak** `[V — paper's own text]`

Abstract read verbatim from arXiv:1702.07956 (v1 25 Feb 2017 … v5 15 Nov 2017; **no venue** — cite as
arXiv-only). The abstract does **not** mention the failure; the body does, and it is more usable than
the second-hand version we banked:

- The algorithm explicitly labels synthesised instances **by human oracles**
  ("Label {G(z₁), G(z₂), …} by human oracles").
- > "In the CIFAR-10 dataset, our human labeler noticed higher chances of bad generated samples,
  > e.g., instances fail to represent either of the categories."
- > "our human labelers failed to confidently identify most of the generated cat and dog images"
- **The consequence is the citable part:** the experiments were **restricted to easy binary pairs**
  (automobile vs horse, digit 5 vs 7) *because* humans could not label the harder generated classes.
  That is Baum–Lang recurring under a modern generative renderer, **with the authors narrowing their
  own experimental scope to route around it**.
- ⇒ **Upgrade the §(b) entry to `[V]` and quote the two sentences.** This is now the strongest single
  piece of evidence that a fluent generative renderer does not buy answerability, and the direct
  argument for the scrambled control.

## 3. LineCoSpar 6 / 6 / 30 — **VERIFIED EXACTLY** `[V — paper text via ar5iv]`

*Tucker, Cheng, Novoseller, Cheng, Yue, Burdick & Ames, "Human Preference-Based Learning for
High-dimensional Optimization of Exoskeleton Walking Gaits", IROS 2020,
doi 10.1109/IROS45743.2020.9341416, arXiv:2003.06495.*

- **6 gait parameters**, named: step length, step duration, step width, maximum step height, pelvis
  roll, pelvis pitch.
- **6 able-bodied human subjects**, on the **Atalante** lower-body exoskeleton.
- **30 trials per subject** in the learning phase, **plus 6 validation trials** (we did not have the
  validation trials).
- Feedback per subject: > "at most 29 pairwise preferences and 30 pieces of coactive feedback".
- ⇒ The `[S]` numbers were right. **Quote them.** The existence proof stands: a synthesised continuous
  query, off any pool, answered by a real person — because the exoskeleton *walks it*. Companion
  CoSpar (ICRA 2020, arXiv:1909.12316) is the same group and setup.

## 4. ADO precision claims — **VERIFIED BUT MIS-ATTRIBUTED; one number was wrong** `[A — abstract verbatim]`

**The numbers are not in ADOpy.** They come from:

*Ahn, Gu, Shen, Haines, Hahn, Teater, Myung & Pitt, "Rapid, precise, and reliable measurement of
delay discounting using a Bayesian learning algorithm", **Scientific Reports 10, 12091 (2020)**,
doi 10.1038/s41598-020-68587-x* (bib key `ahn2020ado`). Abstract, verbatim:

> "In three populations (college students, patients with a substance use disorder, and Amazon
> Mechanical Turk workers) … ADO led to 0.95 or higher test–retest reliability of the discounting
> rate **within 10–20 trials** (under 1–2 min of testing), captured approximately 10% more variance
> in test–retest reliability, was **3–5 times more precise**, and was **3–8 times more efficient**
> than the staircase method."

- **"3–5× more precise" is CORRECT.** ✔ (And there is a bonus: **3–8× more efficient**.)
- **"fewer than 20 trials" is WRONG** — the paper says **10–20 trials**, and attaches it to a
  *reliability* threshold (≥0.95 test–retest), not to the precision claim. Write "10–20 trials".
- **Real human participants, three populations.** ✔ The structural claim ("continuous design space,
  directly perceivable design, real participants ⇒ answerability is free") survives intact.
- **ADOpy** (*Yang, Pitt, Ahn & Myung, Behavior Research Methods 53:874–897, 2021,
  doi 10.3758/s13428-020-01386-4*, `yang2021adopy`) is the **software/tutorial paper only** —
  psychometric functions, delay discounting, risky choice. Cite it for the method and its
  availability; cite `ahn2020ado` for any number. ADO's design space is a **grid over continuous
  design variables** — that is what "design grid" meant on 08-18; do not upgrade it to "continuum".

## 5. LOL-BO's "20×" — **VERIFIED; it means less than it looks** `[A — abstract verbatim]`

*Maus, Jones, Moore, Kusner, Bradshaw & Gardner, "Local Latent Space Bayesian Optimization over
Structured Inputs", NeurIPS 2022, arXiv:2201.11872* (`maus2022lolbo`). Abstract, verbatim:

> "LOL-BO achieves as much as 20 times improvement over state-of-the-art latent space Bayesian
> optimization methods across six real-world benchmarks"

- It is a **best-case ("as much as") optimisation-performance / sample-efficiency improvement over
  prior LSBO methods**, over six benchmarks. It says **nothing** about decode validity,
  on-manifold-ness, or how often a latent point yields a real object.
- ⇒ **Quotable only in that exact form, with "as much as", and only as an optimisation result.** Do
  not use it to argue that the emitted point is *meaningful* — that is precisely what it does not
  measure. The citable content for Paper C stays the **mechanism** (trust regions in latent space +
  retraining the generative model toward the promising region).

## 6. GenMQD — **MECHANISM VERIFIED; BIBLIOGRAPHY UNRESOLVED** `[S — OpenReview abstract via search indexing; the forum, API and PDF endpoints all serve a bot-check]`

Two independent search retrievals of the OpenReview abstract (id `CZvbXXgjrn`, posted 4 Oct 2024)
agree, and the mechanism the 08-18 pass reported second-hand is **correct**:

- **GenMQD = Generative Membership Query Descriptor.** > "generate textual descriptions of the desired
  data, instead of the data samples themselves"; a pre-trained **multi-modal alignment model (e.g.
  CLIP)** transforms the synthesised features into **natural-language text used to gather matching
  real data**.
- Explicitly motivated by our problem: existing membership-query synthesis "often generate[s]
  instances lacking semantic meaning, thereby increasing the difficulty of labeling".
- **★ NEW, and better for us than what we banked:** it reports a **large-scale user study in which
  human oracles PREFER GenMQD's generated TEXT queries over generated IMAGE queries**, and **+2.43%
  average accuracy when gathering and labelling 500 examples**.
- ⇒ **It is the nearest published instance of "verbalise the synthesised query so a human can act on
  it", and it comes with human-preference evidence.** It is vision-AL, not elicitation: the human acts
  by *fetching matching real data*, not by *answering a question about themselves* — so it does not
  take our cell, but it **does** take "rendering a synthesised query as text is a new idea".
- **STILL UNRESOLVED: author list and publication status.** It is an **ICLR 2025 submission** (the
  submission acknowledges the ICLR Code of Ethics); **no accepted/published version was located**.
  **Cite as an OpenReview preprint with the forum id, or not at all — do not attribute authors.**

## 7. Vendrov et al. 2020 — **VERIFIED, and our fallback reading was WRONG** `[V — ar5iv full text]`

*Vendrov, Lu, Huang & Boutilier, "Gradient-based Optimization for Bayesian Preference Elicitation",
AAAI 2020, 34(06):10292–10301, arXiv:1911.09153* — already in the bib as `vendrov2020gradient`.

The 08-18 pass flagged a secondary reading that the relaxation was over **item-selection weights**,
i.e. *soft selection* rather than synthesis. **That is refuted.** The paper's own framing:

> "relaxing the set of items from which recommendations and queries are generated by allowing
> attributes to vary continuously without requiring they correspond to available items"

- The optimisation variables are matrices **X** (query slate) and **Y** (recommendation slate) whose
  columns are **d-dimensional item vectors in continuous attribute space** — **hypothetical items**.
- Query types: **slate / pairwise comparison** (focus k=2, extended to larger slates) and **partial
  comparison queries** where only a subset of attributes is specified — explicitly to reduce the
  user's cognitive burden. *That is an answerability move, in 2020, by Boutilier's group.*
- The mapping back is a **snap**: > "Once optimized in this relaxed fashion, any hypotheticals are
  projected back into the nearest actual items to generate suitable recommendations or queries." —
  via a **deep retrieval** step selecting the real items that maximise expected utility given the
  optimised slate.
- ⇒ **Vendrov is genuine synthesis-then-snap, in recsys, at AAAI 2020, by a likely reviewer.** Three
  consequences, none of them small:
  1. **Add it to the "everyone snaps" list** — now a **four-field** claim (learning theory, active
     learning, preference BO/elicitation, marketing) and one of the four is **our own field**.
  2. **"Optimise the query continuously over hypothetical items" is PRE-EMPTED in recsys.** Paper C's
     novelty cannot rest on it. What Vendrov does *not* do: (a) work in a **collaborative** latent
     rather than an attribute space, (b) **render** the hypothetical as language, (c) leave the point
     **unsnapped**. State our claim against exactly those three.
  3. **The partial-comparison query is prior art for "ask about a subset of the direction"** — check
     it before presenting any partial/sparse rendering as new.

## 8. Tang, LeBel, Jain & Huth — **VERIFIED, and a better fit than expected** `[A — abstract verbatim + null-model description recovered from the bioRxiv full text]`

*Nature Neuroscience **26(5):858–866**, May 2023, doi 10.1038/s41593-023-01304-9; preprint bioRxiv
2022.09.29.509744* (`tang2023semantic`). The preprint is 403/429-walled to our fetcher, but the
abstract and the null-model description were both recovered.

- **What it shows** (abstract, verbatim in part): a **non-invasive fMRI decoder that reconstructs
  continuous language** — "Given novel brain recordings, this decoder generates intelligible word
  sequences that recover the meaning of **perceived speech, imagined speech and even silent videos**";
  language can be **separately decoded from multiple cortical regions**; and "**subject cooperation is
  required both to train and to apply the decoder**".
- **★ The part Paper C needs is the EVALUATION protocol, and it is exactly the control we want:**
  - A **null distribution** is built from **null sequences that carry no information about the
    stimulus**, scored against the reference transcript; observed decoding scores are compared to it
    with a **one-sided nonparametric test**.
  - An **alternative null** is built by **permuting the actual voxel responses** before decoding,
    which **preserves the statistics of the brain responses** while destroying the response→stimulus
    **correspondence**; repeated **10 times** to give 10 alternative null sequences.
  - Test stimulus: a ~10-minute story ("Where There's Smoke").
- ⇒ **This is the scrambled-CORRESPONDENCE control, published in Nature Neuroscience, on a
  latent→language decoder.** Second independent instance of the Heap et al. lesson
  (`llm_verbalisation_2026-08.md`): *permute the mapping, keep the distribution*. **Cite both.** It
  also answers "who validates a rendered latent without asking a human whether it sounds plausible":
  they score the rendering against a reference and a matched null, never against a plausibility
  judgement.
- ⚠ **Do not import their headline as a claim about verbalisation quality.** They decode brain data to
  language against a *paired reference transcript*; we have no reference text for a synthesised CF
  direction. **The transferable object is the NULL CONSTRUCTION, not the metric.**

---

## What this pass CHANGES in the sections above

1. **§(b) Zhu & Bento:** `[S]` → `[V]`; replace "described in the survey literature as producing
   low-quality, class-ambiguous samples" with the two quoted sentences plus the fact that the authors
   **restricted their experiments to easy binary pairs** because of it.
2. **§(b) Baum & Lang:** delete "(Baltimore)"; record that there is **no page range**.
3. **§(b) ADO:** the precision/trial numbers move to **`ahn2020ado` (Sci. Rep. 10:12091, 2020)**;
   "fewer than 20 trials" becomes **"10–20 trials"**, attached to reliability, not precision.
4. **§(a) Q4 item 9 (Vendrov):** the "soft selection, not synthesis" caveat is **removed**. Vendrov
   joins the "everyone snaps" list and **pre-empts continuous query optimisation in recsys**.
5. **§(a) Q4 item 10 (LOL-BO):** the 20× is now quotable, in the exact form "as much as 20 times
   improvement over state-of-the-art latent space Bayesian optimization methods across six real-world
   benchmarks", **and only as an optimisation result**.
6. **§(a) Q4 item 5 / §(b) GenMQD:** mechanism confirmed and strengthened (human oracles *prefer* the
   text queries; +2.43% at 500 labels); **bibliography still unresolved** — OpenReview preprint only,
   no author attribution.
7. **§(c) empty cell 3 (scrambled control) — NARROW IT.** A scrambled-correspondence control is **not**
   novel in general: Heap et al. hold it for SAE explanations and **Tang et al. 2023 hold it for a
   latent→language decoder in Nature Neuroscience**. What remains unclaimed is a scrambled control
   **for a rendered elicitation QUESTION**, where the validated object is a question a user must
   answer rather than a reconstruction scored against a reference. Write that, not "our scrambled
   control appears genuinely novel as an experimental design".
8. **§(d)7 (the debt list) is DISCHARGED** — superseded by this section.

---

# PRINCIPAL-AXIS EMITTER PRIOR-ART PASS — 2026-09-01

*Brief: before building an emitter whose output is (approximately) the **principal axis of the
cohort's residual covariance** at an interview state — a closed-form, training-free rule — establish
exactly what is already published, so we cite rather than reinvent. Five questions, answered against
primary sources. Provenance flags carry the same meaning as at the top of this file.*

**Method note.** This pass re-read our own `rank_targeted_acquisition.md`,
`active_learning_elicitation.md`, `adaptive_elicitation_baselines.md` and `infosynth_baseline_2026-08.md`
first, per HARD RULE 4. **A large fraction of the answer was already in this repository** — see §Q1
and §Q2 — and one of the two decisive facts (our own measured Σ-greedy null) is an in-house number,
not a literature finding.

---

## ★ THE ONE-PARAGRAPH ANSWER — read this before designing anything

**The emitter rule is published, twice, in closed form, and one of the two is eighteen years old.**
For a linear-Gaussian belief the most informative query direction is the top eigenvector of the
posterior covariance; **Ji, Xue & Carin (IEEE TSP 2008) write exactly that sentence and even offer
the synthesise-or-snap choice we thought we were introducing** `[V]`. Info-Synth (2026) re-derives
the same direction under a different (ideal-point, heteroscedastic-probit) model and lands on
`ã = 4r̃v₁` `[V]`. So *nothing about "the principal axis is the best direction" is ours to claim* —
it is a textbook design result with a recent preference-learning restatement. Worse for an
adaptivity story: **a pure Σ-functional emitter provably falls inside the linear-Gaussian
static-optimality null** that `rank_targeted_acquisition.md` §"The null" already documents, and
CASPER has **already measured that class**: Σ-greedy beat random order by **+0.0036**. What remains
genuinely open is not the direction but everything around it — the *space* (a collaborative latent,
not an attribute or ideal-point space), the *renderer* (language), and the *answerability* accounting
that no one in this lineage performs.

---

# Q1 — The canonical OED statement, its exact conditions, and where it fails

## The citable statement (VERIFIED VERBATIM)

**Ji, Shihao; Xue, Ya; Carin, Lawrence. "Bayesian Compressive Sensing." *IEEE Transactions on Signal
Processing* 56(6):2346–2356, 2008, doi 10.1109/TSP.2007.914345** (`ji2008bcs`). `[V — Sec. IV.A of
the authors' preprint PDF; bibliography confirmed against DBLP]`

Their Eqs. (16)–(18), in their own words:

> "h(f) = … = ½ log |Σ| + const"
>
> "hnew(f) = h(f) − ½ log(1 + α₀ rᵀ_{K+1} Σ r_{K+1}), (17) … To minimize hnew, the next projection
> r_{K+1} should hence be designed to maximize rᵀ_{K+1} Σ r_{K+1}."
>
> "this is equivalent to maximizing the variance of the expected measurement g_{K+1}. In other words,
> the next projection r_{K+1} should be selected to constitute the measurement g_{K+1} for which the
> data is most uncertain, and hence access to the associated measurement would be most informative."
>
> **"If it is possible to design new projection r_{K+1} adaptively 'on the fly', then one might
> perform an eigen-decomposition of the matrix Σ, and select for representation of r_{K+1} the
> eigenvector with largest eigenvalue. Alternatively, if from a hardware standpoint such flexibility
> in design of r_{K+1} is not feasible, then one might a priori design a library L of possible next
> projections, with r_{K+1} selected from L with the goal of maximizing (18)."**

★ **That last paragraph is the synthesise-or-snap dichotomy, in 2008.** It is the same fork as
Baum & Lang → local membership queries → Wolpertinger → NN query synthesis → Vendrov → Info-Synth's
Pair-Opt-dist, stated inside Bayesian experimental design and settled the same way. **Add it to the
"everyone snaps" list**, which is now a five-field claim.

**Lineage citations** (bibliographic detail verified, bodies NOT read — cite for provenance only):
- **Lindley, D. V. (1956), *Ann. Math. Statist.* 27(4):986–1005**, doi 10.1214/aoms/1177728069
  (`lindley1956information`) `[A]` — the expected-information-gain criterion itself.
- **Chaloner & Verdinelli (1995), *Statistical Science* 10(3):273–304**, doi 10.1214/ss/1177009939
  (`chaloner1995bayesian`) `[A — abstract + bibliography from Project Euclid; full text paywalled]` —
  the review that puts Bayesian D-/A-/E-optimality in one decision-theoretic frame and *separates the
  linear from the nonlinear design problem*. Cite for the taxonomy; **quote no formula from it until
  someone reads it.**
- **Houlsby, Huszár, Ghahramani & Lengyel (2011), BALD, arXiv:1112.5745** (`houlsby2011bald`, already
  in the bib) `[A — abstract verbatim]`, whose own framing is the failure condition: *"For simple
  regression an optimal myopic policy is easily tractable. However, for other tasks and with more
  complex models, such as classification with nonparametric models, the optimal solution is harder to
  compute."*

## The exact conditions under which it holds

All six must hold together. The result is fragile in a way that is easy to forget once it is written
as "just take the top eigenvector".

1. **Linear response.** The measurement is `y = rᵀθ + ε`.
2. **Gaussian belief and Gaussian noise**, `θ ~ N(µ, Σ)`, `ε ~ N(0, σ²)` with `σ²` known.
3. **A norm/energy constraint on the design.** Without one the objective `rᵀΣr` is unbounded — scale
   `r` up and the information grows without limit. **The "top eigenvector" answer is the answer to a
   constrained problem; the constraint is doing real work and must be stated.**
4. **A continuous, unconstrained design space** (the whole sphere). Restrict to a pool and you are
   back to Ji et al.'s "library L".
5. **A Σ-only criterion.** Information gain / Bayesian D-optimality gives `½log(1 + rᵀΣr/σ²)` ⇒ `v₁`.
   For a *single* rank-one measurement A- and E-optimality select `v₁` too, because the achievable
   trace reduction `λᵢ²/(σ² + λᵢ)` and the max-eigenvalue reduction are both monotone in `λᵢ`
   `[A — one-line derivation, not a citation; do not attribute it to Chaloner & Verdinelli
   without reading them]`. **The three criteria coincide for one query and diverge for a batch.**
6. **Myopia.** Ji et al. call their own criterion "(myopic)" and "one-look-ahead" explicitly `[V]`.

## Where it FAILS — the four that matter for us

**(a) ★ SEQUENTIAL vs BATCH — and this is the one that bites CASPER hardest.**
Under exactly conditions 1–2 the posterior update
`Σ ← Σ − Σr rᵀΣ / (σ² + rᵀΣr)` **does not involve the observed answer `y`**. Therefore the entire
greedy sequence is computable before any answer arrives: the queries are just the leading eigenvectors
of the prior covariance, in order. **A pure principal-axis emitter is not adaptive — it cannot be.**
This is not new to us: `rank_targeted_acquisition.md` §"The null" already records it, with
Jedynak, Frazier & Sznitman's Bayes-optimal policy that *"asks a deterministic set of questions"*
`[V]` and Krause & Guestrin's a-priori designs `[S]`, and with our own measured consequence —
**Σ-greedy over random order = +0.0036**. That file's verdict applies verbatim to this hypothesis:
*it is evidence that we ran an objective inside the theorem's hypothesis.*
⚠ Note what re-introduces adaptivity in Ji et al.: **not** the Gaussian update, but the type-II ML
re-estimation of the sparsity hyperparameters `α` from the data `[V]`. Adaptivity has to come from
somewhere the covariance does not already know.

**(b) Discrete / comparative answers.** With a probit or logistic link the posterior is no longer
Gaussian, the covariance update *does* depend on the answer, and — decisively — **a direction is no
longer a query**. Info-Synth's own optimum has **two** conditions, not one: the query midpoint must
sit at the posterior mean (so the answer is equiprobable) *and* the separation must lie along `v₁`
`[V, §Info-Synth read]`. The direction is half the object. BALD is the reference for the general
intractability `[A]`.

**(c) Nonlinear response.** The posterior covariance becomes data-dependent and the closed form is
gone; this is the linear/nonlinear split Chaloner & Verdinelli organise their review around `[A]`.
For us the mismatch is concrete: **our recommender scores by inner product against a decoder plus a
learned decoder bias, not by a linear-Gaussian measurement of a latent user vector**, and
`infosynth_baseline_2026-08.md` §2.1 already flags that porting an ideal-point posterior onto that
space is *"a substantive assumption we would be making on their behalf"* `[V]`.

**(d) Heteroscedastic noise / answerability.** If `σ²` depends on the design — Info-Synth's entire
contribution — the optimal magnitude is no longer "as large as the constraint allows"; there is a
finite optimal radius found numerically `[V]`. And if the design must additionally be *askable*, the
continuum relaxation is void and you snap (Ji's library L; Vendrov; Pair-Opt-dist) or render.

---

---
# Q2 — Posterior-covariance geometry to choose interview questions, INSIDE recommender systems

**VERDICT: classical optimal design is ALREADY in recsys elicitation — it arrived in 2015 and was
re-derived from a ranking-flavoured loss in 2018 — but it is the TRACE criterion over a POOL, never
the top eigenvector, and never a synthesised direction. "Emit the principal residual direction" is
NOT a named method in RS. The idea is unnamed; the ingredients are all taken.**

Most of this was already in the repository and is restated here so the answer is in one place.

## What IS in recsys, ranked by closeness

1. **★ A-optimality is the recsys elicitation rule, twice.**
   - **Anava, Golan, Golbandi, Karnin, Lempel, Rokhlenko & Somekh, "Budget-constrained item cold-start
     handling in collaborative filtering recommenders via optimal design", WWW 2015**
     (`anava2015budget`, arXiv:1406.2431) `[V — abstract, prior pass]`: elicitation formalised as
     **optimal design** under squared prediction error, objective proved monotone-supermodular.
     ⚠ It is **item** cold-start (which users to ask about a new item), so it is the formalism's
     source, not a comparator.
   - **Sepliarskaia, Kiseleva, Radlinski & de Rijke, "Preference Elicitation as an Optimization
     Problem", RecSys 2018** (`sepliarskaia2018spq`, doi 10.1145/3240323.3240352) `[V — full PDF,
     prior pass]`: the selection rule is **`argmin_B tr[(V_B V_Bᵀ + εE)⁻¹]`** — literally
     **A-optimality over the selected items' factor vectors**, greedy with a Sherman–Morrison
     reduction, derived from a *flat pairwise* loss via Anava et al.
   ⇒ **Posterior-covariance geometry for interview-question selection in RS is settled prior art.**
   Anything we say about "design criteria in elicitation" must cite these two.
   ★ The load-bearing distinction our own files already established: the collapse "task loss ⇒
   A-optimal design" happens **because the loss is untruncated and undiscounted**; NDCG@10 does not
   collapse to a trace criterion.

2. **`goldberg2001eigentaste` — the eigen-basis, 25 years ago.** `[V, this pass]` Every new user
   answers the same universal **gauge set**; PCA on the dense gauge submatrix (eigenvectors of the
   gauge-item correlation matrix) projects and clusters them. **The eigenstructure is the analysis
   basis, not the question-selection rule, and the interview is deliberately fixed.** This is the
   paper that makes "principal directions of a rating covariance" unclaimable as an *idea* in
   recommendation.

3. **`nguyen2024pere` — PERE, Nguyen, Nguyen, Doan & Nguyen, UAI 2024 (PMLR v244:2766–2786,
   arXiv:2406.00973)** `[V-html, prior pass]`. The closest *belief-geometry* elicitation method in
   modern recsys: the user is a **region** in a frozen embedding space, and adaptive information-gain
   queries **cut the region by halfspaces**, weighted by an estimated experience probability;
   recommendation comes from the region's **Chebyshev centre**. Geometry of the belief drives the
   questions — but the questions are **real items**, the geometry is a polytope rather than a
   covariance, and nothing is synthesised.
   ⚠ Standing debt: PERE's experience-probability formula is still `[S]`.

4. **`fonarev2016rectangular` (RMVA) and the representative-based line** (`liu2011wisdom`,
   `shi2017local`) `[prior passes]`: **maximal-volume** selection of seed items in the MF latent — a
   determinant/volume criterion on the item factor matrix, i.e. D-optimality's cousin, applied to a
   **static** seed set chosen from the catalogue.

5. **`rubens2007influence` — Rubens & Sugiyama, RecSys 2007** `[S]`: selects items by the **influence
   of a rating on other items' predictions**. `rank_targeted_acquisition.md` already identifies this
   as the closest pre-existing recsys relative of a `Qᵀ Σ φ` influence term — un-Bayesian and
   untargeted.

6. **The Bayesian-AL-for-CF line, for completeness** `[S/A, prior passes]`: Jin & Si (UAI 2004,
   arXiv:1207.4146) minimise **expected loss over the model posterior** rather than a point estimate;
   Sutherland, Póczos & Schneider (KDD 2013) query the **highest-posterior-variance matrix element**
   — the max-variance baseline that is, per §Q1(a), static-equivalent by construction; Harpale & Yang
   (SIGIR 2008), Zhao et al. (CIKM 2013, `zhao2013interactive`) and Christakopoulou et al.
   (KDD 2016, `christakopoulou2016towards`) run bandit/Thompson-style selection over items.
   **None of these uses an eigen-direction, and none synthesises a query.**

7. **The classic interview-tree line does NOT use covariance geometry at all** `[V, prior passes]`.
   Rashid et al. (IUI 2002 / SIGKDD Expl. 2008) use popularity, entropy, HELF; Golbandi, Koren &
   Lempel (WSDM 2011) explicitly **deviate** from entropy/Gini to a **squared-loss** split
   (verbatim in `golbandi2011adaptive`); Zhou, Yang & Zha (SIGIR 2011) fit latent vectors at tree
   leaves. Elahi, Ricci & Rubens (Computer Science Review 2016) taxonomise these as heuristic /
   personalised / multi-criterion strategies — **the survey does not carry a "variance- or
   covariance-optimal" family name**, which is why the design vocabulary has to be imported from OED.

8. **Synthesis inside recsys is Vendrov's** (`vendrov2020gradient`, AAAI 2020) `[V, 2026-08-21 pass]`:
   continuous **hypothetical item vectors** optimised for **EVOI**, then **projected back to the
   nearest actual items**. ⚠ **Its objective is expected value of information, not covariance
   eigen-structure** — so it pre-empts "optimise the query continuously in recsys", not "use the
   principal axis".

## The blunt answer to "is it already a named method in RS?"

**No. There is no named RS method called anything like "principal residual direction elicitation".**
But every ingredient is held by someone: the **criterion family** by Anava/Sepliarskaia, the
**eigen-basis** by Eigentaste, the **belief-geometry-drives-the-question** move by PERE, the
**synthesise-then-snap** move by Vendrov, and the **eigenvector rule itself** by Ji et al. (outside
recsys) and Info-Synth (outside recsys). **An unnamed combination of fully-owned parts is a weak
novelty claim, and we should not build a headline on it.**

---

## ⚠ AMENDMENT TO §Q2 — 2026-09-01, same day, after the RS-specific sweep returned

The §Q2 above was written from this repository's own verified records while the delegated
recommender-side sweep was still running. **The sweep has since returned, and it found one paper that
changes the answer.** Per the standing rule that a subagent's verdict is a draft, every bibliographic
fact below was re-verified directly against the primary source; mechanism detail that was NOT
re-read here is flagged `[V-delegated]` and must be treated as one level weaker than `[V]`.

### ★ THE DANGEROUS NEIGHBOUR — adaptive A-optimality over a latent user posterior, in a survey

**Zhang, Chelsea; Taylor, Sean J.; Cobb, Curtiss; Sekhon, Jasjeet. "Active matrix factorization for
surveys." *Annals of Applied Statistics* 14(3):1182–1206, 2020, doi 10.1214/20-AOAS1322,
arXiv:1902.07634** (`zhang2020activemf`).

`[A — title, full author list, venue, volume, issue, pages, DOI and the verbatim abstract confirmed
directly against the arXiv abs page and the Project Euclid listing, 2026-09-01. The criterion's
algebra below is [V-delegated]: read from ar5iv full text by the sweep, NOT re-read here.]`

- Abstract, verbatim: *"We propose to ask fewer questions of respondents and impute missing responses
  via probabilistic matrix factorization. **A variance-minimizing active learning criterion chooses
  the most informative questions per respondent.**"* — plus a Facebook survey experiment.
- Criterion `[V-delegated]`: `min_j tr[(Λ_i^{(t+1)})⁻¹]` — **minimise the trace of the posterior
  covariance of the RESPONDENT'S OWN latent factor** after asking question *j*. Adaptive,
  per-respondent, sequential. Reportedly rewritten by the authors in eigen terms as
  `min Σ_ℓ [λ_ℓ(Λ_i^{(t+1)})]⁻¹`, which *"penalizes small eigenvalues of the precision matrix,
  corresponding to directions in latent space with least information."*
- **But it SELECTS among existing survey questions** by their question-factor vectors. It never forms
  an eigenvector and never emits a direction; and it is survey methodology, not recommendation.

**Why it matters anyway:** it is adaptive + latent + posterior-covariance + *written in eigenvalue
language*. A reviewer will find it. **Cite it as nearest prior art.**

### ★ THE UNRESOLVED QUESTION IT OPENS — and it is exactly our question

§Q1(a) above says a criterion that is a functional of `Σ` alone is realisable as a fixed,
answer-independent sequence. **Zhang et al. look like a counterexample and probably are not one:**
under a Gaussian PMF likelihood the update `Λ ← Λ + β v_j v_jᵀ` does not involve the answer, so their
"per-respondent adaptive" sequence would still be precomputable from the respondent's prior. What
would break that is their own listed extension — the abstract says the procedure *"can benefit from …
modelling responses as **ordered logit** rather than Gaussian"*, and under an ordered-logit likelihood
the posterior covariance **does** depend on the answer.

⇒ **NEW DEBT, HIGH, and it is a design lead rather than a citation chore:** read Zhang et al. §§3–4
and determine whether their reported adaptive gains survive under the Gaussian likelihood or arrive
only with the non-Gaussian one. **If the latter, that is the published evidence for where our own
adaptivity has to come from, and it is worth more to us than the citation.** Until it is read, do not
write either "their adaptivity is illusory" or "covariance criteria can be adaptive".

⚠ **Do NOT let this overwrite §Q1(a).** The theorem is about the objective class, not about whether
people run such criteria sequentially. Both of these are true and must be written together: *people
do run covariance criteria adaptively (Zhang et al.), and under a linear-Gaussian belief such a
criterion is still realisable as a fixed sequence.*

### Other additions to §Q2

- **`fonarev2016rectangular` gains its arXiv id (1610.04850) and a gap sentence in the authors' own
  voice** `[V-delegated, ar5iv]`: *"Another interesting direction for future work is to join
  approaches from two classes: based on the maximal-volume concept and based on optimal design
  criteria."* ⇒ **The maximal-volume line and the optimal-design line were still separate as of 2016,
  said so by the maximal-volume authors.** Useful, and cheap to cite.
- **Rubens, Kaplan & Sugiyama, "Active Learning in Recommender Systems" (RecSys Handbook; 2015 ed.
  with Elahi, doi 10.1007/978-1-4899-7637-6\_24)** `[A-partial/S — extraction unreliable across three
  attempts]`. Reported family names: **Uncertainty Reduction · Error Reduction · Variance Reduction ·
  Attention-Based**, with A-optimality under the variance branch. ⇒ **RS's own survey already names a
  variance-reduction family containing A-optimality** — this independently confirms §Q2's verdict.
  ⚠ **Do not quote the chapter's wording.**
- **★ NEW LEAD, and the one most worth chasing: Loepp, Hussein & Ziegler, "Choice-based preference
  elicitation for collaborative filtering recommender systems", CHI 2014,
  doi 10.1145/2556288.2557069** `[S ONLY — ACM 403, two open PDFs would not text-extract]`. Reported
  to build the dialogue by presenting item sets at the **low and high extremes of each latent factor
  in turn**, positioning the user in latent space. **That is "interrogate along a latent direction",
  in a recommender, in 2014** — but along a *fixed factor axis*, not a state-conditional one, and
  **whether the factors are taken in eigenvalue/variance order is UNVERIFIED**. ⚠ **This is the
  single highest-value verification debt from this pass: it is the nearest thing anyone found to
  asking a user about a direction rather than an item. Do not write any novelty sentence about
  "asking along a direction" until it is read.**
- **Boutilier, Zemel & Marlin, "Active Collaborative Filtering", UAI 2003** `[A]` — EVOI with offline
  prototyping and EVOI bounds; the Vendrov family, seventeen years earlier.
- **AUTHOR CORRECTION:** the KDD 2012 online-Bayesian-MF active-learning paper is **Silva & CARIN**
  (Duke), not "Silva & Carvalho" — the name in the brief and in earlier notes was wrong. KDD 2012,
  pp. 325–333, doi 10.1145/2339530.2339584. Criterion: greedy **submodular mutual-information**
  maximisation, with a cheap **trace-of-posterior-covariance** surrogate `[S — the trace detail comes
  from Zhang et al.'s description of it, not from Silva & Carin's own equations; paywalled]`.
- **CORRECTION to how IGCN is described:** Rashid, Karypis & Riedl (SIGKDD Explor. 10(2):90–100,
  2008) score items by **information gain against bisecting-k-means user-cluster labels** (c = 300),
  personalised to the 150 nearest neighbours after 5 ratings — and the paper states outright that a
  direct decision-tree application *"is not practically feasible in our problem domain"*
  `[V-delegated, via a mirror whose printed venue line was absent — page numbers unverified]`.
  **⛔ Never write "IGCN builds a decision tree over user clusters."**

### Sources the sweep could NOT reach — nothing about these may be written from this pass

**Zhao, Zhang & Wang, CIKM 2013** (`zhao2013interactive`) — PDF would not extract. ⚠ A linear-UCB /
Thompson variant would carry a `sqrt(vᵀΣv)` term, which would make it *another* covariance-geometry
instance in recsys; **assume nothing until someone reads it.** Also unreached:
**Christakopoulou et al. KDD 2016**; **Vanchinathan et al. RecSys 2014**; **Sutherland, Póczos &
Schneider KDD 2013** (our `[S]` record stands unrefreshed); **Elahi, Ricci & Rubens, Computer Science
Review 20:29–50, 2016** (ScienceDirect 403) — so §Q2's sentence about what that survey does *not*
contain rests on our own earlier `[S]` reading, not on this pass. **Zhou, Yang & Zha SIGIR 2011** is
`[S]` only (three routes dead); the squared-loss split is our prior record, not re-verified.

⚠ **On Vendrov:** the sweep could not verify the snap step and one of its PDF extractions
*hallucinated* an "optimal design criteria" claim. **Our own 2026-08-21 pass verified the snap at
`[V]` from ar5iv full text, with the quotation banked in that section. Our `[V]` stands; the sweep's
failure to reproduce it is not a refutation** — but it is a reminder that this paper's PDF extracts
badly and that the ar5iv HTML is the route that works.

### What this amendment does NOT change

The §Q2 verdict is unchanged and is if anything strengthened: **every RS-side source anyone reached
reduces the covariance to a SCALAR (trace or determinant) and uses it to SCORE A FINITE POOL. Nobody
takes the argmax-direction, and nobody emits it.** What was already dead stays dead — optimal design
is not new to RS, and covariance-based question *selection* is not new to latent-factor elicitation.
**The narrow live cell is the DIRECTION AS THE EMITTED OBJECT**, and now it must be stated against
`zhang2020activemf` and `fonarev2016rectangular` explicitly, or a reviewer will supply them.


---

# Q3 — Info-Synth: does our proposal COINCIDE with it?

## Bluntly: on the emitter rule, YES.

Confirmed from our own full-PDF read (`infosynth_baseline_2026-08.md` §1.3, `[V]`), and the arXiv abs
page re-checked 2026-09-01 `[A]` (title *Active Query Synthesis for Preference Learning*; Nadagouda,
Ahad, Tucker, Davenport; 25 May 2026; the abstract mentions mutual-information maximisation and the
confidence-aware response model but **not** the eigenvector — the eigenvector is Proposition 1 in the
body). Their synthesis rule, exactly:

- **Term 1** of the MI objective forces the query **midpoint to the posterior mean**: `b̃ = µ`.
- **Term 2** is maximised by `x̃ = r̃ v₁`, with **`v₁` the principal eigenvector of the posterior
  covariance `Σ`**; the magnitude `r̃` is a 1-D numerical entropy minimisation.
- Final: **`ã = 4 r̃ v₁`**, i.e. the pair straddling `µ` along the top principal axis.

**Our proposed emitter and theirs pick the same direction.** If the "cohort's residual covariance at
an interview state" is (as it would be under any Gaussian belief) the posterior covariance over the
user's latent given the answers so far, then `v₁` of that matrix **is** their `v₁`. The two
derivations differ — theirs is MI under an ideal-point/heteroscedastic-probit model, Ji et al.'s is
entropy under linear-Gaussian — and *that they agree is the point*: the rule is overdetermined, and
therefore not a contribution.

## What differs, precisely — and it is not nothing

| Axis | Info-Synth | Ji et al. 2008 | Ours, if built |
|---|---|---|---|
| Direction rule | `v₁` of posterior Σ | `v₁` of posterior Σ | `v₁` of cohort residual Σ — **same object** |
| Query object | ordered **pair** `(p̃, q̃)` straddling `µ` | a **projection row** `r` | a single **direction** to be rendered |
| Space | ideal-point embedding (uniform synthetic; OPT-350M summaries; robot gains) | sparse signal basis | **collaborative-filtering latent** |
| Presented to a person? | **never** — no human answers any query in the paper `[V]` | no | that is the whole point |
| Rendered as language? | no renderer anywhere in the paper `[V]` | no | **yes** |
| Answerability | modelled as **response noise from geometry**; askability never raised `[V]` | "library L" if the hardware cannot do it | the object of study |
| Left unsnapped? | only where the point is itself instantiable (random vectors, controller gains) `[V]` | either, explicitly | yes |

**⇒ The honest position, and the one to write:** *we are the rendered, catalogue-free variant of a
known emitter, and the contribution is the answerability analysis and the causal-faithfulness audit
that the emitter's own literature has never run.* This is materially narrower than "a closed-form
training-free emitter" as a headline, and the design sheet should say so in its own voice.

⚠ **Do not present the closed form as an efficiency contribution either.** Ji et al. is
training-free, closed-form and 2008; Info-Synth's cost is *"just the covariance estimate"*, `O(S·d²)`
`[V]`. "No network needed" is not a claim, it is the default in this family.

---

# Q4 — Answerability bending the ideal direction: is the GAP measured anywhere?

**VERDICT: the cell is unoccupied — but it is one paper away from being taken, and the near-miss is
Info-Synth again.**

1. **★ Info-Synth writes the object down and never reports it.** Their §5.1 Taylor-expands the MI at
   the synthesised optimum, where the gradient vanishes, giving the loss of moving to a realisable
   pair as a Mahalanobis norm **`ΔI(z) = ½‖Δz‖²_M`, `M = −H(z*)`** `[V]`. **Pair M-dist** filters on
   it; **Pair Opt-dist** is the cheap high-dimensional surrogate
   `η(p,q,λ) = Σᵢ(bᵢ−µᵢ)²/Σᵢᵢ + λ‖(p−q) − 2r̃v₁‖²` `[V]`. **They never report ΔI as a number** — no
   table, no plot; the results are downstream MSE / Kendall-tau / accuracy / selection time `[V]`.
   ⇒ **The machinery for "information cost of snapping" exists and is named. Reporting the number
   closes the cell. We should cite it explicitly and stake our claim on the axis they do not touch:
   their constraint is POOL MEMBERSHIP, ours would be HUMAN ANSWERABILITY.**
   ⚠ **Technical caution before we use their ΔI:** it is a *second-order* expansion around `z*`,
   valid for small `Δz`. The interesting case — the nearest askable question is *far* from the
   emitted direction — is exactly where the quadratic understates the loss. **Report the true MI
   difference, not the Mahalanobis surrogate, whenever the gap is large.**

2. **Bıyık et al. 2019 (`biyik2019asking`, already in our bib) is the strongest answerability work
   and it deliberately argues there is NO gap to measure.** `[V — ar5iv full text]` Verified detail
   beyond what we held: their answerability model is a noisily-rational choice model plus an
   explicit **"About Equal" option Υ** governed by a minimum-perceivable-difference `δ ≥ 0`, with
   **`δ = 1` in the experiments**; **Theorem 1** shows the *trivial* query (identical trajectories)
   is a **global optimum of volume removal** and the global *minimum* of information gain. Human
   study: **15 participants** (Driver, Tosser) + **12** (Fetch), 30 randomised blinded queries each,
   7-point Likert; info-gain queries rated easier at **p < 0.005**, fewer indistinguishable queries at
   **p < 0.05 / p < 0.005**, trajectory preference at **p < 0.05**.
   ⚠ **Quote the p-values, never a Likert mean** — the means live only in Figure 6 and were not read.
   Their argument is that information gain *aligns* informativeness with answerability, so the
   trade-off need not be paid. **That is a claim we can test in our setting and they cannot in
   theirs**, because their queries are always physically renderable.

3. **The metric SHAPE already exists, in psychometrics: *p*-optimality** (`yang2020poptimal`)
   `[A/S — landing page only; SAGE 403, PMC captcha; figures from indexing quotations]`. A pool is
   ***p*-optimal** if at every ability estimate some *available* item supplies at least a `p`-fraction
   of the **maximum possible information** there; operational `p = 0.95`, reportedly ≈ 0.35 logit of
   mismatch under Rasch. **Borrow the ratio (best-available ÷ theoretical maximum); do not borrow the
   numbers, and name the disanalogy — their unavailable optimum has the wrong DIFFICULTY, ours would
   be UNASKABLE.** ⚠ Verify both figures against the article before any sentence uses them; the
   originating Reckase source could not be resolved to a full citation.

4. **SEL-BALD** (`gao2024selbald`, in the bib) `[A — NeurIPS proceedings abstract]`: models human
   *discretion to decline*, and argues you must also acquire information about the discretion model.
   **No quantified discretion cost visible at abstract level.** Title varies between listings
   (*"Deep Bayesian Active Learning for Selective Labeling with Instance Rejection"* on the virtual
   site); **no arXiv id found** — check before citing a preprint number.

5. **`wang2015qsnn` — the standing "authors unverified" debt is now CLOSED** at `[A]`:
   **Wang, Liantao; Hu, Xuelei; Yuan, Bo; Lu, Jianfeng, "Active learning via query synthesis and
   nearest neighbour search", *Neurocomputing* 147:426–434 (2015)**, doi 10.1016/j.neucom.2014.06.042.
   Synthesise, then label the **nearest real instance**, explicitly to keep the query recognisable to
   a human oracle. ⚠ **ScienceDirect 403 — the body was NOT read; we still do not know whether they
   quantify the snap's cost.** New verification debt.

6. **Reached-and-rejected / unreachable, recorded so nobody re-runs them:** Canal, Massimino,
   Davenport & Rozell, *Active embedding search via noisy paired comparisons*, ICML 2019
   (arXiv:1905.04363) — **note the author list includes Massimino; our earlier "Canal, Fenu & Rozell"
   was wrong** `[A]`; Chumbalov, Maystre & Grossglauser, ICML 2020 (arXiv:1905.05049) — derives an
   optimal query form but over a discrete object set from the start `[A]`; Schumann & Rehbein,
   CoNLL 2019 (K19-1044), VAE membership-query synthesis for sentences, **PDF extraction failed** —
   a likely source of unlabelable-rate numbers, worth one more attempt `[A]`. Constrained-CAT
   relative-efficiency numbers and van der Linden's shadow-test approach: `[S]`, **paywalled or
   captcha-walled, no figure verified.** GenMQD remains `[S]` and unverifiable.

---

# Q5 — The softmax-weighted centroid as a policy parameterisation

**VERDICT: the operator is published, the exact object is published, and — the correction that
matters — our design sheets have been calling it by the wrong name and asserting it is untried in
recsys when it is not.**

## ★ NAME CORRECTION, BINDING

arXiv:2602.08616 is **DGRL — "Breaking the Grid: Distance-Guided Reinforcement Learning in Large
Discrete Action Spaces", Hoppe, Akkerman, van Heeswijk & Schiffer**, preprint (v1 9 Feb 2026), no
venue `[V — abs page + HTML full text]`. **"DBU" is only its update rule (Distance-Based Updates);
"SDN" (Sampled Dynamic Neighborhoods) is the other half.** Every CASPER sheet that names the method
"DBU / Breaking the Grid" — `PAPER_C_DESIGN.md` §C1, `PAPER_C_TODO.md`, `FABLE_BRIEF_PAPERC.md`,
`PAPER_B_POLICY_DESIGN.md`, `docs/archive/PLAN_NEXT_2026-07-14.md` — is carrying a second-hand
transmission error. **Fix it before it reaches a draft.**

## What DGRL actually does, and two corrections to our record

- **The target IS our object** `[V]`: a value-softmax over sampled candidate actions at temperature
  `τ_b`, with the actor minimising the squared distance from its proto-action to that weighted mean.
- **Theorem 4.2** bounds the per-sample gradient noise by `G² Tr(Cov[ā])` and states *"Both sides are
  independent of |𝒜|"* `[V]`. **That is a bound whose two sides do not scale with the action count —
  not a measured variance-independence claim. Do not overquote it.**
- ⚠ **Correction 1: the candidates are NOT retrieved kNN neighbours.** Algorithm 2 draws them by
  Gaussian perturbation of the proto-action, **rounded onto a metric lattice** `[V]`. DGRL presumes
  an action space with coordinate structure it can perturb inside — that is what "breaking the grid"
  means. **Our candidates are an enumerated question bank with no such geometry, so SDN does not
  transfer and we must say what replaces it.**
- ⚠ **Correction 2: recommendation is one of its evaluation domains** — MovieLens, recommend N of
  343 movies, choose-or-terminate `[V]`. **We may not write that the blend target is untried in
  recsys.** (343 items with a choose-or-terminate simulator is a long way from an interview policy;
  that distance is a sentence we can write, "untried" is not.)

## Is it a NAMED pattern? — the per-source verdict

| Source | Is it literally a utility-softmax-weighted average of candidate embeddings? |
|---|---|
| **DGRL**, `hoppe2026dgrl` `[V]` | **YES** — the exact object, as an explicit regression target |
| **MPO**, Abdolmaleki et al., ICLR 2018 (`abdolmaleki2018mpo`) `[V]` | **NEARLY — the true ancestor.** The E-step's self-normalised weights *are* a softmax over candidate values; for a Gaussian policy the M-step's weighted-ML mean *is* the softmax-weighted centroid. It **contains** the object without naming it. DGRL cites MPO as its own lineage. |
| **AWR** (`peng2019awr`) / **AWAC** (arXiv:2006.09359) `[V]` | **NO** — per-sample exponentiated-advantage-weighted log-likelihood; **no averaged action is ever formed.** ⚠ Citing these as precedent for a blend would be a factual error. |
| **Wolpertinger** (`dulacarnold2015deep`) `[V]` | **NO — confirmed argmax** over the kNN set under the critic. Our record was right. |
| **SAVO**, arXiv:2410.11833 `[A]` | **NO** — samples candidates and **selects the highest-valued**; argmax family. |
| **DeCap** (`li2023decap`) `[V]` | **YES mechanically, but it is not a policy** — softmax over *similarity*, not utility, applied training-free at decoding time over a support memory. |
| **Spatial softmax / soft-argmax**, Levine, Finn, Darrell & Abbeel, JMLR 17(39) 2016 (`levine2016visuomotor`) `[V]` | **The operator, yes; the name, weakly.** Their own term is "spatial softmax"; "soft-argmax" appears **once**, and it is over image pixel coordinates. |

Open-vocabulary searches for "expected embedding action selection", "soft action selection over
candidate embeddings", "Boltzmann-weighted action embedding average" returned **no established term**
`[A — search-level absence, so this is [A] not [V]]`.

**⇒ The correct citation chain is: operator → `levine2016visuomotor`; RL ancestor →
`abdolmaleki2018mpo`; exact statement of our object → `hoppe2026dgrl`; text-bank analogue →
`li2023decap`.** What is *not* covered by DGRL, and is therefore where any novelty must live:
(i) where the candidates come from (an enumerated question bank, not a perturbable lattice);
(ii) where the weights come from (a predicted one-step gain profile, not a bootstrapped critic — which
also voids DGRL's local-improvement guarantee); (iii) that the centroid is left **unsnapped** and
handed to a renderer.

---

# WHAT THIS PASS CHANGES — sentences we may and may not write

## ⛔ DEAD — do not write these

1. **"The best off-catalogue question direction is the principal axis of the residual covariance"
   presented as our finding, insight, or design contribution.** `ji2008bcs` (2008, verbatim) and
   `nadagouda2026infosynth` (2026, Prop. 1) both state it. It may appear only as *"the standard
   design result, which we instantiate"*.
2. **"A closed-form, training-free emitter" as a novelty or efficiency claim.** Both prior instances
   are closed-form and training-free; that is the family's default, not a delta.
3. **"Our emitter adapts to the answers"** — for a *pure* Σ-functional rule this is provably false
   under linear-Gaussian belief; the whole query sequence is precomputable. Anything we claim about
   adaptivity must come from the objective's dependence on the posterior **mean** (the escape already
   documented in `rank_targeted_acquisition.md`), not from the covariance.
4. **"Principal-component structure is new to recommender elicitation."** `goldberg2001eigentaste`
   (Information Retrieval 4:133–151, 2001) built the whole algorithm on eigenvectors of the gauge-item
   correlation matrix, twenty-five years ago.
5. **"DBU / Breaking the Grid"** as a method name — it is **DGRL**; DBU is the update rule.
6. **"The softmax-weighted-blend target is untested in recommendation."** DGRL evaluates on MovieLens.
7. **"Nobody prices answerability against informativeness."** Already dead five times over per
   `active_learning_elicitation.md` §0.5; this pass adds nothing that revives it.
8. **"No one selects interview questions using the posterior covariance of a latent USER vector."**
   Dead — `zhang2020activemf` (AOAS 14(3):1182–1206, 2020) minimises the trace of exactly that
   covariance, adaptively, per respondent, and words it in eigenvalues.
9. **"Optimal experimental design has not been applied to preference elicitation in recommender
   systems."** Dead four times: `anava2015budget`, `fonarev2016rectangular`, `sepliarskaia2018spq`,
   and RS's own handbook chapter, which already names a **Variance Reduction** family containing
   A-optimality.
10. **"Existing interview trees use variance criteria."** Misleading and it will read as sloppy:
    Golbandi and Zhou minimise **squared prediction error on ratings**; Rashid uses **discrete
    entropy**. Squared-error-on-ratings is not covariance-in-latent-space.
11. **"IGCN builds a decision tree over user clusters."** False — Rashid et al. state that a direct
    decision-tree application *"is not practically feasible in our problem domain"* and score items
    by information gain against k-means cluster labels instead.
12. **"Silva & Carvalho"** — wrong author. It is **Silva & Carin** (Duke), KDD 2012.
13. **Anything about Zhao et al. CIKM 2013's UCB term, or about Loepp et al. CHI 2014's factor
    ordering.** Neither was read; see the debt table.

⚠ **Guard against a mis-application of item 3 and safe-sentence 5.** `CLAUDE.md` forbids arguing
"static is optimal", and rightly — that theorem assumes linear-Gaussian **and** variance-only, and we
have neither. Nothing in this pass licenses a general static-is-optimal claim. What it licenses is
much narrower: **if we build an emitter whose rule is a functional of Σ alone, then *that arm* is
static-equivalent and must be labelled as such.** The escape remains the objective's dependence on
the posterior mean, exactly as `rank_targeted_acquisition.md` sets out.

## ✅ SAFE — these survive, stated exactly this way

1. *"The direction our emitter produces is the classical design answer — the leading eigenvector of
   the belief covariance (Lindley 1956; Ji et al. 2008), re-derived for pairwise preference queries by
   Nadagouda et al. (2026). We adopt it rather than propose it."*
2. *"What has never been done with that direction is to **render it as language and put it to a
   person**: Ji et al. offer only 'design it on the fly, or pick from a library L'; Info-Synth poses
   its synthesised pair only where a latent point is itself instantiable — uniform random vectors and
   robot controller gains — and in its one semantic space it snaps to real summaries before any
   response is collected."* `[V, all clauses]`
3. *"Info-Synth derives the Mahalanobis information loss of snapping a synthesised query onto a pool
   (ΔI = ½‖Δz‖²_M) and never reports it as a number. We report the analogous quantity, and against a
   constraint they do not consider: not pool membership but human answerability."* — **the strongest
   single sentence available to us from this pass**, provided we use the true MI difference where the
   gap is large.
4. *"Principal-component structure in recommender elicitation is old (Eigentaste, 2001), but there it
   is the analysis basis over a fixed universal gauge set; nobody uses a leading eigen-direction as
   the **content** of a question."*
5. *"A design criterion that is a functional of the posterior covariance alone is realisable as a
   fixed, answer-independent question sequence; we therefore do not claim adaptivity for it, and we
   report it as the static arm."* — this converts a weakness into a correctly-labelled control.
6. *"We use an MPO-style exponentiated-value reweighting as a soft-argmax over a candidate question
   bank, as in DGRL's distance-based update."* — cite, do not christen.
7. ★ *"Every prior method that uses belief geometry to choose an elicitation question reduces the
   covariance to a **scalar** — a trace or a determinant — and uses it to **score a finite pool**
   (Anava et al. 2015; Fonarev et al. 2016; Sepliarskaia et al. 2018; Zhang et al. 2020). We take the
   **argmax direction itself** as the emitted object."* — this is the sharpest surviving form of the
   claim after the RS sweep, and it must be stated **against `zhang2020activemf` and
   `fonarev2016rectangular` by name**, or a reviewer will supply them for us.
8. *"The maximal-volume and optimal-design families were still unjoined as of 2016, in the words of
   the maximal-volume authors themselves."* `[V-delegated — Fonarev et al.'s own future-work
   sentence; re-read before it goes into prose]`

## New verification debts opened by this pass

| # | Debt | Why it matters |
|---|---|---|
| 1 | **`wang2015qsnn` body** (ScienceDirect 403) — does it quantify the cost of snapping? | If it does, our Q4 cell narrows further |
| 2 | **`yang2020poptimal`'s p = 0.95 ↔ 0.35 logit ↔ 5% information** (SAGE 403 / PMC captcha) | We would be importing the metric's shape and its threshold |
| 3 | **`chaloner1995bayesian` full text** (paywalled) | We currently cite the taxonomy from an abstract |
| 4 | **SEL-BALD's discretion-cost quantity** and its correct title / arXiv id | Cited in our bib with no id |
| 5 | **Schumann & Rehbein CoNLL 2019** (PDF extraction failed) | Possible unlabelable-rate numbers for membership-query synthesis in text |
| 6 | The Bıyık Likert **means** live only in Figure 6 — never quoted here | Guard against a figure-read creeping into prose |
| **7 ★ HIGH** | **`zhang2020activemf` §§3–4** — do their adaptive gains survive under the **Gaussian** likelihood, or arrive only with the **ordered-logit** one? | This is a DESIGN LEAD, not a citation chore: under Gaussian the covariance update is answer-free and their "adaptive" sequence is precomputable; under ordered logit it is not. **If the gains need the non-Gaussian likelihood, that is published evidence for where our own adaptivity has to come from.** |
| **8 ★ HIGH** | **Loepp, Hussein & Ziegler, CHI 2014** (doi 10.1145/2556288.2557069) — ACM 403, two open PDFs would not extract | Reported to elicit by presenting item sets at the **low and high extremes of each latent factor in turn**. If true, "interrogate along a latent direction" exists in recsys since 2014. **Write no novelty sentence about asking along a direction until this is read.** |
| 9 | **`zhao2013interactive` (CIKM 2013)** — PDF would not extract | A linear-UCB / Thompson variant would carry a `sqrt(vᵀΣv)` term, i.e. another covariance-geometry instance inside recsys. **Assume nothing.** |
| 10 | **`elahi2016survey` (Comput. Sci. Rev. 20:29–50)** — ScienceDirect 403 | §Q2's statement about what that survey does NOT contain rests on our own earlier `[S]`, not on this pass |
| 11 | **Christakopoulou KDD 2016 · Vanchinathan RecSys 2014 · Sutherland KDD 2013** — never reached | That corner of the literature is **unchecked, not clear** |
| 12 | **Silva & Carin KDD 2012's trace surrogate** — the detail comes from Zhang et al.'s *description* of it, not their own equations | Do not attribute a criterion to a paper through a third party |
