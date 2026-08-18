# Findings: LLM VERBALISATION OF LATENT VECTORS — and how anyone validates FAITHFULNESS rather than FLUENCY

*Deep research 2026-08-18. Question asked, verbatim from the brief: we want a policy to emit a point in a
recommender's embedding space — possibly a point no catalogue item or attribute occupies — and have an LLM
turn it into a natural-language question. **An LLM will render ANY vector into fluent plausible text,
including a scrambled or random one.** So fluency proves nothing. How (or whether) does anyone establish
that a rendered latent is faithful?*

Serves `docs/design/PAPER_C_TODO.md` STEP A questions 2, 3, 6 (and touches 1 and 5).
Read **with** `text_to_cf_projection.md` (ELM's verified negative on text→embedding; our ridge adapter) and
`llm_elicitation_and_metalearning.md` §6.2–6.6 (ELM, ACING, VDLM, Info-Synth, and the negative sweeps —
those are NOT re-derived here).

**Provenance markers, used on every claim:**
- **[V]** — I extracted the primary PDF/HTML myself and read the sentence/number.
- **[V\*]** — the document was fetched and read in full by the fetch tool; quotes/numbers are reproduced
  from that reading of the primary text, but I did not eyeball the table myself. Strong, but re-verify
  before a number goes in the paper.
- **[A]** — abstract / landing page only.
- **[S]** — second-hand (search snippet, blog, third-party summary). A lead, never evidence.

---

## 0. HEADLINE — five sentences

1. **Faithfulness validation of embedding→text is a THIN literature with a THICK neighbouring one.** In
   recsys/embedding-verbalisation proper (ELM, ctELM, EAGLE) the entire toolkit is: round-trip
   re-embedding cosine ("semantic consistency"), a downstream-task check ("behavioural consistency"), and
   human ratings. In *interpretability* the toolkit is far more developed — and far more sceptical.
2. **★ THE SCRAMBLED/RANDOM CONTROL EXISTS — but not in the verbalisation-of-embeddings literature. It
   exists in interpretability, and when it was run it BROKE the field's metrics.** Heap et al.
   (arXiv:2501.17727) trained SAEs on **randomly initialised** Pythia transformers and got
   auto-interpretability scores *equal to or higher than* the trained model (Pythia-6.9b fuzzing AUROC:
   trained **0.79**, randomised-excl-embeddings **0.87**, step-0 **0.88**, randomised-incl-embeddings
   **0.87**) [V]. Their *one* control that did separate was a **scrambled-embedding** control — resampling
   each token's embedding i.i.d. per occurrence so no consistent vector→meaning map exists — which sits at
   **AUC 0.50, chance** [V]. That is our design, already published, in another field, with the result that
   makes it necessary.
3. **★ The single strongest borrowable idea is Hewitt et al.'s PLUG-IN EVALUATION** (arXiv:2510.08506,
   preprint Oct 2025): a verbalisation is faithful to the degree that **substituting the text back for the
   latent reproduces the latent's causal effect**, scored as *percent of the concept gap closed* between a
   base model (0%) and the latent itself (100%) [V]. It is normalised, causal, cheap, and makes "fluent but
   empty" score ~0 by construction. Their own numbers range from **−6% to 86%** across concepts, i.e. it
   discriminates [V].
4. **Off-manifold decoding demonstrably COLLAPSES, and the published fix is a soft snap.** DeCap (ICLR 2023)
   feeds a CLIP *image* embedding to a decoder trained only on *text* embeddings — the exact off-manifold
   situation of our emitted point — and CIDEr on MSCOCO falls to **8.1** vs **42.1** for the same decoder
   given a softmax-weighted blend over a support memory of text embeddings; a **hard** nearest-neighbour
   snap scores **27.1**, pure retrieval **25.6** [V]. So: *soft blend ≫ hard snap > retrieval ≫ raw
   off-manifold vector*, in a published ablation. This is DBU/Wolpertinger's argument, pre-made, with
   numbers, in captioning.
5. **The cell we care about is empty in the precise sense that matters.** No paper we found renders a
   latent into a **question addressed to a user**, and **no paper in the embedding-verbalisation line runs
   a scrambled-latent control on its renderer**. ELM's own consistency metrics have **no random baseline of
   any kind** [V\*]. That gap is a real, defensible methodological contribution — and it is *cheap*, which
   is the risk: a reviewer will ask why nobody did it, and the honest answer is "because their claim was
   plausibility, and ours is faithfulness".

---

## 1. Direct answers to Q1–Q5, with verdicts

### Q1. How do people validate faithfulness of embedding-to-text? — **VERDICT: five families, and only two of them are causal.**

| # | Family | What it actually measures | Who does it |
|---|---|---|---|
| 1 | **Round-trip re-encoding** (decode → re-embed → cosine to the source vector) | Whether the text lands back near the vector *under the same encoder* | ELM (semantic consistency); ctELM; Vec2Text (as its training objective); ZSInvert (as its search objective) |
| 2 | **Downstream-task utility** (use the generated text in place of the vector, measure task performance) | Whether the text carries the *decision-relevant* content | ELM (behavioural consistency, via a language-based ranker); LatentQA; Activation Oracles |
| 3 | **Causal substitution / plug-in** (put the text back where the latent was, measure the same effect) | Whether the text *does what the latent did* | Hewitt et al. neologism plug-in eval; Huang et al. intervention mode; (partially) LatentQA steering |
| 4 | **Human judgement** (raters score relevance/coherence; or experts try to tell synthetic from real) | Plausibility, coherence — **not** faithfulness | ELM (100 raters, 0–1 scale); ctELM (2 clinicians, 50 pairs, "fools experts 44%"); Manderlier et al. IntRS 2025 (326 participants) |
| 5 | **Discrimination / classification scoring** (given the explanation, can a model pick out the activating cases?) | Whether the description is *predictive* of the latent's behaviour | Bills et al. simulation scoring; Paulo et al. fuzzing/detection AUROC; PRISM; AxBench concept-detection |

**The important structural point:** families 1 and 5 are **correlational and gameable**; 3 is the only one
that is causal end-to-end; 2 is causal only if the downstream task is not the task the renderer was trained
on. ZSInvert makes the gameability of family 1 explicit: it *searches* text to maximise cosine to the target
embedding, achieving **cosine > 90** while F1 against the true text is only **> 50** [V]. **A high
round-trip cosine is therefore evidence that the renderer optimised for round-trip cosine, not that the
text means what the vector means.** Say this in the paper; it is our justification for adding a *control*
rather than adding another *metric*.

### Q2. Does anyone use a SCRAMBLED / RANDOM / SHUFFLED latent as an explicit control? — **VERDICT: YES in interpretability, NO in embedding verbalisation. The design exists; its application to a latent RENDERER is, as far as we found, unoccupied.**

**Where it exists (adopt these as the precedent — do not claim the idea):**

- **Heap, Lawson, Farnik & Aitchison, "Automated Interpretability Metrics Do Not Distinguish Trained and
  Random Transformers", arXiv:2501.17727v2 (27 Jan 2026)** [V]. The canonical citation. Two distinct
  nulls: (a) **randomised weights** — "SAEs trained on randomly initialized transformers produce
  auto-interpretability scores and reconstruction metrics that are similar to those from trained models";
  (b) a **scrambled-embedding control** where "a given token does not have a consistent embedding vector.
  For this variant, we expect auto-interpretability to perform at the level of chance" [V]. Result: the
  weight-randomisation null **fails to separate** (numbers in §0.2); the scrambled control **does**, at AUC
  0.50 [V]. Their closing recommendation is verbatim what we want to cite: *"we recommend treating common
  SAE metrics as useful but insufficient proxies for mechanistic interpretability and argue for routine
  randomized baselines and targeted measures of feature 'abstractness'"* [V].
  ⚠ Their own related-work caveat: Bricken et al. 2023 *did* find auto-interp separated random from trained
  **one-layer** transformers, and Heap et al. find the gap is real for **small** models (Pythia-70m) and
  closes with scale [V]. So "a random control separates" is model-scale-dependent — do not over-generalise.
- **Li, Ceballos Arroyo, Rogers, Saphra & Wallace, "Do (Natural Language Descriptions of) Model Activations
  Convey Privileged Information?", arXiv:2509.13316 (v4), ICML 2026** [V\*]. Controls by **knowledge
  misalignment** rather than by scrambling: PersonaQA-**Shuffled** removes sociodemographic correlations;
  PersonaQA-**Fantasy** uses fabricated personas and verbalizers score **zero accuracy** [V\*]. Their
  headline is the one that should scare us: *"a zero-shot baseline achieves high accuracy (matching or
  surpassing verbalization methods) despite operating on only text inputs"* [V\*] — i.e. the verbaliser's
  own priors, not the activation, were doing the work. **This is the exact failure mode our scrambled
  control is designed to catch, demonstrated in a neighbouring field in 2026.**
- **Oikarinen, Yan & Weng, "Evaluating Neuron Explanations: A Unified Framework with Sanity Checks", ICML
  2025, arXiv:2506.05774** [V\*]. Sanity checks on the **METRIC**: *extra labels* (random superclass of the
  perfect explanation) and *missing labels* (random subclass), with a pass bar of "decreases its score at
  least 90% of the time". *"Almost all commonly used metrics fail at least one of these simple sanity
  checks"*; only **Correlation, F1-score, IoU, cosine similarity and AUPRC** pass both [V\*]. Not a
  latent-side scramble — an **explanation-side** perturbation — but the cleanest published statement of
  "your faithfulness metric must itself pass a control before you trust it".
- **Huang, Geiger, D'Oosterlinck, Wu & Potts, "Rigorously Assessing Natural Language Explanations of
  Neurons", BlackboxNLP @ EMNLP 2023 (pp. 317–331), arXiv:2309.10312** [A]. Splits validation into an
  **observational** mode (does the neuron fire on all and only the strings the explanation picks out?) and
  an **intervention** mode (is the neuron a causal mediator of the explained concept?). Finding: GPT-4's
  explanations of GPT-2 XL neurons have "high error rates and little to no causal efficacy" even at high
  confidence [A]. **The observational/intervention split is the right skeleton for our validation section.**
- **Scattered shuffle controls in other domains** [S only, from search summaries — do not cite unread]:
  shuffled sample↔embedding correspondence dropping R² to ~zero in an auto-bidding study (arXiv:2605.05833);
  shuffled-pairing controls collapsing to chance in a rationale-verifier study (arXiv:2606.21678);
  random-embedding conditions in an EEG intent-decoding paper (arXiv:2601.20447). These establish that the
  **shuffle control is a normal move in ML generally** — good for us (nobody will call it exotic), bad for
  us (nobody will call it novel *as a technique*).

**Where it does NOT exist — the empty cells (see §4):** ELM, ctELM, EAGLE, DeCap, Vec2Text, ZSInvert,
vec2vec, LatentQA, Activation Oracles, Patchscopes/SelfIE, and every recsys "explain the latent factor"
paper we found. **Zero of them decode a random/scrambled latent and report what comes out.**

**⇒ RULING FOR PAPER C.** The scrambled-control *design* is pre-existing (Heap et al. 2026 is the citation;
Adebayo et al.'s saliency-map sanity checks are its ancestor, cited by Heap et al. [V]). What is unoccupied
is (i) applying it to a **generative renderer of a domain embedding** rather than to a metric over
explanations, and (ii) doing so where the rendered object is a **question**. Write "we are aware of no
embedding-verbalisation system that reports what its renderer produces from a scrambled latent" — never
"nobody has used a random control".

### Q3. Embedding inversion: state of the art and FAILURE MODES — **VERDICT: inversion is strong in-distribution, degrades OOD, and its objective *is* round-trip cosine, so it will fluently hallucinate on vectors no text produced.**

- **Vec2Text** (Morris et al., EMNLP 2023, arXiv:2310.06816) — hypothesiser + iterative corrector, trained
  per-encoder on ~5M passage/embedding pairs, 2 days on 4×A6000 [V, as reported by ZSInvert]. Failure modes:
  **sensitive to input length**, **degrades out-of-domain**, **defeated by small Gaussian noise on the
  embedding** [V for the noise claim via ZSInvert; [S] for length/OOD via the RecSys 2025 reproducibility
  study, arXiv:2507.07700 / doi 10.1145/3705328.3748155 — **unread, flagged**].
- **ZSInvert / "Universal Zero-shot Embedding Inversion"** — Collin Zhang, John X. Morris, Vitaly Shmatikov
  (Cornell), arXiv:2504.00147, preprint under review [V]. Zero-shot, encoder-agnostic, based on **adversarial
  decoding**: it *searches text to maximise similarity to the target embedding*. MS-MARCO: **F1 above 50,
  cosine above 90**; remains effective up to **σ = 0.01** embedding noise, where vec2text does not [V].
  **★ Load-bearing for us: cosine 90 with F1 50 is a quantified demonstration that round-trip cosine and
  semantic correctness come apart.**
- **vec2vec / "Harnessing the Universal Geometry of Embeddings"** — Rishi Jha, Collin Zhang, Vitaly
  Shmatikov, John X. Morris, **NeurIPS 2025**, arXiv:2505.12540 [S for venue + numbers: cosine "as high as
  0.96" to ground-truth target vectors, "perfect matching on over 8000 shuffled embeddings"]. Unpaired
  CycleGAN-style translation between embedding spaces. Relevant only as evidence that cross-space alignment
  is learnable without pairs — **not** as a faithfulness method. (Their "shuffled embeddings" is a
  *matching task*, not a control — do not miscite it.)
- **Rep2Text** (arXiv:2511.06571) [S] — decodes full text from a single LLM token representation, claims
  robustness under distribution shift; notes Vec2Text's corrector "provides only marginal gains on OOD data,
  despite converged training metrics" [S]. **Unread — best lead for a primary-source statement of vec2text's
  OOD failure.**
- **Does inversion hallucinate fluently OOD? — we found NO paper that measures this directly.** Every
  inversion paper evaluates on embeddings *produced by real text*, i.e. on-manifold by construction. The
  behaviour of an inverter on a **synthesised** vector (interpolation, extrapolation, random) is, as far as
  we found, **unmeasured and unreported anywhere**. That is the second empty cell, and it is the one our
  scrambled control fills.

**Consequence for the snap debate (already flagged in PAPER_C_TODO):** the correct justification for a snap
is *not* "no decoder exists". Decoders exist and are good. It is (a) **answerability** (our existing
argument) and (b) **DeCap's measured collapse off-manifold**, below.

### Q4. Adapter/soft-prompt (ELM-style) vs prompting with retrieved nearest-neighbour concepts — **VERDICT: the adapter is standard in the ELM lineage and in LLM-RecSys; the soft PROJECTION onto a support memory is standard in captioning; and the one head-to-head we found says soft-projection ≫ hard-NN ≫ retrieval ≫ raw off-manifold vector.**

- **Adapter into token space (ELM-style)** is the dominant 2024–2026 pattern: ELM's "two-layer MLP for the
  domain-embedding adapter layer E_A" [V, recorded previously]; **ctELM** reuses it verbatim for clinical
  trials [V\*]; the LLM-RecSys soft-prompt line (A-LLMRec, LLaRA, E4SRec, PEPLER, semantic-ID prompting)
  injects CF vectors as soft tokens [S, already in `text_to_cf_projection.md`]. **Cost:** training the
  adapter against the frozen LLM (ELM: two-stage, 20k then 300k iterations, batch 32, ≈7 epochs [V\*]).
  **Faithfulness evidence:** ELM/ctELM report semantic + behavioural consistency; **neither compares against
  a retrieval-prompt alternative.**
- **Retrieved-neighbour prompting** is the training-free alternative and is the *baseline* in captioning
  (CLIPRe). It is bounded by the support corpus: it can only say things the corpus already says.
- **★ The one measured comparison — DeCap, Li, Zhu, Wen & Yang, ICLR 2023, arXiv:2303.03032** [V].
  MSCOCO Karpathy-test, decoder trained on CC3M text only:

  | Inference strategy | What it is | B@4 | CIDEr | NoCaps overall |
  |---|---|---|---|---|
  | CLIPRe | pure retrieval, no decoder | 4.6 | 25.6 | 28.2 |
  | DeCap-VD | **feed the off-manifold (image) vector straight to the decoder** | **1.2** | **8.1** | **8.5** |
  | DeCap-NND | **hard snap** to nearest support-memory text embedding | 5.3 | 27.1 | 28.8 |
  | DeCap (PD) | **soft snap**: softmax-weighted blend over the whole support memory | **8.8** | **42.1** | **39.7** |

  Verbatim: *"Due to the modality gap between CLIP image embedding space and text embedding space, it is
  impractical to directly take the CLIP image embedding as the prefix embedding"*; *"this method does not
  achieve satisfying results in most scenarios"* [V]. The projection is
  `v_proj = Σ_i softmax((m_i·v)/τ) · m_i`, τ = 1/100 for images, **training-free**, support memory = 1M text
  embeddings [V].
  **⇒ This is Paper C's DBU / "breaking the grid" argument, already published in captioning, with a 5×
  CIDEr gap between raw off-manifold decoding and the soft blend, and a 1.55× gap between soft and hard
  snap.** Strongest external support for emitting a point and rendering it via a *weighted blend of bank
  questions* rather than a hard snap. Cite it in the method, not just related work.
- **Is one more FAITHFUL? — no direct evidence.** DeCap measures caption quality, not faithfulness to the
  vector. **Empty cell.**

### Q5. Verbalising a QUESTION from a latent (as opposed to an entity or a profile) — **VERDICT: EMPTY, and this pass did not dislodge the prior sweep.**

- Everything found renders a latent into an **entity description** (ELM movies, ctELM trial abstracts,
  DeCap captions), a **profile** (ELM user profiles, Montazeralghaem 2025), an **explanation** (all of
  interpretability), a **retrieval query** (RGAlign-Rec), or a **task instruction** (ACING).
- The question-asking literature goes the other way round: it **selects then phrases** (PEBOL stage 2,
  OPEN's `φ⁻¹`, BED-LLM, CLARINET). In every case the object phrased is a **discrete, enumerated** item /
  attribute / feature-vector, never a free point in a learned space.
- **Closest miss remains Info-Synth (arXiv:2605.26072)** — synthesises queries in a learned latent space and
  then **presents them as a pair of dataset items**, i.e. it re-snaps [V-partial, recorded in
  `llm_elicitation_and_metalearning.md` §6.4].
- **Closest live neighbour, and it needs a full read: EAGLE — Tennenholtz, Chow, Hsu, Shani, Liang &
  Boutilier, "Embedding-Aligned Language Models", NeurIPS 2024, arXiv:2406.00024** [A]. An RL agent treats a
  frozen LLM as an *environment* and "iteratively steers the LLM's generation towards optimal regions of the
  latent embedding space"; MovieLens-25M and Amazon; framed as surfacing **content gaps reflecting latent
  user demand**. That is *a policy choosing a target point in our exact domain and an LLM generating toward
  it* — the ELM group's own sequel, and the nearest thing to Paper C's mechanism. It generates **content**,
  not a question, and there is no user in the loop [A]. **⚠ BLOCKING: read EAGLE in full before writing
  Paper C's related work.** It was not in our INDEX.

---

## 2. Per-paper entries

### A. The embedding-verbalisation line (our direct ancestry)

**A1. ELM — Tennenholtz, Chow, Hsu, Shani, Liang, Boutilier et al., "Demystifying Embedding Spaces using
Large Language Models", arXiv:2310.04475 (Google Research, 2023). Venue: not verified — do NOT write
"ICLR 2024".**
*Mechanism.* A two-layer MLP adapter maps vectors from a frozen domain model (MovieLens CF item/user
embeddings; also Amazon) into the LLM's token-embedding space; the LLM then describes that vector across 24
movie tasks. Decodes **synthesised** points: α = 0.5 movie interpolations, random user-pair interpolations,
CAV extrapolations. [V, recorded previously]
*How they validate faithfulness.* Three things, no controls.
- **Semantic consistency**: *"SC(o,w^s) = d(P(o), w^s), where P(o) projects the output to the domain
  embedding space 𝒲 and d is the distance metric in 𝒲"* — i.e. **re-embed the generated text and take
  cosine to the source vector** [V\*]. Reported range **0.77–0.96** (movies, Table 2), **0.89–0.96**
  (Amazon, Table 4) [V\*].
- **Behavioural consistency**: *"BC(o_u,u) = RankingScore(R_u, ρ(o_u)), where ρ(o_u) are the ranks of the
  movies in 𝒰 for user u provided by the language-based ranker"* — Spearman rank correlation, **0.46–0.96**
  (movies), **0.81–0.94** (Amazon) [V\*].
- **Human evaluation**: 100 raters, three dimensions (plot consistency, linguistic coherence, task quality),
  scale 0–1; **0.37–1.0** (movies), **0.7–0.94** (Amazon) [V\*].
*Baselines.* Text-only LLMs (PaLM 2-L, GPT-4). **No random/scrambled/shuffled embedding baseline of any
kind** [V\*]. **This is the empty cell, in the paper that binds us.**
*Training.* Two stages: 20k iterations with E_0, M_0 frozen, then 300k full-model iterations, batch 32,
≈7 epochs [V\*].
*Status:* **citation-only as a baseline** (no shared ruler, no user, no policy); **mechanism precedent** for
everything Paper C renders.

**A2. ctELM — Ondov, Chang, Zhou, Giuffrè, Xu (Yale), "ctELM: Decoding and Manipulating Embeddings of
Clinical Trials with Embedding Language Models", arXiv:2601.18796v1, 26 Jan 2026. No venue stated.** [V\*]
*Mechanism.* ELM transplanted to clinical trials: two-layer MLP adapter aligns a trial-embedding space to a
base LLM's token space; generates trial abstracts from embeddings; also interpolates and steers along CAVs
for sex/age of subjects.
*How they validate faithfulness.* (i) **Semantic consistency** (round-trip cosine): **0.86–0.87** on
abstract reconstruction vs **0.82** for Vec2Text baselines. (ii) **Expert discrimination**: two annotators
(one MD, one MBBS), 50 pairs; ctELM *"fools the experts 44% of the time"* when they try to distinguish
interpolated synthetic abstracts from real ones — approaching the 50% indistinguishability ceiling.
(iii) **CAV steering**: moving along the sex CAV changes the sex described in the generated abstract.
**No scrambled/random embedding control** [V\*].
*★ Why this matters more than its size suggests:* the 44% expert result is **exactly the trap the brief
names**. Indistinguishability from real text measures *fluency and plausibility*, and would be **equally
satisfied by a renderer fed a scrambled vector**, because the reader has no access to the source vector.
Cite ctELM as the state of the art in ELM-style validation *and* as the demonstration that the state of the
art measures the wrong thing. **Usable as a comparator design, not as a baseline.**

**A3. EAGLE — Tennenholtz, Chow, Hsu, Shani, Liang, Boutilier, "Embedding-Aligned Language Models",
NeurIPS 2024, arXiv:2406.00024.** [A] ⚠ **UNREAD — blocking for Paper C related work.**
RL agent steers a frozen LLM's generation "towards optimal regions of the latent embedding space";
MovieLens-25M + Amazon; application = surfacing content gaps reflecting latent user demand; mentions "an
optimal design of a state-dependent action set". No numbers, no round-trip validation, no control condition
visible from the abstract. **Nearest published relative of Paper C's policy→point→text loop.**

**A4. Manderlier, Lecron, Vu Thanh & Gillis, "From latent factors to language: a user study on
LLM-generated explanations for an inherently interpretable matrix-based recommender system", IntRS 2025
workshop @ RecSys 2025, arXiv:2509.18980.** [V\* for design, [A] for numbers]
*Mechanism.* Constrained matrix factorisation (BSSMF) yields human-interpretable latent user types; prompts
turn the latent factors + internal weights into textual justifications.
*How they validate faithfulness.* **They do not, in the causal sense.** 326 participants rate transparency,
effectiveness, persuasion, trust, satisfaction; explanation strategies are compared by varying what is fed
to the LLM. *"All explanation types are generally well received, with moderate statistical differences
between strategies."* **No control condition with random/scrambled explanations** [V\*].
*Status:* **citation-only.** The RecSys-side confirmation that our concern is unaddressed: the paper uses
the word "faithful" throughout and validates with a satisfaction survey.

### B. The interpretability line (where the controls live — this is the borrow)

**B1. ★ Heap, Lawson, Farnik & Aitchison (University of Bristol), "Automated Interpretability Metrics Do
Not Distinguish Trained and Random Transformers", arXiv:2501.17727v2 (27 Jan 2026).** [V]
*Mechanism.* Train per-layer TopK SAEs (expansion factor R = 64, sparsity k = 32, 100M RedPajama tokens) on
Pythia 70M→6.9B in five variants: **trained**, **Step-0** (at initialisation), **re-randomised excluding
embeddings**, **re-randomised including embeddings**, and a **control** in which token embeddings are
resampled i.i.d. per occurrence so "a given token does not have a consistent embedding vector". Score every
SAE latent with Paulo et al.'s **fuzzing** and **detection** auto-interp AUROC, using
Meta-Llama-3.1-70B-Instruct-AWQ-INT4 as explainer and simulator, 100 latents sampled per SAE.
*Result (Pythia-6.9b, 100M tokens, Fig. 1).* Fuzzing AUC — **Trained 0.79 · Randomized excl emb 0.87 ·
Step 0 0.88 · Randomized incl emb 0.87 · Control 0.50** [V].
*Verbatim.* *"the auto-interpretability scores were far more similar between the trained and randomized
models than with the control"*; *"we recommend treating common SAE metrics as useful but insufficient
proxies for mechanistic interpretability and argue for routine randomized baselines and targeted measures
of feature 'abstractness'"* [V].
*Caveats they state themselves.* Randomised-model AUROC is **low for small models** (Pythia-70m) and the gap
closes with scale; Bricken et al. 2023 found auto-interp *did* separate random from trained one-layer
transformers; Karvonen et al. found SAEs on random **chess** transformers *were* much worse [V]. So the
null's discriminating power is domain- and scale-dependent.
*★ Status: USABLE-DESIGN, and the citation for our control.* Two lessons to take literally:
(1) **use the SCRAMBLE (no consistent vector→meaning map), not just "a random vector"** — their random
*weights* null failed to separate, their scrambled *embeddings* null hit chance;
(2) run the control at our operating scale, because the separation is scale-sensitive.

**B2. ★ Hewitt, Tafjord, Geirhos & Kim (Google DeepMind), "Neologism Learning for Controllability and
Self-Verbalization", arXiv:2510.08506v1 (9 Oct 2025), preprint.** [V]
*Mechanism.* Freeze an LLM; add a new token embedding; train **only** that embedding to elicit a concept
(single-sentence answers, flattery, incorrect answers, AxBench concepts). Then **self-verbalize**: ask the
model for synonyms/definitions of the new token.
*★ How they validate faithfulness — PLUG-IN EVALUATION.* *"we take the prompt with the neologism, and we
replace the neologism with the verbalization. We measure whether the verbalization causes a similar impact
of the concept."* Scored as **percent of the concept gap closed**:
`(x − base-model score) / (training-data score − base-model score)` [V].
*Numbers (Table 4, single-word synonym verbalisations, % gap closed) [V]:*
single-sentence — statement **86**, text **60**, declaration **51**, record **0**;
short-text — identical **57**, precise **48**, exact **36**, accurate **27**;
refusal — nonfunctional **44**, unproductive **38**, ineffective **23**, inefficient **15**;
flattery — unrivaled **25**, unmatched **17**, unique **17**, uncommon **10**;
wrong-answer — tiny **24**, small **14**, little **14**, mini **14**;
long-text — lengthy **13**, extensive **11**, prolific **4**, detailed **−6**;
use-like — single **5**, solitary **2**, lone **1**, isolated **1**.
Anecdote with raw numbers: asking Gemma-3-4B-IT for a *"lack answer"* cut mean sentences **42.9 → 15.8**;
Gemini-2.5-Flash median **37 → 4** [V].
*★ The finding worth a paragraph in Paper C — "machine-only synonyms":* verbalisations "that look odd or
unrelated to humans, but consistently cause the behavior of a given neologism" [V]. **A faithful
verbalisation of a latent need not look right to a person.** A direct warning against validating our
rendered question by human plausibility rating, and a direct argument for a causal round-trip.
*Status: ★ USABLE-DESIGN, the strongest single methodological borrow in this file.* Note the honest
negatives (`detailed` at **−6**, `record` at **0**): the metric discriminates, including against the model's
own top-ranked verbalisation.

**B3. Li, Ceballos Arroyo, Rogers, Saphra & Wallace, "Do (Natural Language Descriptions of) Model
Activations Convey Privileged Information?", arXiv:2509.13316 (v4), ICML 2026.** [V\*]
*Mechanism.* Audits **Patchscopes/SelfIE** (patch a target model's activation into a verbaliser LLM) and
**LIT** (finetune a verbaliser on concatenated layer activations).
*How they validate.* Three attacks on the claim: (i) a **zero-shot text-only baseline** that "achieves high
accuracy (matching or surpassing verbalization methods) despite operating on only text inputs"; (ii)
**activation inversion** — reconstruct the input text from activations and predict from that instead, which
"results in competitive performance, suggesting that some tasks used to evaluate verbalization may just
describe the degree to which inputs are encoded in activations"; (iii) **PersonaQA** knowledge misalignment
— finetune targets on fake personas the verbaliser has never seen; **PersonaQA-Fantasy** gives **zero
accuracy**; **PersonaQA-Shuffled** removes sociodemographic correlations to kill statistical shortcuts
[V\*]. When the verbaliser is trained on the *same* data as the target, performance is comparable —
*"overlap of parametric knowledge"* explains the result [V\*].
*Status: ★ MUST-CITE.* The 2026 peer-reviewed statement that **a verbaliser's world knowledge, not the
latent, can produce the whole result**. Its **text-only baseline** is a second control we should run
alongside the scramble: render the question from the interview *state text* with no latent at all, and see
how much of the effect survives.

**B4. Huang, Geiger, D'Oosterlinck, Wu & Potts, BlackboxNLP @ EMNLP 2023, arXiv:2309.10312** [A] — see Q2.
*Status: MUST-CITE for the observational-vs-intervention framing.*

**B5. Oikarinen, Yan & Weng, ICML 2025, arXiv:2506.05774** [V\*] — see Q2.
*Status: USABLE-DESIGN.* Our own faithfulness metric should be shown to pass a perturbation check before we
report it — cheap, and it pre-empts "your metric is a rubber ruler".

**B6. Others in this family, triaged.**
- **LatentQA — Pan, Chen & Steinhardt, arXiv:2412.08686** [S]. LoRA-finetune a decoder LLM to answer
  open-ended questions about patched-in activations; validates by (a) supervised **reading** tasks with a
  known answer (hidden system prompts, relational knowledge) vs probing baselines, and (b) **control** —
  steering the target to behaviours unseen in training. (b) is a plug-in-style causal check.
  **Unread; no random-control condition confirmed.**
- **Activation Oracles — Karvonen, Chua, Dumas, Fraser-Taliente, Kantamneni, Minder, Ong, Sen Sharma, Wen,
  Evans & Marks, arXiv:2512.15674v2 (Jan 2026)** [V\*]. Norm-matched steering injection of activation
  vectors into placeholder tokens (`h'_i = h_i + ‖h_i‖·v_i/‖v_i‖` after layer 2); trained explainers beat
  the Patchscopes baseline. **No random/shuffled/mismatched activation control** [V\*]; the fetch also
  notes the model "will frequently produce an answer even when confidence is low" — i.e. it confabulates by
  default. **Empty cell, and an admission of the failure mode.**
- **PRISM — "Capturing Polysemanticity with PRISM: A Multi-Concept Feature Description Framework",
  arXiv:2506.15538, NeurIPS 2025 poster** [S]. Multi-concept feature descriptions; a *description score* and
  a *polysemanticity score*. Relevant because our emitted point is a **mixture** by construction. **Unread.**
- **AxBench — Wu, Arora, Geiger, Wang, Huang, Jurafsky, Manning & Potts, ICML 2025 (PMLR v267),
  arXiv:2501.17148** [S]. Steering/concept-detection benchmark; steering score = harmonic mean of concept,
  instruct and fluency scores (0–2, judged by gpt-4o-mini). Finding: **prompting beats all
  representation-based methods; SAEs are not competitive** [S]. Useful as a rubric template — and as a
  caution that the LLM-judge rubric is what the field now uses *in place of* a control.
- **Bills et al. 2023 (simulation scoring) and Paulo et al. 2024 (fuzzing/detection AUROC)** [V via Heap
  et al.'s description] — the two standard auto-interp scorers; fuzzing correlates with simulation scoring
  and is far cheaper. These are the *metrics* Heap et al. showed do not separate trained from random.

### C. Inversion / decoding (the "a decoder exists" family)

**C1. DeCap — Wei Li, Linchao Zhu, Longyin Wen, Yi Yang, "DeCap: Decoding CLIP Latents for Zero-Shot
Captioning via Text-Only Training", ICLR 2023, arXiv:2303.03032.** [V]
*Mechanism.* Train a 4-layer transformer decoder from scratch to **invert the CLIP text encoder** by prefix
LM on text only; at inference, project an off-manifold (image) embedding into text-embedding space via a
softmax-weighted blend over a support memory of ~1M text embeddings, then decode.
*Validation.* Standard captioning metrics (BLEU, METEOR, CIDEr, SPICE, CLIP-S/CLIP-SRef) — **no faithfulness
metric, no random control**. But the **ablation is the evidence we want** (table in §Q4).
*Status: ★ USABLE-DESIGN + MUST-CITE.* The published quantification of "an off-manifold vector decodes
badly, and a soft blend fixes it better than a hard snap".

**C2. Vec2Text — Morris et al., EMNLP 2023, arXiv:2310.06816** [S/V-indirect]; reproducibility study
**"Rethinking the Privacy of Text Embeddings", RecSys 2025, doi 10.1145/3705328.3748155 /
arXiv:2507.07700** [S, UNREAD]. Iterative hypothesiser + corrector. Failure modes reported second-hand:
length sensitivity, OOD degradation, defeated by Gaussian noise. **Verify before citing any number.**

**C3. ZSInvert — Collin Zhang, John X. Morris, Vitaly Shmatikov (Cornell), "Universal Zero-shot Embedding
Inversion", arXiv:2504.00147, preprint.** [V] — see Q3. **No random-embedding control.**
*Status: MUST-CITE for the cosine-vs-F1 gap* — our evidence that round-trip cosine is not faithfulness.

**C4. vec2vec — Jha, Zhang, Shmatikov & Morris, NeurIPS 2025, arXiv:2505.12540** [S]. *Citation-only.*

**C5. Rep2Text, arXiv:2511.06571** [S, UNREAD]. Best lead for a primary-source OOD-failure statement.

### D. Brain decoding — the field with the strictest controls, and we did not close it

**D1. Tang, LeBel, Jain & Huth, "Semantic reconstruction of continuous language from non-invasive brain
recordings", Nature Neuroscience 2023, doi 10.1038/s41593-023-01304-9.** [S — ⚠ **paywalled; the fetch was
blocked by Nature's IdP redirect; nothing here is verified**]
*Why it belongs in this file.* This is a literature whose entire epistemology is our problem: a generative
language model, conditioned on a latent (an fMRI-derived semantic vector), produces fluent text that *could*
be confabulated — and the field's answer is **identification accuracy against a null distribution built from
mismatched stimuli**, plus permutation testing. Code: `github.com/HuthLab/semantic-decoding`; preprint
bioRxiv 2022.09.29.509744.
*★ ACTION: highest-value unread item in this file.* If their null-distribution protocol is what we think it
is, it is the statistical form our scrambled control should take: not "the scramble looks worse", but "the
true pairing exceeds the mismatched-pairing null at p < α".

---

## 3. Table — faithfulness-validation methods and who uses each

| Method | Causal? | Gameable? | Who uses it | What they report |
|---|---|---|---|---|
| **Round-trip re-encode → cosine** | no | **yes — it is the training/search objective of every inverter** | ELM (SC 0.77–0.96 movies, 0.89–0.96 Amazon [V\*]); ctELM (0.86–0.87 vs Vec2Text 0.82 [V\*]); ZSInvert (cosine > 90 with F1 only > 50 [V]); Vec2Text (objective) | a cosine |
| **Downstream-task utility** | partly | yes if the task trained the renderer | ELM behavioural consistency (Spearman 0.46–0.96 movies, 0.81–0.94 Amazon [V\*]); LatentQA reading tasks [S]; Activation Oracles [V\*] | a task metric |
| **Causal substitution / plug-in** | **YES** | hard to game | **Hewitt et al. 2025 (% concept gap closed, −6 to 86 [V])**; Huang et al. 2023 intervention mode [A]; LatentQA steering [S] | normalised effect recovery |
| **Discrimination / simulation scoring** | no | yes | Bills et al. 2023; Paulo et al. 2024 fuzzing/detection AUROC [V via Heap]; PRISM [S]; AxBench concept score [S] | AUROC / rubric 0–2 |
| **Human rating of quality** | no | **yes — this is exactly the fluency trap** | ELM (100 raters, 0.37–1.0 [V\*]); Manderlier et al. (326 participants, 5 dimensions [V\*]) | Likert / 0–1 |
| **Human discrimination (real vs synthetic)** | no | **yes, and it is maximised by a good confabulator** | ctELM (2 experts, 50 pairs, 44% fooled [V\*]) | win rate vs 50% |
| **★ Random / scrambled LATENT control** | n/a (a control) | no | **Heap et al. 2026 (scrambled-embedding control at chance, AUC 0.50; random-WEIGHTS null FAILS to separate: 0.87 vs trained 0.79) [V]**; Li et al. 2026 PersonaQA-Shuffled / -Fantasy (zero accuracy) [V\*] | separation vs the true condition |
| **Prior-only / text-only baseline** | n/a (a control) | no | **Li et al. 2026 zero-shot text-only baseline matching or beating verbalisation [V\*]**; ELM's text-only PaLM 2-L / GPT-4 comparison [V\*] | how much survives without the latent |
| **Metric sanity checks (perturbed explanations)** | n/a | no | Oikarinen et al. ICML 2025 (only 5 common metrics pass both) [V\*] | pass/fail at 90% |
| **Null distribution over mismatched pairings** | n/a | no | brain decoding (Tang et al. 2023) [S, UNVERIFIED] | p-value vs permutation null |

---

## 4. EXPLICIT EMPTY CELLS

Stated as "we are aware of none", per project rule. Each is a candidate contribution, and each is also a
thing a reviewer could find first.

1. **No embedding-verbalisation paper reports what its renderer produces from a scrambled or random
   latent.** Checked: ELM [V\*], ctELM [V\*], EAGLE [A], DeCap [V], Vec2Text [S], ZSInvert [V], vec2vec [S],
   LatentQA [S], Activation Oracles [V\*], Patchscopes/SelfIE [S], Manderlier et al. [V\*].
   **Zero of eleven.**
2. **No inversion paper measures inverter behaviour on SYNTHESISED (interpolated / extrapolated / random)
   vectors.** All evaluate on embeddings produced by real text — on-manifold by construction. "Does an
   inverter hallucinate fluently off-manifold?" is, as far as we found, **unasked**.
3. **No head-to-head between an ELM-style adapter and retrieved-neighbour prompting on FAITHFULNESS.**
   DeCap's ablation compares them on caption quality [V], which is not the same thing.
4. **Nobody verbalises a latent into a QUESTION addressed to a user.** Confirms the prior sweep
   (`llm_elicitation_and_metalearning.md` §6.5) rather than extending it.
5. **No published faithfulness protocol for a point in a COLLABORATIVE-FILTERING latent** that does not
   route through either (a) the same encoder that made the point (round-trip) or (b) a human's sense of
   plausibility. Our text→CF ridge adapter gives us a **second, independently-fitted map** for the round
   trip — that asymmetry is unusual and worth naming.
6. **No paper combines the scrambled-latent control with the prior-only (text-only) control** on the same
   renderer. Li et al. have the second, Heap et al. the first, in different fields.

---

## 5. What Paper C should actually do (design implications, not results)

Consistent with the ruling in `PAPER_C_TODO.md` that validation must be a property of the POINT and must
never route through simulating an answer to it.

1. **Adopt plug-in evaluation, renamed for our setting (Hewitt et al. 2025 [V]).** Render the emitted point
   q into a question text; push the text back through the **frozen text→CF ridge adapter W** — a map fitted
   independently of the renderer, which is what stops this being a self-round-trip — and measure how much of
   q it recovers. Report as a **normalised gap closure**: 0 = the interview state with no question, 1 = q
   itself. Report the *per-item distribution*, not a mean; Hewitt et al.'s own numbers span −6 to 86.
2. **Run THREE controls on the same axis, not one.**
   - **Scramble** — permute q's coordinates (Heap et al.'s lesson: scramble the *correspondence*, don't just
     draw a Gaussian; their Gaussian-weights null failed to separate, their correspondence-scramble hit
     chance) [V].
   - **Prior-only** — render a question from the interview state text with the latent withheld (Li et al.'s
     zero-shot text baseline) [V\*]. This is the harder control and the one a reviewer will ask for.
   - **Nearest bank question** — already specified in `PAPER_C_EMISSION_DESIGN.md`; makes "carries
     information the bank does not" measurable rather than geometric.
3. **State significance as a null distribution over mismatched pairings**, not as a difference of means
   (brain-decoding convention; ⚠ protocol unverified — read Tang et al. first).
4. **Do NOT use human plausibility ratings as the faithfulness evidence.** ctELM's 44% expert-fooling result
   [V\*] and Hewitt et al.'s machine-only synonyms [V] are two independent demonstrations that human
   plausibility is orthogonal to faithfulness — in opposite directions. Human rating may appear as a
   *secondary* readability check only.
5. **Justify the snap on DeCap's numbers, not on "no decoder exists".** VD 8.1 vs PD 42.1 vs NND 27.1 CIDEr
   [V] is the published case for a **soft, weighted blend** over a support memory — exactly the DBU move
   already carried forward from June. Frame our contribution as *removing* the hard snap that Wolpertinger
   and Info-Synth still require, with DeCap as evidence that the soft version is better.
6. **Sanity-check our own faithfulness metric** with an Oikarinen-style perturbation (does it drop when we
   corrupt the rendered text?) before reporting it [V\*].

---

## 6. Sentences to NOT write

- ✗ "We are the first to use a random/scrambled control for a verbalisation." — Heap et al. 2026 (and
  Adebayo et al.'s saliency sanity checks upstream). Write: *we are aware of no embedding-verbalisation
  system that reports its renderer's output on a scrambled latent.*
- ✗ "An LLM verbalising a synthesised latent point is novel." — ELM; already settled in `PAPER_C_TODO.md`.
- ✗ "High round-trip cosine shows the text is faithful." — ZSInvert reaches cosine > 90 with F1 ~50 by
  *searching* for cosine [V].
- ✗ "Human raters could not distinguish our rendered questions, therefore they are faithful." — ctELM's 44%
  is the counterexample; a good confabulator maximises this number.
- ✗ "Random controls always separate." — Heap et al.'s random-*weights* null did **not** separate (0.87 vs
  0.79), and the separation is scale-dependent [V].
- ✗ "No one validates verbalisation faithfulness." — interpretability does, extensively and sceptically.
  The gap is domain-specific, not universal.
- ✗ Any Vec2Text number, until the reproducibility study or the original is read. Currently [S].

---

## 7. Open / unread — for the next pass, so nothing is re-found

**Blocking:**
- **EAGLE (arXiv:2406.00024, NeurIPS 2024)** — full read. Nearest relative of Paper C; MovieLens-25M; ELM
  group. Not previously in our INDEX.
- **Tang et al., Nature Neuroscience 2023** — the identification-accuracy / mismatched-null protocol.
  Paywalled; try the bioRxiv preprint (2022.09.29.509744) or `github.com/HuthLab/semantic-decoding`.

**Should read:**
- Rep2Text (arXiv:2511.06571) — primary source for vec2text OOD failure.
- Vec2Text reproducibility (arXiv:2507.07700 / RecSys 2025) — before quoting any vec2text number.
- LatentQA (arXiv:2412.08686) — whether its steering check is a genuine plug-in eval.
- PRISM (arXiv:2506.15538, NeurIPS 2025) — multi-concept description scoring; our emitted point is a
  mixture by construction.
- Heap et al. Appendices B / E / H — detection-scoring AUROCs, multiple seeds, entropy-vs-AUROC
  ("abstractness" proxy). The numbers here are from the Figure 1–2 discussion text only.
- AxBench (arXiv:2501.17148) — rubric design if we ever use an LLM judge.

**Searched and NOT found (do not redo):**
- "scrambled / random latent control" in embedding-verbalisation, captioning, or LLM-RecSys soft-prompt
  literature — nothing.
- A faithfulness comparison of adapter-injection vs retrieved-neighbour prompting — nothing.
- Any paper rendering a latent into a *question for a user* — nothing (confirms §6.5 of the elicitation
  file).
- Direct measurement of an inverter on synthesised / off-manifold vectors — nothing.

---

## 8. New INDEX / bib entries required

`heap2026autointerp` (arXiv:2501.17727) · `hewitt2025neologism` (arXiv:2510.08506) ·
`li2026verbalization` (arXiv:2509.13316, ICML 2026) · `huang2023neurons` (BlackboxNLP 2023,
arXiv:2309.10312) · `oikarinen2025neuroneval` (ICML 2025, arXiv:2506.05774) ·
`li2023decap` (ICLR 2023, arXiv:2303.03032) · `zhang2025zsinvert` (arXiv:2504.00147) ·
`jha2025vec2vec` (NeurIPS 2025, arXiv:2505.12540) · `ondov2026ctelm` (arXiv:2601.18796) ·
`tennenholtz2024eagle` (NeurIPS 2024, arXiv:2406.00024) · `manderlier2025latentfactors`
(IntRS@RecSys 2025, arXiv:2509.18980) · `karvonen2026oracles` (arXiv:2512.15674) ·
`pan2024latentqa` (arXiv:2412.08686) · `kopf2025prism` (NeurIPS 2025, arXiv:2506.15538) ·
`wu2025axbench` (ICML 2025, arXiv:2501.17148) · `tang2023semantic` (Nature Neuroscience 2023).
⚠ Several author lists are [S] — verify before the bib is compiled.
