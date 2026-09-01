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
   a scrambled-latent control on its renderer**. ~~ELM's own consistency metrics have **no random baseline of
   any kind** [V\*]~~ ⚠ **CORRECTED 2026-08-21 by the PDF (§10): FALSE as written.** ELM's Fig. 5 carries a
   **random baseline ("which rates movies randomly") and a ground-truth baseline** for behavioural
   consistency [V]. What ELM has no instance of is a **random / scrambled EMBEDDING** — a null on the input
   side. Write "no scrambled-latent control"; never "no random baseline of any kind". That gap is a real,
   defensible methodological contribution — and it is *cheap*, which
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
  injects CF vectors as soft tokens [S, already in `text_to_cf_projection.md`]. **Cost:** ELM's two-stage
  schedule, 20k then 300k iterations, batch 32, ≈7 epochs [V].
  ⚠ **CORRECTED 2026-08-21: ~~"training the adapter against the frozen LLM"~~ is WRONG for ELM.** The LLM is
  frozen only in **stage 1**; stage 2 fine-tunes **all** parameters (E_0, M_0, E_A), and the paper reports
  two-stage training is *"essential for convergence"* [V]. Also: **ELM's adapter emits ONE token**, not a
  sequence — longer soft-token slots are explicitly named as future work (§10.3).
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

**A1. ELM — Tennenholtz, Chow, Hsu, Jeong, Shani, Tulepbergenov, Ramachandran, Mladenov & Boutilier,
"Demystifying Embedding Spaces using Large Language Models", **ICLR 2024**; arXiv:2310.04475v2 (13 Mar 2024),
Google Research.**
✔ **VENUE + AUTHOR LIST CORRECTED 2026-08-21 by the PDF (§10).** ~~"Venue: not verified — do NOT write ICLR
2024"~~ — **retracted**: every page of the PDF is headed *"Published as a conference paper at ICLR 2024"*
[V]. ~~"Tennenholtz, Chow, Hsu, Shani, **Liang**, Boutilier et al."~~ — **WRONG, that was EAGLE's author
list pasted onto ELM.** ELM's nine authors are as above; **there is no Liang on ELM**, and Jeong,
Tulepbergenov, Ramachandran and Mladenov appear on ELM but not on EAGLE.
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
  **0.37–1.0** (movies), **0.7–0.94** (Amazon) [V\*, all four ranges re-verified cell-by-cell against
  Tables 2 and 4 of the PDF 2026-08-21 — every number matched]. ⚠ **CORRECTED: ~~"scale 0–1"~~ is the
  REPORTING scale, not the rating scale.** Raters answer on a **1–5 Likert agreement scale**, normalised to
  [0, 1] for the tables (Appendix C, Fig. 8) [V].
*Baselines.* Text-only LLMs (PaLM 2-L, GPT-4). **No random/scrambled/shuffled EMBEDDING baseline** [V —
PDF]. ⚠ **NARROWED 2026-08-21: ELM *does* carry a random baseline on the METRIC side** — Fig. 5's dotted
*"random baseline, which rates movies randomly"*, plus a ground-truth ceiling [V]. The input-side null is
the empty cell, and it is the only one we may claim.
*Training.* PaLM 2-XS (Otter). Two stages: 20k adapter-only iterations with E_0, M_0 **frozen**, then 300k
iterations fine-tuning **all** of (E_0, M_0, E_A), batch 32, ≈7 epochs [V — PDF §4]. ⚠ The second stage
is a **full-model fine-tune, not a frozen-LLM adapter fit** — see §10.3.
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
NeurIPS 2024, arXiv:2406.00024.** ✔ **READ 2026-08-21 — the blocking read is CLEARED. Full entry in
§9 below; this stub is superseded.** Headline: it is **encoder-only** (no decoder anywhere), it has **no
user in the loop and no question of any kind**, its automatic metric **is its own reward function**, and
its only independent evidence is **124 post-hoc raters**. It does *not* occupy any leg of Paper C's
conjunction — but it *does* kill the unqualified sentence "nobody has a policy optimising in a
recommender's latent space whose output is language". See §9 for the exact safe/unsafe wordings.

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
   latent.** Checked: **ELM [V** — upgraded [V\*]→**[V] (PDF)** 2026-08-21, §10**]**, ctELM [V\*], **EAGLE [V** — upgraded [A]→[V\*]→**[V] (PDF)** 2026-08-21, §9**]**, DeCap [V], Vec2Text [S], ZSInvert [V], vec2vec [S],
   LatentQA [S], Activation Oracles [V\*], Patchscopes/SelfIE [S], Manderlier et al. [V\*].
   **Zero of eleven.**
   ⚠ **NARROWED 2026-08-21.** EAGLE *does* carry a random-ACTION baseline (a uniform reference policy over
   its 100-action set: U(z) **0.49–0.51** vs EAGLE's **0.65–0.74** [V — PDF Table 2]). It carries no
   **scrambled-latent / mismatched-user** control. ⇒ write "no scrambled-latent control"; **never** "no
   random baseline of any kind" — that sentence is true of ELM and ctELM and **FALSE of EAGLE**.
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
- ✗ **"No published system has a policy optimising in a recommender's latent space whose output is
  language."** — **EAGLE, NeurIPS 2024, on ML-25M** (§9). The surviving qualifiers are *decode*,
  *interview*, *question*.
- ✗ **"EAGLE verbalises a latent point."** — it has **no decoder**; it edits text and encodes the result
  (§9). Getting this wrong in front of Boutilier would be the retraction.
- ✗ **"EAGLE reports no random baseline."** — it has a uniform reference policy over its action set (§9).

---

## 7. Open / unread — for the next pass, so nothing is re-found

**Blocking:**
- ~~**EAGLE (arXiv:2406.00024, NeurIPS 2024)**~~ — ✔ **DONE 2026-08-21, see §9.** ✔✔ **PDF-VERIFIED
  2026-08-21 (second pass): debt CLEARED.** Both the arXiv v2 PDF and the NeurIPS camera-ready PDF
  (54 pp., `1ce12df021f8fb43cf531c556efcebdc-Paper-Conference.pdf`) were extracted and read. Tables 2–4,
  the utility/MDP/action-space specifications and the 124-rater protocol **all confirmed**; §9 numbers are
  now [V]. **Two corrections landed** — the fifth author is **Ethan Liang**, not "Yi Liang" (both PDFs'
  own author blocks; only the NeurIPS *metadata page* says Yi Liang), and raters answer **eleven**
  questions, not three. **Residual: none.**
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
`tennenholtz2024eagle` was added to `new_chapters/references.bib` on 2026-08-21 with verified fields (§9).

---

## 9. ★ EAGLE — THE BLOCKING READ, CLEARED (2026-08-21)

**Full bibliographic record, verified.** Guy Tennenholtz, Yinlam Chow, Chih-Wei Hsu, Lior Shani, **Ethan
Liang**, Craig Boutilier, **"Embedding-Aligned Language Models"**, *Advances in Neural Information Processing
Systems 37* (**NeurIPS 2024**), poster 94829, Vancouver; arXiv:2406.00024 (v1 24 May 2024, v2 28 Oct 2024,
arXiv comment "Accepted Neurips 2024"); OpenReview `WSu1PPi2UP`; proceedings PDF
`proceedings.neurips.cc/paper_files/paper/2024/file/1ce12df021f8fb43cf531c556efcebdc-Paper-Conference.pdf`.
Affiliations: Tennenholtz, Hsu, Shani, Boutilier — Google Research; Chow, Liang — Google DeepMind.
✔ **AUTHOR NAME RESOLVED 2026-08-21 (PDF pass).** ~~"the NeurIPS proceedings page gives Yi Liang — use Yi
Liang"~~ — **WRONG, retracted.** The **author block of the PDF itself** reads "Ethan Liang" in **both** the
arXiv v2 PDF *and* the NeurIPS camera-ready proceedings PDF. Only the NeurIPS *metadata/listing page* says
"Yi Liang"; the paper of record does not. **Cite Ethan Liang.** `references.bib` updated to match.

**Provenance of the numbers below.** ✔ **PDF-VERIFIED 2026-08-21 (second pass).** Both PDFs (54 pp. each)
were downloaded and text-extracted with `pypdf`, and Tables 2–4, §5, §6 and Appendices B/C/D/E/G were read
directly. Every figure in §9 is now **[V]**. ~~First pass was arXiv HTML v2 only, flagged [V\*]~~ — that
pass turned out to be accurate on every number; the only corrections were the fifth author (above) and the
rater-question count (§9.2c).

### 9.1 What it actually is — and the one-line surprise

**EAGLE has no decoder.** Verbatim: *"EAGLE does not use a decoder, but rather only requires an encoder
E_D : 𝒳 ↦ 𝒵"* [V]. Nothing is ever mapped from a latent point to text. The latent appears **only inside
the reward**. So the mental model "an RL agent decodes an optimal latent point into language" — the one
our own design doc carried — is **WRONG**. EAGLE is *propose-in-text, score-in-latent*: it is exactly the
**variant (a) propose-and-encode** architecture that the Fable ruling disqualified as Paper C's
claim-bearing renderer, published at NeurIPS 2024 by the ELM group, at scale.

**MDP** [V]:
- **State space** — *"𝒳 ⊆ 𝒮 is the state space comprising of all possible entities (in language)"*: a
  movie's textual description (plot summary + five reasons to like + five to dislike, generated by
  prompting Gemini). Episodes **start at a real catalogue movie**: *"At time t = 0 the process is
  initialized at an existing entity x_i ∈ 𝒟"* (the "anchor"). Horizon **H = 5**.
- **Action space** — *"𝒜 ⊂ 𝒮 is an action space, defined by language prompts for a pre-trained
  environment LLM"*. **K = 100 per state: 50 generic + 50 personalized**, all generated offline by
  prompting Gemini Ultra in five categories (verbatim: *"plot changes, character enhancements, visual and
  storytelling improvements, thematic exploration, and audience appeal adjustments"*); the personalized 50
  are produced by also handing Gemini a
  **textual user profile**. Actions are *text-edit instructions*, not points.
- **Transition** — the **frozen environment LLM (Gemini Ultra)** rewrites the description. The **agent** is
  a separate, trainable **Gemini Nano-2 (3.25B)**, fitted by policy gradient with KL regularisation to a
  reference policy.
- **Reward** — terminal only: *"r_t(x) = 0 for all t < H−1, and r_{H−1}(x) = U(E_D(x), 𝒟)"*, with
  **`U(z_m; 𝒟) = ⟨z_u, z_m⟩ + λ Σ_{m′∈NN(m)} ‖z_m − z_m′‖`**, where `NN(m)` is *"the set of three nearest
  neighbors to m in 𝒟"*. The second term is an explicit **novelty / off-catalogue bonus**.
- **E_D** — a Gemini Nano-2 fine-tuned with **ℓ2 regression** to map movie descriptions to their
  **behavioural embeddings** (ML-25M *"matrix factorization using weighted alternating least squares
  [Hu et al., 2008]"* — the paper never uses the abbreviation "WALS"; rating predicted as ⟨z_u, z_m⟩).
- **Output** — *"A final description is then returned after H steps"*: a **hypothetical new movie** (a
  "content gap"), not a question, not a point.

### 9.2 Answers to the five commissioned questions

**(a) What does the agent optimise; is there an interview?**
It optimises **terminal `U(z)`** — user-affinity `⟨z_u, z_m⟩` plus a distance-from-nearest-neighbours
novelty bonus — over a **fixed, offline, historical** `z_u` from WALS on ML-25M ratings. **There is no
interview, no state that accumulates answers, and no user in the loop.** `z_u` is never updated within an
episode. A string search of the **full extracted PDF text (all 54 pages, main paper + all appendices +
checklist)** finds **zero occurrences of "preference elicitation", "conversational", or "elicitation"** [V —
this re-confirms the first pass's HTML search on the paper of record]. The RL "state" is a *movie description*, not a belief or an
interview transcript. It is offline steering, full stop.

**(b) What does the LLM produce; how is the latent fed to it?**
The frozen environment LLM produces **a rewritten movie description**. **The latent is never fed to any
LLM** — no adapter, no soft prompt, no ELM inside EAGLE. ELM is used **only as a comparison baseline**.
The latent enters exclusively through `E_D(x)` at scoring time. Their stated reason is *"learning an ELM
decoder f requires training an LLM to interpret embeddings, a costly process which may negatively impact the
expressive power of the LLM"*, plus *"the latent space metric d_𝒵 may not be known"* [V — §4, exact wording
restored from the PDF; the first pass dropped the decoder's name `f`].

**(c) Controls / baselines / faithfulness beyond raters?**
Baselines are **ELM** and three **reference-policy** variants over the same action set — **Uniform**
(`q_a = 1/K`), **Optimistic Action**, **G-Optimal Design** — plus **environment transfer** and
**action-space ablations**. **No scrambled, shuffled, or mismatched-latent control anywhere**; they never
pair a generated movie with the wrong `z_u`. **No causal / plug-in faithfulness evaluation.**
★ **The sharp point: EAGLE's only automatic metric IS its own reward.** `U(z)` is computed by `E_D` — the
same encoder that defines the objective the agent was trained to maximise. Reporting `U(z) = 0.74` as the
result is reporting the training objective. The only evidence not routed through the reward is the
**124-rater** panel.
Reported numbers [V — **Table 2 of the PDF, checked cell-by-cell 2026-08-21; every value matched**. The
paper prints "0.6" and "0.8" where the table below writes 0.60/0.80 — same numbers, two significant figures]:
| Arm | U(z) | User Utility | Rater Utility | Distance |
|---|---|---|---|---|
| ELM | 0.57 | 0.31 | 0.37 | 0.52 |
| Uniform (reference policy) | 0.49 ± 0.02 | 0.59 | 0.62 | 0.44 |
| Optimistic Action (ref) | 0.58 ± 0.02 | 0.62 | 0.60 | 0.34 |
| G-Optimal Design (ref) | 0.51 ± 0.02 | 0.51 | 0.67 | 0.45 |
| **EAGLE, Uniform** | 0.65 ± 0.03 | 0.69 | 0.64 | 0.43 |
| **EAGLE, Optimistic** | 0.69 ± 0.04 | 0.71 | 0.69 | 0.25 |
| **EAGLE, G-Optimal** | **0.74 ± 0.03** | 0.76 | 0.74 | 0.47 |
Transfer [V — **Table 3**]: Pro→Ultra 0.75 ± 0.03, Ultra→Ultra 0.80 ± 0.02, Pro→**GPT-4** 0.65 ± 0.02 (user
utility 0.86, rater utility 0.78). Ablations [V — **Table 4**]: default 0.75 ± 0.03 · **without personalized
actions 0.56 ± 0.02** · macro actions 0.71 ± 0.02 · combined 0.74 ± 0.04 (and the combined arm posts the
paper's highest rater scores: user utility **0.88**, rater utility **0.92**). Anchor-quality asymmetry
[V — §6, prose not a table]: on movies originally rated poorly (*"scores 1, 2, 3"*), utility 0.7 ± 0.04
(**~+30%**); on highly-rated anchors (*"scores 4, 5"*) 0.83 ± 0.03 (**~−5%**).
**Human study protocol** [V — §5.3 + **Appendix G**]: **124 raters** (*"a pool of 124 human raters"*; paid
contractors, footnote 8); each first rates **10–20 of 600 popular movies and writes their own textual
profile**; then sees a user profile and **two descriptions in random order — one real movie, one EAGLE
variant at H = 5** — *"raters do not know which is the original or generated variant"*.
⚠ **CORRECTED 2026-08-21 by the PDF: ~~"answers three questions"~~ is WRONG — the rater form has ELEVEN
questions.** Appendix G: *"We use eleven questions to evaluate our human raters. Many of the questions were
used to validate the quality of the responses."* **Three of the eleven are reported**, and the mapping is
explicit: **Q1** (*"Which movie would the user prefer to watch? (A, B)"*) → *User Utility*; **Q5** (*"The two
movies are different from each other"*, 5-point strongly-disagree…strongly-agree) → *Distance Score*; **Q8**
(*"If you had to choose one of the movies, which movie would you personally prefer to watch? (A, B)"*) →
*Rater Utility*. The other eight are attention/consistency checks (Q2–Q4 re-ask user utility; Q9/Q10 ask
whether each movie is real). Scores are **the percentage of raters preferring the generated result** to the
original, in [0, 1] — note Q5 is a Likert item read on that same scale. Each rater evaluates
every algorithm **twice** — once against a MovieLens profile, once against their own. This is a
**preference/plausibility panel**, not a faithfulness measurement: raters never see the latent, so it is
the same trap as ctELM's "fools the experts 44%".

**(d) Does EAGLE ask anyone anything, or fold any answer back?**
**No, and no.** No question is generated, put to anyone, or answered. Nothing is folded into `𝒵`. The only
humans are post-hoc raters scoring finished text.

**(e) Bibliographic details** — §9 header above.

### 9.3 Two things EAGLE says that HELP Paper C

1. **They name our problem and decline it.** *"A latent embedding z\* deemed optimal in latent space, may
   be 'off-manifold', and therefore not mapped to a valid entity x ∈ 𝒳"*; EAGLE *"achieves realizability of
   the embedding by construction"* by never leaving text [V]. **The ELM group looked at decoding an
   off-manifold optimal point and routed around it.** That is the cleanest possible motivation sentence for
   Paper C: the nearest published relative avoided the decode, so someone should measure what happens when
   you do it.
2. **They state the price of propose-and-encode, in their own words** — the argument *for* our variant (b):
   *"EAGLE's coverage is inherently limited by the defined action space"*; *"it may be that no action exists
   that would allow one to move in any direction of the embedding space… This is a particular disadvantage
   of EAGLE, and an advantage of ELM, as ELM is not constrained by movements in embedding space"*; and none
   of their coverage remedies *"would guarantee coverage of the embedding space"* [V]. Our variant-(a)
   baseline arm's expected weakness is therefore **pre-diagnosed by the authors of the architecture**.

### 9.4 ★ VERDICT — does EAGLE occupy a leg of the conjunction?

**No leg is occupied. But the daylight is narrower than the design doc assumes, in three specific places,
and one sentence we were relying on is now false.**

- **Leg (i) — a decoded point selected by an interview policy at an interview state: SAFE, and safer than
  expected.** EAGLE has a policy and a latent objective, but it **decodes nothing**, its `z_u` is fixed
  historical CF, and it has no interview, no question, no answer, no state that conditions on a user
  response. Both halves of leg (i) survive intact.
- **Leg (ii) — scrambled-correspondence control: SAFE.** Absent. EAGLE stays in the "zero of N" tally.
  ⚠ **but narrow the wording**: EAGLE *does* have a uniform-random **action** baseline (0.49–0.51 vs
  0.65–0.74). Never write "no random baseline of any kind" about EAGLE.
- **Leg (iii) — plug-in causal faithfulness: SAFE, and the case is stronger than we had it.** Not only is
  there no causal substitution test — their headline automatic metric is **the reward itself, scored by the
  encoder that trained the agent**. The 124 raters are the only non-circular evidence, and they measure
  preference, not faithfulness.

**Where the daylight SHRINKS — say these out loud before a reviewer does:**
1. ✗ **DEAD SENTENCE: any unqualified "no published system has a policy that optimises in a recommender's
   latent space and whose output is natural language".** EAGLE is exactly that, at NeurIPS 2024, on
   **ML-25M**, from the group whose ELM we cite. The surviving qualifiers are **decode** (EAGLE never maps
   a point to text), **interview** (EAGLE has no user), and **question** (EAGLE emits an item).
2. ✗ **Our variant (a) "propose-and-encode" arm is not our baseline design — it is EAGLE's architecture,
   minus the RL.** Present it as *"an EAGLE-style encoder-only arm"* and cite them, or a reviewer finds it
   first. This also means the (a)-vs-(b) row is a **comparison against published prior art**, which is
   better for us than a comparison against a strawman we built.
3. ✗ **"Rewarding distance from the catalogue to push a point off-grid" is published machinery.** EAGLE's
   `λ Σ_{m′∈NN(m)} ‖z_m − z_m′‖` is structurally C1's off-catalogue target. Different channel (item vs
   question), so the claim holds — but **claim the channel, never the trick**. Combined with GeneRec, "we
   generate off-catalogue objects in recommendation" is now doubly taken.
4. ⚠ **The "same group already did the sequel" risk.** ELM (render a point) and EAGLE (steer toward a
   region) are consecutive papers by the same authors; Boutilier is a plausible reviewer. Paper C must
   answer *"what does this add over EAGLE?"* in one sentence, in the intro, not the related work.

**Sentences that are now SAFE to write** (each verified against the primary text at [V]):
- *"EAGLE steers a frozen LLM toward high-utility regions of a MovieLens-25M behavioural embedding, but it
  never decodes a latent point: it requires only an encoder, and its action space is a fixed set of
  text-edit prompts."*
- *"In EAGLE the user embedding is fixed historical collaborative-filtering output; no question is asked,
  no answer is observed, and nothing is folded back into the embedding space."*
- *"EAGLE's automatic evaluation is its own reward function, computed by the encoder that defines the
  training objective; its only independent evidence is a panel of 124 post-hoc raters who never see the
  latent."*
- *"We are aware of no embedding-verbalisation system — EAGLE and ELM included — that reports what its
  renderer produces from a correspondence-scrambled latent."*
- *"EAGLE's authors identify the off-manifold decoding problem explicitly and avoid it by construction,
  and they name action-space coverage as the price."*

**Sentences that are UNSAFE and must not be written:**
- ✗ "No policy optimises in a recommender's own latent space with a language output." (EAGLE.)
- ✗ "EAGLE reports no baselines / no random control." (Uniform reference policy; three action-set designs;
  transfer; ablations.)
- ✗ "EAGLE verbalises a latent point." (It does not decode at all — this is a factual error we were one
  draft away from making.)
- ✗ "Off-catalogue generation in recommendation is unclaimed." (GeneRec + EAGLE's novelty term.)
- ✗ "Propose-and-encode is our baseline construction." (It is EAGLE's contribution.)

**Status: CITATION + EXPLICIT DIFFERENTIATION, and a BASELINE ANCESTOR for variant (a). Not runnable as a
baseline** (no catalogue-ranking metric, no interview, different objective, Gemini-scale environment).

---

## 10. ★ ELM — FULL READ (2026-08-21)

**Provenance.** The arXiv v2 PDF (`2310.04475v2`, 13 Mar 2024, **34 pp.**) was downloaded and text-extracted
with `pypdf`; §§1–6, Tables 1–7, Figs. 1–8 and **Appendices A–H in full** were read directly. Everything in
§10 is **[V]** unless marked otherwise. This section **supersedes the [V\*] entry at §2/A1** and carries
five corrections back into it in place.

**Bibliographic record, verified.** Guy Tennenholtz, Yinlam Chow, Chih-Wei Hsu, **Jihwan Jeong**, Lior Shani,
**Azamat Tulepbergenov**, **Deepak Ramachandran**, **Martin Mladenov**, Craig Boutilier (Google Research),
*"Demystifying Embedding Spaces using Large Language Models"*, **ICLR 2024** — every page is headed
*"Published as a conference paper at ICLR 2024"*. Correspondence: `guytenn@gmail.com`. **Nine authors; no
Liang.** ⇒ two corrections to A1 (venue was flagged unverified; the author list we had banked was **EAGLE's**).

### 10.1 ★ THE CONSISTENCY METRICS — exact definitions, and who computes them

Both are introduced in §3.2 and detailed in Appendix B. They are the paper's own novelty claim: *"bench-
marking with both human evaluations as well as two novel quantitative metrics; namely, semantic and
behavioral consistency"* [V].

**(a) Semantic consistency (SC).** Verbatim definition [V]:
> *"Semantic consistency (SC) compares the semantic embedding of generated text with the original embedding
> that produced that text. In our movie tasks, this metric is computed by re-embedding ELM's output into the
> same semantic space used to embed the original movies. Formally, we define the semantic consistency of an
> output o generated by an embedding w^s ∈ W by  SC(o, w^s) = d(P(o), w^s),  where P(o) projects the output
> to the domain embedding space W and d is the distance metric in W (or alternatively, a similarity score)."*

- **Who computes P?** Footnote 6, verbatim [V]: *"In our experiments we use the **same DLM** to serve as P,
  i.e., to re-embed output text into semantic space."* The DLM is the pretrained dual-encoder
  (Sentence-T5-like / GTR-like) that **produced ELM's input embeddings in the first place**.
- **d in practice** = **cosine similarity** (Table 2 caption: *"We use cosine similarity as our metric for
  SC"*). ⚠ Note the paper defines SC as a **distance** `d` and reports a **similarity**; the "or
  alternatively, a similarity score" clause is doing the reconciliation. Minor, but do not quote the formula
  as a similarity without the caveat.
- **Appendix B restates it differently and the difference matters** [V]: *"we measure the semantic
  consistency by computing cosine similarity between the semantic embedding of the **ground-truth text** and
  the semantic embedding of generated output. For movie tasks, it is more straightforward as the input
  embedding to ELM **is indeed the semantic embedding of the ground-truth text**"*. ⇒ for movie tasks the two
  statements coincide **only because the input vector is on-manifold by construction** (it is a real movie's
  DLM embedding). **For a synthesised point there is no ground-truth text**, so only the §3.2 form applies —
  which is exactly the round-trip. Worth saying in Paper C: ELM's SC is well-defined off-manifold, but its
  *interpretation as agreement with a reference text* is not.
- **Rationale, verbatim** [V]: *"The utility of SC is grounded in the expectation that if the generated
  output o holds true to its source w^s, then its embedded representation should not deviate substantially
  from its generating embedding vector."* And the sentence that makes it our precedent: *"Importantly, this
  approach is not tied to examples in the data, providing the flexibility to measure the semantic consistency
  of **hypothetical entities**."*

**(b) Behavioural consistency (BC).** Verbatim [V]:
> *"A second form of consistency, which we term behavioral consistency (BC), measures the ability to use
> ELM's output text to make good behavioral predictions. […] BC(o_u, u) = RankingScore(R_u, ρ(o_u)), where
> ρ(o_u) are the ranks of the movies in I for user u provided by the **language-based ranker**, and
> R_u = {r(u,m) : m ∈ I} are the (MovieLens) ground-truth ratings for I provided by u (converted to
> rankings)."*

- **RankingScore** is left open — *"any rank-comparison metric"*, with **NDCG** and **Spearman** named.
  Table 2/4 use **Spearman rank correlation**; Figs. 3 and 5 use **NDCG**. (So "BC" denotes two different
  numbers in the same paper — check which one before quoting.)
- **Who computes ρ?** Footnote 7, verbatim [V]: *"In our experiments we use the **dual-encoder DLM** to serve
  as ρ, by embedding test movies and user profiles and using their similarity for scoring/ranking."*
  **Same DLM again.**
- **Implementation detail worth knowing** [V, App. B]: candidate movies are *"randomly selected"*, and
  *"We impute zero-star if a selected movie has no rating."* A zero-star imputation over a random candidate
  set is a soft ruler; the imputation rate is not reported.
- **BC for movies** is defined symmetrically in Appendix B: `BC(o_m, m; U) = RankingScore(R_m ‖ ρ(o_m))`,
  ranking **users** for a movie via a User Ranker whose query is the generated text.

**(c) A third, undiscussed metric — human evaluation.** §3.2: *"we ask raters to assess ELM's output w.r.t.
its consistency with a movie's plot, its linguistic coherence, and overall task quality"* [V]. Not a
consistency metric in their taxonomy, but it is the only evidence outside the DLM.

### 10.2 ★★★ DOES ELM SELF-GRADE? — YES for SC, MOSTLY for BC, and the extrapolation figure is fully circular

This is the answer the two-ruler discipline needed. Three tiers:

| Metric | Input encoder | Scoring encoder | Ground truth | Independent? |
|---|---|---|---|---|
| **SC (all 24 movie tasks, Fig. 4)** | DLM | **the same DLM** (fn. 6) | the source vector itself | **NO — closed loop.** The only thing outside the loop is the LLM's own generation |
| **BC, Table 2/4 + Fig. 3 (user profile)** | WALS behavioural MF | DLM ranker (fn. 7) | **real MovieLens ratings** | **PARTLY.** Input encoder ≠ scoring encoder, and the target is external held-out human data. This one cell is a genuine second ruler |
| **BC, Fig. 5 (CAV extrapolation of users)** | WALS behavioural MF | DLM ranker | ⚠ *"NDCG with the **predicted CF model ratings**"* (Fig. 5 caption) [V] | **NO.** The target is the **CF model's own predictions** — i.e. the same factorisation that produced the input embedding. The one experiment closest to Paper C's use case is graded by the encoder under test |

**⇒ Ruling for Paper C.** Say: *"ELM's semantic consistency re-embeds the generated text with the very
encoder that produced the input vector; its behavioural consistency for the extrapolation experiment scores
against the collaborative-filtering model's own predicted ratings. Only the user-profile behavioural
consistency against real MovieLens ratings crosses rulers, and it is a single cell."* That is precise,
defensible, and not a hit-piece — it is what the footnotes say.

**★ The authors themselves treat SC/BC as an OPTIMISABLE OBJECTIVE.** Appendix H ("Reinforcement Learning
from AI Feedback") specifies a full CoMDP whose terminal reward **is** the consistency metric [V]:
> *"r(o_n; o_{0:n−1}; w_v, w_u) = BC(o; w_u) (or SC(o; w_v)) if o_n = [EOS]; 0 otherwise"*,
> and *"one may fine-tune an ELM with a reward so that it would better align with consistency."*

Policy gradient / REINFORCE with KL regularisation to the pretrained model, plus a soft-actor-critic variant
(Eqs. 2–8). **Appendix H is a PROPOSAL — no RLAIF experiment is reported anywhere in the paper.** But it is a
gift for our argument: **the ELM authors write down, in their own appendix, the recipe for maximising the
faithfulness metric directly.** Combined with ZSInvert's cosine-90/F1-50 result, this is our second, and
now *first-party*, demonstration that round-trip consistency is a training target rather than a verdict.
Quotable in Paper C's method section.

**Explicit limitation the paper states, which we must not paper over** [V, §4]:
> *"That said, to ground our results, **we do not consider arbitrary hypothetical points in embedding space**.
> Instead, we evaluate our model using interpolation of existing entities, as well shifts of such entities in
> specific directions."*

⇒ **ELM never decodes an arbitrary point.** Every synthesised point they evaluate is an interpolation of two
real entities or a CAV shift from one. Paper C's emitted point **is** arbitrary (a policy's argmax in a
learned space), so we are outside ELM's evaluated regime by their own statement. **This is the single most
useful sentence in the paper for our motivation** — better than anything in EAGLE §9.3.

### 10.3 ★ ADAPTER SPEC — ELM vs our C3B build

| | **ELM (ICLR 2024)** [V] | **our C3B plan** |
|---|---|---|
| Adapter | **two-layer MLP**, `E_A : W ↦ Z` | 2-layer MLP ✔ **matches** |
| Soft-token slot | **ℓ = 1 — ONE token** | **8 soft tokens** |
| Base LM | **PaLM 2-XS ("Otter")**, pretrained | **distilgpt2 (82M)** |
| LM frozen? | **stage 1 only.** Stage 2 fine-tunes **all** of (E_0, M_0, E_A) | **PERMANENTLY FROZEN** (`src/instrument/c3b_adapter.py:365`, `lm.requires_grad_(False)` — "Only the adapter trains") ⇒ we run **ELM's stage 1 only** |
| Schedule | 20k (adapter-only) → 300k (full), **batch 32**, ≈7 epochs | — |
| Training pairs | 24 movie tasks + 1 user-profile task, targets **generated by PaLM 2-L** from the movie title; ELM sees **only the embedding** | — |
| Splits | movies: **1000 random test**, rest train. Users: **80/20** | — |
| Domain embeddings | **behavioural** = MF via **WALS** (Hu et al. 2008), movies with **≥5 ratings**; **semantic** = DLM over concatenated plot descriptions + reviews, output vectors averaged | our CF latents / SBERT |
| Dataset | ML-25M (25M ratings, 62,423 movies, 162,541 users); + Amazon "Clothing, Shoes and Jewelry" (5.7M reviews, 1.5M products) | ML-25M ✔ |

**★ Three findings that calibrate C3B directly:**

1. **Our 8-token slot is ELM's stated future work, not a deviation.** Footnote 4, verbatim [V]:
   *"More expressive adapters that map vectors in W to length ℓ ≥ 1 sequences are left for future work."*
   ⇒ ELM injects the domain vector as a **single** token embedding. **8 soft tokens is a strictly broader
   slot than the precedent**, and we can cite footnote 4 as the authors' own flag that ℓ > 1 was untried.
   (⚠ Do **not** write "ELM uses soft tokens (plural)". It is one.)
2. **★ TWO-STAGE TRAINING IS LOAD-BEARING, and Appendix D quantifies it on a toy task.** Verbatim [V]:
   *"we found that training ELM on a pretrained model without two-stage training could not fit to the
   training data. Particularly, fine-tuning the full model for over **100k iterations** always converged to
   outputting either one or two. When two stage training was applied, i.e., first fitting E_A and only then
   the rest of the model, we found the model to converge in **less than 1000 iterations**."* The toy task is
   a 2-D embedding → the tokens "one"/"two", **no textual prompt at all**. The stated mechanism [V]:
   *"as the pretrained embedding layer E_0 readily maps to language space, the newly initialized embedding
   layer E_A may require numerous updates to map to a similar space"* — i.e. a randomly initialised adapter
   pulls the frozen token geometry apart if you unfreeze everything at once. **A 100× convergence difference
   on a 2-D toy problem is a cheap, replicable smoke test we should run on C3B before the real fit.**
3. **⚠ RISK FOR C3B, CONFIRMED AGAINST OUR OWN CODE: ELM's reported quality comes from stage 2 — a FULL
   fine-tune of the LLM.** `src/instrument/c3b_adapter.py:365` sets `lm.requires_grad_(False)` with the
   comment *"FROZEN. Only the adapter trains."* ⇒ **C3B as built is ELM's stage 1 and nothing else**, the
   configuration ELM used purely as a warm-up. Appendix D does not show adapter-only *failing* (it shows
   **full-only** failing), so this is a risk rather than a banked negative — but the consequence is fixed:
   **we may not compare C3B's absolute generation quality to ELM's numbers**, and if C3B underperforms, "we
   never ran ELM's second stage" is the first hypothesis, not the last. Either budget an unfreeze (LoRA on
   distilgpt2 is cheap) or state the limitation explicitly in the paper.

### 10.4 HUMAN-RATER PROTOCOL — what the 100 raters actually judged

- **100 raters per task** (§4 and Appendix C), *"paid contractors […] their standard contracted wage, which
  is above the living wage in their country of employment"* (fn. 8) [V].
- **Per task: 40 randomly generated model utterances**, on **movies not used in training**, each saved with
  its context and questions on a **Google form** (Fig. 8) [V].
- **The rating scale is 1–5 agreement**, normalised to [0,1] for the tables: *"On scale of 1-5, how much do
  you agree with the following: 1 = Strongly Disagree, 5 = Strongly Agree"* [V]; Table 2 caption: *"we show
  normalized scores between 0 (strongly disagree) and 1 (strongly agree)"*. ⇒ **corrects our banked "scale
  0–1"** (that is the reporting scale).
- **What they compare against:** *"Usually we start from ground-truth data of a movie and then model
  utterance generated for the task"* [V] — i.e. raters see the **real plot / real item description** as the
  reference, and judge the generated text against it. **Raters never see the embedding.** Three reported
  dimensions per movie task (consistency-with-plot / language comprehensiveness / task quality) plus
  task-specific items — e.g. `convince1` has **five** questions, `negative review` has three
  ("consistent with the movie plot", "coherent", "**is negative**").
- **Numbers (Table 2, 24 movie tasks), verified cell-by-cell:** SC (cosine) **0.77–0.96** (min `common with
  nn` 0.77, max `pitch` 0.96); BC (Spearman) **0.46–0.96** (min `common with nn` 0.46, max `convince1` 0.96);
  human **0.37–1.0** (min: `similarities` scores **0.37 / 0.40 / 0.38** across all three dimensions — the
  paper's worst cell by a wide margin; max: `neutral review` **1.0 / 1.0**, though its task-quality is
  **0.56**). ⇒ **all four ranges we had banked are CONFIRMED.**
  The three nearest-neighbour two-movie tasks are the weak spot on BC (**0.48 / 0.48 / 0.46**), and the paper
  says so: *"performance tends to be worse on two-movie tasks, due to the additional challenge of
  'conversing' about similarities and differences between two distinct embedding representations."*
- **Amazon (Table 4, 6 tasks):** SC **0.89–0.96**, BC **0.81–0.94**, human **0.7–0.94**. ⇒ **confirmed.**
- **★ An Amazon-only extra we had not banked — a ground-truth rater ceiling.** Tables 5 and 6 give
  question-by-question rater scores for **generated** vs **real** data on the same forms. Finding, verbatim
  [V]: *"our generated results correctly correspond to test items, and perhaps surprisingly, **often
  surpasses the original ground-truth data quality**, as viewed by the human raters."* Example: item
  `description` Q1 ("Both descriptions are equivalent") **0.77 generated vs 0.68 ground truth**; positive
  review Q1 **0.91 vs 0.85**.
  ⇒ **This is a rater-ceiling control, and it is a better design than ctELM's "fools the experts 44%"** —
  but it measures the same thing (plausibility), and **beating the ground truth is precisely what a fluent
  confabulator does**. Use it as evidence for our §5.4 rule, not against it.

### 10.5 INTERPOLATION AND EXTRAPOLATION — what was decoded, and how it was validated

**(a) Movie interpolation.** Linear, **α = 0.5**, of a training-set movie's semantic embedding with
**"Forrest Gump (1994)"**; the interpolated vector is fed to ELM. Validated by **SC only**. Compared against
**PaLM 2-L and GPT-4** given a **one-shot ground-truth example** and asked to write the same task "in the
same embedding space" (Appendix F gives the prompt, ending *"Do not use the movies' names in your
response."*). Result (Fig. 3, a plot — no table): *"text-only LLMs are significantly less consistent than
ELM"* [V]. **⚠ No numeric values are printed for Fig. 3; do not quote a number off that figure.**

**(b) Interpolation sweep over α** (Fig. 4): *"outputs are highly consistent across all degrees of
interpolation values, with a tendency for lower consistency scores for embeddings that are farther away from
the two movies. This result is unsurprising, as we expect performance to degrade as it moves away from
'real' data points. Also note these are linear interpolations, though the true embedding manifold may be
highly non-linear."* [V] ⇒ **the on/off-manifold degradation we care about is present in ELM, acknowledged,
and mild** (an SC curve, magenta, over two tasks).

**(c) User interpolation.** Random pairs of user behavioural embeddings; validated by **BC (NDCG)**;
text-only baselines get the two real user profiles and a zero-shot "generate an interpolation of the
following two user profiles" prompt.

**(d) CAV extrapolation.** CAVs trained on **MovieLens-25M attribute tags** using the procedure of **Göpfert
et al. (WWW 2022)** and the dataset of **Balog et al. (SIGIR 2021)**, *"linear, objective rather than
non-linear and/or subjective CAVs"* [V]. Treated as *"noisy, 'soft' movie attributes, such as 'funny,'
'terrifying,' 'thought-provoking'"*. Movies and users are shifted along a CAV direction by α.
- **Movies (Fig. 4):** *"semantic consistency of movie extrapolations generally decreased as we extrapolated
  further in CAV directions. This is expected as extrapolations outside the data manifold should induce
  higher error."* [V]
- **★ Users (Fig. 5) — the anomaly, and it is the one to remember:** *"Perhaps surprisingly, user profiles
  show **greater** behavior consistency as we increase the degree to which they move in a CAV direction. We
  hypothesize this is due to user profiles becoming 'more predictable' in our CF model by making user
  preferences more extreme."* [V]
  ⇒ **ELM's behavioural faithfulness metric goes UP the further off-manifold the point is pushed.** Their own
  explanation is a **metric artefact**, not a capability. Read with §10.2: Fig. 5's target is the CF model's
  *predicted* ratings, so an extremised user is trivially easier for the CF model to rank. **This is a
  published, first-party example of a faithfulness metric moving the wrong way off-manifold** — cite it in
  Paper C's validation section as the reason we normalise (plug-in gap closure) rather than report a raw
  consistency score.
- **Fig. 5 carries the two baselines we said ELM lacked** [V]: *"Dotted lines show two baselines (for α = 0):
  the **random baseline**, which rates movies randomly, and the **ground-truth baseline**, which uses the
  ground-truth user profile data to compute BC (i.e., BC of the ground-truth training data)."*
  ⇒ a random **ranking** floor and a real-data **ceiling** on the same axis. Good practice, and it is the
  design our own plug-in numbers should copy (floor = no latent, ceiling = the latent itself). It is **not**
  an input-side null.

**(e) Qualitative, Appendix G — an observation of ours, not a claim of theirs.** The sample decode of
"I, Robot (2004)" × "Forrest Gump (1994)" describes *"a mentally challenged man named **Charlie
Baileygates**"* who befriends *"a robot named Harry"* [V]. Charlie Baileygates is the protagonist of *Me,
Myself & Irene* — a third film, entangled in neither input. There is no ground truth for a hypothetical
point, so this is not an error by the paper's own lights; but it is a concrete, quotable instance of
**confident, fluent, specific confabulation at a synthesised point, published by ELM itself**, and SC would
score it on plot-vocabulary overlap alone. Useful as a motivating figure. ⚠ Present as *our reading of their
Appendix G sample*, never as a reported failure.

### 10.6 Corrections landed, and sentences now safe / unsafe

**Corrections to what was banked (all applied in place above, dated):**
1. **Venue** — ELM **is** ICLR 2024. The "do NOT write ICLR 2024" warning is retracted.
2. **Author list** — we had **EAGLE's** authors on ELM. Nine authors, no Liang; Jeong, Tulepbergenov,
   Ramachandran, Mladenov are ELM-only.
3. **"No random baseline of any kind"** (§0.5 and §2/A1) — **FALSE**. Fig. 5 has a random-ranking floor and a
   ground-truth ceiling. Narrow to **"no random or scrambled EMBEDDING baseline"** — i.e. no **input-side**
   null. Everything the empty-cell tally (§4.1) claims still stands under the narrowed wording.
4. **"frozen LLM"** (§1/Q4) — wrong for ELM. Frozen in stage 1 only; stage 2 is a full fine-tune of
   (E_0, M_0, E_A), and two-stage is reported *essential for convergence*.
5. **Human rating scale** — 1–5 Likert normalised to [0,1], not a native 0–1 scale.
6. **"24 movie tasks"** — correct, but the intro says *"25 training tasks"* [V]: 24 movie tasks + the
   user-profile task, and **two separate models** are trained (one joint model for the 24, one for profiles).
   Do not write "a single ELM does all 25".
7. **Everything else we had banked survived**: two-layer MLP adapter ✔, 20k/300k/batch-32/≈7-epochs ✔,
   PaLM 2-L + GPT-4 as the only text-only baselines ✔, α = 0.5 movie interpolations ✔, random user-pair
   interpolations ✔, CAV extrapolations ✔, **all four reported metric ranges ✔ (cell-by-cell)**.

**Newly SAFE to write** (each verified against the PDF at [V]):
- *"ELM's semantic consistency re-embeds the generated text with the same dual-encoder that produced the
  input vector; the metric is therefore closed over the system's own components."*
- *"ELM's authors specify, in an appendix, a reinforcement-learning procedure whose reward is the semantic
  or behavioural consistency score itself — the metric is an optimisation target in the paper that
  introduces it."* (Appendix H; note it is proposed, not run.)
- *"ELM explicitly declines to evaluate arbitrary points: it reports only interpolations of existing entities
  and CAV shifts from them."*
- *"In ELM's own extrapolation experiment, behavioural consistency **increases** as the user embedding is
  pushed further along a concept direction; the authors attribute this to the profile becoming more
  predictable for the collaborative-filtering model that scores it."*
- *"ELM injects a domain vector as a single token embedding; adapters mapping to longer sequences are named
  as future work."*

**UNSAFE — do not write:**
- ✗ *"ELM reports no baseline of any kind."* (PaLM 2-L / GPT-4 text-only; Fig. 5 random + ground-truth;
  Amazon Tables 5–6 ground-truth rater ceiling.)
- ✗ *"ELM decodes arbitrary points in embedding space."* (§4 explicitly says it does not.)
- ✗ *"ELM's adapter uses soft tokens."* (One token; fn. 4.)
- ✗ *"ELM trains an adapter against a frozen LLM."* (Stage 2 unfreezes everything.)
- ✗ *"ELM's behavioural consistency is an independent ruler."* (Ranker = the same DLM; and Fig. 5's target is
  the CF model's own predictions. **Only** the Table-2/Fig-3 user-profile BC against real MovieLens ratings
  crosses rulers.)
- ✗ Any number read off Fig. 3, Fig. 4 or Fig. 5 — they are plots with no printed values.

**Residual unread in ELM: none.** All 34 pages and Appendices A–H were extracted and read. Remaining
uncertainty is confined to figure values that the PDF does not print.
