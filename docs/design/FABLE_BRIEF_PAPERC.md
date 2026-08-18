# Fable brief — Paper C: continuous emission + LLM verbalisation

**Self-contained. Do not read the repo.** Author's framing, 2026-08-18: *"continuous is what we can
marry with LLM for a sexy star-of-the-show paper, maybe more fun, less beating benchmarks … make it
somewhat safe, like PEBOL, so fun idea, cool techniques, no clear baselines to beat, more capability
paper. should ideally be truly original. i am counting on A and B to be thorough and serious ones,
and this is just basically making the whole thesis more current and fun and sexy."*
**Publication is a bonus, not a requirement. No deadline.** Papers A (instrument → TORS) and B
(elicitation policy → ECIR) carry the serious load.

## The system, in one paragraph

CASPER is a cold-start conversational recommender on ML-25M (18,359 items, canonical Liang split,
140,768 train / 10,000 val / 10,000 test). Paper B's policy holds a state (a folded belief `z`) and
emits a continuous query vector `ẑ`, which is snapped to `argmax_q ⟨ẑ, e_q⟩` over a 19,390-question
bank (items + genome concepts). Paper C asks whether `ẑ` can name a question the bank does **not**
contain, and whether an LLM can say it in English.

## The claim as currently drafted

**We can find a point in recommender space worth asking about that no catalogue question occupies,
say it in English, and show that what we said is not noise.** Explicitly NOT: that it improves NDCG.

## Two hard constraints that shaped it

1. **No non-circular SIMULATED answerer exists for a continuous direction.** A geometric answer
   `a = sign(u*·q)` has the model both asking and answering (project ruling, Jul-25). Projecting the
   user's real liked items onto `q` does not escape it — still tower item embeddings, plus added
   error. Non-circular answers exist only where a real person responds.
2. **The author's circularity ruling (2026-08-18):** a simulated answerer is acceptable as a **STATE
   GENERATOR** (it puts the policy at a plausible state so it can emit) but **fatal as a VALIDATOR**
   (simulating an answer to the emitted point and showing the belief improves = the model grading
   itself). ⇒ validation must be a **property of the point**, never an outcome of answering it.

## What two deep-research passes established (2026-08-18)

- **The novelty narrows to LANGUAGE.** Synthesised continuous queries answered by real humans are
  routine where the design can be *physically executed* — CoSpar/LineCoSpar propose a point in
  continuous gait-parameter space with no pool and an exoskeleton *walks it* for a real subject;
  Adaptive Design Optimization and preferential BO likewise. **Nobody renders a synthesised point as
  language and folds the answer back into the space it came from.**
- **Answerability has a 34-year-old named citation:** Baum & Lang 1992, *"Query learning can work
  poorly when a human oracle is used"* — synthesised digits were fusions no human could label; Zhu &
  Bento reproduced it with GANs (2017). The two standard fixes are our two options (snap to a real
  neighbour, or verbalise). This puts the burden of proof on us for removing the snap.
- **The scrambled-latent control is absent everywhere in our cell.** Twelve embedding-verbalisation /
  inversion systems checked (ELM, ctELM, EAGLE, DeCap, Vec2Text, ZSInvert, vec2vec, LatentQA,
  Patchscopes…): **none reports what its renderer produces from a scrambled latent.** ELM reports 100
  human raters and no random baseline of any kind.
- **★ The naive control FAILS.** Heap et al. (arXiv:2501.17727): automated interpretability metrics do
  not separate trained from randomly-initialised transformers — Pythia-6.9b fuzzing AUROC, trained
  **0.79** vs randomised **0.87**. The ONLY null that separated was a **scrambled-EMBEDDING** control
  (each token's embedding resampled per occurrence, destroying the vector→meaning correspondence):
  **AUC 0.50, chance.** ⇒ scramble the CORRESPONDENCE, not the distribution; a Gaussian null fails.
- **The faithfulness metric to borrow — plug-in evaluation** (Hewitt, Tafjord, Geirhos, Kim,
  arXiv:2510.08506): a verbalisation is faithful to the degree that substituting the text back for the
  latent **reproduces the latent's causal effect**, scored as % of concept gap closed; their range
  −6% to 86%, so it discriminates. Their **"machine-only synonyms"** (verbalisations humans judge
  wrong that nonetheless reproduce the behaviour) is the argument against grading by human plausibility.
- **Round-trip cosine is dead as a metric:** ZSInvert reaches **cosine >90 with F1 only >50**.
- **The snap has a real citation with numbers — DeCap (ICLR 2023):** raw off-manifold vector CIDEr
  **8.1** · hard nearest-neighbour snap **27.1** · softmax-weighted blend **42.1**.
- **Nearest neighbours, checked in full text:**
  - **ELM** (Tennenholtz et al., Google, arXiv:2310.04475) — 2-layer MLP adapter maps frozen CF
    embeddings into an LLM's token-embedding space; decodes *synthesised* points (interpolated movies,
    interpolated users, CAV extrapolations); introduces consistency metrics because those points have
    no ground truth. **No loop, no user, nothing selects which vector to decode. It owns
    "an LLM verbalises a synthesised latent point."**
  - **EAGLE** (same group, NeurIPS 2024, arXiv:2406.00024) — we feared it took our cell; it does not.
    Its action space is verbatim *"defined by language prompts for a pre-trained environment LLM"*
    (50 generic + 50 personalised prompts per movie), its output is **movie descriptions** (content
    gaps), there is **no interactive user loop** (124 post-hoc raters), and it has **no faithfulness
    metric and no random/scrambled control**. It is explicitly an *alternative* to ELM that needs only
    an **encoder**, avoiding adapter training.
  - **Info-Synth** (arXiv:2605.26072) — continuous MI-maximising query synthesis with a
    geometry-dependent response model, snapping only when the domain forces it. Our design minus a
    catalogue, minus a renderer, minus any answerability or circularity discussion. **The foil.**

## The staged design

- **C0 — geometry check (free).** Emit `ẑ` for validation states; report cosine to nearest bank
  question and margin `(s₁−s₂)/(s₁−s₁₀)`. Prediction to falsify: trained on one-hot targets, `ẑ` sits
  *on* the nearest question and carries no off-catalogue content.
- **C1 — teach in-between emission.** DBU / "Breaking the Grid" (arXiv:2602.08616): regress `ẑ` onto a
  gain-weighted blend of the top-m **measured** candidates. Off-catalogue by construction; every
  gradient still comes from a question whose value we measured.
- **C2 — is the point MEANINGFUL? (answer-free, the heart).** Three properties: off-catalogue
  distance/margin; **discriminative over real users** (do actual fold-in profiles project differently
  onto it? — computed from real ratings, a fact about the direction); **IM1/IM2 plausibility**
  (Van Looveren & Klaise, user-free autoencoder score). All against a **scrambled-correspondence**
  control.
- **C3 — verbalise.** See the three variants below.
- **C4 — plug-in evaluation.** Text → frozen SBERT→CF adapter → % of causal gap closed, against
  scrambled / prior-only / nearest-bank controls.

## The rendering variants we want you to rule between

**(a) API + nearest verbalisable neighbours ("propose-and-encode").** Prompt a hosted LLM with the
top-m nearest bank phrases and their blend weights, ask for N candidate questions, **encode each
through the frozen text→CF adapter**, keep the one closest to `ẑ`, report the residual distance.
*Pros:* no weights, no GPU, dollars, and the residual is itself a result ("how close can language get
to this point?"). Follows EAGLE's encoder-only structure. *Cons:* the LLM never sees the vector; the
prompt anchors proposals to bank phrases, working against the in-between claim.

**(b) Embedding straight to the LLM (ELM-style trained adapter).** 2-layer MLP maps `ẑ` into the
token-embedding space of an open-weights model; freeze the LLM, train only the adapter on
(embedding, text) pairs from real entities; then decode synthesised points. *Pros:* principled, the
vector genuinely conditions generation, and it is what ELM did. *Cons:* impossible through any hosted
API (no service exposes token-embedding injection or gradients); needs open weights. Feasible at
Flan-T5-base scale (220M) on our CPU box in ~1–3 h, or minutes on free Colab.

**(c) Hybrid** — adapter-conditioned generation, then encoder-based re-ranking of its samples.

## Our own concerns, stated before you rule

1. **★ A circularity we walked into and think we have fixed.** In variant (a), selecting the candidate
   that maximises closeness under adapter `W` and then validating with plug-in evaluation through the
   **same** `W` optimises the metric directly. Proposed fix: fit **two adapters on disjoint halves** of
   the phrase↔embedding pairs — select with `W_A`, validate with `W_B`. Is that sufficient, or is the
   shared training distribution still fatal?
2. **Is the claim thin?** "The point is off-catalogue, discriminative, plausible, and a scrambled
   control is none of those" — is that a paper, or a section of Paper B?
3. **Prompt anchoring.** In (a), proposals conditioned on nearest bank phrases may simply recover bank
   questions. Does that void the in-between claim, or is the encoded residual an adequate defence?
4. **Hardware dependency.** Everything in this thesis reproduces on one ordinary CPU box, which is
   part of Paper A's story. Variant (b) adds Colab or a rented GPU. Worth it?
5. **The 300 quarantined study users** are the project's only non-circular human answer source, and
   the human round-trip is already flagged as blocking the strongest claims across A–E. **Should C be
   where they are spent**, or should they be saved?

## What we are asking you for

1. **Rule between (a), (b) and (c)** — with the reason, and what each buys or costs the claim.
2. **Is the two-adapter split enough** to break the select/validate circularity? If not, what is?
3. **Is the C0–C2 core a result on its own** — publishable as a capability paper, a thesis chapter
   only, or neither?
4. **What does a hostile reviewer say first**, and what is the cheapest thing that defuses it?
5. **The 300 study users — spend here or save?**
6. **Anything in the framing above you think is wrong or over-read.**

## Constraints that bind

No LLM API calls without explicit author approval · never the Reddit API · 173/300 study users
quarantined · every pre-Jul-22 magnitude is demoted and re-establishes on the certified stack before
citation · a capability claim needs a control that a scrambled input fails.
