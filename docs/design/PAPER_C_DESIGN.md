# Paper C — the design (2026-08-18)

Supersedes the "two fixes" sketch in `PAPER_C_EMISSION_DESIGN.md`, which stands as the blocker
analysis. Brief and settled facts: `PAPER_C_TODO.md`. Evidence base:
`external_literature/findings/{continuous_elicitation,llm_verbalisation}_2026-08.md`.

## The claim, in one sentence

**We can find a point in recommender space worth asking about that no catalogue question occupies,
say it in English, and show that what we said is not noise.**

Not: that it improves NDCG. The circularity ruling (author, 2026-08-18) buys the freedom to use a
simulated answerer as a STATE GENERATOR by giving up any performance claim that would route through
it. What is left is a capability plus a validation method — which is the paper.

## Why this is defensible against the literature

- **The narrowed novelty.** Synthesised continuous queries answered by real humans are routine where
  the design can be *physically executed* (CoSpar/LineCoSpar walk a synthesised gait; ADO and
  preferential BO run on real participants). **Nobody renders a synthesised point AS LANGUAGE and
  folds the answer back into the space it came from.** Recommendation has no physical renderer —
  which is exactly why the field snapped to a catalogue.
- **The answerability lineage.** Baum & Lang 1992, *"Query learning can work poorly when a human
  oracle is used"* — synthesised digits came out as fusions no human could label; Zhu & Bento
  reproduced it with GANs (2017). The two standard fixes are our two options: snap to a real
  neighbour, or verbalise. This puts the burden of proof on us for removing the snap, and gives the
  chapter a 34-year lineage instead of a novelty assertion.
- **The gap in validation.** Of eleven embedding-verbalisation/inversion systems checked (ELM,
  ctELM, EAGLE, DeCap, Vec2Text, ZSInvert, vec2vec, LatentQA, Patchscopes…), **none reports what its
  renderer produces from a scrambled latent.** ELM reports 100 human raters and no random baseline
  of any kind. Claimable wording: *"we are aware of no embedding-verbalisation system that reports
  what its renderer produces from a scrambled latent."*

## The stages

### C0 — the geometry check (FREE, decides whether the arc exists)
Emit `ẑ` for every validation state E1 was evaluated on; report cosine to the nearest bank question,
and the margin `(s₁−s₂)/(s₁−s₁₀)`. **Prediction to falsify:** trained on one-hot targets, `ẑ` sits
essentially *on* `e₍₁₎` and carries no off-catalogue content. Minutes on the existing checkpoint.
If it sits between candidates, C1 has a head start.

### C1 — teach in-between emission (cheap)
Replace the one-hot target with the **DBU / "Breaking the Grid" blend**: regress `ẑ` onto a
gain-weighted blend of the top-m *measured* candidates at each state. The target is off-catalogue by
construction and every gradient still comes from a question whose value we actually measured — no
simulation, no clairvoyance. Per-state measured gains already exist (`policy_labels.py`).
**Report:** how far off-catalogue the emissions move, and whether they are stable across seeds.

### C2 — is the point MEANINGFUL? (answer-free, the heart of the method)
Three properties of the point, none of which requires anyone to answer it:
1. **Off-catalogue** — distance and margin, as C0.
2. **Discriminative over real users** — do actual fold-in profiles project differently onto it?
   Computed from real ratings, once. A fact about the direction, not an answer on anyone's behalf.
3. **On-manifold plausibility** — **IM1/IM2** (Van Looveren & Klaise, counterfactual XAI): a
   user-free, answer-free autoencoder plausibility score. **Import it; do not invent one.**

★ **The control, and its exact form matters.** Heap et al. (arXiv:2501.17727) found automated
interpretability metrics *fail* to separate trained from randomly-initialised transformers (0.79 vs
**0.87** AUROC — the random model scored higher). The only null that separated was a
**scrambled-EMBEDDING** control at chance (0.50). ⇒ **scramble the CORRESPONDENCE, not the
distribution** — a Gaussian null demonstrably fails — and the separation is scale-dependent, so run
it at our operating scale.

### C3 — verbalise (BLOCKED on author approval for LLM calls)
ELM-style: a small adapter maps `ẑ` into an LLM's token-embedding space. Alternative to compare:
retrieval-augmented prompting from the nearest bank phrases. The literature says the adapter is
standard; the comparison is worth one row because the prompt version needs no training.

### C4 — plug-in evaluation (the methodological contribution)
**Borrowed from Hewitt, Tafjord, Geirhos & Kim (arXiv:2510.08506):** a verbalisation is faithful to
the degree that **substituting the text back for the latent reproduces the latent's causal effect**,
scored as percent of the concept gap closed, `(x − base)/(latent − base)`. Their range is −6% to 86%,
so it discriminates.

**Our instantiation:** emitted point → text → push that text through the **frozen SBERT→CF adapter**
(Paper A's open-vocab path). That adapter was fitted independently of the emitter, so this is **not a
self-round-trip**. Report gap closure against three controls: **scrambled correspondence**,
**prior-only** (the text without the latent), and **nearest bank question**.

⚠ Their **"machine-only synonyms"** finding — verbalisations that look wrong to humans yet reliably
reproduce the behaviour — is the argument against grading our rendered questions by human
plausibility. Say so explicitly; it pre-empts the obvious reviewer request.

## Baselines: what to replicate, and what not to

**Replicate (cheap, and each earns a specific sentence):**
1. **DeCap's decoding ladder in OUR space** (ICLR 2023, arXiv:2303.03032): raw off-manifold vector
   CIDEr **8.1** · hard nearest-neighbour snap **27.1** · softmax-weighted blend **42.1**. If the
   same ordering holds for us, it justifies the soft snap with a number instead of the assertion
   "no decoder exists" — which 2025 zero-shot inverters have made risky.
2. **Info-Synth** (arXiv:2605.26072) — the nearest neighbour: continuous MI-maximising query
   synthesis with a geometry-dependent response model, no catalogue, no renderer, and **no
   answerability or circularity discussion**. Running it on our space is the "emit freely with no
   answerability constraint" arm, and it is the natural foil for the whole chapter.
3. **The bank snap itself** — our own E1 argmax, which is the "stay on the catalogue" arm.

**Do NOT replicate:** PEBOL (100-item catalogue; already handled in B's positioning), CLARINET
(fine-tunes to emit a *bank* question, so it cannot serve the in-between claim), GATE, BED-LLM.
Citation-only.

**BLOCKING READ before any positioning sentence: EAGLE** (Tennenholtz, Chow, Hsu, Shani, Liang,
Boutilier, **NeurIPS 2024**, arXiv:2406.00024) — an RL agent steering a frozen LLM toward optimal
regions of a **MovieLens-25M** latent. Nearest published relative of the policy→point→text loop, same
group as ELM, and it was absent from our INDEX until 2026-08-18.

## Order of work

1. **C0** — free, and it decides everything downstream.
2. **Read EAGLE** — before writing any claim about what is new.
3. **C1 + C2** — cheap, answer-free, and they constitute a result on their own even if C3/C4 never
   run: *"we can emit off-catalogue points that are discriminative and plausible, and a scrambled
   control is neither."*
4. **C3 + C4** — the LLM half. Needs explicit author approval for API calls.

## What must never be claimed

- That the emitted question would HELP (no non-circular simulated answerer exists for a continuous
  direction; the only honest sources are the 300 quarantined study users or real text).
- Novelty for "an LLM verbalises a synthesised latent point" — **ELM took it**, including
  interpolated and extrapolated points.
- That off-catalogue *emission* is unclaimed — **GeneRec** (arXiv:2304.03516) has an explicit AI
  creator for items outside the corpus. Narrow ours to the **question channel**.
- Anything resting on round-trip cosine alone — ZSInvert reaches **cosine >90 with F1 only >50**.

## Verification debts (do not cite until cleared)

GenMQD · Vendrov's actual query object · LineCoSpar's 6 params / 6 subjects / 30 trials · ADO's
precision claims · LOL-BO's "20×" · Baum & Lang's exact bibliographic details · Tang et al. Nature
Neuroscience 2023 (paywalled; try bioRxiv 2022.09.29.509744).
