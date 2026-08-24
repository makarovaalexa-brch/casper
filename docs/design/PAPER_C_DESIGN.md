# Paper C — the design (2026-08-18)

## ★★ AMENDMENTS OF 2026-08-24 (Fable, after the deep-read + hostile-review + repair pass) — BINDING over the corresponding sections below

1. **The "worth is inherited / bounded by a convex combination" sentence is RETRACTED** (both
   review lenses: it is a bound on vectors asserted as a bound on value, with no affineness
   support; and the ingredient gains are winner's-curse maxima — bootstrap top-5 overlap 1.9/5).
   Replacement, the only claimable form: *"every gradient comes from a candidate whose value we
   measured; we make no claim about the value of the blend point itself, and the ingredient set at
   a given state is an unstable draw from a near-flat measured profile (its direction is only
   moderately stable)."* C1-proper must build targets on the retrained tree's larger cohorts and
   report selection stability before the word "measured" is used of a target.
2. **C2's IM1/IM2 import is DEAD as specified** (full read + the Hvilshøj/Schut critique line: the
   metrics are class-relative ratios over a paired counterfactual; C2 has no classes and no x₀,
   and IM2's ‖x‖₁ denominator would reward C1's arm for shrinking the vector). Replacement: a
   **reconstruction-plausibility score** — one AE fitted on real fold-in latents,
   ‖ẑ − AE(ẑ)‖² against the spectrum-matched null and the correspondence-scrambled control —
   citing Van Looveren & Klaise for the idea and Hvilshøj et al. for why the published form is not
   used. Never call it IM1.
3. **C4's formula as written here was WRONG** (Hewitt full read, verified against their raw
   tables): their denominator is the *training data's* score, not the latent's — the latent is an
   arm, not the normaliser. Our instantiation: the external ruler is the **measured best real
   question at the state** (so latent, text, and all controls sit on one ruler none of them
   defines); the latent-in-denominator form is retired. Additional C4 requirements from the read:
   an **adapter-ceiling row** (a real question's own text through the bridge vs its measured
   effect — the C3B `ceiling` subcommand is this row's prototype); the "machine-only synonyms"
   analogue needs a second independently-fitted text→CF map or the claim weakens to plausibility
   only; and the C4 scalar is fixed in advance. Also: the retrieval-prompting ARM and the
   nearest-bank CONTROL currently coincide — they must be constructed distinctly or merged into
   one row labelled as both.
4. **Every C-series statistic carries a CI, with the declared aggregation unit** — the repair pass
   found the stack had none; DS_C3B clause 5 generalises to all future sheets. Registered reading
   rules must be applied to their null control at design time (the C0 lesson: a cell the null
   occupies decides nothing).
5. Verified-source updates binding on the prose: ELM has an output-side random baseline (narrow
   every "no random baseline" sentence to "no input-side embedding null"); DeCap's 25.6 is a
   decoder-free retrieval baseline, not a ladder rung; Heap's strong null is itself Gaussian —
   the axis is correspondence-vs-weights, not Gaussian-vs-not; Info-Synth replicates ANSWER-FREE
   only (its response model is its acquisition objective — a simulated-answer arm would be
   unfalsifiable).

## ★★★ FABLE VERDICT (2026-08-18) — BINDING. Read before touching anything below.

**"Build it, build it around variant (b), and hold the claim at exactly what the answer-free design
can carry."** Full text of the ruling is in this section; the staged design below stands except where
this contradicts it.

### The claim is TRIMMED. Delete "worth asking about".

The drafted marquee sentence describes a paper that asks somebody something and folds the answer
back. **This one does neither** — the circularity ruling forbids simulated answerers and the 300
study users are ruled OUT (below). Fable: *"trim the claim to match the build and this is a good bonus
paper; keep the claim as drafted and it's your next retraction."*

**The claim now reads:** *the policy emits off-catalogue directions that carry real structure, a
trained renderer can say them in English, and the text — unlike text from a correspondence-scrambled
latent — reproduces the direction's causal effect.*

**Worth is INHERITED, not measured.** C1's regression target is a gain-weighted blend of top-m
**measured** candidates, so the emitted point's value is bounded by a convex combination of measured
question values. Say exactly that, and claim no off-catalogue EXCESS value.

### RENDERER: variant (b), the trained adapter. Variant (a) becomes a BASELINE ARM.

- **(a) propose-and-encode is disqualified as the claim-bearing renderer** — not for engineering
  reasons but because **it breaks the control**: the LLM never sees the vector, so scrambling `ẑ`
  only changes which bank phrases are retrieved, testing RETRIEVAL rather than RENDERING. Per Heap,
  only a control destroying the vector→meaning correspondence separates anything, which requires the
  vector to condition generation directly. It is also Baum–Lang fix #1 (snap to a neighbour) wearing
  the costume of fix #2 (verbalise) — it launders the snap rather than removing it, and it is
  encoder-only, i.e. EAGLE-shaped, shrinking our daylight exactly where we need it.
- **(a) IS kept as a measured baseline:** showing the anchored pipeline recovers bank questions
  (near-zero off-catalogue residual) while (b) produces text the bank does not contain **is itself
  evidence for the in-between claim**. Its fatal flaw becomes a table row.
- **(b) is the only variant under which "the vector conditions what is said" is literally true** —
  the sentence the paper rests on. ELM owning the mechanism is FINE and actually better: our
  contribution is the evaluation its field has never run, and using the standard mechanism makes
  "we audited it properly" cleaner than inventing a bespoke one.
- **(c)** = (b) plus encoder re-ranking of its own samples; the residual-distance number is worth
  having, but it reintroduces the select/validate edge and lives or dies by the discipline below.

### The two-adapter split is NECESSARY BUT NOT SUFFICIENT

*"Two copies of one ruler is one ruler."* Same architecture, same text encoder, same training
distribution ⇒ the same systematic mapping errors, so a candidate gaming `W_A` scores well under
`W_B` for the same wrong reason. Three additions, the third decisive:
1. **Different encoder FAMILY** for the validator (different pretraining corpus), not just a
   different data half.
2. **Report the two rulers' correlation** on held-out real pairs — show independence, do not assert it.
3. **★ Selection and validation must optimise DIFFERENT QUANTITIES.** Select on geometric closeness
   to `ẑ` under `W_A`; validate on **plug-in causal-gap closure through the frozen CF model** under
   `W_B`. Cosine-to-a-point and %-of-downstream-effect are different objectives.
Plus the instrument checks: nearest-bank text must close LESS of the gap than the selected text, and
scrambled text ~none. **If scrambled text closes the gap through `W_B`, the validator is broken.**

### C0–C2 alone: THESIS CHAPTER ONLY, not a paper

Without the renderer it is Info-Synth's cell plus better controls — a section, not a contribution.
Its real value is as a **free kill-gate**: C0's falsifiable prediction (one-hot-trained `ẑ` sits ON
the nearest question) is probably what we will see, which is exactly why C1 exists. **If C1 cannot
pull `ẑ` off the grid, the paper dies for the price of a geometry check** — the correct price. Do not
publish the corpse as a capability result (HARD RULE 6: a failed C1 is a weak method, not a theorem).

### THREE FRAMING ERRORS TO FIX

1. **C2 as specified may be theatre.** In a CF space with dominant principal components, *almost
   every* direction projects real users differently — discriminativeness is nearly free geometry.
   **The null must be norm- and spectrum-MATCHED directions, not Gaussian noise**, and the
   load-bearing control everywhere is **correspondence-scrambling** (scramble which embedding pairs
   with which phrase BEFORE adapter training). **Pre-register the null before running it.**
2. **C must state it claims CAPABILITY, NOT VALUE** — otherwise the thesis argues with itself, since
   Paper B established that *the action space was not the constraint* (a richer bank made selectors
   fail harder). One sentence, but it must be there.
3. **"No baselines to beat" ≠ safe.** The likely C0 outcome is the negative prediction.

### ★ AMENDMENT (author, 2026-08-18): the human-study bar was OVERSTATED by me

The framing above implied Paper C would need the C3 human round-trip. **It does not, and the field's
norm is far cheaper than I implied.** Author: *"as if many people do human studies. i bet no one like
elm etc did them either."* Checked:

- **ELM: 100 human raters. EAGLE: 124 post-hoc raters. ctELM: an expert discrimination test** ("fools
  the experts 44%"). So the neighbours DO use humans — but only to **rate generated text after the
  fact**. A short Prolific task; standard, cheap.
- **Our C3 design (`STUDY_C3_HUMAN_ROUNDTRIP.md`) is a different animal**: import a real film history,
  hide half, run a full interview, score recommendations against the hidden half. Rare because it is
  expensive. Conflating the two made the bar look far higher than it is.

**Paper C needs neither.** Its claim is faithfulness — does the text reproduce the vector's effect —
measured through the frozen CF model, not through people. And there is a principled reason to skip
even the cheap panel: Hewitt's **"machine-only synonyms"** (verbalisations humans judge WRONG that
nonetheless reproduce the behaviour). If that holds, a plausibility panel actively misleads — which
is precisely the audit this paper proposes, and awkward for ELM's 100 raters.

**Net effect on the ruling below: unchanged, but for a better reason.** The 300 stay reserved for A/B
not because C is unworthy, but because C genuinely does not need humans to make its claim, and the
one human instrument the field uses is one we have an argument against.

### The 300 quarantined study users: SAVE THEM. Not "mostly" — save.

Irreplaceable, the only non-circular human answer source, and the serious papers (A, B) have a
standing claim on them; the bonus paper does not. The whole C design was engineered to be answer-free
— spending the one resource it was built to avoid needing would be bad portfolio allocation. **If**
C4 ever completes clean and a headline is judged worth it, the single highest-leverage spend is a
small capped slice on Baum–Lang **answerability** (can real people answer the rendered questions at
all), never on quality grading.

### What is actually interesting — sell the AUDIT

*"'A question nobody has a word for yet' is catchy and will get the paper read; as a contribution on
its own it is a solution in search of a problem — recsys will ask 'does asking it help?' and you have
forbidden yourself the answer."* **The durable contribution is the measurement**: twelve
embedding-verbalisation systems, zero scrambled controls, in a field where (per Heap) the naive null
actively misleads. **Sell the audit, decorate with the sexy sentence — not the reverse.**

### The daylight is the CONJUNCTION, never the legs

(i) the decoded point is **selected by an interview policy at a state** — ELM has nothing choosing
which vector to decode; (ii) the **scrambled-correspondence control** — absent in all twelve;
(iii) **plug-in causal faithfulness** — absent in ELM and EAGLE, whose only evidence is 124 post-hoc
raters. Per the Aug-12 lesson: **claim the conjunction, never the individual legs.**

### Two futures in which this becomes a bad idea

The claim keeps the word "worth", or it starts consuming the 300 users. **Guard those two doors.**

---

Supersedes the "two fixes" sketch in `PAPER_C_EMISSION_DESIGN.md`, which stands as the blocker
analysis. Brief and settled facts: `PAPER_C_TODO.md`. Evidence base:
`external_literature/findings/{continuous_elicitation,llm_verbalisation}_2026-08.md`.

## The claim, in one sentence

~~We can find a point in recommender space *worth asking about*…~~ **RETRACTED before use** — Fable,
2026-08-18: the word "worth" describes a paper that asks somebody something, and this one does not.

**The claim is:** *the policy emits off-catalogue directions that carry real structure, a trained
renderer can say them in English, and the text — unlike text from a correspondence-scrambled latent —
reproduces the direction's causal effect.* Worth is INHERITED from C1's construction (a blend of
MEASURED candidates bounds it), never measured, and no off-catalogue EXCESS value is claimed.

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

**STATUS 2026-08-21 — ALL EIGHT WORKED. 5 cleared, 2 cleared-with-correction, 1 partial; none
refuted.** Evidence per item → `external_literature/findings/continuous_elicitation_2026-08.md`
§"VERIFICATION PASS — 2026-08-21"; bib entries with provenance notes are now in
`new_chapters/references.bib`; INDEX rows under "Verification pass 2026-08-21".

- **GenMQD** — ⚠ **PARTIAL.** *Mechanism CLEARED:* generates a **textual description of the wanted
  instance instead of the instance**, CLIP-style alignment, text used to gather real data; **plus a
  large-scale user study where human oracles PREFER the text queries over generated image queries**,
  and +2.43% accuracy at 500 labels. *Bibliography NOT cleared:* OpenReview `CZvbXXgjrn`, an **ICLR
  2025 submission**, no published version and **no author list** obtainable (bot-check on every
  endpoint). **Cite as an OpenReview preprint or not at all; never attribute authors.**
- **Vendrov's actual query object** — ✅ **CLEARED, and our fallback reading was WRONG.** Not soft
  item-selection weights: it optimises **continuous d-dimensional hypothetical item vectors**
  ("allowing attributes to vary continuously without requiring they correspond to available items")
  and then **projects them back to the nearest actual items** via a *deep retrieval* step; it also has
  **partial comparison queries** over attribute subsets, explicitly to cut cognitive burden.
  ⇒ **synthesis-then-snap inside recsys, at AAAI 2020, by a likely reviewer.** Two consequences for
  this document: "everyone snaps" becomes a **four-field** claim including our own, and **"optimise
  the query continuously" is pre-empted in recommendation** — our claim must be stated against the
  three things Vendrov does not do (collaborative rather than attribute latent; rendered as language;
  left unsnapped).
- **LineCoSpar's 6 params / 6 subjects / 30 trials** — ✅ **CLEARED EXACTLY** from the paper's own
  text (IROS 2020, arXiv:2003.06495): 6 named gait parameters, 6 able-bodied subjects on the Atalante
  exoskeleton, 30 learning trials **+ 6 validation trials**, "at most 29 pairwise preferences and 30
  pieces of coactive feedback". Quotable.
- **ADO's precision claims** — ⚠ **CLEARED WITH CORRECTION, and MIS-ATTRIBUTED.** The numbers are
  **Ahn et al., *Scientific Reports* 10:12091 (2020)**, not ADOpy: ≥0.95 test–retest reliability
  **within 10–20 trials**, ~10% more variance, **3–5× more precise**, **3–8× more efficient** than
  staircase, in three real populations. **"Fewer than 20 trials" was wrong — write "10–20 trials"**,
  and note it attaches to *reliability*, not to the precision figure. Cite ADOpy only for the method.
- **LOL-BO's "20×"** — ✅ **CLEARED, and it is narrower than it reads:** verbatim "as much as 20 times
  improvement over state-of-the-art latent space Bayesian optimization methods across six real-world
  benchmarks". A best-case **optimisation / sample-efficiency** result; **it says nothing about decode
  validity**. Quote only in that exact form, and keep the citable content as the *mechanism* (latent
  trust region + retraining toward the promising region).
- **Baum & Lang's exact bibliographic details** — ⚠ **CLEARED WITH CORRECTION.** *Baum, E. B. and
  Lang, K. (1992), "Query learning can work poorly when a human oracle is used", IJCNN, volume 8.*
  **There is NO page range** — the `pages={8}` in circulating BibTeX is a mis-parse of "volume 8"; do
  not emit a `pages` field. **The venue city is Beijing, not Baltimore** (our note said Baltimore).
  The finding — synthesised digit queries were **fusions (a 3/5 hybrid) no human can label**, yielding
  **noisy labels** — is corroborated across four independent bibliographies, but the 1992 proceedings
  text was never obtained: **cite the claim, never a number.**
- **Tang et al. Nature Neuroscience 2023** — ✅ **CLEARED.** *Nat. Neurosci.* **26(5):858–866**, doi
  10.1038/s41593-023-01304-9 (preprint bioRxiv 2022.09.29.509744, itself 403-walled; abstract and
  null-model description recovered elsewhere). fMRI decoder reconstructing continuous language for
  perceived speech, imagined speech and silent video; **subject cooperation required to train and to
  apply it**. **★ The load-bearing part is the evaluation:** decoded sequences are scored against a
  null distribution of information-free null sequences **plus an alternative null that permutes the
  voxel responses (×10), preserving response statistics while destroying the response→stimulus
  correspondence**, compared by a one-sided nonparametric test. **⇒ NARROW the novelty claim for our
  scrambled control**: scrambled-correspondence controls exist (here, and in Heap et al.); what is
  unclaimed is one for a **rendered elicitation QUESTION**, validated without a paired reference.
  Import the **null construction**, not the metric — they have a reference transcript and we do not.
