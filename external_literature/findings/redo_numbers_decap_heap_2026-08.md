# \REDO number verification: DeCap CIDEr ladder + Heap et al. auto-interp AUROCs

**Date:** 2026-08-21 · **Scope:** verification only.

> **SCOPE NOTE — no downstream edits made.** This file does *not* edit
> `external_literature/INDEX.md`, `references.bib`, or
> `new_chapters/chapterC_v2/chapterC_v2.tex`. The chapter edit that clears the two `\REDO`
> flags (lines ~220–221 and ~241–243 of `chapterC_v2.tex`) happens in a **later coordinated
> pass**. Nothing here is committed.

Both papers were read from their arXiv PDFs (`arxiv.org/pdf/2303.03032`,
`arxiv.org/pdf/2501.17727`, v2, dated 27 Jan 2026), results sections plus the relevant
appendices. `[V]` = verified verbatim against the paper's own table/figure.

---

## 1. DeCap — Li, Zhu, Wen, Yang, ICLR 2023, arXiv:2303.03032

### Where the ladder lives

**Table 1**, p. 6: *"Zero-shot captioning results on MSCOCO Karpathy-test split and NoCaps
validation set. (In: in-domain; Near: near-domain; Out: out-of-domain; B@4: BLEU@4; M:
METEOR; C: CIDEr; S: SPICE)."* `[V]`

All four quoted values sit in the **`C` (CIDEr) column of the MSCOCO block**, all in rows
with pre-training stage **`CLIP+CC3M-text`** (decoder trained on CC3M text only; support
memory = 1M descriptions sampled from the 3M). Verbatim rows: `[V]`

```
Methods        Pre-training stage   MSCOCO                 NoCaps val (CIDEr)
                                    B@4   M     C     S    In   Near  Out   Overall
CLIPRe         CLIP+CC3M-text       4.6  13.3  25.6  9.2  23.3 26.8  36.5  28.2
DeCap-VD       CLIP+CC3M-text       1.2  10.4   8.1  5.8   8.4  8.0  10.2   8.5
DeCap-NND      CLIP+CC3M-text       5.3  13.7  27.1  9.1  24.2 27.1  37.6  28.8
DeCap          CLIP+CC3M-text       8.8  16.0  42.1 10.9  34.8 37.7  49.9  39.7
```

### Per-number verdict

| Chapter's rung | Value | Verdict | Where | What it actually measures |
|---|---|---|---|---|
| raw off-manifold | **8.1** | **CONFIRMED** `[V]` | Table 1, row `DeCap-VD`, col `C` | CIDEr on MSCOCO Karpathy-test, Visual Decoding: the CLIP **image** embedding fed directly to the text-trained decoder as prefix |
| hard NN snap | **27.1** | **CONFIRMED** `[V]` | Table 1, row `DeCap-NND`, col `C` | same metric/split, Nearest-neighbor Decoding: the single nearest **text** embedding in support memory used as prefix |
| softmax blend | **42.1** | **CONFIRMED** `[V]` | Table 1, row `DeCap` (default = projection-based decoding), col `C` | same metric/split, softmax-weighted convex combination of *all* support-memory text embeddings |
| "25.6" | **25.6** | **REAL NUMBER, NOT A RUNG OF THE LADDER** `[V]` | Table 1, row `CLIPRe`, col `C` | CIDEr of a **decoder-free retrieval baseline**: return the highest CLIP-similarity caption from the corpus verbatim |

**On the 25.6.** It is not a fourth decoding strategy of DeCap. `CLIPRe` uses **no decoder at
all** — §3.2.2(1): *"a simple retrieval-based approach that does not require a decoder … CLIPRe
retrieves the most relevant texts from T based on the image-text similarity measured by CLIP …
In all experiments, we use CLIPRe as our baseline, since it can well reflect the zero-shot
performance of the original CLIP without the decoder."* `[V]` So the chapter's four-way
ordering *soft blend ≫ hard snap > retrieval ≫ raw* is **numerically correct**
(42.1 > 27.1 > 25.6 > 8.1) `[V]`, but the sentence must not present 25.6 as a decoder
inference strategy — it is the no-decoder retrieval floor. Note also the paper's own remark
that NND beating CLIPRe is mildly surprising: *"Ideally, NND and CLIPRe should attain similar
performance since the decoder learns to recover the origin text conditioned on the text
embedding. Interestingly, across our experiments, NND achieves better performance than
CLIPRe"* (§3.2.2(3)) `[V]`.

### Definitions, for making the chapter sentence exact (§3.2, pp. 4–5) `[V]`

- **PD (projection-based decoding, = plain "DeCap")**, Eq. 2:
  `v_proj = Σ_i w_i · m_i` with `w_i = softmax((m_i^T v)/τ)` over the support memory `M`,
  then decode `P_θ(v_proj/||v_proj||₂)`. Temperature **τ = 1/100 for image captioning**
  (1/150 for video). *This is exactly our "softmax blend".*
- **VD (visual decoding)**: `P_θ(E_image(I))` — *"a simple approach is to directly use image
  embedding as the preﬁx embedding … this method does not achieve satisfying results in most
  scenarios, indicating that there is a modality gap between CLIP image embeddings and text
  embeddings."* This is the off-manifold rung; the off-manifold-ness is the **CLIP modality
  gap**, not synthetic noise. Worth stating in the chapter, since our own off-manifold point
  arises differently (a synthesised CF latent).
- **NND (nearest-neighbour decoding)**: `P_θ(argmax_{m∈M} sim(E_image(I), m))`.

### Caveats for the chapter sentence

1. The ladder is **CIDEr on MSCOCO Karpathy-test**, zero-shot, decoder trained on
   **CC3M-text only**, frozen **ViT-B/32** CLIP, 4-layer/4-head transformer decoder,
   hidden 768, **no beam search** (§4 Implementation Details) `[V]`.
2. The same ladder reproduces on **NoCaps overall** (8.5 / 28.8 / 39.7, CLIPRe 28.2) `[V]`
   and on **VATEX video captioning** (DeCap-VD 13.8 / DeCap-NND 32.4 / DeCap 43.1 CIDEr,
   Table 5 region, p. ~9) `[V]` — so the ordering is not a one-dataset artifact. Optional
   strengthener for the chapter.
3. Do **not** confuse 42.1 with the SS1M row: `DeCap CLIP+SS1M` = **50.6** CIDEr, and the
   text explicitly contrasts them — *"DeCap trained on SS1M achieves better performance than
   trained on CC3M (CIDEr: 50.6% vs. 42.1%)"* `[V]`. The ladder must stay inside the
   CC3M-text block or the comparison is not controlled.

**Suggested exact form for the chapter (not applied):** "…feeding a genuinely off-manifold
vector (the raw CLIP image embedding) to a text-trained decoder collapses zero-shot MSCOCO
CIDEr to 8.1; a hard nearest-neighbour snap to the support memory recovers 27.1, roughly the
level of decoder-free CLIP retrieval (25.6); and a softmax-weighted blend over that memory
reaches 42.1 (DeCap, Table 1, all rows trained on CC3M text only)."

---

## 2. Heap, Lawson, Farnik, Aitchison — *Automated Interpretability Metrics Do Not
Distinguish Trained and Random Transformers*, arXiv:2501.17727v2

### Where the numbers live

**Figure 1** (p. 4) — **a figure legend, not a table.** There is no table of AUROCs in this
paper; the values are printed in the legend of Fig. 1 and of the Appendix B figures. The
chapter's `\REDO` note says "results tables" — that phrasing should change to "Figure 1".

Caption verbatim: *"Figure 1: 'Fuzzing' ROC curve vs. layer for **Pythia-6.9b** (100 latents
sampled per SAE). The trained model (gray line) and randomized variants (colored) overlap,
whereas the control (black) is near chance (dotted). This suggests aggregate AUROC alone is
insufficient to attribute latents to learned computation. See Figure 2 for other
metrics/model sizes and Appendix E for multiple random seeds."* `[V]`

Legend verbatim (verified against the rendered page image, not only the text layer) `[V]`:

```
Trained (AUC = 0.79)          Control (AUC = 0.50)        Randomized incl emb (AUC = 0.87)
Randomized excl emb (AUC = 0.87)   Step 0 (AUC = 0.88)
```

### Per-number verdict

| Chapter's number | Verdict | Where | What it actually measures |
|---|---|---|---|
| trained **0.79** | **CONFIRMED** `[V]` | Fig. 1 legend | 'Fuzzing' auto-interp AUROC, SAEs on residual-stream activations of **Pythia-6.9b**, 100 latents per SAE |
| randomised **0.87** | **CONFIRMED** `[V]` | Fig. 1 legend | *both* randomised variants read 0.87 — "Randomized excl emb" **and** "Randomized incl emb". A third null, **Step 0 = 0.88**, is *higher still* and is currently unquoted |
| control **0.50** | **CONFIRMED** `[V]` | Fig. 1 legend | the per-token Gaussian-embedding control, i.e. exactly at chance |

Nothing corrected. Three additions the chapter should absorb:

**(a) The scale is load-bearing and must be named.** 0.79/0.87/0.50 are **Pythia-6.9b,
'fuzzing' scoring, SAEs trained on 100M RedPajama tokens**. The chapter currently states the
scale-dependence caveat but does not name the model. The appendix makes the dependence
concrete — the "no separation" result **does not hold at small scale**: `[V]`

| Model | Trained | Rand. excl emb | Rand. incl emb | Step 0 | Control | Source |
|---|---|---|---|---|---|---|
| Pythia-70m (fuzz) | 0.63 | **0.50** | **0.49** | 0.67 | 0.44 | Fig. 6 |
| Pythia-160m (fuzz) | 0.63 | 0.56 | 0.55 | 0.73 | 0.45 | Fig. 8 |
| Pythia-1b (fuzz) | 0.77 | 0.83 | 0.82 | 0.84 | 0.49 | Fig. 12 |
| Pythia-6.9b (fuzz) | **0.79** | **0.87** | **0.87** | 0.88 | **0.50** | **Fig. 1** |
| Pythia-1b (detection) | 0.72 | 0.76 | 0.77 | 0.79 | 0.51 | Fig. 13 |
| Pythia-6.9b (detection) | 0.72 | 0.82 | 0.83 | 0.84 | 0.54 | Fig. 14 |

Their own wording: *"we found that auto-interpretability scores for randomized models were
relatively low for smaller models (e.g., Pythia-70m) but that the gap was narrowed for larger
models (e.g., Pythia-6.9b)"* (§2) `[V]`, and *"In the AUROC plots, we find that for all but
the control variant, AUROC increases with model size"* (§3) `[V]`. This **supports** the
chapter's existing sentence that the control must be run at our operating scale — it can now
be cited with the 70m-vs-6.9b contrast rather than asserted.

**(b) Exact construction of the control — verbatim, §3 (p. 3):** `[V]`

> **Control:** The original, trained model, except where the input token embeddings are
> replaced at inference time by sampling i.i.d. standard Gaussian noise for each token, such
> that a given token does not have a consistent embedding vector. For this variant, we expect
> auto-interpretability to perform at the level of chance.

**Read this carefully — it changes a chapter phrase.** The control is *itself Gaussian*.
Nothing is permuted or "scrambled" in the shuffle sense; per-token embeddings are **replaced
with fresh i.i.d. standard Gaussian noise at every occurrence**, so the destroyed quantity is
the **consistency of the token→vector map** (the correspondence), not the distribution's
shape. The chapter's phrase *"a Gaussian or norm-matched random vector is a demonstrably weak
null in this setting"* is therefore **imprecise as written**: the *weak* nulls are the
Gaussian-**weight** re-randomisations, and the *strong* null is also Gaussian. The separating
axis is **what** is randomised (model parameters vs. the token→embedding correspondence), not
Gaussian-vs-not. The chapter's own maxim — *scramble the correspondence, not the
distribution* — is exactly right and is what should carry the sentence; the "Gaussian null is
weak" clause should be narrowed to "randomising the **weights** — even matching each trained
matrix's mean and variance — is a demonstrably weak null."

The chapter's own summary of the control (*"token embeddings were resampled per occurrence,
so that no consistent vector-to-meaning map exists"*) is **accurate** `[V]` — it is only the
following Gaussian clause that needs narrowing.

**(c) Exact construction of the weak nulls — verbatim, §3 (p. 3):** `[V]`

> **Re-randomized incl. embeddings:** All the model parameters, including the embeddings, are
> re-initialized by sampling Gaussian noise with mean and variance equal to the values for
> each of the original, trained weight matrices.
> **Re-randomized excl. embeddings:** As above, except the embedding and unembedding weight
> matrices are not re-initialized, i.e., are the same as the original, trained model.
> **Step-0:** For Pythia models, the step0 revisions are available, which are the original
> model weights at initialization, i.e., before any learning.

Note this is a *norm-preserving* randomisation by design — §3: *"our randomization procedure
was specifically designed to preserve parameter norms with respect to the trained model"*
`[V]`. So "norm-matched random weights score **higher** than the trained model" is a fair and
strong reading, and is the sharpest version of the chapter's point.

### What "auto-interp score / AUROC" exactly is (§3, p. 4) `[V]`

Not a similarity score. It is the **AUROC of a classifier that is a language model reading a
natural-language explanation of an SAE latent**:

> Feature explanations that identify a concept can be input to a classifier that predicts
> whether the concept appears in the text inputs. Such a classifier may be evaluted by
> traditional metrics, like the area under the receiver operating characteristic (ROC) curve
> (AUROC). Paulo et al. (2024) proposed 'fuzzing' and 'detection' classification tasks to
> evaluate feature explanations. For 'fuzzing' scoring, both positive and negative examples
> of tokens (i.e., with non-zero and zero activation values, respectively) for a given latent
> are delimited with special characters, and a language model is prompted to identify which
> examples have been correctly delimited for the latent given its explanation. For
> 'detection', a language model is asked to identify which examples contain activating tokens
> for each feature.

Reported metric is **'fuzzing'** unless noted (*"because this measure has been demonstrated
to correlate with simulation scoring"*); 'detection' curves are in Appendix B `[V]`.
Explainer/judge model: **Meta-Llama-3.1-70B-Instruct-AWQ-INT4** `[V]`. Sampling: **100
randomly sampled latents per (model, variant, layer)** `[V]`. SAEs: **TopK / k-sparse**,
expansion factor **R = 64**, sparsity **k = 32**, trained on **100M RedPajama tokens**,
activation buffer 10M `[V]`; robustness over R ∈ {16…128}, k ∈ {16, 32} in Fig. 18, 1B-token
replication in Appendix C, five-seed uncertainty for Pythia-70m in Fig. 17 `[V]`.

This is a genuinely close structural analogue of our plug-in test: *a natural-language
rendering of a latent is validated by whether a downstream reader can use it to recover the
latent's behaviour* — which is worth one clause in the chapter, since it upgrades the Heap
citation from "someone else ran a random control" to "someone else ran a random control **on
a verbalisation**".

### Citation hygiene

- The paper's title is *"Automated Interpretability Metrics Do Not Distinguish Trained and
  Random Transformers"* `[V]` — check `references.bib` key `heap2026autointerp` carries this
  (the v2 stamp is 27 Jan 2026; v1 was Jan 2025, so a 2025-vs-2026 year field is a real
  choice to make, not an error either way).
- Cite **Figure 1**, not a table. There is no AUROC table in the paper.
- Their own conclusion is deliberately narrow and matches our use: *"we do not [show] that
  SAEs fail to find meaningful features in trained transformers; only that aggregate
  auto-interpretability measures do not necessarily indicate the [presence of learned
  features]"* (§5, paraphrase of lines around p. 11) — do not over-claim.

---

## Summary verdict table

| # | Quoted | Verdict | Locus |
|---|---|---|---|
| 1 | DeCap raw off-manifold CIDEr **8.1** | **CONFIRMED** | Table 1, `DeCap-VD` / MSCOCO `C` |
| 2 | DeCap hard-snap CIDEr **27.1** | **CONFIRMED** | Table 1, `DeCap-NND` / MSCOCO `C` |
| 3 | DeCap blend CIDEr **42.1** | **CONFIRMED** | Table 1, `DeCap` (CC3M-text) / MSCOCO `C` |
| 4 | DeCap **25.6** as a fourth rung | **REAL VALUE, MISLABELLED** — it is the decoder-free `CLIPRe` retrieval baseline, not a DeCap decoding strategy | Table 1, `CLIPRe` / MSCOCO `C` |
| 5 | Heap trained AUROC **0.79** | **CONFIRMED** (Pythia-6.9b, fuzzing) | Figure 1 legend |
| 6 | Heap randomised AUROC **0.87** | **CONFIRMED** (both re-randomised variants; Step-0 is 0.88, higher) | Figure 1 legend |
| 7 | Heap control AUROC **0.50** | **CONFIRMED** (per-token i.i.d. Gaussian embedding replacement) | Figure 1 legend |

**Chapter changes implied (for the later pass, not made here):** name MSCOCO/CIDEr/CC3M-text
and demote 25.6 to "decoder-free retrieval baseline" in the DeCap sentence; name Pythia-6.9b
+ 'fuzzing' and narrow the "Gaussian null is weak" clause to "randomising the *weights*,
norm-matched, is a weak null" in the Heap sentence; change "results tables" to "Figure 1" in
the Heap citation. Both `\REDO` macros can then be deleted.
