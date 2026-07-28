# Findings: Text → CF-Latent Projection as a PREFERENCE Signal

*Stress-test of the claim: "a single linear ridge adapter W = QᵀS(SᵀS+βI)⁻¹ from frozen SBERT (384-d) into a frozen RecVAE-class CF latent (200-d) turns ANY free-text phrase into a preference direction that folds into the user belief exactly like an item token — no per-concept training, inference-time only."*
Deep research 2026-07-28 (WebSearch/WebFetch). Verification level is marked per item: **[V]** = verified from the paper's own text (HTML/PDF fetched), **[A]** = from the paper's abstract/landing page only, **[S]** = secondary source (search snippet, survey, repo README) — treat [S] as a lead, not evidence.

---

## What we concluded

- **The claim is NOT pre-empted as a whole, but it must be narrowed on two axes**: (i) the *mechanism* (content/attribute → CF-latent regression) is 15+ years old and must be cited as ancestry, not claimed; (ii) the *use* (arbitrary open-vocabulary phrase as a value-carrying elicitation token in a belief that also holds item tokens) is where we found no match.
- **Two distinct literatures bracket us, and neither occupies the cell.**
  - **Text→ITEM-representation** (UniSRec, VQ-Rec, Recformer, MoRec, TIGER/semantic IDs, CB2CF, DropoutNet, EasyRec, A-LLMRec, LLaRA, E4SRec): text *replaces or initialises an item embedding*. Text is never a user-side preference input. NOT our claim.
  - **Attribute/concept-as-preference** (Göpfert/Boutilier CAVs, Biyik et al. 2023, Latent Linear Critiquing, CE-VAE, BK-VAE, M&Ms-VAE, LACE, Balog/Radlinski NL profiles): concepts *are* preference inputs, but every one of them draws from a **closed, curated vocabulary with per-concept supervision** (a probe set, a keyphrase inventory, a tag column, a concept bank). None derives the direction from a *text encoder* so that an unseen phrase works with zero per-concept data.
    > ⚠ **CORRECTED 2026-07-28 [V]:** Balog et al. 2021 is a **partial exception** — its §5.2 Centroid-Based and §5.3 WWD methods derive an attribute direction in a frozen CF latent with **no attribute labels**, by running the phrase as a BM25 query over a review corpus and taking the centroid / a pseudo-label regression over the retrieved items' embeddings. Label-free ≠ unclaimed. See the VERIFIED section at the end of this file.
- **The single most threatening paper is Göpfert et al. (WWW 2022 / TORS 2024), "Discovering Personalized Semantics for Soft Attributes … using Concept Activation Vectors"** — same latent space, same "attribute = direction", frozen CF, used for critiquing. It is *not* a kill shot for one reason **now verified from its own text [V, 2026-07-28]**: its CAVs are trained **per attribute by logistic regression / RankNet / LambdaRank on labelled item exemplars** (user-applied MovieLens-20M **tags** — ⚠ *not* the tag genome — and, separately, the Balog SoftAttributes rater comparisons), so the attribute vocabulary is closed (164 tags; 60/36 attributes). Our contribution is precisely the removal of that per-concept probe set via one global text adapter.
- **The direction of travel in the 2024–2026 LLM×RecSys literature is the OPPOSITE of ours**: nearly everything projects **CF embeddings → LLM token space** (A-LLMRec, LLaRA, E4SRec, ELM, FACE, SAILRec). Our direction (text → CF latent, used as a *user-side* signal) is comparatively under-populated — which is a positioning advantage but also means we should say "we are aware of none", never "there are none".
- **On ridge-vs-contrastive (our median rank 38 vs InfoNCE 206):** we found **no published head-to-head** on off-manifold extrapolation for text→CF alignment. But there is **directly supportive independent evidence**: Wang et al. (SIGIR '26, arXiv:2604.22195) probe semantic→collaborative mappings and find a **linear map holds up on held-out items (R² 0.171 train → 0.166 test) while an MLP-3 collapses to R² = −0.495** [V]. That is the same qualitative phenomenon (simpler map generalises off the fitting manifold) from an independent group, though measured as R² on unseen *items*, not on non-item concept phrases.

---

## Key external methods

### A. Text → ITEM representation (the "not our claim" side — cite to disambiguate)

| System | Cite | What it does | Text is a…? | CF frozen? | Per-concept training? |
|---|---|---|---|---|---|
| UniSRec | Hou et al., KDD 2022, arXiv:2206.05941 | Item *description text* → universal item representation via parametric whitening + MoE adaptor, for cross-domain transfer [S] | item repr. | no (trained end-to-end) | n/a |
| VQ-Rec | Hou et al., WWW 2023 | Item text → **discrete codes**, learns transferable code embeddings; decouples text encoding from representation learning [S] | item repr. | no | n/a |
| Recformer | Li et al., KDD 2023 | Models items *and* user preference as language sequences (item = sentence of attribute key-values) [S] | item repr. | no | n/a |
| MoRec | Yuan et al., SIGIR 2023 | Modality (text/image) encoder replaces item-ID embedding; "ID vs modality" study [S] | item repr. | no | n/a |
| TIGER / semantic IDs | Rajput et al., NeurIPS 2023 | RQ-VAE over item content embeddings → semantic ID tokens for generative retrieval [S] | item repr. | no | n/a |
| CB2CF | Barkan, Koenigstein, Yogev, Katz, **RecSys 2019** | Neural multiview **content → CF vector** regression for completely cold items (Microsoft Store) [S] | item repr. (regressed **into** a CF space) | effectively yes (CF vectors are the target) | no — but items only |
| DropoutNet | Volkovs et al., NeurIPS 2017 | Interaction-dropout so content alone reproduces the warm latent [S; already in INDEX] | item/user repr. | latent is the target | no |
| EasyRec | Ren & Huang, arXiv:2408.08821 / EMNLP 2025 | RoBERTa text encoder **trained contrastively** against collaborative signals; encodes LLM-generated user/item *profiles* for zero-shot rec [V] | both, but profiles are LLM-generated summaries of history, **not arbitrary free-text** [V] | **no** — the text encoder is the trained component | no |
| A-LLMRec | Kim et al., **KDD 2024**, arXiv:2404.11343 | Alignment network between a frozen SASRec and a frozen LLM; **fine-tunes neither** [S] | CF → LLM token space (opposite direction) | yes | no |
| LLaRA | Liao et al., SIGIR 2024, arXiv:2312.02445 | Projector aligns ID embeddings into LLM input space, curriculum from text-only → hybrid prompts [S] | CF → LLM | no | no |
| E4SRec | Li et al., arXiv:2312.02443 | Injects pretrained item-ID embeddings into an LLM as tokens; LoRA [S] | CF → LLM | item embeddings frozen, LLM LoRA-tuned | no |
| RLMRec | Ren et al., **WWW 2024**, arXiv:2310.15950 | LLM-written user/item *profiles* aligned to CF representations by contrastive (`-Con`) or masked-generative (`-Gen`) cross-view alignment [S] | auxiliary training signal that **reshapes** the CF encoder | **no** — CF encoder is retrained | no |
| ELM | Tennenholtz et al., arXiv:2310.04475 (Google) | Adapter maps **domain (CF) embeddings → LLM token space** to *verbalise* embeddings; CF model frozen [V] | embedding → text | yes | uses Göpfert CAVs (per-attribute, see below) |

> **Verified negative on ELM** [V]: we specifically checked whether ELM has a text→embedding path (hypothetical-item descriptions, embedding arithmetic from a phrase). It does not — "trained adapters … map domain embedding vectors into the token-level embedding space of an LLM". Its attribute directions are **CAVs trained per-attribute** using "the procedure of Göpfert et al. (2022) and the dataset of Balog et al. (2021) … linear, objective rather than non-linear and/or subjective CAVs". So ELM *consumes* a closed CAV vocabulary; it does not generate directions from arbitrary text.

### B. Concept / attribute as a PREFERENCE signal (the "our side" — the real competition)

- **`gopfert2022soft` — Göpfert, Haig, Chow, Hsu, Vendrov, Lu, Ramachandran, Pham, Ghavamzadeh, Boutilier. "Discovering Personalized Semantics for Soft Attributes in Recommender Systems using Concept Activation Vectors." WWW 2022 (arXiv:2202.02830, v3 Jun 2023); journal version ACM **TORS** 2024, doi 10.1145/3658675.** [A] + [V via ELM's description of the procedure]
  CAVs "translate between latent item representations learned by a **collaborative filtering** model and the soft attributes users adopt to describe items and preferences"; distinguishes objective vs subjective attributes and per-user senses; demonstrated to "improve recommendations through **interactive item critiquing**" [A].
  **Why it is not a kill shot:** a CAV is *by construction* a probe-trained direction — you need a labelled/scored exemplar set per concept (here: MovieLens tag-genome attribute scores and the Balog et al. 2021 attribute-ranking dataset [V, quoted in ELM]). The attribute vocabulary is therefore **closed**; there is no text encoder and no path to an unseen phrase. **This is exactly the gap our adapter fills.** *(Caveat RESOLVED 2026-07-28: full text obtained from `arxiv.org/pdf/2202.02830v3` (extended = TORS version). Quotes in the VERIFIED section at the end of this file. ACM full-text remains 403 — cite the TORS doi, read the arXiv v3.)*

- **`biyik2023soft` — Biyik, Yao, Chow, Haig, Hsu, Ghavamzadeh, Boutilier. "Preference Elicitation with Soft Attributes in Interactive Recommendation." arXiv:2311.02085 (Oct 2023), cs.IR.** [A]
  Uses "concept activation vectors for soft attribute semantics"; "query users using **both items and soft attributes** to update the recommender system's belief about their preferences"; synthetic + real datasets [A]. This is the closest *functional* analogue — mixed item/attribute elicitation over a belief. **VERIFIED 2026-07-28 from `arxiv.org/pdf/2311.02085v1` [V]:** it inherits the CAV vocabulary closure from Göpfert et al. (**164 tags**, §6), fits each CAV by **per-tag regularized logistic regression** (§2.3, Eq. 1), **trains the CF model and the CAVs separately** (§2.3), has **no text-encoder / zero-shot path**, and reports **NDCG@5 against a simulated ground-truth utility on 16 test users** — no held-out full-catalogue accuracy. Full quotes in the VERIFIED section at the end of this file. **The differentiator holds against this paper.**

- **Critiquing-VAE line** (all keyphrase-vocabulary-bounded):
  - `wu2019deepcritiquing` — Wu, Luo, Sanner, Soh. "Deep Language-based Critiquing for Recommender Systems." RecSys 2019 [S]. Predicts personalized keyphrases and embeds language-based feedback in the latent space to modulate re-recommendation.
  - `luo2020llc` — Latent Linear Critiquing (WWW 2020) [S].
  - **CE-VAE** — zeroes the latent coordinate corresponding to the critiqued keyphrase [S].
  - **BK-VAE** — Luo et al., SIGIR 2021 — **Bayesian** update of the user embedding from keyphrase critiques [S]; **DCE-VAE** adds a keyphrase tree [S].
  - `antognini2021mmvae` — M&Ms-VAE, RecSys 2021 — "Fast Multi-Step Critiquing for VAE-based Recommender Systems", arXiv:2105.00774; user representable from **keyphrases alone**, but via a *second* encoder over a fixed keyphrase vocabulary [S].
  **Common structural limit:** the keyphrase set is a fixed vocabulary extracted from reviews, and each keyphrase gets its own latent coordinate / encoder input. An unseen phrase has no slot.
  - **Items Proxy Bridging** (arXiv:2509.26107) — frictionless critiquing in KG recommenders [S], still schema-bound (KG entities).

- **Natural-language user profiles as the preference representation** (no CF latent involved):
  - `balog2019scrutable` — Balog, Radlinski, Arakelyan. "Transparent, Scrutable and Explainable User Models for Personalized Recommendation." **SIGIR 2019** — user model = set of (tag, weight) over a curated tag vocabulary, scrutable and editable [S].
  - `radlinski2022nlprofiles` — Radlinski, Balog, Diaz, Dixon, Wedin. "On Natural Language User Profiles for Transparent and Scrutable Recommendation." **SIGIR 2022**, arXiv:2205.09403 — argues for NL *instead of* latent user representations [S]. Position paper; explicitly the alternative to what we do (we keep the latent and make text a *port* into it).
  - `mysore2023lace` — Mysore, Jasim, McCallum, Zamani. "Editable User Profiles for Controllable Text Recommendation" (LACE). **SIGIR 2023**, arXiv:2304.04250 — user = small set of human-readable concepts **retrieved from a large concept inventory**, with personalized concept values; users edit the profile and recommendations change; interactive user study [S]. **Closest in spirit on the "concepts as an editable preference state" axis** — but it is *text/document* recommendation (no collaborative latent), and the concepts come from a fixed retrieved inventory, not an arbitrary phrase mapped by an adapter.
  - `langptune2025` — "LangPTune: Optimizing Language-based User Profiles for Recommendation." CIKM 2025 [S].
  - **BLUE** — Tan, Zhai, Zhu, Jiang, Hammad, arXiv:2605.06981 (May 2026) [V]. RL-optimises an LLM-written textual user profile against a **frozen embedding model** (Qwen3-Embedding-0.6B) using a bidirectional **InfoNCE** reward + a text-space next-item reward (GRPO). **Verified: the frozen embedder is a reward provider, not an adapter; there is no projection of text into a CF latent.** Reports NDCG@10/Recall@10 0.173/0.306 source, 0.329/0.516 on Books target [V]. Not a threat, but it is the 2026 state of "text profile ↔ frozen embedding space" and should be cited.

### C. The mechanism's ancestry (must cite, must NOT claim)

- `gantner2010attribute` — Gantner, Drumond, Freudenthaler, Rendle, Schmidt-Thieme. "Learning Attribute-to-Feature Mappings for Cold-Start Recommendations." **ICDM 2010** [S; already in INDEX].
  Maps **user or item attributes → the latent features of an already-trained MF**, so a frozen MF can serve cold entities; the linear variant is BPR-MF + attribute-to-feature mapping. **This is the 15-year-old ancestor of our W.** It differs on: fixed categorical attribute schema (genre, gender, occupation), attributes describe an *entity* (they *are* the cold user's profile), and the output is a full factor estimate — not a composable, signed, value-carrying token in a set-encoder belief alongside item tokens.
- `vasile2016metaprod2vec` — metaProd2vec, RecSys 2016 — metadata treated as **virtual items** in the embedding model [S]. The "concept as a token in item space" precedent, but training-time and closed-vocabulary.
- Kim et al., "Interpretability Beyond Feature Attribution: TCAV." **ICML 2018**, arXiv:1711.11279 — CAV = normal to a hyperplane separating concept-positive from concept-negative activations; **requires a probe set per concept** [S/definitional].
- **The label-free-CBM analogue (our strongest framing precedent, from vision):** Oikarinen, Das, Nguyen, Weng. "Label-Free Concept Bottleneck Models." **ICLR 2023**, arXiv:2304.06129 — converts any network into a CBM **without labelled concept data** by using CLIP-Dissect to project into a concept space, with GPT-3-generated concept sets [S]. Also "Explain via Any Concept: CBM with Open Vocabulary Concepts", ECCV 2024 [S].
  **This is the move we are making, one domain over.** We should cite it as the methodological analogue and state plainly: *label-free/open-vocabulary concept directions are established in vision; we are the first we are aware of to do it in a collaborative-filtering latent and, unlike CBMs, to use the direction as a preference INPUT rather than an explanation.*

### D. Evidence bearing on ridge vs. contrastive alignment

- **Wang et al. (SIGIR '26), "Rethinking Semantic–Collaborative Integration: Why Alignment Is Not Enough", arXiv:2604.22195** [V].
  Attacks the "global low-complexity alignment hypothesis". Evidence: (E1) an independent LightGCN and a semantic baseline share only **6.37% top-list overlap** on Movies with >55% unique hits; (E2) **alignment probe** mapping semantic→collaborative embeddings under a **frozen encoder**: linear map R² = 0.171 train / **0.166 test**, MLP-3 **R² = −0.495 test** — "semantics are insufficient to reconstruct unseen collaborative signals"; (E3) a trivial Norm-Concat-Norm **fusion** beats alignment methods (RLMRec, CARec) by **+41.7% Recall@20** on Movies. No explicit regression-vs-contrastive comparison (their probe uses MSE, their fusion uses InfoNCE) [V].
  **Two-edged for us.** *Supports:* higher-capacity maps overfit the fitting manifold and fail off it — the same shape as our ridge(38) ≪ InfoNCE(206). *Threatens:* it says text can only weakly reconstruct collaborative geometry (R²≈0.17), so a reviewer will ask why our W is good enough. **Our answer must be that we never claim reconstruction — we claim a usable RANK-CORRECT direction**, and our evidence is behavioural (paraphrase overlap@10 = 70–80%; text-vs-genome-centroid overlap@10 = 50% vs 0.2% random). We should pre-empt this by reporting our own R² and being explicit that low R² with high rank agreement is the expected and sufficient regime.
- **Blair, "Embeddings for Preferences, Not Semantics", arXiv:2605.08360 (2026)** [S]. Off-the-shelf text embeddings capture preference only through a *correlation* between semantic and preferential similarity, and fail where that correlation breaks; the authors fine-tune an ST5-XL encoder for preferential similarity. **This is a live objection to our frozen-SBERT choice** and should be cited and answered (our adapter is precisely the correction term that converts semantic similarity into behavioural geometry — and it is fitted, not assumed).
- No paper found reporting that **contrastive** text–CF alignment extrapolates worse than **regression** off the item manifold. Our result appears to be an **unclaimed empirical observation**; frame it as such (a mechanism note with a control), not as a headline contribution.

---

## What is NOVEL vs pre-empted

**PRE-EMPTED — cite, do not claim:**
1. Learning a map from content/attributes into a (frozen) CF latent — `gantner2010attribute` (2010), CB2CF (RecSys 2019), DropoutNet (NeurIPS 2017), Heater/MetaEmb/MWUF/ALDI line.
2. Concepts/attributes as **directions** in a CF latent, used for critiquing and elicitation — Göpfert et al. WWW'22/TORS'24; Biyik et al. 2023; TCAV (ICML'18) upstream.
3. Text as an item representation, including in frozen-CF/frozen-LLM sandwiches — UniSRec, VQ-Rec, Recformer, MoRec, TIGER, A-LLMRec, LLaRA, E4SRec, EasyRec.
4. Keyphrase/critique feedback modifying a VAE user latent — Deep Critiquing (RecSys'19), LLC (WWW'20), BK-VAE (SIGIR'21), M&Ms-VAE (RecSys'21), CE-VAE.
5. Editable/scrutable concept or NL user profiles as the preference state — Balog SIGIR'19, Radlinski SIGIR'22, LACE SIGIR'23, LangPTune CIKM'25.
6. Obtaining concept directions **without per-concept labels** via a text encoder — **established in vision** (Label-Free CBM, ICLR'23; Open-Vocabulary CBM, ECCV'24). Do not claim the idea; claim the transfer + the use.

**NOVEL / defensible (we are aware of no system that does this):**
- The **conjunction**: *one* globally-fitted linear adapter from a frozen general-purpose sentence encoder into a **frozen** CF latent, such that an **arbitrary, never-seen, out-of-catalog phrase** becomes a **signed, value-carrying preference token** that composes in the *same belief* as item tokens and curated concepts, with **zero per-concept supervision and zero retraining**.
- The **open-vocabulary removal of the CAV probe set** in a recommender latent — i.e. Göpfert-style attribute directions *without* the attribute-labelled dataset.
- The **grounding evidence** as a validation protocol: text-fold vs genome-centroid-fold agreement (50% overlap@10 vs 0.2% random, ~250×) is, as far as we found, an unused way to certify that a text-derived direction lands where the behavioural concept actually lives. ⚠ *2026-07-28: note that the "centroid of items as the attribute's embedding-space representation" construction is itself Balog et al. 2021 §5.2.2 (CB). Our use is as a **validation reference**, theirs as a **method** — say so explicitly, and cite them.*
- The **ridge-over-InfoNCE off-manifold observation** — unclaimed in the literature, and independently rhymed with by arXiv:2604.22195's linear-vs-MLP probe.

**Sentences to NOT write:**
- ✗ "We are the first to map text into a collaborative latent space." — Gantner 2010, CB2CF 2019. Dead on arrival.
- ✗ "We are the first to represent concepts as directions in a recommender's latent space." — Göpfert 2022.
- ✗ "No prior work uses text as a preference signal." — Balog 2019, Radlinski 2022, LACE 2023, the whole critiquing line.
- ✗ "There are none." — always "we are aware of none".

---

## Verdict

**(b) Partially pre-empted; narrowing required — but the narrowed claim is strong and survives.**
The mechanism is old and the concepts-as-directions framing is Boutilier-group prior art. What is unoccupied is the *open-vocabulary, zero-per-concept-supervision, inference-time* version of it used as a *preference input to a belief that also holds item tokens*.

**Most threatening paper:** Göpfert et al., WWW 2022 / TORS 2024 (arXiv:2202.02830), with Biyik et al. 2023 (arXiv:2311.02085) as its elicitation-facing sequel. Same latent, same "attribute = direction", same critiquing/elicitation use, same MovieLens domain, frozen CF. **Not a kill shot** because the direction is a probe-trained CAV over a closed attribute vocabulary — there is no text encoder and no route to an unseen phrase. **VERIFIED FROM PRIMARY TEXT 2026-07-28 [V] — this differentiator survives against both papers.** The residual threat moved to **Balog et al. 2021 §5.2–5.3** (label-free centroid/pseudo-label directions). See the VERIFIED section at the end of this file.

**Exact narrowed claim wording — ⚠ SUPERSEDED 2026-07-28; use the replacement in the VERIFIED section at the end of this file (the "no per-concept supervision" phrasing below is unsafe against Balog CB/WWD):**

> We map an **arbitrary, out-of-catalog free-text phrase** into a **frozen** collaborative latent via a **single globally-fitted linear adapter** from a frozen general-purpose sentence encoder, and use the resulting direction as a **signed, value-carrying preference token** that is folded into the user belief **identically to an item or a curated concept** — with **no per-concept supervision, no concept inventory, and no retraining at inference time**. Prior work either maps text into item representations (UniSRec, VQ-Rec, CB2CF, A-LLMRec), or treats concepts as latent directions for critiquing and elicitation but obtains each direction from a **per-concept labelled probe set over a closed attribute vocabulary** (concept-activation vectors: Göpfert et al. 2022; Biyik et al. 2023). We are aware of no system that combines the open vocabulary of the former with the preference-input role of the latter.

**Top 3 threats, in order:**
1. **Göpfert et al. WWW'22/TORS'24 + Biyik et al. 2023** — nearest neighbour on every axis except open vocabulary. Differentiator unverified from primary text.
2. **Wang et al. SIGIR'26 (2604.22195)** — not a novelty threat but a *validity* threat: says semantic→collaborative mapping barely generalises (linear R² 0.166 on held-out items) and that fusion beats alignment. We must report our own R² and argue rank-usability, not reconstruction.
3. **Label-Free CBM (ICLR'23) + Open-Vocabulary CBM (ECCV'24)** — pre-empts the *idea* of text-derived, label-free concept directions. Cheap to defuse by citing them as the cross-domain analogue, expensive if a reviewer finds them first.

---

## Open questions (UNCHECKED / must resolve)

- ~~**[BLOCKING]** Verify from primary text…~~ **RESOLVED 2026-07-28 [V]** — (a)(b)(c)(d) all confirmed for both papers; see the VERIFIED section at the end of this file. **New open item instead:** decide how we position against **Balog CB/WWD**, the label-free-but-per-attribute route we had missed, and whether to run CB as an explicit baseline on our ruler.
- Does anyone in the **cold-start user** literature fold a *linear-mapped attribute vector* into a VAE-style fold-in (rather than replacing the user factor)? Gantner's mapping produces a full factor; ours is an additive token. Unchecked.
- ~~**Balog et al. 2021** attribute-ranking dataset — identify it exactly~~ **RESOLVED 2026-07-28 [V]:** SIGIR 2021, arXiv:2105.09179, author PDF at `krisztianbalog.com/files/sigir2021-softattr.pdf`, data at `github.com/google-research-datasets/soft-attributes`. **60** soft attributes sampled from 173 extracted from the **CCPE-M** dialogue corpus, **100** raters, **249,863** pairwise preferences (52,352 ties) over the 300 most-popular ML-20M movies; plus a second **82**-attribute MovieLens tag collection on 238 movies.
- Is there any 2025–26 industrial paper (Spotify/Netflix/Pinterest) doing "free-text prompt → user-space vector → recommend" that has not surfaced in academic search? Searched generically, not found; not exhaustively checked.
- Our ridge-vs-InfoNCE finding has no published counterpart. Should we run the ablation the literature would want — ridge vs InfoNCE vs orthogonal Procrustes vs CCA — so the comparison is complete rather than a two-point claim? (Procrustes/CCA are the standard alternatives in the embedding-alignment literature.)

## Citation corrections (binding)

- The soft-attribute CAV paper is **Göpfert et al.** (first author Christina Göpfert), *not* Christakopoulou. WWW 2022 (arXiv:2202.02830, v3 2023); journal version **ACM TORS 2024**, doi 10.1145/3658675. Cite the TORS version.
- **Biyik et al. 2023** = arXiv:2311.02085, cs.IR, Oct 2023 — full author list Biyik, Yao, Chow, Haig, Hsu, Ghavamzadeh, Boutilier. No peer-reviewed venue found as of Jul 2026; cite as arXiv preprint.
- **ELM** = Tennenholtz et al., "Demystifying Embedding Spaces using Large Language Models", arXiv:2310.04475 (Google Research, 2023). No venue verified here — do **not** write "ICLR 2024" without checking.
- **A-LLMRec** = Kim et al., KDD 2024 (arXiv:2404.11343). **RLMRec** = Ren et al., WWW 2024 (arXiv:2310.15950). **UniSRec** = Hou et al., KDD 2022 (arXiv:2206.05941). **CB2CF** = Barkan, Koenigstein, Yogev, Katz, RecSys 2019.
- **LACE** = Mysore, Jasim, McCallum, Zamani, SIGIR 2023 (arXiv:2304.04250) — title on the ACM page is "Editable User Profiles for Controllable Text Recommendation**s**"; the arXiv title is singular. Match the ACM version.
- **Label-Free CBM** = Oikarinen, Das, Nguyen, Weng, ICLR 2023, arXiv:2304.06129 (not 2023's other CBM papers).

---

# VERIFIED FROM PRIMARY TEXT (2026-07-28)

**Everything in this section is [V] — quoted from the papers' own PDFs, text-extracted with pypdf.** It **supersedes** the [A]/inferred claims about Göpfert, Bıyık and Balog in the sections above wherever they conflict; conflicts are flagged as ⚠ CORRECTION.

**Fetch record (exactly what worked):**
- Göpfert et al., TORS 2024 — **ACM `dl.acm.org/doi/full/10.1145/3658675` → HTTP 403** (both HTML and PDF). **`arxiv.org/pdf/2202.02830v3` succeeded** (35 pp; this is the *extended* version: "This is an extended version of a paper that appeared at WWW-22"). arXiv abs page and ar5iv gave abstract only.
- Bıyık et al. — **`arxiv.org/pdf/2311.02085v1` succeeded** (22 pp). abs page abstract-only.
- Balog et al., SIGIR 2021 — **ACM PDF 403**; **author copy `krisztianbalog.com/files/sigir2021-softattr.pdf` succeeded** (10 pp). arXiv mirror also exists: **arXiv:2105.09179**. Dataset README at `github.com/google-research-datasets/soft-attributes` also fetched.

## ⚠ VERDICT ON OUR DIFFERENTIATOR: **NEEDS NARROWING** (not collapse)

- **Against Göpfert and Bıyık the differentiator SURVIVES intact and is now [V].** Both obtain a direction **per tag, by fitting a supervised discriminative model on labelled item exemplars for that tag**. Neither contains a text encoder, word vector, LLM embedding, or any zero-shot route. Vocabulary is enumerated and closed (164 MovieLens tags; 60/36 SoftAttributes attributes). CF model is frozen and trained without the tag data.
- **But Balog et al. 2021 §5.2–5.3 DOES contain a label-free, phrase-driven route to a direction in a frozen CF item-embedding space**, and our file above wrongly lumped it in with "per-concept supervision". The **Centroid-Based (CB)** method takes the *attribute phrase as a BM25 query over an Amazon review corpus*, then uses **the centroid of the top-k retrieved items' MF embeddings as the attribute's representation in embedding space** — **no attribute labels at all**. **WWD** extends this to a logistic regression on pseudo-labels from the same retrieval. So **"a direction for an attribute with no labelled data, in a frozen CF latent" is NOT unclaimed** — it is Balog's own unsupervised/weakly-supervised **baseline**, and it is our nearest prior art on the label-free axis, not Göpfert.
- **What still survives, and must now carry the claim:** (i) our map is **one global amortised adapter** — a single matrix multiply per phrase, with **no per-concept retrieval, no per-concept fit, and no requirement that the phrase occur in any corpus**; Balog CB/WWD run a **per-attribute retrieval + per-attribute model fit** and are hard-bounded by review coverage; (ii) the direction is used as a **signed, value-carrying user-side preference token folded into a belief that also holds item tokens** — Balog uses it only to **rank items by attribute degree** (a measurement paper), Göpfert to apply a **critique**, Bıyık to define a **query**; (iii) our evaluation ruler is **held-out full-catalogue NDCG@10 on 10,000 real users**, which none of the three reports.

**Replacement narrowed wording (use this in place of the wording at line ~119):**

> We map an **arbitrary, out-of-catalog free-text phrase** into a **frozen** collaborative latent with a **single globally-fitted linear adapter** from a frozen general-purpose sentence encoder — **amortised across all phrases**, requiring **no per-concept probe set, no per-concept retrieval or model fit at query time, and no corpus evidence for the phrase** — and use the resulting direction as a **signed, value-carrying preference token folded into the user belief identically to an item token**. Prior work obtains such directions **per attribute**: either from **labelled exemplars** for that attribute (concept activation vectors — logistic regression / RankNet / LambdaRank on user tag data or rater comparisons: Göpfert et al. 2024; Bıyık et al. 2023; supervised weighted dimensions: Balog et al. 2021), or **label-free but per-attribute and retrieval-bound**, by running the attribute phrase as a text query over a review corpus and taking the centroid (or a pseudo-label regression) over the retrieved items' embeddings (Balog et al. 2021, §5.2–5.3). We are aware of no system that amortises this into one global text→latent map and uses the result as a user-side preference token in a belief shared with item tokens.

**Sentences to NOT write (additions):**
- ✗ "No prior work derives an attribute direction in a CF latent without labelled data." — **Balog et al. 2021 CB/WWD do exactly that** (§5.2, §5.3).
- ✗ "CAV approaches require the attribute vocabulary to be fixed at recommender-training time." — Göpfert explicitly claims the opposite as an *advantage* (see Q2 quote below); the closure is in the **CAV's need for a labelled source per attribute**, not in the RS.

## Q1 — How is each CAV / attribute direction obtained?

**Per attribute, by fitting a supervised discriminative model on labelled item exemplars for that attribute. Verified in all three papers.**

**Göpfert, §3.2 "Linear Attributes" (Binary Logistic Regression):**
> "We train a logistic regressor 𝜙_g to predict whether an item i 'satisfies' the soft attribute corresponding to tag g. Specifically, P(g(i);𝜙_g) = 𝜎(𝜙_g^⊤𝜙_I(i)) is the predicted probability that i satisfies g, and is trained using (regularized) logistic loss (and labels y ∈ {+1,−1})… **Once trained, the regressor 𝜙_g obtained serves as our CAV.**"

Two further per-attribute variants: **RankNet** (§3.2, per-user pairwise loss) and **LambdaRank**; §4 adds an **EM** variant that clusters raters into per-attribute "senses". §3.1 states the probe-set requirement by analogy to TCAV:
> "Just as in the image setting, where some small set of positive example images (with stripes) and negative examples (non-striped) is used to attempt to identify a CAV, **we use a small set of positive (tagged) and negative (untagged) items for the same purpose**"

Exemplar construction (§3.4, MovieLens-20M): "Positive examples for tag g are user-item pairs to which g has been applied; negatives are those tagged by that user, but not with g".

**Bıyık, §2.3 "Concept Activation Vectors in RSs"** — identical machinery, same Eq. (1):
> "For a tag g ∈ T, we construct a training set D_g in which positive instances (y = +1) are items in T_g and negatives (y = −1) are those in T̄_g. We then learn the CAV 𝜙_g for g by learning a (regularized) logistic regressor"
> "The induced CAV 𝜙_g is the normal to the separating hyperplane of this classifier, and offers a directional semantics for (the attribute corresponding to) tag g in the item embedding space."

**Balog, §5.4 (fully supervised SWD):** "uses a linear ranking support vector machine… **this model represents a soft attribute by a direction in the embedding space (w)**, albeit trained using explicit pairwise preferences between items rather than based on terms extracted from reviews." (§5.3 WWD: "the model parameters w_a that are computed by LR reflect the importance (weight) of each dimension in the item embeddings in predicting the soft attribute.")

⚠ CORRECTION to line ~14 above: Göpfert's CAVs are trained on **user-applied MovieLens-20M tags** (and, separately in §5.3, on the SoftAttributes rater comparisons) — **not** on the "tag genome". Do not write "tag-genome".

## Q2 — Closed or open vocabulary? How many, from where?

**Closed and explicitly enumerated in every experiment.**

**Göpfert, §3.4 (MovieLens-20M):**
> "Given the sparseness of tags—of 30745 unique tags, more than 28000 are applied to fewer than 10 unique movies—to ensure that the analyzed tags are relevant to sufficiently many user-item interactions, **we train CAVs only for those tags that are among the 250 most frequently applied both by unique users and to unique items, which results in 164 tags.**"

**Göpfert, §5.1 (SoftAttributes):** "The resulting SoftAttributes data set consists of approximately 250K pairwise comparisons of movies over these **60 soft attributes**"; "**Of the 60 attributes used in the SoftAttributes data set, only 36 are found (frequently) in the MovieLens-20M data set.** When evaluating CAVs trained using MovieLens tags, we do so only on these 36 in-common attributes." — and footnote 16 **lists all 36 by name** ("animated, artsy, believable, big budget, bizarre, boring, cheesy, …, terrifying, and sappy"). A hand-enumerable closed set.

**Bıyık, §6 (MovieLens 20M):**
> "Due to item-tag sparsity, **we train CAVs only for the 164 most-frequently used tags (w.r.t. unique users, items).** Average CAV test quality is 0.727."
Synthetic domain: "Five of the 25 latent attributes are taggable (|T| = 5)." Query-space size is explicitly linear in |T| (§5.3): "The size of Q depends linearly on the number of tags |T| and combinatorially on the number of items |I|".

**IMPORTANT NUANCE we must not misstate — Göpfert §3 claims vocabulary-openness at the *recommender* level:**
> "(1) The recommender system model can be developed/trained/used **without a pre-commitment to a specific attribute vocabulary—new attributes can be added as needed without retraining the model itself.**"
and §1: "(2) The recommender system model can easily accommodate new attributes without retraining **should new sources of tags, keywords or phrases emerge from which to derive new soft attributes.**"

That second clause is the load-bearing one **for us**: a new attribute is free w.r.t. the recommender, but still requires **a new labelled source to derive it from**. Quote both sentences when we distinguish ourselves, or a reviewer will accuse us of misrepresenting them as vocabulary-closed.

Also §1 advantage (4): "One can learn soft attribute/tag semantics with **relatively small amounts of labelled data**, in the spirit of pre-training and few-shot learning." — few-shot, explicitly **not** zero-shot.

## Q3 — Any path to a direction with NO labelled data? **THE DECISIVE QUESTION**

**Göpfert: NO. Bıyık: NO. Balog: YES (and this is the correction to our file).**

- **Göpfert / Bıyık:** full-text search of both PDFs for `glove | word2vec | word embedding | sentence | BERT | text encoder | language model | zero-shot | unseen tag | out-of-vocabulary` returns **no method-bearing hit**. The only "natural language" mentions are motivational (intro) or in related work (Welch et al. on personalized word embeddings, cited only as a *prior for subjectivity analysis*: "Their methodology could be used for a more granular analysis of which words and phrases are most subjective"). The tag **string is never embedded**; only its labelled item exemplars are used. In Bıyık, an unseen tag is not merely unhandled — it is **not in the query space**.
- **Balog, §5 lead-in:** "**When no explicit training data over soft attributes exists**, we use implicit signals from item reviews, employing established models from entity retrieval. **Taking the centroid of top-ranked items, in the spirit of pseudo-relevance feedback, we represent[] soft attribute using an unsupervised approach (Section 5.2).**"
  **§5.2.2 Centroid-based Ranking:** "We obtain v_a by first ranking items with respect to the soft attribute using a term-based model over review text, and then **taking the centroid of the top-k ranked items' embeddings**" (top-k = 5, 25-d embeddings, §6.1). The retrieval itself (§5.2.1) is **BM25 with the soft attribute used as a search query** over an Amazon-review corpus.
  **§5.3 WWD:** "**In the absence of explicit training labels**, we again use term-based models for obtaining an initial ranking of items. Then, the top- and bottom-ranked items are taken as positive and negative (pseudo-)training examples… to learn a regression model."
  Performance (Table 5, Goodman–Kruskal γ): CB **0.404/0.471** (MovieLens G) and **0.087/0.101** (SoftAttr G′); WWD **0.539/0.517** and **0.194/0.200**; fully supervised SWD **0.485** (SoftAttr G′). **The label-free route works but is much weaker than supervision on the human-rated collection** — a useful point for us: it establishes the *baseline family* our adapter must beat, and it shows the label-free path was known and left underperforming.

## Q4 — Frozen or co-trained?

**Frozen in all three; and both Boutilier-group papers make the separation an explicit design claim.**

**Göpfert, §3:**
> "**Critically, we do not use the tag data when training the collaborative filtering model**—this is akin to work that builds attribute models on top of embeddings to address the cold-start problem, and **stands in contrast with approaches that jointly train models to predict both ratings and attributes**."

**Bıyık, §2.3:**
> "**We train the CF model and learn CAVs separately**, similar to methods that build attribute models on top of embeddings for cold-start, and in contrast to those that jointly train attribute models… After training a two-tower model, we use its item tower 𝜙_I to learn CAVs."

Backbones: Göpfert §3.4 — "We generate d = 50-dimensional user and item embeddings: we use **WALS** as our collaborative-filtering method in the case of linear CAVs, and train **two-tower DNNs** for non-linear CAVs." Bıyık §6 — "We generate d-dimensional (d = 50) user and item embeddings using **alternating least-squares (ALS)** and train CAVs on this latent space." Balog §5.1 — plain **matrix factorization**, 25-d (§6.1).

## Bıyık specifics: elicitation mechanics + accuracy reporting

**Query space (§3):** three types — **item queries** (choose from a slate), **attribute queries** `q = (S, g)`, and **item-plus-attribute (IpA)** queries.
> §3.1: "An attribute query q = (S, g) consists of a slate of items S and a tag g. The RS presents the slate to the user and asks her if she prefers items, relative to those in S, that are more/less—i.e., exhibit a greater degree of—g's attribute (e.g., 'Would you prefer movies that are more/less thought-provoking than those in S?'). The user responds to positively 𝜌 = +1 (i.e., more) or negatively 𝜌 = −1 (i.e., less)."
Response model is defined against a **hypothetical ideal item**, not a catalogue item: "we consider a model where u targets a hypothetical ideal item, 𝜙*_{I,u} ∈ argmax_{𝜙∈Γ} 𝜙_u^⊤𝜙, w.r.t. some mildly constrained space Γ ⊂ X unrelated to I", norm-capped at "‖𝜙‖₂ ≤ max_{i∈I}‖𝜙_I(i)‖₂".

**Belief update (§4):** Bayesian posterior over the user embedding.
> "The prior P_U(u) is the Gaussian user embedding learned by our two-tower model"; "P_U(u|H(K)) ∝ P_U(u) ∏_{k=1}^{K} P(𝜌(k)|q(k), 𝜙_u)   (7)"
Two approximations: **§4.1 parameterized posterior** sampled via Metropolis–Hastings/HMC (with a **batch vs iterative** variant — iterative wins, Fig. 5) and **§4.2 Gaussian posterior via Laplace approximation**. §6: "we use the **parameterized posterior** as the default belief state model". **§3.4 CAV uncertainty**: a CAV belief `P_g(𝜙_g|D_g)` is carried and the response likelihood is an **expectation over sampled CAVs** — "modeling CAV uncertainty offers significant gain in IG and RQ, with a **10-15% NDCG improvement** over PE methods that update their beliefs by treating the mean CAV as 'certain'."

**Query-selection criterion (§5):** acquisition functions **Random, Entropy, Mutual Information, EVOI**; EVOI wins ("EVOI is the most effective AF, outperforming MI and Entropy"). §5.2 **BPER** blends information gain with slate quality: "we define the BPER AF as **𝛾IG(q|H) + (1−𝛾)RQ(q|H)**", RQ being "RQ(q|H) := Σ_{i∈S} E_{𝜙_u∼P_U(u|H)}[𝜙_u^⊤𝜙_I(i)]"; §6 uses **γ = 0.5** and notes "a well-tuned **constant γ** suffices" (i.e., their adaptivity in the blend is flat). §5.3 query optimizers: **random search, continuous relaxation (first-/second-order), Thompson-sampling slate construction**.

**Full-catalogue top-N accuracy: NO — and this is quotable.**
> §6 Metrics: "(ii) **NDCG** is the normalized discounted cumulative gain between the **true top |S| items and the top |S| items estimated using the posterior**." plus "(i) **Cosine** … between the mean user posterior embedding and true user embedding" and "(iii) **Query NDCG**".
> §6 Experiment 4: "we create '**ground-truth' users**, each of whom has rated at least 50 movies… we treat them as the **ground truth utility**… We **sample 16 such test users** for PE… We use **slates of 5 movies**, set γ to 0.5".

So the headline number is **NDCG@5 against a simulated ground-truth utility (the user's own ALS embedding), on 16 users** — an *alignment* metric inside a simulator, **not** held-out top-N accuracy over the catalogue. **Nowhere in the paper is there a Recall@K / NDCG@K against real held-out interactions.** Same for Göpfert's critiquing experiments (§6: UAU on slates of k = 10 over T = 25 critique steps, with the user's *own* embedding forgotten and used as ground-truth utility).
**→ This is a strong, honest positioning line for Paper A:** the CAV elicitation line is evaluated in-simulator on tens of users; our claims are made on the canonical Liang ML-25M ruler (10,000 held-out users, full **and** tail NDCG@10). We should say this plainly and cite the metric definitions above.

## Balog specifics: is it a fixed human-annotated set, and how many?

**Yes — two fixed collections, both enumerated.**

**Collection 1 — MovieLens Attribute Collection (§3.3):** "The top **100** most frequent MovieLens tags are taken as soft attributes, excluding tags that are named entities…, refer to adult content, or contain coarse language."; after the α = 0.15 positive-example threshold, "there were no positive examples for 18 tags, **leaving us with 82 tags as soft attributes**." Items: "the **300 most popular movies** in the MovieLens-20M collection", reduced to "**238** out of the original 300 movies".

**Collection 2 — Soft Attributes Collection (§3.4.1) — the one everyone cites:**
> "We sample soft attributes from the **CCPE-M** dataset. It consists of over 500 English dialogs between a user and an assistant discussing movie preferences… **This yielded 173 unique soft attributes, of which we sample 60 for our evaluation** (the probability of selecting an attribute was proportional to its frequency of use)."
Table 3: "**Number of soft attributes 60**; Sets of movies rated **5,991**; Pairwise preferences **249,863 with 52,352 ties**". **100 raters** (dataset README: "A unique integer id for each rater from 1 to 100"; Göpfert §5.1: "across 100 raters"). Task (§3.4.3): raters place a sample set X of 10 movies into "less a than x / about the same a as x / more a than x" relative to an anchor.

⚠ CORRECTION: Göpfert footnote 15 says Balog's attributes "**are extracted from movie reviews in an Amazon reviews corpus** rather than from explicit tagging of items" — **that is wrong about the attribute source**; per Balog §3.4.1 the 60 attributes come from the **CCPE-M dialogue corpus**. The Amazon review corpus is what Balog uses for the *items'* text (§3.2, §5.2). Cite Balog directly, not Göpfert's footnote.

## Citations we should now add / results that bear on our beliefs

**Cite (new, from primary text):**
1. **Göpfert §6 critiquing update** — `𝜙_U(u) ← 𝜙_U(u) + 𝛼_t(g)·𝜙_g`, "where 𝜙_g is g's CAV, 𝛼_t(g) is a tag-specific step size at iteration t… and **the sign of 𝛼_t(g) reflects the direction ('more,' 'less') of the critique**." This is the **direct published ancestor of our signed additive concept fold** — our signed 4-band SEL fold must cite it as ancestry, not novelty. They also note the limitation we address: "we adopt a **simple heuristic**… More elaborate strategies for updating user embedding are possible, including the use of **Bayesian updates relative to a prior over the user embedding**."
2. **Göpfert §3 "we do not use the tag data when training the collaborative filtering model"** — the canonical citation for our frozen-tower firewall.
3. **Bıyık §5.2 BPER** — the published precedent for blending information gain against slate quality; relevant if we ever argue about elicitation-vs-recommendation trade-off.
4. **Balog §5.2.2 CB** — cite as the **label-free precedent** for a text-derived attribute direction in a CF latent, and as the natural weak baseline for our adapter.
5. **Balog Table 5** — CB 0.087–0.101 vs WWD 0.194–0.200 vs SWD 0.485 (G′, SoftAttr). Numbers we can point at for "label-free was known and weak".

**Contradicts / complicates something we believed:**
- ⚠ **Our "no per-concept supervision" is no longer a clean differentiator** — Balog CB/WWD are label-free. Narrow to *amortised, corpus-free, single global map* (see wording above).
- ⚠ **Göpfert explicitly advertises open-endedness of the attribute set** ("new attributes can be added as needed without retraining the model itself"). Our prose must attack the **per-attribute labelled source**, never claim they are architecturally vocabulary-locked.
- ⚠ **Both Boutilier papers report only simulator/alignment metrics on small user samples** (Bıyık: 16 users, NDCG@5 vs a simulated ideal; Göpfert: UAU/NDCG with forgotten-then-restored user embeddings). This *helps* us on rigour, but it also means **we cannot claim to "beat" them numerically** — there is no comparable number to beat. Any comparison must be a **re-implementation on our ruler**, or framed qualitatively.
- **Göpfert §4/§5 sense-subjectivity (EM over rater clusters, gamma 0.523 → 0.667 with EM) is a capability we do not have**: per-user attribute *senses*. A reviewer may ask why our global adapter ignores that a phrase means different things to different users. We should have an answer (our per-user signed value + belief conditioning is the substitute) rather than be surprised.
