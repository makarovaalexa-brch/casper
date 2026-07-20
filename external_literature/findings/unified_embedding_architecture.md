# Findings: Unified-Embedding Architecture

*Set encoders, RBMF, UNICORN, ConTS, one-embedding-serving-both-roles (recommender + policy action space).* (Paper A / C1 enabler.)
Consolidated & deduped from unified-embedding-novelty-map, NOVELTY_CLAIMS, LIT_IMPLICIT_VS_EXPLICIT. Bib-keys in `../INDEX.md`.

## What we concluded

- **The thesis:** ONE shared, continuous, text-derived (SBERT/two-tower) embedding space that is SIMULTANEOUSLY (i) the recommender — a FROZEN, policy-independent, PERMUTATION-INVARIANT set encoder over an arbitrary MIX of item-rating AND standalone concept/attribute reveals, partial-reveal-robust via masked training, scoring items as popularity-floor + personalization-residual — AND (ii) the ACTION SPACE of a continuous-action (Wolpertinger-style) elicitation policy spanning items + attributes + ARBITRARY OPEN concepts, LLM-verbalised. **No single published system occupies this intersection.**
- **C1 (safe formulation):** "One shared token space in which ITEMS, CONCEPTS and ARBITRARY CONTINUOUS DIRECTIONS are interchangeable VALUE-CARRYING inputs to a single set encoder, at RecVAE-class full-profile accuracy (full-profile NDCG@10 0.4852; teacher a0c 0.4961)." It is the ENABLER, not the headline — it is what makes a continuous, snappable action space possible AT ALL (a dense catalog-vector model cannot take a concept token or a continuous query).
- **The true, narrow architectural gap** (genuinely inexpressible in a one-hot input basis): (i) GRADED/SIGNED VALUES per entity, (ii) OUT-OF-CATALOG CONCEPTS, (iii) ARBITRARY CONTINUOUS DIRECTIONS. Claim ONLY those three.
- **Positive-only VAE recommenders' dislike-blindness is a real, DOCUMENTED architecture/interaction limitation** (Mult-VAE/RecVAE "can only express disagreement... no means of highlighting a desired feature"; critiquing VAEs bolt this on) — invisible on static top-N benchmarks, but exactly the cost that shows up when you move to active elicitation. See `answerability_and_channels.md`.

## Key external methods

- `liu2011wisdom` (RBMF), `fonarev2016rectangular` (RMVA), `shi2017local` (Local RBMF), `zhou2011functional` (Functional MF), `kweon2020deep` (DRE) — MF/learned embeddings as interview SEEDS via representativeness/max-volume, cold user folded into item-factor space. Established.
- `deng2021unified` (UNICORN) — shared space serving recommender AND a unified items+attributes DISCRETE graph-Q policy; no open concepts, not cold-start. Nearest anchor #1.
- `li2021seamlessly` (ConTS) — unifies items+attributes as BANDIT ARMS (an action space), Thompson, cold-start, 4-way feedback; FIXED categorical attributes, discrete argmax, NOT an encoder INPUT, no continuous token. Nearest anchor #2.
- `lei2020estimation` (EAR) — FM fold-in over items + confirmed attributes with a SEPARATE policy (item+attribute scoring in one space but policy-entangled).
- `ma2019eddi` (EDDI/Partial-VAE) — permutation-invariant set encoder over an arbitrary observed SUBSET + info-gain; right architecture, wrong domain (not recsys, no attributes). Nearest anchor #3.
- `liang2018variational` (Mult-VAE), plus EASE (Steck WWW'19), CDAE (Wu WSDM'16) — subset-robust fold-in but ITEM-ONLY, set-INDICATOR input (a 3-item interview is legal-but-OOD, not inexpressible).
- `reimers2019sentence` (SBERT) — the text-derived embedding substrate; `johnson2019billion` (FAISS) — NN grounding.
- `dulacarnold2015deep` (Wolpertinger) — continuous+NN action mechanism, but item selection, no recommender tie-in.
- **Not in references.bib — must add & cite-or-look-ignorant:** Biyik et al. 2023 (soft attributes via CAVs, items+concepts as directions, Bayesian, co-trained encoder); TaNP (WWW 2021) — permutation-invariant encoder mean-pooling (item,rating) pairs, THE ancestor of the set-input value-carrying cold-start recommender; MeLU (KDD 2019); M&Ms-VAE (RecSys 2021) — user from KEYPHRASES ALONE but TWO separate encoders + MoE, not one shared table, no continuous token; P5 (RecSys 2022) — items/attributes/reviews as text tokens in ONE shared space (strongest precedent); Latent Linear Critiquing (Luo WWW'20), CE-VAE/Deep Critiquing (Wu RecSys'19), Göpfert WWW'22; Anava WWW'15 (optimal design); Sun et al. WSDM'13 (multi-question trees).

## What is NOVEL vs pre-empted

**ESTABLISHED (must CITE, must NOT claim):**
- MF embedding as interview seeds / max-volume selection = RBMF/RMVA/Local-RBMF line.
- Decision-tree interview with latent-vector leaves = Functional MF / Golbandi / Sun.
- Shared space + a unified items+attributes DISCRETE policy action space = UNICORN, ConTS, EAR.
- Permutation-invariant set encoder over an arbitrary subset + info-gain = EDDI.
- Items+attributes as concepts/directions with mixed feedback but FIXED schema / Bayesian (not one frozen learned encoder + continuous RL) = Biyik 2023, Latent Linear Critiquing, CE-VAE, PEBOL.
- Set-input value-carrying cold-start recommender (item,rating pairs) = TaNP, MeLU. Text-token unified space = P5.

**NOVEL / defensible:**
- **The CONJUNCTION** — one frozen learned encoder that is BOTH the RecVAE-class recommender AND the continuous-action policy's open (item∪concept∪continuous) action space. No single system occupies this intersection.
- **A CONTINUOUS QUERY as an INPUT TOKEN** in the same entity space as items and concepts — the cleanest, strongest single leg (found nowhere).
- **Two FALSE statements to NOT write:** (1) "Mult-VAE structurally CANNOT represent a 3-answer interview" — WRONG, it's OOD not inexpressible. (2) "interview-native = token input" is TRIVIALLY satisfied by SASRec (token-based). Interview-native must mean VALUE-CARRYING, MIXED-TYPE, CONTINUOUS-CAPABLE tokens.
- Say **"we are aware of none,"** never "there are none."

## Open questions

- Has anyone built a SOTA-class recommender that natively consumes an arbitrary SET of (entity, value) tokens spanning items AND concepts AND continuous directions at RecVAE-class accuracy? (Set-Transformer recommenders? Any CRS with a unified token space at RecVAE-class accuracy?) If yes, C1 dies. UNCHECKED novelty.
- Does the unified encoder preserve item accuracy while ingesting attribute answers? (CRITICAL seam: recommender must INGEST attribute answers as token=(attribute_embedding, response), trained with reveal-subset augmentation mixing item-rating + attribute-pref tokens, GT attribute prefs derived from item ratings — otherwise "entities" silently collapse back to items.) Paper B result says YES in cold-start (items preserved: warm full-profile 0.380/0.185 vs 0.383/0.191).

## Citation corrections (binding)
- DRE = Kweon/Kang/Hwang/Yu, **WWW 2020** (NOT 2024; arXiv:2402.16327 is a reupload).
- There is NO "Liu et al. 2017": RBMF = Liu et al. RecSys 2011; Local-RBMF = Shi/Zhao/Shen TOIS 2017; RMVA = Fonarev et al. ICDM 2016.
- Functional MF = Zhou/Yang/Zha **SIGIR 2011** (not KDD). Golbandi tree = WSDM 2011. Rashid "Getting to Know You" = IUI 2002.
- PPDPP arXiv id = 2311.00262.
