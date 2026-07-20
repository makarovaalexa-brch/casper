# Findings: Paper A — Interview-Compatible Recommender Landscape

*The measuring instrument: a SOTA-full-profile recommender that folds an arbitrary-length, channel-agnostic
evidence set into an uncertainty-native, monotone belief.* (Paper A / pillar 1 of `docs/VISION.md`.)
Deep-research pass 2026-07-20. Reads on top of, and does NOT duplicate, `unified_embedding_architecture.md`
(the one-shared-space / action-space thesis) and `elicitation_and_belief_pool.md` (Golbandi/EDDI/Biyik/VoI).
Read those two first. This file is about the RECOMMENDER-as-instrument: SOTA, taxonomy, gap, baseline bank.

Our five requirements (R) and gates: **R1** SOTA full-profile (tie EASE/RecVAE); **R2** arbitrary-length
evidence set from 0 up (cold-start fold-in, no per-user retrain); **R3** channel-agnostic (items, concepts,
entities, free text through one interface via arbitrary embeddings); **R4** uncertainty-native (covariance
over the belief); **R5** monotone (a truthful answer never lowers ranking quality). Gates **G0** full-strength
preserved, **G1** posterior shrinks per question, **G2** cold-start per-question NDCG rises on full AND tail.

---

## c1 — SOTA for interactively-queried / fold-in / any-length-input recommenders (2023–2026)

Two distinct "SOTA" frontiers matter and must not be conflated:

**(A) Full-profile top-N SOTA on ML-20M-class implicit data (the R1 bar).** On the canonical Liang
strong-generalization split (10k val / 10k test held-out users, NDCG@100), the frontier has been *flat since
2019–2020* and is held by shallow linear/autoencoder models, not deep nets:
- `steck2019ease` **EASE** — closed-form item-item, ML-20M NDCG@100 **0.420**, Recall@20 0.391, Recall@50 0.521.
- `ren2020recvae` **RecVAE** — ML-20M NDCG@100 **0.442**, Recall@20 0.414, Recall@50 0.553 (top of this split).
- `liang2018variational` **Mult-VAE** — NDCG@100 **0.426**; Mult-DAE 0.419.
- `steck2020edlae` **EDLAE** (denoising/higher-order EASE) — ties RecVAE on ML-20M (NDCG@100 ~0.42–0.44 class).
- Classic floors on same split: SLIM 0.401, WMF/iALS 0.386, PureSVD lower. *(numbers as reported in the
  RecVAE and Mult-VAE papers; the split is fixed and widely reused.)*
- **Verdict:** no 2023–2026 method convincingly *beats* EASE/RecVAE on this exact ML-20M split. `dacrema2019progress`
  (weak-baseline critique) is the binding warning — most "SOTA" deep gains vanish here. CASPER's own set-encoder
  (full NDCG@10 ≈ 0.4946, ML-25M) and RecVAE-d512 ruler (0.4998) are on a *different* metric/split (NDCG@10,
  ML-25M) and are NOT directly comparable to these 0.42–0.44 NDCG@100 figures — snap to a shared harness before
  any "ties SOTA" claim.

**(B) Graph-filtering SOTA (the LightGCN benchmark family, Gowalla/Yelp2018/Amazon-book).** A separate,
newer frontier of *training-free / closed-form* filters that are the current speed+accuracy leaders on those
datasets — relevant because they are interactive-friendly (a new user's history is just a new signal to
filter, no retrain):
- `shen2021gfcf` **GF-CF** — closed-form graph signal filter; strong-baseline that matches/beats trainable GNNs.
- `choi2023bspm` **BSPM** — blurring–sharpening (score-SDE-inspired) filtering; SOTA Recall/NDCG on
  Gowalla/Yelp2018/Amazon-book, comparable runtime to fast baselines. *(ML-20M NDCG not reported.)*
- `park2024turbocf` **Turbo-CF** — matrix-decomposition-free polynomial graph filter; BSPM-class accuracy,
  fits on GPU in ~seconds. *(ML-20M NDCG not reported.)*
- **Relevance to us:** these are ITEM-ONLY, POINT-estimate, NO-uncertainty, NO-concept — they set the *accuracy
  ceiling to respect* but fail R3/R4/R5 outright. They matter as R1 sanity-baselines only if we re-run them on
  our split (they don't publish ML-20M/25M strong-gen NDCG@10).

**(C) Interactively-queried / elicitation-native SOTA (the actual Paper-A niche).** No system on this frontier
reports EASE/RecVAE-class full-profile accuracy AND arbitrary-channel input AND uncertainty AND monotonicity.
The strongest *concrete* systems each occupy a corner:
- `austin2024pebol` **PEBOL** (RecSys'24) — Bayesian-optimization NL preference elicitation; Beta posterior per
  item, LLM-extracted aspects, NLI soft scoring. Cold-start MRR@10 0.27 vs 0.17 baseline (+131% MAP@10),
  preferences 100% LLM-simulated. Uncertainty-native but PER-ITEM (no shared user covariance), not full-profile SOTA.
- `biyik2023soft` **Biyik soft-attributes** — multivariate-Gaussian belief over a user embedding, items+soft-
  attributes as directions, EVOI query selection. The nearest thing to R2+R3+R4 together — but it CO-TRAINS the
  encoder (not a frozen SOTA CF tower), posterior "generally not Gaussian," and is not benchmarked at RecVAE-class
  full-profile accuracy.
- `li2021seamlessly` **ConTS** — items+attributes as unified bandit arms, Thompson posterior, cold-start, 4-way
  feedback. Uncertainty-native + unified action space, but FIXED categorical schema (no arbitrary embedding /
  continuous / open concept), discrete argmax, not a full-profile encoder.
- `ma2019eddi` **EDDI / Partial-VAE** — permutation-invariant set encoder over an arbitrary observed subset +
  info-gain. Exactly R2's architecture, but non-recsys domain, no attribute channel, no full-profile CF SOTA.
- LLM-CRS line (`he2023large`, MACRS, τ-Rec 2026 agentic benchmark): strong zero-shot/cold-start on content, but
  documented **weak on collaborative signal** — motivates keeping a strong CF tower in the loop, not replacing it.

---

## c2 — Taxonomy of interview-compatible recommenders (scored against R1–R5, G0–G2)

| Family | Representatives | R1 SOTA-full | R2 any-length set | R3 channel-agnostic | R4 uncertainty | R5 monotone | Verdict for Paper A |
|---|---|---|---|---|---|---|---|
| (i) retrain / fine-tune-per-user MF (meta-learning) | MeLU, MAML-rec, `lin2021tanp` TaNP, CDRNP | mid (cold-start focus, not full-profile SOTA) | yes (support set) | weak (items+side feats, fixed) | TaNP/NP: yes (stochastic process) | no guarantee | Ancestor of value-carrying set-input cold-start rec; NP variants give uncertainty. Not full-profile SOTA, not channel-open. Cite as R2/R4 precedent. |
| (ii) fold-in linear / least-squares projection | `steck2019ease` EASE, SLIM, `shen2021gfcf` GF-CF, `choi2023bspm` BSPM, `park2024turbocf` Turbo-CF, PureSVD/iALS fold-in | **yes (R1 bar)** | yes (new history = new signal, closed-form fold) | **no** (item-only, one-hot basis) | **no** (point) | approx (adding a true positive rarely hurts, no proof) | The full-profile bar to tie (G0). Structurally cannot take graded/signed/concept/continuous tokens. Our R3 gap lives exactly here. |
| (iii) amortized inference encoders | `liang2018variational` Mult-VAE, `ren2020recvae` RecVAE, `steck2020edlae` EDLAE, `ma2019eddi` EDDI, Set-Transformer / deep-set recs, CASPER set encoder | **near-bar** (RecVAE = top of ML-20M split) | yes (set-indicator / partial-VAE / deep-set) | VAE: **no** (item-indicator input); EDDI/deep-set: **partially** (arbitrary token IF trained for it) | VAE: latent posterior exists but item-only & not used as belief; EDDI: yes | not guaranteed | The architecture CASPER lives in. Standard VAEs are item-only & dislike-blind; EDDI has the set-input shape but wrong domain. R3+R4+R5 are the open work. |
| (iv) Bayesian / Gaussian belief over a frozen item space | `biyik2023soft`, `li2021seamlessly` ConTS, `toroghi2023bcie` BCIE, `guo2010real` Gaussian/TrueSkill, bandit posteriors, `zhao2013interactive` ICF, BDECF (`wang2025bdecf`) | no (belief layer, not a full-profile SOTA tower) | yes (sequential Bayesian update) | Biyik: items+attrs as directions (best R3); ConTS: fixed schema | **yes (native)** | conjugate updates are Blackwell-monotone in expectation (Good 1967) — closest to R5 | The R4+R5 heart. Biyik is the central threat (R2+R3+R4). Wedge = EXACT linear-Gaussian conjugacy over a TRULY FROZEN RecVAE-class tower + channel-agnostic tokens. |
| (v) sequence models (implicit fold-in) | `kang2018sasrec` SASRec, `sun2019bert4rec` BERT4Rec, Transformers4Rec | strong on sequential next-item, NOT the ML-20M top-N split bar | yes (re-encode sequence at inference, no retrain) | token-based but ITEM-only & ORDER-sensitive (not a set; not value-carrying) | no (point) | no (a later token can reorder everything) | Interview ≠ ordered session. Token-input is trivially satisfied but NOT value-carrying/mixed-type — do not claim "interview-native = tokens." Baseline only. |
| (vi) LLM-as-recommender / CRS | `austin2024pebol` PEBOL, `li2023eliciting` GATE, `he2023large`, MACRS, RecLLM, τ-Rec | weak on collaborative signal (documented) | yes (NL context) | **yes** (free text native) — but no shared CF geometry | per-item Beta (PEBOL) or none | no | Best R3 (open text) but weakest R1. Our design keeps LLM for render/interpret only; the CF tower does the ranking. PEBOL is the load-bearing PE baseline. |

Key cross-cutting reads:
- **R1 and (R3+R4+R5) are anti-correlated across the literature.** The models that hit the full-profile bar
  (ii, and VAEs in iii) are item-only point estimators; the models with uncertainty + channels (iv, vi) do not
  report full-profile SOTA. No family occupies both corners — this is the structural shape of the gap.
- **Monotonicity (R5) is nowhere proven for a learned ranker.** Conjugate-Gaussian belief updates are
  Blackwell/Good-1967 monotone *in expected utility*, but a dot-product/NDCG ranker does not act optimally under
  the belief, so per-step NDCG monotonicity is NOT automatic (see `elicitation_and_belief_pool.md`). Our G1/G2
  empirical monotone-per-question result (Kalman belief pool) is the contribution, not the theorem.

---

## c3 — Does anything solve it all? (adversarial gap verdict)

**No single published system satisfies all of {R1 SOTA-full-profile ∧ R2 any-length set ∧ R3 channel-agnostic
via arbitrary embeddings ∧ R4 uncertainty-native ∧ R5 monotone}. The binding near-miss is Biyik et al. 2023,
which fails R1 (co-trained encoder, not a frozen RecVAE-class tower, not benchmarked at full-profile SOTA) and
gives only an approximate/non-Gaussian posterior; every other candidate fails R3 or R4 outright.**

Precise failure of each nearest miss:
- **Biyik 2023** — has R2+R3(items+soft-attributes as directions)+R4(Gaussian belief). **Fails R1**: the item
  space is co-trained with the belief model, not a certified SOTA-full-profile CF tower; accuracy is reported on
  their elicitation task, not on the EASE/RecVAE full-profile bar. R3 is attribute-directions, not *arbitrary
  embeddings / open concepts / continuous query tokens*. R5 approximate (sampled EVOI, non-Gaussian posterior).
- **ConTS** — R2+R4+unified action space. **Fails R3** (fixed categorical attribute schema, no arbitrary
  embedding, no continuous/open token) and **R1** (bandit over arms, not a full-profile encoder).
- **EDDI / Partial-VAE** — R2+R4(info-gain)+set-input. **Fails R1** (not recsys / no CF SOTA) and **R3** (no
  attribute/concept channel as demonstrated).
- **RecVAE / EASE** — **R1 yes**. **Fail R3** (item-only one-hot/indicator basis — cannot express a concept,
  a signed value, or a continuous direction) and **R4** (point estimate; RecVAE's latent posterior is item-only
  and not used as an actionable belief).
- **PEBOL** — R4(per-item Beta)+R3(LLM aspects)+cold-start-strong. **Fails R1** (no full-profile CF ranking) and
  R4 is per-item, not a shared user covariance; no monotonicity guarantee.
- **SASRec/BERT4Rec** — R2(re-encode). **Fail R3, R4, R5** (item-only, order-sensitive, point, non-monotone).

**Defensible novelty of Paper A (the instrument):** the CONJUNCTION — one *frozen, certified RecVAE-class*
recommender that ingests an arbitrary-length SET of value-carrying, mixed-type tokens (items ∪ concepts ∪
entities ∪ continuous directions) through one interface, carries an explicit covariance belief, and is
empirically monotone per truthful question (G1/G2). No single system occupies this intersection. Say **"we are
aware of none"**, never "there is none" (per the standing rule). The *architecturally inexpressible-elsewhere*
legs to claim narrowly: (i) graded/signed values per entity, (ii) out-of-catalog concepts, (iii) arbitrary
continuous direction tokens — all impossible in the item-indicator basis of the R1-bar models.

**Pre-empted — must cite, must NOT claim as first:** attribute-answers-as-directions + Bayesian fold (Biyik);
unified item+attribute belief/VoI (Biyik BPER, ConTS); permutation-invariant set-encoder + info-gain (EDDI);
value-carrying set-input cold-start rec (TaNP/MeLU); non-degradation-of-information principle (Good 1967 /
Blackwell); MF-embedding-as-interview-seeds (RBMF/RMVA/Functional-MF/Golbandi). See the two prior findings files.

---

## c4 — Baseline bank for Paper A (ranked, with why each earns a slot)

Tier 1 — classic floors (must include; `dacrema2019progress` demands them):

| Baseline | Type | Reported anchor | Code | Why the slot |
|---|---|---|---|---|
| Most-Popular | non-personalized floor | popb floor (our harness) | trivial | G2 must beat it cold; the honest "did we learn anything" line. |
| Item-kNN (cosine) | neighborhood | ML-20M NDCG@100 ~0.36 class | trivial | Cremonesi-era strong simple baseline; folds new users instantly. |
| PureSVD / iALS-WMF | linear MF fold-in | WMF ML-20M NDCG@100 **0.386** (reported) | Implicit lib | Cheap fold-in reference; the MF latent space RBMF/Functional-MF build on. |
| SLIM | sparse linear item-item | ML-20M NDCG@100 **0.401** (reported) | public | Direct EASE ancestor; sparse fold-in. |

Tier 2 — full-profile SOTA (the R1 / G0 bar — snap to these exactly before any claim):

| Baseline | Type | Reported anchor (ML-20M strong-gen split) | Code | Why the slot |
|---|---|---|---|---|
| **EASE** `steck2019ease` | closed-form item-item | NDCG@100 **0.420**, R@20 0.391, R@50 0.521 | public | The full-profile bar to tie; instant fold-in; item-only (shows R3 gap). |
| **RecVAE** `ren2020recvae` | VAE (amortized) | NDCG@100 **0.442**, R@20 0.414, R@50 0.553 | public | Top of the ML-20M split; our in-house d512 ruler (0.4998 NDCG@10 ML-25M). |
| **Mult-VAE / Mult-DAE** `liang2018variational` | VAE / DAE | NDCG@100 **0.426** / **0.419** | public | Canonical amortized-encoder anchor; set-indicator input = OOD-not-inexpressible demo. |
| EDLAE `steck2020edlae` | denoising higher-order EASE | ML-20M ≈ RecVAE class (reported tie) | public | Confirms the linear frontier is flat; strong closed-form fold-in. |
| GF-CF / BSPM / Turbo-CF `shen2021gfcf`/`choi2023bspm`/`park2024turbocf` | closed-form graph filter | SOTA on Gowalla/Yelp/Amazon (ML-20M NDCG **not reported**) | public | 2023–24 accuracy+speed leaders; item-only point est.; include IF re-run on our split. |
| SASRec / BERT4Rec `kang2018sasrec`/`sun2019bert4rec` | self-attentive sequence | strong sequential next-item (not this split) | public | Sequence fold-in baseline; shows token-input ≠ value-carrying set. Watch BERT4Rec replicability (`petrov2022replicability`). |

Tier 3 — closest elicitation-compatible matches (the real rivals for the instrument claim):

| Baseline | Type | Reported anchor | Code | Why the slot |
|---|---|---|---|---|
| **EDDI / Partial-VAE** `ma2019eddi` | set-encoder + info-gain | non-recsys (UCI/MNIST) | public | The R2 architecture; the myopic info-gain baseline our tree generalizes. |
| **Biyik soft-attributes** `biyik2023soft` | Gaussian belief + EVOI | their elicitation task (not full-profile SOTA) | partial | THE central threat: nearest R2+R3+R4 system; the co-trained-encoder / non-frozen distinction is our wedge. |
| **ConTS** `li2021seamlessly` | Thompson bandit, items+attrs arms | cold-start CRS metrics | public | Unified-space + uncertainty prior-art; fixed schema = R3 gap. |
| **BCIE** `toroghi2023bcie` | conjugate-Gaussian critique fold | frozen-factorization critique | — | Exact-conjugacy-over-frozen-space precedent; closest to our R4/R5 mechanism. |
| **PEBOL** `austin2024pebol` | Bayesian-opt NL-PE (LLM) | cold-start MRR@10 **0.27** vs 0.17 (LLM-sim users) | public | Load-bearing PE baseline; per-item Beta uncertainty; can't represent a novel direction. |
| Golbandi tree `golbandi2011adaptive` | RMSE decision-tree interview | Netflix RMSE | — | THE adaptive cold-start anchor; static-tree instrument comparator. |
| RBMF / Functional-MF `liu2011wisdom`/`zhou2011functional` | MF-embedding seeds / tree-in-latent | elicitation RMSE | — | Established interview-seed line; cite-and-differentiate, not a strong ranker. |

Ranking logic: G0 is decided against **EASE + RecVAE** (Tier 2, R1 bar). G1/G2 (the instrument's own claim) are
decided against **EDDI, Biyik, ConTS, PEBOL, Golbandi** (Tier 3) plus a random-question control. Tier 1 is the
floor every curve must clear. Any deep-net "SOTA" not snapping to Tier-2 numbers on OUR split is inadmissible
(`dacrema2019progress`).

---

## What is NOVEL vs pre-empted (instrument-specific)

**NOVEL / defensible (this file's delta over the two prior findings):**
- The **conjunction as an INSTRUMENT**: one frozen certified RecVAE-class tower that is simultaneously
  channel-agnostic (arbitrary-embedding set input) AND carries an actionable covariance belief AND is
  empirically monotone-per-question. The literature is bifurcated (R1-corner vs R3/R4/R5-corner); nobody bridges.
- **Continuous-direction and out-of-catalog-concept tokens** are *architecturally inexpressible* in every Tier-2
  R1-bar model (item-indicator basis) — the one leg found nowhere.
- **G1/G2 as an acceptance protocol** (posterior strictly shrinks + per-question NDCG rises on full AND tail):
  no elicitation-rec paper reports monotone-per-truthful-question NDCG for a frozen SOTA tower.

**PRE-EMPTED (cite, do not lead):** everything in c3's pre-empted list. Additionally: uncertainty-native
recommenders are now a crowded 2024–25 area (BDECF Bayesian ensembles, aleatoric-uncertainty rec, probabilistic
Gaussian embeddings) — "uncertainty-native recommender" alone is NOT novel; only the *conjunction with the frozen
SOTA tower + channel-agnostic fold + monotone-per-question* is.

## Open questions (for design; not answered here)

- Does any Set-Transformer / deep-set recommender hit RecVAE-class full-profile NDCG on ML-20M/25M while taking
  value-carrying mixed-type tokens? UNCHECKED — if yes, R1∧R2∧R3 is partly pre-empted. (Searches found no such
  reported system; absence-of-evidence, not proof.)
- Do the 2023–24 graph-filter leaders (GF-CF/BSPM/Turbo-CF) beat RecVAE on the ML-20M strong-gen split? They
  don't publish it — must be re-run on our harness to admit them to Tier 2.
- Is BERT4Rec's fold-in advantage real or a replicability artifact (`petrov2022replicability` found it under-tuned)?
- Confirm EASE/RecVAE ML-20M numbers against the primary tables in our own harness before printing them as "ties SOTA".

## Evidence relevant to design (facts, not a recommendation)

- The full-profile frontier on ML-20M has been flat since ~2019 and is held by SHALLOW linear/AE models
  (EASE/RecVAE/EDLAE); deep gains largely vanish under `dacrema2019progress`. A shallow frozen tower is a
  defensible R1 substrate.
- Conjugate-Gaussian / closed-form Kalman updates are the only mechanism with a *principled* monotonicity story
  (Good 1967 / Blackwell) — but only in expected utility, and only if the ranker acts optimally under the belief.
- Item-only bases (EASE, VAEs, graph filters, sequence models) are *structurally* unable to ingest concepts /
  signed values / continuous directions — R3 is an architecture property, not a tuning gap.
- LLM recommenders are documented weak on collaborative signal (`he2023large`) — keep the CF tower as the ranker,
  LLM for render/interpret only, consistent with VISION constraint.
- Neural-Process cold-start recs (TaNP) already deliver a *stochastic-process* uncertainty over a value-carrying
  set input — the R2+R4 shape exists, just not at full-profile SOTA nor channel-agnostic.

## Bib entries to add (do NOT edit references.bib here — hand to author)

- `steck2019ease` — Harald Steck. "Embarrassingly Shallow Autoencoders for Sparse Data." WWW 2019.
- `ren2020recvae` — Shenbin, Alekseev, Tutubalina, Malykh, Nikolenko. "RecVAE." WSDM 2020. (arXiv:1912.11160)
- `steck2020edlae` — Harald Steck. "Autoencoders that don't overfit towards the identity (EDLAE)." NeurIPS 2020.
- `shen2021gfcf` — Shen et al. "How Powerful is Graph Convolution for Recommendation? (GF-CF)." CIKM 2021.
- `choi2023bspm` — Choi, Hong, Park, Cho. "Blurring-Sharpening Process Models for CF (BSPM)." SIGIR 2023.
- `park2024turbocf` — Park et al. "Turbo-CF: Matrix Decomposition-Free Graph Filtering." SIGIR 2024. (arXiv:2404.14243)
- `kang2018sasrec` — Kang & McAuley. "Self-Attentive Sequential Recommendation (SASRec)." ICDM 2018.
- `sun2019bert4rec` — Sun et al. "BERT4Rec." CIKM 2019.
- `petrov2022replicability` — Petrov & Macdonald. "A Systematic Review and Replicability Study of BERT4Rec." RecSys 2022.
- `lin2021tanp` — Lin et al. "Task-adaptive Neural Process for User Cold-Start Recommendation." WWW 2021. (arXiv:2103.06137)
- `biyik2023soft` — Bıyık et al. "Preference Elicitation with Soft Attributes in Interactive Recommendation." arXiv:2311.02085, 2023.
- `toroghi2023bcie` — Toroghi & Sanner. "Bayesian Critique-Improve-Explain (BCIE)." SIGIR 2023.
- `austin2024pebol` — Austin et al. "Bayesian Optimization with LLM Acquisition for NL Preference Elicitation (PEBOL)." RecSys 2024. (arXiv:2405.00981) *(may already be `austin2024bayesian` in bib — reconcile key)*
- `wang2025bdecf` — "Epistemic Uncertainty-aware Recommendation via Bayesian Deep Ensemble Learning." arXiv:2504.10753, 2025.
