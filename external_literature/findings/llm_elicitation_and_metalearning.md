# LLM-era preference elicitation (2023–2026) + meta-learning cold start — Paper B baseline battery

*Deep research 2026-08-05. Question asked: for the Paper B baseline battery, what does the **LLM-era
preference-elicitation** family (PEBOL and successors, LLM CRSs that ASK, HCI/IUI elicitation, NL-PE
benchmarks) and the **meta-learning cold-start** family (MeLU/MAMO/TaNP/CDRNP + 2023–26 successors)
actually offer us, and which of them are baselines vs category errors?*

Verification levels: **[V]** primary full text extracted · **[A]** abstract/landing page only ·
**[S]** secondary (search summary / third party). Nothing is quoted unless it came from [V].

Read **with** `adaptive_elicitation_baselines.md` (the non-LLM elicitation family and the harness audit),
`simulator_circularity_conv_rec.md` (the firewall argument), `critiquing_vocabulary_is_closed.md`
(the closed-vocabulary count), and `text_to_cf_projection.md` (ELM, the reverse direction).

---

## 0. The headline — four sentences

1. **The LLM-era elicitation family is mostly GENERATORS, not SELECTORS — and the three exceptions all put
   the LLM OUTSIDE the selection loop.** Most 2023–2026 methods generate free-form questions by
   prompting/fine-tuning (GATE, Montazeralghaem, BED-LLM) or pick from a tiny closed strategy set (COPE,
   PPDPP, RSO). But **PEBOL** (Beta posteriors + TS/UCB), **OPEN** (EIG over a Bradley–Terry particle
   filter), and **CLARINET** (greedy rank-gain of the true item) each have a real acquisition rule that is
   *specifiable without an LLM* — the LLM only phrases or featurises. **That is the finding that turns this
   family from citation-only into one runnable ablation.**
2. **The measurement contract is the disqualifier, not the LLM.** BLEU/ROUGE against a reconstructed
   profile; mean predicted rating over 10 films; Recall@k against a known dialogue target; p(correct) or
   TIDA on a binary preference classifier; **MRR@10 over a 100-item catalogue** (PEBOL — see the correction
   below), retrieval success over 1,877 documents (CLARINET), 30 articles (OPEN). **Not one of them ranks a
   held-out consumption set over an 18k catalogue.** So the family contributes *citations and one ablation
   row*, never a number in our table.
   ⚠ **CORRECT THE CHAPTER: PEBOL's catalogue is 100 items, not ~1–5k** — and the paper says the cap exists
   only so the MonoLLM baseline fits in a GPT-3.5 context window. **Ours is 184× larger.**
3. **Bıyık et al. 2023 — the standing verification debt — is now closed, and it is weaker than we feared.**
   It is an **arXiv-only preprint** (no venue in 2.5 years), its attribute vocabulary is a **closed 164-tag
   ML-20M set**, its evaluation is **16 sampled simulated users**, and its "NDCG" is agreement with the
   simulator's own ground-truth embedding at |S|=5 — **there is no held-out ranking metric anywhere in it.**
4. **The meta-learning cold-start line is a literature of ANSWER CONSUMERS.** Exactly one paper in it
   (MeLU, KDD'19) selects evidence candidates; that selection is a **static global popularity×gradient-norm
   ranking**, validated only on **39 human subjects** — and in doing so MeLU independently **reproduced our
   answerability collapse** (≈25% of discriminative items rated vs ≈50% of popular ones) and moved on.
   Presenting MAMO / TaNP / CDRNP as elicitation baselines is a category error.

---

## 1. SCOPE A — LLM-era preference elicitation

### 1.1 PEBOL — Austin\*, Korikov\*, Toroghi, Sanner, RecSys 2024, arXiv:2405.00981v2, DOI 10.1145/3640457.3688142 **[V, full PDF]**
Indexed twice (`austin2024bayesian` / `austin2024pebol`) — **reconcile the duplicate bib-key.**
This pass verified the whole method from primary text and produced **two corrections that matter**.

- **Beliefs.** Independent Beta per item, `p(u) = ∏ᵢ p(uᵢ)`, `p(uᵢ) = Beta(αᵢ⁰,βᵢ⁰)`, uniform cold-start
  prior `(1,1)`, support [0,1] read as a normalised rating. Ranking point estimate `μᵢᵗ = E[uᵢ]`.
- **Update.** PEBOL-B: Beta-Bernoulli conjugacy, `αᵢᵗ = αᵢᵗ⁻¹ + rᵢᵗ`, `βᵢᵗ = βᵢᵗ⁻¹ + (1−rᵢᵗ)`.
  PEBOL-P: `rᵢᵗ` is replaced by an **NLI entailment probability** `wᵢᵗ ∈ [0,1]` — the binary answer is turned
  into a preference description ("patriotic" / "not patriotic"), then `facebook/bart-large-mnli` scores
  whether the item description entails it (entailment+contradiction logits temperature-scaled then
  softmaxed). **PEBOL-P beats PEBOL-B** — binary feedback *"discard[s] valuable information from the
  entailment probabilities."*
- **★ ACQUISITION IS TWO-STAGE, AND STAGE 1 IS LLM-FREE.** Stage 1 (**selection**) is a closed-form bandit
  rule over the Beta posteriors: **TS** (`ûᵢ ∼ p(uᵢ|·)`, argmax), **UCB** (argmax of the k-th percentile of
  the Beta), **ER** (argmax variance, explore-only), **Greedy** (argmax mean), Random. Stage 2 (**phrasing**)
  is where GPT-3.5-turbo-0613 lives: extract a ≤3-word aspect from the chosen item's description (excluding
  aspects already used), then phrase a yes/no query — *"Are you interested in movies with patriotic themes?"*
  **The LLM is NOT in the selection loop.** This is exactly the decomposition that makes an LLM-free
  acquisition ablation legitimate.
- **★★ CORRECTION — THE CATALOGUE IS 100 ITEMS, NOT 1–5k.** Verbatim [V]: *"we have to limit |I| to 100 for
  fair comparison to the MonoLLM baseline"* and *"After the filtering steps below for Yelp and MovieLens, we
  randomly sample 100 items to create x."* The limit is a pure **baseline artefact** — MonoLLM must fit every
  item description into a GPT-3.5 context window; for MovieLens they had to switch to the 16k variant just to
  fit 100. Sources: MovieLens-25M (top 10% by rating count, ≥20 tags; description = title + genres + 20
  most-common tags), Yelp (Philadelphia restaurants, ≥50 reviews, ≥5 categories), Recipe-MPR (the sampling
  sentence names only Yelp and ML, so Recipe-MPR's exact count is **one-sentence ambiguous** — treat as 100
  with a flag). **Our 18,359-item catalogue is 184× larger. Fix this number wherever the chapter states it.**
- **Oracle: 100% LLM-simulated** [V]: *"we simulate 100 users, each of which likes a single item i ∈ I. Each
  user is simulated by GPT-3.5-turbo-0613, which is given item description xᵢ and instructed to provide only
  'yes' or 'no' responses to a query qᵗ as if it was a user who likes item i."* ⚠ Note the **compounded**
  circularity: the simulated user reads the *same* description string that the NLI scorer entails against —
  the user and the measurement instrument share their text.
- **Metrics:** MRR@10 over 10 turns (*"equivalent to MAP@10 for the case of a single preferred item"*), noise
  levels ∈ {0, 0.25, 0.5} (fraction of answers flipped). **Correction to the headline:** the abstract's
  "0.27 vs 0.17" mixes datasets. Per-dataset at turn 10, PEBOL vs GPT-MonoLLM: **0.27/0.12 Yelp,
  0.18/0.09 MovieLens, 0.17/0.11 Recipe-MPR**; vs Gemini-MonoLLM: 0.27/0.17, 0.18/0.15, 0.17/0.16.
- **★ A quotable precedent for our own protocol argument** [V]: PEBOL's authors **decline** ConvRec baselines
  ("not cold-start systems") and **decline ReDial** — *"made up of pre-recorded conversation histories and
  cannot be used to evaluate active, cold-start NL-PE systems."* Cite this verbatim when a reviewer asks why
  we do not run a CRS dialogue benchmark.
- **Code:** https://github.com/D3Mlab/llm-pe (verified, resolves; Python, YAML config,
  `experiment_manager.py` / `eval_manager.py` / `pe_modules/` / `dataloaders/` / `llms/`).

**Verdict: UNFIT for the primary table. MUST-INCLUDE as one LLM-free ablation row** — see §7.1.

**Successors:** 26 citations; **no PEBOL v2 and no NL-PE benchmark**. NL-PE did not become an adopted named
task — the term appears essentially only in PEBOL and in the group's own survey (Korikov et al.,
"Large Language Model Driven Recommendation", arXiv:2408.10946) [S]. Same-group adjacent work: Kemper et al.
SIGIR 2024 (arXiv:2406.00033), Korikov et al. IR-RAG@SIGIR 2024 (arXiv:2408.00878) [S].

### 1.1b Pep — Bose et al., "Cold-Start Personalization via Training-Free Priors from Structured World Models", arXiv:2602.15012 (Feb 2026) **[A — abstract only]**
**The nearest live competitor to our offline-prior + online-selection architecture.** Offline: learn a
structured world model encoding preference *correlations* from complete profiles. Online: training-free
Bayesian inference both **selects informative questions** and **imputes unasked dimensions**. Reports 80.8%
alignment vs 68.5% for RL with 3–5× fewer interactions. Domains are reasoning tasks (medical / mathematical /
social / commonsense), **not a fixed item catalogue** — so it is not a baseline for us.
**★ Adopt its adaptivity diagnostic:** *when two users give different answers to the same question, Pep
changes its follow-up **39–62%** of the time versus **0–28%** for RL.* That is a cheap, sharp, contract-free
measure of whether a policy is genuinely adaptive — and our own harness audit says Act 1 currently has zero
adaptive selectors, so this diagnostic would expose that immediately. **Read in full before Act 1 runs.**

### 1.2 Montazeralghaem, Tennenholtz, Boutilier & Meshi 2025 — "Asking Clarifying Questions for Preference Elicitation With LLMs", arXiv:2510.12015 **[V, arXiv HTML]**
Google Research — the same group as ELM. **A diffusion analogy for question ordering.**
- **Forward (corruption):** start from a complete LLM-written user profile in JSON with tags ranked
  general→specific; generate one question per tag; then progressively DELETE the information that answers
  each question: `JP^t_u = JP^u \ ⋃_{i=t}^{n-1} T_i`, down to an empty profile.
- **Reverse (training):** fine-tune an LLM to ask the questions that RECONSTRUCT the profile,
  `p_θ(Q_{t-1} | P_{t-1})`. The learned artefact is a *funnel ordering* — general questions first.
- **Action space: FREE-FORM GENERATION.** No candidate set, no acquisition function, **no continuous latent
  action**.
- **Data:** MovieLens LLM-written profiles from the ELM line (Tennenholtz et al. 2024). **User/item counts
  are not reported.**
- **Metrics: BLEU / ROUGE** against the ground-truth profile, % unanswered questions, and a "Weighted Rank"
  `WR = Σ i·p(i)` diagnostic for funnel ordering. **No NDCG, no ranking metric of any kind.**
- **Answers 100% LLM-simulated** [V]: *"In each conversation between our Questioner and user simulator, we
  provide the user profile to the LLM and ask it to find the answer from the profile if possible … Otherwise,
  we instruct the LLM to respond with 'I don't know' in cases where the answers are not present in the user
  profile."*
- **No code URL.** Also appeared at the SIGIR 2025 GenNext workshop [S].

**Verdict: UNFIT as a baseline. MUST-CITE.** It is the 2025 statement of the "let the LLM ask" position,
by the group whose embedding-verbalisation work is nearest Papers C/E, and its metric choice (BLEU/ROUGE on
a reconstructed profile) is the cleanest available evidence for our sentence that *this family does not
measure recommendation quality at all*. It is also, usefully, a **profile-RECONSTRUCTION** objective — the
same objective shape as Paper B's set-encoder — arrived at from the language side.

### 1.3 BED-LLM — Choudhury, Williamson, Goliński, Miao, Bickford Smith, Kirchhof, Zhang, Rainforth — "Intelligent Information Gathering with LLMs and Bayesian Experimental Design", arXiv:2508.21184, **ICLR 2026** (OpenReview `qyylZMLYT8`) **[V, arXiv HTML v2]**
Apple + Oxford. **The strongest principled acquisition paper in the LLM family.**
- **Mechanism:** sequential Bayesian experimental design. At each turn the LLM *proposes* a set of candidate
  questions (unconstrained generation, "a targeted strategy for proposing candidate queries"); a purpose-built
  **expected-information-gain estimator** scores them; **argmax EIG is asked**. Conditioning is not left to
  in-context updating — a probabilistic model over the LLM's beliefs is maintained explicitly.
- **Tasks:** 20-Questions, and **film preference elicitation**. Turns `t ∈ {0,…,5}` (six turns);
  **200 user profiles** as ground-truth targets.
- **Metric for the film task: mean rating across 10 film recommendations** (1–5 scale, 0.5 increments),
  mean ± SE over the 200 users. **Not a ranking metric.**
- **Answerer is a separate LLM** given the ground-truth `θ*` (a user profile), answering without access to
  the questioner's context. ⇒ **LLM-simulated preferences.**
- ⚠ A first extraction pass reported "MovieLens, 1,682 movies"; this was **NOT confirmed** on the HTML pass.
  **Do not quote a catalogue size for BED-LLM.**
- **Successor:** Hartmann, Harvey, Navott, Wang, Melo, Cipcigan, Zhang & Abate, *"Amortising Bayesian
  Experimental Design for Sequential Information Gathering in LLMs"*, arXiv:2607.03426, **FoGen@ICML 2026
  workshop (non-archival)** [A]. RL-fine-tunes the policy with an EIG reward so the design is amortised into
  the weights; 20 Questions + MediQ; ">2× the success rate of the 7B base model", ">25× cost reduction vs
  BED-LLM". Text action space. No recsys ranking task.

**Verdict: UNFIT as a baseline. MUST-CITE.** BED-LLM is the 2026 peer-reviewed evidence that **greedy
one-step EIG over proposed candidates is still the state-of-the-art acquisition rule** — the same criterion
our greedy prefix-optimal selector uses. Together with `wang2026aipersona` (arXiv:2605.00696) this makes two
independent 2026 anchors for "our selector is not a weak choice". Cite both in the same sentence.

### 1.4 COPE / InPE — Xia, Zhang & Wang 2026, "When and How to Ask: Dynamic Preference Elicitation Strategies for Conversational Recommendation", arXiv:2607.06765 **[V, arXiv HTML]**
- **InPE** = the INSPIRED corpus re-annotated: **6,719 turn-level samples** labelled with (1) task type
  (elicit / recommend / chat), (2) **elicitation strategy type — attribute-based / item-based / hybrid**,
  (3) target items, (4) comparative response pairs. Data: https://github.com/juanfacabian/InPE
- **COPE** = a hierarchical router (task level + strategy level) over task-specialised experts on a shared
  Qwen3-8B backbone. **It selects a STRATEGY TYPE, not a question** — a closed 3-way action space, i.e. the
  same shape as PPDPP and RSO.
- Metrics: Recall@1/@10/@50 (best **Recall@10 = 0.314**), task/strategy accuracy, pairwise win rate 0.604;
  a preliminary user study with 328 crowdworkers.
- **★ The finding that matters to us** [V]: *"Attribute-based strategies are more effective for early-stage
  preference exploration, whereas item-based strategies become more suitable once users have formed clearer
  and more specific preferences."*

**Verdict: UNFIT as a baseline** (INSPIRED dialogue corpus, Recall@k against a known conversational target,
3-way strategy action space). **MUST-CITE**: it is an independent, 2026, dialogue-side statement of
**exactly our concept-beats-item-at-small-budget result**, arrived at with a completely different method,
dataset and metric. Put it next to `CONCEPT_VS_ITEM_BANK_SWEEP.md`. Convergent evidence from a different
contract is worth more than another number in our own table.

### 1.5 GATE — Li, Tamkin, Goodman & Andreas, "Eliciting Human Preferences with Language Models", arXiv:2310.11589, **ICLR 2025** **[V for content, S for venue]**
Already indexed as `li2023eliciting`; this pass upgrades and corrects it.
- **Question generation = pure prompting of GPT-4 (`gpt-4-0613`). No learned selection, no acquisition
  function.** Three elicitation policies: *generative active learning* (LM generates example inputs for the
  user to label), *yes/no question generation*, *open-ended question generation*; compared against
  supervised learning, a user-written prompt, and pool-based active learning — six methods total.
- **Three domains:** email validation; **content recommendation** (would the user want to read a held-out
  article — test cases are popular newspaper/magazine articles collected by the authors); moral reasoning.
- **REAL humans, preregistered:** *"We recruited 20–30 participants for each domain-method pair (6
  elicitation methods across 3 domains), for a total of **388 participants**"* (Prolific + CS students);
  by domain: content recommendation 183, moral reasoning 121, email validation 84. Preregistration
  https://osf.io/5v6nd/.
- **Metric: p(correct)** — the probability the model assigns the user-preferred answer — summarised as **area
  under the Δp(correct)-vs-elicitation-TIME curve**, i.e. accuracy gain integrated against *user effort in
  wall-clock time*. That is a notably honest cost axis and worth borrowing if we ever run humans.
  AUROC in appendices. **No NDCG / Recall / MRR anywhere**, and no item catalogue at all — the recommendation
  domain is binary per-article preference prediction, **not ranking**.
- **Code:** https://github.com/alextamkin/generative-elicitation
- ⚠ **Correction to a floating claim:** a summariser once attached "Microsoft News Dataset, 160k articles"
  to GATE's recommendation domain. The verbatim text says the articles were *collected by the authors*.
  **Treat MIND/160k as unverified — do not cite it.**

**Verdict: UNFIT as a baseline. MUST-CITE, and note what it does and does not threaten.** GATE is the
genuine **open-vocabulary** counterpoint in this space, and it is the only paper here with **real human
participants**. But it has no learned selection, no catalogue, and no ranking metric — so it threatens a
*framing* claim about question format, **not** a selection claim.

### 1.5b OPEN — Handa, Gal, Pavlick, Goodman, Andreas, Tamkin & Li, "Bayesian Preference Elicitation with Language Models", arXiv:2403.05534 (Mar 2024) **[V]**
GATE's own EIG-based successor, and **the closest published thing to a principled LLM-free acquisition
wrapped in an LLM shell**. Not previously in our INDEX.
- Feature space `|F| = 10`; a query is a pair of binary feature vectors `(o_a, o_b)`, each containing exactly
  `K = 2` features. User model is **Bradley–Terry**,
  `p(x_a | (x_a,x_b), θ) = σ(θ·φ(x_a) − θ·φ(x_b))`; the `θ` prior is initialised from an LM-generated
  feature-importance ranking with variance scaled `w_i = 1.2 − 0.12·i`.
- **Acquisition: `EIG(q) = E_{p(y|q)}[IG(q,y)]`, posterior approximated by a particle filter over N personas**,
  reweighted by the BT likelihood after each response. **The LM does not choose the query** — it supplies only
  the two inverse maps: `φ⁻¹` verbalises a feature vector into an NL question, `φ` featurises test items into
  feature vectors.
- Evaluation: content recommendation over **30 New York Times articles** (15 persistent pairwise test
  comparisons), **~40 real users per condition** across six settings. Metric = **TIDA** (time-integrated delta
  accuracy), GATE's effort-integrated measure. Baselines: LM open-ended, LM pairwise, user self-mapping.
- **Code promised in the reproducibility statement, no URL in v1; no repo located.**

**Verdict: UNFIT as a baseline (30 articles, TIDA, pairwise). NICE as an ABLATION and MUST-CITE.** Its
acquisition — EIG over a particle-filter posterior with a `K=2`-of-10 binary query space — **maps directly
onto our concept bank**, with the one adaptation that it is pairwise rather than pointwise yes/no. It is the
strongest available demonstration that the *principled* half of an LLM elicitation system can be run with no
LLM in the loop, which is precisely the argument our ablation row makes.

### 1.5c CLARINET — Chi, Lin, Lin & Klein (UC Berkeley), "Augmenting Language Models to Ask Clarification Questions for Retrieval", arXiv:2405.15784v1 **[V]**
Not previously in our INDEX, and **the most useful missing comparator for our greedy-selector result.**
- Retriever = DPR finetuned on WhatsThatBook (11,552 query–document pairs). Belief state is **not** Bayesian:
  GPT-3 condenses the dialogue history into a **"language posterior"** — a synthesised NL query fed to the
  retriever.
- **Question selection with an explicitly UTILITY-maximising criterion:** GPT-3.5 generates 50 candidates
  conditioned on the query, history and top-3 retrieved items; each candidate is scored by **simulating the
  user's answer and measuring the RANK IMPROVEMENT OF THE TRUE ITEM**; only questions that put the target in
  the top-10 or improve its rank by ≥10 survive as training data; Flan-T5-base (Fusion-in-Decoder) is then
  finetuned to emit the rank-maximising question directly, so inference is one forward pass.
- **★ It implements EIG and KL-divergence acquisition as baselines and reports them as WORSE** — CLARINET
  beats *"traditional heuristics such as information gain"* by **17% relative** on retrieval success, and
  vanilla-prompted LLMs by 39%. Both baselines are noted as expensive at inference.
- Data: tip-of-the-tongue book retrieval (Bhargav et al. 2022), filtered to 784 interactions
  (156 train / 628 test); **catalogue = 1,877 documents**. Metrics: top-1 accuracy, MRR, cumulative and
  non-cumulative retrieval success.
- Oracle LLM-simulated [V]: *"We implement the user simulator as a GPT-3 model that has access to the true
  item and is prompted to answer questions about the item vaguely."* — ⚠ the simulator **sees the true item**,
  i.e. the PEPPER target-biased failure mode, verbatim.
- **No code found.**

**Verdict: UNFIT as a baseline (1,877 documents, retrieval-success metric, target-biased simulator).
MUST-CITE, and it is a gift.** Its selection objective — *maximise the rank of the true item under the
current retriever* — **is our greedy prefix-optimal objective in different notation**, and its finding that a
learned rank-gain rule beats EIG is an independent, published pre-support for our own
`prefix-optimal-greedy-is-the-right-object` result. Cite it there.

### 1.6 The strategy-selection line (for completeness)
PPDPP (`deng2023plug`, ICLR 2024, arXiv:2311.00262 — applied to negotiation/emotional-support/tutoring,
**not recommendation with an item catalogue** [S]), RSO (`zhao2025reinforced`, arXiv:2509.26093), MACRS
(`fang2024multi`, arXiv:2402.01135 [A]), COPE (§1.4), and Zhang, Knox & Choi ICLR 2025 (`zhang2025modeling`,
arXiv:2410.13788 [A]) all share one shape: **a small learned policy picks from a closed discrete set of
dialogue STRATEGIES and a frozen LLM speaks.**

New detail on **RSO** [V]: the planner (a RoBERTa-based LM) maps dialogue state to one of **13 macro
strategies** in three groups — 8 *Sociable*, **3 *Preference Elicitation* (Opinion Inquiry, Experience
Inquiry, Rephrase Preference)**, and generic chat; an Actor expert instantiates the strategy into text.
Training = SFT on INSPIRED strategy annotations, then entropy-regularised policy gradient with an
**LLM-as-judge reward** (Qwen2.5-7B-Instruct, 1–5 rubric, 10 samples averaged). Data: INSPIRED (1,001
dialogues / **1,783 items**) and ReDial (10,006 dialogues / **51,699 items**); Recall@1/@5; evaluation is
LLM-as-judge **on pre-recorded dialogues, not interactive** — which structurally cannot measure an active
policy at all.
**The crisp related-work sentence: act-level planning is not acquisition.** RSO chooses *that* it will ask
about an experience; nothing in it chooses *which* item or attribute. There is no content-level acquisition
to ablate.

**The observation worth making in the paper: the action space has not grown in three years** — still ~3–13
strategy labels from PPDPP (2024) through COPE (2026). That is the standing gap Papers C/E aim at.

### 1.7 Adjacent question-asking work found, triaged, NOT read
Listed so the next pass does not re-find them. All **[S]**, none verified:
`STaR-GATE` (arXiv:2403.19154) · `TO-GATE` (arXiv:2506.02827, AAAI 2025) · `QuestBench` (arXiv:2503.22674,
NeurIPS 2025) · `MediQ` (arXiv:2406.00922, NeurIPS 2024) · `Active Preference Inference using Language Models
and Probabilistic Reasoning` (arXiv:2312.12009) · `Comparison-based Active Preference Learning for
Multi-dimensional Personalization` (arXiv:2411.00524) · `Snippet-based CRS` (arXiv:2411.06064) ·
`CaRT: Teaching LLM Agents to Know When They Know Enough` (arXiv:2510.08517) · `AGENT-CQ` (arXiv:2410.19692) ·
`Accelerated Preference Elicitation with LLM-Based Proxies` (Huang, Marmolejo-Cossío, Lock, Parkes, WINE'25,
arXiv:2501.14625 — LLM proxies + DNF-proper-learning for combinatorial **auctions**, ~5× fewer queries; wrong
domain) · `Decisive` (ACL 2026, arXiv:2604.18122) · `Ontology-Guided Hierarchical Preference Elicitation with
Bayesian Active Querying` (ICAART 2026, DOI 10.5220/0014476800004052) · `Feedback-Driven Gradual Discovery for
Expanding Musical Preferences` (RecSys 2025, DOI 10.1145/3705328.3748025).
⚠ **"STARDUST" could not be verified to exist** in this literature — do not cite it.
⚠ **PersonalWAB is entirely unverified** (Semantic Scholar rate-limited) — do not cite it.

---

## 2. SCOPE A′ — Bıyık et al., the closed verification debt **[V, full PDF]**

**Bıyık (USC), Yao (UVA), Chow, Haig, Hsu, Ghavamzadeh, Boutilier (Google Research),
"Preference Elicitation with Soft Attributes in Interactive Recommendation", arXiv:2311.02085v1, 22 Oct 2023.**

**⚠ VENUE: arXiv-ONLY.** Boutilier's own publication page lists it as *"arXiv:2311.02085 (2023)"* with no
venue; DBLP returns no match. **Index it as an unrefereed preprint** — and weight it accordingly in the
Paper A threat model. (Contrast: the same page lists Göpfert et al. as *ACM TORS 2(4):1–37, 2024*.)

- **Belief:** Gaussian prior over the user embedding, taken from their two-tower CF model; Bayes update
  `P_U(u|H^{(K)}) ∝ P_U(u) ∏_k P(ρ^{(k)}|q^{(k)}, φ_u)`, assuming conditional independence of responses.
  **The posterior is explicitly NOT Gaussian** [V]: *"Unlike the prior, generally the posterior is not
  Gaussian, and thus, we use the following two methods to tractably approximate it."* — (a) a parameterised
  posterior fitted by max-likelihood to MH/HMC samples (batch vs iterative; iterative wins), (b) a
  **Laplace approximation**. Empirically the *sampled* posterior beats the Gaussian one.
- **Action space is JOINT:** a query is `q = (S, g)` — a **slate of items plus a tag**; item queries are
  `q = S` alone. This is the paper's headline claim (items and attributes in one action space).
- **Acquisition:** Random / Entropy / Mutual Information / **EVOI**; *"EVOI is the most effective AF,
  outperforming MI and Entropy"*. **BPER** blends it with recommendation quality,
  `γ·IG(q|H) + (1−γ)·RQ(q|H)`, tuned to **γ = 0.5**. Separately, slate optimisers (random search, sequential
  greedy, continuous relaxation + gradient, joint, Thompson). Honest negative: *"PE with random queries
  performs relatively well"* — attributed to inaccurate early priors and non-convergent slate optimisers.
- **Attributes = Göpfert CAVs, verbatim adopted** [V]: *"Göpfert et al. [21] apply CAVs to identify the
  semantics of tags w.r.t. the item representation learned by CF models in RSs. We adopt this approach in our
  work."* Regularised logistic probe on **frozen** item embeddings; the CAV is the normal to the separating
  hyperplane; g-score `c_g(i) = φ_g^T φ_I(i)`. **Explicitly two-stage/frozen**: *"We train the CF model and
  learn CAVs separately … in contrast to those that jointly train attribute models."*
  ⇒ **CORRECTION to our INDEX row**, which says its encoder is co-trained and therefore fails R1. It does
  **not** co-train. The R1 wedge must be re-argued on *tower strength* (their two-tower/ALS model is not a
  certified RecVAE-class full-profile SOTA and no full-profile benchmark is reported), **not** on freezing.
- **Vocabulary is CLOSED:** |T| = 10 (synthetic), 5 (RecSim NG), **164 tags (ML-20M)**.
- **Their novel bit vs Göpfert: CAV UNCERTAINTY** — a belief `P_g(φ_g|D_g)` over the CAV itself, marginalised
  in both the belief update and the query optimisation. Their largest reported gain (10–20% on their NDCG).
- **Scope limit worth knowing** [V, footnote 5]: *"Different users may have different interpretations of an
  attribute. Such subjectivity can be uncovered using CAVs [21], **but we do not consider this here.**"*
- **Query format: comparative, binary** [V]: *"Would you prefer movies that are more/less thought-provoking
  than those in S?"*, `ρ = ±1`. Response models: probit for attribute queries, multinomial logit for item
  queries, and **IpA (item-plus-attribute)** = pick the preferred item in the slate, then critique it on `g`
  — **IpA is the best query type**. The user is modelled as comparing against a *hypothetical ideal item*
  `φ*_{I,u} = max_i‖φ_I(i)‖₂ · φ_u/‖φ_u‖₂`, which "may fall outside of I".
- **Experiments: 100% simulated, no human study.** Synthetic (|I| = 1,000, 10 users × 5 seeds, 10 queries);
  RecSim NG (|I| = 10,000, 25,000 users, 20 queries); **ML-20M (27K movies, 50-d ALS, 164 tags, 20 queries,
  slate 5) with only 16 sampled test users** [V]: *"We sample 16 such test users for PE."*
- **★ THE LOAD-BEARING FINDING — its "NDCG" is simulator-internal** [V]: *"NDCG is the normalized discounted
  cumulative gain between the true top |S| items and the top |S| items estimated using the posterior."*
  That is **agreement with the simulator's own ground-truth embedding ranking at |S| = 5**, not NDCG against
  held-out ratings. Every number in the paper is measured inside the simulator's closed loop, and the
  "ground truth" is itself just an ALS embedding of a ≥50-rating user.
- **Baselines: internal ablations only** (query type × AF × posterior × slate optimiser), plus **PITF** as an
  alternative attribute-semantics source (CAVs beat PITF). **No external elicitation baseline** — no Golbandi,
  no PEBOL, no bandit CRS.
- **Code: NONE.** The only GitHub URL in the paper is a third-party PITF implementation used for a baseline.
  Implementation is TF + TF-Probability (HMC). (The public `google-research-datasets/soft-attributes` release
  is the *dataset* from the Göpfert line, not this method.)

**Verdict: NICE, not MUST — and materially less threatening than the INDEX currently implies.**
It is the nearest published-adjacent artifact to a belief layer over a frozen tower with items and attributes
in one action space. But it is an unrefereed preprint, closed-vocabulary, 16 simulated users, no held-out
ranking metric, no external baseline, no code. **Do not order its numbers against ours.** If the concept arm
needs a named rival, this is it — but reimplementation is a real project (MH/HMC posterior + slate optimiser).

### 2.1 Göpfert et al. — corrections to our INDEX row **[V, full PDF]**
*Göpfert (Bielefeld), Chow, Hsu, Vendrov, Lu, Ramachandran, Boutilier, "Discovering Personalized Semantics
for Soft Attributes … using Concept Activation Vectors."*
- **Venue: BOTH** — TheWebConf/WWW 2022 (Lyon), **and** extended as **ACM TORS 2(4):1–37, 2024**,
  DOI 10.1145/3658675; arXiv:2202.02830. The arXiv author list (which adds Haig and Pham) is the TORS list;
  the WWW'22 PDF has 7 authors.
- **CORRECTION:** our INDEX says the vocabulary comes from "tag-genome / Balog attribute scores". **It does
  not.** It is built by filtering raw ML-20M tag applications [V]: lowercase, keep only user-item-tags with
  rating ≥ 4, restrict to the top 250 tags by unique tagged movies, then top 250 by unique users → *"This
  leaves a 164 tags for evaluation."* **Balog et al. 2021 is cited only as motivation**; their scores are
  not used. Fix the row.
- Beyond the logistic probe they add a **per-user pairwise ranking loss** (RankNet-style) and **LambdaRank**,
  motivated by *degree subjectivity* — and **the ranking losses beat plain logistic regression**.
- "Personalized" means two concrete things: **degree subjectivity** (per-user thresholds, handled by the
  ranking loss) and **sense subjectivity** (an **EM** procedure assigning each user one of several latent
  senses of a tag, each with its own CAV), validated on 13 subjective ML tags, on artificially sense-conflated
  meta-tags, and against a control tag `odd year` for which no good CAV should exist (and none is found — a
  clean negative control worth imitating).
- **Metrics: tag-prediction Accuracy on held-out tag data + Spearman (synthetic only).** No NDCG/Recall.
  §5 is example critiquing with a **simulated** user [V]: *"In lieu of live experiments, we adopt a stylized
  but plausible user response model…"*; metric = UMU (user max utility), not ranking. **No real humans.**

### 2.2 Consequence for the closed-vocabulary count
`critiquing_vocabulary_is_closed.md` records **8 systems, 0 exceptions**. Bıyık and Göpfert make it
**10 systems, 0 exceptions** — and Bıyık is the strongest possible test case, since it is the paper that most
loudly advertises *soft* attributes. Update that file's count and add the two rows.

---

## 3. SCOPE B — meta-learning cold start: consumers, not selectors

**The distinction the author asked to be enforced, adjudicated per method.**

### 3.1 MeLU — Lee, Im, Jang, Cho, Chung, KDD 2019, arXiv:1908.00413 — **SELECTS (static, global, one-shot)** **[V]**
Code https://github.com/hoyeoplee/MeLU (resolves).
- **§3.3 "Evidence Candidate Selection", verbatim** [V]: *"the larger the average Frobenius norm of whole
  users' gradient for personalization is (i.e., the gradient of the local update, ‖∇_{θ2i} L_i(f_{θ1,θ2i})‖_F),
  the better the distinction among users' preferences."* … *"for each item, we calculate the gradient-based
  value and popularity-based value as the average Frobenius norm and the number of interactions for each item
  using the whole existing user-item pairs … we normalize the values to range from zero to one and assign the
  score to each item by multiplying two normalized values. Finally, we define the top k items with the highest
  scores as evidence candidates."*
  ⇒ **score(item) = norm(mean local-update gradient magnitude) × norm(popularity)**, top-k. **k = 20.**
- **GLOBAL, not per-user — adjudicated on three pieces of full text** [V]: both components are population
  aggregates over *"the whole existing user-item pairs"*; the user study used **one fixed printed list**
  (*"we listed the two movie lists in Appendix A"* — a list enumerable in an appendix is the same list for
  everyone); and the paper disclaims adaptivity outright: *"We only considered the one-round interaction …
  but it could be expanded into a multiple-round interactions."* The one sentence that reads adaptive
  (*"the scores can vary when entering new user"*) says only that the **global aggregates shift as the
  population grows**.
- **Its selection is evaluated ONLY by a user study**, n = 84 total, **39 in the treatment arm** [V]:
  *"Forty-five of the users rated movies extracted by popularity only, and the others rated movies extracted
  using our strategy."*
- **★★ INDEPENDENT EXTERNAL CORROBORATION OF OUR ANSWERABILITY COLLAPSE** [V]: *"our evidence candidates were
  rarely rated by users compared to the popularity-based evidence candidates. While approximately one-half of
  the popularity-based evidence candidates were chosen, **only one-quarter of our evidence candidates were
  selected.**"* MeLU measured, in **humans**, the same effect our sweep measured in behaviour: a
  discrimination-ranked ITEM bank is answered at roughly half the rate of a popular one. They reported it as
  an aside and shipped anyway. This is a citable precedent for the concept-vs-item argument — cite it in
  `CONCEPT_VS_ITEM_BANK_SWEEP.md` and in Paper B.
- **Contract** [V]: ML-1M (6,040 u / **3,706 i** / 1,000,209 r) and Bookcrossing (278,858 u / 271,379 i).
  Histories filtered to 13–100; **10 random items = query set**, remainder = support. Metrics **MAE,
  nDCG@1, nDCG@3**.
- ⚠ **ORDERABILITY FLAG** [V]: MeLU's nDCG (Eq. 5) averages `DCG_k/IDCG_k` over the 10-item query set with
  gains `2^R − 1` from the real 1–5 rating. It is a **graded re-ranking of the user's own 10 held-out rated
  items**, not full-catalogue ranking. Not orderable against our NDCG@10 over 18,359 candidates.
- **Verdict: NICE — as a BANK-RANKING baseline, not as an elicitation baseline.** MeLU's score is a named,
  published instance of *discrimination × popularity*, i.e. the closest published relative of our `expinfo`
  family. Reimplementable in an afternoon on the certified tower, no retraining. Cite it for (a) the origin of
  the term "evidence candidates", (b) the answerability-collapse corroboration, (c) a bank ranker.
  **Do NOT cite it as adaptive.**

### 3.2 MAMO — Dong, Yuan, Yao, Xu, Zhu, KDD 2020, arXiv:2007.03183 — **CONSUMER** **[V via ar5iv]**
Code https://github.com/dongmanqing/Code-for-MAMO (resolves). MAML + two memories: a **profile memory**
`M_P ∈ R^{K×d_u}` of K preference prototypes with cosine attention `a_u = attention(p_u, M_P)`; a
**feature-specific user-embedding memory** `M_U` giving a personalised offset to the global init
`θ_u ← φ_u − τ b_u`, `b_u = a_u^T M_U`; and a **task-specific memory cube** `M_{U,I} ∈ R^{K×d_e×2d_e}`
retrieved as fast weights. Memories updated by EMA.
**Selection: NONE.** The support set is fixed by a recency rule [V]: *"we consider the first 15 items for a
user as the support set and the others as the query set … we sort these items according to the user review
time."* Metrics MAE + NDCG@N over the query set.
**Verdict: UNFIT as an elicitation baseline (category error). Recommender row only.**

### 3.3 TaNP — Lin, Wu, Zhou, Pan, Cao, Wang, WWW 2021, arXiv:2103.06137 — **CONSUMER** **[V via ar5iv]**
Code https://github.com/IIEdm/TaNP (resolves). **Already wired in our harness.**
Encoder: FC per `(x,y)` support pair → **mean-pool** (permutation invariance) → reparameterisation for
`q(z|τ)` (full task) and `q(z|S)` (support only). **Customization:** a task-identity network `m_φ` encodes the
support set to `t_i`; a **global pool** `A = [a_1..a_k]` of soft cluster centroids; **Student's-t** kernel
similarity gives assignment `c_i`; `o_i = σ(W_o(t_i + A c_i^T))`. **Adaptive decoder: FiLM**,
`γ_i^l = tanh(W_γ^l o_i)`, `β_i^l = tanh(W_β^l o_i)`, plus a Gating-FiLM variant.
Loss = ELBO: prediction + `KL(q(z|τ)‖q(z|S))` + unsupervised clustering loss on the pool.
**Selection: NONE** [V]: *"τ_i contains few interactions as the support set S_i and remaining interactions as
query set Q_i"* — a partition of what exists, with fixed `N_{S_i} ∈ {20,15,10}`.
⚠ **Trap:** TaNP's `N_S` sweep {10,15,20} looks like a budget sweep but is a **given/random** support size.
TaNP is therefore only a fair comparison against a **random-bank** CASPER arm.
Contract: ML-1M (6,040 / 3,706), Last.FM (1,872 / 3,846), Gowalla (2,692 / 27,237); Precision@N, NDCG@N,
MAP@N for N ∈ {5,7,10} — but ranked **within each user's query set**, not over the catalogue, and not a
Liang strong-generalization split. **Numbers not orderable.**
**Verdict: correctly wired as a CONSUMER baseline. In write-up, place it under "adapts a recommender to a
given support set", explicitly contrasted with question selection.** Its FiLM/gating modulation is also the
closest published relative of our fusion-token line — worth one related-work sentence.

### 3.4 CDRNP — Li, Sheng, Cao, Zhang, Li, Liu, WSDM 2024, arXiv:2401.12732 — **CONSUMER, and off-contract** **[V via ar5iv]**
Neural-process cross-domain recommendation: characteristic embedding over source-domain interactions →
preference encoder aggregating context pairs into `z_i` → adaptive conditional decoder with FiLM-style
preference modulation + a "preference reminder".
**Selection: NONE, and not even user-level support** — the context set is composed of **overlapping users**
(present in both domains), a population-level context. **Nothing is asked of the cold user at all**; the cold
signal is their *source-domain* history.
Data: three Amazon scenarios (Movie→Music, Book→Movie, Book→Music). **Metrics MAE and RMSE only.**
**No code repository given.**
**Verdict: UNFIT — different information contract (cross-domain transfer), no ranking metric. Related work only.**

### 3.5 2023–2026 successors — the selection thread was never picked up
Searched specifically for active meta-learning / learning-to-query / support-set selection, 2023–2026.
**No meta-learning cold-start method was found that selects its own support set.** [S] The field converged on
*better consumption* — better modulation, better init, better auxiliary structure.

| Method | Citation | Selects? | Code |
|---|---|---|---|
| **ColdNAS** | Wu et al., WWW 2023, arXiv:2306.03387, DOI 10.1145/3543507.3583344 | **No** — NAS searches the *modulation function and its position*, not items | `LARS-research/ColdNAS` ✔ |
| **MetaKG** | Du et al., TKDE 2022, arXiv:2202.03851 | **No** — KG meta-learners, support given | `ZJU-DAILY/MetaKG` ✔ |
| **M2EU** | Wu, Zhou et al., SIGIR 2023, DOI 10.1145/3539618.3591719 | **No** — selects **similar USERS to borrow from**, never items to ask. *Easy to misread as selection; it is not.* | `zhenchaowu/M2EU` ✔ |
| **MetaHIN** | Lu, Fang, Shi, KDD 2020 | **No** — the "task constructor" builds meta-paths, not question lists | — |
| **PAML** | IJCAI 2021 | **No** — social relations for generalisation | — |
| **ClusterSeq** | arXiv:2307.13766 | **No** — support = sequence prefix | — |
| **Popularity-aware meta-learning** | arXiv:2411.11225 | **No** — *item* cold start, not user elicitation | — |

**★ A defensible gap statement, which survived a deliberate attempt to falsify it:** *the meta-learning
cold-start literature is a literature of answer-consumers. Exactly one paper in the line (MeLU, 2019) selects
evidence candidates; it does so with a single static popularity×discrimination ranking, evaluates it on 39
human subjects, and observes — but does not address — the answerability collapse it causes.*

### 3.6 Surveys and one high-value critique
- **Buffelli, Gupta, Strzalka, Plachouras, "Is Meta-Learning the Right Approach for the Cold-Start Problem in
  Recommender Systems?", arXiv:2308.08354, 2023** [A]. Claims properly-tuned conventional deep models match
  specialised meta-learning on standard cold-start benchmarks, and that meta-learning is impractical at
  industrial latency. **Strategic value: this lets us decline to build a meta-learning arm without appearing
  to dodge** — the burden of proof is already contested in print. ⚠ Abstract only; experimental section not
  verified.
- **Zhang, Bei, Yang et al., "Cold-Start Recommendation towards the Era of LLMs: A Comprehensive Survey and
  Roadmap", arXiv:2501.01945, 2025** [V abstract]. Organises cold-start by *information source* — content →
  graph → domain → LLM world knowledge. ⚠ **Its taxonomy has no elicitation / active-selection axis at all.**
  Quotable as evidence that the field frames cold start as "find more side information", never "ask a better
  question".
- **Zawia et al., "Comprehensive Review of Meta-Learning Methods for Cold-Start Issue in Recommendation
  Systems", IEEE Access 13:24622–24641, 2025**, Xplore doc 10857336 [S]. PRISMA review, 47 studies, window
  2018–Jun 2024. ⚠ **DOI unverified** (search surfaced 10.1109/ACCESS.2025.3536025; the IEEE fetch returned
  empty). Verify before citing.
- **Wang, Zhu, Liu, Zang, Yu, Tang, "Deep Meta-learning in Recommendation Systems: A Survey",
  arXiv:2206.04415, 2022** [V] — preprint only, no venue documented. Secondary.

---

## 4. Benchmarks — is there a consensus one? (answers §7)

| Benchmark | What it actually measures | Ranking? | Relevant to us? |
|---|---|---|---|
| **PrefEval** (Zhao, Hong, Liu, Hazarika, Lin; **ICLR 2025 oral**; arXiv:2502.09597; https://prefeval.github.io/) | **Preference FOLLOWING**, not elicitation: 3,000 curated preference/query pairs, 20 topics, explicit+implicit, contexts to 100k tokens; generation + classification. Headline: following accuracy **<10% by turn 10** for most models zero-shot | ✘ | ✘ — wrong task (following, not asking) |
| **InPE** (arXiv:2607.06765, §1.4) | Turn-level elicitation-strategy labels on INSPIRED; Recall@k against the dialogue's known target | partial | ✘ as a benchmark, ✔ as a citation |
| **iEvaLM** (`wang2023rethinking`, EMNLP 2023) | Interactive re-evaluation of CRSs on ReDial/OpenDialKG with an LLM user | Recall@k vs known target | ✘ — LLM user + known target |
| **Latent-information-discovery benchmark**, arXiv:2510.17132 [S] | 20-Questions / personalised QA / personalised summarisation, tri-agent User–Assistant–Judge, turn-level elicitation scoring | ✘ | ✘ |
| **LaMP** (Salemi, Mysore, Bendersky, Zamani, arXiv:2304.11406) [A] | 7 personalized tasks (3 classification, 4 generation) with retrieval augmentation over a **stored** user profile | ✘ | ✘ — uses a profile, never asks |
| ReDial (10,006 dialogues / **51,699 items**) / INSPIRED (1,001 / **1,783**) / DuRecDial / TG-ReDial / OpenDialKG / CRSLab | Pre-recorded human-human transcripts; dialogue quality + Recall@k of a target named in the conversation | Recall@k vs known target | ✘ — **structurally incapable**: a fixed transcript cannot answer a question the policy invents. Plus >15% of INSPIRED targets are pre-named (He CIKM'23) |
| Qulac / ClariQ / ConvSim / ClarQ-LLM / ClarifyMT-Bench / ProactBench | Clarifying questions for **search** or generic dialogue, not catalogue recommendation | MRR/nDCG over documents | ✘ — different task |

**VERDICT: there is NO consensus benchmark for conversational preference elicitation-as-ASKING that a
reviewer of a cold-start RANKING paper would expect us to run.** The 2025–26 benchmark energy went to
*preference following* (PrefEval, LaMP) and *simulator human-likeness* (MirrorBench), not to
elicitation-then-rank. Every asking-method above evaluates on its own bespoke setup — **PEBOL on 100 items,
CLARINET on 1,877, OPEN on 30, GATE on no catalogue at all** — which is itself the evidence that no shared
protocol exists.

**★ And PEBOL's own authors made our argument for us** [V]: they decline ReDial because it is *"made up of
pre-recorded conversation histories and cannot be used to evaluate active, cold-start NL-PE systems."*
Quote that sentence. A reviewer cannot demand "the standard benchmark" because there isn't one — but they
can demand we justify our protocol, and the contract argument (Liang strong-generalization + full/tail
NDCG@10 + behavioural answers + the named elicitation baselines in `adaptive_elicitation_baselines.md`)
is an argument we are better placed to win than any paper in this table.
Related survey worth one row: *"Evaluating User Experience in CRS: A Systematic Review Across Classical and
LLM-Powered Approaches"*, arXiv:2508.02096 [S].

---

## 5. Simulator validity — 2024–2026 update to `simulator_circularity_conv_rec.md`

- **Yoon, He, Echterhoff & McAuley, "Evaluating Large Language Models as Generative User Simulators for
  Conversational Recommendation", NAACL 2024, arXiv:2403.09738** [A]. **Five probe tasks** on which a
  synthetic user must behave like a real one: (1) choosing which items to discuss, (2) expressing binary
  preferences, (3) expressing open-ended preferences, (4) requesting recommendations, (5) providing feedback.
  Finds measurable deviations from human behaviour and offers model-selection/prompting mitigations.
  **THE citation for "LLM simulators demonstrably diverge from real users on exactly the behaviours an
  elicitation study depends on."**
- **Balog & Zhai, "User Simulation for Evaluating Information Access Systems", Foundations and Trends in IR
  18(1–2):1–261, 2024; arXiv:2306.08550** [V metadata]; also a SIGIR-AP 2023 tutorial (10.1145/3624918.3629549).
  The field's monograph — cite as the standard reference for simulation-based evaluation.
- **Balog et al., "User Simulation in the Era of Generative AI: User Modeling, Synthetic Data Generation, and
  System Evaluation", arXiv:2501.04410, 2025** [S].
- **MirrorBench**, arXiv:2601.08118, 2026 — "A Benchmark to Evaluate Conversational User-Proxy Agents for
  Human-Likeness" [S, unfetched].
- Simulator-CONSTRUCTION work found but **not** validity studies (do not cite as evidence either way):
  CSHI (OpenReview H9BYGURN9M), RecUserSim, PersonaLens, arXiv:2605.05250, arXiv:2606.22803, arXiv:2607.00010 [S].
- **Every LLM-era elicitation method in §1 and §2 uses a simulated answerer, and not one runs a human study
  except GATE.** That is the strongest single sentence available for our firewall argument: the family that
  most needs a validity argument has, with one exception, not made one.

---

## 6. Q1 — is the "policy emits a continuous vector, LLM verbalises it" bet taken?

### 6.1 VERDICT: **PARTIALLY TAKEN. Both halves exist separately; one paper joins them. The specific conjunction is still OPEN — but the framing must change.**

The claim *"nobody has a continuous policy action rendered by an LLM"* is **dead**. Do not write it.

**★ THE PAPER TO CITE AND SEPARATE FROM — LDPP, "Simulation-Free Hierarchical Latent Policy Planning for
Proactive Dialogues", arXiv:2412.14584, AAAI 2025 [V, full text].** Not previously known to this project.
- Latent policies are **discovered from raw dialogue logs** by a VQ-VAE variant giving a codebook
  `Z = {Z_k ∈ R^d}, k = 1..K`.
- **The action is NOT a hard codebook pick** — verbatim: `z_t = Σ_k Z_k · p_{θ,k}(z|u_t)`, a weighted sum;
  the preliminaries define the high-level action as `z_t ∈ A_h` with **`A_h = R^d`**.
- Policy planner (RoBERTa-Large) trained by **offline hierarchical RL (IQL)**,
  `L_high(φ) = −E exp(τ(Q_α(h_t,z_t) − V_β(h_t))) log p_φ(·)`.
- **Rendering: a P-Former adapter maps `z_t` to T "policy-related tokens" prepended to the dialogue history
  for a FROZEN LLM**, so it "can understand and follow the guidance of latent policies."

**What LDPP is missing, i.e. our three separation lines:**
1. **Domain.** Emotional support (ESConv, ExTES) and donation persuasion (P4G). Its utterances are supportive
   /persuasive **responses**, not questions posed to elicit preferences. **There is no user model updated
   from answers** — no belief, no fold-in, no downstream ranking.
2. **Which space.** The latent space is a space of *dialogue strategies* induced from transcripts. It carries
   no information about *what the system is uncertain about* in a preference model.
3. **"Continuous" with an asterisk — and this is our cleanest line.** The reachable action set is the
   **convex hull of K codebook vectors** (the policy's free parameter is a K-simplex). LDPP is the
   **continuous relaxation of a discrete strategy set** — the continuous generalisation of PPDPP — not an
   unconstrained direction in a pretrained semantic latent. Say exactly this.
- Citation sweep: 16 citing papers, **none** extends it to elicitation, question-asking, or recommendation
  (METRO 2604.11427, PRINCIPLES 2509.17459, DialogXpert 2505.17795, DecoupledESC 2505.16995, DPP-CL WWW'25,
  UP-NRPA 2606.13683) [S]. Nearest tangent: CURP, arXiv:2602.00742 [A].

### 6.2 ELM — the renderer half is TAKEN, including the hard part **[V, ar5iv full text]**
This closes another standing question. `tennenholtz2023elm` (arXiv:2310.04475):
- Adapter = *"a two-layer MLP for the domain-embedding adapter layer E_A"* projecting domain embeddings into
  the LLM token-embedding space.
- **It DOES decode hypothetical vectors** — verbatim: *"one could even extract information from the embedding
  representations of **non-existent or hypothetical** entities."* Concretely it **linearly interpolates movie
  embeddings (α = 0.5)**, interpolates random pairs of **user** embeddings, and **extrapolates along CAV
  directions** ("make an existing movie funnier"), introducing semantic/behavioural **consistency metrics**
  precisely because these points have no ground truth.
- **No interactive loop** (embedding → text, one direction, the system never asks anything) and **no learned
  policy** (all vectors are precomputed or analyst-supplied; nothing selects *which* vector to decode).

⇒ **We cannot claim novelty for "an LLM verbalises a synthesised latent point."** ELM took it, including
synthesised points. Cite it as the mechanism precedent.

### 6.3 The other neighbours, sorted by which half they take
**Continuous action → text, but not a question to a user:**
- **ACING**, arXiv:2411.12736 [V] — actor-critic emits a **continuous soft-prompt vector**, converted to NL
  text by the LLM. But the output is a *task instruction* optimised for downstream accuracy; no user, no
  belief state.
- **VDLM**, arXiv:2602.15870 [V-abstract] — a masked-diffusion **planner operating in embedding space**,
  RL-post-trained with **embedding-space rewards**, *"avoiding text decoding inside the RL loop"*, with a
  **Vec2Text renderer** converting planned embeddings to text. **The best available precedent for
  "RL in embedding space + vec2text as the renderer."** Generic long-form generation; not dialogue, not a
  question, no user.
- CoLA arXiv:2503.21383 [A]; BWArea arXiv:2405.17039 [S]; soft→hard prompt translation arXiv:2605.27642 [S];
  neologism self-verbalization arXiv:2510.08506 [S].

**Elicitation with a real acquisition rule, but a DISCRETE action:** BED-LLM (arXiv:2508.21184), CA-BED
(arXiv:2606.01182), amortised BED (arXiv:2607.03426), Active Task Disambiguation (arXiv:2502.04485), BALAR
(arXiv:2605.05386), GATE, TO-GATE, PPDPP, dual-process planning (arXiv:2406.05374). All maximise EIG or a
learned score **over sampled/enumerated candidate questions.**

**★ 6.4 The strongest single piece of evidence that the gap is real — Info-Synth, "Active Query Synthesis for
Preference Learning", arXiv:2605.26072 (2026) [V-partial].** It synthesises queries **in a learned latent
space** by maximising a mutual-information objective — and then **presents the query as a pair of items
selected from the dataset.** That is **Wolpertinger's kNN snap, reproduced in 2026, in our exact problem
setting.** Nobody in preference elicitation has replaced the snap with an LLM renderer. That is the
unoccupied step.

**6.5 Negative results worth keeping (so the next pass does not redo them).** arXiv-API `abs:` conjunctions
returning **zero** hits: `"preference elicitation" AND "latent space"`; `"conversational
recommender"|"conversational recommendation" AND "continuous action"|"latent action"`; `"cold-start" AND
"interview" AND "question" AND "large language model"`; `"ask questions" AND "latent" AND "user preference"
AND "LLM"`; `"policy" AND "question" AND "soft prompt" AND "user"`. Embedding-inversion/vec2text (35 papers
surveyed) is **overwhelmingly a privacy-attack literature**; the only inversion-as-controller-output
instances are VDLM and Vec2Summ (arXiv:2508.07017) — **nobody uses inversion as an elicitation policy's
output channel.** Latent-reasoning recsys (24 papers, incl. arXiv:2604.26760, arXiv:2505.19092) decodes
latents to **items, semantic IDs, intents or reasoning tokens** — **zero decode a latent into a question
addressed to a user** (closest miss: RGAlign-Rec arXiv:2602.12968, "predicted user intent queries" = a
retrieval query). The CF-embedding-as-soft-prompt line (arXiv:2401.04858, 2502.19178, 2306.01475, 2603.17540)
injects user embeddings into an LLM; **none produces a question.**

### 6.6 The framing the author should adopt
The contribution is **not** "continuous action" (LDPP) and **not** "LLM as renderer of a synthesised latent"
(ELM). It is **closing the loop inside ONE latent space**: the action, the belief, and the answer all live in
the same geometry, and the LLM is only the I/O surface. Two sentences of separation:
*this generalises PPDPP/LDPP's strategy codebook to a semantically grounded space*, and *it removes the
kNN snap that Wolpertinger (2015) and Info-Synth (2026) both still require.*
**Both sentences are now supportable from primary text. The bet survives — but only in this reframed form.**

---

## 7. What belongs in the 2026 Paper B baseline battery

**From this pass, exactly two things change the battery:**

1. **PEBOL acquisition, behavioural answers** — ONE ablation row. Beta-Bernoulli posterior per item +
   Thompson/UCB, run over our tower with real held-out answers, no LLM anywhere in the path. **~0.5 agent-day**
   (the posterior and the harness already exist). It closes "you ignored the LLM-era PE method" without
   importing the circularity.
2. **MeLU evidence-candidate score as a bank ranker** — `norm(mean gradient magnitude) × norm(popularity)`,
   slotted beside `pop` / `expinfo` / `polar` / `entropy` in the existing bank comparison. **~0.5 agent-day**,
   no retraining. It is the only published prior instance of the exact object our bank sweep studies.

**Everything else in this family is a CITATION, not a run.** The full battery still comes from
`adaptive_elicitation_baselines.md` §7: **DRE (static bar), PERE (adaptive bar), GLM-UCB/Thompson over our
belief posterior, EDDI, Golbandi's tree as a genuine selector** — and the standing harness finding that
**Act 1 currently has zero adaptive question selectors** is unchanged and remains the first job.

**Do NOT build:** a meta-learning arm (Buffelli 2023 contests its value; and MAMO/TaNP/CDRNP do not select),
a Bıyık reimplementation (unrefereed preprint, no code, MH/HMC + slate optimiser), or any LLM-in-the-loop
baseline (project rule + circularity ruling).

---

## 8. Verification debts and corrections raised by this pass

**Corrections to existing records — apply these:**
1. `biyik2023soft` INDEX row says the encoder is **co-trained**. **It is not** — CAVs are learned on a frozen
   item tower in an explicitly two-stage design. Re-argue the R1 wedge on *tower strength and the absence of a
   full-profile benchmark*, not on freezing.
2. `gopfert2022discovering` / `gopfert2022soft` INDEX rows say the vocabulary is **tag-genome / Balog scores**.
   **It is not** — it is 164 tags filtered from raw ML-20M tag applications; Balog is cited as motivation only.
3. `biyik2023soft` should be marked **arXiv-only preprint, no venue as of Aug 2026**.
4. `austin2024bayesian` and `austin2024pebol` are the **same paper under two bib-keys** — reconcile.
5. `li2023eliciting` (GATE) — venue is **ICLR 2025**; **388 real participants**; metric is p(correct), no
   ranking metric. The "MIND / 160k articles" detail is **unverified — do not cite**.
6. `critiquing_vocabulary_is_closed.md` — count goes from **8 systems, 0 exceptions** to **10, 0**.

**Still unverified / do not quote:**
- BED-LLM catalogue size (a first pass said "1,682 movies"; **not confirmed**).
- Zawia et al. IEEE Access DOI and page range.
- MirrorBench (arXiv:2601.08118), arXiv:2510.17132, and the 2025–26 simulator-construction papers — [S] only.
- Buffelli 2023 experimental section — abstract only.
- Yoon et al. NAACL 2024 — abstract only; the concrete measured deviations are not yet extracted.
