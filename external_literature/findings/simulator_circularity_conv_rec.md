# Findings: User-Simulator Circularity / Self-Play Validity in Conversational Rec & Elicitation

*Is a user-simulator whose ANSWERS are defined by the recommender's own model/embedding recognized as a
circularity / evaluation-validity problem, and how is it handled?* (Feeds Papers B & C answer-simulator design.)
Web deep-research 2026-07-25. Bib-keys mirror the block appended to `../INDEX.md`. Confidence flags per record.

## The two CASPER simulators under scrutiny
- **(1) GEOMETRIC answer** — for a query, the simulated like/dislike is whichever response moves the recommender's
  BELIEF vector CLOSER to `u* = encoder-fold(known-half profile)`. NOT a held-out-target leak (u* uses only the
  fold-in half), but the simulator's oracle target IS the recommender's own latent estimate; simulator and
  recommender share one latent geometry, and "was this answer good?" is judged in the model's own coordinates.
- **(2) BEHAVIORAL answer (current)** — attribute affinity = NPMI/PMI watch-lift of the attribute's member items in
  the user's actual consumption (implicit) + shrunk residual rating (explicit). Computed from behavior/logs,
  INDEPENDENT of the recommender's embedding.

## Bottom line (see synthesis at end)
The field DOES treat a simulator coupled to / sharing representations with the system under test as a recognized
validity problem — under three named banners: **target-biased-simulator shortcutting** (CRS-specific),
**self-preference / self-recognition bias** (LLM-judge/simulator), and **sim2real calibration + agent–environment
separation** (RecSim). The accepted fix is exactly the behavioral direction: ground the simulated answer in
held-out user behavior / independent labels, not in the agent's own estimate. No source names the precise
"cosine-closeness-to-own-fold" construction verbatim — the mapping to CASPER's geometric simulator is analogical
(strong, but inferred). So simulator (2) is the defensible design; simulator (1) is exposed to a documented class
of criticism and should be used (if at all) only as an elicitation/training device, never as the evaluation oracle.

---

## LEAK / CIRCULARITY side — coupling simulator to the evaluated model inflates results

- **`kim2024targetfree` — Kim et al. 2024, "Stop Playing the Guessing Game! Target-free User Simulation for
  Evaluating CRS" (PEPPER), arXiv:2411.16160.** [full-text HTML, HIGH]
  THE closest CRS-specific analogue. Argues existing **target-biased simulators** — pre-encoded with the target
  item's attributes — let the CRS "take shortcuts to the target items," turning elicitation into a guessing game
  rather than genuine preference discovery. Key: even **attribute-only** priming is contaminating — "providing
  target attributes can still shortcut the recommendation process by implicitly narrowing the candidate space."
  Diagnostic disparity: ChatGPT 0.86 recall on selected/known-target items vs 0.12 on residual items (the
  shortcut signature). Fix = **target-free simulation**: build the simulated user from **general preferences
  derived from real reviews / interaction histories** (behavior-grounded), decoupled from the evaluation target,
  so the target is discovered through authentic elicitation. Simulator = INDEPENDENT (data-derived) by design.
  → Direct support that binding the simulator to information the recommender will exploit is a validity flaw, and
    that the fix is behavior-grounded, target-decoupled answers. Maps onto CASPER: geometric-fold answers are a
    softer form of the same coupling (target = model's own estimate, not held-out item, but still model-internal).

- **`panickssery2024self` — Panickssery, Gao et al. 2024, "LLM Evaluators Recognize and Favor Their Own
  Generations," NeurIPS 2024, arXiv:2404.13076.** [full-text PDF, HIGH]
  THE strongest shared-representation-bias source. An LLM scores its OWN outputs higher than others' while humans
  rate them equal; establishes a causal link from **self-recognition → self-preference**. Mechanism = generator
  and evaluator operate in the **same learned representation**, so the evaluator "recognizes" and rewards patterns
  it produced. Cross-domain (LLM-judge, not recsys) but it is the canonical citation for the abstract failure mode
  CASPER's geometric simulator instantiates: an oracle that scores answers by closeness to the SAME model's latent
  will systematically favour that model. Simulator = MODEL-COUPLED = biased.

- **`wei2025selfpref` — "Do LLM Evaluators Prefer Themselves for a Reason?" 2025, arXiv:2504.03846.** [abstract
  only, MEDIUM; authorship unverified] Follow-up probing whether self-preference tracks genuine quality vs bias.
  Secondary cite reinforcing that same-family simulator↔evaluator coupling is an active, named concern.

- **`wang2023rethinking` (in bib) — Wang et al. 2023, iEvaLM, EMNLP.** Static one-turn matching under-measures
  interactive CRS; argues for interactive/simulator evaluation — but the flip side (a coupled simulator inflates)
  is precisely what the target-free line then flagged. Use for the "static is inadequate → simulate, but simulate
  independently" framing.

- **`bernard2025limitations` — Bernard & Balog 2025, "Limitations of Current Evaluation Practices for CRS and the
  Potential of User Simulation," SIGIR-AP 2025, arXiv:2510.05624.** [abstract/metadata, MEDIUM]
  Documents a "striking disconnect between self-reported user satisfaction and performance scores reported in prior
  literature" across 9 CRSs; positions user simulation (with reward/cost metrics) as the fix. Does NOT explicitly
  treat simulator–model coupling/leakage (not in the available text) — cite for "reported scores ≠ real
  satisfaction," not for the circularity claim itself.

- **(context, in bib) `yoon2024` — Yoon et al. NAACL'24** — LLM-simulated users are unfaithful to the profiles
  they're given; behavioral grounding beats stated/LLM answers. Directional support for simulator (2).

---

## ACCEPTABLE / LEGITIMATE side — a generative/model user-state is fine IF separated + calibrated to data

- **`ie2019recsim` — Ie et al. 2019, "RecSim: A Configurable Simulation Platform for Recommender Systems,"
  arXiv:1909.04847 (Google).** [blog+abstract, HIGH on architecture]
  The reference architecture legitimizes a **latent user-state generative model** as ground truth — but with two
  guardrails CASPER's geometric simulator violates: (i) the environment (user model + choice model) is
  **structurally SEPARATE from the agent**; RecSim deliberately "does not provide the learning algorithms," and you
  configure "which parts of the user state are observable to the recommender." The oracle user state is NOT the
  agent's own estimate. (ii) Fidelity requires calibration: the "sim2real gap can only be achieved by calibrating
  the simulator's model to observed data." → Legitimacy of a model-defined user state is conditional on
  independence-from-agent + calibration-to-behavior — both of which the behavioral simulator honours and the
  geometric one does not.

- **`mladenov2021recsimng` — Mladenov et al. 2021, RecSim NG, arXiv:2103.08057.** [abstract, HIGH]
  Successor; formalizes fitting the latent-user generative model to real observational data ("run in reverse" via
  EM / adversarial training). Reinforces: a model-based user is acceptable when it is an INDEPENDENTLY-FITTED
  generative process, not a mirror of the recommender's encoder.

- **`hsu2024minimizing` — Hsu et al. 2024, "Minimizing Live Experiments… User Simulation to Evaluate Preference
  Elicitation Policies," arXiv:2409.17436 (Google/YouTube Music).** [abstract, MEDIUM]
  Builds "counterfactually robust user behavior models" fit to usage logs and **validated to predict live
  metrics** before trusting them to rank elicitation policies. The accepted-practice template: simulator fit to
  independent behavior + externally validated, then used to compare policies. (Full methodology behind PDF; the
  independence-from-policy detail is implied by log-fitting + live-validation, not quoted — flag MEDIUM.)

- **`luo2020llc` — Luo et al. 2020, "Latent Linear Critiquing for CRS," WWW 2020** (+ RankCrit RecSys'20;
  M&Ms-VAE `antognini2021mmvae`; BK-VAE `luo2021bkvae`). [snippet+thesis, MEDIUM-HIGH]
  Critiquing evaluators simulate a user who iteratively critiques keyphrases drawn from a **randomly selected
  target/liked item's ACTUAL review keyphrases** — i.e. the critique labels are DATA-derived (real review
  vocabulary of a held-out liked item), then folded into the model's latent. So even the critiquing line, which
  co-embeds critiques in the model space, sources the SIMULATED ANSWER from independent item-level labels, not from
  the model's own latent estimate of the user. This is the important distinction for CASPER: co-embedding the
  answer channel in the model space is fine; DEFINING the answer's correctness by closeness to the model's own
  user-estimate is the step that crosses into circularity.
  (Caveat: BK-VAE/BCIE fold CAV *directions* that are learned from the item space — the CHANNEL is model-derived,
  but the per-turn ANSWER/critique is still item/label-driven. Medium confidence this holds across all variants.)

- **(in bib) classic CRS elicitation — `christakopoulou2016towards`, `lei2020estimation` (EAR),
  `lei2020interactive` (SCPR), `deng2021unified` (UNICORN), `li2021seamlessly` (ConTS).** Standard practice: the
  simulated user answers from the **held-out ground-truth ratings / attribute labels** of the target user, i.e.
  behavior/label-derived, independent of the policy's own belief. This is the field default the behavioral
  simulator matches.

---

## Mapping to CASPER's two simulators

| | Geometric answer (1) | Behavioral answer (2) |
|---|---|---|
| Oracle target | recommender's own `fold(known-half)` u* | held-out watch-lift (NPMI) + residual rating |
| Coupling to evaluated model | SHARED latent geometry (soft self-play) | INDEPENDENT of embedding |
| Nearest recognized critique | target-biased shortcut (PEPPER); self-preference/shared-rep bias (Panickssery) | — (this IS the accepted fix) |
| RecSim guardrails met? | fails agent–environment SEPARATION; "calibration" is to the model, not data | meets both (independent + behavior-grounded) |
| Verdict | acceptable ONLY as an elicitation/training signal, NEVER the eval oracle | defensible evaluation oracle |

## Caveats / confidence
- No source names the exact "answer = argmax cosine to the recommender's own fold" construction; the geometric↔
  self-preference/target-biased mapping is ANALOGICAL (strong but inferred). Confidence: MEDIUM-HIGH on the
  principle, MEDIUM that a reviewer will accept the analogy without the author pre-empting it.
- `bernard2025limitations`, `hsu2024minimizing`, `wei2025selfpref`: read from abstract/metadata only — the
  circularity-specific claims are NOT directly quotable from those; do not over-attribute.
- PEPPER (`kim2024targetfree`) + Panickssery (`panickssery2024self`) are the two load-bearing, full-text-verified
  citations — lead with them.
