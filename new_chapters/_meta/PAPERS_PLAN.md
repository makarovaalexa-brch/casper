# Two-Paper Sprint Plan (10 days, June 2026)

**Machine verdict:** no Colab needed. i7-7820HQ (4C/8T), 16GB RAM, PyTorch 2.2.2 CPU, sbert 2.7.0, faiss, openai 1.109.1 all working; MovieLens-25M + cached SBERT embeddings on disk. The LSTM recommender at this scale trains in tens of minutes on CPU. Watch: only 27GB disk free.

---

## Paper 1 (testbed) — keeps the CASPER name via re-backronym

**Title:** *CASPER: Controlled Assessment of Strategic Preference Elicitation in Conversational Recommendation*

**Proposed abstract (draft):**
> Conversational recommender systems are increasingly built around large language models, yet their defining capability — strategically eliciting user preferences — is asserted more often than measured. Published evaluations embed elicitation inside whole-system comparisons, where its effect is confounded with recommendation and generation quality, and the candidate approaches now in play (graph-pruned Q-networks, Bayesian acquisition, prompted LLMs, learned policies) have never been compared on equal terms. We introduce CASPER, a controlled testbed that isolates elicitation strategy: a fixed, polarity-aware recommendation model serves as the measurement instrument, simulated users answer verifiably from held-out MovieLens ground truth, and policies are scored on recommendation-relevant information gained per conversational turn. We first show that instrument design is decisive — under common configurations (small candidate pools, popularity-saturated targets, polarity-conflating user representations), per-turn reward signals are an order of magnitude smaller than their noise, rendering both learning and comparison meaningless. We then benchmark eight elicitation policies spanning random and popularity baselines, greedy information gain, prompt-only LLM questioning in several styles, and a reinforcement-learned bot-play policy. [Results sentence — to be filled: e.g., "Prompt-only LLMs underperform greedy information gain by X% at equal turn budgets, while exhibiting strong opening-question diversity; learned policies..."] CASPER's protocol, simulator, and baselines are released to enable measurement-grounded progress on strategic elicitation.

**Venue targets:** RecSys 2026 (short or resource), ECIR 2027, UMAP 2027, CIKM 2026 resource track. Workshops fallback: KaRS, IntRS.

**To-do (P1):**
- [ ] **M1 — Measurement instrument.** Train polarity-aware recommender: SBERT content embedding ⊕ rating one-hot per item, LSTM+attention, *disliked items included as inputs*, gated fusion so the 3-dim rating channel can't be ignored. Acceptance tests defined before training: (a) like-vs-dislike recommendation overlap < 30%; (b) accuracy improves monotonically with number of revealed preferences; (c) flipping one rating flips the ranking direction (Star Wars test). **Fallback:** existing one-hot model (already passed monotonicity 71.5→75.5%).
- [ ] **M2 — Harness.** Pool=500, min_targets=5–10, n≥150 held-out users, fixed seeds, per-turn accuracy curves with CIs; rule-based simulator answering entity questions from ground-truth profile (genome-tag match + rating lookup) — verifiable, free, deterministic.
- [ ] **M3 — Policies.** random | popularity | greedy expected-info-gain (one-step lookahead under the instrument) | prompt-only LLM askers ×3–4 (vanilla, CoT-strategist, GATE-style open interview) | IJCNN bot-play policy (retrained small REINFORCE if checkpoint unusable).
- [ ] **M4 — Experiments.** Main comparison; ablations: turn budget (5/10/20), pool size (100 vs 500 — reuses the V1–V4 broken-instrument evidence), instrument ablation (polarity-aware vs SBERT-conflating). Figures + stats.
- [ ] **M5 — Draft.** 8–10pp.

**Cost:** training = recommender only (~20–60 min/run CPU, several runs); API ≈ $5–30 (gpt-4o-mini askers).

---

## Paper 2 (method) — original CASPER meaning preserved

**Title (2a primary):** *Learning What to Ask: Continuous Semantic Action Spaces for Preference Elicitation in Conversational Recommendation*
(System name in-paper: **CASPER-CA** — Continuous-Action Strategic Preference Elicitation via Reinforcement — "the continuous-action extension of the CASPER testbed's bot-play family".)

**Proposed abstract (2a draft):**
> Reinforcement-learned preference elicitation in conversational recommendation has been confined to discrete action spaces enumerating catalogue attributes, bounding both what can be asked and how policies generalise across related questions. We extend the Wolpertinger architecture — continuous proto-actions grounded to discrete actions by nearest-neighbour search with critic re-ranking — to elicitation question selection: the policy emits a point in a pretrained sentence-embedding space, candidate concepts are retrieved by approximate nearest-neighbour search, the critic selects among them, and a language model verbalises the chosen concept as a natural question. We show that the naive configuration (single-neighbour snapping, language-model-in-the-loop training, ranking-delta rewards) is unlearnable for quantified reasons — action aliasing, two-orders-of-magnitude sample deficits, and reward signal-to-noise below 0.05 — and that each defect has a principled repair: critic re-ranking over k candidates, a verifiable programmatic simulator enabling 10^5 free episodes, and an accuracy-delta reward with SNR > 1.5. On the CASPER testbed, the repaired continuous policy [results: vs discrete PPO over the same concept vocabulary, vs greedy information gain, vs prompt-only LLMs]. Our ablation isolates when the continuous formulation pays for its sample cost — [finding].

**Title (2b fallback/companion):** *Verbal Bot-Play: Improving LLM Elicitation Strategies without Gradient Training*
(System name: **CASPER-V**.) QBot strategy prompt is iteratively rewritten from reward-scored episodes (Reflexion/OPRO-style verbal reinforcement on the same bot-play loop and reward as the IJCNN paper).

**Decision rule (from Paper 1 leaderboard):** greedy ≫ prompt-LLM → 2a is the story (learned policy chases greedy's ceiling with open vocabulary). Prompt-LLM ≈ greedy → 2b is the story (cheap optimisation of an already-strong asker); 2a becomes ablation/negative-result section. Either way both codebases get pre-written this sprint.

**To-do (P2):**
- [ ] Rule-based simulator (shared with P1-M2, parameterised for training speed).
- [ ] 2a trainer: TD3-style actor-critic, k-candidate critic re-ranking, executed-entity embeddings in replay, accuracy-delta reward, entropy-regularised discrete PPO comparator over clustered concept vocabulary (~300 genome-tag clusters).
- [ ] 2b loop: episode batch → score → strategy-prompt rewrite → repeat; prompt archive + best-of selection.
- [ ] Smoke tests for both (100-episode runs overnight on this machine).
- [ ] Paper 2 skeleton: abstract, intro, method written; results sections templated against the testbed's output format.
- [ ] REASONING.md (design rationale for every choice, for Opus continuation) + HANDOFF.md.

---

## Realistic timeline (calibrated to actual working speed)

| Day | Deliverable |
|---|---|
| 1 (today) | Plan ✓, M1 code + training launched, acceptance tests written |
| 2 | M1 verdict (retrain if needed), M2 harness, M3 policies coded |
| 3 | M4 experiments run (API + CPU), figures; M1 fallback decision final |
| 4 | Paper 1 full draft → user review |
| 5 | P2a trainer + simulator code, smoke run launched overnight |
| 6 | P2b loop code + smoke; Paper 2a skeleton draft |
| 7 | Paper 1 revisions; 2a/2b smoke results → decision-rule memo |
| 8 | Paper 2 drafts as far as results allow; REASONING.md |
| 9 | HANDOFF.md, repo hygiene, everything committed + push instructions |
| 10 | Buffer (reviewer-proofing Paper 1, supervisor summary) |

Waiting is never blocking: training runs go to background; LLM experiments are minutes, not hours. If any milestone stalls > half a day, its fallback activates (documented per-milestone above).
