# Findings: Continuous-Action Elicitation Policy

*PEBOL, HyAR, Wolpertinger, DBU, continuous-vs-discrete action RL, embedding→NL inversion.* (Paper C spine.)
Consolidated & deduped from paperC-continuous-deep-research, continuous-ndcg-policy-research, NOVELTY_CLAIMS, LitReview_Verification. Bib-keys in `../INDEX.md`.

## What we concluded

- **CLEANEST FRAMING:** the FIRST unified continuous-action elicitation policy that warm-starts/distills a discrete teacher via Wolpertinger/HyAR-style action representations over a FROZEN unified item∪concept space, closing the loop to open-vocab NL via embedding inversion. The load-bearing novelty is the **NOVEL-DIRECTION capability** — propose a taste direction that NO existing attribute/item name represents — not open-vocab generation (PEBOL/GATE have it) nor principled selection (PEBOL has it).
- **The snap CANNOT fail** — nearest-neighbour into a bank of REAL phrases means every point in R^d maps to a valid, askable question. So there is no snap-VALIDITY problem (unlike HyAR/VQ whose learned decoder can emit garbage), only a snap-FIDELITY one. → No consistency constraint is needed; any commitment/consistency loss is DIRECTLY ADVERSARIAL to the thesis (it optimizes away the quantity we claim is valuable).
- **Justify the snap on ASKABILITY/ANSWERABILITY, not on "no decoder exists."** 2025 zero-shot inverters (ZSInvert, vec2vec) claim encoder-agnostic recovery, so "no decoder for all-MiniLM" is now RISKY. Bank phrases are real, calibrated, known-answerable; an inverted string is an uncontrolled question of unknown answerability — the stronger argument anyway.
- **DROP "lossless / without loss" language.** Chandak 2019's "lossless value" Theorem 1 was REFUTED in verification. Frame discrete→continuous transfer as knowledge-PRESERVING warm-start/distill.
- **Continuous-with-LLM-in-the-loop at ~100 episodes is unlearnable** (proven by CASPER V1–V4: LLM capped training at ~1-2.5k transitions; k=1 FAISS snap aliased actions; reward SNR ~0.025). Continuous done Wolpertinger-style on FREE rule-based-simulator episodes (50k+) is a fair experiment — report continuous-vs-discrete-PPO as an explicit ablation.
- **A differentiable amortized 1-step selector trains cleanly and beats the champion** (DKQS q2: +0.0197 vs var-greedy +0.0186 vs static +0.0171; answer-contingent prize +0.0054, passes perm-null) — exact conjugacy makes elicitation differentiable in the query (Biyik can't). BUT the T=10 UNROLL COLLAPSES to an input-independent, worse-than-greedy static sequence — a TRAINER failure (conditioning-collapse + gradient-starvation + relaxation≠hard-optimum + BPTT credit-starvation), NOT an adaptivity verdict.

## Key external methods

- `dulacarnold2015deep` (Wolpertinger) — DDPG proto-action + kNN-snap + critic re-rank; demonstrated on rec with 13k actions. k=1 snap FAILS → retrieve k>1 (~5% of bank), critic re-rank, train Q on the EXECUTED entity's embedding (not the pre-snap proto-action). THE mechanism ancestor.
- `wang2024efficient` (ECoC) — continuous-action RL for sequential rec, items-only, non-conversational. Honest prior-art.
- `lillicrap2015continuous` (DDPG), `fujimoto2018addressing` (TD3), `haarnoja2018soft` (SAC), `mnih2015human` (DQN), `williams1992simple` (REINFORCE) — continuous/deep RL toolbox.
- `austin2024bayesian` (PEBOL) — BO acquisition over Beta posteriors + LLM aspect extraction + NLI; strong cold-start (MRR@10 0.27 vs 0.17, +131% MAP@10). THE PE baseline; proves selection+LLM-phrasing works → sharpens "why decode a lossy embedding when an LLM phrases for free?" → answer = novel-direction.
- `li2023eliciting` (GATE) — LM generates open-ended questions by prompting; topic-level, binary target; no continuous actor, no unified CF space.
- `deng2023plug` (PPDPP), `zhao2025reinforced` (RSO) — policy-chooses/LLM-speaks over CLOSED discrete strategy sets; RSO is the nearest rival (13 strategy types, not semantic content).
- `li2021seamlessly` (ConTS) — unifies attributes+items as arms but FIXED set + DISCRETE argmax.
- `makarova2024learning` — the author's own discrete bot-play policy (self prior-art; scope out of any "first").
- **Not in references.bib — must add:** HyAR (Li, ICLR 2022, closest decodable-hybrid template: embedding table + cVAE, argmin-L2 snap; LSC = OUTPUT BOX CONSTRAINT not a regularizer, RSC = replay-buffer relabeling — a HyAR-literate reviewer will catch "LSC/RSC penalize off-manifold" as a factual error); **DBU / "Breaking the Grid" (arXiv:2602.08616)** = train the actor by regressing onto a Q-softmax-weighted BLEND of retrieved candidates (all gradient from executable evaluated phrases, target generically off-catalog → teaches landing BETWEEN phrases; SAME object as Paper E's OMP sparse-blend renderer — "ask the blend, not the nearest phrase"); Chandak 2019 (ICML, action representations from dynamics; "lossless" refuted); Vec2Text/GEIA/DeCap/ZSInvert/vec2vec (embedding→NL inverters); P-DQN/Hybrid-SAC (scale poorly to ~1361 entities → background only); **Huang et al. NeurIPS 2024 (arXiv:2411.02064)** amortized BED for decision-making (Transformer NDP trained on downstream DECISION UTILITY not EIG — near-blueprint, substitute utility:=NDCG); **Blau et al. ICML 2022 (arXiv:2202.00821)** sequential-BED-as-MDP-by-deep-RL (best fit for NDCG-as-reward, handles black-box); DAD (Foster ICML'21), iDAD, Step-DAD; NeuralNDCG (Pobrotyn 2021), LambdaLoss; GSMRL (intermediate rewards), DAgger (Ross AISTATS'11).

## What is NOVEL vs pre-empted

**PRE-EMPTED / not novel as machinery:**
- Continuous proto-action + NN-snap = Wolpertinger (even on rec).
- Open-vocab NL question generation alone = GATE (direct LLM prompt) + PEBOL (BO-select-then-LLM-phrase) + 2510.12015 (diffusion-framed clarifier). None use a continuous embedding actor, a unified item∪concept CF space, or embedding→NL inversion.
- Decodable hybrid discrete+continuous = HyAR/P-DQN/Hybrid-SAC.
- Training acquisition on downstream task utility (not info-gain) = TNDP/DUG (Huang NeurIPS'24), ALINE 2025. Amortizing acquisition into one forward pass = DAD (Foster ICML'21). Value-function-argmax = fitted-Q. **A reviewer WILL say "this is DUG applied to recommendation" — do not frame as "we invented a decision-theoretic policy."**
- Unified attributes+items action space = ConTS (fixed + discrete argmax).

**NOVEL / defensible = the CONJUNCTION of three, for elicitation (no surveyed work does all three):**
1. CONTINUOUS actor (not discrete argmax),
2. OPEN-VOCAB unified item∪concept space (not a fixed candidate set),
3. decode-to-NL (embedding inversion) — earning its keep ONLY via the NOVEL-DIRECTION it can propose.
- **A continuous query as an INPUT TOKEN** in the same entity space as items and concepts — the sweep found NOTHING, anywhere (not steering a latent, not generating one, not a retrieval query). The cleanest, strongest leg of C1.
- Off-catalog emission as a PRIZE (not a bug to suppress) appears UNCLAIMED — but means no borrowable solution exists; we must supply one.
- The delta over TNDP/DAD = the APPLICATION + the ACTION SPACE + the BOUNDARY, not the estimator. TNDP/ALINE are BED, not recsys — no catalog, no collaborative prior, no answerability, no refusals, no strong recommender to beat.

## Open questions

- **Continuous-vs-discrete oracle HEADROOM first:** does a continuous oracle (best point in embedding space, unrestricted to the 1361 pool) beat the discrete best-subset oracle on full/tail NDCG@10? If not, Paper C is a capability/framing story, not a performance story — know this before investing in a decoder.
- **Is the snap top-1 an artifact?** Wolpertinger k=1 fails on a 13k-action recommender; run proper k~120 + critic re-rank BEFORE defending the snap-loss number.
- **Densify the bank 10×** (composed phrases): if snap loss COLLAPSES, continuity is merely a COVERAGE claim (weak); if it PERSISTS, the prize is about directions NO PHRASE NAMES (strong).
- **Fix the T=10 trainer collapse:** highest-leverage = BC/DAgger distillation from the tree/greedy teacher (guarantees ≥ greedy-static, seeds out of the collapse basin) + fine-tune with straight-through Gumbel top-1; then dense per-step reward (Blau); entropy reg as band-aid only. Start Family 2 (Blau MDP + Wolpertinger snap) warm-started by Family 5 (distil the NDCG-tree), NeuralNDCG-through-Kalman as a low-variance aux.
- **The snap-loss lam-sweep** should show a NON-MONOTONE curve with an INTERIOR optimum (precedent PLAS eps=0→44.6, 0.1→66.9, 0.5→39.2) — run only as pre-registered, never as a fixed constraint.

## Bibliography / factual fixes (binding)
- HyAR LSC = output box constraint; RSC = replay-buffer relabeling. NOT loss-term regularizers.
- The snap tension DISSOLVES: our NN snap can't fail → no consistency loss.
- Do NOT compare CASPER's 0.4852 to SASRec/BERT4Rec's ~0.15 — INCOMPATIBLE protocols (leave-one-out vs strong-generalization fold-in). If needed, run SASRec ON OUR RULER as a baseline row.
