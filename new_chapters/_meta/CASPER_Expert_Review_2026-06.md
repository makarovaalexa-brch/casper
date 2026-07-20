# CASPER — Independent Expert Review
**Date:** 2026-06-11 · **Scope:** full codebase, experiments V1–V4, diagnostics, lit review, supervision documents

---

## Executive Summary

Your engineering is good and your self-diagnosis is honest, but the project has **three structural flaws, not small bugs**. Together they make the current formulation close to unlearnable, which is why every fix has failed:

1. **The sample budget is off by 2–3 orders of magnitude.** You are training DDPG — a notoriously sample-hungry, brittle algorithm — with a 384-dimensional continuous action space on **~1,000–2,500 transitions** (90–250 episodes × ~10 turns). Published DDPG results need 10⁵–10⁶ transitions on far easier problems. The LLM-in-the-loop makes episodes cost money and hours, capping you at a data scale where *no* RL algorithm could succeed. The V3 "fixed policy" outcome is not a bug — it is the expected behavior of a deterministic policy trained on this little data from identical initial states.

2. **The reward channel is information-theoretically broken.** Your own diagnostic proves it: per-turn NDCG deltas average ~0.005 against per-user noise of ~0.20 (SNR ≈ 0.025), and the pool=100/min_targets=25 evaluation design gives a 0.52 baseline with no headroom. Also, NDCG-delta rewards telescope — the episode return is just (final NDCG − baseline), so the critic must do per-turn credit assignment through noise 40× larger than the signal. No amount of reward shaping fixes this.

3. **The like/dislike problem is a representation-choice error, not a hard blocker — and you already have the working answer.** Trying to make SBERT encode polarity inside text ("likes: X | dislikes: Y") cannot work; SBERT measures topical similarity. But your own unified evaluation shows the CSMAI-19-style model — SBERT for *content*, a separate one-hot channel for *polarity* — improves monotonically (71.5% → 75.5%) while the concept model plateaus. The contrastive fix collapsed (98% overlap, embedding std 0.0006) because during training the rating one-hot was effectively constant (positives drawn only from liked items) and 3 dims were drowned by 384. Keep polarity **structural**, never textual.

**The hard message:** the headline novelty — a continuous 384-dim action space — is the main thing making the problem unlearnable, and it is partly illusory: the action is immediately snapped to a discrete entity via FAISS, so the *effective* action space is discrete anyway, while the continuous formulation adds action aliasing (many actions → one entity) that makes the critic's job nearly ill-posed. Be prepared to reframe this contribution.

**The good news:** a publishable paper is reachable with the components you already have, if you (a) fix the recommender with polarity-as-structure, (b) take the LLM out of the training loop so episodes are free, (c) simplify the RL formulation, and (d) add the baselines reviewers will demand. Detailed plan below.

---

## 1. What is solid

- Clean modular architecture: preference extraction, FAISS entity mapping, GPT question generation, user simulator with tool-calling all work.
- The diagnostic work (`experiments/diagnostics/DIAGNOSTIC_REPORT.md`) is genuinely good science — it isolated the reward-signal failure quantitatively. Most students never do this.
- The research gap is still defensible: 2024–25 literature (and your monitoring report) agree that **preference elicitation is a documented weakness of LLM-based CRS**. Multi-agent/RAG trends changed the landscape, but nobody has solved *learned* elicitation strategy — your niche survives if you can show a learned policy beating non-adaptive baselines.
- Infrastructure (parallel trainer, checkpointing, logging, $1.76/run cost tracking) is publication-grade plumbing.

## 2. The major flaws, in causal order

### Flaw A — Sample budget vs. problem size (root cause of "fixed policy")
DDPG learns a *deterministic* policy. At turn 1 every user produces the same (near-empty) state, so a fixed opener is mathematically the optimal deterministic response — "archaeology 40%" is the algorithm working as designed. Adaptivity can only emerge from turn 2 onward, conditioned on responses, and only with enough data to estimate Q over a 768-dim (state ⊕ action) input. You have ~1k–2.5k transitions; the 10k replay buffer never even fills. Exploration noise decaying 0.20 → 0.14 over 90 episodes cannot rescue this.

### Flaw B — Reward channel
- Baseline NDCG 0.52 (pool=100, min_targets=25) ⇒ ~39% of the pool is ground truth; popular picks win by chance; revealing preferences often *lowers* NDCG (V1's negative rewards).
- Per-step delta ~0.005, per-user std ~0.20. Even your improved concept-accuracy reward (SNR 1.69 vs NDCG's 0.81) is still being consumed by a per-turn actor-critic that must decompose a telescoping return.
- Secondary code issues that compound it: reward-std EMA can decay toward zero and blow up normalized rewards (`rl_actor_critic.py:308`); silent preference-extraction failures return empty dicts (`preference_extractor.py:164`); `self.device` is undefined in `load_weights` (`rl_actor_critic.py:508` — crashes on resume).

### Flaw C — Polarity representation
Putting likes and dislikes in one SBERT string conflates them by construction (0.75–0.78 similarity). All three failed fixes share the same root:
- **Signed encoding** (likes − dislikes): destroys content information; NDCG 0.28.
- **Contrastive training**: collapsed (98% overlap) because positives came only from liked items — the rating channel carried no variance during training — and a 3-dim one-hot cannot compete with a 384-dim embedding without architectural support (e.g., FiLM-style gating or separate towers).
- **Similarity penalty**: addressed a different problem (within-episode repetition) and worked for it; it was never going to fix cross-episode convergence (Flaw A).
Your own data shows the fix: one-hot polarity channel + content embedding, trained with *both* polarities represented in the input sequences, improves monotonically.

### Flaw D — Action space design
The continuous action is L2-normalized, FAISS-snapped to the nearest entity, filtered, then verbalized by GPT. The decision that matters is "which entity to ask about" — a discrete choice over a tag vocabulary. The continuous wrapper (the paper's novelty) gives no gradient through the snap, creates aliasing, and multiplies sample complexity. A discrete stochastic policy (with entropy regularization, which also fixes the diversity problem properly) over a few hundred clustered concepts is a dramatically easier learning problem with the same expressiveness in practice.

### Flaw E — Evaluation and positioning gaps
No pure-LLM baseline, no greedy information-gain baseline, no human study. The single biggest *publication* risk: a non-learned greedy info-gain policy (pick the entity with highest expected accuracy gain under the concept model) may match or beat RL. You must know this before writing the paper, because it determines what you can claim.

## 3. Path forward (ordered by dependency — each step gates the next)

**Step 1 — Fix the user-state/recommender model offline (highest priority, fully supervised, no LLM cost).**
Architecture: per-item SBERT content embedding ⊕ rating one-hot → LSTM+attention (you have this built). Critical training changes: include disliked items as *inputs* with their true one-hot (so the rating channel has variance), balanced sampling, and consider gating (rating modulates the embedding multiplicatively) rather than concatenation so 3 dims can't be ignored. **Define acceptance tests before training:** (i) flipping one rating flips the ranking direction (your Star Wars sanity check), (ii) like-vs-dislike recommendation overlap < 30%, (iii) monotonic accuracy improvement with sequence length. Your unified-evaluation numbers say this is achievable.

**Step 2 — Take the LLM out of the training loop.**
Build a rule-based simulator that answers entity questions directly from the MovieLens profile (rating lookup + genome-tag matching). Deterministic, free, instant. Train on 50k–500k episodes; reserve the LLM simulator (and later humans) for *evaluation only*. This single change closes the 100× sample gap and removes LLM non-stationarity from the MDP.

**Step 3 — Simplify the RL formulation.**
- Reward: concept-model accuracy delta (your V4 direction — correct instinct, right reward, wrong policy class) + discovery bonus.
- Action space: discrete over a clustered concept vocabulary (~200–500 genome-tag clusters).
- Policy: stochastic with entropy bonus (PPO, or even REINFORCE+baseline to start). Entropy regularization solves cross-episode diversity *properly*, unlike penalties.
- **Run a contextual-bandit (myopic) version first.** Because rewards telescope, the sequential aspect may matter less than assumed. If the bandit works, "when does lookahead help elicitation?" becomes a clean, publishable research question — and the full RL becomes an ablation rather than a prerequisite.

**Step 4 — Baselines (non-negotiable for publication).**
Random entity; popularity-ordered entities; **greedy information gain** (the one that can kill or crown the paper); pure-LLM question asking (GPT decides what to ask, no policy). Your contribution claim is whatever the learned policy beats.

**Step 5 — Fix the evaluation protocol.**
Pool = 500, min_targets = 5–10 (your diagnostics show this restores headroom: baseline ~0.26, +0.10 achievable), n ≥ 100 held-out users, report CIs. Pre-register the metrics.

**Step 6 — Paper strategy.**
- *Primary target:* "Learned preference-elicitation policies outperform non-adaptive baselines in conversational recommendation" — RecSys (short or full), UMAP, ECIR, CIKM.
- *Hedge:* if RL only matches greedy info-gain, the diagnostic story you already have ("why per-turn ranking rewards are unlearnable; what reward signals work") is an honest, useful analysis paper — RecSys/SIGIR workshops (KaRS, IntRS) take exactly this.
- Drop or soften the "continuous action space" framing; "embedding-space policy with discrete concept grounding" is defensible and true.

**Rough sequencing:** Step 1 ≈ 2–3 weeks; Step 2 ≈ 1–2 weeks; Steps 3–4 ≈ 4–6 weeks of cheap iteration once episodes are free; evaluation + writing ≈ 6–8 weeks. A submission by late 2026 is realistic *if Step 1's acceptance tests pass*; if they don't, pivot to the hedge paper immediately rather than iterating further on the recommender.

## 4. Small errors worth fixing regardless
- `rl_actor_critic.py:508` — `self.device` undefined; crashes checkpoint resume.
- `preference_extractor.py:164` — JSON failures silently return empty preferences; log them.
- `rl_actor_critic.py:308` — reward-std EMA can collapse; use Welford for std too.
- Noise-then-clip-then-normalize in action selection distorts exploration; add noise after normalization, renormalize.
- Parallel trainer exists but the main notebook trains sequentially — free 10× speedup unused.
