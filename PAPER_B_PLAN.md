# Paper B — Adaptive, answerability-aware, continuous-action concept elicitation

> Status: motivation FINALISED (this doc); detailed technical plan being grounded by a deep-research pass on
> continuous-action RL for elicitation (Wolpertinger / offline-to-online / CRS-RL). Recommender foundation (Paper A)
> is LOCKED: calibrated reconstruction encoder = EXPO exposure-propensity negatives + contrastive value-channel
> polarity (CL=0.2), `data/.cache/enc_unified.pt`. Numbers below are committed in RESULTS.md (PARTS A–O).

---

## 1. Motivation (the long version) — why a continuous concept policy is the right (and novel) object

Paper A delivers the **recommender-side foundation** and, in doing so, *quantifies* the open problem. Four findings
compose into the mandate for Paper B.

### 1.1 The instrument is solved; the open problem is SELECTION
The calibrated encoder converts answers into rankings well: beats the closed-form ridge fold-in on the full catalogue
(q8 0.339 vs 0.305) **and** the long tail (0.116 vs 0.098), in NDCG and Recall, audited by interpretable probes
(genre purity 48%, like/dislike polarity +42pp). "How to *use* an answer" is no longer the bottleneck. What remains is
**which question to ask** — selection.

### 1.2 The selection headroom is large, and only ADAPTIVITY captures any of it
On the canonical encoder (held-out, full + long-tail):
- Static heuristics (random / popularity / entropy / HELF / RMVA) **cluster** (~+0.04 full, ~+0.06 tail): *which*
  item you ask barely matters if the rule is non-adaptive — the recurring negative across the whole project.
- A realizable greedy **information-gain (EIG)** policy is the **first** selector to beat random/static (full +0.079,
  tail +0.127) and is genuinely **deployable** (belief-only; answer look-ahead was worth ~0).
- The privileged **oracle** (best subset) is far higher: **full +0.273, tail +0.327**. EIG captures only ~30–40% ⇒ a
  better-than-greedy policy is worth a lot = the Paper B prize.
- Ceiling reframing: 8 *adaptive* questions on the tail (0.194) exceed folding the **entire** profile (0.157).
  Elicitation is fundamentally a **selection** problem.

### 1.3 The action space must be CONCEPTS, not items — answerability × information (PART O)
| type | answer-rate if asked at random | info per answered Q (oracle NDCG@10, q8 full / tail) |
|---|---|---|
| items | **2.1%** | 0.488 / 0.356 (highest) |
| genres | **63%** | 0.347 / 0.157 (coarse, saturates by q4) |
| concepts | **57%** | 0.479 / 0.331 (≈98% of item info) |

- **Items**: maximally informative but **practically unanswerable** (~30× less answerable than concepts) → value
  unreachable in cold start.
- **Genres**: answerable but **coarse** (18 directions, plateaus).
- **Concepts**: **answerable AND nearly item-level informative** — realizable value (info × answerability) favours
  concepts decisively. But the concept space is **large, open, continuous** → cannot be enumerated → must act **in the
  shared embedding directly**. This is the *structural* reason a continuous-action policy is necessary.
- Honest caveat: in PART O concepts are second-class **centroids** folded by the *linear* instrument; making them
  **first-class** can only raise their value (coupled novelty below).

### 1.4 Why this is novel
- Discrete-attribute CRS-RL (EAR/SCPR/UNICORN/ConTS): small **discrete** action set, cannot ask open concepts.
- Wolpertinger (continuous-action RL + kNN): generic control, never applied to elicitation, not answerability-aware.
- Active feature acquisition / EDDI / PEBOL / GATE: greedy/Bayesian over fixed feature sets; no continuous concept
  policy.
- **CASPER** = one **shared embedding** that is simultaneously the recommender AND the **continuous action space** of
  an **answerability-aware** policy asking **open concepts**, trained to beat greedy EIG and close the oracle gap.

### 1.5 The two coupled novelties Paper B must deliver
1. **Continuous-action concept policy**: actor emits an embedding point → snap to nearest **answerable** concept
   (Wolpertinger); reward = downstream long-tail recommendation quality. Gate: beat deployable EIG.
2. **First-class concepts**: co-factorise items+attributes+concepts into the shared space (remove the centroid
   appendage; PART K) so policy *acts* and recommender *folds in* one trained space; genome/free-text concepts native.

---

## 2. Recommender foundation (DONE — reused, frozen)
- Calibrated encoder (EXPO + contrastive, CL=0.2): `freeze_unified_encoder.py`, `enc_unified.pt`/`Ql_unified.npy`.
- Teacher: deployable greedy **EIG** (`eval_all.py`); **oracle** ceiling/teacher.
- Regression harness with **collapse gates** (G1 enc>ridge, G2 policy>random, G3 EIG-policy monotone, G4 polarity).

---

## 3. Detailed technical plan (continuous-space construction + optimization)
> TO BE FINALISED from the deep-research pass (run `wf_dd1ba079-acb`): Wolpertinger & successors; offline-to-online RL
> (BC/DAgger → DDPG/TD3/SAC or PPO; AWAC/IQL/CQL); reward shaping (EIG vs end-metric vs reconstruction); credit
> assignment over short dialogues; action masking / answerability; CRS-RL action spaces & gains; diagnostics for
> policy collapse / imitation gap. Plan, losses, gates, ablations written here with citations once research returns.

### Design skeleton (to be expanded + lit-justified)
- **State**: encoder user vector u_t + dialogue features (turn index, asked-set summary, belief entropy).
- **Action**: continuous proto-concept a_t ∈ R^D → snap to top-k nearest **answerable** concepts (feasible set per
  user/turn) → critic picks (Wolpertinger).
- **Answer model**: simulate like/dislike/don't-know from held-in profile (answerable iff ≥k relevant items).
- **Reward**: long-tail NDCG gain (end metric) ± dense reconstruction-info-gain shaping; discounted over ≤T turns.
- **Training**: BC/distill pretrain on EIG (±oracle) → off-policy actor-critic finetune. Per user's request, finetune
  **separately** on reconstruction / NDCG / Recall / RMSE and compare ALL on reconstruction + every metric (head+tail);
  **answerable-only** actions.
- **Gates** (reuse harness): beat deployable EIG on tail NDCG; monotone; no collapse to popular/static concept.
