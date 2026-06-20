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
> Grounded in deep-research run `wf_dd1ba079-acb` (23 sources, 25/25 claims verified). Citations inline.

### 3.0 The sober prior (read this first)
**Beating a greedy/heuristic baseline in CRS is genuinely hard.** A *non-RL* decision tree (FacT-CRS, CIKM 2022,
arXiv:2208.14614) **beats EAR/SCPR/UNICORN/FPAN by 10–42% SR@10**. So our deployable greedy EIG (PART G+) is a
*strong* baseline, and RL machinery alone will not beat it. The win must come from (i) **non-myopic** credit
assignment, (ii) **exploration** over the embedding manifold, and (iii) the **concept action space** greedy can't
search — not from "adding DDPG." Every claim of a win must be **ablated** against same-instrument + greedy.

### 3.1 Architecture — Wolpertinger actor–critic over the shared embedding (construction)
Canonical fit: **Wolpertinger** (Dulac-Arnold et al. 2015, arXiv:1512.07679) — actor emits a proto-action in a
continuous embedding, **kNN-snaps** to valid actions in log time, scales to ~1M actions. This *is* our
"snap-to-nearest-askable-entity."
- **State** `s_t` = frozen-encoder user vector `u_t` ⊕ dialogue features (turn `t`, #answered, belief summary:
  `‖u_t‖`, predicted-like entropy, asked-set pooled embedding).
- **Actor** `μ(s_t) → a_t ∈ R^D` (MLP): a proto-concept in the *shared* embedding (item ∪ attribute ∪ concept).
- **Feasible set** `A_t` = **answerable, unasked** entities for this user/turn (UNICORN-style preference/entropy
  pruning of the action space, SIGIR 2021 arXiv:2105.09710). Answerability is enforced **at snap time** (mask the
  kNN candidate pool), the cleanest encoding of "don't-know" (research open-Q; we ablate vs a reward penalty).
- **Snap + select (robustness)**: take top-k kNN of `a_t` in `A_t`, then **pick the highest-Q candidate** — **SAVO**
  multi-proposal selection (Sehgal et al. RLJ/RLC 2025, arXiv:2410.11833), which directly counters Wolpertinger's
  worst failure mode: DDPG/TD3 **stuck in multimodal-Q local optima** + irregular-manifold exploration (DGRL, arXiv
  2602.08616; DNC, ICLR 2024 arXiv:2305.19891). (Caveat: DNC/DGRL gains shown on combinatorial/logistics, **not**
  recsys — treat as optional, validate empirically.)
- **Critic** `Q(s_t, e)` over the *action embedding* `e` (so it generalizes across concepts, the Wolpertinger point).
- **Action space = first-class concepts** (couples with B0): co-factorize items+attributes+concepts into ONE trained
  space (removes the PART-K centroid appendage), so the actor *acts* and the encoder *folds in* the same space.

### 3.2 Optimization — oracle-distill, then exceed greedy (training pipeline)
Three stages; **do not** use naive BC (error compounds **quadratically** in horizon T — Ross et al. AISTATS 2011).
1. **Teacher rollouts**: collect trajectories from the **oracle** (privileged ceiling) and the **deployable EIG**
   (PART G+) on train users → `(s_t, action, reward)`.
2. **Imitation pretrain via DAgger** (Ross et al. 2011), not BC: roll out the current policy, **query the oracle on
   policy-visited states**, aggregate, retrain → error **linear** in T (no-regret). We have a queryable oracle, so
   DAgger is directly applicable and gives a strong, distribution-matched init.
3. **Finetune to EXCEED the (suboptimal) teacher** — two complementary options, both avoid the documented
   "naive offline→online collapse":
   - **AWAC** (Nair et al. 2020, arXiv:2006.09359): advantage-weighted actor updates + off-policy value learning;
     designed exactly for "pretrain on demos → keep improving online," sample-efficient for our short dialogues.
   - **RLIF** (Luo et al. ICLR 2024, arXiv:2311.12996): use **oracle-intervention advantage as reward**; beats DAgger
     2–3× *especially when the expert is suboptimal* — and our EIG teacher *is* suboptimal vs the oracle. Good fit to
     "go past greedy."
- **Actor update = denoised regression (DBU/DGRL)**: regress the actor toward a softmax-Q-weighted target over the
  kNN candidates; gradient variance provably **independent of |A|** (arXiv 2602.08616) → stabilizes learning in the
  large embedding action space and dodges the local-optima trap.
- **Imitation-gap lens**: track learner-vs-oracle gap (Weihs et al. NeurIPS 2021 privileged-info gap; Swamy et al.
  ICML 2021 moment-matching) to know whether the gap is closable or the oracle is fundamentally privileged.

### 3.3 Reward (the genuinely open question — so we ABLATE it)
Research did **not** resolve what beats greedy for 1–10 turn horizons. Plan = potential-based shaping (keeps the
optimum invariant) with a head-to-head ablation, per the user's request to finetune **separately** on each objective
and compare ALL on reconstruction + every metric (NDCG/Recall/RMSE, head+tail):
- **R1 terminal tail-NDCG** (true objective; sparse → hard credit assignment).
- **R2 + per-turn EIG shaping** (dense; *risk*: reproduces the greedy teacher — must show it doesn't).
- **R3 + reconstruction-info-gain shaping** (dense, our Paper-A signal).
- **R4 RMSE-driven** (rating reconstruction).
Compare all four variants × {reconstruction quality, NDCG, Recall, RMSE} × {head, tail}.

### 3.4 Answerability (feasible-action handling)
- **Primary**: mask the kNN candidate pool to answerable entities (UNICORN entropy/preference pruning).
- **Ablation**: open-catalogue asking with a "don't-know" reward penalty (PART H setting) — tests deployability when
  the answerable set is unknown. Concepts shine here (PART O: 57% answerable vs items 2.1%).

### 3.5 Anti-collapse diagnostics (our own scars + research caution)
- **Playlist collapse** (policy ignores answers → static action sequence): monitor action diversity and
  answer-dependence; entropy bonus; the existing **collapse-gate harness** (`eval_all.py`).
- **Reward hacking / imitation gap**: the FacT-CRS caution → require the ablation in §3.6.

### 3.6 Gates & ablations (make-or-break; reuse `eval_all.py` harness)
- **B0** first-class concepts ≥ centroid concepts (no item regression). [enables the action space]
- **B1** a *distilled* policy (DAgger-from-oracle, discrete over the answerable set) **beats deployable EIG** on tail
  NDCG @q{2,4,8}. [is there learnable signal beyond greedy at all? cheapest test]
- **B2** the *continuous* Wolpertinger+SAVO actor ≥ B1 (continuous beats discrete/greedy).
- **B3** online finetune (AWAC/RLIF) > B2; report the 4 reward variants.
- **B4 ABLATION (decisive)**: same instrument + greedy vs + policy — the delta is the policy's contribution, *not*
  the encoder's. If B1 already fails → **EIG is the honest contribution**; report it and stop (anti-thrash).
- Baselines throughout: random, HELF, Golbandi, **deployable EIG**, **oracle** ceiling. Metrics: NDCG/Recall/RMSE,
  head+tail, per #answered.

### 3.7 De-risked staged ladder (one variable at a time; stop/rethink on gate failure)
`B0 co-factorize concepts → B1 distilled discrete policy vs EIG → B2 continuous Wolpertinger+SAVO → B3 AWAC/RLIF
finetune (4 rewards) → B4 open-concept/free-text + long-tail vs oracle`. Each step through the collapse-gate harness.

### 3.8 Key references (Paper B bib seed)
Wolpertinger (arXiv:1512.07679) · SAVO (2410.11833) · DGRL/DBU (2602.08616) · DNC (2305.19891) · AWAC (2006.09359) ·
DAgger (Ross 2011) · RLIF (2311.12996) · imitation gap (Swamy 2021; Weihs 2021) · UNICORN (2105.09710) ·
ConTS (2005.12979) · FacT-CRS (2208.14614) · PEBOL (RecSys 2024). [verify each before citing in the paper]

## 4. MAIN EXPERIMENTS — answerability + continuous-space value (deep-research wf_bb2aa0ee-418)

### 4.1 Setup (lit-grounded)
- Frozen instrument = Paper A calibrated encoder (items+attributes+concepts one space; additive fold = ConTS arm space).
- FAKE-USER concept-answer model: GRADED + profile-based + "don't know" (NOT hard-binary-from-target = the leakage pattern).
  - answerable iff user has >=k HELD-IN profile items with concept-relevance>tau; else "don't know" (no fold).
  - answer = graded affinity = mean debiased residual over those items (sign=like/dislike). [PEBOL graded>binary RecSys24;
    MIMConv UUSFF inherent-interest+don't-know; MCMIPL Others/zero option]
  - LEAKAGE CONTROL (critical, "How Reliable is Your Simulator?" WWW24: leakage inflates Recall@50 13-22%): affinity &
    answerability from held-in profile ONLY; held-out TARGET excised from history AND simulator replies.
- CONCEPT VOCABS (3 tiers to separate enumerable vs uncountable): genres(18 coarse) / tag-genome(~745-1128 fine,
  enumerable) / OPEN-NL (SBERT of LLM-generated/arbitrary phrases = UNCOUNTABLE) <- where continuity is needed.
- EVAL (RankCrit RecSys20 / PEBOL loop): random target from HELD-OUT test, ask/answer/fold <=T turns, success@N{1,5,10,20}
  + NDCG@10(full+tail) per #ASKED, report Avg Success Rate + Session Length + ANSWER-RATE (Montazeralghaem25 precedent),
  multiple sessions/user, leakage-excised.

### 4.2 Experiments + GATES
E1 ANSWERABILITY value: open setting (ask pool, don't-know for unanswerable). item vs genre vs concept asking (EIG-within-
   type + learned policy). Metric NDCG/success vs #ASKED (don't-know turns COST). GATE A1: concept-asking > item-asking
   at fixed #asked. (PART O predicts; control answer-rate, sweep tau/k.)
E2 CONTINUOUS-space value (NOVEL — research found NO prior continuous-vs-discrete concept head-to-head): on OPEN-NL space,
   discrete-small(genres) vs discrete-large(tag-genome enum+EIG) vs CONTINUOUS actor(emit embedding->snap nearest
   answerable, Wolpertinger+SAVO). GATE A2: continuous > best discrete WHEN space is uncountable (open-NL). HONEST: on the
   enumerable tag vocab continuous should ~= discrete-large (no continuity advantage there) -> continuous win REQUIRES
   open-NL adding realizable value (the linchpin/risk).
E3 NON-MYOPIC payoff (continuity is where RL beats greedy, Blau22): continuous learned policy (PPO+GAE + potential info-
   gain shaping Ng99, or AWAC from concept-EIG init) vs greedy concept-EIG. GATE A3: continuous policy > greedy concept-EIG.

### 4.3 Model (continuous exploration)
State u_t + dialogue feats. Actor emits a_t in R^D -> snap to nearest ANSWERABLE concept (kNN over vocab; or all+don't-know)
-> SAVO pick best-Q among kNN -> fold via frozen encoder. Train: distill concept-EIG (realizable) -> PPO+GAE + potential
info-gain shaping finetune (NOT REINFORCE). Answerability = feasible-action masking (UNICORN) vs don't-know penalty (ablate).

### 4.4 Variants/ablations: answer graded-vs-binary (expect graded>binary PEBOL); don't-know on/off; vocab tier;
selector random/within-type-EIG/discrete-large/continuous; leakage full-vs-excised (expect drop, report); budget per-asked
vs per-answered; reward terminal±shaping; tau/k sweep.

### 4.5 RISK REGISTER (honest): (1) LEAKAGE = #1 threat -> excise target everywhere, report excised numbers. (2) Sim
faithfulness: our answer model is rule-based profile-grounded; validate vs an LLM simulator on a subset (Yoon NAACL24 5-task
spirit); state limits. (3) CONTINUOUS may only MATCH discrete-large on enumerable vocab -> continuous win NEEDS open-NL to
add realizable value; if it doesn't, honest fallback contribution = answerability + unified graded-concept instrument
(still novel). (4) PEBOL graded>binary was 2-1/tuning-sensitive -> replicate before relying.
BIB seed: PEBOL(Austin/Korikov/Sanner RecSys24), RankCrit(Li/Luo/Wu/Sanner RecSys20), ConTS(TOIS21), UNICORN(SIGIR21),
MCMIPL(2112.11775), MIMConv-UUSFF(TOIS23), iEvaLM(2305.13112), HowReliableSimulator(WWW24 2403.16416),
Montazeralghaem25(2510.12015), Yoon-NAACL24(2403.09738), Biyik-soft-attr(2023), Wolpertinger(1512.07679), Ng99-shaping.

## 5. THOROUGH RECONSTRUCTION-FIRST TEST LADDER (one variable at a time; ultrathink June 2026)

### 5.0 METRIC REFRAME (foundational, confirmed T1): predict the PROFILE, not NDCG.
PRIMARY = profile reconstruction on the TAIL: Recall@50 (un-saturated) + recon-AUC. NDCG@10/full-AUC are popularity-
SATURATED (q0 AUC already 0.85+) and HID the signal. Per #asked AND #answered. T1 result: elicitation ~DOUBLES tail
Recall@50 (q0 0.107 -> 0.18-0.24) => "it works". BUT realizable selection ~= random profile-restricted; oracle ~2x
(privileged). So selection isn't the win; ANSWERABILITY + folding + direct-embedding + objective are the levers.

### 5.1 BOUNDS (goalposts every test reports): FLOOR-0 (popularity prior) ; FLOOR-rand (random answerable) ;
realizable method ; CEIL-full (fold whole profile) ; CEIL-subset (oracle best-K, peek) ; CEIL-dir (oracle best
direction, continuous ceiling). A method "works" if > FLOOR-rand and climbs toward CEIL.

### 5.2 LADDER (tail Recall@50 + recon-AUC; one variable at a time):
T1 [DONE] metric reframe + bounds: elicitation doubles tail recall; selection~=random; oracle 2x privileged.
T2 [RUNNING] ANSWERABILITY (open setting): item vs concept asking, Recall@50 + answer-rate. Expect concept >> item
   (items ~2% answerable -> wasted turns; concepts ~90%). THE answerability win, on the right metric.
T3 CONCEPT FOLDING: retrain encoder WITH genome-concept reveals (currently OOD - trained on item+genre only) and/or
   co-factorize concepts first-class (PART K). Gate: better concept fold -> higher concept reconstruction value.
T4 DIRECT-EMBEDDING realizable policy (USER FAVOURITE, done right): learned actor predicts a_t in R^D, folds (a_t,
   answer(a_t)) DIRECTLY (no snap), trained on RECONSTRUCTION objective. vs concept-snap / item / CEIL-dir. Does
   probing arbitrary directions (active taste-space search) beat discrete asking on tail recall?
T5 ANSWERABILITY PREDICTION (USER INSIGHT): predict P(answerable | concept popularity/coverage, prior answers);
   answerability-aware selection (info x P(answerable)). FLOOR=ignore answerability, CEIL=oracle answerability. Does
   predicted answerability recover most of the oracle-answerability value in the open setting?
T6 CONCEPT-EIG PRETRAIN + RL: distill concept-EIG -> PPO+GAE+potential-info-gain-shaping, RECONSTRUCTION reward. vs
   concept-EIG. (continuity is where non-myopic RL wins, Blau22).
T7 SYNTHESIS: best folding (T3) + best policy (T4/T6) + answerability (T5), open setting, tail Recall@50 + NDCG/RMSE.

### 5.3 ARCH VARIANTS: encoder {frozen / +concept-retrained / co-factorized}; policy {random / EIG / distilled /
learned-continuous-actor / answerability-aware}; action {snap-concept / direct-embedding}; objective {reconstruction /
Recall / RMSE}. Each test: report all 6 bounds, gate vs FLOOR-rand and vs CEIL, commit to RESULTS.
