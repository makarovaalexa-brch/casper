# CASPER — Why adaptive elicitation "didn't work", and how to capture it
**Overnight investigation, 2026-06-14. Status: diagnosis proven; method built and training.**

---

## 0. Executive summary

The premise of the thesis — *personalised (adaptive) preference elicitation is valuable* — **is correct and survives**. What we discovered is sharper and more interesting than "it doesn't matter":

1. **Adaptive elicitation has large LATENT value** but most of it is **privileged** (needs information the model can't observe at decision time). A clairvoyant routing oracle reaches **AUAC ≈ 0.796**; the best realizable method reaches **~0.72**; naive static **~0.70**.
2. **Naive RL (PPO) collapsed to a static policy for three compounding, now-proven reasons** — low reward signal-to-noise, a wrong (task-metric) reward, and a wrong (flat-MLP) actor. None is "the premise is wrong."
3. We **fixed the architecture** (an equivariant per-item actor now produces genuinely adaptive, branching policies) and identified the right learning signal (**dense information-gain reward, non-myopic**), which is **novel in conversational recommendation**.
4. We **proved a hard limit**: the oracle's edge is *not imitable by action-cloning* (an imitation gap) — most of the 0.72→0.80 gap is privileged. This decomposition (realizable vs. privileged adaptivity) is *itself a publishable contribution*, and it is exactly what theory predicts.

**Net:** the project pivots from "show RL beats baselines" (weak) to **"quantify and capture the realizable adaptive edge; prove the rest is privileged"** (strong, honest, novel). This is supported by the literature (adaptive-submodularity; Wang et al. ICML 2025 "adaptivity pays on the tail").

---

## 1. What went wrong — three proven mechanisms

### M1. Low advantage signal-to-noise → policy gradient collapses to the best FIXED order
`scripts/paper2/snr_diagnostic.py` (computed from held-out episode logs, n=300):

- Realizable adaptive edge (best adaptive heuristic − static): **+0.017 AUAC** (CI [+0.007,+0.027]); over the *best* static (learned PPO order) it is **~+0.002** (not significant).
- Oracle edge: **+0.092 AUAC** — **5× larger**, and privileged.
- Per-step routing reward: mean gain **+0.006** vs **cross-user std 0.063** → **effect-size ratio 0.099 (≪1)**.
- Variance decomposition: routing **signal fraction = 2.1%** of per-step reward variance.

**Conclusion:** the per-step routing signal is ~2% of the noise. A softmax policy gradient is noise-limited and *correctly* converges to the marginal-best (static) action. This is the documented composition of: softmax-PG premature commitment (Mei et al. 2020; Agarwal et al. 2021), PG variance (Tucker et al. 2018), and the gradient-noise-scale regime (McCandlish et al. 2018). It also matches the **(1−1/e) adaptive-submodularity bound** (Golovin & Krause): non-adaptive greedy is provably near-optimal when the objective is adaptive-submodular, so a small realizable edge is *expected*, not a bug.

### M2. Wrong actor architecture → no inductive bias for routing
PPO's flat MLP → N-way softmax has no structure tying logit *i* to item *i*'s belief. Under M1's low SNR it defaults to a position/marginal bias (PPO produced **2 distinct trajectories** over 300 users). **Fix proven:** the permutation-equivariant per-item actor (`scripts/paper2/equivariant_actor.py`) scores each candidate from its own belief slice with shared weights + pooled context; under behavior cloning it already produces **169 distinct trajectories, branches=True**. The architecture was a real cause; it is fixed.

### M3. Wrong reward → task-metric Monte-Carlo instead of information gain
The reward was per-step held-out BCE reduction — a high-variance, weakly-coupled, often-flat function of a single answer (this *is* M1's 2% signal at the source). The fix is a **dense expected-information-gain reward** computed from the instrument's own predictive entropy over held-out targets (no label-sampling noise), which is also the objective that lets RL be **non-myopic** and potentially exceed the myopic greedy heuristic (Foster et al. 2021 DAD; Blau et al. 2022 RL-BOED; Huang et al. 2024 decision-aware BED).

---

## 2. The hard limit we proved: the oracle edge is privileged (imitation gap)

`scripts/paper2/oracle_distill.py` — distill the clairvoyant oracle into a belief-state policy (Choudhury et al. RSS 2017 framing):

- teacher (clairvoyant) AUAC 0.815 (train) / ~0.796 ceiling.
- student (behavior-cloned, equivariant actor): **0.7085**, branches=True, 169/300 unique trajectories.
- oracle-action top-1 predictability from the belief state: **14%**.

**Interpretation (imitation gap; Weihs et al. ADVISOR NeurIPS 2021; Vuorio et al. 2024):** the oracle picks attributes using the *true labels*, which are not determined by the observable belief state, so cloning its action gives a policy *worse* than value-based greedy. **Most of the +0.10 oracle edge is privileged and not realizable by imitation.** The right objective is action *value* (information gain), not action *identity* — hence the method in §3.

---

## 3. The method — non-myopic amortized Expected-Information-Gain policy

`scripts/paper2/train_eig.py`. Combines the three independently-recommended fixes:
- **equivariant per-item actor** (M2 fix — routing inductive bias),
- **dense target-EIG reward** = reduction in the instrument's predictive entropy over held-out targets (M3 fix — high SNR, computed from model probabilities, no label sampling),
- **group-relative (RLOO) baseline** over G rollouts of the same user (M1 fix — cancels between-user variance; Ahmadian et al. 2024),
- **non-myopic** (γ=1, return = total EIG) so it can exceed the *myopic* greedy-infogain (0.716) — the only realizable way to beat the heuristic.

Equivalent to a discrete-action RL-BOED (Blau et al. 2022) with target-predictive EIG (Huang et al. 2024) on the equivariant policy. **Novelty (confirmed by survey):** amortized × non-myopic × recommendation-elicitation is unclaimed.

**Alternative held in reserve:** DAgger distillation (`train_dagger.py`) — but §2 shows the clairvoyant teacher is only partly followable, so DAgger is expected to reach ~greedy, not the oracle. Reserve the asymmetric-critic / Elf-distillation (Walsman et al. ICLR 2023) variant if we add privileged value (not action) signals.

### RESULT (filled as training lands)
- EIG-RLOO on stratified: **[training — `bfe2g6jz8`]**. Targets: beat myopic greedy 0.716, branch, close part of 0.716→0.796.

---

## 4. Literature grounding (4 parallel surveys; key anchors)

- **Amortized BOED:** Foster et al. *Deep Adaptive Design* ICML 2021; Ivanova et al. *iDAD* NeurIPS 2021; **Blau et al. *RL-BOED* ICML 2022** (discrete designs — our case); Huang et al. *Amortized BED for Decision-Making* NeurIPS 2024 (target/decision-aware EIG).
- **Privileged/oracle imitation:** **Choudhury et al. *Adaptive Information Gathering via Imitation Learning* RSS 2017** (direct precedent); Ross et al. *DAgger* AISTATS 2011; Chen et al. *Learning by Cheating* CoRL 2019; Pinto et al. *Asymmetric AC* RSS 2018; **Weihs et al. *ADVISOR* NeurIPS 2021** & Vuorio et al. 2024 (imitation gap); Walsman et al. ICLR 2023 (Elf distillation).
- **PG collapse / SNR:** Mei et al. 2020; Agarwal et al. 2021; Tucker et al. 2018; McCandlish et al. 2018; GRPO (Shao 2024) / RLOO (Ahmadian 2024); Deep Sets (Zaheer 2017), Pointer Nets (Vinyals 2015), Wolpertinger (Dulac-Arnold 2015).
- **CRS elicitation / novelty:** CRM (Sun 2018), EAR (Lei WSDM 2020), SCPR (KDD 2020), UNICORN (Deng SIGIR 2021), **PEBOL (Austin RecSys 2024** — Bayesian/Thompson, *myopic*), GATE (Li ICLR 2025), OPEN (Handa 2024), Mazzaccara (EMNLP 2024, EIG+DPO but myopic), **Wang et al. ICML 2025** (adaptivity pays 5–10× on tail/atypical targets), Rashid IUI 2002 / Elahi 2016 ("static/popularity competitive" is documented). Your IJCNN 2024 (Makarova et al.) is the RL-elicitation predecessor.
  - **Metric note:** "AUAC" is our own metric; PEBOL reports MAP@10/MRR@10 — do not attribute AUAC to it.

---

## 5. Paper framing (the strongest, honest story)

**"The realizable value of adaptive preference elicitation: a decomposition, and a method."**
1. Testbed with a fixed instrument + held-out-target metric (gaming-proof).
2. Finding: on realistic (tail-inclusive) catalogs, adaptive > static but the edge is small *and theory says so* ((1−1/e); tail-concentrated, Wang 2025); a clairvoyant oracle shows a 5× larger ceiling.
3. Diagnosis: naive RL collapses (SNR 2% — proven); the oracle edge is largely privileged (imitation gap — proven).
4. Method: non-myopic amortized-EIG policy with an equivariant actor captures the *realizable* edge (novel in CRS); the residual to the oracle quantifies the *privileged* value.
This reframes a "null" into a measured decomposition + a method — a stronger and more defensible contribution than "RL beats random."

---

## 6. Next steps
- [running] EIG-RLOO on stratified → numbers.
- Then: EIG on Yelp multi-city (disjoint cities → routing forced; the regime where the realizable edge should be largest).
- Then: synthetic indicator world (known large edge) as the method's upper-bound sanity (PPO already branches there; EIG should match/exceed).
- Optional: DAgger reaching ~greedy (confirms the imitation-gap ceiling empirically).
- Write Paper-2 method section from §3–§5; integrate the decomposition figure (static / realizable / oracle bars + per-turn curves).
