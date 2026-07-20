# Session iteration log (internal — NOT for the paper)
Purpose: record everything tried this session so mistakes aren't repeated. Terse on purpose.

## Iterations (what was tried → outcome)
1. **PPO on stratified looked like a win** (AUAC 0.728) → FALSE: it was STATIC (no branching) and the "win" was **metric-gaming via target-probing** (asking the items it's graded on).
2. **Held-out-target metric** (exclude directly-asked targets from scoring) + held-out reward → re-bench flipped: adaptive heuristics on top, PPO/DQN dropped to ~popularity. Confirmed the gaming.
3. **Adaptivity ceiling**: clairvoyant oracle AUAC 0.836 (with target-probing) / **0.796 attr-only**; realizable methods ~0.72. **SNR diagnostic**: per-step routing signal = **2.1% of reward variance** → softmax PG correctly collapses to best fixed order (matches (1−1/e) adaptive-submodular bound).
4. **Oracle action-cloning (BC)** → 0.7085 (< greedy); oracle action only 14% predictable from belief state = **imitation gap** → oracle edge largely *privileged*.
5. **EIG-RLOO from scratch** (entropy-reduction reward) → COLLAPSED (~0.70). Entropy reduction misaligned with accuracy.
6. **Greedy-distillation → equivariant actor** → 0.7222 (best realizable, branches). **+ non-myopic RLOO finetune** → 0.7260. (binary-accuracy metric)
7. **Dataset pivot**: stratified-300 is bespoke/not reviewer-proof → moved to **ML-1M full catalog** (3706 movies + 88 attrs).
8. **ML-1M BCE instrument** → WEAK: oracle ceiling 0.715, lift +0.035, FAILS >0.72 gate (same failure mode as LastFM).
9. **Metric reconsideration (key turn)**: binary per-item accuracy is **base-rate-dominated**. Switched to **NDCG@10/Hit@10**. ranking_headroom: ML-1M BCE instrument ranks **WORSE than random** (0.003 vs 0.031) → instrument broken *for ranking*.
10. **Instrument deep-think**: original (lstm_attention_recommender.py) was already a two-tower + **InfoNCE ranking** model; current set-encoder regressed to **per-item BCE** (calibration, not ranking). Fix = ranking loss.
11. **Ranking-loss instrument v1** (flat head, listwise softmax) → WORKS. Stratified NDCG 0.36→0.52 (2× BCE). ML-1M NDCG 0.39→0.51, lift +0.12.
12. **Headroom-test confusion**: "full reveal" had revealed only **attributes** (coarse) → understated. **Item-reveal (Test B)** confirmed knowing the user helps (+0.098). Absolute top-10 gut-check looked popularity-dominated; **permutation risers** showed real taste (Toy Story→animation, Silence→thrillers, L.A.Conf→arthouse) with an **era confound** (Gladiator→2000 films).
13. **Hard LOO protocol** (popularity-matched negatives) = the honest metric. v1: Hit@10 **0.433 (K0) → 0.647 (K20), spread +0.214**; clean monotonic per-turn curve. In published range (sampled-neg).
14. **Two-tower rank2** (normalized dot-product item emb, d128) to widen spread → **FAILED**: spread +0.160 (worse than v1's +0.214), risers noisier. v1 (flat head) is better.
15. **No-decades v1** → IN PROGRESS (test whether dropping decade attributes clears the era confound).

## Lessons / DO-NOT-REPEAT
- **Metric**: never use per-item binary accuracy for elicitation (base-rate dominated). Use **NDCG@10/Hit@10 with popularity-matched negatives** (hard LOO).
- **Instrument**: must be trained with a **ranking loss**; BCE → predicts the prior, ranks ~random on large catalogs.
- **Headroom/gut-check**: use **item-reveal** (not attribute-only) and **permutation risers** (not absolute top-10, which is popularity-dominated).
- **Target-probing = leakage**: always score on held-out (un-asked) targets.
- **RL collapse is an SNR/effect-size problem**, not capacity; from-scratch RL on the noisy/misaligned reward fails. Distill a realizable teacher (greedy) into an equivariant actor, then finetune.
- **Don't assume architecture upgrades help** — two-tower underperformed the flat head here; always A/B.
- **Background jobs**: check liveness with `ps ... grep [p]ython` (the `[r]ank2`-style name grep is unreliable and falsely showed 0 → caused me to relaunch a still-alive job). Give **every run a UNIQUE log + checkpoint path**; two runs sharing paths collided → shape-mismatch crash. Several "deaths" were OOM (d=256) — size to memory, one heavy run at a time.
- **Instrument training**: needs enough epochs (early runs undertrained; lift kept climbing); stratified npz has 100k users → subsample per epoch.
