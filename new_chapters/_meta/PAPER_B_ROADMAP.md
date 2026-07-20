# Paper B — Roadmap & TODO

**Owner plan (Jun 2026):** squeeze all we can from DISCRETE → port to CONTINUOUS + optimise on top (gain a bit more) → add OPEN (free-text) questions → Paper B done. "Maybe some baselines need a closer look." Endorsed + expanded below. Sequence is right: discrete-first de-risks the rest (the discrete winner is a *realizable* teacher → distil the continuous actor from it, no privileged-imitation gap; open-Q reuses the same encoder+reward).

Current discrete winner: **clean-354 FULL 0.326 / TAIL 0.134** (beats entropy 0.322/0.130 + conc_pop 0.321/0.122 on both axes; answerability-efficiency story). Checkpoint `policy_poracle_reg_best.pt`. See POLICY_RESULTS.md. **[UPDATE Jun 24] de-biased unified-lever policy (`policy_dbfpk.pt`) is the new best on te[300:]: FULL 0.347 / TAIL 0.144, beats entropy on BOTH axes, SEED-ROBUST (6/6 seeds: full +0.0055±0.0021, tail +0.0108±0.0046).**

## ★★★ THREE KEY MILESTONES FOR PAPER B (owner, Jun 24 2026) — after the discrete push is locked ★★★
1. **CONTINUOUS** unified-action actor (Wolpertinger proto-action + snap to nearest answerable item/concept) — novelty lever (Phase 2).
2. **OPEN free-text questions** (arbitrary concept, not pool-restricted) — the real NDCG/answer-bits lever (Phase 3).
3. **BOT-PLAY co-trained answerer** (VisDial-style; learn the fold/representation, NOT the info content — answer stays grounded to sign(u*·E)) — the ENABLER that realizes continuous headroom (Phase 2b; place with/before continuous). *Owner first forgot, then re-added — do not drop.*

## Phase 1 — SQUEEZE THE DISCRETE WIN
- [ ] **Attention state (P6)** — RUNNING (re-distilled oracle+attn, history-aware). Does it widen the margin?
- [ ] **Feature ablation** — isolate which features drive the win (divisiveness vs avg-rating vs answerability); drop dead weight.
- [ ] **Variance reduction** — actor-critic / PPO (P7) for a cleaner finetune; revisit deeper/wider scorer (P5) *with* attn.
- [ ] **SEED + SUBSET average** (seeds 123,1,2,3,7,11 × user subsets) → mean±std. The +0.004 is single-seed.
- [ ] **Significance** — paired test (same users, policy vs entropy) or bootstrap CIs. A bare +0.004 gets killed by reviewers.
- [ ] **Ablation table** for the paper — BC-floor / +RL / +best-ckpt / feature drops → show WHICH component wins (best-ckpt did the heavy lifting; entropy-bonus was a wash).
- [ ] **FULL-objective counterpart** — everything so far is TAIL-optimized (reward + oracle teacher + selection all `REWTAIL=1`). After the tail winner is picked (attn vs non-attn on clean-354), rerun *that config* with `REWTAIL` OFF → full-NDCG teacher + reward + selection. Report BOTH the tail-optimized and full-optimized policies (shows the method is objective-agnostic + the tail↔full tradeoff). Code is one-flag; teacher caches separately (`oracle_demos_full_h.npz`).
- [ ] **DENSER reward (owner idea, Jun 2026)** — tail-NDCG@10 reward is SPARSE/near-binary (few tail likes in top-10 → noisy gradient = the expert's reward-SNR flaw). Try **AUC** (all tail-like vs non-like pairs) or **MRR** as the RL reward for a smoother gradient (keep NDCG@10 for EVAL). Likely the most direct lever on the imitation gap.
- [ ] **SATURATION + FOLD finding (Jun 2026):** elicitation SATURATES at ~q2 — even the conc_oracle peaks at 2 questions (tail 0.408@q2 → 0.400@q8) then DECLINES; the realizable winner is flat from q2. ⇒ (i) MORE questions (T=10/20) won't help (owner asked; confirm with a T=16 run but oracle predicts no gain); (ii) the q2→q8 decline is the ENCODER FOLD degrading with #answers (NOT monotone, even for the oracle) → fix = monotone fold OR learn-to-stop early (efficiency story); (iii) the imitation gap is present FROM q2 = a per-question DISCRIMINATION gap, not a horizon problem.
- **GATE:** lock the strongest discrete policy + robust numbers before porting.

## EXPERIMENT QUEUE (owner agenda, Jun 2026) — RECORD ALL ATTEMPTS, CONSISTENT RULER (val te[:300], test te[300:]), report BOTH full+tail
1. **Natural cut-off / more questions — CUT-OFF DONE (Jun 2026, 30-user q0→q20):** the ORACLE saturates at **q2–q4** (tail 0.429, then slowly DECLINES to 0.418@q20 — info is front-loaded, fold degrades after). But the realizable heuristics keep slowly creeping up to q20 (conc_pop tail 0.174@q8 → 0.205@q20). cos(u,u*) keeps rising to q20 (oracle 0.622→0.754). ⇒ optimal policy needs ~q4; realizable one benefits to ~q16. **[TODO] RETRAIN the policy at T=16** (it's OOD past its train-T=8, so eval-only past q8 is invalid) — the honest test of "more questions help the policy." T + QPTS now env-configurable.
2. **Denser reward (better SNR)** — tail-NDCG@10 reward is sparse/near-binary. Try **AUC** (all tail-like vs non-like pairs) or **MRR** as the RL reward; KEEP NDCG for val/eval.
3. **De-biased full reward** — if tail-reward beats full-reward, try a full reward weighted by **INVERSE popularity** (de-popularized full; fixes the popularity-saturation that makes plain full a poor objective and causes the q2→q8 decline).
4. **cos(u,u*) diagnostic** — belief-to-true-taste alignment is a STRONG signal (rose 0.681→0.801 over full-run training = belief sharpening even as NDCG saturates). FAILED as a reward (O7 reconstruction stalled) but track it everywhere + try as an **auxiliary loss** alongside the NDCG reward.
5. **Both axes always** — report full AND tail for every attempt.
6. **Consistent ruler + record all** — every attempt on val te[:300] / test te[300:]; logged in POLICY_RESULTS.md.
7. **Iterate on the ORACLE/teacher objective (owner idea, Jun 2026)** — the teacher shapes what BC distils, so the teacher's objective is a KEY lever on the imitation gap. Test BC-pretraining on DIFFERENT oracles + inspect the strategy each learns + which yields the best REALIZABLE policy: (a) tail-NDCG [current], (b) full-NDCG, (c) **cos(u,u*) belief-oracle** — pick the concept moving the belief closest to the true taste (DENSER/smoother; the strong cos signal #4), (d) AUC / MRR oracle. **HYPOTHESIS (ties to §gap):** a smoother/denser teacher may be MORE REALIZABLE — its picks depend more on the *observable* belief-direction (residual toward u*) than on the sparse held-out-NDCG ranking, so the distilled policy recovers more of it ⇒ SMALLER privileged-imitation gap, possibly closing the realizable↔oracle gap. Implementable: teacher = argmax_c of the chosen objective on held-out, then BC the feature-rich policy; compare realizable evals + learned strategies. (The same smoother objective can also be the RL reward, #2.)

## Phase 2 — PORT TO CONTINUOUS + OPTIMISE (paper headline novelty)
- [x] **continuous-oracle vs discrete-oracle headroom — DONE (Jun 2026, 40-user read):** cont_oracle tail **0.416** vs conc_oracle **0.400** = **+0.016 tail** (full TIED 0.382/0.383); novel-dir picks **7%** (93% of the time the best point IS a pool concept → pool is DENSE). VERDICT: continuity offers a SMALL tail headroom at the *privileged* level; realizable fraction ~0.002–0.005 (marginal, within noise). ⇒ the continuous port is a **REPRESENTATIONAL** contribution (open-vocab unified actor = paper novelty), NOT an NDCG lever — UNLESS the encoder co-adapts (bot-play, Phase 2b) to enlarge + realize the novel-direction headroom. [confirm on a bigger user set; 40 is noisy]
- [ ] **Distil continuous actor FROM the discrete winner** (realizable teacher → recovers it). Snap ⇒ continuous is a superset of discrete ⇒ no loss at the optimum; the only loss is optimisation (PC collapsed → must anchor).
- [ ] **PC2: anchored/critic-regularised emission** — warm-start from discrete winner, frozen-scorer critic, on-manifold + within-session diversity terms, per-user conditioning, NO snap → reach novel answerable middle-grounds. (cont_elicit.py)
- [ ] **Frozen-encoder OOD check** — does enc() fold novel between-concept points correctly? May need encoder generalisation.

## Phase 2b — BOT-PLAY: co-train the ANSWERER ("user learns to answer") [owner idea, Jun 2026]
**Lineage:** the cited VisDial framework (Das et al.\ 2017) co-trains BOTH QBot (asker) AND ABot (answerer); the owner's published paper simplified ABot to deterministic. Re-introducing a LEARNED answerer restores the original framework AND is the continuous-space enabler.
**Reframing (why it matters):** with a FIXED geometric answerer the frozen encoder is OOD on novel continuous points, so Phase-2a headroom is capped by pool-density + OOD. If the answerer/fold CO-ADAPTS, novel continuous questions become correctly interpretable → the continuous headroom becomes REALIZABLE. "User learns to answer" = co-train the FOLD (encoder) to interpret arbitrary `(e, answer)` pairs, not just the 761 fixed concepts.
- [ ] Co-train encoder(fold) + QBot policy via bot-play (REINFORCE on recommender-loss reduction — exactly the cited reward).
- **HARD CONSTRAINT (else metric-gaming → unpublishable):** the answer bit stays GROUNDED, a faithful function of true taste (`sign(u*·e)`); only the REPRESENTATION/fold is learned, never the information content. Co-adapt representation, NOT information.
- **RISKS:** emergent-comm instability/collapse (this project's prior collapses were exactly co-training: signed-encoding, contrastive); private-code leakage of u*. **MITIGATIONS:** held-out-user generalization test; leakage ablation (answerer cannot reveal more than its one grounded bit/question); collapse monitors.
- **PLACEMENT: BEFORE open concepts** — it is the continuous enabler; open free-text (Phase 3) then builds on the co-trained grounded protocol. HIGH-risk/HIGH-reward → do the cheap headroom + discrete squeeze FIRST.

## Phase 3 — OPEN (FREE-TEXT) QUESTIONS (suspected decisive bits-per-answer lever)
- [ ] Free-text question → embedding → geometric answer; quantify free-text answerability. Concepts ≈ 1 bit/answer; free-text = many bits → the lever that could actually reach the oracle's ~0.35 tail.
- **PRIORITY TENSION:** if discrete+continuous both plateau ~0.13 tail, open-Q is the real *impact* lever while continuous is the *novelty* lever. May deserve higher priority than the continuous port. Decide after the Phase-2 headroom number.

## Cross-cutting — BASELINES & PAPER INTEGRITY (the "closer look")
- [ ] **HARMONISE all tables to ONE ruler** — paper's tab:disc + tab:answer_cold use *different* splits than the new clean-354. Re-run everything on the same ruler. Integrity blocker.
- [ ] **Strongest heuristic** — is divisiveness the strongest? Add **concept-EIG** (one-step info-gain over concepts), not just divisiveness/popularity. The win must beat the STRONGEST reasonable heuristic, not a soft one.
- [ ] **Lit baselines on the concept ruler** — EAR / ConTS / UNICORN / NICF replicated + run on the SAME ruler (replicate-first).
- [ ] **UPDATE THE NARRATIVE** — abstract/contributions say "none beats greedy"; now true only for discrete ITEMS. Discrete CONCEPTS admit a learned win. Re-frame the arc: items = greedy frontier (neg) → concepts = first learned win (pos, sec:learned) → continuous = novelty → open = impact.

## Done-when
Robust (seed-avg + significance) discrete win + continuous port (gain OR honest representational framing, decided by the headroom number) + open-question demonstration + harmonised baselines + updated narrative.
