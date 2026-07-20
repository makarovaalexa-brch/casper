# Paper B — 3-Milestone Novelty Research (deep-research wf wsjz8y92u, Jun 24 2026)

105 agents, 23 primary sources, 25/25 claims verified (0 refuted). Verdicts/recipes = design synthesis on verified facts.

## TL;DR
- **Order: M1 → M2 → M3.** M1 is the single most defensible methods-paper result. M3 needs M1/M2's novel continuous questions to make the answerer OOD (and must ground-first per Lowe/VisDial).
- **Shared infra (all 3):** frozen set-encoder fold; geometric answerer sign(u*·e); shared item+concept embedding; FAISS/ANN snap; discrete policy as BC-teacher + control; full-catalogue NDCG@10 + tail ruler.
- **⚠ CRITICAL CAVEAT (research + our own memory agree):** M1/M3's novelty case rests on the continuous headroom being REAL/NDCG-relevant, not merely representational. We already measured it on ML-1M: cont-oracle beats disc-oracle by only **+0.016 tail (pool is dense)** → on ML-1M continuity is **representational**, NOT a big NDCG lever. ⇒ the NDCG story for continuity likely needs the **sparser dataset** (replication phase) where the pool isn't dense. Measure cont-vs-disc oracle headroom on any new ruler FIRST.

## MILESTONE 1 — CONTINUOUS UNIFIED-ACTION  → **NARROW-IT-DOWN (cleanest novelty; do FIRST)**
**Prior art (gap):**
- **Wolpertinger** (Dulac-Arnold 2015, arXiv:1512.07679): continuous proto-action + approx-NN snap, but ONE homogeneous action set.
- **HyAR** (Li/Tang/Hao, ICLR 2022, arXiv:2109.05490): single decodable latent for hybrid discrete+continuous action; TD3-in-latent + decode; **anti-collapse = "semantic smoothness" via dynamics prediction + Latent Space Constraint** (cite for the collapse fix).
- **ConTS** (Li/Lei/He/Chua, TOIS 2021): unifies items+attributes as undifferentiated arms (A=P∪V), ask-vs-rec emerges — but DISCRETE Thompson-Sampling argmax over enumerated FM/BPR arms, NOT a continuous actor+snap.
- **GAP (unclaimed):** a continuous RL actor snapping to a HETEROGENEOUS union of {items} ∪ {open-vocab concepts} on equal footing (one snap, not split ask-then-rec). Wolpertinger snaps within one type; ConTS unifies types but by discrete enumeration.
**Recipe:** DDPG/TD3 actor → FAISS snap argmax-cos over {items}∪{concepts} → critic on snapped action → reward NDCG@10 + tail. A/B: (a) TD3 from scratch vs (b) **BC-warm-start from the discrete winner + RL-finetune** (lower variance, more defensible given REINFORCE drift). Measure NDCG full+tail vs discrete + entropy heuristic + snap diversity (item:concept mix, unique-entity rate).
**Failure modes:** (1) proto-action collapse → HyAR smoothness + entropy/diversity bonus; (2) heterogeneous-snap imbalance (one modality dominates) → temperature-balance item vs concept NN pools; (3) continuity = representation not NDCG → pre-measure cont-vs-disc ORACLE headroom on the ruler FIRST.

## MILESTONE 2 — OPEN FREE-TEXT QUESTIONS  → **NARROW-IT-DOWN but THIN (crowded); ship as a COMPONENT, lock fast**
**Prior art (crowded):**
- **PEBOL** (Austin/Sanner, RecSys 2024, arXiv:2405.00981): NL pref-elicitation as Bayesian Opt; TS/UCB steer LLM queries; NLI grounding to item content; per-item Beta beliefs; **no MF / no user embeddings**.
- **GATE** (B.Z.Li/Tamkin/Goodman/Andreas, ICLR 2025, arXiv:2310.11589): LM generates open elicitation queries, beats user-written prompts/labels in content-rec; but all-in-text label prediction, **no CF**.
- **Montazeralghaem 2025** (Google, arXiv:2510.12015, ~8mo old): open free-text clarifying Qs ON MOVIELENS, general→specific funnel; but TEXT/JSON profile + BLEU/ROUGE eval, **no CF set-encoder, no NDCG**. ← biggest crowding signal.
- **OPEN** (Handa/Goodman/Andreas/Tamkin/Li 2024, arXiv:2403.05534): BOED/EIG picks query, LM verbalizes, interpretable linear utility θ·φ(x) + particle filter.
- **GAP (thin):** fold the yes/no answer to an arbitrary free-text concept through the SAME frozen CF encoder (score=pop+q·u), eval full-catalogue NDCG@10+tail. None do CF-fold + NDCG. **Time-sensitive — lock the CF-fold differentiator fast.**
**Recipe:** decode target concept from belief geometry → LLM verbalize "do you like films about <X>?" → fold via the frozen encoder. **Failure modes:** metric-gaming → keep answer GEOMETRIC; novel concept OOD for frozen answerer → motivates M3 / restrict to in-distribution early; non-snappable free-text → constrain decoder to the encoder's concept manifold + verify round-trip fidelity.

## MILESTONE 3 — BOT-PLAY CO-TRAINED ANSWERER  → **NARROW-IT-DOWN, HIGHEST ceiling + risk; do LAST**
**Prior art:**
- **VisDial** (Das/Batra, ICCV 2017, arXiv:1703.06585): QBot+ABot co-trained end-to-end RL; documents the goal (ask what ABot is good at) AND the risk (metric-gaming, emergent private codes from scratch).
- **Montazeralghaem 2025** (arXiv:2510.12015): co-trains questioner+simulator (fine-tuning the simulator is necessary), but SUPERVISED (profile reconstruction), text-space — not self-play, not geometric.
- **Lowe S2P** (ICLR 2020, arXiv:2002.01093): ground-first-then-self-play beats reverse; "not beneficial to emerge from scratch"; keep BOTH grounding + self-play signals.
- **DwD** (Cogswell/Batra, NeurIPS 2020, arXiv:2007.12750): anti-drift via factorize-intention-from-language + DISCRETE info bottleneck (N K-way Concrete vars); FREEZES the A-bot (warning that fully co-training the answerer is the risky choice).
- **GAP (unclaimed):** co-adapt the fold under a HARD geometric grounding constraint (answer = sign(u*·e) to TRUE taste in the CF embedding) so novel continuous Qs become interpretable → realize the continuous headroom.
**Recipe (S2P):** BC/ground the answerer to true taste → co-adapt with self-play reward while a grounding loss STAYS ON; never from scratch. **Failure modes/mitigations:** emergent private codes / drift / two-sided collapse → hard geometric grounding (answer bit never learnable), info bottleneck / discrete code, persistent S2P grounding loss, information-leakage audit (answer carries nothing beyond the taste sign).

## OPEN QUESTIONS (from the research)
1. M1: does continuity expose NDCG headroom over the discrete winner on a fixed ruler, or representational only? (We: small on dense ML-1M.)
2. M1: TD3-from-scratch vs distill-from-discrete+RL — which is cleaner/lower-variance?
3. M2: is CF-fold measurably > per-item belief / fixed-genome on NDCG? (Carries the thin novelty.)
4. M3: can hard geometric grounding prevent collapse AND leave room for co-adaptation gains?
5. M3: which anti-collapse regularizer transfers to a geometric (non-language) answer channel?

## CITATIONS TO ADD (Paper B novelty positioning)
Wolpertinger 1512.07679 · HyAR 2109.05490 · ConTS TOIS2021 · PEBOL 2405.00981 · GATE 2310.11589 · Montazeralghaem 2510.12015 · OPEN 2403.05534 · VisDial 1703.06585 · Lowe-S2P 2002.01093 · DwD 2007.12750
