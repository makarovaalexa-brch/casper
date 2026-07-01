# CASPER PhD — 4-Paper / 5-Chapter Plan (Jun 2026)

**Vision:** a cold-start conversational preference-elicitation recommender, built bottom-up: *instrument → understand the problem → method → deployable NL system.* Each paper maps to a PhD chapter.

**⚠️ FILE MAP (do not confuse — verified Jun 25):**
- **Paper A (instrument) = `papers/paper1_casper/casper_u_chapter.tex`** ("CASPER-U: A Unified-Embedding Instrument", ML-1M, RQ1–RQ2, structurally complete/locked, single-seed).
- **Paper B (method) = `papers/paper2_casper/paper2_casper.tex`** ("Answerability-Aware…", active draft).
- `papers/_ARCHIVE/paper1_casper_OUTDATED_*.tex` = DEAD old "Controlled Assessment" testbed/bot-play draft (ML-25M). IGNORE.

## Shared foundation (reused by all four)
- Frozen learned **set-encoder** folding heterogeneous reveals (items, attributes, open-vocab concepts) into ONE collaborative-filtering user embedding that is **also the recommender** (score = pop + q·u).
- **Geometric answers** (sign of u*·e — like/dislike by closeness to true taste).
- Shared item+concept **embedding** + FAISS nearest-neighbour snap.
- Full-catalogue **NDCG@10 + Cremonesi long-tail** ruler.
- **Two testbeds:** ML-1M (dense, real) and a **synthetic hierarchical world** (where branching/adaptivity is structurally required).

---
## Paper A — The Calibrated Reconstruction Instrument  [DONE — Chapter 3]
**Thesis:** one frozen set-encoder folds items + attributes + open-vocab concepts into a single CF embedding that doubles as the recommender, with calibrated like/dislike polarity.
**Results:** monotone + no-harm fold; beats MF/WRMF/FM on RMSE/Recall/NDCG; open-vocab concept→item retrieval ("dinosaurs"→Jurassic Park) as a frozen bolt-on; paraphrase-robust (~70% overlap@10).
**Novelty:** unified item+attr+concept fold + open-vocab retrieval (vs EDDI/Partial-VAE, EAR, P5/LLM-recommenders).

## Paper B — When Does Conversational Elicitation Pay?  [THIS — in progress — Chapter 4]
**Thesis:** characterize *when* elicitation/adaptivity helps cold-start, and the *realizable frontier*.
**Results:**
1. **Answerability lever** — answerable concepts beat (unanswerable) items by **+36% tail NDCG** in realistic cold-start (items ~2% answerable, concepts ~57%).
2. **Realizable discrete frontier** — greedy/heuristic (divisiveness) is near-optimal: BC, oracle-distillation, REINFORCE, two-step lookahead all *match* it; the privileged-oracle headroom is *unreachable* (privileged-imitation gap, Weihs et al.).
3. **A learned policy mildly beats the (corrected, FAIR) entropy heuristic on the tail** (+0.010, 6 seeds × 3 epochs) by **distilling the heuristic + RL-refining**; **oracle-distillation ties** across the whole lever grid (the two-part ablation: oracle = answerability *mechanism* item→concept; entropy-distill = the *beater*). Methods-integrity: found + fixed a *crippled* entropy baseline.
4. **★ Synthetic-world existence proof** — on a hierarchical world where branching is structurally required, **oracle-adaptive has a large headroom that every static/heuristic baseline class fails to capture**, but **our BC-distill→RL policy captures a real share of it by branching on the belief** (the same recipe that wins on ML-1M, here in a world where it has room). ⇒ elicitation is ~flat on dense real catalogs (CF generalizes) but has large headroom **when structure demands it.** *(This is what converts "mild ML-1M win" into a real characterization.)*
   - **Same ruler as ML-1M (rebuilt Jun-24):** NDCG@10 full(+tail) + the current model lineup (prior, pop_item, random, conc_pop, FAIR entropy, static_best, oracle_adaptive, our policy) — the old AUAC + paper-1 PPO/DQN/REINFORCE numbers are DISCARDED. Harness `scripts/paper2/synth_ndcg.py`.
   - **One intentional regime difference (documented):** synthetic = COLD-START with TASTE-based answerability (you must *ask* to discover the group); ML-1M = warm profile-split. This is the scientific point, not an inconsistency — the synthetic world is the controlled regime where adaptivity is *forced*. (Forcing the synthetic world into the warm-profile harness would let the known half reveal the group → answerability pre-routes → headroom vanishes == the ML-1M finding.) Budget `T=4` so a static seq cannot cover both groups' clusters.
**Novelty:** answerability-granularity framing + realizable-frontier characterization + the synthetic existence-proof. (cite Rashid, Golbandi, Elahi; Blau/Bakker; Weihs; Cremonesi.)
**To do:** finish results rewrite (fair entropy + entropy-distill — IN PROGRESS); **synthetic world rebuilt on consistent ruler — VALIDATING headroom + policy>entropy, then seed-avg + write subsection**; harmonize tables to te[300:]; regenerate the decision tree from the winner.

## Paper C — Continuous Open-Vocab Unified-Action Elicitation  [next — Chapter 5]
**Thesis:** one continuous RL actor asks a continuous *mixture* of items AND open-vocab concepts via nearest-neighbour snap (not the split ask-attribute-then-recommend pipeline).
**= Milestone 1 (continuous unified-action) + Milestone 2 (open free-text fold).**
**Novelty (deep-research wsjz8y92u):** NARROW-IT-DOWN, the **cleanest** of the three — no prior work snaps a continuous actor to a *heterogeneous* {items}∪{open-concepts} union (Wolpertinger snaps one homogeneous type; ConTS unifies but by *discrete* argmax). (cite Wolpertinger 1512.07679, HyAR 2109.05490, ConTS TOIS2021; PEBOL/GATE/OPEN/Montazeralghaem for M2.)
**Recipe:** TD3/DDPG actor → FAISS snap over {items}∪{concepts} → critic on snapped action; A/B train-from-scratch vs **distil-from-discrete-winner + RL** (lower variance). **Failure modes:** proto-action collapse (→ HyAR semantic-smoothness + diversity bonus); snap imbalance (→ temperature-balance pools); **continuity = representation not NDCG on dense ML-1M** (cont-oracle only +0.016 tail) → the NDCG payoff likely needs the **sparse catalog**.

## Paper D — Grounded Conversational Elicitation: Co-Training the Answerer  [last — Chapter 6]
**Thesis:** a *deployable NL* system — an LLM verbalizes the chosen concept and parses the answer, and a **co-trained answerer stays grounded to true taste** (no metric-gaming), realizing the continuous headroom.
**= Milestone 3 (bot-play co-trained answerer) + the LLM deployment loop.**
**Genuine research:** the **geometric grounding constraint** (answer = sign u*·e) that prevents self-play collapse (already hit in *bot-play v2 fixed-sequence collapse*) while enabling open continuous questions; the **NL-robustness eval** (does the LLM loop preserve the strategy's gains?).
**Novelty (deep-research):** grounded-geometric-answerer-in-a-CF-embedding (vs VisDial 1703.06585, Lowe-S2P 2002.01093, DwD 2007.12750, Montazeralghaem 2510.12015). Highest ceiling, highest risk.
**NOT the contribution:** LLM verbalize/parse plumbing — engineering (git: decode "~free"; the naked-LLM-asker is a *negative* baseline). **Frame D around grounding, not plumbing.**
**Recipe (S2P):** ground-first then self-play, both signals on; anti-collapse = discrete info-bottleneck + persistent grounding loss + information-leakage audit.

---
## Sequencing
Squeeze ML-1M discrete (Paper B) → milestones M1→M2→M3 (Papers C, D) → **replicate on a 2nd/sparse dataset** (strengthens B's "elicitation pays when structure demands it" *and* C's NDCG payoff). Owner order: **milestones before the dataset**, but the sparse dataset is coupled to C's NDCG story.

## LLM: research vs engineering (the owner's worry, answered)
The LLM verbalize/parse loop is **engineering**; the genuine research is (a) the answerability/adaptivity **characterization** (Paper B + synthetic), (b) the open-vocab **fold** (M2/Paper C), (c) the **grounded co-trained answerer** (M3/Paper D). Paper D is publishable *iff* framed around grounding.
