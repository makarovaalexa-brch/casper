---
name: paperc-continuous-deep-research
description: "Paper C (continuous-action elicitation) deep-research verdicts — what's novel, the SBERT decode-to-NL risk, \"lossless\" refuted, cleanest framing, headroom-first next step"
metadata: 
  node_type: memory
  type: project
  originSessionId: f896d008-8222-4f33-8f51-8f8389a9b8c1
---

June 2026 deep-research (102 agents, 20 primary sources, 22 verified / 3 killed) on the continuous-action Paper C built on CASPER-R. Per-angle verdicts:

- **Decode embedding→NL**: NARROW-IT-DOWN. Vec2Text (Morris, EMNLP 2023) ~92-94% exact recovery but ONLY on GTR/ada-002, NOT SBERT. SBERT-family generative inversion (GEIA, ACL-Findings 2023) only ~53-63% token-F1 = semantic gist not verbatim; collapses OOD + with text length. DeCap (ICLR 2023, CLIP). No published decoder for CASPER's all-MiniLM-L6-v2 384-dim over Tag-Genome → must TRAIN one per-encoder. Decode-to-nearest-SBERT is trivial (projection/kNN); decode-to-NL is the real risk.
- **Discrete→continuous transfer**: NARROW-IT-DOWN. Chandak 2019 "Learning Action Representations" (ICML) = exact structure (internal policy + deterministic f→discrete) but learns geometry from DYNAMICS; CASPER uses FROZEN SBERT (the narrowable twist + it's decodable). Wolpertinger 2015 (Dulac-Arnold) = canonical emit-proto-action + kNN-snap. CRITICAL: Chandak's "lossless value" Theorem 1 was REFUTED in verification (1-2) — frame transfer as knowledge-PRESERVING warm-start/distill, NOT provably lossless. DROP "without loss" language (the one phrase reviewers would kill).
- **Hybrid discrete+continuous (keep emitting discrete concepts)**: NOT-NOVEL as machinery. HyAR (Li, ICLR 2022) = closest decodable-hybrid template (embedding table + cVAE, argmin-L2 snap); its LSC (latent-space constraint, clip to on-manifold) + RSC (representation-shift correction) = ready-made snap-collapse fixes. P-DQN / Hybrid SAC established; P-DQN scales poorly to ~1361 entities → Wolpertinger/HyAR snap is the right fit, cite P-DQN as background only.
- **Open-vocab NL concept generation**: RESOLVED (followup search Jun 2026). NOT-NOVEL for generation alone. GATE (Li/Tamkin/Goodman/Andreas, arXiv Oct 2023→ICLR 2025) = LM directly generates open-ended free-form questions/edge cases by prompting (domains incl. content recommendation). PEBOL (Austin/Korikov/Toroghi/Sanner, RecSys 2024, arXiv 2405.00981) = BO acquisition (Thompson/UCB) over item-utility Beta posteriors + LLM extracts ≤3-word ASPECTS on-the-fly + phrases yes/no queries via NLI beliefs — strong cold-start (MRR@10 0.27 vs 0.17; +131% MAP@10). 2510.12015 (2025) = diffusion-framed ("denoise the user profile") LLM clarifier. NONE use a continuous embedding actor, a unified item∪concept CF space, or embedding→NL inversion — all generate by DIRECT LLM prompting (GATE/2510) or BO-select-then-LLM-phrase (PEBOL). SHARPENED IMPLICATION: PEBOL already proves principled-selection + LLM-phrasing works for NL-PE → a reviewer asks "why decode a lossy embedding to NL when an LLM phrases fluently for free?". Paper C's decode-to-NL only earns its keep via the NOVEL-DIRECTION capability (propose a taste direction with NO existing attribute/item name, which fixed-candidate PEBOL cannot represent) — and that is EXACTLY what HyAR-LSC (on-manifold clip) would blunt. So novel-direction is the LOAD-BEARING differentiator vs PEBOL, not open-vocab generation (PEBOL has it) nor principled selection (PEBOL has it).
- **Combined novelty**: ConTS (Li, TOIS 2021) unifies attributes+items in one arm space BUT fixed predefined set + DISCRETE argmax. Paper C's novel delta = (a) CONTINUOUS actor not argmax, (b) OPEN-VOCAB not fixed, (c) decode-to-NL. Combination of all three for elicitation = genuinely NOVEL (no surveyed work does all three).

CLEANEST FRAMING: FIRST unified continuous-action elicitation policy that warm-starts/distills discrete CASPER-R via Wolpertinger/HyAR-style action representations over a FROZEN unified item∪concept space, closing the loop to open-vocab NL via embedding inversion; HyAR LSC/RSC to control snap-collapse.

## ⚠ CORRECTIONS (2026-07-14 lit sweep) — the above is WRONG in three places
1. **HyAR's LSC/RSC are NOT regularizers / NOT loss terms.** LSC = an OUTPUT BOX CONSTRAINT (the actor's
   latent output is clipped to the 96th-percentile range of observed latents; c=80% "severely degrades").
   RSC = REPLAY-BUFFER RELABELING (refresh stale latents as the encoder keeps training). Saying "LSC/RSC
   regularizers penalize off-manifold actions" is a factual error a HyAR-literate reviewer WILL catch.
2. **THE TENSION LARGELY DISSOLVES.** HyAR/VQ need snap-consistency because their LEARNED DECODER CAN FAIL
   (a latent can decode to garbage). OUR SNAP CANNOT FAIL — nearest-neighbour into a bank of REAL phrases
   means EVERY point in R^512 maps to a valid, askable question. We have no snap-VALIDITY problem, only a
   snap-FIDELITY one. => We need NO consistency constraint, and any commitment/consistency loss
   (`lam*||q - snap(q)||^2`, VQ's commitment term) is DIRECTLY ADVERSARIAL to the thesis: it optimizes away
   the quantity we claim is valuable. Run it ONLY as a pre-registered lam-sweep showing a NON-MONOTONE curve
   with an INTERIOR optimum (precedent: PLAS, CoRL 2020, eps=0 -> 44.6; eps=0.1 -> 66.9; eps=0.5 -> 39.2).
3. **"No decoder exists for all-MiniLM" is now RISKY.** 2025 zero-shot inverters (ZSInvert arXiv 2504.00147;
   vec2vec arXiv 2505.12540) claim encoder-agnostic recovery. The snap DECISION stands, but re-justify it on
   ASKABILITY/ANSWERABILITY (bank phrases are real, calibrated, known-answerable; an inverted string is an
   UNCONTROLLED question of unknown answerability) — NOT on decoder non-existence. Stronger argument anyway.

**THREAT TO THE SNAP-LOSS NUMBER: is our snap top-1?** Wolpertinger reports k=1 snapping FAILS to learn on a
13k-action recommender (a low-Q action can sit nearest to the proto-action). If our snap is top-1 with no
critic re-rank, part of the measured snap loss is a k=1 ARTIFACT, not the continuity prize. Run proper
Wolpertinger (retrieve k~120 = 5% of the bank, re-rank with a value head) BEFORE defending the number.

**THE ARCHITECTURE TO BUILD (Breaking the Grid, arXiv 2602.08616, 2026 — preprint):** DBU trains the actor by
regressing onto a **Q-softmax-weighted BLEND of retrieved candidates**, not onto the nearest one. All gradient
signal comes from EXECUTABLE, EVALUATED phrases (no snap-blindness); gradient variance is independent of bank
size; and the regression target is GENERICALLY OFF-CATALOG (a blend of phrases is no phrase) => it TEACHES the
actor to land BETWEEN phrases. It is the SAME OBJECT as our Paper E OMP sparse-blend renderer: the mechanism we
use to SPEAK the query is the mechanism we should use to TRAIN it. **Ask the blend, not the nearest phrase.**

**THE EXPERIMENT MOST LIKELY TO HURT (run it ourselves first):** densify the bank 10x (composed phrases). If
snap loss COLLAPSES, continuity is merely a COVERAGE claim (weak thesis). If it PERSISTS against a 10x bank,
the prize is about directions NO PHRASE NAMES (strong thesis). Precedent for held-out action sets: Jain/Szot/
Lim, "Generalization to New Actions in RL", ICML 2020.

Also: off-catalog emission as a PRIZE (rather than a bug to suppress) appears to be UNCLAIMED GROUND — good for
novelty, but it means there is no solution to borrow; we must supply one.

TENSION (superseded — see correction 2): LSC constrains the actor to the on-manifold decodable region → may KILL the "propose a novel direction" benefit that motivates continuity over discrete argmax.

NEXT STEP (disciplined, per [[paperB-roadmap]]): measure continuous-vs-discrete oracle HEADROOM FIRST — does a continuous oracle (best point in embedding space, UNRESTRICTED to the 1361 pool) beat the discrete best-subset oracle on full/tail NDCG@10? If not, Paper C is a capability/framing story, not a performance story — know that before investing in a decoder. Then fill the Angle-4 gap (GATE/PEBOL/OPEN). See [[unified-embedding-novelty-map]] and [[answerability-concept-channel-works]].
