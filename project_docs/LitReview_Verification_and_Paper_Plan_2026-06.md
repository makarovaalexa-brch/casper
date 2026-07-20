# Literature Verification & Two-Paper Plan
**Date:** 2026-06-11 · **Basis:** adversarially-verified literature sweep (101 research agents, 19 primary sources read in full, 24/25 claims confirmed by 3-0 verifier votes)

---

## Executive summary

**The novelty claim survives — narrowed, not dead. No pivot needed, but the framing must change.** The entire RL-based CRS lineage — CRM, EAR, SCPR, UNICORN, and the 2023–24 successors HutCRS, DAHCR, CoCHPL — is discrete-action, template-question, and LLM-free. Nobody applies continuous embedding actions to *question selection*, and nobody combines that with LLM verbalisation. **But** the core mechanism (DDPG actor outputs continuous vector → nearest-neighbour snap to discrete entity) is the **Wolpertinger architecture** (Dulac-Arnold et al., 2015, arXiv:1512.07679) — and that paper even demonstrated it on a recommendation task with up to 13,138 item actions. The gap was real; the mechanism was under-cited.

**The defensible claim:**
> *"First to apply Wolpertinger-style continuous semantic-embedding actions to elicitation question selection in conversational recommendation, with LLM-based question verbalisation."*

**Pleasant discovery:** the "competing" paper flagged at Reading's repository (centaur.reading.ac.uk/116044) is *our own IJCNN 2024 publication* — Makarova, Shahzad, Hong & Lester, "Learning a Strategy for Preference Elicitation in Conversational Recommender Systems". The thesis already has paper #1; the new chapter cites it as the foundation.

---

## Key conclusions from the verification

1. **"LLMs are weak at strategic elicitation" still holds — but only for prompt-only systems.** The supporting evidence is mostly motivation statements in 2025 preprints rather than controlled measurement. RL/DPO-trained LLM systems (RSO, Zhang et al. ICLR 2025) are closing the gap fast. This is both a warning (cite carefully, scope the claim) and an opportunity (nobody has *measured* it properly).
2. **The closest rival is RSO (Zhao et al., Sept 2025, arXiv:2509.26093):** an RL planner + frozen LLM actor — structurally the same design, but choosing among 13 discrete *strategy types*, not the semantic *content* of questions. Must be cited and differentiated.
3. **Honest prior-art attributions now required:** Wolpertinger (mechanism, incl. recommendation demo), ECoC (continuous-action RL for sequential recommendation, arXiv:2408.08047), PPDPP/RSO (policy-chooses/LLM-speaks decomposition), GATE/PEBOL (LLM elicitation without learned strategy).
4. **A verified hole in the field:** no established protocol exists for measuring elicitation strategy in isolation — published comparisons evaluate whole systems where elicitation is confounded with generation and recommendation quality. This is a gap we can fill cheaply.

---

## Deliverables (committed `08d84c5`)

- **`lit_review/overleaf_report/chapters/literature_review_2026.tex`** — full thesis chapter, *"Preference Elicitation in the Era of Large Language Models: An Updated Review"*, in the report's voice and template (Harvard/natbib, cross-references to `ch:main`/`ch:con`). Covers: discrete-action RL lineage; Wolpertinger/ECoC honestly attributed; LLM-CRS 2023–26 (zero-shot, iEvaLM, multi-agent, RAG, PPDPP/RSO); LLM-era elicitation (GATE, PEBOL, usage questions, DPO clarifying-question training); evaluation norms; positioning table deriving the narrowed gap.
- **`lit_review/overleaf_report/references_2026.bib`** — 21 verified entries (full author lists checked against arXiv/ACL/ACM pages).
- **To integrate in Overleaf:** add `\input{chapters/literature_review_2026.tex}` after `background.tex` in `main_cs_uor_report.tex`, and merge `references_2026.bib` into `references.bib`.

---

## The two-paper plan

### Paper 1 (recommended, start now): evaluation/benchmark paper
**"Do LLMs know what to ask? Measuring strategic preference elicitation."**

Build the controlled testbed: a fixed, polarity-aware recommender as the measurement instrument; simulated users answering verifiably from held-out MovieLens ratings; compare heterogeneous elicitation policies on information gained per turn:

- random / popularity baselines
- greedy information-gain (the strong non-learned baseline)
- prompt-only GPT asking (several prompting styles)
- the published IJCNN 2024 bot-play policy

**Properties:** no custom model training beyond the supervised recommender fix; ~90% reuse of CASPER infrastructure (simulator, FAISS, logging, diagnostics — the failed V1–V4 runs become motivating evidence that naive reward signals are unlearnable). Publishable whatever the result. Fills the verified evaluation gap. Realistic in ~2–3 months. Venues: RecSys (short/resource), ECIR, UMAP, CIKM.

### Paper 2 (choose based on Paper 1's leaderboard): method paper

- **2a — keeps most of CASPER:** the narrowed continuous-action claim — Wolpertinger-style elicitation policy + LLM verbalisation, trained cheaply with a rule-based simulator (50k+ free episodes), per the recovery plan in `CASPER_Expert_Review_2026-06.md`.
- **2b — zero gradient training, maximally LLM-native:** **"verbal bot-play"** — keep the bot-play loop exactly, but replace DDPG with prompt-space optimisation: the LLM QBot's questioning strategy is iteratively rewritten based on which past episodes scored well on the recommender-based reward (Reflexion/OPRO-style verbal reinforcement). Same reward, same QBot/ABot framing, no PyTorch training. Nothing in the verified sweep does this for elicitation strategy.

**Decision rule:** if greedy info-gain beats prompt-only LLMs by a wide margin in Paper 1, the learned policy (2a) has a clear target; if prompt-only LLMs are nearly optimal, 2b's cheap improvement loop is the honest story. The new chapter's gap section is written to support both.

### Is the continuous action space realistic after all? (reconciling with the expert review)

The expert review called the continuous formulation "the main thing making the problem unlearnable" — that critique was about *how CASPER trained it*, not the formulation itself. Three conditions made it unlearnable: (i) LLM-in-the-loop capped training at ~1k–2.5k transitions; (ii) k=1 FAISS snapping created action aliasing (many proto-actions map to one entity, so the critic's target is ill-posed); (iii) the reward SNR was ~0.025. Wolpertinger prescribes the fix for (ii) directly, and the recovery plan fixes (i) and (iii):

1. **Critic re-ranking (Wolpertinger's k>1):** the actor proposes a proto-action, retrieve the k nearest entities (k = 10–100), then the *critic* evaluates Q(s, entity) for each candidate and executes the argmax. This converts the brittle hard snap into propose-then-refine and largely dissolves the aliasing problem. CASPER already retrieves top-10 but filters heuristically — switching to critic re-ranking is a small, principled change.
2. **Train the critic on the executed entity's embedding,** not the pre-snap proto-action (Wolpertinger trains Q on the applied action). CASPER currently stores the proto-action in the replay buffer — this mismatch must be fixed regardless.
3. **Sample budget still rules:** Wolpertinger itself needed orders of magnitude more steps than CASPER ever ran. Continuous is realistic *only after* the rule-based simulator makes episodes free (50k+), with the high-SNR concept-accuracy reward.

So: continuous-with-LLM-in-the-loop at 100 episodes = unrealistic (proven by V1–V4). Continuous done Wolpertinger-style on free episodes = a fair experiment, and Paper 2a should report continuous vs discrete-PPO as an explicit ablation — either outcome is a publishable finding, since nobody has run that comparison for elicitation.

---

## Housekeeping status

- `casper` repo committed (`ada4a86`): all V1–V4 failure summaries, diagnostics, and training logs now tracked in git (`.pt` weights and regenerable caches ignored).
- PhD root repo (`200ebd4`…`08d84c5`): lit review, expert review, graphs, Overleaf source, new chapter.
- **Open risk:** both repos are local-only — push to a private remote (GitHub/backup drive) for real safety.
- Re-check 2026 publications (esp. anything Wolpertinger-style for dialogue acts, and PEBOL/GATE/MACRS internals) shortly before any submission — the sweep under-covers the last ~6 months.
