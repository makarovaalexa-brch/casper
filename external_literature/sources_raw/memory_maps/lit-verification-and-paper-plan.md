---
name: lit-verification-and-paper-plan
description: "June 2026 adversarially-verified literature check — novelty verdict (narrowed, survives), IJCNN 2024 paper exists, new thesis chapter written, two-paper plan options"
metadata: 
  node_type: memory
  type: project
  originSessionId: f896d008-8222-4f33-8f51-8f8389a9b8c1
---

Deep-research verification (June 2026, 101 agents, 19 primary sources, 24/25 claims confirmed 3-0):

**Novelty verdict: SURVIVES NARROWED.** All RL-CRS lineage (CRM, EAR, SCPR, UNICORN, HutCRS, DAHCR, CoCHPL) is discrete-action, template questions, no LLM. BUT: Wolpertinger (Dulac-Arnold et al. 2015, arXiv:1512.07679) is exact mechanism prior art (DDPG proto-action + kNN snap, demonstrated ON recommendation with 13k actions) — must be cited as architectural ancestor. ECoC (arXiv:2408.08047) = continuous-action RL for sequential rec. RSO (arXiv:2509.26093, 2025) = nearest analogue: RL planner over 13 discrete strategies + frozen LLM actor. Defensible claim: "first to apply Wolpertinger-style continuous semantic-embedding actions to elicitation question selection in CRS, with LLM verbalization." "LLMs weak at strategic elicitation" still holds but ONLY scoped to prompt-only systems (medium confidence, fast-moving).

**Key discovery: the user's bot-play paper IS PUBLISHED** — Makarova, Shahzad, Hong, Lester, "Learning a Strategy for Preference Elicitation in Conversational Recommender Systems", IJCNN 2024 (centaur.reading.ac.uk/116044). Thesis paper #1 exists.

**Thesis context**: UoR Confirmation report on Overleaf (project 640d0e27d59da61b2bedf2ab); local copy at lit_review/overleaf_report/. Supervisor Dr Muhammad Shahzad. Chapters: intro, background, reinforcement_learning, recommendation_module, bot_play, conclusions. Labels: ch:main (bot_play), ch:con (conclusions).

**Deliverables written (committed 08d84c5)**: chapters/literature_review_2026.tex (drop-in LLM-era lit review chapter, label ch:llm_lit) + references_2026.bib (21 verified entries). User must add \input and merge bib in Overleaf.

**User direction (June 2026)**: wants LLMs heavy, 2 papers, low labour, keep as much of old idea as possible. Proposed plan: Paper 1 = evaluation/benchmark paper (controlled testbed for strategic elicitation: random/popularity/greedy info-gain/prompt-only LLM/IJCNN policy; no custom model training beyond supervised recommender fix; fills verified gap — no standardized elicitation evaluation exists). Paper 2 = method paper (narrowed CASPER claim on that testbed, or verbal bot-play/prompt-space optimization fallback).

No verified screening exists yet for PEBOL/GATE/MACRS internals beyond abstracts; 2026 pubs under-covered — re-check before submission.

**User approved Paper 1 (June 2026).** 10-day sprint constraint: user has intensive help for 10 days only, then continues with Opus. Strategy: front-load ALL code (Paper 1 testbed first, then pre-write 2a Wolpertinger-style trainer + 2b verbal bot-play), avoid blocking waits, write HANDOFF.md for continuation. Chapter expanded to ~6.3k words (~14-15pp) with RQ1-RQ3, committed f6f7df9. Chapter file: lit_review/overleaf_report/chapters/literature_review_2026.tex + references_2026.bib (user must \input in main tex + merge bib on Overleaf).
