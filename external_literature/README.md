# External Literature Knowledge Base — CASPER

A retrievable, deduped index of all external prior work relevant to the CASPER thesis
(cold-start conversational recommender, ML-25M; papers A=instrument/recommender,
B=set-encoder reconstruction elicitation, C=continuous-action elicitation,
D=open-vocab free-recall, E=policy→dialogue deployment). Purpose: never redo deep research —
every external finding is captured here with its bib-key so claims stay traceable.

## File map
- `INDEX.md` — table of ALL `references.bib` entries: bib-key, authors/year, title, venue, 1-line takeaway (or "citation only"), topic-tag, relevant-to (A–E). Sorted by tag then year; tag legend at top. **Every paper researched must have a row here.**
- `papers/<bib-key>.md` — **one durable record per individually-read paper** (schema: `papers/TEMPLATE.md` — essence, method, relevance, baseline-candidate, verdict). Written the first time a paper is read; the source is never reread after. Only papers that matter get a file; pure-background citations stay INDEX-row-only.
- `findings/elicitation_and_belief_pool.md` — Golbandi trees, EDDI, Biyik Gaussian belief, VoI, adaptive-vs-static, the info-vs-value boundary.
- `findings/continuous_action_policy.md` — PEBOL, HyAR, Wolpertinger, DBU, continuous-vs-discrete RL, embedding→NL inversion (Paper C).
- `findings/unified_embedding_architecture.md` — set encoders, RBMF, UNICORN, ConTS, one-embedding-both-roles (Paper A / C1 enabler).
- `findings/answerability_and_channels.md` — item vs concept vs open questions, answerability, implicit vs explicit, dislike semantics (Papers B & D).
- `summaries_raw/`, `updated_lit_review_2025.md`, `LitReview_Verification_and_Paper_Plan_2026-06.md` — the raw source material these were distilled from.

## How to retrieve (before re-doing ANY lit search)
1. Search `INDEX.md` by topic-tag or bib-key for the paper/area.
2. Read the matching `findings/<topic>.md` — each has **What we concluded / Key external methods / What is NOVEL vs pre-empted / Open questions**. The NOVEL-vs-pre-empted section is load-bearing: it records which claims are already taken (e.g. "Biyik 2023 pre-empts the Gaussian-fold unified-VoI") vs still open.
3. Only if the answer is genuinely absent should you launch a new deep-research pass — then fold the result back into the relevant `findings/` file and add the paper to `INDEX.md`.

Note: several load-bearing external works are cited in `findings/` but are NOT yet in `references.bib` (flagged inline: Biyik 2023, HyAR, DBU, Huang NeurIPS'24, Blau ICML'22, TaNP, P5, Hu-Koren-Volinsky 2008, etc.) — add them before submission.
