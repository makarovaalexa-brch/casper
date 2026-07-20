# Findings: Answerability & Elicitation Channels

*Item vs concept vs open questions; answerability; implicit vs explicit; dislike semantics.* (Papers B & D.)
Consolidated & deduped from answerability-concept-channel-works, LIT_IMPLICIT_VS_EXPLICIT, the deep-research maps. Bib-keys in `../INDEX.md`.

## What we concluded

- **In realistic cold-start, answerable-concept elicitation BEATS item-asking:** tail NDCG 0.116 vs 0.085 (+36%), tail Recall 0.153 vs 0.136, full NDCG 0.315 vs 0.307. Mechanism: cold items are mostly UNANSWERABLE (even a popular item is answered only 1.9/8 turns; random catalogue 0.1/8) while concepts are answerable (8/8) and fold usefully. Concepts survive as the ANSWERABILITY channel — for film-mute users who cannot answer item questions.
- **Two fixes were both NECESSARY** (before them, concepts HURT): (1) OOD bug — the encoder was trained only on items+18 genres, never on the 761 genome concepts it folded at eval; fix = train concepts in as a LEARNED channel (learned embeddings Ec). (2) Answer derivation — replace noisy mean-residual with a GEOMETRIC answer: like/dislike = whichever fold moves belief toward true taste u* = fold(known-half profile); must be used in BOTH training and eval.
- **The concept ANSWER, not the concept, was the bottleneck.** IMPLICIT watch-signal (watch-lift, popularity-corrected) explains ~40% of concept-affinity variance; the stated LLM ordinal explains only ~1.7% (~24× fidelity gap; SEL 0.376, VAL residual-rating 0.042, SEL+VAL 0.398, LLM ordinal 0.017). Item-derived answers are HIGHLY viable and non-circular (pure watch counts) — the author's implicit-signal bet was right (~9× explicit).
- **Static vs elicitation "dislike value" discrepancy is a DEFINITION/DATASET mismatch, cleanly resolved.** Static top-N literature's "explicit negatives add little" is about LOGGED item-level low ratings feeding a point-ranker; conversational literature's "dislike" is overwhelmingly a live, ATTRIBUTE-scoped negative acting as a REGION-PRUNING operator in a small cold-start action space — a genuinely different, cleaner signal. The two literatures measure different objects; Golbandi is the one item-level counterexample.
- **The LLM era SHARPENED attribute-dominance, not reversed it.** Every dedicated question-generation policy (OPEN, PEBOL, GATE, the 2510.12015 funnel) asks exclusively at the attribute/aspect/topic level — never "did you like Titanic." Item-level negatives appear only when volunteered unprompted (human dialogue, CCPE) or pulled from logged history (static prompt context), never as a designed live probe.
- **No surveyed system elicits GRADED two-axis confidence** like CASPER's {hated..loved} × {no_clue..know_well}. Every specified answer format is binary/small-categorical (OPEN pick-A-or-B; PEBOL yes/no; 2510.12015 yes/no/"No Preference"; GATE binary classifier). A genuine open gap.

## Key external methods

- `austin2024bayesian` (PEBOL) — aspect-level short phrases (`"not " + aspect`), SOFT Beta-posterior downweighting via NLI entailment (not hard pruning); preferences 100% LLM-simulated (GPT-3.5). [full-text, high confidence]
- `li2023eliciting` (GATE) — open-ended LM-generated questions; content arm is TOPIC-level (news), binary "would you read this" target; no explicit dislike mechanism; real human participants. [full-text, high confidence]
- `li2021seamlessly` (ConTS) — the one system with an explicit item-level REJECT action alongside attribute dislikes, but "reject item" = reject a shown recommendation this turn (simulated), not a logged low rating.
- `lei2020interactive` (SCPR) — graph-based candidate pruning after attribute rejection; flags its own leaky heuristic (rejecting an attribute ≠ disliking it). `lei2020estimation` (EAR), `deng2021unified` (UNICORN) — attribute-level, discrete.
- `golbandi2011adaptive` — the item-level, point-ranking-relevant exception: splits users by their actual rating on a probe item (lovers/haters/unknowns); real-like item-level rating data doing real structural work.
- `rashid2002getting` / `rashid2008learning` — item-level ratings, but the "value" measured is which items to ASK about (popularity/entropy/HELF), not dislike-semantics.
- `radlinski2019coached` (CCPE-M) — WoZ human-human; free-form ENTITY_PREFERENCE (entity OR aspect), both likes and dislikes; the sole case a human user volunteers item-level dislikes live. No graded scale.
- `kostric2024generating` — USAGE questions ("how will you use it?") vs attribute questions; helps low-expertise users. `kostric2025should` — conversational STYLE axis (how, not what).
- `liang2018variational` (Mult-VAE) + RecVAE — positive-only, implicit; strong top-N is exactly why dislike-blindness is treated as NOT first-order on static benchmarks; but they structurally CANNOT take a "no" as input.
- **Not in references.bib (context):** Hu-Koren-Volinsky 2008 (implicit-only weighted MF is sufficient for top-N — founding result); BPR (Rendle 2009, one-class problem); OPEN (Handa et al. arXiv:2403.05534, attribute-level pairwise, strictly binary, [full-text]); the 2510.12015 clarifying-questions funnel (attribute funnel, no dislike branch, [full-text]); the movie user-study arXiv:2404.19093 (item-level dislike only as static prompt context); CUPID ~2026 (separates likes/dislikes but post-hoc dialogue-MINED, not asked); Yoon et al. NAACL'24 (LLM users unfaithful); M&Ms-VAE / critiquing VAEs (keyphrase-level critique); "How to deal with negative preferences: a theoretical framework" (Springer ~2022, states negative-preference research is thin and no theory of dislike propagating to neighbors exists — a citable "gap unfilled" statement).

## What is NOVEL vs pre-empted

**PRE-EMPTED / established:**
- "Implicit beats explicit" as a DIRECTION = Hu-Koren-Volinsky 2008.
- "LLM stated preferences are unfaithful" = Yoon NAACL'24.
- Attribute-level dislike as a pruning/downweighting signal = SCPR/EAR/UNICORN/PEBOL/ConTS/critiquing-VAEs (the whole conversational line).
- Item-based CF's founding assumption ("similar items get similar ratings") = Sarwar 2001 — used operationally, never independently validated as a diagnostic.

**NOVEL / defensible:**
- **The implicit-vs-stated VARIANCE DECOMPOSITION of latent concept affinity on the same users** (~40% vs ~1.7%, ~24× gap) — no one does this specific decomposition. THE strongest claim (Claim 4). Frame as measurement of a proxy affinity, dataset-specific.
- **Answerable-concept-beats-item in realistic cold-start with a strong recommender**, mechanistically attributed to answerability (items unanswerable 1.9/8 vs concepts 8/8) — the realizable, answerable version of the privileged concept-EIG.
- **Two-axis graded {valence} × {familiarity/confidence} elicitation** — nothing in the 2023-2026 dialogue-elicitation literature pairs a valence scale with an independent confidence scale (genuine open gap).
- **The MF-neighborhood rating-generalization/transitivity diagnostic** (does a user's rating on item i predict their rating on collaboratively-similar items, split by like vs dislike direction) — NO direct hit; treat as a genuine small clean empirical gap, explicitly named unresolved by the 2022 negative-preferences survey. (Medium-low confidence on absence — search pass, not exhaustive.)

## Open questions / caveats (do NOT oversell)

- conc_pop ≥ conc_eig: myopic info-gain optimizes a popularity-coverage PROXY misaligned with the user's-own-likes metric → simple frequency beats it; smarter/non-myopic concept selection is OPEN (oracle 0.587 full / 0.439 tail, uses concepts 22%).
- Full-NDCG gain from concepts is MODEST (+0.008); the clear win is tail/recall — concepts are a TAIL instrument.
- Single dense dataset (ML-1M) + idealized honest-preference user sim; likely STRONGER on a sparse/large catalogue.
- Whether disliking one item tells you much about neighboring items is OPEN/thin — treat "negative preference has neighborhood structure" as UNVERIFIED, not a citable fact.
- Confidence: PEBOL/GATE/OPEN/2404.19093/2510.12015 claims are full-text (high); ConTS/UNICORN/SCPR/CUPID are snippet-level (medium). RecLLM (`friedman2023leveraging`) and Zero-Shot CRS (arXiv:2308.10053) full text could not be extracted — elicitation mechanics flagged "not determinable," not guessed.
