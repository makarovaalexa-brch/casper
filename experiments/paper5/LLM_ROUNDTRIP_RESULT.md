# Paper E: triangulation + LLM-rendering round-trip (2026-07-01) — RECORDED, NOT IN PAPER YET

## Triangulation (TRIANGULATE block, continuous_actor.py) — interpretability WITHOUT snap-loss
Continuous D1 query -> sparse SIGNED k-phrase blend via OMP over the 6724-phrase bank. Metrics stay continuous (0.377/0.175).
Reconstruction fidelity cos(q,blend): k=1(=snap) 0.726, k=2 0.800, k=3 0.842, k=5 0.896, k=8 0.942.

## LLMDROP (real NDCG cost of verbalising) — n=150 users, q=8
| | full / tail |
|---|---|
| CONTINUOUS (fold true q) | 0.388/0.182  (sanity ~= canonical 0.377/0.175) |
| TRIANG (fold OMP-5 q_hat, NO LLM) | 0.380/0.175   representation drop **-0.008/-0.007** |
| VERBALISED gpt-4.1-mini (fold LLM->SBERT q_hat) | 0.290/0.102   language drop -0.090/-0.073 |
| VERBALISED gpt-5 | 0.278/0.096   language drop -0.102/-0.079 |
KEY: (1) TRIANGULATION IS NEARLY LOSSLESS (repr -0.008) => the interpretable blend captures the query for free.
(2) The ~0.09 "language" drop is LLM-INDEPENDENT (gpt-5 ~= gpt-4.1-mini, even slightly worse) => bottleneck is the SBERT
back-projection (NL->phrases), a SIMULATION PROXY, NOT the LLM. In real deployment the user answers the NL directly (no
re-embed), so 0.09 is an inflated UPPER BOUND. Cheap model suffices (deployable). gpt-5.5 = quota-blocked (429) + reasoning
ate the token budget (400). To get a paper-grade verbalised cost: use an LLM-ANSWERER (profile+NL -> answer) instead of SBERT.

## ⚠⚠ RETRACTION (user-caught, Jul 1): the SBERT back-projection is BOGUS -> "language cost" is INVALID ⚠⚠
The VERBALISED path re-derived q_hat from the NL via SBERT (NL text -> nearest phrase labels -> reconstruct). MEASURED on 13
frontier-quality (Claude) renders: round-trip cos(q, SBERT-q_hat) = 0.026 (near ZERO, several NEGATIVE) vs OMP-3 ceiling 0.84.
=> SBERT embeds QUESTION sentence-semantics, which land nowhere near the short-noun-phrase labels; the NL does NOT translate
back. So the LLMDROP "verbalised NDCG ~0.29 / language drop ~0.09" was measuring SBERT's FAILURE (folding a ~random q_hat ->
NDCG ~= popularity floor), NOT the LLM or deployment cost. RETRACTED. CORRECT deployment fold = the TRIANGULATED q_hat we
ALREADY have (do NOT re-embed the NL): that's the TRIANG rollout = -0.008 NEARLY LOSSLESS. The NL is the human SURFACE; its
geometric meaning IS the q_hat it was built from. LLM's only job = DESCRIBE q_hat in words (interpretability). Whether a HUMAN
answers the sentence as they'd answer q_hat geometrically = a HUMAN STUDY (SBERT cannot proxy it, cos~0). Drop SBERT entirely.
Claude-manual renders saved: experiments/paper5/renders_claude-manual.json (13 axis-synthesis examples, incl 4 collinearity
fallbacks). LLMDROP VERBALISED path + roundtrip.py fidelity metric should be REMOVED/reframed; keep TRIANG (repr -0.008).

## Rich vocab vs single point (over 2087 dumped queries)
nearest single CONCEPT (genome tags/genres, n=1114): mean cos 0.661, p50 0.677, 12% below 0.5.
nearest ANY phrase (full rich vocab 6724, +pairs/people/clusters): mean 0.679 (barely better).
OMP blend fidelity: concepts-dict vs rich-dict -> k1 0.704/0.720(+.016), k3 0.808/0.838(+.030), k5 0.857/0.893(+.036).
=> rich vocab BARELY helps single-SNAP (+0.016) but modestly helps the BLEND (+0.03-0.04, grows with k). Query is
FUNDAMENTALLY off-manifold: even 6724 phrases x 5-blend caps at 0.89, not 1.0. Rich vocab's value = the blend DICTIONARY,
NOT snapping fidelity (consistent with rich vocab not helping the discrete policy either).

## Semantic-geometric GAP (illustrate.py, NOLLM geometric half)
The geometrically-nearest concept is often SEMANTICALLY unrelated: a Tim-Burton-ish q snaps to [ghosts/afterlife] cos 0.36;
[William H Macy]+[kids] blend -> nearest concept [midlife crisis] cos 0.45. Embedding = co-viewing geometry, not meaning.
=> No single named concept is close in BOTH meaning and geometry; only the weighted blend is faithful to both. LLM synthesises
a semantic AXIS; the concept its WORDING text-matches to may sit even FURTHER from q geometrically (to be measured when quota back).

## "Decode through SBERT instead of snapping?" = embedding INVERSION = the AVOIDED risk
q lives in 64-d co-viewing (Q_svd) space, NOT SBERT text space; the phrase bank is the ONLY bridge (each phrase has both a
Q_svd centroid and a text label). Decoding a vector->text needs Vec2Text/GEIA: NO reliable decoder for all-MiniLM (Vec2Text
GTR/ada-only, GEIA 53-63% F1). Triangulation SIDESTEPS this: decompose onto REAL phrases (text already known), never invert;
LLM paraphrases the phrase BLEND, not the raw vector.

## Cost (OpenAI July 2026, 300-query sample): gpt-4.1-mini $0.04, gpt-5 $0.17, gpt-5.5 $0.61; full 6-model grid ~$3.
## MISS: first LLMDROP run did NOT persist renders (in-process cache) -> outputs lost. FIXED: renders now saved to
experiments/paper5/renders_<MODEL>.json (LLMDROP + illustrate.py share it; re-runs free). RERUN when OpenAI quota topped up.
## Files: TRIANGULATE/LLMDROP blocks in scripts/paper2/continuous_actor.py; scripts/paper5/roundtrip.py, illustrate.py.
See memory [[triangulation-llm-rendering]]. DO NOT put in paper until verbalised cost re-run with saved renders + (ideally) LLM-answerer.
