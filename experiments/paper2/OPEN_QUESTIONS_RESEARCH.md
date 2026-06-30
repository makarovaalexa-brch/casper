# Open-vocabulary questions in our continuous space — deep-research + ultrathink (2026-06-30)

**Premise (user):** bot-play/ABot/refusal is a no-go (confirmed). Pivot to OPEN questions ("what's your favourite
genre/actor/movie?"). Open-vocab elicitation is not novel in general, but driving it with OUR learned CONTINUOUS-LATENT
POLICY might be. Is there something there?

## The literature (what exists, and the gap)
- **PEBOL** (Austin/Korikov/Sanner, RecSys 2024, arXiv 2405.00981): Bayesian Optimization + LLM. NL preference elicitation;
  NLI between user utterances and item descriptions maintains a Bayesian belief over a ~100-item SHORTLIST. Acquisition
  (which aspect to probe) = a decision-theoretic/BO rule over the shortlist. **Discrete; not a learned policy; ranks a
  100-item subset, not the full catalog; not trained end-to-end on NDCG.**
- **GATE** (Generative Active Task Elicitation, ICLR 2025): an LLM asks open-ended questions and infers preferences.
  **LLM-driven question generation; no learned selection policy; not grounded in a recommender's embedding.**
- **Asking Clarifying Questions for Preference Elicitation w/ LLMs** (arXiv 2510.12015, Oct 2025): diffusion-inspired;
  fine-tune Gemma-7B to ask "funnel" questions (general->specific). Question SELECTION = **LLM-prompted heuristic** (rank
  profile attributes by generality, prompt to ask). Metric = **BLEU/ROUGE on profile reconstruction**, % unanswered — NOT
  recommendation NDCG. **Discrete, heuristic, no learned latent policy.**
- Movie NL-assistant (Springer JIIS 2023), "Generating Usage-related Questions" (ACM TORS), attribute active learning
  (1805.09023): ask about objective features (genre/actor/director); discrete attribute selection.

**THE GAP:** every open-vocab elicitor selects WHICH question to ask via a DISCRETE heuristic / LLM prompt / BO over a
shortlist. **None uses a learned CONTINUOUS-LATENT policy, grounded in the recommender's own embedding, trained
end-to-end on recommendation NDCG.** That is exactly what our D1 actor is.

## The novelty (as a combination), honestly scoped
**A learned continuous-latent policy that decides WHICH preference aspect to elicit — as a direction in the recommender's
factor space — realized as an open natural-language question by an LLM, and trained end-to-end on NDCG.** Differentiators:
- vs PEBOL: continuous learned policy over the FULL catalog's taste space (not BO over a 100-item shortlist); NDCG-trained.
- vs GATE/2510: the question selection is LEARNED + CONTINUOUS + grounded in the recommender, not an LLM-prompted funnel.
- vs our own Paper B (discrete concepts): open-vocab NL + free-text answers + LLM rendering of OFF-manifold directions.

## The hard problems (ultrathink the risks)
1. **Decode-to-NL (THE risk).** The query is a direction in V1 SVD factor space (NOT SBERT) -> no off-the-shelf decoder
   (Vec2Text/GEIA are GTR/ada-only, lossy). RESOLUTION (already in hand): don't invert -> SNAP to a phrase bank (genome
   tags + mined phrases) and let the LLM RENDER the nearest phrases as an open question. QPROBE already shows D1's
   directions snap to interpretable themes (eerie -> sci-fi -> superhero -> china/paris) at cos ~0.7. Snap-loss is the
   honest cost of naming.
2. **Answer grounding.** Free-text answer ("I love Tarantino", "slow-burn thrillers") -> embed with SBERT -> nearest
   items/concepts -> factor centroid = the (q,a) token. NO inversion needed; the geometric sim already dissolves this for
   train/eval (answer = sign/grade of u*.dir). The LLM/NL is interpretability + deployment only.
3. **Evaluation / simulator.** To score on NDCG we need a simulator that answers open questions. Options: (a) geometric
   sim if the open question maps to a direction (cheap, our ruler); (b) LLM user-simulator (realistic but the documented
   leakage/over-cooperation issues; cf bot-play). Honest path: geometric ruler for the NDCG claim + LLM rendering for the
   deployment demo.
4. **The answerability<->precision tradeoff (the same tension we found).** Open questions ("favourite genre") are MORE
   answerable but COARSER (a genre is a low-resolution direction) than D1's off-manifold queries. So open questions sit at
   the ANSWERABLE end of the spectrum we already mapped: high answerability, lower per-question info. Likely => MORE
   realizable, NOT higher NDCG.

## Honest assessment — where the value is (and isn't)
- **NDCG:** open NL questions ~= concept-asking with NL rendering (concepts = genome tags = genres/moods). Our concept
  baselines sit ~0.36; D1 ~0.378. Open questions are bounded by the concept ceiling -> WILL NOT beat D1 on NDCG. The
  answerable-coarse end of the tradeoff.
- **The real contribution = DEPLOYABILITY + the learned-continuous-policy-drives-open-questions novelty.** A systems/agent
  contribution (Paper D), not a pure NDCG win: the FIRST learned continuous-latent policy rendered as an open conversational
  agent, with the off-manifold value (snap-loss) preserved through LLM rendering. The NDCG story stays Paper C (D1); Paper
  D is the deployable realization + interpretability (named tree) + (optional) bot-play realism as motivation.

## Concrete plan (Paper D), gated
0. Reuse QPROBE/snap: D1's directions -> nearest genome phrases (already interpretable). Build the expanded phrase bank
   (genome + mined review phrases) for richer naming (Paper D task #67).
1. **Aspect grounding:** cluster the factor space into open-question ASPECTS (genre/actor/director/mood/era) = labeled
   subspaces; or just snap each emitted direction to the nearest phrase(s).
2. **LLM render:** prompt the LLM to turn the nearest phrase(s) into a natural open question ("Do you enjoy slow-burn,
   eerie films?" / "Any director you love?"). Render off-manifold = "between" phrases (preserve the snap-loss value).
3. **Answer -> token:** SBERT-embed the free-text answer -> retrieve nearest items/concepts -> factor centroid -> fold.
4. **Evaluate:** NDCG on the geometric ruler for the directions; a small LLM-simulator + human-readability study for the
   open agent. Compare the LEARNED continuous selection vs the funnel/entropy heuristic (PEBOL/2510-style) -> does learned
   continuous selection ask better open questions?
5. **Named decision tree** (the interpretable strategy) as the headline Paper-D figure.

## Verdict
Open questions are NOT a higher-NDCG lever (they're the answerable-coarse end). But "a learned continuous-latent policy
realized as an open conversational agent, grounded in the recommender, with off-manifold value preserved by snap+LLM
rendering" IS a genuine, novel-as-combination Paper-D contribution — and it's the natural deployable home for everything
Paper C proved. The decode-to-NL risk is handled by snap+render (never invert); the geometric answer dissolves inversion.
Cite PEBOL/GATE/2510 as the discrete-selection prior; our differentiator is the learned continuous policy + NDCG training.
