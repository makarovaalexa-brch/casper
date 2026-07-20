# POLICY DESIGN SHEET — STANDING TODOS (Fable, 2026-07-09; inputs for the post-D-ANS sheet)

1. GRADIENT/SNAPPED-CONTINUOUS POLICY CLASS ("ask the gradient"): direction = d(ranking objective)/dz
   through the frozen decoder; candidate score = E_outcomes[ Dz(q, outcome) ] . direction, where the
   expectation runs over the distilled answerer's outcome distribution per level and EACH outcome
   contributes TWO tokens (implicit knowledge token + explicit value token — author correction: two
   signal sources, never a scalar answerability discount; the k=0 branch also contributes, a
   never-heard is implicit evidence). Snap = argmax over askable questions in the user's predicted
   territory. Wolpertinger lineage; fidelity-boundary "snapping denoises" = the mechanism; reconciles
   Papers C & E ("the best question has no name — imagine it, ask its nearest named neighbor").
   Warning baked in: direction must be VALUE-bearing (ranking gradient), never geometric novelty
   (measured anti-correlated, twice).

2. BASELINE LADDER for any policy claim (E7 symmetry throughout, all trained on the same synthetic
   world where applicable):
   a) concept divisiveness-entropy static — continuity row (Paper B's old champion);
   b) LEARNED static, population-scale (greedy sequence trained on synthetic users) — the bar;
   c) static+skip / conditional static — the cheap-adaptivity assassin;
   d) MODEL-BASED MYOPIC GREEDY — no learning: per-user per-turn one-step expected-gain maximizer
      computed with the SAME distilled answerer + fold the policy uses. The sharpest threat: beating
      it isolates the value of look-ahead/policy structure beyond pure computation;
   e) oracle ladder rows (labelled) for headroom context; open-recall reference rows (out of probing
      scope, reviewer-anticipation only).

3. DEFERRED (author): b5 LLM-INTERVIEWER baseline (cross-family, transcript-only, <=$8) - run
   after a working policy exists; pre-empts the 'LLM as interviewer' reviewer challenge.
