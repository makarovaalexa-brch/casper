# Rich-feature learned answerer (ABot) — confidence = FAMILIARITY, calibration PASSES

**Date 2026-06-29.** Pivot from |cos|/movie-distance to a LEARNED confidence over rich features
[popularity/familiarity, divisiveness, experience-max, experience-mean, taste-cos]. ABot(u*,q,feats)->(mu,log sigma^2),
heteroscedastic NLL on 808k real (user,item) ratings. Confidence = 1/sigma^2.

## Calibration gate — ALL PASS (the user's definition holds in the data)
- (1) sigma^2-vs-|err| corr **+0.253** (>0 => knows where it's wrong).
- (2) **THE TEST: popular-unrated => CONFIDENT, niche-unrated => UNSURE.** sigma^2 POPULAR(held) **0.602** < NICHE **0.764**;
  corr(sigma^2, popularity) **-0.236**. Familiarity/popularity IS the answerability driver (NOT |cos|, NOT movie-distance).
- (3) experience boost: experienced-dir sigma^2 **0.726** < random-unexperienced **0.906**.

So a user can confidently answer a MAINSTREAM theme even if unrated; a NICHE one they haven't encountered they cannot.
This reproduces the user's exact intuition and fixes the earlier (rejected) experience-distance-only framing.

## Code
continuous_actor.py: ABot class (D+D+5 rich feats), BOTPLAY=train P0 builds feats from #3 fields (field_pop/field_div) +
experience + taste; 3-test calibration gate. ABot-refusal TRAINING: ABOTREF=1 ABTAU=<sigma^2 cutoff> -> rollout refuses
niche (ABot sigma^2>ABTAU) straight-through so the actor learns FAMILIAR-and-informative asking. Checkpoint .cache/abot.pt.

## Next
Asker co-train vs this learned answerer (ABOTREF), warm from D1, levers: ABTAU (answer-rate), objective, joint. Open
question: does pushing the asker to FAMILIAR+divisive (popular controversial) questions beat D1 0.378/0.178? (|cos| refusal
tied; familiarity refusal is a genuinely different lever.)
