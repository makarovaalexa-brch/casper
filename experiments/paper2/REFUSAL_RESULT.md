# Refusal / answerability-as-SOTA-booster — NEGATIVE (does not beat D1)

**Date 2026-06-29.** Hypothesis (user): an answerer that REFUSES washed-out questions (binary, |cos(u*,q)|<TAUR =
this user is indifferent => refuse, turn WASTED) forces the policy to ask per-user-DECISIVE questions => sharper belief
=> beat D1 (0.378/0.178). Wasted turns kept (no re-ask, per user). Mechanism: straight-through refusal gate in the
training rollout (continuous_actor.py, _REFUSE/TAUR/REFTMP), warm-start from D1, DIVW kept.

## Diagnostic (DIAGWASH) — headroom EXISTS
D1's questions, |cos(u*,q)| per-user: mean 0.384; ~15-20% washed-out (|cos|<0.15-0.20). Concentrated in the OPENER
(turn0 frac<0.15 = 25%) and the SATURATING LATE TURNS (turn6 30%, turn7 36%). So there was a real target.

## Result — TIE tail, LOSE full (two objectives, seed-avg graded vs D1 0.378/0.178)
- refuse1 (OBJ=ustar recon + DIVW + refuse): best ep6 0.372/0.179.
- refuse2 (recon + 0.3*softNDCG + DIVW + refuse): best ep8 0.370/0.179.
- Both: TAIL ~0.179 (= D1 0.178, inside noise) ; FULL ~0.370 (< D1 0.378). NO win.
- TAUR=0.15, REFTMP=0.05, warm from D1, EP=10, graded post-hoc eval (binary VAL is the confound, ignored for selection).

## Why (interpretable) — refusal trades head-signal for tail-sharpening the divisiveness reward already has
Refusing washed-out (low-|cos|) questions slightly sharpens TAIL-relevant belief, but those weak answers carried diffuse
signal that helped rank the POPULAR HEAD => FULL drops. Net: wash on tail, loss on full. Divisiveness already captures
the per-population-informative directions; per-user refusal adds little on top and costs the head.

## Verdict
Answerability/refusal is NOT a SOTA booster here. D1 (0.378/0.178) stays the policy. Frame answerability as the bot-play
ROBUSTNESS / honesty stress-test (the gain is sensitive to realistic refusing answerers) + the motivation for Paper D's
LLM renderer (which makes off-manifold directions answerable), NOT a performance lever. Checkpoints policy_refuse_d1_*,
policy_refuse2_d1_*. Code: continuous_actor.py _REFUSE gate + DIAGWASH block.

## DECISIVE: drop-oracle headroom is OVERFITTING (2026-06-29)
User insight: ML ratings have red herrings; Paper-B oracle best-subset >> full; build a DROP-oracle on D1 (no injected
noise) -- if dropping answers raises held NDCG, headroom for a learned refusal. Backward-greedy drop on D1's 8 geometric
(noiseless) answers:
- PRIVILEGED (select subset on held likes = eval on same): TAIL 0.178 -> **0.243 (+0.066)**, drops ~1.5/8. Looks huge.
- **OVERFIT TEST (select on half the held likes, EVAL on the disjoint half): gain +0.066 -> -0.008. VANISHES.**
So the +0.066 was the oracle exploiting noise in the small held-like set (dropping answers that help the SPECIFIC held
items). NO generalizable red herrings. A realizable predictor has strictly LESS signal than this generous test -> NO
realizable refusal headroom. Whether an answer "hurts" depends on the unobserved target = noise to the policy.
Code: SUBSETORACLE block (OPT=tail|full, OPTSPLIT=1 for the disjoint test) in continuous_actor.py.

## FINAL VERDICT on answerability/refusal
Conclusively NOT a SOTA booster: |cos| refusal ties, familiarity refusal hurts, noisy-frozen no-recover, drop-oracle =
overfitting. D1 0.378/0.178 stays SOTA. KEEPABLE results: (1) the calibrated learned answerer (rich ABot,
familiarity=answerability, ABOT_CONFIDENCE_RESULT.md); (2) the answerability<->informativeness TENSION; (3) false info
devastates NDCG (noisy 0.378->0.308) -- motivates Paper D's LLM renderer. The negative is the honest bot-play story.
