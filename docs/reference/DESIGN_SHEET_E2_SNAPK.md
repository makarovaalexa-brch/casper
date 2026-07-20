# DESIGN SHEET — E2 (continuity vs adaptivity) + SNAP-K (defend the snap-loss centerpiece)
2026-07-14. Author signature required before any run. Both are CHEAP and both defend EXISTING,
ON-DEADLINE claims (Paper C, ECIR 2027, 2 Oct 2026). Neither requires new data or LLM calls.

HARD RULE #1 restated: no sampling/capping/subsampling/truncating/top-N of users, items, questions,
answers or candidates without the author's explicit sign-off. All test users, all seeds, full pool.

---

# RUN A — SNAP-K: is our snap-loss real, or a k=1 retrieval artifact?

## The question / the decision
**[V] `continuous_actor.py:187`: our snap is `argmax(q . POOL)` — TOP-1 cosine, NO critic re-rank.**
**[V] Wolpertinger (Dulac-Arnold et al., arXiv:1512.07679) reports k=1 FAILS** on a 13,138-action
recommender: "actions with low Q-value may occasionally sit closest to a-hat even in a region where most
actions have high Q-value." They need k ~ 5-10% of |A| PLUS a critic re-rank to recover full performance.
Our snap-loss headline is **-0.037 full / -0.040 tail** (D1 un-snapped 0.378/0.178 -> concept-snap
0.341/0.138), and per our own memory **snap-loss is the CENTERPIECE / "the deciding honesty test"** of
Paper C's continuity claim.
=> **Part of that -0.037 may be a k=1 RETRIEVAL ARTIFACT, not the continuity prize.** A reviewer who knows
Wolpertinger asks this immediately and we currently have no answer.

**DECISION THIS RUN MAKES:** does Paper C's centerpiece survive a *properly implemented* Wolpertinger snap?
- Snap-loss COLLAPSES under k-NN + re-rank => our centerpiece was inflated; the continuity story weakens to
  a coverage claim; we must say so.
- Snap-loss PERSISTS => the prize is about directions **no phrase names**; the claim is now DEFENDED against
  the obvious attack, and stronger for it.

## Design
Arms (identical checkpoint, seeds, ruler, answerer, graded fold; only the SNAP differs):
1. **UNSNAPPED** (upper bound, not deployable) — fold `q` directly. Anchor: 0.378/0.178.
2. **SNAP-1** (current) — `argmax(q . POOL)`. Anchor: 0.341/0.138.
3. **SNAP-K + RE-RANK** (proper Wolpertinger) — retrieve top-k by cosine, then pick
   `argmax_{p in kNN_k(q)} V(state, p)` where V = the value head (the belief's expected NDCG improvement, or
   the existing critic if one is available; if none exists, use the one-step NDCG of the folded belief =
   an EXACT, non-learned re-rank — state which was used).
   **k SWEEP: k in {1, 5, 20, 60, 120, 250} (250 ~ 10% of the 2,428 bank). This is a sweep, NOT a cap** —
   k=|POOL| (full re-rank) is included as the ceiling arm.
4. **SNAP-|POOL|** (full re-rank over the WHOLE bank) — the ceiling of the re-rank family. If this ties the
   unsnapped arm, then the "snap loss" was ENTIRELY a retrieval artifact and there is no continuity prize
   at all in this measurement. **This is the arm that can kill the paper's centerpiece — run it.**

Metrics: full + tail NDCG@10, seed-averaged {1,2,3,7,11}, te[300:], q=8. Report snap-loss as
(un-snapped - snapped) for EACH k.
**Pre-registered MDE:** the disputed effect is ~0.037; with seed-avg{5} the observed run-to-run sd is
~0.003-0.005 => a 0.037 effect is ~8-12 sigma and easily resolved. A NULL here is therefore interpretable
(unlike E2 below).

## Traps
- **Baseline drift:** verify the un-snapped and SNAP-1 arms reproduce 0.378/0.178 and 0.341/0.138 EXACTLY
  before believing any new arm (the corrupted-pool_entavg lesson: a stale cache silently moved baselines).
- **Re-rank leak:** the re-rank value head must NOT see the true user factor `u*`. If the only available
  "critic" peeks at u*, the arm is an ORACLE and must be labelled as such (it then bounds, not realizes).
- Report the ANSWERABILITY of the snapped pick per k — if larger k lands on more obscure phrases, part of any
  gain/loss is answerability, not retrieval.

---

# RUN B — E2: was Paper C's win CONTINUITY or ADAPTIVITY?

## The question / the decision
Paper C's headline: continuous actor **0.366/0.162** vs discrete CASPER-R **0.360/0.152** (+0.006 full ~2 sigma,
+0.010 tail ~3 sigma). **That comparison changes TWO variables at once: the ACTION SPACE (continuous vs
discrete catalog) AND the POLICY (learned/adaptive vs static).** We have NEVER run the one-variable control:
**a continuous STATIC schedule.** (R2b ran it under NOISE and found a tie, p=.82; under CLEAN answers it was
never run.)
Nothing in the non-adaptivity theorem forbids a continuous *static* schedule from beating a discrete one — a
continuous query is simply a BETTER QUESTION, whether or not it ever conditions on an answer.

**DECISION:** which mechanism does Paper C actually own?
- Continuous-STATIC ~= continuous-ADAPTIVE => the win is **CONTINUITY**. Clean, survives, and it aligns with
  everything else we know ("the learning is in the recommender + answer; policy ~ static entropy").
- Continuous-ADAPTIVE > continuous-STATIC => **adaptivity IS realizable in our arena**, the H(Theta)
  explanation is wrong or incomplete, and the whole 2026-07-14 synthesis needs revision. **Either outcome is
  a real finding.**

## Design
Arms (identical action space, answers, ruler, seeds, budget q=8; ONLY the conditioning differs):
1. **Continuous ADAPTIVE actor** (existing, `policy_phase2_cont_v1_best`).
2. **Continuous STATIC schedule, construction A — INDEPENDENT**: greedy-forward selection of a FIXED sequence
   of q vectors on the TRAIN split, maximizing population NDCG; frozen; applied to every test user.
3. **Continuous STATIC schedule, construction B — ACTOR-DISTILLED (R2b style)**: the modal/mean query
   sequence extracted from the actor's own rollouts, frozen.
**BOTH constructions are required.** They bracket the answer, and each alone can LIE (below).

## The two ways this test LIES (must be closed, or the result is worthless)
1. **MULTIPLICITY ASYMMETRY.** The actor is a RECORD-THE-PEAK best-checkpoint selected over many runs; a
   one-shot static schedule loses to selection noise alone => a FALSE "adaptivity is real".
   **CLOSURE:** give the static arm the SAME selection budget — construct N=(number of actor candidates)
   static schedules (different train seeds / greedy tie-breaks) and select the best on the SAME disjoint val
   split the actor's checkpoint was chosen on. Symmetric selection or no comparison.
2. **CONSTRUCTION BIAS.** An actor-distilled schedule INHERITS the actor's discoveries => biased toward a tie
   => a FALSE "it's just continuity".
   **CLOSURE:** construction A (independent greedy) is the unbiased arm; B is the upper bracket. Report both.

## Pre-registered MDE (mandatory — an underpowered "tie" is uninterpretable)
The effect in dispute is **+0.006 full / +0.010 tail**. Seed-avg{1,2,3,7,11} run-to-run sd ~0.003-0.005.
=> we need **>= 5 seeds** and we should report the CI, not a p-value alone. **If the CI on
(adaptive - static) spans +/-0.006, the run is INCONCLUSIVE and must be reported as such — NOT as a tie.**
This is the trap that would let us "confirm" the theorem with noise.

## Traps
- Verify baselines snap to canonical (CASPER-R 0.360/0.152, entropy 0.361/0.140) BEFORE trusting any arm.
- GRADED answers only (on binary, actor and CASPER-R already tie — a binary run would manufacture a tie).
- Same fold/curriculum/recommender in every arm.

---

# WHY THESE TWO FIRST (and not the belief/refusal work)
E0 (with a PRIVILEGED answerability table) and REFUSAL_RESULT.md (refusal-forced training already run) and
Golbandi's own w_unknown=0.02 **jointly predict the refusal wiring is worth ~zero** (see WHY_ADAPTIVITY_DIED.md
v2). Building it before the E4-DUAL gate is the week we'd waste. By contrast SNAP-K and E2 each defend or
correct an EXISTING, ON-DEADLINE Paper C claim, and each is cheap. **SNAP-K can kill our centerpiece; better we
find that out than a reviewer.**

## ORDER
1. **SNAP-K** (defends/kills the centerpiece; MDE-resolvable; no policy training).
2. **E2** (adjudicates continuity vs adaptivity; needs the symmetric-selection closure).
3. **E4-DUAL** (Golbandi on our arena, two rulers, unknown-branch ablation — the discriminator between the
   answerability and H(Theta) theses). See WHY_ADAPTIVITY_DIED.md §4.
4. **H(Theta)-LITE** (mixture-posterior collapse rate — measures the theorem's bound on our own data).
