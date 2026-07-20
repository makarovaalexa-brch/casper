# DESIGN SHEET — FOLD-V4: de-OOD the belief encoder (author-approved 2026-07-10, "try that")
Status: APPROVED verbally; recorded for the no-silent-shortcuts rule. Supersedes ARENA_V2 as the
real next step (the fold must be honest before any policy comparison means anything).

## 0. The bug being fixed
fold-v3.1 trained interviews with on_profile=0.7 (privileged: assumes the interviewer already knows
the user's taste region) + budget 1-24. Its strong self-eval (interview T=24 = 0.384) is therefore
OPTIMISTIC. At deployment, blind selection lands OFF-profile -> the fold is OOD -> arena elicitation
barely lifts (+0.02) and the expected-gain order COLLAPSES over turns. The fold isn't broken; it was
trained/evaluated on an unrealistically easy interview distribution.

## 1. Principle (author-designed)
Never fake answers (v2.1 answerer is fixed + realistic); randomize SELECTION broadly so the fold is
robust to whatever policy deploys. Generate interviews by running an ENSEMBLE of REAL selection
policies through the REAL answerer (so answer-statistics stay valid by construction — no synthetic
parameter grids that produce impossible combinations).

## 2. Training distribution construction
GENERATOR = ensemble of real selection policies over TRAIN population users, real v2.1 answerer:
 - random, popularity, entropy, blind-EIG (info-gain), on-profile-relevant (as ONE flavor, not 70%),
   off-profile/adversarial (all-niche, high-refusal),
 - MIXED SCHEDULES: explore-early -> exploit-late, coarse->fine (the real adaptive-interview shape).
MIXTURE: bias toward the HARD TAILS (off-profile/high-refusal/redundant) beyond their realistic
 frequency — that's where the fold currently collapses and where robustness matters (author choice:
 FAT TAILS over match-realism). The fold's fidelity/surprise/know features let it down-weight junk,
 so diversity teaches discrimination, not corruption.
LENGTH (author decision 2026-07-10): PHASE 1 = SINGLE FIXED LENGTH B=8. Deployment target is a
 fixed-budget STATIC policy whose belief is consumed only at the final turn, so the fold only needs
 accuracy at length 8 — specialized, simpler, and the "collapse over turns" problem DISSOLVES (no
 intermediate turns). Train on de-OOD ensemble-selected 8-question interviews. Multi-length (nested
 prefixes 1..24 + no-decrease penalty) is DEFERRED to the ADAPTIVE phase — it's only needed for
 per-turn curves or a policy that reads the belief each turn.
CLEAN: retain ~30% clean full profiles (keeps real-profile skill; the "ok on real profiles" half).
 So two regimes trained: length-8 de-OOD interviews + full clean profiles.
INPUT DIVERSITY: varied refusal rates, fidelity classes (real/EASE/llm), channel mix.

## 3. Evaluation / gates (prove robustness, don't assume it)
 - HOLD OUT >=1 selection policy entirely from training; headline eval on the ACTUAL deployed
   baselines (never in the training mix).
 - G-clean: clean full-profile fold >= native.
 - G-nocollapse: per-turn NDCG@10 non-decreasing over interview turns (the currently-broken thing).
 - G-generalize: fold performs on the held-out (unseen) selection strategy within tolerance of seen ones.

## 4. The comparison (what "run baselines against retrained one" means)
Rerun random / popularity / entropy / blind-EIG per-turn @10 on the SAME 3k test split, on the
RETRAINED (v4) fold, and put them next to the v3.1-fold baselines. Question: does de-OOD-ing kill the
collapse (expected-gain no longer goes negative) and change the honest lift? All arms on the SAME v4
fold (policy-agnostic) = fair.

## 5. Co-training — DEFERRED, fenced (author asked "cheating?")
NOT cheating IF: train/test split clean, no answer-peek, AND the fair head-to-head uses the
policy-agnostic v4 fold. CHEATING version = co-tune fold to the winning policy then judge baselines
on it (unfair comparator). Legit use = a SEPARATE deployment result (full co-adapted system), each
arm with its own matched fold if used as a controlled comparison, and gated by the 173/human check
(co-adaptation raises the distilled-answerer quirk risk). Not in this phase.

## 6. Scope / cost
DEV/synthetic only, $0, NO LLM, 173 untouched. Retrain (~10 epochs CPU) + baseline rerun on v4 fold.
Non-idling execution.

## SIGN-OFF: author "ok, try that and then run the baselines against retrained one and compare" (2026-07-10).
