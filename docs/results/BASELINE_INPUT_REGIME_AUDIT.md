# Baseline input-regime audit — settled: the canonical numbers are the correct ones

**Date:** 2026-07-29 · **Trigger:** author question — "why did we rebuild them binary, if our goal was to
create a graded instrument? and probably made them artificially worse?" · **Adjudicated by Fable.**

## ⚠ THIS FILE PREVIOUSLY ASSERTED THE OPPOSITE. It was wrong. Read the verdict.

## Verdict
**The canonical on-ruler numbers stand: Golbandi-node 0.3065/0.1895, RBMF 0.2534/0.1764, TaNP 0.2645/0.1635.**
There is no "lower bound" caveat to make, because numbers obtained under a different information contract
are **not ordered** against these — not higher, not lower, not comparable. The ruler binarising is not a
handicap: every method on it receives identical information.

The paper says only this, in one sentence: three rows are ratings-native by publication and their
mechanisms are adapted to the implicit contract, which is what Liang et al. do for the ratings-native
baselines in the study that defines this split. Standard practice, one citation, done.

## What was attempted and why it is void
Built a graded all-bands fold-in (1.41M nnz vs the binary 626k, 55.7% sub-3.5) and a graded train matrix
(21.7M vs 10.85M), then re-ran Golbandi and RBMF on it. Two evaluation regimes, two opposite results, both
artifacts:

| | Golbandi likes-only | Golbandi all-bands binary | Golbandi all-bands graded | RBMF graded |
|---|---|---|---|---|
| **Regime 1** (harness default: fold-in is input AND mask) | 0.3065 | 0.3676 | **0.3894** | 0.2422 |
| **Regime 2** (input decoupled; mask = canonical `te_tr`) | 0.3065 | 0.2003 | 0.2641 | **0.1377** |

- **Regime 1 = masking leak.** Targets are always likes, so a sub-3.5 item can never be a target. Masking
  them deletes *guaranteed negatives*, and specifically the hardest ones — items the user chose to watch
  and that sit near their taste. Removing 2.26× more candidates inflates NDCG for **any** scorer.
  *The decisive control never run: binary model + graded mask. It would jump too.*
- **Regime 2 = pool pollution.** The graded arm's extra input items stay *in* the candidate pool, violating
  the universal convention that you mask everything the model was given. A mean-of-100-neighbours scorer
  reproduces its own input, so the user's own dislikes flood the top-10 as unmaskable distractors.

So Regime 1 over-masks one arm and Regime 2 under-masks the other. **Neither difference is attributable to
the input signal, and the graded question remains unmeasured.**

## Seven variables changed at once (not two)
train matrix · fold-in support · fold-in value semantics · candidate pool / input–mask contract ·
dislikes-encoded-as-likes in the binarised all-bands arm · RBMF regressing graded targets onto factors fit
by *implicit* iALS (internally inconsistent regardless of evaluation) · model–metric alignment (rating
predictors rank by $\hat r$; the ruler rewards "will be among held-out likes").

## Why "faithful on this ruler" was incoherent from the start
Faithful to Golbandi/RBMF/TaNP means their task (rating prediction), their metric (RMSE), their data.
Faithful to the ruler means their *mechanism* under the ruler's contract. Feeding graded values into a
binary top-N harness is **neither** — a hybrid no publication describes and no contract governs. A faithful
comparison requires picking one contract: (a) on-ruler, identical information for all — which the canonical
rows already are; (b) their setting as a separate study, never in the same table; or (c) if the question is
genuinely "does graded input help", give it to **every** method including EASE/RecVAE/the tower, fix the
mask to the union of each arm's input for all arms, and vary one factor at a time.

## ★ The process rule this cost
**Two missed sanity gates, in one evening.** A 2011 user-kNN vaulting past RecVAE, and a personalised model
landing on the Most-Popular floor (0.1377 vs 0.1345) — both should have triggered *debug the regime*, not
*update the science*. Both times the reflex was to change the harness and re-interpret, rather than hold the
model fixed and perturb the evaluation to test whether the number was about the model at all.

**Gate, from now on:** validate any new evaluation regime by running Most-Popular and one strong baseline
through it first. If a regime moves Most-Popular, drops a personalised model to the floor, or reorders the
frontier, fix the regime before believing anything it says.

**Surprise → single-factor falsification test → only then interpretation.**

## Must not enter the paper
The 0.3894; both result tables; any claim in either direction about graded input; the "lower bound"
framing; and the archaeology of this episode. Per the project rule, the paper presents the working version —
here that is the canonical table plus the one-line adaptation footnote.
