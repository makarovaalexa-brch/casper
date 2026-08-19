# M4b — one policy over the seven-tree forest

**Code** `src/instrument/m4_distill_gen.py` (trajectories) · `m4b_policy.py` (distillation) ·
`m4b_eval.py` (arm C) · `m4b_val_fidelity.py`, `m4b_fidelity_by_pos.py` (diagnostics) ·
`tree_route.py` (routing). **Artifacts** `experiments/battery/m4/{m4b_policy.pt, m4b_armc.json,
m4b_val_fidelity.json}`. Run 2026-08-19.
**VALIDATION ONLY — the single TEST look remains unspent.**

## The question

M4a established that at an opener state a dedicated teacher beats the incumbent cold tree. Seven
trees is not a deployable artifact — it is seven lookup structures behind a hand-written router that
must be told which anchor a user volunteered. **Can one parametric policy, consuming only the state,
hold all seven?**

## ★ THE RESULT: yes, with no measurable loss

Arm C is the distilled policy picking its **own** questions from the full 1,000-candidate pool. The
anchor is folded into all three arms and is free everywhere. K=15, validation.

| anchor | n | cold | teacher | **policy** | P−C | P−T | whole interview identical |
|---|---|---|---|---|---|---|---|
| dialogue | 730 | .26437 | .27927 | **.27940** | **+0.01502** | +0.00013 | 99.04% |
| teenagers | 786 | .28633 | .29923 | **.29931** | **+0.01297** | +0.00008 | 99.11% |
| marvel | 734 | .42452 | .43451 | .43441 | +0.00989 | −0.00010 | 99.86% |
| arthouse | 587 | .30144 | .30664 | .30709 | +0.00565 | +0.00045 | 96.93% |
| gore | 747 | .36198 | .36767 | .36763 | +0.00564 | −0.00004 | 99.73% |
| inspiring | 1,822 | .26911 | .27425 | .27403 | +0.00492 | −0.00021 | 99.23% |

**|policy − teacher| ≤ 0.00045 everywhere**, three positive and three negative — indistinguishable
from zero in both directions. The policy's edge over the cold tree reproduces its teacher's almost
exactly (`dialogue` +0.01502 against M4a's +0.01490; `teenagers` +0.01297 against +0.01289).

**Root agreement with the teacher is 1.0000 in all six cohorts**, and teacher-forced agreement on
validation states is ≥0.9977 pooled for all seven trees, ≈1.000 at nearly every position.

## The interference falsifier did not fire

Declared before the run: if learning the six teachers costs the cold tree, the single-policy claim
fails at seven teachers and is reported dead. Held-out TRAIN dev, selected checkpoint (epoch 2):

| cold | arthouse | dialogue | marvel | gore | teenagers | inspiring | mean |
|---|---|---|---|---|---|---|---|
| 0.9991 | 0.9988 | 0.9995 | 1.0000 | 1.0000 | 0.9991 | 0.9987 | **0.9993** |

The incumbent is held as tightly as the teachers. L2a had shown one policy could hold the tree plus
four **static** genre orders; this is seven **branching** teachers, which was the real test.

⚠ Training is non-monotone — epoch 1 scored 0.9699, epoch 2 0.9993, epoch 3 0.9926. Best-checkpoint
selection on a held-out train slice is doing real work; early-stopping on the epoch-3 dip would have
shipped a worse model, as would stopping at epoch 1. (In the pre-fix run the ordering was different
again: 0.9974 / 0.9932 / 0.9997.)

## What this establishes, and what it does not

**Does:** the forest collapses into ONE 526,542-parameter object that consumes only the state, with
no hand-written router and no seven artifacts, and loses nothing on held-out users. That is what makes
the forest a *method* rather than an existence proof.

**Does NOT:** beat its teachers. E1 established that a distilled policy reproduces its teacher at best
and never exceeds it, and that is exactly what happened — the claim is *equivalence at one seventh of
the artifacts*, never superiority. Nor does it establish generalisation of the underlying advantage:
M4a's teacher−cold margins are an upper bound because each teacher was fitted on precisely the cohort
it is scored on (see `M4A_TEACHER_VS_COLDTREE.md`). **Arm C is the only arm whose parameters never saw
these users**, which is the one asymmetry running in its favour.

## Design decisions that turned out to be load-bearing

1. **The anchor enters as an ANSWERED QUESTION** — history token + LIKE outcome + masked out of the
   candidate vocabulary — not merely folded into the latent. L1 measured the fold-only encoding at
   0.00% correct branching (memory `asif-asked-encoding-verified`).
2. **Four of the six anchors are NOT in the 1,000-candidate pool** (`arthouse`, `marvel`, `gore`,
   `teenagers`; only `dialogue` and `inspiring` are). The extended phi table that lets an anchor be a
   history token without being a candidate is therefore essential, not defensive: without it the
   policy would have seen an anchored state as an ordinary cold one for exactly the teachers that
   matter most.
3. **Selection on a held-out TRAIN slice, not on validation** — stricter than the design sheet, which
   says "select on val". Validation is where arm C is scored against the trees; selecting there would
   contaminate the comparison.
4. **Fallback positions masked out of the loss** (author, 2026-08-19). Where a user routes outside a
   tree's fitted support the label is `tree_route`'s continuation rule, not the teacher's decision.
   351 of 3.24M positions (0.011%). Evaluation never masks.

## ⚠ The first build was wrong, and how the policy exposed it

The first pipeline reported policy − teacher of −0.004 to −0.012 with every CI excluding zero, and
whole-interview identity of only 55–82%. **All of it was a routing defect**, not distillation loss:
the router rebuilt each interview by accumulating splitters during the descent instead of reading the
tree's stored `path + steps`, which supplied a below-floor child's splitter twice. See
`M4A_TEACHER_VS_COLDTREE.md` and memory `routing-rebuilt-what-the-artifact-already-stored`.

**The policy was the instrument that caught it.** A policy cannot ask the same question twice, so
wherever the router made a teacher repeat, the policy was forced to differ —
`frac_never_deviates` matched **1 − repeat-rate to within 0.002 in all six cohorts**. Two earlier
explanations (majority-branch collapse, rollout compounding) were proposed and killed by measurement
before the correspondence check identified the real cause.

## Next

M5, the single TEST look, pre-registered in `DS_M3_FOREST_TREES.md` and still unspent. Arm C is now a
legitimate third arm for it: one policy, seven teachers, no router.
