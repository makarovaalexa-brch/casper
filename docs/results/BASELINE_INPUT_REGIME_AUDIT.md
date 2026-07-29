# Baseline input-regime audit — is every cited baseline fed the input its published form uses?

**Date:** 2026-07-29 · **Trigger:** author question — "why did we rebuild them binary, if our goal was to
create a graded instrument? and probably made them artificially worse?"

## The standard applied
Not "nothing may be binarised". The canonical Liang ruler binarises at $r>3.5$, and for the
implicit-feedback family **binary is the published input** — EASE, RecVAE and Mult-VAE were published on
binarised data, and feeding them ratings would depart from the published systems rather than repair them.
The defect is narrower and one-directional: **a ratings-native system fed binary input has had a channel
removed that its design uses.** That is the same information loss R5 names and V5 measures at $-0.2985$
when imposed on our own tower.

## Correct as implemented — binary is native
| row | published input | our input | verdict |
|---|---|---|---|
| Most-Popular | counts | counts | ✅ |
| Item-kNN (Cremonesi) | implicit binary | binary | ✅ |
| PureSVD | implicit binary | binary | ✅ |
| iALS / WMF (Hu–Koren–Volinsky) | implicit + confidence from counts | `c_ui = 1 + alpha` | ✅ HKV lets the count in as confidence, never as sign — the assumption our signed estimator breaks |
| EASE, EDLAE (Steck) | binary implicit | binary | ✅ |
| Mult-VAE, Mult-DAE (Liang) | binary implicit multinomial | binary | ✅ |
| RecVAE | binary implicit | binary | ✅ |
| Turbo-CF | binary implicit | binary | ✅ |
| SASRec / BERT4Rec | implicit ID sequences | ID sequences | ✅ |

## ❌ Ratings-native, fed binary — three cases
| row | published form | what our implementation does | consequence |
|---|---|---|---|
| **Golbandi tree** | RMSE decision tree; routes users by **lovers / haters / unknowns** | `golbandi_node.py`: *"raw binary rows (the node-mean averages these)"* | the three-way split collapses to two — a hater and a non-viewer become the same zero, which is the tree's core mechanism |
| **RBMF / Functional-MF** | rating regression, elicitation RMSE | ridge LS on the **binary** support | fits ones instead of the graded targets it was designed for |
| **TaNP** | *"episodic meta-learning for rating prediction"*, support = `(item, rating)` pairs | `tanp_bestefffort.py`: *"rating := 1 for every observed like … the rating channel is therefore constant; the signal is the item set"* | the rating channel is dead; a scalar is still concatenated "for architectural fidelity" but carries no information |

**All three were documented in their own source files and none of it reached the chapter.** The paper
promised "best-effort replication of that core **with every substitution stated**" and did not state this one.

## ⚠ One further regime mismatch, different in kind
**ICF** (Zhao 2013) is interactive collaborative filtering over **Bayesian probabilistic MF on ratings**.
The chapter represents it by asserting its full-profile ceiling is the measured **iALS** row — but iALS is
implicit-with-confidence, a different input regime. The substitution is stated as a model-family
equivalence and should also be stated as a regime change.

## Status
- Disclosed in the chapter for Golbandi and RBMF (commit `eb63c99`); **TaNP and ICF still to add.**
- TaNP's number is not yet in the paper (the run was still converging at val 0.2591), so its row can be
  marked correctly on arrival rather than corrected after.
- **The fix is cheap and structural:** the binarisation lives in `metrics.load_train`, not in the models.
  RBMF's ridge LS and Golbandi's node means are value-agnostic — they consume whatever the matrix carries.
  A graded loader over the identical user and item sets makes both ratings-native with *no model change*.
  TaNP needs its rating channel un-pinned, which is a one-line change to a constant.

## Why this matters beyond fairness
The chapter criticises the elicitation literature for reporting on shortlist metrics that flatter their
systems (\S on what the elicitation literature reports). Publishing binarised reconstructions of
ratings-native rivals and then reading the gap as evidence about their designs is the same error pointed
the other way. The numbers stay as measured — they are on the one ruler and honestly obtained — but they
are **lower bounds on the designs**, and the chapter must say so for all three rows, not two.
