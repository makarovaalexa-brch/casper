# Design sheet — M3/M4/M5: grow the forest, distil it, and spend ONE test look

**Author go, 2026-08-17:** *"we need to grow actual forest on these clusters … can we use val and see
on test if those trees beat on ndcg on those clusters? and then fit policy on all trees and assemble
test set with different openers and without and compare on that?"* · *"make sure to measure them all
@ various k, not just 16."*

Governing plan: `PAPER_B_POLICY_DESIGN.md` Part M (+M.1–M.4). Inputs: `docs/results/M1_TASTE_CLUSTERS.md`
(the six anchors) and `docs/results/M2_SCREEN.md` (the screen passed: pooled +0.00384
[+0.00061,+0.00712], `marvel` +0.01540 [+0.00320,+0.02787]).

## Why validation, not train

Growing a tree requires **measuring NDCG gains**, and the tower was trained on the 140,768 train
users — any NDCG measured there is memorisation (D.6b). So trees are grown on the **10,000 validation
users**, exactly as the cold tree was. Train's only role was *defining* the clusters, which needs no
NDCG. This also makes the comparison symmetric: **both arms are validation-fitted and test-measured**,
so the forest enjoys no advantage the incumbent did not have.

## M3 — six trees, one per anchor

- **Cohort = every validation user who STATES the anchor** (endorses it under the env's SEL rule) —
  the deployable routing signature, not secret cluster membership. Sizes: `inspiring` 1,822 ·
  `teenagers` 786 · `gore` 747 · `marvel` 734 · `dialogue driven` 730 · `art house` 587.
- **The anchor is folded as volunteered evidence and costs no turn**, in every arm. It is masked from
  re-asking.
- Everything else is the cold tree's own recipe, unchanged: same 1,000-candidate combined pool, same
  ternary branch rule, same min-leaf floor, same objective (measured NDCG@10 gain), same certified
  pair, policy-B mask.
- **Persist per node** (HARD RULE 11): partial JSON after every pick, contract-signed heap sidecar,
  resumable.
- Cost: eval time scales with cohort size; the cold tree took 10.33 h on 10,000 users, so ~45 min per
  tree at these sizes. **Budget 5–8 h.** Bench the first tree before trusting that.

## M4 — distil the forest into one policy

L2a machinery, warm-started from E1. Gates: every teacher's fidelity ≥ 0.999 at its own states; the
**cold tree's fidelity unchanged** (zero interference re-proven at N=7, having been shown at N=2);
shuffle control (permuted teacher assignment must destroy fidelity).

## M5 — ONE test look, pre-registered here, taken once

**Arms, all at K ∈ {1, 2, 4, 8, 16}** (author's instruction — not K=16 alone; the screen measured a
one-question horizon and the whole open question is whether an early advantage *persists* or the cold
tree catches up):

1. **Per-anchor:** cluster tree vs cold tree, on the test users who state that anchor, anchor folded
   in both arms. Reports whether each teacher beats the incumbent at its own states.
2. **The assembled mix:** every test user routed to their teacher if they state an anchor, cold tree
   otherwise, versus the cold tree everywhere. This is the deployable comparison.
3. **With and without openers:** the same mix run with the anchor withheld, isolating what the opener
   contributes over and above the fold.
4. **Controls:** shuffled routing (users sent to a random teacher) must lose; canonical m=0 snap on
   test asserted before any number is read.

**Paired percentile bootstrap, 10,000 resamples, seed 98765, on per-user difference vectors. Full AND
tail NDCG@10 everywhere.**

## Pre-registered read-out

- **Headline** = the mix at K=16, forest-vs-cold-tree, full NDCG@10.
- A per-anchor cell is descriptive; test cohorts are ~600–1,800 so their intervals are wide
  (roughly ±0.013 at n=700), and no per-anchor null should be read as a teacher failing.
- **The K-curve is a first-class result, not a robustness check.** If the advantage is large at K=1–2
  and gone by K=16, the honest claim is that an opener *accelerates* the interview rather than
  improving its endpoint — a different and weaker claim, and it must be written as such.

## Failure modes, declared

- Forest ≤ cold tree at K=16 on the mix ⇒ the teachers do not survive a full interview. Report the
  K-curve, state that the screen's advantage closes, and **do not** market the K=1 gain as the result.
- Fidelity gates fail at N=7 ⇒ one policy cannot hold seven teachers; report the interference and fall
  back to reporting the trees themselves.
- Shuffled routing wins ⇒ the harness is broken, not the finding.

## Standing constraints

Certified pair `t2i26c_ep31` + `concept_cv3_ep08`; policy-B mask/pool; canonical snap
`0.15686538118354684`; commit before launch; one heavy job at a time; **the test cohort is not built,
loaded or named until M5**, and M5 runs once.
