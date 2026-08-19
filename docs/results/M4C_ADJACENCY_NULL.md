# M4c — does the policy generalise to cluster-adjacent facts? **No.**

**Code** `src/instrument/m4c_adjacent.py` · **artifact** `experiments/battery/m4/m4c_adjacent.json` ·
run 2026-08-19. **VALIDATION ONLY — the TEST look remains unspent.**

**Author's question:** the policy has seen six opener-conditioned teachers, so has it learned
*"an anchored state gets a specialised interview"* as a function of the state? If so, a user
volunteering something **adjacent** to a cluster — a concept never taught — should land in a similar
state and be routed to that cluster's interview anyway.

## The design

Adjacent concepts = highest co-endorsement **lift with an anchor, computed on TRAIN users**, so the
cohort definition never touches the measurement split. Cohort = validation users endorsing the
neighbour and **no** anchor. Three arms, the fact folded free in all of them:

| arm | the volunteered fact is… |
|---|---|
| **anchor-slot** | folded **and announced** — history token + LIKE + masked from the vocabulary (the L1 encoding the six anchors get) |
| **fold-only** | folded, **not announced** — the anchor slot stays empty |
| **cold** | the incumbent cold tree, fact folded |

**The adjacency construct works.** The lift picked semantically obvious neighbours:
`arthouse` → *intimate, melancholic, enigmatic*; `dialogue` → *entirely dialogue, non-linear, ironic*;
`marvel` → *mutants, based on comic, comics*; `gore` → *splatter, blood, gory*;
`teenagers` → *teens, teen*. Lifts 4.5–8.6.

## ★ THE RESULT: no generalisation, and announcing an untaught fact is actively harmful

18 adjacent cohorts, K=15, full NDCG@10:

| | root agreement with its OWN anchor's teacher | root agreement with the cold tree | anchor-slot − cold | fold-only − cold |
|---|---|---|---|---|
| **adjacent** (18) | **0.000** | 0.000 | **−0.04368** | **+0.00006** |
| far (18) | — | 0.000 | −0.04332 | +0.00003 |

1. **Zero routing.** Not weak — **exactly 0.000** in all 18 cohorts. `mutants` and `comics` do not
   route to `marvel`; `splatter`/`blood`/`gory` do not route to `gore`. The policy memorised six
   anchor tokens; it did not learn a state→specialisation rule.
2. **Announcing an untaught fact costs −0.0437 on average, up to −0.0854** (`comics`). The opening
   question then matches **no tree at all** — not the teacher, not even the cold tree. This is
   off-manifold improvisation and reproduces the banked −0.047 for random concepts
   (`asif-asked-encoding-verified`).
3. **The evidence itself is handled perfectly.** Fold without announcing → **+0.00006** and **root
   agreement 1.000 with the cold tree**: the policy reproduces the incumbent interview exactly.

**The mechanism, precisely:** the anchor slot is a **six-valued categorical**. A seventh value is
out of distribution. The fold is fine; the *announcement* is what breaks.

## ⚠ Defect in the FAR control — disclosed, and it does not carry the finding

The far set was meant to be lowest-max-lift concepts matched on answer rate. The code sorted the
low-lift candidates **by answer-rate match**, which destroyed the low-lift ordering, so "far" is
really *answer-rate-matched arbitrary concepts* — with lifts 2.5–5.6, overlapping the adjacent range,
and one entry (`entirely dialogue`) present in **both** sets.

**Consequence:** adjacent-vs-far *magnitude* comparisons (−0.04368 vs −0.04332) are **not valid** and
must not be quoted as "adjacency makes no difference". The headline does not depend on the control:
0.000 root agreement with the target teacher is an **absolute** measurement, not a contrast. If
anything the contamination makes the null harder to escape.

## Two implications

1. **★ DEPLOYMENT RULE for M5: announce only the six taught anchors; fold everything else silently.**
   In a mixed population users volunteer facts outside the six, and announcing those costs up to
   −0.085 per user. Fold-don't-announce is free and provably safe (+0.00006, exact cold-tree
   behaviour). Without this rule a mixed-population test is actively damaged by its own realism.
2. **Generalisation needs more anchors in TRAINING, not a better encoding.** The slot must see many
   distinct concepts before it can learn descriptor→routing instead of six lookups. Same conclusion
   L1 reached: *more teachers, not more encoding*.

## Next

If adjacency generalisation is wanted, the experiment is to distil with a **broad** set of
volunteered facts occupying the anchor slot — dozens of concepts, most with no dedicated teacher, the
teacher-labelled six among them — and re-run this probe. That is a training-set change, not an
architecture change.
