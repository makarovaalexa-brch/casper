# M6a — the teachers are ANCHOR teachers, not CLUSTER teachers. Gate FAILED.

**Design sheet** `docs/design/DS_M6_REDDIT_SHAPED_OPENERS.md` · **code**
`src/instrument/m6a_opener_shapes.py` · **artifact** `experiments/battery/m6/m6a_opener_shapes.json`
· run 2026-08-19. **VALIDATION ONLY — the TEST look remains unspent.**

**The gate.** The author proposed training on many realistic in-cluster opener states — a film, two
films, a concept, two films and a concept — labelled with that cluster's tree teacher, to populate the
space between clusters and buy the generalisation M4c/M4e could not find. Every teacher, however, was
fitted at exactly **one** state: its anchor concept folded at the root. So before building anything:

> at Reddit-shaped **non-anchor** opener states inside cluster A, does tree-A still beat the cold tree?

The opener is folded into **both** arms and instantiated from each user's **own fold-in half**, never
from held-out targets.

## ★ THE RESULT: the advantage exists only at the anchor

Teacher − cold, full NDCG@10 at K=15, mean over the six clusters:

| opener shape | census share it stands for | mean teacher − cold | clusters positive | separating |
|---|---|---|---|---|
| **anchor** (the fitted state) | — | **+0.00897** | **6 / 6** | +2 / −0 |
| `film1` | 27.1% films only | −0.00202 | 2 / 6 | +0 / **−1** |
| `film2` | 27.1% films only | +0.00078 | 3 / 6 | +0 / −0 |
| `conc1` | 19.9% signature only | −0.00143 | 3 / 6 | +0 / **−1** |
| `film2conc1` | 35.9% films + signature | −0.00241 | 3 / 6 | +0 / −0 |

`arthouse` is the sharpest case: **+0.00520** at its own anchor, and **−0.01173 / −0.00612 /
−0.00489 / −0.00773** at the four realistic openers — every one negative, the first separating.

**The teacher's advantage is a response to the specific fact volunteered, not a property of the
cluster.** Move the same users to an opener a real requester would actually give, and it does not
shrink — it disappears.

## Three consequences

**1. M6b as designed would have been built on a false premise.** Labelling varied in-cluster opener
states with the cluster's tree attaches a teacher that is, at those states, no better than the
incumbent and sometimes worse. The policy would have learned it faithfully — **high fidelity, zero
value** — and it would have surfaced only in the final NDCG. This is exactly what the gate was
written to catch, which is why it ran before the training set rather than after.

**2. It retro-explains M4c / M4d / M4e completely.** Those runs asked why cluster-adjacent facts do
not route to a cluster's interview, and tried encoding (token vs state-only) and coverage as
explanations. **There was never a cluster-level regularity to find.** The cluster interview is not
the right answer at those states anyway, so no encoding or coverage change could have surfaced it.
The M4e conclusion "the blocker is coverage, not encoding" is superseded: **the blocker is that the
target does not exist.**

**3. ⚠ A real scope limit on the Act 5 reach claim, and it belongs in the chapter.** Real requesters
volunteer films and bundles — the M1b census puts 63.0% naming a film and 93.6% carrying ≥2 slots. At
exactly those openers, cluster-conditioned questions buy nothing measurable. Section
`sec:volunteered`'s claim stands **as stated** — a volunteered *attribute* changes which questions
are worth asking, measured where that attribute is what was volunteered — but it must not be extended
to "users in cluster X should get interview X".

## What is NOT affected

- **M4a and M4b stand.** Both measure the anchor state, which is the deployment case where the user
  volunteers exactly that fact. Nothing here touches those numbers.
- **The opener itself is still valuable.** Cold-arm levels *rise* when a film opener is folded
  (`arthouse` 0.30144 → 0.31229 with `film2`). Volunteered evidence pays through the **recommender** —
  the first channel Act 5 already documents — it just does not pay a second time through
  cluster-conditioned question selection.

## If this is to be pursued

The honest alternative is **per-shape teachers**: fit a tree at the state a film opener actually
produces, rather than reusing the anchor's tree there. That is a much larger experiment (a tree per
cluster × shape) and it should not start until someone wants that claim specifically. On this
evidence, the more promising direction is not more teachers of this kind at all.
