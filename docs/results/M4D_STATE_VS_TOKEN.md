# M4d — the policy routes on the announced token, not on the state

**Code** `src/instrument/m4d_state_vs_token.py` · **artifact**
`experiments/battery/m4/m4d_state_vs_token.json` · run 2026-08-19.
**VALIDATION ONLY — the TEST look remains unspent.**

**Author's question:** *"what happens if a user comes with a state that resembles one of the clusters
but slightly different? it will not figure out a question that is relevant for that cluster?"*

M4c answered half of it (a cluster-adjacent concept, folded but not announced, gets the identical
cold interview) but left the premise unverified — maybe the adjacent fold never moved the state near
the cluster, in which case nothing was tested.

## The decisive control

Take a **taught** anchor's **own** cohort — the exact users its teacher was fitted on, a state as
close to the cluster as a state can be — and fold the anchor **without announcing it** in the history
slot.

| arm | K=15 root agreement with the teacher | with the cold tree |
|---|---|---|
| **announced** (history token + LIKE + masked) | **1.000** (whole interview 0.969–0.999) | — |
| **fold-only** (same evidence, silent) | **0.000** | **1.000** (whole 0.978–0.996) |

Per anchor, fold-only against the two references:

| anchor | n | teacher | cold | announced | fold-only | fold-only − cold | fold-only − teacher |
|---|---|---|---|---|---|---|---|
| arthouse | 587 | .30664 | .30144 | .30709 | .30169 | +0.00025 | −0.00495 |
| dialogue | 730 | .27927 | .26437 | .27940 | .26438 | +0.00001 | −0.01489 |
| marvel | 734 | .43451 | .42452 | .43441 | .42458 | +0.00005 | −0.00993 |
| gore | 747 | .36767 | .36198 | .36763 | .36179 | −0.00020 | −0.00589 |
| teenagers | 786 | .29923 | .28633 | .29931 | .28626 | −0.00007 | −0.01296 |
| inspiring | 1,822 | .27425 | .26911 | .27403 | .26922 | +0.00011 | −0.00503 |

**Withhold the announcement and the policy gives the COLD interview to `marvel` fans, on the marvel
cohort, with the marvel concept folded — forfeiting the entire teacher advantage.**

## ★ The state moves a lot, and the policy does not read it

| anchor | ‖Δ‖ of the mean latent when the anchor is folded | cosine to the cold state |
|---|---|---|
| arthouse | 2.7391 | 0.8313 |
| dialogue | 2.8644 | 0.8533 |
| marvel | 2.7558 | 0.8467 |
| gore | 2.6196 | 0.8472 |
| teenagers | 1.9206 | 0.9096 |
| inspiring | 2.8367 | 0.8774 |

This is not a subtle displacement — a cosine of 0.83–0.91 away from the cold state. **The routing
failure is not that near-cluster states are too similar to tell apart. Question selection never
consults the state at all.**

⚠ The run also prints a "×the cohort spread" ratio of ~2.7e9. That statistic is **meaningless and
must not be quoted**: before any evidence is folded, every cold user has an *identical* latent, so
the denominator is ~0. It is itself worth stating — "a state that resembles a cluster" only becomes a
meaningful notion once evidence is folded, and the anchor fold is that evidence. ‖Δ‖ and the cosine
are unaffected.

## The mechanism, and why it was predictable in hindsight

During distillation the announced token was a **perfectly reliable, low-dimensional cue** for which
teacher to imitate: six values, each mapping to exactly one tree. Given a redundant categorical that
predicts the label exactly, the network took it and never learned the harder continuous mapping from
the latent. Same shape as the banked R2 result — *sensitive to the evidence, not informed by it*
(`capabilities-programme-structural-vs-behavioural`).

Note what this does **not** say: the fold is not useless. It pays substantially through the
**recommender** (the ranking improves), which is the first channel Act 5 measures. What it does not
buy is a **different question**.

## Consequences

1. **Answer to the author: no.** A state resembling a cluster gets no cluster-relevant question. Not
   because the resemblance is too weak, but because routing is a six-way lookup on an announced
   token.
2. **★ The fix is a training change, and it is cheap: ANCHOR-TOKEN DROPOUT.** Randomly withhold the
   announcement on a fraction of seeded distillation examples so the token stops being a reliable
   cue and the policy must recover the teacher from the latent. Combine with broadening the slot to
   dozens of concepts (M4c's conclusion) and this run becomes the measurement of whether it worked:
   fold-only root agreement with the teacher should rise from 0.000.
3. **Binding for M5 regardless:** announce only the six taught anchors, fold everything else
   silently (`M4C_ADJACENCY_NULL.md`). Announcing an untaught fact costs up to −0.085; folding it is
   free.
