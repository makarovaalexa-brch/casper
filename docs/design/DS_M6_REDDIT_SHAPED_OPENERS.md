# DS-M6 — Reddit-shaped opener states, and whether a cluster's teacher survives them

**Status:** design sheet, written before any run (project rule). Author's proposal, 2026-08-19:
*"train it on a good selection of states within the clusters, based on reddit-shaped volunteered data
— a movie, a concept, a couple movies and a concept etc, not satiated state, and then apply
within-cluster tree teacher."*

## The problem this addresses

M4c/M4d/M4e established that the forest policy does not generalise to cluster-adjacent facts, under
**either** encoding, and that the cause is **coverage, not encoding**: each of the six clusters was
represented in training by exactly **one** state — its anchor concept folded at the root. Seven
points, not seven regions, so the partition between them is arbitrary and nothing was ever trained
there.

The proposal replaces each point with a **cloud** of realistic in-cluster opener states, so the
policy must learn *"these varied states all belong to cluster A"* rather than *"token 68 ⇒ tree 2"*.
It also aligns the training distribution with deployment, which is what the Act 5 Reddit census was
for.

## ⚠ THE ASSUMPTION THAT MUST BE CHECKED FIRST (this document's experiment)

Every teacher was fitted at **one** state: its anchor concept folded, cohort = that anchor's
endorsers. **Its optimality has only ever been measured there.** Labelling a state produced by "two
films and a concept" with tree-A's interview asserts that tree-A is still right at a state it was
never fitted for. If that is false, the training set is labelled with a teacher that does not suit
the states attached to it, and the policy would faithfully learn a wrong mapping — a failure that
would look like success at distillation time and only surface in NDCG.

**GATE M6a.** For each cluster A, at Reddit-shaped **non-anchor** opener states drawn from users'
own evidence: does tree-A still beat the cold tree, with the opener folded into **both** arms?

- **PASS** ⇒ the labelling is justified; build the training set (M6b).
- **FAIL** ⇒ the teacher is anchored to its own opener; the design needs per-shape teachers, which is
  a different and much larger experiment. Report and stop.

## M6a design

**Cohorts.** Validation users who endorse anchor A — that is how cluster membership is *known* for
this gate. (Identifying the cluster without being told is the policy's job later, not this gate's.)

**Opener shapes**, drawn from the M1b census of real requests and instantiated from each user's own
fold-in evidence, never from held-out targets:

| shape | census share it stands for | content |
|---|---|---|
| `film1` | 27.1% films only | one liked film from the user's fold-in half |
| `film2` | 27.1% films only | two liked films |
| `conc1` | 19.9% signature only | one endorsed concept, **not** the anchor and not any anchor |
| `film2conc1` | 35.9% films + signature | two liked films and one non-anchor concept |
| `anchor` | — | the anchor alone: the state the teacher was fitted at, as the reference row |

**Arms**, per shape, with the opener folded free into **both**:
- **cold** — the incumbent cold tree (depth 11), routed by the user's own answers;
- **teacher** — cluster A's tree, routed the same way.

**Budgets** K ∈ {1,2,4,8,15} counting asked questions; the opener is free. Paired percentile
bootstrap over per-user differences, 10,000 resamples, seed 98765. NaN tail pairs dropped, never
imputed. Canonical m=0 snap asserted. **VALIDATION ONLY.**

**Reading it.** The `anchor` row reproduces M4a and is the reference. The question is how much of
that margin survives when the opener is a film or a non-anchor concept instead. A teacher whose
advantage exists only at its own anchor state is not a cluster teacher — it is an anchor teacher, and
the M6b training set would be mislabelled.

**Declared before the run.** The teacher's margin is expected to *shrink* at non-anchor openers,
because a film carries more specific evidence than a concept and the tree's early questions may be
redundant with it. The gate is directional (does it still beat cold), not a demand that the margin be
preserved.

## M6b, only if M6a passes

Training set: for each train user in a cluster, sample an opener **shape** from the census
distribution, instantiate it from that user's own fold-in evidence, fold it, and label the trajectory
with that cluster's teacher. Cold users keep the cold tree. Then re-run M4c: adjacent facts should
now route to their cluster, because the space between anchors is populated.

**What M6b would claim, and what it would not.** It would claim that a single policy routes a
*realistic, previously unseen* volunteered opener to the right cluster's interview. It would not
claim the clusters themselves are discovered at ask time — cluster identity still comes from the
teacher assignment during training.
