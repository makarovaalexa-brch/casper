# M4e — state-only distillation: the token is unnecessary, and it was never the blocker

**Code** `src/instrument/m4b_policy.py --anchor_encoding {token,none,dropout}` · **artifacts**
`experiments/battery/m4/{m4b_stateonly_policy.pt, m4b_stateonly_policy.json}`, arm C in
`m4b_armc.json` (token run preserved as `m4b_armc_TOKEN.json`, checkpoint as `m4b_policy_TOKEN.pt`).
Run 2026-08-19. **VALIDATION ONLY — the TEST look remains unspent.**

**Author's question:** *"what if we distill based on only the state?"* — i.e. withhold the announced
anchor token so routing must come from the folded latent.

## Why it was worth running (two cheap probes first)

1. **The information is present.** A held-out multinomial linear probe on `z` at position 0 recovers
   which of {cold, 6 anchors} was folded with accuracy **1.0000**, perfect recall on all seven.
   Centroids sit 1.9–4.2 apart against a within-cohort spread of 0.04–0.06.
2. **The geometry favours generalisation.** **14 of 14** cluster-adjacent folds land nearest their
   intended anchor's centroid, not cold — `based on comic` 0.572 from `marvel` vs 2.620 from cold;
   `splatter` 0.744 from `gore` vs 2.281.

## ★ Result 1: the token is unnecessary. State-only reproduces the teachers exactly.

Held-out train dev, mean top-1: **0.9994** state-only vs **0.9993** token. Arm C on validation,
K=15, |policy − teacher|:

| anchor | state-only | token |
|---|---|---|
| dialogue | **+0.00000** | +0.00013 |
| marvel | **+0.00000** | −0.00010 |
| teenagers | +0.00004 | +0.00008 |
| inspiring | −0.00005 | −0.00021 |
| arthouse | −0.00017 | +0.00045 |
| gore | −0.00042 | −0.00004 |

Root agreement with the teacher **1.0000** in all six cohorts, under both encodings.

**⚠ This puts a SCOPE LIMIT on a banked rule.** L1 concluded *"a volunteered fact must enter as an
answered question (history token + fold + mask)"*, citing 0.00% branching for the fold-only encoding.
That was measured on a **frozen** E1 policy which had never seen a volunteered fact in training —
the L1 note itself says *"a wiring property, not a property of distillation."* **Train for it and the
fold alone suffices.** The rule governs inference-time re-encoding of a policy not trained for it; it
is not a law about distillation. M4b's design carried it forward unexamined.

## ★ Result 2: it does NOT generalise — and the token was never the blocker

Cluster-adjacent facts (18 cohorts, `marvel`→*mutants/comics*, `gore`→*splatter/blood/gory*, …):

| policy | root agreement with intended teacher | root agreement with cold | K=15 vs cold |
|---|---|---|---|
| token | 0.000 | **1.000** | **+0.00006** |
| state-only | **0.000** | 0.041 | **−0.03526** |

Adjacent facts route to their cluster **0.000** of the time under *either* encoding. Removing the
token did not buy generalisation.

**And state-only is strictly worse off-manifold.** The token model, meeting a fact it does not
recognise, has a safe default — no token, therefore the cold interview, exactly (root agreement
1.000, +0.00006). The state-only model has no default: it must route from `z`, emits something that
is neither cold nor any teacher, and **loses −0.035** (far cohorts −0.025). This is L1's
*"inertness is worth +0.047"* reappearing. **The token's real function was never routing information
— it was providing a fallback.**

## Why the favourable geometry did not deliver

The centroid probe says adjacent folds are nearest their intended anchor. The policy is not a
nearest-centroid rule — it is a **discriminative partition** fitted over seven populated regions, and
training never placed a single example *between* clusters, so the boundaries out there are arbitrary.
Proximity to `marvel`'s centroid does not put a state inside the policy's `marvel` region.

**So the blocker is COVERAGE, not ENCODING** — the same conclusion M4c reached, now with the encoding
explanation eliminated by experiment rather than assumed.

## The trade-off, measured

| | on the six taught anchors | on everything else |
|---|---|---|
| **token** | exact | **safe** — reverts to the cold interview, +0.00006 |
| **state-only** | exact | **harmful** — improvises, −0.035 |

For deployment the token encoding is preferable *because* it fails safe, not because it routes
better. The M5 rule stands: announce only the six taught anchors, fold everything else.

## Next, if adjacency generalisation is still wanted

Distil with **dozens** of anchors, most without a dedicated teacher, so the space between the six is
populated and the partition has to interpolate rather than extrapolate. `--anchor_encoding dropout`
is implemented and untried; on this evidence it is a robustness lever, not a generalisation one.
