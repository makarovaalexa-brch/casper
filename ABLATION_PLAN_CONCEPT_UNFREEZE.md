# ABLATION PLAN — what to unfreeze to give concepts a fair fight (2026-07-17)  [needs author sign-off before runs]

## The question
Concepts are currently handicapped vs items: FIXED whitened centroids (not learned), a 165k bolt-on encoder, a
decoder tuned for items, additive-only (no item<->concept fusion). Which unfreezing actually converts the
concept signal (oracle ceiling +188% tail) into realized uplift, and at what cost to the item tower?

## Frozen scaffold = the baseline B0 (the run just before A1)
item embeddings FROZEN | decoder Wd/bd FROZEN | item tower FROZEN | concept emb = FIXED whitened centroid |
only the fold-in encoder learns | base = popb floor | additive composition. Operator = fold-to-a-point
(Paper B). This is the "minimal safe" cell.

## The knobs (each unfreezes ONE thing from B0 -> attributable)
| id | unfreeze | hypothesis it tests | cost | risk |
|----|----------|---------------------|------|------|
| **B0** | nothing (fixed centroids) | frozen scaffold suffices if operator is right (Paper B was ~this) | done | none |
| **A1** | concept EMBEDDINGS (init whitened, anchored L2->init) | fixed centroids underpower concepts | low | pop-drift |
| **A2** | concept EMBEDDINGS, FREE (no anchor) | does the anchor matter? free-train re-adds popularity? | low | pop-drift |
| **A3** | + DECODER co-adapt (final proj / LoRA) + ITEM REPLAY | decoder tuned-for-items is the handicap | med | item-tower drift |
| **A4** | + BIGGER fold-in encoder (dh 128->512, +depth) | 165k is too small to fuse the concept set | low | overfit |
| **A5** | UNIFIED pool (item+concept tokens, ONE attention) + floor | additive blocks item<->concept interaction | med | lose floor (fusj failure) |
| **A6** | + ITEM embeddings co-adapt + replay | item factors could better host concepts | high | item-tower drift |
| **F**  | concept emb + decoder + item replay (author's agnostic unified space) | the FAIR ceiling: concepts first-class like items | high | item-tower drift |

## Order to run (value x cost)
1. **A1** (running) — the fair-representation test, cheapest.
2. **A3** — the biggest suspected handicap (decoder space); the decisive one.
3. **A5** — the fusion question (does item<->concept cross-talk help, with a floor guard).
4. **A2, A4** — secondary attributions (anchor, capacity).
5. **F** — only if A1/A3/A5 each help; the fully-fair ceiling.

## Metrics (EVERY cell, same ruler, disjoint val, all 150k users, 173 quarantined)
- **Concept-only k-curve** tail@10 at kc=1,2,4,8,16,32,all -> does it COMPOUND? bars: beat V4 head (+0.028), approach Paper B (+0.052).
- **No-harm**: full NDCG >= intercept - 0.003 (structural via zero-init gate; verify).
- **★ ITEM-TOWER GUARD** (critical for A3/A6/F): item-only FULL-PROFILE NDCG must stay >= 0.4859 - eps. Unfreezing must NOT weaken the item recommender. This is the whole reason we froze; any co-adapt run that drops it fails.
- **Warm compose**: concept on top of K=3,5 known items -> does concept ADD over items (the real interview)?
- **Diagnostics**: knockout (gate->0 == intercept); popularity-drift (concept-emb top-PC alignment with popb) for A1/A2; super-additivity on low-overlap loved pairs (A5).

## Decision rules
- A1 > B0 -> learned representation matters; carry it forward as the new base for A3/A5.
- A3 > A1 AND item guard held -> decoder handicap real -> co-adapt decoder.
- A5 > additive AND floor held -> fusion matters -> unified operator.
- Nothing beats Paper B compounding despite unfreezing -> the ceiling is the operator/data, not the freezing; stop unfreezing.
- Any cell that beats +0.028 with full preserved AND item-tower held = the shippable concept channel.

## Discipline (HARD RULES)
No data reduction (all users/items/concepts). Save per-cell checkpoints + peak on disjoint val. Design sheet
signed before A3/A5/A6/F (they touch the item tower / are heavier). A1/A2/A4 are frozen-tower probes (like B0).
