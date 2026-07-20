# Paper E interpretability: phrase bank + SNAP-LOSS (2026-06-30)

## Phrase bank (scripts/paper5/build_phrasebank.py -> .cache/phrasebank.npz)
2542 phrases in R^64 (centroid of matched movies): 1096 genome tags + 18 genres + 1161 actors + 258 directors + 9 eras.
Catalogue-direction coverage: nearest-phrase cos mean 0.60 (p10 0.43).

## SNAP-LOSS (SNAPBANK block in continuous_actor.py): naming the Paper-C continuous query
Roll D1 continuous policy; snap each off-manifold query to nearest phrase; user answers the NAMED question. seed-avg{1,2,3}.
| | FULL | TAIL |
|---|---|---|
| un-snapped (continuous D1) | 0.3769 | 0.1745 |
| SNAPPED (named phrases) | 0.3324 | 0.1363 |
| **SNAP-LOSS (cost of naming)** | **+0.0445** | **+0.0382** |
snap cos to nearest phrase: mean 0.675, p10 0.531. Opener snaps to one phrase ("pg-13") for all users (cold-start query
identical). Top phrases: pg-13, entertaining, oscar-visual-effects, cult classic, 1930s, period piece, highly quotable,
classic, exciting, original plot.

## RICH BANK (6724 phrases) + SNAP-TRAINED policy + the DISCRETE>SNAPPED insight (2026-07-01)
Rich bank (build_phrasebank_rich.py): 1096 tags + 2600 tag-pairs + 2101 people(>=3 films) + 900 tag-labeled KMeans
cluster centroids + genres/eras. Coverage cos 0.60->0.72 (p10 0.43->0.49). Backup: phrasebank_v1_2542.npz.
| policy | FULL | TAIL |
|---|---|---|
| continuous D1 (off-manifold) | 0.377 | 0.175 |
| discrete entropy-over-concepts (Paper B) | 0.361 | 0.140 |
| discrete PEBOL-geometric [LEAKY-see below] | 0.369 | 0.168 |
| discrete ConTS [LEAKY-see below] | 0.351 | 0.151 |
| **PEBOL-geometric FAIR (no known-item peek)** | **0.327** | **0.110** |
| **ConTS FAIR (no known-item peek)** | **0.306** | **0.112** |
NOTE (Jul 1 LEAK FIX): PEBOL/ConTS as first run grounded their Beta/arms on the user's KNOWN-HALF items (cand=list(half))
= a SELECTION-side PEEK at the profile they must elicit. FAIR re-run (LITBASE FAIRCAND=1: shared popularity-stratified
800-item pool, no per-user peek): PEBOL 0.369->0.327/0.110, ConTS 0.351->0.306/0.112 — both now clearly BELOW CASPER-R
0.360/0.152 (esp tail). Leak was inflating TAIL most. paper4 tab:baselines CORRECTED; paper3 unaffected (ConTS/PEBOL are
related-work mentions there, not a baseline table; its 0.351 = FieldActor, different quantity).
| naive-snap D1 -> rich bank | 0.334 | 0.140 |
| snap-TRAINED (warm-D1, REINFORCE) | 0.334 | 0.139 |
KEY INSIGHT (user-spotted): **DISCRETE (0.361-0.369) > SNAPPED (0.334)** and it MAKES SENSE: the continuous policy is
TRAINED to emit OFF-MANIFOLD queries (value lives BETWEEN named concepts) => snapping lands on a BAD nearest phrase. A
discrete policy optimises DIRECTLY over named actions => picks genuinely informative concepts. Snapping an off-manifold
policy != optimising a discrete policy. Rich bank does NOT reduce naive snap-loss (+0.044->+0.043) — off-manifold is
off-manifold. Snap-TRAINED warm-D1 tied naive (REINFORCE barely moved the D1 off-manifold bias).
=> Right approach = a NATIVE discrete policy over the rich bank (train to pick phrases on NDCG), NOT continuous-then-snap;
or distill from a good discrete policy. Running: SNAPTRAIN from-scratch (SNAPINIT=random) STAU 0.15/0.30, SNEP=140.
Target: beat naive-snap 0.334 and >= discrete 0.361; ideally the RICHER vocab (pairs/clusters) beats concept-only 0.361.

## VERDICT (honest, the Paper-E centerpiece)
The closed continuous probe is accurate but its value is OFF-MANIFOLD (Paper C thesis) => even a rich 2542-phrase bank
loses +0.044 full / +0.038 tail when naming it. So the closed probe is NOT cheaply verbalisable.
CONTRAST: OPEN recall is NATIVELY NAMED => SNAP-LOSS = 0. Every open question is already a phrase. This is the
deployability advantage of recall over latent probing, made quantitative.
=> Deployed agent: OPEN = default (zero naming cost); CLOSED = "deepen, approximate when verbalised" (snap-loss documented);
MIXED = best-of-both. The UI selector (open/closed/mixed/discrete-closed) shows each mode's honest cost.
Enriching the bank (tag-pairs, mined phrases) would shrink but not eliminate the snap-loss (off-manifold by design).
Repro: SNAPBANK=1 EVALSEEDS=1,2,3 python scripts/paper2/continuous_actor.py
