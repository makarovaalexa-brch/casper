# ENTITY-NATIVE TRAINING — the bottom rung of the action-set richness ladder

**Date:** 2026-07-02. **Question:** PAIRTRAIN (PAIRTRAIN_RESULT.md) showed that putting the realizer INSIDE the
differentiable unroll rescues a pair-native policy (6.9M pair directions) to 0.3701/0.1659. What does the SAME recipe
achieve on the POOREST realizable set — the unified 1,361 NAMED ENTITIES (600 pool items + 761 concepts, exactly the
vocabulary CASPER-R scores over)? This completes the action-set richness ladder for Paper C: identical training
protocol at every rung, only the realization set varies.
**Verdict: entity-native training lands at 0.3559 ± 0.0022 / 0.1515 ± 0.0035 (seed-avg {1,2,3,7,11}, te[300:], q8,
graded, entity-realized fold). That is (i) a large rescue over post-hoc entity-snap of the same D1 policy
(+0.019 FULL / +0.017 TAIL), (ii) a statistical TIE with discrete CASPER-R (0.3600/0.1520) — a continuous policy
straight-through-trained onto the 1,361-point vocabulary recovers the discrete SOTA but no more, (iii) clearly below
the pair rung (−0.014/−0.014) and the free D1 (−0.022/−0.027). The ladder is MONOTONE in realization-set richness:
what you can ask bounds what you can learn to elicit.**

## THE LADDER (all rows: same D1 recipe — warm-start from the D1 winner, straight-through realization in the training
## unroll where applicable, DIVW=1.0 DTAU=2.0 field reward, OBJ=ustar, graded geometric answers, TRSEED=0, val-best
## SELVAL=tail on te[:300]; eval = seed-avg {1,2,3,7,11}, te[300:], q8, graded, realized fold)
| realization set (size) | FULL | TAIL |
|---|---|---|
| unconstrained / off-manifold (∞; NOT deployable as a named question) | 0.3780 ± 0.0032 | 0.1782 ± 0.0065 |
| item-PAIR differences (~6.9M directions; pair-native, ts0 best ep4) | 0.3701 ± 0.0047 | 0.1659 ± 0.0049 |
| **NAMED ENTITIES (1,361 points; entity-native, ts0 best ep4)** | **0.3559 ± 0.0022** | **0.1515 ± 0.0035** |
| — reference: uent+GRAW (static entropy, graded, canonical) | 0.3667 ± 0.0045 | 0.1577 ± 0.0078 |
| — reference: CASPER-R (discrete learned policy, canonical) | 0.3600 ± 0.0052 | 0.1520 ± 0.0071 |
| — reference: D1 post-hoc ENTITY-snap, unified 1361 (no in-loop training; this run) | 0.3372 ± 0.0031 | 0.1346 ± 0.0037 |
| — reference: D1 post-hoc concept-snap (SNAPLOSS/ACTSNAP, answerable concepts) | 0.3414 | 0.1384 |
- Per-eval-seed (entity-native best): s1 0.3568/0.1564, s2 0.3544/0.1514, s3 0.3533/0.1475, s7 0.3553/0.1479,
  s11 0.3596/0.1542.
- vs post-hoc entity-snap of the SAME warm-start (0.3372/0.1346): **+0.019 FULL / +0.017 TAIL** — in-loop
  straight-through training again rescues most of the naming loss (same mechanism as the pair rung).
- vs CASPER-R (0.3600/0.1520): −0.004 FULL / −0.001 TAIL ≈ TIE. Sensible ceiling: on a 1,361-point action set a
  policy that must SNAP cannot beat a policy that optimises DIRECTLY over the same named actions (cf. the
  DISCRETE>SNAPPED insight, SNAPBANK_RESULT.md) — but straight-through snap-training now REACHES that discrete
  optimum from the continuous side, where post-hoc snapping fell 0.023 below it.
- vs the pair rung: −0.0142 FULL / −0.0144 TAIL; vs free D1: −0.0221 FULL / −0.0267 TAIL. Monotone ladder:
  ∞ > 6.9M pairs > 1,361 entities, at both full and tail, with the SAME optimizer/recipe at every rung — the
  action-set richness itself is the load-bearing variable.

## Mechanism (ENTTRAIN=1 in scripts/paper2/continuous_actor.py)
Identical to PAIRTRAIN except the realizer: each turn inside `rollout()`,

    emit q  →  qn = q/||q||
    k = argmax_e cos(qn, POOL_e)          (unified NP=1361 pool, cosine over unit entities POOLnt)
    qn ← qn + (ê_k − qn).detach()          # STRAIGHT-THROUGH: fwd = unit entity direction, bwd gradient ~ q
    fold the ACTUAL entity vector POOL_k (native norm, = SNAPLOSS/ACTSNAP convention), graded answer along it

The per-epoch VAL rollout (te[:300]) realizes the same way — validation IS deployment; best checkpoint by val TAIL.
Deployment eval = new ENTSNAP=1 block (mirrors the PAIRSNAP harness exactly: same seeds/users/ruler/graded fold;
folds POOL[k], logs realization cos + repeat rate). Defaults unchanged; both features env-gated.

## Realization-cosine trajectory (same "exploit, don't migrate" signature as pairs — stronger)
Warm run, per-epoch mean cos(q, snapped entity) over 408 train-rollout turns:
ep1 0.6174 → ep2 0.5533 → ep3 0.5589 → ep4 0.5784 (best-val) → ep5 0.5783 → ep6 0.5981 → ep7 0.5919 → ep8 0.5810
→ ep9 0.5766 → ep10 0.5846.
- Post-hoc reference (frozen D1 queries snapped to the same 1361 pool): mean **0.7442** (median 0.7408, p10 0.5527).
- Eval-time distribution of the TRAINED entity-native policy: mean **0.5775** (median 0.5527, p10 0.3698, min 0.2859).
- So cos DROPPED ~0.17 below the frozen-policy reference while val/test rose: the policy does NOT move onto the
  entity manifold — it learns to emit off-manifold "steering" queries whose NEAREST-entity image folds informatively.
  Exactly the pair-rung mechanism, amplified by the coarser set (1,361 snap targets vs 6.9M).
- Snapped-question profile at eval (2,432 turns, seed 1): 85% concepts / 15% items; 24.8% of turns re-snap to an
  already-asked entity (post-hoc D1: 21.4%, 96% concepts) — with only 1,361 targets some duplication is inherent;
  duplicates waste part of the 8-question budget and are one reason this rung sits below pairs.

## Training run (TAG=enttrain_d1warm, TRSEED=0, EP=10, LR 5e-4)
Val trajectory (full/tail): ep1 0.321/0.114, ep2 0.312/0.111, ep3 0.323/0.109, **ep4 0.315/0.116 ← BEST(tail)**,
ep5 0.324/0.115, ep6 0.313/0.106, ep7 0.319/0.114, ep8 0.318/0.106, ep9 0.318/0.109, ep10 0.324/0.115.
Stable throughout (no destabilization; no LR retry needed). One training seed per the plan; the pair rung showed
±0.002 across training seeds under this protocol.

## Honest caveats
1. Same answer-model caveat as the whole ladder: answers are GEOMETRIC (graded cos along the realized direction).
   Here, uniquely, every realized action IS a named catalog entity, so an NL rendering is immediate — this rung has
   the weakest deployability caveat of the three.
2. Snap is unconstrained by per-user answerability (matches the pair rung; ACTSNAP's answerable-concepts-only variant
   is the stricter historical convention — its post-hoc number 0.3414/0.1384 is quoted for continuity).
3. CASPER-R tie is at seed-σ resolution; no per-user bootstrap run (the pair rung's bootstrap showed seed-σ
   understates user-level noise — treat "tie" as the claim, not any ±0.004 ordering).
4. Single training seed (TRSEED=0), per plan.

## Checkpoints / repro
- `data/movielens/.cache/policy_enttrain_d1warm{_best,_last,_ep1..10}.pt` (_best = ep4, sha1 9ce0914b, the headline);
  peak file `peak_enttrain_d1warm.txt` (val 0.3155/0.1163 @ep4/10, sel=tail); run row in `policy_runs.tsv`.
- Warm-start = locked D1 `policy_phase3_d1divw_last.pt` (sha1 a63fec1e, verified untouched before and after).
- Logs: `experiments/paper2/enttrain_d1warm.log` (train), `enttrain_d1warm_eval.log` (headline ENTSNAP eval),
  `entsnap_posthoc_d1.log` (post-hoc entity-snap control).
```
# train (this run):
ENTTRAIN=1 TRSEED=0 INIT=data/movielens/.cache/policy_phase3_d1divw_last.pt DIVW=1.0 DTAU=2.0 NUMAT=2500 \
  CONTMODE=cont GRADED=1 OBJ=ustar NOBC=1 FEATS=ext,ans EP=10 SELVAL=tail TAG=enttrain_d1warm \
  python scripts/paper2/continuous_actor.py
# eval (canonical ruler, entity-realized fold):
NOBC=1 EP=0 ENTSNAP=1 EVALSEEDS=1,2,3,7,11 CONTMODE=cont FEATS=ext,ans ANSF=1 \
  ACTORCK=data/movielens/.cache/policy_enttrain_d1warm_best.pt python scripts/paper2/continuous_actor.py
# post-hoc control: same ENTSNAP with ACTORCK=...policy_phase3_d1divw_last.pt ; ENTBIN=1 adds binary variant
```
