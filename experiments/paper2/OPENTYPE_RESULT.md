# Paper D P3 — "just an open asker policy?" HEADROOM = OVERFIT ARTIFACT (2026-06-30)

Question: does a LEARNED open-asker that adaptively SELECTS the question TYPE per turn/user (fav/gem/hate) beat the best
FIXED open schedule? Deterministic within-type recall (apples-to-apples): fav=most-popular liked, gem=distinctive
(high-align/low-pop), hate=disliked. 8 turns. Greedy-ORACLE = privileged upper bound (picks the type whose next entity
maximises held NDCG). OPENTYPE block. seed-avg {1,2,3}, te[300:].

## Held-peek oracle (NOT disjoint) — looks like big headroom
| schedule | FULL | TAIL |
|---|---|---|
| fav8 | 0.3761 | 0.1418 |
| gem8 (best fixed) | 0.4047 | 0.2097 |
| hate8 | 0.2714 | 0.0740 |
| fav4gem4 | 0.4046 | 0.1942 |
| fav2gem4hate2 | 0.4035 | 0.1974 |
| ORACLE (tail-opt) | 0.3962 | 0.2578 | -> +0.048 tail vs gem8 |
| ORACLE (full-opt) | 0.4613 | 0.1854 | -> +0.057 full vs gem8 |
NB ORACLE full-opt 0.461 > half-fold ceiling 0.41 = tell-tale HELD-SET OVERFIT (peeks at the eval targets).

## DISJOINT eval (OPTSPLIT: select type on held-half-A, eval on disjoint held-half-B) — headroom VANISHES
| schedule | FULL | TAIL |
|---|---|---|
| gem8 (best fixed) | 0.2129 | 0.1132 |
| ORACLE tail-opt | 0.2060 | 0.0875 | -> **-0.026 tail** vs gem8 |
| ORACLE full-opt | 0.1827 | 0.0961 | -> **-0.030 full** vs gem8 |
(abs numbers lower = eval on half the held likes; valid comparison is gem8 vs ORACLE on the SAME disjoint set.)

## TRAINED POLICY (REINFORCE on held-NDCG, held-out model selection) — the RIGHT test (supersedes the oracle)
The greedy oracle is a BAD proxy (overfits the noisy selection-half; also static is a SUBSET of adaptive so a learned
policy cannot lose to static except by optimisation failure). The correct test = TRAIN a policy and eval on TEST.
POLOPEN block: MLP(belief u, turn)->softmax over types, REINFORCE, reward=WF*full+WT*tail, train on 3000 trU users
(per-epoch resampled splits), model-select on te[:300] VAL, eval te[300:] TEST seed-avg{1,2,3}.

WF=1,WT=1, 80ep, types{fav,gem,hate}:
| | FULL | TAIL |
|---|---|---|
| LEARNED (best-val ckpt) | 0.4035 | 0.2086 |
| fixed gem8 | 0.4047 | 0.2097 |
| LEARNED - gem8 | -0.0011 | -0.0011 (= TIE, within noise) |
Per-turn greedy usage on TEST: ~93% gem, ~6% fav, ~0% hate at EVERY turn -> the policy CONVERGED to ~always-hidden-gem.

INTERPRETATION (honest, matches user's prediction "worst case = boring strategy replicating heuristic"):
- The trained adaptive policy TIES the best fixed schedule and does NOT lose (consistent with static subset of adaptive).
- It REDISCOVERS the framing: learned "tree" = essentially "always ask for a hidden-gem favourite". No interesting
  conditional structure for {fav,gem,hate}; hidden-gem is near-universally the best open question.
- This is the convincing version of "the lever is the FRAMING not the SEQUENCING" — shown by a real trained policy that
  CHOOSES to be ~static, NOT by a flawed oracle. (Probing richer type sets {+genre} and tail-weighted reward for whether
  any interesting tree emerges: runs bjyutchzu.)

## (SUPERSEDED as a verdict) Greedy-oracle headroom: an OVERFIT artifact, NOT evidence about a learned asker.
NOTE: the greedy oracle is NOT a valid headroom proxy here (it overfits the noisy selection-half; and static is a SUBSET
of adaptive, so a learned policy cannot truly lose to static). Use the TRAINED-POLICY result above as the verdict, not
this. Kept for the record / as the overfitting illustration only.
The privileged adaptive oracle, evaluated out-of-sample, LOSES to the best FIXED schedule (gem8 = always ask
"an underrated film you love"). The +0.048/+0.057 was entirely held-peek overfitting (same failure as the drop-oracle
SUBSETORACLE, which went +0.066 -> -0.008 under OPTSPLIT). This shows the GREEDY-SELECTION estimator overfits, NOT that
dominates. Consistent with the whole project: adaptive policy ~/< static on this ruler; the lever is the FRAMING (which
fixed question), not per-user SEQUENCING. Strengthens novelty #1 (framing) and closes the pure-open learned-asker question.

Repro: NOBC=1 EP=0 CONTMODE=cont OPENTYPE=1 [OPTSPLIT=1] OPT=tail|full EVALSEEDS=1,2,3 python scripts/paper2/continuous_actor.py
NOTE: joint open+CLOSED jointly-trained asker is a SEPARATE question (HYBRID bolt-on was also negative); pure-open is settled.
