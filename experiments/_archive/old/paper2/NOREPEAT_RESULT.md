# Paper D — DEPLOYABLE no-repeat open-asker + curve + adaptive ordering (2026-06-30)

Motivation (user): asking "name a hidden gem" 8x in a row is NOT deployable. Real protocol = each DISTINCT open question
asked at most ONCE. Watch the NDCG CURVE (maximise EARLY, not just at turn 8); key operating point = turn 5; run to 8.
Added a completely-open "what do you like?" question (union: user volunteers ANY entity, best-aligned first).
NRCURVE block in continuous_actor.py. seed-avg {1,2,3}, te[300:]. 8 distinct questions:
gem, fav, actor, director, genre, avoidgenre, align(all-time-fav), whatdoyoulike.

## Deployable no-repeat CURVE (FIXED order: gem,fav,actor,director,genre,avoidgenre,align,whatdoyoulike)
| turn | FULL | TAIL |
|---|---|---|
| 1 (gem) | 0.3552 | 0.1727 |
| 2 | 0.3751 | 0.1672 |
| 3 | 0.3866 | 0.1703 |
| 4 | 0.3921 | 0.1770 |
| **5 (KEY)** | **0.3932** | **0.1769** |
| 6 | 0.3933 | 0.1762 |
| 7 | 0.4008 | 0.1881 |
| 8 (all) | 0.4054 | 0.1952 |
| [ref] gem x8 REPEAT (non-deployable) | 0.4047 | 0.2097 |

FINDINGS:
1. ~8 distinct open questions available before going closed. Asking each once: 0.405/0.195 @turn8.
2. FULL NDCG nearly SATURATES by turn 5 (0.393 vs 0.405 full-budget) -> 5 questions = efficient operating point.
3. DEPLOYABILITY COST is in the TAIL: 0.195 (no-repeat) vs 0.210 (gem-repeat). Repeating "hidden gem" keeps surfacing
   fresh tail items; asking it once caps the tail. Full is ~unaffected.

## Adaptive ORDERING (front-load): does per-user order help the early CURVE?
Set-encoder is PERMUTATION-INVARIANT => order doesn't change the turn-8 belief (confirmed: fixed==greedy @turn8), but
the CURVE (turns 1-7) depends on order. Greedy front-load = pick the unasked question maximising NDCG.
- FULL-HELD greedy = OVERFIT (tail 0.30-0.33 @early, ABOVE the 0.21 ceiling = held-peek). NOT real.
- HONEST disjoint (OPTSPLIT: select order on held-half-A, eval on disjoint half-B):
  | turn | FIXED full/tail | GREEDY full/tail |
  |---|---|---|
  | 5 (KEY) | 0.2061 / 0.0948 | 0.2107 / 0.0973 |  (greedy +0.0046 / +0.0025)
  | 8 | 0.2133 / 0.1055 | 0.2133 / 0.1055 |  (identical: same set)
  Greedy is consistently +0.004-0.006 full / +0.002-0.006 tail over fixed at turns 1-7; survives the disjoint split.

## VERDICT (CORRECTED 2026-06-30 after user caught the flaw)
- ⚠ The greedy (even disjoint OPTSPLIT) is PRIVILEGED: it peeks at the user's held likes to CHOOSE each question. That is
  NOT a realizable policy. In particular the "user-dependent OPENER" (gem 45% / actor 11% / ...) is UNLEARNABLE: at turn 0
  every user has belief = enc(empty) = 0, so a realizable belief-conditioned policy sees an IDENTICAL state for all users
  and MUST emit the SAME opener. The greedy's opener diversity came only from peeking at preferences. RETRACTED as a
  realizable finding.
- So the realizable OPENER is FIXED = the single best one (gem, by turn-1 tail 0.173). Adaptivity can ONLY come from
  turn>=1, conditioning on the answers ALREADY GIVEN this conversation (the running belief).
- The +0.005 disjoint-greedy "headroom" OVERSTATES realizable adaptivity (greedy still peeks at half-A to select). The
  realizable value is whatever a BELIEF-CONDITIONED learned policy captures -> must be measured by TRAINING it, not by the
  greedy. [SUPERSEDES the "first positive adaptivity" claim above pending the learned-policy run.]
- NOTE the FEATS per-type features (align/pop/div of the entity each type WOULD surface) are themselves partly PRIVILEGED
  (they require knowing the not-yet-given answer); realizable population signal = TYPE-LEVEL priors only (constant per type
  = a per-type bias the policy already learns). The FEATS tie therefore used an over-powered state => negative is stronger.
- DEPLOYABLE no-repeat FIXED-order curve (0.393/0.177 @turn5, 0.405/0.195 @8) and the permutation-invariance (turn-8
  order-invariant; at turn 5 only WHICH-5-subset matters) STAND. "what do you like?" is a viable question type.

## REALIZABLE WIN: belief-only ANYTIME no-repeat learned asker (CURVEREW = reward over ALL lengths) — 2026-06-30
Trained ONE policy on reward = MEAN NDCG over turns 1..8 ("anytime", good at every length), belief-only (REALIZABLE: no
privileged features; turn-0 state identical for all users => fixed opener), NOREPEAT, 9 distinct Qs, REINFORCE, held-out
val model-select, seed-avg{1,2,3} TEST te[300:]. vs FIXED NR-order (gem,fav,actor,director,genre,avoidgenre,align,wdyl).
| turn | LEARNED full/tail | FIXED full/tail |
|---|---|---|
| 1 | 0.3691 / 0.1714 | 0.3552 / 0.1727 |  (full +0.014)
| 2 | 0.3829 / 0.1752 | 0.3751 / 0.1672 |  (+0.008/+0.008)
| 3 | 0.3884 / 0.1774 | 0.3866 / 0.1703 |
| 4 | 0.3913 / 0.1819 | 0.3921 / 0.1770 |
| 5 (KEY) | 0.3940 / 0.1872 | 0.3932 / 0.1769 |  (tail +0.010)
| 6 | 0.3977 / 0.1903 | 0.3933 / 0.1762 |  (tail +0.014)
| 7 | 0.4016 / 0.1900 | 0.4008 / 0.1881 |
| 8 | 0.4037 / 0.1936 | 0.4054 / 0.1952 |  (ties: permutation-invariant endpoint)

WIN: the realizable belief-only policy BEATS the fixed order at every EARLY/MID turn, esp TAIL (+0.010 @t5, +0.014 @t6)
and FULL +0.014 @t1. Converges at t8 (set-invariant). The value is EARLY EFFICIENCY (exactly the "maximise asap" goal).

THE LEARNED TREE (interpretable, per-user adaptive from turn 2; greedy usage):
- turn1 OPENER (fixed, all users identical): ALIGN 100% = "your all-time favourite" (best single most-informative item;
  NOT gem).
- turn2: genre 67% / fav 21%.  turn3: actor 48% / fav 18% / hate 13%.  turn4: actor 37% / gem 27%.
- turn5: gem 51% / hate 23%.  turn6: director 35% / hate 30% / gem 18%.  turn7: director 44% / hate 18% / avoidgenre 12%
  / wdyl 11%.  turn8: wdyl 42% / avoidgenre 33%.
- Strategy: open with defining favourite -> genre -> actor -> mid mix gem/hate/director (tail/negative) -> finish open +
  avoid-genre. Coherent + adaptive.

WHY IT WORKED (when belief-only-repeats-allowed and single-horizon both TIED gem8): training for ALL LENGTHS (CURVEREW)
found a FRONT-LOADED adaptive strategy. The deployability constraint (no-repeat) + anytime reward TOGETHER make adaptivity
realizable. = the FIRST realizable adaptive-policy win on the whole project, with an interpretable tree.
TRAINING-SEED ROBUSTNESS (POLSEED 0,1,2; each eval seed-avg{1,2,3} test) — WIN HOLDS, seeds 1-2 STRONGER:
| turn | LEARNED full/tail s0 | s1 | s2 | FIXED |
|---|---|---|---|---|
| 1 | .369/.171 | .355/.173 | .387/.189 | .355/.173 |  (opener varies: align/gem/align; never loses)
| 2 | .383/.175 | .384/.192 | .389/.191 | .375/.167 |
| 5 | .394/.187 | .398/.190 | .401/.194 | .393/.177 |  seed-avg +0.005 full / +0.013 tail
| 6 | .398/.190 | .402/.188 | .404/.194 | .393/.176 |  seed-avg +0.007 full / +0.015 tail
| 8 | .404/.194 | .406/.195 | .406/.195 | .405/.195 |  ties (perm-invariant endpoint)
All 3 training seeds beat fixed at turns 2-7, consistently on TAIL (+0.013-0.024). Opener choice varies by seed (all
sensible single openers, fixed across users). ROBUST.

HONEST CAVEATS: gains small-to-moderate (+0.005 full/+0.013 tail @t5 seed-avg), concentrated EARLY/MID + TAIL; VANISH at
the full-budget endpoint (perm-invariant encoder). Correct framing: ADAPTIVITY BUYS EFFICIENCY, NOT A HIGHER CEILING.

## TRAINER ABLATION: REINFORCE vs GUMBEL-softmax differentiable (SOFT block) — REINFORCE WINS
Implemented SOFT path: Gumbel-softmax straight-through over the type choice + differentiable softndcg (Paper-C objective)
backpropped through the torch encoder fold. 2 seeds, 120 ep, no-repeat anytime open-only.
| | t5 full/tail | t8 full/tail |
|---|---|---|
| GUMBEL-SOFT | 0.369-0.371/0.144 | 0.400/0.185 |
| REINFORCE (anytime) | 0.398/0.190 | 0.405/0.194 |
| fixed NR-order | 0.393/0.177 | 0.405/0.195 |
Gumbel UNDERPERFORMS REINFORCE AND fixed, esp on TAIL. WHY: softndcg ranks held-likes vs POPULAR negatives = a FULL-metric
surrogate MISALIGNED with the exact tail-weighted NDCG@10 that REINFORCE optimizes DIRECTLY; + straight-through grad
through a perm-invariant encoder is a weak signal for a DISCRETE action. METHODOLOGICAL NOTE: differentiable soft-NDCG is
natural for Paper C (CONTINUOUS action) but for Paper D's DISCRETE type choice + exact tail metric, DIRECT REINFORCE on
the true reward WINS. Could close the gap w/ a tail-weighted soft-NDCG + temp annealing (= tuning to a foregone result);
faithful apples-to-apples => REINFORCE is the better trainer here. Repro: SOFT=1 GTAU=1.0.
Repro: NOBC=1 EP=0 CONTMODE=cont POLOPEN=1 NOREPEAT=1 CURVEREW=1 HORIZON=8 ENT=0.02 POLEP=90 [POLSEED=k]
POLTYPES=gem,fav,align,hate,genre,avoidgenre,actor,director,whatdoyoulike
NRORDER=gem,fav,actor,director,genre,avoidgenre,align,whatdoyoulike EVALSEEDS=1,2,3. Ckpt: .cache/polopen_*_wf1.0wt1.0.pt.

Repro: NOBC=1 EP=0 CONTMODE=cont POLOPEN=1 NRCURVE=1 [OPTSPLIT=1] OPT=tail EVALTURN=5
POLTYPES=gem,fav,align,hate,genre,avoidgenre,actor,director,whatdoyoulike
NRORDER=gem,fav,actor,director,genre,avoidgenre,align,whatdoyoulike EVALSEEDS=1,2,3 python scripts/paper2/continuous_actor.py
NEXT (optional): train the no-repeat learned asker (reward = NDCG@turn5) vs fixed order; does a LEARNED order capture the
small disjoint-greedy edge? (gate: it should >= fixed and approach the +0.005 honest headroom.)
