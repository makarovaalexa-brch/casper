# Paper D open+closed COMBO (open-anchor -> closed-refine) — 2026-06-30

Hybrid: OPENK open favourite-movie folds (high-bandwidth ANCHOR), then (8-OPENK) Paper-C D1 continuous closed PROBES
(graded geometric) on the running belief. HYBRID block in continuous_actor.py. seed-avg {1,2,3}, te[300:].
Repro: NOBC=1 EP=0 CONTMODE=cont HYBRID=1 HEUR=popweight|distinct OPENK=k EVALSEEDS=1,2,3 python .../continuous_actor.py

| HEUR | open0+cl8 (pure D1) | open2+cl6 | open4+cl4 | open6+cl2 | open8 (pure open) |
|---|---|---|---|---|---|
| popweight FULL | 0.377 | 0.341 | 0.356 | 0.378 | **0.387** |
| popweight TAIL | 0.175 | 0.132 | 0.148 | 0.163 | **0.170** |
| distinct FULL | 0.377 | 0.377 | 0.393 | 0.399 | **0.405** |
| distinct TAIL | 0.175 | 0.184 | 0.197 | 0.209 | **0.211** |

Endpoints FAITHFUL: open0 = pure-D1 0.377/0.175 (= D1 0.378/0.178); open8 = pure-open (popweight 0.387/0.170, distinct
0.405/0.211). Harness validated.

## VERDICT: naive bolt-on open+closed does NOT beat pure open recall.
1. PURE OPEN WINS both rows; adding closed D1 probes only DILUTES toward D1 (distinct monotone up with more open).
2. popweight mid-mixes DIP BELOW BOTH endpoints (open2 0.341/0.132 << both) -> naive interleaving can actively HURT.
3. WHY: (a) closed probe = 1 scalar < a named 64-d factor (lower bandwidth, as the headline argues); (b) the D1 actor was
   trained to roll from an EMPTY belief, so feeding it a belief already seeded with open anchors is OFF-DISTRIBUTION ->
   its probes aren't targeted to the residual. The probe's Paper-C value is SUBSUMED by open recall, not additive.

## Implication (HONEST, motivates P3 — do NOT sell as a win)
Complementarity, IF any, requires a JOINTLY-TRAINED open+closed asker (the policy chooses, per turn/user, whether to ask
an open-recall question or a closed probe, trained on NDCG), NOT a frozen-probe bolt-on. This experiment is the ABLATION
that justifies P3's joint action space, and warns that a fixed schedule of open-then-closed underperforms pure open.
The user's hypothesis ("value from combining") = TRUE ONLY under a learned joint policy; the naive combine is a negative.

## JOINTLY-TRAINED open+closed asker (POLOPEN + 'closed' action) — 2026-06-30, CONFIRMS the above
Added a repeatable 'closed' action = Paper-C D1 continuous probe (query from belief -> geometric graded answer; NOT capped
by no-repeat since each probe is a fresh adaptive direction). Belief-only, NOREPEAT (open types once + closed repeatable),
CURVEREW anytime, 3 training seeds, seed-avg{1,2,3} test.
| | t5 full/tail | t8 full/tail |
|---|---|---|
| fixed NR-order | 0.393/0.177 | 0.405/0.195 |
| OPEN-only (anytime) | 0.398/0.190 | 0.405/0.194 |
| OPEN+CLOSED (joint) | 0.400/0.193 | 0.405/0.197 |
RESULT: jointly-trained open+closed BEATS fixed and SLIGHTLY beats open-only (~+0.003 tail mid/late). The policy LEARNS
the open-anchor->closed-refine structure: 'closed' ~0% early, DOMINATES late turns (t7 42-48%, t8 54-88% across seeds) —
open recall localizes the belief, closed probes refine. This is the complementarity the FROZEN BOLT-ON could NOT realize.
CAVEATS: gain over open-only is SMALL (~+0.003 tail, near seed noise 0.194-0.199); OPENER INSTABILITY (seed0 opened genre
-> weak t1 0.331/0.120 then recovered; seeds1-2 opened whatdoyoulike -> 0.387/0.189). HONEST STORY: combining open+closed
helps MODESTLY and ONLY jointly-trained (bolt-on NEG + joint POS = the clean pair). Repro: POLTYPES=...,closed.
