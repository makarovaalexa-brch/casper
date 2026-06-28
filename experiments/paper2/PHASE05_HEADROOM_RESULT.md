# Paper C Phase 0.5 — continuity headroom (DECISIVE GATE: PASS)
Privileged greedy oracle, q=8, graded answer a=u*·q, tail-target, te[300:][:40] users, seed 1.
Both oracles equally privileged => margin = PURE continuity benefit. Frozen V1 encoder (NOT continuous-trained => lower bound).

| oracle                         | FULL   | TAIL   | off-catalog picks |
|--------------------------------|--------|--------|-------------------|
| DISCRETE (catalog dirs only)   | 0.3481 | 0.3425 | 0%                |
| CONTINUOUS (off-catalog ok)    | 0.3771 | 0.3910 | 18%               |
| **margin (continuity)**        | +0.029 | +0.049 | —                 |

VERDICT: continuity headroom EXISTS (+14% tail) even in the linear world on an un-adapted encoder.
The prize is real; remaining work is OPTIMIZATION (realize it with a belief-only policy). Proceed to
Phase 0 (continuous-capable from-scratch encoder) + Phase 2-3 (differentiable-unroll continuous policy).
Caveats: privileged ceiling (not realizable); 40 users/1 seed (directional; margin >> seed noise).
NEXT scale-ups: more users + seeds; full-target (NORT=1); answerability/ABot world (adaptivity nonlinearity).
Harness: CONTORACLE block in continuous_actor.py; run with NOBC=1 (skip BC floor) + DISCONLY=1 for discrete.
