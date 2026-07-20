# Paper C Phase 0.5 — continuity headroom + honesty controls (GATE: PASS, REAL continuity)
Privileged greedy oracle, q=8, graded answer a=u*·q, tail-target, te[300:][:40] users, seed 1, frozen V1 encoder.
Same answer model + encoder across variants => margin isolates the ACTION SPACE. Lower bound (encoder not trained on off-pool).

| oracle (NDCG@10)                          | FULL   | TAIL   | pick-types                  |
|-------------------------------------------|--------|--------|-----------------------------|
| discrete — concepts only (cold-start)     | 0.3481 | 0.3425 | concept 100%                |
| CONTINUOUS (off-pool dirs allowed)        | 0.3771 | 0.3910 | concept 82%, novel 17%, u* 1% |
| CONTINUOUS rand-only (no u*/residual)     | 0.3697 | 0.3854 | concept 84%, novel 16%      |
| discrete — concepts+items (DIAGNOSTIC)    | 0.3822 | 0.4184 | concept 71%, item 29%       |

NOVEL-pick diagnostics (continuous): mean max-cos nearest ITEM 0.43-0.45 | nearest CONCEPT 0.28-0.32 | to u* ~0.00.

## Verdict
1. Continuity REAL: +0.043 tail over askable concepts; survives rand-only control (0.385, ~unchanged) => NOT a u* shortcut
   (u* offered but picked 1%). Driven by random off-pool dirs (16-17% of asks).
2. Novel dirs genuinely OFF-MANIFOLD: far from any item (cos .44), concept (cos .30), and u* (cos .00) — not items in disguise.
3. ASKABILITY GAP: discrete+items (0.418) > continuous (0.391) => most discriminative dirs are item-aligned, but items are
   UNASKABLE in cold-start. Continuous = realizable bridge to that locked headroom (ceiling >=0.418 since continuous ⊇ items).
   Continuity ADDS to concepts (still asked 82%), not replaces.

Caveats: privileged ceiling (not realizable — Phase 2-3 is the realization); 40 users / 1 seed / tail-target (directional;
margin >> seed noise). NEXT: seed-avg + full sample + full-target (NORT=1) + ABot answerability world; then Phase 0 encoder + Phase 2 unroll.
Code: CONTORACLE block in scripts/paper2/continuous_actor.py (NOBC=1; DISCONLY/ITEMINC/NOUSTAR/NCAND/CCAP).
