# P0 DIAGNOSTICS — C-B4 graded-discrete TEST eval + answerability-oracle entropy + NDCG-vs-answered curve

**Date:** 2026-07-02. Three small eval experiments, no training, winners untouched. Ruler everywhere: ML-1M, te[300:]
(304 users), q8, eval seeds {1,2,3,7,11}, NDCG@10 FULL + Cremonesi TAIL, frozen V1 encoder.

---

## 1) C-B4: graded-discrete control TEST eval — **Paper C's claim SURVIVES**

**Question (review C-B4):** the continuous actor gets 8/8 answerable by construction vs CASPER-R's ~5.3/8. Decisive
control = a DISCRETE policy RETRAINED natively on GRADED answers with full answerability over the same 761 concepts.
Opus trained it (`policy_gradeddisc_rerun`, 30 epochs, SKIPVAL, log `reruns_batch1.log` "RERUN C: graded-discrete
control (SNAP concepts, graded)"); only the TEST eval was pending.

**Eval:** canonical Paper C ruler = COMPARE4 ONLYACTOR block in `scripts/paper2/continuous_actor.py`, with **ACTSNAP=1**
(emit-then-snap to nearest ANSWERABLE concept => discrete realization, 8/8 answered — exactly the C-B4 spec).
Sanity: unsnapped eval of the same checkpoint is far worse (0.276/0.078 seed 1, graded) => the policy is snap-native;
ACTSNAP is its faithful/best realization. Epoch scan (seed 1, graded/snapped): ep6 0.328/0.130, ep12 0.326/0.126,
ep18 0.320/0.126, ep24 0.317/0.127, ep30 0.320/0.126 — flat; best scan epoch = ep6 (test-generous selection: there is
no val checkpoint because SKIPVAL; giving the control its best-on-test epoch only makes the conclusion safer).

**Result (5-seed COMPARE4, mean±std over seeds):**

| condition | FULL | TAIL |
|---|---|---|
| D1 continuous, un-snapped, graded (headline) | 0.3780 ± 0.0032 | 0.1782 ± 0.0065 |
| uent+GRAW static (graded ruler variant) | 0.367 / — | 0.158 / — |
| entropy binary (canonical discrete heuristic) | 0.3618 ± 0.0026 | 0.1393 ± 0.0041 |
| CASPER-R binary (discrete SOTA) | 0.3594 ± 0.0055 | 0.1467 ± 0.0049 |
| entropy graded | 0.3529 ± 0.0038 | 0.1328 ± 0.0065 |
| CASPER-R graded (binary-trained, graded answers) | 0.3432 ± 0.0023 | 0.1382 ± 0.0034 |
| D1 concept-snap (SNAPLOSS reference) | 0.3414 | 0.1384 |
| **graded-discrete control, ep6 (best-scan, graded)** | **0.3247 ± 0.0029** | **0.1247 ± 0.0040** |
| **graded-discrete control, ep30=last (graded)** | **0.3215 ± 0.0027** | **0.1195 ± 0.0040** |
| graded-discrete control, ep30 (binary) | 0.3261 ± 0.0045 | 0.1256 ± 0.0032 |

**Verdict: the natively-graded discrete policy does NOT reach ~0.37/0.16 — it lands at 0.32/0.12, BELOW static entropy
(0.353/0.133 graded), below binary-trained CASPER-R on graded (0.343/0.138), and −0.053/−0.053 below D1.** Full
answerability (8/8 by construction via ACTSNAP) plus native graded training does not rescue the discrete action space.
Paper C's narrowed claim — the graded win needs the CONTINUOUS query, "two inseparable continuities" — survives the
C-B4 control. Note the training return improved monotonically (−0.155→−0.133) while test NDCG stayed flat at ~0.32
across all 30 epochs: the snap-constrained policy optimizes its reconstruction objective without converting it into
ranking gains, mirroring the 8 failed discrete policy-learning attempts of the Paper C campaign.

Caveats: one training run (training-seed robustness not re-checked for this control; D1 itself is ±0.003/±0.006 over
3 training seeds); exact training env reconstructed from Opus's log heading (SNAP concepts, graded); epoch choice is
test-selected (generous to the control — both ep6 and last give the same verdict).

Repro:
```
NOBC=1 EP=0 COMPARE4=1 ONLYACTOR=1 ACTSNAP=1 EVALSEEDS=1,2,3,7,11 CONTMODE=cont FEATS=ext,ans ANSF=1 \
  ACTORCK=data/movielens/.cache/policy_gradeddisc_rerun_ep6.pt python scripts/paper2/continuous_actor.py
```
Scan log: `experiments/paper2/gradeddisc_scan.log`. Checkpoints: `policy_gradeddisc_rerun{_ep1..30,_last,.pt}`
(last == ep30 == .pt, verified identical weights; no `_best` exists — SKIPVAL run).

---

## 2) P0-a: answerability-oracle entropy (Paper B harness) — headroom is FRONT-LOADING only

**Setup:** `scripts/paper2/continuous_policy2.py`, new opt-in mode `entropy_ansoracle` (env-gated via
MODES/EVALBASE, never default): SAME fixed population entropy(divisiveness) ranking, but SKIP concepts unanswerable
for THIS user (peeks at the >=2-tagged-profile-items answerability criterion of `cans_np` ONLY — never at answer
values) => all 8 turns answered. Baseline = canonical `entropy` (unanswerable asks waste the turn). 5 seeds,
QPTS=0..8, binary answers (Paper B convention).

**Result (seed-avg, NDCG@10):** baseline answers/8 = **5.16**; oracle = **8.00**.

| q | entropy FULL | ansoracle FULL | Δ | entropy TAIL | ansoracle TAIL | Δ |
|---|---|---|---|---|---|---|
| 1 | 0.3193 ± 0.0047 | 0.3302 ± 0.0019 | **+0.0109** | 0.1034 ± 0.0038 | 0.1111 ± 0.0063 | **+0.0077** |
| 2 | 0.3482 ± 0.0073 | 0.3569 ± 0.0029 | **+0.0087** | 0.1201 ± 0.0047 | 0.1343 ± 0.0036 | **+0.0143** |
| 4 | 0.3588 ± 0.0037 | 0.3627 ± 0.0023 | +0.0039 | 0.1353 ± 0.0038 | 0.1394 ± 0.0037 | +0.0041 |
| 8 | 0.3609 ± 0.0014 | 0.3618 ± 0.0026 | +0.0010 | 0.1397 ± 0.0039 | 0.1392 ± 0.0040 | −0.0005 |

**Verdict: the ENTIRE realizable wasted-turn headroom is an EFFICIENCY (low-q front-loading) effect — up to
+0.011 FULL @q1 / +0.014 TAIL @q2 (~2–3σ) — and vanishes at q8 (+0.001/−0.001, a tie).** Answerability knowledge lets
the asker reach the q4 plateau ~2 questions sooner, but does not raise the plateau. For Paper B this bounds any
answerability-aware selection mechanism: it can buy speed, not endpoint NDCG@8. (Baseline entropy q8 0.3609/0.1397
reproduces the canonical 0.361/0.140 — ruler verified.)

Repro:
```
NOBC=1 NOTRAIN=1 EVALCKS=',' EVALBASE=entropy,entropy_ansoracle EVALSEEDS=1,2,3,7,11 QPTS=0,1,2,3,4,5,6,7,8 \
  EVALCSV=data/movielens/.cache/ansoracle_grid.csv ANSDUMP=experiments/paper2/ansdump_entropy.csv \
  python scripts/paper2/continuous_policy2.py
```

---

## 3) P0-b: NDCG vs ANSWERED-count under static entropy — 5.16 answered IS the plateau

From the per-user dump of run 2's baseline (`ansdump_entropy.csv`: seed,mode,tail,user,q,answered-so-far,NDCG@10;
env-gated ANSDUMP, inert by default). Two views:

**A) Cross-user @q8, bucketed by total answered (CONFOUNDED — users with more answerable concepts have denser/easier
profiles):** FULL rises 0.06→0.70 and TAIL 0.03→0.34 from ans=2 to ans=8. This gradient is a selection effect over
users, NOT the causal value of answers — do not read it as headroom.

**B) Within-user paired view (honest): NDCG at the first turn where k answers have accumulated; marginal value of the
k-th answered question = paired per-user difference:**

| k-th answered Q | FULL Δ | TAIL Δ | n users×seeds |
|---|---|---|---|
| 1 | +0.0203 | +0.0304 | 1520/1515 |
| 2 | +0.0270 | +0.0234 | 1497/1492 |
| 3 | −0.0005 | +0.0042 | 1417/1412 |
| 4 | +0.0074 | +0.0017 | 1216/1214 |
| 5 | +0.0013 | −0.0004 | 929 |
| **6** | **−0.0011** | **+0.0007** | 673 |
| **7** | **−0.0042** | **+0.0008** | 439 |
| **8** | **−0.0042** | **+0.0009** | 150 |

**Verdict: performance saturates at ~2–4 answered questions; at the baseline's 5.16 answered the policy is already ON
the 8-answer plateau. The marginal value of the 6th/7th/8th answered question is ≤ +0.001 TAIL and ~0/negative FULL.**
Consistent with (2): converting the ~2.8 wasted turns into answered turns cannot help at q8 because answers 6–8 are
worthless under this (fixed-ranking) instrument — the binding constraint is the INFORMATIVENESS of later questions,
not their answerability.

Analysis script: scratchpad `agg_ansdump.py` (bucketing + within-user first-reach pairing). Dump:
`experiments/paper2/ansdump_entropy.csv` (27k rows, 5 seeds × 304 users × 9 q-points × full/tail).

---

## Code provenance
- `continuous_policy2.py`: added opt-in mode `entropy_ansoracle` + `ANSDUMP` per-user CSV dump + `DUMPSEED` tag line
  in EVALCKS loop. All env-gated / non-default; canonical modes and defaults untouched. NOT committed.
- `continuous_actor.py`: no changes (pre-existing COMPARE4/ACTSNAP used as-is; working-tree PAIRSNAP diff untouched).
- No locked checkpoints touched; no training runs.
