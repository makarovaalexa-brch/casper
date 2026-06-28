# Paper C (draft) — Continuous-Action Preference Elicitation

## Goal 1 — RESULT: a continuous actor replicates the discrete winner (CASPER-R) to sub-noise

**Architecture.** The discrete policy (CASPER-R, Paper B) scores all 1361 pool entities and takes argmax.
We replace that with a **continuous actor**:
- `Actor: (belief u ∈ R^64, turn t) → query q ∈ R^64` — a small MLP (64→128→128→64) that emits a *point* in the
  shared item∪concept embedding space (not a choice from a menu).
- **Snap (cosine):** `k* = argmax_k cos(q, E_k)` over the pool — the entity whose embedding *direction* best matches q.
  (Cosine, not dot-product: dot is norm-biased and snaps to high-norm popular items; cosine matches the cos-loss training.)
- **Fold:** ask `k*`, user answers (geometric like/dislike), fold (E_k*, answer) through the **frozen set-encoder** → new belief u.

**Training = distillation from CASPER-R.** Roll out CASPER-R (the discrete winner, `policy_entdistill_ep4`, din=72 =
FEATS=ext+ANSF) on 1500 train users → 12 000 (state, its-exact-pick) demos. Train the actor so its query points at the
chosen entity's embedding: `loss = 1 − cos(actor(u,t), E_{k_CASPER})`. cos-loss 0.59 → 0.034 over 25 epochs.

**Result (te[300:], seed-avg {1,2,3,7,11}, NDCG@10):**

| @q | distilled CONTINUOUS actor (cosine snap) | CASPER-R (discrete) |
|---|---|---|
| q2 | 0.362 / 0.128 | — |
| q4 | 0.365 / 0.137 | — |
| **q8** | **0.362 ± 0.005 / 0.150 ± 0.003** (cos 0.747, ans 5.2/8) | **0.360 / 0.152** (cos 0.745, ans 5.3/8) |

→ The continuous actor reproduces the discrete winner to **within ±0.002 NDCG (sub-noise)**, with matching cos and
answered-rate. **The continuous "emit a point → snap" parametrisation is a lossless re-expression of the discrete policy.**
No Wolpertinger rescore was needed; pure-query distillation + cosine snap suffices.

Checkpoint: `PAPER_C_GOAL1/policy_cont_distill_replicates_casperR.pt` (sha e0d3b77d). Code: `scripts/paper2/continuous_actor.py`
(`DISTILL=1 FEATS=ext ANSF=1`; eval `COSSNAP=1`). Isolated copy — Paper B harness untouched.

## Why this matters / what continuity unlocks (Goal 2)
The discrete policy could only pick from a **fixed menu** of 1361 entities. The continuous actor emits a **point**, so it can
reach **beyond** the menu — novel directions / abstract concepts between the named ones — while the snap guarantees we can
always fall back to a real, answerable entity. Goal 1 proves continuity **costs nothing** (it matches discrete); Goal 2 tests
whether it **buys** anything:
- snap to a **richer / expanded** answerable concept bank (more concepts, same machinery), and/or
- treat the emitted point as an **abstract concept grounded in its nearest movies** — assume answerable (the PEBOL move),
  derive answer + popularity/answerability features from kNN real items (keeps the fold *in-distribution*, which the OOD
  fold of arbitrary points does NOT — see overnight result: folding arbitrary off-pool points plateaus at cos 0.46).

## Goal 2 — RESULT: continuity headroom is REAL, off-manifold, and not a u*-shortcut

We test whether continuity *buys* NDCG with a privileged greedy **oracle** (q=8 questions, graded geometric answer
`a = u*·q`; greedy = pick the direction maximising held-out NDCG@10 over the user's relevant items). The oracle is a
*ceiling*, not a realizable policy — but by giving the discrete and continuous variants the **same** privileged answer
model and the **same** frozen set-encoder, the *margin* isolates the value of the continuous action space itself.
*(directional: te[300:][:40] users, seed 1, tail-target, frozen V1 encoder ⇒ a lower bound — the encoder is not trained
to fold off-pool queries; seed-avg + full sample pending.)*

| oracle (q=8, NDCG@10 full / tail) | FULL | TAIL | pick-types |
|---|---|---|---|
| discrete — **concepts only** (cold-start askable) | 0.348 | 0.342 | concept 100% |
| **continuous** (off-pool directions allowed) | 0.377 | **0.391** | concept 82%, **novel 17%**, u* 1% |
| continuous — **random novel dirs only** (no u*/residual) | 0.370 | 0.385 | concept 84%, **novel 16%** |
| discrete — **concepts + items** (diagnostic; items NOT askable cold-start) | 0.382 | **0.418** | concept 71%, item 29% |

**Three findings, all pointing the same way:**

1. **Continuity is real, +0.043 tail over the askable concept catalog**, and it survives the strict control: with u* and
   its residual *removed* from the candidate set (`random novel directions only`), the gain barely changes (0.385 vs 0.391
   tail). So it is **not** a privileged "query your own taste vector" shortcut — when u* *is* offered, the oracle picks it
   only 1% of the time. The win comes from **random off-pool directions** chosen 16–17% of the time.

2. **The winning novel directions are genuinely off-manifold — not items in disguise.** For the novel picks, mean max-cosine
   to the nearest real **item = 0.43–0.45**, to the nearest **concept = 0.28–0.32**, and to **u* ≈ 0.00**. They match no
   catalog entity and are orthogonal to the taste vector — new directions in the embedding space.

3. **The "askability gap" reframes the contribution.** If items *were* askable, a discrete oracle does even better
   (0.418 tail) — i.e. the most discriminative directions live **near items**. But in cold-start you cannot ask a user about
   a specific unseen movie. **Continuous queries are the realizable bridge**: an answerable soft direction can point toward
   item-level discriminative structure that the askable *concept* catalog cannot reach — recovering +0.043 tail of that
   locked-away headroom *without* asking unaskable items, and (being ⊇ item directions) with a ceiling of ≥0.418 once the
   policy/encoder learn to aim there. Continuity does not *replace* concepts (still asked 82% of the time); it *adds* a
   targeted off-catalog edge on the ~17% of asks where no named concept fits.

Code: `CONTORACLE` block in `scripts/paper2/continuous_actor.py` (run `NOBC=1`; `DISCONLY`/`ITEMINC`/`NOUSTAR`/`NCAND`/`CCAP`
knobs). Result log: `experiments/paper2/PHASE05_HEADROOM_RESULT.md`, `experiments/paper2/phase05_continuity_1506.log`.

## Open question carried forward → now an OPTIMIZATION problem
The headroom exists and is realizable *in principle* (random exploration found it; no u* needed in the *choice*). The
remaining problem is to **realize it with a belief-only policy**. Plan: (Phase 0) train a from-scratch, length/type-balanced
*continuous-capable* encoder that folds off-pool (q,a) tokens in-distribution (raising the 0.46-cos frozen plateau); (Phase
2–3) train the continuous actor by **differentiable unroll** — backprop a reconstruction→NDCG-surrogate reward through the
8-step rollout using each training user's u* for the *gradient/simulator only* (the actor sees belief-only ⇒ realizable),
which gives a dense analytic gradient instead of the high-variance REINFORCE that has collapsed to the entropy basin.
Honesty controls to ship with any win: **SNAP-LOSS** (un-snapped vs snapped-to-phrase-bank — novel dirs at cos 0.3 to
concepts will be hard to verbalise, the real risk) and **adaptive vs static-schedule**.

*Theory note (why learned adaptivity has only ever tied entropy):* for a ~linear-Gaussian answer model the optimal
experimental design is **non-adaptive** (top eigenvectors regardless of answers). Adaptive continuous value therefore
requires nonlinearity — the ranking (NDCG) reward and/or **answerability** (taste-dependent, via the ABot bot-play) — so
continuous + adaptive + bot-play are one story, not three.
