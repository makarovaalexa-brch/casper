# E0d / E0e — coarse→fine DESCENT probe + emergent-switch (ML-25M, RecVAE-d512, 298 users)

Date 2026-07-07. Scripts `scripts/e0d_descent.py` (E0d hard-coded descent), `scripts/e0e_emergent.py`
(E0e emergent switch + emergence diagnostic), `scripts/e0e_reweight.py` (E0e P3 fold re-weighting).
**NO LLM calls** — all cached grids. Reuses the E0 harness EXACTLY via `import e0_gonogo as E`
(fold-in, belief `z'=z+eta·a·q` with `a=cos(z*,q)`, `eta=16`, NDCG@10, per-user paired bootstrap).
Backing JSON: `experiments/E0d_descent.json`, `E0e_emergent.json`, `E0e_reweight.json`.

Motivating context: FABLE_AGENT_DESIGN §"E0's GAP, CAUGHT BY THE OWNER" — E0's routers never actually
performed coarse→fine channel descent, so E0's NO-GO killed *routing-within-tested-scorers*, not
*channel descent*. E0d/E0e test descent directly.

## Reproduction check (done FIRST, same harness)

- static B: **anytime 0.2517 / endpoint 0.2812** ✓ (E0: 0.2517 / 0.2812)
- static+skip: **anytime 0.2516 / endpoint 0.2796** ✓ (E0: 0.2514 / 0.2785; within bootstrap noise of
  the priority-list tail ordering)
- O-full endpoint 0.5681 inherited from the E0 G2 reproduction (context only).
- fold-reweight `static_base` reproduces endpoint **0.2812** exactly.

All arms below run on the **same 298 eligible users** (prof and target-like both non-empty).

---

## VERDICT: DESCENT LOSES — the coarse→fine thesis does NOT revive

Every descent arm — every switch point, every value model, hard-coded or emergent, true-table or
p̂ — is **significantly below static B** (bootstrap CI excludes 0). E0's NO-GO stands, and the
dilution mechanism the design *inferred* is now confirmed **directly**: switching to the item channel
after concept saturation actively *lowers* NDCG.

### P1 — E0d hard-coded descent (phase 1 = k schedule concepts for all; phase 2 = 8−k per-user items)

| arm | anytime | endpoint | vs static B (anytime) | 95% CI | vs static+skip (anytime) |
|---|---|---|---|---|---|
| **static B** (ref) | 0.2517 | 0.2812 | 0 | — | — |
| static+skip | 0.2516 | 0.2796 | −0.0001 | — | 0 |
| descent k=4, true table, pop-value | 0.2406 | 0.2614 | **−0.0111** | [−0.0176, −0.0047] | −0.0110 |
| descent k=4, true table, divisiveness | 0.2359 | 0.2517 | −0.0159 | [−0.0229, −0.0091] | −0.0157 |
| descent k=3, true table, pop-value | 0.2343 | 0.2511 | −0.0174 | [−0.0258, −0.0089] | −0.0172 |
| descent k=5, true table, pop-value | 0.2441 | 0.2557 | −0.0076 | [−0.0122, −0.0029] | −0.0075 |
| descent k=4, **p̂ surrogate** (realizable) | 0.2384 | 0.2573 | −0.0133 | [−0.0199, −0.0068] | −0.0132 |

**The per-turn curve is the whole story** (descent k=4 true pop-value vs static B):

| t | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| static B | 0.176 | 0.227 | 0.246 | 0.267 | 0.268 | 0.272 | 0.278 | 0.281 |
| descent (switch after t4) | 0.176 | 0.227 | 0.246 | 0.267 | **0.248** | 0.248 | 0.252 | 0.261 |

Turns 1–4 are identical (same 4 concepts). At turn 5 static's 5th concept edges **up** 0.267→0.268
(concepts saturate — the owner's premise is confirmed), but the descent's first item pick **drops**
0.267→0.248 and never recovers to the static line. The item answer (geometric `cos(z*, item_dir)`,
typically small / partly off-axis) injects an `eta=16` step that *dilutes* rather than sharpens the
belief — the non-monotone-fold dilution (Paper B) made visible. Higher k (more concepts, later switch)
loses least (k=5 −0.0076 < k=4 −0.0111 < k=3 −0.0174): the more you delay leaving the concept channel,
the less you lose. There is no switch point at which descent wins.

The p̂-surrogate arm (b) loses about the same as the true-table arm (−0.0133 vs −0.0111): surrogate
decision loss is second-order here — the item channel is *value-destroying whether or not you route to
answerable items*, so a better answerability estimate cannot help.

### P2 — E0e emergent switch (greedy over the whole bank; belief-only marginal-info × answerability)

`score(q) = marginal_info(q | answered dirs) × answerability(q) [× divisiveness(q)]`, where
`marginal_info` = residual norm of q against the orthonormal span of already-**answered** directions
(never touches z*). "true" arm restricts candidates to answerable; "phat" arm scores all candidates by
p̂ with a refused pick wasting the turn.

| arm | anytime | endpoint | vs static B (anytime) | 95% CI |
|---|---|---|---|---|
| static B (ref) | 0.2517 | 0.2812 | 0 | — |
| emergent · true table · no-div | 0.1489 | 0.2136 | −0.1028 | [−0.1214, −0.0848] |
| emergent · true table · × div | 0.2039 | 0.2385 | −0.0478 | [−0.0640, −0.0314] |
| emergent · p̂ · no-div | 0.1735 | 0.2211 | −0.0782 | [−0.0948, −0.0625] |
| emergent · p̂ · × div | 0.2036 | 0.2402 | −0.0481 | [−0.0641, −0.0322] |

All lose, and by more than the hard-coded descent — a marginal-information objective is **actively
anti-correlated** with ranking value in this arena, because the most "informative" (most orthogonal)
directions are specific items whose geometric answer dilutes the belief.

### THE EMERGENCE DIAGNOSTIC (per user, turn of first item pick)

| arm | switched to item | first-item turn (q25 / median / q75) | never-switch score gap (concept−item) |
|---|---|---|---|
| emergent · true · no-div | **292/298 (98%)** | 4 / **4** / 4 | mean 0.082 (for the 6 who don't) |
| emergent · true · ×div | 294/298 (99%) | 1 / 1 / 1 | 0.004 |
| emergent · p̂ · no-div | **22/298 (7%)** | 7 / 7 / 8 | mean 0.052 |
| emergent · p̂ · ×div | 298/298 (100%) | 1 / 1 / 1 | — |

The mechanism the owner asked for **does fire**: with the true table and pure marginal-info, the
selector emergently abandons the concept channel at a **tight median turn 4** (histogram: 262/298 users
switch at exactly t4, after the 3–4 coarse concepts are folded and items become the most-orthogonal
remaining directions). This is precisely the "switch when concepts saturate" behaviour — it just
*costs* NDCG. The only thing that suppresses the switch is p̂ (concepts are far more answerable, so the
p̂·no-div arm keeps 93% of users on concepts until t7–8, never-switch gap 0.052 in favour of concepts).
Emergence is real; its NDCG payoff is negative.

### P3 — E0e fold re-weighting (down-weight redundant answer tokens; fair on both schedules)

Fold weight `w_t = marginal_info(q_t | previously answered span)`, so near-duplicate tokens contribute
less to z.

| schedule | base anytime | base endpoint | reweight Δ anytime | 95% CI | reweight Δ endpoint |
|---|---|---|---|---|---|
| static B | 0.2517 | 0.2812 | −0.0004 | [−0.0012, +0.0003] | +0.0002 |
| descent k=4 true pop | 0.2406 | 0.2614 | **+0.0031** | [+0.0019, +0.0043] | **+0.0059** |

Re-weighting does **nothing** to static B (already diversified — CI spans 0) and **helps descent**
(anytime +0.0031, endpoint +0.0059, both CI-clean) — exactly where the dilution lives, confirming the
diagnosis. But it does **not close the gap**: reweighted descent endpoint 0.2673 is still well below
static's 0.2812. Re-weighting repairs part of the self-inflicted item dilution; it cannot make descent
worth doing.

---

## What this means for the design

E0d/E0e answer the owner's challenge in the owner's own stated form and **confirm E0's NO-GO, now for
channel descent specifically** (not merely routing-within-scorers):

- Coarse→fine descent was tested hard-coded (3 switch points × 2 value models × true/p̂) and emergent
  (4 selector variants). **It loses everywhere, significantly.**
- The dilution/low-marginal-info explanation is now **direct** (per-turn NDCG drops at the switch;
  re-weighting helps only the descent arm), not inferred.
- Concepts *do* saturate (static gains ~0.001/turn after t4; the emergent selector wants to leave at
  t4) — but the item channel's geometric answer is a *worse* use of the remaining turns than a fifth
  concept, so the two-regime "coarse concepts, then stop" story proceeds unchanged, with a stronger
  footnote: *we tried descent five ways and it strictly loses; the item channel's marginal information
  does not translate into belief that improves ranking under the geometric answer model at η=16*.

---

## ASSUMPTIONS / judgment calls (explicit)

1. **Item answer values = geometric, NOT real ratings.** The E0 harness (and therefore this one)
   computes every answer as `a = cos(z*, q)` where `z*` is the RecVAE encoding of the user's liked
   known-profile items — for **concepts and items alike**. Real star ratings never enter the answer
   value; the LLM-judged answerability grid only **gates which questions are askable** (`rec["ans"]`).
   Consequently there is **no rated/unrated answer-value distinction** to make in phase-2 item routing:
   an "answerable" item's answer is the same geometric quantity whether or not that item is in the
   user's rating history. **No SENSITIVITY-ONLY arm was needed** (there are no LLM-predicted answer
   values anywhere in this harness). This is the single most load-bearing assumption; if a future
   harness used real-rating item answers, the descent verdict would need re-checking.
2. **Phase-1 concepts** = the concept keys of the static schedule in schedule order
   (`C:77, C:11, C:136, C:10, C:67, C:70, C:5`; the schedule's one item slot `I:1279` is excluded from
   the concept phase). For k=4: `C:77, C:11, C:136, C:10`. A phase-1 slot that a user cannot answer is
   a refusal that **consumes the turn** (hard-coded schedule) — matching `E.sel_static`. Concepts are
   ~universally answerable (37.8/39 per user) so phase-1 refusals are rare.
3. **Phase-2 item value model.** Primary = **population value-when-answered single-question NDCG**
   (`popval[k]` = mean over users who can answer k of the one-shot NDCG), the same population greedy
   value selector B is built from. Secondary = **divisiveness** `div(q)=qᵀCov(W)q` (belief-independent).
   pop-value beats divisiveness among the descent arms but both lose. NOTE: many item keys are answered
   by very few users (216 distinct item keys over 300 users), so the top `popval` items are estimated
   from as few as 1 answerer and are noisy; this favours the descent arm if anything (it cherry-picks
   high-estimated-value items) and it still loses.
4. **P1 arm (b) p̂ routing** ranks ALL of a user's item candidates by `p̂ · max(popval,0)` (expected
   value) and picks the top (8−k); a pick that is truly unanswerable is a refusal (turn consumed, no
   belief update). This is the realizable analogue of arm (a).
5. **E0e marginal_info** = residual norm of a candidate direction against the Gram-Schmidt orthonormal
   span of the directions **actually answered so far** (unit-norm candidate ⇒ marginal ∈ [0,1]).
   Belief-only; never uses z*. At t1 all candidates have marginal ≈ 1, so ties break to the first
   candidate (a concept), which is why even the emergent arms open on a concept.
6. **E0e "true table" arm** implements `answerability(q)` as a hard 0/1 restriction to answerable
   candidates; the "p̂" arm uses the continuous p̂ multiplier with a true-table-gated outcome.
7. **Divisiveness multiplier** clamped at 0 (`max(div,0)`) before multiplying, so a rare negative
   numerical `qᵀCq` cannot flip the argmax sign.
8. **P3 fold re-weighting** applies `w_t = marginal_info(q_t | previously answered span)` at the fold
   step (`z += eta·a·w_t·q`) and is applied to BOTH static B and descent k=4-true-pop for a fair
   comparison; the answered direction is added to the span **unweighted** (the weight scales only the
   belief step, not the redundancy bookkeeping).
9. **Candidate pool** = the gate grid's per-user bank (39 concepts + 36 items per user), identical to
   the E0 selectors, so static/static+skip reproduce 0.2812/0.2796. Users average 10.7 answerable items
   (median 11; 89% have ≥4), so the phase-2 item budget is fillable for 295/298 users.
10. **Deterministic**: `np.random.default_rng(0)` for all bootstraps (5000 resamples); RecVAE fold-in
    and concept bags are cached and deterministic.
