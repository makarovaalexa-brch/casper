# Conversational bandits on our ruler: the concept channel finally has a published opponent

Run 2026-08-07, 2h23m, `experiments/baselines/interview/adaptive_table_convbandit_uniform_v2.json`
(+ `_per_user.npz`). Selectors from `src/baselines/conversational_bandit_selectors.py`; contract:
canonical Liang ML-25M, **policy-B pool**, NO-FLOOR, behavioural answers, all 10,000 test users,
consumer = certified tower i26c ep31 + concept channel cv3 ep08, **published key-term schedule**
`b(t)=5⌊log t⌋`, key-term representation `uniform_member_mean` (ConUCB Eq. 4's own definition).
MostPop prior .1626/.0262.

**Why this family and no other.** Every other baseline in the battery is item-only — Golbandi, DRE,
RBMF, IGCN, fMF, EDDI, ICF, Greedy SLIM, and all six of our static arms. ConUCB (Zhang WWW'20) →
ConLinUCB (Wang AAAI'23) → CLiSK/CLiME (Liu KDD'25) is the only published line that asks about a
**key-term (concept) or an item in one algorithm**, with no target item and no simulator, so it is
the only available opponent for our concept channel.

## The table

| selector | k=1 | k=2 | k=4 | k=8 | k=16 | items answered@16 |
|---|---|---|---|---|---|---|
| CLiME | .1606/.0400 | .1634/.0452 | .1776/.0549 | .1914/.0701 | **.2218/.0897** | 4.80 |
| ConLinUCB-BS | .1606/.0400 | .1634/.0452 | .1780/.0546 | .1942/.0683 | .2215/.0864 | 4.75 |
| ConLinUCB-MCR | .1606/.0400 | .1634/.0452 | .1743/.0574 | .1906/.0693 | .2190/.0865 | 4.77 |
| ConLinUCB-BS-static *(concepts only)* | .1644/.0374 | .1659/.0402 | .1724/.0468 | .1829/.0598 | .1891/.0700 | 0.00 |
| ConUCB (the WWW'20 original) | .1584/.0346 | .1591/.0364 | .1651/.0447 | .1805/.0574 | .1874/.0699 | 1.80 |
| *entropy₀ (strongest heuristic)* | .1635/.0433 | .1706/.0528 | .1831/.0636 | .2053/.0792 | *.2303/.0924* | 5.52 |
| *greedy items-only (strongest static)* | .1760/.0434 | .1907/.0581 | .2066/.0703 | .2247/.0776 | *.2449/.0967* | 4.20 |
| *greedy concepts-only* | .1783/.0478 | .1909/.0548 | .2147/.0775 | .2362/.1022 | *.2473/.1154* | 12.87 c |

## Four findings

**1. The Act-1 null holds against the published family.** The best conversational bandit (CLiME,
.2218/.0897) beats our own bandit reimplementations (GLM-UCB .2161/.0833, Thompson .2135/.0878) but
loses to entropy₀ by −.0085 full and to the strongest static by −.0231. The strongest published
adaptive line whose action space matches ours does not beat a good static order.

**2. ★ The concept channel now has a published opponent, and it loses — but the first number we
wrote was a cross-contract subtraction.** `ConLinUCB-BS-static` is the pure concept arm — zero item
answers, all sixteen questions are key-terms, its order is the barycentric spanner of the key-term
feature set, i.e. **the published method for choosing which concepts to ask**.

**CORRECTED 2026-08-07.** The originally reported *"+.0582 full / +.0454 tail, same contract"*
differenced two runs that use different **concept ANSWER conventions**: the convbandit harness
builds answers with `signed_values(apply_neg_cap=True)` — a population-wide negative-mass cap over
all 1,031 concepts — while `build_greedy_env`, the certified-era path **cv3 was trained under**,
uses `apply_neg_cap=False` plus `cap_negatives` over the *revealed* list only. The same 16 greedy
concepts score `.2473/.1154` under the certified-era convention and only `.2298/.0977` under the
convbandit one, so the two published figures were never ordered. Every static row here has now been
**re-scored in-harness** (controls reproduce at 0.0e+00 per user), giving two like-for-like
readings, **both separated at every budget on both axes (10/10 each)**:

| reading | full @k=16 | tail @k=16 |
|---|---|---|
| in the **convbandit** answer contract | **+.0407** [+.0381,+.0432] | **+.0276** [+.0246,+.0308] |
| under **certified-era** answers (BS-static rises to .1921/.0745) | **+.0553** [+.0524,+.0582] | **+.0408** [+.0378,+.0440] |

The claim survives on both readings — our concept bank beats the published concept selector by a
wide, separated margin — only the original number was wrong. This is the same class of error as the
candidate-mask contract: an apparent ordering manufactured by differencing across conventions.

**3. ConUCB, the 2020 original, trails its three item-asking successors, and answerability is why.**
1.80 answered items at k=16 against ~4.8 for them, and it trails by −.0316…−.0344 full at k=16,
**all separated**. The same answerability failure measured for Thompson and the latent-only
info-gain arm, now in a **published** method purpose-built for a mixed item/attribute action space
— a far stronger version of the point than our own reimplementations could support.
**CORRECTED:** *"worst arm in its own family"* is not supported — against `ConLinUCB-BS-static`,
which answers **zero** items, ConUCB is **ns at k=8 and k=16 on both axes**. The supportable claim
is that it trails the three arms that ask items.

**4. The short-budget rows say nothing about concepts, by construction.** Under the published
schedule `b(t)=5⌊log t⌋` only turns 3 and 8 are key-term turns, so k=1 and k=2 fire **zero**
concept questions and the three adaptive variants are numerically identical there
(.1606/.0400, .1634/.0452). Even k=16 spends only **2 of 16** turns on concepts. **The published
method is item-dominated by its own schedule** — which is itself worth reporting, and means this
family contests the concept channel only weakly except through the BS-static arm.

## Reproducibility notes

* The frozen iALS basis is now banked to disk (`load_or_fit_belief`), because `implicit`'s ALS is
  **not bit-reproducible across BLAS thread counts** despite `random_state=98765` — the cause of
  the ±.0026 bandit drift recorded earlier. This run and any resume share one basis.
* Caches are LRU-bounded with float32 stored widths; verified 0/1600 argmax disagreements versus
  float64 and 0/8000 differing actions end to end, against a smallest top-1/top-2 gap of 1.82e-02.
* Interviews checkpoint per turn to contract-signed sidecars; each scored cell is flushed
  atomically. A mid-interview resume reproduced all ten per-user NDCG vectors bit-identically.

## ★ C5 — every concept number this family produced is a FLOOR, not an estimate

The answer-convention split above is not only a comparison problem. `ConLinUCB-BS`, `-MCR`, `CLiME`
and `BS-static` all produced their concept answers under `apply_neg_cap=True`, **a convention cv3
was not trained on**, and the split costs a concept-only order up to **+.0177 tail**. So the
published values for this family understate it: they are floors. **Re-run the family with
`apply_neg_cap=False` before any of its concept numbers is quoted as an estimate** — including,
symmetrically, before the concept-channel margin above is quoted at its larger value.

## Owed

* Paired CIs are DONE → `docs/results/CONVBANDIT_PAIRED_CIS.md` (10,000 resamples, seed 98765;
  arm-vs-itself and the schedule self-test both exact).
* The other two key-term representations (`representative` per Hier-UCB, `whitened_centroid` =
  ours) — that trio is the concept-fold adjudication and is a separate experiment.
* CLiME under its own width rule (task #10): under a pure uncertainty trigger it asks **zero**
  concept questions, which is a measured negative worth reporting.

## ★ ConUCB is slow for ONE reason: Eq. 8 is VALUE-FREE but is cached as if it were not

**Symptom.** ConUCB spent ~14 h at 86% CPU inside a single k=8 turn while its k=1, 2, 4 cells took
seconds to minutes. Profiled 2026-08-10 (`--profile only`, no ConUCB cell re-run).

**The rule.** `ConUCBSelector._keyterm_scores` implements Eq. 8:

```
    s_k = || X_t M_t^-1 Mtil_{t-1}^-1 xtilde_k ||^2  /  ( 1 + xtilde_k^T Mtil_{t-1}^-1 xtilde_k )
```

**Cost per call, measured at the run's true shapes** (f=200, C=1031, pool=18,359; `G = Y^T Y` is
200×200 and built once per pool):

| step | complexity | ms |
|---|---|---|
| `Mti @ Xt.T` | O(f²C) | 1.6 |
| `solve(M, f×C rhs)` | O(f³ + f²C) | **44.0** |
| `Vk.T @ G` | O(C f²) | 1.7 |
| denominator — recomputes `Mti @ Xt.T` a second time | O(f²C) | 1.8 |
| **whole call** | **O(f³ + f²C)** | **102.6** |

ConLinUCB-MCR's key-term rule on the same shapes costs **3.95 ms — 26× less**.

**The actual defect is not the 103 ms, it is the CALL COUNT.** Every quantity Eq. 8 consumes is
value-free: `M = (1−λ)I + λ Y[cols]^T Y[cols]` depends only on WHICH items were answered, and
`Mtil^-1 = (λ̃I + Xk^T Xk)^-1` only on WHICH key-terms were. Confirmed empirically — holding the
answered SETS fixed and changing every answer VALUE gives `max|Δs| = 0.000e+00`. Yet
`_keyterm_scores` is **not cached at all**, and `select()` calls it once per **value-bearing**
interview state, every turn. MCR's equivalent is cached on `_wkey` — the value-FREE
(item-ids, concept-ids) key — which is precisely the collapse ConUCB never gets.

From the run's own banked telemetry (`adaptive_table_cfold_representative_PARTIAL.json`):

| budget | value-states (`_mu` misses) | asked-set states (`_width` misses) | ratio |
|---|---|---|---|
| k=4 (ConUCB) | 3,612 | **8** | **451×** |
| k=8 (MCR) | 17,599 | 145 | 121× |
| k=16 (MCR) | 77,370 | 13,550 | 5.7× |

The model checks out at the one ConUCB budget that completed: k=4 has ~3.6k distinct states over 4
turns; at 103 ms/call that predicts ~18 min against **15.85 min observed**. The `_width` cache
(the 18,359-item acquisition) is already keyed value-free and sat at **99.8% hit rate with 8
misses** — the item side is not the problem.

**Complexity, stated.** Total ConUCB key-term cost is

```
    O( SUM_t S_t  ·  (f^3 + f^2 C) )        S_t = distinct VALUE-bearing states at turn t
```

where `S_t` grows toward `n_users` as soon as a key-term turn fires, because concept answers are
continuous (signed implicit-watch V) and so nearly every user forms his own state. It should be

```
    O( SUM_t A_t  ·  (f^3 + f^2 C) )        A_t = distinct ASKED-SET states  (A_t << S_t)
```

**Honest bound on the attribution.** With `S_t` capped at 10,000 users over 8 turns, Eq. 8 alone
accounts for **at most 2.3 h** of the observed ~14 h. The k=4 agreement says the per-call cost and
the mechanism are right; what is NOT measured is ConUCB's own k=8 state count, because the cell was
killed and was deliberately not re-run. So: Eq. 8 is the dominant term and the only one with a
451× headroom, but the full 14 h is not yet fully attributed.

**The fix (not applied — diagnosis only).** In order of payoff:
1. Cache `_keyterm_scores` on `self._wkey(key)`, exactly as `_width_vec_kt` is cached. Provably
   sound, since Eq. 8 is value-free. At k=4 that is 8 calls instead of 3,612.
2. Skip Eq. 8 entirely on non-key-term turns. `select()` computes it unconditionally, but
   `keyterm_asks_per_turn` shows k=8 fires key-terms on only turns 3 and 8 — 6 of 8 turns compute
   a 103 ms quantity and discard it.
3. Hoist the duplicated `Mti @ Xt.T` (computed once for `Vk`, again for the denominator).
4. `_mtil`/`_mu` are LRU-capped at 4,096 against 17.6k–77.4k distinct states (13,503 evictions
   measured at k=8), so they thrash. Keying (1) value-free removes most of the pressure.
