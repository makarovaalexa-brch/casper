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
