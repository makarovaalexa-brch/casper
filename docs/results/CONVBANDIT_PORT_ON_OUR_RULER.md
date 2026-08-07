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

**2. ★ The concept channel now has a published opponent, and the gap is large.**
`ConLinUCB-BS-static` is the pure concept arm — zero item answers, all sixteen questions are
key-terms, its order is the barycentric spanner of the key-term feature set, i.e. **the published
method for choosing which concepts to ask**. It reaches **.1891/.0700** where our greedy
concepts-only reaches **.2473/.1154**: **+.0582 full and +.0454 tail**, static-versus-static inside
the same channel, same budget, same contract. This is the comparison the battery was missing —
until now the concept result had no rival because the literature asks about items.

**3. ConUCB, the 2020 original, is the worst arm in its own family, and answerability is why.**
1.80 answered items at k=16 against ~4.8 for its successors, and it trails them by ~.034 full. The
same answerability failure measured for Thompson and the latent-only info-gain arm, now in a
**published** method purpose-built for a mixed item/attribute action space — a far stronger version
of the point than our own reimplementations could support.

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

## Owed

* **Paired CIs on every gap here** — `--per_user` vectors are on disk. The load-bearing number is
  the +.0582/+.0454 concept-channel gap.
* The other two key-term representations (`representative` per Hier-UCB, `whitened_centroid` =
  ours) — that trio is the concept-fold adjudication and is a separate experiment.
* CLiME under its own width rule (task #10): under a pure uncertainty trigger it asks **zero**
  concept questions, which is a measured negative worth reporting.
