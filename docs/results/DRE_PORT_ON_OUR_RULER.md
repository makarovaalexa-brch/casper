# DRE on our ruler: the peer-reviewed learned-static bar loses to a 2008 heuristic

Run 2026-08-07/08. Training `scripts/baselines/dre_train_ours.py --budgets 1,2,4,8,16` (~10.5 h,
one artifact per budget, train users only). Cells `src/baselines/adaptive_harness.py --selectors dre
--only ours,dre --label dre_port_v1`. Native snap PASSED
(`docs/results/SNAP3_DRE_NATIVE.md`: ML-1M k=50, .5593 ± .0088 vs published .5688, ≈1.1 seed-sd).
Contract: canonical Liang ML-25M, policy-B pool, NO-FLOOR, behavioural answers, all 10,000 test
users. MostPop prior .1626/.0262.

DRE (Kweon et al., WWW'20) learns a **fixed length-k seed set** end-to-end with Gumbel-Softmax, so
in our vocabulary it is a **static selector** — the peer-reviewed *learned*-static bar, and the row
that makes "we compared against the strongest published static method" a true sentence rather than
a comparison against our own greedy order.

## The cells

| consumer | k=1 | k=2 | k=4 | k=8 | k=16 |
|---|---|---|---|---|---|
| **(X) certified tower** | .1658/.0396 | .1693/.0465 | .1763/.0586 | .2074/.0789 | **.2185/.0898** |
| **(P) DRE's own retrained decoder** | .1677/.0365 | .1709/.0488 | .1831/.0623 | .2071/.0779 | .2183/.0895 |

**CORRECTED 2026-08-08 by paired CIs.** An earlier version read *"the two consumers agree to within
.002 at every budget"* and inferred that a learned static seed set is *"a property of the data
rather than of the recommender that learned it"*. **Both are false as stated.** The two consumers
are **separated at k=1, 2 and 4 on both axes** (k=4 full gap .0068); they converge only from k=8.
The supportable claim is narrower: *DRE's seed set transfers between its own decoder and our tower
at long budgets, and does not at short ones.*

## Where it lands — every published method lands below the optimised static order

**CORRECTED: "every published method loses to entropy₀" is FALSE.** IGCN separably beats entropy₀
at k=1/2/4 (+.0050/+.0095/+.0113 full). Its `igcn_personalised_rate` is **exactly 0.0% at k≤4**,
so that early win is *not* adaptivity — it is IGCN's global information-gain **static** order
outperforming entropy₀'s. What survives is the weaker and still-useful claim below.

k=16 on the certified tower, one contract. **NB the DRE−entropy₀ gap is separated on FULL
(−.0118 [−.0153,−.0083]) but NOT on tail** — the axes disagree, and the tail column below is a
point estimate only:

| selector | full | tail | kind |
|---|---|---|---|
| greedy items-only (ours) | **.2449** | **.0967** | optimised static |
| entropy₀ (Rashid 2008) | .2303 | .0924 | published heuristic |
| CLiME (KDD'25) | .2218 | .0897 | published adaptive |
| **DRE (WWW'20)** | **.2185** | **.0898** | **published learned static** |
| GLM-UCB | .2161 | .0833 | acquisition family |
| Thompson | .2135 | .0878 | acquisition family |
| Golbandi tree (WSDM'11) | .2117 | .0844 | published adaptive |
| latent-only info-gain | .1917 | .0749 | acquisition family |

**The strongest published learned-static method is beaten on full accuracy by a 2008 entropy
heuristic** (−.0118 [−.0153,−.0083], separated) **and by our greedy order** (−.0264
[−.0296,−.0232], separated). The claim that survives across the whole battery is the greedy one:
*published elicitation methods of every kind — learned static, classical adaptive, modern
conversational bandit — land below a fixed order chosen by greedy search on the recommender they
are scored with.* Against entropy₀ specifically the picture is mixed (IGCN beats it early), which
is why the comparator that matters is greedy, not the heuristic.

## The duplicate-seed defect does NOT fire at our budgets

Its native snap found DRE selects k seeds as independent categoricals with **no duplicate guard**,
and one of three native runs at k=50 duplicated an item (costing ≈.015 NDCG@10). **At k ∈
{1,2,4,8,16} the trained sets have zero duplicates at every budget** (distinct = k throughout). The
criticism is real at large budgets and inapplicable at ours — recorded rather than asserted either
way.

Seed sets (train users only): k=1 `[142]`; k=2 `[695, 111]`; k=4 `[549, 142, 619, 304]`;
k=8 `[89, 137, 174, 350, 143, 10, 36, 480]`; k=16 `[136, 530, 549, 77, 87, 25, 1174, 131, 350, 1,
142, 668, 98, 619, 10, 480]`. Note the sets are **not nested** — the k=2 set is not the first two of
k=16 — which is the paper's own point: the columns are trained jointly, so one artifact per budget.

## ⚠ The convergence gate is UNEVALUABLE for these artifacts

We required proof that the learned selection distribution committed before any cell was reported.
**It cannot be checked after the fact for this run**: the checkpoints persist the resulting seed set
and the retrained decoder, but not the selection logits. Distinct argmaxes do **not** prove
commitment — a diffuse distribution can still yield k different argmaxes.

Indirect evidence is good but not proof: the temperature annealed to its endpoint T=0.1 (hard
selection by construction), joint loss plateaued (.003364 → .003362 over the last five epochs), the
retrain phase on the hard seeds converged (17352 → 16178), and the **native snap passed**, which is
the strongest single argument that the implementation is faithful. The trainer now logs *and
persists* per-slot peak probability, entropy against uniform, and a HARD/MIXED/DIFFUSE verdict
(committed 2026-08-08), so any future artifact is auditable. **Until one budget is retrained under
the instrumented trainer, these cells carry the caveat that peakedness was not verified.**

## Owed

* Paired CIs against entropy₀ and greedy items-only (`--per_user` vectors are on disk; re-score the
  comparator in-harness rather than differencing means).
* One budget retrained under the instrumented trainer to discharge the convergence gate.
* Optional: the ML-20M native cell (published NDCG@10 .6077) if a reviewer wants the larger dataset.
