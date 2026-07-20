# T6 — Paper D FAIR comparison: access-controlled open-recall vs probe baselines (2026-07-05)

**Review charge (T6).** Paper D's open-recall arm ("name a movie you love") folds a full 64-d known-item token
`(Q[j], resid[x][j])` per question, while the probe baselines (PEBOL, ConTS, entropy) only ever fold a *scalar*
geometric answer over arms they must **spend turns discovering**. The reviewer's point: open-recall consumes
volunteered known-item tokens the baselines **never receive** — an unequal information budget, not an apples-to-apples
elicitation comparison.

**Fix (access-controlled arms).** Give **every** arm the SAME `m` matched volunteered known-item tokens (realistic
`popweight` recall, identical per user), then each arm spends the remaining `p = T-m` turns with its own policy:
open-recall names `p` more favourites; PEBOL/ConTS/etc probe `p` concepts. Equal total budget `T=8` for all arms.
Sweep `m` to show the transition. Realistic answerer = headline; oracle-recall rows **labeled as BOUNDS** (not
realizable). Bootstrap 95% CIs (2000 resamples over pooled seed×user per-user scores) on every arm.

- Ruler (unchanged): ML-1M, frozen V1 set-encoder `s = popb + Ql·u`, NDCG@10 full+tail, `te[300:]`, seeds {1,2,3}.
- Code: `ACCESS` block in `scripts/paper2/continuous_actor.py` (additive; OPENQ/LITBASE untouched).
- Repro: `ACCESS=1 EVALSEEDS=1,2,3 MSWEEP=0,2,4,6 HEADM=4 BOOT=2000 python scripts/paper2/continuous_actor.py`
- Numbers: `data/movielens/.cache/paperD/t6_paperD_fair.json`

## Access-controlled arms — NDCG@10 (full / tail) with bootstrap 95% CI

### m = 0 (OLD, UNFAIR setup: baselines get 0 volunteered tokens, 8 probes)
| arm | FULL [95% CI] | TAIL [95% CI] |
|---|---|---|
| **OPEN-recall (ours, realistic)** | **0.3872** [0.3690,0.4060] | **0.1745** [0.1615,0.1888] |
| PEBOL+profile [PRIMARY] | 0.3690 [0.3504,0.3873] | 0.1682 [0.1558,0.1811] |
| ConTS+profile | 0.3510 [0.3333,0.3695] | 0.1465 [0.1355,0.1598] |
| concept-pop+profile | 0.3119 [0.2948,0.3294] | 0.1057 [0.0957,0.1167] |
| random-concept+profile | 0.3201 [0.3029,0.3380] | 0.1176 [0.1068,0.1288] |
| profile-only (control) | 0.3087 [0.2921,0.3260] | 0.0826 [0.0744,0.0911] |
| [BOUND] oracle-align | 0.4126 [0.3926,0.4320] | 0.2110 [0.1961,0.2266] |
| [BOUND] oracle-distinct | 0.4047 [0.3851,0.4237] | 0.2106 [0.1958,0.2261] |

### m = 4 (HEADLINE, EQUAL BUDGET: baselines get 4 matched volunteered tokens + 4 probes; open names 8)
| arm | FULL [95% CI] | TAIL [95% CI] |
|---|---|---|
| **OPEN-recall (ours, realistic)** | 0.3926 [0.3740,0.4115] | 0.1776 [0.1635,0.1922] |
| **PEBOL+profile [PRIMARY]** | 0.3899 [0.3714,0.4088] | 0.1767 [0.1630,0.1903] |
| ConTS+profile | 0.3796 [0.3610,0.3980] | 0.1649 [0.1515,0.1789] |
| concept-pop+profile | 0.3678 [0.3502,0.3860] | 0.1584 [0.1455,0.1718] |
| random-concept+profile | 0.3668 [0.3491,0.3854] | 0.1548 [0.1423,0.1678] |
| profile-only (control) | 0.3778 [0.3591,0.3963] | 0.1576 [0.1445,0.1711] |
| [BOUND] oracle-align (not realizable) | 0.4126 [0.3926,0.4320] | 0.2110 [0.1961,0.2266] |
| [BOUND] oracle-distinct (not realizable) | 0.4047 [0.3851,0.4237] | 0.2106 [0.1958,0.2261] |

### Transition (open-realistic vs PEBOL+profile) as baselines get more matched volunteered tokens
| m (volunteered) | OPEN full | PEBOL+prof full | Δfull | OPEN tail | PEBOL+prof tail | Δtail |
|---|---|---|---|---|---|---|
| 0 (unfair) | 0.3872 | 0.3690 | **+0.0182** | 0.1745 | 0.1682 | +0.0063 |
| 2 | 0.3926 | 0.3801 | +0.0125 | 0.1746 | 0.1703 | +0.0043 |
| **4 (equal budget)** | 0.3926 | 0.3899 | **+0.0028** | 0.1776 | 0.1767 | **+0.0009** |
| 6 | 0.3877 | 0.3831 | +0.0046 | 0.1695 | 0.1730 | **−0.0035** |

## Verdict

**Open-recall's bandwidth advantage does NOT survive once the baselines receive equal known-item access.**
At m=0 (the old setup) open-recall leads PEBOL+profile by +0.018 full / +0.006 tail — but that gap is an
**information-access artifact**: the baselines were simply denied the volunteered known-item tokens. As soon as
PEBOL is given the same matched tokens, the gap collapses monotonically. At the equal-budget headline (m=4),
open-recall (0.3926/0.1776) and PEBOL+profile (0.3899/0.1767) are a **statistical tie** — d=+0.0028 full,
+0.0009 tail, with heavily overlapping 95% CIs (CI-disjoint = False on both axes). By m=6, PEBOL+profile even
edges ahead on tail (−0.0035). The `profile-only` control confirms the mechanism: at m=4 the shared volunteered
tokens alone already deliver 0.3778/0.1576, and open-recall's 4 extra names (+0.015/+0.020) buy essentially the
same increment as PEBOL's 4 concept probes (+0.012/+0.019).

**Honest tail statement.** Under the realistic (`popweight`, availability-proxy) answerer that is the headline,
open-recall tail = 0.1776, which is **−0.033 below** the oracle-recall BOUND (oracle-distinct 0.2106; oracle-align
full 0.4126). The oracle rows are upper bounds from an answerer that names its single most-informative /
most-distinctive favourite — **not realizable** — and are labeled as such. The realistic tail loss vs the bound is
the honest cost of popularity-biased human recall.

**Implication for the paper.** The load-bearing claim can no longer be "open-recall beats the probe baselines." The
defensible, fair claim is narrower: (i) volunteered known-item tokens are a large lever *for every method* — arming
PEBOL with them lifts it from 0.369/0.168 to 0.390/0.177; (ii) at equal information budget, *how* you spend the
remaining turns (naming vs probing) is a wash. Open-recall's value is **operational** (it is the natural, low-friction
way to elicit those tokens), not a raw NDCG edge over an equally-armed probe policy.
