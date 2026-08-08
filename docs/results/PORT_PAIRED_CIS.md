# Paired CIs on the DRE, DPE and IGCN port cells

Task #21, run 2026-08-08. Analysis `src/baselines/port_paired.py`; machine-readable output
`experiments/baselines/interview/port_paired.json`; log `…/port_paired.log`.
**Nothing is committed — the author reviews first.**

**Method.** Paired percentile bootstrap over the per-user **difference** vector, 10,000 resamples,
seed 98765. Never the unpaired `1.96*sqrt(2)*SE` form (`ARM_N_PHASE1.md` §4b: that form overstated
the true interval ~4.6× and forced a retraction). `boot_ci` is the `gtport_paired` implementation.

**Contract (all three runs).** Canonical Liang ML-25M, policy-B pool, NO-FLOOR, 10,000 test users
(9,281 with ≥1 tail target), consumer = certified tower i26c ep31. DPE ran under
`--concept_neg_cap 0` (certified-era answers); the item-only comparators are untouched by that flag.

## Gates — all three PASS

| gate | result |
|---|---|
| **Self-test** — `dre`, `dpe`, `igcn` each against **itself**, 5 budgets × 2 axes | Δ = 0.0000, CI [0.0000, 0.0000] on all 30. **PASS** |
| **Bootstrap equivalence** — the memory-chunked `boot_ci` vs the `gtport_paired` reference | **bit-identical** on both real probes. **PASS** |
| **Cross-run admission** — in-run `entropy0-static` vs the banked act1 vector, per user | max abs diff = **0.000e+00** on all 10 cells. **PASS** |

The chunked bootstrap exists only because this box had 1.3–3.0 GB free while a heavy twin ran, and
the reference form allocates a 10000×10000 float64 (~1.2 GB) per call. It is proven identical, not
assumed: `RandomState` fills `size=(n_boot, n)` row-major, so consecutive `(chunk, n)` blocks from
the same stream are the same resamples.

### Comparator discipline — what is exact and what is admitted

* **In-run (exact by construction).** The IGCN run carries `entropy0-static` and
  `fixedorder:greedy_items` *inside itself* (the new `fixedorder:` spec), so every IGCN gap, every
  budget-ladder contrast, every `dpe_concept`−`dpe` row and the DRE consumer comparison are exact.
* **Cross-run (admitted).** DRE and DPE ran without comparators. Their gaps against entropy₀ /
  greedy items-only difference a DRE/DPE vector against the **IGCN run's** comparator. This is
  admitted on the control above: the item-only static order reproduces **bit-for-bit (0.0e+00 per
  user)** across independent harness runs on this contract, and the user list is a fixed on-disk
  split with no sampling. Every such row is stamped `cross-run (admitted)` in the JSON.
  **This is weaker than an in-run comparator** — the honest fix is to re-run DRE and DPE with
  `fixedorder:greedy_items,entropy0-static` in the same invocation, which is now cheap.

---

## 1 ★ DPE-ILD degrades with budget — CONFIRMED, separated below zero

`dpe` (the ILD/diversity arm) is the only arm in the battery that gets **worse** with more
questions: .1627 → .1621 → .1613 → .1609 → .1603 full across k=1…16.

**Paired, per user, `dpe` at k versus `dpe` at k=1:**

| contrast | full Δ | CI | tail Δ | CI |
|---|--:|---|--:|---|
| k2 − k1 | −0.0006 | [−0.0011, −0.0000] **SEP** | +0.0003 | [−0.0004, +0.0010] ns |
| k4 − k1 | −0.0014 | [−0.0020, −0.0006] **SEP** | −0.0001 | [−0.0009, +0.0008] ns |
| k8 − k1 | −0.0018 | [−0.0025, −0.0009] **SEP** | −0.0008 | [−0.0017, +0.0001] ns |
| **k16 − k1** | **−0.0024** | **[−0.0032, −0.0015] SEP** | **−0.0024** | **[−0.0034, −0.0013] SEP** |
| k16 − k8 | −0.0006 | [−0.0011, −0.0001] **SEP** | −0.0015 | [−0.0022, −0.0008] **SEP** |

**The claim survives.** Fifteen extra questions make the recommendation *separably worse* on both
axes. On full the degradation separates from k=2 onward; on tail it only separates at k=16.

**The control says this is unique to `dpe`.** Same contrast, every other arm — all eight separated
**positive**:

| arm | full k16−k1 | tail k16−k1 |
|---|--:|--:|
| `dpe` | **−0.0024 SEP** | **−0.0024 SEP** |
| dpe_hellinger | +0.0566 SEP | +0.0527 SEP |
| dpe_oracle | +0.1399 SEP | +0.1246 SEP |
| dpe_concept | +0.0398 SEP | +0.0452 SEP |
| dre | +0.0527 SEP | +0.0502 SEP |
| igcn | +0.0554 SEP | +0.0505 SEP |
| igcn_i1 | +0.0502 SEP | +0.0481 SEP |
| entropy₀ | +0.0668 SEP | +0.0491 SEP |
| greedy items-only | +0.0689 SEP | +0.0532 SEP |

### The mechanism is corroborated, and it is stronger than expected

`answered` for `dpe`: **0.3176 / 0.3179 / 0.3185 / 0.3190 / 0.3205** at k=1/2/4/8/16.

Questions 2 through 16 — fifteen of them — bought **0.003 answers between them**. The arm's entire
answer yield is its popularity-seeded *first* pick; after that its diversity criterion maximises
distance from the profile and it selects items essentially **nobody has seen**. Compare
`dpe_hellinger` (0.32 → 4.46), entropy₀ (0.44 → 5.52), greedy items (0.30 → 4.20).

Note this makes the *degradation itself* a small effect (−.0024) on a near-flat curve: the arm is not
actively destroying signal so much as adding 15 no-answer turns that slightly perturb the fold-in.
**Report it as "the only arm that gets separably worse with budget", not as a large effect.**

---

## 2 ★ The oracle ceiling — separated, large, at every budget

**`dpe_oracle` is a CEILING, never a baseline.** It reads the user's own held-out ratings at
selection time to choose what to ask. It is an upper bound on what perfect question selection could
buy on this ruler, and it must never be listed alongside real methods without that label.

| budget | oracle − greedy items (full) | oracle − entropy₀ (full) | oracle − greedy items (tail) | oracle − entropy₀ (tail) |
|---|--:|--:|--:|--:|
| k=1 | +0.0133 [+.0106,+.0160] | +0.0258 [+.0233,+.0283] | +0.0246 [+.0220,+.0273] | +0.0247 [+.0221,+.0274] |
| k=2 | +0.0287 [+.0255,+.0318] | +0.0488 [+.0456,+.0519] | +0.0385 [+.0353,+.0417] | +0.0437 [+.0404,+.0469] |
| k=4 | +0.0477 [+.0443,+.0511] | +0.0713 [+.0678,+.0748] | +0.0578 [+.0542,+.0614] | +0.0645 [+.0609,+.0682] |
| k=8 | +0.0683 [+.0645,+.0720] | +0.0877 [+.0839,+.0914] | +0.0822 [+.0781,+.0862] | +0.0805 [+.0765,+.0844] |
| **k=16** | **+0.0843 [+.0804,+.0881]** | **+0.0989 [+.0950,+.1027]** | **+0.0960 [+.0917,+.1002]** | **+0.1002 [+.0960,+.1045]** |

**Every cell separated, all 20.** The headroom argument holds with an interval: at k=16 an oracle
selector is worth **+.084 full / +.096 tail** over the best real method (greedy items-only,
.2449/.0967), and **+.099 / +.100** over entropy₀. The oracle reaches .3292/.1927 — i.e. it recovers
a large share of the full-profile gap using 16 questions.

This is the strongest available counter to "selection doesn't matter": with the recommender, the
answer model, the pool and the budget all held fixed, **changing only which questions are asked**
moves NDCG by ~.08–.10, and the interval excludes zero by a wide margin. The caveat is symmetric and
must be stated: the oracle is not achievable, so this bounds the prize, not the realisable gain.
`answered` for the oracle runs 0.84 → 12.39, against greedy's 0.30 → 4.20, so a large part of the
ceiling is simply **asking things the user can actually answer**.

---

## 3 DRE — the peer-reviewed learned-static bar

### 3a vs entropy₀ and greedy items-only (cross-run, admitted)

| budget | DRE − entropy₀ (full) | DRE − greedy items (full) | DRE − entropy₀ (tail) | DRE − greedy items (tail) |
|---|--:|--:|--:|--:|
| k=1 | **+0.0023** [+.0005,+.0041] SEP | −0.0102 [−.0123,−.0081] SEP | −0.0037 [−.0060,−.0015] SEP | −0.0038 [−.0057,−.0020] SEP |
| k=2 | −0.0013 [−.0036,+.0010] **ns** | −0.0214 [−.0240,−.0188] SEP | −0.0064 [−.0090,−.0038] SEP | −0.0116 [−.0141,−.0092] SEP |
| k=4 | −0.0068 [−.0097,−.0039] SEP | −0.0303 [−.0334,−.0272] SEP | −0.0050 [−.0080,−.0019] SEP | −0.0117 [−.0147,−.0088] SEP |
| k=8 | +0.0021 [−.0011,+.0052] **ns** | −0.0173 [−.0201,−.0143] SEP | −0.0003 [−.0034,+.0027] **ns** | +0.0013 [−.0015,+.0043] **ns** |
| **k=16** | **−0.0118 [−.0153,−.0083] SEP** | **−0.0264 [−.0296,−.0232] SEP** | **−0.0027 [−.0058,+.0006] ns** | −0.0069 [−.0100,−.0037] SEP |

**The headline holds on full at k=16**: −.0118 [−.0153,−.0083] vs entropy₀ and −.0264
[−.0296,−.0232] vs greedy items — both separated. The peer-reviewed learned-static method is
separably beaten by a 2008 heuristic and by our greedy order.

**But it does not hold uniformly, and two cells cut against the doc's framing:**
* **On TAIL at k=16, DRE−entropy₀ is ns** (−0.0027, [−.0058,+.0006]). The .0898 vs .0924 tail gap in
  the source table is *not* separated.
* **At k=1 DRE BEATS entropy₀ on full, separated** (+0.0023). At k=2 and k=8 the full gap is ns.
  DRE loses to entropy₀ at k=16, not "everywhere".

DRE also has the lowest answer yield of any real arm — `answered` 0.21 → 2.88 at k=16, against
entropy₀'s 5.52 — which is the plainest reading of why a jointly-trained seed set underperforms a
1-D entropy order here.

### 3b DRE's two consumers against each other (IN-RUN, exact) — **the .002 agreement claim fails**

`ours` (certified tower) minus `dre` (DRE's own retrained decoder), same seed sets, same users:

| budget | full Δ | CI | tail Δ | CI |
|---|--:|---|--:|---|
| k=1 | −0.0019 | [−0.0036, −0.0002] **SEP** | +0.0031 | [+0.0012, +0.0049] **SEP** |
| k=2 | −0.0016 | [−0.0032, −0.0001] **SEP** | −0.0023 | [−0.0042, −0.0005] **SEP** |
| **k=4** | **−0.0068** | **[−0.0085, −0.0051] SEP** | **−0.0037** | **[−0.0059, −0.0016] SEP** |
| k=8 | +0.0003 | [−0.0017, +0.0023] ns | +0.0010 | [−0.0010, +0.0029] ns |
| k=16 | +0.0003 | [−0.0020, +0.0025] ns | +0.0003 | [−0.0020, +0.0026] ns |

The two consumers **agree (ns) only at k=8 and k=16**. At k=1, 2 and 4 they are **separated on both
axes**, and the k=4 full gap is **.0068 — 3.4× the claimed .002 bound**. The correct statement is:
*the two consumers converge as the budget grows, and are indistinguishable from k=8; below that the
consumer still reorders things.* The instrument-dependence the doc says is absent is present at
short budgets — which is exactly the regime the chapter cares about.

---

## 4 dpe_hellinger vs entropy₀ on TAIL at k=16 — **ns, a real dead heat**

.0925 vs .0924 → **Δ = +0.0001, CI [−0.0026, +0.0029], not separated.** The point-estimate dead heat
is a genuine tie, not a photo-finish that an interval would resolve. Full context:

| budget | Hellinger − entropy₀ (full) | Hellinger − entropy₀ (tail) | Hellinger − greedy items (tail) |
|---|--:|--:|--:|
| k=1 | −0.0008 ns | −0.0035 SEP | −0.0036 SEP |
| k=2 | −0.0025 SEP | −0.0048 SEP | −0.0101 SEP |
| k=4 | −0.0014 ns | −0.0026 SEP | −0.0094 SEP |
| k=8 | −0.0051 SEP | −0.0027 SEP | −0.0011 **ns** |
| **k=16** | **−0.0111 SEP** | **+0.0001 ns** | −0.0041 SEP |

So `dpe_hellinger` **ties entropy₀ on tail at k=16 while losing separably on full** (−.0111). Worth
saying explicitly: the two axes disagree about this arm, and a full-only table would report a loss
where the tail reports a tie.

---

## 5 IGCN — all in-run, exact by construction

**★ Every IGCN cell with its `igcn_personalised_rate`.** The published method's warm-up threshold is
expressed in **ratings** (switch after 5), and our answer rate is ~0.3/question, so in our regime
`igcn` is a **static order at k ≤ 4 and ~98% static at k = 8**. Read every IGCN claim in that light.

| budget | igcn full/tail | **pers. rate** | igcn_i1 full/tail | **pers. rate** | igcn_page15 | **pers. rate** |
|---|---|--:|---|--:|---|--:|
| k=1 | .1685/.0407 | **0.0%** | .1685/.0407 | **0.0%** | .1685/.0407 | **0.0%** |
| k=2 | .1800/.0569 | **0.0%** | .1816/.0584 | 12.9% | .1800/.0569 | **0.0%** |
| k=4 | .1944/.0701 | **0.0%** | .1960/.0690 | 29.7% | .1944/.0701 | **0.0%** |
| k=8 | .2031/.0747 | **1.9%** | .2017/.0760 | 46.8% | .2028/.0749 | **0.0%** |
| k=16 | .2239/.0912 | **13.0%** | .2187/.0889 | 61.9% | .2217/.0918 | 2.3% |

The smoke's 1.9% at k=8 reproduces exactly on all 10,000 users (1,529 / 80,000 asks).
`igcn_degenerate_falls = 0` everywhere.

### vs entropy₀ and greedy items-only

| budget | igcn − entropy₀ (full) | igcn − greedy (full) | igcn − entropy₀ (tail) | igcn − greedy (tail) |
|---|--:|--:|--:|--:|
| k=1 | **+0.0050** [+.0031,+.0070] SEP | −0.0075 SEP | −0.0026 SEP | −0.0027 SEP |
| k=2 | **+0.0095** [+.0071,+.0119] SEP | −0.0106 SEP | **+0.0040** SEP | −0.0012 ns |
| k=4 | **+0.0113** [+.0083,+.0141] SEP | −0.0123 SEP | **+0.0065** SEP | −0.0003 ns |
| k=8 | −0.0022 [−.0055,+.0011] ns | −0.0216 SEP | −0.0045 SEP | −0.0028 SEP |
| k=16 | **−0.0065** [−.0096,−.0033] SEP | −0.0211 SEP | −0.0012 ns | −0.0055 SEP |

**IGCN beats entropy₀ at short budgets and loses at long ones, both separated.** The crossover sits
between k=4 and k=8. The load-bearing caveat: **at k ≤ 4 the personalised rate is exactly 0**, so the
early win is *not* adaptivity — it is IGCN's global cluster-information order beating entropy₀'s
order as a **static** ranking. IGCN never beats greedy items-only on full at any budget (all SEP
negative); on tail it is ns at k=2 and k=4.

### What adaptivity is actually worth here (`igcn_i1` / `igcn_page15` vs `igcn`)

| budget | i1 − igcn (full) | i1 − igcn (tail) | page15 − igcn (full) | page15 − igcn (tail) |
|---|--:|--:|--:|--:|
| k=1 | 0.0000 (identical) | 0.0000 | 0.0000 (identical) | 0.0000 |
| k=2 | +0.0015 SEP | +0.0016 SEP | 0.0000 (identical) | 0.0000 |
| k=4 | +0.0016 ns | −0.0011 ns | 0.0000 (identical) | 0.0000 |
| k=8 | −0.0013 ns | +0.0012 ns | −0.0003 ns | +0.0001 ns |
| **k=16** | **−0.0051 SEP** | **−0.0023 SEP** | **−0.0021 SEP** | +0.0006 ns |

`igcn_page15` is **numerically identical** to `igcn` at k ≤ 4 (its re-rank period cannot fire), which
is a clean internal control that the harness is doing what the spec says. The deviation row
`igcn_i1`, which raises the personalised rate from 13% to 62% at k=16, is **separably WORSE** than
the published form at k=16 on both axes. **Running the personalized step more does not help here** —
so the near-static behaviour of `igcn` in our regime is not costing it anything.

---

## 6 Secondary: `dpe` and `dpe_concept`

| budget | dpe − entropy₀ (full) | dpe − entropy₀ (tail) | dpe_concept − dpe (full) | dpe_concept − dpe (tail) |
|---|--:|--:|--:|--:|
| k=1 | −0.0008 ns | −0.0035 SEP | −0.0006 ns | −0.0054 SEP |
| k=2 | −0.0085 SEP | −0.0128 SEP | +0.0040 SEP | +0.0005 ns |
| k=4 | −0.0218 SEP | −0.0239 SEP | +0.0116 SEP | +0.0102 SEP |
| k=8 | −0.0444 SEP | −0.0402 SEP | +0.0223 SEP | +0.0225 SEP |
| k=16 | −0.0700 SEP | −0.0550 SEP | **+0.0416 SEP** | **+0.0421 SEP** |

The concept-channel version of the same DPE criterion beats the item version by a widening,
separated margin (+.042/+.042 at k=16) — consistent with the recorded finding that concepts
decouple discrimination from answerability (`dpe_concept` answers via concept tokens and never burns
a turn on an unseen item, where `dpe` answers 0.32 questions in 16).

---

## ★ Verdict — sentences in the source docs the intervals do NOT support

Four sentences, all in `DRE_PORT_ON_OUR_RULER.md`; `SNAPIGCN_NATIVE.md` survives intact (its port
section correctly claimed nothing, and its smoke-measured 1.9% personalised rate at k=8 reproduced
exactly at full scale). **(1)** *"**The two consumers agree to within .002 at every budget.**"* — false
on the point estimates (the k=4 full gap is .0068) and separated at k=1, k=2 and k=4 on **both** axes;
agreement (ns) holds only at k=8 and k=16. **(2)** The inference built on it — *"A learned static seed
set appears to be a property of the data rather than of the recommender that learned it."* — must be
restricted to k ≥ 8, because below that the consumer does reorder things, which is the short-budget
regime the chapter is about. **(3)** The k=16 table caption *"one contract, all separated where CIs
exist"* — CIs now exist and **DRE − entropy₀ on TAIL is ns** (−.0027, [−.0058,+.0006]), so the
.0898-vs-.0924 tail column in that table is not a separated loss. **(4)** The section heading *"Where
it lands — **every published method loses to entropy₀**"* — IGCN, a published method now measured on
this ruler, **separably beats entropy₀ on full at k=1, 2 and 4** (+.0050/+.0095/+.0113) and only loses
from k=8; the claim is true at k=16 on full, not in general. The load-bearing DRE headline itself
survives unscathed: −.0118 [−.0153,−.0083] vs entropy₀ and −.0264 [−.0296,−.0232] vs greedy
items-only, both separated on full at k=16. The DPE claims in the brief also survive — the ILD arm's
degradation is separated below zero on both axes, and the oracle ceiling is separated at all 20 cells
— but "DPE-ILD degrades with budget" should be reported as **the only arm that gets separably worse**
rather than as a large effect, since the magnitude is only −.0024 and the arm is effectively flat
(0.32 answers at k=1, 0.32 at k=16).

## Owed / next

* **Re-run DRE and DPE with `fixedorder:greedy_items,entropy0-static` in the same invocation.** Every
  cross-run row above would then be exact by construction rather than admitted on a control. Cheap
  now that the `fixedorder:` spec exists.
* DRE's convergence gate is still unevaluable for these artifacts (`DRE_PORT_ON_OUR_RULER.md` §⚠) —
  intervals do not touch that.
