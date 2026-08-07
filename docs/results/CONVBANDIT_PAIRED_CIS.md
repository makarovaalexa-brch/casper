# Paired CIs on the conversational-bandit port — and the contract mix inside its headline gap

Task #17. Paired **percentile bootstrap over the per-user difference vector** (10,000 resamples, seed 98765); users are resampled and the per-user delta recomputed. The unpaired `1.96*sqrt(2)*SE` form is banned — it overstated intervals ~4.6x and forced a retraction (`ARM_N_PHASE1.md` §4b).

Contract: canonical Liang ML-25M, policy-B pool, NO-FLOOR, 10,000 test users, tower i26c ep31 + concept channel cv3 ep08, key-term schedule b(t)=5*floor(log t), repr uniform_member_mean. n_full = 10,000, n_tail = 9,281 (the users with ≥1 tail target).

Sources: `experiments/baselines/interview/adaptive_table_convbandit_uniform_v2_per_user.npz` (the run), `convbandit_rescore_per_user.npz` and `convbandit_rescore_nocap_per_user.npz` (the fixed static orders re-scored in-harness by `src/baselines/convbandit_greedy_rescore.py`), analysis `src/baselines/convbandit_paired.py` → `convbandit_paired.json`.

## Controls, all passed before any gap was read

* **Self-test** — an arm minus itself: delta 0.0000, CI [0.0000, 0.0000] at every budget and axis. `self_test_pass = True`.
* **Schedule self-test** — under `b(t)=5⌊log t⌋` no key-term turn fires before t=3, so `conlinucb_mcr`, `conlinucb_bs` and `clime` must be numerically identical at k=1 and k=2. All three pairwise deltas are **exactly 0.0000 with CI [0.0000, 0.0000]** on both axes at both budgets. `schedule_self_test_pass = True`.
* **Cross-run admission, item side** — the re-scored `entropy0` top-16 order reproduces the banked ladder vector **per user**: max abs diff 0.0e+00. Its means come back `.1635/.0433 · .1706/.0528 · .1831/.0636 · .2053/.0792 · .2303/.0924`, identical to the published ladder. Greedy items-only likewise reproduces `.1760/.0434 · .1907/.0581 · .2066/.0703 · .2247/.0776 · .2449/.0967`.
* **Cross-run admission, concept side** — `ConLinUCB-BS-static` is a *fixed* key-term order in this run (asserted: identical acts for all 10,000 users), so it can be re-scored like any static arm. Re-scoring it reproduces the banked convbandit vectors **per user**: max abs diff 0.0e+00. That is what licenses differencing a re-scored *concept* sequence at all.

No comparison in this file crosses runs on a mean anchor. Every static row was re-scored inside this harness (precedent: task #16, `gtport_greedy_rescore.py`).

## ★ The finding that comes before the intervals: two concept-answer conventions

The greedy concepts-only row could not be reproduced in this harness at first. The cause is a real difference in the **answer**, not in the question:

* `conversational_bandit_selectors.build_concept_stack` builds concept answers with `signed_values(...)` at its default `apply_neg_cap=True` — the **population-wide** negative-mass cap, spread across all 1,031 concepts at once.
* `strategy_channel_suite.build_greedy_env` builds them with `apply_neg_cap=False` and applies `cap_negatives` over the **revealed** list instead — which its own comment calls "the certified convention", and is what the concept channel cv3 was trained on.

Re-scoring the greedy concepts-only order under the second convention reproduces the published row **exactly** — `.1783/.0478 · .1909/.0548 · .2147/.0775 · .2362/.1022 · .2473/.1154` — while under the convbandit run's own convention the same 16 questions reach only `.2298/.0977` at k=16. So the published `.2473/.1154` and the published `.1891/.0700` were measured under **different information contracts** and were never ordered (`surprise-means-debug-the-harness`).

| fixed order | answers | k=1 | k=2 | k=4 | k=8 | k=16 |
|---|---|---|---|---|---|---|
| greedy concepts-only | in-contract (population cap) | 0.1773/0.0471 | 0.1845/0.0510 | 0.2048/0.0691 | 0.2189/0.0863 | 0.2298/0.0977 |
| greedy concepts-only | certified-era (revealed cap) | 0.1783/0.0478 | 0.1909/0.0548 | 0.2147/0.0775 | 0.2362/0.1022 | 0.2473/0.1154 |
| ConLinUCB-BS-static | in-contract (population cap) | 0.1644/0.0374 | 0.1659/0.0402 | 0.1724/0.0468 | 0.1829/0.0598 | 0.1891/0.0700 |
| ConLinUCB-BS-static | certified-era (revealed cap) | 0.1643/0.0379 | 0.1663/0.0412 | 0.1732/0.0494 | 0.1822/0.0618 | 0.1921/0.0745 |

The convention costs the *greedy* arm far more than it costs the *bandit* arm (§C5 below): the bandit's spanner concepts carry little negative mass, so shrinking negatives barely touches it. Both readings are given below; both are like-for-like.

## Point estimates used (all from the same per-user vectors as the CIs)

| arm | k=1 | k=2 | k=4 | k=8 | k=16 |
|---|---|---|---|---|---|
| CLiME | 0.1606/0.0400 | 0.1634/0.0452 | 0.1776/0.0549 | 0.1914/0.0701 | 0.2218/0.0897 |
| ConLinUCB-BS | 0.1606/0.0400 | 0.1634/0.0452 | 0.1780/0.0546 | 0.1942/0.0683 | 0.2215/0.0864 |
| ConLinUCB-MCR | 0.1606/0.0400 | 0.1634/0.0452 | 0.1743/0.0574 | 0.1906/0.0693 | 0.2190/0.0865 |
| ConLinUCB-BS-static *(concepts only)* | 0.1644/0.0374 | 0.1659/0.0402 | 0.1724/0.0468 | 0.1829/0.0598 | 0.1891/0.0700 |
| ConUCB | 0.1584/0.0346 | 0.1591/0.0364 | 0.1651/0.0447 | 0.1805/0.0574 | 0.1874/0.0699 |
| *entropy₀ (re-scored)* | 0.1635/0.0433 | 0.1706/0.0528 | 0.1831/0.0636 | 0.2053/0.0792 | 0.2303/0.0924 |
| *greedy items-only (re-scored)* | 0.1760/0.0434 | 0.1907/0.0581 | 0.2066/0.0703 | 0.2247/0.0776 | 0.2449/0.0967 |
| *greedy concepts-only (re-scored, in-contract)* | 0.1773/0.0471 | 0.1845/0.0510 | 0.2048/0.0691 | 0.2189/0.0863 | 0.2298/0.0977 |

## The intervals

### C1 ★ the load-bearing one, IN CONTRACT

Both arms under the convbandit run's own concept-answer convention. Static-versus-static, same channel, same budget, same users.

| comparison | axis | k=1 | k=2 | k=4 | k=8 | k=16 |
|---|---|---|---|---|---|---|
| greedy concepts-only − ConLinUCB-BS-static | full | +0.0128 [+0.0111, +0.0146] **SEP** | +0.0187 [+0.0167, +0.0206] **SEP** | +0.0324 [+0.0299, +0.0349] **SEP** | +0.0360 [+0.0335, +0.0385] **SEP** | +0.0407 [+0.0381, +0.0432] **SEP** |
| greedy concepts-only − ConLinUCB-BS-static | tail | +0.0096 [+0.0077, +0.0116] **SEP** | +0.0108 [+0.0087, +0.0129] **SEP** | +0.0223 [+0.0195, +0.0252] **SEP** | +0.0265 [+0.0236, +0.0295] **SEP** | +0.0276 [+0.0246, +0.0308] **SEP** |

### C1b the same two orders under the CERTIFIED-ERA answers

The convention cv3 was trained on and the published greedy row was measured under. Both arms re-scored, so this is still like-for-like.

| comparison | axis | k=1 | k=2 | k=4 | k=8 | k=16 |
|---|---|---|---|---|---|---|
| greedy concepts-only − ConLinUCB-BS-static | full | +0.0140 [+0.0121, +0.0160] **SEP** | +0.0246 [+0.0223, +0.0269] **SEP** | +0.0414 [+0.0387, +0.0441] **SEP** | +0.0540 [+0.0512, +0.0568] **SEP** | +0.0553 [+0.0524, +0.0582] **SEP** |
| greedy concepts-only − ConLinUCB-BS-static | tail | +0.0099 [+0.0078, +0.0121] **SEP** | +0.0136 [+0.0112, +0.0160] **SEP** | +0.0282 [+0.0252, +0.0312] **SEP** | +0.0403 [+0.0372, +0.0435] **SEP** | +0.0408 [+0.0378, +0.0440] **SEP** |

### C2 the Act-1 null: the best conversational bandit against the strongest statics

Static rows re-scored in-harness; concepts row is the in-contract one.

| comparison | axis | k=1 | k=2 | k=4 | k=8 | k=16 |
|---|---|---|---|---|---|---|
| CLiME − entropy₀ | full | -0.0029 [-0.0046, -0.0012] **SEP** | -0.0072 [-0.0092, -0.0052] **SEP** | -0.0055 [-0.0076, -0.0033] **SEP** | -0.0138 [-0.0163, -0.0114] **SEP** | -0.0085 [-0.0111, -0.0060] **SEP** |
| CLiME − entropy₀ | tail | -0.0033 [-0.0054, -0.0013] **SEP** | -0.0076 [-0.0097, -0.0054] **SEP** | -0.0087 [-0.0110, -0.0065] **SEP** | -0.0092 [-0.0117, -0.0066] **SEP** | -0.0028 [-0.0052, -0.0003] **SEP** |
| CLiME − greedy items-only | full | -0.0154 [-0.0174, -0.0133] **SEP** | -0.0273 [-0.0298, -0.0247] **SEP** | -0.0290 [-0.0318, -0.0262] **SEP** | -0.0332 [-0.0361, -0.0305] **SEP** | -0.0231 [-0.0262, -0.0201] **SEP** |
| CLiME − greedy items-only | tail | -0.0035 [-0.0056, -0.0014] **SEP** | -0.0128 [-0.0152, -0.0105] **SEP** | -0.0155 [-0.0184, -0.0125] **SEP** | -0.0075 [-0.0104, -0.0046] **SEP** | -0.0070 [-0.0102, -0.0038] **SEP** |
| CLiME − greedy concepts-only | full | -0.0167 [-0.0188, -0.0145] **SEP** | -0.0211 [-0.0238, -0.0185] **SEP** | -0.0272 [-0.0299, -0.0245] **SEP** | -0.0275 [-0.0302, -0.0248] **SEP** | -0.0080 [-0.0111, -0.0049] **SEP** |
| CLiME − greedy concepts-only | tail | -0.0071 [-0.0095, -0.0048] **SEP** | -0.0057 [-0.0085, -0.0030] **SEP** | -0.0142 [-0.0175, -0.0110] **SEP** | -0.0163 [-0.0197, -0.0129] **SEP** | -0.0080 [-0.0118, -0.0044] **SEP** |

### C3 ConUCB against its own successors (the answerability claim)

Answered items at k=16: ConUCB **1.80**, MCR 4.77, BS 4.75, CLiME 4.80, BS-static 0.00 (it asks only key-terms).

| comparison | axis | k=1 | k=2 | k=4 | k=8 | k=16 |
|---|---|---|---|---|---|---|
| ConUCB − ConLinUCB-MCR | full | -0.0022 [-0.0036, -0.0007] **SEP** | -0.0043 [-0.0064, -0.0021] **SEP** | -0.0092 [-0.0117, -0.0067] **SEP** | -0.0100 [-0.0126, -0.0075] **SEP** | -0.0316 [-0.0348, -0.0284] **SEP** |
| ConUCB − ConLinUCB-MCR | tail | -0.0054 [-0.0071, -0.0036] **SEP** | -0.0088 [-0.0113, -0.0065] **SEP** | -0.0126 [-0.0154, -0.0099] **SEP** | -0.0120 [-0.0149, -0.0091] **SEP** | -0.0166 [-0.0200, -0.0133] **SEP** |
| ConUCB − ConLinUCB-BS | full | -0.0022 [-0.0036, -0.0007] **SEP** | -0.0043 [-0.0064, -0.0021] **SEP** | -0.0129 [-0.0154, -0.0104] **SEP** | -0.0137 [-0.0163, -0.0111] **SEP** | -0.0341 [-0.0374, -0.0308] **SEP** |
| ConUCB − ConLinUCB-BS | tail | -0.0054 [-0.0071, -0.0036] **SEP** | -0.0088 [-0.0113, -0.0065] **SEP** | -0.0099 [-0.0126, -0.0072] **SEP** | -0.0110 [-0.0139, -0.0082] **SEP** | -0.0165 [-0.0198, -0.0133] **SEP** |
| ConUCB − CLiME | full | -0.0022 [-0.0036, -0.0007] **SEP** | -0.0043 [-0.0064, -0.0021] **SEP** | -0.0126 [-0.0149, -0.0102] **SEP** | -0.0109 [-0.0133, -0.0085] **SEP** | -0.0344 [-0.0376, -0.0313] **SEP** |
| ConUCB − CLiME | tail | -0.0054 [-0.0071, -0.0036] **SEP** | -0.0088 [-0.0113, -0.0065] **SEP** | -0.0101 [-0.0127, -0.0076] **SEP** | -0.0127 [-0.0154, -0.0101] **SEP** | -0.0198 [-0.0230, -0.0165] **SEP** |
| ConUCB − ConLinUCB-BS-static | full | -0.0060 [-0.0074, -0.0046] **SEP** | -0.0068 [-0.0083, -0.0052] **SEP** | -0.0073 [-0.0090, -0.0056] **SEP** | -0.0024 [-0.0047, +0.0000] ns | -0.0017 [-0.0045, +0.0011] ns |
| ConUCB − ConLinUCB-BS-static | tail | -0.0028 [-0.0045, -0.0013] **SEP** | -0.0038 [-0.0057, -0.0019] **SEP** | -0.0020 [-0.0040, -0.0000] **SEP** | -0.0025 [-0.0051, +0.0003] ns | -0.0001 [-0.0032, +0.0029] ns |

### C4 the rest of the family, pairwise

k=1 and k=2 are the schedule self-test cells: exactly 0.0000, CI [0.0000, 0.0000].

| comparison | axis | k=1 | k=2 | k=4 | k=8 | k=16 |
|---|---|---|---|---|---|---|
| CLiME − ConLinUCB-BS | full | +0.0000 [+0.0000, +0.0000] ns | +0.0000 [+0.0000, +0.0000] ns | -0.0004 [-0.0017, +0.0010] ns | -0.0028 [-0.0044, -0.0011] **SEP** | +0.0003 [-0.0015, +0.0020] ns |
| CLiME − ConLinUCB-BS | tail | +0.0000 [+0.0000, +0.0000] ns | +0.0000 [+0.0000, +0.0000] ns | +0.0003 [-0.0012, +0.0016] ns | +0.0017 [-0.0001, +0.0036] ns | +0.0032 [+0.0013, +0.0051] **SEP** |
| CLiME − ConLinUCB-MCR | full | +0.0000 [+0.0000, +0.0000] ns | +0.0000 [+0.0000, +0.0000] ns | +0.0034 [+0.0016, +0.0052] **SEP** | +0.0009 [-0.0006, +0.0024] ns | +0.0028 [+0.0013, +0.0042] **SEP** |
| CLiME − ConLinUCB-MCR | tail | +0.0000 [+0.0000, +0.0000] ns | +0.0000 [+0.0000, +0.0000] ns | -0.0025 [-0.0044, -0.0006] **SEP** | +0.0007 [-0.0008, +0.0023] ns | +0.0031 [+0.0017, +0.0046] **SEP** |
| CLiME − ConLinUCB-BS-static | full | -0.0038 [-0.0054, -0.0022] **SEP** | -0.0025 [-0.0046, -0.0004] **SEP** | +0.0052 [+0.0031, +0.0074] **SEP** | +0.0085 [+0.0059, +0.0112] **SEP** | +0.0327 [+0.0295, +0.0358] **SEP** |
| CLiME − ConLinUCB-BS-static | tail | +0.0025 [+0.0005, +0.0046] **SEP** | +0.0051 [+0.0027, +0.0075] **SEP** | +0.0081 [+0.0056, +0.0106] **SEP** | +0.0103 [+0.0073, +0.0133] **SEP** | +0.0196 [+0.0163, +0.0230] **SEP** |
| ConLinUCB-BS − ConLinUCB-MCR | full | +0.0000 [+0.0000, +0.0000] ns | +0.0000 [+0.0000, +0.0000] ns | +0.0037 [+0.0019, +0.0055] **SEP** | +0.0037 [+0.0020, +0.0053] **SEP** | +0.0025 [+0.0008, +0.0043] **SEP** |
| ConLinUCB-BS − ConLinUCB-MCR | tail | +0.0000 [+0.0000, +0.0000] ns | +0.0000 [+0.0000, +0.0000] ns | -0.0028 [-0.0048, -0.0008] **SEP** | -0.0010 [-0.0028, +0.0009] ns | -0.0001 [-0.0020, +0.0018] ns |
| ConLinUCB-BS − ConLinUCB-BS-static | full | -0.0038 [-0.0054, -0.0022] **SEP** | -0.0025 [-0.0046, -0.0004] **SEP** | +0.0056 [+0.0034, +0.0078] **SEP** | +0.0114 [+0.0088, +0.0139] **SEP** | +0.0324 [+0.0294, +0.0356] **SEP** |
| ConLinUCB-BS − ConLinUCB-BS-static | tail | +0.0025 [+0.0005, +0.0046] **SEP** | +0.0051 [+0.0027, +0.0075] **SEP** | +0.0078 [+0.0054, +0.0103] **SEP** | +0.0085 [+0.0057, +0.0115] **SEP** | +0.0164 [+0.0131, +0.0196] **SEP** |
| ConLinUCB-MCR − ConLinUCB-BS-static | full | -0.0038 [-0.0054, -0.0022] **SEP** | -0.0025 [-0.0046, -0.0004] **SEP** | +0.0019 [-0.0005, +0.0043] ns | +0.0077 [+0.0049, +0.0105] **SEP** | +0.0299 [+0.0267, +0.0331] **SEP** |
| ConLinUCB-MCR − ConLinUCB-BS-static | tail | +0.0025 [+0.0005, +0.0046] **SEP** | +0.0051 [+0.0027, +0.0075] **SEP** | +0.0106 [+0.0078, +0.0135] **SEP** | +0.0095 [+0.0064, +0.0127] **SEP** | +0.0165 [+0.0132, +0.0199] **SEP** |

### C5 what the answer convention alone is worth

Same fixed order, same consumer, same users — only the concept-answer construction changes.

| comparison | axis | k=1 | k=2 | k=4 | k=8 | k=16 |
|---|---|---|---|---|---|---|
| greedy concepts-only: certified − in-contract | full | +0.0011 [+0.0002, +0.0019] **SEP** | +0.0064 [+0.0052, +0.0076] **SEP** | +0.0098 [+0.0084, +0.0112] **SEP** | +0.0173 [+0.0157, +0.0189] **SEP** | +0.0176 [+0.0161, +0.0190] **SEP** |
| greedy concepts-only: certified − in-contract | tail | +0.0007 [-0.0001, +0.0016] ns | +0.0038 [+0.0025, +0.0051] **SEP** | +0.0084 [+0.0069, +0.0100] **SEP** | +0.0159 [+0.0139, +0.0178] **SEP** | +0.0177 [+0.0158, +0.0196] **SEP** |
| ConLinUCB-BS-static: certified − in-contract | full | -0.0001 [-0.0007, +0.0004] ns | +0.0004 [-0.0002, +0.0011] ns | +0.0008 [-0.0001, +0.0018] ns | -0.0007 [-0.0018, +0.0004] ns | +0.0030 [+0.0016, +0.0043] **SEP** |
| ConLinUCB-BS-static: certified − in-contract | tail | +0.0005 [-0.0001, +0.0009] ns | +0.0010 [+0.0004, +0.0016] **SEP** | +0.0026 [+0.0015, +0.0036] **SEP** | +0.0020 [+0.0008, +0.0032] **SEP** | +0.0045 [+0.0031, +0.0058] **SEP** |


## Verdict — what `CONVBANDIT_PORT_ON_OUR_RULER.md` may and may not say now

**Survives, with the interval attached.** Finding 1's two numbers are both real: CLiME loses to
entropy₀ by **−0.0085 [−0.0111, −0.0060]** full at k=16 and to greedy items-only by **−0.0231
[−0.0262, −0.0201]**, both SEP, and CLiME is behind *every* static row on *both* axes at *every*
budget — **thirty of thirty** separated comparisons in C2. The Act-1 null against the
published family is the strongest-supported claim in the doc. Finding 4's schedule argument is
verified exactly as stated: the three adaptive variants are bit-identical at k=1 and k=2 (delta
0.0000, CI [0.0000, 0.0000]). Finding 3's direction holds — ConUCB trails MCR/BS/CLiME by −0.0316 to
−0.0344 full at k=16, all SEP.

**Does NOT survive as written — one sentence, and it is the load-bearing one.**

> *"It reaches **.1891/.0700** where our greedy concepts-only reaches **.2473/.1154**: **+.0582 full
> and +.0454 tail**, static-versus-static inside the same channel, same budget, same contract."*

The words "same contract" are false. `.1891/.0700` was measured with concept answers built under
`signed_values(apply_neg_cap=True)` — the population-wide negative cap in
`build_concept_stack` — and `.2473/.1154` was measured under `apply_neg_cap=False` plus a
revealed-set cap, the convention `concept_cv3_ep08` was trained on. Re-scoring the greedy order in
the convbandit harness reproduces `.2298/.0977`, not `.2473/.1154`; re-scoring it under the
certified-era convention reproduces `.2473/.1154` to four decimals. `+.0582/+.0454` is a subtraction
across two information contracts and is not a measured gap. The two honest readings, both
like-for-like and both SEP at every budget, are:

* **in the convbandit contract** (both arms population-capped): **+0.0407 [+0.0381, +0.0432]** full
  and **+0.0276 [+0.0246, +0.0308]** tail at k=16;
* **under the certified-era answers** (both arms re-scored, ConLinUCB-BS-static rises to
  `.1921/.0745`): **+0.0553 [+0.0524, +0.0582]** full and **+0.0408 [+0.0378, +0.0440]** tail.

So the *claim* — the published method for choosing which concepts to ask is beaten decisively by our
greedy concept order, static-versus-static, at every budget from k=1 on both axes — **survives, and
separates everywhere** (10/10 comparisons under either convention). Only the number is wrong: by
0.0175 full / 0.0178 tail against the in-contract reading (about 3.4 and 2.9 interval widths) and by
0.0029 / 0.0046 against the certified-era reading (inside half an interval width). Write one of the
two bracketed pairs, name the convention, and drop "same contract".

**Two smaller corrections.**

1. Finding 3 attributes ConUCB's deficit to answerability. The intervals do not license that
   attribution as stated: against `ConLinUCB-BS-static`, which answers **zero** items, ConUCB is
   **ns** at k=8 (−0.0024 [−0.0047, +0.0000]) and k=16 (−0.0017 [−0.0045, +0.0011]) on full and
   ns on tail at both. ConUCB is separated from the arms that *answer more items* and tied with the
   arm that answers *none*; that is consistent with the answerability story but it also means
   "worst arm in its own family" is only true against the three item-capable successors. Say
   "trails its three item-asking successors" and drop "worst in its family".
2. The doc's own "Owed" line — *"Paired CIs on every gap here … the load-bearing number is the
   +.0582/+.0454 concept-channel gap"* — is discharged, but the answer-convention split it
   uncovered is now the more consequential open item: **every concept number the conversational
   bandits produced is under a convention the certified concept channel was not trained on**, and
   §C5 prices that at up to +0.0177 tail on a concept-only order. `ConLinUCB-BS`, `-MCR` and `CLiME`
   should be re-run under `apply_neg_cap=False` before their concept-turn rows are cited; their
   published values are floors, not estimates.
