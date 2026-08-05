# Migration to candidate-pool POLICY B (2026-08-05)

> Ruling: `external_literature/findings/candidate_masking_protocol.md` §3.
> Diagnostic that set the acceptance numbers: `docs/results/MASK_CONTRACT_DIAGNOSTIC.md`.
> Reference implementation: `src/baselines/arm_n.py::build_pool`.
> **Nothing here is committed.** Code changed, verification run, deltas recorded; the author reviews first.

## The rule, stated once

> We never remove an item from the candidate list because it was asked about. The pool is fixed per
> user from their rated history before the interview begins — every rated fold-in-side item, likes
> **and** dislikes — so no arm can shrink or enlarge its own haystack, and no method is either
> rewarded or punished for asking well. The held-out target matrix is never modified.

Policy A (liked fold-in only, Liang verbatim) is **retained** for the full-profile certification
table and for the greedy run's certification-bridge gate, and is retained as a **second printed
column** in G9 because a Paper A number depends on it.

## Code changed

| file | change |
|---|---|
| `src/instrument/mask_contract.py` | **NEW.** `policy_b_pool()` (cohort analogue of `arm_n.build_pool`, with the pool/target leak check inside) + the two invariant asserts. |
| `src/instrument/run_battery_phaseA.py` | `ctx_pool_b(ctx)` — the pool, built once and cached on the ctx. `ndcg10_from_scores(..., contract=(tag, rows))` and `eval_tokens(..., contract_tag=)` — the invariant is enforced inside the only funnel an instrument arm can turn scores into a number through. `gate_g9` now reports **two pool columns**. |
| `src/instrument/strategy_ladder.py` | `eval_budget`: asked-item mask **deleted**, `te[j,i] = 0` target deletion **deleted**; scores against the fixed policy-B pool and the untouched `ctx.va_te`. |
| `src/instrument/strategy_channel_suite.py` | `genv_eval`: `ctx.va_tr[ch] + ans[cj]` → `E["pool"][ch]`, built once per environment in `build_greedy_env`. New `genv_intercept_policyA` — takes no sequence — carries gate (a)'s certification bridge on policy A. |
| `src/instrument/concept_cold_check.py` | Both interview builders carry the user's policy-B pool in the pack; `score` masks it via the new `apply_pool_mask` (contract + leak assert). Was `sc[r, e[6]] = -inf` with no fold-in mask at all. |
| `src/instrument/concept_vs_item_paired.py` | `per_user` was a copy of the same masking line; migrated to `apply_pool_mask` so the CIs and the table they qualify can never sit on different pools. `--out`. |
| `src/instrument/bank_sweep.py` | `--out` so the policy-B sweep lands on a new filename and the pre-B artifact stays readable. |

### The two asserts (ruling §3, item 3)

Enforced in the stronger form that is actually true under policy B: **for a given cohort slice the
mask and the target set must be bit-identical across every arm AND every budget.**

* (a) mask identical → `MASK CONTRACT (a) VIOLATED`
* (b) target matrix never modified → `TARGET CONTRACT (b) VIOLATED`

They live inside `ndcg10_from_scores` (csr paths) and `apply_pool_mask` (the per-user cold-check
path), i.e. inside the scoring funnel, not at the call sites. Negative control run: injecting one
extra masked item for one user is rejected with `MASK CONTRACT (a) VIOLATED [smoke/user=1]`.

## Verification

### Canary — `src/baselines/arm_n_canary.py`, ALL GATES PASS (10.4 m)

| model | policy A (full/tail) | policy B (full/tail) | Δ full | Δ tail |
|---|---|---|---|---|
| MostPop | 0.1345 / 0.0226 | 0.1626 / 0.0262 | +0.0280 | +0.0037 |
| EASE | 0.3476 / 0.2441 | 0.4135 / 0.2941 | +0.0658 | +0.0500 |

Both arm-A rows snap to the recorded canonical values exactly; both rise; the pair does not reorder
(EASE 0.4135 ≫ MostPop 0.1626). The rise is uneven (ratio 2.35) — flagged INSPECT by the gate, and
expected: EASE ranks the user's own rated-and-disliked items much higher than popularity does, so it
has more junk to gain from removing.

### Regression — `adaptive_harness.py --regression --budgets 1,2,4,8,16`

PASS: 30 (arm, budget) cells byte-identical through the sequential path, including the named
acceptance cell entropy0 / k=8.

### G9 column A is byte-unchanged — proved on the ep4 snapshot

The first G9 re-run went out on the script's **default** snapshot `t2i25_EP4_SNAP.pt` rather than the
certified tower, and column A came back `0.14927 / 0.18299 / 0.22296 / 0.27103` (pop, q2/4/8/16) —
which is **exactly the g9.json that commit 4f8574e replaced** (`0.14926706…`, `0.18298954…`). So the
published-column code path reproduces its own historical output to every recorded digit; the policy-B
column is genuinely additive. (The committed artifact, `0.1574 → 0.2609`, is the certified-tower
re-measurement from 4f8574e — that file was restored from backup and re-run on
`models/i26c_ep31/t2i26c_ep31.pt`; numbers below.)

Policy-A → B on the ep4 snapshot, pop bank full@10:
`q2 .1493→.1746 · q4 .1830→.2094 · q8 .2230→.2458 · q16 .2710→.2836` (random bank
`.1408→.1676 … .1447→.1703`). The A→B lift **shrinks with budget** (+.025 at q2, +.013 at q16),
which is the expected shape: policy A's extra masking of answered items grows with the budget and
partially closes the gap that the unmasked dislikes open at q=0.

### Acceptance gate — `genv_eval` must reproduce the diagnostic

Certified pair (tower `t2i26c_ep31.pt` + concept `concept_cv3_ep08.pt`), 10,000 TEST users,
no truncation. Script: scratchpad `gate_policyB.py`.

| check | expected | measured | verdict |
|---|---|---|---|
| pool object == `arm_n.build_pool` | identical | 0 differing cells, nnz 1,411,761 both, 141.2 items/user | **PASS** |
| m=0 policy-A bridge | 0.13139 / 0.02343 (`greedy_static_i26c.json` `intercept_eval`) | 0.13139 / 0.02343 | **PASS** |
| m=0 policy-B intercept | .1586 / .0282 | 0.15855 / 0.02819 | **PASS** |
| greedy-combined @ q16 | .2556 / .1098 | 0.2556 / 0.1098 | **PASS** |
| entropy0 top-16 vs published ladder | 4 dp | **5 dp, all five budgets, |Δ| < 0.000005** | **PASS** |

entropy0 top-16 under the migrated `genv_eval`, against `interview_table_ladder_ep31.json`:

| q | measured | published ladder | Δ |
|---|---|---|---|
| 1 | 0.16348 / 0.04328 | 0.16348 / 0.04328 | +0.00000 / −0.00000 |
| 2 | 0.17056 / 0.05282 | 0.17056 / 0.05282 | +0.00000 / +0.00000 |
| 4 | 0.18310 / 0.06358 | 0.18310 / 0.06358 | +0.00000 / +0.00000 |
| 8 | 0.20526 / 0.07922 | 0.20526 / 0.07922 | +0.00000 / +0.00000 |
| 16 | 0.23031 / 0.09241 | 0.23031 / 0.09241 | −0.00000 / +0.00000 |

The mask contract was live throughout: 20 cohort slices × 22 scored sequences, zero violations
(`contract={'genv:GATE(test):test': 20}`).

## Numbers that move

### 1. The greedy best-static table — `docs/results/GREEDY_BEST_STATIC_I26C.md` (test cohort, 10k users)

Same tower, same concept channel, same frozen sequences, **only the pool changes**.

| arm | q | policy A (recorded) | **policy B** |
|---|---|---|---|
| intercept m=0 | – | .1314 / .0234 | **.1586 / .0282** |
| items-only | 1 | .1471 / .0351 | **.1760 / .0434** |
| | 2 | .1563 / .0441 | **.1907 / .0581** |
| | 4 | .1719 / .0542 | **.2066 / .0703** |
| | 8 | .1922 / .0598 | **.2247 / .0776** |
| | 16 | .2104 / .0761 | **.2449 / .0967** |
| concepts-only | 1 | .1487 / .0398 | **.1783 / .0478** |
| | 2 | .1590 / .0457 | **.1909 / .0548** |
| | 4 | .1738 / .0620 | **.2147 / .0775** |
| | 8 | .1898 / .0815 | **.2362 / .1022** |
| | 16 | .1997 / .0921 | **.2473 / .1154** |
| combined | 1 | .1487 / .0398 | **.1783 / .0478** |
| | 2 | .1590 / .0457 | **.1909 / .0548** |
| | 4 | .1738 / .0588 | **.2135 / .0755** |
| | 8 | .1918 / .0763 | **.2335 / .0962** |
| | 16 | .2115 / .0880 | **.2556 / .1098** |

**★ One qualitative claim in that doc flips.** It currently reads *"items own full at long budgets
(.2104 vs .1997 at q16)"*. Under policy B **concepts-only wins full at q16 too: .2473 vs .2449**, and
it already led on tail by a wide margin (.1154 vs .0967). The item channel's long-budget advantage on
full was, at least in part, the free lift policy-A+C-answered handed it — masking the ~4 answered,
just-revealed items an item arm collects per user and a concept arm structurally cannot. That
sentence and finding 2 of `GREEDY_BEST_STATIC_I26C.md` must be re-cut before the doc is cited again.

Combined still leads full (.2556) and concepts-only still leads tail (.1154), so the "objective
steers the composition" reading survives; the *channel* verdict at q16 does not.

### 2. G9 — both columns (certified tower `t2i26c_ep31.pt`, val cohort)

**Column A is bit-reproduced.** Every published cell comes back identical, including the Paper A
R3‡ row: `0.1574 → 0.2609` full, `0.0505 → 0.1062` tail, rise **+0.10354 CI [0.10003, 0.10707]**
(chapter prints `+0.1035`, CI `[0.1000, 0.1071]`). Nothing in Paper A moves.

| bank | q | A full/tail (published) | **B full/tail** |
|---|---|---|---|
| pop | 2 | .1574 / .0505 | **.1817 / .0623** |
| | 4 | .1879 / .0657 | **.2127 / .0785** |
| | 8 | .2206 / .0821 | **.2422 / .0969** |
| | 16 | .2609 / .1062 | **.2785 / .1251** |
| rand | 2 | .1435 / .0474 | **.1721 / .0561** |
| | 4 | .1476 / .0523 | **.1769 / .0623** |
| | 8 | .1483 / .0545 | **.1772 / .0650** |
| | 16 | .1486 / .0548 | **.1768 / .0651** |

Gate verdicts: `PASS = True` (column A, unchanged) and `PASS_policyB = True`. Under B the fixed-bank
rise is **+0.0968 CI [0.0933, 0.1004]** (vs +0.1035 under A) and pop still beats random at every
budget, separated (+.0097 / +.0358 / +.0650 / +.1018). **The gate's conclusion is pool-invariant**;
only the numbers move, and the rise shrinks slightly because policy B lifts the q=2 baseline more
than the q=16 endpoint.

### 3. The concept-vs-item bank sweep — `docs/results/CONCEPT_VS_ITEM_BANK_SWEEP.md`

Re-run: `bank_sweep.py --ckpt models/i26c_ep31/concept_cv3_ep08.pt --out
experiments/instrument/bank_sweep_policyB.json` (certified pair, 1,500 held-out train users, no
truncation). **The pre-B artifact `bank_sweep.json` is untouched.** Arms-matched m=0 canary asserted
and passing for all four banks.

**The floor moves: `.1277 / .0267` → `.2154 / .0354`.** Every absolute number in that doc is on the
old pool and must be re-read from the new file.

**★ The prediction in the ruling (§Q3) is confirmed: the concept margin WIDENS under policy B, on
both axes, at every budget, under the load-bearing `polar` bank.** The old sweep gave the item arm a
free lift the concept arm structurally could not receive — masking the ~4 answered, just-revealed
items per user — so the published result was conservative.

`polar` bank, cv3 ep8, concept − item (point estimates):

| m | old Δ full / Δ tail (C-answered) | **new Δ full / Δ tail (policy B)** |
|---|---|---|
| 1 | +.0045 / +.0109 | **+.0099 / +.0106** |
| 2 | +.0075 / +.0151 | **+.0166 / +.0171** |
| 4 | +.0103 / +.0192 | **+.0178 / +.0254** |
| 8 | +.0189 / +.0310 | **+.0376 / +.0407** |
| 16 | +.0146 / +.0265 | **+.0358 / +.0385** |

Absolute `polar` cells (conc / item, full/tail): m1 .2240/.0548 vs .2141/.0442 · m2 .2305/.0631 vs
.2139/.0461 · m4 .2358/.0709 vs .2180/.0455 · m8 .2562/.0910 vs .2186/.0503 · m16 .2632/.0974 vs
.2274/.0589.

**Paired CIs on the same pool** (`concept_vs_item_paired.py --out
experiments/instrument/concept_vs_item_paired_policyB.json`, 10k-resample, n=1500 full / 1448 tail;
pre-B artifact untouched). `polar`: **all ten cells separated positive**, as before but with roughly
doubled margins:

| m | full Δ [CI] | tail Δ [CI] |
|---|---|---|
| 1 | +.0099 [+.0054, +.0146] | +.0106 [+.0047, +.0167] |
| 2 | +.0166 [+.0115, +.0217] | +.0171 [+.0104, +.0240] |
| 4 | +.0178 [+.0118, +.0238] | +.0254 [+.0181, +.0327] |
| 8 | **+.0376** [+.0308, +.0446] | **+.0407** [+.0325, +.0487] |
| 16 | +.0358 [+.0287, +.0432] | +.0385 [+.0310, +.0462] |

Off `polar`, with intervals: `pop` tail is now **ns at every budget** (was: m4 separated positive);
`pop` full separates negative at m1, m8, m16. `entropy` keeps its shape exactly — separated positive
on both axes at m1–2, ns at m4, separated negative by m8. `expinfo` is **ns everywhere up to m4** and
separated negative on both axes at m8/m16. So claim (1) of that doc ("under a pure-discrimination
bank concepts beat items on both axes at every budget") is **strengthened**; claim (2) ("off that
bank items win full decisively, tail a wash or a loss") stands.

The other three banks keep their qualitative shape: items win full decisively off `polar`
(`pop` m16 Δfull −.0307, `entropy` −.0468, `expinfo` −.0340), and `entropy`'s tail advantage still
flips from positive at m1–4 to separated-negative by m8. So **the reading of that doc survives the
pool change; the numbers do not.**

### 4. Nothing else in `docs/` moves

`strategy_ladder`'s outputs are cited nowhere (grep-verified) and were stale on two independent
grounds already; the code is fixed but **not re-run** — a citable ladder needs a fresh run on the
certified tower, which is a separate decision, not a patch. Paper A's full-profile certification
table is policy A by design and is untouched. Paper A's R3‡ interview curve is G9 column A, bit-
reproduced above. The graded-vs-likes curve (`premium_vs_k_t2final.json`) is pure policy A and was
not in scope.
