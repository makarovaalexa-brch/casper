# Diagnostic: greedy IS optimal — the "entropy0 beats greedy" reading was two rulers

Run 2026-08-05 (author question: *"why is greedy below entropy, greedy is optimal by definition, no?"*).
Script + logs in scratchpad (`decisive.py`, `decisive.json`); nothing re-run, nothing committed to
`experiments/` — this is a re-scoring of two FIXED sequences under two masks.

## The two masks (quoted from code)

* **Greedy env** — `src/instrument/strategy_channel_suite.py:543`, in `genv_eval`:
  `mask = ctx.va_tr[ch] + ans[cj]` where `ctx.va_tr` = the binary >3.5 fold-in = **62.6 items/user**,
  and `ans` = only the asked items the user actually rated.
* **Interview table** — `src/baselines/run_interview_table.py:285`, `mask_X=D["pool"]` from
  `arm_n.build_pool` = `binarize(g_te_tr) ∪ binarize(te_tr)` = every rated fold-in item, likes AND
  dislikes = **141.2 items/user**.

The greedy mask is a strict subset of the arm-N pool (asserted). Metric code is shared
(`ndcg10_from_scores` → `M.NDCG_binary_at_k_batch`, identical head-mask logic), and the ladder run
had `floored: 0` on every cell, so neither the metric nor the popularity floor is a confound.

## Cause B (bank restriction) is DEAD

entropy0's top-16 item ids are `[8,121,59,123,5,4,631,0,113,14,83,157,12,19,22,155]`; their
popularity ranks are `[2,1,4,3,0,5,22,13,18,15,11,7,6,9,16,14]`. **All 16 are inside greedy's
top-500-popularity pool — in fact all are in the top 23.** Greedy could have picked every one of
them and chose not to. (entropy0 is close to a permutation of the popularity head: both are
dominated by the 11th "not rated" band, which is prevalence.)

## The decisive table — same sequences, both contracts, 10,000 test users

**Greedy contract** (mask = likes-only fold-in + answered), full/tail NDCG@10:

| sequence | q1 | q2 | q4 | q8 | q16 |
|---|---|---|---|---|---|
| *m=0 intercept* | .1314/.0234 | | | | |
| greedy-combined | **.1487/.0398** | **.1590/.0457** | **.1738/.0588** | **.1918/.0763** | **.2115/.0880** |
| greedy items-only | .1471/.0351 | .1563/.0441 | .1719/.0542 | .1922/.0598 | .2104/.0761 |
| entropy0 top-16 | .1381/.0338 | .1446/.0422 | .1557/.0509 | .1750/.0616 | .1993/.0732 |

**Arm-N pool contract** (identical env, `ctx.va_tr` swapped for `D["pool"]`, one line):

| sequence | q1 | q2 | q4 | q8 | q16 |
|---|---|---|---|---|---|
| *m=0 intercept* | .1586/.0282 | | | | |
| greedy-combined | **.1783/.0478** | **.1909/.0548** | **.2135/.0755** | **.2335/.0962** | **.2556/.1098** |
| greedy items-only | .1760/.0434 | .1907/.0581 | .2066/.0703 | .2247/.0776 | .2449/.0967 |
| entropy0 top-16 | .1635/.0433 | .1706/.0528 | .1831/.0636 | .2053/.0792 | .2303/.0924 |
| *published ladder entropy0* | .16348/.04328 | .17056/.05282 | .18310/.06358 | .20526/.07922 | .23031/.09241 |

**Two exact reproductions validate the diagnostic**: greedy under the greedy mask matches
`greedy_static_i26c.json` to 4 dp at every budget, and entropy0 under the arm-N pool matches
`interview_table_ladder_ep31.json` to 4 dp at every budget. Two harnesses, same numbers, once the
mask is aligned.

## Verdict

**Greedy is optimal, as expected — in BOTH contracts, at EVERY budget, on BOTH axes.**
Greedy contract: +0.0122 full / +0.0148 tail over entropy0 at q16. Arm-N contract: +0.0253 full /
+0.0174 tail. The published ".2303 > .2115" comparison read one row from each table; those numbers
were never ordered. Swapping only the mask is worth **+0.0272 full at zero questions** and +0.0441
on greedy-combined at q16 — more than twice the disputed 0.0188 "deficit", and it has nothing to do
with question selection.

Greedy also wins on *fewer answers*: entropy0 collects 5.53 answered items at q16, greedy-combined
2.25 items + 6.66 concepts, greedy items-only 4.20. Greedy buys more NDCG per answer.

## ★ CONSEQUENCE FOR ACT 1 — the null can now be stated against the strongest static

The adaptive selectors were run under the arm-N pool. The arm-N rows above put greedy in that same
contract, so the comparison is now legitimate (consumer = certified tower, 10k test users,
NO-FLOOR, arm-N pool). The item-bank comparison — the fair one, since the adaptive selectors have
no concept channel — at q16:

| selector / order | full | tail |
|---|---|---|
| **greedy items-only (strongest static)** | **.2449** | **.0967** |
| entropy0 (strongest lit heuristic) | .2303 | .0924 |
| GLM-UCB | .2161 | .0833 |
| Thompson | .2135 | .0878 |
| EDDI info-gain | .1917 | .0749 |

**The null hardens: the strongest static beats the best adaptive selector by +0.0288 full /
+0.0134 tail at q16**, versus +0.0142/+0.0091 against entropy0. Chapter B's open REDO on this point
is discharged for the item bank.

Two honesty notes that must travel with the row. (1) The greedy sequence was *selected* under the
likes-only mask on validation users; scoring it under arm-N is therefore a **lower bound** on what a
greedy search optimised for arm-N would reach — conservative, which is the right direction for a
null. (2) A greedy re-run with the arm-N pool as its selection contract is still owed for a fully
like-for-like row, and the paired CIs remain outstanding.
