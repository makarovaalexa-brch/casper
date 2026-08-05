# Greedy prefix-optimal best-static sequence on the CERTIFIED stack (i26c ep31 + cv3 ep08)

Run 2026-08-04 15:49 → 2026-08-05 03:52 (**723.6 min**, CPU, detached PID 2404).
Script `src/instrument/strategy_channel_suite.py --greedy --eval_test --full_threads --n_items 500
--n_concepts 500 --objective full`. Output `experiments/battery/greedy_static_i26c.json` (+ .png).
Supersedes the Jul-27 run (`GREEDY_BEST_STATIC_RESULT.md`), which used the pre-certification tower
`t2i25_EP4_SNAP.pt` + fold `cd_s1_l10_best.pt` and is now **old-tower, do not cite beside i26c rows**.

## Contract (why the Jul-27 numbers could not simply be re-labelled)

Tower `models/i26c_ep31/t2i26c_ep31.pt` (ep31, frozen) + concept channel
`models/i26c_ep31/concept_cv3_ep08.pt` (ep8, 464k params, `use_refusal=True`, `keep_meh=False`).
NO-FLOOR, learned decoder bias, full **and** tail NDCG@10, canonical Liang ML-25M split.
Every asked question burns the budget:

* an asked item the user has not rated → folded as the i26 **UNSEEN exposure token** (the i25 path
  dropped it, or invented a mild-negative level the tower never trained on; `--fold_refusals` now
  hard-errors);
* an unanswerable concept → folded as a **cv3 refusal token** (`v=0.0, ref=1.0`); 41.6% of cv3's
  training concept tokens were refusals, so dropping them would make an unanswered question free;
* only **answered** items leave the candidate list (asked-but-unseen was never revealed, so it
  neither burns a target nor leaves the ranking; asserted `ans ∩ targets = ∅`).

**User split.** The greedy search ran on the **10,000 canonical VALIDATION users and nothing else**;
the test cohort was not constructed until all three sequences were frozen, and is read once per
(arm, budget). No question, bank, pool or checkpoint was chosen from a test number.

**Gates.** (a) m=0 empty interview on val: **0.12987 vs the tower's recorded cold k0 0.12987,
delta +0.00000** (tol 0.002) → PASS, so the fold/mask/metric path is the certified one.
(b) Concept-net load asserts architecture, answer contract, vocabulary and **tower pairing**
(`snapshot == t2i26c_ep31.pt`) and a bit-exact G0 item-tie (empty concept set ⇒ fold is the
identity). (c) Banks resolve through a **no-default lookup + monotonicity assert**, so the
`--bank expinfo` silent-fallthrough-to-RANDOM class of bug (fixed 26fe795) is unreachable here.
Banks: 500 items ranked by popularity, 500 concepts ranked by train answer rate (Jul-27 parity),
combined 1,000. 3,173 evaluations at 13.54 s each; 1,169 exact memo hits (shared seeding pass).

## Result (TEST cohort, 10,000 users; full/tail NDCG@10)

Empty-interview intercept on test: **0.1314 / 0.0234**.

| arm | order | q1 | q2 | q4 | q8 | q16 | answered@16 |
|---|---|---|---|---|---|---|---|
| items-only | `iiiiiiiiiiiiiiii` | .1471/.0351 | .1563/.0441 | .1719/.0542 | **.1922**/.0598 | .2104/.0761 | 4.20 items |
| concepts-only | `cccccccccccccccc` | .1487/.0398 | .1590/.0457 | .1738/**.0620** | .1898/**.0815** | .1997/**.0921** | 12.87 concepts |
| combined | **`c c c i c c i i i i c c i i i i`** | .1487/.0398 | .1590/.0457 | .1738/.0588 | .1918/.0763 | **.2115**/.0880 | 2.25 i / 6.66 c |

First five greedy gains, combined arm: +0.0166, +0.0100, +0.0074, +0.0075, +0.0060.

## What this establishes

1. **Coarse-to-fine falls out of pure optimisation, on the certified stack.** Nobody specified a
   curriculum: the combined arm's greedy order opens with **three concepts**, and concepts hold six
   of the first seven slots before items take over the back half. This is the certified-stack
   re-establishment of the demoted Jul-14 probe's direction (genre opener → niche congruent film)
   and of the leg-1 channel-flip prediction (concept-optimal when cold, item-optimal once placed).
   It is a property of the loss surface, not of a trained policy.
2. **~~The channels split the axes.~~ RE-CUT 2026-08-05 under candidate-mask policy B —
   the item channel's "win" on full was an artefact of the pool.** Under policy A + answered-item
   masking the numbers above read "concepts own the tail, items own full at long budgets
   (.2104 vs .1997 at q16)". Re-scored under policy B (every rated fold-in item masked, identically
   for every arm — see `MASK_POLICY_B_MIGRATION.md`), **concepts-only wins BOTH axes at q16:
   .2473/.1154 against items-only .2449/.0967**, and combined reaches .2556/.1098. The item channel
   had been receiving a free lift that the concept channel could not receive, because answered
   *items* were deleted from the candidate list (5.42/user) while answered *concepts* delete
   nothing. **A `--objective tail` run remains not optional** for the paper's tail claims.
3. **Answerability is the mechanism, quantified again:** 16 item questions yield 4.20 answers;
   16 concept questions yield 12.87. Same budget, 3.1× the evidence.
4. **This is Paper B's strongest-static comparator.** Any adaptive policy is measured against the
   combined arm, which is prefix-optimal (Paper A) and optimised on the same consumer it is scored
   on — i.e. the hardest honest bar, not a heuristic strawman.

## Limits / follow-ups

* Objective = `full` only. The tail-objective sequence is a one-line relaunch (separate filenames
  and sidecar) and is **required before any tail-optimised claim**.
* Concept bank = `answerable` (Jul-27 parity). The Aug-4 record says concepts beat items on both
  axes only under a **pure-discrimination** bank; `--conc_bank polar` here uses `std|v|`, which is
  *not* the same estimator as `concept_cold_check`'s polar (`H(bands|answered)`). Porting that
  estimator is open work — do not present this run's bank as "the polar bank".
* Greedy is prefix-optimal, not globally optimal; that is Paper A's result and is cited, not
  re-proved here.
* Sidecar `greedy_static_i26c.partial.json` holds each frozen arm signed with the full contract;
  a relaunch resumes, a changed contract is refused.
