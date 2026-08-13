# Overnight handoff — night of 2026-08-13/14

**AUTHOR SAID (verbatim, before sleeping):** *"ok, i am very confused now. why we even need the heads?
did we just discard the tree distillation? was it not better to just emit q embedding to maximise ndcg?
i am not sure where this phase led us"* … *"the cheap part works, but not well? and expensive part is
still as much of a bottleneck as if we had just one head? this whole design is confusing, but i am
going to sleep, you proceed as planned and we reassess in the morning."*

Everything below is MINE (working notes). Governing doc: `docs/design/PAPER_B_POLICY_DESIGN.md`,
**Part G** (the staged plan). Read that before acting on anything here.

## ★ THE AUTHOR'S TWO CRITIQUES ARE BOTH SUBSTANTIALLY CORRECT — answer them first in the morning

1. **"The cheap part works, but not well."** True. The outcome head beats the per-question popularity
   prior but the margin is modest (AUC 0.98243 vs 0.96944; CE 0.05476 vs 0.06974, −21%), and **it is
   5× WORSE CALIBRATED than the baseline it beats** (ECE 0.00147 vs 0.00029). Calibration is not a
   nicety here: the design MULTIPLIES this probability by a value estimate, so two learned errors
   compound. Part F.1 already made calibration a hard reported requirement; this is the first evidence
   it will bind.
2. **"The expensive part is as much a bottleneck as with one head."** ★ **Correct in DATA terms and I
   should have said so unprompted.** Splitting the head added zero examples to the value side — it is
   still **7 complete states**. The only argument for the split is that V becomes an *easier function*
   once answerability variance is factored out. **That is a hypothesis, and tonight's lift-head run is
   its test.** If V fails at 7 states, the honest reading is "the design is fine, the data was never
   there" ⇒ Phase 4 becomes necessary, not optional.

## Where phase 2 stands

| | |
|---|---|
| outcome head, epoch 1 | trained 118 min; held-out CE 0.05476 **better** than train 0.05798 ⇒ no overfitting |
| verdict vs popularity prior | **PASS** (`beats_popularity: true`), numbers above; suite cost 32.0 min |
| global base rates | **0.00817 like / 0.02587 dislike / 0.96596 no-answer** over the whole (user × catalogue) grid — accuracy is a dead statistic here (always-"no-answer" scores 0.966) and the report must judge on AUC/ECE/CE |
| stopped at 1 epoch | deliberate, on my instruction — the author's ask was *"train on all and see if it even works"*, and 44 bn label terms answers that. **Say in the report that one epoch sufficed; that is itself informative.** |
| still running | lift head (7 complete states, primary) + the declared 22-state secondary arm + combined P×V policy on validation + shuffle control. ETA ~04:00–05:00. |

## Running / queued

| what | state |
|---|---|
| phase-2 chain | RUNNING, PID varies, ~04:00–05:00 |
| **phase 3** (Part G.B) | agent launched, **INSTRUCTED TO WAIT** until no python job remains, then `--threads 6`. Deep labels (~3.3 h) → fine-tune with label REPLAY → **the deeper-tree control, which is mandatory** |
| phase 4 (Part G.C) | **BLOCKED on explicit author sampling sign-off (HARD RULE 1).** "proceed with this list" is NOT that sign-off — ask it as its own question with the design in hand |
| phase 5 (Part G.D) | gated shut; opens only if phase 3 saturates with floor-stop gain demonstrably remaining |

## Corrections made today that must not be un-made

- **I told the author "the fixed list currently beats the tree". FALSE.** Best static combined
  order .2590 vs tree .2694 on test — the tree beats it by +0.01035. The true statement is a
  comparison of LEVER SIZES: conditioning is worth +0.01035, improving the question BANK is worth
  +0.01430. Both real, second larger. Never restate it the other way.
- **The policy MUST beat the tree** — that is both the target and, from phase 3 on, the floor (phase 1
  is an exact copy of it). The bank number is the yardstick for how much the win is worth, not a
  substitute comparator.
- The tree's split criterion and plain greedy gain are **algebraically identical** (verified 5.6e-17
  over 7 nodes). My earlier "two different scalars" claim is withdrawn.
- `.cache/baselines/recvae_ml25m_liang.pt` is **load-bearing by path**, restored as a hardlink; see
  `models/DELETED_CACHE_CHECKPOINTS.md`.

## Method lessons banked tonight (worth a memory note in the morning)

- **`torch.argmax` on a (3, 1024, 19390) tensor costs 5.5 s — against 0.08 s for `gather` and 0.8 s
  for `logsumexp` on the same tensor.** It computed the *accuracy printout*, not the loss. The hot
  path was dominated by a diagnostic nobody needed (accuracy is meaningless at a 96.6% base rate).
- **Bench the assembled path, not the part you wrote.** Synthetic-tensor bench said 42 min/epoch; real
  data said 104 and then *degraded superlinearly* to ~148 as the working set stopped fitting.
- **We were part of the load.** Measured: training 8.23 cores + this session's agents 0.93 + others
  ≈ **10.37 cores demanded on 4 physical cores**. Do not run agent work beside a training job, and
  prefer `--threads 6` on this chip. **No cost figure in this workstream is a property of the method.**
- Batch-by-user (contiguous memmap reads) beat batch-by-(user,position) by **190×**, not the 40%
  estimated.
