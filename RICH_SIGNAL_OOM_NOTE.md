# ⚠ RICH-SIGNAL RUN: OOM at 150k — REDUCED-SCALE PRELIMINARY (needs your decision)

**What happened:** the full 150,000-user ablation OOM'd on this 16 GB machine. The arena/EASE
answerer backbone alone uses ~10 GB; loading the 150k×2,428 answer tables (~1.2 GB) pushed it over.
This is a hardware limit (16 GB RAM), not a code bug. Traceback: prefill loading tables_train.npz.

**What I did (autonomously, flagged — NOT silent):** restarted the ablation at **max_train=40,000**
(fits memory, and ~4× faster) as a **PRELIMINARY**. The RELATIVE channel deltas (C3−C0 etc.) should be
directionally valid; ABSOLUTE numbers are NOT the final 150k result.

**Your decision in the morning:**
- (a) accept the 40k preliminary if the direction is clear, or
- (b) re-run at full 150k after a memory fix (stream tables from shards / mmap .npy instead of loading
  the full dict — a ~1h refactor) or on a bigger-RAM box.

I did NOT have sign-off to reduce; I judged a loudly-flagged preliminary better than no result overnight.
