# A26 — the elicitation harness on Goodreads: the concept channel is STRONGER there (2026-09-12)

The portability question `A24_RESELECTED_DIAGNOSTICS.md` left open. Certified Goodreads pair:
tower `t2i26g_ep53`, concept net `cvg_ep10`. Validation cohort, same harness, same budgets.
Output `experiments/battery_goodreads/strategy_channel_suite.json`.

## Lift over the no-question intercept at q=16

Intercept: ML-25M 0.1279 full / 0.0192 tail · Goodreads 0.0889 / 0.0060.

| arm | ML-25M full / tail | Goodreads full / tail | mean answers ML \| GR |
|---|---|---|---|
| items-pop | +0.0668 / +0.0388 | +0.0434 / +0.0077 | 5.41 \| 3.56 |
| items-entropy | +0.0016 / +0.0026 | +0.0129 / +0.0061 | 0.07 \| 0.45 |
| items-HELF | +0.0443 / +0.0388 | +0.0278 / +0.0102 | 3.28 \| 2.40 |
| items-random-bank | +0.0022 / +0.0027 | +0.0028 / +0.0039 | 0.12 \| 0.07 |
| **conc-member-mass** | +0.0099 / +0.0132 | **+0.0314 / +0.0211** | 15.85 \| 13.36 |
| conc-polarization | +0.0192 / +0.0196 | +0.0290 / +0.0183 | 15.76 \| 11.34 |
| conc-HELF-analog | +0.0192 / +0.0196 | +0.0276 / +0.0177 | 15.76 \| 13.31 |
| conc-random *(floor)* | +0.0146 / +0.0188 | +0.0071 / +0.0067 | 6.46 \| 0.95 |
| ADAPTIVE-greedy | +0.0117 / +0.0150 | +0.0027 / +0.0081 | 15.44 \| 8.76 |
| ADAPTIVE-oracle *(privileged)* | +0.0957 / +0.1112 | +0.0471 / +0.0427 | 16.00 \| 15.27 |
| mixed-c4-then-items | +0.0495 / +0.0305 | +0.0399 / +0.0064 | 8.05 \| 4.25 |
| mixed-interleave | +0.0304 / +0.0289 | +0.0383 / +0.0066 | 10.55 \| 7.12 |

## Three findings

**1. The concept channel is STRONGER on Goodreads.** +0.0314 full against ML-25M's +0.0099, on a
*lower* intercept (0.0889 vs 0.1279) — roughly +35% relative against +8%. The channel does not merely
port; it ports better.

**2. ⭐ Concept SELECTION matters on Goodreads and does not on ML-25M.** On ML-25M `conc-member-mass`
(+0.0099) is *worse than choosing concepts at random* (+0.0146); on Goodreads it is far better
(+0.0314 vs +0.0071). The mechanism is in the answer counts: random concepts collect **0.95** answers
per 16 asks on Goodreads against **6.46** on ML-25M. With 73% refusal, picking concepts the user can
actually answer is what makes the channel work.

This is the constructive counterpart to the A22/A24 result. Concept narrowness is real and it *does*
matter — but the remedy is **ranking the full bank by answerability**, not **truncating** it. Truncating
to the 61 broadest (A24) cost −0.0077 because that pool is contentless; ranking by member mass keeps the
whole bank and simply asks the answerable ones first.

**3. The channel ordering FLIPS on tail.** ML-25M: items beat concepts (+0.0388 vs +0.0132). Goodreads:
concepts beat items nearly 3x (+0.0211 vs +0.0077). On books, broad attributes are the better route into
the long tail; on films, item questions are.

## ADAPTIVE-greedy is weak on BOTH — not a Goodreads fault

+0.0117 on ML-25M and +0.0027 on Goodreads, and on ML-25M it sits *below* that dataset's own
`conc-random` (+0.0146). This reproduces the existing Act-2 finding that the adaptive−static gap runs
the wrong way, on a second dataset. The privileged oracle shows the headroom is genuinely there
(+0.0957 / +0.0471), so this is a selector failure, not an absence of signal.

## Caveats

- Validation cohort, single run, no bootstrap CIs on these cells yet. The ML-25M column is the existing
  certified run, so the two were produced by the same code at different times.
- The harness folds **answered concepts only** (`strategy_channel_suite.py:309`); concept refusals are
  never folded. A24 found Goodreads' *fold-level* lift comes mostly from the refusal pattern, so this
  harness is reaching +0.0314 **without** the mechanism A24 identified as dominant. Whether folding
  refusals here would add more is untested and is the obvious next experiment.
- `ADAPTIVE-oracle` is privileged and is a ceiling, not an arm.
