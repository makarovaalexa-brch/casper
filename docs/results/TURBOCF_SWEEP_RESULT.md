# Turbo-CF on the canonical ML-25M ruler — the val sweep and what it settles

**Date:** 2026-07-29 · **Runner:** `src/baselines/run_queue2.py --only turbocf` ·
**Log:** `experiments/baselines/ml25m_liang/turbocf_fixed.log`

## Why there was a sweep at all
Turbo-CF (Park, Kim & Shin, SIGIR 2024) publishes no ML-20M/25M strong-generalisation number, so there is
nothing to snap to under HARD RULE 2. The hyper-parameters $(\alpha, s, \text{filter})$ are therefore
selected on **our validation split** and the row is flagged best-effort tuned, like iALS and EDLAE.

## The grid (val full NDCG@10, 10,000 validation users)

| $\alpha$ | $s$ | $\rho(\bar P)$ | filter 1 (linear) | filter 2 (2nd-order) |
|---|---|---|---|---|
| 0.5 | 0.6 | 26.59 | 0.2373 | 0.0001 |
| **0.5** | **1.0** | **1.000** | 0.2313 | **0.2555** ← selected |
| 0.7 | 0.6 | 49.63 | 0.2107 | 0.0001 |
| 0.7 | 1.0 | 6.51 | 0.2041 | 0.0086 |
| 1.0 | 0.6 | 269.21 | 0.1920 | 0.0001 |
| 1.0 | 1.0 | 321.31 | 0.1845 | (not run — $\rho$ makes it degenerate) |

## Test result (selected config, scored once)
**full/tail NDCG@10 = 0.2622 / 0.1621**, NDCG@100 $0.3569$, R@20 $0.3170$, R@50 $0.4427$ —
`experiments/baselines/ml25m_liang/turbocf.json`, $\alpha=0.5$, $s=1.0$, filter 2, $\rho(\bar P)=1.000$.
Mid-pack in the accuracy corner: above RBMF ($0.2534$) and iALS ($0.2442$) on full, below Golbandi's node
recommender ($0.3065$) and well below EASE ($0.3476$). Its tail ($0.1621$) is below iALS's ($0.1853$).
*Note:* the `sweep` field inside `turbocf.json` records only the scoring run's single grid point
(`--turbocf_grid 0.5,1.0,2`); the full grid is the table above and `turbocf_fixed.log`.

## What the sweep settles
The public Turbo-CF code does **not** re-normalise $\bar P$ after the Hadamard power. The polynomial
filters $2P - P^2$ and the 3rd-order approximation are low-pass only while $\rho(\bar P)\le 1$. The table
shows this exactly:

- $\rho = 1.000$ **only** at $\alpha=0.5$, $s=1.0$ — symmetric normalisation with no Hadamard power — and
  that is the **one** cell where the 2nd-order filter helps (0.2313 → 0.2555, $+0.024$).
- Every cell with $\rho > 1$ collapses under the 2nd-order filter: 0.0001, 0.0086, 0.0001. At
  $\alpha=1.0$, $s=0.6$, $\rho=269$ and $F$ reaches $-1.9\times10^{3}$ with 99.9% of entries negative —
  the ranking is inverted, not weak.
- $s<1$ raises every entry of a sub-unit matrix, so the blow-up grows with catalogue size. On a 130-item
  smoke $\rho=5.9$ and the 2nd-order filter still scores fine; at 18,359 items it is fatal. **That is why
  the smoke test passed while the real run returned 0.0000 everywhere.**

The winner is the public repository's own default $\alpha=0.5$, $s=1$, plus the 2nd-order filter. The
paper's tuned region ($\alpha=0.7$, $s=0.6$) does not transfer to a catalogue this size.

## Provenance note
The earlier 0.0000 result (2026-07-28, $\alpha=0.7$, $s=0.6$, filter 2) is archived as
`turbocf_DEGENERATE_alpha0.7_s0.6_filt2.json`. It was a degenerate configuration, not a measurement of
Turbo-CF, and must never be cited as one.

## Guard added
`turbocf.build_filter` now estimates $\rho(\bar P)$ by power iteration, warns when a polynomial filter is
used outside its stable region, logs $F$'s min/max/negative fraction, and rejects a non-finite $F$.
**Lesson: a smoke test on toy data cannot catch a scale-dependent numerical degeneracy — the guard has to
be a property check on the real matrix.**
