# Graded-vs-binarized interview-length curve ("the star curve") — RESULT

> Ran 2026-07-27 (`src/instrument/premium_vs_k.py` → `experiments/battery/premium_vs_k.json`;
> chart `src/instrument/plot_graded_curve.py` → `new_chapters/chapterA_v2/fig_graded_curve.png`,
> figure `\ref{fig:gradedcurve}`). Frozen graded-native i25 tower (`t2i25_EP4_SNAP.pt`), VAL cohort,
> parity mask, full NDCG@10. Answers the author's "does gradedness help on short interviews?" and the
> designed gate `docs/design/GATE_BATTERY_INSTRUMENT.md:34` ("graded vs binarized, k=2..full").

## The curve (full@10, same revealed set fed three ways)
| k | graded (true) | likes-only | all-as-like | graded−likes | graded−all-as-like |
|---|---|---|---|---|---|
| 2 | 0.1778 | 0.1661 | 0.1751 | **+0.0117** | +0.0027 |
| 4 | 0.2016 | 0.1945 | 0.1775 | +0.0071 | +0.0241 |
| 8 | 0.2248 | 0.2258 | 0.1774 | −0.0010 | +0.0474 |
| 16 | 0.2575 | 0.2587 | 0.1871 | −0.0012 | +0.0703 |
| full | 0.3478 | 0.3435 | 0.2283 | +0.0043 | +0.1195 |

Flip control (mirror star signs): −0.038 (k2) → −0.294 (full), CI-clean throughout.

## The reading — stars help over binary on the SHORT interview
**Headline (vs the fair positive-only baseline `likes-only` — what RecVAE/Mult-VAE do at r>3.5):** graded
input beats binarized input on the short interview and the edge is **CI-clean**:

| k | graded − likes-only | 95% CI | verdict |
|---|---|---|---|
| 2 | **+0.0117** | [+0.0099, +0.0135] | clean win |
| 4 | **+0.0071** | [+0.0053, +0.0090] | clean win |
| 8 | −0.0010 | [−0.0027, +0.0008] | tie |
| 16 | −0.0012 | [−0.0027, +0.0003] | tie |

Stars help **most when the interview is short** (few items known → each answer's sign/level is decisive), and
the advantage washes out by k=8 as collaborative signal pins the user down. **Mechanism:** a signed answer
encodes a *dislike* — a negative observation the positive-only item-indicator basis cannot express; that is the
expressive gap the graded leg fills. (Not answerability.)

**Why the first-reported number looked "opposite" (grows with k):** that used the `all-as-like` control (every
answered item = a 4-star like — naive implicit "any interaction = positive"), which **mislabels dislikes as
likes**. Its premium grows +0.003 → +0.12 with k only because more revealed items = more dislikes corrupted —
a control artifact, not a marginal-value-of-stars effect. The fair baseline is `likes-only`.

## REPLICATED ON THE CERTIFIED TOWER (2026-07-28)

Re-ran on `t2final_best.pt` (`premium_vs_k_t2final.json`; the pre-certification run is preserved as
`premium_vs_k_ep4snap.json`). Same shape, slightly STRONGER at the short budgets:

| k | ep4 snapshot | **certified t2final** |
|---|---|---|
| 2 | +0.0117 [+.0099,+.0135] | **+0.0147 [+.0127,+.0167]** |
| 4 | +0.0071 [+.0053,+.0090] | **+0.0083 [+.0063,+.0103]** |
| 8 | -0.0010 (tie) | -0.0006 [-.0024,+.0012] (tie) |
| 16 | -0.0012 (tie) | -0.0007 (tie) |
| full | - | -0.0002 (tie) |

Figure and caption updated to the certified numbers; the \REDO marker is cleared.

**Paper:** Figure~\ref{fig:gradedcurve} — panel (b) leads with the short-interview graded-over-binary
premium + CI. Re-run on the certified tower when it lands. (full-profile point pending; not needed for the
interview-range claim.)
