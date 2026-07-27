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

## The reading (important — the naive premium is misleading)
There are **two** binary baselines and they tell opposite stories:
- **vs `all-as-like`** (every answered item = a 4-star like; naive implicit "any interaction = positive"):
  premium GROWS +0.003 → **+0.12** with k. This is NOT a marginal-value-of-stars effect — the control
  **mislabels dislikes as likes**, and the damage scales with how many dislikes are revealed (∝ k). It is a
  control artifact, not a property of gradedness. (This is the number that first looked "opposite from
  expectation.")
- **vs `likes-only`** (drop sub-threshold answers — exactly what RecVAE/Mult-VAE do at r>3.5, the fair
  positive-only baseline): premium is **+0.012 @k2, fading to ≈0 by k8**, +0.004 at full. This is the
  author's intuition confirmed — fine star levels help most on the SHORT interview and wash out as
  collaborative signal fills in. Small.

**Conclusion for the instrument.** The graded/signed leg's value is **representing a dislike at all** (a
negative observation the item-indicator basis cannot encode), NOT fine gradation among likes. Against the fair
positive-only baseline the level premium is small and short-concentrated; against naive binarization it is huge
because that control corrupts dislike information. The sign matters far more than the gradations (flip −0.29 at
full). Not answerability. Paper: provisional Figure~\ref{fig:gradedcurve}; re-run on the certified tower.
