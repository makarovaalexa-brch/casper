# Paper A (chapterA_v2) — meticulous number / chart / prose / bib audit

> Run 2026-07-27. Every numeric claim in `new_chapters/chapterA_v2/chapterA_v2.tex` was checked against the
> committed JSON or run log that produced it. Verdicts below are one of: **OK** (matches source),
> **FIXED** (was wrong, corrected), **CERTIFIED** (was `\TODO`, now a measured number), **RE-RUN QUEUED**
> (measured, but on a superseded stack — marked `\REDO{}` in the text).

## Provenance convention introduced
The chapter now carries two distinct status macros, because a reader must be able to tell "no evidence yet"
from "evidence from an older instrument":

- `\TODO{...}` — never run / not yet landed.
- `\REDO{...}` — **was** measured, but on a superseded stack (earlier fold recipe, or the pre-certification
  tower); the re-run is queued.

A "provenance convention" paragraph at the head of §6 states once which stack each class of number came from.

## 1. CERTIFIED — numbers that were `\TODO` and are now measured

Source: `experiments/baselines/t2final_train.out` (train log) →
`experiments/baselines/ml25m_liang/tower_t2.json`.

| quantity | value | where it now appears |
|---|---|---|
| T2′ tower TEST full NDCG@10 | **0.3482** | abstract, `tab:g0`, `tab:master`, gap map, §5.1, §7 |
| T2′ tower TEST tail NDCG@10 | **0.2462** | same |
| T2′ tower TEST NDCG@100 | 0.4486 | `tab:g0` note |
| G0 tie vs RecVAE bar 0.3540 | diff **−0.0057**, CI **0.0069**, `tie=True` | §7 G0 paragraph |
| G0 identity (empty interview) | \|z\|=0.000000; Spearman 1.0000 vs frozen decoder bias | §7 G0 paragraph |
| best val full@10 (selection) | 0.3510 (converged ep13, val flat 4 epochs) | not printed (val, not test) |

**Consequence for the chapter's claims:** R1/G0 is no longer an open question — it is passed, on the same
ruler as every baseline row, with the identity half holding exactly rather than statistically. The gap map
gained a *measured* green diamond for T2′ at (x=2.5, y=0.3482) alongside the target star; the star no longer
carries a "TODO" label.

## 2. FIXED — numbers that were wrong

| location | was | is | source |
|---|---|---|---|
| `tab:g0`, Most-Popular note | @100 `0.1961` | **`0.1975`** | `ml25m_liang/pop.json` = 0.197477 |
| §6.4, volume-leak probe | R² `0.025 → 0.204` | **`0.025 → 0.249`** | `CONCEPT_CHANNEL_RESULT.md` = 0.02365 → 0.24928 (§6.5 already said 0.249 — the two were self-inconsistent) |

## 3. C1 bridge — completed honestly, including a failure

Source: `experiments/baselines/snap_ml20m/*.json` vs the published ML-20M Liang numbers.

| model | published @100 | ours | verdict |
|---|---|---|---|
| EASE | 0.420 | 0.4203 | **PASS** (+.0003) |
| RecVAE | 0.442 | 0.4425 | **PASS** (+.0005); R@20 .4144 vs .414, R@50 .5525 vs .553 |
| Mult-VAE | 0.426 | 0.4223 | **near-snap** (−.0037); admitted, labelled as near-snap not a reproduction |
| **Mult-DAE** | 0.419 | 0.3969 | **FAIL** (−.022) → **DROPPED** from the baseline bank entirely |
| iALS (vs WMF anchor) | 0.386 | 0.358 | −.028; iALS rows are a **floor**, not a tuned reproduction |

The chapter now says this outright: "a bridge that only ever passes is not a bridge." C1 moved from
*partially met* to *met, with one disqualification*.

## 4. RE-RUN QUEUED — measured, but on a superseded stack

The Jul-25 strategy×channel suite ran on the **superseded concept fold** (`cfold_signed_best.pt`, capture
1–6%). The Jul-27 recipe fix (signed four-band SEL + kc/member-drop curriculum, `cd_s1_l10_best.pt`, capture
~25%) **inverts its long-interview conclusion**. Actions taken:

- **Figure `fig:strategychannel` REMOVED from the PDF.** It visually asserted "items win the long interview,"
  which is now known to be a broken-fold artifact, and it also still carried the design flaws the author
  flagged (unequal y-axes, a privileged oracle arm, an artificial "mixed" strategy). A `\REDO{}` placeholder
  names the replacement now in flight.
- **A named retraction paragraph added** ("*A retraction: 'items win the long interview' was an artifact of a
  broken fold*") rather than a silent restatement.
- **NO cross-channel claim is made in the chapter at all**, pending the greedy run. *(This was corrected
  mid-audit: I had first substituted the fixed-fold D1 table — concept-ask .1591/.1803/.1972/.2107 vs
  item-ask .1374/.1493/.1839/.2274 — as the replacement. The author caught that D1's methodology is
  `item-ask = popularity order` vs `concept-ask = polarization order` (`answer_contrast.py:10-11`): two
  different hand-picked heuristics, one per channel. That is the same asymmetric-heuristic defect that made
  the original suite fragile, and it is exactly what the agreed best-static-greedy criterion replaces.
  Swapping one arbitrary pairing for another is not a fix. The table was pulled.)*
- **The agreed replacement criterion** is stated in the chapter instead: the best static question sequence
  for NDCG by greedy maximisation at each step, over items-only / concepts-only / combined banks, built on
  val and scored on the disjoint test cohort. The chapter spells out why this is strategy-free: the combined
  bank *contains* the items-only bank, so combined can only fall below items-only by overfitting the
  construction set — making the combined−items gap a direct readout of what concepts add.
- The chapter now retains only the weak claim the instrument paper actually needs: **both channels fold and
  both move the ranking off the intercept**, with no ranking between them.
- **Adaptive-headroom magnitudes withdrawn** (oracle−static gap, greedy closure %). The *direction* is
  re-grounded on fold-independent evidence: the D3 one-level probe, item channel Δfull +0.0015
  CI[+0.0007,+0.0024], Δtail +0.0016 CI[+0.0008,+0.0024]; concept channel positive-not-significant.
- Other `\REDO{}` marks: fold-operator magnitudes (§6.3), signed triple-gate deltas (§6.5), frozen-fold vs
  tokens-in-tower concept curves, graded-vs-binarized curve on the certified tower.

**Note:** the G0 numbers inside the tokens-in-tower comparison (0.3487/0.2471, tie vs 0.3540) are on the
certified ruler and stand; only the concept curves that decide *between* the two homes are queued.

## 5. Verified OK (spot-checked against JSON, no change)

All `tab:g0` / `tab:master` baseline rows match their JSONs to the printed precision: Most-Popular
.1345/.0226, belief-MF .2019/.1200, Item-kNN .2428/.1297, iALS .2442/.1853, Golbandi-node .3065/.1895,
EDLAE .3433/.2395, EASE .3476/.2441, RecVAE .3540/.2497, and all their @100 values (except the Most-Popular
one fixed above). Star-curve numbers match `GRADED_CURVE_RESULT.md` (+0.0117@k2 CI[+.0099,+.0135],
+0.0071@k4, tie by k8). Signed triple-gate deltas match `signed_sel_gate_signed_retrain.json`. Cold intercept
0.1279/0.0192 matches `strategy_channel_suite.json:intercept`.

## 6. Bibliography

- **All 35 cited keys resolve**; no duplicates; final compile pass has **0 undefined citations and 0 undefined
  references** (the 308 warnings in the old `compile.log` were first-pass only — a false alarm).
- **3 entries added** for claims that were cited in prose but had no bib entry:
  `good1967principle`, `blackwell1953equivalent` (the non-degradation-of-information principle, invoked 4×),
  and `mu2018allbutthetop` (the all-but-the-top whitening, named but uncited). Bibliography: 35 → 38.
- `references.bib` has 78 uncited entries — expected, it is the shared thesis bib.

## 7. Prose / build hygiene

- **9 raw-UTF-8 `Bıyık`** normalised to `B\i y\i k` (the two spellings were mixed; raw dotless-ı under the
  cm-font stack is a silent-render risk). The file is now **100% ASCII**.
- Abstract rewritten: no longer says "the tower is training and all instrument results are `\TODO`".
- Gate preamble, §5 opener, §5.1, `\paragraph{Metrics}`, and the C1 gate/battery rows all de-staled to match
  what is now measured.
- Master-table and G0-table captions now state the **train vocabulary 18,359** alongside the 18,430 arena
  catalog, which were previously conflated.
- Build: `tectonic chapterA_v2.tex` → clean, 0 undefined. 21 overfull hboxes remain (cosmetic, mostly the
  sideways master table and the wide `\paragraph` blocks); not addressed.

## 8. A run-integrity catch made during this audit

The in-flight best-static greedy run (PID 6512) was loading the **superseded** `cfold_signed_best.pt` — the
script's default — and would have re-derived the very broken-fold artifact this audit exists to retire. It was
killed 23 min in (no results lost) and relaunched on `cd_s1_l10_best.pt`. A provenance line now logs which
fold the sclite rung rides (commit `e745f6f`), so this cannot recur silently.
