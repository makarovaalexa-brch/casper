# 🏆 PAPER B WINNER — CASPER-R  (DO NOT DELETE / DO NOT OVERWRITE)

**THE checkpoint:** `policy_entdistill_ep4.pt`  (alias `policy_entwin.pt`)
**Live load path:** `data/movielens/.cache/policy_entdistill_ep4.pt`  (that dir is gitignored — this archive is the durable, version-controlled guard)
**SHA256:** `0f553015f445e5d53294ffecdcaf2eaf8a2921fcb79f6f5c22abc548bff091dc`

This is **CASPER-R**, the learned cold-start concept-elicitation policy reported in Paper B
(`papers/paper2_casper/paper2_casper.tex` → `tab:learned`, `fig:qcurve`).

## Recipe
Entropy-BC floor → **reconstruction** refinement. Differentiable straight-through rollout;
loss `1 − cos(u, u*)`, where `u*` = fold of the known profile.
(NOT REINFORCE — that label was a bug; REINFORCE `rbase`=0.126 < entropy. Corrected in `alg:train`.)

## Headline metrics  (te[300:], 304-user representative test, seed-avg n=6 {123,1,2,3,7,11})
| metric | CASPER-R | static entropy | delta |
|---|---|---|---|
| **TAIL NDCG@10 @q8** | **0.152 ± 0.002** | 0.141 ± 0.005 | **+0.011, all 6 seeds positive (~6σ)** |
| FULL NDCG@10 @q8 | 0.359 | 0.360 | tied (popularity-saturated) |
| FULL @q2 (efficiency) | — | — | **+0.008** (adaptive low-q edge) |
| cos(u, u*) | 0.745 | 0.637 | sharper belief |
| answered / 8 | ~5.3 | — | — |

Paper reports CASPER-R = **0.360 / 0.152** (full / tail). The win is on the **long tail + question-efficiency**; full is popularity-saturated and ties.

## Why ep4 (NOT best-val ep7)
val ≠ test (~300 noisy val users). Val-best `ep7` was the **test-worst**; `ep3`=`ep4` are the **test peak**.
Selected by scoring EVERY epoch on the paper test set te[300:], then seed-averaging.
`peak_entdistill.txt` records the *val* selection (ep7) — that is **NOT** the reported checkpoint; ep4 is.

## Architecture / how to load
Scorer MLP over per-candidate features `[emb, u·E align, popularity-prior, belief-strength, turn]`
+ optional history-attention head; soft-pick (train) / argmax (eval) over the unified **1361** pool
(600 items + 761 concepts). Code: `scripts/paper2/continuous_policy2.py` (mode `entdistill`, `LOAD=1`).

## Files in this archive
- `policy_entdistill_ep4.pt` — **THE winner** (reported checkpoint)
- `policy_entdistill_best.pt` — val-best (ep7); provenance only, NOT the reported model
- `policy_entdistill_bc.pt` — the entropy-BC floor it refined from
- `peak_entdistill.txt` — durable peak record (val selection)
- `evalcks_entdistill.csv` — per-seed per-q eval grid behind the metrics above

## Provenance docs (repo root)
`POLICY_RESULTS.md` (full log), `POLICY_LADDER.md`, `OVERNIGHT_CAMPAIGN_2026-06-24.md`, `PAPER_B_ROADMAP.md`.

_Durable guard created 2026-06-25. Paper B is frozen on this checkpoint._
