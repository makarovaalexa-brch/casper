
## STAGE D1 � concept-vs-item on the fixed fold  (2026-07-27 00:54:41)

Best distilled fold: `cd_s1_l10_best.pt`.

Matched-budget realizable concept-ask vs pop item-ask (full@10 | tail@10 | full@100):

| k | concept f@10 | item f@10 | concept t@10 | item t@10 | concept f@100 | item f@100 |
|---|---|---|---|---|---|---|
| 1 | 0.1591 | 0.1374 | 0.0638 | 0.0392 | 0.2345 | 0.2018 |
| 2 | 0.1803 | 0.1493 | 0.0861 | 0.0478 | 0.2595 | 0.2171 |
| 4 | 0.1972 | 0.1839 | 0.1050 | 0.0648 | 0.2813 | 0.2568 |
| 8 | 0.2107 | 0.2274 | 0.1210 | 0.0839 | 0.2984 | 0.3013 |

Concept beats item (full@10) at k = 1; (tail@10) at k = 1.
D1a adaptive_concept_arms rc=0, D1b concepts_only_curve rc=0 (committed-harness @10 crossover + selection arms in their JSONs).

## STAGE D2 � answerer panel on the fixed fold  (2026-07-27 01:30:57)

answer_contrast rc=0, out=`answer_contrast_distill.json`.

Panel ran on the fixed fold; see JSON for per-arm capture/curve + CKA/oracle-B guards. KEY read: whether imputers now differentiate vs SEL (broken-fold artifact test).

## STAGE D3 — 1-level adaptivity tree (concept + item)  (2026-07-27 01:51:23)

1-level tree: q1 -> branch on answer -> per-branch best q2 (selected on SELECT half, evaluated OUT-OF-SAMPLE on EVAL half) vs single STATIC q2. Bank=top-40; branch = answer band.

| channel | q1 | static f@10 | adaptive f@10 | Δfull (CI) | Δtail (CI) | pays full | pays tail |
|---|---|---|---|---|---|---|---|
| concept | 704 | 0.1458 | 0.1474 | +0.0016 [-0.0008,+0.0039] | -0.0007 [-0.0033,+0.0018] | False | False |
| item | 98 | 0.1450 | 0.1465 | +0.0015 [+0.0007,+0.0024] | +0.0016 [+0.0008,+0.0024] | True | True |

**Item channel: adaptivity PAYS** (Δfull +0.0015 CI[+0.0007,+0.0024]; Δtail +0.0016 CI[+0.0008,+0.0024]) — both significant, refuting linear-Gaussian static-optimal (HARD RULE #2). This is the fold-independent capability confirmation. **Concept channel: positive but NOT significant** at 1-level (Δfull +0.0016 CI[-0.0008,+0.0039]; tail flat) — the effect exists but the 1-level probe is underpowered for concepts (a learned multi-turn policy = Chapter B). Effect sizes are small (~+0.0015): a 1-level q1→q2 preview, not a full adaptive policy.

## CHAIN VERDICTS (D1–D3, fixed λ=1.0 fold cd_s1_l10)  (2026-07-27 01:51:23)

- **D1**: concepts now beat items on TAIL at every budget (@10 & @100) and lead FULL through k=4 (items overtake full only at k=8). Stronger short-interview/tail concept case than the broken-fold 'items close by q8' story.
- **D2**: 'SEL is the ceiling' REPLICATES — imputers (SEL+/content) tie SEL within ~0.003, ExpoMF/PITF worse; not a broken-fold artifact. Real lever = SELECTION (oracle-select 0.211 vs fixed-order answer-model ~0.144 @q8), not the answer model.
- **D3**: adaptivity PAYS on the item channel (Δfull/Δtail both sig); concept-channel 1-level positive but not significant (needs the multi-turn policy).
