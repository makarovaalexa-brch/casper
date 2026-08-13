# Checkpoints deleted from `.cache/baselines/` on 2026-08-13 — identity manifest

Deleted to recover disk (the box was at 0.97 GB free). **These four binaries are gone and can only be
restored by retraining.** This manifest exists so their loss stays *detectable* and any replacement
stays *verifiable* — the lesson of the pbC provenance hole (memory `anchor-provenance-hole-pbc-recvae`),
where weights vanished with no record and a recorded number became unreproducible.

Machine-readable copy: `.cache/baselines/DELETED_CHECKPOINTS_MANIFEST.json`.

| file | sha256 | size | trained | what it backed |
|---|---|---|---|---|
| `recvae_ml20m.pt` | `E547493C…87297CA3` | 394,826,414 B (376.5 MB) | 2026-07-21 | the ML-20M leg of the Liang replication snap (HARD RULE 2) |
| `multvae_ml20m.pt` | `C95185DF…4FCA66DF` | 392,200,678 B (374.0 MB) | 2026-07-26 | ditto, Mult-VAE arm |
| `dae_ml20m.pt` | `65351259…7D3D0C41` | 390,276,086 B (372.2 MB) | 2026-07-25 | Mult-DAE — **already DROPPED** for failing its snap (`g0-certified-and-paperA-audit-2026-07-27`) |
| `recvae_ml25m.pt` | `B0574CAA…3E4D215F3D` | 365,267,182 B (348.3 MB) | 2026-07-21 | the **pre-Liang** ML-25M RecVAE, i.e. the retired 500/500 arena whose numbers are never cited |

## Why this was judged acceptable

The test applied is the same one in `models/baselines_ml25m_liang/README.md`: *does a number the
chapter still cites die if this file is lost?*

- The ML-20M files back a **replication snap that is complete and banked** — the chapter cites the
  snap's *numbers*, which live in their result artifacts, not the weights.
- `dae_ml20m.pt` backs an arm that was **dropped on evidence**; nothing cites it.
- `recvae_ml25m.pt` is on the **retired arena ruler**, whose figures (0.5078/0.5230) the project rule
  says never to cite.

## What was deliberately NOT deleted

- `models/baselines_ml25m_liang/*.pt` — the five rescued checkpoints that back live `tab:battery`
  rows. Their `.cache/` duplicates were removed instead, after MD5-verifying byte-identity.
- The certified pair `t2i26c_ep31.pt` + `concept_cv3_ep08.pt` (34 MB total).
- `data/reddit_he/` (5.86 GB) — **irreplaceable**: the standing project rule is never to touch the
  Reddit API, so nothing there can be re-fetched. Large, tempting, and permanently off-limits.

## If one of these is ever needed again

Retrain, then check the new file's sha256 against the row above. A match proves the reconstruction is
bit-identical; a mismatch means the recipe drifted and any number derived from it must be re-measured,
not assumed.
