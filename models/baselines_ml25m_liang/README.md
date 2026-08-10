# Baseline checkpoints that back table rows — ML-25M Liang split

Rescued from `.cache/baselines/` on 2026-08-10. **The binaries here are deliberately NOT committed**
(1.05 GB, no LFS on this repo); this manifest is, so their identity stays verifiable and their loss stays
detectable.

## Why these five and not the other forty

The test is: *does a number the chapter still cites die if this file is lost?*

| file | sha256 (first 16) | size | what dies without it |
|---|---|---|---|
| `recvae_ml25m_liang.pt` | `8dcc875d41d1f881` | 347.2 MB | **the frozen teacher AND the top table row** — see below |
| `multvae_ml25m_liang.pt` | `bed9568c107a58f6` | 342.0 MB | Mult-VAE row, $.3202/.2194$ LIANG |
| `eddi_ml25m_liang.pt` | `068330ec0b96b693` | 242.8 MB | EDDI row, $.2153/.1077$ LIANG |
| `tanp_ml25m_liang.pt` | `54a4576b935e050c` | 61.0 MB | TaNP LIANG row, $.2645/.1635$ |
| `tanp_native_armn_rmse.pt` | `bb95b4ec4777588a` | 61.0 MB | the TaNP-as-published result, $0.0060$ (§ native limits) |

## The one that matters most

`recvae_ml25m_liang.pt` is **load-bearing twice over**, and until today it existed only in gitignored
scratch:

1. **The certified tower cannot be loaded without it.** `train_tower_t2.load_recvae_teacher` reads it at
   `RECVAE_CKPT = .cache/baselines/recvae_ml25m_liang.pt`, and `build_i26` must construct the architecture
   from that frozen decoder *before* `t2i26c_ep31.pt`'s state dict can be loaded. So
   `models/i26c_ep31/` was **not self-sufficient**: rescuing the certified weights while leaving the
   decoder they are built on in a directory we were about to prune is the same failure as
   `anchor-provenance-hole-pbc-recvae`, one level up.
2. It is separately the strongest row in the master table ($.3540/.2497$ LIANG, $.4273/.3051$ NATIVE) —
   the row the instrument is measured against.

`models/i26c_ep31/README.md` states the provenance chain as *RecVAE → i26c*. This directory is the RecVAE
end of that arrow.

## Not rescued, deliberately

`*_ml20m.pt` (recvae, multvae, dae — 1.1 GB) back the ML-20M bridge/snap checks. Those checks are
recorded as numbers in `experiments/baselines/snap_ml20m/` and are not re-run; if a bridge ever needs
re-establishing it should be re-run against the published paper rather than against a local checkpoint.

## If a file here is lost

Re-run its trainer against the canonical split and confirm the sha256 changes nothing that matters — the
val-selected number in the corresponding `experiments/baselines/ml25m_liang/*.json` is the acceptance
gate. Note that a re-trained checkpoint will **not** be byte-identical, so a mismatch against the table
above means "regenerated", not necessarily "wrong"; the number is the contract, the hash is provenance.
