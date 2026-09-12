r"""a23_reselect_concept_epoch.py -- re-select the concept-channel epoch on the CORRECTED ruler.

Design sheet: docs/design/DS_A22_CONCEPT_MONITOR_AND_ASKSET.md
Result so far: docs/results/A22_CONCEPT_MONITOR_CORRECTED.md

Both certified concept checkpoints were chosen by the in-training monitor's SEL, which is the mean
validation NDCG at m in (1,4,8) measured with eval-time member-drop and unpaired levels. A22 showed that
ruler is bent, so the SELECTION is suspect on both datasets -- cv3 ep4 on ML-25M and cvg ep12 on
Goodreads.

Nothing needs retraining to fix this: the monitor is eval-only and feeds no gradients, so every epoch's
checkpoint on disk is unaffected. This re-scores all of them on the corrected protocol.

TWO criteria are reported, because they answer different questions:

  * `sel_corrected` -- mean NDCG at m in (1,4,8), paired and drop-free. The direct analogue of the
    shipped SEL. On a dataset where the channel adds nothing this mostly tracks the m0 intercept, i.e.
    it selects on the TOWER's state rather than on the concept net, which is exactly why it is not
    sufficient on its own.
  * `fold_mean` -- mean paired FOLD EFFECT at m in (1,4,8): what the concept net adds with the context
    held fixed. This is the criterion the concept channel actually exists to maximise.

The interview packs do not depend on the net, so they are built ONCE per seed and reused across every
epoch -- that is what makes a 12-epoch sweep affordable.

  CASPER_DATASET=goodreads python src/instrument/a23_reselect_concept_epoch.py \
      --snapshot .cache/instrument/t2i26g_ep53.pt --prefix cvg --tag gr
"""
import os
import sys
import json
import time
import argparse

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_ROOT, "src", "baselines"))

import concept_train_v2 as CT
from concept_fold import ConceptFoldNet
import a22_concept_curve_paired as A22

CKPT_DIR = os.path.join(_ROOT, ".cache", "instrument")
OUT = os.path.join(_ROOT, "experiments", "instrument")
MS = (0, 1, 4, 8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", required=True)
    ap.add_argument("--prefix", required=True, help="checkpoint prefix, e.g. cvg or cv3")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--ask", default="broad")
    # Selecting the max over 12 epochs on the SAME seeds that ranked them is selection on noise.
    # --seed0 gives an independent seed set so the ranking can be replicated out of sample.
    ap.add_argument("--seed0", type=int, default=4242)
    a = ap.parse_args()

    import metrics as _M
    from train_tower_t2 import compute_head_mask, PROC as _PROC
    import concept_distill_train as CD

    S = CD._build_train_stack(argparse.Namespace(snapshot=a.snapshot, max_users=None))
    mc = CT._member_counts(S)
    head_mask, _ = compute_head_mask(_M.load_train(S["ni"], _PROC), S["ni"])

    # packs are a function of (users, rng, ask policy) only -- NOT of the net, so build once
    packsets = []
    for si in range(a.seeds):
        seed = a.seed0 + 1000 * si
        p, kept = A22.build_paired(S, mc, MS, A22.N_VAL, seed, False, a.ask, False)
        packsets.append(p)
        A22.log("[packs] seed " + str(seed) + " built, n=" + str(kept))

    ckpts = sorted(f for f in os.listdir(CKPT_DIR)
                   if f.startswith(a.prefix + "_ep") and f.endswith(".pt"))
    rows = []
    for f in ckpts:
        blob = torch.load(os.path.join(CKPT_DIR, f), map_location="cpu")
        net = ConceptFoldNet(len(S["tags"]), d=S["enc"].d_out, h=int(blob.get("hidden", 256)),
                             conc_init=S["d_c"].numpy(),
                             use_refusal=bool(blob.get("use_refusal", True)))
        net.load_state_dict(blob["net"])
        t0 = time.time()
        per_m = {m: {"full": [], "fold": []} for m in MS}
        for p in packsets:
            for m in MS:
                f_on, _ = A22.ndcg_rows(net, S, p[m], head_mask, strip=False)
                f_off, _ = A22.ndcg_rows(net, S, p[m], head_mask, strip=True)
                per_m[m]["full"].append(float(f_on.mean()))
                per_m[m]["fold"].append(float((f_on - f_off).mean()))
        agg = {str(m): {"full": float(np.mean(per_m[m]["full"])),
                        "fold": float(np.mean(per_m[m]["fold"]))} for m in MS}
        sel_c = float(np.mean([agg[str(m)]["full"] for m in (1, 4, 8)]))
        fold_m = float(np.mean([agg[str(m)]["fold"] for m in (1, 4, 8)]))
        rows.append({"ckpt": f, "epoch": int(blob.get("epoch", -1)), "nll": blob.get("nll"),
                     "per_m": agg, "sel_corrected": sel_c, "fold_mean": fold_m})
        A22.log(f + "  m0=" + format(agg["0"]["full"], ".4f") +
                "  sel_corr=" + format(sel_c, ".4f") +
                "  fold_mean=" + format(fold_m, "+.4f") +
                "  (" + format(time.time() - t0, ".0f") + "s)")

    best_sel = max(rows, key=lambda r: r["sel_corrected"])
    best_fold = max(rows, key=lambda r: r["fold_mean"])
    A22.log("[a23] best by sel_corrected : " + best_sel["ckpt"] +
            "  sel=" + format(best_sel["sel_corrected"], ".4f"))
    A22.log("[a23] best by fold_mean     : " + best_fold["ckpt"] +
            "  fold=" + format(best_fold["fold_mean"], "+.4f"))

    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, "a23_reselect_" + a.tag +
                     ("" if a.seed0 == 4242 else "_s" + str(a.seed0)) + ".json")
    json.dump({"tag": a.tag, "prefix": a.prefix, "snapshot": os.path.basename(a.snapshot),
               "ask": a.ask, "seeds": a.seeds, "rows": rows,
               "best_by_sel_corrected": best_sel["ckpt"],
               "best_by_fold_mean": best_fold["ckpt"]}, open(p, "w"), indent=2)
    A22.log("[a23] -> " + p)


if __name__ == "__main__":
    main()
