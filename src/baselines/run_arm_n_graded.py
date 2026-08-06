r"""run_arm_n_graded.py -- the arm-N NATIVE cell for RBMF: a ridge seed on CENTRED graded targets.

SPEC: PROTOCOL_DISLIKE_DISCARD.md section 11.1, stage (iv). RBMF/Functional-MF is ratings-native by
publication, so its arm-N input is its real rated history rather than the binarised likes. run_arm_n.py
deliberately does not cover it -- that runner is re-eval-only for models whose published contract IS
binary implicit feedback.

WHY CENTRING IS THE WHOLE POINT, and why the 2026-07-29 attempt was void for a second reason beyond its
pool bug. rbmf_seed regresses the fold-in VALUES on the frozen item basis (`rhs = Ys.T @ Xc.data`).
Handed RAW ratings on 0.5..5 every target is POSITIVE, so a 1-star review pulls the seed TOWARD that
region -- it reads as a weak endorsement. That is the "1-star as weak positive" scar section 11.1 names, and
it is not a graded contract at all, merely a reweighted positive one. Centring at the protocol threshold
(r - 3.5, the same boundary Liang's filter and our own sign convention use) is what turns a sub-threshold
rating into a NEGATIVE target that pulls the seed away. We therefore report the CENTRED arm as the
native cell and the raw arm only as a diagnostic, because the difference between them is exactly the
capability the row is supposed to exercise.

POOL. The arm-N pool excludes every rated fold-in item, and the graded fold-in support IS that set
(1,411,761 nnz both). So input support and mask coincide here and the input/mask coupling that voided the
July run cannot arise; we still pass the pool explicitly rather than relying on that coincidence.

CAVEAT TO CARRY INTO THE WRITE-UP, stated in rbmf_seed itself: the frozen basis Y is fitted on BINARY
data, so the latent space has no dislike structure. That caps what a negative target can buy and is a
property of the method as published, not of this harness.

  python src/baselines/run_arm_n_graded.py
"""
import os
import sys
import json
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, _HERE)

import metrics as M
import rbmf_seed
from arm_n import load_arm_n

OUTDIR = os.path.join(_ROOT, "experiments", "baselines", "arm_n")
THRESH = 3.5          # the protocol's own like/dislike boundary


def logln(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(os.path.join(OUTDIR, "rbmf_graded.log"), "a") as f:
        f.write(line + "\n")


def centred(g):
    """Copy of the graded fold-in with values re-centred on the protocol threshold."""
    c = g.copy()
    c.data = (c.data.astype(np.float64) - THRESH)
    return c


def main():
    D = load_arm_n(log=logln)
    ni = D["n_items"]
    g_te_tr, pool, te_te, hm = D["g_te_tr"], D["pool"], D["te_te"], D["head_mask"]

    # Frozen basis from the CERTIFIED binary fit -- the published method is a fold-in onto a frozen
    # basis, so only the seed target moves. iALS is now seeded (ials.py), so this is reproducible.
    predict = rbmf_seed.fit(D["train"], ni, log=logln)

    g_c = centred(g_te_tr)
    neg = float((g_c.data < 0).mean())
    logln(f"centred at r-{THRESH}: {neg:.1%} of fold-in targets are now NEGATIVE "
          f"(raw ratings would have made every one of them positive)")

    res = {}
    for tag, X in (("centred", g_c), ("raw_diagnostic", g_te_tr)):
        r = M.evaluate(predict, X, te_te, batch_size=500, head_mask=hm, mask_X=pool)
        res[tag] = {"ndcg@10": r["ndcg@10"], "tail_ndcg@10": r["tail_ndcg@10"],
                    "ndcg@100": r["ndcg@100"]}
        logln(f"[{tag:14s}] full={r['ndcg@10']:.4f} tail={r['tail_ndcg@10']:.4f} "
              f"@100={r['ndcg@100']:.4f}")

    # The binary arm-N row from run_arm_n.py, for the contrast the cell exists to draw.
    bpath = os.path.join(OUTDIR, "rbmf_seed.json")
    if os.path.exists(bpath):
        b = json.load(open(bpath))["N"]
        res["binary_arm_n"] = {"ndcg@10": b["ndcg@10"], "tail_ndcg@10": b["tail_ndcg@10"]}
        d = res["centred"]["ndcg@10"] - b["ndcg@10"]
        logln(f"[contrast] centred graded minus binary, arm N: {d:+.4f} full")

    res["_meta"] = {"model": "rbmf_seed", "arm": "N", "threshold": THRESH,
                    "native_cell": "centred",
                    "note": "Frozen Y from the binary iALS fit (seeded). Raw arm is a diagnostic only: "
                            "uncentred ratings make every target positive, so a 1-star reads as a weak "
                            "endorsement rather than a dislike."}
    out = os.path.join(OUTDIR, "rbmf_seed_graded_armn.json")
    json.dump(res, open(out, "w"), indent=2)
    logln(f"done -> {out}")


if __name__ == "__main__":
    main()
