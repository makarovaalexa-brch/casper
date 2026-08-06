r"""run_tanp_graded.py -- TaNP meta-trained on RATINGS, the last arm-N native cell.

SPEC: PROTOCOL_DISLIKE_DISCARD.md section 11.1, stage (iv). TaNP is ratings-native by publication -- its
paper meta-learns over explicit ratings for rating regression -- but our ruler binarises at r>3.5, so the
existing row was trained with its rating channel pinned to a constant 1. This retrains it with the channel
carrying real values and evaluates on the arm-N pool.

CENTRED, for the reason RBMF demonstrated. TaNP mean-pools item embeddings weighted by the support value.
Handed RAW ratings on 0.5..5 every weight is POSITIVE, so a 1-star still pulls the user representation
TOWARD that item -- the "1-star as weak positive" scar of section 11.1, and a reweighted positive contract
rather than a graded one. Centring at the protocol threshold (r-3.5) makes a sub-threshold rating a
NEGATIVE weight that pushes the representation away, which is the capability the cell exists to test.
Only the centred arm is run: RBMF's raw arm also lost, so a second full meta-training run is not worth
the compute, and this is recorded rather than left implicit.

The encoder change in tanp_bestefffort.py is bit-identical on binary input, so the existing binary row and
its checkpoint are unaffected.

  python src/baselines/run_tanp_graded.py [--epochs 200]
"""
import os
import sys
import json
import time
import argparse

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, _HERE)

import metrics as M
import tanp_bestefffort as tanp
from arm_n import load_arm_n

OUTDIR = os.path.join(_ROOT, "experiments", "baselines", "arm_n")
CKPT = os.path.join(_ROOT, ".cache", "baselines", "tanp_graded_armn.pt")
THRESH = 3.5


def logln(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    os.makedirs(OUTDIR, exist_ok=True)
    with open(os.path.join(OUTDIR, "tanp_graded.log"), "a") as f:
        f.write(line + "\n")


def centred(g):
    c = g.copy()
    c.data = c.data.astype(np.float64) - THRESH
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--max_minutes", type=float, default=1e9)
    a0 = ap.parse_args()

    D = load_arm_n(log=logln)
    ni = D["n_items"]

    g_train_c = centred(D["g_train"])
    g_te_tr_c = centred(D["g_te_tr"])
    neg = float((g_train_c.data < 0).mean())
    logln(f"graded+centred train: {g_train_c.nnz} nnz, {neg:.1%} negative "
          f"(binary train was {D['train'].nnz} nnz, all +1)")

    a = tanp._defaults()
    a.epochs, a.max_minutes = a0.epochs, a0.max_minutes
    predict = tanp.fit(g_train_c, ni, evaluator=None, args=a, ckpt=CKPT, log=logln)

    r = M.evaluate(predict, g_te_tr_c, D["te_te"], batch_size=500,
                   head_mask=D["head_mask"], mask_X=D["pool"])
    res = {"model": "tanp", "arm": "N", "contract": "graded ratings centred at %.1f" % THRESH,
           "ndcg@10": r["ndcg@10"], "tail_ndcg@10": r["tail_ndcg@10"], "ndcg@100": r["ndcg@100"],
           "epochs_requested": a0.epochs}
    logln(f"[tanp graded arm N] full={r['ndcg@10']:.4f} tail={r['tail_ndcg@10']:.4f}")

    bp = os.path.join(OUTDIR, "tanp.json")
    if os.path.exists(bp):
        b = json.load(open(bp))["N"]
        res["binary_arm_n"] = {"ndcg@10": b["ndcg@10"], "tail_ndcg@10": b["tail_ndcg@10"]}
        logln(f"[contrast] graded minus binary, arm N: "
              f"{r['ndcg@10'] - b['ndcg@10']:+.4f} full, "
              f"{r['tail_ndcg@10'] - b['tail_ndcg@10']:+.4f} tail")

    out = os.path.join(OUTDIR, "tanp_graded_armn.json")
    json.dump(res, open(out, "w"), indent=2)
    logln(f"done -> {out}")


if __name__ == "__main__":
    main()
