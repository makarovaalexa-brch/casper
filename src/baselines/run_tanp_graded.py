r"""run_tanp_graded.py -- TaNP meta-trained on RATINGS, the last arm-N native cell.

SPEC: PROTOCOL_DISLIKE_DISCARD.md section 11.1, stage (iv). TaNP is ratings-native by publication -- its
paper meta-learns over explicit ratings for rating regression -- but our ruler binarises at r>3.5, so the
existing row was trained with its rating channel pinned to a constant 1. This retrains it with the channel
carrying real values and evaluates on the arm-N pool.

RAW RATINGS ARE THE DEFAULT AND THE NATIVE CELL. An earlier version of this script centred at the
protocol threshold (r-3.5) on the reasoning that raw ratings are all positive, so a 1-star still pulls the
representation toward the item. That reasoning was wrong twice over, and the resulting row (.1745/.0695,
-0.1579 against binary) is SUPERSEDED and must not be cited.

  * ARITHMETIC. The MovieLens scale is 0.5..5.0, so r-3.5 spans [-3.0,+1.5]: a 0.5-star gets magnitude
    3.0 while a 5-star gets 1.5, i.e. dislikes outweigh likes 2:1, with total negative mass 1.15x
    positive. That is not centring, it is an asymmetric reweighting toward dislikes.
  * CONTRACT. The NATIVE column's stated purpose is that each system receives ITS OWN PUBLISHED INPUT
    CONTRACT. TaNP's is raw explicit ratings into an end-to-end network. Hand-transforming the input is
    precisely the deviation the column exists to avoid -- and our own instrument uses a BOUNDED level-to-
    value map capped at +/-1, so the centred arm handed baselines a cruder encoding than we use ourselves
    and then reported that graded input does not help them.

The mechanism the centred arm actually measured: TaNP mean-pools item embeddings WEIGHTED BY THE SUPPORT
VALUE, so signed weights let a user's likes and dislikes cancel and the context vector collapses toward a
negatively-biased zero. Under the published contract every weight is positive and that cancellation never
occurs -- which is why TaNP fell furthest of the three baselines under an encoding it never sees.
`--centre` is retained only to reproduce the superseded arm.

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
CKPT_FMT = os.path.join(_ROOT, ".cache", "baselines", "tanp_graded_armn_%s.pt")
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
    ap.add_argument("--centre", action="store_true",
                    help="Subtract the protocol threshold from every rating. NOT the native contract "
                         "and NOT the default -- see the module docstring. Kept only so the superseded "
                         "2026-08-06 arm can be reproduced.")
    a0 = ap.parse_args()
    tag = "centred" if a0.centre else "raw"

    D = load_arm_n(log=logln)
    ni = D["n_items"]

    xf = centred if a0.centre else (lambda m: m)
    g_train_c, g_te_tr_c = xf(D["g_train"]), xf(D["g_te_tr"])
    neg = float((g_train_c.data < 0).mean())
    logln(f"graded[{tag}] train: {g_train_c.nnz} nnz, {neg:.1%} negative, "
          f"range [{g_train_c.data.min():.2f},{g_train_c.data.max():.2f}] "
          f"(binary train was {D['train'].nnz} nnz, all +1)")

    # EPOCH SELECTION ON THE SAME CONTRACT WE TRAIN ON. Passing evaluator=None would be silently
    # destructive here: tanp.fit only records best_state (and only writes the checkpoint) inside
    # `if sc > state["best"]`, and with no evaluator sc is -1.0 every epoch against an initial best of
    # -1.0, so the branch never fires -- 200 epochs, no checkpoint written, no early stop possible, and
    # FINAL-epoch weights returned. The binary TaNP row is val-selected (best 0.2623 @ep98), so a
    # final-epoch graded row would not be comparable to it.
    from metrics import load_val
    from graded_data import load_graded_val
    va_tr_bin, va_te = load_val(ni, os.path.join(_ROOT, "data", "ml-25m", "proc"))
    g_va = xf(load_graded_val(va_te, log=logln))

    def ev(pred):
        m = M.evaluate(pred, g_va, va_te, batch_size=500, head_mask=D["head_mask"])
        return m["ndcg@10"], m

    a = tanp._defaults()
    a.epochs, a.max_minutes = a0.epochs, a0.max_minutes
    predict = tanp.fit(g_train_c, ni, evaluator=ev, args=a, ckpt=CKPT_FMT % tag, log=logln)
    logln(f"val-selected epoch {getattr(predict, 'best_epoch', '?')} "
          f"(val NDCG@10 {getattr(predict, 'best_val', float('nan')):.4f})")

    r = M.evaluate(predict, g_te_tr_c, D["te_te"], batch_size=500,
                   head_mask=D["head_mask"], mask_X=D["pool"])
    res = {"model": "tanp", "arm": "N", "contract": ("graded ratings centred at %.1f" % THRESH) if a0.centre else "raw catalogue ratings (published contract)",
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

    out = os.path.join(OUTDIR, "tanp_graded_armn_%s.json" % tag)
    json.dump(res, open(out, "w"), indent=2)
    logln(f"done -> {out}")


if __name__ == "__main__":
    main()
