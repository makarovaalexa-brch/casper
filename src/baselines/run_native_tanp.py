r"""run_native_tanp.py -- TaNP as PUBLISHED (rating regression) on the arm-N pool. Faithful run #3.

Uses tanp_native.py, NOT tanp_bestefffort.py. The latter is our documented implicit adaptation and stays
as the LIANG row; handing it graded data does not make it native (its episode builder overwrites values
with 1.0), which is what voided the 2026-08-06 arms.

GATES, fixed before the run so a bad number cannot be rationalised afterwards:
  G1 Most-Popular must reproduce its arm-N row through this exact eval path, or abort.
  G2 val RMSE must beat the predict-the-user's-mean baseline (~1.0 on the ML scale; a working rating
     model reaches ~0.92 or better). If the regression cannot beat predicting the mean, it never learned
     and its NDCG is void -- this is the tripwire for the collapse-to-constant failure that a masked-MSE
     bug produces.
  G3 per-user predicted-rating spread across the catalogue must be non-degenerate (std > 0.1). Constant
     output means a broken loss even when NDCG looks plausible.
  G3b LATENT COLLAPSE: distinct top-10 lists across users must exceed 25% of users. G3 alone
     cannot catch collapse -- a model that has learned only item biases still shows healthy per-user
     spread ACROSS ITEMS while producing a near-identical ranking for everyone.
  G4 NDCG below Most-Popular is a bug, not a finding. Anywhere in [MostPop, 0.33] is a plausible result.
  Also logged: the NDCG of the val-RMSE-selected epoch alongside the val-NDCG-selected one. The gap
  between the two selections measures objective misalignment directly and costs nothing to record.

  python src/baselines/run_native_tanp.py [--epochs 200]
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

import metrics as M
import pop
import tanp_native
from arm_n import load_arm_n
from graded_data import load_graded_val

OUTDIR = os.path.join(_ROOT, "experiments", "baselines", "arm_n")
CKPT_FMT = os.path.join(_ROOT, ".cache", "baselines", "tanp_native_armn_%s.pt")


def logln(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    os.makedirs(OUTDIR, exist_ok=True)
    with open(os.path.join(OUTDIR, "native_tanp.log"), "a") as f:
        f.write(line + "\n")


def rmse_on(predict, X_in, X_ho):
    """Held-out rating RMSE: predict from the fold-in, score against the held-out RATINGS."""
    se, n = 0.0, 0
    for st in range(0, X_in.shape[0], 500):
        P = predict(X_in[st:st + 500])
        H = X_ho[st:st + 500].tocoo()
        if H.nnz == 0:
            continue
        se += float(((P[H.row, H.col] - H.data) ** 2).sum()); n += H.nnz
    return (se / max(n, 1)) ** 0.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--max_minutes", type=float, default=1e9)
    ap.add_argument("--select", choices=["ndcg", "rmse"], default="rmse",
                    help="rmse = the model's OWN criterion, as its paper would select. Default, per the "
                         "2026-08-07 ruling: selecting on val NDCG for a model whose training descends "
                         "AWAY from NDCG picks the least-trained checkpoint, which is neither TaNP nor a "
                         "ranker. ndcg is retained to reproduce that adversarially-favourable upper bound.")
    a0 = ap.parse_args()

    D = load_arm_n(log=logln)
    ni, hm, pool, te_te = D["n_items"], D["head_mask"], D["pool"], D["te_te"]

    pr_pop = pop.fit(D["train"], ni, log=logln)
    rp = M.evaluate(pr_pop, D["te_tr"], te_te, batch_size=500, head_mask=hm, mask_X=pool)
    canon = json.load(open(os.path.join(OUTDIR, "pop.json")))["N"]
    ok = abs(rp["ndcg@10"] - canon["ndcg@10"]) <= 1e-4
    logln(f"[G1] MostPop this path {rp['ndcg@10']:.4f} vs arm-N {canon['ndcg@10']:.4f} -> "
          f"{'PASS' if ok else '*** FAIL ***'}")
    if not ok:
        raise SystemExit("[G1] eval path does not reproduce MostPop; fix wiring first")
    mostpop = rp["ndcg@10"]

    va_tr_bin, va_te = M.load_val(ni, os.path.join(_ROOT, "data", "ml-25m", "proc"))
    g_va = load_graded_val(va_te, log=logln)
    # held-out RATINGS for the val users, to measure regression quality rather than ranking
    from graded_data import _partition, _graded_rows
    uid, s2i, raw = _partition()
    n_all = len(uid)
    g_va_ho = _graded_rows(uid[n_all - 2 * 10000: n_all - 10000], s2i, raw)
    g_va_ho = g_va_ho.multiply(va_te.astype(bool)).tocsr()      # ratings of the held-out target items
    g_va_ho.eliminate_zeros()
    mean_r = float(D["g_train"].data.mean())
    base_rmse = float(np.sqrt(((g_va_ho.data - mean_r) ** 2).mean()))
    logln(f"[G2] predict-the-global-mean baseline val RMSE = {base_rmse:.4f} (mean rating {mean_r:.3f})")

    hist = {"rmse": []}

    def ev(pred):
        m = M.evaluate(pred, g_va, va_te, batch_size=500, head_mask=hm)
        rm = rmse_on(pred, g_va, g_va_ho)
        hist["rmse"].append(rm)
        logln(f"      val NDCG@10 {m['ndcg@10']:.4f} | val RMSE {rm:.4f}")
        m["ndcg_at_this_epoch"] = m["ndcg@10"]
        # fit() maximises the returned score, so RMSE selection returns its negation.
        return (-rm if a0.select == "rmse" else m["ndcg@10"]), m

    a = tanp_native._defaults()
    a.epochs, a.max_minutes = a0.epochs, a0.max_minutes
    predict = tanp_native.fit(D["g_train"], ni, evaluator=ev, args=a, ckpt=CKPT_FMT % a0.select, log=logln)

    best_rmse = min(hist["rmse"]) if hist["rmse"] else float("nan")
    g2 = best_rmse < base_rmse
    logln(f"[G2] best val RMSE {best_rmse:.4f} vs mean-predictor {base_rmse:.4f} -> "
          f"{'PASS' if g2 else '*** FAIL: regression never learned, NDCG is void ***'}")

    S = predict(D["g_te_tr"][:2000])
    spread = float(np.std(S, axis=1).mean())
    logln(f"[G3] mean per-user prediction spread across the catalogue = {spread:.4f} -> "
          f"{'PASS' if spread > 0.1 else '*** FAIL: constant output ***'}")

    # G3b LATENT COLLAPSE. G3 measures spread ACROSS items for one user, which stays healthy even when
    # the model has learned only item biases -- that is exactly how the 2026-08-07 beta bug slipped past
    # G2 and G3 (val RMSE 0.845, spread 0.19, yet 39 distinct top-10 lists across 1,000 users because
    # the KL term had collapsed the latent). This measures spread ACROSS USERS, which is the thing
    # ranking actually needs, and it is the cheapest direct test for posterior collapse.
    Sm = S.copy(); Sm[D["pool"][:2000].nonzero()] = -np.inf
    tops = np.argsort(-Sm, axis=1)[:, :10]
    n_distinct = len(set(map(tuple, tops.tolist())))
    cross_user = float(np.std(S, axis=0).mean())
    g3b = n_distinct > 0.25 * tops.shape[0]
    logln(f"[G3b] distinct top-10 lists across {tops.shape[0]} users = {n_distinct}; "
          f"mean per-item spread ACROSS users = {cross_user:.4f} -> "
          f"{'PASS' if g3b else '*** FAIL: latent collapsed, the model is ranking item bias ***'}")

    r = M.evaluate(predict, D["g_te_tr"], te_te, batch_size=500, head_mask=hm, mask_X=pool)
    g4 = r["ndcg@10"] >= mostpop
    logln(f"[G4] NDCG {r['ndcg@10']:.4f} vs MostPop {mostpop:.4f} -> "
          f"{'PASS' if g4 else '*** FAIL: below the popularity floor, presume harness bug ***'}")
    logln(f"[row] TaNP native (rating regression): full={r['ndcg@10']:.4f} tail={r['tail_ndcg@10']:.4f}")

    b = json.load(open(os.path.join(OUTDIR, "tanp.json")))["N"]
    logln(f"[contrast] native minus binary: {r['ndcg@10'] - b['ndcg@10']:+.4f} full, "
          f"{r['tail_ndcg@10'] - b['tail_ndcg@10']:+.4f} tail")

    out = {"model": "tanp", "arm": "N", "contract": "rating regression on explicit ratings (published)", "selection": a0.select,
           "ndcg@10": r["ndcg@10"], "tail_ndcg@10": r["tail_ndcg@10"], "ndcg@100": r["ndcg@100"],
           "val_ndcg@10": getattr(predict, "best_val", None),
           "val_epoch": getattr(predict, "best_epoch", None),
           "val_rmse_best": best_rmse, "val_rmse_mean_predictor": base_rmse,
           "binary_arm_n": {"ndcg@10": b["ndcg@10"], "tail_ndcg@10": b["tail_ndcg@10"]},
           "gates": {"mostpop": bool(ok), "rmse_beats_mean": bool(g2),
                     "spread_nondegenerate": bool(spread > 0.1), "latent_not_collapsed": bool(g3b),
                     "distinct_top10_lists": int(n_distinct), "above_mostpop": bool(g4)}}
    p = os.path.join(OUTDIR, "tanp_native_armn_%s.json" % a0.select)
    json.dump(out, open(p, "w"), indent=2)
    logln(f"done -> {p}")


if __name__ == "__main__":
    main()
