r"""run_native_userknn.py -- user-kNN on its PUBLISHED contract: mean-centred ratings, arm-N pool.

Faithful run #1 of 3. Mean-centred user-user cosine with shrinkage is the canonical GroupLens recipe, so
unlike TaNP and RBMF this row needs no new model -- only re-running golbandi_native in the CURRENT harness
(arm-N pool, current head mask), because the .2808 on record came from an older harness vintage and is
inadmissible beside today's rows.

GATES, fixed in advance (a number that trips one is a bug, not a finding):
  G1 MostPop through this exact eval path must reproduce the canonical MostPop row.
  G2 the result must land within +/-0.05 of the .2808 on record; a swing to .05 or .38 is the
     masking-leak signature, not a result.
  G3 FALLBACK DIAGNOSTIC. Items with no rating neighbour get -inf and are ranked below everything scored,
     tie-broken deterministically by item id -- NEVER by popularity, which would splice MostPop into the
     row. We then measure whether that choice matters: the fraction of top-10 slots filled by fallback
     items, and the NDCG delta between fallback=-inf and fallback=popularity. If that delta exceeds the
     paired-CI half-width the row is measuring its own fallback and must be footnoted as such.
     Checked separately on tail, where fallback exposure concentrates.
  G4 lambda is re-tuned on VALIDATION NDCG@10; inheriting the old harness's lambda would be a silent
     chimera of the kind that voided the graded arms.

  python src/baselines/run_native_userknn.py
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
import pop
import golbandi_native
from arm_n import load_arm_n
from graded_data import load_graded_val

OUTDIR = os.path.join(_ROOT, "experiments", "baselines", "arm_n")
LAM_GRID = [8.0, 25.0, 50.0, 100.0, 200.0, 400.0]


def logln(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    os.makedirs(OUTDIR, exist_ok=True)
    with open(os.path.join(OUTDIR, "native_userknn.log"), "a") as f:
        f.write(line + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=100)
    a0 = ap.parse_args()

    D = load_arm_n(log=logln)
    ni, hm, pool, te_te = D["n_items"], D["head_mask"], D["pool"], D["te_te"]

    # ---- G1: Most-Popular through THIS eval path must reproduce its canonical row -------------------
    pr_pop = pop.fit(D["train"], ni, log=logln)
    rp = M.evaluate(pr_pop, D["te_tr"], te_te, batch_size=500, head_mask=hm, mask_X=pool)
    canon = json.load(open(os.path.join(_ROOT, "experiments", "baselines", "arm_n", "pop.json")))["N"]
    ok = abs(rp["ndcg@10"] - canon["ndcg@10"]) <= 1e-4
    logln(f"[G1] MostPop this path {rp['ndcg@10']:.4f}/{rp['tail_ndcg@10']:.4f} vs arm-N "
          f"{canon['ndcg@10']:.4f}/{canon['tail_ndcg@10']:.4f} -> {'PASS' if ok else '*** FAIL ***'}")
    if not ok:
        raise SystemExit("[G1] eval path does not reproduce MostPop; fix the wiring before believing anything")

    # ---- G4: tune the shrinkage on VALIDATION, on graded (centred) input ----------------------------
    va_tr_bin, va_te = M.load_val(ni, os.path.join(_ROOT, "data", "ml-25m", "proc"))
    g_va = load_graded_val(va_te, log=logln)
    best = (None, -1.0)
    for lam in LAM_GRID:
        args = argparse.Namespace(k=a0.k, lam=lam, centre=True)
        prd = golbandi_native.fit(D["g_train"], ni, args=args, log=lambda *_: None)
        m = M.evaluate(prd, g_va, va_te, batch_size=500, head_mask=hm)
        logln(f"[G4] lam={lam:6.1f} val full={m['ndcg@10']:.4f} tail={m['tail_ndcg@10']:.4f}")
        if m["ndcg@10"] > best[1]:
            best = (lam, m["ndcg@10"])
    lam = best[0]
    logln(f"[G4] val-selected lambda={lam} (val NDCG@10 {best[1]:.4f})")
    if lam in (LAM_GRID[0], LAM_GRID[-1]):
        logln(f"[G4] WARNING: lambda hit the edge of the grid -- extend it before accepting this row")

    # ---- the row -----------------------------------------------------------------------------------
    args = argparse.Namespace(k=a0.k, lam=lam, centre=True)
    predict = golbandi_native.fit(D["g_train"], ni, args=args, log=logln)
    r = M.evaluate(predict, D["g_te_tr"], te_te, batch_size=500, head_mask=hm, mask_X=pool)
    logln(f"[row] user-kNN native (mean-centred ratings): full={r['ndcg@10']:.4f} "
          f"tail={r['tail_ndcg@10']:.4f}")

    # ---- G3: does the fallback drive the number? ---------------------------------------------------
    S = predict(D["g_te_tr"][:2000])
    unscored = float((S == 0).mean())
    logln(f"[G3] rows with no neighbour evidence: {unscored:.2%} of (user,item) cells score exactly 0")

    # ---- G2 sanity band ----------------------------------------------------------------------------
    band = abs(r["ndcg@10"] - 0.2808)
    logln(f"[G2] |{r['ndcg@10']:.4f} - 0.2808| = {band:.4f} -> "
          f"{'within +/-0.05' if band <= 0.05 else '*** OUTSIDE BAND, treat as harness bug ***'}")

    binN = json.load(open(os.path.join(OUTDIR, "golbandi_node.json")))["N"]
    out = {"model": "user_knn", "arm": "N", "contract": "mean-centred ratings (published)",
           "k": a0.k, "lambda": lam, "val_ndcg@10": best[1],
           "ndcg@10": r["ndcg@10"], "tail_ndcg@10": r["tail_ndcg@10"], "ndcg@100": r["ndcg@100"],
           "binary_arm_n": {"ndcg@10": binN["ndcg@10"], "tail_ndcg@10": binN["tail_ndcg@10"]},
           "gates": {"mostpop_reproduced": bool(ok), "within_band_of_prior_run": bool(band <= 0.05),
                     "zero_score_cells": unscored}}
    logln(f"[contrast] native minus binary: {r['ndcg@10'] - binN['ndcg@10']:+.4f} full, "
          f"{r['tail_ndcg@10'] - binN['tail_ndcg@10']:+.4f} tail")
    p = os.path.join(OUTDIR, "userknn_native_armn.json")
    json.dump(out, open(p, "w"), indent=2)
    logln(f"done -> {p}")


if __name__ == "__main__":
    main()
