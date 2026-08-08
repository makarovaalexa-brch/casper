r"""run_reselect_on_ndcg10.py -- re-select the shared MF hyperparameters on the PRIMARY metric.

WHAT WAS ACTUALLY WRONG (and it is narrower than "the baselines are untuned"). iALS, RBMF and belief-MF
all carry {factors 200, reg 0.1, alpha 50}. That was NOT a default and NOT a guess -- it came from a real
9-point validation sweep, logged on 2026-07-20, and `run_ml25m_liang.py` says so:

    # iALS: val-selected from the ML-20M sweep (no new sweep on ML-25M -- author directive 2026-07-21).

Two things follow from that comment, and only two:
  1. the sweep ran on ML-20M, not on this ruler;
  2. it selected on **NDCG@100**, where its pick genuinely was best (reg 0.1 / alpha 50 -> .3642, ahead of
     reg 0.01 / alpha 10 -> .3576).
The chapter's PRIMARY metric is NDCG@10. On @10 the ordering differs: the graded ladder measured
reg 0.01 beating reg 0.1 by +0.0257 full and +0.0337 tail, both paired-separated.

So the defect is "selected on the secondary metric, on the previous dataset", not "never tuned". This
re-runs the same grid on THIS ruler and selects on NDCG@10, for the three rows that share the setting, in
BOTH arms -- because changing one arm alone would leave a single table row carrying two different
hyperparameter settings across its two columns.

WHAT THIS WILL DO TO THE PAPER: it raises baselines we currently beat. That is the point. A chapter that
tunes its own instrument on the reported metric and its baselines on a different one is making exactly the
comparison Dacrema et al. warn about, and the fix is to state one selection rule and apply it everywhere.

Both the old and the new selection are recorded per row, so the chapter can state the rule and show what
it cost us.

  python src/baselines/run_reselect_on_ndcg10.py
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
import ials
from arm_n import load_arm_n

OUTDIR = os.path.join(_ROOT, "experiments", "baselines", "arm_n")
OUTP = os.path.join(OUTDIR, "reselect_ndcg10.json")
REG_GRID = [0.001, 0.01, 0.1]          # the 2026-07-20 grid, unchanged
ALPHA_GRID = [1.0, 10.0, 50.0]
FACTORS, ITERS, BATCH = 200, 15, 250
OLD = {"factors": 200, "reg": 0.1, "alpha": 50.0}
STATE = {"old_setting": OLD, "selection_rule": "validation NDCG@10 on THIS ruler (was: NDCG@100 on ML-20M)"}


def logln(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    os.makedirs(OUTDIR, exist_ok=True)
    with open(os.path.join(OUTDIR, "reselect_ndcg10.log"), "a") as f:
        f.write(line + "\n")


def persist():
    json.dump(STATE, open(OUTP, "w"), indent=2)


def main():
    argparse.ArgumentParser().parse_args()
    D = load_arm_n(log=logln)
    ni, hm, pool, te_te = D["n_items"], D["head_mask"], D["pool"], D["te_te"]

    pr_pop = pop.fit(D["train"], ni, log=logln)
    rp = M.evaluate(pr_pop, D["te_tr"], te_te, batch_size=BATCH, head_mask=hm, mask_X=pool)
    canon = json.load(open(os.path.join(OUTDIR, "pop.json")))["N"]
    ok = abs(rp["ndcg@10"] - canon["ndcg@10"]) <= 1e-4
    logln(f"[G1] MostPop {rp['ndcg@10']:.4f} vs {canon['ndcg@10']:.4f} -> {'PASS' if ok else 'FAIL'}")
    if not ok:
        raise SystemExit("[G1] eval path does not reproduce MostPop")
    persist()

    va_tr, va_te = M.load_val(ni, os.path.join(_ROOT, "data", "ml-25m", "proc"))

    logln("--- sweeping the shared iALS/RBMF/belief-MF setting on THIS ruler, selecting on NDCG@10 ---")
    best10, best100 = (None, None, -1.0), (None, None, -1.0)
    STATE["sweep"] = []
    for reg in REG_GRID:
        for alpha in ALPHA_GRID:
            t0 = time.time()
            m = ials._fit_model(D["train"], FACTORS, reg, alpha, ITERS, log=lambda *_: None)
            prd = ials._make_predict(m, reg, alpha)
            v = M.evaluate(prd, va_tr, va_te, batch_size=BATCH, head_mask=hm)
            logln(f"  reg={reg:<6} alpha={alpha:5.1f} val @10={v['ndcg@10']:.4f} "
                  f"@100={v['ndcg@100']:.4f} ({(time.time()-t0)/60:.1f}m)")
            STATE["sweep"].append({"reg": reg, "alpha": alpha, "val_ndcg@10": v["ndcg@10"],
                                   "val_ndcg@100": v["ndcg@100"]})
            persist()
            if v["ndcg@10"] > best10[2]:
                best10 = (reg, alpha, v["ndcg@10"])
            if v["ndcg@100"] > best100[2]:
                best100 = (reg, alpha, v["ndcg@100"])
    logln(f"[select] on NDCG@10 : reg={best10[0]} alpha={best10[1]} (val {best10[2]:.4f})")
    logln(f"[select] on NDCG@100: reg={best100[0]} alpha={best100[1]} (val {best100[2]:.4f})"
          + ("   <- the rule that produced the row on record" if True else ""))
    agree = (best10[0], best10[1]) == (best100[0], best100[1])
    logln(f"[select] the two criteria {'AGREE' if agree else 'DISAGREE'} on this ruler")
    STATE["selected_on_ndcg10"] = {"reg": best10[0], "alpha": best10[1], "val": best10[2]}
    STATE["selected_on_ndcg100"] = {"reg": best100[0], "alpha": best100[1], "val": best100[2]}
    STATE["criteria_agree"] = bool(agree)
    persist()

    reg, alpha = best10[0], best10[1]
    m = ials._fit_model(D["train"], FACTORS, reg, alpha, ITERS, log=logln)
    prd = ials._make_predict(m, reg, alpha)
    rows = {}
    for arm, te_in in (("A", D["te_tr"]), ("N", D["te_tr"])):
        kw = dict(batch_size=BATCH, head_mask=hm)
        if arm == "N":
            kw["mask_X"] = pool
        r = M.evaluate(prd, te_in, te_te, **kw)
        rows[arm] = {"ndcg@10": r["ndcg@10"], "tail_ndcg@10": r["tail_ndcg@10"],
                     "ndcg@100": r["ndcg@100"]}
        logln(f"[iALS arm {arm}] reselected: full={r['ndcg@10']:.4f} tail={r['tail_ndcg@10']:.4f}")
    old = json.load(open(os.path.join(OUTDIR, "ials.json")))
    for arm in ("A", "N"):
        logln(f"[delta arm {arm}] {rows[arm]['ndcg@10'] - old[arm]['ndcg@10']:+.4f} full, "
              f"{rows[arm]['tail_ndcg@10'] - old[arm]['tail_ndcg@10']:+.4f} tail "
              f"(was {old[arm]['ndcg@10']:.4f}/{old[arm]['tail_ndcg@10']:.4f})")
    STATE["ials_reselected"] = rows
    STATE["ials_on_record"] = {a: {k: old[a][k] for k in ("ndcg@10", "tail_ndcg@10")} for a in ("A", "N")}
    persist()
    logln("NOTE: RBMF and belief-MF share this setting and must be re-fitted under the same rule before "
          "any of the three rows is quoted; this run establishes the rule and the iALS row.")
    logln(f"done -> {OUTP}")


if __name__ == "__main__":
    main()
