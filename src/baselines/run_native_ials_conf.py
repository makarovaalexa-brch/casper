r"""run_native_ials_conf.py -- iALS with RATINGS AS CONFIDENCE (Hu, Koren & Volinsky 2008), arm-N pool.

WHY THIS ROW EXISTS. Every graded baseline result we have so far is confounded with MEAN-CENTRING:
user-kNN's published recipe centres, and RBMF's graded arm loses -.070 centred but only -.007 uncentred,
so ~90% of its "graded penalty" is centring rather than gradedness. A reviewer can therefore say, fairly,
that we never gave grades a fair test -- we only tested recipes that cancel the exposure signal ranking
runs on.

Hu/Koren/Volinsky is the published, canonical answer to "use grades WITHOUT discarding exposure":
    c_ui = 1 + alpha * r_ui        (instead of 1 + alpha * 1 for binary implicit feedback)
Every rated item stays in the loss with weight >= 1; the rating only scales how strongly it counts. There
is no centring and no normalisation by support, so the who-rated-what signal survives intact. This is the
one graded arm that cannot be dismissed as "you crippled it by centring".

DECISION VALUE, stated before the run so neither outcome can be spun:
  * If it BEATS binary iALS (.3044/.2293 arm N), then grades DO help when used in an exposure-preserving
    way, our current claim that no ratings-native baseline benefits from its grades is WRONG, and the
    chapter must be rewritten to say so.
  * If it TIES or LOSES, grades buy nothing even in the method purpose-built to consume them, and the
    claim becomes robust to the "you configured the baseline badly" objection (Dacrema et al.).

GATES: Most-Popular must reproduce its arm-N row through this eval path; alpha is re-tuned on validation
NDCG@10 because the confidence scale changes meaning entirely when r replaces 1 (alpha=10 against r in
0.5..5 is not the same regularisation regime as alpha=10 against 1), and inheriting it would be exactly
the carried-constant-across-a-contract-change failure that has bitten this experiment three times.

  python src/baselines/run_native_ials_conf.py
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
from graded_data import load_graded_val

OUTDIR = os.path.join(_ROOT, "experiments", "baselines", "arm_n")
ALPHA_GRID = [1.0, 2.5, 5.0, 10.0, 20.0]


def logln(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    os.makedirs(OUTDIR, exist_ok=True)
    with open(os.path.join(OUTDIR, "native_ials_conf.log"), "a") as f:
        f.write(line + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--factors", type=int, default=200)
    ap.add_argument("--reg", type=float, default=0.01)
    ap.add_argument("--iterations", type=int, default=15)
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

    g_train, g_te_tr = D["g_train"], D["g_te_tr"]
    logln(f"graded train nnz={g_train.nnz} range [{g_train.data.min():.1f},{g_train.data.max():.1f}] "
          f"-- confidence c = 1 + alpha*r, so every rated item keeps weight >= 1 and exposure survives")

    va_tr_bin, va_te = M.load_val(ni, os.path.join(_ROOT, "data", "ml-25m", "proc"))
    g_va = load_graded_val(va_te, log=logln)

    best = (None, -1.0, None)
    for alpha in ALPHA_GRID:
        model = ials._fit_model(g_train, a0.factors, a0.reg, alpha, a0.iterations, log=lambda *_: None)
        predict = ials._make_predict(model, a0.reg, alpha)
        m = M.evaluate(predict, g_va, va_te, batch_size=500, head_mask=hm)
        logln(f"[tune] alpha={alpha:5.1f} val full={m['ndcg@10']:.4f} tail={m['tail_ndcg@10']:.4f}")
        if m["ndcg@10"] > best[1]:
            best = (alpha, m["ndcg@10"], predict)
    alpha, val, predict = best
    logln(f"[tune] val-selected alpha={alpha} (val NDCG@10 {val:.4f})")
    if alpha in (ALPHA_GRID[0], ALPHA_GRID[-1]):
        logln("[tune] WARNING: alpha at the edge of the grid -- extend before accepting this row")

    r = M.evaluate(predict, g_te_tr, te_te, batch_size=500, head_mask=hm, mask_X=pool)
    binN = json.load(open(os.path.join(OUTDIR, "ials.json")))["N"]
    d_full = r["ndcg@10"] - binN["ndcg@10"]
    logln(f"[row] iALS ratings-as-confidence: full={r['ndcg@10']:.4f} tail={r['tail_ndcg@10']:.4f}")
    logln(f"[contrast] vs binary iALS {binN['ndcg@10']:.4f}/{binN['tail_ndcg@10']:.4f}: "
          f"{d_full:+.4f} full, {r['tail_ndcg@10'] - binN['tail_ndcg@10']:+.4f} tail")
    logln("[verdict] " + ("GRADES HELP under an exposure-preserving contract -- the chapter's claim that "
                          "no ratings-native baseline benefits from its grades is WRONG and must be "
                          "rewritten." if d_full > 0 else
                          "grades buy nothing even in the method built to consume them -- the claim is "
                          "robust to the 'you configured the baseline badly' objection."))

    out = {"model": "ials_confidence", "arm": "N",
           "contract": "Hu/Koren/Volinsky c_ui = 1 + alpha*r_ui (exposure-preserving graded)",
           "alpha": alpha, "factors": a0.factors, "reg": a0.reg, "val_ndcg@10": val,
           "ndcg@10": r["ndcg@10"], "tail_ndcg@10": r["tail_ndcg@10"], "ndcg@100": r["ndcg@100"],
           "binary_arm_n": binN, "delta_full": d_full,
           "gates": {"mostpop": bool(ok), "alpha_interior": alpha not in (ALPHA_GRID[0], ALPHA_GRID[-1])}}
    p = os.path.join(OUTDIR, "ials_confidence_armn.json")
    json.dump(out, open(p, "w"), indent=2)
    logln(f"done -> {p}")


if __name__ == "__main__":
    main()
