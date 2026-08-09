r"""run_foldin_weight_vs_estimand.py -- the mechanism test, scoped to the fold-in and made tractable.

WHY THIS REPLACES run_weight_vs_estimand.py. That script trained an explicit-feedback MF from scratch to
contrast with iALS. The contrast was right; the implementation was not. It solved a 200-dimensional ridge
per user per iteration in a pure Python loop over 140,768 training users, where every other model in this
project delegates ALS to the compiled `implicit` library. It ran 17 hours without finishing a single arm
and logged nothing while doing it, because the logging sat after the fit rather than inside it. Killed
2026-08-09.

WHAT THIS MEASURES INSTEAD. Freeze ONE item basis -- the certified iALS factorisation, the same Y that
RBMF and belief-MF already fold onto -- and vary ONLY how a held-out user's answers become a user vector:

  arm W  WEIGHT    the iALS confidence solve. The rating scales how strongly each observation counts;
                   the thing being fitted is still "did this user consume it", target 1.
  arm E  ESTIMAND  ridge least squares of the user's ACTUAL RATINGS on the same Y, support items only
                   (this is exactly RBMF's seed rule, which is why RBMF is the natural comparison).
  arm Eb CONTROL   the same ridge LS regressing a CONSTANT on the same support. Separates "the estimand
                   formulation is weak" from "the values are doing the work inside it".

Same basis, same 200 factors, same fold-in support, same pool, same metric. The only thing that moves is
the role the rating plays. If W beats E, the weight/estimand distinction survives with the model held
fixed, which is the objection the chapter cannot currently answer -- that iALS simply is the better model.

WHAT IT DOES NOT TEST, and the chapter must say so: this varies the fold-in rule, not the training
objective. A method whose *training* regresses ratings (TaNP as published) is a stronger version of the
same idea and is reported separately at 0.0060.

Ridge strength for E and Eb is val-selected; alpha for W is val-selected. Nothing is inherited.

  python src/baselines/run_foldin_weight_vs_estimand.py
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
OUTP = os.path.join(OUTDIR, "foldin_weight_vs_estimand.json")
FACTORS, ITERS, BATCH = 200, 15, 250
BASIS_ALPHA, BASIS_REG = 10.0, 0.1     # the re-selected setting (NDCG@10 rule, 2026-08-08)
W_ALPHA_GRID = [1.0, 2.5, 5.0, 10.0]
RIDGE_GRID = [0.1, 1.0, 10.0, 100.0]
STATE = {}


def logln(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    os.makedirs(OUTDIR, exist_ok=True)
    with open(os.path.join(OUTDIR, "foldin_weight_vs_estimand.log"), "a") as f:
        f.write(line + "\n")


def persist():
    json.dump(STATE, open(OUTP, "w"), indent=2)


def make_ridge_predict(Y, ridge, use_values):
    """Fold in by ridge LS on the SUPPORT items only: min_x |Y_S x - t|^2 + ridge|x|^2.

    use_values=True  -> t is the user's real ratings   (the ESTIMAND arm)
    use_values=False -> t is a constant 1 on the same support (the CONTROL arm)

    Support-only, unweighted, no term for the unobserved items: that is what makes it an estimand rule
    rather than a confidence rule, and it is RBMF's published seed.
    """
    f = Y.shape[1]
    I = (ridge * np.eye(f)).astype(np.float32)
    YT = np.ascontiguousarray(Y.T)          # (f x m); the score Y @ x is a single gemv per user
    from scipy.linalg import cho_factor, cho_solve

    def predict(X_csr):
        X = X_csr.tocsr().astype(np.float32)
        out = np.zeros((X.shape[0], Y.shape[0]), dtype=np.float32)
        for u in range(X.shape[0]):
            lo, hi = X.indptr[u], X.indptr[u + 1]
            if lo == hi:
                continue
            idx = X.indices[lo:hi]
            t = X.data[lo:hi] if use_values else np.ones(hi - lo, dtype=np.float32)
            Ys = Y[idx]
            # Cholesky rather than a general solve: the normal-equations matrix is symmetric positive
            # definite by construction (Gram + positive ridge), and cho_solve is ~2x the general LU path.
            G = Ys.T @ Ys
            G += I
            x = cho_solve(cho_factor(G, lower=True, check_finite=False), Ys.T @ t, check_finite=False)
            out[u] = x @ YT
        return out
    return predict


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
    STATE["gates"] = {"mostpop": bool(ok)}
    persist()

    va_tr, va_te = M.load_val(ni, os.path.join(_ROOT, "data", "ml-25m", "proc"))
    g_va = load_graded_val(va_te, log=logln)
    g_te = D["g_te_tr"]

    logln(f"[basis] freezing ONE iALS item basis (f={FACTORS}, reg={BASIS_REG}, alpha={BASIS_ALPHA}) "
          "-- every arm below folds onto this same Y")
    t0 = time.time()
    base = ials._fit_model(D["train"], FACTORS, BASIS_REG, BASIS_ALPHA, ITERS, log=logln)
    Y = np.asarray(base.item_factors, dtype=np.float32)
    logln(f"[basis] done in {(time.time()-t0)/60:.1f}m; Y is {Y.shape}")
    STATE["basis"] = {"factors": FACTORS, "reg": BASIS_REG, "alpha": BASIS_ALPHA}
    persist()

    # ---- arm W: the confidence fold-in (the rating as a WEIGHT) --------------------------------------
    logln("--- arm W: iALS confidence fold-in, rating scales the weight, target 1 ---")
    bw = (None, -1.0, None)
    for a in W_ALPHA_GRID:
        prd = ials._make_predict(base, BASIS_REG, a)
        v = M.evaluate(prd, g_va, va_te, batch_size=BATCH, head_mask=hm)
        logln(f"  [W] alpha={a:5.1f} val full={v['ndcg@10']:.4f}")
        if v["ndcg@10"] > bw[1]:
            bw = (a, v["ndcg@10"], prd)
    aW, valW, prdW = bw
    rW = M.evaluate(prdW, g_te, te_te, batch_size=BATCH, head_mask=hm, mask_X=pool)
    logln(f"[W] alpha={aW} -> TEST full={rW['ndcg@10']:.4f} tail={rW['tail_ndcg@10']:.4f}")
    STATE["W_weight"] = {"alpha": aW, "val": valW, "ndcg@10": rW["ndcg@10"],
                         "tail_ndcg@10": rW["tail_ndcg@10"]}
    persist()

    # ---- arms E and Eb: the ridge fold-in, with and without the values ------------------------------
    for tag, use_vals, note in (("E_estimand", True, "ridge LS regressing the user's RATINGS (RBMF's seed rule)"),
                                ("Eb_control", False, "the same ridge LS regressing a CONSTANT")):
        logln(f"--- arm {tag}: {note} ---")
        be = (None, -1.0, None)
        for ridge in RIDGE_GRID:
            prd = make_ridge_predict(Y, ridge, use_vals)
            v = M.evaluate(prd, g_va, va_te, batch_size=BATCH, head_mask=hm)
            logln(f"  [{tag}] ridge={ridge:6.1f} val full={v['ndcg@10']:.4f}")
            if v["ndcg@10"] > be[1]:
                be = (ridge, v["ndcg@10"], prd)
        rg, valE, prdE = be
        rE = M.evaluate(prdE, g_te, te_te, batch_size=BATCH, head_mask=hm, mask_X=pool)
        logln(f"[{tag}] ridge={rg} -> TEST full={rE['ndcg@10']:.4f} tail={rE['tail_ndcg@10']:.4f}")
        STATE[tag] = {"ridge": rg, "val": valE, "ndcg@10": rE["ndcg@10"],
                      "tail_ndcg@10": rE["tail_ndcg@10"], "note": note}
        persist()

    d = STATE["W_weight"]["ndcg@10"] - STATE["E_estimand"]["ndcg@10"]
    dv = STATE["E_estimand"]["ndcg@10"] - STATE["Eb_control"]["ndcg@10"]
    logln("=" * 78)
    logln(f"[verdict] WEIGHT minus ESTIMAND, one frozen basis, only the fold-in rule varying: {d:+.4f}")
    logln("[verdict] " + ("mechanism SURVIVES with the model held fixed -- the role of the rating, not "
                          "the model family, separates the rows." if d > 0.005 else
                          "mechanism NOT supported once the basis is held fixed. The four-row pattern is "
                          "about model family. Delete the paragraph."))
    logln(f"[control] values INSIDE the estimand rule are worth {dv:+.4f} (E minus Eb)")
    logln("=" * 78)
    STATE["verdict"] = {"weight_minus_estimand": d, "values_within_estimand": dv,
                        "mechanism_supported": bool(d > 0.005),
                        "scope": "fold-in rule only; does NOT vary the training objective"}
    persist()
    logln(f"done -> {OUTP}")


if __name__ == "__main__":
    main()
