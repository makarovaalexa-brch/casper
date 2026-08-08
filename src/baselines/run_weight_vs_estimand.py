r"""run_weight_vs_estimand.py -- the within-model test of the chapter's mechanism claim.

THE CLAIM. sec:nativemech argues that a rating helps when it enters as a WEIGHT on an observation and
hurts when it becomes the ESTIMAND the model predicts. The evidence for it is four rows: user-kNN, RBMF
and TaNP all regress or centre the rating and all lose; iALS uses it as confidence and gains.

WHY THAT EVIDENCE IS NOT ENOUGH. Model family is perfectly confounded with the category. iALS is a global
200-factor matrix factorisation; the three losers are a neighbourhood method, a ridge fold onto a frozen
binary basis, and a neural process. "iALS is simply the better model on this ruler" explains the same four
observations exactly as well, and it is the first thing a referee will say. One positive case cannot carry
a two-category mechanism.

THE TEST. Hold the model family fixed and vary ONLY the role of the rating.

  arm W  WEIGHT   c_ui = 1 + alpha*r_ui on the observed support, target 1  (Hu/Koren/Volinsky; = r3)
  arm E  ESTIMAND same ALS machinery, same factors/iterations, but the least-squares target is the
                  RATING ITSELF on observed entries, unobserved entries carrying weight 0 -- i.e. classic
                  explicit-feedback MF (Koren/Bell/Volinsky), scored by ranking the predicted rating.

Same optimiser, same factor count, same data, same support, same pool, same metric. The ONLY difference is
whether r multiplies the confidence or occupies the target. If W still beats E, the mechanism survives with
family held fixed. If E matches or beats W, the four-row pattern was about model quality and the paragraph
must come out.

CONTROL. arm E is also run with the rating replaced by 1 on the same support (arm E-bin): an explicit MF
that regresses a constant. If E-bin ~ E, then arm E's weakness is the explicit-MF *estimand* rather than
the values, which is the reading the chapter wants; if E-bin is much worse than E, the values are doing
real work inside the estimand formulation and the claim needs softening.

alpha for W and the ridge for both are re-selected on validation NDCG@10 -- not inherited. Every arm is
scored on the arm-N pool through the same evaluate() call, and MostPop must reproduce first.

  python src/baselines/run_weight_vs_estimand.py
"""
import os
import sys
import json
import time
import argparse

import numpy as np
from scipy import sparse

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, _HERE)

import metrics as M
import pop
import ials
from arm_n import load_arm_n
from graded_data import load_graded_val

OUTDIR = os.path.join(_ROOT, "experiments", "baselines", "arm_n")
OUTP = os.path.join(OUTDIR, "weight_vs_estimand.json")
ALPHA_GRID = [1.0, 2.5, 5.0, 10.0]
REG_GRID = [0.01, 0.1]
FACTORS, ITERS, BATCH = 200, 15, 250
STATE = {}


def logln(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    os.makedirs(OUTDIR, exist_ok=True)
    with open(os.path.join(OUTDIR, "weight_vs_estimand.log"), "a") as f:
        f.write(line + "\n")


def persist():
    json.dump(STATE, open(OUTP, "w"), indent=2)


def fit_explicit(train, factors, reg, iters, seed=98765):
    """Explicit-feedback MF by ALS: min sum_{observed} (r_ui - p_u q_i)^2 + reg(|P|^2+|Q|^2).

    Unobserved entries carry weight ZERO -- they are missing, not negative. That is the defining
    difference from the implicit/confidence formulation, and it is exactly the estimand/weight contrast
    this script exists to measure. Prediction for ranking is the predicted rating p_u . q_i.
    """
    R = train.tocsr().astype(np.float32)
    Rt = R.T.tocsr()
    n_u, n_i = R.shape
    rng = np.random.RandomState(seed)
    P = rng.normal(0, 0.01, (n_u, factors)).astype(np.float32)
    Q = rng.normal(0, 0.01, (n_i, factors)).astype(np.float32)
    I = (reg * np.eye(factors)).astype(np.float32)

    def side(X, A, B):
        for u in range(X.shape[0]):
            lo, hi = X.indptr[u], X.indptr[u + 1]
            if lo == hi:
                continue
            idx, val = X.indices[lo:hi], X.data[lo:hi]
            Bs = B[idx]
            A[u] = np.linalg.solve(Bs.T @ Bs + I, Bs.T @ val)

    for it in range(iters):
        side(R, P, Q)
        side(Rt, Q, P)
    return P, Q


def make_explicit_predict(P, Q, train_rows_lookup=None):
    def predict(X_csr):
        # fold in a held-out user: solve the same ridge for their observed ratings against fixed Q
        n = X_csr.shape[0]
        F = Q.shape[1]
        I = (make_explicit_predict.reg * np.eye(F)).astype(np.float32)
        out = np.zeros((n, Q.shape[0]), dtype=np.float32)
        X = X_csr.tocsr().astype(np.float32)
        for u in range(n):
            lo, hi = X.indptr[u], X.indptr[u + 1]
            if lo == hi:
                continue
            idx, val = X.indices[lo:hi], X.data[lo:hi]
            Bs = Q[idx]
            p = np.linalg.solve(Bs.T @ Bs + I, Bs.T @ val)
            out[u] = Q @ p
        return out
    return predict


def main():
    ap = argparse.ArgumentParser()
    a0 = ap.parse_args()

    D = load_arm_n(log=logln)
    ni, hm, pool, te_te = D["n_items"], D["head_mask"], D["pool"], D["te_te"]

    pr_pop = pop.fit(D["train"], ni, log=logln)
    rp = M.evaluate(pr_pop, D["te_tr"], te_te, batch_size=BATCH, head_mask=hm, mask_X=pool)
    canon = json.load(open(os.path.join(OUTDIR, "pop.json")))["N"]
    ok = abs(rp["ndcg@10"] - canon["ndcg@10"]) <= 1e-4
    logln(f"[G1] MostPop this path {rp['ndcg@10']:.4f} vs arm-N {canon['ndcg@10']:.4f} -> "
          f"{'PASS' if ok else '*** FAIL ***'}")
    if not ok:
        raise SystemExit("[G1] eval path does not reproduce MostPop")
    STATE["gates"] = {"mostpop": bool(ok)}
    persist()

    va_tr_bin, va_te = M.load_val(ni, os.path.join(_ROOT, "data", "ml-25m", "proc"))
    g_va = load_graded_val(va_te, log=logln)
    g_tr, g_te = D["g_train"], D["g_te_tr"]

    def binarise(X):
        B = X.tocsr().copy().astype(np.float32); B.data[:] = 1.0
        return B

    # ---- arm W: the rating as CONFIDENCE ------------------------------------------------------------
    logln("--- arm W: rating as WEIGHT (c = 1 + alpha*r), target 1 ---")
    best = (None, None, -1.0, None)
    for reg in REG_GRID:
        for alpha in ALPHA_GRID:
            m = ials._fit_model(g_tr, FACTORS, reg, alpha, ITERS, log=lambda *_: None)
            prd = ials._make_predict(m, reg, alpha)
            v = M.evaluate(prd, g_va, va_te, batch_size=BATCH, head_mask=hm)
            logln(f"  [W] reg={reg} alpha={alpha:5.1f} val full={v['ndcg@10']:.4f}")
            if v["ndcg@10"] > best[2]:
                best = (reg, alpha, v["ndcg@10"], prd)
    regW, alphaW, valW, prdW = best
    rW = M.evaluate(prdW, g_te, te_te, batch_size=BATCH, head_mask=hm, mask_X=pool)
    logln(f"[W] selected reg={regW} alpha={alphaW} -> TEST full={rW['ndcg@10']:.4f} "
          f"tail={rW['tail_ndcg@10']:.4f}")
    STATE["W_weight"] = {"reg": regW, "alpha": alphaW, "val": valW,
                         "ndcg@10": rW["ndcg@10"], "tail_ndcg@10": rW["tail_ndcg@10"]}
    persist()

    # ---- arm E: the rating as ESTIMAND, and its constant-target control ------------------------------
    for tag, tr, va_in, te_in, note in (
            ("E_estimand", g_tr, g_va, g_te, "explicit MF regressing the RATING"),
            ("Ebin_control", binarise(g_tr), binarise(g_va), binarise(g_te),
             "explicit MF regressing a CONSTANT on the same support")):
        logln(f"--- arm {tag}: {note} ---")
        bestE = (None, -1.0, None)
        for reg in REG_GRID:
            t0 = time.time()
            P, Q = fit_explicit(tr, FACTORS, reg, ITERS)
            make_explicit_predict.reg = reg
            prd = make_explicit_predict(P, Q)
            v = M.evaluate(prd, va_in, va_te, batch_size=BATCH, head_mask=hm)
            logln(f"  [{tag}] reg={reg} val full={v['ndcg@10']:.4f} ({(time.time()-t0)/60:.1f}m)")
            if v["ndcg@10"] > bestE[1]:
                bestE = (reg, v["ndcg@10"], prd)
        regE, valE, prdE = bestE
        rE = M.evaluate(prdE, te_in, te_te, batch_size=BATCH, head_mask=hm, mask_X=pool)
        logln(f"[{tag}] selected reg={regE} -> TEST full={rE['ndcg@10']:.4f} tail={rE['tail_ndcg@10']:.4f}")
        STATE[tag] = {"reg": regE, "val": valE, "ndcg@10": rE["ndcg@10"],
                      "tail_ndcg@10": rE["tail_ndcg@10"], "note": note}
        persist()

    d = STATE["W_weight"]["ndcg@10"] - STATE["E_estimand"]["ndcg@10"]
    dv = STATE["E_estimand"]["ndcg@10"] - STATE["Ebin_control"]["ndcg@10"]
    logln("=" * 78)
    logln(f"[verdict] WEIGHT minus ESTIMAND, model family held fixed: {d:+.4f} full")
    logln("[verdict] " + ("mechanism SURVIVES the within-model test -- the role of the rating, not the "
                          "model family, is what separates the rows." if d > 0.005 else
                          "mechanism NOT supported once the family is held fixed -- the four-row pattern "
                          "was model quality. Delete the paragraph."))
    logln(f"[control] values inside the estimand formulation are worth {dv:+.4f} "
          "(E minus E-bin; near zero means arm E's weakness is the formulation, not the values)")
    logln("=" * 78)
    STATE["verdict"] = {"weight_minus_estimand": d, "values_within_estimand": dv,
                        "mechanism_supported": bool(d > 0.005)}
    persist()
    logln(f"done -> {OUTP}")


if __name__ == "__main__":
    main()
