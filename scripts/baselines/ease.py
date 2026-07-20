"""ease.py -- EASE (Embarrassingly Shallow AutoEncoder), Steck, WWW 2019.

Closed-form item-item model. Given the binary train matrix X (r users x m items) and Gram G = X^T X:
    P = (G + lambda * I)^-1
    B = I - P * diagMat(1 / diag(P))          # zero-diagonal Lagrangian solution (Steck 2019, Eq. 8)
    B_ii = 0
Fold-in score for a user with item vector x:  score = x @ B.

Hyperparameter (PAPER value, no tuning):  lambda = 500, as reported for ML-20M in Steck 2019
(Section 4 / Table: "l2-norm regularization lambda = 500"). A validation sweep is available via
--sweep for cross-checking but the DEFAULT and the snap use lambda = 500 verbatim.

Precision: float64 for the solve (the (G+lambda I) inverse is ill-conditioned at m~20k; float32 loses
several NDCG points -- Steck's reference uses double). RAM: G and P are each m x m float64.
  ML-20M  m=20108 -> 20108^2 * 8 B = 3.23 GB per matrix; np.linalg.inv needs a working copy ->
  peak ~ 3 matrices ~ 10 GB. ML-25M m=18430 -> 2.72 GB each, ~8 GB peak. Documented in DESIGN_SHEET.

Interface: fit(train, n_items, evaluator=None, args=None, ckpt=None, log=print) -> predict_fn
"""
import os
import sys
import json
import time
import argparse
import numpy as np
from scipy import sparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import metrics as M

DEFAULTS = {"lam": 500.0}


def build_gram(train, dtype=np.float64):
    """G = X^T X (m x m dense). X binary user x item."""
    X = train.tocsr().astype(np.float32)
    G = (X.T @ X).todense()
    return np.asarray(G, dtype=dtype)


def inv_spd_lowmem(A, block=2048):
    """Exact inverse of SPD A via IN-PLACE Cholesky + block column solves.
    DESTROYS A (it becomes the factor). Peak extra RAM ~= one m x m output + one m x block RHS,
    vs np.linalg.inv's ~2 extra m x m copies -- needed because this box has ~8.5 GB free
    (m=20108 float64 = 3.23 GB per matrix). Same math, no approximation."""
    from scipy import linalg as sla
    m = A.shape[0]
    c, low = sla.cho_factor(A, overwrite_a=True, check_finite=False)
    P = np.empty((m, m), dtype=A.dtype)
    for st in range(0, m, block):
        en = min(st + block, m)
        E = np.zeros((m, en - st), dtype=A.dtype)
        E[np.arange(st, en), np.arange(en - st)] = 1.0
        P[:, st:en] = sla.cho_solve((c, low), E, overwrite_b=True, check_finite=False)
    return P


def ease_B(G, lam):
    """B = I - P diagMat(1/diag(P)); P = (G + lam I)^-1.  float64.
    DESTROYS G (in-place ridge + in-place Cholesky) to fit low RAM; rebuild the Gram to reuse."""
    m = G.shape[0]
    G[np.diag_indices(m)] += lam            # A = G + lam*I, in place (no 3.2 GB eye)
    P = inv_spd_lowmem(G)                   # G is destroyed here
    d = 1.0 / np.diag(P).copy()
    P *= -d[np.newaxis, :]                  # in place: columns scaled by -1/diag
    np.fill_diagonal(P, 0.0)
    return P.astype(np.float32)


def fit(train, n_items, evaluator=None, args=None, ckpt=None, log=print):
    lam = (args.lam if args and getattr(args, "lam", None) is not None else DEFAULTS["lam"])
    t0 = time.time()
    log(f"[ease] building Gram ({n_items}x{n_items} float64)")
    G = build_gram(train)
    log(f"[ease] solving EASE lambda={lam} ...")
    B = ease_B(G, lam)
    del G
    log(f"[ease] B ready ({(time.time()-t0)/60:.1f}m), |B|={np.linalg.norm(B):.2f}")

    def predict(X_csr):
        return np.asarray(X_csr @ B, dtype=np.float32)
    predict.B = B
    predict.lam = lam
    return predict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lam", type=float, default=DEFAULTS["lam"])
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    meta = M.load_meta(); n_items = meta["n_items"]
    train = M.load_train(n_items)
    te_tr, te_te = M.load_test(n_items)
    t0 = time.time()
    lam = args.lam
    if args.sweep:
        va_tr, va_te = M.load_val(n_items)
        best_v = -1.0
        for l in (100.0, 200.0, 500.0, 1000.0):
            G = build_gram(train)               # ease_B destroys G -> rebuild per lambda (~12 s)
            B = ease_B(G, l); del G
            v = M.evaluate(lambda Xc, _B=B: np.asarray(Xc @ _B, np.float32), va_tr, va_te)["ndcg@100"]
            print(f"[ease] VAL lambda={l} ndcg@100={v:.4f}")
            if v > best_v:
                best_v, lam = v, l
    predict = fit(train, n_items, args=argparse.Namespace(lam=lam))
    res = M.evaluate(predict, te_tr, te_te)
    res["lambda"] = lam; res["seconds"] = time.time() - t0
    print(f"[ease] test {res}")
    if args.out:
        json.dump(res, open(args.out, "w"), indent=2)


if __name__ == "__main__":
    main()
