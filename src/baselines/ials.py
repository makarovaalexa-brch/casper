"""ials.py -- iALS / WMF (Weighted Matrix Factorization, Hu-Koren-Volinsky 2008) via the `implicit` lib.

Uses implicit.als.AlternatingLeastSquares (Cython/BLAS, CPU). Confidence weighting c_ui = 1 + alpha,
preference p_ui = 1 for observed (binary) interactions -- the Hu 2008 formulation implicit implements.

Cold-start fold-in (strong generalization): a held-out user is never in the trained user factors, so
we solve the ALS user-step in closed form against the FROZEN item factors Y (implicit's own
`recalculate_user` math, vectorized):
    x_u = (Y^T Y + reg*I + alpha * Y_S^T Y_S)^-1 * (1+alpha) * Y_S^T 1        (S = the user's items)
    score = Y @ x_u
This is exactly the least-squares user embedding implicit computes for a new user.

Hyperparameters: WMF has no single canonical ML-20M constant in the Mult-VAE/RecVAE tables (they report
NDCG@100 ~0.386 for a tuned WMF). We fix factors=200 (= the VAE latent dim) and select regularization
and alpha on validation, reporting the chosen values. **Flagged in DESIGN_SHEET as a tuned baseline**
(no verbatim paper hyperparameters exist for their WMF).

Interface: fit(train, n_items, evaluator=None, args=None, ckpt=None, log=print) -> predict_fn
"""
import os
import sys
import json
import time
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import metrics as M

DEFAULTS = {"factors": 200, "reg": 0.01, "alpha": 10.0, "iterations": 15}


def _fit_model(train, factors, reg, alpha, iterations, log=print):
    from implicit.als import AlternatingLeastSquares
    from scipy import sparse
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")  # implicit warns; avoid oversubscription
    # implicit >=0.5 expects a user-items CSR; confidence handled by multiplying the matrix by alpha.
    Cui = (train.tocsr().astype(np.float32)) * alpha
    model = AlternatingLeastSquares(factors=factors, regularization=reg,
                                    iterations=iterations, use_gpu=False, calculate_training_loss=False)
    t0 = time.time()
    model.fit(Cui, show_progress=False)
    log(f"[ials] trained f={factors} reg={reg} alpha={alpha} it={iterations} "
        f"({(time.time()-t0)/60:.1f}m)")
    return model


def _make_predict(model, reg, alpha):
    Y = np.asarray(model.item_factors, dtype=np.float64)   # (m x f)
    f = Y.shape[1]
    YtY = Y.T @ Y
    base = YtY + reg * np.eye(f)

    def predict(X_csr):
        Xc = X_csr.tocsr()
        b = Xc.shape[0]
        scores = np.zeros((b, Y.shape[0]), dtype=np.float32)
        for r in range(b):
            cols = Xc.indices[Xc.indptr[r]:Xc.indptr[r + 1]]
            if len(cols) == 0:
                continue
            Ys = Y[cols]                                   # (|S| x f)
            A = base + alpha * (Ys.T @ Ys)
            rhs = (1.0 + alpha) * Ys.sum(axis=0)           # Y_S^T 1 * (1+alpha)
            x_u = np.linalg.solve(A, rhs)
            scores[r] = (Y @ x_u).astype(np.float32)
        return scores
    return predict


def fit(train, n_items, evaluator=None, args=None, ckpt=None, log=print):
    a = args or argparse.Namespace(**DEFAULTS)
    factors = getattr(a, "factors", DEFAULTS["factors"])
    reg = getattr(a, "reg", DEFAULTS["reg"])
    alpha = getattr(a, "alpha", DEFAULTS["alpha"])
    iterations = getattr(a, "iterations", DEFAULTS["iterations"])
    model = _fit_model(train, factors, reg, alpha, iterations, log=log)
    predict = _make_predict(model, reg, alpha)
    predict.hp = dict(factors=factors, reg=reg, alpha=alpha, iterations=iterations)
    return predict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--factors", type=int, default=DEFAULTS["factors"])
    ap.add_argument("--reg", type=float, default=DEFAULTS["reg"])
    ap.add_argument("--alpha", type=float, default=DEFAULTS["alpha"])
    ap.add_argument("--iterations", type=int, default=DEFAULTS["iterations"])
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    meta = M.load_meta(); n_items = meta["n_items"]
    train = M.load_train(n_items)
    te_tr, te_te = M.load_test(n_items)
    t0 = time.time()
    best = vars(args).copy()
    if args.sweep:
        va_tr, va_te = M.load_val(n_items)
        best_v = -1.0
        for reg in (0.001, 0.01, 0.1):
            for alpha in (1.0, 10.0, 50.0):
                pr = fit(train, n_items, args=argparse.Namespace(
                    factors=args.factors, reg=reg, alpha=alpha, iterations=args.iterations))
                v = M.evaluate(pr, va_tr, va_te)["ndcg@100"]
                print(f"[ials] VAL reg={reg} alpha={alpha} ndcg@100={v:.4f}")
                if v > best_v:
                    best_v = v; best["reg"] = reg; best["alpha"] = alpha
    predict = fit(train, n_items, args=argparse.Namespace(**{k: best[k] for k in
                  ("factors", "reg", "alpha", "iterations")}))
    res = M.evaluate(predict, te_tr, te_te)
    res.update(predict.hp); res["seconds"] = time.time() - t0
    print(f"[ials] test {res}")
    if args.out:
        json.dump(res, open(args.out, "w"), indent=2)


if __name__ == "__main__":
    main()
