"""turbocf.py -- Turbo-CF (Park, Kim & Shin, SIGIR 2024, "Turbo-CF: Matrix Decomposition-Free
Graph Filtering for Fast Recommendation", arXiv:2404.14243).

A TRAINING-FREE, matrix-decomposition-FREE graph filter. Instead of eigendecomposing the item-item
graph (GF-CF / BSPM), Turbo-CF applies a low-order POLYNOMIAL low-pass filter directly, so the whole
model is a couple of dense mat-muls -- seconds on a benchmark, minutes at our 18k-item catalog.

ALGORITHM (verbatim from the paper's equations + the public code github.com/jindeok/Turbo-CF/main.py,
fetched 2026-07-22; provenance pinned inline):

  1. Asymmetric degree normalization of the binary user-item matrix R  (paper Eq. 4):
        R_tilde = D_U^{-alpha} @ R @ D_I^{alpha-1}
     D_U = diag(row-sums of R) (user degrees), D_I = diag(col-sums of R) (item degrees).
     alpha in [0,1]; alpha=0.5 => symmetric D^{-1/2} R D^{-1/2}.  (code kwarg `alpha`).

  2. Item-item similarity graph + Hadamard (element-wise) power `s`  (code: `P = R_norm.T@R_norm;
     P.data **= power`):
        P_bar = (R_tilde^T R_tilde)^{o s}          (o s = element-wise exponent, R_tilde>=0 so P>=0)

  3. Polynomial low-pass filter (code `args.filter` in {1,2,3}):
        filter=1 (linear, 1st order):   F = P_bar
        filter=2 (second order):        F = 2*P_bar - P_bar@P_bar
        filter=3 (ideal-LPF approx):    F = P_bar + 0.01*(-P_bar^3 + 10*P_bar^2 - 29*P_bar)
     (frequency responses 1-lambda, 1-lambda^2, and a 3rd-order Chebyshev-like approx respectively.)

  4. Score:   S = R @ F      (fold-in items masked to -inf by metrics.evaluate, as everywhere here).

HYPERPARAMETERS -- provenance documented (two sources disagree; we take the PAPER's tuned values):
  * PAPER sensitivity analysis (Sec 4.5 / Fig 4): alpha=0.7, s=0.6 are the reported best region across
    the dense benchmarks; linear (filter=1) "is often sufficient", second-order helps on denser graphs.
  * The public repo's argparse DEFAULTS are alpha=0.5, power=1, filter=1 (generic CLI defaults, NOT the
    per-dataset tuned values used to produce the paper table).
  We DEFAULT to the paper-tuned alpha=0.7, s=0.6, filter=2 (second-order) -- ML-25M @ 18k items is dense
  enough that the 2nd-order term is affordable (F is one 18k^2 float32 = ~1.35 GB dense matrix, one extra
  18k^3 matmul ~ a couple minutes) -- and expose --alpha/--power/--filter for a val sweep. The chosen
  values are recorded in the output `hp`. NO published ML-20M/25M Turbo-CF number exists to snap to;
  flagged best-effort-tuned at review (like ials/edlae here).

Shape is IDENTICAL to EASE: fit() builds a dense item-item matrix F, predict(X)=X@F.

Interface: fit(train, n_items, evaluator=None, args=None, ckpt=None, log=print) -> predict_fn
CLI: python src/baselines/turbocf.py --smoke     # tiny SYNTHETIC code-path check (NOT the real split)
     python src/baselines/turbocf.py             # full fit+eval on the Liang ML-25M split (no caps)
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

# Paper-tuned defaults (Sec 4.5 / Fig 4). filter=2 (second-order) as the dense-catalog default.
DEFAULTS = {"alpha": 0.7, "power": 0.6, "filter": 2}


def normalize_R(train, alpha):
    """R_tilde = D_U^{-alpha} R D_I^{alpha-1}  (paper Eq. 4). Sparse, float32."""
    R = train.tocsr().astype(np.float32)
    du = np.asarray(R.sum(axis=1)).ravel()                 # user degrees
    di = np.asarray(R.sum(axis=0)).ravel()                 # item degrees
    du_s = np.power(du, -alpha, where=du > 0, out=np.zeros_like(du))
    di_s = np.power(di, alpha - 1.0, where=di > 0, out=np.zeros_like(di))
    # R_tilde = diag(du_s) @ R @ diag(di_s)
    Rn = sparse.diags(du_s) @ R @ sparse.diags(di_s)
    return Rn.tocsr().astype(np.float32)


def build_filter(train, alpha, power, filt, log=print):
    """Return the dense (n_items x n_items) Turbo-CF filter matrix F (float32)."""
    Rn = normalize_R(train, alpha)
    log(f"[turbocf] R_tilde nnz={Rn.nnz} (alpha={alpha}); building P_bar (dense {Rn.shape[1]}^2 f32)")
    P = np.asarray((Rn.T @ Rn).todense(), dtype=np.float32)   # item-item, dense
    # Hadamard power s on the (nonnegative) similarities -- code: P.data **= power
    if power != 1.0:
        np.power(P, power, out=P)                            # element-wise, in place
    if filt == 1:
        F = P
    elif filt == 2:
        F = 2.0 * P - (P @ P)                                # second order: 2P - P^2
    elif filt == 3:
        P2 = P @ P
        F = P + 0.01 * (-(P2 @ P) + 10.0 * P2 - 29.0 * P)    # ideal-LPF poly approx
    else:
        raise ValueError(f"filter must be 1/2/3, got {filt}")
    return np.asarray(F, dtype=np.float32)


def fit(train, n_items, evaluator=None, args=None, ckpt=None, log=print):
    a = args or argparse.Namespace(**DEFAULTS)
    alpha = getattr(a, "alpha", DEFAULTS["alpha"])
    power = getattr(a, "power", DEFAULTS["power"])
    filt = int(getattr(a, "filter", DEFAULTS["filter"]))
    t0 = time.time()
    F = build_filter(train, alpha, power, filt, log=log)
    log(f"[turbocf] F ready ({(time.time()-t0)/60:.1f}m) alpha={alpha} power={power} filter={filt} "
        f"|F|={np.linalg.norm(F):.2f}")

    def predict(X_csr, _F=F):
        return np.asarray(X_csr @ _F, dtype=np.float32)
    predict.F = F
    predict.hp = dict(alpha=alpha, power=power, filter=filt,
                      role="Turbo-CF (Park SIGIR2024) paper-tuned alpha=0.7 s=0.6 2nd-order")
    return predict


# --------------------------------------------------------------------------- CLI / smoke
def _head_mask(train, n_items):
    cnt = np.asarray(train.sum(axis=0)).ravel().astype(np.float64)
    order = np.argsort(-cnt)
    cumfrac = np.cumsum(cnt[order]) / cnt.sum()
    head = np.zeros(n_items, dtype=bool)
    head[order[:np.searchsorted(cumfrac, 0.33) + 1]] = True
    return head


def _smoke():
    print("[turbocf][SMOKE] synthetic tiny data (code-path only, NOT the Liang split)")
    rng = np.random.RandomState(0)
    n_users, n_items = 120, 130
    X = (sparse.random(n_users, n_items, density=0.15, random_state=rng,
                       data_rvs=lambda s: np.ones(s)) > 0).astype(np.float32).tocsr()
    tr = X[:90]
    te_tr = X[90:]
    te_te = (sparse.random(30, n_items, density=0.1, random_state=rng,
                           data_rvs=lambda s: np.ones(s)) > 0).astype(np.float32).tocsr()
    for filt in (1, 2, 3):
        a = argparse.Namespace(alpha=0.7, power=0.6, filter=filt)
        pr = fit(tr, n_items, args=a, log=print)
        res = M.evaluate(pr, te_tr, te_te, batch_size=10, head_mask=_head_mask(tr, n_items))
        print(f"[turbocf][SMOKE] filter={filt} OK full@10={res['ndcg@10']:.4f} "
              f"tail@10={res['tail_ndcg@10']:.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", type=float, default=DEFAULTS["alpha"])
    ap.add_argument("--power", type=float, default=DEFAULTS["power"])
    ap.add_argument("--filter", type=int, default=DEFAULTS["filter"], choices=[1, 2, 3])
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        _smoke(); return
    _HERE = os.path.dirname(os.path.abspath(__file__))
    PROC = os.path.abspath(os.path.join(_HERE, "..", "..", "data", "ml-25m", "proc"))
    meta = M.load_meta(PROC); n_items = meta["n_items"]
    train = M.load_train(n_items, PROC)
    te_tr, te_te = M.load_test(n_items, PROC)
    hm = _head_mask(train, n_items)
    alpha, power, filt = args.alpha, args.power, args.filter
    if args.sweep:
        va_tr, va_te = M.load_val(n_items, PROC)
        best_v = -1.0
        for al in (0.5, 0.7, 1.0):
            for pw in (0.6, 1.0):
                for fl in (1, 2):
                    pr = fit(train, n_items, args=argparse.Namespace(alpha=al, power=pw, filter=fl))
                    v = M.evaluate(pr, va_tr, va_te, batch_size=500)["ndcg@10"]
                    print(f"[turbocf] VAL alpha={al} power={pw} filter={fl} full@10={v:.4f}")
                    if v > best_v:
                        best_v, alpha, power, filt = v, al, pw, fl
    t0 = time.time()
    predict = fit(train, n_items, args=argparse.Namespace(alpha=alpha, power=power, filter=filt))
    res = M.evaluate(predict, te_tr, te_te, batch_size=500, head_mask=hm)
    res["seconds"] = time.time() - t0; res["hp"] = predict.hp
    print(f"[turbocf] test full@10={res['ndcg@10']:.4f} tail@10={res['tail_ndcg@10']:.4f} "
          f"ndcg@100={res['ndcg@100']:.4f}")
    if args.out:
        json.dump(res, open(args.out, "w"), indent=2)


if __name__ == "__main__":
    main()
