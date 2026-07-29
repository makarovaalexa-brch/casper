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

HYPERPARAMETERS -- provenance documented (two sources disagree):
  * PAPER sensitivity analysis (Sec 4.5 / Fig 4): alpha=0.7, s=0.6 are the reported best region across
    the dense benchmarks; linear (filter=1) "is often sufficient", second-order helps on denser graphs.
  * The public repo's argparse DEFAULTS are alpha=0.5, power=1, filter=1 (generic CLI defaults, NOT the
    per-dataset tuned values used to produce the paper table).
  There is NO published ML-20M/25M Turbo-CF number to snap to, so the hyper-parameters are chosen on OUR
  VAL split (`--sweep`, the default) and the row is flagged best-effort-tuned at review, like ials/edlae.

STABILITY -- why the config matters, and why it is checked (added 2026-07-29 after a null run).
  The public code applies the Hadamard power to P_bar and then feeds it straight to the polynomial: it
  does NOT re-normalize afterwards. The polynomials 2P - P^2 and the 3rd-order approximation are low-pass
  only while rho(P) <= 1. Symmetric normalization (alpha=0.5, s=1) gives exactly that. But s < 1 raises
  every entry of a sub-unit matrix, and the resulting growth in rho scales with the catalogue: harmless on
  a 130-item toy, ruinous at 18,359 items, where the -P^2 term dominates and the ranking INVERTS -- which
  is exactly how alpha=0.7 / s=0.6 / filter=2 returned 0.0000 on every metric here. build_filter now
  estimates rho(P_bar) by power iteration and logs a loud warning when a polynomial filter is used outside
  its stable region, so a degenerate configuration can never again look like a weak baseline.

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

# Public-repo defaults (alpha=0.5, s=1, linear) -- the only configuration guaranteed stable at any
# catalogue size. The paper's tuned region (alpha=0.7, s=0.6) is reachable through --sweep, selected
# on OUR val split. See the STABILITY note above for why the tuned region is not a safe default here.
DEFAULTS = {"alpha": 0.5, "power": 1.0, "filter": 1}


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


def _rho(P, iters=30, seed=0):
    """Power-iteration estimate of the spectral radius of the (symmetric, nonnegative) P_bar."""
    v = np.random.RandomState(seed).rand(P.shape[0]).astype(np.float32)
    v /= np.linalg.norm(v)
    lam = 0.0
    for _ in range(iters):
        w = P @ v
        lam = float(np.linalg.norm(w))
        if lam == 0.0:
            return 0.0
        v = w / lam
    return lam


def _blocked(P, fn, block=2048):
    """Evaluate a row-blocked polynomial in P without materialising a second full temporary."""
    F = np.empty_like(P)
    for i in range(0, P.shape[0], block):
        F[i:i + block] = fn(P[i:i + block])
    return F


def build_filter(train, alpha, power, filt, log=print):
    """Return the dense (n_items x n_items) Turbo-CF filter matrix F (float32)."""
    Rn = normalize_R(train, alpha)
    log(f"[turbocf] R_tilde nnz={Rn.nnz} (alpha={alpha}); building P_bar (dense {Rn.shape[1]}^2 f32)")
    P = np.asarray((Rn.T @ Rn).todense(), dtype=np.float32)   # item-item, dense
    # Hadamard power s on the (nonnegative) similarities -- code: P.data **= power
    if power != 1.0:
        np.power(P, power, out=P)                            # element-wise, in place
    rho = _rho(P)
    log(f"[turbocf] rho(P_bar) ~ {rho:.3f} (alpha={alpha}, s={power})")
    if filt != 1 and rho > 1.05:
        log(f"[turbocf] *** WARNING: rho(P_bar)={rho:.3f} > 1, so the order-{filt} polynomial is NOT "
            f"low-pass here -- the -P^2 term dominates and the ranking inverts. Expect ~0 accuracy. ***")
    if filt == 1:
        F = P
    elif filt == 2:                                          # second order: 2P - P^2
        F = _blocked(P, lambda B: 2.0 * B - (B @ P))
    elif filt == 3:                                          # ideal-LPF poly approx
        F = _blocked(P, lambda B: B + 0.01 * (-((B @ P) @ P) + 10.0 * (B @ P) - 29.0 * B))
    else:
        raise ValueError(f"filter must be 1/2/3, got {filt}")
    F = np.asarray(F, dtype=np.float32)
    if not np.isfinite(F).all():
        raise FloatingPointError(f"[turbocf] non-finite entries in F (alpha={alpha} s={power} filt={filt})")
    log(f"[turbocf] F stats: min={F.min():.3g} max={F.max():.3g} frac_neg={float((F < 0).mean()):.3f}")
    return F


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
                      role=f"Turbo-CF (Park SIGIR2024) alpha={alpha} s={power} filter={filt}")
    return predict


# Grid = the paper's reported sensitivity region (Sec 4.5 / Fig 4) union the repo defaults.
SWEEP_GRID = [(al, pw, fl) for al in (0.5, 0.7, 1.0) for pw in (0.6, 1.0) for fl in (1, 2)]


def fit_sweep(train, n_items, va_tr, va_te, log=print, grid=None):
    """Select (alpha, s, filter) on OUR val split -- there is no published ML-20M/25M number to snap
    to, so the row is best-effort tuned and the whole grid is recorded alongside the winner."""
    trials = []
    best = None
    for al, pw, fl in (grid or SWEEP_GRID):
        pr = fit(train, n_items, args=argparse.Namespace(alpha=al, power=pw, filter=fl), log=log)
        v = M.evaluate(pr, va_tr, va_te, batch_size=500)["ndcg@10"]
        log(f"[turbocf] VAL alpha={al} s={pw} filter={fl} full@10={v:.4f}")
        trials.append(dict(alpha=al, power=pw, filter=fl, val_ndcg10=round(float(v), 5)))
        if best is None or v > best[0]:
            best = (v, pr)
        else:
            del pr
    val, pr = best
    pr.hp["selected_on"] = "val full NDCG@10 (no published ML-20M/25M number to snap to)"
    pr.hp["val_ndcg@10"] = round(float(val), 5)
    pr.hp["sweep"] = trials
    log(f"[turbocf] SELECTED {pr.hp['role']} (val {val:.4f})")
    return pr


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
    ap.add_argument("--no_sweep", action="store_true",
                    help="use the fixed --alpha/--power/--filter instead of val selection")
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
    t0 = time.time()
    if args.no_sweep:
        predict = fit(train, n_items,
                      args=argparse.Namespace(alpha=args.alpha, power=args.power, filter=args.filter))
    else:
        va_tr, va_te = M.load_val(n_items, PROC)
        predict = fit_sweep(train, n_items, va_tr, va_te)
    res = M.evaluate(predict, te_tr, te_te, batch_size=500, head_mask=hm)
    res["seconds"] = time.time() - t0; res["hp"] = predict.hp
    print(f"[turbocf] test full@10={res['ndcg@10']:.4f} tail@10={res['tail_ndcg@10']:.4f} "
          f"ndcg@100={res['ndcg@100']:.4f}")
    if args.out:
        json.dump(res, open(args.out, "w"), indent=2)


if __name__ == "__main__":
    main()
