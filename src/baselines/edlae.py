"""edlae.py -- EDLAE full-rank (Emphasized Denoising Linear AutoEncoder), Steck, NeurIPS 2020.

Status: REIMPLEMENTED from the paper's closed form (no official standalone repo was located; the
method is a few lines on top of EASE). Verified against the primary source by extracting the equations
from the NeurIPS 2020 PDF (proceedings.neurips.cc/paper/2020/file/e33d974aae13e4d877477d51d8bafdc4).

Full-rank EDLAE with full emphasis (b=0) -- paper Eqs. 4, 8, 9:
    Lambda = (p / q) * diagMat(diag(X^T X)),   q = 1 - p        (Eq. 4)
    C      = (X^T X + Lambda)^-1                                 (Eq. 9)
    B      = I - C * diagMat(1 / diag(C))                        (Eq. 8)  [same zero-diagonal form as EASE]
Prediction (no dropout at test time):  score = x @ B.

=> EDLAE is EXACTLY EASE with the uniform ridge lambda*I replaced by a POPULARITY-SCALED diagonal
   Lambda = (p/q) * diag(G).  The single hyperparameter is the dropout probability p.

Hyperparameter: p (dropout). The paper selects hyperparameters on the validation set (standard); we
mirror that by sweeping p on validation and reporting the chosen value (documented in DESIGN_SHEET,
flagged as a deviation from a single "verbatim" constant because the paper reports no single ML-20M p).
An additive uniform ridge `l2` (default 0.0) is exposed for parity with Steck's released variants but
is 0 by default (pure paper closed form).

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
from ease import build_gram, inv_spd_lowmem

DEFAULTS = {"p": 0.5, "l2": 0.0}       # p selected on val in main(); 0.5 = paper's typical drop rate


def edlae_B(G, p, l2=0.0):
    """Full-rank EDLAE closed form (Steck 2020, Eq. 8/9). G = X^T X (float64)."""
    m = G.shape[0]
    q = 1.0 - p
    diagG = np.diag(G).copy()
    Lam = (p / q) * diagG                      # Eq. 4 diagonal
    Lam[diagG == 0.0] = 1.0                    # items with ZERO train interactions (possible on the
                                               # fixed ML-25M catalog, unlike the Liang train-vocab
                                               # split) -> keep A SPD; their B columns carry no signal
    G[np.diag_indices(m)] += Lam + l2          # A = G + Lam (+ l2*I), in place; DESTROYS G
    C = inv_spd_lowmem(G)                       # Eq. 9 (in-place Cholesky, low RAM)
    d = 1.0 / np.diag(C).copy()
    C *= -d[np.newaxis, :]                      # Eq. 8: I - C diagMat(1/diag(C)), in place
    np.fill_diagonal(C, 0.0)
    return C.astype(np.float32)


def fit(train, n_items, evaluator=None, args=None, ckpt=None, log=print):
    p = (args.p if args and getattr(args, "p", None) is not None else DEFAULTS["p"])
    l2 = (args.l2 if args and getattr(args, "l2", None) is not None else DEFAULTS["l2"])
    t0 = time.time()
    log(f"[edlae] building Gram ({n_items}x{n_items} float64)")
    G = build_gram(train)
    log(f"[edlae] solving full-rank EDLAE p={p} l2={l2} ...")
    B = edlae_B(G, p, l2)
    del G
    log(f"[edlae] B ready ({(time.time()-t0)/60:.1f}m), |B|={np.linalg.norm(B):.2f}")

    def predict(X_csr):
        return np.asarray(X_csr @ B, dtype=np.float32)
    predict.B = B; predict.p = p; predict.l2 = l2
    return predict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p", type=float, default=None, help="dropout prob; if unset, sweep on val")
    ap.add_argument("--l2", type=float, default=0.0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--export_B", default=None,
                    help="path to np.save the fitted EDLAE B (n_items x n_items float32) for distillation "
                         "(distill_edlae.py teacher). Saved AFTER the final (val-selected p) refit.")
    args = ap.parse_args()
    meta = M.load_meta(); n_items = meta["n_items"]
    train = M.load_train(n_items)
    te_tr, te_te = M.load_test(n_items)
    t0 = time.time()
    p = args.p
    if p is None:
        va_tr, va_te = M.load_val(n_items)
        best_v = -1.0
        import gc
        B = None
        for pp in (0.3, 0.4, 0.5, 0.6, 0.7):
            B = None; gc.collect()               # free prev B BEFORE the next 3.2GB Gram (OOM guard)
            G = build_gram(train)               # edlae_B destroys G -> rebuild per p (~12 s)
            B = edlae_B(G, pp, args.l2); del G; gc.collect()
            v = M.evaluate(lambda Xc, _B=B: np.asarray(Xc @ _B, np.float32), va_tr, va_te)["ndcg@100"]
            print(f"[edlae] VAL p={pp} ndcg@100={v:.4f}")
            if v > best_v:
                best_v, p = v, pp
    G = build_gram(train)
    B = edlae_B(G, p, args.l2); del G
    if args.export_B:
        os.makedirs(os.path.dirname(os.path.abspath(args.export_B)), exist_ok=True)
        np.save(args.export_B, B.astype(np.float32))
        print(f"[edlae] exported teacher B -> {args.export_B} ({B.shape} float32) for distill_edlae.py")
    res = M.evaluate(lambda Xc: np.asarray(Xc @ B, np.float32), te_tr, te_te)
    res["p"] = p; res["l2"] = args.l2; res["seconds"] = time.time() - t0
    print(f"[edlae] test {res}")
    if args.out:
        json.dump(res, open(args.out, "w"), indent=2)


if __name__ == "__main__":
    main()
