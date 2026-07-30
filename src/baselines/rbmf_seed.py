"""rbmf_seed.py -- RBMF / Functional-MF best-effort: MF-seed reconstruction (POINT least-squares).

WHAT THIS IS (documented best-effort row; no canonical ML-20M/25M number to snap to):
  The "reconstruct a cold user's latent by least squares on a frozen item basis, then score" family
  -- RBMF (Reconstruction-Based MF) / Functional MF / the classic WMF cold-start fold-in. We FREEZE the
  same certified iALS factorization used by ials.py / belief_mf.py (factors=200, reg=0.1, alpha=50,
  15 it -- the ML-20M val-selected constants), take the frozen item factors Y (m x f), and for each
  held-out user solve a RIDGE-REGULARIZED least-squares seed from their fold-in items S:

        x_u = argmin_x || Y_S x - 1 ||^2 + lam * ||x||^2
            = (Y_S^T Y_S + lam I)^{-1} Y_S^T 1                          (closed form)
        score(user) = Y @ x_u

  where Y_S = Y[fold-in items], target = 1 for each observed like. One f x f solve per user (f=200),
  same shape as the iALS cold-start fold-in.

DIFFERENCE vs belief_mf.py (documented on purpose -- these are DISTINCT table rows):
  * rbmf_seed  = POINT estimate. Ridge LS with an ISOTROPIC prior lam*I and target r_i=1. No user-prior
                 mean, no learned covariance -- a frequentist MAP/ridge seed. (RBMF / functional-MF.)
  * belief_mf  = BAYESIAN. Conjugate Gaussian posterior with an EMPIRICAL prior N(mu0, Sigma0) fit from
                 the TRAIN user factors; precision Sigma0^{-1} + (1/s2) Y_S^T Y_S; scores the posterior
                 MEAN. (Biyik-belief + ConTS-mean.)
  Concretely rbmf_seed == belief_mf with (mu0=0, Sigma0 = (1/lam) I, obs_noise=1): the *isotropic-prior,
  zero-mean* special case. Kept as its own module so the ablation "empirical Gaussian prior vs plain
  ridge" is an explicit two-row comparison, not a hyperparameter of one row.

Note vs ials.py: iALS fold-in uses the CONFIDENCE-weighted WMF user step
  (Y^TY + reg I + alpha Y_S^T Y_S)^{-1}(1+alpha)Y_S^T 1  -- it folds ALL items (with the global Y^TY
  term) and confidence alpha. rbmf_seed is the plain unweighted ridge LS on the SUPPORT items only
  (no global Y^TY, no confidence) -- the textbook functional-MF reconstruction. Both frozen-Y, closed
  form; documented so the three MF-fold-in rows (iALS / RBMF / belief) are not confused.

Interface: fit(train, n_items, evaluator=None, args=None, ckpt=None, log=print) -> predict_fn
CLI: python src/baselines/rbmf_seed.py --smoke   # tiny SYNTHETIC code-path check (NOT the real split)
     python src/baselines/rbmf_seed.py           # full fit+eval on the Liang ML-25M split (no caps)
"""
import os
import sys
import json
import time
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import metrics as M
import ials

# Certified frozen-MF hyperparameters (== run_ml25m_liang.IALS_HP; ML-20M val-selected constants).
MF_HP = {"factors": 200, "reg": 0.1, "alpha": 50.0, "iterations": 15}
DEFAULTS = {"ls_reg": 1.0}   # ridge on the per-user LS seed (the only knob)


def _make_predict(Y, ls_reg, log=print):
    f = Y.shape[1]
    reg_I = ls_reg * np.eye(f)

    def predict(X_csr):
        Xc = X_csr.tocsr()
        b = Xc.shape[0]
        scores = np.zeros((b, Y.shape[0]), dtype=np.float32)
        for r in range(b):
            s0, e0 = Xc.indptr[r], Xc.indptr[r + 1]
            cols = Xc.indices[s0:e0]
            if len(cols) == 0:
                continue                                    # no evidence -> zero seed -> zero scores
            Ys = Y[cols]                                     # (|S| x f)
            A = Ys.T @ Ys + reg_I                            # (f x f) ridge normal-equations
            # TARGET = the VALUE carried by the input matrix. Binary contract: every entry is 1.0, so
            # Y_S^T r reduces EXACTLY to the old Y_S^T 1 and every published row is bit-identical.
            # Signed interview contract: pass centred ratings and a dislike becomes a NEGATIVE target,
            # pulling the ridge seed AWAY from that region instead of toward it. The frozen basis Y is
            # untouched -- this method is DEFINED as a fold-in onto a frozen basis, so only the target
            # moves. Caveat for the write-up: Y was itself fitted on binary data, so the latent space
            # has no dislike structure, which caps what sign can buy here.
            rhs = Ys.T @ Xc.data[s0:e0].astype(np.float64)   # Y_S^T r
            x_u = np.linalg.solve(A, rhs)
            scores[r] = (Y @ x_u).astype(np.float32)
        return scores
    return predict


def fit(train, n_items, evaluator=None, args=None, ckpt=None, log=print):
    a = args or argparse.Namespace(**DEFAULTS, **MF_HP)
    hp = {k: getattr(a, k, MF_HP[k]) for k in MF_HP}
    ls_reg = getattr(a, "ls_reg", DEFAULTS["ls_reg"])
    t0 = time.time()
    model = ials._fit_model(train, hp["factors"], hp["reg"], hp["alpha"], hp["iterations"], log=log)
    Y = np.asarray(model.item_factors, dtype=np.float64)     # (m x f) frozen item basis
    log(f"[rbmf_seed] frozen iALS f={hp['factors']} ({(time.time()-t0)/60:.1f}m); ls_reg={ls_reg}")
    predict = _make_predict(Y, ls_reg, log=log)
    predict.hp = dict(hp, ls_reg=ls_reg, role="RBMF/functional-MF point-LS seed (no prior)")
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
    print("[rbmf_seed][SMOKE] synthetic tiny data (code-path only, NOT the Liang split)")
    from scipy import sparse
    rng = np.random.RandomState(0)
    n_users, n_items = 120, 130
    X = (sparse.random(n_users, n_items, density=0.15, random_state=rng,
                       data_rvs=lambda s: np.ones(s)) > 0).astype(np.float32).tocsr()
    tr = X[:90]
    te_tr = X[90:]
    te_te = (sparse.random(30, n_items, density=0.1, random_state=rng,
                           data_rvs=lambda s: np.ones(s)) > 0).astype(np.float32).tocsr()
    a = argparse.Namespace(ls_reg=1.0, **{"factors": 8, "reg": 0.1, "alpha": 50.0, "iterations": 3})
    pr = fit(tr, n_items, args=a, log=print)
    res = M.evaluate(pr, te_tr, te_te, batch_size=10, head_mask=_head_mask(tr, n_items))
    print(f"[rbmf_seed][SMOKE] OK full@10={res['ndcg@10']:.4f} tail@10={res['tail_ndcg@10']:.4f} "
          f"hp={pr.hp}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ls_reg", type=float, default=DEFAULTS["ls_reg"])
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
    a = argparse.Namespace(ls_reg=args.ls_reg, **MF_HP)
    t0 = time.time()
    predict = fit(train, n_items, args=a)
    res = M.evaluate(predict, te_tr, te_te, batch_size=500, head_mask=hm)
    res["seconds"] = time.time() - t0; res["hp"] = predict.hp
    print(f"[rbmf_seed] test full@10={res['ndcg@10']:.4f} tail@10={res['tail_ndcg@10']:.4f} "
          f"ndcg@100={res['ndcg@100']:.4f}")
    if args.out:
        json.dump(res, open(args.out, "w"), indent=2)


if __name__ == "__main__":
    main()
