"""belief_mf.py -- Bayesian linear-Gaussian user belief over a FROZEN MF latent.

WHAT THIS IS (two citations, one module -- documented on purpose):
  * "Bıyık-style" best-effort belief recommender. Bıyık et al. (learning reward/utility functions from
    observations under a Gaussian belief) has no canonical *recommender* on ML-scale CF, so this is a
    faithful *adaptation* of the idea, not a verbatim reimplementation: a conjugate Gaussian belief over
    a user's latent taste vector, updated in closed form from observed likes, scoring items by the
    posterior-mean utility.  Flag at review as best-effort (no published ML-20M/25M number to snap to).
  * ConTS (Contextual Thompson Sampling, Li et al. / Zhang et al. conversational-rec line) posterior CORE.
    ConTS maintains EXACTLY this Bayesian linear-Gaussian posterior over the user vector and, per turn,
    draws a Thompson sample from it to pick an arm.  At FULL profile we do not need to explore, so the
    Thompson draw collapses to the posterior MEAN (the greedy exploit).  => the *same math* serves both
    baselines; only the read-out differs (mean here; a sample under the bandit).  Documented so one module
    legitimately backs both the Bıyık and the ConTS rows of the table.

MODEL (Bayesian linear regression over a frozen item basis):
  Freeze an iALS/WMF factorization (certified hp: factors=200, reg=0.1, alpha=50, 15 it -- the ML-20M
  val-selected constants also used by run_ml25m_liang).  Let Y (m x f) = frozen item factors.
    prior:       u ~ N(mu0, Sigma0),  mu0 = mean of TRAIN user factors, Sigma0 = their empirical cov.
    observation: each liked item i is a measurement  r_i = <u, Y_i> + eps,  eps ~ N(0, s2),  r_i := 1.
                 (s2 = obs_noise, the one tunable knob; smaller s2 = trust each like more.)
    posterior (conjugate, closed form):
                 A        = Sigma0^{-1} + (1/s2) * Y_S^T Y_S            (S = the user's observed items)
                 mu_post  = A^{-1} ( Sigma0^{-1} mu0 + (1/s2) * Y_S^T 1 )
    score(user) = Y @ mu_post          (posterior-mean utility over all items).
  No gradient training beyond the frozen iALS solve; the belief update is one f x f solve per user
  (f=200), identical in shape to the iALS cold-start fold-in already used in ials.py.

Interface: fit(train, n_items, evaluator=None, args=None, ckpt=None, log=print) -> predict_fn
  (evaluator/ckpt accepted for runner-convention parity; unused -- this method is closed-form.)
  predict_fn(fold_in_csr) -> dense np.float32 (batch x n_items) scores.
CLI: python src/baselines/belief_mf.py            # fit + eval on the Liang ML-25M split (full, no caps)
     python src/baselines/belief_mf.py --smoke     # tiny SYNTHETIC code-path check (NOT the real split)
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

# Certified frozen-MF hyperparameters (== run_ml25m_liang.IALS_HP; the ML-20M val-selected constants).
MF_HP = {"factors": 200, "reg": 0.1, "alpha": 50.0, "iterations": 15}
DEFAULTS = {"obs_noise": 1.0, "prior_scale": 1.0, "jitter": 1e-6}


def _fit_prior_and_basis(train, hp, log=print):
    """Freeze iALS -> (Y item factors, mu0, Sigma0_inv, Sigma0_inv @ mu0)."""
    model = ials._fit_model(train, hp["factors"], hp["reg"], hp["alpha"], hp["iterations"], log=log)
    Y = np.asarray(model.item_factors, dtype=np.float64)   # (m x f)
    U = np.asarray(model.user_factors, dtype=np.float64)   # (r x f) TRAIN users -> empirical prior
    mu0 = U.mean(axis=0)                                    # (f,)
    Sig0 = np.cov(U, rowvar=False)                          # (f x f) empirical covariance
    return Y, mu0, Sig0


def _make_predict(Y, mu0, Sig0, obs_noise, prior_scale, jitter, log=print):
    f = Y.shape[1]
    Sig0 = prior_scale * Sig0 + jitter * np.eye(f)         # jitter: keep SPD if cov is rank-deficient
    Sig0_inv = np.linalg.inv(Sig0)
    Sig0_inv_mu0 = Sig0_inv @ mu0
    inv_s2 = 1.0 / float(obs_noise)

    def predict(X_csr):
        Xc = X_csr.tocsr()
        b = Xc.shape[0]
        scores = np.zeros((b, Y.shape[0]), dtype=np.float32)
        for r in range(b):
            s0, e0 = Xc.indptr[r], Xc.indptr[r + 1]
            cols = Xc.indices[s0:e0]
            if len(cols) == 0:
                # no evidence -> posterior == prior -> score by prior-mean utility
                scores[r] = (Y @ mu0).astype(np.float32)
                continue
            Ys = Y[cols]                                   # (|S| x f)
            A = Sig0_inv + inv_s2 * (Ys.T @ Ys)            # posterior precision
            # OBSERVATION = the VALUE carried by the input matrix. Binary contract: every entry is 1.0,
            # so this reduces EXACTLY to the old Y_S^T 1 and the published row is bit-identical. Signed
            # interview contract: a dislike enters as a NEGATIVE measurement of the user's utility.
            # Frozen basis and empirical prior unchanged -- see rbmf_seed for the caveat about Y.
            obs = Ys.T @ Xc.data[s0:e0].astype(np.float64)
            rhs = Sig0_inv_mu0 + inv_s2 * obs              # Y_S^T r
            mu_post = np.linalg.solve(A, rhs)
            scores[r] = (Y @ mu_post).astype(np.float32)
        return scores
    return predict


def fit(train, n_items, evaluator=None, args=None, ckpt=None, log=print):
    a = args or argparse.Namespace(**DEFAULTS, **MF_HP)
    hp = {k: getattr(a, k, MF_HP[k]) for k in MF_HP}
    obs_noise = getattr(a, "obs_noise", DEFAULTS["obs_noise"])
    prior_scale = getattr(a, "prior_scale", DEFAULTS["prior_scale"])
    jitter = getattr(a, "jitter", DEFAULTS["jitter"])
    t0 = time.time()
    Y, mu0, Sig0 = _fit_prior_and_basis(train, hp, log=log)
    log(f"[belief_mf] frozen iALS f={hp['factors']} ({(time.time()-t0)/60:.1f}m); "
        f"prior mu0|={np.linalg.norm(mu0):.3f} tr(Sig0)={np.trace(Sig0):.3f}; obs_noise={obs_noise}")
    predict = _make_predict(Y, mu0, Sig0, obs_noise, prior_scale, jitter, log=log)
    predict.hp = dict(hp, obs_noise=obs_noise, prior_scale=prior_scale, role="Biyik-belief + ConTS-mean")
    return predict


# --------------------------------------------------------------------------- CLI / smoke
def _head_mask(train, n_items):
    """HEAD = smallest item set covering 33% of TRAIN mass (== run_ml25m_liang.compute_head_mask)."""
    cnt = np.asarray(train.sum(axis=0)).ravel().astype(np.float64)
    order = np.argsort(-cnt)
    cumfrac = np.cumsum(cnt[order]) / cnt.sum()
    head = np.zeros(n_items, dtype=bool)
    head[order[:np.searchsorted(cumfrac, 0.33) + 1]] = True
    return head


def _smoke():
    """SYNTHETIC code-path check ONLY -- not the real split, no snap to any published number."""
    print("[belief_mf][SMOKE] synthetic tiny data (code-path only, NOT the Liang split)")
    from scipy import sparse
    rng = np.random.RandomState(0)
    n_users, n_items = 120, 130   # >100 items: metrics.evaluate uses NDCG@100 / Recall@50
    X = sparse.random(n_users, n_items, density=0.15, random_state=rng, data_rvs=lambda s: np.ones(s))
    X = (X > 0).astype(np.float32).tocsr()
    tr = X[:90]
    te_tr, te_te = X[90:], (sparse.random(30, n_items, density=0.1, random_state=rng,
                                          data_rvs=lambda s: np.ones(s)) > 0).astype(np.float32).tocsr()
    a = argparse.Namespace(**DEFAULTS, **{"factors": 8, "reg": 0.1, "alpha": 50.0, "iterations": 3})
    pr = fit(tr, n_items, args=a, log=print)
    hm = _head_mask(tr, n_items)
    res = M.evaluate(pr, te_tr, te_te, batch_size=10, head_mask=hm)
    print(f"[belief_mf][SMOKE] OK  full@10={res['ndcg@10']:.4f} tail@10={res['tail_ndcg@10']:.4f} hp={pr.hp}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obs_noise", type=float, default=DEFAULTS["obs_noise"])
    ap.add_argument("--prior_scale", type=float, default=DEFAULTS["prior_scale"])
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
    a = argparse.Namespace(obs_noise=args.obs_noise, prior_scale=args.prior_scale,
                           jitter=DEFAULTS["jitter"], **MF_HP)
    t0 = time.time()
    predict = fit(train, n_items, args=a)
    res = M.evaluate(predict, te_tr, te_te, batch_size=500, head_mask=hm)
    res["seconds"] = time.time() - t0; res["hp"] = predict.hp
    print(f"[belief_mf] test full@10={res['ndcg@10']:.4f} tail@10={res['tail_ndcg@10']:.4f} "
          f"ndcg@100={res['ndcg@100']:.4f}")
    if args.out:
        json.dump(res, open(args.out, "w"), indent=2)


if __name__ == "__main__":
    main()
