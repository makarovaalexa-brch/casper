r"""imputer_models.py -- deferred NON-CIRCULAR concept-answer imputer arms E and P (author overnight
directive 2026-07-25). Both are SEPARATE latent-factor models (no recommender tower / decoder / u*
parameters) fit on TRAIN users, then applied to cold VAL users from their fold-in only -> the concept
answer value. Shared implicit weighted-MF (Hu et al. 2008) ALS core.

  E -- ExpoMF-style exposure model: content-conditioned exposure. The UNOBSERVED (negative) confidence
       per concept g is its content exposure prior mu_g (normalized member mass) instead of a flat 1 --
       niche concepts a user never watched are weak negatives, popular ones strong negatives (the
       exposure/popularity-confound fix, Liang et al. 2016 in spirit).
  P -- TagMF / PITF-style: a plain separate user x concept implicit factor model (flat negative
       confidence). The tag-affinity factorization baseline (Rendle PITF / tag-MF lineage; genome is
       item-level so the user x tag interaction is "rated a member").

Value pipeline (both): affinity(u,g) = theta_u . beta_g (cold theta solved from fold-in) -> per-concept
z-score -> signed value in [-1,1] -> the SAME four bands + C_NEG cap as signed_answers. Answerable where
the user has a real fold-in profile AND the concept has train support (coverage-complete, like content).
"""
import os
import numpy as np
from scipy import sparse

K_FACT = 16
ALS_ITERS = 6
ALPHA = 20.0
LAM = 1e-1


def _user_counts(Xbin, Mm):
    """(users x concepts) member-rating counts from a user x item binary matrix."""
    return (Xbin @ Mm).tocsr()


def solve_factors(Xc, other, A0, w_neg, alpha, lam):
    """Solve one ALS side: for each row u of Xc (m x C counts), theta_u given `other` (C x k) factors.
    A0 = other^T diag(w_neg) other (k x k). Returns (m x k)."""
    k = other.shape[1]
    m = Xc.shape[0]
    Th = np.zeros((m, k), np.float32)
    Xc = Xc.tocsr()
    A0lam = A0 + lam * np.eye(k, dtype=np.float64)
    for u in range(m):
        s, e = Xc.indptr[u], Xc.indptr[u + 1]
        if e == s:
            continue
        cols = Xc.indices[s:e]
        cnt = Xc.data[s:e].astype(np.float64)
        c = 1.0 + alpha * cnt                                    # positive confidence
        bo = other[cols].astype(np.float64)                     # (o, k)
        A = A0lam + (bo * (c - w_neg[cols])[:, None]).T @ bo     # A0 + correction
        b = (bo * c[:, None]).sum(0)                             # since p=1 on observed
        try:
            Th[u] = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            Th[u] = np.linalg.lstsq(A, b, rcond=None)[0]
    return Th


def fit_wmf(Xtr, C, w_neg, k=K_FACT, iters=ALS_ITERS, alpha=ALPHA, lam=LAM, seed=4242, log=print):
    """Weighted-MF ALS on the train user x concept count matrix. w_neg: (C,) per-concept negative
    confidence (exposure prior for E, flat 1 for P). Returns beta (C x k)."""
    rng = np.random.default_rng(seed)
    n = Xtr.shape[0]
    theta = (rng.standard_normal((n, k)) * 0.01).astype(np.float32)
    beta = (rng.standard_normal((C, k)) * 0.01).astype(np.float32)
    Wn = w_neg.astype(np.float64)
    XtrT = Xtr.T.tocsr()
    for it in range(iters):
        A0b = (beta.astype(np.float64) * Wn[:, None]).T @ beta.astype(np.float64)   # beta^T diag(w) beta
        theta = solve_factors(Xtr, beta, A0b, Wn, alpha, lam)
        A0t = theta.astype(np.float64).T @ theta.astype(np.float64)                 # user side w=1
        wone = np.ones(n)
        beta = solve_factors(XtrT, theta, A0t, wone, alpha, lam)
        log(f"[wmf] iter {it+1}/{iters} done (|theta|={np.linalg.norm(theta):.1f} "
            f"|beta|={np.linalg.norm(beta):.1f})")
    return beta


def _bands(V, prereg):
    from signed_answers import (TAU_REF, C_NEG, BAND_LIKE, BAND_MEH, BAND_DISLIKE, BAND_REFUSE)
    t_like = prereg["t_like_p60pos"]; t_neg = prereg["t_neg_absp25"]
    B = np.full(V.shape, BAND_MEH, np.int8)
    B[V > t_like] = BAND_LIKE
    B[V < -t_neg] = BAND_DISLIKE
    return B, C_NEG, BAND_MEH, BAND_DISLIKE, BAND_REFUSE


def _build(ctx, sh, Mm, pexp, exposure_weighted, smoke, cache_name, log):
    """Shared builder for E/P. Returns (V, F, answerable) at (n, C)."""
    from signed_answers import load_prereg
    import metrics as M
    prereg, item_mean = load_prereg()
    C = len(sh["tags"])
    CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".cache",
                         "instrument", cache_name)
    # per-concept content exposure prior (normalized member mass) for E; flat for P
    gmass = ctx.cnt @ np.asarray(Mm.todense())
    mu = (gmass / max(gmass.max(), 1e-9)).astype(np.float64)
    mu = np.clip(mu, 1e-3, 1.0)
    w_neg = mu if exposure_weighted else np.ones(C)
    # TRAIN counts (user x concept). smoke: reuse val fold-in as a stand-in train set.
    if smoke or not hasattr(ctx, "tr_set"):
        Xb_tr = ctx.allb_mask if hasattr(ctx, "allb_mask") else ctx.va_tr
        Xtr = _user_counts(Xb_tr, Mm)
    else:
        train = M.load_train(ctx.ni, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                   "..", "..", "data", "ml-25m", "proc"))
        Xtr = _user_counts(train, Mm)
    if (not smoke) and os.path.exists(CACHE):
        beta = np.load(CACHE)["beta"]
        log(f"[imp] loaded cached factors {cache_name}")
    else:
        beta = fit_wmf(Xtr, C, w_neg, log=log)
        if not smoke:
            os.makedirs(os.path.dirname(CACHE), exist_ok=True)
            np.savez(CACHE, beta=beta)
    # cold val theta from fold-in (allb binary), then affinity
    items_l = [np.asarray(s, np.int64) for s, l in ctx.allb]
    r_idx = np.concatenate([np.full(len(s), i) for i, s in enumerate(items_l)]) if items_l else np.array([])
    c_idx = np.concatenate(items_l) if items_l else np.array([])
    Xb_val = sparse.csr_matrix((np.ones(len(c_idx), np.float32), (r_idx, c_idx)),
                               shape=(ctx.n, ctx.ni))
    Xval = _user_counts(Xb_val, Mm)
    A0b = (beta.astype(np.float64) * w_neg[:, None]).T @ beta.astype(np.float64)
    theta = solve_factors(Xval, beta, A0b, w_neg, ALPHA, LAM)
    aff = (theta @ beta.T).astype(np.float32)                    # (n, C) affinity
    dl = np.asarray(Xb_val.sum(axis=1)).ravel()
    nc = np.asarray(Xval.todense())
    mu_a = aff.mean(0, keepdims=True); sd = aff.std(0, keepdims=True) + 1e-6
    z = (aff - mu_a) / sd
    V = np.clip(0.5 * z, -1.0, 1.0).astype(np.float32)
    B, C_NEG, BAND_MEH, BAND_DISLIKE, BAND_REFUSE = _bands(V, prereg)
    trained_support = (np.asarray(Xtr.sum(0)).ravel() > 0)
    answerable = (dl[:, None] >= 3) & trained_support[None, :]
    B[~answerable] = BAND_REFUSE
    F = (B != BAND_REFUSE)
    V = np.where(B == BAND_MEH, 0.0, V).astype(np.float32)
    negmask = (B == BAND_DISLIKE) & F
    negsum = np.abs(np.where(negmask, V, 0.0)).sum(axis=1)
    scale = np.minimum(1.0, C_NEG / np.maximum(negsum, 1e-9))
    V = np.where(negmask, V * scale[:, None], V).astype(np.float32)
    return V, F, answerable


def build_expomf_arrays(ctx, sh, Mm, pexp, smoke=False, log=print):
    """(E) ExpoMF-style content-conditioned exposure imputer."""
    return _build(ctx, sh, Mm, pexp, exposure_weighted=True, smoke=smoke,
                  cache_name="imputer_expomf_beta.npz", log=log)


def build_tagmf_arrays(ctx, sh, Mm, pexp, smoke=False, log=print):
    """(P) TagMF / PITF-style separate user x concept factor imputer."""
    return _build(ctx, sh, Mm, pexp, exposure_weighted=False, smoke=smoke,
                  cache_name="imputer_tagmf_beta.npz", log=log)


class FactorImputer:
    """Concept answer from a fitted factor model; items = the real fold-in rating (concept-only imputer)."""
    geometric = False

    def __init__(self, V, F, answerable, lvl_lookup):
        self.V = V; self.F = F; self.ans = answerable; self.lk = lvl_lookup

    def concept_value(self, r, c):
        if self.ans[r, c] and self.F[r, c]:
            return float(self.V[r, c]), True
        return 0.0, False

    def item_value(self, r, i):
        d = self.lk[r]
        return (float(d[i]), True) if i in d else (0.0, False)


def _smoke():
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "baselines"))
    rng = np.random.RandomState(0)
    C = 12; ni = 200; n = 40
    Mm = sparse.random(ni, C, density=0.15, format="csr", random_state=1)
    Mm.data[:] = 1.0
    Xtr = sparse.random(n, C, density=0.3, format="csr", random_state=2)
    Xtr.data = np.rint(Xtr.data * 5) + 1
    w = np.ones(C)
    beta = fit_wmf(Xtr, C, w, k=8, iters=3, log=lambda *a: None)
    assert beta.shape == (C, 8) and np.isfinite(beta).all()
    A0 = (beta.astype(np.float64)).T @ beta.astype(np.float64)
    th = solve_factors(Xtr, beta, A0, w, ALPHA, LAM)
    assert th.shape == (n, 8) and np.isfinite(th).all()
    print("[SMOKE imputer_models] PASS: WMF-ALS fits, factors finite, cold-solve finite")


if __name__ == "__main__":
    _smoke()
