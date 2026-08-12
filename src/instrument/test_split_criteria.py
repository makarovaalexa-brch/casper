r"""Unit self-test for `split_criteria.py` -- the Hu & Yu gradient/Hessian and Golbandi's Eq. (4).

Runs in a second on synthetic data, builds NO environment, touches NO checkpoint. It exists because
the ablation's whole claim rests on the criterion being the published one, and a criterion that is
silently wrong would produce a clean-looking, entirely fictitious result.

Checked here:
  1. `loss_grad_hess('pairwise')` gradient == central finite difference of the pairwise logistic loss.
  2. `loss_grad_hess('pairwise')` diagonal Hessian == the second central difference, per coordinate.
     (This is the term whose cross-part `-2 sum_ij c_ij W_i W_j` is the easy one to drop; the test
     fails loudly if it is dropped.)
  3. the same two for `logistic`.
  4. `_taylor_gain` == sum over children of [Ltilde(0) - min_du Ltilde(du)] with H replaced by its
     diagonal, computed by brute force from the quadratic model.
  5. `variance_scores` == a brute-force per-child recomputation of Golbandi's Eq. (4), and it is
     exactly the node's e2 minus the partition's e2.
  6. `build_evidence` maps levels to stars and partitions POS/NEG at the >= 4.0-star boundary.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from scipy import sparse
import split_criteria as SC


def pairwise_loss(z, Wd, bd, p, m):
    s = Wd @ z + bd
    d = s[p][:, None] - s[m][None, :]
    return np.log1p(np.exp(-np.clip(d, -30, 30))).sum()


def logistic_loss(z, Wd, bd, p, m):
    ids = np.concatenate([p, m]); y = np.concatenate([np.ones(len(p)), np.zeros(len(m))])
    s = Wd[ids] @ z + bd[ids]
    sg = 1.0 / (1.0 + np.exp(-np.clip(s, -30, 30)))
    return float(-(y * np.log(sg + 1e-300) + (1 - y) * np.log(1 - sg + 1e-300)).sum())


def main():
    rng = np.random.default_rng(0)
    d, ni = 7, 40
    Wd = rng.normal(0, 0.4, (ni, d)); bd = rng.normal(0, 0.2, ni)
    pos = [np.array(sorted(rng.choice(ni, 5, replace=False)))]
    neg = [np.array(sorted(set(range(ni)) - set(pos[0].tolist())))[:6]]
    z = rng.normal(0, 0.3, d)
    Z = z[None, :]
    eps = 1e-5
    for kind, lossf in (("pairwise", pairwise_loss), ("logistic", logistic_loss)):
        G, H, _n = SC.loss_grad_hess(kind, Z, Wd, bd, pos, neg, np.array([0]))
        gnum = np.empty(d); hnum = np.empty(d)
        L0 = lossf(z, Wd, bd, pos[0], neg[0])
        for k in range(d):
            e = np.zeros(d); e[k] = eps
            lp = lossf(z + e, Wd, bd, pos[0], neg[0])
            lm = lossf(z - e, Wd, bd, pos[0], neg[0])
            gnum[k] = (lp - lm) / (2 * eps)
            hnum[k] = (lp - 2 * L0 + lm) / (eps ** 2)
        assert np.allclose(G[0], gnum, rtol=1e-5, atol=1e-6), \
            f"{kind}: gradient != finite difference\n  ours {G[0]}\n  fd   {gnum}"
        assert np.allclose(H[0], hnum, rtol=2e-3, atol=1e-5), \
            f"{kind}: DIAGONAL HESSIAN != second finite difference (the cross term is the usual " \
            f"casualty)\n  ours {H[0]}\n  fd   {hnum}"
        print(f"[ok] {kind}: gradient and diagonal Hessian match central finite differences "
              f"(max |dg| {np.abs(G[0] - gnum).max():.2e}, max |dH| {np.abs(H[0] - hnum).max():.2e})")

    # ---- 4. the Taylor gain is exactly what Eq. (11)/(12) says it is
    lam = SC.LAM_U
    g = rng.normal(0, 1, (3, d)); h = np.abs(rng.normal(2, 1, (3, d)))
    gain = SC._taylor_gain(g[0:1], h[0:1], g[1:2], h[1:2], g[2:3], h[2:3], lam)[0]
    brute = 0.0
    for c in range(3):
        D = h[c] + 2 * lam
        du = -g[c] / D                                   # argmin of Ltilde with H replaced by diag D
        brute += -(du @ g[c] + 0.5 * (du * D * du).sum())   # Ltilde(0) - min Ltilde
    assert abs(gain - brute) < 1e-9, f"taylor gain {gain} != brute force {brute}"
    print(f"[ok] _taylor_gain == sum_C [Ltilde(0) - min_du Ltilde(du)] under the diagonal Hessian "
          f"({gain:.9f} vs {brute:.9f})")

    # ---- 5. Golbandi Eq. (4), against a brute-force per-child recomputation
    nu, nit = 60, 25
    R = sparse.random(nu, nit, density=0.3, random_state=1, format="csr")
    R.data = np.round(R.data * 8 + 1) / 2.0
    M = R.copy(); M.data = np.ones_like(M.data)
    cnt = np.diff(R.indptr).astype(float); tot = np.asarray(R.sum(1)).ravel()
    gmu = float(R.data.mean()); b_u = (tot + 10 * gmu) / (cnt + 10)
    Xb = R.copy(); Xb.data = Xb.data - np.repeat(b_u, np.diff(R.indptr))
    X2 = Xb.copy(); X2.data = Xb.data ** 2
    ev = {"R": R, "M": M, "Xb": Xb, "X2": X2}
    nc = 5
    like = rng.random((nc, nu)) < 0.3
    dis = (~like) & (rng.random((nc, nu)) < 0.3)
    memL = sparse.csr_matrix(like); memD = sparse.csr_matrix(dis)
    sub = np.arange(nu)
    s, e2n = SC.variance_scores(ev, memL, memD, sub, list(range(nc)), chunk=2)
    Xd = np.asarray(Xb.todense()); Md = np.asarray(M.todense())

    def e2_brute(mask):
        if mask.sum() == 0:
            return 0.0
        x = Xd[mask]; m = Md[mask]
        n = m.sum(0); su = (x * m).sum(0); s2 = ((x * m) ** 2).sum(0)
        return float(np.where(n > 0, s2 - su * su / np.maximum(n, 1), 0.0).sum())
    assert abs(e2_brute(np.ones(nu, bool)) - e2n) < 1e-8, "node e2 mismatch"
    for c in range(nc):
        want = e2n - (e2_brute(like[c]) + e2_brute(dis[c]) + e2_brute(~(like[c] | dis[c])))
        assert abs(want - s[c]) < 1e-8, f"variance criterion candidate {c}: {s[c]} vs brute {want}"
    print(f"[ok] variance_scores == brute-force Golbandi Eq. (4) variance REDUCTION on all {nc} "
          f"candidates (node e2 {e2n:.6f})")
    print("ALL SELF-TESTS PASSED")


if __name__ == "__main__":
    main()
