r"""golbandi_native.py -- the Golbandi node recommender on its PUBLISHED input: ratings, mean-centred.

SPEC: docs/results/PROTOCOL_DISLIKE_DISCARD.md section 11 (arm N).

WHY THIS MODULE EXISTS. `golbandi_node.py` is the same node model on the canonical BINARY split: rows
are 0/1, similarity is plain cosine, the node score is a neighbour mean of indicators. That is the
right arm-A row, but it is not what Golbandi, Koren & Lempel (WSDM 2011) describe -- their tree routes
users by lovers / haters / unknowns and the node model is a RATING estimate. On binarised data the
hater branch cannot exist, so the arm-A row measures the like/unknown skeleton of the algorithm.

Feeding raw ratings into the binary code path is NOT the fix, and the void 29-Jul run shows why: a
cosine over unnormalised rating rows treats a 1.0-star rating as a positive contribution, so a user who
HATED a film is pulled toward neighbours who also rated it, and the node mean returns rating magnitudes
in which 0.5 and 5.0 both count upward. That run scored 0.2641 -- BELOW the binary 0.3065 -- which is
the signature of the bug, not evidence that ratings do not help.

THE FAITHFUL MODEL (standard neighbourhood formulation, Koren; matches the archived ML-100k tree's
shrunk `profile()` with LAM=8):
    mu_u   = user u's mean rating over their rated items
    c_ui   = r_ui - mu_u                            centred: a dislike is NEGATIVE, which is the point
    s_uv   = cos(c_u, c_v)                          adjusted-cosine / Pearson user-user similarity
    N      = top-k neighbours by s, negatives clipped (an anti-correlated user is not evidence)
    score_i = SUM_{v in N, v rated i} s_uv * c_vi  /  ( lambda + SUM_{v in N, v rated i} s_uv )
The shrinkage lambda is load-bearing for RANKING (as opposed to RMSE): without it an item rated by a
single enthusiastic neighbour outranks an item loved by forty, and the top-10 fills with noise.

This model is genuinely sign-aware: a neighbour who hated item i pushes i DOWN, which no binary
neighbour mean can express.

Interface: fit(train_graded, n_items, args=None, log=print) -> predict_fn
  predict_fn(fold_in_graded_csr) -> dense np.float32 (batch x n_items) scores.
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

DEFAULTS = {"k": 100, "lam": 8.0}


def _centre(X):
    """Row-centre a ratings CSR by each row's own mean, keeping the sparsity pattern.
    Returns (centred_csr, row_means). Empty rows keep mean 0 and stay empty."""
    X = X.tocsr().astype(np.float32)
    cnt = np.diff(X.indptr)
    tot = np.asarray(X.sum(axis=1)).ravel()
    mu = np.zeros_like(tot, dtype=np.float32)
    nz = cnt > 0
    mu[nz] = (tot[nz] / cnt[nz]).astype(np.float32)
    C = X.copy()
    C.data = C.data - np.repeat(mu, cnt)
    return C, mu


def _l2_normalize(X):
    X = X.tocsr().astype(np.float32)
    norms = np.sqrt(np.asarray(X.multiply(X).sum(axis=1)).ravel())
    norms[norms == 0] = 1.0
    return (sparse.diags(1.0 / norms) @ X).tocsr()


def fit(train, n_items, evaluator=None, args=None, ckpt=None, log=print):
    a = args or argparse.Namespace(**DEFAULTS)
    k = int(getattr(a, "k", DEFAULTS["k"]))
    lam = float(getattr(a, "lam", DEFAULTS["lam"]))

    C, _mu = _centre(train)                 # centred train ratings: the values the node mean averages
    Cn = _l2_normalize(C)                   # cosine basis over centred rows
    A = C.copy()
    A.data = np.ones_like(A.data)           # rated-indicator, for the per-item support denominator
    r = C.shape[0]
    kk = min(k, r)
    log(f"[golbandi_native] centred user-kNN: {r} train users, k={kk}, lambda={lam}")

    def predict(X_csr):
        Cq, _ = _centre(X_csr)
        Qn = _l2_normalize(Cq)
        b = Qn.shape[0]
        S = np.asarray((Qn @ Cn.T).todense(), dtype=np.float32)      # (b x r) adjusted cosine
        scores = np.zeros((b, n_items), dtype=np.float32)
        for i in range(b):
            sims = S[i]
            nbr = np.argpartition(-sims, kk - 1)[:kk] if kk < r else np.arange(r)
            w = np.maximum(sims[nbr], 0.0)                            # anti-correlated != evidence
            if w.sum() <= 0:
                continue
            num = w @ C[nbr].toarray()                                # signed: haters push DOWN
            den = lam + (w @ A[nbr].toarray())                        # shrink toward 0 on thin support
            # An item no neighbour rated has num == den == 0. With lambda > 0 that is 0/lambda = 0;
            # at lambda == 0 it is 0/0, and a NaN score sorts unpredictably rather than ranking last.
            # Score it 0 -- "no neighbourhood evidence", the same place shrinkage would put it.
            np.divide(num, den, out=scores[i], where=den > 0)
        return scores
    predict.k = kk
    predict.lam = lam
    return predict


# --------------------------------------------------------------------------- CLI
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=100)
    ap.add_argument("--lam", type=float, default=8.0)
    ap.add_argument("--sweep", action="store_true", help="val-select (k, lambda) on the arm-N protocol")
    ap.add_argument("--grid", default=None,
                    help="semicolon-separated k,lam pairs to val-sweep, e.g. '100,50;100,100'. Use to "
                         "extend a sweep whose winner sat on the grid edge -- an edge winner means the "
                         "baseline is under-tuned, and an under-tuned baseline is not a fair baseline.")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--binary_control", action="store_true",
                    help="EXISTENTIAL CONTROL. Run this same centred/shrunk estimator on the BINARY "
                         "matrices. If it lands near the binary user-kNN row (0.3756), the estimator "
                         "is sound and the ratings INPUT is what costs. If it also lands near 0.28, "
                         "the estimator itself is broken and the ratings-native row means nothing.")
    a = ap.parse_args()

    if a.smoke:
        rng = np.random.RandomState(0)
        n_u, n_i = 200, 130
        X = sparse.random(n_u, n_i, density=0.25, random_state=rng,
                          data_rvs=lambda s: rng.choice([0.5, 1.0, 2.0, 3.0, 4.0, 5.0], s)).tocsr()
        tr, te_tr = X[:150], X[150:]
        te_te = (sparse.random(50, n_i, density=0.1, random_state=rng,
                               data_rvs=lambda s: np.ones(s)) > 0).astype(np.float32).tocsr()
        cnt = np.asarray((tr > 0).sum(axis=0)).ravel().astype(np.float64)
        order = np.argsort(-cnt); cum = np.cumsum(cnt[order]) / cnt.sum()
        hm = np.zeros(n_i, bool); hm[order[:np.searchsorted(cum, 0.33) + 1]] = True
        pr = fit(tr, n_i, args=argparse.Namespace(k=20, lam=8.0))
        res = M.evaluate(pr, te_tr, te_te, batch_size=25, head_mask=hm)
        print(f"[golbandi_native][SMOKE] full@10={res['ndcg@10']:.4f} tail@10={res['tail_ndcg@10']:.4f}")
        return 0

    from arm_n import load_arm_n, _ROOT
    OUT = os.path.join(_ROOT, "experiments", "baselines", "arm_n")
    os.makedirs(OUT, exist_ok=True)

    def logln(m):
        line = f"[{time.strftime('%H:%M:%S')}] {m}"
        print(line, flush=True)
        with open(os.path.join(OUT, "golbandi_native.log"), "a") as f:
            f.write(line + "\n")

    D = load_arm_n(log=logln)
    if a.grid:
        grid = [(int(p.split(",")[0]), float(p.split(",")[1])) for p in a.grid.split(";")]
        a.sweep = True
    elif a.sweep:
        grid = [(100, 0.0), (100, 8.0), (100, 25.0), (300, 8.0), (300, 25.0)]
    else:
        grid = [(a.k, a.lam)]

    # Val selection uses the VAL cohort with the same protocol. Graded val fold-in is rebuilt here
    # rather than cached: it is only needed when --sweep is on.
    if a.sweep:
        from graded_data import _partition, _graded_rows
        unique_uid, show2id, raw = _partition()
        n_all = len(unique_uid)
        val_ids = unique_uid[n_all - 20000:n_all - 10000]
        va_tr, va_te = M.load_val(D["n_items"], os.path.join(_ROOT, "data", "ml-25m", "proc"))
        g_va_tr = _graded_rows(val_ids, show2id, raw, exclude=va_te)
        b_v = g_va_tr.copy(); b_v.data[:] = 1.0
        b_t = va_tr.copy(); b_t.data[:] = 1.0
        va_pool = (b_v + b_t).tocsr(); va_pool.data[:] = 1.0
        assert va_pool.multiply(va_te).nnz == 0, "val pool/target leak"
        logln(f"[golbandi_native] val graded fold-in nnz={g_va_tr.nnz}")

    best, results = None, []
    for (k, lam) in grid:
        ts = time.time()
        pr = fit(D["g_train"], D["n_items"], args=argparse.Namespace(k=k, lam=lam), log=logln)
        if a.sweep:
            v = M.evaluate(pr, g_va_tr, va_te, batch_size=500, head_mask=D["head_mask"],
                           mask_X=va_pool)
            logln(f"[golbandi_native] VAL k={k} lam={lam}: full={v['ndcg@10']:.4f} "
                  f"tail={v['tail_ndcg@10']:.4f} ({(time.time() - ts) / 60:.1f} m)")
            results.append({"k": k, "lam": lam, "val": v})
            if best is None or v["ndcg@10"] > best[0]:
                best = (v["ndcg@10"], k, lam)
        else:
            best = (None, k, lam)

    _, k, lam = best
    logln(f"[golbandi_native] TEST with k={k} lam={lam}"
          + (" [BINARY CONTROL: same estimator, binary input]" if a.binary_control else ""))
    ts = time.time()
    train_X, foldin_X = (D["train"], D["te_tr"]) if a.binary_control else (D["g_train"], D["g_te_tr"])
    pr = fit(train_X, D["n_items"], args=argparse.Namespace(k=k, lam=lam), log=logln)
    t = M.evaluate(pr, foldin_X, D["te_te"], batch_size=500, head_mask=D["head_mask"],
                   mask_X=D["pool"])
    logln(f"[golbandi_native] ARM-N TEST full@10={t['ndcg@10']:.4f} tail@10={t['tail_ndcg@10']:.4f} "
          f"ndcg@100={t['ndcg@100']:.4f} ({(time.time() - ts) / 60:.1f} m)")
    with open(os.path.join(OUT, ("golbandi_native_binctrl.json" if a.binary_control else "golbandi_native.json")), "w") as f:
        json.dump({"arm": "N", "model": "golbandi_native", "hp": {"k": k, "lam": lam},
                   "sweep": results, "test": t}, f, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
