"""itemknn.py -- Item-based k-nearest-neighbour CF with cosine similarity (standard settings).

The Cremonesi-era strong-simple neighbourhood baseline. Standard implicit-feedback item-kNN:
  - cosine similarity between item columns of the binary train matrix X (r users x m items):
        S = normalize_cols(X)^T @ normalize_cols(X);  S_ii := 0 (exclude self).
  - keep top-k neighbours per item (k=200 default, a common ML-20M-scale setting), zero the rest.
  - fold-in score for a new user with item-set u:  score = x_u @ S   (sum of similarities to held items).

k=200 and cosine are the conventional defaults (e.g. RecBole / Dacrema replicability item-kNN);
this is a standard baseline, not a paper with one canonical number, so we sweep k on validation and
report the chosen k (documented in the result JSON). No shrinkage term (plain cosine).

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

DEFAULTS = {"topk": 200}


def _cosine_topk(train, topk, log=print):
    X = train.tocsc().astype(np.float32)
    # column (item) L2 norms
    norms = np.sqrt(np.asarray(X.multiply(X).sum(axis=0)).ravel())
    norms[norms == 0] = 1.0
    Xn = X.multiply(1.0 / norms)           # normalized columns
    Xn = Xn.tocsr()
    S = (Xn.T @ Xn).tocsr()                # m x m cosine similarity (dense-ish sparse)
    S.setdiag(0.0)
    S.eliminate_zeros()
    # keep top-k per row (per item)
    m = S.shape[0]
    data, indices, indptr = [], [], [0]
    for i in range(m):
        lo, hi = S.indptr[i], S.indptr[i + 1]
        d = S.data[lo:hi]; c = S.indices[lo:hi]
        if len(d) > topk:
            sel = np.argpartition(-d, topk)[:topk]
            d = d[sel]; c = c[sel]
        data.append(d); indices.append(c); indptr.append(indptr[-1] + len(d))
        if i % 4000 == 0:
            log(f"  [itemknn] top-{topk} pruning {i}/{m}")
    S = sparse.csr_matrix((np.concatenate(data), np.concatenate(indices), np.array(indptr)),
                          shape=(m, m), dtype=np.float32)
    return S


def fit(train, n_items, evaluator=None, args=None, ckpt=None, log=print):
    topk = (args.topk if args and getattr(args, "topk", None) else DEFAULTS["topk"])
    t0 = time.time()
    S = _cosine_topk(train, topk, log=log)
    log(f"[itemknn] built cosine S (top-{topk}), nnz={S.nnz}, {(time.time()-t0)/60:.1f}m")

    def predict(X_csr):
        return np.asarray((X_csr @ S).todense(), dtype=np.float32)
    predict.topk = topk
    return predict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--topk", type=int, default=DEFAULTS["topk"])
    ap.add_argument("--sweep", action="store_true", help="sweep topk on validation, report best")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    meta = M.load_meta(); n_items = meta["n_items"]
    train = M.load_train(n_items)
    va_tr, va_te = M.load_val(n_items)
    te_tr, te_te = M.load_test(n_items)
    t0 = time.time()
    best_k, best_v = args.topk, -1.0
    if args.sweep:
        for k in (50, 100, 200, 500):
            pr = fit(train, n_items, args=argparse.Namespace(topk=k))
            v = M.evaluate(pr, va_tr, va_te)["ndcg@100"]
            print(f"[itemknn] VAL topk={k} ndcg@100={v:.4f}")
            if v > best_v:
                best_v, best_k = v, k
    predict = fit(train, n_items, args=argparse.Namespace(topk=best_k))
    res = M.evaluate(predict, te_tr, te_te)
    res["topk"] = best_k; res["seconds"] = time.time() - t0
    print(f"[itemknn] test {res}")
    if args.out:
        json.dump(res, open(args.out, "w"), indent=2)


if __name__ == "__main__":
    main()
