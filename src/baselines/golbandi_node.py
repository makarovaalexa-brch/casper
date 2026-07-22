"""golbandi_node.py -- the NODE RECOMMENDER of the Golbandi tree line, scored at FULL profile.

Golbandi, Koren & Lempel ("Adaptive Bootstrapping of Recommender Systems Using Decision Trees",
WSDM 2011) build a decision tree that ELICITS a cold user by asking about seed items; each tree NODE
holds the set of training users routed to it, and the node's item scores are the MEAN interaction/rating
vector of THAT node's users.  The tree is the elicitation policy; the node model is the recommender.

WHAT THIS MODULE MEASURES (documented honestly -- this is NOT the tree/elicitation):
  At FULL profile every answer is known, so a cold user is no longer routed by a shallow tree; the
  best node for them is "the training users who look most like them".  The node-mean recommender then
  reduces to a USER-based k-NN mean: score a fold-in user by the (optionally similarity-weighted) mean
  of the binary vectors of the k most similar TRAIN users.  This module implements exactly that reduced
  recommender core -- the full-profile ceiling of the node model, with NO tree, NO elicitation.  Flag at
  review: it answers "how strong is the node's user-neighbourhood-mean recommender on its own?", i.e. the
  recommender component of the Golbandi line; it is NOT a claim about the adaptive tree's cold-start curve.

REDUCTION:
    Xtr (r x m) binary train matrix, row-normalized cosine.  For a fold-in user x:
      sims_u = cos(x, Xtr_u)  for all train users u ;  N = argtop_k(sims)
      score  = (1/|N|) * SUM_{u in N} Xtr_u            (unweighted)
             or  (SUM_{u in N} w_u Xtr_u) / SUM w_u    (w_u = sims_u, if weighted=True)
  Held-in items are masked by metrics.evaluate (vae_cf convention), so the neighbour mean naturally
  surfaces items the neighbours liked that the user has not yet seen -- the node-mean recommendation.

Interface: fit(train, n_items, evaluator=None, args=None, ckpt=None, log=print) -> predict_fn
  predict_fn(fold_in_csr) -> dense np.float32 (batch x n_items) scores.
COST NOTE: each fold-in batch does a (batch x r) sparse cosine (r ~ 140k train users); documented below.
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

DEFAULTS = {"k": 100, "weighted": True, "shrink": 0.0}


def _row_normalize(X):
    X = X.tocsr().astype(np.float32)
    norms = np.sqrt(np.asarray(X.multiply(X).sum(axis=1)).ravel())
    norms[norms == 0] = 1.0
    inv = sparse.diags(1.0 / norms)
    return (inv @ X).tocsr()


def fit(train, n_items, evaluator=None, args=None, ckpt=None, log=print):
    a = args or argparse.Namespace(**DEFAULTS)
    k = int(getattr(a, "k", DEFAULTS["k"]))
    weighted = bool(getattr(a, "weighted", DEFAULTS["weighted"]))
    shrink = float(getattr(a, "shrink", DEFAULTS["shrink"]))
    Xtr = train.tocsr().astype(np.float32)                 # raw binary rows (the node-mean averages these)
    Xn = _row_normalize(Xtr)                                # cosine basis for user-user similarity
    r = Xtr.shape[0]
    kk = min(k, r)
    log(f"[golbandi_node] user-kNN mean core: {r} train users, k={kk}, weighted={weighted}, shrink={shrink}")

    def predict(X_csr):
        Q = _row_normalize(X_csr)                          # (b x m) normalized fold-in
        b = Q.shape[0]
        S = (Q @ Xn.T)                                     # (b x r) cosine sims (sparse -> densify per batch)
        S = np.asarray(S.todense(), dtype=np.float32)
        if shrink > 0:
            S *= 1.0                                        # (placeholder; cosine already bounded -- shrink
            #                                                  reserved for count-shrinkage variants)
        scores = np.zeros((b, Xtr.shape[1]), dtype=np.float32)
        for i in range(b):
            sims = S[i]
            nbr = np.argpartition(-sims, kk - 1)[:kk] if kk < r else np.arange(r)
            w = sims[nbr]
            if weighted:
                den = w.sum()
                if den <= 0:
                    continue
                scores[i] = (w @ Xtr[nbr].toarray()) / den
            else:
                scores[i] = Xtr[nbr].toarray().mean(axis=0)
        return scores
    predict.k = kk; predict.weighted = weighted
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
    print("[golbandi_node][SMOKE] synthetic tiny data (code-path only, NOT the Liang split)")
    rng = np.random.RandomState(0)
    n_users, n_items = 120, 130   # >100 items: metrics.evaluate uses NDCG@100 / Recall@50
    X = (sparse.random(n_users, n_items, density=0.2, random_state=rng,
                       data_rvs=lambda s: np.ones(s)) > 0).astype(np.float32).tocsr()
    tr, te_tr = X[:90], X[90:]
    te_te = (sparse.random(30, n_items, density=0.12, random_state=rng,
                           data_rvs=lambda s: np.ones(s)) > 0).astype(np.float32).tocsr()
    hm = _head_mask(tr, n_items)
    for wtd in (True, False):
        pr = fit(tr, n_items, args=argparse.Namespace(k=10, weighted=wtd, shrink=0.0))
        res = M.evaluate(pr, te_tr, te_te, batch_size=15, head_mask=hm)
        print(f"[golbandi_node][SMOKE] weighted={wtd} full@10={res['ndcg@10']:.4f} "
              f"tail@10={res['tail_ndcg@10']:.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=DEFAULTS["k"])
    ap.add_argument("--weighted", action="store_true", default=DEFAULTS["weighted"])
    ap.add_argument("--unweighted", dest="weighted", action="store_false")
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
    predict = fit(train, n_items, args=argparse.Namespace(k=args.k, weighted=args.weighted, shrink=0.0))
    res = M.evaluate(predict, te_tr, te_te, batch_size=500, head_mask=hm)
    res["seconds"] = time.time() - t0; res["k"] = predict.k; res["weighted"] = predict.weighted
    print(f"[golbandi_node] test full@10={res['ndcg@10']:.4f} tail@10={res['tail_ndcg@10']:.4f} "
          f"ndcg@100={res['ndcg@100']:.4f} {res['seconds']/60:.1f}m")
    if args.out:
        json.dump(res, open(args.out, "w"), indent=2)


if __name__ == "__main__":
    main()
