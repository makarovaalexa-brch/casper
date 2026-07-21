"""pop.py -- Most-Popular (non-personalized) baseline.

Score every item by its training popularity (number of users who interacted with it), identical for
all users. The `dacrema2019progress` sanity floor: any personalized method must beat this. We rank by
log(1+count) which is order-identical to raw count (ranking is invariant to the monotone transform) --
kept for numerical parity with the CASPER `popb` floor.

Interface (shared by all baselines):
  fit(train_csr, n_items, evaluator=None, args=None, ckpt=None, log=print) -> predict_fn
  predict_fn(fold_in_csr) -> dense np.float32 (batch x n_items) scores
"""
import os
import sys
import json
import time
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import metrics as M

DEFAULTS = {}


def fit(train, n_items, evaluator=None, args=None, ckpt=None, log=print):
    cnt = np.asarray(train.sum(axis=0)).ravel().astype(np.float64)  # item interaction counts
    popb = np.log(cnt + 1.0).astype(np.float32)

    def predict(X_csr):
        b = X_csr.shape[0]
        return np.tile(popb, (b, 1))
    log(f"[pop] fitted item popularity over {train.shape[0]} train users, {n_items} items")
    return predict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="pop")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    meta = M.load_meta(); n_items = meta["n_items"]
    train = M.load_train(n_items)
    te_tr, te_te = M.load_test(n_items)
    t0 = time.time()
    predict = fit(train, n_items)
    res = M.evaluate(predict, te_tr, te_te)
    res["seconds"] = time.time() - t0
    print(f"[pop] test {res}")
    if args.out:
        json.dump(res, open(args.out, "w"), indent=2)


if __name__ == "__main__":
    main()
