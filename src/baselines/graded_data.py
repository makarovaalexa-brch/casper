r"""graded_data.py -- ratings-valued matrices over the canonical Liang ML-25M user/item sets.

WHY. Three baselines in the bank are ratings-native by publication: Golbandi's interview tree (RMSE, routes
users by lovers / haters / unknowns), RBMF / Functional-MF (a rating regression), and TaNP (episodic
meta-learning for rating prediction over (item, rating) support pairs). The canonical split keeps only
r>3.5, so running them off `train.csv` / `test_tr.csv` hands them binary support and, for Golbandi,
removes the branch its design routes on. This module rebuilds the same matrices with RATINGS as values so
each of those systems can be run faithful to its published design.

WHAT IS AND IS NOT CHANGED.
  * user set, item set and the train / val / test partition -- IDENTICAL to the canonical split
    (unique_uid order, unique_sid vocabulary, the same 140,768 / 10,000 / 10,000 cohorts).
  * HELD-OUT TARGETS -- identical and still binary. Only model INPUT changes, so NDCG@10 (full and tail)
    remains the number every other row in the bank reports, and the rows stay comparable.
  * values -- the user's catalogue rating, all bands (0.5 .. 5.0), for every rated (user, item) pair in
    the item vocabulary, EXCLUDING that user's held-out items. This is the same reconstruction the
    instrument's interview uses (run_battery_phaseA.build_real_ctx), so no system gets a private view of
    the data: the dataset is the shared resource and each design reads what it can from it.

  from graded_data import load_graded_train, load_graded_test
"""
import os
import numpy as np
import pandas as pd
from scipy import sparse

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
PROC = os.path.join(_ROOT, "data", "ml-25m", "proc")
RAW = os.path.join(_ROOT, "data", "movielens", "ratings.csv")


def _partition():
    """The canonical user order, item vocabulary and ALL-BANDS catalogue ratings, via the instrument's
    own reproducer -- which calls liang_split's helpers and constants. `unique_uid.txt` is not written by
    the split, so the order MUST come from here rather than be re-derived: a different permutation would
    silently produce a different test cohort and every number would be incomparable."""
    import sys
    sys.path.insert(0, os.path.join(_ROOT, "src", "instrument"))
    from train_tower_t2 import reproduce_partition
    unique_uid, _tr, _vd, _te, _ntr, raw, show2id, _usid = reproduce_partition()
    return unique_uid, show2id, raw[["userId", "movieId", "rating"]]


def _graded_rows(user_ids, show2id, raw, exclude=None):
    """(n_users x n_items) CSR of catalogue ratings for `user_ids`, minus each user's excluded items."""
    n_items = len(show2id)
    uid2row = {int(u): i for i, u in enumerate(user_ids)}
    df = raw[raw["userId"].isin(set(int(u) for u in user_ids))].copy()
    df["sid"] = df["movieId"].map(show2id)
    df = df[df["sid"].notna()]
    df["sid"] = df["sid"].astype(np.int64)
    rows = df["userId"].map(uid2row).to_numpy(np.int64)
    cols = df["sid"].to_numpy(np.int64)
    vals = df["rating"].to_numpy(np.float32)
    if exclude is not None:                       # drop each user's held-out targets
        keep = np.ones(len(rows), dtype=bool)
        for r in range(exclude.shape[0]):
            te = set(exclude[r].indices.tolist())
            if te:
                m = rows == r
                keep[m] &= ~np.isin(cols[m], list(te))
        rows, cols, vals = rows[keep], cols[keep], vals[keep]
    X = sparse.csr_matrix((vals, (rows, cols)), shape=(len(user_ids), n_items), dtype=np.float32)
    X.sum_duplicates()
    return X


def load_graded(te_te, log=print):
    """(graded_train, graded_test_foldin). Train = the canonical 140,768; test = the last 10,000 with
    held-out targets removed. Both carry catalogue ratings across all bands."""
    unique_uid, show2id, raw = _partition()
    n_all = len(unique_uid)
    train_ids = unique_uid[:n_all - 2 * 10000]
    test_ids = unique_uid[n_all - 10000:]
    Xtr = _graded_rows(train_ids, show2id, raw)
    log(f"[graded] train {Xtr.shape} nnz={Xtr.nnz} mean={Xtr.data.mean():.3f} "
        f"frac_sub3.5={float((Xtr.data <= 3.5).mean()):.3f}")
    Xte = _graded_rows(test_ids, show2id, raw, exclude=te_te)
    log(f"[graded] test fold-in {Xte.shape} nnz={Xte.nnz} mean={Xte.data.mean():.3f} "
        f"frac_sub3.5={float((Xte.data <= 3.5).mean()):.3f}")
    return Xtr, Xte
