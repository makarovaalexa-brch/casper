"""
eval_liang.py -- Liang et al. data loading + full-catalogue (non-sampled) metrics.

Metrics (verbatim from vae_cf/RecVAE):
  NDCG@100, Recall@20, Recall@50, computed over the full item catalogue,
  with fold-in ("tr") items masked out (-inf) before ranking.

Shared by multvae.py and recvae.py.
"""
import os, json
import numpy as np
from scipy import sparse

PROC = os.path.join("data", "ml20m", "proc")


def load_meta():
    with open(os.path.join(PROC, "meta.json")) as f:
        return json.load(f)


def _load_train_data(csv_file, n_items):
    import pandas as pd
    tp = pd.read_csv(csv_file)
    n_users = tp["uid"].max() + 1
    rows, cols = tp["uid"], tp["sid"]
    data = sparse.csr_matrix((np.ones_like(rows, dtype="float32"), (rows, cols)),
                             dtype="float32", shape=(n_users, n_items))
    return data


def _load_tr_te_data(csv_file_tr, csv_file_te, n_items):
    import pandas as pd
    tp_tr = pd.read_csv(csv_file_tr)
    tp_te = pd.read_csv(csv_file_te)
    start_idx = min(tp_tr["uid"].min(), tp_te["uid"].min())
    end_idx = max(tp_tr["uid"].max(), tp_te["uid"].max())
    rows_tr, cols_tr = tp_tr["uid"] - start_idx, tp_tr["sid"]
    rows_te, cols_te = tp_te["uid"] - start_idx, tp_te["sid"]
    data_tr = sparse.csr_matrix((np.ones_like(rows_tr, dtype="float32"), (rows_tr, cols_tr)),
                                dtype="float32", shape=(end_idx - start_idx + 1, n_items))
    data_te = sparse.csr_matrix((np.ones_like(rows_te, dtype="float32"), (rows_te, cols_te)),
                                dtype="float32", shape=(end_idx - start_idx + 1, n_items))
    return data_tr, data_te


def load_train(n_items):
    return _load_train_data(os.path.join(PROC, "train.csv"), n_items)


def load_val(n_items):
    return _load_tr_te_data(os.path.join(PROC, "validation_tr.csv"),
                            os.path.join(PROC, "validation_te.csv"), n_items)


def load_test(n_items):
    return _load_tr_te_data(os.path.join(PROC, "test_tr.csv"),
                            os.path.join(PROC, "test_te.csv"), n_items)


def NDCG_binary_at_k_batch(X_pred, heldout_batch, k=100):
    batch_users = X_pred.shape[0]
    idx_topk_part = np.argpartition(-X_pred, k, axis=1)
    topk_part = X_pred[np.arange(batch_users)[:, np.newaxis], idx_topk_part[:, :k]]
    idx_part = np.argsort(-topk_part, axis=1)
    idx_topk = idx_topk_part[np.arange(batch_users)[:, np.newaxis], idx_part]
    tp = 1.0 / np.log2(np.arange(2, k + 2))
    DCG = (heldout_batch[np.arange(batch_users)[:, np.newaxis], idx_topk].toarray() * tp).sum(axis=1)
    IDCG = np.array([(tp[:min(n, k)]).sum() for n in heldout_batch.getnnz(axis=1)])
    return DCG / IDCG


def Recall_at_k_batch(X_pred, heldout_batch, k=20):
    batch_users = X_pred.shape[0]
    idx = np.argpartition(-X_pred, k, axis=1)
    X_pred_binary = np.zeros_like(X_pred, dtype=bool)
    X_pred_binary[np.arange(batch_users)[:, np.newaxis], idx[:, :k]] = True
    X_true_binary = (heldout_batch > 0).toarray()
    tmp = (np.logical_and(X_true_binary, X_pred_binary).sum(axis=1)).astype(np.float32)
    recall = tmp / np.minimum(k, X_true_binary.sum(axis=1))
    return recall


def evaluate(predict_fn, data_tr, data_te, batch_size=500):
    """predict_fn: csr (batch x n_items) fold-in -> dense np (batch x n_items) scores.
    Returns dict of mean metrics. tr items masked with -inf before ranking."""
    n = data_tr.shape[0]
    n100, r20, r50 = [], [], []
    for st in range(0, n, batch_size):
        en = min(st + batch_size, n)
        X = data_tr[st:en]
        X_pred = predict_fn(X)
        X_pred[X.nonzero()] = -np.inf  # mask fold-in items
        he = data_te[st:en]
        n100.append(NDCG_binary_at_k_batch(X_pred, he, k=100))
        r20.append(Recall_at_k_batch(X_pred, he, k=20))
        r50.append(Recall_at_k_batch(X_pred, he, k=50))
    n100 = np.concatenate(n100); r20 = np.concatenate(r20); r50 = np.concatenate(r50)
    return {
        "ndcg@100": float(np.mean(n100)), "ndcg@100_std": float(np.std(n100) / np.sqrt(len(n100))),
        "recall@20": float(np.mean(r20)), "recall@20_std": float(np.std(r20) / np.sqrt(len(r20))),
        "recall@50": float(np.mean(r50)), "recall@50_std": float(np.std(r50) / np.sqrt(len(r50))),
    }
