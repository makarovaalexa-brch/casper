"""metrics.py -- Liang et al. (Mult-VAE, WWW 2018) data loaders + full-catalogue ranking metrics.

Shared by every baseline module and the two runners. The metric functions are a VERBATIM port of
the official vae_cf evaluation code (github.com/dawenl/vae_cf, `Rec_eval` in the WWW2018 notebook /
RecVAE `utils.py`), so that any "snap to published numbers" comparison uses the SAME truncation and
ideal-DCG conventions the papers used.

Provenance (dawenl/vae_cf, VAE_ML20M_WWW2018.ipynb, "Evaluate the model" cells):
  - NDCG_binary_at_k_batch: argpartition top-k, then argsort within top-k; gains tp = 1/log2(2..k+1);
    IDCG = sum of the first min(n_pos, k) gains  ->  DCG / IDCG.   (our NDCG@100 and NDCG@10)
  - Recall_at_k_batch: top-k indicator; recall = |pred_topk ∩ true| / min(k, |true|).  (R@20, R@50)
  - evaluate(): fold-in ("tr") items are masked to -inf BEFORE ranking, exactly as vae_cf does
    (`X_pred[X.nonzero()] = -np.inf`).

NDCG@10 is added here (same formula, k=10) for the CASPER ML-25M bridge; NDCG@100/R@20/R@50 are the
ML-20M published quantities.

Data layout (Liang split, produced by liang_split.py): PROC = data/ml-20m/proc/
  train.csv, validation_tr.csv, validation_te.csv, test_tr.csv, test_te.csv, unique_sid.txt, meta.json
"""
import os
import json
import numpy as np
from scipy import sparse

# All project data lives under data/ (project rule). Resolve relative to repo root.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
PROC = os.path.join(_ROOT, "data", "ml-20m", "proc")


# ----------------------------------------------------------------------------- loaders
def load_meta(proc=PROC):
    with open(os.path.join(proc, "meta.json")) as f:
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


def load_train(n_items, proc=PROC):
    return _load_train_data(os.path.join(proc, "train.csv"), n_items)


def load_val(n_items, proc=PROC):
    return _load_tr_te_data(os.path.join(proc, "validation_tr.csv"),
                            os.path.join(proc, "validation_te.csv"), n_items)


def load_test(n_items, proc=PROC):
    return _load_tr_te_data(os.path.join(proc, "test_tr.csv"),
                            os.path.join(proc, "test_te.csv"), n_items)


# ----------------------------------------------------------------------------- metrics (verbatim vae_cf)
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


def evaluate(predict_fn, data_tr, data_te, batch_size=500, ks=(100, 20, 50), head_mask=None):
    """predict_fn: csr (batch x n_items) fold-in -> dense np (batch x n_items) scores.
    Returns dict of mean NDCG@100, NDCG@10, Recall@20, Recall@50. tr items masked with -inf before
    ranking (vae_cf convention). Users with 0 held-out items are skipped.

    TAIL metric (head_mask given): head_mask is a boolean array over items (True == HEAD item, i.e. in
    the smallest set of items covering 33% of TRAIN interaction mass). When supplied we ALSO report
    `tail_ndcg@10`, computed by reusing NDCG_binary_at_k_batch (no duplicated metric): head-item scores
    are set to -inf so they can never be ranked, head held-out targets are dropped, and users whose
    held-out set is ALL head (no tail target) are skipped -- identical to signed_latent.ndcg10(tail=True).
    """
    n = data_tr.shape[0]
    acc = {"ndcg@100": [], "ndcg@10": [], "recall@20": [], "recall@50": []}
    do_tail = head_mask is not None
    if do_tail:
        head_mask = np.asarray(head_mask, dtype=bool)
        tail_row = (~head_mask).astype("float32")[np.newaxis, :]  # (1 x n_items), 1 on tail, 0 on head
        acc["tail_ndcg@10"] = []
    for st in range(0, n, batch_size):
        en = min(st + batch_size, n)
        X = data_tr[st:en]
        he = data_te[st:en]
        keep = np.asarray(he.getnnz(axis=1)).ravel() > 0  # skip users with no held-out target
        if not keep.any():
            continue
        X_pred = predict_fn(X)
        X_pred[X.nonzero()] = -np.inf  # mask fold-in items
        X_pred = X_pred[keep]
        he = he[keep]
        acc["ndcg@100"].append(NDCG_binary_at_k_batch(X_pred, he, k=100))
        acc["ndcg@10"].append(NDCG_binary_at_k_batch(X_pred, he, k=10))
        acc["recall@20"].append(Recall_at_k_batch(X_pred, he, k=20))
        acc["recall@50"].append(Recall_at_k_batch(X_pred, he, k=50))
        if do_tail:
            he_tail = he.multiply(tail_row).tocsr()          # drop head targets from held-out
            he_tail.eliminate_zeros()
            tkeep = np.asarray(he_tail.getnnz(axis=1)).ravel() > 0  # users with >=1 tail target
            if tkeep.any():
                Xp_t = X_pred[tkeep].copy()
                Xp_t[:, head_mask] = -np.inf                 # head items never rankable
                acc["tail_ndcg@10"].append(
                    NDCG_binary_at_k_batch(Xp_t, he_tail[tkeep], k=10))
    out = {}
    for key, chunks in acc.items():
        if not chunks:
            out[key] = float("nan"); out[key + "_se"] = float("nan"); continue
        v = np.concatenate(chunks)
        out[key] = float(np.mean(v))
        out[key + "_se"] = float(np.std(v) / np.sqrt(len(v)))
    return out
