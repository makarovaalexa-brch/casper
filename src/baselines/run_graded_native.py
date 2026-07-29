r"""run_graded_native.py -- re-run the ratings-native baselines faithful to their published design.

Golbandi's interview tree, RBMF / Functional-MF and TaNP are ratings-native by publication. The canonical
split keeps only r>3.5, so on it they receive binary support and 55.7% of the ratings their designs consume
are absent. This re-runs them on the graded catalogue ratings (graded_data.py) over the IDENTICAL user and
item sets, with held-out targets excluded, and scores them with the same metrics.evaluate against the same
binary targets -- so only the model INPUT changes and the numbers stay comparable to every other row.

  golbandi_node   value-agnostic: the node score is a mean over neighbour rows, which becomes a mean
                  RATING, and the user-user cosine is computed on graded rows. Fully faithful.
  rbmf_seed       value-agnostic in the seed step: the per-user ridge LS now regresses the user's actual
                  ratings on the frozen item factors instead of ones. SUBSTITUTION STATED: the frozen Y
                  still comes from the implicit iALS fit (as the existing row does); only the elicitation
                  seed step -- the mechanism the interview exercises -- becomes graded.
  tanp            needs its rating channel un-pinned and a retrain; run separately with --tanp.

  python src/baselines/run_graded_native.py            # golbandi + rbmf (closed form, minutes)
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "6")
import sys
import json
import time
import argparse
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, _HERE)

import metrics as M
import golbandi_node, rbmf_seed
from graded_data import load_graded
from run_ml25m_liang import compute_head_mask

PROC = os.path.join(_ROOT, "data", "ml-25m", "proc")
OUTDIR = os.path.join(_ROOT, "experiments", "baselines", "ml25m_liang")


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)


def evaluate_decoupled(predict_fn, input_X, mask_X, data_te, head_mask, batch_size=500):
    """metrics.evaluate mirrored, with the model INPUT decoupled from the RANKING MASK.

    metrics.evaluate uses one matrix for both, which is correct when they are the same object. Handing it
    a richer fold-in silently also masks more items out of the ranking -- and the extra items are the
    user's sub-3.5 ratings, i.e. exactly the confusable candidates a recommender would otherwise rank
    high. That inflates NDCG and makes the row incomparable to every other row in the bank. Here the
    model reads `input_X` while the mask stays `mask_X` (the canonical fold-in), so the ranking task is
    byte-identical across arms and only the input regime varies."""
    from metrics import NDCG_binary_at_k_batch, Recall_at_k_batch
    n = mask_X.shape[0]
    acc = {"ndcg@100": [], "ndcg@10": [], "recall@20": [], "recall@50": [], "tail_ndcg@10": []}
    head_mask = np.asarray(head_mask, dtype=bool)
    tail_row = (~head_mask).astype("float32")[np.newaxis, :]
    for st in range(0, n, batch_size):
        en = min(st + batch_size, n)
        Xin, Xmask, he = input_X[st:en], mask_X[st:en], data_te[st:en]
        keep = np.asarray(he.getnnz(axis=1)).ravel() > 0
        if not keep.any():
            continue
        X_pred = predict_fn(Xin)
        X_pred[Xmask.nonzero()] = -np.inf          # canonical mask, NOT the (richer) input
        X_pred, he = X_pred[keep], he[keep]
        acc["ndcg@100"].append(NDCG_binary_at_k_batch(X_pred, he, k=100))
        acc["ndcg@10"].append(NDCG_binary_at_k_batch(X_pred, he, k=10))
        acc["recall@20"].append(Recall_at_k_batch(X_pred, he, k=20))
        acc["recall@50"].append(Recall_at_k_batch(X_pred, he, k=50))
        he_tail = he.multiply(tail_row).tocsr(); he_tail.eliminate_zeros()
        tkeep = np.asarray(he_tail.getnnz(axis=1)).ravel() > 0
        if tkeep.any():
            Xp_t = X_pred[tkeep].copy(); Xp_t[:, head_mask] = -np.inf
            acc["tail_ndcg@10"].append(NDCG_binary_at_k_batch(Xp_t, he_tail[tkeep], k=10))
    out = {}
    for key, chunks in acc.items():
        v = np.concatenate(chunks) if chunks else np.array([np.nan])
        out[key] = float(np.mean(v))
        out[key + "_se"] = float(np.std(v) / np.sqrt(len(v)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="golbandi_node_graded,rbmf_seed_graded")
    args = ap.parse_args()
    want = set(args.only.split(","))
    t0 = time.time()

    meta = M.load_meta(PROC); n_items = meta["n_items"]
    bin_train = M.load_train(n_items, PROC)
    te_tr, te_te = M.load_test(n_items, PROC)
    hm, _ = compute_head_mask(bin_train, n_items)
    log(f"ruler: {n_items} items, {te_tr.shape[0]} test users, head={int(hm.sum())}")

    g_train, g_te_tr = load_graded(te_te, log=log)
    assert g_te_tr.multiply(te_te).nnz == 0, "graded fold-in overlaps held-out targets"
    log(f"graded fold-in carries {g_te_tr.nnz / te_tr.nnz:.2f}x the binary fold-in's interactions; "
        f"{float((g_te_tr.data <= 3.5).mean()):.1%} of them are sub-3.5 and absent from the split")

    # CONTROL. The graded fold-in carries 2.26x the binary one's interactions, so a graded gain could
    # simply be a volume gain. This arm holds the interaction SET at all-bands and binarises the values,
    # isolating "more interactions" from "graded values". Without it no graded number is interpretable.
    b_train = g_train.copy(); b_train.data[:] = 1.0
    b_te_tr = g_te_tr.copy(); b_te_tr.data[:] = 1.0

    out = {}
    if "golbandi_node_allbands_binary" in want:
        log("--- CONTROL: golbandi_node, all-bands interaction SET, values binarised ---")
        pr = golbandi_node.fit(b_train, n_items, log=log)
        r = evaluate_decoupled(pr, b_te_tr, te_tr, te_te, hm)
        r["input_regime"] = "all-bands interaction set, values binarised (volume control)"
        out["golbandi_node_allbands_binary"] = r
        log(f"[done] golbandi_node_allbands_binary: full@10={r['ndcg@10']:.4f} "
            f"tail@10={r['tail_ndcg@10']:.4f}  (likes-only binary was 0.3065/0.1895)")
        json.dump(r, open(os.path.join(OUTDIR, "golbandi_node_allbands_binary.json"), "w"), indent=2)

    if "golbandi_node_graded" in want:
        log("--- golbandi_node on graded ratings (fully faithful: node mean = mean rating) ---")
        pr = golbandi_node.fit(g_train, n_items, log=log)
        r = evaluate_decoupled(pr, g_te_tr, te_tr, te_te, hm)
        r["input_regime"] = "graded catalogue ratings, all bands, targets excluded; canonical ranking mask"
        out["golbandi_node_graded"] = r
        log(f"[done] golbandi_node_graded: full@10={r['ndcg@10']:.4f} tail@10={r['tail_ndcg@10']:.4f} "
            f"ndcg@100={r['ndcg@100']:.4f}  (binary row was 0.3065/0.1895)")
        json.dump(r, open(os.path.join(OUTDIR, "golbandi_node_graded.json"), "w"), indent=2)

    if "rbmf_seed_graded" in want:
        log("--- rbmf_seed on graded ratings (seed step regresses real ratings; frozen Y still iALS) ---")
        pr = rbmf_seed.fit(bin_train, n_items, log=log)     # Y from the certified implicit fit
        r = evaluate_decoupled(pr, g_te_tr, te_tr, te_te, hm)
        r["input_regime"] = ("graded catalogue ratings in the per-user LS seed; frozen Y from the "
                             "implicit iALS fit (stated substitution)")
        out["rbmf_seed_graded"] = r
        log(f"[done] rbmf_seed_graded: full@10={r['ndcg@10']:.4f} tail@10={r['tail_ndcg@10']:.4f} "
            f"ndcg@100={r['ndcg@100']:.4f}  (binary row was 0.2534/0.1764)")
        json.dump(r, open(os.path.join(OUTDIR, "rbmf_seed_graded.json"), "w"), indent=2)

    log(f"complete in {(time.time() - t0)/60:.1f}m")


if __name__ == "__main__":
    main()
