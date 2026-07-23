"""eval_cold_baselines.py -- CLOSED-FORM baselines at the tower's random cold-k protocol (VAL cohort).

Evaluates {pop, itemknn, ease, ials, belief_mf, golbandi_node} on the canonical Liang-25M VAL cohort
(NOT test) under three fold-in conditions, exactly mirroring train_tower_t2's cold val:
  full : the user's full binary tr-half fold-in (the ordinary val protocol)
  k=8  : fold-in truncated to the SAME fixed random 8-subset the tower uses (COLD_SEED+1)
  k=2  : fold-in truncated to the SAME fixed random 2-subset the tower uses (COLD_SEED)

SUBSET IDENTITY WITH THE TOWER (documented): truncate_graded + COLD_SEED are IMPORTED from
train_tower_t2 (same sampling code, same seeds). The tower samples from its graded val matrix L_val;
we sample from the binary validation_tr matrix. These have IDENTICAL sparsity patterns (same (row,item)
sets: build_graded_eval_matrix asserts nnz == validation_tr rows, both CSR with sorted indices), and
RandomState draws depend only on the per-row lengths and the seed => the chosen (row,item) subsets are
IDENTICAL to the tower's. Assert included (per-row nnz match against the tower's own truncation rule).

DISLIKES (documented decision: DROPPED): binary baselines have no dislike concept; feeding a hated film
as a like poisons them. On THIS protocol the rule is VACUOUS: the Liang val fold-in contains ONLY
binarized likes (rating > 3.5) by split construction -- the graded val matrix carries levels 7-9 only
(verified at tower build). So tower and baselines see the SAME item subsets, all likes. Rows left EMPTY
by truncation are counted, logged per baseline, and scored as the popularity intercept (train
like-count); with k>=2 and every val user having >=4 tr items, the expected count is 0.

MASKING PARITY: like the tower's cold eval, the fold-in the ENCODER/baseline sees is the k-subset, but
metrics.evaluate still masks the FULL tr fold-in and keeps the full te targets -- cold numbers are
directly comparable to the full-fold numbers (same candidate sets).

Output: experiments/baselines/ml25m_liang/cold_baselines_val.json (updated incrementally per baseline)
        + a printed table (full/tail NDCG@10 x {full, k8, k2}).
Thread-capped (OMP=4): safe to run alongside a live training run. Scoring only, no training runs here
(fits are the usual closed-form/ALS solves).
"""
import os
os.environ["OMP_NUM_THREADS"] = "4"          # HARD cap: runs alongside the live tower training
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
import sys
import json
import time
import argparse
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, os.path.join(_ROOT, "src", "baselines"))
sys.path.insert(0, os.path.join(_ROOT, "src", "instrument"))
import metrics as M
import pop as pop_mod
import itemknn
import ease
import ials
import belief_mf
import golbandi_node
from train_tower_t2 import truncate_graded, COLD_SEED     # SAME sampling code + seeds as the tower

PROC = os.path.join(_ROOT, "data", "ml-25m", "proc")
OUTP = os.path.join(_ROOT, "experiments", "baselines", "ml25m_liang", "cold_baselines_val.json")
IALS_HP = {"factors": 200, "reg": 0.1, "alpha": 50.0, "iterations": 15}   # == run_ml25m_liang.IALS_HP


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def compute_head_mask(train, n_items):
    """Identical to run_ml25m_liang (top-33%-train-mass head)."""
    cnt = np.asarray(train.sum(axis=0)).ravel().astype(np.float64)
    order = np.argsort(-cnt)
    cumfrac = np.cumsum(cnt[order]) / cnt.sum()
    head = np.zeros(n_items, dtype=bool)
    head[order[:np.searchsorted(cumfrac, 0.33) + 1]] = True
    return head, cnt


def make_trunc_predict(base_predict, Xk, intercept, n_empty_box):
    """Wrapper predict_fn: metrics.evaluate hands us the FULL fold-in batch (used for masking); the
    baseline SCORES from the aligned truncated batch (cursor over Xk, same row order). Rows with an
    empty k-subset are scored as the popularity intercept (counted)."""
    state = {"pos": 0}

    def predict(X_full):
        pos = state["pos"]; rows = X_full.shape[0]
        Xs = Xk[pos:pos + rows]; state["pos"] = pos + rows
        assert Xs.shape[0] == rows, "truncated fold-in misalignment"
        sc = np.asarray(base_predict(Xs), dtype=np.float32)
        empty = np.asarray(Xs.getnnz(axis=1)).ravel() == 0
        if empty.any():
            sc[empty] = intercept[np.newaxis, :]
            n_empty_box[0] += int(empty.sum())
        return sc
    return predict


def eval_cond(base_predict, Xk, va_tr, va_te, head_mask, intercept):
    """One condition: baseline folds Xk; evaluate masks the FULL va_tr (tower parity)."""
    box = [0]
    pred = make_trunc_predict(base_predict, Xk, intercept, box)
    m = M.evaluate(pred, va_tr, va_te, batch_size=500, head_mask=head_mask)
    return {"ndcg@10": m["ndcg@10"], "tail_ndcg@10": m["tail_ndcg@10"], "empty_rows": box[0]}


def build(name, train, n_items):
    ns = argparse.Namespace
    if name == "pop":
        return pop_mod.fit(train, n_items, log=log)
    if name == "itemknn":
        return itemknn.fit(train, n_items, args=ns(topk=itemknn.DEFAULTS["topk"]), log=log)
    if name == "ease":
        return ease.fit(train, n_items, args=ns(lam=ease.DEFAULTS["lam"]), log=log)
    if name == "ials":
        return ials.fit(train, n_items, args=ns(**IALS_HP), log=log)
    if name == "belief_mf":
        return belief_mf.fit(train, n_items, log=log)
    if name == "golbandi_node":
        return golbandi_node.fit(train, n_items, log=log)
    raise ValueError(name)


def main():
    meta = M.load_meta(PROC); ni = meta["n_items"]
    train = M.load_train(ni, PROC)
    va_tr, va_te = M.load_val(ni, PROC)
    head_mask, cnt = compute_head_mask(train, ni)
    intercept = cnt.astype(np.float32)                       # popularity intercept for empty rows

    # SAME subsets as the tower: same code (truncate_graded), same seeds (k2: COLD_SEED, k8: +1)
    Xk2 = truncate_graded(va_tr, 2, COLD_SEED)
    Xk8 = truncate_graded(va_tr, 8, COLD_SEED + 1)
    for k, Xk in ((2, Xk2), (8, Xk8)):
        per_row = np.asarray(Xk.getnnz(axis=1)).ravel()
        want = np.minimum(k, np.asarray(va_tr.getnnz(axis=1)).ravel())
        assert (per_row == want).all(), f"k={k} truncation rule mismatch vs tower"
        log(f"[cold] k={k}: nnz={Xk.nnz}, empty rows={(per_row == 0).sum()} (expected 0)")

    results = {"_protocol": {"cohort": "validation", "seed_k2": COLD_SEED, "seed_k8": COLD_SEED + 1,
                             "dislikes": "DROPPED (vacuous: val fold-in is likes-only by construction)",
                             "masking": "full tr fold-in masked (tower cold-val parity)"}}
    if os.path.exists(OUTP):                                 # resumable: skip finished baselines
        results.update({k: v for k, v in json.load(open(OUTP)).items() if not k.startswith("_")})

    order = ["pop", "itemknn", "ease", "ials", "belief_mf", "golbandi_node"]  # cheap -> expensive
    for name in order:
        if name in results:
            log(f"[skip] {name}: already in {OUTP}")
            continue
        log(f"--- {name} ---"); t0 = time.time()
        predict = build(name, train, ni)
        row = {}
        row["full"] = eval_cond(predict, va_tr, va_tr, va_te, head_mask, intercept)  # Xk = full fold-in
        row["k8"] = eval_cond(predict, Xk8, va_tr, va_te, head_mask, intercept)
        row["k2"] = eval_cond(predict, Xk2, va_tr, va_te, head_mask, intercept)
        row["seconds"] = round(time.time() - t0, 1)
        results[name] = row
        os.makedirs(os.path.dirname(OUTP), exist_ok=True)
        json.dump(results, open(OUTP, "w"), indent=2)
        log(f"[done] {name}: full={row['full']['ndcg@10']:.4f}/{row['full']['tail_ndcg@10']:.4f} "
            f"k8={row['k8']['ndcg@10']:.4f}/{row['k8']['tail_ndcg@10']:.4f} "
            f"k2={row['k2']['ndcg@10']:.4f}/{row['k2']['tail_ndcg@10']:.4f} "
            f"({row['seconds']/60:.1f}m)")

    print("\n=== COLD BASELINES, VAL cohort (full/tail NDCG@10) ===")
    print(f"{'baseline':<14} {'full':>13} {'k=8':>13} {'k=2':>13}")
    for name in order:
        r = results[name]
        print(f"{name:<14} "
              f"{r['full']['ndcg@10']:.4f}/{r['full']['tail_ndcg@10']:.4f} "
              f"{r['k8']['ndcg@10']:.4f}/{r['k8']['tail_ndcg@10']:.4f} "
              f"{r['k2']['ndcg@10']:.4f}/{r['k2']['tail_ndcg@10']:.4f}")
    print(f"JSON -> {OUTP}")


if __name__ == "__main__":
    main()
