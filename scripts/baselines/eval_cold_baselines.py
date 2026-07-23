"""eval_cold_baselines.py v2 -- CLOSED-FORM baselines at the tower's random cold-k protocol (VAL cohort),
now with POPULARITY-FLOOR BLENDED arms and a k0 (zero fold-in) column.

Conditions per arm: full / k=8 / k=2 / k=0, mirroring train_tower_t2's cold val exactly
(same truncate_graded code, same COLD_SEED subsets; k0 = zero items).

SUBSET IDENTITY WITH THE TOWER (documented): truncate_graded + COLD_SEED are IMPORTED from
train_tower_t2 (same sampling code, same seeds). The tower samples from its graded val matrix L_val; we
sample from the binary validation_tr matrix. These have IDENTICAL sparsity patterns (same (row,item)
sets; both CSR with sorted indices), and RandomState draws depend only on per-row lengths and the seed
=> the chosen (row,item) subsets are IDENTICAL to the tower's. Per-row nnz asserts included.

DISLIKES (documented decision: DROPPED): binary baselines have no dislike concept; feeding a hated film
as a like poisons them. On THIS protocol the rule is VACUOUS: the Liang val fold-in contains ONLY
binarized likes (rating > 3.5) by split construction (graded val levels are 7-9 only, verified at tower
build), so tower and baselines see the SAME item subsets, all likes.

MASKING PARITY: the baseline scores from the k-subset, but metrics.evaluate still masks the FULL tr
fold-in and keeps the full te targets -- cold numbers are directly comparable to full-fold numbers.

RAW ARMS ({pop, itemknn, ease, ials, belief_mf, golbandi_node}): fit on train as usual, score the
truncated binary fold-in. k0 note: raw arms score their NATURAL zero-input output -- for EASE/itemknn/
ials/golbandi that is an all-constant (zero) score row, i.e. UNDEFINED ranking with ties broken by item
index inside metrics' argpartition; reported honestly (expect ~0). pop ignores the fold-in entirely.

FLOOR ARMS ('<name>+floor', for ease / ials / itemknn / golbandi_node):
    score = alpha_k * popnorm + (1 - alpha_k) * foldnorm
  NORMALIZATION (documented): foldnorm = the raw fold score row Z-SCORED OVER ITEMS PER USER ROW
  ((s - mean_i)/max(std_i, 1e-8), computed within each user batch); popnorm = the train like-count
  vector z-scored over items once (identical for every user). alpha_k selected on the grid
  {0, .2, .4, .6, .8, 1} PER baseline PER k on the VAL cohort ITSELF -- deliberately GENEROUS to the
  baselines (documented; not an honest-tuning claim). Selection metric: full NDCG@10.
  GRID SHORTCUTS (exact, not approximations): per-row z-scoring is strictly monotone within a row, so
  alpha=0 ranks IDENTICALLY to the raw arm and alpha=1 IDENTICALLY to pop -- their metrics are reused;
  only alpha in {.2,.4,.6,.8} needs fresh ranking passes. k0 floor = pop BY CONSTRUCTION (foldnorm of a
  constant row is 0 under the std guard): reported as pop's metrics with alpha=1.0.
  belief_mf gets NO floor arm: its Gaussian PRIOR (mu0 fit on train users) IS its floor -- at k=0 the
  posterior collapses to the prior mean, the model's own intercept. Noted in the output.

TOWER REFERENCE: the printed table includes the tower's k0 intercept from the --eval_cold line for
context (not recomputed here).

Output: experiments/baselines/ml25m_liang/cold_baselines_val.json (v2 schema, incremental per baseline;
the raw-only v1 results are preserved in cold_baselines_val_raw.json) + a printed table.
Thread-capped (OMP=4); scoring only (closed-form refits sanctioned -- no artifact caching).
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
from scipy import sparse

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
ALPHAS_EVAL = (0.2, 0.4, 0.6, 0.8)     # 0 and 1 reused exactly from raw / pop (monotone-z shortcut)
TOWER_K0_REF = ("tower t2warm_v3_best ep1 (--eval_cold 2026-07-23)", 0.1283)


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


def score_matrix(base_predict, Xk, batch=500):
    """One pass: the baseline's raw scores for every val row under fold-in Xk. float32 (n_users, ni)."""
    n = Xk.shape[0]
    out = None
    for st in range(0, n, batch):
        sc = np.asarray(base_predict(Xk[st:st + batch]), dtype=np.float32)
        if out is None:
            out = np.empty((n, sc.shape[1]), dtype=np.float32)
        out[st:st + sc.shape[0]] = sc
    return out


def make_slice_predict(S):
    """predict_fn over a precomputed score matrix (cursor; metrics.evaluate masks the FULL fold-in)."""
    state = {"pos": 0}

    def predict(X_full):
        pos = state["pos"]; rows = X_full.shape[0]
        state["pos"] = pos + rows
        return S[pos:pos + rows].astype(np.float32, copy=True)   # copy: evaluate writes -inf into it
    return predict


def make_blend_predict(Z, popz, alpha):
    """predict_fn: alpha * popnorm + (1-alpha) * per-row-z-scored fold scores (Z pre-z-scored)."""
    state = {"pos": 0}

    def predict(X_full):
        pos = state["pos"]; rows = X_full.shape[0]
        state["pos"] = pos + rows
        return (alpha * popz[np.newaxis, :] + (1.0 - alpha) * Z[pos:pos + rows]).astype(np.float32)
    return predict


def zscore_rows_inplace(S):
    """Per-user-row z-score over items (strictly monotone per row => ranking-preserving)."""
    mu = S.mean(axis=1, keepdims=True)
    sd = np.maximum(S.std(axis=1, keepdims=True), 1e-8)
    S -= mu; S /= sd
    return S


def ev(pred, va_tr, va_te, head_mask):
    m = M.evaluate(pred, va_tr, va_te, batch_size=500, head_mask=head_mask)
    return {"ndcg@10": m["ndcg@10"], "tail_ndcg@10": m["tail_ndcg@10"]}


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
    popz = ((cnt - cnt.mean()) / max(cnt.std(), 1e-8)).astype(np.float32)   # popnorm (z over items)

    Xk2 = truncate_graded(va_tr, 2, COLD_SEED)
    Xk8 = truncate_graded(va_tr, 8, COLD_SEED + 1)
    Xk0 = sparse.csr_matrix(va_tr.shape, dtype=np.float32)
    for k, Xk in ((2, Xk2), (8, Xk8)):
        want = np.minimum(k, np.asarray(va_tr.getnnz(axis=1)).ravel())
        assert (np.asarray(Xk.getnnz(axis=1)).ravel() == want).all(), f"k={k} rule mismatch vs tower"
    conds = [("full", va_tr), ("k8", Xk8), ("k2", Xk2), ("k0", Xk0)]

    results = {"_protocol": {
        "cohort": "validation", "seed_k2": COLD_SEED, "seed_k8": COLD_SEED + 1,
        "dislikes": "DROPPED (vacuous: val fold-in is likes-only by construction)",
        "masking": "full tr fold-in masked (tower cold-val parity)",
        "floor": "alpha*popnorm+(1-alpha)*foldnorm, both z-scored over items (fold: per user row); "
                 "alpha per baseline per k selected on VAL itself (GENEROUS), grid {0,.2,.4,.6,.8,1}, "
                 "selection metric full NDCG@10; alpha=0/1 metrics reused exactly (monotone-z)",
        "k0_raw": "natural zero-input scores; constant rows = undefined ranking, index tie-break",
        "belief_mf": "no floor arm: its Gaussian prior mu0 IS its floor (posterior->prior at k=0)",
        "tower_k0_reference": {"source": TOWER_K0_REF[0], "ndcg@10": TOWER_K0_REF[1]}}}
    if os.path.exists(OUTP):                                 # resumable per baseline (v2 schema only)
        results.update({k: v for k, v in json.load(open(OUTP)).items() if not k.startswith("_")})

    order = ["pop", "itemknn", "ease", "ials", "belief_mf", "golbandi_node"]
    floored = {"itemknn", "ease", "ials", "golbandi_node"}
    for name in order:
        if name in results:
            log(f"[skip] {name}: already in {OUTP}")
            continue
        log(f"--- {name} ---"); t0 = time.time()
        predict = build(name, train, ni)
        row = {"raw": {}}
        if name in floored:
            row["floor"] = {}
        for cname, Xk in conds:
            S = score_matrix(predict, Xk)                    # ONE baseline pass per condition
            row["raw"][cname] = ev(make_slice_predict(S), va_tr, va_te, head_mask)
            if name in floored:
                pm = results["pop"]["raw"]["full"]           # pop metrics (fold-in-invariant)
                if cname == "k0":                            # floor(k0) = pop BY CONSTRUCTION
                    row["floor"][cname] = {"alpha": 1.0, "note": "pop by construction", **pm}
                else:
                    Z = zscore_rows_inplace(S)               # in place; raw eval already done
                    cand = [(0.0, row["raw"][cname])] + \
                           [(a, ev(make_blend_predict(Z, popz, a), va_tr, va_te, head_mask))
                            for a in ALPHAS_EVAL] + [(1.0, pm)]
                    best_a, best_m = max(cand, key=lambda t: t[1]["ndcg@10"])
                    row["floor"][cname] = {"alpha": best_a,
                                           "ndcg@10": best_m["ndcg@10"],
                                           "tail_ndcg@10": best_m["tail_ndcg@10"],
                                           "grid": {str(a): m["ndcg@10"] for a, m in cand}}
            del S
        row["seconds"] = round(time.time() - t0, 1)
        if name == "belief_mf":
            row["note"] = "no floor arm: own Gaussian prior is its floor"
        results[name] = row
        os.makedirs(os.path.dirname(OUTP), exist_ok=True)
        json.dump(results, open(OUTP, "w"), indent=2)
        msg = f"[done] {name} ({row['seconds']/60:.1f}m): " + " ".join(
            f"{c}={row['raw'][c]['ndcg@10']:.4f}" for c, _ in conds)
        if name in floored:
            msg += " | floor " + " ".join(
                f"{c}={row['floor'][c]['ndcg@10']:.4f}(a={row['floor'][c]['alpha']})" for c, _ in conds)
        log(msg)

    def fmt(m):
        return f"{m['ndcg@10']:.4f}/{m['tail_ndcg@10']:.4f}"
    print("\n=== COLD BASELINES v2, VAL cohort (full/tail NDCG@10; floor cells show @alpha) ===")
    print(f"{'arm':<22} {'full':>17} {'k=8':>17} {'k=2':>17} {'k=0':>17}")
    for name in order:
        r = results[name]
        print(f"{name:<22} " + " ".join(f"{fmt(r['raw'][c]):>17}" for c in ("full", "k8", "k2", "k0")))
        if "floor" in r:
            cells = " ".join(f"{(fmt(r['floor'][c]) + '@' + str(r['floor'][c]['alpha'])):>17}"
                             for c in ("full", "k8", "k2", "k0"))
            print(f"{name + '+floor':<22} {cells}")
    print(f"(reference) tower k0 intercept: full@10={TOWER_K0_REF[1]} [{TOWER_K0_REF[0]}]")
    print("(note) belief_mf: no floor arm -- its Gaussian prior IS its floor")
    print(f"JSON -> {OUTP}")


if __name__ == "__main__":
    main()
