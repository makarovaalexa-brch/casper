r"""arm_n.py -- the NATIVE-REGIME (arm N) protocol object: data, candidate pool, canary gate.

SPEC: docs/results/PROTOCOL_DISLIKE_DISCARD.md section 11 (author decision 2026-07-30).

THE TWO ARMS.
  ARM A  canonical Liang, verbatim, unchanged. Certification anchor. Nothing here touches it.
  ARM N  same 10k test users, same held-out targets, same 18,359-item catalogue, full-rank NDCG@10
         (full + tail). Two changes relative to A:
           1. INPUT  -- every model reads its own PUBLISHED input contract. Binary-native models
              (EASE, RecVAE, Mult-VAE, TurboCF, iALS, SASRec, MostPop) read exactly what they read in
              arm A. Ratings-native models (Golbandi, RBMF, TaNP) read catalogue ratings. The tower
              reads signed/graded. Nobody is forced to eat a representation their design cannot state.
           2. POOL   -- the candidate pool is a property of the PROTOCOL, not of the model: every rated
              fold-in-side item (likes AND dislikes) is -inf for EVERY model, identically, whatever
              that model consumed.

WHY THE POOL IS DEFINED THIS WAY. Under arm A only the >3.5 fold-in is excluded, so each user's own
sub-3.5 ratings sit in the candidate pool as permanent non-targets that no model is allowed to see.
That is (a) unlike deployment -- a served system filters everything the user has already rated -- and
(b) the exact channel that makes a signed model's advantage un-scoreable: give it the dislikes and it
collects a free lift by demoting items that were guaranteed wrong anyway. Excluding them for everyone
kills the free lift, so any residual edge is off-support generalisation: "knowing you hated this moved
the neighbourhood", measured on held-out likes the model never saw.

THE COUPLING THIS FILE EXISTS TO PREVENT. metrics.evaluate historically derived the -inf mask from the
model's own input matrix. Handing it a 2.26x richer graded fold-in therefore deleted 2.26x more
candidates and produced the void "Golbandi 0.3894". metrics.evaluate now takes `mask_X`; arm N always
passes N_POOL, so input and pool vary independently and only one of them varies at a time.

  from arm_n import load_arm_n
  D = load_arm_n()
  M.evaluate(predict, D["te_tr"], D["te_te"], head_mask=D["head_mask"], mask_X=D["pool"])
"""
import os
import sys
import time
import numpy as np
from scipy import sparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import metrics as M
from run_ml25m_liang import compute_head_mask

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
PROC = os.path.join(_ROOT, "data", "ml-25m", "proc")
CACHE = os.path.join(_ROOT, ".cache", "baselines", "graded_ml25m.npz")


def _log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _save_npz(path, mats):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    out = {}
    for name, X in mats.items():
        X = X.tocsr()
        out[f"{name}.data"] = X.data
        out[f"{name}.indices"] = X.indices
        out[f"{name}.indptr"] = X.indptr
        out[f"{name}.shape"] = np.asarray(X.shape, dtype=np.int64)
    np.savez(path, **out)


def _load_npz(path, names):
    z = np.load(path)
    return {n: sparse.csr_matrix((z[f"{n}.data"], z[f"{n}.indices"], z[f"{n}.indptr"]),
                                 shape=tuple(z[f"{n}.shape"])) for n in names}


def load_graded_cached(te_te, log=_log):
    """(graded_train, graded_test_foldin), cached. The uncached rebuild re-reads the 25M-row ratings
    csv and re-runs the whole partition reproducer (~47 s) -- fine once, absurd per baseline."""
    names = ["g_train", "g_te_tr"]
    if os.path.exists(CACHE):
        mats = _load_npz(CACHE, names)
        log(f"[arm_n] graded matrices from cache: train nnz={mats['g_train'].nnz} "
            f"test nnz={mats['g_te_tr'].nnz}")
        return mats["g_train"], mats["g_te_tr"]
    from graded_data import load_graded
    g_train, g_te_tr = load_graded(te_te, log=log)
    _save_npz(CACHE, {"g_train": g_train, "g_te_tr": g_te_tr})
    log(f"[arm_n] graded matrices cached -> {CACHE}")
    return g_train, g_te_tr


def build_pool(te_tr, g_te_tr, log=_log):
    """The arm-N candidate exclusion: every rated fold-in-side item, likes union dislikes.

    g_te_tr already excludes each user's held-out targets, so the targets stay rankable -- they are the
    thing being measured and must never enter the pool mask. te_tr is unioned in defensively rather
    than assumed to be a subset: the binary fold-in comes from the split's own csv while g_te_tr is a
    catalogue-ratings reconstruction, and nothing in the repo has ever asserted the two agree."""
    b_g = g_te_tr.copy()
    b_g.data[:] = 1.0
    b_t = te_tr.copy()
    b_t.data[:] = 1.0
    pool = (b_g + b_t).tocsr()
    pool.data[:] = 1.0
    pool.eliminate_zeros()

    # Provenance check nobody had run: is the canonical binary fold-in inside the graded rebuild?
    only_in_te_tr = (b_t - b_t.multiply(b_g)).nnz
    log(f"[arm_n] pool nnz={pool.nnz} (te_tr {te_tr.nnz}, graded {g_te_tr.nnz}); "
        f"te_tr items absent from the graded rebuild: {only_in_te_tr}")
    if only_in_te_tr:
        log(f"[arm_n] NOTE: {only_in_te_tr} binary fold-in cells are not in the graded reconstruction; "
            f"the union keeps them, so the pool is a superset of arm A's exclusion by construction.")
    return pool


def load_arm_n(log=_log):
    """Everything arm N needs, with the invariants checked once, here, rather than per baseline."""
    n_items = len(open(os.path.join(PROC, "unique_sid.txt")).read().split())
    train = M.load_train(n_items, PROC)
    te_tr, te_te = M.load_test(n_items, PROC)
    head_mask, _cnt = compute_head_mask(train, n_items)   # from the BINARY train matrix, always
    g_train, g_te_tr = load_graded_cached(te_te, log=log)
    pool = build_pool(te_tr, g_te_tr, log=log)

    # Invariants. Any of these firing means the arm is not comparable to arm A and must not be run.
    assert te_tr.shape == te_te.shape == g_te_tr.shape == pool.shape, "test-side shape mismatch"
    assert train.shape[1] == n_items == 18359, f"catalogue drift: {train.shape[1]} vs 18359"
    assert te_tr.shape[0] == 10000, f"test cohort size {te_tr.shape[0]}"
    assert g_train.shape[0] == 140768, f"train cohort size {g_train.shape[0]}"
    leak = pool.multiply(te_te).nnz
    assert leak == 0, f"POOL/TARGET LEAK: {leak} held-out targets are inside the exclusion mask"
    log(f"[arm_n] ruler: {n_items} items, {te_tr.shape[0]} test users, head={int(head_mask.sum())}; "
        f"pool excludes {pool.nnz / te_tr.shape[0]:.1f} items/user vs "
        f"{te_tr.nnz / te_tr.shape[0]:.1f} in arm A; target leak 0")
    return {"n_items": n_items, "train": train, "g_train": g_train, "te_tr": te_tr, "te_te": te_te,
            "g_te_tr": g_te_tr, "pool": pool, "head_mask": head_mask}


if __name__ == "__main__":
    D = load_arm_n()
    g = D["g_te_tr"]
    print(f"[arm_n] graded fold-in: mean={g.data.mean():.3f} frac<=3.5={float((g.data <= 3.5).mean()):.3f}")
