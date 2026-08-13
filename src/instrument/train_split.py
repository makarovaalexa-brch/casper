r"""THE TRAIN-SIDE FOLD-IN / TARGET SPLIT -- workstream E1 (`docs/design/PAPER_B_POLICY_DESIGN.md` D.6b).

WHY THIS EXISTS, AND IT IS NOT A CONVENIENCE
--------------------------------------------
The policy trains on the 140,768 TRAIN users (author ruling D.6b: apply the frozen tree to them, never
regrow it, never measure NDCG there). Interview STATES are all we take from those users -- but a state
is a function of the ANSWERS, and answerability is the single most load-bearing property in this
chapter.

Validation users answer only from their 80% fold-in half: `build_real_ctx` builds their answer lookup
from `ctx.allb`, which is every catalogue rating of the user MINUS the 20% held-out target half. Train
users have no such split on disk, so answering them from 100% of their ratings would give train
interviews a systematically HIGHER answer rate than the deployment distribution the policy is selected
and reported under. The policy would then be trained on states that cannot occur at test time.

So: apply the SAME per-user 80/20 recipe, SAME seed 98765, SAME code path, to the train users, and
answer only from the 80% half. The 20% is discarded -- no NDCG is ever measured on a train user, so it
is needed for nothing except to be removed from the answer set.

FAITHFULNESS, PROVEN NOT ASSERTED
---------------------------------
`liang_split.split_train_test_proportion` walks pandas groups in sorted-userId order and draws
`np.random.RandomState(98765).choice(n_u, int(0.2*n_u), replace=False)` per group, sequentially, so the
draw a user receives depends on every earlier user's size. That sequential stream cannot be vectorised,
but it CAN be reproduced exactly with a numpy loop over group boundaries -- and this module proves it
does, by rebuilding the VALIDATION split from the raw ratings and asserting it reproduces
`validation_te.csv` row-for-row (`verify_against_validation`). A faster reimplementation that
reproduces the shipped artifact byte-for-byte is the same code path; one that merely looks similar is
not (HARD RULE 2).

FIREWALL: this module never reads `test_tr.csv` / `test_te.csv` and never builds the test cohort.
"""
import os
import sys
import json
import time
import hashlib
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, os.path.join(_ROOT, "src", "baselines"))
sys.path.insert(0, os.path.join(_ROOT, "scripts", "baselines"))
import metrics as M
import liang_split as LS

PROC = os.path.join(_ROOT, "data", "ml-25m", "proc")
CACHE = os.path.join(_ROOT, ".cache", "instrument")
CACHE_NPZ = os.path.join(CACHE, "train_foldin_split_98765.npz")


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def te_mask_liang(group_key, seed=LS.SEED, test_prop=LS.TEST_PROP):
    """The per-user 20% target mask, reproducing `liang_split.split_train_test_proportion` EXACTLY.

    `group_key` is the per-row user key IN THE ORIGINAL ROW ORDER of the plays frame. Groups are taken
    in sorted-key order (pandas `groupby` default) and rows keep their original order inside a group
    (pandas `groupby` is stable), which is what a stable argsort gives. One RandomState per call ==
    the notebook's single `np.random.seed(98765)`; draws are sequential over groups, so the group ORDER
    and the group SIZES are both part of the contract.
    """
    key = np.asarray(group_key)
    order = np.argsort(key, kind="stable")                   # sorted groups, original order within
    ks = key[order]
    starts = np.flatnonzero(np.concatenate([[True], ks[1:] != ks[:-1]]))
    sizes = np.diff(np.concatenate([starts, [len(ks)]]))
    rng = np.random.RandomState(seed)
    mask_sorted = np.zeros(len(ks), bool)
    for s, n in zip(starts, sizes):
        n = int(n)
        if n >= 5:                                           # as in dawenl/vae_cf
            pick = rng.choice(n, size=int(test_prop * n), replace=False).astype("int64")
            mask_sorted[s + pick] = True
    mask = np.zeros(len(ks), bool)
    mask[order] = mask_sorted
    return mask


def _binarised_plays(raw):
    """`raw` (catalog-restricted, ALL bands) -> the binarised, min_uc-filtered frame liang split on."""
    binr = raw[raw["rating"] > LS.RATING_GT]
    binr, _ua, _ic = LS.filter_triplets(binr)
    return binr


def verify_against_validation(raw, unique_uid, show2id):
    """G-TS1. Rebuild the VALIDATION 80/20 from raw with `te_mask_liang` and assert it reproduces the
    shipped `validation_tr.csv` / `validation_te.csv` EXACTLY (as (uid, sid) sets, and on counts).

    This is the gate that licenses using the same function on the train users. It fails loudly rather
    than warning: an approximate reproduction is a different split, and a different split is a
    different answer distribution.
    """
    import pandas as pd
    n_all = len(unique_uid)
    vd_users = unique_uid[n_all - 2 * LS.N_HELDOUT:n_all - LS.N_HELDOUT]
    binr = _binarised_plays(raw)
    vad = binr[binr["userId"].isin(set(vd_users.tolist()))]
    vad = vad[vad["movieId"].isin(show2id)]
    mask = te_mask_liang(vad["userId"].values)
    profile2id = {int(u): i for i, u in enumerate(unique_uid)}
    uid = vad["userId"].map(profile2id).values.astype(np.int64)
    sid = vad["movieId"].map(show2id).values.astype(np.int64)
    got_te = set(zip(uid[mask].tolist(), sid[mask].tolist()))
    got_tr = set(zip(uid[~mask].tolist(), sid[~mask].tolist()))
    ref_te = pd.read_csv(os.path.join(PROC, "validation_te.csv"))
    ref_tr = pd.read_csv(os.path.join(PROC, "validation_tr.csv"))
    want_te = set(zip(ref_te["uid"].tolist(), ref_te["sid"].tolist()))
    want_tr = set(zip(ref_tr["uid"].tolist(), ref_tr["sid"].tolist()))
    assert got_te == want_te, (
        f"G-TS1 FAILED: the rebuilt validation TARGET half differs from validation_te.csv "
        f"({len(got_te ^ want_te)} rows differ of {len(want_te)}). The train-side split would NOT be "
        f"the canonical recipe -- REFUSING.")
    assert got_tr == want_tr, "G-TS1 FAILED: the rebuilt validation FOLD-IN half differs"
    log(f"[G-TS1] the 80/20 reproducer is BYTE-FAITHFUL: rebuilt validation split == "
        f"validation_tr.csv ({len(want_tr):,} rows) + validation_te.csv ({len(want_te):,} rows), exactly")
    return True


def build_train_split(raw=None, unique_uid=None, show2id=None, force=False):
    """The train users' 80/20, cached. Returns a dict with `te_uid` / `te_sid` (the DISCARDED target
    half) and the contract it was built under.

    The cache is contract-signed (HARD RULE 11b): a cache written under a different recipe, seed,
    vocabulary or train.csv is REFUSED, not merged.
    """
    meta = M.load_meta(PROC)
    sig = {"v": 1, "recipe": "liang_split.split_train_test_proportion", "seed": LS.SEED,
           "test_prop": LS.TEST_PROP, "n_items": int(meta["n_items"]),
           "n_train_users": int(meta["n_train_users"]),
           "train_csv_bytes": int(os.path.getsize(os.path.join(PROC, "train.csv")))}
    if os.path.exists(CACHE_NPZ) and not force:
        z = np.load(CACHE_NPZ, allow_pickle=False)
        got = json.loads(str(z["sig"]))
        if got == sig:
            log(f"[train-split] cache hit ({os.path.basename(CACHE_NPZ)}): "
                f"{len(z['te_uid']):,} target-half rows over {sig['n_train_users']:,} train users")
            return {"te_uid": z["te_uid"], "te_sid": z["te_sid"], "sig": sig}
        log(f"[train-split] cache REFUSED (contract changed): {got} != {sig}")

    if raw is None:
        from train_tower_t2 import reproduce_partition
        unique_uid, _tr_set, _vd, _te, _ntr, raw, show2id, _usid = reproduce_partition()
    verify_against_validation(raw, unique_uid, show2id)

    n_all = len(unique_uid)
    tr_users = unique_uid[:n_all - 2 * LS.N_HELDOUT]
    binr = _binarised_plays(raw)
    plays = binr[binr["userId"].isin(set(tr_users.tolist()))]
    plays = plays[plays["movieId"].isin(show2id)]
    # the train item vocabulary IS derived from these users, so nothing may be dropped here
    assert len(plays) == int(M.load_train(meta["n_items"], PROC).nnz), (
        f"train plays {len(plays):,} != train.csv nnz -- the reproduced partition is not the shipped one")
    t0 = time.time()
    mask = te_mask_liang(plays["userId"].values)
    profile2id = {int(u): i for i, u in enumerate(unique_uid)}
    uid = plays["userId"].map(profile2id).values.astype(np.int32)
    sid = plays["movieId"].map(show2id).values.astype(np.int32)
    te_uid, te_sid = uid[mask], sid[mask]
    log(f"[train-split] {len(plays):,} liked train interactions -> {int(mask.sum()):,} target-half "
        f"({mask.mean() * 100:.2f}%) / {int((~mask).sum()):,} fold-in half, {time.time() - t0:.1f}s")
    os.makedirs(CACHE, exist_ok=True)
    np.savez_compressed(CACHE_NPZ + ".tmp.npz", te_uid=te_uid, te_sid=te_sid,
                        sig=np.asarray(json.dumps(sig, sort_keys=False)))
    os.replace(CACHE_NPZ + ".tmp.npz", CACHE_NPZ)
    return {"te_uid": te_uid, "te_sid": te_sid, "sig": sig}


def train_tr_te(n_items, lo=None, hi=None, **kw):
    """(va_tr, va_te) CSR for train uids in `[lo, hi)`, matching `metrics.load_val`'s shape contract.

    va_tr = the 80% fold-in half, va_te = the 20% discarded target half. Row r of the returned matrices
    is train uid `lo + r`, exactly as `_load_tr_te_data` reindexes a held-out cohort by its own uid
    offset.
    """
    from scipy import sparse
    blob = build_train_split(**kw)
    full = M.load_train(n_items, PROC)                       # 140,768 x n_items, ALL of each profile
    lo = 0 if lo is None else int(lo)
    hi = full.shape[0] if hi is None else int(hi)
    te_u, te_s = blob["te_uid"].astype(np.int64), blob["te_sid"].astype(np.int64)
    keep = (te_u >= lo) & (te_u < hi)
    n = hi - lo
    va_te = sparse.csr_matrix((np.ones(int(keep.sum()), np.float32), (te_u[keep] - lo, te_s[keep])),
                              shape=(n, n_items), dtype=np.float32)
    va_tr = (full[lo:hi] - va_te).tocsr()
    va_tr.eliminate_zeros()
    assert va_tr.min() >= 0.0, "the target half is not a subset of the train profile"
    assert va_tr.nnz + va_te.nnz == int(full[lo:hi].nnz), "the 80/20 halves do not partition the profile"
    return va_tr, va_te


if __name__ == "__main__":
    build_train_split(force="--force" in sys.argv)
