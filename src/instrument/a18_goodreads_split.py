"""A18 -- the Liang strong-generalisation split for the Goodreads arena.

Design sheet: docs/design/DS_A17_GOODREADS_ARENA.md

Mirrors `data/ml-25m/proc/` exactly in schema and recipe. From that split's own meta.json:

    "provenance": "dawenl/vae_cf VAE_ML20M_WWW2018.ipynb (bit-faithful recipe; applied to ML-25M raw)"
    "catalog_restriction": "raw pre-filtered to the project-wide 18430-item catalog ...;
                            user-side Liang-verbatim"

So: restrict the catalogue FIRST (done in A17), then apply Liang's user-side recipe verbatim.

  * like <=> rating > 3.5. Goodreads is integers 1-5, so this is rating >= 4, which is the same
    predicate ML-25M's half-star scale expresses as > 3.5. No adaptation needed.
  * min_uc = 5 likes; min_sc = 0
  * 10,000 validation + 10,000 test users held out; per-user 80/20 fold-in/target on the held-out
    cohorts; seed 98765
  * train.csv carries uid,sid for LIKES ONLY -- the graded ratings live separately (A17's graded.npz),
    exactly as ML-25M keeps them in data/movielens/ratings.csv

THE SECOND MATCHING DECISION, stated because it is a reduction:
ML-25M yields 140,768 train users. Goodreads has ~788k arena users, and using all of them would give
the Goodreads instrument roughly five times the training population. That is not a fairer
replication, it is a DIFFERENT and better-resourced instrument, and it would make any difference
between the two datasets uninterpretable -- as well as taking the port from ~60 h to ~150 h. So the
train cohort is sampled to 140,768 users, matching ML-25M, under the same seed. This is the same
calibrate-to-the-reference principle as the arena size, applied to the cohort.

Run:  python src/instrument/a18_goodreads_split.py
"""
import json
import os
import time

import numpy as np
import pandas as pd

_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
DATA = os.path.join(_ROOT, "data", "goodreads_ucsd")
ARENA = os.path.join(DATA, "arena")
PROC = os.path.join(DATA, "proc")
SEED = 98765
LIKE_GT = 3.5
MIN_UC = 5
N_VAL = N_TEST = 10000
N_TRAIN = 140768         # matched to ML-25M
TEST_PROP = 0.2


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def main():
    os.makedirs(PROC, exist_ok=True)
    t0 = time.time()
    z = np.load(os.path.join(ARENA, "graded.npz"))
    u_all, i_all, r_all = z["user"], z["item"], z["rating"]
    log(f"[graded] {len(u_all):,} rated events, {len(np.unique(u_all)):,} users, "
        f"{len(np.unique(i_all)):,} arena books")

    like = r_all > LIKE_GT
    ul, il = u_all[like], i_all[like]
    log(f"[likes] {len(ul):,} like events (rating > {LIKE_GT}) = {like.mean():.1%} of ratings")

    uu, cnt = np.unique(ul, return_counts=True)
    ok = uu[cnt >= MIN_UC]
    log(f"[min_uc] {len(ok):,} users with >= {MIN_UC} likes (of {len(uu):,})")
    need = N_TRAIN + N_VAL + N_TEST
    if len(ok) < need:
        raise SystemExit(f"only {len(ok)} eligible users, need {need}")

    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(ok))
    te_u = np.sort(ok[perm[:N_TEST]])
    va_u = np.sort(ok[perm[N_TEST:N_TEST + N_VAL]])
    tr_u = np.sort(ok[perm[N_TEST + N_VAL:N_TEST + N_VAL + N_TRAIN]])
    log(f"[cohorts] {len(tr_u):,} train / {len(va_u):,} validation / {len(te_u):,} test "
        f"(sampled from {len(ok):,} eligible, matching ML-25M's 140,768/10,000/10,000)")

    # GLOBAL CONTIGUOUS uid numbering: train 0..N-1, then validation, then test -- the layout
    # data/ml-25m/proc uses and that metrics._load_train_data requires (it indexes rows BY uid).
    uid = {}
    for block in (tr_u, va_u, te_u):
        for x in block:
            uid[int(x)] = len(uid)
    log(f"[uid] remapped {len(uid):,} raw Goodreads user ids to contiguous 0..{len(uid)-1}")

    # sid vocabulary is defined by the TRAIN cohort's likes, as in the ML-25M build
    tr_mask = np.isin(ul, tr_u)
    train_items = np.unique(il[tr_mask])
    sid = {int(b): k for k, b in enumerate(train_items)}
    log(f"[vocab] train vocabulary = {len(train_items):,} books "
        f"(arena 18,430; ML-25M: 18,359 of 18,430)")
    np.savetxt(os.path.join(PROC, "unique_sid.txt"), train_items, fmt="%d")

    def emit(users, name, split_fold):
        m = np.isin(ul, users)
        uu_, ii_ = ul[m], il[m]
        keep = np.array([b in sid for b in ii_])
        uu_, ii_ = uu_[keep], ii_[keep]
        s = np.array([sid[int(b)] for b in ii_], np.int32)
        uu_ = np.array([uid[int(x)] for x in uu_], np.int64)
        if not split_fold:
            pd.DataFrame({"uid": uu_, "sid": s}).to_csv(
                os.path.join(PROC, f"{name}.csv"), index=False)
            return len(uu_)
        order = np.argsort(uu_, kind="stable")
        uu_, s = uu_[order], s[order]
        bnd = np.searchsorted(uu_, np.unique(uu_), side="left").tolist() + [len(uu_)]
        tr_rows, te_rows = [], []
        for a, b in zip(bnd[:-1], bnd[1:]):
            idx = rng.permutation(b - a) + a
            ntg = max(1, int(round(TEST_PROP * (b - a))))
            te_rows.append(idx[:ntg]); tr_rows.append(idx[ntg:])
        tri = np.concatenate(tr_rows); tei = np.concatenate(te_rows)
        pd.DataFrame({"uid": uu_[tri], "sid": s[tri]}).to_csv(
            os.path.join(PROC, f"{name}_tr.csv"), index=False)
        pd.DataFrame({"uid": uu_[tei], "sid": s[tei]}).to_csv(
            os.path.join(PROC, f"{name}_te.csv"), index=False)
        return len(uu_)

    n_tr = emit(tr_u, "train", False)
    n_va = emit(va_u, "validation", True)
    n_te = emit(te_u, "test", True)
    log(f"[emit] train {n_tr:,} events | validation {n_va:,} | test {n_te:,}")

    meta = {"n_items": int(len(train_items)), "n_train_users": int(len(tr_u)),
            "n_val_users": int(len(va_u)), "n_test_users": int(len(te_u)),
            "n_events_total": int(n_tr + n_va + n_te), "seed": SEED, "min_uc": MIN_UC,
            "min_sc": 0, "rating_gt": LIKE_GT, "n_heldout_users": N_VAL, "test_prop": TEST_PROP,
            "dataset": "goodreads-ucsd-full",
            "raw": os.path.join(DATA, "goodreads_interactions.csv"),
            "provenance": "dawenl/vae_cf VAE_ML20M_WWW2018.ipynb recipe, user-side verbatim, "
                          "applied to the Goodreads arena built by a17_goodreads_arena.py",
            "catalog_restriction": "arena = the 18,430 most-rated books, matched to ML-25M's "
                                   "18,430-film arena SIZE (its >=20-rating rule leaves 434,531 "
                                   "books here, untrainable); keeps 56.7% of rating events",
            "cohort_restriction": "train cohort SAMPLED to 140,768 users to match ML-25M; using all "
                                  "~788k arena users would give a better-resourced instrument and "
                                  "make the cross-dataset comparison uninterpretable",
            "seconds": time.time() - t0}
    np.savez_compressed(os.path.join(PROC, "uid_map.npz"),
                        raw=np.array(list(uid.keys()), np.int64),
                        contiguous=np.array(list(uid.values()), np.int64))
    json.dump(meta, open(os.path.join(PROC, "meta.json"), "w"), indent=2)
    log(f"\n[done] {PROC} ({meta['seconds']/60:.1f}m)")
    log(json.dumps({k: meta[k] for k in ("n_items", "n_train_users", "n_val_users",
                                         "n_test_users", "n_events_total")}))


if __name__ == "__main__":
    main()
