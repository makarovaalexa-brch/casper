"""liang_split.py -- BIT-FAITHFUL replication of the Liang et al. (Mult-VAE, WWW 2018) ML-20M
strong-generalization preprocessing.

Source of truth (verified 2026-07-20 by fetching the raw notebook):
  github.com/dawenl/vae_cf, VAE_ML20M_WWW2018.ipynb  ("Data preprocessing" section).
The RecVAE repo (github.com/ilya-shenbin/RecVAE, preprocessing.py) reproduces this exact procedure.

EVERY parameter below is copied from that notebook; provenance noted inline "as in dawenl/vae_cf".
Getting any of these wrong makes every published-number snap fail, so they are pinned, not tunable.

PARAMETERS (verbatim from the notebook):
  - Rating binarization: keep ratings STRICTLY GREATER THAN 3.5.
        notebook: `raw_data = raw_data[raw_data['rating'] > 3.5]`
        For ML-20M (half-star ratings, min increment 0.5) `> 3.5` is EXACTLY `>= 4.0`; we write the
        `> 3.5` form to be byte-identical to the source.
  - filter_triplets(min_uc=5, min_sc=0):
        notebook: `raw_data, user_activity, item_popularity = filter_triplets(raw_data)`  (defaults 5, 0)
        users must have >= 5 kept interactions; NO item minimum.
  - Held-out users: n_heldout_users = 10000  (10k validation + 10k test), remainder = train.
        notebook: `n_heldout_users = 10000`
                  `tr_users = unique_uid[:(n_users - n_heldout_users * 2)]`
                  `vd_users = unique_uid[(n_users - n_heldout_users*2):(n_users - n_heldout_users)]`
                  `te_users = unique_uid[(n_users - n_heldout_users):]`
  - RNG seed = 98765 for BOTH the user permutation and the per-user tr/te split.
        notebook: `np.random.seed(98765); idx_perm = np.random.permutation(unique_uid.size)`
        notebook: (inside split_train_test_proportion) `np.random.seed(98765)` set once, then sequential
        `np.random.choice` draws over pandas-groupby(userId) order.  We reproduce this exactly by
        creating ONE np.random.RandomState(98765) per call (equivalent to setting the global seed once)
        and drawing in sorted-userId group order (pandas groupby default = sorted keys).
  - Item vocabulary from TRAIN users only:
        notebook: `unique_sid = pd.unique(train_plays['movieId'])`
        vad/test interactions are then restricted to `movieId in unique_sid`.
  - Per-user fold-in / target split: test_prop = 0.2, only for users with >= 5 items.
        notebook: `if n_items_u >= 5: idx[np.random.choice(n_items_u, size=int(test_prop*n_items_u),
        replace=False)] = True` -> 20% target ("te"), 80% fold-in ("tr").
        (all our held-out users have >=5 items by the min_uc=5 filter above.)

OUTPUTS (data/ml-20m/proc/): unique_sid.txt, train.csv, validation_tr.csv, validation_te.csv,
  test_tr.csv, test_te.csv, meta.json.  New contiguous ids: uid = profile2id[userId], sid = show2id[movieId].

Usage:  python scripts/baselines/liang_split.py
"""
import os
import json
import numpy as np
import pandas as pd

SEED = 98765          # as in dawenl/vae_cf VAE_ML20M_WWW2018.ipynb
MIN_UC = 5            # as in dawenl/vae_cf (filter_triplets default)
MIN_SC = 0            # as in dawenl/vae_cf
N_HELDOUT = 10000     # as in dawenl/vae_cf (n_heldout_users)
TEST_PROP = 0.2       # as in dawenl/vae_cf (test_prop)
RATING_GT = 3.5       # as in dawenl/vae_cf (raw_data['rating'] > 3.5)

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
DATA_DIR = os.path.join(_ROOT, "data", "ml-20m")
RAW = os.path.join(DATA_DIR, "ratings.csv")
OUT = os.path.join(DATA_DIR, "proc")


def get_count(tp, id_col):
    return tp[[id_col]].groupby(id_col, as_index=False).size()


def filter_triplets(tp, min_uc=MIN_UC, min_sc=MIN_SC):
    """as in dawenl/vae_cf data preprocessing (filter_triplets)."""
    if min_sc > 0:
        itemcount = get_count(tp, "movieId")
        keep = itemcount[itemcount["size"] >= min_sc]["movieId"]
        tp = tp[tp["movieId"].isin(keep)]
    if min_uc > 0:
        usercount = get_count(tp, "userId")
        keep = usercount[usercount["size"] >= min_uc]["userId"]
        tp = tp[tp["userId"].isin(keep)]
    usercount = get_count(tp, "userId")
    itemcount = get_count(tp, "movieId")
    return tp, usercount, itemcount


def split_train_test_proportion(data, test_prop=TEST_PROP):
    """Per-user split: `test_prop` of each user's items -> te, rest -> tr.
    as in dawenl/vae_cf (split_train_test_proportion). One RandomState(98765) per call reproduces the
    notebook's `np.random.seed(98765)`-then-sequential-draws behaviour over sorted-userId groups."""
    data_grouped_by_user = data.groupby("userId")
    tr_list, te_list = [], []
    rng = np.random.RandomState(SEED)
    for _, group in data_grouped_by_user:
        n_items_u = len(group)
        if n_items_u >= 5:
            idx = np.zeros(n_items_u, dtype="bool")
            idx[rng.choice(n_items_u, size=int(test_prop * n_items_u),
                           replace=False).astype("int64")] = True
            tr_list.append(group[np.logical_not(idx)])
            te_list.append(group[idx])
        else:
            tr_list.append(group)
    data_tr = pd.concat(tr_list)
    data_te = pd.concat(te_list)
    return data_tr, data_te


def numerize(tp, profile2id, show2id):
    uid = tp["userId"].map(profile2id)
    sid = tp["movieId"].map(show2id)
    return pd.DataFrame(data={"uid": uid, "sid": sid}, columns=["uid", "sid"])


def main():
    if not os.path.exists(RAW):
        raise SystemExit(f"[liang] {RAW} not found. Run scripts/baselines/download_ml20m.py first.")
    os.makedirs(OUT, exist_ok=True)
    print(f"[liang] loading {RAW}", flush=True)
    raw = pd.read_csv(RAW)
    print(f"[liang] raw interactions: {len(raw):,}", flush=True)

    # binarize: keep ratings > 3.5  (== >= 4.0 for half-star data)  -- as in dawenl/vae_cf
    raw = raw[raw["rating"] > RATING_GT]
    print(f"[liang] after rating>{RATING_GT}: {len(raw):,}", flush=True)

    raw, user_activity, item_popularity = filter_triplets(raw)
    sparsity = 1.0 * raw.shape[0] / (user_activity.shape[0] * item_popularity.shape[0])
    print(f"[liang] after filter(min_uc={MIN_UC}, min_sc={MIN_SC}): {raw.shape[0]:,} events, "
          f"{user_activity.shape[0]:,} users, {item_popularity.shape[0]:,} items, "
          f"sparsity {sparsity*100:.3f}%", flush=True)

    unique_uid = user_activity["userId"].values
    np.random.seed(SEED)                                  # as in dawenl/vae_cf
    idx_perm = np.random.permutation(unique_uid.size)
    unique_uid = unique_uid[idx_perm]

    n_users = unique_uid.size
    tr_users = unique_uid[:(n_users - N_HELDOUT * 2)]
    vd_users = unique_uid[(n_users - N_HELDOUT * 2):(n_users - N_HELDOUT)]
    te_users = unique_uid[(n_users - N_HELDOUT):]
    print(f"[liang] users: train={len(tr_users):,} val={len(vd_users):,} test={len(te_users):,}",
          flush=True)

    train_plays = raw[raw["userId"].isin(tr_users)]
    unique_sid = pd.unique(train_plays["movieId"])        # vocab from TRAIN users only
    print(f"[liang] item vocab (from train users): {len(unique_sid):,}", flush=True)

    show2id = dict((sid, i) for (i, sid) in enumerate(unique_sid))
    profile2id = dict((pid, i) for (i, pid) in enumerate(unique_uid))

    with open(os.path.join(OUT, "unique_sid.txt"), "w") as f:
        for sid in unique_sid:
            f.write(f"{sid}\n")

    vad_plays = raw[raw["userId"].isin(vd_users)]
    vad_plays = vad_plays[vad_plays["movieId"].isin(unique_sid)]
    vad_plays_tr, vad_plays_te = split_train_test_proportion(vad_plays)

    test_plays = raw[raw["userId"].isin(te_users)]
    test_plays = test_plays[test_plays["movieId"].isin(unique_sid)]
    test_plays_tr, test_plays_te = split_train_test_proportion(test_plays)

    numerize(train_plays, profile2id, show2id).to_csv(os.path.join(OUT, "train.csv"), index=False)
    numerize(vad_plays_tr, profile2id, show2id).to_csv(os.path.join(OUT, "validation_tr.csv"), index=False)
    numerize(vad_plays_te, profile2id, show2id).to_csv(os.path.join(OUT, "validation_te.csv"), index=False)
    numerize(test_plays_tr, profile2id, show2id).to_csv(os.path.join(OUT, "test_tr.csv"), index=False)
    numerize(test_plays_te, profile2id, show2id).to_csv(os.path.join(OUT, "test_te.csv"), index=False)

    meta = {
        "n_items": int(len(unique_sid)),
        "n_train_users": int(len(tr_users)),
        "n_val_users": int(len(vd_users)),
        "n_test_users": int(len(te_users)),
        "n_events_total": int(raw.shape[0]),
        "seed": SEED, "min_uc": MIN_UC, "min_sc": MIN_SC, "rating_gt": RATING_GT,
        "n_heldout_users": N_HELDOUT, "test_prop": TEST_PROP,
        "provenance": "dawenl/vae_cf VAE_ML20M_WWW2018.ipynb (bit-faithful)",
    }
    with open(os.path.join(OUT, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[liang] DONE. meta={meta}", flush=True)
    # Expected (published): ~9,990,682 events, ~136,677 users, 20,108 items.
    if len(unique_sid) != 20108:
        print(f"[liang] NOTE: item vocab = {len(unique_sid)} (published Liang split = 20108). "
              f"Verify the raw ratings.csv is the canonical ML-20M before trusting snaps.", flush=True)


if __name__ == "__main__":
    main()
