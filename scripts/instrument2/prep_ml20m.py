"""
prep_ml20m.py -- Liang et al. (WWW 2018) strong-generalization preprocessing for ML-20M.

Faithful port of the canonical Mult-VAE data-splitting notebook
(github.com/dawenl/vae_cf, data.py / VAE_ML20M_WWW2018.ipynb) and the RecVAE
preprocessing (github.com/ilya-shenbin/RecVAE, preprocessing.py), which are identical.

Protocol (verbatim):
  - keep ratings >= 4 as implicit positives (binarize)
  - filter: users with >= 5 interactions (min_uc=5), no item minimum (min_sc=0)
  - strong generalization: 10,000 validation users + 10,000 test users held out,
    remainder = train users
  - item vocabulary built from TRAIN users only; vad/test restricted to those items
  - for each held-out user: 80% of items = fold-in input ("tr"), remaining 20% =
    prediction target ("te"); split done per-user with test_prop=0.2

Outputs (to data/ml20m/proc/):
  unique_sid.txt          item vocabulary (original movieIds, ordinal = column index)
  show2id / profile2id    (implicit via ordering)
  train.csv               uid,sid  (new contiguous ids) -- train users' items
  validation_tr.csv       vad users fold-in items
  validation_te.csv       vad users held-out items
  test_tr.csv             test users fold-in items
  test_te.csv             test users held-out items
  meta.json               n_items, n_users counts
"""
import os, sys, json
import numpy as np
import pandas as pd

SEED = 98765  # canonical seed used in the Mult-VAE notebook
np.random.seed(SEED)

DATA_DIR = os.path.join("data", "ml20m")
RAW = os.path.join(DATA_DIR, "ml-20m", "ratings.csv")
OUT = os.path.join(DATA_DIR, "proc")
os.makedirs(OUT, exist_ok=True)

MIN_UC = 5
MIN_SC = 0
N_HELDOUT = 10000
TEST_PROP = 0.2


def get_count(tp, id_col):
    return tp[[id_col]].groupby(id_col, as_index=False).size()


def filter_triplets(tp, min_uc=MIN_UC, min_sc=MIN_SC):
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
    """Per-user split: test_prop of each user's items -> te, rest -> tr.
    Matches the Mult-VAE notebook (users with < 5 items go entirely to tr;
    all our held-out users have >=5)."""
    data_grouped_by_user = data.groupby("userId")
    tr_list, te_list = [], []
    rng = np.random.RandomState(98765)
    for _, group in data_grouped_by_user:
        n_items_u = len(group)
        if n_items_u >= 5:
            idx = np.zeros(n_items_u, dtype="bool")
            idx[rng.choice(n_items_u, size=int(test_prop * n_items_u), replace=False).astype("int64")] = True
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
    print(f"[prep] loading {RAW}", flush=True)
    raw = pd.read_csv(RAW)
    print(f"[prep] raw interactions: {len(raw):,}", flush=True)

    # binarize: keep ratings >= 4
    raw = raw[raw["rating"] >= 4.0]
    print(f"[prep] after rating>=4: {len(raw):,}", flush=True)

    raw, user_activity, item_popularity = filter_triplets(raw)
    sparsity = 1.0 * raw.shape[0] / (user_activity.shape[0] * item_popularity.shape[0])
    print(f"[prep] after filter(min_uc={MIN_UC}): {raw.shape[0]:,} events, "
          f"{user_activity.shape[0]:,} users, {item_popularity.shape[0]:,} items, "
          f"sparsity {sparsity*100:.3f}%", flush=True)

    unique_uid = user_activity["userId"].values
    rng = np.random.RandomState(SEED)
    idx_perm = rng.permutation(unique_uid.size)
    unique_uid = unique_uid[idx_perm]

    n_users = unique_uid.size
    n_heldout = N_HELDOUT

    tr_users = unique_uid[:(n_users - n_heldout * 2)]
    vd_users = unique_uid[(n_users - n_heldout * 2):(n_users - n_heldout)]
    te_users = unique_uid[(n_users - n_heldout):]
    print(f"[prep] users: train={len(tr_users):,} val={len(vd_users):,} test={len(te_users):,}", flush=True)

    train_plays = raw[raw["userId"].isin(tr_users)]
    unique_sid = pd.unique(train_plays["movieId"])
    print(f"[prep] item vocab (from train users): {len(unique_sid):,}", flush=True)

    show2id = dict((sid, i) for (i, sid) in enumerate(unique_sid))
    profile2id = dict((pid, i) for (i, pid) in enumerate(unique_uid))

    with open(os.path.join(OUT, "unique_sid.txt"), "w") as f:
        for sid in unique_sid:
            f.write(f"{sid}\n")

    # validation
    vad_plays = raw[raw["userId"].isin(vd_users)]
    vad_plays = vad_plays[vad_plays["movieId"].isin(unique_sid)]
    vad_plays_tr, vad_plays_te = split_train_test_proportion(vad_plays)

    # test
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
        "seed": SEED,
        "min_uc": MIN_UC, "min_sc": MIN_SC, "rating_thresh": 4.0,
        "test_prop": TEST_PROP,
    }
    with open(os.path.join(OUT, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[prep] DONE. meta={meta}", flush=True)


if __name__ == "__main__":
    main()
